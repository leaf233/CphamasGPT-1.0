"""
skills/structure_selection.py

跨房室结构选择：在 Phase 4 结束、Phase 5 开始前，
收集 1/2/3 房室的最终 OFV 与参数数，用 BIC 选出最优结构。

跨进程协作（并行房室）：
  - 每个房室进程把 (n_cmt, ofv, n_params, best_code_path) 写入共享 JSON
  - 本 Skill 等待所有房室写入完成（或超时），然后做 BIC 比较
  - 若本房室不是最优，则设置 state.force_stop=True 优雅退出
  - 外层调度器只保留最优房室继续跑 Phase 5
"""
import json
import math
import os
import time
from .base import Skill, SkillResult
from utils.compartment_registry import CompartmentRegistry



class StructureSelectionSkill(Skill):
    name = 'structure_selection'
    phase = [4, 5]          # 允许在 4 末尾或 5 开头触发
    tools = []
    prompt_template = ''

    REGISTRY_PATH = None    # 由 Orchestrator 初始化时注入

    # ★ 新增：懒加载的 registry 实例
    @property
    def registry(self) -> CompartmentRegistry:
        if self._registry is None:
            if not self.REGISTRY_PATH:
                raise RuntimeError(
                    "StructureSelectionSkill.REGISTRY_PATH not set — "
                    "Orchestrator must inject it before calling run()")
            self._registry = CompartmentRegistry(self.REGISTRY_PATH)
        return self._registry

    def __init__(self):
        super().__init__()
        self._registry = None


    def run(self, state, llm, tools) -> SkillResult:
        n_cmt = state.locked_compartments
        ofv = state.best_ofv
        n_params = self._count_parameters(state.best_code)
        n_obs = state.data_profile.get('n_subjects', 0)

        # 若运行配置中只启用了一个房室，直接跳过等待
        enabled = state.data_profile.get('enabled_compartments', [1, 2, 3])
        if len(enabled) <= 1:
            print(f"  [STRUCTURE] Single-compartment mode — skipping BIC")
            return SkillResult(done=True, updates={'structure_selected': True})

        # 1) 本房室结果写入共享注册表
        self._write_registry(n_cmt, ofv, n_params, state.best_code)

        # self.registry.write( n_cmt=n_cmt, ofv=ofv, n_params=n_params, best_code_path=getattr(state, 'best_code_path', None))

        # 2) 等待其他房室写入（或超时）
        '''
        expected = set(state.data_profile.get('enabled_compartments',
                                              [1, 2, 3]))
        if len(expected) <= 1:
            print(f"  [STRUCTURE] Single-compartment mode — skipping BIC wait")
            registry = self.registry.read()
        else:
            registry = self.registry.wait_for(expected, timeout_s=300)
            if not expected.issubset({int(k) for k in registry.keys()}):
                print(f"  [STRUCTURE] Timeout waiting for all compartments; "
                      f"proceeding with available: {list(registry.keys())}")
        '''
        expected = {1, 2, 3}
        timeout_s = 300
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            registry = self._read_registry()
            if expected.issubset(registry.keys()):
                break
            time.sleep(5)
        else:
            print(f"  [STRUCTURE] Timeout waiting for all compartments; "
                  f"proceeding with available: {list(registry.keys())}")

        registry = self._read_registry()


        # 3) BIC 比较
        best_n = None
        best_bic = float('inf')
        qualified = 0

        for n, entry in registry.items():
            ok, reason = self._passes_quality_gate(entry)
            if not ok:
                print(f"  [STRUCTURE-GATE] {n}cmt disqualified: {reason}")
                continue
            qualified += 1
            bic = entry['ofv'] + entry['n_params'] * math.log(max(n_obs, 2))
            print(f"  [STRUCTURE] {n}cmt: OFV={entry['ofv']:.2f}, "
                  f"n_params={entry['n_params']}, BIC={bic:.2f}")
            if bic < best_bic:
                best_bic = bic
                best_n = n

        if best_n is None:
            # 没有房室通过质量门 → 不做 BIC 决策，保守降级
            print(f"  [STRUCTURE] No compartment passed quality gate — "
                  f"continuing Phase 5 conservatively")
            return SkillResult(done=True, updates={'structure_selected': True})

        # 4) 若非最优 → 优雅退出
        if best_n != n_cmt:
            print(f"  [STRUCTURE] This branch ({n_cmt}cmt) lost to {best_n}cmt "
                  f"— stopping Phase 5 for this branch")
            state.force_stop = True
            state.force_stop_reason = (
                f"structure_selection: {best_n}cmt wins (BIC={best_bic:.2f})")
            return SkillResult(done=True, updates={
                'force_stop': True,
                'force_stop_reason': state.force_stop_reason,
            })

        # 5) 最优 → 继续 Phase 5
        print(f"  [STRUCTURE] This branch ({n_cmt}cmt) wins — proceeding to SCM")
        return SkillResult(done=True, updates={'structure_selected': True})

    # ---- 辅助 ---- #
    def _count_parameters(self, code: str) -> int:
        """统计 THETA + OMEGA(diag) + SIGMA 数量（不含 FIX）。"""
        import re
        if not code:
            return 0
        n_theta = len(re.findall(r'THETA\s*\(\s*\d+\s*\)', code))
        # OMEGA 数字数
        omega_m = re.search(r'\$OMEGA[^\$]*', code, re.IGNORECASE | re.DOTALL)
        n_omega = 0
        if omega_m:
            for line in omega_m.group(0).split('\n'):
                line = line.split(';')[0].strip()
                if line and not line.upper().startswith('$OMEGA'):
                    n_omega += len(re.findall(r'[-+]?\d+\.?\d*', line))
        sigma_m = re.search(r'\$SIGMA[^\$]*', code, re.IGNORECASE | re.DOTALL)
        n_sigma = 0
        if sigma_m:
            for line in sigma_m.group(0).split('\n'):
                line = line.split(';')[0].strip()
                if line and not line.upper().startswith('$SIGMA'):
                    n_sigma += len(re.findall(r'[-+]?\d+\.?\d*', line))
        return n_theta + n_omega + n_sigma

    def _passes_quality_gate(entry: dict) -> tuple:
        """
        房室进入 BIC 比较前的质量门。

        返回 (ok, reason)。ok=False 时该房室不参与 BIC。
        """
        # 1) 协方差必须成功（Hessian 可逆）
        if not entry.get('covariance_successful', False):
            return False, "covariance step failed"

        # 2) 收缩率 < 80%（避免拿 OMEGA 塌缩模型做 BIC）
        shrink = entry.get('avg_eta_shrinkage')
        if shrink is not None and shrink > 80:
            return False, f"ETA shrinkage {shrink:.1f}% > 80%"

        # 3) OFV 有效
        ofv = entry.get('ofv')
        if ofv is None or ofv > 1e4 or ofv < -1e5:
            return False, f"OFV={ofv} is invalid (overflow/negative)"

        # 4) 至少一次最小化成功
        if not entry.get('minimization_successful', False):
            return False, "minimization never succeeded"

        return True, "ok"

    def _write_registry(self, n_cmt, ofv, n_params, best_code):
        path = self.REGISTRY_PATH
        if not path:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        registry = self._read_registry()
        registry[str(n_cmt)] = {
            'n_cmt': n_cmt,
            'ofv': ofv,
            'n_params': n_params,
            'best_code_path': None,   # 由外层调度器补充
            'ts': time.time(),
        }
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp, path)

    def _read_registry(self) -> dict:
        path = self.REGISTRY_PATH
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
