"""
skills/phase3_reduce.py

Phase 3: 过拟合控制。

触发条件：shrinkage > 95% 或 OFV < -50
退出条件：shrinkage < 90% 或 OFV > -50

职责：
1. 移除不支持随机效应
2. OMEGA block → diagonal
3. 简化误差模型
4. 固定边界参数
5. 至少保留 1 个 OMEGA（防止非群体模型）
"""

from .base import Skill, SkillResult, CompartmentLockError
from modules.prompts.phase3_reduce import Phase3Reduce
from utils.nonmem_utils import extract_code as _extract_code, extract_theta_param_map
from utils.nonmem_utils import _count_omega
import re

class ReduceOverfittingSkill(Skill):
    name = 'phase3_reduce'
    phase = [3]
    tools = ['cs_editor', 'nonmem_runner', 'composite_scorer']
    prompt_template = 'phase3_reduce'

    def run(self, state, llm, tools) -> SkillResult:

        parsed = state.last_run.get('parsed_data', {})
        omega_count = _count_omega(state.current_code)

        # ★ THETA 撞边界检测（不只是 OMEGA）
        boundary_thetas = _identify_boundary_theta(
            state.last_run.get('full_output', ''))
        boundary_param_names = _theta_param_names(state.current_code, boundary_thetas)

        # ★ 关键策略 1：结构性参数撞边界 → 检查是否需要 ALAG
        STRUCTURAL_MISSING_ALAG = (
                state.data_profile.get('route', 'oral').lower() == 'oral'
                and 'ALAG' not in state.current_code.upper()
                and ('KA' in boundary_param_names or 'Ka' in boundary_param_names)
                and len(boundary_param_names) >= 2  # 至少 2 个结构参数撞边界
        )
        if STRUCTURAL_MISSING_ALAG:
            print(f"  [PHASE3-STRUCT] Ka + other structural params at boundary "
                  f"with NO ALAG — recommending ALAG addition")

            # 更新标志，让 Phase 3 的 prompt 包含 ALAG 要求
            state.phase3_alag_recommended = True
            state.phase3_boundary_params = boundary_param_names

        # 独立判定 2：OMEGA 塌缩 → 建议简化误差模型
        max_shrink = max((s['shrinkage'] for s in parsed.get('eta_shrinkage', [])),
                         default=None)
        OMEGA_COLLAPSE = (
                omega_count == 1
                and max_shrink is not None and max_shrink > 90
        )
        if OMEGA_COLLAPSE:
            print(f"  [PHASE3-OMEGA] OMEGA=1 with shrinkage {max_shrink:.1f}% — "
                  f"model is over-parameterized. Will attempt error model "
                  f"simplification (NOT compartment reduction).")
            state.phase3_error_model_simplify = True

        prompt = Phase3Reduce.generate_prompt(
            iteration=state.iteration,
            current_code=state.current_code,
            parsed_results=state.last_run.get('parsed_data', {}),
            shrinkage_data=state.last_run.get('shrinkage', []),
            current_omega_count=_count_omega(state.current_code),
            n_subjects=state.data_profile['n_subjects'],
        )
        prompt = self._prepend_warning(prompt, state)  # ★

        # 注入 ALAG 提示
        if state.phase3_alag_recommended:
            prompt += (
                f"\n\n{'!' * 70}\n"
                f"STRUCTURAL FIX RECOMMENDED: Add ALAG1 to $PK\n"
                f"{'!' * 70}\n"
                f"The following structural parameters are at their bounds: "
                f"{state.phase3_boundary_params}\n"
                f"This often indicates a missing absorption lag time (ALAG1).\n"
                f"Add ALAG1 = THETA(n) * EXP(ETA(k)) in $PK (range: 0, 0.5, 2 h).\n"
                f"Keep the model at {state.locked_compartments}-compartment.\n"
                f"{'!' * 70}\n"
            )

        response = llm.generate(prompt, model_type=state.model)
        new_code = _extract_code(response)

        # ★ 强制房室锁
        try:
            new_code = self._enforce_compartment_lock(
                state, new_code, llm, prompt)
        except CompartmentLockError as e:
            print(f"  [PHASE3-LOCK-FAIL] {e}")
            return self.fallback(state, str(e))

        # 防止误删最后一个 OMEGA
        if _count_omega(new_code) < 1:
            return self.fallback(state, "Phase 3 would remove the last OMEGA — refused.")

        return SkillResult(updates={'current_code': new_code})

    def fallback(self, state, error: str) -> SkillResult:
        return SkillResult(ok=False, error=error,
                           updates={'current_code': state.current_code})


def _identify_boundary_theta(lst_output: str) -> list:
    """从 .lst 的 GRADIENT 行提取触边界的 THETA 索引。"""
    import re
    lines = re.findall(
        r'GRADIENT:\s+((?:[+-]?\s*\d+\.\d+E[+-]\d+\s*)+)',
        lst_output, re.IGNORECASE)
    if not lines:
        return []
    values = re.findall(r'[+-]?\d+\.\d+E[+-]\d+', lines[-1])
    return [i for i, v in enumerate(values, start=1) if float(v) == 0.0]


def _theta_param_names(code: str, indices: list) -> list:
    """将 THETA 索引映射为参数名（Ka/CL/V1/V2/Q 等）。"""
    mapping = extract_theta_param_map(code)
    return [mapping.get(i, f'THETA({i})') for i in indices]

    '''
    mapping = {}
    patterns = [
        (r'TVKA?\s*=\s*THETA\((\d+)\)', 'Ka'),
        (r'TVCL\s*=\s*THETA\((\d+)\)', 'CL'),
        (r'TVV1?\s*=\s*THETA\((\d+)\)', 'V1'),
        (r'TVV2\s*=\s*THETA\((\d+)\)', 'V2'),
        (r'TVQ2?\s*=\s*THETA\((\d+)\)', 'Q'),
    ]
    for pat, name in patterns:
        for m in re.finditer(pat, code, re.IGNORECASE):
            mapping[int(m.group(1))] = name
    return [mapping.get(i, f'THETA({i})') for i in indices]
    '''

'''
def _count_omega(code: str) -> int:
    """统计 $OMEGA 块中实际的 OMEGA 数量（BLOCK 形式按其维度计数）"""
    m = re.search(r'\$OMEGA\s*(.*?)(?=\n\s*\$|\Z)', code, re.DOTALL | re.IGNORECASE)
    if not m:
        return 0
    block_body = m.group(1)

    # 检测 BLOCK(n) 形式
    block_m = re.search(r'BLOCK\s*\(\s*(\d+)\s*\)', block_body, re.IGNORECASE)
    if block_m:
        return int(block_m.group(1))

    # DIAGONAL / 默认形式：逐行统计数值
    count = 0
    for line in block_body.split('\n'):
        line = line.split(';')[0].strip()
        if not line:
            continue
        # 匹配 FIX、SAME 等关键字后的数值
        nums = re.findall(r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', line)
        count += len(nums)
    return count
'''
'''
block = re.search(r'\$OMEGA\s*(.*?)(?=\n\s*\$|\Z)', code, re.DOTALL | re.IGNORECASE)
if not block:
    return 0
count = 0
for line in block.group(1).split('\n'):
    line = line.split(';')[0].strip()
    if line:
        count += len(re.findall(r'\d+\.?\d*(?:[eE][+-]?\d+)?', line))
return count
'''
