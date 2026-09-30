"""
agents/orchestrator.py

OrchestratorAgent：PKGPT 2.0 主循环状态机。

协调 Bootstrap → Phase 1-5 → Termination 的完整流程。
本文件不直接调用 NONMEM 或 LLM；所有工具调用经由 ToolRegistry / SkillRegistry。
"""

from __future__ import annotations
from typing import Optional, List
import re
from skills import build_skill_registry
from utils.nonmem_utils import extract_theta_param_map
from .skill_selector import SkillSelector
from .human_gateway import HumanGateway
from modules.optimizer import ModelPhase
from utils.safe_numeric import safe_float, extract_matrix_diagonal_values
from utils.parameter_guidance import build_parameter_stabilization_guidance
import os
from utils.cov_diagnosis import (
    diagnose_covariance_failure as _diagnose_cov_failure,
    extract_theta_param_map as _extract_theta_map_util, identify_boundary_theta
)
from skills.phase4_optimize_iiv import _collapse_omega_to_cl_only, _count_omega
from utils.shrinkage_threshold import get_shrinkage_threshold
from utils.nonmem_utils import _count_omega


# --------------------------------------------------------------------------- #
# GuardrailCritic：在每次代码生成后执行结构 / 语法守卫
# --------------------------------------------------------------------------- #

class GuardrailCritic:
    """
    在 Phase 1-4 的每轮代码生成后执行结构守卫。

    调用顺序（一旦失败即回退到 best model）：
      1. compartment_invariance  → 房室数量不得变化
      2. advan_trans_validity    → ADVAN/TRANS 组合合法性

    Phase 5 的结构冻结（phase5_structure_guard）由 CovariateSCMSkill 内部调用。
    Phase 5 的数值安全（numeric_safety_gate）由 SCM winner 判定时调用。
    """

    @staticmethod
    def check(state, tools):
        # Phase 5 由 Skill 内部处理
        if state.phase == 5:
            return state

        new_code = state.current_code
        if not new_code:
            return state

        # ---- 1) 房室不变性 -------------------------------------------- #
        parsed = state.last_run.get('parsed_data') or {}
        prev_ofv = parsed.get('objective_function')
        # prev_ofv = state.last_run.get('parsed_data', {}).get('objective_function')
        guard = tools['compartment_guard']
        msg = guard.check(new_code, prev_ofv, state.locked_compartments)

        if msg:
            print(f"  [GUARD] {msg}")
            base = state.best_code
            if base is None:
                # best_code 不存在（Phase 1 首轮）→ 用 iter0
                import os
                iter0_path = f"{state.output_base}_iter0.txt"
                if os.path.exists(iter0_path):
                    with open(iter0_path, 'r', encoding='utf-8') as f:
                        base = f.read()
            if base:
                state.current_code = guard.apply(
                    new_code, base, state.locked_compartments)
                # 若回退后的 code 仍违规（说明 base 本身违规），走失败路径
                from utils.compartment_lock import validate_lock
                ok, actual, advan = validate_lock(
                    state.current_code, state.locked_compartments)
                if not ok:
                    print(f"  [GUARD-CRITICAL] base_code 本身违规: "
                          f"ADVAN{advan} ({actual}-cmt). 强制停止。")
                    state.force_stop = True
                    state.force_stop_reason = (
                        f"base_code violates compartment lock "
                        f"({actual}-cmt vs locked {state.locked_compartments}-cmt). "
                        f"Run cannot recover.")
                    return state

        '''
        if msg:
            print(f"  [GUARD] {msg}")
            if state.best_code:
                # ★ 部分恢复：只替换 $SUBROUTINES，保留 $PK / $THETA
                new_code = guard.apply(
                    new_code, state.best_code, state.locked_compartments)
                state.current_code = new_code
                # 部分恢复后重新检查，避免"恢复不彻底"
                msg2 = guard.check(new_code, prev_ofv, state.locked_compartments)
                if msg2:
                    # apply() 未能修复 → 整体回退
                    print(f"  [GUARD] apply() 未能修复，整体回退到 best_code")
                    state.current_code = state.best_code
                    state.last_revert_info = {
                        'reason': 'compartment_invariance_violation',
                        'message': msg,
                        'reverted_to_iteration': state.best_iteration,
                        'recovery': 'full_revert',
                    }
                else:
                    state.last_revert_info = {
                        'reason': 'compartment_invariance_violation',
                        'message': msg,
                        'recovery': 'partial_apply_subroutines_only',
                    }
        '''
        '''
        state.current_code = state.best_code
        state.last_revert_info = {
            'reason':                'compartment_invariance_violation',
            'message':               msg,
            'reverted_to_iteration': state.best_iteration,
        }
        '''

        # ---- 2) ADVAN/TRANS 合法性 ------------------------------------ #
        advan_guard = tools['advan_trans_validity']
        msg = advan_guard.check(new_code)
        if msg:
            print(f"  [GUARD] {msg}")
            if state.best_code:
                new_code = advan_guard.apply(state.current_code, state.best_code)
                state.current_code = new_code
                msg2 = advan_guard.check(new_code)
                if msg2:
                    print(f"  [GUARD] apply() 未能修复，整体回退到 best_code")
                    state.current_code = state.best_code
                    state.last_revert_info = {
                        'reason': 'advan_trans_invalid',
                        'message': msg,
                        'reverted_to_iteration': state.best_iteration,
                        'recovery': 'full_revert',
                    }
                else:
                    state.last_revert_info = {
                        'reason': 'advan_trans_invalid',
                        'message': msg,
                        'recovery': 'partial_apply_subroutines_only',
                    }

                '''
                state.current_code = state.best_code
                state.last_revert_info = {
                    'reason':                'advan_trans_invalid',
                    'message':               msg,
                    'reverted_to_iteration': state.best_iteration,
                }
                '''
        # return state

        # ★ 3) 协方差失败诊断（新增）
        # GuardrailCritic._diagnose_covariance_failure(state)
        _diagnose_cov_failure(state)
        return state
    

    @staticmethod
    def _diagnose_covariance_failure(state):
        """
        当协方差步失败时，识别具体的失败原因并写入 state，
        供下一轮 prompt 生成使用。

        覆盖三类根因（按优先级）：
          A. STRUCTURAL_BOUNDARY: 结构参数（CL/V/Ka/Q/V2/V3）触边界
          B. ERROR_MODEL_BOUNDARY: 误差模型参数（Additive）触下界
          C. OMEGA_COLLAPSE: 某个 ETA 方差塌缩为 0
        """
        # parsed = state.last_run.get('parsed_data', {})
        # cov = parsed.get('covariance_step', {})
        parsed = state.last_run.get('parsed_data') or {}
        cov = parsed.get('covariance_step') or {}

        if not cov.get('attempted') or cov.get('successful'):
            state.cov_failure_diagnosis = None
            return

        # 读 .lst 全文，用于识别 boundary 参数
        lst_output = state.last_run.get('full_output', '')
        boundary_indices = GuardrailCritic._identify_boundary_theta(lst_output)
        theta_map = GuardrailCritic._extract_theta_param_map(state.current_code)

        STRUCTURAL_PK = {'CL', 'V1', 'V2', 'V3', 'V', 'Q', 'Q2', 'Q3', 'Q4', 'KA', 'Ka'}
        boundary_params = [
            (idx, theta_map.get(idx, f"THETA({idx})"))
            for idx in boundary_indices
        ]

        diagnosis = None
        if boundary_params:
            structural_hits = [(i, n) for i, n in boundary_params if n in STRUCTURAL_PK]
            if structural_hits:
                idx_str = ", ".join(f"THETA({i})" for i, _ in structural_hits)
                names = ", ".join(n for _, n in structural_hits)
                diagnosis = {
                    'type': 'STRUCTURAL_BOUNDARY',
                    'params': [i for i, _ in structural_hits],
                    'names': names,
                    'message': (
                        f"STRUCTURAL_BOUNDARY_DETECTED: Structural PK parameter(s) "
                        f"({names}) [{idx_str}] converged to their bounds, causing "
                        f"Hessian singularity and covariance failure. "
                        f"MANDATORY: Widen the lower/upper bounds for these THETA(s) "
                        f"(e.g., change (0.1, 1.0, 10) to (0.001, 1.0, 10)), OR "
                        f"re-initialize them closer to the previous best estimates. "
                        f"Do NOT change the error model."
                    ),
                }
            else:
                diagnosis = {
                    'type': 'ERROR_MODEL_BOUNDARY',
                    'params': [i for i, _ in boundary_params],
                    'names': ", ".join(n for _, n in boundary_params),
                    'message': (
                        f"ERROR_MODEL_BOUNDARY_DETECTED: Error model THETA "
                        f"({', '.join(n for _, n in boundary_params)}) hit its lower "
                        f"boundary, making the Hessian singular. "
                        f"MANDATORY FIX: Remove the additive error THETA and use "
                        f"proportional-only error ($ERROR: W = THETA(n)*IPRED). "
                        f"Do NOT change the structural model."
                    ),
                }

        # OMEGA collapse 检查（若 boundary 未命中）
        if diagnosis is None:
            omega_values = [
                o.get('value') for o in
                (parsed.get('parameter_estimates') or {}).get('omega', [])
                if isinstance(o, dict) and o.get('value') is not None
            ]
            collapsed = [v for v in omega_values if v < 0.001]
            if collapsed and len(collapsed) == len(omega_values):
                diagnosis = {
                    'type': 'OMEGA_COLLAPSE',
                    'message': (
                        f"OMEGA_COLLAPSE_DETECTED: All {len(omega_values)} OMEGA "
                        f"variances have collapsed below 0.001. The random-effect "
                        f"structure is over-parameterized for this dataset. "
                        f"MANDATORY: Remove the OMEGA with the highest ETA shrinkage, "
                        f"or switch from BLOCK to DIAGONAL OMEGA structure."
                    ),
                }

        state.cov_failure_diagnosis = diagnosis
        if diagnosis:
            print(f"  [COV-DIAGNOSIS] {diagnosis['type']}: {diagnosis['message'][:120]}...")

    @staticmethod
    def _identify_boundary_theta(lst_output):
        # return identify_boundary_theta_util(lst_output)
        return identify_boundary_theta(lst_output)
        '''
        """从 .lst 的 GRADIENT 行中提取 gradient=0 的 THETA 索引（即触边界参数）。"""
        if not lst_output:
            return []
        lines = re.findall(r'GRADIENT:\s+((?:[+-]?\s*\d+\.\d+E[+-]\d+\s*)+)',
                           lst_output, re.IGNORECASE)
        if not lines:
            return []
        values = re.findall(r'[+-]?\d+\.\d+E[+-]\d+', lines[-1])
        return [i for i, v in enumerate(values, start=1) if float(v) == 0.0]
        '''

    @staticmethod
    def _extract_theta_param_map(code):
        return _extract_theta_map_util(code)
        # return extract_theta_param_map(code)
        '''
        """THETA 索引 → 参数名映射。"""
        import re
        mapping = {}
        for pattern, name in [
            (r'TVCL\s*=\s*THETA\((\d+)\)', 'CL'),
            (r'TVV1?\s*=\s*THETA\((\d+)\)', 'V1'),
            (r'TVV2\s*=\s*THETA\((\d+)\)', 'V2'),
            (r'TVV3\s*=\s*THETA\((\d+)\)', 'V3'),
            (r'TVQ2?\s*=\s*THETA\((\d+)\)', 'Q'),
            (r'TVQ3\s*=\s*THETA\((\d+)\)', 'Q3'),
            (r'TVQ4\s*=\s*THETA\((\d+)\)', 'Q4'),
            (r'TVKA?\s*=\s*THETA\((\d+)\)', 'Ka'),
        ]:
            m = re.search(pattern, code, re.IGNORECASE)
            if m:
                mapping.setdefault(int(m.group(1)), name)
        return mapping
        '''

# --------------------------------------------------------------------------- #
# StateReducer：将 SkillResult.updates 合并到 State
# --------------------------------------------------------------------------- #

class StateReducer:
    """
    将 SkillResult.updates 写入 OptimizationState。

    规则：
    - 字典字段（last_run / flags / data_profile 等）：浅合并
    - 列表字段（improvement_history 等）：替换
    - SCM 子对象：逐字段更新
    - 其余标量字段：直接赋值
    """

    _MERGE_DICT_FIELDS = {
        'last_run', 'flags', 'scm_state', 'data_profile',
        'plausibility_bounds', 'plausibility_report',
        'strategy_repeat_count', 'prior_info',
        'theta_widen_count',
    }

    @staticmethod
    def apply(state, result):
        if result is None:
            return state
        for key, value in (result.updates or {}).items():
            current_phase = getattr(state, 'phase', 1)
            if isinstance(value, int) and value < current_phase:
                print(f"  [StateReducer] BLOCKED backward phase transition: "
                      f"{current_phase} → {value} (forward-only constraint)")
                continue    # 跳过，不写入
            if key in StateReducer._MERGE_DICT_FIELDS and isinstance(value, dict):
                existing = getattr(state, key, None) or {}
                setattr(state, key, {**existing, **value})
            else:
                setattr(state, key, value)
        return state


# --------------------------------------------------------------------------- #
# OrchestratorAgent：主循环状态机
# --------------------------------------------------------------------------- #

class OrchestratorAgent:
    """PKGPT 2.0 主循环状态机。"""

    def __init__(self, state, skills: dict = None, tools=None):
        self.state = state
        self.tools = tools
        self.skills = skills or build_skill_registry(tools)
        self.human = HumanGateway()

    # ── 主入口 ───────────────────────────────────────────────────────── #

    def run(self) -> dict:
        s = self.state

        print("=" * 70)
        print("PKGPT 2.0 — Agent Orchestrator")
        print("=" * 70)
        print(f"Phase locked to {s.locked_compartments}-compartment "
              f"(--compartments / data-driven hint)")
        print(f"Model: {s.model}")
        print(f"Max iterations (Phase 1-4): {s.max_iterations}")
        print("=" * 70)

        # ---- 1) Bootstrap -------------------------------------------- #
        s = self._run_bootstrap(s)

        # ---- 2) 生成初始代码（Phase 1 首次生成）---------------------- #
        print("\n[Phase 1] Generating initial control stream...")
        result = self.skills['phase1_establish'].run(s, s.llm, self.tools)
        s = StateReducer.apply(s, result)
        if not result.ok:
            print(f"[WARNING] Initial generation failed: {result.error}")

        # ---- 3) 主循环 ---------------------------------------------- #
        s = self._main_loop(s)

        # ---- 4) 最终保存 -------------------------------------------- #
        return self.skills['termination'].finalize(s)

    # ── Bootstrap ────────────────────────────────────────────────────── #

    def _run_bootstrap(self, s):
        for skill_name in ('data_introspection',
                           'plausibility_bounds',
                           'dose_unit_check'):
            print(f"\n[Bootstrap] Running {skill_name}...")
            result = self.skills[skill_name].run(s, s.llm, self.tools)
            s = StateReducer.apply(s, result)
            if not result.ok:
                print(f"[WARNING] {skill_name} failed: {result.error}")
        return s

    # ── 主循环 ───────────────────────────────────────────────────────── #

    def _main_loop(self, s):
        while True:
            if s.force_stop:
                print(f"\n[STOP] {s.force_stop_reason}")
                break
            s.iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION {s.iteration} | PHASE {s.phase} | "
                  f"iter_in_phase={s.iterations_in_phase}")
            print(f"{'='*70}")

            # ---- 终止判断 --------------------------------------------- #
            stop, reason = self.skills['termination'].should_stop(s)
            if stop:
                print(f"\n[STOP] {reason}")
                break

            # ★ Rescue 流程：统一处理两个触发源
            #   触发源 A：TerminationSkill._early_stop 设置 s.rescue_pending
            #   触发源 B：PhaseTransitionManager 设置 phase_manager.rescue_triggered
            #   （B 在 PhaseTransitionSkill.run → determine_next_phase 中被置位，
            #     由于该 Skill 在本轮 _run_phase1_to_4 末尾执行，所以 B 的信号
            #     会在【下一轮】iteration 顶部被检测到，符合"先完成本轮、再 rescue"
            #     的语义）
            if s.phase_manager is not None and \
                    getattr(s.phase_manager, 'rescue_triggered', False):
                s.phase_manager.rescue_triggered = False
                s.rescue_pending = True


            # ★ Rescue 流程：连续 3 次失败时
            if getattr(s, 'rescue_pending', False):
                s.rescue_pending = False
                s = self._apply_rescue_strategy(s)
                continue

            # ---- Phase 5 特殊分支 ------------------------------------ #
            if s.phase == 5:

                # ★ 新增 1：进入 Phase 5 之前的跨房室结构选择（一次性）
                # if not getattr(s, '_structure_selected', False):
                #     s = self._run_structure_selection(s)
                #     if s.force_stop:
                #         print(f"\n[STOP] {s.force_stop_reason}")
                #         break

                s = self._run_phase5(s)
                if s.scm_complete:
                    print("\n[SCM] Phase 5 complete")
                    break
                # Human Review 在 SCM 结束后统一处理
                if self.skills['human_review'].required(s):
                    decision = self.human.pause(s)
                    if decision == 'abort':
                        print("\n[HUMAN] Abort requested — exiting main loop")
                        s.flags['abort'] = True
                        s.force_stop = True
                        if not s.force_stop_reason:
                            s.force_stop_reason = "aborted via human gateway"
                        break
                    elif decision == 'revert' and s.best_code:
                        s.current_code = s.best_code
                continue

            # ---- Phase 1-4 通用流程 ---------------------------------- #
            s = self._run_phase1_to_4(s)

            # ---- 终止信号 ------------------------------------------- #
            if s.flags.get('stop') or s.flags.get('abort'):
                break

            # ---- Human Review ---------------------------------------- #
            if self.skills['human_review'].required(s):
                decision = self.human.pause(s)
                if decision == 'abort':
                    print("\n[HUMAN] Abort requested — exiting main loop")
                    s.flags['abort'] = True
                    s.force_stop = True
                    if not s.force_stop_reason:
                        s.force_stop_reason = "aborted via human gateway"
                    break
                elif decision == 'revert':
                    if s.best_code:
                        s.current_code = s.best_code
                        s.last_revert_info = {
                            'reason':                'human_revert',
                            'reverted_to_iteration': s.best_iteration,
                        }
                # decision == 'continue' → 继续循环

        return s

    def _apply_rescue_strategy(self, s):
        """
    连续失败后的救援策略（不改变 Phase，符合单向阶段约束）：
    1. 回退到 best_code（若无，用 iter0.txt）
    2. 放宽所有 THETA 边界（×10 下限，×10 上限）
    3. OMEGA 从 BLOCK 改为 DIAGONAL
    4. $ERROR 从 combined 改为 proportional-only
    5. 保持当前 Phase 号不变，让救援后的代码在**当前 Phase**内继续迭代
        """
        print(f"\n{'!' * 70}")
        print("RESCUE STRATEGY TRIGGERED — 3+ consecutive failures")
        print(f"{'!' * 70}")

        iter0 = f"{s.output_base}_iter0.txt"
        if os.path.exists(iter0):
            try:
                with open(iter0, 'r', encoding='utf-8') as f:
                    s.best_code = f.read()
                s.best_iteration = 0
                print(f"  [RESCUE] No best_code — falling back to iter0.txt")
            except Exception as e:
                print(f"  [RESCUE] Failed to read iter0.txt: {e}")


        if not s.best_code:
            s.force_stop = True
            s.force_stop_reason = ("rescue failed: no best_code and no iter0.txt "
                                   "— Phase 1 never produced a runnable model")
            return s

        code = s.best_code

        # ★ 步骤 1：oral 给药时强制加 ALAG（如果还没有）
        route = s.data_profile.get('route', 'oral').lower()
        if route == 'oral' and 'ALAG' not in code.upper():
            print(f"  [RESCUE] Adding ALAG1 (oral route, missing)")
            # 在 $PK 末尾插入 ALAG1
            pk_match = re.search(
                r'(\$PK[\s\S]*?)(?=\n\s*\$[A-Z]|\Z)', code, re.IGNORECASE)
            if pk_match:
                pk_block = pk_match.group(1)
                new_pk = pk_block.rstrip() + \
                         '\nALAG1 = THETA(98) * EXP(ETA(98))  ; rescue: ALAG for oral'
                code = code.replace(pk_block, new_pk, 1)
                # $THETA 追加
                theta_match = re.search(
                    r'(\$THETA[\s\S]*?)(?=\n\s*\$[A-Z]|\Z)', code, re.IGNORECASE)
                if theta_match:
                    new_theta = theta_match.group(1).rstrip() + \
                                '\n(0, 0.5, 2)   ; 98 ALAG1 (h)'
                    code = code.replace(theta_match.group(1), new_theta, 1)
                # $OMEGA 追加对应 ETA
                omega_match = re.search(
                    r'(\$OMEGA[\s\S]*?)(?=\n\s*\$[A-Z]|\Z)', code, re.IGNORECASE)
                if omega_match:
                    new_omega = omega_match.group(1).rstrip() + '\n0.1   ; ETA(98) ALAG'
                    code = code.replace(omega_match.group(1), new_omega, 1)

        # ★ 步骤 2：oral 给药时加 WT allometric（如果还没有）
        if route == 'oral' and 'WT' in s.data_profile.get('columns', []):
            if '/WT' not in code and '/ 70' not in code and '/70' not in code:
                print(f"  [RESCUE] Adding WT allometric scaling (CL, V)")
                code = re.sub(
                    r'TVCL\s*=\s*THETA\((\d+)\)',
                    r'TVCL = THETA(\1) * (WT/70)**0.75',
                    code, count=1, flags=re.IGNORECASE)
                code = re.sub(
                    r'TVV(1?)\s*=\s*THETA\((\d+)\)',
                    r'TVV\1 = THETA(\2) * (WT/70)**1.0',
                    code, count=1, flags=re.IGNORECASE)

        # 1) 放宽所有 THETA 边界
        def _widen_theta(m):
            lo, init, hi = m.group(1), m.group(2), m.group(3)
            try:
                lo_f = float(lo);
                hi_f = float(hi)
                new_lo = lo_f * 0.1 if lo_f > 0 else lo_f * 10
                new_hi = hi_f * 10
                return f"({new_lo:.6g}, {init}, {new_hi:.6g})"
            except ValueError:
                return m.group(0)

        code = re.sub(
            r'\(([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?),\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?),\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\)',
            _widen_theta, code)

        # 2) BLOCK → DIAGONAL
        code = re.sub(r'\$OMEGA\s+BLOCK\s*\(\s*\d+\s*\)',
                      '$OMEGA', code, flags=re.IGNORECASE)

        # 3) $ERROR 简化
        code = re.sub(
            r'W\s*=\s*SQRT\s*\(\s*THETA\s*\(\s*\d+\s*\)\s*\*\*\s*2\s*\+\s*\(\s*THETA\s*\(\s*(\d+)\s*\)\s*\*\s*IPRED\s*\)\s*\*\*\s*2\s*\)',
            r'W = THETA(\1) * IPRED',
            code, flags=re.IGNORECASE)

        if getattr(s, 'phase3_alag_recommended', False):
            print(f"  [RESCUE] ALAG will be injected in Phase 3 prompt")
            # 不重置 phase，让 Phase 3 Skill 在下一轮迭代中通过 prompt 注入 ALAG
            # 标志位保留，Phase 3 Skill 会在生成 prompt 时读取并注入

        # 4) 写回 state
        s.current_code = code
        s.failed_strategies.append("RESCUE_WIDEN_BOUNDS_OMEGA_DIAG_ERROR_PROP")
        print(f"  [RESCUE] Applied: widen bounds + OMEGA diag + proportional error")
        print(f"  [RESCUE] Phase {s.phase} retained (forward-only); code rescale "
              f"applied to next iteration")
        # print(f"  [RESCUE] Re-entering Phase 1 with rescaled code")
        return s




    # ── Phase 1-4 通用流程 ───────────────────────────────────────────── #

    def _should_revert_to_best(self, s) -> bool:
        """基于 composite_score / OFV 判断是否回退到 best model。"""
        if not s.best_code or len(s.improvement_history) < 2:
            return False
        if s.iteration - s.best_iteration < 2:
            return False

        last = s.improvement_history[-1]
        current_composite = last.get('composite_score', float('inf'))
        current_ofv = last.get('ofv')

        # 条件1：composite 2x 差
        if s.best_composite != float('inf') and current_composite > s.best_composite * 2:
            print(f"  [REVERT] composite {current_composite:.0f} > 2x best {s.best_composite:.0f}")
            return True
        # 条件2：OFV 2x 差
        if s.best_ofv is not None and current_ofv is not None and current_ofv > s.best_ofv * 2:
            print(f"  [REVERT] OFV {current_ofv:.1f} > 2x best {s.best_ofv:.1f}")
            return True
        # 条件3：连续 3 次恶化
        if len(s.improvement_history) >= 3:
            recent = [h.get('composite_score', float('inf'))
                      for h in s.improvement_history[-3:]]
            if (recent[0] < recent[1] < recent[2] and
                    recent[2] > s.best_composite * 1.5):
                print(f"  [REVERT] 连续 3 次恶化 ({recent})")
                return True
        return False

    def _build_failed_strategies_warning(self, s) -> str:
        if not s.failed_strategies:
            s.failed_strategies_warning = ""
            return ""
        lines = [
            "=" * 70,
            "FAILED STRATEGIES - DO NOT REPEAT THESE",
            "=" * 70,
            "The following configurations have been tried and FAILED:",
        ]
        for sig in s.failed_strategies[-5:]:
            lines.append(f"  ✗ {sig}")
        lines += [
            "",
            "IMPORTANT: Do NOT generate code matching these patterns.",
            "=" * 70,
        ]

        # 针对 ERROR_MODEL 失败追加专门提示
        if any('ERROR_MODEL' in sig for sig in s.failed_strategies):
            failed = next((sig for sig in s.failed_strategies if 'ERROR_MODEL' in sig), '')
            lines += [
                "",
                "⚠️  RESIDUAL ERROR MODEL FAILURE DETECTED:",
                f"   The current error model structure ({failed}) caused",
                "   repeated boundary/covariance failures in this dataset.",
                "   Try a DIFFERENT error model structure:",
            ]
            if 'COMBINED' in failed:
                lines.append("   → Switch to proportional-only: W = THETA(n)*IPRED, $SIGMA 1 FIX")
            else:
                lines.append("   → Consider combined error: W = SQRT(THETA(a)**2 + (THETA(b)*IPRED)**2)")

        lines.append("=" * 70)

        warning = "\n".join(lines)
        s.failed_strategies_warning = warning
        return warning
        # return "\n".join(lines)

    def _run_phase1_to_4(self, s):
        # 1) 执行 NONMEM
        result = self.skills['nonmem_execution'].run(s, s.llm, self.tools)
        s = StateReducer.apply(s, result)

        # 2) 解析 .lst
        result = self.skills['output_parsing'].run(s, s.llm, self.tools)
        s = StateReducer.apply(s, result)
        if not result.ok:
            print(f"[WARNING] Parsing failed: {result.error}")

        # 3) 生理范围检查
        result = self.skills['plausibility_check'].run(s, s.llm, self.tools)
        s = StateReducer.apply(s, result)

        # 4) 追加迭代历史（必须在阶段推进之前，保证 iteration 编号一致）
        s = self._append_history(s)
        if s.phase != 5 and self._should_revert_to_best(s):
            s.current_code = s.best_code
            s.last_revert_info = {
                'reason': 'composite_ofv_degradation',
                'reverted_to_iteration': s.best_iteration,
            }

        # ★ 4.5) Phase 2 边界数不减少检测（必须在 phase_transition 之前）
        if s.phase == 2:
            lst_output = s.last_run.get('full_output', '') or ''
            boundary_thetas = identify_boundary_theta(lst_output)
            current_boundary_count = len(boundary_thetas)

            if not hasattr(s, '_phase2_boundary_history'):
                s._phase2_boundary_history = []
            s._phase2_boundary_history.append(current_boundary_count)

            print(f"  [PHASE2-TRACK] Boundary THETA count = "
                  f"{current_boundary_count} "
                  f"(history={s._phase2_boundary_history[-3:]})")
            '''
            if len(s._phase2_boundary_history) >= 3:
                recent = s._phase2_boundary_history[-3:]
                if recent[0] >= recent[1] >= recent[2] and recent[2] >= 3:
                    print(f"  [PHASE2-STUCK] Boundary count not decreasing "
                          f"({recent}) — forcing escalation to Phase 4")
                    s.phase = 4
            '''
            if len(s._phase2_boundary_history) >= 5:  # ← 3 → 5，给协变量注入更多机会
                recent = s._phase2_boundary_history[-5:]
                if all(recent[i] >= recent[i + 1] for i in range(4)) and recent[-1] >= 3:
                    print(f"  [PHASE2-STUCK] Boundary not decreasing for 5 iters "
                          f"({recent}) — escalating with OMEGA pre-check")

                    # ★ 升级前先做一次 OMEGA 塌陷检测
                    try:
                        p2_skill = self.skills['phase2_diagnose']

                        max_shrink = self._max_shrinkage(
                            s.last_run.get('parsed_data', {}))
                        if max_shrink is not None and max_shrink > 90:
                            s.current_code = _collapse_omega_to_cl_only(s.current_code)
                            print(f"  [PHASE2-STUCK] OMEGA collapse applied "
                                  f"(shrinkage {max_shrink:.1f}%) before escalation")
                    except Exception as e:
                        print(f"  [PHASE2-STUCK] OMEGA pre-check failed: {e}")

                    # ★ 升级到 Phase 3（而非 Phase 4），保留过拟合控制
                    s.phase = 3
                    s.iterations_in_phase = 0
                    if s.phase_manager is not None:
                        s.phase_manager.current_phase = ModelPhase.REDUCE_OVERFITTING
                        s.phase_manager.iterations_in_phase = 0
                    s._phase2_boundary_history = []

                    s.iterations_in_phase = 0
                    if s.phase_manager is not None:
                        s.phase_manager.current_phase = ModelPhase.OPTIMIZE_IIV
                        s.phase_manager.iterations_in_phase = 0
                    # 复位，避免下一轮又触发
                    s._phase2_boundary_history = []
        else:
            # 离开 Phase 2 时复位
            if hasattr(s, '_phase2_boundary_history'):
                s._phase2_boundary_history = []


        # 5) 阶段推进（前向唯一）
        result = self.skills['phase_transition'].run(s, s.llm, self.tools)
        s = StateReducer.apply(s, result)

        # ★ 5.5) 参数稳定化指导：唯一注入点
        #         放在阶段推进之后，可覆盖"Phase 2→4 推进后立即需要 guidance"的场景
        if s.phase in (1, 3, 4) and s.parameter_history:
            guidance = build_parameter_stabilization_guidance(s.parameter_history)
            if guidance:
                existing = s.last_run.get('full_output') or ''   # ← None → ''
                s.last_run['full_output'] = (
                        existing + "\n\n" + "=" * 70 + "\n" +
                        "AUTO-GENERATED PARAMETER STABILIZATION GUIDANCE\n" +
                        "=" * 70 + "\n" + guidance
                )

        # 6) AI 质量评价（Phase 1-4 通用）
        qe_result = self.skills['quality_evaluation'].run(s, s.llm, self.tools)
        s = StateReducer.apply(s, qe_result)
        s.last_ai_recommendations = qe_result.updates.get(
            'last_ai_recommendations', [])
        s.last_ai_critical_issues = qe_result.updates.get(
            'last_ai_critical_issues', [])

        # 7) AI 判定可停 + 严格复核
        if not qe_result.meta.get('should_continue', True):
            if self._can_stop(s, qe_result):
                print("\n[STOP] AI quality evaluation accepted")
                s.flags['stop'] = True
                return s  # 不再生成下一轮代码

        # 8) 选择并执行阶段 Skill（生成下一轮代码）
        # 8b) 刷新 failed_strategies 警告（已有的逻辑）
        self._build_failed_strategies_warning(s)

        # 8c) 选择并执行阶段 Skill
        skill = SkillSelector.select(s, self.skills)
        print(f"\n[Skill] {skill.name}")
        result = skill.run(s, s.llm, self.tools)
        s = StateReducer.apply(s, result)
        '''
        skill = SkillSelector.select(s, self.skills)
        print(f"\n[Skill] {skill.name}")
        self._build_failed_strategies_warning(s)
        result = skill.run(s, s.llm, self.tools)
        # s = StateReducer.apply(s, result)
        '''

        if getattr(s, 'phase3_alag_recommended', False):
            print(f"\n{'!' * 70}")
            print(f"[PHASE3-ALAG] Structural params at boundary without ALAG")
            print(f"[PHASE3-ALAG] Boundary params: {s.phase3_boundary_params}")
            print(f"[PHASE3-ALAG] ALAG was injected into this iteration's prompt")
            print(f"{'!' * 70}")

            # 重置标志，避免重复触发
            s.phase3_alag_recommended = False
            # 记录失败策略（可选）
            s.failed_strategies.append("PHASE3_ALAG_RECOMMENDED")

        # 9) Guardrail 检查（房室不变性 + ADVAN/TRANS 合法性）
        s = GuardrailCritic.check(s, self.tools)
        # skill = SkillSelector.select(s, self.skills)
        # print(f"\n[Skill] {skill.name}")
        # result = skill.run(s, s.llm, self.tools)

        return s

    '''
    # ── Phase 5 特殊流程 ─────────────────────────────────────────────── #
    def _run_structure_selection(self, s) -> "OptimizationState":
        """
        在 Phase 5 开始前执行一次跨房室结构选择。

        并行架构下的行为：
          - 每个房室进程独立写入共享 registry（JSON）
          - 所有进程等待齐备后做 BIC 比较
          - 非最优房室设 force_stop=True，优雅退出
          - 最优房室设 _structure_selected=True，进入 Phase 5

        串行架构下的行为：
          - registry 中只有本房室条目，直接判定为最优，继续 Phase 5
        """
        skill = self.skills['structure_selection']

        # 注入 registry 路径（与 output_base 同目录）
        if not skill.REGISTRY_PATH:
            skill.REGISTRY_PATH = os.path.join(
                os.path.dirname(s.output_base) or '.',
                '_compartment_bic_registry.json')
            # 重置懒加载的 registry，确保路径生效
            skill._registry = None

        result = skill.run(s, s.llm, self.tools)
        s = StateReducer.apply(s, result)
        s._structure_selected = True

        if not result.ok:
            print(f"[WARNING] structure_selection failed: {result.error}")
        return s
    '''


    def _run_phase5(self, s):
        """
        Phase 5：CovariateSCMSkill 内部完成
        - 生成单协变量代码
        - 执行 NONMEM
        - 解析结果
        - 记录到 scm.round_results
        - revert 到 round base
        - 轮次结束时选 winner / loser
        """
        skill = self.skills['phase5_covariate_scm']
        result = skill.run(s, s.llm, self.tools)
        s = StateReducer.apply(s, result)

        if result.done:
            s.scm_complete = True
            s.scm.complete = True
            return s

        # ★ 防止死循环：若本轮无任何 state 变化，视为 SCM 完成
        #   判据：scm.round_tested 未增加、scm.round 未增加、scm.mode 未变化
        scm = s.scm
        prev_snapshot = getattr(s, '_last_scm_snapshot', None)
        current_snapshot = (
            len(scm.round_tested),
            scm.round,
            scm.mode,
            len(scm.confirmed),
        )
        if prev_snapshot == current_snapshot:
            s._scm_no_progress_count = getattr(s, '_scm_no_progress_count', 0) + 1
            if s._scm_no_progress_count >= 3:
                print(f"\n[SCM] No progress for 3 consecutive iterations — marking complete")
                s.scm_complete = True
                s.scm.complete = True
                return s
        else:
            s._scm_no_progress_count = 0
        s._last_scm_snapshot = current_snapshot

        return s

    # ── 停止条件细化 ─────────────────────────────────────────────────── #

    def _can_stop(self, s, qe_result) -> bool:
        """
        AI 建议停止时，进一步检查：
          1. 最小化必须成功
          2. shrinkage 达到数据量允许的阈值
          3. Phase 5 尚未执行但存在协变量时，强制推进到 Phase 5
          4. 质量分 >= 60
        """
        parsed = s.last_run.get('parsed_data', {})

        # 1) 最小化
        if not parsed.get('minimization_successful', False):
            print("[OVERRIDE] AI stop rejected — minimization failed")
            return False

        # 2) shrinkage
        n_subjects = s.data_profile.get('n_subjects', 100)
        shrink = parsed.get('eta_shrinkage', [])
        max_shrink = max((x['shrinkage'] for x in shrink), default=None)
        threshold = 70 if n_subjects < 20 else 60 if n_subjects < 50 else 50
        if max_shrink is not None and max_shrink > threshold:
            print(f"[OVERRIDE] AI stop rejected — shrinkage "
                  f"{max_shrink:.1f}% > threshold {threshold}% for N={n_subjects}")
            return False

        # 3) Phase 5 前置强制
        # if s.phase < 5 and s.data_profile.get('covariates'):
        if s.phase == 4 and s.data_profile.get('covariates'):
            best_entry = next(
                (e for e in s.improvement_history
                 if e.get('iteration') == s.best_iteration), None)
            if best_entry and best_entry.get('covariance_successful'):
                print("[OVERRIDE] AI stop deferred — advancing to Phase 5 SCM")
                shrink_val = best_entry.get('avg_eta_shrinkage')
                n = s.data_profile.get('n_subjects', 100)
                threshold = 70 if n < 20 else 60 if n < 50 else 50
                if shrink_val is not None and shrink_val < threshold:
                    print("[OVERRIDE] AI stop deferred — advancing to Phase 5 SCM")
                    # ★ 记录 phase_history，保持与实际 phase_manager 一致
                    s.phase_history.append({
                        'iteration': s.iteration,
                        'from_phase': s.phase,
                        'to_phase': 5,
                        'iterations_in_previous_phase': s.iterations_in_phase,
                        'reason': 'can_stop_forced_phase5',
                    })
                    s.phase = 5
                    s.iterations_in_phase = 0
                    # ★ 同步 phase_manager（若已初始化）
                    if s.phase_manager is not None:
                        s.phase_manager.current_phase = ModelPhase.COVARIATE_ANALYSIS
                        s.phase_manager.iterations_in_phase = 0
                    return False
                else:
                    print(f"[OVERRIDE] AI stop rejected — shrinkage "
                          f"{shrink_val}% not within threshold for Phase 5")

                '''
                best_entry = next(
                    (e for e in s.improvement_history
                     if e.get('iteration') == s.best_iteration), None)
                if best_entry and best_entry.get('covariance_successful'):
                    print("[OVERRIDE] AI stop deferred — advancing to Phase 5 SCM")
                    s.phase = 5
                    s.iterations_in_phase = 0
                    return False
                '''

        # 4) 质量分
        if qe_result and 'evaluation' in qe_result.meta:
            score = qe_result.meta['evaluation'].get('score', 0)
            if score < 60:
                print(f"[OVERRIDE] AI stop rejected — quality score {score} < 60")
                return False

        return True



    # ---- 辅助：shrinkage / composite / 数值安全 -------------------------- #

    @staticmethod
    def _max_shrinkage(parsed: dict) -> Optional[float]:
        """
        返回最大 ETA shrinkage (%)。
        - parsed['eta_shrinkage'] 期望是 [{eta: i, shrinkage: x}, ...]
        - 若条目缺 shrinkage 或为 None，则跳过
        - 若列表为空或无有效值，返回 None
        """
        entries = parsed.get('eta_shrinkage') or []
        if not isinstance(entries, list):
            return None
        vals = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            v = safe_float(e.get('shrinkage'))
            if v is not None:
                vals.append(v)
        return max(vals) if vals else None

    @staticmethod
    def _safe_float_list(values) -> List[float]:
        """把任意可迭代对象中的数值安全转为 float 列表，过滤 None / NaN / 非数值。"""
        out = []
        for v in values or []:
            f = safe_float(v)
            if f is not None:
                out.append(f)
        return out

    def _composite_score(self, s, parsed: dict) -> float:
        """
        调用 composite_scorer Tool 计算复合质量分（越低越好）。
        若 Tool 不可用，回退到内联实现（与 tools/composite_scorer.py 行为一致）。
        """
        # 优先走 Tool（保持与 Skill 层一致）
        try:
            scorer = self.tools.get('composite_scorer')
            if scorer is not None:
                return scorer.compute(parsed, s)
        except Exception as e:
            print(f"  [WARNING] composite_scorer tool failed, using fallback: {e}")



        # 内联回退
        ofv = safe_float(parsed.get('objective_function'))
        shrink = self._max_shrinkage(parsed)
        cov_ok = parsed.get('covariance_step', {}).get('successful', False)
        min_ok = parsed.get('minimization_successful', False)
        omega_values = extract_matrix_diagonal_values(
            parsed.get('parameter_estimates', {}).get('omega') or []
        )
        max_rse = safe_float(parsed.get('rse_percent', {}).get('max_rse'))

        score = 0.0
        '''
        # OFV
        if ofv is None:
            score += 100000.0
        elif ofv < -50:
            score += 50000.0
        elif ofv < 0:
            score += 10000.0
        else:
            score += ofv
        '''
        # OFV
        # OFV_OVERFLOW_THRESHOLD = 1e10  # PopPK 数据集的 OFV 都不可能超过 1e10
        OFV_OVERFLOW_THRESHOLD = 1e4    # OFV 溢出检测：PopPK 数据集 OFV 通常 < 1e4
        if ofv is None:
            score += 100000.0
        elif ofv > OFV_OVERFLOW_THRESHOLD:
            # 浮点溢出 / 数值爆炸：视为完全失败
            print(f"  [CRITICAL] OFV overflow ({ofv:.2e} > {OFV_OVERFLOW_THRESHOLD:.0e}): +100000")
            score += 100000.0
        elif ofv < -50:
            score += 50000.0
        elif ofv < 0:
            score += 10000.0
        else:
            score += ofv

        # Shrinkage
        '''
        if shrink is not None:
            if shrink > 95:   score += 20000.0
            elif shrink > 90: score += 10000.0
            elif shrink > 70: score += 2000.0
            elif shrink > 50: score += 500.0
        '''
        '''
        if shrink is not None:
            if shrink > 95:   score += 500.0
            elif shrink > 90: score += 300.0
            elif shrink > 70: score += 100.0
            elif shrink > 50: score += 50.0
        '''
        n_subjects = s.data_profile.get('n_subjects', 100)
        omega_count = _count_omega(s.current_code) if hasattr(
            self, '_count_omega') else 0
        shrink_threshold = get_shrinkage_threshold(n_subjects, omega_count)


        if shrink is not None:
            if shrink > shrink_threshold:
                # 超出阈值越高，惩罚越大（连续）
                score += (shrink - shrink_threshold) * 10.0
            elif shrink > shrink_threshold * 0.7:
                score += (shrink - shrink_threshold * 0.7) * 2.0


        # OMEGA near-zero
        if omega_values:
            collapsed = [o for o in omega_values if o < 0.0001]
            if collapsed:
                score += len(collapsed) * 500.0

        # Covariance
        if not cov_ok:
            if shrink is not None and shrink < 40:
                score += 100.0
            elif shrink is not None and shrink < 60:
                score += 200.0
            else:
                score += 300.0

        # RSE（Phase 5 不惩罚，与旧逻辑一致）
        in_phase5 = (s.phase == 5)
        if not in_phase5 and max_rse is not None and cov_ok:
            if max_rse > 200:
                score += min((max_rse - 200) * 2.0 + 100, 1000.0)
            elif max_rse > 100:
                score += (max_rse - 100) * 1.0

        # Minimization
        if not min_ok:
            score += 2000.0

        return score

    # ── 历史记录 ─────────────────────────────────────────────────────── #
    def _append_history(self, s) -> "OptimizationState":
        """
        将当前迭代结果追加到 improvement_history。

        Phase 1-4 中，如果 composite_score 更低，同时更新 best model。
        Phase 5 由 SCM Skill 显式管理 best。
        """
        parsed = s.last_run.get('parsed_data', {}) or {}
        params = parsed.get('parameter_estimates', {}) or {}
        cov = parsed.get('covariance_step', {})

        shrink_list = parsed.get('eta_shrinkage', [])
        max_shrink = max((x['shrinkage'] for x in shrink_list), default=None)

        entry = {
            'iteration':               s.iteration,
            'status':                  'success'
                                       if parsed.get('minimization_successful')
                                       else 'failed',
            'ofv':                     parsed.get('objective_function'),
            'avg_eta_shrinkage':       max_shrink,
            'covariance_successful':   cov.get('successful', False),
            'minimization_successful': parsed.get('minimization_successful', False),
            'max_rse':                 parsed.get('rse_percent', {}).get('max_rse'),
            'omega_values': [
                float(o.get('value', 1.0))
                for o in (parsed.get('parameter_estimates', {}) or {}).get('omega', [])
                if 'value' in o
            ],
            'n_subjects':              s.data_profile.get('n_subjects'),
            'composite_score':         s.last_run.get('composite_score', float('inf')),
        }

        omega_values = extract_matrix_diagonal_values(params.get('omega') or [])

        # s.improvement_history = s.improvement_history + [entry]
        '''
        s.improvement_history.append({
            'iteration': s.iteration,
            'ofv': safe_float(parsed.get('objective_function')),
            'omega_values': omega_values,
            'avg_eta_shrinkage': self._max_shrinkage(parsed),
            'composite_score': self._composite_score(s, parsed),
            'covariance_successful': parsed.get('covariance_step', {}).get('successful', False),
            'minimization_successful': parsed.get('minimization_successful', False),
        })
        '''

        ofv = safe_float(parsed.get('objective_function'))
        shrink = self._max_shrinkage(parsed)
        cov_ok = parsed.get('covariance_step', {}).get('successful', False)
        min_ok = parsed.get('minimization_successful', False)
        max_rse = safe_float(parsed.get('rse_percent', {}).get('max_rse'))
        omega_values = extract_matrix_diagonal_values(
            (parsed.get('parameter_estimates') or {}).get('omega') or []
        )
        # ★ 修复 1：composite_score 必须取实际计算值
        composite = self._composite_score(s, parsed)
        s.improvement_history.append({
            'iteration': s.iteration,
            'status': 'success' if min_ok else 'failed',
            'ofv': ofv,
            'max_rse': max_rse,
            'high_rse_count': (parsed.get('rse_percent') or {}).get('high_rse_count', 0),
            'avg_eta_shrinkage': shrink,
            'issues': s.last_run.get('issues', []),
            'minimization_successful': min_ok,
            'covariance_successful': cov_ok,
            'composite_score': composite,
            'omega_values': omega_values,
            'n_subjects': s.data_profile.get('n_subjects'),
        })
        '''
        s.improvement_history.append({
            'iteration': s.iteration,
            'status': 'success' if min_ok else 'failed',
            'ofv': ofv,
            'omega_values': omega_values,
            'avg_eta_shrinkage': shrink,
            'max_rse': max_rse,
            'covariance_successful': cov_ok,
            'minimization_successful': min_ok,
            'n_subjects': s.data_profile.get('n_subjects'),
            'composite_score': composite,
        })
        '''

        def _is_valid_model(ofv, composite):
            if ofv is None:
                return False
            if ofv > 1e10 or ofv < -1e5:  # 溢出或极端负值
                return False
            if composite > 1e9:  # composite 也溢出
                return False
            return True

        # ★ 修复 2：best_* 更新用实际 composite，而非 entry['composite_score']
        # ★ 修复 2：best_* 更新用实际 composite，而非 entry['composite_score']
        if s.phase != 5:
            shrink_val = self._max_shrinkage(parsed)
            shrinkage_too_high = (
                shrink_val is not None and shrink_val > 95
            )

            # ---- 正常 best 更新（仅在 shrinkage 未超阈值时执行） ---- #
            if shrinkage_too_high:
                print(f"  [BEST] Rejected: shrinkage {shrink_val:.1f}% > 95% "
                      f"(OMEGA collapsed — not a valid population model)")
            else:
                if (min_ok and composite < s.best_composite
                        and _is_valid_model(ofv, composite)):
                    s.best_composite = composite
                    s.best_ofv = ofv
                    s.best_iteration = s.iteration
                    s.best_code = s.current_code
                    print(f"  [BEST] New best: iter={s.best_iteration}, "
                          f"OFV={ofv}, composite={composite:.1f}")
                elif min_ok and not _is_valid_model(ofv, composite):
                    print(f"  [BEST] Rejected: OFV={ofv} is not a valid model "
                          f"(overflow/negative)")

            # ---- 兜底 1：First-seen fallback（仅当 shrinkage 未超阈值） ---- #
            if (s.best_code is None and s.current_code
                    and _is_valid_model(ofv, composite)
                    and not shrinkage_too_high):
                s.best_composite = composite
                s.best_ofv = ofv
                s.best_iteration = s.iteration
                s.best_code = s.current_code
                print(f"  [BEST] First-seen fallback: iter={s.best_iteration}, "
                      f"OFV={ofv}, composite={composite:.1f}")

            # ---- 兜底 2：iter0.txt 兜底（无论 shrinkage 如何） ---------- #
            if s.best_code is None and s.current_code:
                iter0_path = f"{s.output_base}_iter0.txt"
                if os.path.exists(iter0_path):
                    try:
                        with open(iter0_path, 'r',
                                  encoding='utf-8', errors='replace') as f:
                            s.best_code = f.read()
                        s.best_iteration = 0
                        s.best_ofv = None
                        shrink_str = (f"{shrink_val:.1f}%"
                                      if shrink_val is not None else "N/A")
                        print(f"  [BEST] Fallback to iter0.txt "
                              f"(shrinkage={shrink_str}, no valid best — "
                              f"using iter0 as placeholder)")
                    except (OSError, UnicodeDecodeError) as e:
                        print(f"  [BEST] iter0.txt read failed: {e}")
                else:
                    print(f"  [BEST] iter0.txt not found: {iter0_path}")

        return s

        '''
        if s.phase != 5:
            # ★ 硬门槛：shrinkage > 95% 视为非群体模型，不允许成为 best
            shrink_val = self._max_shrinkage(parsed)
            if shrink_val is not None and shrink_val > 95:
                print(f"  [BEST] Rejected: shrinkage {shrink_val:.1f}% > 95% "
                      f"(OMEGA collapsed — not a valid population model)")
                # 但若当前无 best 且这是首次有效 iteration，仍写入兜底
                if s.best_code is None and s.current_code:
                    s.best_composite = float('inf')  # 保持无效状态
                    print(f"  [BEST] No fallback: shrinkage too high")

            # 优先在最小化成功时更新
            if min_ok and composite < s.best_composite and _is_valid_model(ofv, composite):
                s.best_composite = composite
                s.best_ofv = ofv
                s.best_iteration = s.iteration
                s.best_code = s.current_code
                print(f"  [BEST] New best: iter={s.best_iteration}, "
                      f"OFV={ofv}, composite={composite:.1f}")
            elif min_ok and not _is_valid_model(ofv, composite):
                print(f"  [BEST] Rejected: OFV={ofv} is not a valid model (overflow/negative)")

            # 兜底：仅在无任何有效 best 时使用，且必须 OFV 有效
            #        否则 finalize 永远找不到 iter0 以外的文件
            # if s.best_code is None and s.current_code:
            if s.best_code is None and s.current_code and _is_valid_model(ofv, composite):
                s.best_composite = composite
                s.best_ofv = ofv
                s.best_iteration = s.iteration
                s.best_code = s.current_code
                print(f"  [BEST] First-seen fallback: iter={s.best_iteration}, "
                      f"OFV={ofv}, composite={composite:.1f}")

            if s.best_code is None and s.current_code:
                iter0_path = f"{s.output_base}_iter0.txt"
                if os.path.exists(iter0_path):
                    with open(iter0_path, 'r') as f:
                        s.best_code = f.read()
                    s.best_iteration = 0
                    s.best_ofv = None  # 标记为无效，但不阻断流程
                    print(f"  [BEST] Fallback to iter0.txt (shrinkage {shrink_val:.1f}% too high, "
                          f"but no valid best exists — using iter0 as placeholder)")
          '''
        '''
        # Phase 1-4：更新 best model
        if s.phase != 5:
            if entry['composite_score'] < s.best_composite:
                s.best_composite = entry['composite_score']
                s.best_ofv = entry['ofv']
                s.best_iteration = s.iteration
                s.best_code = s.current_code
                print(f"  [BEST] New best: iteration {s.best_iteration}, "
                      f"OFV={s.best_ofv}, composite={s.best_composite:.2f}")
        '''


        # ★ 有效性检查：OFV 必须非空、非溢出、非负异常

    def _check_iteration_progress(self, s) -> bool:
        """
        检测上一次 iteration 是否生成了与之前相同的代码。
        返回 True 表示有进展，False 表示停滞。
        """
        if len(s.code_history) < 2:
            return True

        curr = s.code_history[-1].get('code', '')
        prev = s.code_history[-2].get('code', '')

        # 规范化后比较（去空白/注释）
        def _normalize(c):
            import re
            c = re.sub(r';.*$', '', c, flags=re.MULTILINE)
            c = re.sub(r'\s+', ' ', c)
            return c.strip()

        if _normalize(curr) == _normalize(prev):
            s._no_progress_count = getattr(s, '_no_progress_count', 0) + 1
            print(f"  [NO-PROGRESS] iter {s.iteration} produced IDENTICAL code to "
                  f"iter {s.iteration - 1} (count={s._no_progress_count})")

            if s._no_progress_count >= 3:
                print(f"  [NO-PROGRESS] 3 consecutive identical iterations — "
                      f"LLM is ignoring prompts. Forcing rescue.")
                s.rescue_pending = True
                s._no_progress_count = 0
                return False
            return False
        else:
            s._no_progress_count = 0
            return True

