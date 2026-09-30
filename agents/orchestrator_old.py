"""
agents/orchestrator.py

OrchestratorAgent：PKGPT 2.0 主循环状态机。

协调 Bootstrap → Phase 1-5 → Termination 的完整流程。
本文件不直接调用 NONMEM 或 LLM；所有工具调用经由 ToolRegistry / SkillRegistry。
"""

from __future__ import annotations

from typing import Optional

from skills import build_skill_registry
from .skill_selector import SkillSelector
from .human_gateway import HumanGateway


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
        prev_ofv = state.last_run.get('parsed_data', {}).get('objective_function')
        guard = tools['compartment_guard']
        msg = guard.check(new_code, prev_ofv, state.locked_compartments)
        if msg:
            print(f"  [GUARD] {msg}")
            if state.best_code:
                state.current_code = state.best_code
                state.last_revert_info = {
                    'reason':                'compartment_invariance_violation',
                    'message':               msg,
                    'reverted_to_iteration': state.best_iteration,
                }

        # ---- 2) ADVAN/TRANS 合法性 ------------------------------------ #
        advan_guard = tools['advan_trans_validity']
        msg = advan_guard.check(new_code)
        if msg:
            print(f"  [GUARD] {msg}")
            if state.best_code:
                state.current_code = state.best_code
                state.last_revert_info = {
                    'reason':                'advan_trans_invalid',
                    'message':               msg,
                    'reverted_to_iteration': state.best_iteration,
                }

        return state


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
    }

    @staticmethod
    def apply(state, result):
        if result is None:
            return state
        for key, value in (result.updates or {}).items():
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

            # ---- Phase 5 特殊分支 ------------------------------------ #
            if s.phase == 5:
                s = self._run_phase5(s)
                if s.scm_complete:
                    print("\n[SCM] Phase 5 complete")
                    break
                # Human Review 在 SCM 结束后统一处理
                if self.skills['human_review'].required(s):
                    decision = self.human.pause(s)
                    if decision == 'abort':
                        s.flags['abort'] = True
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

    # ── Phase 1-4 通用流程 ───────────────────────────────────────────── #

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

        # 5) 阶段推进（前向唯一）
        result = self.skills['phase_transition'].run(s, s.llm, self.tools)
        s = StateReducer.apply(s, result)

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
        skill = SkillSelector.select(s, self.skills)
        print(f"\n[Skill] {skill.name}")
        result = skill.run(s, s.llm, self.tools)
        s = StateReducer.apply(s, result)

        # 9) Guardrail 检查（房室不变性 + ADVAN/TRANS 合法性）
        s = GuardrailCritic.check(s, self.tools)

        return s

    # ── Phase 5 特殊流程 ─────────────────────────────────────────────── #

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
        if s.phase < 5 and s.data_profile.get('covariates'):
            best_entry = next(
                (e for e in s.improvement_history
                 if e.get('iteration') == s.best_iteration), None)
            if best_entry and best_entry.get('covariance_successful'):
                print("[OVERRIDE] AI stop deferred — advancing to Phase 5 SCM")
                s.phase = 5
                s.iterations_in_phase = 0
                return False

        # 4) 质量分
        if qe_result and 'evaluation' in qe_result.meta:
            score = qe_result.meta['evaluation'].get('score', 0)
            if score < 60:
                print(f"[OVERRIDE] AI stop rejected — quality score {score} < 60")
                return False

        return True

    # ── 历史记录 ─────────────────────────────────────────────────────── #

    def _append_history(self, s):
        """
        将当前迭代结果追加到 improvement_history。

        Phase 1-4 中，如果 composite_score 更低，同时更新 best model。
        Phase 5 由 SCM Skill 显式管理 best。
        """
        parsed = s.last_run.get('parsed_data', {})
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

        s.improvement_history = s.improvement_history + [entry]

        # Phase 1-4：更新 best model
        if s.phase != 5:
            if entry['composite_score'] < s.best_composite:
                s.best_composite = entry['composite_score']
                s.best_ofv = entry['ofv']
                s.best_iteration = s.iteration
                s.best_code = s.current_code
                print(f"  [BEST] New best: iteration {s.best_iteration}, "
                      f"OFV={s.best_ofv}, composite={s.best_composite:.2f}")

        return s