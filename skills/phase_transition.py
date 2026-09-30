"""
skills/phase_transition.py

前向唯一阶段推进。
- Phase 1 → Phase 2（若未过拟合）或 Phase 3（若过拟合）
- Phase 2 → Phase 3（若过拟合）或 Phase 4
- Phase 3 → Phase 4
- Phase 4 → Phase 5（前提满足）
- Phase 5 → 终止

Phase 2 不再负责房室升级（房室由 --compartments 锁定）。
"""

from .base import Skill, SkillResult
from modules.phase_transition_manager import PhaseTransitionManager
from modules.optimizer import ModelPhase


class PhaseTransitionSkill(Skill):
    name = 'phase_transition'
    phase = [1, 2, 3, 4, 5]
    tools = []
    prompt_template = ''

    def run(self, state, llm, tools) -> SkillResult:
        if state.phase_manager is None:
            state.phase_manager = PhaseTransitionManager(
                data_loader=state.data_loader,
                current_phase=ModelPhase(state.phase),
                iterations_in_phase=state.iterations_in_phase,
                improvement_history=state.improvement_history,
            )

        state.phase_manager.current_phase = ModelPhase(state.phase)
        state.phase_manager.iterations_in_phase = state.iterations_in_phase

        parsed = state.last_run.get('parsed_data', {})
        new_phase = state.phase_manager.determine_next_phase(parsed)

        # ★★★ FIX 4：N<20 + 3cmt 强制不跳过 Phase 2 ★★★
        # 小样本 + 高房室数时，OMEGA 塌缩风险极高，
        # 必须经过 Phase 2 的 OMEGA 早期塌陷检测（90% 阈值）。
        # 触发条件：当前 Phase 1，N<20，房室数 >=3，且原计划跳到 Phase >= 3
        n_subjects = state.data_profile.get('n_subjects', 100)
        n_cmt = state.locked_compartments or 1
        if (state.phase == 1
                and n_subjects < 20
                and n_cmt >= 3
                and new_phase is not None
                and getattr(new_phase, 'value', 1) >= 3):
            print(f"  [PHASE-TRANSITION-OVERRIDE] N={n_subjects} + "
                  f"{n_cmt}cmt — cannot skip Phase 2 (OMEGA collapse risk). "
                  f"Forcing 1 → 2.")
            new_phase = ModelPhase.DIAGNOSE_STRUCTURE  # Phase 2
            # 记录 override 事件到 phase_history
            state.phase_history.append({
                'iteration': state.iteration,
                'from_phase': 1,
                'to_phase': 2,
                'iterations_in_previous_phase': state.iterations_in_phase,
                'reason': f'small_n_3cmt_requires_phase2 '
                          f'(N={n_subjects}, {n_cmt}cmt)',
            })
            # 同步 phase_manager
            if state.phase_manager is not None:
                state.phase_manager.current_phase = ModelPhase.DIAGNOSE_STRUCTURE
                state.phase_manager.iterations_in_phase = 0

        # ★ 防御 1：new_phase 必须存在且为 ModelPhase
        if new_phase is None:
            print(f"  [PHASE-TRANSITION] determine_next_phase returned None "
                  f"— staying at Phase {state.phase}")
            return SkillResult(updates={'iterations_in_phase': state.iterations_in_phase + 1})

        # ★ 防御 2：new_phase.value 必须 ≥ 1（防止 0 写入）
        if not hasattr(new_phase, 'value') or new_phase.value < 1:
            print(f"  [PHASE-TRANSITION] Invalid phase value: {new_phase}. "
                  f"Coercing to current phase {state.phase}")
            return SkillResult(updates={'iterations_in_phase': state.iterations_in_phase + 1})

        if new_phase.value < state.phase:
            print(f"  [PHASE-TRANSITION] Blocked backward: "
                  f"{state.phase} → {new_phase.value}")
            return SkillResult(updates={'iterations_in_phase': state.iterations_in_phase + 1})

        # ★ 检查强制终止标记
        if getattr(state.phase_manager, 'force_terminate', False):
            reason = state.phase_manager.terminate_reason
            print(f"\n[PHASE TRANSITION] Forced termination: {reason}")
            return SkillResult(updates={
                'force_stop': True,
                'force_stop_reason': reason
            })

        # 前向唯一
        if new_phase.value < state.phase:
            return SkillResult()  # 向后被阻止

        if new_phase.value != state.phase:
            history_entry = {
                'iteration': state.iteration,
                'from_phase': state.phase,
                'to_phase': new_phase.value,
                'iterations_in_previous_phase': state.iterations_in_phase,
            }
            # ★ 阶段推进时清空 failed_strategies（旧阶段的失败不再适用）
            print(f"  [TRACK] Clearing {len(state.failed_strategies)} failed strategies on phase transition")
            return SkillResult(updates={
                'phase': new_phase.value,
                'iterations_in_phase': 0,
                'phase_history': state.phase_history + [history_entry],
            })
        return SkillResult(updates={'iterations_in_phase': state.iterations_in_phase + 1})

    '''
    def _get_next_phase(self, parsed_results: Dict):
        """所有返回路径必须返回 ModelPhase 枚举成员。"""
        result = self._compute_next_phase(parsed_results)

        # ★ 最终防御：返回前断言
        if not isinstance(result, type(self.current_phase)):
            print(f"  [PHASE-MGR] _get_next_phase returned invalid type: "
                  f"{type(result)}. Forcing current phase.")
            return self.current_phase

        if result.value < 1 or result.value > 5:
            print(f"  [PHASE-MGR] Invalid phase value {result.value}. "
                  f"Forcing current phase.")
            return self.current_phase

        return result
    '''