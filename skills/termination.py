"""
skills/termination.py

终止判断与最终模型保存。

终止条件：
1. 达到 max_iterations（Phase 1-4）
2. Phase 5 SCM 完成
3. 早停（连续 catastrophic shrinkage / 负 OFV / OMEGA 全塌缩 / 分数持续恶化）
4. AI quality evaluation 判定可停且质量足够

finalize：保存 best model 为 {output_base}_final.txt。
"""

import os
import shutil
from .base import Skill, SkillResult
from utils.compartment_lock import validate_lock, parse_advan


class TerminationSkill(Skill):
    name = 'termination'
    phase = [1, 2, 3, 4, 5]
    tools = []
    prompt_template = ''

    # ---- 终止判断 -------------------------------------------------------- #

    def should_stop(self, state) -> tuple[bool, str]:

        # ★ 最高优先级：PhaseTransitionManager 判定无法收敛
        if getattr(state, 'force_stop', False):
            return True, state.force_stop_reason or "forced stop"

        # 1) 达到迭代上限（Phase 1-4）
        if state.phase != 5 and state.iteration > state.max_iterations:
            return True, f"max_iterations ({state.max_iterations}) reached"

        # 2) Phase 5 完成（scm done）
        if state.phase == 5 and state.scm_complete:
            return True, "SCM complete"

        # 3) 早停
        stop, reason = _early_stop(state)
        if stop:
            return True, reason

        return False, ""

    # ---- 最终保存 -------------------------------------------------------- #
    def finalize(self, state) -> dict:
        final_file = f"{state.output_base}_final.txt"
        best_file = f"{state.output_base}_iter{state.best_iteration}.txt"

        # 读 best_code（或 iter0 兜底）
        best_code = None
        if os.path.exists(best_file):
            with open(best_file, 'r', encoding='utf-8') as f:
                best_code = f.read()
        elif os.path.exists(f"{state.output_base}_iter0.txt"):
            best_file = f"{state.output_base}_iter0.txt"
            with open(best_file, 'r', encoding='utf-8') as f:
                best_code = f.read()

        # ★ 最终房室校验
        if best_code and state.locked_compartments:
            ok, actual, advan = validate_lock(best_code, state.locked_compartments)
            if not ok:
                print(f"\n{'!' * 70}")
                print(f"❌ FINAL VALIDATION FAILED: best model has ADVAN{advan} "
                      f"({actual}-cmt) but --compartments was "
                      f"{state.locked_compartments}")
                print(f"{'!' * 70}")
                print(f"[ERROR] Refusing to save non-conforming model.")
                return {
                    'total_iterations': state.iteration,
                    'best_iteration': state.best_iteration,
                    'best_ofv': state.best_ofv,
                    'best_compartments': actual,
                    'locked_compartments': state.locked_compartments,
                    'history': state.improvement_history,
                    'final_file': None,
                    'error': (f"Final model compartment mismatch: "
                              f"ADVAN{advan} ({actual}-cmt) != "
                              f"{state.locked_compartments}-cmt"),
                }
            else:
                print(f"[OK] Final compartment validation passed: "
                      f"ADVAN{advan} = {actual}-cmt")

        # 保存
        if best_code and os.path.exists(best_file):
            shutil.copy(best_file, final_file)
            print(f"[OK] Best model saved to: {final_file}")
        else:
            final_file = None

        self._print_detailed_summary(state)

        # 兜底 1：best_file 不存在时尝试 iter0.txt
        if not os.path.exists(best_file):
            iter0 = f"{state.output_base}_iter0.txt"
            if os.path.exists(iter0):
                print(f"[WARNING] best_file {best_file} missing, falling back to iter0.txt")
                best_file = iter0

        # 兜底 2：仍不存在时尝试最近一个 iterN.txt
        if not os.path.exists(best_file):
            import glob, re as _re
            candidates = sorted(glob.glob(f"{state.output_base}_iter*.txt"),
                                key=lambda p: int(_re.search(r'_iter(\d+)\.txt', p).group(1)))
            if candidates:
                best_file = candidates[-1]
                print(f"[WARNING] No best_file, falling back to latest: {best_file}")

        if os.path.exists(best_file):
            shutil.copy(best_file, final_file)
            print(f"[OK] Best model saved to: {final_file}")
        else:
            print(f"[ERROR] No model file found for {state.output_base}")
            final_file = None

        # 详细摘要
        self._print_detailed_summary(state)

        return {
            'total_iterations': state.iteration,
            'best_iteration': state.best_iteration,
            'best_ofv': state.best_ofv,
            'locked_compartments': state.locked_compartments,
            'best_composite': state.best_composite,
            'force_stop': state.force_stop,
            'force_stop_reason': state.force_stop_reason,
            'history': state.improvement_history,
            'final_file': final_file,
        }

    def _print_detailed_summary(self, state):
        """Phase Progression + Base Model Development + SCM 结果。"""
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        # Phase Progression
        if state.phase_history:
            print(f"\n{'=' * 70}\nPHASE PROGRESSION\n{'=' * 70}")
            for t in state.phase_history:
                print(f"  Iter {t['iteration']:>3}: Phase {t['from_phase']} -> {t['to_phase']} "
                      f"({t['iterations_in_previous_phase']} iters)")
        # Base Model Development 表格
        print(f"\n{'=' * 70}\nBASE MODEL DEVELOPMENT (Phase 1-4)\n{'=' * 70}")
        print(f"{'Iter':<6} {'Status':<8} {'OFV':<12} {'Composite':<12} {'Shrink':<10} {'CoV':<5}")
        print("-" * 70)
        for e in state.improvement_history:
            it = e.get('iteration', '?')
            status = "OK" if e.get('minimization_successful') else "FAIL"
            ofv = f"{e['ofv']:.2f}" if e.get('ofv') is not None else "N/A"
            comp = e.get('composite_score', float('inf'))
            comp_s = f"{comp:.1f}" if comp != float('inf') else "N/A"
            shrink = e.get('avg_eta_shrinkage')
            shrink_s = f"{shrink:.1f}%" if shrink is not None else "N/A"
            cov = "Yes" if e.get('covariance_successful') else "No"
            star = " *" if it == state.best_iteration else ""
            print(f"{it:<6} {status:<8} {ofv:<12} {comp_s:<12} {shrink_s:<10} {cov:<5}{star}")
        print("-" * 70)
        # Best model quality
        best = next((e for e in state.improvement_history
                     if e.get('iteration') == state.best_iteration), None)
        if best:
            shrink = best.get('avg_eta_shrinkage')
            grade = ("EXCELLENT" if shrink and shrink < 30 else
                     "GOOD" if shrink and shrink < 50 else
                     "ACCEPTABLE" if shrink and shrink < 70 else
                     "CONCERNING" if shrink and shrink < 90 else "CRITICAL")
            print(f"\nFINAL MODEL QUALITY")
            print(f"  ETA Shrinkage : {shrink:.1f}%  [{grade}]" if shrink else "  ETA Shrinkage : N/A")
            print(f"  Covariance    : {'SUCCESS' if best.get('covariance_successful') else 'FAILED'}")
            # if best.get('ofv') is not None:
            #     print(f"  OFV           : {best['ofv']:.2f}")

            ofv_val = best.get('ofv')
            if ofv_val is None:
                print(f"  OFV           : N/A")
            elif ofv_val > 1e10 or ofv_val < -1e5:
                print(f"  OFV           : INVALID ({ofv_val:.2e} — numerical overflow)")
            else:
                print(f"  OFV           : {ofv_val:.2f}")
        print("=" * 70)


    '''
    def finalize(self, state) -> dict:
        final_file = f"{state.output_base}_final.txt"
        best_file = f"{state.output_base}_iter{state.best_iteration}.txt"
        if os.path.exists(best_file):
            shutil.copy(best_file, final_file)
            print(f"[OK] Best model saved to: {final_file}")

        return {
            'total_iterations': state.iteration,
            'best_iteration':   state.best_iteration,
            'best_ofv':         state.best_ofv,
            'history':          state.improvement_history,
            'final_file':       final_file if os.path.exists(best_file) else None,
        }
    '''
    def run(self, state, llm, tools) -> SkillResult:
        """Skill 接口；由 Orchestrator 显式调用 should_stop / finalize。"""
        stop, reason = self.should_stop(state)
        return SkillResult(meta={'should_stop': stop, 'reason': reason})


def _early_stop(state) -> tuple[bool, str]:
    """复制优化器 _should_stop_early 逻辑，Phase 5 不适用。"""
    if state.phase == 5:
        return False, ""
    h = state.improvement_history
    if len(h) < 5:
        return False, ""

    recent = h[-6:]
    # 连续 >95% shrinkage
    cat = [e for e in recent[-4:] if (e.get('avg_eta_shrinkage') or 0) > 95]
    if len(cat) >= 4:
        return True, "catastrophic shrinkage (>95%) for 4+ iterations"
    # 连续负 OFV
    neg = [e for e in recent[-4:] if (e.get('ofv') or 0) < -50]
    if len(neg) >= 4:
        return True, "negative OFV (<-50) for 4+ iterations"

    # ★ 检查连续 3 次 FAIL（minimization 失败）+ 无 best 更新
    #   若满足，触发一次"rescue"策略（放宽所有 THETA 边界 + 简化 OMEGA）
    #   而不是立即停止
    recent_fail = [e for e in recent[-3:]
                   if not e.get('minimization_successful')]
    if len(recent_fail) >= 3:
        # 只有第一次触发 rescue
        if not getattr(state, '_rescue_attempted', False):
            state._rescue_attempted = True
            state.rescue_pending = True
            return False, ""  # 让 Orchestrator 处理 rescue
        # rescue 后再失败 → 真的停止
        return True, "3+ consecutive minimization failures after rescue"

    # 分数持续恶化
    scores = [e.get('composite_score', float('inf')) for e in recent]
    if all(s != float('inf') for s in scores):
        if all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)) \
           and scores[-1] > scores[0] * 1.5:
            return True, "composite score degrading for 6 iterations"
    return False, ""