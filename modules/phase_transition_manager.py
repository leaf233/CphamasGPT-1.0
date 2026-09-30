"""
Phase Transition Manager - Cleaner State Machine Pattern
Manages phase transitions with explicit, understandable logic
"""

from typing import Optional, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizer import ModelPhase


class PhaseTransitionManager:
    """Manages phase transitions with clear, explicit logic"""

    MAX_ITERATIONS_PER_PHASE = 8

    def __init__(self, data_loader, current_phase, iterations_in_phase, improvement_history):
        self.data_loader = data_loader
        self.current_phase = current_phase
        self.iterations_in_phase = iterations_in_phase
        self.improvement_history = improvement_history
        # ★ 新增：强制终止标记
        self.force_terminate = False
        self.terminate_reason = ""
        # ★ 新增：Phase 1 连续失败触发的 rescue 标记
        self.rescue_triggered = False
        # ★ 新增：阶段历史（用于判断"是否已经诊断过"）
        self.phase_history = []



    



    def determine_next_phase(self, parsed_results: Dict):
        """
        Determine next phase using clear state machine logic

        Returns:
            Next phase (always >= current_phase, never backward!)
        """
        # Step 1: Check if current phase is COMPLETE
        ModelPhase = type(self.current_phase)

        # ---- 前置：确保 current_phase 是 ModelPhase ----
        if not isinstance(self.current_phase, ModelPhase):
            print(f"  [PHASE-MGR] current_phase invalid: {type(self.current_phase)}. "
                  f"Forcing to ESTABLISH_BASE")
            self.current_phase = ModelPhase.ESTABLISH_BASE

        '''
        # ★ Phase 1 连续失败早停
        if self.current_phase == ModelPhase.ESTABLISH_BASE:
            # ★ 连续 3 次进入 Phase 2 且 covariance 仍失败 → 升级到 Phase 3
            recent_phase2_entries = sum(
                1 for h in self.improvement_history[-5:]
                if h.get('phase') == 2
            )
            if recent_phase2_entries >= 3:
                cov_ok = parsed_results.get('covariance_step', {}).get('successful', False)
                if not cov_ok:
                    print(f"[PHASE2-ESCALATE] Phase 2 stuck for "
                          f"{recent_phase2_entries} iterations with covariance "
                          f"failure — escalating to Phase 3")
                    self.current_phase = ModelPhase.REDUCE_OVERFITTING
                    self.iterations_in_phase = 0
                    return ModelPhase.REDUCE_OVERFITTING
        '''

        '''
        # ★ Phase 1 连续失败早停 → 触发 rescue
        if self.current_phase == ModelPhase.ESTABLISH_BASE:
            recent_fails = [
                h for h in self.improvement_history[-4:]
                if not h.get('minimization_successful', False)
            ]
            if len(recent_fails) >= 4:
                print(f"[EARLY FAIL] Phase 1 failed {len(recent_fails)} consecutive "
                      f"times — triggering rescue")
                self.rescue_triggered = True
                return ModelPhase.ESTABLISH_BASE  # 仍留在 Phase 1，rescue 在下一轮处理
        '''
        # ---- Phase 2 死循环检测 ----
        if self.current_phase == ModelPhase.DIAGNOSE_STRUCTURE:
            recent_phase2_entries = sum(
                1 for h in self.improvement_history[-5:]
                if h.get('phase') == 2
            )
            if recent_phase2_entries >= 3:
                cov_ok = parsed_results.get('covariance_step', {}).get('successful', False)
                if not cov_ok:
                    print(f"[PHASE2-ESCALATE] Phase 2 stuck for "
                          f"{recent_phase2_entries} iterations — escalating to Phase 3")
                    self.current_phase = ModelPhase.REDUCE_OVERFITTING
                    self.iterations_in_phase = 0
                    return ModelPhase.REDUCE_OVERFITTING


        if self._is_phase_complete(parsed_results):
            # Move to next logical phase
            next_phase = self._get_next_phase(parsed_results)
            return next_phase

        # Step 2: Check if STUCK in current phase (max iterations)



        '''
            recent_fails = [
                h for h in self.improvement_history[-4:]
                if not h.get('minimization_successful', False)
            ]
            if len(recent_fails) >= 4:
                print(f"[EARLY FAIL] Phase 1 failed {len(recent_fails)} consecutive times "
                      f"— triggering rescue")
                self.force_terminate = False  # 走 rescue 而非直接停
                self.rescue_triggered = True
                self.terminate_reason = (
                    f"Phase 1 could not establish a runnable model after "
                    f"{len(recent_fails)} consecutive attempts. This usually indicates "
                    f"the selected compartment count is not supported by the data."
                )
                return ModelPhase.ESTABLISH_BASE  # 触发 rescue
            '''

        # if self._is_phase_complete(parsed_results):
        #     return self._get_next_phase(parsed_results)
        # if self.iterations_in_phase >= self.MAX_ITERATIONS_PER_PHASE:
        #     return self._force_next_phase(parsed_results)

        if self._is_phase_complete(parsed_results):
            next_phase = self._get_next_phase(parsed_results)
            # ★ 防御：不允许返回比当前 phase 更小的值
            if next_phase.value < self.current_phase.value:
                print(f"[PHASE-GUARD] Blocked backward transition "
                      f"{self.current_phase.value} → {next_phase.value}")
                return self.current_phase
            return next_phase


        if self.iterations_in_phase >= self.MAX_ITERATIONS_PER_PHASE:
            # Force transition to avoid infinite loop
            return self._force_next_phase(parsed_results)

        return self.current_phase

        # Step 3: Stay in current phase (not complete, not stuck)
        # return self.current_phase

    def _is_phase_complete(self, parsed_results: Dict) -> bool:
        """
        Check if current phase has completed its objectives

        EXPLICIT completion criteria for each phase
        """
        minimization_ok = parsed_results.get('minimization_successful', False)
        ofv = parsed_results.get('objective_function')
        shrinkage = parsed_results.get('eta_shrinkage', [])
        avg_shrink = max([s['shrinkage'] for s in shrinkage]) if shrinkage else None

        metadata = self.data_loader.get_metadata()
        n_subjects = metadata.get('n_subjects', 100)

        # Get ModelPhase enum from current_phase
        ModelPhase = type(self.current_phase)

        # Phase 1: ESTABLISH_BASE
        if self.current_phase == ModelPhase.ESTABLISH_BASE:
            # Complete when: Minimization successful
            return minimization_ok

        # Phase 2: DIAGNOSE_STRUCTURE
        if self.current_phase == ModelPhase.DIAGNOSE_STRUCTURE:
            # Complete when: minimization successful + covariance step succeeded +
            # no boundary warning + at least 1 iteration of diagnosis.
            #
            # NOTE: this previously only checked minimization_ok, which let Phase 2
            # move on to Phase 3/4 even while a boundary/covariance issue (often a
            # structural or error-model misspecification — exactly what Phase 2's
            # own prompt is supposed to diagnose) was still present. Phase 3/4 don't
            # touch structure/error model, and phases never move backward, so an
            # unresolved Phase-2-shaped problem had nowhere to be fixed once Phase 2
            # rubber-stamped itself complete. Requiring covariance success and no
            # boundary warning keeps the model in Phase 2 until that's actually
            # resolved, before Phase 3/4 (OMEGA-only) ever see it.
            #
            # Rationale for iterations_in_phase >= 1 (unchanged): forcing 2+
            # iterations risks structural regression when the model already
            # converges well after 1 iter (LLM may unnecessarily complicate it).
            # MAX_ITERATIONS_PER_PHASE forced transition remains as safety net if
            # the boundary/covariance issue never resolves.
            cov_ok = parsed_results.get('covariance_step', {}).get('successful', False)
            nonmem_warnings = parsed_results.get('warnings', []) or []
            nonmem_errors = parsed_results.get('errors', []) or []
            boundary_hit = any('BOUNDARY' in str(w).upper() for w in nonmem_warnings + nonmem_errors)
            if minimization_ok and cov_ok and not boundary_hit and self.iterations_in_phase >= 1:
                return True

            # ★ 修改：Phase 2 连续 3 次 boundary → 不再 rescue，而是直接升级到 Phase 3
            if self.iterations_in_phase >= 3 and boundary_hit:
                print(f"[PHASE2-ESCALATE] {self.iterations_in_phase} iterations "
                      f"in Phase 2 with unresolved boundary — escalating to Phase 3")
                # 直接返回 True → determine_next_phase → _get_next_phase
                self.force_terminate = False
                self.rescue_triggered = False
                # 不再进入 rescue 循环，让 PhaseTransitionManager 正常推进到下一阶段
                return True  # ★ 让上层走 _get_next_phase → Phase 3/4

            '''
            # ★ 新增：Phase 2 连续 3 次都无法解决 boundary → 触发 rescue
            if self.iterations_in_phase >= 3 and boundary_hit:
                # 已达 Phase 2 迭代上限，仍在 boundary
                print(f"[PHASE2-STUCK] {self.iterations_in_phase} iterations "
                      f"in Phase 2 with unresolved boundary — triggering rescue")
                self.rescue_triggered = True
                return False  # 不推进 phase，让 rescue 处理
            '''

            return False

        # Phase 3: REDUCE_OVERFITTING
        if self.current_phase == ModelPhase.REDUCE_OVERFITTING:
            # Complete when:
            # - Shrinkage reduced to <90% (goal achieved)
            # - OR OFV no longer extremely negative (goal achieved)
            # Note: If goal NOT achieved, forced transition (5 iterations) handles it

            '''
            if avg_shrink and avg_shrink < 90:
                return True
            if ofv and ofv > -50:
                return True
            return False  # Let forced transition handle max iterations
            '''

            # 完成条件：shrinkage 达标 + covariance 成功 + OFV 合理
            covariance_ok = parsed_results.get('covariance_step', {}).get('successful', False)
            ofv_val = parsed_results.get('objective_function')
            ofv_ok = ofv_val is not None and -1e5 < ofv_val < 1e10

            if avg_shrink and avg_shrink < 90 and covariance_ok and ofv_ok:
                return True
            # 若 shrinkage 达标但 covariance 未成功，继续留在 Phase 3
            return False



        # Phase 4: OPTIMIZE_IIV
        if self.current_phase == ModelPhase.OPTIMIZE_IIV:
            # Complete when:
            # - Shrinkage is good (<50%) + OFV stable + Covariance OK → Ready for covariates
            # Note: If goal NOT achieved, forced transition (5 iterations) handles it
            covariance_ok = parsed_results.get('covariance_step', {}).get('successful', False)
            if avg_shrink and avg_shrink < 50:
                # Check if covariates available
                covariates_available = len(metadata.get('covariates', [])) > 0
                if covariates_available and minimization_ok and covariance_ok:
                    # Check OFV stability
                    if self._is_ofv_stable():
                        return True
            return False  # Let forced transition handle max iterations

        # Phase 5: COVARIATE_ANALYSIS
        if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            # Final phase - NEVER complete (let forced transition handle it)
            # This prevents infinite loop where Phase 5 → Phase 5
            return False

        return False


    def _compute_next_phase(self, parsed_results: Dict):
        """
        Determine the NEXT phase to transition to
        EXPLICIT phase progression rules
        纯逻辑计算下一阶段。返回 ModelPhase 或 None（未完成）。
        不在此方法内做任何防御性检查——防御由 _get_next_phase 包装。
        """
        minimization_ok = parsed_results.get('minimization_successful', False)
        ofv = parsed_results.get('objective_function')
        shrinkage = parsed_results.get('eta_shrinkage', [])
        avg_shrink = max([s['shrinkage'] for s in shrinkage]) if shrinkage else None

        metadata = self.data_loader.get_metadata()
        n_subjects = metadata.get('n_subjects', 100)

        # Get ModelPhase enum from current_phase
        ModelPhase = type(self.current_phase)

        # From Phase 1: ESTABLISH_BASE
        if self.current_phase == ModelPhase.ESTABLISH_BASE:
            # ★ 首次迭代 shrinkage>95% 时，先进入 Phase 2 诊断结构
            #   只有在 Phase 2 也无法解决时才判定为"真过拟合"
            if (self._is_true_overfitting(ofv, avg_shrink)
                    and self.iterations_in_phase >= 1):  # 至少 1 次迭代
                # 检查是否已有 Phase 2 诊断历史
                phase_history = getattr(self, 'phase_history', [])
                # already_diagnosed = any(
                #     h.get('from_phase') == ModelPhase.DIAGNOSE_STRUCTURE.value
                #     for h in phase_history
                # )

                # ★ 检查 improvement_history 中是否已进入过 Phase 2
                already_diagnosed = any(
                    h.get('phase') == ModelPhase.DIAGNOSE_STRUCTURE.value
                    for h in self.improvement_history
                )

                if not already_diagnosed:
                    print("[TRANSITION] Phase 1 → Phase 2 (initial overfitting — "
                          "diagnose structure first)")
                    return ModelPhase.DIAGNOSE_STRUCTURE

                print("[TRANSITION] Phase 1 → Phase 3 (overfitting confirmed)")
                return ModelPhase.REDUCE_OVERFITTING

            '''
            if self.current_phase == ModelPhase.ESTABLISH_BASE:
                # Check for overfitting first
                if self._is_true_overfitting(ofv, avg_shrink):
                    print("[TRANSITION] Phase 1 → Phase 3 (overfitting detected)")
                    return ModelPhase.REDUCE_OVERFITTING
            '''

            # Check dataset size
            if n_subjects < 20:
                print("[TRANSITION] Phase 1 → Phase 4 (N<20, skipping structure diagnosis)")
                return ModelPhase.OPTIMIZE_IIV

            # Normal progression
            print("[TRANSITION] Phase 1 → Phase 2 (minimization successful, diagnosing structure)")
            return ModelPhase.DIAGNOSE_STRUCTURE

        # From Phase 2: DIAGNOSE_STRUCTURE
        if self.current_phase == ModelPhase.DIAGNOSE_STRUCTURE:
            # Check for overfitting
            if self._is_true_overfitting(ofv, avg_shrink):
                print("[TRANSITION] Phase 2 → Phase 3 (overfitting detected)")
                return ModelPhase.REDUCE_OVERFITTING

            print("[TRANSITION] Phase 2 → Phase 4 (structure diagnosis complete)")
            return ModelPhase.OPTIMIZE_IIV

        # From Phase 3: REDUCE_OVERFITTING
        if self.current_phase == ModelPhase.REDUCE_OVERFITTING:
            print("[TRANSITION] Phase 3 → Phase 4 (overfitting reduced)")
            return ModelPhase.OPTIMIZE_IIV

        # From Phase 4: OPTIMIZE_IIV
        if self.current_phase == ModelPhase.OPTIMIZE_IIV:
            # Check if ready for covariate analysis
            covariates_available = len(metadata.get('covariates', [])) > 0
            covariance_ok = parsed_results.get('covariance_step', {}).get('successful', False)
            if (covariates_available and
                minimization_ok and
                covariance_ok and
                avg_shrink and avg_shrink < 50 and
                self._is_ofv_stable()):
                print("[TRANSITION] Phase 4 → Phase 5 (covariance OK + shrinkage OK)")
                return ModelPhase.COVARIATE_ANALYSIS

            if not covariance_ok:
                print("[TRANSITION] Phase 4 NOT complete — covariance step failed (prerequisite for Phase 5)")
            elif not (avg_shrink and avg_shrink < 50):
                print(f"[TRANSITION] Phase 4 NOT complete — shrinkage {avg_shrink:.1f}% still too high")
            else:
                print("[TRANSITION] Phase 4 complete (no covariates or OFV unstable)")
            return ModelPhase.OPTIMIZE_IIV  # Stay in Phase 4

        # From Phase 5: COVARIATE_ANALYSIS (final phase)
        print("[TRANSITION] Phase 5 complete (final phase)")
        return ModelPhase.COVARIATE_ANALYSIS

    def _get_next_phase(self, parsed_results: Dict):
        """
        防御性包装 _compute_next_phase，保证返回 ModelPhase 枚举。

        兜底策略：
          1. _compute_next_phase 返回 None → 返回当前 phase
          2. 返回值不是 ModelPhase → 返回当前 phase
          3. 返回值 value < 1 或 > 5 → 返回当前 phase
          4. 返回值 value < 当前 phase（反向）→ 返回当前 phase
        """
        # ---- 前置检查：self.current_phase 必须是 ModelPhase ----
        ModelPhase = type(self.current_phase)
        if not isinstance(self.current_phase, ModelPhase):
            print(f"  [PHASE-MGR] current_phase is not ModelPhase: "
                  f"{type(self.current_phase)}. Forcing to ESTABLISH_BASE")
            # 尝试修正为枚举
            try:
                self.current_phase = ModelPhase(self.current_phase)
            except (ValueError, TypeError):
                self.current_phase = ModelPhase.ESTABLISH_BASE

        # ---- 计算下一阶段 ----
        try:
            result = self._compute_next_phase(parsed_results)
        except Exception as e:
            print(f"  [PHASE-MGR] _compute_next_phase raised: {e}. "
                  f"Staying at {self.current_phase}")
            return self.current_phase

        # ---- 兜底 1：None ----
        if result is None:
            print(f"  [PHASE-MGR] _compute_next_phase returned None. "
                  f"Staying at {self.current_phase}")
            return self.current_phase

        # ---- 兜底 2：类型检查 ----
        if not isinstance(result, ModelPhase):
            print(f"  [PHASE-MGR] Invalid return type: {type(result)} "
                  f"(value={result!r}). Coercing to ModelPhase.")
            # 尝试强制转换
            try:
                coerced = ModelPhase(result)
                print(f"  [PHASE-MGR] Coerced {result} → {coerced}")
                result = coerced
            except (ValueError, TypeError):
                print(f"  [PHASE-MGR] Cannot coerce {result!r}. "
                      f"Staying at {self.current_phase}")
                return self.current_phase

        # ---- 兜底 3：值域检查 ----
        if not isinstance(result.value, int) or result.value < 1 or result.value > 5:
            print(f"  [PHASE-MGR] Invalid phase value: {result.value}. "
                  f"Staying at {self.current_phase}")
            return self.current_phase

        # ---- 兜底 4：反向检查 ----
        if result.value < self.current_phase.value:
            print(f"  [PHASE-MGR] Backward transition blocked: "
                  f"{self.current_phase.value} → {result.value}. "
                  f"Staying at {self.current_phase}")
            return self.current_phase

        return result


    def _force_next_phase(self, parsed_results: Dict):
        """
        Force transition when stuck (max iterations reached)
        强制推进（超时兜底）。保证返回 ModelPhase 且不反向。
        Always moves forward, never backward
        """
        print(f"\n{'!'*70}")
        print(f"[FORCE TRANSITION] Max iterations ({self.MAX_ITERATIONS_PER_PHASE}) reached in {self.current_phase}")
        print(f"{'!'*70}")

        # Get ModelPhase enum from current_phase
        ModelPhase = type(self.current_phase)

        # ---- 前置检查 ----
        if not isinstance(self.current_phase, ModelPhase):
            print(f"  [FORCE] current_phase invalid: {type(self.current_phase)}. "
                  f"Defaulting to ESTABLISH_BASE")
            self.current_phase = ModelPhase.ESTABLISH_BASE

        minimization_ok = parsed_results.get('minimization_successful', False)
        metadata = self.data_loader.get_metadata()
        covariates_available = len(metadata.get('covariates', [])) > 0

        # All forced transitions go to Phase 4 (safest) or Phase 5   Phase 1/2/3 → Phase 4
        if self.current_phase.value < ModelPhase.OPTIMIZE_IIV.value:
            print(f"[FORCE] {self.current_phase} → Phase 4 (safest fallback)")
            return ModelPhase.OPTIMIZE_IIV

        # From Phase 4: Try Phase 5 if possible   Phase 4 → Phase 5 或停留
        if self.current_phase == ModelPhase.OPTIMIZE_IIV:
            covariance_ok = parsed_results.get('covariance_step', {}).get('successful', False)

            if covariates_available and minimization_ok and covariance_ok:
                print(f"[FORCE] Phase 4 → Phase 5 (attempting covariates, covariance OK)")
                return ModelPhase.COVARIATE_ANALYSIS
            elif not covariance_ok:
                # ★ Phase 4 已耗尽 MAX_ITERATIONS_PER_PHASE，covariance 仍失败
                #   继续留在 Phase 4 只会无限 FORCE TRANSITION
                print(f"[FORCE] Phase 4 → TERMINATE — covariance failed "
                      f"after {self.MAX_ITERATIONS_PER_PHASE} forced iterations")
                self.force_terminate = True
                self.terminate_reason = (
                    f"Phase 4 exhausted {self.MAX_ITERATIONS_PER_PHASE} iterations "
                    f"without a successful covariance step. Further Phase 4 iterations "
                    f"are unlikely to resolve the issue within the current model. "
                    f"Recommend human review of residual error model / OMEGA structure / "
                    f"parameter bounds."
                )
                return ModelPhase.OPTIMIZE_IIV
            else:
                print(f"[FORCE] Phase 4 complete (no covariates, staying)")
                return ModelPhase.OPTIMIZE_IIV

        '''
        if self.current_phase == ModelPhase.OPTIMIZE_IIV:
            covariance_ok = parsed_results.get('covariance_step', {}).get('successful', False)
            if covariates_available and minimization_ok and covariance_ok:
                print(f"[FORCE] Phase 4 → Phase 5 (attempting covariates, covariance OK)")
                return ModelPhase.COVARIATE_ANALYSIS
            elif not covariance_ok:
                print(f"[FORCE] Phase 4 → Phase 5 BLOCKED — covariance step failed")
                print(f"[FORCE] Fix base model covariance before covariate analysis")
                return ModelPhase.OPTIMIZE_IIV
            else:
                print(f"[FORCE] Phase 4 complete (no covariates, staying)")
                return ModelPhase.OPTIMIZE_IIV
        '''

        # Phase 5: Stay (final phase)  Phase 5 → 停留
        if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            print(f"[FORCE] Phase 5 complete (final phase)")
            return ModelPhase.COVARIATE_ANALYSIS

        # ---- 未知情况 ----
        print(f"  [FORCE] Unknown phase: {self.current_phase}. Staying put.")
        return self.current_phase

    def _is_ofv_stable(self) -> bool:
        """Check if OFV has stabilized"""
        if len(self.improvement_history) < 2:
            return False

        recent_ofvs = [h.get('ofv') for h in self.improvement_history[-2:]]
        if not all(o is not None for o in recent_ofvs):
            return False

        ofv_change = abs(recent_ofvs[-1] - recent_ofvs[-2])
        return ofv_change < 5

    def _is_true_overfitting(self, ofv, avg_shrink) -> bool:
        """Check if TRUE overfitting (vs underparameterization)"""
        # Get ModelPhase enum from current_phase
        ModelPhase = type(self.current_phase)

        # Only detect overfitting if not already past Phase 3
        if self.current_phase.value >= ModelPhase.OPTIMIZE_IIV.value:
            return False

        # Criteria for TRUE overfitting
        if ofv is not None and ofv < -50:
            return True

        if avg_shrink is not None and avg_shrink > 95:
            return True

        return False
