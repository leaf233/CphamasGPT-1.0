"""
skills/phase2_diagnose.py

Phase 2: 诊断与稳定性修复。

重要变更：本 Skill 不再执行房室升级。
房室数量由 --compartments 或数据推断锁定（state.locked_compartments）。
不同房室数量由并行运行的独立子进程探索。

职责：
1. 读取 .lst 诊断（边界参数 / 协方差失败 / 残差模式）
2. 残差系统性偏差 → 调整 THETA 初值/边界 或 误差模型（绝不动 ADVAN）
3. 边界参数 → 放宽 THETA 边界
4. 协方差失败 → 调整误差模型
5. 全部通过 → done=True，由 PhaseTransitionSkill 推进
"""

from .base import Skill, SkillResult, ValidationResult, CompartmentLockError
from modules.prompts.phase2_diagnose import Phase2Diagnose
from utils.nonmem_utils import extract_code, extract_theta_param_map as _extract_theta_param_map
import re
from .phase4_optimize_iiv import _collapse_omega_to_cl_only
from utils.cov_diagnosis import diagnose_covariance_failure, identify_boundary_theta as _identify_boundary_theta
from utils.nonmem_utils import _count_omega



class DiagnoseStructureSkill(Skill):
    name = 'phase2_diagnose'
    phase = [2]
    # 注意：不含 advan_switch
    # tools = ['nonmem_parser', 'boundary_widener',
    #          'error_model_editor', 'compartment_guard',
    #          'nonmem_runner']
    tools = ['nonmem_parser', 'compartment_guard', 'nonmem_runner']
    prompt_template = 'phase2_diagnose'



    def run(self, state, llm, tools) -> SkillResult:
        parser = self._tool(tools, 'nonmem_parser')
        parsed = parser.parse(state.last_run['lst_path'])

        # ★ OMEGA 早期塌陷检测（必须在结构诊断之前）
        max_shrink = max(
            (s['shrinkage'] for s in parsed.get('eta_shrinkage', [])),
            default=None)
        omega_count = _count_omega(state.current_code)

        if max_shrink is not None and max_shrink > 90 and omega_count > 1:
            print(f"  [PHASE2-OMEGA-EARLY] Shrinkage {max_shrink:.1f}% > 90% "
                  f"with OMEGA={omega_count} — auto-collapsing to CL-only")
            new_code = _collapse_omega_to_cl_only(state.current_code)
            state.failed_strategies.append("PHASE2_OMEGA_EARLY_COLLAPSE")
            return SkillResult(updates={'current_code': new_code})

        # ★ 最高优先级：结构性边界（≥3 个结构参数撞边界）
        # boundary_thetas = parser.identify_boundary_theta(state.last_run.get('full_output', ''))
        boundary_thetas = _get_boundary_thetas(state)
        theta_map = _extract_theta_param_map(state.current_code)
        boundary_param_names = [theta_map.get(i, f'THETA({i})') for i in boundary_thetas]
        STRUCTURAL_PK = {'CL', 'V1', 'V2', 'V3', 'V', 'Q', 'Q2', 'Q3', 'Q4', 'Ka'}
        n_structural_boundary = sum(1 for n in boundary_param_names if n in STRUCTURAL_PK)

        # 标记位跨分支传递"截断"意图
        structural_boundary_mode = False

        if n_structural_boundary >= 3:
            print(f"  [PHASE2-STRUCTURAL-BOUNDARY] {n_structural_boundary} structural "
                  f"params at boundary — skipping bound-widening, applying structural fix")
            # 尝试加协变量
            result = self._apply_mandatory_covariates(state)
            if result:
                return result
            # 尝试加 ALAG（oral route）
            result = self._apply_alag_if_missing(state)
            if result:
                return result
            structural_boundary_mode = True
            print(f"  [PHASE2-STRUCTURAL-BOUNDARY] No structural fix available — "
                  f"limiting bound-widening to top-2 THETA")
            # 兜底：至少不要把 6 个参数全部扩边界
            # 只扩最严重的 1-2 个，其余保持不变
            # boundary_thetas = boundary_thetas[:2]

        '''
        # 新增：OMEGA 早期塌陷检测
        max_shrink = max((s['shrinkage'] for s in parsed.get('eta_shrinkage', [])), default=None)
        omega_count = _count_omega(state.current_code)

        if max_shrink is not None and max_shrink > 95 and omega_count > 1:
            print(f"  [PHASE2-OMEGA-EARLY] Shrinkage {max_shrink:.1f}% > 95% — "
                  f"OMEGA over-parameterized, auto-simplifying before structural diagnosis")
            new_code = _collapse_omega_to_cl_only(state.current_code)
            state.failed_strategies.append("PHASE2_OMEGA_EARLY_COLLAPSE")
            return SkillResult(updates={'current_code': new_code})
        '''

        diag = parser.diagnose_structural_model(parsed)

        # 1) 残差系统性偏差 → LLM 调整 THETA 或误差模型（绝不改 ADVAN）
        if not diag.get('structural_adequate', True):
            prompt = Phase2Diagnose.generate_prompt(
                iteration=state.iteration,
                current_code=state.current_code,
                nonmem_output=state.last_run.get('full_output', ''),
                parsed_results=parsed,
                warnings=parsed.get('warnings', []),
                n_subjects=state.data_profile['n_subjects'],
                plausibility_bounds=state.plausibility_bounds,
            )
            prompt = self._prepend_warning(prompt, state)  # ★
            response = llm.generate(prompt, model_type=state.model)
            new_code = extract_code(response)

            # ★ 强制房室锁
            try:
                new_code = self._enforce_compartment_lock(
                    state, new_code, llm, prompt)
            except CompartmentLockError as e:
                print(f"  [PHASE2-LOCK-FAIL] {e}")
                return self.fallback(state, str(e))

            # 结构冻结守卫：即使 LLM 违规改 ADVAN，也在此恢复
            base = state.best_code or state.current_code
            new_code = self._tool(tools, 'compartment_guard').apply(new_code, base_code=base,hint_compartments=state.locked_compartments)

            return SkillResult(updates={'current_code': new_code})


        # ★ 误差模型边界 → 优先简化而非扩边界
        if getattr(state, 'error_model_needs_simplification', False):
            simplified = self._try_auto_simplify_error_model(state, parsed, force=True)
            if simplified:
                print(f"  [PHASE2-ERROR-SIMPLIFY] {simplified}")
                state.error_model_needs_simplification = False
                return SkillResult(updates={'current_code': state.current_code})

        # ★ 结构性边界失败 → 主动加药理先验协变量
        # boundary_thetas = parser.identify_boundary_theta(state.last_run.get('full_output', ''))
        boundary_thetas = _get_boundary_thetas(state)
        if structural_boundary_mode:
            # 按索引升序取前 2 个，避免"9 个全扩"造成 Ka 爆炸
            boundary_thetas = sorted(boundary_thetas)[:2]
            print(f"  [PHASE2-BOUND-LIMITED] Widening only THETA {boundary_thetas}")

        if boundary_thetas:
            new_code = _widen_theta_bounds(state.current_code, boundary_thetas, state=state)
            return SkillResult(updates={'current_code': new_code})

        theta_map = _extract_theta_param_map(state.current_code)
        boundary_param_names = [theta_map.get(i, f'THETA({i})')
                                for i in boundary_thetas]
        STRUCTURAL_PK = {'CL', 'V1', 'V2', 'V3', 'Q', 'Q2', 'Q3', 'Q4', 'Ka'}

        # 3+ 个结构参数撞边界 → 说明缺少关键协变量
        n_structural_boundary = sum(
            1 for n in boundary_param_names if n in STRUCTURAL_PK)

        structural_boundary_mode = False  # ★ 标志位跨分支传递
        '''
        if n_structural_boundary >= 3:
            print(f"  [PHASE2-COVARIATE] {n_structural_boundary} structural "
                  f"params at boundary — likely missing pharmacological covariates")
            
            # 从状态中推断药物类别，选择强制协变量
            drug = (state.plausibility_bounds or {}).get('drug_identified', '').lower()
            covariates = state.data_profile.get('columns', [])

            # 药理先验映射
            MANDATORY_COVS = {
                'tobramycin': [('CLCR', 'CL', 0.75), ('WT', 'V1', 1.0)],
                'gentamicin': [('CLCR', 'CL', 0.75), ('WT', 'V1', 1.0)],
                'amikacin': [('CLCR', 'CL', 0.75), ('WT', 'V1', 1.0)],
                'warfarin': [('WT', 'CL', 0.75), ('WT', 'V', 1.0)],
                'theophylline': [('WT', 'CL', 0.75), ('WT', 'V', 1.0)],
            }

            mandatory = None
            for drug_key, cov_list in MANDATORY_COVS.items():
                if drug_key in drug:
                    mandatory = cov_list
                    break

            if mandatory:
                available = {c.upper() for c in covariates}
                added = []
                new_code = state.current_code
                for cov, param, exp in mandatory:
                    if cov.upper() in available:
                        if cov.upper() not in new_code.upper():
                            new_code = _add_power_covariate(
                                new_code, cov, param, exp)
                            added.append(f"{cov} on {param} (power {exp})")

                if added:
                    print(f"  [PHASE2-COVARIATE] Added: {', '.join(added)}")
                    state.failed_strategies.append(
                        f"PHASE2_ADDED_COVARIATES_{'_'.join(added)}")
                    return SkillResult(updates={'current_code': new_code})
            '''

        if n_structural_boundary >= 3:
            print(f"  [PHASE2-STRUCTURAL-BOUNDARY] {n_structural_boundary} "
                  f"structural params at boundary — skipping bound-widening, "
                  f"applying structural fix")

            # 优先级 1：药理先验协变量
            result = self._apply_mandatory_covariates(state)
            if result:
                return result

            # 优先级 2：oral 给药注入 ALAG1
            result = self._apply_alag_if_missing(state)
            if result:
                return result

            # 优先级 3：只扩最严重的 2 个
            structural_boundary_mode = True
            print(f"  [PHASE2-STRUCTURAL-BOUNDARY] No covariate/ALAG available — "
                  f"limiting bound-widening to top-2 THETA")


        # 2) 边界参数 → 放宽 THETA 边界
        # boundary_thetas = parser.identify_boundary_theta(state.last_run.get('full_output', ''))
        boundary_thetas = _get_boundary_thetas(state)
        if structural_boundary_mode:
            # ★ 关键：截断逻辑在此生效，不会被重新获取的列表覆盖
            boundary_thetas = sorted(boundary_thetas)[:2]
            print(f"  [PHASE2-BOUND-LIMITED] Widening only THETA {boundary_thetas}")

        if boundary_thetas:
            # new_code = self._tool(tools, 'boundary_widener').apply(
            #     state.current_code, boundary_thetas)
            new_code = _widen_theta_bounds(state.current_code, boundary_thetas, state=state)
            return SkillResult(updates={'current_code': new_code})

        # 3) 协方差失败 → 调整误差模型
        if not parsed.get('covariance_step', {}).get('successful', False):
            # ★ 当场重新诊断，避免用上一轮迭代的陈旧诊断
            # ★ 当场刷新诊断（用当前迭代的 parsed + last_run）
            diagnose_covariance_failure(state)
            diag = getattr(state, 'cov_failure_diagnosis', None)
            force = (diag is not None
                     and diag.get('type') == 'ERROR_MODEL_BOUNDARY')
            # ★ 4.3a: 检测 Additive Error 是否触边界，自动简化
            simplified = self._try_auto_simplify_error_model(state, parsed, force=force)
            if simplified:
                print(f"  [AUTO-SIMPLIFY] {simplified}")
                return SkillResult(updates={'current_code': state.current_code})

            # 4.3b: 未命中简化条件 → 走 LLM

            base_prompt = Phase2Diagnose.generate_prompt(
                iteration=state.iteration,
                current_code=state.current_code,
                nonmem_output=state.last_run.get('full_output', ''),
                parsed_results=parsed,
                warnings=parsed.get('warnings', []),
                n_subjects=state.data_profile['n_subjects'],
                plausibility_bounds=state.plausibility_bounds,
            )

            # ★ 追加协方差专项修复指令
            cov_issue = parsed.get('covariance_step', {})
            cov_issues_text = '\n'.join(cov_issue.get('issues', []))
            prompt = base_prompt + f"""

            {'=' * 70}
            COVARIANCE STEP FAILURE — TARGETED FIX
            {'=' * 70}
            Covariance issues reported:
            {cov_issues_text}

            MANDATORY FIX PRIORITY (least destructive first):
            1. If a THETA is near its boundary: widen its lower/upper bounds
            2. If Additive Error THETA collapsed: switch to proportional-only error
            3. If OMEGA has collapsed: remove the highest-shrinkage ETA (keep ≥1)
            4. Do NOT change ADVAN or compartment count

            {'=' * 70}
            """

            '''
            prompt = Phase2Diagnose.generate_covariance_prompt(
                iteration=state.iteration,
                current_code=state.current_code,
                parsed_results=parsed,
                n_subjects=state.data_profile['n_subjects'],
            )
            '''
            prompt = self._prepend_warning(prompt, state)  # ★
            response = llm.generate(prompt, model_type=state.model)
            new_code = extract_code(response)
            base = state.best_code or state.current_code
            new_code = self._tool(tools, 'compartment_guard').apply(
                new_code, base_code=base,hint_compartments=state.locked_compartments)
            return SkillResult(updates={'current_code': new_code})

        # 4) 诊断通过 → 让 PhaseTransitionSkill 推进
        return SkillResult(done=True)


    def _apply_mandatory_covariates(self, state) -> SkillResult | None:
        """
        当 ≥3 个结构参数同时撞边界时，按药物类别注入药理先验协变量。

        返回 SkillResult（成功注入）或 None（无可用协变量 / 药物未匹配）。
        """
        drug = (state.plausibility_bounds or {}).get('drug_identified', '').lower()
        covariates = state.data_profile.get('columns', []) or []
        available = {str(c).upper() for c in covariates}

        MANDATORY_COVS = {
            'tobramycin':   [('CLCR', 'CL', 0.75), ('WT', 'V1', 1.0)],
            'gentamicin':   [('CLCR', 'CL', 0.75), ('WT', 'V1', 1.0)],
            'amikacin':     [('CLCR', 'CL', 0.75), ('WT', 'V1', 1.0)],
            'warfarin':     [('WT',  'CL', 0.75), ('WT', 'V1', 1.0)],
            'theophylline': [('WT',  'CL', 0.75), ('WT', 'V1', 1.0)],
        }

        mandatory = None
        for drug_key, cov_list in MANDATORY_COVS.items():
            if drug_key in drug:
                mandatory = cov_list
                break
        if not mandatory:
            return None

        new_code = state.current_code
        added = []
        for cov, param, exp in mandatory:
            if cov.upper() not in available:
                continue

            already = re.search(
                rf'(?:TV)?{re.escape(param)}\w*\s*=\s*[^\n]*\(\s*{re.escape(cov)}\s*/',
                new_code, re.IGNORECASE)
            if already:
                print(f"  [PHASE2-COVARIATE-SKIP] {cov} on {param} already injected")
                continue

            # 已注入该协变量则跳过
            if re.search(rf'\(\s*{re.escape(cov)}\s*/', new_code, re.IGNORECASE):
                continue

            trial = _add_power_covariate(new_code, cov, param, exp)
            if trial != new_code:
                new_code = trial
                added.append(f"{cov} on {param} (power {exp})")
            else:
                print(f"  [PHASE2-COVARIATE-FAIL] Cannot inject {cov} on {param} — "
                      f"LHS not matched (tried: {param}, TV{param})")

        if not added:
            return None

        state.current_code = new_code
        state.failed_strategies.append(
            f"PHASE2_ADDED_COVARIATES_{'_'.join(added)}")
        print(f"  [PHASE2-COVARIATE] Injected: {', '.join(added)}")
        return SkillResult(updates={'current_code': new_code})
    

    def _apply_alag_if_missing(self, state) -> SkillResult | None:
        """
        oral 给药 + Ka 撞边界 + 无 ALAG1 时，自动注入 ALAG1。

        返回 SkillResult（成功注入）或 None（不满足注入条件）。
        """
        route = str(state.data_profile.get('route', 'oral')).lower()
        if route != 'oral':
            return None
        if 'ALAG' in state.current_code.upper():
            return None

        # 仅当 Ka 撞边界时才注入
        boundary_thetas = _get_boundary_thetas(state)
        theta_map = _extract_theta_param_map(state.current_code)
        boundary_names = [theta_map.get(i, f'THETA({i})')
                          for i in boundary_thetas]
        if not any(n.upper() == 'KA' for n in boundary_names):
            return None

        new_code = _inject_alag1(state.current_code)
        if new_code == state.current_code:
            return None

        state.current_code = new_code
        state.failed_strategies.append("PHASE2_ADDED_ALAG1")
        print(f"  [PHASE2-ALAG] Injected ALAG1 (oral route, Ka at boundary)")
        return SkillResult(updates={'current_code': new_code})

    def _try_auto_simplify_error_model(self, state, parsed, force=False):
        """
        当协方差失败且 Additive Error 参数塌缩时，将 $ERROR 简化为比例误差。
        force=False（默认）：走原有阈值判断（abs(add_val) <= 0.01 才简化）
        force=True        ： cov_failure_diagnosis.type == 'ERROR_MODEL_BOUNDARY'就简化，不管 add_val 的具体数值
        判据：
          - $ERROR 中 W = SQRT(THETA(a)**2 + (THETA(b)*IPRED)**2) 形式
          - 提取的 THETA(a) 估计值 < 1e-4（塌缩）
        动作：
          - $ERROR 改为 W = THETA(b) * IPRED
          - 从 $THETA 移除 Additive 参数
          - $SIGMA 改为 1 FIX（或保持原值）
        """
        code = state.current_code
        if 'SQRT' not in code.upper() or 'IPRED' not in code.upper():
            return None

        # 从 parsed 结果找 Additive THETA
        thetas = (parsed.get('parameter_estimates') or {}).get('theta', [])
        if not thetas:
            return None

        # 尝试匹配 (THETA(a)**2 + (THETA(b)*IPRED)**2)
        m = re.search(
            r'SQRT\s*\(\s*THETA\s*\(\s*(\d+)\s*\)\s*\*\*\s*2\s*\+\s*\(\s*THETA\s*\(\s*(\d+)\s*\)\s*\*\s*IPRED\s*\)\s*\*\*\s*2\s*\)',
            code, re.IGNORECASE)
        if not m:
            return None
        add_idx = int(m.group(1))
        prop_idx = int(m.group(2))

        # 找 Additive THETA 估计值
        add_val = None
        for t in thetas:
            if t.get('index') == add_idx:
                add_val = t.get('value')
                break
        # if add_val is None or abs(add_val) > 1e-4:
        #     return None

        if not force:
            # 原有阈值判断
            if add_val is None or abs(add_val) > 0.01:
                return None
        else:
            # force 模式：只要诊断类型是 ERROR_MODEL_BOUNDARY 就简化
            diag = getattr(state, 'cov_failure_diagnosis', None)
            if not (diag and diag.get('type') == 'ERROR_MODEL_BOUNDARY'):
                return None

        '''
        # ★ 放宽阈值：THETA 从 1e-4 放宽到 0.01（因为 THETA 是 SD 而非方差）
        # if add_val is None or abs(add_val) > 0.01: return None
        # ★ 或者：结合 boundary 诊断
        # 如果 GuardrailCritic 已标记 ERROR_MODEL_BOUNDARY，则直接简化
        diag = getattr(state, 'cov_failure_diagnosis', None)

        # if diag and diag.get('type') == 'ERROR_MODEL_BOUNDARY':
        #     force_simplify = True
        force_simplify = (diag is not None
                          and diag.get('type') == 'ERROR_MODEL_BOUNDARY')
        if not force_simplify and (add_val is None or abs(add_val) > 0.01):
            return None
        '''

        # 执行简化
        new_code = re.sub(
            rf'W\s*=\s*SQRT\(.*?\)',
            f'W = THETA({prop_idx}) * IPRED',
            code, count=1, flags=re.IGNORECASE | re.DOTALL)

        # 从 $THETA 中移除 Additive 行（行号或值匹配）
        new_code = re.sub(
            rf'[^\n]*THETA\({add_idx}\)[^\n]*\n', '', new_code, count=1)

        state.current_code = new_code
        state.failed_strategies.append(
            f"ERROR_MODEL_SIMPLIFIED_COMBINED_TO_PROPORTIONAL_ADDITIVE_COLLAPSED")
        return (f"Auto-simplified $ERROR: removed additive THETA({add_idx}) "
                f"(est={add_val:.2e} → boundary), kept proportional THETA({prop_idx})")

        '''
        def _apply_mandatory_covariates(self, state):
            """尝试注入药理先验协变量。成功返回 SkillResult，失败返回 None。"""
            drug = (state.plausibility_bounds or {}).get('drug_identified', '').lower()
            covariates = state.data_profile.get('columns', [])
            available = {c.upper() for c in covariates}

            MANDATORY_COVS = {
                'tobramycin': [('CLCR', 'CL', 0.75), ('WT', 'V1', 1.0)],
                'gentamicin': [('CLCR', 'CL', 0.75), ('WT', 'V1', 1.0)],
                'amikacin': [('CLCR', 'CL', 0.75), ('WT', 'V1', 1.0)],
                'warfarin': [('WT', 'CL', 0.75), ('WT', 'V', 1.0)],
                'theophylline': [('WT', 'CL', 0.75), ('WT', 'V', 1.0)],
            }

            mandatory = None
            for drug_key, cov_list in MANDATORY_COVS.items():
                if drug_key in drug:
                    mandatory = cov_list
                    break
            if not mandatory:
                return None

            new_code = state.current_code
            added = []
            for cov, param, exp in mandatory:
                if cov.upper() not in available:
                    continue
                # 检查是否已经加了该协变量
                if re.search(rf'\({cov}\s*/', new_code, re.IGNORECASE):
                    continue
                new_code = _add_power_covariate(new_code, cov, param, exp)
                if new_code != state.current_code:
                    added.append(f"{cov} on {param} (power {exp})")
                    state.current_code = new_code

            if added:
                state.failed_strategies.append(
                    f"PHASE2_ADDED_COVARIATES_{'_'.join(added)}")
                return SkillResult(updates={'current_code': new_code})
            return None

        def _apply_alag_if_missing(self, state):
            """oral 给药且缺少 ALAG1 时，注入 ALAG1。成功返回 SkillResult。"""
            route = state.data_profile.get('route', 'oral').lower()
            if route != 'oral':
                return None
            if 'ALAG' in state.current_code.upper():
                return None
            # 只有当 Ka 撞边界时才注入
            boundary_thetas = _identify_boundary_theta(
                state.last_run.get('full_output', ''))
            theta_map = _extract_theta_param_map(state.current_code)
            boundary_names = [theta_map.get(i, '') for i in boundary_thetas]
            if 'Ka' not in boundary_names:
                return None

            new_code = _inject_alag1(state.current_code)
            if new_code == state.current_code:
                return None

            state.failed_strategies.append("PHASE2_ADDED_ALAG1")
            return SkillResult(updates={'current_code': new_code})
        '''

    def validate(self, output, state) -> ValidationResult:
        """房室数量不得变化。"""
        new_code = output.updates.get('current_code', '')
        if not new_code or not state.locked_compartments:
            return ValidationResult(ok=True)

        m = re.search(r'ADVAN(\d+)', new_code, re.IGNORECASE)
        if not m:
            return ValidationResult(ok=True)
        advan = int(m.group(1))
        advan_compartments = {1: 1, 2: 1, 3: 2, 4: 2, 11: 3, 12: 3}.get(advan)
        if advan_compartments and advan_compartments != state.locked_compartments:
            return ValidationResult(
                ok=False,
                reason=f"ADVAN{advan} implies {advan_compartments} compartments; "
                       f"locked to {state.locked_compartments}",
                corrective_instruction=(
                    f"Keep the model at {state.locked_compartments}-compartment. "
                    f"Different compartment counts are explored by separate parallel runs."
                ),
            )
        return ValidationResult(ok=True)

'''
def _count_omega(code: str) -> int:
    """
    统计 $OMEGA 块中实际的 OMEGA（随机效应）数量。

    规则：
      - BLOCK(n) 形式：返回 n（不论块体有几行）
      - 非 BLOCK 形式（DIAGONAL / 默认）：逐行统计数值个数
      - 跳过注释（; 后内容）和空行
    """
    m = re.search(r'\$OMEGA\s*(.*?)(?=\n\s*\$|\Z)', code,
                  re.DOTALL | re.IGNORECASE)
    if not m:
        return 0

    body = m.group(1)

    # BLOCK(n) 形式：直接返回维度 n
    block_m = re.search(r'BLOCK\s*\(\s*(\d+)\s*\)', body, re.IGNORECASE)
    if block_m:
        return int(block_m.group(1))

    # DIAGONAL / 默认形式：逐行统计数值
    count = 0
    for line in body.split('\n'):
        line = line.split(';')[0].strip()
        if not line:
            continue
        nums = re.findall(r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', line)
        count += len(nums)
    return count
'''

def _get_boundary_thetas(state) -> list:
    """
    统一的 boundary_thetas 获取入口。

    优先从 state.last_run['full_output'] 解析；
    若为空，尝试从 state.cov_failure_diagnosis 中恢复
    （GuardrailCritic 已在上一轮写入）。
    """
    lst_output = state.last_run.get('full_output', '') or ''
    boundary = _identify_boundary_theta(lst_output)
    if boundary:
        return boundary

    # 回退：从 GuardrailCritic 的诊断结果中恢复
    diag = getattr(state, 'cov_failure_diagnosis', None)
    if diag and diag.get('type') in ('STRUCTURAL_BOUNDARY',
                                     'ERROR_MODEL_BOUNDARY'):
        return list(diag.get('params', []))
    return []

'''
def _identify_boundary_theta(lst_output: str) -> list:
    """
    从 .lst 的 GRADIENT 行中提取 THETA 索引（1-based），
    判据：gradient 值严格等于 0.0（即参数贴在边界上）。

    与 phase3_reduce.py / orchestrator.GuardrailCritic 中的同名函数逻辑一致。
    取最后一个 GRADIENT 行（最终迭代的梯度）。
    """
    if not lst_output:
        return []
    lines = re.findall(
        r'GRADIENT:\s+((?:[+-]?\s*\d+\.\d+E[+-]\d+\s*)+)',
        lst_output, re.IGNORECASE)
    if not lines:
        return []
    values = re.findall(r'[+-]?\d+\.\d+E[+-]\d+', lines[-1])
    return [i for i, v in enumerate(values, start=1) if float(v) == 0.0]
'''
def _widen_theta_bounds(code: str, boundary_indices: list, state=None) -> str:
    """
    将指定 THETA 索引的 lower/upper 边界放宽 ×10。

    boundary_indices: 1-based THETA 索引列表
    匹配格式: (lower, initial, upper) 每个值可含小数、科学计数法、负号
    """

    # ★ 识别误差模型 THETA
    error_thetas = set()
    # error_match = re.search(r'(\$ERROR[^\$]*?)(?=\n\s*\$|\Z)', code, re.DOTALL | re.IGNORECASE)
    # if error_match:
    #     for m in re.finditer(r'THETA\((\d+)\)', error_match.group(1)):
    #         error_thetas.add(int(m.group(1)))

    error_match = re.search(r'(\$ERROR[^\$]*?)(?=\n\s*\$|\Z)',
                            code, re.DOTALL | re.IGNORECASE)
    if error_match:
        for m in re.finditer(r'THETA\s*\(\s*(\d+)\s*\)',
                             error_match.group(1), re.IGNORECASE):
            error_thetas.add(int(m.group(1)))

    # 误差模型 THETA 不扩边界，标记为需要简化
    error_boundary = [i for i in boundary_indices if i in error_thetas]
    structural_boundary = [i for i in boundary_indices if i not in error_thetas]

    if error_boundary and state is not None:
        state.error_model_needs_simplification = True
        print(f"  [PHASE2-BOUND] Error model THETA {error_boundary} at boundary — "
              f"flagged for simplification (NOT widening)")

    # ★ 关键：只对结构参数扩边界
    boundary_indices = structural_boundary
    if not boundary_indices:
        return code

    # ★ 从 state 读取/初始化扩边界计数
    if state is not None:
        widen_count = getattr(state, 'theta_widen_count', None)
        if widen_count is None:
            state.theta_widen_count = {}
            widen_count = state.theta_widen_count
    else:
        widen_count = {}

    MAX_WIDEN_PER_THETA = 2  # 每个 THETA 最多扩 2 次

    # 逐个 THETA 行处理：匹配 $THETA 块内每行
    theta_block_match = re.search(
        r'(\$THETA[^\$]*?)(?=\n\s*\$|\Z)', code,
        re.DOTALL | re.IGNORECASE)
    if not theta_block_match:
        return code

    theta_block = theta_block_match.group(1)
    lines = theta_block.split('\n')
    new_lines = []
    theta_idx = 0

    for line in lines:
        # 跳过 $THETA 行本身
        if line.strip().upper().startswith('$THETA'):
            new_lines.append(line)
            continue

        # 分离注释（; 后内容）
        if ';' in line:
            head, comment = line.split(';', 1)
            comment_suffix = ' ; ' + comment.rstrip()
        else:
            head, comment_suffix = line, ''

        # 匹配 (lower, initial, upper)
        m = re.match(
            r'(\s*)\(([^,]+),\s*([^,]+),\s*([^)]+)\)(.*)$', head)
        if not m:
            new_lines.append(line)
            continue

        theta_idx += 1
        if theta_idx not in boundary_indices:
            new_lines.append(line)
            continue

        # ★ 检查是否已达扩边界次数上限
        current_count = widen_count.get(theta_idx, 0)
        if current_count >= MAX_WIDEN_PER_THETA:
            print(f"  [PHASE2-BOUND-SKIP] THETA({theta_idx}) 已扩 "
                  f"{current_count} 次，跳过（上限 {MAX_WIDEN_PER_THETA}）")
            new_lines.append(line)
            continue

        # 放宽边界
        indent, lo_s, init_s, hi_s, tail = m.groups()
        try:
            lo = float(lo_s)
            hi = float(hi_s)
            new_lo = lo * 10 if lo > 0 else lo * 10
            new_hi = hi * 10 if hi > 0 else hi * 10
            # 若 new_lo 仍为正，降低 10 倍更保险
            if lo > 0:
                new_lo = lo / 10
            new_line = (f"{indent}({new_lo:g}, {init_s.strip()}, "
                        f"{new_hi:g}){tail.rstrip()}{comment_suffix}")
            print(f"  [PHASE2-BOUND] THETA({theta_idx}) 边界 "
                  f"({lo_s},{init_s},{hi_s}) → "
                  f"({new_lo:g},{init_s},{new_hi:g})")
        except ValueError:
            new_line = line

        new_lines.append(new_line)

    new_block = '\n'.join(new_lines)
    return code.replace(theta_block, new_block, 1)

def _add_power_covariate(code: str, cov: str, param: str, exponent: float) -> str:
    """
    在 $PK 中为指定参数添加幂函数协变量。
    例如 _add_power_covariate(code, 'WT', 'CL', 0.75):
        TVCL = THETA(1)  →  TVCL = THETA(1) * (WT/70)**0.75
    """
    # 匹配 TVCL = THETA(n) 或 TVCL = THETA(n) * (....)
    # 只处理尚未添加该协变量的情况
    patterns = [
        rf'(TV{param}\s*=\s*THETA\((\d+)\))(?!\s*\*\s*\({re.escape(cov)})',
        rf'(TV{param}\s*=\s*THETA\((\d+)\)\s*\*\s*EXP\s*\(ETA)',
        rf'(\b{param}\s*=\s*THETA\((\d+)\))(?!\s*\*\s*\({re.escape(cov)})',  # 无 TV 前缀
    ]

    # 别名展开
    param_aliases = [param]
    if param.upper() in ('V1', 'V'):
        param_aliases = ['V1', 'V', 'TVV1', 'TVV']
    elif param.upper() == 'CL':
        param_aliases = ['CL', 'TVCL', 'CLPOP']

    for alias in param_aliases:
        # 模式 1：包含 EXP(ETA) 的完整形式
        m = re.search(
            rf'((?:TV)?{re.escape(alias)}\s*=\s*THETA\(\d+\)\s*\*\s*EXP\s*\(ETA[^)]*\))',
            code, re.IGNORECASE)
        if m:
            old = m.group(1)
            # 在 THETA(...) 后、EXP 前插入协变量
            theta_part = re.match(r'.*THETA\(\d+\)', old, re.IGNORECASE).group(0)
            eta_part = old[len(theta_part):].strip()
            new = f'{theta_part} * ({cov}/70)**{exponent} {eta_part}'
            return code.replace(old, new, 1)

        # 模式 2：简单形式 TVX = THETA(n)
        m = re.search(
            rf'((?:TV)?{re.escape(alias)}\s*=\s*THETA\(\d+\))(?!\s*\*\s*\()',
            code, re.IGNORECASE)
        if m:
            old = m.group(1)
            new = f'{old} * ({cov}/70)**{exponent}'
            return code.replace(old, new, 1)

    '''
    for pat in patterns:
        m = re.search(pat, code, re.IGNORECASE)
        if m:
            # 使用居中幂函数形式
            if pat == patterns[0]:
                # TVCL = THETA(1)  →  TVCL = THETA(1) * (WT/70)**0.75
                old = m.group(1)
                new = f'{old} * ({cov}/70)**{exponent}'
                code = code.replace(old, new, 1)
            else:
                # TVCL = THETA(1) * EXP(ETA(...))  →  TVCL = THETA(1) * (WT/70)**0.75 * EXP(ETA(...))
                theta_part = m.group(1)
                eta_part = m.group(2)
                new = f'{theta_part} * ({cov}/70)**{exponent} {eta_part}'
                code = code.replace(m.group(0), new, 1)
            break
    '''

    return code

def _inject_alag1(code: str) -> str:
    """
    在 $PK 中插入 ALAG1 及其对应的 $THETA / $OMEGA 条目。

    THETA / ETA 编号使用"当前最大编号 + 1"，避免与已有条目冲突。
    $OMEGA 为 BLOCK 形式时跳过（由 Phase 4 统一处理），避免块体错位。
    """
    if 'ALAG' in code.upper():
        return code

    # 1. 计算当前 THETA / ETA 最大编号
    theta_nums = [int(m.group(1))
                  for m in re.finditer(r'THETA\s*\(\s*(\d+)\s*\)', code)]
    eta_nums = [int(m.group(1))
                for m in re.finditer(r'ETA\s*\(\s*(\d+)\s*\)', code)]
    new_theta = (max(theta_nums) + 1) if theta_nums else 1
    new_eta = (max(eta_nums) + 1) if eta_nums else 1

    # 2. $PK 追加 ALAG1
    pk_m = re.search(r'(\$PK[\s\S]*?)(?=\n\s*\$[A-Z]|\Z)', code, re.IGNORECASE)
    if not pk_m:
        return code
    pk_block = pk_m.group(1).rstrip()
    new_pk = pk_block + (
        f"\nALAG1 = THETA({new_theta}) * EXP(ETA({new_eta}))"
        f"  ; Phase2 auto-added ALAG"
    )
    code = code.replace(pk_m.group(1), new_pk, 1)

    # 3. $THETA 追加
    theta_m = re.search(r'(\$THETA[\s\S]*?)(?=\n\s*\$[A-Z]|\Z)',
                        code, re.IGNORECASE)
    if theta_m:
        new_theta_block = theta_m.group(1).rstrip() + \
            f"\n(0, 0.5, 2)   ; {new_theta} ALAG1 (h)"
        code = code.replace(theta_m.group(1), new_theta_block, 1)

    # 4. $OMEGA 追加（仅 DIAGONAL 形式）
    omega_m = re.search(r'(\$OMEGA[\s\S]*?)(?=\n\s*\$[A-Z]|\Z)',
                        code, re.IGNORECASE)
    if omega_m:
        omega_body = omega_m.group(1)
        if 'BLOCK' in omega_body.upper():
            # BLOCK 形式跳过，交由 Phase 4 处理
            print(f"  [PHASE2-ALAG] $OMEGA is BLOCK — skipping OMEGA entry "
                  f"for ETA({new_eta}), will be handled in Phase 4")
        else:
            new_omega_block = omega_body.rstrip() + \
                f"\n0.1   ; ETA({new_eta}) ALAG1"
            code = code.replace(omega_body, new_omega_block, 1)

    return code



