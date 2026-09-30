"""
skills/phase5_covariate_scm.py

Phase 5: 确定性 SCM（前向选择 + 后向剔除）。

前向：ΔOFV < -3.84 (p<0.05, df=1) 且 cov_ok 且 numeric_safe
后向：ΔOFV on removal < 6.63 (p<0.01, df=1) 且 cov_ok 且 numeric_safe

结构冻结：ADVAN / $ERROR / $OMEGA / $ESTIMATION 不得变化。
Guardrail：phase5_structure_guard 自动恢复结构变更。
"""

from .base import Skill, SkillResult, CompartmentLockError
from modules.prompts.phase5_covariates import Phase5Covariates
from utils.safe_numeric import extract_matrix_diagonal_values
from utils.nonmem_utils import extract_code as _extract_code


class CovariateSCMSkill(Skill):
    name = 'phase5_covariate_scm'
    phase = [5]
    tools = ['cs_editor', 'nonmem_runner', 'nonmem_parser',
             'scm_runner', 'numeric_safety_gate',
             'phase5_structure_guard']
    prompt_template = 'phase5_covariate_scm'

    # ---- 主入口 ---------------------------------------------------------- #

    def run(self, state, llm, tools) -> SkillResult:
        scm = state.scm

        # 1) 初始化本轮 base
        if scm.round_base_ofv is None:
            scm.round_base_ofv = state.best_ofv
            scm.round_base_code = state.best_code

        # ★ 校验 base_code 是否房室合规
        if scm.round_base_code and state.locked_compartments:
            from utils.compartment_lock import validate_lock
            ok, actual, advan = validate_lock(
                scm.round_base_code, state.locked_compartments)
            if not ok:
                print(f"  [PHASE5-LOCK-FAIL] round_base_code is ADVAN{advan} "
                      f"({actual}-cmt) != locked {state.locked_compartments}-cmt.")
                return SkillResult(
                    done=True,
                    ok=False,
                    error=f"round_base_code violates compartment lock")

        # 2) 取下一个待测候选
        next_cand = self._next_candidate(state)
        if next_cand is None and scm.round_results:
            return self._complete_round(state, tools)
        if next_cand is None:
            return SkillResult(done=True)

        # 3) 生成 prompt（add 或 remove）
        if next_cand.get('mode') == 'remove':
            prompt = Phase5Covariates.generate_removal_prompt(
                iteration=state.iteration,
                current_code=scm.round_base_code,
                parsed_results=state.last_run.get('parsed_data', {}),
                target_covariate_name=next_cand['name'],
                remaining_covariates=[c['name'] for c in scm.confirmed],
                n_subjects=state.data_profile['n_subjects'],
            )
        else:
            prompt = Phase5Covariates.generate_prompt(
                iteration=state.iteration,
                current_code=scm.round_base_code,
                parsed_results=state.last_run.get('parsed_data', {}),
                shrinkage_data=state.last_run.get('shrinkage', []),
                available_covariates=[next_cand],
                current_covariates_in_model=[c['name'] for c in scm.confirmed],
                n_subjects=state.data_profile['n_subjects'],
            )

        # 4) LLM 生成
        response = llm.generate(prompt, model_type=state.model)
        new_code = _extract_code(response)

        # 5) Guardrail：结构冻结
        new_code = self._tool(tools, 'phase5_structure_guard').apply(
            new_code, base_code=scm.round_base_code)

        # 6) 执行 NONMEM + 解析
        run_result = self._tool(tools, 'nonmem_runner').execute(new_code, state)
        parsed = self._tool(tools, 'nonmem_parser').parse(run_result['lst_path'])

        # 7) 记录本轮结果
        self._record_result(state, next_cand, parsed, new_code)

        # 8) 回退到 round base，等待本轮所有候选测完
        return SkillResult(updates={
            'current_code': scm.round_base_code,
            'last_run': run_result,
            'scm': scm,
        })

    # ---- 候选选择 -------------------------------------------------------- #

    def _next_candidate(self, state):
        scm = state.scm
        if scm.mode == 'backward':
            for c in scm.confirmed:
                if c['name'] not in scm.round_tested:
                    return {'name': c['name'], 'mode': 'remove',
                            'covariate': c.get('covariate', ''),
                            'parameter': c.get('parameter', '')}
            return None

        confirmed = {c['name'] for c in scm.confirmed}
        for cand in _all_candidates(state):
            if cand['name'] in confirmed or cand['name'] in scm.round_tested:
                continue
            cand = dict(cand)
            cand['mode'] = 'add'
            return cand
        return None

    # ---- 记录与轮次完成 -------------------------------------------------- #

    def _record_result(self, state, cand, parsed, code):
        scm = state.scm
        ofv = parsed.get('objective_function')
        cov_ok = parsed.get('covariance_step', {}).get('successful', False)
        delta = (ofv - scm.round_base_ofv
                 if ofv is not None and scm.round_base_ofv is not None else None)

        # ★ 输出 SCM 测试详情
        label = "Backward" if cand.get('mode') == 'remove' else "Forward"
        delta_s = f"ΔOFV={delta:+.2f}" if delta is not None else "ΔOFV=N/A"
        cov_s = "cov=OK" if cov_ok else "cov=FAIL"
        print(f"  [SCM {label} Round {scm.round}] "
              f"Test '{cand['name']}': {delta_s}, {cov_s}")

        scm.round_tested.add(cand['name'])

        omega_values = extract_matrix_diagonal_values(
            parsed.get('parameter_estimates', {}).get('omega') or []
        )

        scm.round_results.append({
            'name': cand['name'],
            'covariate': cand.get('covariate', ''),
            'parameter': cand.get('parameter', ''),
            'mode': cand.get('mode', 'add'),
            'ofv': ofv, 'delta_ofv': delta, 'cov_ok': cov_ok,
            'code': code, 'iteration': state.iteration,
            'omega_values': omega_values
        })

    def _complete_round(self, state, tools) -> SkillResult:
        if state.scm.mode == 'backward':
            return self._complete_backward(state, tools)
        return self._complete_forward(state, tools)

    def _complete_forward(self, state, tools) -> SkillResult:
        scm = state.scm
        significant = [
            r for r in scm.round_results
            if r.get('delta_ofv') is not None
            and r['delta_ofv'] < -3.84 and r.get('cov_ok', False)
        ]

        # 数值安全 gate
        safe = []
        for r in significant:
            ok, reason = self._tool(tools, 'numeric_safety_gate').check(
                r.get('ofv'), r.get('omega_values'), r.get('avg_eta_shrinkage'))
            if ok:
                safe.append(r)
            else:
                print(f"  [SCM] '{r['name']}' met ΔOFV/cov but unsafe: {reason}")

        if not safe:
            # 无 winner → 若已有 confirmed，则进入 backward
            if scm.confirmed:
                scm.mode = 'backward'
                scm.round = 1
                scm.round_tested = set()
                scm.round_results = []
                scm.round_base_ofv = state.best_ofv
                scm.round_base_code = state.best_code
                return SkillResult(updates={'scm': scm})
            return SkillResult(done=True)

        winner = min(safe, key=lambda r: r['delta_ofv'])
        scm.confirmed.append({
            'name': winner['name'],
            'covariate': winner.get('covariate', ''),
            'parameter': winner.get('parameter', ''),
            'delta_ofv': winner['delta_ofv'],
            'code': winner['code'],
            'ofv': winner['ofv'],
            'iteration': winner['iteration'],
            'round': scm.round,
        })

        # 下一轮
        scm.round += 1
        scm.round_tested = set()
        scm.round_results = []
        scm.round_base_ofv = winner['ofv']
        scm.round_base_code = winner['code']

        return SkillResult(updates={
            'scm': scm,
            'current_code': winner['code'],
            'best_code': winner['code'],
            'best_ofv': winner['ofv'],
            'best_iteration': winner['iteration'],
        })

    def _complete_backward(self, state, tools) -> SkillResult:
        scm = state.scm
        removable = [
            r for r in scm.round_results
            if r.get('delta_ofv') is not None
            and r['delta_ofv'] < 6.63 and r.get('cov_ok', False)
        ]

        safe = []
        for r in removable:
            ok, _ = self._tool(tools, 'numeric_safety_gate').check(
                r.get('ofv'), r.get('omega_values'), r.get('avg_eta_shrinkage'))
            if ok:
                safe.append(r)

        if not safe:
            return SkillResult(done=True)

        loser = min(safe, key=lambda r: r['delta_ofv'])
        scm.confirmed = [c for c in scm.confirmed if c['name'] != loser['name']]
        scm.eliminated.append({
            'name': loser['name'], 'delta_ofv': loser['delta_ofv'],
            'iteration': loser['iteration'], 'round': scm.round,
        })

        if not scm.confirmed:
            return SkillResult(done=True)

        scm.round += 1
        scm.round_tested = set()
        scm.round_results = []
        scm.round_base_ofv = loser['ofv']
        scm.round_base_code = loser['code']

        return SkillResult(updates={
            'scm': scm,
            'current_code': loser['code'],
            'best_code': loser['code'],
            'best_ofv': loser['ofv'],
            'best_iteration': loser['iteration'],
        })


# ---- 辅助函数 ------------------------------------------------------------ #

def _all_candidates(state):
    """从 state.data_profile 生成 covariate × parameter 候选列表。"""
    info = state.data_profile.get('covariate_info', {})
    candidates = []

    for cov, meta in info.items():
        cov_type = meta.get('type', 'continuous')
        for param in ('CL', 'V1'):
            name = f"{cov} on {param}"
            priority = _priority_score(cov, param)
            if cov_type == 'categorical':
                # ★ 分类变量：必须用 IF 语句，不能用幂/线性
                example = f"IF ({cov}.EQ.1) TV{param} = TV{param} * THETA(X)"
                # model_type = 'categorical_if'
                candidates.append({
                    'name': name,
                    'covariate': cov,
                    'parameter': param,
                    'model_type': 'categorical_if',
                    'example': example,
                    'priority': priority,
                })
            else:
                model_type = meta.get('suggested_model', 'linear')
                candidates.append({
                    'name': name,
                    'covariate': cov,
                    'parameter': param,
                    'model_type': model_type,
                    'example': None,
                    'priority': priority,
                })

                '''
                example = None
                candidates.append({
                    'name': name, 'covariate': cov, 'parameter': param,
                    'model_type': model_type, 'example': example,
                })
                '''
    '''
    for cov, meta in info.items():
        model_type = meta.get('suggested_model', 'linear')
        for param in ('CL', 'V1'):
            name = f"{cov} on {param}"
            priority = _priority_score(cov, param)
            candidates.append({
                'name': name, 'covariate': cov, 'parameter': param,
                'model_type': model_type,
                'priority': priority,
            })
    '''

    # 按优先级排序（低=优先测试）
    candidates.sort(key=lambda c: c['priority'])
    return candidates


def _priority_score(cov, param):
    """pharmacology-informed 优先级。数字越小越优先。"""
    """
    Pharmacology-informed 优先级。数字越小越优先。
    通用规则（不绑定具体药物）：
      1. 体表面积指标（WT/BW/BSA）优先于 CL，其次 V1
      2. 肾功能指标优先（肾清除药物类）
      3. 年龄次之
      4. 性别最末
    """

    cov_u = cov.upper()
    param_u = param.upper()

    '''
    # ★ 药物类别加权（非硬编码）
    drug_boost = 0
    if drug_class:
        dc = drug_class.lower()
        if 'warfarin' in dc and cov_u in ('WT', 'BW', 'BSA'):
            drug_boost = -1   # 提升优先级
        elif ('aminoglycoside' in dc or 'gentamicin' in dc
              or 'tobramycin' in dc) and cov_u in ('CLCR', 'CRCL'):
            drug_boost = -1
    '''

    # 体表面积指标 → CL 优先，V1 次之
    if cov_u in ('WT', 'WEIGHT', 'BW', 'BSA'):
        return 0 if param_u == 'CL' else 1
    # 肾功能指标 → CL 优先
    if cov_u in ('CLCR', 'CRCL', 'GFR', 'EGFR', 'SCR', 'CREAT'):
        return 0 if param_u == 'CL' else 2
    # 年龄
    if cov_u in ('AGE',):
        return 2
    # 性别
    if cov_u in ('SEX', 'GENDER', 'RACE'):
        return 4
    # 其他
    return 5


    '''
    cov_u = cov.upper()
    # 药物类别关键 covariate 优先
    if cov_u in ('WT', 'WEIGHT', 'BW'):
        return 0 if param == 'CL' else 1
    if cov_u in ('CLCR', 'CRCL', 'GFR', 'EGFR'):
        return 0
    if cov_u in ('AGE',):
        return 2
    if cov_u in ('SEX', 'GENDER'):
        return 3
    return 5
    '''
