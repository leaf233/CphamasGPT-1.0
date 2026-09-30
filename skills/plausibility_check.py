"""
skills/plausibility_check.py

封装 _check_plausibility：
- 解析 THETA 映射（param_name → index）
- 与 state.plausibility_bounds 的 min/max 对比
- 输出 violations / plausibility_score，写入 state.plausibility_report
"""
from utils.safe_numeric import safe_float
from .base import Skill, SkillResult


class PlausibilityCheckSkill(Skill):
    name = 'plausibility_check'
    phase = [1, 2, 3, 4, 5]
    tools = []
    prompt_template = ''

    def run(self, state, llm, tools) -> SkillResult:
        report = _check(state)
        return SkillResult(updates={'plausibility_report': report})


def _check(state) -> dict:
    import re
    bounds = state.plausibility_bounds or {}
    params_bounds = bounds.get('parameters', {})
    if not params_bounds:
        return {}

    parsed = state.last_run.get('parsed_data', {})
    theta_list = parsed.get('parameter_estimates', {}).get('theta', [])
    if not theta_list:
        return {}

    theta_map = _extract_theta_map(state.current_code)
    violations, checked, score = [], [], 100

    for item in theta_list:
        '''
        idx = item.get('index')
        est = item.get('value') or item.get('estimate')
        if idx is None or est is None:
            continue
        '''
        idx = item.get('index')
        est = safe_float(item.get('value'))
        if est is None:
            est = safe_float(item.get('estimate'))
        if idx is None or est is None:
            continue
        name = theta_map.get(idx)
        if not name or name not in params_bounds:
            continue
        b = params_bounds[name]
        lo, hi = b.get('min'), b.get('max')
        checked.append(f"{name}={est:.3g}")

        if hi is not None and est > hi:
            fold = est / hi
            sev = 'SEVERE' if fold > 10 else 'MODERATE' if fold > 3 else 'MILD'
            score -= {'SEVERE': 30, 'MODERATE': 15, 'MILD': 5}[sev]
            violations.append(f"{sev}: {name}={est:.3g} exceeds max {hi} ({fold:.1f}x)")
        elif lo is not None and est < lo:
            fold = lo / max(est, 1e-9)
            sev = 'SEVERE' if fold > 10 else 'MODERATE' if fold > 3 else 'MILD'
            score -= {'SEVERE': 30, 'MODERATE': 15, 'MILD': 5}[sev]
            violations.append(f"{sev}: {name}={est:.3g} below min {lo} ({fold:.1f}x)")

    return {
        'violations': violations,
        'plausibility_score': max(0, score),
        'checked_parameters': checked,
        'drug': bounds.get('drug_identified', 'unknown'),
    }


def _extract_theta_map(code: str) -> dict:
    import re
    mapping = {}
    patterns = [
        (r'TVCL\s*=\s*THETA\((\d+)\)', 'CL'),
        (r'TVV1\s*=\s*THETA\((\d+)\)', 'V1'),
        (r'TVV\s*=\s*THETA\((\d+)\)',  'V1'),
        (r'TVQ\s*=\s*THETA\((\d+)\)',  'Q'),
        (r'TVV2\s*=\s*THETA\((\d+)\)', 'V2'),
        (r'TVKA?\s*=\s*THETA\((\d+)\)', 'Ka'),
    ]
    for pat, name in patterns:
        m = re.search(pat, code, re.IGNORECASE)
        if m:
            mapping.setdefault(int(m.group(1)), name)
    return mapping