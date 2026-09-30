"""
utils/cov_diagnosis.py

协方差失败诊断的独立模块。

从 agents/orchestrator.py 的 GuardrailCritic 抽出，用于打破
    skills/phase2_diagnose.py  →  agents/orchestrator.py
的反向依赖（skills 层不应 import agents 层）。

导出：
  - diagnose_covariance_failure(state)  协方差失败根因诊断
  - identify_boundary_theta(lst_output) 从 GRADIENT 行提取边界 THETA
  - extract_theta_param_map(code)       THETA 索引 → 参数名映射（薄封装）
"""
import re

from utils.nonmem_utils import extract_theta_param_map as _extract_theta_param_map


# --------------------------------------------------------------------------- #
# 1) THETA 索引 → 参数名映射
# --------------------------------------------------------------------------- #

def extract_theta_param_map(code: str) -> dict:
    """THETA 索引（1-based） → 参数名（CL/V1/V2/Q/Ka/...）。"""
    return _extract_theta_param_map(code)


# --------------------------------------------------------------------------- #
# 2) 从 .lst 的 GRADIENT 行提取边界 THETA
# --------------------------------------------------------------------------- #

def identify_boundary_theta(lst_output: str) -> list:
    """
    从 .lst 的 GRADIENT 行中提取 gradient 严格等于 0.0 的 THETA 索引。

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


# --------------------------------------------------------------------------- #
# 3) 协方差失败根因诊断
# --------------------------------------------------------------------------- #

_STRUCTURAL_PK = {
    'CL', 'V1', 'V2', 'V3', 'V', 'Q', 'Q2', 'Q3', 'Q4', 'KA', 'Ka'
}


def diagnose_covariance_failure(state) -> None:
    """
    当协方差步失败时，识别具体失败原因并写入 state.cov_failure_diagnosis。

    覆盖三类根因（按优先级）：
      A. STRUCTURAL_BOUNDARY  : 结构参数（CL/V/Ka/Q/V2/V3）触边界
      B. ERROR_MODEL_BOUNDARY : 误差模型参数（Additive）触下界
      C. OMEGA_COLLAPSE       : 所有 ETA 方差塌缩

    无论是否命中，都会更新 state.cov_failure_diagnosis（未命中则为 None）。
    """
    parsed = state.last_run.get('parsed_data') or {}
    cov = parsed.get('covariance_step') or {}

    # 未失败 → 清空诊断
    if not cov.get('attempted') or cov.get('successful'):
        state.cov_failure_diagnosis = None
        return

    lst_output = state.last_run.get('full_output', '')
    boundary_indices = identify_boundary_theta(lst_output)
    theta_map = extract_theta_param_map(state.current_code)

    boundary_params = [
        (idx, theta_map.get(idx, f"THETA({idx})"))
        for idx in boundary_indices
    ]

    diagnosis = None

    # ---- A / B：边界 THETA 分类 ----------------------------------------- #
    if boundary_params:
        structural_hits = [(i, n) for i, n in boundary_params
                           if n in _STRUCTURAL_PK]
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

    # ---- C：OMEGA 塌陷（仅当 A/B 未命中） ------------------------------- #
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
        print(f"  [COV-DIAGNOSIS] {diagnosis['type']}: "
              f"{diagnosis['message'][:120]}...")