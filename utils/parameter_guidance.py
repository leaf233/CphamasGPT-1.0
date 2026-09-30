"""参数稳定化指导：从历史估计值生成建议的 THETA/OMEGA/SIGMA 边界。"""
'''
def build_parameter_stabilization_guidance(parameter_history: list) -> str:
    if not parameter_history:
        return ""
    last = parameter_history[-1]
    lines = [
        "Use the following estimates to tighten initial values and bounds",
        "for the NEXT iteration (do NOT change structural model unless required).",
        "",
    ]
    def _bounds(v):
        if v == 0: return (-1.0, 1.0)
        return (v * 0.5, v * 1.5) if abs(v) >= 1e-3 else (v * 0.1, v * 10.0)

    for label, names, vals in (
        ("THETA", last.get('theta_names', []), last.get('theta_vals', [])),
        ("OMEGA", last.get('omega_names', []), last.get('omega_vals', [])),
        ("SIGMA", last.get('sigma_names', []), last.get('sigma_vals', [])),
    ):
        if vals:
            lines.append(f"{label}:")
            for i, v in enumerate(vals):
                name = names[i] if i < len(names) else f"{label}({i+1})"
                lo, hi = _bounds(v)
                lines.append(f"  - {name}: estimate={v:.6g}, bounds ≈ [{lo:.6g}, {hi:.6g}]")
            lines.append("")
    return "\n".join(lines)
'''

"""
utils/parameter_guidance.py

从 parameter_history 生成"建议收紧边界"的文本，
供 Phase 1-4 的 prompt 使用。
"""


def build_parameter_stabilization_guidance(parameter_history: list) -> str:
    if not parameter_history:
        return ""
    last = parameter_history[-1]

    lines = [
        "=" * 70,
        "AUTO-GENERATED PARAMETER STABILIZATION GUIDANCE",
        "=" * 70,
        "Use the following estimates to tighten initial values and bounds",
        "for the NEXT iteration (do NOT change structural model unless required).",
        "",
    ]

    def _bounds(v):
        if v == 0:
            return -1.0, 1.0
        if 0 < abs(v) < 1e-3:
            return v * 0.1, v * 10.0
        return v * 0.5, v * 1.5

    for label, names_key, vals_key in [
        ("THETA (fixed effects)", 'theta_names', 'theta_vals'),
        ("OMEGA (IIV variances)", 'omega_names', 'omega_vals'),
        ("SIGMA (residual variances)", 'sigma_names', 'sigma_vals'),
    ]:
        names = last.get(names_key, []) or []
        vals = last.get(vals_key, []) or []
        if not vals:
            continue
        lines.append(f"{label}:")
        for i, v in enumerate(vals):
            name = names[i] if i < len(names) else f"{label.split()[0]}({i+1})"
            lo, hi = _bounds(v)
            lines.append(
                f"  - {name}: estimate={v:.6g}, recommended bounds ≈ "
                f"[{lo:.6g}, {hi:.6g}]")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)