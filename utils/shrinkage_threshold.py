"""
utils/shrinkage_threshold.py

根据样本量和 OMEGA 数量动态调整 shrinkage 阈值。

PKGPT 2.0 的 Phase 3 触发阈值固定为 95%，但论文承认这导致 85.2% 的
shrinkage 被漏检。本模块提供动态阈值，小样本时更早触发简化。
"""


def get_shrinkage_threshold(n_subjects: int, omega_count: int) -> float:
    """
    根据样本量和 OMEGA 数量返回 shrinkage 阈值（%）。

    规则：
      - N < 20（小样本，如 theo N=12）：OMEGA>1 时阈值降至 80%
      - 20 ≤ N < 50：阈值 85%
      - N ≥ 50：阈值 90%
      - OMEGA = 1（已是单 IIV）：统一 95%，避免误触发

    返回：阈值（浮点数，单位 %）
    """
    if omega_count <= 1:
        return 95.0
    if n_subjects < 20:
        return 80.0
    elif n_subjects < 50:
        return 85.0
    else:
        return 90.0