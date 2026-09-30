"""
utils/safe_numeric.py

从解析结果（AI 或正则）中安全提取数值。
AI 有时会对不确定项显式输出 null；正则回退路径也偶有缺值。
所有 float(x) 调用点应统一经过此模块，避免 float(None) 崩溃。
"""

from typing import Any, Iterable, List, Optional


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    安全转换为 float。
    - None → default
    - 空字符串 → default
    - 非数值字符串 → default
    - NaN / inf → default（由调用方决定是否允许）
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default if default is None else float(default)
    if isinstance(value, (int, float)):
        v = float(value)
        if v != v or v in (float('inf'), float('-inf')):  # NaN / inf
            return default
        return v
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {'null', 'none', 'na', 'n/a', '.'}:
            return default
        try:
            v = float(s)
            if v != v or v in (float('inf'), float('-inf')):
                return default
            return v
        except ValueError:
            return default
    return default


def extract_matrix_diagonal_values(matrix_entries: Iterable[dict],
                                   default: Optional[float] = None) -> List[float]:
    """
    从 OMEGA/SIGMA 条目中提取对角线元素的数值。
    条目形如 {"row": i, "col": j, "value": float|None}。
    - 只保留 row == col 的对角线
    - 丢弃 value 为 None 或非数值的项
    - 若条目缺少 row/col（如 AI 只给一个扁平列表），全部接受
    """
    diag: List[float] = []
    for entry in matrix_entries or []:
        if not isinstance(entry, dict):
            continue
        row = entry.get('row')
        col = entry.get('col')
        # 若有 row/col 且非对角线，跳过
        if row is not None and col is not None and row != col:
            continue
        v = safe_float(entry.get('value'), default=safe_float(entry.get('estimate'), default))
        if v is None:
            continue
        diag.append(v)
    return diag


def extract_theta_values(theta_entries: Iterable[dict]) -> List[float]:
    """提取 THETA 数值，跳过 None。"""
    out: List[float] = []
    for entry in theta_entries or []:
        if not isinstance(entry, dict):
            continue
        v = safe_float(entry.get('value'), default=safe_float(entry.get('estimate')))
        if v is None:
            continue
        out.append(v)
    return out