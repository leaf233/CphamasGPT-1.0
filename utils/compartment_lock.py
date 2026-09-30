"""
utils/compartment_lock.py

房室锁定工具：ADVAN ↔ 房室数 ↔ route 的唯一映射。
所有 Skill / Guardrail 都通过此模块校验房室一致性。
"""

from __future__ import annotations
import re
from typing import Optional, Tuple

# ADVAN → 房室数（固定 NONMEM 事实）
ADVAN_TO_COMPARTMENTS = {
    1:  1,   # IV bolus 1-cmt
    2:  1,   # Oral 1-cmt
    3:  2,   # IV bolus 2-cmt
    4:  2,   # Oral 2-cmt
    11: 3,   # IV bolus 3-cmt
    12: 3,   # Oral 3-cmt
}

# (route, compartments) → 推荐的 ADVAN
COMPARTMENTS_TO_ADVAN = {
    ('iv',   1): 1, ('oral', 1): 2,
    ('iv',   2): 3, ('oral', 2): 4,
    ('iv',   3): 11, ('oral', 3): 12,
}


def parse_advan(code: str) -> Optional[int]:
    """从控制流中提取 ADVAN 编号。"""
    m = re.search(r'ADVAN\s*(\d+)', code, re.IGNORECASE)
    return int(m.group(1)) if m else None


def get_actual_compartments(code: str) -> Optional[int]:
    """返回控制流的实际房室数；无法判断时返回 None。"""
    advan = parse_advan(code)
    if advan is None:
        return None
    return ADVAN_TO_COMPARTMENTS.get(advan)


def get_required_advan(route: str, compartments: int) -> Optional[int]:
    """根据 route 与目标房室数返回应有的 ADVAN。"""
    return COMPARTMENTS_TO_ADVAN.get((route.lower(), compartments))


def validate_lock(code: str, locked_compartments: int
                  ) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    校验代码的房室数是否与锁定值一致。

    返回：(ok, actual_compartments, advan_num)
      ok=True  → 一致
      ok=False → 不一致（actual 与 advan 供调用方使用）
    """
    advan = parse_advan(code)
    if advan is None:
        # 没有 ADVAN（可能用 $DES）→ 保守放行，交由 advan_validator 处理
        return True, None, None
    actual = ADVAN_TO_COMPARTMENTS.get(advan)
    if actual is None:
        # 未识别的 ADVAN（5-10/13，$DES 模式）→ 保守放行
        return True, None, advan
    return (actual == locked_compartments), actual, advan


def build_correction_prompt(code: str, locked_compartments: int,
                            route: str) -> str:
    """
    生成强硬纠正提示：告知 LLM 必须用哪个 ADVAN。
    """
    advan = parse_advan(code)
    actual = ADVAN_TO_COMPARTMENTS.get(advan) if advan else None
    required_advan = get_required_advan(route, locked_compartments)

    return (
        f"\n\n{'!'*70}\n"
        f"🚫 HARD CONSTRAINT VIOLATION — REGENERATE IMMEDIATELY\n"
        f"{'!'*70}\n"
        f"Your generated code used ADVAN{advan} "
        f"({actual}-compartment).\n"
        f"This run is HARD-LOCKED to {locked_compartments}-compartment "
        f"(specified by --compartments {locked_compartments}).\n"
        f"\n"
        f"⚠️  You are NOT allowed to override the lock based on:\n"
        f"   - Sample size concerns (N=12, N=32, etc.)\n"
        f"   - Parameter count concerns\n"
        f"   - Your own pharmacokinetic judgment\n"
        f"   - Data-driven inference\n"
        f"\n"
        f"REQUIRED: Use $SUBROUTINE ADVAN{required_advan} TRANS4 or TRANS2\n"
        f"          (for {route.upper()} {locked_compartments}-compartment).\n"
        f"\n"
        f"REGENERATE NOW with ADVAN{required_advan}. "
        f"Do NOT add explanatory comments about parameter count or N.\n"
        f"{'!'*70}\n"
    )