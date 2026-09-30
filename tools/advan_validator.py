"""
tools/advan_validator.py

ADVAN 语法与 ADVAN/TRANS 组合校验。

提供：
1. validate(code)          → (ok, error_message)
   - 检查 ADVAN4/5/6 的 S2/S3 定义
   - 检查 V1 命名
2. check_advan_trans(code) → error_message or None
   - 检查 (ADVAN, TRANS) 组合合法性
   - 检查该组合所需的 $PK 参数是否齐备

注意：Phase 2 房室升级已移除。本工具只做语法与组合校验，
     不提出任何房室升级建议。

不调用 LLM。
"""

from __future__ import annotations
import re
from typing import Optional, Tuple

# ADVAN -> 房室数量（固定 NONMEM 事实）
_ADVAN_COMPARTMENTS = {1: 1, 2: 1, 3: 2, 4: 2, 11: 3, 12: 3}

# (ADVAN, TRANS) -> 必需 $PK 参数（固定 NONMEM 事实）
_ADVAN_TRANS_PARAMS = {
    (1, 1):  ['K', 'V'],
    (1, 2):  ['CL', 'V'],
    (2, 1):  ['K', 'V', 'KA'],
    (2, 2):  ['CL', 'V', 'KA'],
    (3, 1):  ['K', 'K12', 'K21'],
    (3, 4):  ['CL', 'V1', 'Q', 'V2'],
    (4, 1):  ['K', 'K23', 'K32', 'KA'],
    (4, 4):  ['CL', 'V2', 'Q', 'V3', 'KA'],
    (11, 1): ['K', 'K12', 'K21', 'K13', 'K31'],
    (11, 4): ['CL', 'V1', 'Q2', 'V2', 'Q3', 'V3'],
    (12, 1): ['K', 'K23', 'K32', 'K24', 'K42', 'KA'],
    (12, 4): ['CL', 'V2', 'Q3', 'V3', 'Q4', 'V4', 'KA'],
}


class AdvanValidator:
    """ADVAN 语法与组合校验器。"""

    # ---- 主入口 ---------------------------------------------------------- #

    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        校验 ADVAN 语法。返回 (is_valid, error_message)。
        """
        advan_match = re.search(r'ADVAN(\d+)', code, re.IGNORECASE)
        if not advan_match:
            return True, None  # 无 ADVAN 则交给 NONMEM 处理

        advan_num = int(advan_match.group(1))

        # ADVAN4/5/6（2 室）语法校验
        if advan_num in [4, 5, 6]:
            error = self._check_2compartment_syntax(code, advan_num)
            if error:
                return False, error

        return True, None

    def check_advan_trans(self, code: str) -> Optional[str]:
        """
        校验 (ADVAN, TRANS) 组合及必需 $PK 参数。返回错误信息或 None。
        """
        m = re.search(
            r'\$SUBROUTINES?\s+ADVAN(\d+)\s+TRANS(\d+)',
            code, re.IGNORECASE,
        )
        if not m:
            return None

        advan_num = int(m.group(1))
        trans_num = int(m.group(2))
        key = (advan_num, trans_num)
        required = _ADVAN_TRANS_PARAMS.get(key)

        if required is None:
            known_trans = sorted(
                t for (a, t) in _ADVAN_TRANS_PARAMS if a == advan_num)
            if known_trans and trans_num not in known_trans:
                valid = ', '.join(f"TRANS{t}" for t in known_trans)
                return (f"REJECTED: ADVAN{advan_num} TRANS{trans_num} is not a "
                        f"valid combination. ADVAN{advan_num} supports: {valid}.")
            return None  # 通用 ADVAN，跳过

        pk_match = re.search(r'\$PK(.*?)(?=\n\s*\$|\Z)',
                             code, re.DOTALL | re.IGNORECASE)
        pk_block = pk_match.group(1) if pk_match else ''
        missing = [
            p for p in required
            if not re.search(rf'\b{re.escape(p)}\s*=', pk_block, re.IGNORECASE)
        ]
        if missing:
            return (f"REJECTED: ADVAN{advan_num} TRANS{trans_num} requires: "
                    f"{', '.join(required)}. Missing: {', '.join(missing)}.")
        return None

    def advan_to_compartments(self, code: str) -> Optional[int]:
        """从 code 推断 ADVAN 对应房室数量。"""
        m = re.search(r'ADVAN(\d+)', code, re.IGNORECASE)
        if not m:
            return None
        return _ADVAN_COMPARTMENTS.get(int(m.group(1)))

    # ---- 内部方法 --------------------------------------------------------- #

    def _check_2compartment_syntax(
        self, code: str, advan_num: int
    ) -> Optional[str]:
        """ADVAN4/5/6（2 室）语法校验。"""
        pk_match = re.search(r'\$PK(.*?)(?=\$|$)', code,
                             re.DOTALL | re.IGNORECASE)
        if not pk_match:
            return None

        pk_content = pk_match.group(1)
        has_s3 = re.search(r'S3\s*=', pk_content, re.IGNORECASE)
        has_v3 = re.search(r'\bV3\s*=', pk_content, re.IGNORECASE)

        # 常见错误：定义 S3 但未定义 V3
        if has_s3 and not has_v3:
            return (
                f"ADVAN{advan_num} syntax error: S3 is defined but V3 is not. "
                f"For 2-compartment models with ADVAN4, use V1 and V2, "
                f"define S2 = V1, and do NOT define S3 unless V3 is explicitly defined."
            )

        # 常见错误：用 V 而非 V1
        has_v_not_v1 = re.search(r'\bV\s*=\s*TV', pk_content, re.IGNORECASE)
        has_v1 = re.search(r'\bV1\s*=\s*TV', pk_content, re.IGNORECASE)
        if advan_num == 4 and has_v_not_v1 and not has_v1:
            return (
                "ADVAN4 requires explicit V1 for central volume, not just 'V'. "
                "Use 'V1 = TVV1 * EXP(ETA(2))' instead of 'V = TVV * EXP(ETA(2))'."
            )

        return None