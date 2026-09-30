"""
guardrails/advan_trans_validity.py

ADVAN/TRANS 组合合法性守卫。

职责：
1. 校验 (ADVAN, TRANS) 是否为合法组合
2. 校验该组合所需的 $PK 参数是否在 $PK 块中齐备
3. 若使用通用 ADVAN（5-10/13，$DES 模式），跳过参数检查

注意：Phase 2 房室升级已移除。本守卫只做合法性校验，
     不提出任何房室升级建议。

封装的原始逻辑：NONMEMOptimizer._check_advan_trans_validity

不调用 LLM。
"""

from __future__ import annotations
import re
from typing import Optional

# # (ADVAN, TRANS) -> 必需 $PK 参数（固定 NONMEM 事实）
# _ADVAN_TRANS_PARAMS = {
#     (1, 1):  ['K', 'V'],
#     (1, 2):  ['CL', 'V'],
#     (2, 1):  ['K', 'V', 'KA'],
#     (2, 2):  ['CL', 'V', 'KA'],
#     (3, 1):  ['K', 'K12', 'K21'],
#     (3, 4):  ['CL', 'V1', 'Q', 'V2'],
#     (4, 1):  ['K', 'K23', 'K32', 'KA'],
#     (4, 4):  ['CL', 'V2', 'Q', 'V3', 'KA'],
#     (11, 1): ['K', 'K12', 'K21', 'K13', 'K31'],
#     (11, 4): ['CL', 'V1', 'Q2', 'V2', 'Q3', 'V3'],
#     (12, 1): ['K', 'K23', 'K32', 'K24', 'K42', 'KA'],
#     (12, 4): ['CL', 'V2', 'Q3', 'V3', 'Q4', 'V4', 'KA'],
# }

"""ADVAN/TRANS 组合合法性 + $PK 必需参数检查。"""

# ADVAN → 房室数（固定 NONMEM 事实）
_ADVAN_COMPARTMENTS = {1: 1, 2: 1, 3: 2, 4: 2, 11: 3, 12: 3}

# (ADVAN, TRANS) → $PK 必需参数（固定 NONMEM 事实）
_ADVAN_TRANS_PARAMS = {
    (1, 1): ['K', 'V'],
    (1, 2): ['CL', 'V'],
    (2, 1): ['K', 'V', 'KA'],
    (2, 2): ['CL', 'V', 'KA'],
    (3, 1): ['K', 'K12', 'K21'],
    (3, 4): ['CL', 'V1', 'Q', 'V2'],
    (4, 1): ['K', 'K23', 'K32', 'KA'],
    (4, 4): ['CL', 'V2', 'Q', 'V3', 'KA'],
    (11, 1): ['K', 'K12', 'K21', 'K13', 'K31'],
    (11, 4): ['CL', 'V1', 'Q2', 'V2', 'Q3', 'V3'],
    (12, 1): ['K', 'K23', 'K32', 'K24', 'K42', 'KA'],
    (12, 4): ['CL', 'V2', 'Q3', 'V3', 'Q4', 'V4', 'KA'],
}

class AdvanTransValidityGuard:
    """ADVAN/TRANS 组合合法性守卫。"""

    @staticmethod
    def _has_pk_assignment(pk_block: str, param: str) -> bool:
        """
        匹配以下任意一种赋值：
          CL = ...
          TVCL = ...
          TV_CL = ...
        且 param 之前不能是字母/数字/下划线（避免 V2 被 TVV2 命中）。
        """
        # pattern = rf'(?<![A-Za-z0-9_])(?:TV_?)?{re.escape(param)}\s*='

        # 支持：CL = / TVCL = / TV_CL = / CL= / TVCL=
        # 且允许 param 后跟空格、括号等
        pattern = rf'(?<![A-Za-z0-9_])(?:TV_?)?{re.escape(param)}\s*=[^=]'
        return bool(re.search(pattern, pk_block, re.IGNORECASE))



    def check(self, code: str) -> Optional[str]:
        """
        校验 (ADVAN, TRANS) 组合与必需 $PK 参数。

        返回：
          若违规则返回修正指令；否则返回 None
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

        # 未知 ADVAN（如通用 ADVAN5-10/13，$DES 模式）→ 跳过
        if required is None:
            known_trans = sorted(
                t for (a, t) in _ADVAN_TRANS_PARAMS if a == advan_num)
            if known_trans and trans_num not in known_trans:
                valid = ', '.join(f"TRANS{t}" for t in known_trans)
                return (
                    f"REJECTED: ADVAN{advan_num} TRANS{trans_num} 不是合法组合。"
                    f"ADVAN{advan_num} 仅支持：{valid}。请选择其中之一，"
                    f"并使用其对应的 $PK 参数。"
                )
            return None

        # 检查 $PK 必需参数
        pk_match = re.search(r'\$PK(.*?)(?=\n\s*\$|\Z)',
                             code, re.DOTALL | re.IGNORECASE)
        # pk_block = pk_match.group(1) if pk_match else ''

        pk_block = _extract_pk_block(code)

        # ★ 若 $PK 块为空或极短，说明提取失败 → 跳过本次检查
        #   避免"全参数缺失"误判导致代码被错误回退
        if not pk_block or len(pk_block.strip()) < 10:
            print(f"  [GUARD-WARN] $PK block empty or too short "
                  f"(len={len(pk_block.strip())}), skipping param check")
            return None

        clean_pk = '\n'.join(
            line.split(';')[0].strip()
            for line in pk_block.split('\n')
            if line.split(';')[0].strip()
        )
        if not clean_pk or len(clean_pk) < 10:
            print(f"  [GUARD-WARN] $PK block has no effective content after "
                  f"comment stripping — skipping")
            return None

        # missing = [p for p in required if not self._has_pk_assignment(pk_block, p)]
        missing = [p for p in required if not self._has_pk_assignment(clean_pk, p)]

        # ★ 若所有参数都"缺失"，说明正则匹配系统性问题 → 跳过
        if len(missing) == len(required):
            print(f"  [GUARD-WARN] All {len(required)} required params reported as missing "
                  f"— likely regex/extraction issue. Sample from $PK (first 200 chars):")
            print(f"    {pk_block.strip()[:200]!r}")
            return None

        if missing:
            return (
                f"REJECTED: ADVAN{advan_num} TRANS{trans_num} 需要以下参数在 $PK 中赋值："
                f"{', '.join(required)}。缺失：{', '.join(missing)}。"
            )
        return None

    def apply(self, generated_code: str, base_code: str) -> str:
        """
        检测并恢复违规的 ADVAN/TRANS 组合。

        恢复策略：将 $SUBROUTINES 行替换为 base 的对应行。
        """
        violation = self.check(generated_code)
        if not violation or not base_code:
            return generated_code

        print(f"  [GUARD] {violation}")
        base_sub = re.search(r'\$SUBROUTINES?[^\$]+', base_code,
                             re.IGNORECASE | re.DOTALL)
        gen_sub = re.search(r'\$SUBROUTINES?[^\$]+', generated_code,
                            re.IGNORECASE | re.DOTALL)
        if base_sub and gen_sub:
            return generated_code.replace(gen_sub.group(0),
                                          base_sub.group(0), 1)
        return generated_code

def _extract_pk_block(code: str) -> str:
    """
    多模式 $PK 提取，应对 LLM 生成的各种格式：
      1. 标准：$PK ... \n$XXX
      2. 无换行分隔：$PK ... \n$T
      3. 注释壳：$PK\n; 注释\nCL=THETA(1)...
      4. Unicode 干扰：先规范化 em dash 等
    """
    # ★ 步骤 1：规范化 Unicode 破折号和不可见字符
    normalized = code.replace('\u2014', '--').replace('\u2013', '-')
    normalized = normalized.replace('\u00a0', ' ')  # non-breaking space

    # ★ 步骤 2：多模式提取
    patterns = [
        # 优先：$PK 到下一个 $XXX（严格）
        r'\$PK\b\s*\n(.*?)(?=\n\s*\$[A-Z])',
        # 次选：$PK 到下一个 $（宽松）
        r'\$PK\b\s*\n(.*?)(?=\n\s*\$)',
        # 兜底：$PK 后到文件尾
        r'\$PK\b(.*?)(?=\$[A-Z]{2,})',
    ]
    for pat in patterns:
        m = re.search(pat, normalized, re.DOTALL | re.IGNORECASE)
        if m:
            block = m.group(1)
            # ★ 步骤 3：去除注释行，只看有效内容
            clean_lines = []
            for line in block.split('\n'):
                stripped = line.split(';')[0].strip()
                if stripped:
                    clean_lines.append(stripped)
            clean_block = '\n'.join(clean_lines)
            # 至少有 1 个形如 X = ... 的赋值
            if re.search(r'\b\w+\s*=', clean_block):
                return block  # 返回**原始** block（保留注释，供 LLM 参考）
    return ''