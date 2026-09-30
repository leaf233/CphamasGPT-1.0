"""
tools/foce_enforcer.py

强制 FOCE-I 估计。

PKGPT 的 OFV 比较、过拟合检测、阶段推进均基于 FOCE-I（-2 log-likelihood 尺度，
典型 PopPK 数据集 OFV 在 100-500）。EM / Bayesian 方法的 OFV 尺度完全不同，
会破坏所有比较逻辑。

本工具检测并替换任何非 FOCE-I 的 $ESTIMATION 块，同时清理 EM 专用关键字。

不调用 LLM。
"""

from __future__ import annotations
import re


class FOCEEnforcer:
    """FOCE-I 强制器。"""

    # 非 FOCE-I 方法的关键字
    _NON_FOCE_PATTERN = re.compile(
        r'\$EST(?:IMATION)?[^\$]*(?:'
        r'METHOD\s*=\s*(?:SAEM|ITS|BAYES|MCMC|NUTS)'
        r'|NBURN\s*='
        r'|NITER\s*='
        r')[^\$]*',
        re.IGNORECASE | re.DOTALL,
    )

    _FOCE_BLOCK = (
        "$ESTIMATION METHOD=1 INTER MAXEVAL=9999 "
        "PRINT=5 POSTHOC NOABORT\n"
    )

    def apply(self, code: str) -> str:
        """检测并替换非 FOCE-I 估计块。"""
        if not self._NON_FOCE_PATTERN.search(code):
            return code

        # 记录被替换的方法名
        method_match = re.search(
            r'METHOD\s*=\s*(SAEM|ITS|BAYES|MCMC|NUTS)',
            code, re.IGNORECASE,
        )
        method = method_match.group(1).upper() if method_match else 'non-FOCE-I'
        print(f"  [{method}-BLOCK] {method} detected — replacing with FOCE-I")

        # 替换整个 $ESTIMATION 块
        est_pattern = re.compile(r'\$EST(?:IMATION)?[^\$]*',
                                 re.IGNORECASE | re.DOTALL)
        code = est_pattern.sub(self._FOCE_BLOCK, code, count=1)

        # 清理 EM 专用关键字
        for kw in ('NBURN', 'NITER', 'ISAMPLE', 'NSIG'):
            code = re.sub(rf'\b{kw}\s*=\s*\d+[^\n]*\n', '',
                          code, flags=re.IGNORECASE)

        return code

    def is_foce(self, code: str) -> bool:
        """判断当前 $ESTIMATION 是否已是 FOCE-I。"""
        m = re.search(r'\$EST(?:IMATION)?[^\$]*',
                      code, re.IGNORECASE | re.DOTALL)
        if not m:
            return False
        block = m.group(0)
        return bool(re.search(r'METHOD\s*=\s*1\b', block) and
                    re.search(r'\bINTER\b', block))