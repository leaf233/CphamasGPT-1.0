"""
guardrails/omega_structure_guard.py

OMEGA 结构合理性守卫。

职责：
- 检查 $OMEGA BLOCK(n) 的维度是否超出 N 支持的上限
- 超限时自动降维为 DIAGONAL（保留对角线元素）

不调用 LLM。
"""

from __future__ import annotations
import re
from typing import Optional


class OmegaStructureGuard:
    """OMEGA BLOCK 维度守卫。"""

    # (N 下限, N 上限) → 允许的最大 BLOCK 维度
    BLOCK_DIM_LIMITS = [
        (0,   30,  2),   # N<30:   BLOCK(2) 最大
        (30,  100, 3),   # 30≤N<100: BLOCK(3) 最大
        (100, 300, 4),   # 100≤N<300: BLOCK(4) 最大
        (300, 10**9, 5), # N≥300: BLOCK(5) 最大
    ]

    def _max_dim_for_n(self, n: int) -> int:
        for lo, hi, limit in self.BLOCK_DIM_LIMITS:
            if lo <= n < hi:
                return limit
        return 5

    def check(self, code: str, n_subjects: int) -> Optional[str]:
        """返回违规描述；合规返回 None。"""
        m = re.search(r'\$OMEGA\s+BLOCK\s*\(\s*(\d+)\s*\)',
                      code, re.IGNORECASE)
        if not m:
            return None
        block_dim = int(m.group(1))
        max_dim = self._max_dim_for_n(n_subjects)
        if block_dim > max_dim:
            n_params = block_dim * (block_dim + 1) // 2
            return (
                f"OMEGA BLOCK({block_dim}) requires {n_params} parameters, "
                f"but N={n_subjects} supports at most BLOCK({max_dim}). "
                f"Convert to DIAGONAL OMEGA with {block_dim} parameters."
            )
        return None

    def apply(self, code: str, n_subjects: int) -> str:
        """检测并自动将超限 BLOCK 转为 DIAGONAL。"""
        violation = self.check(code, n_subjects)
        if not violation:
            return code

        print(f"  [OMEGA-GUARD] {violation}")

        # 匹配 $OMEGA BLOCK(n) 之后的块内容（到下一个 $ 或文件尾）
        m = re.search(
            r'(\$OMEGA\s+BLOCK\s*\(\s*\d+\s*\))\s*\n([^\$]+)',
            code, re.IGNORECASE | re.DOTALL)
        if not m:
            return code

        block_header = m.group(1)
        block_body = m.group(2)

        # 提取每一行的最后一个数值作为对角线元素
        diag_values = []
        for line in block_body.strip().split('\n'):
            line = line.split(';')[0].strip()
            if not line:
                continue
            nums = line.split()
            if nums:
                diag_values.append(nums[-1])

        if not diag_values:
            return code

        # 构造 DIAGONAL OMEGA
        new_block_lines = ["$OMEGA"]
        for i, v in enumerate(diag_values):
            new_block_lines.append(
                f"{v}   ; ETA({i+1}) — auto-converted from BLOCK")
        new_block = '\n'.join(new_block_lines)

        return code.replace(m.group(0), new_block, 1)