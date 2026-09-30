"""
tools/cs_editor.py

控制流局部编辑工具。

提供：
1. set_theta_bounds    — 修改单个 THETA 的 (lower, initial, upper)
2. remove_omega        — 删除指定索引的 OMEGA
3. collapse_omega_to_diagonal — BLOCK Ω → DIAGONAL Ω
4. simplify_error_model — combined → proportional-only
5. fix_theta           — 用 FIX 固定 THETA

不提供 advan_switch：Phase 2 房室升级已移除，房室由 --compartments 锁定。
"""

from __future__ import annotations
import re
from typing import List, Optional


class ControlStreamEditor:
    """控制流编辑器。"""

    # ---- THETA ----------------------------------------------------------- #

    def set_theta_bounds(
        self,
        code: str,
        theta_index: int,
        lower: float,
        initial: float,
        upper: float,
        fix: bool = False,
    ) -> str:
        """修改指定 THETA 的 (lower, initial, upper)。"""
        fix_str = " FIX" if fix else ""
        new_line = f"({lower}, {initial}, {upper}){fix_str}"

        pattern = re.compile(
            rf"(THETA\(\s*{theta_index}\s*\)\s*=\s*)\([^)]*\)(?:\s*FIX)?",
            re.IGNORECASE,
        )
        if pattern.search(code):
            return pattern.sub(rf"\g<1>{new_line}", code)

        # 若 THETA 用 3 值列表形式（$THETA 块），逐行替换
        return self._replace_theta_by_position(code, theta_index, new_line)

    def fix_theta(self, code: str, theta_index: int, value: float) -> str:
        """用 FIX 固定指定 THETA。"""
        pattern = re.compile(
            rf"THETA\(\s*{theta_index}\s*\)\s*=\s*[^\n]+",
            re.IGNORECASE,
        )
        if pattern.search(code):
            return pattern.sub(f"THETA({theta_index}) = {value} FIX", code)
        return code

    # ---- OMEGA ----------------------------------------------------------- #

    def remove_omega(self, code: str, eta_index: int) -> str:
        """
        删除指定索引的 OMEGA。

        仅支持 DIAGONAL $OMEGA；BLOCK 结构请先用 collapse_omega_to_diagonal()。
        至少保留 1 个 OMEGA。
        """
        omega_values = self._parse_omega_values(code)
        if len(omega_values) <= 1:
            raise ValueError("Cannot remove last OMEGA (would become non-population model)")

        if eta_index < 1 or eta_index > len(omega_values):
            raise IndexError(f"eta_index {eta_index} out of range [1, {len(omega_values)}]")

        remaining = [v for i, v in enumerate(omega_values, start=1) if i != eta_index]
        new_block = "$OMEGA\n" + "\n".join(f"{v}" for v in remaining) + "\n"
        return self._replace_block(code, 'OMEGA', new_block)

    def collapse_omega_to_diagonal(self, code: str) -> str:
        """将 BLOCK Ω 转成 DIAGONAL Ω（只保留对角线）。"""
        block_match = re.search(
            r'\$OMEGA\s+BLOCK\s*\(\s*(\d+)\s*\)\s*(.*?)(?=\n\s*\$|\Z)',
            code, re.DOTALL | re.IGNORECASE,
        )
        if not block_match:
            return code

        n = int(block_match.group(1))
        body = block_match.group(2)
        # 提取每行数字
        rows = []
        for line in body.split('\n'):
            line = line.split(';')[0].strip()
            if line:
                rows.append(re.findall(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', line))

        # 对角线值：第 i 行的第 i 个数字（1-based）
        diag = []
        for i, row in enumerate(rows[:n]):
            if i < len(row):
                diag.append(row[i])
            else:
                diag.append("0.1")

        new_block = "$OMEGA\n" + "\n".join(diag) + "\n"
        return self._replace_block(code, 'OMEGA', new_block)

    # ---- ERROR ----------------------------------------------------------- #

    def simplify_error_model(self, code: str, sigma_index: int = 1) -> str:
        """
        将 combined error（加法 + 比例）简化为 proportional-only。

        $ERROR 改为：
            IPRED = F
            Y = IPRED * (1 + EPS(1))
        $SIGMA 改为：
            0.04  ; ~20% CV
        """
        error_block = "$ERROR\nIPRED = F\nY = IPRED * (1 + EPS(1))\n"
        code = self._replace_block(code, 'ERROR', error_block)

        sigma_block = "$SIGMA\n0.04\n"
        code = self._replace_block(code, 'SIGMA', sigma_block)

        # 同时删除 $THETA 中的加法误差参数（若有）
        code = self._remove_additive_error_theta(code)
        return code

    # ---- 内部辅助 --------------------------------------------------------- #

    def _replace_block(self, code: str, block_name: str, new_block: str) -> str:
        """替换 $BLOCK_NAME ... 直到下一个 $ 或结尾。"""
        pattern = re.compile(
            rf'\${block_name}\b.*?(?=\n\s*\$|\Z)',
            re.DOTALL | re.IGNORECASE,
        )
        if pattern.search(code):
            return pattern.sub(new_block, code, count=1)
        # 不存在则追加
        return code + "\n" + new_block

    def _parse_omega_values(self, code: str) -> List[str]:
        """提取 DIAGONAL $OMEGA 中的数值。"""
        block = re.search(
            r'\$OMEGA\s*(?!BLOCK)(.*?)(?=\n\s*\$|\Z)',
            code, re.DOTALL | re.IGNORECASE,
        )
        if not block:
            return []
        values = []
        for line in block.group(1).split('\n'):
            line = line.split(';')[0].strip()
            if line:
                values.extend(
                    re.findall(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', line))
        return values

    def _replace_theta_by_position(
        self, code: str, theta_index: int, new_line: str
    ) -> str:
        """在 $THETA 块中按位置替换。"""
        block = re.search(
            r'\$THETA\b(.*?)(?=\n\s*\$|\Z)',
            code, re.DOTALL | re.IGNORECASE,
        )
        if not block:
            return code

        body = block.group(1)
        lines = body.split('\n')
        theta_count = 0
        for i, line in enumerate(lines):
            content = line.split(';')[0].strip()
            if not content:
                continue
            theta_count += 1
            if theta_count == theta_index:
                # 保留注释
                comment = ''
                if ';' in line:
                    comment = '  ;' + line.split(';', 1)[1]
                lines[i] = f"{new_line}{comment}"
                break

        new_body = '\n'.join(lines)
        return code.replace(body, new_body, 1)

    def _remove_additive_error_theta(self, code: str) -> str:
        """删除带 'additive' 注释的 THETA 行。"""
        lines = code.split('\n')
        out = []
        for line in lines:
            if ';' in line:
                comment = line.split(';', 1)[1].lower()
                if 'additive' in comment or 'add ' in comment:
                    continue
            out.append(line)
        return '\n'.join(out)