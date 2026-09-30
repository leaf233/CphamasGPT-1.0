"""
tools/input_normalizer.py

强制 $INPUT 与 CSV 列序一致。

NONMEM 按位置映射列，若 LLM 重排列名会导致 DV/MDV/RATE 静默错配。
本工具在每次代码生成后运行，确保 $INPUT 与数据文件列序完全一致。

不调用 LLM。
"""

from __future__ import annotations
import re


class InputNormalizer:
    """$INPUT 列序强制器。"""

    def apply(self, code: str, state=None) -> str:
        """
        将 $INPUT 行替换为 CSV 实际列序。

        参数：
          code  : NONMEM 控制流
          state : OptimizationState（可选；若为 None，需在构造函数中预先提供列信息）

        返回：
          已修正的 code
        """
        csv_columns = self._get_csv_columns(state)
        if not csv_columns:
            return code

        correct_input = "$INPUT " + " ".join(csv_columns)

        # 替换现有 $INPUT 块（到下一个 $ 前）
        pattern = re.compile(r'\$INPUT\b[^\$]*', re.IGNORECASE | re.DOTALL)
        if pattern.search(code):
            new_code = pattern.sub(correct_input + "\n\n", code, count=1)
            if new_code != code:
                print(f"  [INPUT-FIX] $INPUT normalized: {' '.join(csv_columns)}")
            return new_code

        # 若无 $INPUT，插入到 $DATA 之后
        data_pat = re.compile(r'(\$DATA[^\n]*\n)', re.IGNORECASE)
        if data_pat.search(code):
            print("  [INPUT-FIX] $INPUT missing — inserted after $DATA")
            return data_pat.sub(r'\1' + correct_input + "\n", code, count=1)

        return code

    def _get_csv_columns(self, state) -> list:
        """从 state 或 data_loader 读取 CSV 列序。"""
        if state is None:
            return []

        # 优先从 data_loader 读取
        loader = getattr(state, 'data_loader', None)
        if loader is not None:
            try:
                meta = loader.get_metadata()
                cols = meta.get('columns', [])
                if cols:
                    return cols
            except Exception:
                pass

        # 其次从 data_profile 读取
        profile = getattr(state, 'data_profile', None) or {}
        return profile.get('columns', [])