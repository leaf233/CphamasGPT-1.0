"""
guardrails/compartment_invariance.py

房室数量不变性守卫。

设计变更：
- 旧逻辑：仅在错误恢复期（prev_ofv is None）拒绝房室变化，
          允许 Phase 2 的"诊断驱动升级"
- 新逻辑：Phase 2 房室升级已移除，房室由 --compartments 锁定，
          本守卫拒绝任何 ADVAN 变化（无论 prev_ofv 是否为 None）

不同房室数量由并行运行的独立子进程探索（每个子进程锁定一个 N）。

封装的原始逻辑：NONMEMOptimizer._check_compartment_invariance

不调用 LLM。
"""

from __future__ import annotations
import re
from typing import Optional
from utils.compartment_lock import validate_lock



# ADVAN -> 房室数量（固定 NONMEM 事实）
_ADVAN_COMPARTMENTS = {1: 1, 2: 1, 3: 2, 4: 2, 11: 3, 12: 3}


class CompartmentInvarianceGuard:
    """房室数量不变性守卫。"""

    def check(
        self,
        code: str,
        prev_ofv: Optional[float],
        hint_compartments: Optional[int],
    ) -> Optional[str]:
        """
        检查 code 的 ADVAN 是否与锁定的房室数量一致。

        参数：
          code              : 待检查的控制流
          prev_ofv          : 上一轮 OFV（保留参数以兼容旧调用；新逻辑不再使用）
          hint_compartments : 锁定的房室数量（来自 state.locked_compartments）

        返回：
          若违规则返回修正指令字符串；否则返回 None
        """
        if hint_compartments is None:
            return None

        advan_match = re.search(r'ADVAN(\d+)', code, re.IGNORECASE)
        if not advan_match:
            return None

        new_compartments = _ADVAN_COMPARTMENTS.get(int(advan_match.group(1)))
        if new_compartments is None or new_compartments == hint_compartments:
            return None

        return (
            f"REJECTED: 该代码将结构改为 {new_compartments}-compartment "
            f"(ADVAN{advan_match.group(1)})，但本次运行锁定的房室数量为 "
            f"{hint_compartments}（--compartments / PKGPT_STRUCT_HINT）。"
            f"请保持 {hint_compartments}-compartment 模型。"
            f"不同房室数量由并行的独立子进程探索，不在单次运行内变更。"
        )

    def apply(
        self,
        generated_code: str,
        base_code: str,
        hint_compartments: Optional[int],
    ) -> str:
        """
        检测并恢复违规的房室变更。

        参数：
          generated_code    : LLM 生成的新代码
          base_code         : 上一轮有效代码
          hint_compartments : 锁定的房室数量

        返回：
          恢复后的代码
        """

        """
        检测到房室违规时，完整回退到 base_code。

        ⚠️  仅替换 $SUBROUTINES 是不够的：$PK 中针对不同房室的参数
           （如 TVQ3, TVV3 在 3cmt 中）在恢复 SUBROUTINES 后会成为悬空变量，
           NONMEM 会报错或忽略它们，导致隐式降房室。
           因此必须整体回退。
        """
        if hint_compartments is None or not base_code:
            return generated_code

        ok, actual, advan = validate_lock(generated_code, hint_compartments)
        if ok:
            return generated_code

        print(f"  [GUARD] 房室数不符: ADVAN{advan} ({actual}-cmt) != "
              f"locked {hint_compartments}-cmt → 完整回退到 base_code")
        return base_code  # ← 完整回退，不保留任何 LLM 修改

        '''
        violation = self.check(generated_code, prev_ofv=None,
                               hint_compartments=hint_compartments)
        if not violation or not base_code:
            return generated_code

        print(f"  [GUARD] {violation}")
        # 从 base 恢复 $SUBROUTINES
        base_sub = re.search(r'\$SUBROUTINES?[^\$]+', base_code,
                             re.IGNORECASE | re.DOTALL)
        gen_sub = re.search(r'\$SUBROUTINES?[^\$]+', generated_code,
                            re.IGNORECASE | re.DOTALL)
        if base_sub and gen_sub:
            return generated_code.replace(gen_sub.group(0),
                                          base_sub.group(0), 1)
        return generated_code
        '''
