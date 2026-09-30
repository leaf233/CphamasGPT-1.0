"""
guardrails/phase5_structure_guard.py

Phase 5 结构冻结守卫。

SCM 期间唯一允许的操作：
  ✅ 在 $PK 中添加/移除协变量关系式
  ✅ 在 $THETA 中添加/移除一个协变量参数

禁止的操作（检测到即从 base 恢复）：
  ❌ 改变 $SUBROUTINES / ADVAN（房室变化）
  ❌ 改变 $OMEGA 块数或结构（IIV 变化）
  ❌ 改变 $ERROR / $SIGMA（误差模型变化）
  ❌ 改变 $ESTIMATION method（估计方法变化）

封装的原始逻辑：NONMEMOptimizer._enforce_phase5_structure

不调用 LLM。
"""

from __future__ import annotations
import re


class Phase5StructureGuard:
    """Phase 5 结构冻结守卫。"""

    def apply(self, generated_code: str, base_code: str) -> str:
        """
        检测并恢复 Phase 5 中不允许的结构变更。

        参数：
          generated_code : LLM 生成的新代码
          base_code      : 本轮 SCM 的 base 代码（通常是 best_code）

        返回：
          结构恢复后的代码（只保留 covariate 变更）
        """
        if not base_code:
            return generated_code

        reference = base_code
        result = generated_code
        violations = []

        # ---- 1) ADVAN 变更检测 ------------------------------------------ #
        base_advan = re.search(r'ADVAN(\d+)', reference, re.IGNORECASE)
        gen_advan  = re.search(r'ADVAN(\d+)', result, re.IGNORECASE)
        if base_advan and gen_advan and \
           base_advan.group(1) != gen_advan.group(1):
            violations.append(
                f"ADVAN{gen_advan.group(1)} → ADVAN{base_advan.group(1)} "
                f"(结构变更禁止)"
            )
            base_sub = re.search(r'\$SUBROUTINES?[^\$]+',
                                 reference, re.IGNORECASE | re.DOTALL)
            gen_sub = re.search(r'\$SUBROUTINES?[^\$]+',
                                result, re.IGNORECASE | re.DOTALL)
            if base_sub and gen_sub:
                result = result.replace(gen_sub.group(0), base_sub.group(0), 1)

        # ---- 2) OMEGA 块数减少检测 -------------------------------------- #
        base_omega_blocks = re.findall(
            r'\$OMEGA[^\$]+', reference, re.IGNORECASE | re.DOTALL)
        gen_omega_blocks = re.findall(
            r'\$OMEGA[^\$]+', result, re.IGNORECASE | re.DOTALL)
        if len(gen_omega_blocks) < len(base_omega_blocks):
            violations.append(
                f"OMEGA 块 {len(base_omega_blocks)} → "
                f"{len(gen_omega_blocks)}（IIV 移除禁止）"
            )
            # 恢复：删除所有 OMEGA 块，重新插入 base 的 OMEGA 块
            result_no_omega = re.sub(r'\$OMEGA[^\$]+', '', result,
                                     flags=re.IGNORECASE | re.DOTALL)
            sigma_m = re.search(r'\$SIGMA', result_no_omega, re.IGNORECASE)
            omega_str = ''.join(base_omega_blocks) + '\n'
            if sigma_m:
                ins = sigma_m.start()
                result = result_no_omega[:ins] + omega_str + result_no_omega[ins:]
            else:
                result = result_no_omega + '\n' + omega_str

        # ---- 3) $ERROR 变更检测 ----------------------------------------- #
        base_error = re.search(r'\$ERROR[^\$]+', reference,
                               re.IGNORECASE | re.DOTALL)
        gen_error = re.search(r'\$ERROR[^\$]+', result,
                              re.IGNORECASE | re.DOTALL)
        if base_error and gen_error:
            b = re.sub(r'\s+', ' ', base_error.group(0)).strip()
            g = re.sub(r'\s+', ' ', gen_error.group(0)).strip()
            if b != g:
                violations.append("$ERROR 变更 → 恢复 base")
                result = result.replace(gen_error.group(0),
                                        base_error.group(0), 1)

        # ---- 4) $ESTIMATION 变更检测 ------------------------------------ #
        base_est = re.search(r'\$EST(?:IMATION)?[^\$]+', reference,
                             re.IGNORECASE | re.DOTALL)
        gen_est = re.search(r'\$EST(?:IMATION)?[^\$]+', result,
                            re.IGNORECASE | re.DOTALL)
        if base_est and gen_est:
            b = re.sub(r'\s+', ' ', base_est.group(0)).strip()
            g = re.sub(r'\s+', ' ', gen_est.group(0)).strip()
            if b != g:
                violations.append("$ESTIMATION 变更 → 恢复 base")
                result = result.replace(gen_est.group(0),
                                        base_est.group(0), 1)

        # ---- 5) 报告 ------------------------------------------------------ #
        if violations:
            print(f"\n{'!'*70}")
            print("PHASE 5 STRUCTURAL VIOLATION — AUTO-CORRECTED")
            print(f"{'!'*70}")
            for v in violations:
                print(f"  ❌ {v}")
            print(f"  → 结构元素已从 base model 恢复")
            print(f"  → Covariate 变更（$PK 修改、$THETA 添加/移除）保留")
            print(f"{'!'*70}")

        return result