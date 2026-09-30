"""
guardrails/numeric_safety.py

数值安全 gate。

判定条件（任一即不安全）：
1. OFV < -50        — 严重过拟合
2. OMEGA 全部 < 0.001 — IIV 结构完全塌缩
3. shrinkage > 95%   — 灾难性 shrinkage，ETA 不可信

与 _should_stop_early 的区别：
- _should_stop_early 使用"N 次连续"形式判定早期终止
- 本 gate 使用"单次判定"形式，用于 Phase 5 winner 候选的即时判定
  （SCM 每候选只测试一次，不会形成连续序列）

封装的原始逻辑：NONMEMOptimizer._is_numerically_safe

不调用 LLM。
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from utils.safe_numeric import safe_float



class NumericSafetyGate:
    """数值安全 gate。"""

    # 判定阈值
    OFV_NEGATIVE_THRESHOLD = -50.0
    OMEGA_COLLAPSE_THRESHOLD = 0.001
    SHRINKAGE_CATASTROPHIC_THRESHOLD = 95.0

    def check(
        self,
        ofv: Optional[float],
        omega_values: Optional[List[float]],
        shrinkage: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        判定候选模型是否数值安全。

        参数：
          ofv           : 目标函数值
          omega_values  : OMEGA 对角值列表
          shrinkage     : 最大 ETA shrinkage（%）

        返回：
          (is_safe: bool, reason: str)
          - is_safe=True 时 reason 为空字符串
          - is_safe=False 时 reason 说明具体失败原因
        """
        ofv = safe_float(ofv)
        shrinkage = safe_float(shrinkage)
        omega_values = [v for v in (safe_float(o) for o in (omega_values or [])) if v is not None]

        if ofv is not None and ofv < -50:
            return False, f"negative OFV ({ofv:.2f} < -50)"
        if omega_values:
            collapsed = [o for o in omega_values if o < 0.001]
            if len(collapsed) == len(omega_values):
                return False, f"all {len(omega_values)} OMEGA(s) collapsed (<0.001)"
        if shrinkage is not None and shrinkage > 95:
            return False, f"catastrophic ETA shrinkage ({shrinkage:.1f}% > 95%)"
        return True, ""

        '''
        # 1) 负 OFV（严重过拟合）
        if ofv is not None and ofv < self.OFV_NEGATIVE_THRESHOLD:
            return False, (
                f"negative OFV ({ofv:.2f} < "
                f"{self.OFV_NEGATIVE_THRESHOLD:.0f}, 严重过拟合)"
            )

        # 2) OMEGA 全部塌缩
        if omega_values:
            collapsed = [o for o in omega_values
                         if o < self.OMEGA_COLLAPSE_THRESHOLD]
            if len(collapsed) == len(omega_values):
                return False, (
                    f"all {len(omega_values)} OMEGA(s) collapsed "
                    f"(< {self.OMEGA_COLLAPSE_THRESHOLD})"
                )

        # 3) 灾难性 shrinkage
        if shrinkage is not None and \
           shrinkage > self.SHRINKAGE_CATASTROPHIC_THRESHOLD:
            return False, (
                f"catastrophic ETA shrinkage "
                f"({shrinkage:.1f}% > "
                f"{self.SHRINKAGE_CATASTROPHIC_THRESHOLD:.0f}%)"
            )

        return True, ""
        '''

    def check_parsed(self, parsed: dict) -> Tuple[bool, str]:
        """
        便捷入口：直接从 NONMEMParser 的 parsed_data 判定。

        参数：
          parsed : NONMEMParser.get_parsed_data() 结果

        返回：
          (is_safe, reason)
        """
        ofv = parsed.get('objective_function')

        params = parsed.get('parameter_estimates', {}) or {}
        omega_values = [
            float(o.get('value', 1.0))
            for o in params.get('omega', [])
            if 'value' in o
        ]

        shrink_list = parsed.get('eta_shrinkage', [])
        max_shrink = max((s['shrinkage'] for s in shrink_list), default=None) \
            if shrink_list else None

        return self.check(ofv, omega_values, max_shrink)