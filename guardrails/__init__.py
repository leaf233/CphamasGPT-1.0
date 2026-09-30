"""
guardrails/__init__.py

Guardrail 层统一导出。

Guardrail 职责：
- 在 LLM 生成代码后、NONMEM 执行前，拦截违反约束的代码
- 在 NONMEM 执行后、结果写入 State 前，判定数值安全
- 在 Phase 5 winner 判定时，拒绝不安全的候选

所有 Guardrail 均为纯确定性逻辑，不调用 LLM。
"""

from .phase5_structure_guard import Phase5StructureGuard
from .compartment_invariance import CompartmentInvarianceGuard
from .advan_trans_validity import AdvanTransValidityGuard
from .numeric_safety import NumericSafetyGate
from .plausibility_guard import PlausibilityGuard
from .omega_structure_guard import OmegaStructureGuard          # ★ 新增


__all__ = [
    'Phase5StructureGuard',
    'CompartmentInvarianceGuard',
    'AdvanTransValidityGuard',
    'NumericSafetyGate',
    'PlausibilityGuard',
    'OmegaStructureGuard',
]