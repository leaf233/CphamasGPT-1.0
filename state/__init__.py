"""
state/__init__.py

State 层统一导出。

State 层职责：
- 定义 OptimizationState 主数据类（唯一状态容器）
- 定义 SCMState 子状态（Phase 5 SCM 专用）
- 定义 history 条目结构（improvement / phase / code / parameter / covariate）
- 提供 StateReducer.apply，将 SkillResult.updates 合并回 State

设计原则：
- State 是纯数据，不含业务逻辑
- 所有 Agent / Skill / Tool / Guardrail 读写同一份 State 引用
- StateReducer 是唯一的写入路径，避免散落赋值
"""

from .state import OptimizationState, ModelPhase
from .scm_state import SCMState
from .history import (
    ImprovementEntry,
    PhaseTransitionEntry,
    CodeHistoryEntry,
    ParameterHistoryEntry,
    CovariateHistoryEntry,
    append_improvement,
    append_phase_transition,
    append_code,
    append_parameter,
    append_covariate,
)
from .reducer import StateReducer

__all__ = [
    'OptimizationState',
    'ModelPhase',
    'SCMState',
    'ImprovementEntry',
    'PhaseTransitionEntry',
    'CodeHistoryEntry',
    'ParameterHistoryEntry',
    'CovariateHistoryEntry',
    'append_improvement',
    'append_phase_transition',
    'append_code',
    'append_parameter',
    'append_covariate',
    'StateReducer',
]