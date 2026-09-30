"""
state/history.py

历史条目数据结构与追加辅助函数。

包含：
- ImprovementEntry        — 每轮迭代的模型结果（OFV、shrinkage、covariance 等）
- PhaseTransitionEntry    — 阶段推进记录
- CodeHistoryEntry        — 控制流历史（供 prompt 回看）
- ParameterHistoryEntry   — THETA/OMEGA/SIGMA 估计值历史（供收紧边界）
- CovariateHistoryEntry   — SCM 协变量测试历史

辅助函数：
- append_improvement(state, entry)      — 追加并更新 best model
- append_phase_transition(state, entry)
- append_code(state, iteration, code, description)
- append_parameter(state, entry)
- append_covariate(state, entry)

设计原则：
- 数据结构为普通 dict，便于 JSON 序列化与测试
- 使用 dataclass 仅作为类型提示与构造辅助，不强制
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class ImprovementEntry:
    """单轮迭代结果记录。"""
    iteration: int
    status: str = 'unknown'                       # 'success' | 'failed'
    ofv: Optional[float] = None
    max_rse: Optional[float] = None
    high_rse_count: int = 0
    avg_eta_shrinkage: Optional[float] = None
    issues: List[str] = field(default_factory=list)
    minimization_successful: bool = False
    covariance_successful: bool = False
    composite_score: float = float('inf')
    omega_values: List[float] = field(default_factory=list)
    n_subjects: int = 0
    changes: str = 'N/A'
    ai_evaluation: Optional[Dict] = None          # {quality_score, grade, should_continue, reason}
    reverted_to_round_base: bool = False          # Phase 5 专用标记

    def to_dict(self) -> dict:
        """转为普通 dict 以便存入 state.improvement_history。"""
        return {
            'iteration':               self.iteration,
            'status':                  self.status,
            'ofv':                     self.ofv,
            'max_rse':                 self.max_rse,
            'high_rse_count':          self.high_rse_count,
            'avg_eta_shrinkage':       self.avg_eta_shrinkage,
            'issues':                  self.issues,
            'minimization_successful': self.minimization_successful,
            'covariance_successful':   self.covariance_successful,
            'composite_score':         self.composite_score,
            'omega_values':            self.omega_values,
            'n_subjects':              self.n_subjects,
            'changes':                 self.changes,
            'ai_evaluation':           self.ai_evaluation,
            'reverted_to_round_base':  self.reverted_to_round_base,
        }


@dataclass
class PhaseTransitionEntry:
    """阶段推进记录。"""
    iteration: int
    from_phase: Any           # int 或 ModelPhase
    to_phase: Any
    iterations_in_previous_phase: int

    def to_dict(self) -> dict:
        return {
            'iteration':                   self.iteration,
            'from_phase':                  self.from_phase,
            'to_phase':                    self.to_phase,
            'iterations_in_previous_phase': self.iterations_in_previous_phase,
        }


@dataclass
class CodeHistoryEntry:
    """控制流历史条目。"""
    iteration: int
    code: str
    description: str = 'Not specified'

    def to_dict(self) -> dict:
        return {
            'iteration':   self.iteration,
            'code':        self.code,
            'description': self.description,
        }


@dataclass
class ParameterHistoryEntry:
    """THETA/OMEGA/SIGMA 估计值历史（供参数稳定化指引）。"""
    iteration: int
    theta_names: List[str] = field(default_factory=list)
    theta_vals: List[float] = field(default_factory=list)
    omega_names: List[str] = field(default_factory=list)
    omega_vals: List[float] = field(default_factory=list)
    sigma_names: List[str] = field(default_factory=list)
    sigma_vals: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'iteration':   self.iteration,
            'theta_names': self.theta_names,
            'theta_vals':  self.theta_vals,
            'omega_names': self.omega_names,
            'omega_vals':  self.omega_vals,
            'sigma_names': self.sigma_names,
            'sigma_vals':  self.sigma_vals,
        }


@dataclass
class CovariateHistoryEntry:
    """SCM 协变量测试历史条目。"""
    name: str
    covariate: str = ''
    parameter: str = ''
    mode: str = 'add'                       # 'add' | 'remove'
    delta_ofv: Optional[float] = None
    cov_ok: bool = False
    result: str = 'TESTED'                  # TESTED | ACCEPTED | REJECTED | RETAINED |
                                            # ELIMINATED | REJECTED_UNSAFE | RETAINED_UNSAFE
    iteration: int = 0
    round: int = 0

    def to_dict(self) -> dict:
        return {
            'name':       self.name,
            'covariate':  self.covariate,
            'parameter':  self.parameter,
            'mode':       self.mode,
            'delta_ofv':  self.delta_ofv,
            'cov_ok':     self.cov_ok,
            'result':     self.result,
            'iteration':  self.iteration,
            'round':      self.round,
        }


# --------------------------------------------------------------------------- #
# 追加辅助函数
# --------------------------------------------------------------------------- #

def append_improvement(state, entry) -> None:
    """
    追加一条 improvement 记录并更新 best model。

    best 判定规则：
      - 非 Phase 5：composite_score 更低者优先
      - Phase 5：由 SCM Skill 显式管理，不在此处更新 best
    """
    if isinstance(entry, ImprovementEntry):
        entry = entry.to_dict()

    state.improvement_history = state.improvement_history + [entry]

    # Phase 5 不做通用 best 更新（SCM Skill 显式管理）
    if getattr(state, 'phase', None) == 5:
        return

    # best model 更新
    score = entry.get('composite_score', float('inf'))
    if score < state.best_composite:
        state.best_composite = score
        state.best_ofv = entry.get('ofv')
        state.best_iteration = entry.get('iteration', state.iteration)
        state.best_code = state.current_code


def append_phase_transition(state, entry) -> None:
    """追加一条阶段推进记录。"""
    if isinstance(entry, PhaseTransitionEntry):
        entry = entry.to_dict()
    state.phase_history = state.phase_history + [entry]


def append_code(state, iteration: int, code: str, description: str = 'Not specified') -> None:
    """追加一条控制流历史。"""
    state.code_history = state.code_history + [{
        'iteration':   iteration,
        'code':        code,
        'description': description,
    }]


def append_parameter(state, entry) -> None:
    """追加一条参数估计历史。"""
    if isinstance(entry, ParameterHistoryEntry):
        entry = entry.to_dict()
    state.parameter_history = state.parameter_history + [entry]


def append_covariate(state, entry) -> None:
    """追加一条 SCM 协变量历史。"""
    if isinstance(entry, CovariateHistoryEntry):
        entry = entry.to_dict()
    state.covariate_history = state.covariate_history + [entry]