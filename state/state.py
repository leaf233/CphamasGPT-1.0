"""
state/state.py

OptimizationState：PKGPT 2.0 的唯一状态容器。

组成：
- 配置字段（构造时注入，运行期不变）
- 运行态字段（iteration / phase / iterations_in_phase）
- 模型快照（current_code / best_code / best_ofv / best_composite / best_iteration）
- 历史列表（improvement / phase / code / parameter / covariate）
- 上下文（data_profile / plausibility_bounds / plausibility_report / dose_scaling_hint）
- SCM 子状态（scm: SCMState）
- Guardrail 状态（failed_strategies / last_revert_info / flags）
- 外部依赖引用（llm / tools / data_loader / skills / phase_manager）
- 房室锁定（structure_locked / forced_compartments / locked_compartments）

设计原则：
- 数据与逻辑分离：本文件不实现任何业务逻辑
- 显式初始值：所有字段有明确默认值，便于测试
- 序列化友好：除外部依赖外，其余字段均可 JSON 序列化
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .scm_state import SCMState


# --------------------------------------------------------------------------- #
# 阶段枚举
# --------------------------------------------------------------------------- #

class ModelPhase(Enum):
    """
    建模阶段。

    编号说明：
      - 1 / 3 / 4 / 5 与论文阶段编号对齐
      - 缺口 2 是刻意的（Phase 2 已移除房室升级，仅保留诊断，
        但为了与论文编号对齐，Phase 3 仍为 3）
      - Phase 2 保留在流程中，仅移除其中的房室升级逻辑

    前向唯一：阶段只能单调不减，由 PhaseTransitionSkill 强制
    """
    ESTABLISH_BASE     = 1
    DIAGNOSE_STRUCTURE = 2   # 保留：诊断与稳定性，但不升级房室
    REDUCE_OVERFITTING = 3
    OPTIMIZE_IIV       = 4
    COVARIATE_ANALYSIS = 5

    def __str__(self):
        names = {
            ModelPhase.ESTABLISH_BASE:     "Phase 1: Establish Base Model",
            ModelPhase.DIAGNOSE_STRUCTURE: "Phase 2: Diagnose Structure",
            ModelPhase.REDUCE_OVERFITTING: "Phase 3: Reduce Overfitting",
            ModelPhase.OPTIMIZE_IIV:       "Phase 4: Optimize IIV",
            ModelPhase.COVARIATE_ANALYSIS: "Phase 5: Covariate Analysis",
        }
        return names.get(self, "Unknown Phase")


# --------------------------------------------------------------------------- #
# 主状态
# --------------------------------------------------------------------------- #

@dataclass
class OptimizationState:
    """PKGPT 2.0 优化过程的唯一状态容器。"""

    # ── 配置（构造时注入）────────────────────────────────────────────── #
    data_file: str
    output_base: str
    api_key: str
    min_iterations: int = 3
    max_iterations: int = 20
    nmfe_command: str = 'nmfe75.bat'
    model: str = 'flash'
    prior_info: Dict[str, Any] = field(default_factory=dict)
    forced_compartments: Optional[int] = None


    # ── 房室锁定（Agent/Skill/Guardrail 读取）────────────────────────── #
    structure_locked: bool = True
    locked_compartments: Optional[int] = None

    # ── 运行态 ───────────────────────────────────────────────────────── #
    iteration: int = 0
    phase: int = 1
    iterations_in_phase: int = 0

    # ── 模型快照 ─────────────────────────────────────────────────────── #
    current_code: Optional[str] = None
    best_code: Optional[str] = None
    best_ofv: Optional[float] = None
    best_composite: float = float('inf')
    best_iteration: int = 0

    # ── 历史列表 ─────────────────────────────────────────────────────── #
    improvement_history: List[dict] = field(default_factory=list)
    phase_history: List[dict] = field(default_factory=list)
    code_history: List[dict] = field(default_factory=list)
    parameter_history: List[dict] = field(default_factory=list)
    covariate_history: List[dict] = field(default_factory=list)

    # ── 上下文 ───────────────────────────────────────────────────────── #
    # data_profile: Optional[Dict[str, Any]] = None
    # plausibility_bounds: Optional[Dict[str, Any]] = None
    # plausibility_report: Optional[Dict[str, Any]] = None
    # dose_scaling_hint: Optional[str] = None

    data_profile: dict = field(default_factory=dict)
    # data_loader: Any = None
    plausibility_bounds: Optional[dict] = None
    plausibility_report: Optional[dict] = None
    dose_scaling_hint: Optional[str] = None



    # ── 最近一次运行结果 ─────────────────────────────────────────────── #
    # last_run: {
    #   'lst_path':    str,
    #   'txt_path':    str,
    #   'exit_code':   int,
    #   'full_output': str,
    #   'issues':      list[str],
    #   'parsed_data': dict,          # NONMEMParser.get_parsed_data()
    #   'shrinkage':   list[dict],
    #   'summary':     str,
    # }
    last_run: Dict[str, Any] = field(default_factory=dict)

    # ── AI 质量评价上下文（供下一轮 prompt 使用）─────────────────────── #
    last_ai_recommendations: List[str] = field(default_factory=list)
    last_ai_critical_issues: List[str] = field(default_factory=list)

    # ── SCM 子状态 ───────────────────────────────────────────────────── #
    scm: SCMState = field(default_factory=SCMState)
    scm_complete: bool = False

    # ── Guardrail 状态 ───────────────────────────────────────────────── #
    failed_strategies: List[str] = field(default_factory=list)
    strategy_repeat_count: Dict[str, int] = field(default_factory=dict)
    last_revert_info: Optional[dict] = None
    # last_revert_info: Optional[Dict[str, Any]] = None

    # ★ failed_strategies 的 prompt-ready 文本形式（由 Orchestrator 每轮刷新，
    #   Skill 在 run() 中 prepend 到 prompt）
    failed_strategies_warning: str = ""

    # ── 控制标志 ─────────────────────────────────────────────────────── #
    # flags: {
    #   'stop':  bool,     # TerminationSkill 判定可停止
    #   'abort': bool,     # HumanGateway 判定中止
    # }
    flags: Dict[str, bool] = field(default_factory=dict)
    # ★ Rescue 流程（v2.2 新增）
    rescue_pending: bool = False
    _rescue_attempted: bool = False

    _no_progress_count: int = 0
    '''
    force_stop: bool = False
    force_stop_reason: str = ""
    human_review_count: int = 0
    last_best_iteration_at_review: int = 0
    cov_failure_diagnosis: Optional[dict] = None
    '''

    # ★ 强制停止信号（v2.1 新增）
    force_stop: bool = False
    force_stop_reason: str = ""

    # ★ Human Review 计数（v2.1 新增）
    human_review_count: int = 0
    last_best_iteration_at_review: int = 0

    # ★ 协方差失败诊断（v2.2 新增）
    cov_failure_diagnosis: Optional[dict] = None

    # ★ THETA 扩边界计数（v2.2 新增）
    theta_widen_count: dict = field(default_factory=dict)

    # ── Phase 3 结构性信号（v2.2 新增）────────────────────
    phase3_alag_recommended: bool = False
    phase3_boundary_params: List[str] = field(default_factory=list)
    phase3_error_model_simplify: bool = False

    # ── 外部依赖引用（不参与序列化）──────────────────────────────────── #
    llm: Any = None                 # MultiModelOpenRouterClient
    tools: Any = None               # ToolRegistry
    skills: Any = None              # SkillRegistry（dict）
    data_loader: Any = None         # PKDataLoader
    phase_manager: Any = None       # PhaseTransitionManager


    # ── 便捷方法 ─────────────────────────────────────────────────────── #

    def current_phase(self) -> ModelPhase:
        """当前阶段枚举。"""
        return ModelPhase(self.phase)

    def is_phase5(self) -> bool:
        """是否为 Phase 5。"""
        return self.phase == 5

    def best_entry(self) -> Optional[dict]:
        """返回 best_iteration 对应的 improvement 记录。"""
        for entry in self.improvement_history:
            if entry.get('iteration') == self.best_iteration:
                return entry
        return None

    def last_improvement(self) -> Optional[dict]:
        """返回最近一条 improvement 记录。"""
        return self.improvement_history[-1] if self.improvement_history else None

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        生成用于最终汇总的字典（不含外部依赖与完整历史）。
        """
        return {
            'data_file':          self.data_file,
            'output_base':        self.output_base,
            'model':              self.model,
            'locked_compartments': self.locked_compartments,
            'total_iterations':   self.iteration,
            'final_phase':        self.phase,
            'best_iteration':     self.best_iteration,
            'best_ofv':           self.best_ofv,
            'best_composite':     self.best_composite,
            'phase_history':      self.phase_history,
            'scm': {
                'mode':       self.scm.mode,
                'round':      self.scm.round,
                'confirmed':  [c['name'] for c in self.scm.confirmed],
                'eliminated': [e['name'] for e in self.scm.eliminated],
                'complete':   self.scm.complete,
            },
        }