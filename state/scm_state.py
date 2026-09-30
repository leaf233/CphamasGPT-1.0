"""
state/scm_state.py

SCMState：Phase 5 Stepwise Covariate Modeling 子状态。

设计要点：
- forward / backward 两个模式共享同一结构，mode 字段区分
- round_base_ofv / round_base_code 是每轮冻结的 base（所有候选对比基准）
- confirmed / eliminated 是跨轮累积的最终结果
- round_tested / round_results 是当前轮的临时记录，轮结束由 Skill 清零

与 optimizer.py 中的 SCM 实现对应：
  self.scm_mode / scm_confirmed / scm_eliminated / scm_current_round /
  scm_round_tested / scm_round_results / scm_round_base_ofv / scm_round_base_code
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class SCMState:
    """Phase 5 SCM 子状态。"""

    # ---- 模式与轮次 ------------------------------------------------------ #
    mode: str = 'forward'          # 'forward' | 'backward'
    round: int = 1                 # 当前轮次（forward/backward 各自从 1 起）

    # ---- 跨轮累积结果 ---------------------------------------------------- #
    # confirmed: 已被 forward 接受、尚未被 backward 剔除的协变量
    #   每项: {name, covariate, parameter, model_type, delta_ofv, code, ofv,
    #          iteration, round}
    confirmed: List[dict] = field(default_factory=list)

    # eliminated: 已被 backward 剔除的协变量
    #   每项: {name, delta_ofv, iteration, round}
    eliminated: List[dict] = field(default_factory=list)

    # ---- 当前轮临时记录 -------------------------------------------------- #
    # round_tested: 本轮已测试过的候选名（用于快速判定轮是否完成）
    round_tested: Set[str] = field(default_factory=set)

    # round_results: 本轮每个候选的测试结果
    #   每项: {name, covariate, parameter, model_type, mode, ofv, delta_ofv,
    #          cov_ok, omega_values, avg_eta_shrinkage, code, iteration, round}
    round_results: List[dict] = field(default_factory=list)

    # ---- 本轮 base（所有候选对比基准）----------------------------------- #
    round_base_ofv: Optional[float] = None
    round_base_code: Optional[str] = None

    # ---- 阈值与完成标志 -------------------------------------------------- #
    forward_threshold: float = -3.84       # ΔOFV < -3.84（p<0.05, df=1）接受
    backward_threshold: float = 6.63       # ΔOFV on removal < 6.63（p<0.01, df=1）剔除
    complete: bool = False                 # SCM 整体完成标志

    # ---- 本轮待测指令（供 Skill 生成 prompt）---------------------------- #
    # current_instruction:
    #   mode='add' 时: {name, covariate, parameter, model_type, example, mode}
    #   mode='remove' 时: {name, covariate, parameter, mode}
    current_instruction: Optional[dict] = None

    # ---- 便捷方法 -------------------------------------------------------- #

    def reset_round(self) -> None:
        """进入新轮次时清空本轮临时记录。"""
        self.round_tested = set()
        self.round_results = []
        self.round_base_ofv = None
        self.round_base_code = None
        self.current_instruction = None

    def confirmed_names(self) -> Set[str]:
        """已确认的协变量名集合。"""
        return {c['name'] for c in self.confirmed}

    def eliminated_names(self) -> Set[str]:
        """已剔除的协变量名集合。"""
        return {e['name'] for e in self.eliminated}

    def switch_to_backward(self) -> None:
        """从 forward 切换到 backward，重置轮次与 base。"""
        self.mode = 'backward'
        self.round = 1
        self.reset_round()


