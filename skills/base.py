"""
skills/base.py

Skill 抽象基类与统一的返回值/校验结构。

设计原则：
- 所有 Skill 共享相同的 run/validate/fallback 接口
- Skill 通过 SkillResult.updates 向 State 写入，不直接修改 State
- Skill 通过 ValidationResult 表达校验结果，由 Orchestrator 决定是否回退
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from utils.compartment_lock import validate_lock, build_correction_prompt
from utils.nonmem_utils import extract_code as _extract_code
import re


# -----------------------------------------------
# ---------------------------- #
# 返回值结构
# --------------------------------------------------------------------------- #

class CompartmentLockError(Exception):
    """房室锁违规且无法修正。"""
    pass


@dataclass
class ValidationResult:
    """Skill 内部校验结果。"""
    ok: bool = True
    reason: str = ""
    corrective_instruction: Optional[str] = None  # 供 LLM 重试时附带的指令


@dataclass
class SkillResult:
    """
    Skill 执行结果。

    updates: 需要写回 OptimizationState 的字段（由 StateReducer.apply 应用）
    done:    True 表示该 Skill 已完成其阶段职责，可触发 PhaseTransitionSkill
    ok:      False 表示执行失败，Orchestrator 可触发 fallback 或 HumanReview
    error:   失败原因
    meta:    额外信息（如 SCM round winner、parse 统计等），不写回 State
    """
    updates: Dict[str, Any] = field(default_factory=dict)
    done: bool = False
    ok: bool = True
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Skill 抽象基类
# --------------------------------------------------------------------------- #

class Skill(ABC):
    """
    所有 Skill 的抽象基类。

    子类必须声明：
      name        : 唯一标识（与 SkillRegistry key 一致）
      phase       : 适用阶段列表（[1] / [2] / [1,2,3,4,5] / [0] 表示 bootstrap）
      tools       : 允许调用的 Tool 名称列表
      prompt_template: prompt 模板标识（供 PromptTemplates 路由）
    """

    name: str = "base_skill"
    phase: List[int] = []
    tools: List[str] = []
    prompt_template: str = ""

    # ---- 生命周期 --------------------------------------------------------- #

    @abstractmethod
    def run(self, state, llm, tools) -> SkillResult:
        """
        执行 Skill 主逻辑。

        参数：
          state : OptimizationState 实例
          llm   : MultiModelOpenRouterClient 实例（供 LLM 调用）
          tools : ToolRegistry 实例（按 name 取用 Tool）
        """
        raise NotImplementedError

    def validate(self, output, state) -> ValidationResult:
        """默认不校验；子类可覆盖。"""
        return ValidationResult(ok=True)

    def fallback(self, state, error: str) -> SkillResult:
        """默认失败返回；子类可覆盖为具体回退策略。"""
        return SkillResult(ok=False, error=error)

    # ---- 便捷方法 --------------------------------------------------------- #

    def _tool(self, tools, name: str):
        """按名称取 Tool，若未注册则抛错，避免静默失败。"""
        if name not in tools:
            raise KeyError(
                f"[{self.name}] Tool '{name}' not registered. "
                f"Available: {list(tools.keys())}"
            )
        return tools[name]

    @staticmethod
    def _prepend_warning(prompt: str, state) -> str:
        """ 前置 failed_strategies 警告 + 协方差失败诊断。 """
        blocks = []

        warning = getattr(state, 'failed_strategies_warning', '') or ''
        if warning:
            blocks.append(warning)

        # 2) 协方差失败诊断（★ 新增）
        diag = getattr(state, 'cov_failure_diagnosis', None)
        if diag and diag.get('message'):
            blocks.append(
                "=" * 70 + "\n"
                "COVARIANCE FAILURE DIAGNOSIS — MANDATORY FIX\n" +
                "=" * 70 + "\n" +
                f"Root cause: {diag['type']}\n" +
                f"{diag['message']}\n" +
                "=" * 70
            )

        if not blocks:
            return prompt
        return "\n\n".join(blocks) + "\n\n" + prompt
        '''
        if not warning:
            return prompt
        return warning + "\n\n" + prompt
        '''



    # ---- 房室锁（Phase 1-4 通用） --------------------------------------- #

    def _enforce_compartment_lock(self, state, code: str, llm, prompt: str,
                                   max_retries: int = 2) -> str:
        """
        强制房室锁：若 code 的 ADVAN 对应的房室数与
        state.locked_compartments 不一致，则：
          - 若 LLM 明显"故意改房室"（与 base_code 的 ADVAN 不同），立即失败
          - 否则追加纠正 prompt 让 LLM 重试

        参数：
          state        : OptimizationState
          code         : LLM 生成的代码
          llm          : LLM 客户端
          prompt       : 原始 prompt（重试时追加 correction）
          max_retries  : 最多纠正重试次数（默认 2，共 3 次尝试）

        返回：
          修正后的 code（房室数匹配 locked_compartments）

        抛出：
          CompartmentLockError —— max_retries 次仍失败 或 LLM 故意改房室

        说明：
          - 若 state.locked_compartments 为 None，直接返回 code
          - 使用 utils.compartment_lock.validate_lock 做判定
          - 使用 utils.compartment_lock.build_correction_prompt 生成纠正
        """
        if not getattr(state, 'locked_compartments', None):
            return code

        # ★ 计算 base_advan：用于区分"LLM 意外生成" vs "LLM 故意改结构"
        base_code = getattr(state, 'best_code', None) or state.current_code
        base_advan = None
        if base_code:
            m = re.search(r'ADVAN(\d+)', base_code, re.IGNORECASE)
            base_advan = int(m.group(1)) if m else None

        for attempt in range(max_retries + 1):
            ok, actual, advan = validate_lock(code, state.locked_compartments)
            if ok:
                return code

            #  快速失败：LLM 故意改房室（base_advan 与生成 advan 不同）
            #  前提：base_advan 已知，且 prompt 中未授权改房室
            if (base_advan is not None
                    and advan is not None
                    and advan != base_advan
                    and 'advan_switch' not in prompt.lower()):
                raise CompartmentLockError(
                    f"LLM attempted to change ADVAN{base_advan} → ADVAN{advan} "
                    f"without explicit instruction. "
                    f"Locked to {state.locked_compartments}-cmt."
                )

            if attempt == max_retries:
                raise CompartmentLockError(
                    f"[{self.name}] After {max_retries + 1} attempts, "
                    f"still got ADVAN{advan} ({actual}-cmt), "
                    f"locked to {state.locked_compartments}-cmt"
                )

            print(f"  [{self.name}-LOCK] Attempt {attempt + 1}: "
                  f"ADVAN{advan} ({actual}-cmt) != locked "
                  f"{state.locked_compartments}-cmt. Retrying...")

            route = state.data_profile.get('route', 'oral')
            correction = build_correction_prompt(
                code, state.locked_compartments, route)
            response = llm.generate(prompt + correction,
                                    model_type=state.model)
            code = _extract_code(response)

        return code

