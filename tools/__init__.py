"""
tools/__init__.py

ToolRegistry：集中注册所有 Tool。

- 键为工具名称（字符串）
- 值为 Tool 实例（有状态；共享同一个 OptimizationState 与 LLM client）
- Skill / Agent / Guardrail 通过 tools['name'] 取用
"""

from __future__ import annotations
from typing import Any, Dict
# 延迟导入，避免循环依赖
from .nonmem_runner import NonmemRunner
from .cs_editor import ControlStreamEditor
from .input_normalizer import InputNormalizer
from .foce_enforcer import FOCEEnforcer
from .advan_validator import AdvanValidator
from .scm_runner import SCMRunner
from .composite_scorer import CompositeScorer

# Guardrail 封装（来自 guardrails/ 目录）
from guardrails.compartment_invariance import CompartmentInvarianceGuard
from guardrails.advan_trans_validity import AdvanTransValidityGuard
from guardrails.numeric_safety import NumericSafetyGate
from guardrails.phase5_structure_guard import Phase5StructureGuard
from guardrails.plausibility_guard import PlausibilityGuard

# Parser 直接复用 modules/nonmem_parser
from modules.nonmem_parser import NONMEMParser
from guardrails.omega_structure_guard import OmegaStructureGuard  # ★ 新增


class ToolRegistry(dict):
    """
    字典风格的 Tool 容器。

    用途：
      tools = ToolRegistry(state, llm)
      tools['nonmem_runner'].execute(code, state)
      tools['composite_scorer'].compute(parsed, state)

    支持属性访问：
      tools.nonmem_runner  与 tools['nonmem_runner'] 等价
    """

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Tool '{name}' not registered") from exc


def build_tool_registry(state, llm) -> ToolRegistry:
    """
    构造 ToolRegistry。

    参数：
      state : OptimizationState（工具共享同一 state 引用）
      llm   : MultiModelOpenRouterClient（供内部 AI 解析使用）

    说明：
      - NONMEM runner 需要 state.nmfe_command / output_base / iteration / data_file
      - Parser 需要 llm 用于 AI 解析（带正则回退）
      - Guardrail 类工具（compartment_guard / advan_trans_validity / numeric_safety_gate
        / phase5_structure_guard）也在 ToolRegistry 中注册，供 Skill / Agent 取用
    """


    class _ParserTool:
        """薄封装：把 NONMEMParser 变成无状态 Tool 接口。"""
        def __init__(self, llm):
            self.llm = llm

        def parse(self, lst_path: str) -> dict:
            parser = NONMEMParser(lst_path, gemini_client=self.llm,
                                  use_ai_parsing=True)
            return parser.get_parsed_data()

        def diagnose_structural_model(self, parsed: dict) -> dict:
            # 复用 NONMEMParser.diagnose_structural_model 的静态逻辑
            class _P:
                def __init__(self, parsed):
                    self.parsed_data = parsed
                def get_parsed_data(self):
                    return self.parsed_data
            return NONMEMParser.diagnose_structural_model(_P(parsed))

        def identify_boundary_theta(self, lst_output: str) -> list:
            import re
            if not lst_output:
                return []
            grad_lines = re.findall(
                r'GRADIENT:\s+((?:[+-]?\s*\d+\.\d+E[+-]\d+\s*)+)',
                lst_output, re.IGNORECASE)
            if not grad_lines:
                return []
            last = grad_lines[-1]
            vals = re.findall(r'[+-]?\d+\.\d+E[+-]\d+', last)
            return [i for i, v in enumerate(vals, start=1) if float(v) == 0.0]

    tools = ToolRegistry()

    # ---- 执行 / 解析 ------------------------------------------------------ #
    tools['nonmem_runner']      = NonmemRunner()
    tools['nonmem_parser']      = _ParserTool(llm)

    # ---- 控制流编辑 ------------------------------------------------------- #
    tools['cs_editor']          = ControlStreamEditor()
    tools['input_normalizer']   = InputNormalizer()
    tools['foce_enforcer']      = FOCEEnforcer()
    tools['advan_validator']    = AdvanValidator()

    # ---- SCM 专用 ---------------------------------------------------------- #
    tools['scm_runner']         = SCMRunner()

    # ---- 评分 -------------------------------------------------------------- #
    tools['composite_scorer']   = CompositeScorer()

    # ---- Guardrail --------------------------------------------------------- #
    tools['compartment_guard']  = CompartmentInvarianceGuard()
    tools['advan_trans_validity'] = AdvanTransValidityGuard()
    tools['numeric_safety_gate']  = NumericSafetyGate()
    tools['phase5_structure_guard'] = Phase5StructureGuard()
    tools['plausibility_guard']     = PlausibilityGuard()
    tools['omega_structure_guard'] = OmegaStructureGuard()

    return tools


__all__ = ['ToolRegistry', 'build_tool_registry']