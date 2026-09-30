"""
skills/__init__.py

SkillRegistry：集中注册所有 Skill。

- 键为 Skill.name（如 'phase1_establish'）
- 值为 Skill 实例（无状态；所有状态保存在 OptimizationState）
"""

from .base import Skill, SkillResult, ValidationResult

# bootstrap
from .bootstrap.data_introspection import DataIntrospectionSkill
from .bootstrap.plausibility_bounds import PlausibilityBoundsSkill
from .bootstrap.dose_unit_check import DoseUnitCheckSkill

# 五阶段
from .phase1_establish import EstablishBaseModelSkill
from .phase2_diagnose import DiagnoseStructureSkill
from .phase3_reduce import ReduceOverfittingSkill
from .phase4_optimize_iiv import OptimizeIIVSkill
from .phase5_covariate_scm import CovariateSCMSkill

# 横切
from .nonmem_execution import NonmemExecutionSkill
from .output_parsing import OutputParsingSkill
from .quality_evaluation import QualityEvaluationSkill
from .plausibility_check import PlausibilityCheckSkill
from .phase_transition import PhaseTransitionSkill
from .termination import TerminationSkill
from .human_review import HumanReviewSkill
from .structure_selection import StructureSelectionSkill

def build_skill_registry(tools) -> dict:
    """
    构造 SkillRegistry 字典。所有 Skill 实例共享同一个 ToolRegistry。
    Skill 本身无状态，可安全复用。
    """
    return {
        # bootstrap
        'data_introspection':   DataIntrospectionSkill(),
        'plausibility_bounds':  PlausibilityBoundsSkill(),
        'dose_unit_check':      DoseUnitCheckSkill(),

        # 五阶段
        'phase1_establish':     EstablishBaseModelSkill(),
        'phase2_diagnose':      DiagnoseStructureSkill(),
        'phase3_reduce':        ReduceOverfittingSkill(),
        'phase4_optimize_iiv':  OptimizeIIVSkill(),
        'phase5_covariate_scm': CovariateSCMSkill(),
        'structure_selection': StructureSelectionSkill(),

        # 横切
        'nonmem_execution':     NonmemExecutionSkill(),
        'output_parsing':       OutputParsingSkill(),
        'quality_evaluation':   QualityEvaluationSkill(),
        'plausibility_check':   PlausibilityCheckSkill(),
        'phase_transition':     PhaseTransitionSkill(),
        'termination':          TerminationSkill(),
        'human_review':         HumanReviewSkill(),
    }


# 兼容别名：旧代码可能直接 import SKILLS
SKILLS = None  # 由 Orchestrator 在启动时用 build_skill_registry() 填充


__all__ = [
    'Skill', 'SkillResult', 'ValidationResult',
    'build_skill_registry', 'SKILLS',
    'DataIntrospectionSkill', 'PlausibilityBoundsSkill', 'DoseUnitCheckSkill',
    'EstablishBaseModelSkill', 'DiagnoseStructureSkill',
    'ReduceOverfittingSkill', 'OptimizeIIVSkill', 'CovariateSCMSkill',
    'NonmemExecutionSkill', 'OutputParsingSkill', 'QualityEvaluationSkill',
    'PlausibilityCheckSkill', 'PhaseTransitionSkill',
    'TerminationSkill', 'HumanReviewSkill',
]