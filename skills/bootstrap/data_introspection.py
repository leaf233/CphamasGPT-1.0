"""
skills/bootstrap/data_introspection.py

封装 PKDataLoader，完成：
- NONMEM 标准列识别（ID/TIME/DV/AMT/EVID/MDV/CMT/RATE）
- 协变量检测与类型（continuous / categorical）
- 给药途径推断（IV / oral）
- 隔室数量数据驱动推断（1 / 2 / 3）
- AMT 剂量统计（用于剂量单位判定）
"""

from ..base import Skill, SkillResult
from modules.data_loader import PKDataLoader


class DataIntrospectionSkill(Skill):
    name = 'data_introspection'
    phase = [0]  # bootstrap
    tools = []   # 直接调用 PKDataLoader，不经过 ToolRegistry
    prompt_template = ''

    def run(self, state, llm, tools) -> SkillResult:
        # 用当前数据文件 state.data_file 创建 PKDataLoader
        loader = PKDataLoader(state.data_file)
        metadata = loader.get_metadata()

        # 若用户强制指定房室，覆盖数据推断值
        if state.forced_compartments is not None:
            metadata['compartments'] = state.forced_compartments
        # 列摘要、数据摘要、协变量摘要
        # 标准列识别结果：columns、nonmem_columns
        # 协变量列表与类型信息
        # 受试者数 n_subjects
        # 给药途径 route，默认口服
        # 房室数 compartments，默认 1
        # 剂量统计 dose_stats，后面给 DoseUnitCheckSkill 用
        # 先验信息 prior_information
        data_profile = {
            'column_summary': loader.get_column_summary(),
            'data_summary':   loader.get_data_summary(),
            'covariate_summary': loader.get_covariate_summary(),
            'columns':        metadata.get('columns', []),
            'nonmem_columns': metadata.get('nonmem_columns', {}),
            'covariates':     metadata.get('covariates', []),
            'covariate_info': metadata.get('covariate_info', {}),
            'n_subjects':     metadata.get('n_subjects', 0),
            'route':          metadata.get('route', 'oral'),
            'compartments':   metadata.get('compartments', 1),
            'dose_stats':     metadata.get('dose_stats', {}),
            'prior_information': loader.get_metadata().get('prior_information', None),
        }
        # 把原始数据文件变成结构化画像，包括列信息、协变量、途径、房室数、剂量统计
        return SkillResult(updates={
            'data_profile': data_profile,
            'data_loader':  loader,       # 供 Tool 复用，不重复 IO
        })