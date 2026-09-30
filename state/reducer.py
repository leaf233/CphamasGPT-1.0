"""
state/reducer.py

StateReducer：将 SkillResult.updates 合并回 OptimizationState。

合并规则：
- 字典字段（last_run / flags / scm / data_profile / plausibility_bounds /
  plausibility_report / strategy_repeat_count）：浅合并（保留旧键）
- 列表字段（improvement_history / phase_history / code_history /
  parameter_history / covariate_history）：若 updates 提供则替换
- 集合字段（scm.round_tested）：替换
- 其余标量字段（current_code / best_ofv 等）：直接赋值

设计原则：
- 唯一写入路径：所有 Skill 都通过 SkillResult.updates → StateReducer.apply 写入
- 显式白名单：只有 State 上已存在的字段才允许写入
- 不修改 State 以外对象：StateReducer 不做业务判定
"""

from __future__ import annotations
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, Set

from .state import OptimizationState
from .scm_state import SCMState


class StateReducer:
    """SkillResult.updates → OptimizationState 合并器。"""

    # ── 合并策略分类 ─────────────────────────────────────────────────── #

    # 浅合并的字典字段
    _MERGE_DICT_FIELDS: Set[str] = {
        'last_run',
        'flags',
        'data_profile',
        'plausibility_bounds',
        'plausibility_report',
        'strategy_repeat_count',
        'prior_info',
    }

    # 替换的列表字段（Skill 提供完整列表）
    _REPLACE_LIST_FIELDS: Set[str] = {
        'improvement_history',
        'phase_history',
        'code_history',
        'parameter_history',
        'covariate_history',
        'failed_strategies',
        'last_ai_recommendations',
        'last_ai_critical_issues',
    }

    # SCM 子对象需特殊处理（内部字段合并）
    _SCM_FIELD = 'scm'

    # ── 主入口 ───────────────────────────────────────────────────────── #

    @staticmethod
    def apply(state: OptimizationState, result) -> OptimizationState:
        """
        将 SkillResult.updates 合并到 state。

        参数：
          state  : OptimizationState
          result : SkillResult（可能为 None）

        返回：
          修改后的 state（同一引用）
        """
        if result is None:
            return state

        updates = getattr(result, 'updates', None) or {}
        if not updates:
            return state

        # 允许写入的字段白名单（State 已声明的字段）
        valid_fields = {f.name for f in dataclass_fields(OptimizationState)}

        for key, value in updates.items():
            if key not in valid_fields:
                # 忽略未知字段，避免拼写错误污染 State
                print(f"  [StateReducer] Unknown field '{key}' ignored")
                continue

            # SCM 子状态合并
            if key == StateReducer._SCM_FIELD:
                StateReducer._merge_scm(state, value)
                continue

            # 字典浅合并
            if key in StateReducer._MERGE_DICT_FIELDS and isinstance(value, dict):
                existing = getattr(state, key, None) or {}
                merged = {**existing, **value}
                setattr(state, key, merged)
                continue

            # 列表 / 集合替换
            if key in StateReducer._REPLACE_LIST_FIELDS:
                setattr(state, key, value)
                continue

            # 其余直接赋值
            setattr(state, key, value)

        return state

    # ── SCM 子状态合并 ───────────────────────────────────────────────── #

    @staticmethod
    def _merge_scm(state: OptimizationState, value: Any) -> None:
        """
        合并 SCMState。

        - 若 value 是 SCMState：直接替换
        - 若 value 是 dict：逐字段更新到现有 state.scm
        """
        if isinstance(value, SCMState):
            state.scm = value
            return

        if not isinstance(value, dict):
            return

        scm = state.scm
        scm_fields = {f.name for f in dataclass_fields(SCMState)}
        for k, v in value.items():
            if k in scm_fields:
                setattr(scm, k, v)

    # ── 便捷方法 ─────────────────────────────────────────────────────── #

    @staticmethod
    def append_to_list(state: OptimizationState, field_name: str, item: Any) -> None:
        """
        向列表字段追加单个条目。

        供 Skill 在需要增量追加（而非整体替换）时使用。
        """
        current = getattr(state, field_name, None) or []
        setattr(state, field_name, current + [item])

    @staticmethod
    def snapshot(state: OptimizationState) -> Dict[str, Any]:
        """
        生成 State 快照（仅可序列化字段），便于调试与日志。
        """
        skip = {'llm', 'tools', 'skills', 'data_loader', 'phase_manager'}
        snap = {}
        for f in dataclass_fields(OptimizationState):
            if f.name in skip:
                continue
            value = getattr(state, f.name)
            if f.name == 'scm':
                snap['scm'] = {
                    'mode':              value.mode,
                    'round':             value.round,
                    'confirmed_count':   len(value.confirmed),
                    'eliminated_count':  len(value.eliminated),
                    'round_tested':      list(value.round_tested),
                    'round_base_ofv':    value.round_base_ofv,
                    'complete':          value.complete,
                }
            else:
                snap[f.name] = value
        return snap