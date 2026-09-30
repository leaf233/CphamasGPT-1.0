"""
agents/skill_selector.py

SkillSelector：依据 State 决定当前迭代应该调用哪个 Skill。

设计原则：
- 确定性优先：能用规则判断就不调用 LLM
- 与 skills/__init__.py 中的 registry key 一致
- Phase 2 不再返回 advan_switch 相关路径；房室由 --compartments 锁定
"""

from __future__ import annotations


class SkillSelector:
    """根据当前 phase 与 last_run 选择 Skill。"""

    @staticmethod
    def select(state, skills: dict):
        """
        参数：
          state  : OptimizationState
          skills : SkillRegistry（由 build_skill_registry() 构造）

        返回：
          Skill 实例
        """
        # ── Phase 1：建立基础模型 ──────────────────────────────────────
        if state.phase == 1:

            # 无代码 → 首次生成
            if not state.current_code:
                return skills['phase1_establish']

            # ★ 新增：minimization 未成功 → 必须生成修复代码，而非推进
            parsed = state.last_run.get('parsed_data') or {}
            if not parsed.get('minimization_successful', False):
                return skills['phase1_establish']

            # 已达阶段内迭代上限但仍未收敛 → 修复
            if state.iterations_in_phase >= 3:
                return skills['phase1_establish']

            # 首次生成由 orchestrator 显式调用，不经过这里
            # 若上一轮有 compile/syntax 错误，继续用 phase1_establish 修复
            if state.last_run.get('errors'):
                return skills['phase1_establish']

            # 只有上一轮成功最小化，交给 phase_transition 推进
            return skills['phase_transition']

        # ── Phase 2：诊断与稳定性（不升级房室）────────────────────────
        if state.phase == 2:
            return skills['phase2_diagnose']

        # ── Phase 3：过拟合控制 ────────────────────────────────────────
        if state.phase == 3:
            return skills['phase3_reduce']

        # ── Phase 4：IIV 优化 ─────────────────────────────────────────
        if state.phase == 4:
            return skills['phase4_optimize_iiv']

        # ── Phase 5：SCM（Skill 内部完成执行 + 解析）───────────────────
        if state.phase == 5:
            return skills['phase5_covariate_scm']

        # ── fallback ──────────────────────────────────────────────────
        return skills['quality_evaluation']