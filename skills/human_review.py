"""
skills/human_review.py

人工审核门控。

触发条件（任一）：
1. covariance step failure
2. 参数超出 SEVERE 生理范围
3. 多次独立运行结构不一致
4. 未识别强制药理协变量
5. shrinkage 落在 20-30% 与 95% 之间的未刻画区间
6. numeric_safety_gate 返回 False

输出：审查包（best model .lst / SCM 历史 / plausibility report / 建议方向），
      暂停自动流程，等待专家决策。
"""

import json
import os
from .base import Skill, SkillResult


class HumanReviewSkill(Skill):
    name = 'human_review'
    phase = [1, 2, 3, 4, 5]
    tools = []
    prompt_template = ''

    def required(self, state) -> bool:
        parsed = state.last_run.get('parsed_data', {})

        # 1) covariance failure
        if not parsed.get('covariance_step', {}).get('successful', True):
            return True

        # 2) SEVERE 生理范围
        report = state.plausibility_report or {}
        if any('SEVERE' in v for v in report.get('violations', [])):
            return True

        # 3) numeric_safety 失败
        if state.last_run.get('numeric_safety_failed'):
            return True

        # 4) 未识别强制协变量（示例：CLCR 在 tobramycin 类）
        #    实际判定由 domain checklist 提供，此处仅示意
        mandatory = state.data_profile.get('mandatory_covariates', [])
        if mandatory:
            confirmed = {c['covariate'] for c in state.scm.confirmed}
            missing = set(mandatory) - confirmed
            if missing and state.phase == 5 and state.scm_complete:
                return True

        # 5) shrinkage 中间区
        shrink = (parsed.get('eta_shrinkage') or [])
        if shrink:
            mx = max(s['shrinkage'] for s in shrink)
            if 30 <= mx <= 95 and state.phase == 4 and state.iterations_in_phase >= 5:
                return True

        return False

    def pause(self, state) -> SkillResult:
        """输出审查包，暂停自动流程。"""
        review_dir = f"{state.output_base}_review"
        os.makedirs(review_dir, exist_ok=True)

        best_lst = f"{state.output_base}_iter{state.best_iteration}.lst"
        if os.path.exists(best_lst):
            import shutil
            shutil.copy(best_lst, os.path.join(review_dir, 'best_model.lst'))

        package = {
            'iteration':          state.iteration,
            'phase':              state.phase,
            'best_iteration':     state.best_iteration,
            'best_ofv':           state.best_ofv,
            'best_composite':     state.best_composite,
            'plausibility_report': state.plausibility_report,
            'scm_confirmed':      [c['name'] for c in state.scm.confirmed],
            'scm_eliminated':     [e['name'] for e in state.scm.eliminated],
            'recommended_actions': [
                'Review best_model.lst for covariance / boundary issues',
                'Check plausibility_report for SEVERE violations',
                'Compare with parallel 1/2/3-compartment runs',
            ],
        }
        with open(os.path.join(review_dir, 'package.json'), 'w', encoding='utf-8') as f:
            json.dump(package, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n[HUMAN REVIEW] Package written to {review_dir}")
        print(f"[HUMAN REVIEW] Waiting for expert decision...")

        return SkillResult(meta={
            'paused': True,
            'review_dir': review_dir,
            'package': package,
        })

    # ========== 新增：实现抽象方法 run ==========
    def run(self, state, llm, tools) -> SkillResult:
        """
        Skill 入口。若满足人工审核条件，则输出审查包并暂停；
        否则返回一个正常的、未暂停的结果。
        """
        if self.required(state):
            return self.pause(state)
        return SkillResult(ok=True, done=False)   # 无需人工介入，继续自动流程
    # ==========================================