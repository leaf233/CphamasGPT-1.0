"""
agents/human_gateway.py

HumanGateway：人工审核门控。

当 HumanReviewSkill.required() 为 True 时被 Orchestrator 调用。
职责：
1. 调用 HumanReviewSkill.pause() 输出审查包
2. 等待专家决策（continue / revert / abort）
3. 返回决策字符串给 Orchestrator

决策入口：
- 默认非交互式：读取环境变量 PKGPT_HUMAN_DECISION
  - 'continue'（默认，仅在 PKGPT_HUMAN_INTERACTIVE=0 时）
  - 'revert'
  - 'abort'
- 交互式（PKGPT_HUMAN_INTERACTIVE=1）：在终端提示用户输入
"""

from __future__ import annotations

import os
import sys


class HumanGateway:
    """人工审核门控。"""

    def pause(self, state) -> str:
        """
        暂停自动流程并收集专家决策。

        返回：
          'continue'  — 采纳审查包中的建议，继续自动流程
          'revert'    — 回退到 best model 后继续
          'abort'     — 终止优化
        """
        # 1) 通过 Skill 输出审查包（写文件 + 打印摘要）
        review_result = state.skills['human_review'].pause(state) \
            if hasattr(state, 'skills') else None

        # 2) 打印审查摘要
        self._print_summary(state)

        # 3) 收集决策
        # ★ 非交互模式：累计 review 次数，连续无 best 更新则强制 abort
        if os.getenv('PKGPT_HUMAN_INTERACTIVE', '0') != '1':
            state.human_review_count = getattr(state, 'human_review_count', 0) + 1
            MAX_NONINTERACTIVE_REVIEWS = 5

            if state.human_review_count >= MAX_NONINTERACTIVE_REVIEWS:
                if state.best_iteration == state.last_best_iteration_at_review:
                    print(f"\n[HUMAN] {state.human_review_count} consecutive reviews "
                          f"with no best-model improvement — forcing ABORT.")
                    state.force_stop = True
                    state.force_stop_reason = (
                        f"{state.human_review_count} consecutive human-review cycles "
                        f"without improvement to best model "
                        f"(best_iter={state.best_iteration}). "
                        f"Likely persistent covariance failure or model misspecification."
                    )
                    return 'abort'
                state.last_best_iteration_at_review = state.best_iteration

        if os.getenv('PKGPT_HUMAN_INTERACTIVE', '0') == '1':
            return self._prompt_interactive(state)

        decision = os.getenv('PKGPT_HUMAN_DECISION', 'continue').lower()
        if decision not in ('continue', 'revert', 'abort'):
            print(f"[HUMAN] Unknown PKGPT_HUMAN_DECISION='{decision}', "
                  f"defaulting to 'continue'")
            decision = 'continue'

        print(f"[HUMAN] Non-interactive decision: {decision}")
        return decision

    # ---- 内部方法 -------------------------------------------------------- #

    def _print_summary(self, state):
        print("\n" + "!" * 70)
        print("HUMAN REVIEW REQUIRED")
        print("!" * 70)
        print(f"  Iteration       : {state.iteration}")
        print(f"  Phase           : {state.phase}")
        print(f"  Best iteration  : {state.best_iteration}")
        print(f"  Best OFV        : {state.best_ofv}")
        print(f"  Composite score : {state.best_composite}")

        report = state.plausibility_report or {}
        if report.get('violations'):
            print(f"  Plausibility    :")
            for v in report['violations']:
                print(f"    - {v}")

        confirmed = [c['name'] for c in state.scm.confirmed]
        if confirmed:
            print(f"  SCM confirmed   : {confirmed}")

        print("!" * 70)

    def _prompt_interactive(self, state) -> str:
        prompt = (
            "\n[HUMAN] Choose action:\n"
            "  [c] continue (accept review package and resume)\n"
            "  [r] revert   (revert to best model and resume)\n"
            "  [a] abort    (stop optimization)\n"
            "> "
        )
        try:
            choice = input(prompt).strip().lower()
        except EOFError:
            print("[HUMAN] No input available, defaulting to 'continue'")
            return 'continue'

        mapping = {'c': 'continue', 'r': 'revert', 'a': 'abort',
                   '': 'continue'}
        decision = mapping.get(choice, 'continue')
        print(f"[HUMAN] Decision: {decision}")
        return decision