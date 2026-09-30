"""
skills/quality_evaluation.py

封装 _ai_quality_check 与 _calculate_composite_score。
输出写入 state.last_ai_recommendations / state.last_ai_critical_issues，
并更新 improvement_history 中的 ai_evaluation 字段。
"""

from .base import Skill, SkillResult
from modules.prompt_templates import PromptTemplates
from utils.safe_numeric import safe_float


class QualityEvaluationSkill(Skill):
    name = 'quality_evaluation'
    phase = [1, 3, 4]   # Phase 5 不做（SCM 有自身确定性阈值）
    tools = ['composite_scorer']
    prompt_template = 'quality_evaluation'

    def run(self, state, llm, tools) -> SkillResult:
        parsed = state.last_run.get('parsed_data', {})
        if not parsed:
            return SkillResult(ok=False, error='No parsed data for quality evaluation')

        # 1) 复合评分
        scorer = self._tool(tools, 'composite_scorer')
        composite = scorer.compute(parsed, state)

        # 2) AI 质量评价
        lst_output = state.last_run.get('full_output', '')
        prompt = PromptTemplates.quality_evaluation_prompt(
            iteration=state.iteration,
            parsed_data=parsed,
            previous_improvements=state.improvement_history,
            overfitting_warnings=[],
            # max_rse=parsed.get('rse_percent', {}).get('max_rse'),
            # high_rse_count=parsed.get('rse_percent', {}).get('high_rse_count', 0),
            max_rse=safe_float(parsed.get('rse_percent', {}).get('max_rse')),
            high_rse_count = int(safe_float(parsed.get('rse_percent', {}).get('high_rse_count', 0), default=0) or 0),
            plausibility_report=state.plausibility_report,
            lst_output=lst_output,
        )

        try:
            response = llm.generate(prompt, model_type=state.model)
            evaluation = _parse_json(response)
        except Exception as e:
            print(f"  [WARNING] AI quality evaluation failed: {e}")
            evaluation = {}

        should_continue = evaluation.get('should_continue', True)
        recs = evaluation.get('recommendations_if_continuing', [])
        critical = evaluation.get('critical_issues', [])

        return SkillResult(
            updates={
                'last_ai_recommendations': recs,
                'last_ai_critical_issues': critical,
            },
            meta={
                'composite_score': composite,
                'should_continue': should_continue,
                'evaluation': evaluation,
            },
        )


def _parse_json(text: str) -> dict:
    import re, json
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)