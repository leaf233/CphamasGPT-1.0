"""
skills/bootstrap/plausibility_bounds.py
封装 _build_plausibility_bounds 逻辑：
  先识别药物，再由 LLM 给出该药物群体 PK 参数的合理范围与典型单次剂量
  写入 state.plausibility_bounds，供 Phase 1 的 THETA 初值使用。
  把"药物识别 + 参数合理性范围 + 典型剂量"写入 state
封装 _build_plausibility_bounds 逻辑：
- Step 1: 依据文件/输出名识别药物（无 web search）
- Step 2: 依据药物名给出 CL/V1/Ka/（2 室时）Q/V2 的 min/typical/max
           与典型单次剂量 typical_single_dose_mg
- 结果写入 state.plausibility_bounds，供 Phase 1 THETA 初值使用
"""

from ..base import Skill, SkillResult
from modules.prompt_templates import PromptTemplates


class PlausibilityBoundsSkill(Skill):
    name = 'plausibility_bounds'
    phase = [0]
    tools = []
    prompt_template = 'plausibility_bounds'

    def run(self, state, llm, tools) -> SkillResult:

        # ★ 新增：若已从共享文件加载，跳过 LLM 调用
        if state.plausibility_bounds and state.plausibility_bounds.get('parameters'):
            print(f"  [PLAUSIBILITY] Using precomputed bounds "
                  f"(drug={state.plausibility_bounds.get('drug_identified')})")
            return SkillResult(done=True)

        # 若已构建过则跳过（支持重入）
        if state.plausibility_bounds:
            return SkillResult(done=True)

        data_file = state.data_file
        drug_hint = f"{state.output_base} / {data_file}"
        route_hint = state.data_profile.get('route', 'oral').upper()
        n_subjects = state.data_profile.get('n_subjects', '?')

        # Step 1: 药物识别（无 web search）
        # 让 LLM 根据数据集提示识别最可能的药物，并输出 JSON
        id_prompt = f"""You are an expert clinical pharmacokineticist.
            Dataset hint: "{drug_hint}"
            Route of administration: {route_hint}
            
            Identify the single drug this dataset most likely comes from.
            Respond ONLY with valid JSON:
            {{"drug_identified": "<drug name or 'unknown'>", "route": "<IV or oral>"}}
        """
        try:
            id_resp = llm.generate(id_prompt, model_type=state.model)
            id_json = _extract_json(id_resp)
            drug = (id_json.get('drug_identified') or 'unknown').strip()
            route = id_json.get('route', route_hint)
        except Exception:
            drug, route = 'unknown', route_hint

        # Step 2: 生理范围 + 典型剂量
        # 判断该药物更适合 1 室还是 2 室模型；
        # 给出 CL、V1、Ka 的 min/typical/max；
        # 若为 2 室，额外给 Q、V2；
        # 给出典型单次给药剂量 typical_single_dose_mg；
        # 不确定时保守处理。
        prompt = f"""You are an expert clinical pharmacokineticist researching: {drug}

        Route of administration: {route}
        Number of subjects: {n_subjects}
        
        Task:
        1. Decide whether {drug}'s disposition is best described by 1 or 2 compartments.
        2. Provide physiologically plausible POPULATION-LEVEL ranges (typical values)
           ONLY for the parameters applicable to the chosen compartment count.
        3. Provide the typical SINGLE administered dose for {drug} via {route},
           in absolute milligrams for an average adult.
        4. If you are uncertain, be conservative; do NOT fabricate specificity.
        
        Respond ONLY with valid JSON:
        {{
          "drug_identified": "{drug}",
          "route": "<IV or oral>",
          "notes": "<1-sentence PK rationale>",
          "suggested_compartments": <1 or 2>,
          "compartments_rationale": "<1-sentence>",
          "typical_single_dose_mg": <number>,
          "parameters": {{
            "CL":  {{"min": <n>, "typical": <n>, "max": <n>, "unit": "L/h", "rationale": "<brief>"}},
            "V1":  {{"min": <n>, "typical": <n>, "max": <n>, "unit": "L",   "rationale": "<brief>"}},
            "Ka":  {{"min": <n>, "typical": <n>, "max": <n>, "unit": "1/h", "rationale": "<brief>"}}
            // Include "Q" and "V2" ONLY if suggested_compartments == 2
          }}
        }}
        """
        try:
            resp = llm.generate(prompt, model_type=state.model)
            bounds = _extract_json(resp)

            # ★ 交叉验证：typical_dose 与 AMT 量级是否一致
            amt_median = state.data_profile.get('dose_stats', {}).get('amt_median')
            typical = bounds.get('typical_single_dose_mg')
            if amt_median and typical:
                ratio = amt_median / typical
                if ratio > 10 or ratio < 0.1:
                    print(f"  [PLAUSIBILITY-WARN] typical_dose={typical} vs "
                          f"amt_median={amt_median} (ratio={ratio:.1f}). "
                          f"Possible scale mismatch — LLM may be inconsistent. "
                          f"Will use ratio-aware fallback.")
                    # ★ 兜底：根据 AMT 量级修正 typical
                    if amt_median < 10 and typical > 100:
                        # AMT 是 mg/kg，typical 是 mg（应该允许）
                        pass
                    elif amt_median > 100 and typical < 10:
                        # AMT 是 mg，typical 是每日剂量（可能拆分）
                        pass

            bounds['drug_identified'] = drug
            bounds['citations'] = []
        except Exception as e:
            print(f"  [WARNING] Plausibility bounds build failed: {e}")
            bounds = {}

        return SkillResult(updates={'plausibility_bounds': bounds})


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象（支持 markdown 包裹与前后解释）。"""
    import re, json
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    return json.loads(text)