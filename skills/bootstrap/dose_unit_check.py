"""
skills/bootstrap/dose_unit_check.py

封装 _detect_dose_unit_mismatch 逻辑：
- 比较 CV(AMT/WT) 与 CV(AMT*WT)的变异系数
- 结合 典型剂量 ，判断数据中的剂量单位是否可能不是 mg，并给出 F1 缩放建议，选择最可能的单位解释
- 结果写入 state.dose_scaling_hint，供 Phase 1 prompt 使用
"""

from ..base import Skill, SkillResult


class DoseUnitCheckSkill(Skill):
    name = 'dose_unit_check'
    phase = [0]
    tools = []
    prompt_template = ''

    def run(self, state, llm, tools) -> SkillResult:
        hint = _detect_dose_unit_mismatch(state)

        # ★ 临时调试打印
        stats = state.data_profile.get('dose_stats', {})
        print(f"  [DOSE-DEBUG] amt_mean={stats.get('amt_mean')}")
        print(f"  [DOSE-DEBUG] amt_median={stats.get('amt_median')}")
        print(f"  [DOSE-DEBUG] wt_mean={stats.get('wt_mean')}")
        print(f"  [DOSE-DEBUG] typical_dose_mg={state.plausibility_bounds.get('typical_single_dose_mg')}")
        print(f"  [DOSE-DEBUG] hint={hint!r}")


        return SkillResult(updates={'dose_scaling_hint': hint})


def _detect_dose_unit_mismatch(state) -> str:
    """纯确定性判定，不调用 LLM。"""
    stats = state.data_profile.get('dose_stats', {})  # 取剂量统计
    bounds = state.plausibility_bounds or {}   # 取典型单次剂量
    amt_mean = stats.get('amt_mean')
    typical_dose = bounds.get('typical_single_dose_mg')

    if amt_mean is None or not typical_dose or typical_dose <= 0:
        return ""

    wt_mean = stats.get('wt_mean')
    cv_div = stats.get('cv_amt_div_wt')
    cv_times = stats.get('cv_amt_times_wt')

    # 依据 CV 比较决定是否允许 mg/kg 类候选
    # 若 AMT/WT 的 CV 明显更小，说明 AMT 更像绝对 mg。
    # 若 AMT*WT 的 CV 明显更小，说明 AMT 更像 mg/kg 比例，需要乘体重换算成绝对 mg。
    weight_conversion_needed = None
    CV_CONFIDENT, CV_MARGIN = 0.3, 0.5
    if cv_div is not None and cv_times is not None:
        if cv_div < CV_CONFIDENT and cv_div < cv_times * CV_MARGIN:
            weight_conversion_needed = False   # AMT 已是绝对 mg
        elif cv_times < CV_CONFIDENT and cv_times < cv_div * CV_MARGIN:
            weight_conversion_needed = True    # AMT 是未换算的 mg/kg 比例

    # 有平均体重且没有明确“不需要体重换算”时，才允许 mg/kg 类候选
    allow_wt = bool(wt_mean) and (weight_conversion_needed is not False)

    # 绝对 mg：因子 1.0，无需缩放。
    # mg/kg：因子 WT，即 F1 = WT。
    # mcg：因子 0.001，即 F1 = 0.001。
    # mcg/kg：因子 WT * 0.001，即 F1 = WT * 0.001。
    # g：因子 1000，即 F1 = 1000。

    # ★ 识别"weekly vs daily"剂量差异
    #   warfarin 典型 5 mg/day = 35 mg/week，AMT 可能是周剂量
    #   或 5 mg/day × 7 = 35 mg，或 105 mg/week
    # ★ 新增：weekly/daily 换算候选
    WEEKLY_FACTORS = [1.0, 7.0, 30.0, 7 * 7.0]  # 1, 7, 30, 49

    candidates = [("absolute mg (no scaling needed)", 1.0)]
    if allow_wt:
        candidates.append((f"mg/kg (F1 = WT, mean WT={wt_mean:.1f})", wt_mean))
    candidates.append(("mcg, not mg (F1 = 0.001)", 0.001))
    if allow_wt:
        candidates.append((f"mcg/kg (F1 = WT * 0.001, mean WT={wt_mean:.1f})", wt_mean * 0.001))
    candidates.append(("g, not mg (F1 = 1000)", 1000.0))

    for wf in WEEKLY_FACTORS:
        if wf != 1.0:
            candidates.append(
                (f"weekly dosing ({wf}× daily)", wf))

    # 对每个候选计算“换算后的隐含剂量” implied_dose。
    # 与典型剂量比较，比例在 0.3 - 3.0 为合理。
    # 选最接近典型剂量的候选，即 score = abs(1 - ratio) 最小者。
    best_label, best_factor, best_score = None, None, None
    for label, factor in candidates:
        implied_dose = amt_mean * factor
        ratio = implied_dose / typical_dose
        # 放宽 ratio 范围（3.0 → 10.0）以容纳 weekly-daily 差异
        if 0.05 <= ratio <= 50.0:
        # if 0.1 <= ratio <= 10.0:
        # if 0.3 <= ratio <= 3.0:
            score = abs(1.0 - ratio)
            if best_score is None or score < best_score:
                best_label, best_factor, best_score = label, factor, score

    if best_label is None or best_factor == 1.0:
        return ""

    # ★ weekly-daily 场景：返回提示但不强制缩放 F1
    if any(wf > 1.0 and best_factor == wf for wf in WEEKLY_FACTORS):
        return (
            f"NOTE: dataset AMT mean = {amt_mean:.3g}, "
            f"typical clinical dose for this drug ≈ {typical_dose}mg/day. "
            f"The AMT appears to be WEEKLY dosing ({best_factor:.0f}× daily). "
            f"This is not a unit mismatch — ensure the $PK model handles "
            f"weekly dosing (e.g., divide dose by {best_factor:.0f} or "
            f"use appropriate dosing interval)."
        )
    # 返回强提示字符串，告知 Phase 1：
    # 检测到剂量单位可能不匹配；
    # 数据集 AMT 均值、典型临床剂量；
    # 最可能匹配的单位解释；
    # 必须在 $PK 中加入合适的 F1 缩放，使进入房室的剂量为 mg，与 DV 浓度单位一致。
    return (
        f"DOSE UNIT MISMATCH DETECTED: dataset AMT mean = {amt_mean:.3g}, "
        f"typical clinical dose for this drug ≈ {typical_dose}mg. "
        f"This matches: {best_label}. "
        f"MANDATORY: add the appropriate F1 scaling in $PK "
        f"(F1 = WT if mg/kg, F1 = 0.001 if mcg, F1 = WT*0.001 if mcg/kg, etc.) "
        f"so the actual administered dose reaching the compartment is in mg, "
        f"consistent with DV's concentration units."
    )