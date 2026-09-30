"""
定义所有可用技能的元数据，用于LLM调度和前端展示。
"""

SKILLS = {
    "init": {
        "description": "加载CSV数据集，推断给药途径和房室数，生成初始NONMEM控制流（$PROBLEM到$TABLE完整代码）。",
        "params": {},
        "example": "用户上传数据后，系统自动调用此技能初始化模型。"
    },
    "run_nonmem": {
        "description": "执行当前控制流，调用NONMEM，生成.lst输出文件。",
        "params": {},
        "example": "运行NONMEM以获取新模型的OFV和诊断信息。"
    },
    "parse_output": {
        "description": "解析最近一次NONMEM输出的.lst文件，提取OFV、参数估计、RSE%、收缩率、警告/错误。",
        "params": {},
        "example": "解析运行结果以评估模型质量。"
    },
    "phase1_fix": {
        "description": "进入Phase 1：修复语法错误、边界问题、估计方法，使模型能够成功执行最小化。",
        "params": {},
        "example": "当运行失败或出现边界警告时调用。"
    },
    "phase2_diagnose": {
        "description": "进入Phase 2：诊断结构模型，检查CWRES模式，必要时升级房室（如ADVAN2→ADVAN4）。",
        "params": {},
        "example": "当残留误差存在系统趋势时调用。"
    },
    "phase3_reduce": {
        "description": "进入Phase 3：减少过拟合，包括移除不显著的OMEGA参数、简化误差模型、固定边界THETA。",
        "params": {},
        "example": "当收缩率>90%或OFV<-50时强制调用。"
    },
    "phase4_optimize": {
        "description": "进入Phase 4：优化随机效应结构，根据数据集大小调整OMEGA的DIAGONAL/BLOCK，调整THETA初值。",
        "params": {},
        "example": "在基础模型稳定后精细调整IIV。"
    },
    "phase5_covariate": {
        "description": "进入Phase 5：对下一个候选协变量进行单步SCM测试（forward或backward），仅添加/移除一个协变量。",
        "params": {},
        "example": "在Phase 4完成后启动，逐步筛选协变量。"
    },
    "summarize": {
        "description": "生成最终模型报告，包括参数表、诊断图（GOF/VPC）和房室对比。",
        "params": {},
        "example": "建模完成后调用，展示结果。"
    }
}

def get_all_skill_descriptions():
    """返回格式化的技能描述文本，用于系统提示。"""
    lines = []
    for name, info in SKILLS.items():
        lines.append(f"- {name}: {info['description']}")
    return "\n".join(lines)