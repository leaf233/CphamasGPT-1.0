# agent.py (新增文件)
import os
from typing import Optional, Dict, List, Callable, Any
from modules.optimizer import NONMEMOptimizer
from modules.nonmem_parser import NONMEMParser
from modules.prompt_templates import PromptTemplates
import json
from skill_descriptions import SKILLS, get_all_skill_descriptions

class PKGPTAgent:
    def __init__(self, data_file: str, output_dir: str,
                 min_iter: int, max_iter: int, model: str,
                 api_key: str, nmfe: str = "nmfe75.bat"):
        self.data_file = data_file
        self.output_dir = output_dir
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.model = model
        self.api_key = api_key
        self.nmfe = nmfe

        # 状态上下文
        self.context = {
            "iteration": 0,
            "current_code": None,
            "best_code": None,
            "best_ofv": None,
            "history": [],        # 每个元素为 {iteration, ofv, status, phase, ...}
            "last_lst": None,
            "last_parsed": None,
            "current_phase": "INIT",   # 当前阶段描述
            "scm_confirmed": [],       # 已确认协变量
            "scm_round": 0,
            "covariate_candidates": [],  # 待测试协变量列表
            "user_confirmed": False,   # 用于交互确认标志
        }

        # 状态
        self.phase = 0                 # 当前 Phase 编号 (1-5)
        self.iteration = 0
        self.history = []
        self.best_ofv = None
        self.best_code = None
        self.current_code = None

        # 等待用户确认标志
        self.waiting_for_user = False
        self.user_decision = None

        # 回调函数（由前端注册）
        self.on_log = None
        self.on_phase_change = None
        self.on_iteration_end = None
        self.on_need_confirm = None

        # 延迟初始化 Optimizer
        self.optimizer = None
        self.initialize_optimizer()

        # 注册技能
        self.skills = {}
        self.register_skills()

        # 对话历史
        self.conversation_history = []  # 存储 (user, assistant)


    def initialize_optimizer(self):
        """创建 NONMEMOptimizer 实例，但仅用于工具调用"""
        if self.optimizer is None:
            self.optimizer = NONMEMOptimizer(
                data_file=self.data_file,
                output_base=os.path.join(self.output_dir, 'result'),
                api_key=self.api_key,
                min_iterations=self.min_iter,
                max_iterations=self.max_iter,
                nmfe_command=self.nmfe,
                model=self.model,
                forced_compartments=None
            )
            # 继承一些属性
            self.current_code = self.optimizer.current_code if hasattr(self.optimizer, 'current_code') else None

    # ---------- 工具方法 ----------
    def tool_run_nonmem(self, code: str) -> Dict:
        """执行 NONMEM，返回结果摘要"""
        self.optimizer.current_code = code
        success = self.optimizer._run_nonmem()
        lst_path = f"{self.optimizer.output_base}_iter{self.optimizer.iteration}.lst"
        return {
            'success': success,
            'lst_path': lst_path if os.path.exists(lst_path) else None,
            'iteration': self.optimizer.iteration
        }

    def tool_parse_output(self, lst_path: str) -> Dict:
        """解析 .lst 文件，返回结构化数据"""
        parser = NONMEMParser(lst_path, gemini_client=self.optimizer.ds_client.clients.get(self.model))
        return parser.get_parsed_data() if parser else {}

    def tool_generate_code(self, phase: int, feedback: Optional[Dict] = None) -> str:
        """根据阶段和反馈生成新代码"""
        # 直接调用 optimizer 的生成方法
        if feedback is None:
            # 初始生成
            self.optimizer._generate_initial_code()
        else:
            # 改进生成：需要构造一个 parser 对象或直接传参
            # 这里我们可以用模拟 parser 或直接调用 _generate_improved_code
            # 但 _generate_improved_code 期望 parser 参数，我们可传入 None 并使用现有状态
            self.optimizer._generate_improved_code(None)
        return self.optimizer.current_code

    def tool_plot_gof_vpc(self, model_dir: str, compartment: int) -> str:
        """调用 R 脚本生成诊断图，返回图片路径"""
        import subprocess
        r_script = os.path.join(os.path.dirname(__file__), 'plot_gof_vpc.R')
        out_png = os.path.join(model_dir, f'gof_vpc_{compartment}.png')
        cmd = ['Rscript', r_script, model_dir, str(compartment), out_png]
        subprocess.run(cmd, check=False)
        return out_png if os.path.exists(out_png) else None

    # ---------- 核心步进方法 ----------
    def step(self) -> Dict:
        """执行一步完整的迭代（生成→执行→解析→评估）"""
        if self.optimizer is None:
            self.initialize_optimizer()

        # 1. 生成代码（Phase 1 或后续）
        if self.iteration == 0:
            self.tool_generate_code(1)
        else:
            # 根据当前 Phase 生成改进代码
            self.tool_generate_code(self.phase)

        # 2. 执行 NONMEM
        result = self.tool_run_nonmem(self.current_code)
        if not result['success'] or result['lst_path'] is None:
            # 执行失败，记录并等待用户确认是否重试
            self.waiting_for_user = True
            if self.on_need_confirm:
                self.on_need_confirm(f"迭代 {self.iteration} 执行失败，是否重试？")
            return {'status': 'error', 'iteration': self.iteration}

        # 3. 解析输出
        parsed = self.tool_parse_output(result['lst_path'])
        ofv = parsed.get('objective_function')
        minimization_ok = parsed.get('minimization_successful', False)

        # 4. 评估并决定是否继续
        # 这里简化，实际可调用 _evaluate_improvement
        # 记录历史
        self.history.append({
            'iteration': self.iteration,
            'ofv': ofv,
            'minimization_ok': minimization_ok,
            'parsed': parsed
        })

        # 更新最佳模型
        if ofv is not None and (self.best_ofv is None or ofv < self.best_ofv):
            self.best_ofv = ofv
            self.best_code = self.current_code

        self.iteration += 1

        # 5. 检查是否达到停止条件
        if self.iteration >= self.max_iter or (self.iteration >= self.min_iter and minimization_ok):
            return {'status': 'done', 'iteration': self.iteration, 'best_ofv': self.best_ofv}

        # 6. 检查是否需要切换 Phase
        # 这里调用 _determine_current_phase（可复用 optimizer 的方法）
        new_phase = self.optimizer._determine_current_phase(None)  # 需传入 parser
        if new_phase != self.phase:
            self.phase = new_phase
            if self.on_phase_change:
                self.on_phase_change(self.phase)
            # 暂停等待确认
            self.waiting_for_user = True
            if self.on_need_confirm:
                self.on_need_confirm(f"即将进入 Phase {self.phase}，确认继续？")

        return {'status': 'running', 'iteration': self.iteration}

    def confirm(self, decision: bool = True):
        """用户确认后继续"""
        self.waiting_for_user = False
        self.user_decision = decision

    def _register_skills(self):
        """将技能函数绑定到技能名。"""
        self.skills = {
            "init": self._skill_init,
            "run_nonmem": self._skill_run_nonmem,
            "parse_output": self._skill_parse_output,
            "phase1_fix": self._skill_phase1_fix,
            "phase2_diagnose": self._skill_phase2_diagnose,
            "phase3_reduce": self._skill_phase3_reduce,
            "phase4_optimize": self._skill_phase4_optimize,
            "phase5_covariate": self._skill_phase5_covariate,
            "summarize": self._skill_summarize,
        }

    # ---------- 技能实现 ----------
    def _skill_init(self, params: Dict = None) -> str:
        """加载数据并生成初始代码，使用优化器的公开方法。"""
        if self.optimizer is None:
            self._initialize_optimizer()
        # 调用公开方法（我们将在optimizer.py中将其改为公开）
        self.optimizer.generate_initial_code()  # 原 _generate_initial_code
        self.context["current_code"] = self.optimizer.current_code
        self.context["iteration"] = 0
        msg = "已加载数据并生成初始NONMEM控制流，请运行 'run_nonmem' 执行模型。"
        return msg

    def _skill_run_nonmem(self, params: Dict = None) -> str:
        """执行NONMEM。"""
        if self.optimizer.current_code is None:
            return "没有可执行的代码，请先调用 init。"
        self.context["iteration"] += 1
        success = self.optimizer.run_nonmem()  # 原 _run_nonmem
        lst_path = f"{self.optimizer.output_base}_iter{self.context['iteration']}.lst"
        if success and os.path.exists(lst_path):
            self.context["last_lst"] = lst_path
            msg = f"NONMEM 执行完成，输出文件：{lst_path}。请使用 parse_output 解析结果。"
        else:
            msg = "NONMEM 执行失败，请检查语法或边界，建议调用 phase1_fix。"
        return msg

    def _skill_parse_output(self, params: Dict = None) -> str:
        """解析最近的.lst文件。"""
        if self.context["last_lst"] is None:
            return "没有可用的.lst文件，请先运行 run_nonmem。"
        parser = NONMEMParser(self.context["last_lst"],
                              gemini_client=self.optimizer.ds_client.clients.get(self.model))
        parsed = parser.get_parsed_data()
        self.context["last_parsed"] = parsed
        ofv = parsed.get("objective_function")
        success = parsed.get("minimization_successful")
        msg = f"解析完成：OFV={ofv if ofv is not None else 'N/A'}, 最小化{'成功' if success else '失败'}。"
        if parsed.get("warnings"):
            msg += f" 警告：{', '.join(parsed['warnings'][:2])}"
        return msg

    def _skill_phase1_fix(self, params: Dict = None) -> str:
        """调用优化器的Phase 1改进方法。"""
        # 需要传入一个模拟的parser或None，因为_generate_improved_code需要parser参数
        # 我们将优化器中的_generate_improved_code公开为generate_improved_code，并允许传入None
        self.optimizer.generate_improved_code(None)  # 内部会根据当前阶段判断
        self.context["current_code"] = self.optimizer.current_code
        return "Phase 1 修复完成，代码已更新。建议运行 run_nonmem 重新测试。"

    def _skill_phase2_diagnose(self, params: Dict = None) -> str:
        """类似Phase 2，但需要先有解析结果。"""
        if self.context["last_parsed"] is None:
            return "请先解析输出（parse_output）。"
        # 设置当前阶段为DIAGNOSE_STRUCTURE
        self.optimizer.current_phase = self.optimizer.ModelPhase.DIAGNOSE_STRUCTURE
        self.optimizer.generate_improved_code(None)
        self.context["current_code"] = self.optimizer.current_code
        return "Phase 2 结构诊断完成，代码已更新。"

    def _skill_phase3_reduce(self, params: Dict = None) -> str:
        if self.context["last_parsed"] is None:
            return "请先解析输出。"
        self.optimizer.current_phase = self.optimizer.ModelPhase.REDUCE_OVERFITTING
        self.optimizer.generate_improved_code(None)
        self.context["current_code"] = self.optimizer.current_code
        return "Phase 3 过拟合简化完成，代码已更新。"

    def _skill_phase4_optimize(self, params: Dict = None) -> str:
        if self.context["last_parsed"] is None:
            return "请先解析输出。"
        self.optimizer.current_phase = self.optimizer.ModelPhase.OPTIMIZE_IIV
        self.optimizer.generate_improved_code(None)
        self.context["current_code"] = self.optimizer.current_code
        return "Phase 4 随机效应优化完成，代码已更新。"

    def _skill_phase5_covariate(self, params: Dict = None) -> str:
        if self.context["last_parsed"] is None:
            return "请先解析输出。"
        # 需要从上下文中获取下一个协变量指令
        # 此处简化，直接调用优化器的协变量步骤
        self.optimizer.current_phase = self.optimizer.ModelPhase.COVARIATE_ANALYSIS
        # 需要设置current_covariate_instruction
        if self.context.get("scm_mode") == "backward":
            # 选择下一个待移除的协变量
            pass
        else:
            # 选择下一个待添加的协变量
            pass
        self.optimizer.generate_improved_code(None)
        self.context["current_code"] = self.optimizer.current_code
        return "Phase 5 协变量测试完成，代码已更新。"

    def _skill_summarize(self, params: Dict = None) -> str:
        """生成最终报告和图片。"""
        # 调用优化器的_final_summary，并调用R绘图
        summary = self.optimizer.generate_final_summary()
        # 绘图部分略
        msg = f"建模完成！最佳OFV: {summary.get('best_ofv')}，最终文件: {summary.get('final_file')}"
        return msg

    # ---------- 对话调度 ----------
    def process_user_message(self, user_input: str) -> str:
        """核心入口：接收用户消息，决定调用哪个技能，执行并返回回复。"""
        # 1. 构建包含上下文和技能描述的提示
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(user_input)
        full_prompt = system_prompt + "\n\n" + user_prompt

        # 2. 调用LLM生成动作
        response = self.optimizer.ds_client.generate(full_prompt, model_type=self.model)

        # 3. 解析动作（期望JSON）
        try:
            # 提取JSON（可能被```包裹）
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                action_json = json_match.group(1)
            else:
                # 尝试直接解析
                action_json = response.strip()
            action = json.loads(action_json)
        except Exception as e:
            # 如果LLM未返回有效JSON，尝试用启发式匹配
            return f"无法解析指令，请重新描述。错误：{e}。原始响应：{response[:200]}"

        skill_name = action.get("skill")
        params = action.get("params", {})
        if skill_name not in self.skills:
            return f"未知技能 '{skill_name}'，可用技能：{', '.join(self.skills.keys())}"

        # 执行技能
        msg = self.skills[skill_name](params)

        # 记录对话
        self.conversation_history.append((user_input, msg))
        return msg

    def _build_system_prompt(self) -> str:
        """构建系统提示，包含可用技能列表和上下文信息。"""
        desc = get_all_skill_descriptions()
        context_info = f"当前迭代次数: {self.context['iteration']}\n"
        if self.context.get("best_ofv"):
            context_info += f"最佳OFV: {self.context['best_ofv']:.2f}\n"
        if self.context.get("last_parsed"):
            parsed = self.context["last_parsed"]
            ofv = parsed.get("objective_function")
            success = parsed.get("minimization_successful")
            context_info += f"最新OFV: {ofv if ofv else 'N/A'}, 最小化{'成功' if success else '失败'}\n"
        prompt = f"""你是一个药代动力学建模助手（PKGPT），可以调用以下技能来完成建模任务：
{desc}

当前状态：
{context_info}

用户消息可能是自然语言，你需要根据用户意图和当前状态，选择一个最合适的技能，并以JSON格式返回，例如：
{{"skill": "run_nonmem", "params": {{}}}}
或
{{"skill": "phase1_fix", "params": {{}}}}

只返回JSON，不要其他解释。"""
        return prompt

    def _build_user_prompt(self, user_input: str) -> str:
        return f"用户请求：{user_input}\n请决定调用哪个技能。"