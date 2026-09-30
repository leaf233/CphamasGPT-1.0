# gradio_app.py (新增文件)
import gradio as gr
import threading
import time
import os
from agent import PKGPTAgent
import tempfile
import shutil

# 全局状态
agent_state = gr.State(None)
run_thread = None
stop_flag = False

# 全局agent实例（简化，实际可用session）
current_agent = None

'''
# 旧有agent网页
def set_callbacks(agent, log_box, progress_bar, confirm_btn):
    """为 Agent 注册回调函数"""
    def on_log(msg):
        log_box.value += msg + "\n"
        return log_box

    def on_phase_change(phase):
        log_box.value += f"\n--- Phase {phase} 开始 ---\n"
        return log_box

    def on_need_confirm(msg):
        log_box.value += f"\n⚠️ {msg}\n请点击「继续」按钮\n"
        confirm_btn.visible = True
        return log_box, confirm_btn

    agent.on_log = on_log
    agent.on_phase_change = on_phase_change
    agent.on_need_confirm = on_need_confirm

def run_agent(agent, log_box, progress_bar, confirm_btn, results_df, image_gallery):
    """后台运行 Agent，直到完成或用户停止"""
    global stop_flag
    stop_flag = False
    while not stop_flag:
        if agent.waiting_for_user:
            # 等待用户确认
            while agent.waiting_for_user and not stop_flag:
                time.sleep(0.5)
            if stop_flag:
                break
            # 用户已确认
            agent.confirm(True)
        # 执行一步
        status = agent.step()
        # 更新日志和进度
        log_box.value += f"Iteration {status['iteration']} 完成\n"
        progress_bar.value = min(100, (status['iteration'] / agent.max_iter) * 100)
        # 如果完成，则停止
        if status.get('status') == 'done':
            log_box.value += f"优化完成！最佳 OFV: {status['best_ofv']}\n"
            # 生成结果对比和图片
            results = compare_models(agent.output_dir)  # 需要实现
            results_df.value = results['table']
            image_gallery.value = results['images']
            break
        # 检查停止标志
        if stop_flag:
            break

def start_run(data_file, output_dir, min_iter, max_iter, model_choice, api_key,
              log_box, progress_bar, confirm_btn, results_df, image_gallery):
    """点击「开始运行」触发的函数"""
    # 创建 Agent
    agent = PKGPTAgent(
        data_file=data_file.name,
        output_dir=output_dir,
        min_iter=int(min_iter),
        max_iter=int(max_iter),
        model=model_choice,
        api_key=api_key
    )
    # 保存到全局状态
    agent_state.value = agent
    # 注册回调
    set_callbacks(agent, log_box, progress_bar, confirm_btn)
    # 启动后台线程
    global run_thread
    run_thread = threading.Thread(
        target=run_agent,
        args=(agent, log_box, progress_bar, confirm_btn, results_df, image_gallery)
    )
    run_thread.start()
    # 返回更新后的组件（日志、进度、确认按钮可见性）
    return log_box, progress_bar, gr.update(visible=True)

def confirm_click(confirm_btn, log_box):
    """点击「继续」按钮"""
    if agent_state.value is not None:
        agent_state.value.waiting_for_user = False
    confirm_btn.visible = False
    return confirm_btn, log_box

def stop_run():
    global stop_flag
    stop_flag = True
    return "运行已停止"

# 定义 Gradio 界面
def create_interface():
    with gr.Blocks(title="PKGPT - 交互式建模平台") as demo:
        gr.Markdown("# PKGPT - 药代动力学建模助手")
        with gr.Row():
            with gr.Column(scale=1):
                data_file = gr.File(label="上传 CSV 数据文件", file_types=[".csv"])
                output_dir = gr.Textbox(label="输出目录", value="./results")
                min_iter = gr.Number(label="最小迭代次数", value=1, precision=0)
                max_iter = gr.Number(label="最大迭代次数", value=1, precision=0)
                model_choice = gr.Dropdown(
                    choices=['flash', 'flash-lite', 'pro', 'claude-sonnet', 'gpt-4.1'],
                    value='flash', label="LLM 模型"
                )
                api_key = gr.Textbox(label="DeepSeek API Key", type="password")
                with gr.Row():
                    start_btn = gr.Button("开始运行", variant="primary")
                    stop_btn = gr.Button("停止", variant="stop")
                confirm_btn = gr.Button("继续", visible=False, variant="secondary")
            with gr.Column(scale=2):
                log_output = gr.Textbox(label="运行日志", lines=20, interactive=False)
                progress_bar = gr.Slider(0, 100, value=0, label="进度 (%)", interactive=False)
        with gr.Row():
            results_table = gr.DataFrame(label="房室对比结果")
        with gr.Row():
            image_gallery = gr.Gallery(label="GOF/VPC 图", columns=3)

        # 事件绑定
        start_btn.click(
            fn=start_run,
            inputs=[data_file, output_dir, min_iter, max_iter, model_choice, api_key,
                    log_output, progress_bar, confirm_btn, results_table, image_gallery],
            outputs=[log_output, progress_bar, confirm_btn]
        )
        confirm_btn.click(
            fn=confirm_click,
            inputs=[confirm_btn, log_output],
            outputs=[confirm_btn, log_output]
        )
        stop_btn.click(
            fn=stop_run,
            outputs=[]
        )
    return demo

# 在 gradio_app.py 中新增
def compare_models(output_dir):
    """汇总三个房室结果，生成表格和图片列表"""
    import pandas as pd
    import glob
    data = []
    images = []
    for comp in [1, 2, 3]:
        model_dir = os.path.join(output_dir, f'{comp}cmt')
        # 读取最佳 OFV（可从 history 或最终文件获取）
        # 简化：假设存在 result_final.txt，从中提取 OFV
        final_file = os.path.join(model_dir, 'result_final.txt')
        ofv = None
        if os.path.exists(final_file):
            with open(final_file, 'r') as f:
                content = f.read()
                # 简单查找 OFV
                import re
                match = re.search(r'OFV\s*[:=]\s*([\d.]+)', content)
                if match:
                    ofv = float(match.group(1))
        data.append({'房室': comp, '最佳 OFV': ofv})
        # 查找图片
        pngs = glob.glob(os.path.join(model_dir, 'gof_vpc_*.png'))
        images.extend(pngs)
    df = pd.DataFrame(data)
    return {'table': df, 'images': images}
'''

'''
def upload_file(file):
    """上传CSV文件并初始化Agent。"""
    global current_agent
    if file is None:
        return "请上传一个CSV文件。"

    # Gradio 已把文件保存在临时目录，file.name 即真实路径，直接使用
    file_path = file.name

    # 检查文件是否存在
    if not os.path.exists(file_path):
        return f"文件不存在：{file_path}"

    # 检查 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "请设置 DEEPSEEK_API_KEY 环境变量（可在 .env 文件中配置）。"

    # 输出目录：可以让用户在前端指定，或使用默认值
    output_dir = "./results"          # 如需用户自定义，可添加 gr.Textbox
    os.makedirs(output_dir, exist_ok=True)

    # 初始化 Agent
    try:
        current_agent = PKGPTAgent(
            data_file=file_path,
            output_dir=output_dir,
            api_key=api_key,
            model="flash"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"初始化 Agent 失败：{e}"

    return f"文件已加载：{os.path.basename(file_path)}，Agent 初始化完成。请输入指令，如“开始建模”或“运行一次”。"

def respond(message, history):
    """处理用户消息并返回回复。"""
    global current_agent
    if current_agent is None:
        return "请先上传一个CSV数据文件。"
    reply = current_agent.process_user_message(message)
    return reply

# 构建界面
with gr.Blocks() as demo:
    gr.Markdown("# PKGPT 对话式建模助手")
    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="上传CSV数据文件", file_types=[".csv"])
            upload_status = gr.Textbox(label="状态", interactive=False)
            upload.upload(upload_file, inputs=upload, outputs=upload_status)
        with gr.Column(scale=3):
            chat = gr.ChatInterface(
                fn=respond,
                title="对话区",
                description="输入指令，如：'开始建模'、'运行NONMEM'、'解析结果'、'进入Phase 1'等。"
            )
'''

# 每个浏览器会话独立的 Agent 实例（多用户安全）
def get_new_agent_state():
    return {"agent": None, "data_file": None, "output_dir": "./results"}

def handle_upload_and_message(user_message, chat_history, state):
    """
    处理 MultimodalTextbox 的输入。
    user_message 格式: {"text": "...", "files": ["/tmp/xxx.csv", ...]}
    """
    text = (user_message.get("text") or "").strip()
    files = user_message.get("files") or []

    # ---- 1. 若上传了新文件，尝试初始化 Agent ----
    file_notice = ""
    if files:
        # Gradio 会把上传文件放到临时目录，路径已在 files 中
        uploaded_path = files[0]
        if not os.path.exists(uploaded_path):
            file_notice = f"⚠️ 文件不存在：{uploaded_path}"
        else:
            # 复制到会话专属目录，避免 Gradio 清理临时文件
            session_dir = tempfile.mkdtemp(prefix="pkgpt_session_")
            local_path = os.path.join(session_dir, os.path.basename(uploaded_path))
            shutil.copy(uploaded_path, local_path)

            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                file_notice = "⚠️ 未检测到 DEEPSEEK_API_KEY，请在 .env 中配置。"
            else:
                output_dir = state.get("output_dir", "./results")
                os.makedirs(output_dir, exist_ok=True)
                try:
                    state["agent"] = PKGPTAgent(
                        data_file=local_path,
                        output_dir=output_dir,
                        api_key=api_key,
                        model="flash",
                    )
                    state["data_file"] = local_path
                    file_notice = (f"✅ 已加载数据文件：**{os.path.basename(local_path)}**\n"
                                   f"输出目录：`{output_dir}`\n"
                                   f"您可以输入“开始建模”，或直接说“并行运行三个房室模型”。")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    file_notice = f"⚠️ Agent 初始化失败：{e}"

    # ---- 2. 调用 Agent 处理文本 ----
    if state.get("agent") is None:
        # 未上传文件
        if text:
            reply = "请先上传一个 CSV 数据文件（点击输入框左侧的回形针图标）。"
        else:
            reply = "请上传 CSV 数据文件以开始建模。"
    else:
        if not text:
            reply = file_notice or "请告诉我下一步要做什么，例如“开始建模”或“运行 NONMEM”。"
        else:
            try:
                reply = state["agent"].process_user_message(text)
            except Exception as e:
                import traceback
                traceback.print_exc()
                reply = f"处理请求时出错：{e}"
            if file_notice:
                reply = file_notice + "\n\n" + reply

    # ---- 3. 更新对话历史 ----
    chat_history = chat_history + [
        {"role": "user", "content": text if text else "（已上传数据文件）"},
        {"role": "assistant", "content": reply},
    ]
    return chat_history, state, ""   # 清空输入框


# ---------------- 界面构建 ----------------
with gr.Blocks(title="PKGPT 建模助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧬 PKGPT · 对话式 PopPK 建模助手
        在下方对话框里**上传 CSV 数据文件**（点击 📎 图标），然后输入指令开始建模。
        例如：`开始建模`、`并行运行三个房室`、`运行 NONMEM`、`生成报告`。
        """
    )

    state = gr.State(get_new_agent_state())

    chatbot = gr.Chatbot(
        label="对话",
        type="messages",
        height=560,
        show_copy_button=True,
        avatar_images=(None, "🧬"),
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="single",
        file_types=[".csv"],
        placeholder="上传 CSV 数据文件，或输入指令…（例：开始建模）",
        show_label=False,
        sources=["upload"],   # 只显示上传按钮，不显示麦克风
    )

    # 事件绑定：用户提交 -> 处理 -> 更新对话
    chat_input.submit(
        fn=handle_upload_and_message,
        inputs=[chat_input, chatbot, state],
        outputs=[chatbot, state, chat_input],
    )






if __name__ == "__main__":
    demo.queue().launch(share=False, inbrowser=True)

    # demo.launch()
    # demo = create_interface()
    # demo.launch(share=False)