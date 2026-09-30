#!/usr/bin/env python
"""
PKGPT 2.0 - Pharmacokinetic NONMEM Optimizer (Agent architecture)

CLI 入口。仅做两件事：
  1. 解析参数并构造 OptimizationState
  2. 分发到单房室运行或并行 1/2/3 房室运行

所有建模逻辑位于：
  agents/orchestrator.py  → 主循环状态机
  skills/                 → 五阶段与横切 Skill
  tools/                  → NONMEM 执行 / 解析 / 编辑 / 评分
  guardrails/             → 结构冻结 / 房室不变性 / 数值安全 / 生理合理性
  state/                  → OptimizationState

默认行为：不传 --compartments → 并行运行 1/2/3 房室
          传 --compartments N → 仅运行该房室
"""

# --------------------------------------------------------------------------- #
# 依赖导入
# --------------------------------------------------------------------------- #


import sys
import os
import argparse
import subprocess
import json
from dotenv import load_dotenv
from agents.orchestrator import StateReducer
# from tools import build_tool_registry
from modules.prior_info import load_prior_info
from state import OptimizationState
from agents import OrchestratorAgent
from tools import ToolRegistry, build_tool_registry
from skills import build_skill_registry
from modules.openrouter_client import MultiModelOpenRouterClient

# --------------------------------------------------------------------------- #
# Windows 控制台 UTF-8 修正与实时刷新
# --------------------------------------------------------------------------- #
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace',
                            line_buffering=True)

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
# 日志 Tee
# --------------------------------------------------------------------------- #

class _Tee:
    """将 stdout/stderr 同时写入控制台与日志文件。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return False


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    """CLI 入口。"""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description='PKGPT 2.0 - Agent-based NONMEM PopPK Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 默认并行运行 1/2/3 房室
  python pkgpt_optimizer.py dataset/theo.csv output_theo

  # 仅运行 2 房室
  python pkgpt_optimizer.py dataset/theo.csv output_theo --compartments 2

  # 指定 NONMEM 命令
  python pkgpt_optimizer.py data.csv output --nmfe nmfe74

  # 指定 LLM 模型
  python pkgpt_optimizer.py data.csv output --model claude-sonnet

环境变量：
  OPENROUTER_API_KEY    OpenRouter API key（必需，或通过 --api-key 指定）
  DEEPSEEK_API_KEY      DeepSeek API key（可选，优先级高于 OPENROUTER）

输出文件：
  <output_base>_iter0.txt       初始模型
  <output_base>_iterN.txt       第 N 轮模型
  <output_base>_iterN.lst       第 N 轮 NONMEM 输出
  <output_base>_final.txt       最优模型
  <output_base>_terminal.txt    完整终端记录
""")

    # ── 位置参数 ──────────────────────────────────────────────────────── #
    parser.add_argument('data_file', help='输入 CSV 数据集（NONMEM 格式）')
    parser.add_argument('output_base', help='输出文件基名（不含扩展名）')

    # ── 迭代控制 ──────────────────────────────────────────────────────── #
    parser.add_argument('--min-iter', type=int, default=3, metavar='N',
                        help='最小迭代次数（默认 3）')
    parser.add_argument('--max-iter', type=int, default=20, metavar='N',
                        help='Phase 1-4 最大迭代次数（默认 20）。'
                             'Phase 5 SCM 由候选测试完成度控制，不受此限制。')

    # ── NONMEM ───────────────────────────────────────────────────────── #
    parser.add_argument('--nmfe', default='nmfe75.bat', metavar='CMD',
                        help='NONMEM 执行命令（默认 nmfe75.bat）')

    # ── 房室控制 ──────────────────────────────────────────────────────── #
    parser.add_argument('--compartments', type=int, choices=[1, 2, 3],
                        default=None, metavar='N',
                        help='仅运行指定的房室模型（1/2/3）。'
                             '若不指定，则并行运行 1/2/3。')

    # ── LLM 模型 ─────────────────────────────────────────────────────── #
    parser.add_argument('--model',
                        choices=['flash', 'flash-lite', 'pro',
                                 'claude-sonnet', 'claude-opus',
                                 'gemini-flash', 'gemini-flash-lite',
                                 'gemini-pro', 'gpt-4.1', 'gpt-5.5'],
                        default='flash',
                        help='LLM 模型 profile（默认 flash）')

    # ── 输出目录 ──────────────────────────────────────────────────────── #
    parser.add_argument('--output-dir', default='./results', metavar='DIR',
                        help='输出根目录（默认 ./results）')

    # ── API Key ──────────────────────────────────────────────────────── #
    parser.add_argument('--api-key', help='API key（覆盖环境变量）')

    # ── 先验信息 ──────────────────────────────────────────────────────── #
    parser.add_argument('--prior-info', metavar='FILE',
                        help='可选 JSON/YAML 先验信息文件')

    parser.add_argument('--version', action='version',
                        version='PKGPT v2.0.0 (Agent)')

    parser.add_argument(
        '--shared-bounds',
        metavar='FILE',
        help='Path to precomputed plausibility_bounds JSON (used by parallel subprocesses)'
    )

    args = parser.parse_args()

    # ── 参数校验 ──────────────────────────────────────────────────────── #
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)

    if args.min_iter < 1:
        print("Error: --min-iter must be >= 1")
        sys.exit(1)

    if args.max_iter < args.min_iter:
        print("Error: --max-iter must be >= --min-iter")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 加载 prior_info ──────────────────────────────────────────────── #
    prior_info = {}
    if args.prior_info:
        try:
            prior_info = load_prior_info(args.prior_info)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    # ── API Key 解析 ──────────────────────────────────────────────────── #
    api_key = (args.api_key
               or os.getenv('DEEPSEEK_API_KEY')
               or os.getenv('OPENROUTER_API_KEY'))
    if not api_key:
        print("Error: API key not set.")
        print("Set OPENROUTER_API_KEY (or DEEPSEEK_API_KEY), "
              "or pass --api-key.")
        sys.exit(1)

    # ── 分发 ──────────────────────────────────────────────────────────── #
    if args.compartments is not None:
        # 单房室模式：直接运行
        return run_single_agent(args, api_key, prior_info, args.compartments)
    else:
        # 默认：并行运行 1/2/3 房室
        return run_parallel_agents(args, api_key, prior_info)


# --------------------------------------------------------------------------- #
# 单房室运行
# --------------------------------------------------------------------------- #

def run_single_agent(args, api_key, prior_info, compartments):
    """
    通过 Agent 流水线运行单个房室模型的优化。

    房室数量由 `compartments` 参数锁定，全程不变。
    Phase 2 不再执行房室升级；不同房室数量由并行运行的独立子进程探索。
    """
    log_path = f"{args.output_base}_terminal.txt"
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    log_file = open(log_path, 'w', encoding='utf-8', errors='replace')
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)
    print(f"[LOG] Full terminal output will be saved to: {log_path}")

    try:
        # ── 1. 构造 State（纯数据）────────────────────────────────── #
        state = OptimizationState(
            data_file=args.data_file,
            output_base=args.output_base,
            api_key=api_key,
            min_iterations=args.min_iter,
            max_iterations=args.max_iter,
            nmfe_command=args.nmfe,
            model=args.model,
            prior_info=prior_info,
            forced_compartments=compartments,
            # 显式声明结构锁定
            structure_locked=True,
            locked_compartments=compartments,
        )

        # ★ 新增：加载共享 bounds（如果提供）
        if getattr(args, 'shared_bounds', None):
            import json
            if os.path.exists(args.shared_bounds):
                with open(args.shared_bounds, 'r', encoding='utf-8') as f:
                    state.plausibility_bounds = json.load(f)
                print(f"[SHARED-BOUNDS] Loaded from: {args.shared_bounds}")
                print(f"  drug: {state.plausibility_bounds.get('drug_identified')}")
                print(f"  typical_dose_mg: "
                      f"{state.plausibility_bounds.get('typical_single_dose_mg')}")
            else:
                print(f"[SHARED-BOUNDS-WARN] File not found: {args.shared_bounds} "
                      f"— will compute fresh")

        # ── 2. 组装 Agent（依赖注入）────────────────────────────────── #
        llm = MultiModelOpenRouterClient(api_key)
        # main_llm = MultiModelOpenRouterClient(api_key)
        # shared_bounds_path = os.path.join(output_dir, '_shared_plausibility_bounds.json')

        state.llm = llm

        tools = build_tool_registry(state, llm)
        state.tools = tools

        skills = build_skill_registry(tools)
        state.skills = skills

        agent = OrchestratorAgent(state, skills=skills, tools=tools)

        # ── 3. 启动 Agent ──────────────────────────────────────────── #
        results = agent.run()

        # ★ CLI 层最终校验
        expected_n = compartments
        actual_file = results.get('final_file')
        if actual_file:
            from utils.compartment_lock import validate_lock
            with open(actual_file, 'r', encoding='utf-8') as f:
                final_code = f.read()
            ok, actual, advan = validate_lock(final_code, expected_n)
            if not ok:
                print(f"\n{'=' * 70}")
                print(f"❌ FATAL: Final model compartment mismatch")
                print(f"   Expected: {expected_n}-compartment")
                print(f"   Got:      ADVAN{advan} = {actual}-compartment")
                print(f"{'=' * 70}")
                return 1  # 返回失败
            print(f"\n[OK] Compartment validation: {expected_n}-cmt "
                  f"(ADVAN{advan}) — matches --compartments")

        # ── 4. 结果报告 ────────────────────────────────────────────── #
        if results.get('final_file'):
            print(f"\n[OK] Optimization completed successfully!")
            print(f"[OK] Best model saved to: {results['final_file']}")
            print(f"[OK] Total iterations: {results.get('total_iterations')}")
            print(f"[OK] Best iteration:   {results.get('best_iteration')}")
            print(f"[OK] Best OFV:         {results.get('best_ofv')}")
            return 0
        else:
            print("\n[WARNING] Optimization completed with issues")
            print("[WARNING] Check iteration files for details")
            return 1

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        return 130

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_file.close()


# --------------------------------------------------------------------------- #
# 并行运行 1/2/3 房室
# --------------------------------------------------------------------------- #

def run_parallel_agents(args, api_key, prior_info):
    """
    并行启动三个独立的单房室 Agent 流水线（1/2/3）。

    每个子进程走完整的 OrchestratorAgent 流程，forced_compartments 为其
    分配的值。房室数量在每个子进程内锁定；Phase 2 只做诊断与稳定性修复，
    不做 ADVAN 升级。
    """
    output_dir = args.output_base
    os.makedirs(output_dir, exist_ok=True)

    data_file = args.data_file
    base_name = os.path.splitext(os.path.basename(data_file))[0]
    output_base = args.output_base

    # ★ 步骤 1：主进程预先计算共享的 plausibility_bounds
    shared_bounds_path = os.path.join(output_dir, '_shared_plausibility_bounds.json')
    if not os.path.exists(shared_bounds_path):
        print(f"\n{'='*70}")
        print("PRE-COMPUTING SHARED PLAUSIBILITY BOUNDS (main process)")
        print(f"{'='*70}")
        _precompute_shared_bounds(
            data_file=data_file,
            output_dir=output_dir,
            api_key=api_key,
            model=args.model,
            prior_info=prior_info,
            save_path=shared_bounds_path,
        )
    else:
        print(f"[SHARED-BOUNDS] Reusing existing: {shared_bounds_path}")

    # ── 构造子进程参数（排除 data_file / output_base / compartments）── #
    extra_args = []
    for key, value in vars(args).items():
        if key in ['data_file', 'output_base', 'compartments']:
            continue
        if value is None or value is False:
            continue
        if isinstance(value, bool) and value is True:
            extra_args.append(f'--{key.replace("_", "-")}')
        else:
            extra_args.extend([f'--{key.replace("_", "-")}', str(value)])

    # ★ 新增：把共享 bounds 路径传给子进程
    extra_args.extend(['--shared-bounds', shared_bounds_path])

    # ── 启动 3 个子进程 ──────────────────────────────────────────────── #
    processes = []
    for comp in [1, 2, 3]:
        subdir = os.path.join(output_dir, f'{comp}cmt')
        os.makedirs(subdir, exist_ok=True)
        output_prefix = os.path.join(subdir, f"{base_name}_{comp}cmt")

        # 子进程通过 CLI 重新进入本文件，带 --compartments N
        # → 子进程走 run_single_agent()，其内部结构锁定为该 N
        cmd = ([sys.executable, __file__, data_file, output_prefix]
               + extra_args
               + ['--compartments', str(comp)])

        print(f"[PARALLEL] Starting compartment {comp} in {subdir}")
        proc = subprocess.Popen(cmd)
        processes.append((comp, proc))

    # ── 等待完成 ─────────────────────────────────────────────────────── #
    for comp, proc in processes:
        proc.wait()
        print(f"[PARALLEL] Compartment {comp} finished "
              f"(exit code {proc.returncode})")

    # ── 结果汇总 ─────────────────────────────────────────────────────── #
    print(f"\n{'='*70}")
    print("PARALLEL OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    for comp in [1, 2, 3]:
        subdir = os.path.join(output_dir, f'{comp}cmt')
        final_file = os.path.join(subdir, f"{base_name}_{comp}cmt_final.txt")
        status = "OK" if os.path.exists(final_file) else "FAILED"
        print(f"  {comp}-compartment: {status}")
        if os.path.exists(final_file):
            print(f"    → {final_file}")
    print(f"{'='*70}")
    print(f"[PARALLEL] Results saved under: {output_dir}")

    return 0


def _precompute_shared_bounds(data_file, output_dir, api_key, model,
                               prior_info, save_path):
    """
    主进程：跑一次 data_introspection + plausibility_bounds，
    保存 JSON 供子进程加载。
    """

    # 构造临时 state（output_base 用 _shared 占位）
    temp_state = OptimizationState(
        data_file=data_file,
        output_base=os.path.join(output_dir, '_shared'),
        api_key=api_key,
        model=model,
        prior_info=prior_info,
    )
    temp_state.llm = MultiModelOpenRouterClient(api_key)

    # 组装 skills & tools
    tools = build_tool_registry(temp_state, temp_state.llm)
    skills = build_skill_registry(tools)

    # 只跑 bootstrap（data_introspection + plausibility_bounds）
    '''
    for skill_name in ('data_introspection', 'plausibility_bounds'):
        print(f"\n[Bootstrap-Main] Running {skill_name}...")
        skill = skills[skill_name] if isinstance(skills, dict) else skills.get(skill_name)
        result = skill.run(temp_state, temp_state.llm, tools)
        temp_state = StateReducer.apply(temp_state, result)
    '''

    # ---- 只跑 bootstrap 两个 Skill ----
    for skill_name in ('data_introspection', 'plausibility_bounds'):
        print(f"\n[Bootstrap-Main] Running {skill_name}...")
        skill = skills.get(skill_name)
        if skill is None:
            print(f"[SHARED-BOUNDS-WARN] Skill '{skill_name}' not registered. "
                  f"Available: {list(skills.keys())}")
            continue
        try:
            result = skill.run(temp_state, temp_state.llm, tools)
            temp_state = StateReducer.apply(temp_state, result)
            if not result.ok:
                print(f"[SHARED-BOUNDS-WARN] {skill_name} failed: {result.error}")
        except Exception as e:
            print(f"[SHARED-BOUNDS-ERROR] {skill_name} crashed: {e}")
            import traceback
            traceback.print_exc()

    # 保存到 JSON
    bounds = temp_state.plausibility_bounds or {}
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(bounds, f, indent=2, ensure_ascii=False)
    print(f"[SHARED-BOUNDS] Saved to: {save_path}")
    print(f"  drug: {bounds.get('drug_identified', 'unknown')}")
    print(f"  typical_dose_mg: {bounds.get('typical_single_dose_mg')}")
    if not bounds:
        print(f"  [WARN] Empty bounds — subprocesses will recompute")

# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    sys.exit(main())