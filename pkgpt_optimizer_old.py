#!/usr/bin/env python
"""
PKGPT - LLM-assisted Pharmacokinetic NONMEM Optimizer
Recursive optimization of NONMEM control stream files
"""

import sys
import os
import argparse

import SkillRegistry
import ToolRegistry
from dotenv import load_dotenv
import multiprocessing as mp
from functools import partial
import subprocess
import multiprocessing

from agents.orchestrator import OrchestratorAgent
from modules.openrouter_client import MultiModelOpenRouterClient
from state.state import OptimizationState

# Windows consoles often default to a legacy code page (e.g. cp949 for Korean
# locales) instead of UTF-8. LLM-generated text frequently contains unicode
# punctuation (en/em dashes, multiplication sign, etc.) that cp949 cannot
# encode, which crashes bare print() calls throughout the optimizer (and
# silently empties out try/except-guarded steps like plausibility bounds
# generation). Force UTF-8 on stdout/stderr regardless of the console's
# active code page so no print() call can fail on encoding.
#
# Separately: when stdout/stderr are redirected to a file or pipe (e.g. any
# background/non-interactive run), Python switches from line-buffered to
# full block buffering, so nothing appears in the output file until the
# internal buffer fills or the process exits. line_buffering=True forces a
# flush after every newline so progress is visible in real time.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

# Add modules directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.optimizer import NONMEMOptimizer
from modules.prior_info import load_prior_info


class _Tee:
    """Duplicates every write to multiple streams (e.g. the real console +
    a log file), so a run's full terminal transcript is saved automatically
    instead of having to be copy-pasted out of the console by hand."""

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


def main():
    """Main CLI entry point"""

    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description='PKGPT - Recursive NONMEM Model Optimizer using OpenRouter models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pkgpt_optimizer.py dataset/theophylline_nonmem.csv output_theo

  # Specify iterations
  python pkgpt_optimizer.py dataset/warfarin_nonmem.csv output_warf --min-iter 5 --max-iter 15

  # Custom NONMEM command
  python pkgpt_optimizer.py data.csv output --nmfe nmfe74

Environment Variables:
  OPENROUTER_API_KEY    OpenRouter API key (required)

The optimizer will:
  1. Analyze your pharmacokinetic dataset
  2. Generate an initial NONMEM control stream
  3. Execute NONMEM and parse results
  4. Recursively improve the model through phase-specific optimization
  5. Save the best model as <output>_final.txt

Output files:
  <output>_iter0.txt       Initial model
  <output>_iter1.txt       First iteration
  <output>_iter1.lst       NONMEM output for iteration 1
  ...
  <output>_final.txt       Best model (copy of best iteration)
        """
    )

    parser.add_argument(
        'data_file',
        help='Path to input CSV dataset (NONMEM format)'
    )

    parser.add_argument(
        'output_base',
        help='Base name for output files (without extension)'
    )

    parser.add_argument(
        '--min-iter',
        type=int,
        default=3,
        metavar='N',
        help='Minimum number of iterations (default: 3)'
    )

    parser.add_argument(
        '--max-iter',
        type=int,
        default=20,
        metavar='N',
        help='Maximum number of iterations for Phase 1-4 (default: 20). Phase 5 (SCM) runs until all covariate candidates are tested regardless of this value.'
    )

    parser.add_argument(
        '--nmfe',
        default='nmfe75.bat',
        metavar='CMD',
        help='NONMEM execution command (default: nmfe75.bat)'
    )

    parser.add_argument(
        '--model',
        choices=['flash', 'flash-lite', 'pro','claude-sonnet', 'claude-opus', 'gemini-flash', 'gemini-flash-lite','gemini-pro', 'gpt-4.1', 'gpt-5.5'],
        default='flash',
        help='Deepseek model profile to use (default: flash)'
    )
    # 入参默认为3个房室模型运行,可通过指定1,2,3来运行单个对应房室模型
    parser.add_argument(
        '--compartments',
        type=int,
        choices=[1, 2, 3],
        default=None,
        metavar='N',
        help='Run a single specified compartment model (1, 2, or 3). If not set, all three compartments will be run in parallel.'
    )
    '''
    # 入参指定单个运行房室
    parser.add_argument(
        '--compartments',
        type=int,
        choices=[1, 2, 3],
        default=None,
        metavar='N',
        help='Force a specific number of compartments (1, 2, or 3). If not set, PKGPT will auto-infer from data.'
    )

    # 入参运行多房室并行模式
    parser.add_argument(
        '--run-all-compartments',
        action='store_true',
        help='Run optimization for 1, 2, and 3 compartments in parallel (ignores --compartments).'
    )
    '''

    # 指定输出目录
    parser.add_argument(
        '--output-dir',
        default='./results',
        metavar='DIR',
        help='Directory to save all output files (default: current directory).'
    )

    parser.add_argument(
        '--api-key',
        help='DeepSeek API key (overrides DEEPSEEK_API_KEY env var)'
    )

    parser.add_argument(
        '--prior-info',
        metavar='FILE',
        help='Optional JSON/YAML prior-information file. With covariates.mode=user_selected, SCM tests only the listed covariates.'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='PKGPT v2.0.0-rc1'
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Validation
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)

    if args.min_iter < 1:
        print("Error: Minimum iterations must be >= 1")
        sys.exit(1)

    if args.max_iter < args.min_iter:
        print("Error: Maximum iterations must be >= minimum iterations")
        sys.exit(1)

    prior_info = {}
    if args.prior_info:
        try:
            prior_info = load_prior_info(args.prior_info)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    # Check for API key
    # api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    api_key = args.api_key or os.getenv('DEEPSEEK_API_KEY')or os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it in your environment or .env file, or use --api-key")
        print("\nExample:")
        print("  set OPENROUTER_API_KEY=your-api-key-here")
        print("  python pkgpt_optimizer.py data.csv output")
        sys.exit(1)

    # 如果指定了 --compartments，则单次运行；否则并行运行三个房室
    if args.compartments is not None:
        # 单次运行模式（原逻辑，但需将 output_base 直接使用）
        return run_single(args, api_key, prior_info)
    else:
        # 并行运行三个房室
        return run_parallel(args, api_key, prior_info)


    '''
    # 指定并行运行3个房室模型,将对应结果输出至指定目录
    if args.parallel:
        if not args.output_dir:
            print("Error: --output-dir is required when using --parallel")
            sys.exit(1)
        # 创建子目录
        for n in [1, 2, 3]:
            subdir = os.path.join(args.output_dir, f"_{n}cmt")
            os.makedirs(subdir, exist_ok=True)

        # 准备子进程命令
        import subprocess
        base_cmd = [sys.executable, __file__, args.data_file]
        common_args = [
            "--nmfe", args.nmfe,
            "--model", args.model,
            "--min-iter", str(args.min_iter),
            "--max-iter", str(args.max_iter),
        ]
        if args.api_key:
            common_args.extend(["--api-key", args.api_key])
        if args.prior_info:
            common_args.extend(["--prior-info", args.prior_info])

        processes = []
        for n in [1, 2, 3]:
            output_base = os.path.join(args.output_dir, f"{n}cmt", "result")
            cmd = base_cmd + [output_base] + common_args + ["--compartments", str(n)]
            print(f"[INFO] Starting {n}-compartment optimization...")
            p = subprocess.Popen(cmd)
            processes.append((n, p))

        # 等待完成并报告
        for n, p in processes:
            p.wait()
            final_file = os.path.join(args.output_dir, f"{n}cmt", "result_final.txt")
            if os.path.exists(final_file):
                print(f"[OK] {n}-compartment model saved to: {final_file}")
            else:
                print(f"[WARNING] {n}-compartment optimization did not produce final model.")

        # 汇总
        print("\n" + "=" * 70)
        print("PARALLEL OPTIMIZATION COMPLETE")
        print("=" * 70)
        for n in [1, 2, 3]:
            final_file = os.path.join(args.output_dir, f"{n}cmt", "result_final.txt")
            status = "OK" if os.path.exists(final_file) else "FAILED"
            print(f"  {n}-compartment: {status} - {final_file if os.path.exists(final_file) else 'not found'}")
        print("=" * 70)
        return 0
    '''

    '''
    # Save the full terminal transcript of this run to a .txt file automatically,
    # so it doesn't have to be copy-pasted out of the console by hand afterward.
    log_path = f"{args.output_base}_terminal.txt"
    log_file = open(log_path, 'w', encoding='utf-8', errors='replace')
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)
    print(f"[LOG] Full terminal output will be saved to: {log_path}")

    try:
        # Create optimizer

        if args.run_all_compartments:
            def run_for_compartments(cmt, output_base, data_file, api_key, min_iter, max_iter, nmfe, model, prior_info,
                                     output_dir):
                # 构造带房室数的输出基名
                base_name = os.path.basename(output_base)
                dir_name = os.path.dirname(output_base) or '.'
                # 如果指定了输出目录，则使用；否则使用原路径
                if output_dir != '.':
                    dir_name = output_dir
                cmt_output_base = os.path.join(dir_name, f"{base_name}_{cmt}cmt")
                # 创建优化器
                optimizer = NONMEMOptimizer(
                    data_file=data_file,
                    output_base=cmt_output_base,
                    api_key=api_key,
                    min_iterations=min_iter,
                    max_iterations=max_iter,
                    nmfe_command=nmfe,
                    model=model,
                    prior_info=prior_info
                )
                results = optimizer.run()
                # 返回结果摘要
                return {
                    'compartments': cmt,
                    'final_file': results.get('final_file'),
                    'best_ofv': results.get('best_ofv'),
                    'best_iteration': results.get('best_iteration')
                }

            # 使用进程池并行执行
            with mp.Pool(processes=3) as pool:
                # 准备参数
                task = partial(
                    run_for_compartments,
                    output_base=args.output_base,
                    data_file=args.data_file,
                    api_key=api_key,
                    min_iter=args.min_iter,
                    max_iter=args.max_iter,
                    nmfe=args.nmfe,
                    model=args.model,
                    prior_info=prior_info,
                    output_dir=args.output_dir
                )
                results_list = pool.map(task, [1, 2, 3])

                # 打印汇总
                print("\n" + "=" * 70)
                print("PARALLEL OPTIMIZATION COMPLETE")
                print("=" * 70)
                for res in results_list:
                    print(f"  {res['compartments']}-compartment model:")
                    print(
                        f"    Best OFV: {res['best_ofv']:.2f}" if res['best_ofv'] is not None else "    Best OFV: N/A")
                    print(f"    Final file: {res['final_file']}")
                print("=" * 70)
                return 0

        else:
            # 原有逻辑
            optimizer = NONMEMOptimizer(
                data_file=args.data_file,
                output_base=args.output_base,
                api_key=api_key,
                min_iterations=args.min_iter,
                max_iterations=args.max_iter,
                nmfe_command=args.nmfe,
                model=args.model,
                prior_info=prior_info,
                forced_compartments=args.compartments
            )
            # Run optimization
            results = optimizer.run()

            # Success
            if results['final_file']:
                print(f"\n[OK] Optimization completed successfully!")
                print(f"[OK] Best model saved to: {results['final_file']}")
                return 0
            else:
                print("\n[WARNING] Optimization completed with issues")
                print("[WARNING] Check iteration files for details")
                return 1

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        return 130

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_file.close()
    '''

# 运行指定数字对应的单个房室
def run_single(args, api_key, prior_info):
    """Run a single compartment model (used when --compartments is specified)."""
    # 日志重定向（使用 args.output_base）
    log_path = f"{args.output_base}_terminal.txt"
    log_file = open(log_path, 'w', encoding='utf-8', errors='replace')
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)
    print(f"[LOG] Full terminal output will be saved to: {log_path}")

    try:
        # 1) 构造 State（纯数据，不含逻辑）
        state = OptimizationState(
            data_file=args.data_file,
            output_base=args.output_base,
            api_key=api_key,
            min_iterations=args.min_iter,
            max_iterations=args.max_iter,
            nmfe_command=args.nmfe,
            model=args.model,
            prior_info=prior_info,
            forced_compartments=args.compartments,
        )

        # 2) 组装 Agent（依赖注入）
        llm = MultiModelOpenRouterClient(api_key)
        tools = ToolRegistry(state, llm)
        skills = SkillRegistry(tools)
        agent = OrchestratorAgent(state, skills, tools)

        # 3) 启动 Agent
        results = agent.run()
        ''' # 9.14 修改为agent架构前
        optimizer = NONMEMOptimizer(
            data_file=args.data_file,
            output_base=args.output_base,
            api_key=api_key,
            min_iterations=args.min_iter,
            max_iterations=args.max_iter,
            nmfe_command=args.nmfe,
            model=args.model,
            prior_info=prior_info,
            forced_compartments=args.compartments  # 需要同步修改 optimizer.py 支持此参数
        )
        results = optimizer.run()
        if results.get('final_file'):
            print(f"\n[OK] Optimization completed successfully!")
            print(f"[OK] Best model saved to: {results['final_file']}")
            return 0
        else:
            print("\n[WARNING] Optimization completed with issues")
            return 1
        '''
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        return 130
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_file.close()


# 默认并行3个房室模型
def run_parallel(args, api_key, prior_info):
    """Run three compartment models (1, 2, 3) in parallel."""
    output_dir = args.output_base
    os.makedirs(output_dir, exist_ok=True)

    data_file = args.data_file
    base_name = os.path.splitext(os.path.basename(data_file))[0]

    output_base = args.output_base

    # 构建额外参数列表（排除 data_file 和 output_base）
    extra_args = []
    # arg_dict = vars(args)
    for key, value in vars(args).items():
        if key in ['data_file', 'output_base', 'compartments']:
            continue
        # 其他参数照常添加
        if value is None or value is False:
            continue
        if isinstance(value, bool) and value is True:
            extra_args.append(f'--{key.replace("_", "-")}')
        else:
            extra_args.extend([f'--{key.replace("_", "-")}', str(value)])

    processes = []
    for comp in [1, 2, 3]:
        subdir = os.path.join(output_dir, f'{comp}cmt')
        os.makedirs(subdir, exist_ok=True)
        output_prefix = os.path.join(subdir, f"{base_name}_{comp}cmt")
        cmd = [sys.executable, __file__, data_file, output_prefix] + extra_args + ['--compartments', str(comp)]
        print(f"[PARALLEL] Starting compartment {comp} in {subdir}")
        proc = subprocess.Popen(cmd)
        processes.append((comp, proc))

    for comp, proc in processes:
        proc.wait()
        print(f"[PARALLEL] Compartment {comp} finished (exit code {proc.returncode})")

    print(f"\n[PARALLEL] All three compartment models completed.")
    print(f"[PARALLEL] Results saved under: {output_dir}")


if __name__ == '__main__':
    sys.exit(main())
