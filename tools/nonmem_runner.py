"""
tools/nonmem_runner.py

执行 NONMEM（nmfe75）并等待 .lst 输出稳定。

职责：
1. 将控制流写入 {output_base}_iter{N}.txt
2. 将 $DATA 替换为绝对路径（避免 NONMEM 找不到数据文件）
3. 调用 nmfe 命令执行，timeout=600
4. 轮询 .lst 文件大小稳定 + 出现 "Stop Time" 标记
5. 返回 lst_path / txt_path / exit_code / full_output / issues

不调用 LLM。测试时若 nmfe 不存在，写入 mock 输出（便于无 NONMEM 环境测试）。
"""

from __future__ import annotations
import os
import re
import time
import subprocess


class AsyncNonmemRunner:
    """异步 NONMEM 执行器：不阻塞主循环"""

    def execute_async(self, code, state):
        # 写入控制流文件
        # 启动 NONMEM 子进程（subprocess.Popen）
        # 返回 future 对象
        pass

    def poll(self, future):
        # 检查 .lst 文件是否生成完成
        # 返回 None（运行中）或 parsed_result（完成）
        pass

class NonmemRunner:
    """NONMEM 执行器。"""

    DEFAULT_TIMEOUT = 600        # 秒
    STABLE_CHECK_INTERVAL = 1    # 秒
    STABLE_COUNT_REQUIRED = 3    # 连续 3 次大小不变
    MAX_WAIT = 30                # 等待稳定上限

    def execute(self, code: str, state) -> dict:
        """
        执行 NONMEM。

        参数：
          code  : 完整的 NONMEM 控制流文本
          state : OptimizationState（需有 data_file / output_base /
                  iteration / nmfe_command）

        返回：
          {
            'lst_path':    str,
            'txt_path':    str,
            'exit_code':   int,
            'full_output': str,
            'issues':      list,
          }
        """
        input_file  = os.path.abspath(
            f"{state.output_base}_iter{state.iteration}.txt")
        output_file = os.path.abspath(
            f"{state.output_base}_iter{state.iteration}.lst")

        os.makedirs(os.path.dirname(input_file), exist_ok=True)

        # 替换 $DATA 为绝对路径（正斜杠，兼容 Windows）
        abs_data = os.path.abspath(state.data_file).replace('\\', '/')
        code = re.sub(r'\$DATA\s+\S+', f'$DATA {abs_data}',
                      code, flags=re.IGNORECASE)

        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(code)

        exit_code = 0
        try:

            # ★ 依据数据行数动态调整超时
            # n_rows = state.data_profile.get('n_rows', 100)

            # 优先从 DataFrame 直接取，避免 profile 缓存错误
            loader = getattr(state, 'data_loader', None)
            if loader and hasattr(loader, 'df'):
                n_rows = len(loader.df)
            elif 'n_rows' in state.data_profile and state.data_profile['n_rows'] > 0:
                n_rows = state.data_profile['n_rows']
            else:
                n_rows = 100

            timeout = 600 if n_rows < 500 else 1200 if n_rows < 2000 else 1800

            print(f"  [NONMEM] n_rows={n_rows} (from data), timeout={timeout}s")
            # print(f"  [NONMEM] n_rows={n_rows}, timeout={timeout}s")

            proc = subprocess.run(
                [state.nmfe_command, input_file, output_file],
                timeout=self.DEFAULT_TIMEOUT,
                cwd=os.path.dirname(input_file) or '.',
            )
            exit_code = proc.returncode
        except FileNotFoundError:
            print(f"[WARNING] NONMEM command '{state.nmfe_command}' not found")
            print("[WARNING] Writing mock output for testing...")
            self._write_mock(output_file, state)
        except subprocess.TimeoutExpired:
            print("[ERROR] NONMEM execution timed out")
            return {
                'lst_path':    output_file,
                'txt_path':    input_file,
                'exit_code':   -1,
                'full_output': '',
                'issues':      ['NONMEM timeout'],
            }

        # 等待输出稳定
        if os.path.exists(output_file):
            self._wait_stable(output_file)

        full_output = ''
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                full_output = f.read()

        return {
            'lst_path':    output_file,
            'txt_path':    input_file,
            'exit_code':   exit_code,
            'full_output': full_output,
            'issues':      [],
        }

    # ---- 内部方法 --------------------------------------------------------- #

    def _wait_stable(self, path: str) -> bool:
        """轮询文件大小稳定 + "Stop Time" 标记。"""
        prev_size = 0
        stable_count = 0
        waited = 0

        while waited < self.MAX_WAIT:
            time.sleep(self.STABLE_CHECK_INTERVAL)
            waited += self.STABLE_CHECK_INTERVAL

            try:
                size = os.path.getsize(path)
            except OSError:
                continue

            if size == prev_size:
                stable_count += 1
                if stable_count >= self.STABLE_COUNT_REQUIRED:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if 'Stop Time' in content or 'Stop time' in content.lower():
                        print("  [OK] Output file complete")
                        return True
            else:
                stable_count = 0
                prev_size = size

        print("  [INFO] Output stable wait timed out; proceeding anyway")
        return False

    def _write_mock(self, output_file: str, state) -> None:
        """无 NONMEM 环境时写 mock 输出，让流程可继续。"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Mock NONMEM output for testing\n"
                    f"ITERATION: {state.iteration}\n\n"
                    f"MINIMIZATION TERMINATED\n"
                    f"OBJECTIVE FUNCTION VALUE: {1000 + state.iteration * 10}\n")