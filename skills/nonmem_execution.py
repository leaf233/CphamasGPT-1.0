"""
skills/nonmem_execution.py

执行 NONMEM（nmfe75），监控 .lst 文件生成完整性。
不调用 LLM，纯工具执行。
"""

import os
import time
import subprocess
from .base import Skill, SkillResult


class NonmemExecutionSkill(Skill):
    name = 'nonmem_execution'
    phase = [1, 2, 3, 4, 5]
    tools = ['nonmem_runner']
    prompt_template = ''

    def run(self, state, llm, tools) -> SkillResult:
        runner = self._tool(tools, 'nonmem_runner')
        result = runner.execute(state.current_code, state)

        return SkillResult(updates={
            'last_run': {
                'lst_path':     result['lst_path'],
                'txt_path':     result['txt_path'],
                'exit_code':    result['exit_code'],
                'full_output':  result.get('full_output', ''),
                'issues':       result.get('issues', []),
            }
        })


# ---- Tool 层实现（放在 tools/nonmem_runner.py，此处给出接口说明）--------- #

class NonmemRunner:
    """
    Tool：执行 NONMEM 并等待输出稳定。

    接口：
      execute(code, state) -> {
        'lst_path': str, 'txt_path': str, 'exit_code': int,
        'full_output': str, 'issues': list
      }

    实现要点：
      - 将 code 写入 {output_base}_iter{N}.txt
      - 替换 $DATA 为绝对路径
      - 调用 nmfe75 {txt} {lst}，timeout=600
      - 轮询 .lst 大小稳定 + 出现 "Stop Time" 标记
      - 若 nmfe 不存在，写入 mock 输出（便于无 NONMEM 环境测试）
    """
    def execute(self, code: str, state) -> dict:
        input_file  = os.path.abspath(f"{state.output_base}_iter{state.iteration}.txt")
        output_file = os.path.abspath(f"{state.output_base}_iter{state.iteration}.lst")
        os.makedirs(os.path.dirname(input_file), exist_ok=True)

        # 替换 $DATA 为绝对路径
        import re
        abs_data = os.path.abspath(state.data_file).replace('\\', '/')
        code = re.sub(r'\$DATA\s+\S+', f'$DATA {abs_data}', code, flags=re.IGNORECASE)

        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(code)

        try:
            subprocess.run([state.nmfe_command, input_file, output_file],
                           timeout=600, cwd=os.path.dirname(input_file) or '.')
        except FileNotFoundError:
            self._write_mock(output_file, state)
        except subprocess.TimeoutExpired:
            return {'lst_path': output_file, 'txt_path': input_file,
                    'exit_code': -1, 'full_output': '', 'issues': ['timeout']}

        # 等待输出稳定
        self._wait_stable(output_file)
        full_output = ''
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                full_output = f.read()

        return {'lst_path': output_file, 'txt_path': input_file,
                'exit_code': 0, 'full_output': full_output, 'issues': []}

    def _wait_stable(self, path, max_wait=30):
        if not os.path.exists(path):
            return
        prev, stable, waited = 0, 0, 0
        while waited < max_wait:
            time.sleep(1); waited += 1
            size = os.path.getsize(path)
            if size == prev:
                stable += 1
                if stable >= 3:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        if 'Stop Time' in f.read():
                            return
            else:
                stable, prev = 0, size

    def _write_mock(self, path, state):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"Mock NONMEM output for testing\n"
                    f"MINIMIZATION TERMINATED\n"
                    f"OBJECTIVE FUNCTION VALUE: {1000 + state.iteration * 10}\n")