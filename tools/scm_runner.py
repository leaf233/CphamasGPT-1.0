"""
tools/scm_runner.py

SCM 单候选执行工具。

职责：
1. 接收待测代码、state、round base
2. 调用 nonmem_runner 执行
3. 调用 nonmem_parser 解析
4. 计算 ΔOFV = current_ofv - round_base_ofv
5. 返回结构化记录，供 CovariateSCMSkill 记录到 scm.round_results

不调用 LLM。
"""

from __future__ import annotations
from typing import Optional


class SCMRunner:
    """SCM 候选执行器。"""

    def run_candidate(
        self,
        code: str,
        state,
        tools,
        base_ofv: Optional[float],
    ) -> dict:
        """
        执行单个 SCM 候选并返回结果记录。

        参数：
          code     : 候选模型的控制流
          state    : OptimizationState
          tools    : ToolRegistry
          base_ofv : 本轮 base OFV（用于计算 ΔOFV）

        返回：
          {
            'ofv':               float | None,
            'delta_ofv':         float | None,
            'cov_ok':            bool,
            'minimization_ok':   bool,
            'omega_values':      list,
            'avg_eta_shrinkage': float | None,
            'lst_path':          str,
            'code':              str,
            'parsed':            dict,
          }
        """
        runner = tools['nonmem_runner']
        parser = tools['nonmem_parser']

        run_result = runner.execute(code, state)
        parsed = parser.parse(run_result['lst_path'])

        ofv = parsed.get('objective_function')
        cov_ok = parsed.get('covariance_step', {}).get('successful', False)
        min_ok = parsed.get('minimization_successful', False)

        omega_values = [
            float(o.get('value', 1.0))
            for o in (parsed.get('parameter_estimates', {}) or {}).get('omega', [])
            if 'value' in o
        ]
        shrink_list = parsed.get('eta_shrinkage', [])
        max_shrink = max((s['shrinkage'] for s in shrink_list), default=None)

        delta_ofv = None
        if ofv is not None and base_ofv is not None:
            delta_ofv = ofv - base_ofv

        return {
            'ofv':               ofv,
            'delta_ofv':         delta_ofv,
            'cov_ok':            cov_ok,
            'minimization_ok':   min_ok,
            'omega_values':      omega_values,
            'avg_eta_shrinkage': max_shrink,
            'lst_path':          run_result['lst_path'],
            'code':              code,
            'parsed':            parsed,
        }

    # ---- winner 选择 ----------------------------------------------------- #

    @staticmethod
    def select_forward_winner(
        round_results: list,
        threshold: float = -3.84,
        numeric_safe_fn=None,
    ) -> Optional[dict]:
        """
        从一轮 forward 结果中选择 winner。

        规则：
          1. ΔOFV < threshold (-3.84, p<0.05, df=1)
          2. cov_ok = True
          3. numeric_safe_fn(ofv, omega_values, shrinkage) = True
          4. 取 ΔOFV 最负者
        """
        significant = [
            r for r in round_results
            if r.get('delta_ofv') is not None
            and r['delta_ofv'] < threshold
            and r.get('cov_ok', False)
        ]
        if numeric_safe_fn:
            significant = [
                r for r in significant
                if numeric_safe_fn(
                    r.get('ofv'),
                    r.get('omega_values'),
                    r.get('avg_eta_shrinkage'),
                )[0]
            ]
        if not significant:
            return None
        return min(significant, key=lambda r: r['delta_ofv'])

    @staticmethod
    def select_backward_loser(
        round_results: list,
        threshold: float = 6.63,
        numeric_safe_fn=None,
    ) -> Optional[dict]:
        """
        从一轮 backward 结果中选择 loser（可剔除者）。

        规则：
          1. ΔOFV（移除成本）< threshold (6.63, p<0.01, df=1)
          2. cov_ok = True
          3. numeric_safe_fn = True
          4. 取 ΔOFV 最小者（移除后最不坏）
        """
        removable = [
            r for r in round_results
            if r.get('delta_ofv') is not None
            and r['delta_ofv'] < threshold
            and r.get('cov_ok', False)
        ]
        if numeric_safe_fn:
            removable = [
                r for r in removable
                if numeric_safe_fn(
                    r.get('ofv'),
                    r.get('omega_values'),
                    r.get('avg_eta_shrinkage'),
                )[0]
            ]
        if not removable:
            return None
        return min(removable, key=lambda r: r['delta_ofv'])