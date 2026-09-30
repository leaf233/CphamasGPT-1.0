"""
guardrails/plausibility_guard.py

生理合理性守卫。

职责：
1. 从 code 解析 THETA 索引 → 参数名映射（CL / V1 / V2 / Q / Ka）
2. 从 parsed_data 提取 THETA 估计值
3. 与 state.plausibility_bounds 的 min/max 对比
4. 输出 violations（MILD / MODERATE / SEVERE）与 plausibility_score

封装的原始逻辑：NONMEMOptimizer._check_plausibility

不调用 LLM。
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional


class PlausibilityGuard:
    """生理合理性守卫。"""

    # 严重度 → 扣分
    _SEVERITY_PENALTY = {
        'SEVERE':   30,
        'MODERATE': 15,
        'MILD':      5,
    }

    # THETA 索引 → 参数名映射模式
    _THETA_PATTERNS = [
        (r'TVCL\s*=\s*THETA\((\d+)\)',   'CL'),
        (r'TVV1\s*=\s*THETA\((\d+)\)',   'V1'),
        (r'TVV(?!2)\s*=\s*THETA\((\d+)\)', 'V1'),
        (r'TVQ\s*=\s*THETA\((\d+)\)',    'Q'),
        (r'TVV2\s*=\s*THETA\((\d+)\)',   'V2'),
        (r'TVK(?:A|a)\s*=\s*THETA\((\d+)\)', 'Ka'),
        (r'^\s*CL\s*=\s*THETA\((\d+)\)', 'CL'),
        (r'^\s*V1\s*=\s*THETA\((\d+)\)', 'V1'),
        (r'^\s*V\s*=\s*THETA\((\d+)\)',  'V1'),
        (r'^\s*Q\s*=\s*THETA\((\d+)\)',  'Q'),
        (r'^\s*V2\s*=\s*THETA\((\d+)\)', 'V2'),
    ]

    def check(
        self,
        parsed_data: dict,
        code: str,
        bounds: Optional[dict],
    ) -> dict:
        """
        检查参数估计值与生理范围的偏离。

        参数：
          parsed_data : NONMEMParser.get_parsed_data() 结果
          code        : 当前控制流（用于提取 THETA 映射）
          bounds      : state.plausibility_bounds

        返回：
          {
            'violations':          list[str],
            'plausibility_score':  int (0-100),
            'checked_parameters':  list[str],
            'drug':                str,
            'notes':               str,
          }
          若 bounds 为空，返回 {}
        """
        if not bounds:
            return {}

        bounds_params = bounds.get('parameters', {})
        if not bounds_params:
            return {}

        params = parsed_data.get('parameter_estimates', {})
        theta_list = params.get('theta', [])
        if not theta_list:
            return {}

        theta_map = self._extract_theta_map(code)
        violations: List[str] = []
        checked: List[str] = []
        score = 100

        for entry in theta_list:
            idx = entry.get('index')
            estimate = entry.get('value') or entry.get('estimate')
            if idx is None or estimate is None:
                continue

            param_name = theta_map.get(idx)
            if not param_name or param_name not in bounds_params:
                continue

            bound = bounds_params[param_name]
            min_val = bound.get('min')
            max_val = bound.get('max')
            unit = bound.get('unit', '')
            rationale = bound.get('rationale', '')

            checked.append(f"{param_name}={estimate:.3g} {unit}")

            violation = None

            # 超上限
            if max_val is not None and estimate > max_val:
                fold = estimate / max_val
                severity = self._severity_from_fold(fold)
                score -= self._SEVERITY_PENALTY[severity]
                violation = (
                    f"{severity}: {param_name} = {estimate:.2g} {unit} "
                    f"超出上限 ({max_val} {unit}, {fold:.1f}× 超). {rationale}"
                )
            # 低于下限
            elif min_val is not None and estimate < min_val:
                fold = min_val / max(estimate, 1e-9)
                severity = self._severity_from_fold(fold)
                score -= self._SEVERITY_PENALTY[severity]
                violation = (
                    f"{severity}: {param_name} = {estimate:.2g} {unit} "
                    f"低于下限 ({min_val} {unit}, {fold:.1f}× 低). {rationale}"
                )

            if violation:
                violations.append(violation)

        return {
            'violations':         violations,
            'plausibility_score': max(0, score),
            'checked_parameters': checked,
            'drug':               bounds.get('drug_identified', 'unknown'),
            'notes':              bounds.get('notes', ''),
        }

    # ---- 内部方法 --------------------------------------------------------- #

    def _extract_theta_map(self, code: str) -> Dict[int, str]:
        """从 $PK 中提取 THETA 索引 → 参数名映射。"""
        param_map: Dict[int, str] = {}
        for pattern, param_name in self._THETA_PATTERNS:
            match = re.search(pattern, code, re.IGNORECASE | re.MULTILINE)
            if match:
                idx = int(match.group(1))
                if idx not in param_map:
                    param_map[idx] = param_name
        return param_map

    def _severity_from_fold(self, fold: float) -> str:
        """依据偏离倍数确定严重度。"""
        if fold > 10:
            return 'SEVERE'
        if fold > 3:
            return 'MODERATE'
        return 'MILD'

    # ---- 便捷判定 --------------------------------------------------------- #

    @staticmethod
    def has_severe(report: dict) -> bool:
        """判断 report 是否包含 SEVERE 违规。"""
        if not report:
            return False
        return any('SEVERE' in v for v in report.get('violations', []))

    @staticmethod
    def violation_summary(report: dict) -> str:
        """生成 violations 摘要文本。"""
        if not report or not report.get('violations'):
            return "无生理合理性违规"
        lines = [f"生理合理性分数: {report.get('plausibility_score', 'N/A')}/100"]
        for v in report['violations']:
            lines.append(f"  - {v}")
        return "\n".join(lines)