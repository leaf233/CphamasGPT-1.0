"""
tools/composite_scorer.py

复合质量评分（lower is better）。

评分维度（权重参考论文 + README）：
  Convergence (30%)  — 最小化成功 + 协方差成功
  Shrinkage   (25%)  — ETA shrinkage 阈值
  Stability   (20%)  — OFV 合理性 + OMEGA 塌缩
  Precision   (15%)  — max RSE
  Utility     (10%)  — 模型复杂度与数据量匹配

不调用 LLM。
"""

from __future__ import annotations
from typing import List, Optional
from utils.safe_numeric import extract_matrix_diagonal_values



class CompositeScorer:
    """复合质量评分器。"""

    # 惩罚项常量
    PENALTY_NEG_OFV_SEVERE = 50000.0
    PENALTY_NEG_OFV_MILD   = 10000.0
    PENALTY_SHRINK_95      = 20000.0
    PENALTY_SHRINK_90      = 10000.0
    PENALTY_SHRINK_70      = 2000.0
    PENALTY_SHRINK_50      = 500.0
    PENALTY_OMEGA_COLLAPSE = 500.0     # 每个塌缩 OMEGA
    PENALTY_COV_FAIL_GOOD  = 100.0
    PENALTY_COV_FAIL_MID   = 200.0
    PENALTY_COV_FAIL_POOR  = 300.0
    PENALTY_MIN_FAIL       = 2000.0
    PENALTY_NO_OFV         = 100000.0

    def compute(self, parsed: dict, state=None) -> float:
        """
        计算复合评分。

        参数：
          parsed : NONMEMParser.get_parsed_data() 结果
          state  : OptimizationState（可选；若提供 phase，Phase 5 中不惩罚 RSE）

        返回：
          复合评分（lower is better）
        """
        ofv           = parsed.get('objective_function')
        min_ok        = parsed.get('minimization_successful', False)
        cov_step      = parsed.get('covariance_step', {})
        cov_success   = cov_step.get('successful', False)
        eta_shrink    = parsed.get('eta_shrinkage', [])
        avg_shrink    = max((s['shrinkage'] for s in eta_shrink), default=None) \
                        if eta_shrink else None
        params        = parsed.get('parameter_estimates', {}) or {}
        # omega_values  = [
        #     float(o.get('value', 1.0)) for o in params.get('omega', [])
        #     if 'value' in o
        # ]
        omega_values = extract_matrix_diagonal_values(
            parsed.get('parameter_estimates', {}).get('omega') or []
        )
        rse_data      = parsed.get('rse_percent', {})
        max_rse       = rse_data.get('max_rse')

        score = 0.0

        # ── OFV ─────────────────────────────────────────────────────────
        if ofv is None:
            score += self.PENALTY_NO_OFV
        elif ofv < -50:
            score += self.PENALTY_NEG_OFV_SEVERE
            print(f"  [WARNING] Negative OFV penalty: +{self.PENALTY_NEG_OFV_SEVERE:.0f}")
        elif ofv < 0:
            score += self.PENALTY_NEG_OFV_MILD
            print(f"  [WARNING] Small negative OFV penalty: +{self.PENALTY_NEG_OFV_MILD:.0f}")
        else:
            score += ofv

        # ── Shrinkage ───────────────────────────────────────────────────
        if avg_shrink is not None:
            if avg_shrink > 95:
                score += self.PENALTY_SHRINK_95
                print(f"  [CRITICAL] Shrinkage >95% penalty: +{self.PENALTY_SHRINK_95:.0f}")
            elif avg_shrink > 90:
                score += self.PENALTY_SHRINK_90
                print(f"  [CRITICAL] Shrinkage >90% penalty: +{self.PENALTY_SHRINK_90:.0f}")
            elif avg_shrink > 70:
                score += self.PENALTY_SHRINK_70
                print(f"  [WARNING] Shrinkage >70% penalty: +{self.PENALTY_SHRINK_70:.0f}")
            elif avg_shrink > 50:
                score += self.PENALTY_SHRINK_50

        # ── OMEGA 塌缩 ──────────────────────────────────────────────────
        if omega_values:
            collapsed = [o for o in omega_values if o < 0.0001]
            if collapsed:
                penalty = len(collapsed) * self.PENALTY_OMEGA_COLLAPSE
                score += penalty
                print(f"  [WARNING] {len(collapsed)} OMEGA(s) collapsed: +{penalty:.0f}")

        # ── 协方差失败 ──────────────────────────────────────────────────
        if not cov_success:
            if avg_shrink is not None and avg_shrink < 40:
                score += self.PENALTY_COV_FAIL_GOOD
                print(f"  [INFO] Cov fail but shrinkage <40%: +{self.PENALTY_COV_FAIL_GOOD:.0f}")
            elif avg_shrink is not None and avg_shrink < 60:
                score += self.PENALTY_COV_FAIL_MID
                print(f"  [WARNING] Cov fail + moderate shrinkage: +{self.PENALTY_COV_FAIL_MID:.0f}")
            else:
                score += self.PENALTY_COV_FAIL_POOR
                print(f"  [WARNING] Cov fail + poor shrinkage: +{self.PENALTY_COV_FAIL_POOR:.0f}")

        # ── RSE（Phase 5 中不惩罚）─────────────────────────────────────
        in_phase5 = (state is not None and getattr(state, 'phase', None) == 5)
        if not in_phase5 and max_rse is not None and cov_success:
            if max_rse > 200:
                penalty = min((max_rse - 200) * 2.0 + 100, 1000)
                score += penalty
                print(f"  [WARNING] Severe RSE penalty: +{penalty:.0f}")
            elif max_rse > 100:
                penalty = (max_rse - 100) * 1.0
                score += penalty
                print(f"  [INFO] High RSE penalty: +{penalty:.0f}")

        # ── 最小化失败 ──────────────────────────────────────────────────
        if not min_ok:
            score += self.PENALTY_MIN_FAIL

        return score

    # ---- 五维评分（0-100）------------------------------------------------- #

    def dimension_scores(self, parsed: dict, state=None) -> dict:
        """
        返回五维分数（0-100），用于 README / 报告展示。

        维度：
          convergence / precision / shrinkage / stability / utility
        """
        min_ok = parsed.get('minimization_successful', False)
        cov_ok = parsed.get('covariance_step', {}).get('successful', False)

        # Convergence
        if min_ok and cov_ok:
            convergence = 100
        elif min_ok:
            convergence = 75
        else:
            convergence = 30

        # Precision
        max_rse = parsed.get('rse_percent', {}).get('max_rse')
        if max_rse is None:
            precision = 50
        elif max_rse < 30:
            precision = 100
        elif max_rse < 50:
            precision = 85
        elif max_rse < 100:
            precision = 70
        else:
            precision = 45

        # Shrinkage
        eta = parsed.get('eta_shrinkage', [])
        shrink = max((s['shrinkage'] for s in eta), default=None)
        if shrink is None:
            shrinkage = 50
        elif shrink < 30:
            shrinkage = 100
        elif shrink < 50:
            shrinkage = 85
        elif shrink < 70:
            shrinkage = 65
        elif shrink < 90:
            shrinkage = 40
        else:
            shrinkage = 20

        # Stability
        ofv = parsed.get('objective_function')
        if ofv is None:
            stability = 30
        elif ofv < -50:
            stability = 20
        elif ofv < 0:
            stability = 60
        else:
            stability = 95

        # Utility
        n_subjects = (state.data_profile.get('n_subjects', 100)
                      if state else 100)
        omega_count = len((parsed.get('parameter_estimates', {}) or {}).get('omega', []))
        if n_subjects < 15:
            utility = 100 if omega_count <= 2 else 60
        elif n_subjects < 30:
            utility = 100 if omega_count <= 3 else 70
        else:
            utility = 100 if omega_count <= 4 else 80

        return {
            'convergence': convergence,
            'precision':   precision,
            'shrinkage':   shrinkage,
            'stability':   stability,
            'utility':     utility,
        }

    def weighted_score(self, dimensions: dict) -> int:
        """将五维分数按权重合成为 0-100 总分。"""
        weights = {
            'convergence': 0.30,
            'precision':   0.15,
            'shrinkage':   0.25,
            'stability':   0.20,
            'utility':     0.10,
        }
        return round(sum(dimensions.get(k, 0) * w for k, w in weights.items()))