"""
Portfolio weight computation for regime-based allocation.

Implements:
1. PortfolioOptimizer.compute_regime_weights() — per-regime MV optimal weights
2. PortfolioOptimizer.compute_daily_weights()  — soft allocation (probability blend)
3. PortfolioOptimizer.compute_hard_weights()   — hard allocation (dominant regime)
4. PortfolioOptimizer.determine_positions()    — long / short / flat classification

References:
  Guidolin & Timmermann (2007) Eq. 3-4, 7-10
  Kritzman et al. (2012) — hard regime switching
"""

import numpy as np
import polars as pl
from scipy import linalg
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Mean-variance portfolio weights optimised per regime, then blended by HMM
    filtered probabilities to produce soft-allocation daily weights.
    """

    # ─── Per-regime weights ───────────────────────────────────────────────

    def compute_regime_weights(self, model, canonical_idx: list,
                               gamma: float = 5,
                               allow_short: bool = False) -> dict:
        """
        Compute per-regime MV optimal weights (Guidolin & Timmermann 2007 Eq. 3-4).

        Unconstrained closed form: w* = (1/γ) Σ⁻¹ μ
        Then projected onto:
          - Long-only:  w ≥ 0, sum(w) = 1, gross leverage ≤ 1.5
          - Long/short: |w| ≤ 1.5, sum(w) = 1, gross leverage ≤ 1.5

        Parameters
        ----------
        model        : fitted GaussianHMM
        canonical_idx: [crash_raw, correction_raw, bull_raw, moderate_growth_raw]
        gamma        : risk-aversion coefficient
        allow_short  : whether to permit negative weights

        Returns
        -------
        dict  {regime_name: np.ndarray shape (3,)}  — weights for [SPY, IWM, TLT]
        """
        regime_names = ["crash", "correction", "bull", "moderate_growth"]
        weights_dict = {}

        for regime_idx, regime_name in enumerate(regime_names):
            raw_idx = canonical_idx[regime_idx]
            mu      = model.means_[raw_idx]          # (3,)
            Sigma   = model.covars_[raw_idx]         # (3, 3)

            # Ridge regularisation prevents singular-matrix errors
            try:
                Sigma_inv = linalg.inv(Sigma + 1e-6 * np.eye(3))
            except linalg.LinAlgError as e:
                raise ValueError(
                    f"Singular covariance for regime '{regime_name}'."
                ) from e

            w_unconstrained = (1.0 / gamma) * (Sigma_inv @ mu)

            if allow_short:
                w_final = self._optimize_long_short(w_unconstrained, mu, Sigma, gamma)
            else:
                w_final = self._optimize_long_only(w_unconstrained)

            weights_dict[regime_name] = w_final

        return weights_dict

    # ─── Daily soft-allocation weights ────────────────────────────────────

    def compute_daily_weights(self, results_df: pl.DataFrame,
                              regime_weights: dict,
                              min_prob_threshold: float = 0.0) -> pl.DataFrame:
        """
        Blend per-regime weights with filtered probabilities (G&T 2007 Eq. 7-10).

        w_blended = Σ_k  p_k * w_regime[k]

        Parameters
        ----------
        results_df          : pl.DataFrame with prob_crash/correction/bull/moderate_growth
        regime_weights      : dict from compute_regime_weights()
        min_prob_threshold  : if max(p_k) < threshold, go 100% cash

        Returns
        -------
        pl.DataFrame  columns: date, w_spy, w_iwm, w_tlt, w_cash
        """
        dates = results_df["date"].to_numpy()
        probs = results_df.select([
            "prob_crash", "prob_correction", "prob_bull", "prob_moderate_growth"
        ]).to_numpy()                              # (n, 4)

        # Normalise rows (numerical safety)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs    = probs / np.where(row_sums > 0, row_sums, 1.0)

        # Stack regime weights: (4, 3)
        W = np.vstack([
            regime_weights["crash"],
            regime_weights["correction"],
            regime_weights["bull"],
            regime_weights["moderate_growth"],
        ])

        w_blended = probs @ W                      # (n, 3)

        # Low-confidence days → 100% cash
        max_probs = probs.max(axis=1)
        w_blended[max_probs < min_prob_threshold] = 0.0

        w_cash = 1.0 - w_blended.sum(axis=1)

        # Handle over-leverage (risky > 1.0)
        over_lev = w_cash < 0
        if over_lev.any():
            risky_sum = w_blended[over_lev].sum(axis=1)
            w_blended[over_lev] /= risky_sum[:, np.newaxis]
            w_cash[over_lev] = 0.0

        w_cash = np.maximum(w_cash, 0.0)

        assert np.allclose(w_blended[:, 0] + w_blended[:, 1] + w_blended[:, 2] + w_cash, 1.0, atol=1e-6)

        return pl.DataFrame({
            "date":  dates,
            "w_spy": w_blended[:, 0],
            "w_iwm": w_blended[:, 1],
            "w_tlt": w_blended[:, 2],
            "w_cash": w_cash,
        })

    # ─── Daily hard-allocation weights ────────────────────────────────────

    def compute_hard_weights(self, results_df: pl.DataFrame,
                             regime_weights: dict) -> pl.DataFrame:
        """
        Assign each day the weight vector of the dominant (Viterbi) regime.
        Implements hard regime switching (Kritzman et al. 2012).

        Returns
        -------
        pl.DataFrame  columns: date, w_spy, w_iwm, w_tlt, w_cash
        """
        dates   = results_df["date"].to_numpy()
        regimes = results_df["regime"].to_numpy()
        n       = len(dates)

        w_spy_arr = np.zeros(n)
        w_iwm_arr = np.zeros(n)
        w_tlt_arr = np.zeros(n)

        for i, regime_name in enumerate(regimes):
            w = regime_weights[regime_name]
            w_spy_arr[i], w_iwm_arr[i], w_tlt_arr[i] = w

        w_cash = 1.0 - (w_spy_arr + w_iwm_arr + w_tlt_arr)
        w_cash = np.maximum(w_cash, 0.0)

        return pl.DataFrame({
            "date":  dates,
            "w_spy": w_spy_arr,
            "w_iwm": w_iwm_arr,
            "w_tlt": w_tlt_arr,
            "w_cash": w_cash,
        })

    # ─── Position classification ──────────────────────────────────────────

    def determine_positions(self, daily_weights_df: pl.DataFrame,
                            threshold: float = 0.02,
                            results_df: pl.DataFrame = None,
                            crash_threshold: float = 0.4,
                            bull_threshold:  float = 0.5) -> pl.DataFrame:
        """
        Classify each asset's daily weight as 'long', 'short', or 'flat'.

        Parameters
        ----------
        daily_weights_df : output of compute_daily_weights / compute_hard_weights
        threshold        : |weight| below this → 'flat'
        results_df       : optional; if provided, overlay crash/bull signals
        crash_threshold  : prob_crash > this → crash_signal = True
        bull_threshold   : prob_bull  > this → bull_signal = True

        Returns
        -------
        pl.DataFrame  columns: date, pos_spy, pos_iwm, pos_tlt, w_spy, w_iwm, w_tlt
                      + (regime_overlay, crash_signal, bull_signal) if results_df given
        """
        dates = daily_weights_df["date"].to_numpy()
        w_spy = daily_weights_df["w_spy"].to_numpy()
        w_iwm = daily_weights_df["w_iwm"].to_numpy()
        w_tlt = daily_weights_df["w_tlt"].to_numpy()

        def _classify(weights):
            return np.where(weights >= threshold, "long",
                   np.where(weights <= -threshold, "short", "flat"))

        positions_data = {
            "date":    dates,
            "pos_spy": _classify(w_spy),
            "pos_iwm": _classify(w_iwm),
            "pos_tlt": _classify(w_tlt),
            "w_spy":   w_spy,
            "w_iwm":   w_iwm,
            "w_tlt":   w_tlt,
        }

        if results_df is None:
            return pl.DataFrame(positions_data)

        # Optional regime overlay
        pos_df = pl.DataFrame(positions_data).join(
            results_df.select(["date", "prob_crash", "prob_bull"]),
            on="date", how="inner"
        )
        prob_crash = pos_df["prob_crash"].to_numpy()
        prob_bull  = pos_df["prob_bull"].to_numpy()

        regime_overlay = np.full(len(prob_crash), "neutral", dtype=object)
        crash_signal   = prob_crash > crash_threshold
        bull_signal    = (prob_bull > bull_threshold) & ~crash_signal

        regime_overlay[crash_signal] = "crash"
        regime_overlay[bull_signal]  = "bull"

        return pos_df.drop(["prob_crash", "prob_bull"]).with_columns([
            pl.Series("regime_overlay", regime_overlay),
            pl.Series("crash_signal",   crash_signal),
            pl.Series("bull_signal",    bull_signal),
        ])

    # ─── Constrained optimisation helpers ────────────────────────────────

    @staticmethod
    def _optimize_long_only(w_unconstrained: np.ndarray) -> np.ndarray:
        """Project unconstrained weights onto long-only simplex."""
        w = np.maximum(w_unconstrained, 0.0)
        s = w.sum()
        if s > 1e-10:
            w = w / s
        else:
            w = np.array([1.0, 0.0, 0.0])   # fallback: 100% large cap
        return w

    @staticmethod
    def _optimize_long_short(w_unconstrained: np.ndarray,
                             mu: np.ndarray, Sigma: np.ndarray,
                             gamma: float) -> np.ndarray:
        """Constrained optimisation for long/short (|w| ≤ 1.5, sum = 1)."""
        def objective(w):
            return -(np.dot(w, mu) - (gamma / 2.0) * np.dot(w, Sigma @ w))

        def jac(w):
            return -(mu - gamma * Sigma @ w)

        w0 = np.clip(w_unconstrained, -1.5, 1.5)
        constraints = [
            {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: 1.5 - np.sum(np.abs(w))},
        ]
        result = minimize(objective, w0, method="SLSQP", jac=jac,
                          bounds=[(-1.5, 1.5)] * 3, constraints=constraints,
                          options={"ftol": 1e-9, "maxiter": 1000})
        if not result.success:
            w = np.clip(w_unconstrained, -1.5, 1.5)
            return w / w.sum()
        return result.x
