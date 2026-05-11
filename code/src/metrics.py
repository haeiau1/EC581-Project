"""
Performance metrics and strategy comparison for regime-based portfolio allocation.

PerformanceAnalyzer computes:
  - Annualised return, volatility, Sharpe, Sortino, Calmar, max drawdown
  - Power utility (Guidolin & Timmermann 2007 Eq. 15-16)
  - Regime-conditional Sharpe ratios
"""

import numpy as np
import pandas as pd
import polars as pl

# Numerical safety: use Taylor expansion for very large gamma
_GAMMA_TAYLOR_THRESHOLD = 100


class PerformanceAnalyzer:
    """Computes and displays performance metrics for one or many strategies."""

    # ─── Core metrics ─────────────────────────────────────────────────────

    def compute_metrics(self, returns_series, rf_daily,
                        trading_days: int = 252) -> dict:
        """
        Compute comprehensive performance metrics from daily log returns.

        Parameters
        ----------
        returns_series : pl.DataFrame (with 'net_return' column) or np.ndarray
        rf_daily       : pl.DataFrame (with 'rf_daily' column) or np.ndarray
        trading_days   : business days per year for annualisation

        Returns
        -------
        dict with keys: annualized_return, annualized_vol, sharpe, max_drawdown,
                        calmar, sortino, win_rate, avg_win, avg_loss, turnover_mean
        """
        returns, rf, turnover = self._extract_arrays(returns_series, rf_daily)

        excess = returns - rf
        ann_return = np.mean(returns) * trading_days
        ann_vol    = np.std(returns, ddof=1) * np.sqrt(trading_days)

        excess_std = np.std(excess, ddof=1)
        sharpe = (np.mean(excess) / excess_std) * np.sqrt(trading_days) if excess_std > 1e-12 else 0.0

        cum     = np.cumsum(returns)
        max_dd  = np.min(cum - np.maximum.accumulate(cum))
        calmar  = ann_return / abs(max_dd) if max_dd < -1e-12 else (np.inf if ann_return > 0 else 0.0)

        downside = returns[returns < 0]
        ds_std   = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
        sortino  = (np.mean(excess) / ds_std) * np.sqrt(trading_days) if ds_std > 1e-12 else 0.0

        pos = returns[returns > 0]
        neg = returns[returns < 0]

        return {
            "annualized_return": float(ann_return),
            "annualized_vol":    float(ann_vol),
            "sharpe":            float(sharpe),
            "max_drawdown":      float(max_dd),
            "calmar":            float(calmar),
            "sortino":           float(sortino),
            "win_rate":          float(np.sum(returns > 0) / len(returns)),
            "avg_win":           float(np.mean(pos)) if len(pos) > 0 else 0.0,
            "avg_loss":          float(np.mean(neg)) if len(neg) > 0 else 0.0,
            "turnover_mean":     float(np.mean(turnover)) if turnover is not None else None,
        }

    def compute_realized_utility(self, returns_series, gamma: float = 5) -> float:
        """
        Power utility U = E[(1+r)^(1-γ) / (1-γ)]  (G&T 2007 Eq. 15-16).
        Input should be a pl.DataFrame with 'net_return' (log returns) or an ndarray
        of simple returns.
        """
        if np.isclose(gamma, 1.0):
            raise ValueError("gamma=1.0 undefined for power utility.")

        if isinstance(returns_series, pl.DataFrame):
            simple = np.exp(returns_series["net_return"].to_numpy()) - 1.0
        else:
            simple = np.asarray(returns_series)

        one_plus_r = np.maximum(1.0 + simple, 1e-8)

        if gamma > _GAMMA_TAYLOR_THRESHOLD:
            utility = np.log(one_plus_r)
        else:
            utility = (np.power(one_plus_r, 1.0 - gamma) - 1.0) / (1.0 - gamma)

        return float(np.mean(utility))

    # ─── Comparison table ─────────────────────────────────────────────────

    def print_comparison_table(self, strategies_dict: dict,
                               rf_daily, gamma: float = 5) -> None:
        """
        Print a formatted comparison table for all strategies.

        Parameters
        ----------
        strategies_dict : {name: pl.DataFrame}  — each DataFrame has 'net_return'
        rf_daily        : pl.Series or np.ndarray of daily risk-free rates
        gamma           : risk-aversion for utility calculation
        """
        print("=" * 110)
        print("  STRATEGY PERFORMANCE COMPARISON (OOS)")
        print("=" * 110)

        rows = []
        for name, df in strategies_dict.items():
            m = self.compute_metrics(df, rf_daily)
            u = self.compute_realized_utility(df, gamma=gamma)
            rows.append({
                "Strategy":          name,
                "Ann. Return":       f"{m['annualized_return']*100:.2f}%",
                "Ann. Vol":          f"{m['annualized_vol']*100:.2f}%",
                "Sharpe":            f"{m['sharpe']:.2f}",
                "Max DD":            f"{m['max_drawdown']*100:.2f}%",
                "Sortino":           f"{m['sortino']:.2f}",
                "Calmar":            f"{m['calmar']:.2f}",
                "Win Rate":          f"{m['win_rate']*100:.1f}%",
                "Turnover":          f"{m['turnover_mean']*100:.2f}%" if m["turnover_mean"] else "N/A",
                f"Utility(γ={gamma})": f"{u:.6f}",
            })

        df_out = pd.DataFrame(rows)
        print(df_out.to_string(index=False))
        print("=" * 110)

    # ─── Regime-conditional analysis ──────────────────────────────────────

    def regime_conditional_analysis(self, results_df: pl.DataFrame,
                                    strategy_returns: np.ndarray,
                                    strategy_name: str = "Strategy") -> None:
        """
        Print Sharpe ratios broken down by the active HMM regime.

        Parameters
        ----------
        results_df       : pl.DataFrame with 'regime' and 'rf_daily' columns
        strategy_returns : 1D np.ndarray of daily log returns (aligned with results_df)
        strategy_name    : label for the header
        """
        print("\n" + "=" * 65)
        print(f"  REGIME-CONDITIONAL PERFORMANCE: {strategy_name.upper()}")
        print("=" * 65)
        print(f"  {'Regime':<18} | {'Days':<6} | {'Ann. Ret':<10} | {'Ann. Vol':<10} | {'Sharpe':<6}")
        print("  " + "-" * 59)

        regimes = results_df["regime"].to_numpy()
        rf      = results_df["rf_daily"].to_numpy()

        for regime in ["bull", "moderate_growth", "correction", "crash"]:
            mask = regimes == regime
            n    = mask.sum()
            if n == 0:
                continue
            r   = strategy_returns[mask]
            rf_ = rf[mask]
            ann_ret = np.mean(r) * 252
            ann_vol = np.std(r, ddof=1) * np.sqrt(252) if n > 1 else 0.0
            exc_std = np.std(r - rf_, ddof=1) if n > 1 else 0.0
            sharpe  = (np.mean(r - rf_) / exc_std) * np.sqrt(252) if exc_std > 1e-12 else 0.0
            print(f"  {regime:<18} | {n:<6} | {ann_ret*100:>8.2f}% | {ann_vol*100:>8.2f}% | {sharpe:>6.2f}")

        print("=" * 65)

    # ─── Helper ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_arrays(returns_series, rf_daily):
        """Convert polars DataFrames or numpy arrays to aligned numpy arrays."""
        if isinstance(returns_series, pl.DataFrame):
            returns  = returns_series["net_return"].to_numpy()
            turnover = returns_series["turnover"].to_numpy() if "turnover" in returns_series.columns else None
        else:
            returns  = np.asarray(returns_series)
            turnover = None

        if isinstance(rf_daily, pl.DataFrame):
            rf = rf_daily["rf_daily"].to_numpy()
        elif isinstance(rf_daily, pl.Series):
            rf = rf_daily.to_numpy()
        else:
            rf = np.asarray(rf_daily)

        if len(rf) != len(returns):
            # Broadcast scalar or use mean
            rf = np.full(len(returns), rf.mean()) if len(rf) > 1 else np.full(len(returns), float(rf))

        return returns, rf, turnover
