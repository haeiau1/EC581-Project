"""
Benchmark strategies for comparison with HMM soft regime allocation.

BenchmarkRunner implements three reference portfolios:
1. equal_weight    — 1/N daily rebalance (SPY, IWM, TLT equally weighted)
2. buy_and_hold    — 60/40 equity/bond, no rebalancing after entry
3. static_mv       — MV optimal weights fixed from in-sample data, held OOS

All benchmarks apply the same 5 bps transaction-cost model.
"""

import numpy as np
import polars as pl
from datetime import datetime

from src.config import OOS_START, TRANSACTION_COST


class BenchmarkRunner:
    """Compute walk-forward returns for the three reference portfolios."""

    def __init__(self, transaction_cost: float = TRANSACTION_COST,
                 oos_start: str = OOS_START):
        self.transaction_cost = transaction_cost
        self.oos_start = datetime.strptime(oos_start, "%Y-%m-%d").date()

    # ─── 1/N Equal-weight ────────────────────────────────────────────────

    def equal_weight(self, results_df: pl.DataFrame) -> pl.DataFrame:
        """
        Equal-weight benchmark: 1/3 each to SPY, IWM, TLT, rebalanced daily.

        Returns
        -------
        pl.DataFrame  columns: date, gross_return, net_return, turnover
        """
        oos_df = self._filter_oos(results_df)
        n = len(oos_df)
        w = np.full((n, 4), [1/3, 1/3, 1/3, 0.0])   # [spy, iwm, tlt, cash]

        gross = self._portfolio_returns(oos_df, w)
        turnover = self._compute_turnover(w)
        net = self._apply_costs(gross, turnover)

        return self._build_df(oos_df["date"], gross, net, turnover, w)

    # ─── 60/40 Buy-and-hold ───────────────────────────────────────────────

    def buy_and_hold(self, results_df: pl.DataFrame,
                     equity_weight: float = 0.6,
                     bond_weight:   float = 0.4) -> pl.DataFrame:
        """
        Buy-and-hold benchmark: fixed equity/bond allocation, no rebalancing.

        Parameters
        ----------
        equity_weight : fraction to SPY (all equity in SPY, 0 in IWM)
        bond_weight   : fraction to TLT

        Returns
        -------
        pl.DataFrame  columns: date, gross_return, net_return, turnover
        """
        oos_df = self._filter_oos(results_df)
        n = len(oos_df)
        cash = 1.0 - equity_weight - bond_weight
        w = np.tile([equity_weight, 0.0, bond_weight, cash], (n, 1))

        gross = self._portfolio_returns(oos_df, w)
        turnover = self._compute_turnover(w)   # only day-0 has cost
        net = self._apply_costs(gross, turnover)

        return self._build_df(oos_df["date"], gross, net, turnover, w)

    # ─── Static mean-variance ────────────────────────────────────────────

    def static_mv(self, results_df: pl.DataFrame,
                  model, canonical_idx: list,
                  gamma: float = 5) -> pl.DataFrame:
        """
        Static MV benchmark: single MV optimisation on in-sample data, held OOS.
        Uses the first-day OOS soft-allocation weights as a fixed portfolio.

        Parameters
        ----------
        model         : GaussianHMM fitted on full dataset
        canonical_idx : canonical regime ordering

        Returns
        -------
        pl.DataFrame  columns: date, gross_return, net_return, turnover
        """
        from src.portfolio import PortfolioOptimizer

        oos_df = self._filter_oos(results_df)
        n = len(oos_df)

        opt = PortfolioOptimizer()
        regime_weights = opt.compute_regime_weights(model, canonical_idx,
                                                    gamma=gamma, allow_short=False)
        # Fix weights = first OOS day's soft-allocation blend
        first_day_weights = opt.compute_daily_weights(oos_df.slice(0, 1), regime_weights)
        w0 = first_day_weights.select(["w_spy", "w_iwm", "w_tlt", "w_cash"]).to_numpy().flatten()

        w = np.tile(w0, (n, 1))

        gross = self._portfolio_returns(oos_df, w)
        turnover = self._compute_turnover(w)
        net = self._apply_costs(gross, turnover)

        return self._build_df(oos_df["date"], gross, net, turnover, w)

    # ─── Private helpers ─────────────────────────────────────────────────

    def _filter_oos(self, results_df: pl.DataFrame) -> pl.DataFrame:
        """Filter DataFrame to OOS period only."""
        dates = results_df["date"].to_list()
        mask  = [
            (d.date() if hasattr(d, "date") else d) >= self.oos_start
            for d in dates
        ]
        oos = results_df.filter(pl.Series(mask))
        if len(oos) == 0:
            raise ValueError(f"No data found on or after {self.oos_start}")
        return oos

    @staticmethod
    def _portfolio_returns(oos_df: pl.DataFrame,
                           w: np.ndarray) -> np.ndarray:
        """Vectorised log portfolio return from daily weight matrix."""
        r_large = oos_df["r_large"].to_numpy()
        r_small = oos_df["r_small"].to_numpy()
        r_bond  = oos_df["r_bond"].to_numpy()

        gross_simple = (
            w[:, 0] * np.exp(r_large)
            + w[:, 1] * np.exp(r_small)
            + w[:, 2] * np.exp(r_bond)
            + w[:, 3]  # cash at 0% return
        )
        return np.log(np.maximum(gross_simple, 1e-8))

    @staticmethod
    def _compute_turnover(w: np.ndarray) -> np.ndarray:
        """Turnover = sum(|Δw|)/2; first day = 1.0 (entering from cash)."""
        n = len(w)
        turnover = np.zeros(n)
        turnover[0] = 1.0
        if n > 1:
            diffs = np.abs(np.diff(w, axis=0)).sum(axis=1) / 2.0
            turnover[1:] = diffs
        return turnover

    def _apply_costs(self, gross: np.ndarray,
                     turnover: np.ndarray) -> np.ndarray:
        """Deduct transaction costs from log returns."""
        net_factor = np.exp(gross) - turnover * self.transaction_cost
        return np.log(np.maximum(net_factor, 1e-8))

    @staticmethod
    def _build_df(dates, gross: np.ndarray, net: np.ndarray,
                  turnover: np.ndarray, w: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({
            "date":         dates,
            "gross_return": gross,
            "net_return":   net,
            "turnover":     turnover,
            "w_spy":        w[:, 0],
            "w_iwm":        w[:, 1],
            "w_tlt":        w[:, 2],
            "w_cash":       w[:, 3],
        })
