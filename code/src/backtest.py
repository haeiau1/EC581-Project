"""
Walk-forward backtesting engine with Backtrader integration.

Two engines are available:
  WalkForwardEngine         — standard HMM soft allocation (original notebook logic)
  EnhancedWalkForwardEngine — adds asymmetric gamma, EMA probability smoothing,
                              weight EMA + rebalancing threshold

After computing walk-forward weights, BacktraderEngine runs the strategy through
Backtrader's Cerebro to produce an independently verified equity curve and
a broker-level audit trail.

References:
  Guidolin & Timmermann (2007); Nystrup et al. (2015)
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

import backtrader as bt
import backtrader.feeds as btfeeds

from hmmlearn.hmm import GaussianHMM

from src.config import (
    GAMMA, GAMMA_BULL, GAMMA_DEFENSIVE, TRANSACTION_COST, RETRAIN_FREQ,
    OOS_START, PROB_EMA_HALFLIFE, WEIGHT_EMA_ALPHA, REBALANCE_THRESH,
)
from src.portfolio import PortfolioOptimizer

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Walk-forward weight computation (CPU-side, separate from Backtrader loop)
# ═════════════════════════════════════════════════════════════════════════════

class WalkForwardEngine:
    """
    Standard walk-forward backtest: expanding training window, periodic HMM
    refit every RETRAIN_FREQ business days, soft-allocation daily weights.
    """

    def __init__(self, gamma: float = GAMMA,
                 transaction_cost: float = TRANSACTION_COST,
                 retrain_freq:     int   = RETRAIN_FREQ,
                 oos_start:        str   = OOS_START):
        self.gamma            = gamma
        self.transaction_cost = transaction_cost
        self.retrain_freq     = retrain_freq
        self.oos_start        = datetime.strptime(oos_start, "%Y-%m-%d").date()
        self._opt             = PortfolioOptimizer()

    def run(self, results_df: pl.DataFrame,
            model: GaussianHMM,
            canonical_idx: list) -> tuple[pl.DataFrame, list]:
        """
        Execute the walk-forward backtest.

        Parameters
        ----------
        results_df    : full feature DataFrame from DataLoader (sorted by date)
        model         : initially fitted GaussianHMM
        canonical_idx : [crash_raw, correction_raw, bull_raw, moderate_growth_raw]

        Returns
        -------
        output_df     : pl.DataFrame with date, gross_return, net_return,
                        turnover, regime, w_spy, w_iwm, w_tlt, w_cash
        retrain_dates : list of dates when HMM was refitted
        """
        self._validate_inputs(results_df, model, canonical_idx)

        dates_list = results_df["date"].to_list()
        oos_dates  = [d.date() if hasattr(d, "date") else d for d in dates_list]
        oos_idx    = self._find_oos_idx(oos_dates)

        current_model = model
        regime_weights = self._opt.compute_regime_weights(current_model, canonical_idx,
                                                          gamma=self.gamma)
        all_weights = self._opt.compute_daily_weights(results_df, regime_weights)

        results_list        = []
        previous_weights    = None
        retrain_dates       = []
        days_since_retrain  = 0

        for t in range(oos_idx, len(results_df)):
            if t > oos_idx:
                days_since_retrain += 1

            # Periodic HMM refit (expanding window)
            if days_since_retrain >= self.retrain_freq and t > oos_idx:
                current_model, regime_weights, all_weights, retrain_dates = self._refit(
                    results_df, t, canonical_idx, current_model,
                    regime_weights, all_weights, oos_dates, retrain_dates
                )
                days_since_retrain = 0

            w_t      = all_weights[t].select(["w_spy", "w_iwm", "w_tlt", "w_cash"]).to_numpy().flatten()
            gross, net, turnover = self._day_pnl(results_df, t, w_t, previous_weights)

            results_list.append({
                "date":         oos_dates[t],
                "gross_return": float(gross),
                "net_return":   float(net),
                "turnover":     float(turnover),
                "regime":       results_df[t, "regime"],
                "w_spy":        float(w_t[0]),
                "w_iwm":        float(w_t[1]),
                "w_tlt":        float(w_t[2]),
                "w_cash":       float(w_t[3]),
            })
            previous_weights = w_t

        return pl.DataFrame(results_list), retrain_dates

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _day_pnl(self, results_df, t, w_t, prev):
        r_large = results_df[t, "r_large"]
        r_small = results_df[t, "r_small"]
        r_bond  = results_df[t, "r_bond"]

        gross_simple = (w_t[0] * np.exp(r_large) + w_t[1] * np.exp(r_small)
                        + w_t[2] * np.exp(r_bond) + w_t[3])
        gross  = np.log(np.maximum(gross_simple, 1e-8))

        turnover = 1.0 if prev is None else np.sum(np.abs(w_t - prev)) / 2.0
        net_factor = np.exp(gross) - turnover * self.transaction_cost
        net = np.log(np.maximum(net_factor, 1e-8))

        return gross, net, turnover

    def _refit(self, results_df, t, canonical_idx, prev_model,
               regime_weights, all_weights, oos_dates, retrain_dates):
        try:
            X = results_df[:t].select(["r_large", "r_small", "r_bond"]).to_numpy()
            refitted = GaussianHMM(n_components=4, covariance_type="full", random_state=42)
            refitted.fit(X)
            new_weights = self._opt.compute_regime_weights(refitted, canonical_idx, self.gamma)
            new_daily   = self._opt.compute_daily_weights(results_df, new_weights)
            retrain_dates.append(oos_dates[t])
            return refitted, new_weights, new_daily, retrain_dates
        except Exception as e:
            logger.warning(f"HMM refit failed on {oos_dates[t]}: {e}. Using previous model.")
            return prev_model, regime_weights, all_weights, retrain_dates

    @staticmethod
    def _find_oos_idx(oos_dates):
        for idx, d in enumerate(oos_dates):
            if d >= datetime.strptime(OOS_START, "%Y-%m-%d").date():
                return idx
        raise ValueError(f"OOS_START {OOS_START} not found in data")

    @staticmethod
    def _validate_inputs(results_df, model, canonical_idx):
        required = ["date", "r_large", "r_small", "r_bond", "rf_daily",
                    "regime", "prob_crash", "prob_correction",
                    "prob_bull", "prob_moderate_growth"]
        missing = [c for c in required if c not in results_df.columns]
        if missing:
            raise ValueError(f"results_df missing columns: {missing}")
        if not hasattr(model, "means_"):
            raise ValueError("model must be a fitted GaussianHMM")
        if len(canonical_idx) != 4:
            raise ValueError("canonical_idx must have 4 elements")


# ═════════════════════════════════════════════════════════════════════════════
# Enhanced walk-forward (asymmetric gamma + prob EMA + weight EMA)
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedWalkForwardEngine(WalkForwardEngine):
    """
    Three improvements over the base engine:
    1. Asymmetric gamma: higher risk aversion during crash / correction
    2. EMA on filtered regime probabilities (confirmation lag)
    3. Weight EMA + minimum rebalance threshold (reduces unnecessary turnover)
    """

    def __init__(self, gamma_bull:          float = GAMMA_BULL,
                       gamma_defensive:     float = GAMMA_DEFENSIVE,
                       transaction_cost:    float = TRANSACTION_COST,
                       retrain_freq:        int   = RETRAIN_FREQ,
                       oos_start:           str   = OOS_START,
                       prob_ema_halflife:   float = PROB_EMA_HALFLIFE,
                       weight_ema_alpha:    float = WEIGHT_EMA_ALPHA,
                       rebalance_threshold: float = REBALANCE_THRESH):
        super().__init__(gamma=gamma_bull, transaction_cost=transaction_cost,
                         retrain_freq=retrain_freq, oos_start=oos_start)
        self.gamma_bull          = gamma_bull
        self.gamma_defensive     = gamma_defensive
        self.prob_ema_halflife   = prob_ema_halflife
        self.weight_ema_alpha    = weight_ema_alpha
        self.rebalance_threshold = rebalance_threshold

    def run(self, results_df: pl.DataFrame,
            model: GaussianHMM,
            canonical_idx: list) -> tuple[pl.DataFrame, list]:
        """Enhanced walk-forward with three improvements applied."""

        # Improvement 2: smooth regime probabilities before computing weights
        smoothed_df = self._smooth_probs(results_df)

        self._validate_inputs(results_df, model, canonical_idx)
        dates_list = smoothed_df["date"].to_list()
        oos_dates  = [d.date() if hasattr(d, "date") else d for d in dates_list]
        oos_idx    = self._find_oos_idx(oos_dates)

        current_model  = model
        # Improvement 1: asymmetric gamma
        regime_weights = self._asymmetric_weights(current_model, canonical_idx)
        all_weights    = self._opt.compute_daily_weights(smoothed_df, regime_weights)

        results_list       = []
        prev_w             = None
        held_w             = None     # actual holdings (after EMA + threshold)
        retrain_dates      = []
        days_since_retrain = 0

        for t in range(oos_idx, len(results_df)):
            if t > oos_idx:
                days_since_retrain += 1

            if days_since_retrain >= self.retrain_freq and t > oos_idx:
                try:
                    X = results_df[:t].select(["r_large", "r_small", "r_bond"]).to_numpy()
                    refitted = GaussianHMM(n_components=4, covariance_type="full", random_state=42)
                    refitted.fit(X)
                    current_model  = refitted
                    regime_weights = self._asymmetric_weights(refitted, canonical_idx)
                    all_weights    = self._opt.compute_daily_weights(smoothed_df, regime_weights)
                    retrain_dates.append(oos_dates[t])
                    days_since_retrain = 0
                except Exception as e:
                    logger.warning(f"Enhanced refit failed on {oos_dates[t]}: {e}")

            w_target = all_weights[t].select(["w_spy", "w_iwm", "w_tlt", "w_cash"]).to_numpy().flatten()

            # Improvement 3: weight EMA + rebalancing threshold
            if held_w is None:
                w_t = w_target.copy()
            else:
                w_smooth = self.weight_ema_alpha * w_target + (1 - self.weight_ema_alpha) * held_w
                w_smooth = np.maximum(w_smooth, 0.0)
                w_smooth /= w_smooth.sum()
                # Only trade if max drift exceeds threshold
                if np.max(np.abs(w_smooth - held_w)) >= self.rebalance_threshold:
                    w_t = w_smooth
                else:
                    w_t = held_w

            gross, net, turnover = self._day_pnl(results_df, t, w_t, prev_w)

            results_list.append({
                "date":         oos_dates[t],
                "gross_return": float(gross),
                "net_return":   float(net),
                "turnover":     float(turnover),
                "regime":       results_df[t, "regime"],
                "w_spy":        float(w_t[0]),
                "w_iwm":        float(w_t[1]),
                "w_tlt":        float(w_t[2]),
                "w_cash":       float(w_t[3]),
            })
            prev_w  = w_t.copy()
            held_w  = w_t.copy()

        return pl.DataFrame(results_list), retrain_dates

    def _asymmetric_weights(self, model, canonical_idx):
        """Compute per-regime weights with different gamma for bull vs defensive."""
        regime_gammas = {
            "crash":           self.gamma_defensive,
            "correction":      self.gamma_defensive,
            "bull":            self.gamma_bull,
            "moderate_growth": self.gamma_bull,
        }
        import numpy as np
        from scipy import linalg

        regime_names  = ["crash", "correction", "bull", "moderate_growth"]
        weights_dict  = {}

        for i, name in enumerate(regime_names):
            raw_idx   = canonical_idx[i]
            mu        = model.means_[raw_idx]
            Sigma     = model.covars_[raw_idx]
            g         = regime_gammas[name]
            Sigma_inv = linalg.inv(Sigma + 1e-6 * np.eye(3))
            w_raw     = (1.0 / g) * (Sigma_inv @ mu)
            w         = np.maximum(w_raw, 0.0)
            s         = w.sum()
            weights_dict[name] = w / s if s > 1e-10 else np.array([0.0, 0.0, 1.0])

        return weights_dict

    def _smooth_probs(self, results_df: pl.DataFrame) -> pl.DataFrame:
        """Apply EMA to regime filtered probabilities (confirmation lag)."""
        alpha    = 1 - np.exp(-np.log(2) / self.prob_ema_halflife)
        cols     = ["prob_crash", "prob_correction", "prob_bull", "prob_moderate_growth"]
        probs    = results_df.select(cols).to_numpy()
        smoothed = np.zeros_like(probs)
        smoothed[0] = probs[0]

        for t in range(1, len(probs)):
            smoothed[t] = alpha * probs[t] + (1 - alpha) * smoothed[t - 1]
            s = smoothed[t].sum()
            if s > 0:
                smoothed[t] /= s

        df = results_df.clone()
        for i, col in enumerate(cols):
            df = df.with_columns(pl.Series(col, smoothed[:, i]))
        return df


# ═════════════════════════════════════════════════════════════════════════════
# Backtrader integration
# ═════════════════════════════════════════════════════════════════════════════

class _PandasOHLCV(btfeeds.PandasData):
    """Backtrader pandas feed that treats close price as the only meaningful field."""
    params = (
        ("datetime", None),
        ("open",  "open"),
        ("high",  "high"),
        ("low",   "low"),
        ("close", "close"),
        ("volume", -1),
        ("openinterest", -1),
    )


class _RegimeAllocationStrategy(bt.Strategy):
    """
    Backtrader strategy that rebalances to pre-computed target weights each day.
    Weights come from the walk-forward engine (run before Cerebro).
    """

    params = (
        ("weights_df",        None),   # pl.DataFrame: date, w_spy, w_iwm, w_tlt
        ("commission",        TRANSACTION_COST),
        ("tickers",           ["SPY", "IWM", "TLT"]),
    )

    def __init__(self):
        # Build lookup: date → (w_spy, w_iwm, w_tlt)
        df = self.params.weights_df
        self._weights = {}
        for row in df.iter_rows(named=True):
            d = row["date"] if not hasattr(row["date"], "date") else row["date"]
            self._weights[d] = np.array([row["w_spy"], row["w_iwm"], row["w_tlt"]])

    def next(self):
        dt = self.datas[0].datetime.date(0)
        if dt not in self._weights:
            return

        targets = self._weights[dt]  # [w_spy, w_iwm, w_tlt]
        for data, target in zip(self.datas, targets):
            self.order_target_percent(data, target=float(target))

    def stop(self):
        pass


class BacktraderEngine:
    """
    Runs pre-computed walk-forward weights through Backtrader's Cerebro for
    an independent execution simulation with broker-level trade accounting.

    Usage
    -----
    engine = BacktraderEngine()
    bt_results = engine.run(weights_df, price_df)
    """

    def __init__(self, initial_cash: float = 1_000_000,
                 commission:    float = TRANSACTION_COST):
        self.initial_cash = initial_cash
        self.commission   = commission

    def run(self, weights_df: pl.DataFrame,
            price_df:   pd.DataFrame) -> dict:
        """
        Run Cerebro with pre-computed weights.

        Parameters
        ----------
        weights_df : pl.DataFrame  columns: date, w_spy, w_iwm, w_tlt
                     (OOS period only, aligned dates)
        price_df   : pd.DataFrame  index=date, columns=[spy, iwm, tlt]
                     (full date range; Backtrader starts from OOS dates)

        Returns
        -------
        dict with keys:
          final_value  : float — final portfolio value
          returns      : np.ndarray — daily log returns (from Backtrader)
          equity_curve : pd.Series — cumulative portfolio value
        """
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        # 5 bps commission on each trade (applied per side)
        cerebro.broker.setcommission(commission=self.commission)

        tickers  = ["spy", "iwm", "tlt"]
        oos_dates = weights_df["date"].to_list()
        start = min(oos_dates) if not hasattr(min(oos_dates), "strftime") else min(oos_dates)

        for ticker in tickers:
            if ticker not in price_df.columns:
                raise ValueError(f"price_df missing column '{ticker}'")
            col_df = self._make_ohlcv(price_df[[ticker]].rename(columns={ticker: "close"}))
            feed = _PandasOHLCV(dataname=col_df)
            cerebro.adddata(feed, name=ticker.upper())

        cerebro.addstrategy(
            _RegimeAllocationStrategy,
            weights_df=weights_df,
            commission=self.commission,
            tickers=["SPY", "IWM", "TLT"],
        )

        # Record portfolio values daily
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
        cerebro.addanalyzer(bt.analyzers.DrawDown,   _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                            riskfreerate=0.0, annualize=True)

        strats = cerebro.run()
        strat  = strats[0]

        time_ret  = strat.analyzers.time_return.get_analysis()
        dates_ret = list(time_ret.keys())
        rets_arr  = np.array(list(time_ret.values()))
        log_rets  = np.log(1 + rets_arr)

        equity_curve = pd.Series(
            np.exp(np.cumsum(log_rets)),
            index=pd.to_datetime(dates_ret),
            name="Backtrader Equity",
        )

        return {
            "final_value":  cerebro.broker.getvalue(),
            "returns":      log_rets,
            "equity_curve": equity_curve,
            "dates":        dates_ret,
            "drawdown":     strat.analyzers.drawdown.get_analysis(),
            "sharpe":       strat.analyzers.sharpe.get_analysis(),
        }

    @staticmethod
    def _make_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Convert a close-only Series to OHLCV (O=H=L=C, V=0) for Backtrader."""
        c = df["close"]
        return pd.DataFrame({
            "open":   c,
            "high":   c,
            "low":    c,
            "close":  c,
            "volume": 0,
        }, index=df.index)
