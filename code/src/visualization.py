"""
Visualisation for regime-based portfolio allocation.

Visualizer wraps all chart-generation functions in a single class.
All methods display charts via plt.show() and optionally save to file.
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import to_rgba

from src.config import REGIME_COLORS, EVENTS

sns.set_theme(style="whitegrid")


class Visualizer:
    """Chart factory for regime and performance visualisations."""

    def __init__(self, save_dir: str = None):
        """
        Parameters
        ----------
        save_dir : optional path; if set, charts are also saved as PNG files
        """
        self.save_dir = save_dir

    # ─── Regime probability chart ─────────────────────────────────────────

    def plot_regimes(self, dates, state_probs: np.ndarray) -> None:
        """
        Plot filtered state probabilities for all four regimes.

        Parameters
        ----------
        dates       : date-like iterable aligned with rows of state_probs
        state_probs : np.ndarray (n, 4) canonical order: crash, correction, bull, mod_growth
        """
        from src.config import REGIME_ORDER

        fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True,
                                 gridspec_kw={"hspace": 0.08})
        fig.suptitle(
            "Filtered State Probabilities — 4-State Gaussian HMM\n"
            "Guidolin & Timmermann (2007) | er_large, er_small, er_bond",
            fontsize=12, fontweight="bold", y=0.98,
        )

        for ax, regime_name, col_idx in zip(axes, REGIME_ORDER, range(4)):
            probs = state_probs[:, col_idx]
            color = REGIME_COLORS[regime_name]

            ax.fill_between(dates, probs, alpha=0.45, color=color)
            ax.plot(dates, probs, color=color, linewidth=0.9, label=regime_name)
            ax.axhline(0.5, color="black", linestyle="--", linewidth=0.55, alpha=0.45)
            ax.fill_between(dates, 0, probs, where=(probs > 0.5),
                            alpha=0.65, color=color, interpolate=True)

            for evt_date, evt_label in EVENTS:
                evt_dt = pd.Timestamp(evt_date)
                if pd.Timestamp(dates.min()) <= evt_dt <= pd.Timestamp(dates.max()):
                    ax.axvline(evt_dt, color="black", linestyle=":", linewidth=0.9, alpha=0.7)
                    ax.text(evt_dt, 0.92, evt_label, fontsize=7,
                            ha="center", color="black", alpha=0.75)

            ax.set_ylim(-0.02, 1.05)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_ylabel("Prob.", fontsize=9)
            ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
            ax.grid(axis="x", alpha=0.25, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(1))
        axes[-1].set_xlabel("Date", fontsize=10)
        plt.tight_layout()
        self._show_save("regime_probabilities")

    # ─── Equity curves ────────────────────────────────────────────────────

    def plot_equity_curves(self, returns_dict: dict, dates,
                           title: str = "Equity Curves (Log Scale)") -> None:
        """
        Plot cumulative log-return equity curves for multiple strategies.

        Parameters
        ----------
        returns_dict : {strategy_name: np.ndarray of daily log returns}
        dates        : date-like iterable aligned with returns arrays
        """
        plt.figure(figsize=(12, 6))
        for name, rets in returns_dict.items():
            plt.plot(dates, np.cumsum(rets), label=name, linewidth=1.5)

        plt.title(title, fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Log Return", fontsize=12)
        plt.legend(loc="upper left")
        plt.tight_layout()
        self._show_save("equity_curves")

    # ─── Drawdown chart ───────────────────────────────────────────────────

    def plot_drawdowns(self, returns_dict: dict, dates,
                       title: str = "Strategy Drawdowns") -> None:
        """
        Plot drawdown (percentage) for multiple strategies.

        Parameters
        ----------
        returns_dict : {strategy_name: np.ndarray of daily log returns}
        dates        : date-like iterable
        """
        plt.figure(figsize=(12, 4))
        for name, rets in returns_dict.items():
            cum      = np.cumsum(rets)
            run_max  = np.maximum.accumulate(cum)
            dd_pct   = (np.exp(cum - run_max) - 1) * 100
            plt.plot(dates, dd_pct, label=name, linewidth=1)

        plt.title(title, fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Drawdown (%)", fontsize=12)
        plt.legend(loc="lower left")
        plt.tight_layout()
        self._show_save("drawdowns")

    # ─── Weight evolution ─────────────────────────────────────────────────

    def plot_weight_evolution(self, weights_df: pl.DataFrame,
                              title: str = "Portfolio Weight Evolution") -> None:
        """
        Stacked area chart of portfolio weights over time.

        Parameters
        ----------
        weights_df : pl.DataFrame with columns: date, w_spy, w_iwm, w_tlt, w_cash
        """
        dates   = weights_df["date"].to_list()
        w_spy   = np.maximum(weights_df["w_spy"].to_numpy(),  0)
        w_iwm   = np.maximum(weights_df["w_iwm"].to_numpy(),  0)
        w_tlt   = np.maximum(weights_df["w_tlt"].to_numpy(),  0)
        w_cash  = np.maximum(weights_df["w_cash"].to_numpy(), 0)
        total   = w_spy + w_iwm + w_tlt + w_cash
        total   = np.where(total > 0, total, 1.0)

        plt.figure(figsize=(12, 5))
        plt.stackplot(dates, w_spy / total, w_iwm / total, w_tlt / total, w_cash / total,
                      labels=["SPY (Large Cap)", "IWM (Small Cap)", "TLT (Bond)", "Cash"],
                      alpha=0.8)
        plt.title(title, fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Weight", fontsize=12)
        plt.legend(loc="upper left")
        plt.margins(x=0)
        plt.tight_layout()
        self._show_save("weight_evolution")

    # ─── Regime overlay on equity curve ───────────────────────────────────

    def plot_regime_performance(self, returns: np.ndarray,
                                regime_series, dates,
                                title: str = "Strategy Performance by Regime") -> None:
        """
        Equity curve with active regime shaded in the background.

        Parameters
        ----------
        returns       : 1D np.ndarray of daily log returns
        regime_series : array of regime names per day
        dates         : date-like iterable
        """
        cum     = np.cumsum(returns)
        regimes = np.array(regime_series)

        plt.figure(figsize=(14, 6))
        plt.plot(dates, cum, color="black", linewidth=1.5, label="Strategy Return")

        for regime, color in REGIME_COLORS.items():
            mask = regimes == regime
            if not mask.any():
                continue
            plt.fill_between(dates, cum.min(), cum.max(),
                             where=mask,
                             color=to_rgba(color, alpha=0.2),
                             label=f"Regime: {regime}")

        plt.title(title, fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Log Return", fontsize=12)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(dict(zip(labels, handles)).values(),
                   dict(zip(labels, handles)).keys(), loc="upper left")
        plt.margins(x=0)
        plt.tight_layout()
        self._show_save("regime_performance")

    # ─── Private ──────────────────────────────────────────────────────────

    def _show_save(self, name: str) -> None:
        if self.save_dir:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, f"{name}.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()
