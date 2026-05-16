"""
plots.py — replicate the four paper figures for USA (S&P 500).

Figure 1 : Realized volatility time series + ACF  (3 frequencies)
Figure 2 : Actual vs AIC/SIC log-vol forecasts    (OOS period, 3 frequencies)
Figure 3 : Predicted sign-probability time series  (3 models × 3 frequencies)
Figure 4a: Individual Brier(Abs) scatter — Nonparametric vs Baseline (low vol)
Figure 4b: Individual Brier(Abs) scatter — Extended     vs Baseline (low vol)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import FIG_DIR, FREQ_LABEL, FREQ_NAME, MARKET

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 130})


# ── shared axis formatter ──────────────────────────────────────────────────────

def _fmt(ax, step_years: int = 5, rotate: bool = True):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(step_years))
    if rotate:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.tick_params(labelsize=7)


# ── Figure 1 ───────────────────────────────────────────────────────────────────

def fig1_realized_volatility(period_data: dict):
    """
    2 rows × 3 columns
    Row 0 : log realized vol time series (1M | 2M | 3M)
    Row 1 : ACF of log realized vol up to lag 20
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))

    for col, h in enumerate([1, 2, 3]):
        lv    = period_data[h]["LogRealVol"].values
        dates = period_data[h].index
        lv_ok = lv[~np.isnan(lv)]
        n     = len(lv_ok)
        ci    = 1.96 / np.sqrt(n)

        # — vol series —
        ax = axes[0, col]
        ax.plot(dates, lv, linewidth=0.8, color="#1f77b4")
        ax.set_title(FREQ_NAME[h])
        if col == 0:
            ax.set_ylabel("Log Realized Vol")
        _fmt(ax, step_years=10)

        # — ACF —
        ax = axes[1, col]
        acf_vals = [1.0] + [
            float(np.corrcoef(lv_ok[:-lag], lv_ok[lag:])[0, 1])
            for lag in range(1, 21)
        ]
        ax.bar(range(21), acf_vals, color="#1f77b4", alpha=0.7, width=0.4)
        ax.axhline( ci, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(0,   color="black", linewidth=0.5)
        ax.set_xlim(-0.5, 20.5)
        ax.set_ylim(-0.45, 1.05)
        ax.set_xlabel("Lag", fontsize=7)
        if col == 0:
            ax.set_ylabel("ACF")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, linewidth=0.4)

    fig.suptitle(f"Figure 1 — Realized Volatility  [{MARKET}]", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "fig1_realized_volatility.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ── Figure 2 ───────────────────────────────────────────────────────────────────

def fig2_vol_forecasts(period_data: dict, vol_forecasts: dict):
    """
    1 row × 3 columns
    Each panel: actual log-vol (black) + AIC forecast (blue dashed) +
                SIC forecast (red dotted) over the OOS period.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for col, h in enumerate([1, 2, 3]):
        ax     = axes[col]
        fc     = vol_forecasts[h]
        actual = period_data[h].loc[fc.index, "LogRealVol"]

        ax.plot(actual.index, actual.values,
                "k-",  lw=0.9, label="Actual",   zorder=3)
        ax.plot(fc.index, fc["LogVolFcast_AIC"].values,
                "b--", lw=0.8, label="AIC Fcst", zorder=2)
        ax.plot(fc.index, fc["LogVolFcast_SIC"].values,
                "r:",  lw=0.8, label="SIC Fcst", zorder=1)

        ax.set_title(FREQ_NAME[h])
        if col == 0:
            ax.set_ylabel("Log Realized Vol")
        if col == 2:
            ax.legend(fontsize=7, loc="upper right")
        _fmt(ax, step_years=5)

    fig.suptitle(f"Figure 2 — Realized Volatility Forecasts  [{MARKET}]", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "fig2_vol_forecasts.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ── Figure 3 ───────────────────────────────────────────────────────────────────

def fig3_predicted_probabilities(forecast_results: dict):
    """
    3 rows × 3 columns
    Rows    : 1M, 2M, 3M
    Columns : Baseline | Nonparametric | Extended
    Both AIC (blue solid) and SIC (red dotted) plotted for Nonpar and Extended.
    """
    col_defs = [
        ("Baseline",      "Pr_Baseline",     None),
        ("Nonparametric", "Pr_Nonpar_AIC",   "Pr_Nonpar_SIC"),
        ("Extended",      "Pr_Extended_AIC", "Pr_Extended_SIC"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=True)

    for row, h in enumerate([1, 2, 3]):
        fdf = forecast_results[h]
        for col, (title, key_aic, key_sic) in enumerate(col_defs):
            ax = axes[row, col]

            ax.plot(fdf.index, fdf[key_aic].values,
                    linewidth=0.65, color="#1f77b4", label="AIC")
            if key_sic:
                ax.plot(fdf.index, fdf[key_sic].values,
                        linewidth=0.5, color="#d62728", linestyle=":", label="SIC")

            ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
            ax.set_ylim(0, 1)
            _fmt(ax, step_years=5)

            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(FREQ_LABEL[h], fontsize=8)
            if row == 0 and col == 2:
                ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(f"Figure 3 — Predicted Probabilities  [{MARKET}]", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "fig3_predicted_probabilities.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ── Figure 4 helper ────────────────────────────────────────────────────────────

def _scatter_panel(ax, pr_base, pr_alt, positive, rv, freq_lbl, alt_lbl):
    """
    One scatter panel for Figure 4.
    x-axis : baseline individual Brier(Abs) score
    y-axis : competing model individual Brier(Abs) score
    Only low-volatility observations (1st–33rd percentile of RV) included.
    """
    z        = positive.astype(float)
    low_mask = rv <= np.nanpercentile(rv, 33)

    base_abs = np.abs(pr_base[low_mask] - z[low_mask])
    alt_abs  = np.abs(pr_alt[low_mask]  - z[low_mask])

    ax.scatter(base_abs, alt_abs,
               s=7, alpha=0.45, color="steelblue", edgecolors="none")

    # gridlines at 0.5 + 45-degree line
    ax.axhline(0.5, color="gray",  linewidth=0.7)
    ax.axvline(0.5, color="gray",  linewidth=0.7)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Baseline Brier(Abs)", fontsize=7)
    ax.set_ylabel(f"{alt_lbl} Brier(Abs)", fontsize=7)
    ax.set_title(freq_lbl, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, linewidth=0.4)

    # quadrant labels
    for xr, yr, txt in [
        (0.25, 0.94, "both right"),
        (0.75, 0.94, "right→wrong"),
        (0.25, 0.06, "wrong→right"),
        (0.75, 0.06, "both wrong"),
    ]:
        ax.text(xr, yr, txt, fontsize=5, ha="center",
                transform=ax.transAxes, color="gray")


def fig4(forecast_results: dict, alt_col: str, alt_label: str, fig_id: str):
    """
    Figure 4a or 4b — 1 row × 3 columns (one per frequency).
    alt_col   : column name in forecast DataFrame (e.g. 'Pr_Nonpar_AIC')
    alt_label : short model name for axis label
    fig_id    : '4a' or '4b'
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for col, h in enumerate([1, 2, 3]):
        fdf = forecast_results[h]
        _scatter_panel(
            axes[col],
            pr_base  = fdf["Pr_Baseline"].values,
            pr_alt   = fdf[alt_col].values,
            positive = fdf["Positive"].values,
            rv       = fdf["RealVol"].values,
            freq_lbl = FREQ_LABEL[h],
            alt_lbl  = alt_label,
        )

    fig.suptitle(
        f"Figure {fig_id} — Comparative Brier(Abs), Low Volatility: "
        f"{alt_label} vs Baseline  [{MARKET}]",
        fontsize=9,
    )
    fig.tight_layout()
    fname = f"fig{fig_id}_brier_scatter.png"
    path  = FIG_DIR / fname
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")
