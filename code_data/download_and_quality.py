"""
EC581 Project — S&P 500 Sector Indexes
Step 1: Data Download + Quality Report

Downloads daily price data for all 11 S&P 500 GICS sector indexes
from Yahoo Finance (earliest available to today), then produces a
comprehensive data quality report.

Output
------
data/raw/<SECTOR>.csv          — raw daily OHLCV data
results/quality_report/
    quality_summary.csv        — one row per sector, key stats
    quality_report.txt         — human-readable report
    plots/
        01_price_levels.png    — log-price series, all sectors
        02_daily_returns.png   — daily return series, all sectors
        03_missing_heatmap.png — monthly missing-day heatmap
        04_return_dist.png     — return distribution (hist + QQ)
"""

import os
import warnings
import textwrap
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
RAW_DIR    = BASE / "data" / "raw"
REPORT_DIR = BASE / "results" / "quality_report"
PLOT_DIR   = REPORT_DIR / "plots"

for d in [RAW_DIR, REPORT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Sector definitions ─────────────────────────────────────────────────────────
SECTORS = {
    "IT":           ("^SP500-45", "S&P 500 Information Technology"),
    "CommSvcs":     ("^SP500-50", "S&P 500 Communication Services"),
    "ConDisc":      ("^SP500-25", "S&P 500 Consumer Discretionary"),
    "ConStaples":   ("^SP500-30", "S&P 500 Consumer Staples"),
    "Financials":   ("^SP500-40", "S&P 500 Financials"),
    "HealthCare":   ("^SP500-35", "S&P 500 Health Care"),
    "Industrials":  ("^SP500-20", "S&P 500 Industrials"),
    "Energy":       ("^GSPE",     "S&P 500 Energy"),
    "Materials":    ("^SP500-15", "S&P 500 Materials"),
    "Utilities":    ("^SP500-55", "S&P 500 Utilities"),
    "RealEstate":   ("^SP500-60", "S&P 500 Real Estate"),
}

TODAY = date.today().isoformat()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_all() -> dict[str, pd.DataFrame]:
    """Download daily Close prices for every sector. Returns dict of DataFrames."""
    raw: dict[str, pd.DataFrame] = {}

    print("=" * 65)
    print("DOWNLOADING S&P 500 SECTOR DATA")
    print("=" * 65)

    for key, (ticker, label) in SECTORS.items():
        print(f"  {label:<45} [{ticker}]", end="  ")
        try:
            df = yf.download(
                ticker,
                start="1989-01-01",   # request as far back as possible
                end=TODAY,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                print("NO DATA")
                continue

            # flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            df.to_csv(RAW_DIR / f"{key}.csv")

            raw[key] = df
            print(f"{df.index[0].date()} → {df.index[-1].date()}  "
                  f"({len(df):,} rows)")
        except Exception as exc:
            print(f"ERROR: {exc}")

    print()
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def expected_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Rough estimate: US market open ~252 days/year."""
    years = (end - start).days / 365.25
    return int(years * 252)


def largest_gap(index: pd.DatetimeIndex) -> int:
    """Largest calendar-day gap between consecutive observations."""
    if len(index) < 2:
        return 0
    deltas = pd.Series(index).diff().dt.days.dropna()
    return int(deltas.max())


def outlier_count(returns: pd.Series, z_threshold: float = 4.0) -> int:
    """Number of daily returns beyond z_threshold standard deviations."""
    z = (returns - returns.mean()) / returns.std()
    return int((z.abs() > z_threshold).sum())


def zero_return_count(returns: pd.Series) -> int:
    return int((returns == 0).sum())


def quality_check(key: str, df: pd.DataFrame) -> dict:
    close = df["Close"].dropna()
    ret   = compute_daily_returns(close).dropna()

    first_date = close.index[0]
    last_date  = close.index[-1]
    n_obs      = len(close)
    n_missing  = int(df["Close"].isna().sum())
    pct_miss   = n_missing / max(len(df), 1) * 100

    exp_days   = expected_trading_days(first_date, last_date)
    coverage   = n_obs / exp_days * 100 if exp_days > 0 else np.nan

    max_gap    = largest_gap(close.index)

    # Return stats
    ann_ret  = ret.mean() * 252
    ann_vol  = ret.std() * np.sqrt(252)
    skew     = float(ret.skew())
    kurt     = float(ret.kurtosis())          # excess kurtosis
    min_ret  = float(ret.min())
    max_ret  = float(ret.max())
    n_out    = outlier_count(ret, z_threshold=4)
    n_zero   = zero_return_count(ret)

    # Jarque-Bera normality test
    jb_stat, jb_pval = stats.jarque_bera(ret.dropna())

    # Autocorrelation of returns at lag 1
    ac1 = float(ret.autocorr(lag=1))

    # ADF stationarity test on log-price
    from statsmodels.tsa.stattools import adfuller
    log_price = np.log(close)
    adf_stat, adf_pval, *_ = adfuller(log_price, autolag="AIC")

    return {
        "sector":           key,
        "label":            SECTORS[key][1],
        "ticker":           SECTORS[key][0],
        "first_date":       first_date.date().isoformat(),
        "last_date":        last_date.date().isoformat(),
        "n_obs":            n_obs,
        "n_missing_close":  n_missing,
        "pct_missing":      round(pct_miss, 4),
        "exp_trading_days": exp_days,
        "coverage_pct":     round(coverage, 2),
        "max_gap_days":     max_gap,
        "ann_return":       round(ann_ret * 100, 2),
        "ann_volatility":   round(ann_vol * 100, 2),
        "skewness":         round(skew, 4),
        "excess_kurtosis":  round(kurt, 4),
        "min_daily_ret_pct":round(min_ret * 100, 4),
        "max_daily_ret_pct":round(max_ret * 100, 4),
        "n_outliers_4sd":   n_out,
        "n_zero_returns":   n_zero,
        "jb_stat":          round(float(jb_stat), 2),
        "jb_pval":          round(float(jb_pval), 6),
        "ret_autocorr_lag1":round(ac1, 4),
        "adf_stat":         round(float(adf_stat), 4),
        "adf_pval":         round(float(adf_pval), 4),
    }


def run_quality_checks(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("=" * 65)
    print("RUNNING DATA QUALITY CHECKS")
    print("=" * 65)

    rows = []
    for key, df in raw.items():
        try:
            result = quality_check(key, df)
            rows.append(result)
            print(f"  {key:<14}  OK")
        except Exception as exc:
            print(f"  {key:<14}  FAILED: {exc}")

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORT_DIR / "quality_summary.csv", index=False)
    print(f"\n  Saved → results/quality_report/quality_summary.csv\n")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HUMAN-READABLE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_text_report(summary: pd.DataFrame):
    lines = []
    lines.append("=" * 70)
    lines.append("EC581 PROJECT — S&P 500 SECTOR DATA QUALITY REPORT")
    lines.append(f"Generated: {TODAY}")
    lines.append("=" * 70)

    # ── Coverage table ──────────────────────────────────────────────────────
    lines.append("\n[1] DATA COVERAGE\n")
    lines.append(f"{'Sector':<16} {'First Date':<12} {'Last Date':<12} "
                 f"{'N Obs':>7} {'Exp Days':>9} {'Coverage':>9}  {'Max Gap':>8}")
    lines.append("-" * 72)
    for _, r in summary.iterrows():
        lines.append(
            f"{r['sector']:<16} {r['first_date']:<12} {r['last_date']:<12} "
            f"{r['n_obs']:>7,} {r['exp_trading_days']:>9,} "
            f"{r['coverage_pct']:>8.1f}%  {r['max_gap_days']:>7}d"
        )

    # ── Missing values ──────────────────────────────────────────────────────
    lines.append("\n[2] MISSING VALUES\n")
    lines.append(f"{'Sector':<16} {'Missing (Close)':>16} {'Pct Missing':>12} {'Zero Returns':>13}")
    lines.append("-" * 60)
    for _, r in summary.iterrows():
        lines.append(
            f"{r['sector']:<16} {r['n_missing_close']:>16,} "
            f"{r['pct_missing']:>11.4f}%  {r['n_zero_returns']:>12,}"
        )

    # ── Return statistics ───────────────────────────────────────────────────
    lines.append("\n[3] DAILY RETURN STATISTICS (annualized where noted)\n")
    lines.append(f"{'Sector':<16} {'Ann Ret%':>9} {'Ann Vol%':>9} "
                 f"{'Skew':>8} {'Ex.Kurt':>8} {'Min%':>8} {'Max%':>8} {'Outliers(4σ)':>13}")
    lines.append("-" * 85)
    for _, r in summary.iterrows():
        lines.append(
            f"{r['sector']:<16} {r['ann_return']:>9.2f} {r['ann_volatility']:>9.2f} "
            f"{r['skewness']:>8.4f} {r['excess_kurtosis']:>8.4f} "
            f"{r['min_daily_ret_pct']:>8.3f} {r['max_daily_ret_pct']:>8.3f} "
            f"{r['n_outliers_4sd']:>13,}"
        )

    # ── Normality & autocorrelation ─────────────────────────────────────────
    lines.append("\n[4] NORMALITY TEST (Jarque-Bera) & AUTOCORRELATION\n")
    lines.append(f"{'Sector':<16} {'JB Stat':>10} {'JB p-val':>10} "
                 f"{'AC(1) ret':>10} {'ADF p-val (log-px)':>20}")
    lines.append("-" * 70)
    for _, r in summary.iterrows():
        normal_flag = "NON-NORMAL" if r["jb_pval"] < 0.05 else "normal"
        statio_flag = "non-stationary" if r["adf_pval"] > 0.05 else "stationary"
        lines.append(
            f"{r['sector']:<16} {r['jb_stat']:>10,.1f} {r['jb_pval']:>10.4f} "
            f"{r['ret_autocorr_lag1']:>10.4f} "
            f"{r['adf_pval']:>8.4f} ({statio_flag}, {normal_flag})"
        )

    # ── Flags / warnings ────────────────────────────────────────────────────
    lines.append("\n[5] FLAGS & WARNINGS\n")
    flags_found = False
    for _, r in summary.iterrows():
        sector_flags = []
        if r["coverage_pct"] < 85:
            sector_flags.append(f"Low coverage ({r['coverage_pct']:.1f}%)")
        if r["max_gap_days"] > 10:
            sector_flags.append(f"Large gap ({r['max_gap_days']} calendar days)")
        if r["n_missing_close"] > 0:
            sector_flags.append(f"{r['n_missing_close']} missing Close values")
        if r["n_zero_returns"] > 5:
            sector_flags.append(f"{r['n_zero_returns']} zero-return days")
        if r["n_outliers_4sd"] > 20:
            sector_flags.append(f"{r['n_outliers_4sd']} outliers beyond 4σ")
        if r["adf_pval"] > 0.05:
            sector_flags.append("Log-price appears non-stationary (expected for price levels)")

        if sector_flags:
            flags_found = True
            lines.append(f"  {r['sector']:<14}: " + " | ".join(sector_flags))

    if not flags_found:
        lines.append("  No critical flags detected.")

    lines.append("\n" + "=" * 70)

    report_text = "\n".join(lines)
    path = REPORT_DIR / "quality_report.txt"
    path.write_text(report_text)
    print(report_text)
    print(f"\n  Saved → results/quality_report/quality_report.txt\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = plt.cm.tab20.colors  # 20 distinct colours

def plot_price_levels(raw: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(
        len(raw), 1, figsize=(14, 2.2 * len(raw)), sharex=False
    )
    if len(raw) == 1:
        axes = [axes]

    for ax, (key, df), color in zip(axes, raw.items(), COLORS):
        close = df["Close"].dropna()
        ax.plot(close.index, np.log(close), color=color, linewidth=0.8)
        ax.set_ylabel("Log Price", fontsize=8)
        ax.set_title(f"{SECTORS[key][1]}  [{SECTORS[key][0]}]",
                     fontsize=9, loc="left", pad=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, linewidth=0.4)

    fig.suptitle("S&P 500 Sector Indexes — Log Price Levels", fontsize=12, y=1.002)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_price_levels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → 01_price_levels.png")


def plot_daily_returns(raw: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(
        len(raw), 1, figsize=(14, 2.2 * len(raw)), sharex=False
    )
    if len(raw) == 1:
        axes = [axes]

    for ax, (key, df), color in zip(axes, raw.items(), COLORS):
        ret = df["Close"].pct_change().dropna()
        ax.plot(ret.index, ret * 100, color=color, linewidth=0.4, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Ret (%)", fontsize=8)
        ax.set_title(f"{SECTORS[key][1]}", fontsize=9, loc="left", pad=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, linewidth=0.4)

    fig.suptitle("S&P 500 Sector Indexes — Daily Returns (%)", fontsize=12, y=1.002)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_daily_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → 02_daily_returns.png")


def plot_missing_heatmap(raw: dict[str, pd.DataFrame]):
    """
    Monthly heatmap: shade = fraction of trading days with data in that month.
    Useful for spotting structural data gaps.
    """
    # Build a union calendar from all available dates across all sectors
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*[df.index for df in raw.values()]))
    )
    monthly_idx = pd.period_range(all_dates.min(), all_dates.max(), freq="M")

    matrix = pd.DataFrame(index=[k for k in raw], columns=monthly_idx, dtype=float)
    for key, df in raw.items():
        close = df["Close"]
        for month in monthly_idx:
            mask = (close.index.year == month.year) & (close.index.month == month.month)
            expected = mask.sum()   # trading days Yahoo returned for this month
            if expected == 0:
                matrix.loc[key, month] = np.nan
            else:
                actual = close[mask].notna().sum()
                matrix.loc[key, month] = actual / expected

    # Downsample to yearly for readability
    yearly_cols = {}
    for month in monthly_idx:
        yr = str(month.year)
        yearly_cols.setdefault(yr, []).append(month)

    yr_matrix = pd.DataFrame(index=matrix.index, columns=sorted(yearly_cols.keys()), dtype=float)
    for yr, months in yearly_cols.items():
        vals = matrix[months].values.astype(float)
        yr_matrix[yr] = np.nanmean(vals, axis=1)

    fig, ax = plt.subplots(figsize=(max(14, len(yr_matrix.columns) * 0.55), 4))
    im = ax.imshow(yr_matrix.values.astype(float), aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(yr_matrix.index)))
    ax.set_yticklabels(yr_matrix.index, fontsize=8)
    ax.set_xticks(range(len(yr_matrix.columns)))
    ax.set_xticklabels(yr_matrix.columns, rotation=90, fontsize=7)
    plt.colorbar(im, ax=ax, label="Fraction of days with data")
    ax.set_title("Data Availability Heatmap (annual avg, green=full, red=missing)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_missing_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → 03_missing_heatmap.png")


def plot_return_distributions(raw: dict[str, pd.DataFrame]):
    ncols = 3
    nrows = int(np.ceil(len(raw) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, (key, df) in enumerate(raw.items()):
        ax = axes[i]
        ret = df["Close"].pct_change().dropna() * 100

        ax.hist(ret, bins=80, density=True, color=COLORS[i], alpha=0.6,
                edgecolor="none", label="Empirical")

        # Fit and overlay a normal distribution
        mu, sigma = ret.mean(), ret.std()
        x = np.linspace(ret.min(), ret.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "k--", linewidth=1.2, label="Normal fit")

        sk = ret.skew()
        ku = ret.kurtosis()
        ax.set_title(f"{key}\nskew={sk:.2f}  ex.kurt={ku:.2f}", fontsize=8)
        ax.set_xlabel("Daily Return (%)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3, linewidth=0.4)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("S&P 500 Sector Indexes — Daily Return Distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_return_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → 04_return_distributions.png")


def run_plots(raw: dict[str, pd.DataFrame]):
    print("=" * 65)
    print("GENERATING PLOTS")
    print("=" * 65)
    plot_price_levels(raw)
    plot_daily_returns(raw)
    plot_missing_heatmap(raw)
    plot_return_distributions(raw)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    raw     = download_all()
    summary = run_quality_checks(raw)
    write_text_report(summary)
    run_plots(raw)

    print("=" * 65)
    print("ALL DONE")
    print(f"  Raw CSVs    → code_data/data/raw/")
    print(f"  Summary CSV → code_data/results/quality_report/quality_summary.csv")
    print(f"  Text report → code_data/results/quality_report/quality_report.txt")
    print(f"  Plots       → code_data/results/quality_report/plots/")
    print("=" * 65)
