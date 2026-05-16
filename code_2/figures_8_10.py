"""
Christoffersen & Diebold (2006) — Figures 8–10
Empirical S&P 500 application (Section 6).

Methodology:
  - Volatility: RiskMetrics EWMA  σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1},  λ=0.94
  - Sign model: logit regression  I_{t+h} = F(α + β·(1/σ_t)) + ε
    estimated with 5-year rolling window (1 250 trading days)
  - Daily out-of-sample forecasts at horizons h = 1 … 250 days
  - Refit frequency is configurable via CD_UPD_FREQ. Default is 21 trading days
    because it matches the paper's rough model count; CD_UPD_FREQ=1 refits daily.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

OUTDIR  = os.path.join(os.path.dirname(__file__), "results")
DATADIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

def _int_env(name, default, min_value=1):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return value


def _bool_env(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


LAMBDA_RM = 0.94    # RiskMetrics smoothing
TRAIN_WIN = 1_250   # 5-year rolling window (250 × 5)
UPD_FREQ  = _int_env("CD_UPD_FREQ", 21)  # 1 = daily refit; 21 ~= monthly refit
FIT_INTERCEPT = _bool_env("CD_LOGIT_INTERCEPT", False)
FIG10_HORIZONS = range(1, 251)
VALID_ALIGNMENTS = {"strict", "paper_like"}
RETURN_TARGET = os.environ.get("CD_RETURN_TARGET", "compound").strip().lower()
FIG10_CORR_SAMPLING = os.environ.get("CD_FIG10_CORR", "overlap").strip().lower()
FIG10_NONOVERLAP_OFFSET = os.environ.get(
    "CD_FIG10_NONOVERLAP_OFFSET", "sample_start"
).strip().lower()
VALID_RETURN_TARGETS = {"compound", "sum_simple"}
VALID_FIG10_CORR_SAMPLING = {"overlap", "nonoverlap"}
VALID_NONOVERLAP_OFFSETS = {"sample_start", "first_valid"}
if RETURN_TARGET not in VALID_RETURN_TARGETS:
    raise ValueError(f"CD_RETURN_TARGET must be one of {sorted(VALID_RETURN_TARGETS)}")
if FIG10_CORR_SAMPLING not in VALID_FIG10_CORR_SAMPLING:
    raise ValueError(
        f"CD_FIG10_CORR must be one of {sorted(VALID_FIG10_CORR_SAMPLING)}"
    )
if FIG10_NONOVERLAP_OFFSET not in VALID_NONOVERLAP_OFFSETS:
    raise ValueError(
        "CD_FIG10_NONOVERLAP_OFFSET must be one of "
        f"{sorted(VALID_NONOVERLAP_OFFSETS)}"
    )


# ── Data ────────────────────────────────────────────────────────────────────
def load_sp500(start='1963-01-01', end='2004-01-01',
               min_last_date='2003-12-31', force_download=False):
    """Download S&P 500 (^GSPC) from Yahoo Finance, cache locally.

    Yahoo's ``end`` parameter is exclusive, so ``2004-01-01`` is used to include
    observations through 2003-12-31 when available.
    """
    external_csv = os.environ.get("CD_SP500_CSV")
    if external_csv:
        df = pd.read_csv(external_csv, parse_dates=True)
        date_col = next((c for c in df.columns if c.lower() in {"date", "caldt"}), None)
        close_col = next(
            (c for c in df.columns if c.lower() in {"close", "spindx", "price", "level"}),
            None,
        )
        if date_col is None or close_col is None:
            raise ValueError(
                "CD_SP500_CSV must contain a date column named Date/CALDT and "
                "a price column named Close/SPINDX/Price/Level"
            )
        df = df[[date_col, close_col]].dropna()
        df.columns = ["Date", "Close"]
        df = df.set_index("Date").sort_index()
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
        if df.empty:
            raise ValueError("CD_SP500_CSV has no rows in the requested date range")
        print(f"  Loaded {len(df)} rows from {external_csv}.")
        return df

    cache = os.path.join(DATADIR, "sp500_1963_2003.csv")
    if os.path.exists(cache) and not force_download:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if df.index.max() >= pd.Timestamp(min_last_date):
            print(f"  Loaded {len(df)} rows from cache.")
            return df
        print(
            f"  Cache ends at {df.index.max().date()}, before {min_last_date}; "
            "trying to refresh from Yahoo Finance."
        )
    import yfinance as yf
    print("  Downloading ^GSPC from Yahoo Finance …")
    try:
        df = yf.download('^GSPC', start=start, end=end,
                         auto_adjust=True, progress=False)[['Close']]
        df.columns = ['Close']
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("empty Yahoo Finance response")
        df.to_csv(cache)
        print(f"  Downloaded {len(df)} trading days.")
        return df
    except Exception as exc:
        if os.path.exists(cache):
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            print(f"  Yahoo refresh failed ({exc}); using cached {len(df)} rows.")
            return df
        raise


# ── RiskMetrics EWMA volatility ──────────────────────────────────────────────
def riskmetrics(log_returns, lam=LAMBDA_RM):
    """
    Annualised standard deviation from RiskMetrics EWMA.
      σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}
    Returns array of length len(log_returns).
    """
    n   = len(log_returns)
    var = np.zeros(n)
    var[0] = log_returns[0]**2
    for t in range(1, n):
        var[t] = lam * var[t-1] + (1.0 - lam) * log_returns[t-1]**2
    return np.sqrt(var * 250)   # annualise


# ── Rolling logit sign model ─────────────────────────────────────────────────
def rolling_logit(X, y, h=1, train_win=TRAIN_WIN, upd_freq=UPD_FREQ,
                  alignment="strict", fit_intercept=FIT_INTERCEPT):
    """
    Out-of-sample logit forecasts with rolling 5-year estimation window.
    X : (N,1) predictor = 1/σ_t
    y : (N,)  binary label I(R_{t:t+h} > 0)
    h : forecast horizon (days)

    alignment:
      - "strict": real-time OOS alignment. At forecast origin T, training uses
        the previous 5-year calendar/trading window, after dropping labels
        whose h-day return has not yet been realized.
      - "paper_like": common rolling-window replication alignment. The window
        ends at T-1, which matches the paper's Figure 10 behavior much more
        closely, but for h > 1 it is not a real-time trading forecast because
        some training labels finish after T.

    Returns P_oos (N,) with NaN where no forecast available.
    """
    if alignment not in VALID_ALIGNMENTS:
        raise ValueError(f"alignment must be one of {sorted(VALID_ALIGNMENTS)}")

    n   = len(y)
    P   = np.full(n, np.nan)
    mdl = LogisticRegression(
        fit_intercept=fit_intercept,
        penalty=None,
        max_iter=1_000,
        solver='lbfgs',
    )
    last_fit = -upd_freq

    start_T = train_win

    for T in range(start_T, n):
        if (T - last_fit) >= upd_freq:
            train_start = T - train_win
            if alignment == "strict":
                # At forecast origin T, y[j] is known iff j + h <= T. Keep the
                # 5-year rolling origin window but drop labels not yet realized.
                train_end = T - h + 1
            else:
                # Paper-like pseudo-OOS window used for replication diagnostics.
                train_end = T
            if train_start < 0:
                continue
            X_tr = X[train_start : train_end]
            y_tr = y[train_start : train_end]
            if len(np.unique(y_tr)) < 2:
                continue
            mdl.fit(X_tr, y_tr)
            last_fit = T
        try:
            P[T] = mdl.predict_proba(X[T : T+1])[0, 1]
        except Exception:
            pass
    return P


# ── Prepare features & labels for horizon h ──────────────────────────────────
def prepare_horizon(log_prices, sig_ann, h, simple_returns=None,
                    return_target=RETURN_TARGET):
    """
    Returns (X, y, dates_y):
      X   — predictor 1/σ_t,  shape (N-h, 1)
      y   — I(R_{t+1:t+h}>0), shape (N-h,)
    """
    if return_target == "compound":
        lr_h = log_prices[h:] - log_prices[:-h]
        y = (lr_h > 0).astype(int)
    elif return_target == "sum_simple":
        if simple_returns is None:
            raise ValueError("simple_returns is required for CD_RETURN_TARGET=sum_simple")
        csum = np.r_[0.0, np.cumsum(simple_returns)]
        # Forecast origin t is the close at log_prices[t], corresponding to
        # simple_returns[t + 1] as the next one-day return.
        r_h = csum[1 + h:] - csum[1:-h]
        y = (r_h > 0).astype(int)
    else:
        raise ValueError(f"return_target must be one of {sorted(VALID_RETURN_TARGETS)}")
    X = (1.0 / np.maximum(sig_ann[:len(y)], 1e-8)).reshape(-1, 1)
    return X, y


def _alignment_label(alignment):
    if alignment == "strict":
        return "Strict real-time OOS"
    if alignment == "paper_like":
        return "Paper-like rolling window"
    raise ValueError(alignment)


def _refit_label(upd_freq=UPD_FREQ):
    if upd_freq == 1:
        return "daily refit"
    if upd_freq == 21:
        return "21-day refit"
    return f"{upd_freq}-day refit"


def _logit_label(fit_intercept=FIT_INTERCEPT):
    return "with intercept" if fit_intercept else "no intercept"


def _logit_formula(fit_intercept=FIT_INTERCEPT):
    return "α + β·(1/σ_t)" if fit_intercept else "β·(1/σ_t)"


def _figure_suffix(alignment, upd_freq=UPD_FREQ):
    parts = []
    if alignment != "strict":
        parts.append(alignment)
    if FIT_INTERCEPT:
        parts.append("intercept")
    if RETURN_TARGET != "compound":
        parts.append(RETURN_TARGET)
    if FIG10_CORR_SAMPLING != "overlap":
        parts.append(FIG10_CORR_SAMPLING)
    if upd_freq != 21:
        parts.append("daily" if upd_freq == 1 else f"refit{upd_freq}")
    return "" if not parts else "_" + "_".join(parts)


def _selected_figures():
    raw = os.environ.get("CD_FIGURES", "8,9,10")
    selected = {item.strip() for item in raw.split(",") if item.strip()}
    allowed = {"8", "9", "10"}
    unknown = selected - allowed
    if unknown:
        raise ValueError(f"CD_FIGURES contains unknown figures: {sorted(unknown)}")
    return selected


def _return_target_label():
    if RETURN_TARGET == "compound":
        return "compound h-day return"
    if RETURN_TARGET == "sum_simple":
        return "sum of daily simple returns"
    raise ValueError(RETURN_TARGET)


def _fig10_corr_label():
    if FIG10_CORR_SAMPLING == "overlap":
        return "overlapping daily origins"
    return f"non-overlapping horizons ({FIG10_NONOVERLAP_OFFSET})"


def _forecast_corr(P, y, h):
    valid = ~np.isnan(P)
    if valid.sum() < 30:
        return np.nan, 0
    if FIG10_CORR_SAMPLING == "overlap":
        idx = np.where(valid)[0]
    else:
        if FIG10_NONOVERLAP_OFFSET == "first_valid":
            offset = np.where(valid)[0][0] % h
        else:
            offset = 0
        idx = np.arange(offset, len(y), h)
        idx = idx[valid[idx]]
    if len(idx) < 20:
        return np.nan, len(idx)
    P_i = P[idx]
    y_i = y[idx].astype(float)
    if P_i.std() < 1e-8 or y_i.std() < 1e-8:
        return np.nan, len(idx)
    return np.corrcoef(P_i, y_i)[0, 1], len(idx)


# ── Figure 8 ─────────────────────────────────────────────────────────────────
def figure8(df):
    """Daily annualised RiskMetrics volatility, 1963–2003."""
    lr  = np.log(df['Close'].values)
    ret = np.diff(lr)
    sig = riskmetrics(ret)          # length = len(ret)
    dates = df.index[1:]            # shift by 1 for diff

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, sig * 100, 'steelblue', lw=0.7)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Volatility (% per year)', fontsize=12)
    ax.set_title(
        'Figure 8  —  Daily Annualised RiskMetrics Volatility (λ=0.94)  '
        '—  S&P 500 1963–2003', fontsize=11)
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = f'{OUTDIR}/figure8.png'
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Figure 9 ─────────────────────────────────────────────────────────────────
def figure9(df, alignment="strict"):
    """Conditional sign-probability forecasts at 6 horizons — S&P 500."""
    print(f"  Figure 9: rolling logit for 6 horizons ({alignment}) …")
    lr   = np.log(df['Close'].values)
    ret  = np.diff(lr)
    simple_ret = np.diff(df['Close'].values) / df['Close'].values[:-1]
    sig  = riskmetrics(ret)
    # log_prices aligned with sig: sig[t] is known at day t
    lp   = lr[1:]          # drop first so len(lp) == len(sig)

    H6     = [1, 5, 21, 63, 126, 250]
    LABELS = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semiannual', 'Annual']

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for ax, h, lab in zip(axes.flat, H6, LABELS):
        X, y = prepare_horizon(lp, sig, h, simple_returns=simple_ret)
        P    = rolling_logit(
            X, y, h=h, upd_freq=UPD_FREQ, alignment=alignment,
            fit_intercept=FIT_INTERCEPT
        )
        dates = df.index[1:-h]              # forecast-origin dates

        valid = ~np.isnan(P)
        pbar = y[valid].mean() if valid.any() else np.nan
        ax.plot(dates, P, 'steelblue', lw=0.7, alpha=0.9)
        ax.axhline(pbar, color='k', lw=1.1, ls='--', label=f'Unconditional = {pbar:.2f}')
        ax.set_title(f'{lab}  (h={h})', fontsize=11)
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Pr(R > 0)', fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        print(f"    h={h:3d}: uncond = {pbar:.3f}  (valid = {valid.sum()})")

    fig.suptitle(
        'Figure 9  —  Conditional Sign-Probability Forecasts  —  S&P 500 1963–2003\n'
        f'(Logit model: logit(P) = {_logit_formula()},  5-year rolling window; '
        f'{_alignment_label(alignment)}; {_refit_label()}; {_logit_label()}; '
        f'{_return_target_label()})',
        fontsize=11, y=1.01)
    plt.tight_layout()
    path = f'{OUTDIR}/figure9{_figure_suffix(alignment)}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ── Figure 10 ────────────────────────────────────────────────────────────────
def figure10(df, alignment="strict"):
    """
    Figure 10: forecast-realisation correlation and sign autocorrelation
    vs horizon h — S&P 500 empirical analogue of Figure 6.

    Sign autocorrelation: lag-1 autocorr of adjacent h-day sign sequences.
    Forecast correlation: Corr(P_oos, I) over the out-of-sample period.
    """
    print(f"  Figure 10: computing for horizons 1–250 ({alignment}) …")
    lr   = np.log(df['Close'].values)
    ret  = np.diff(lr)
    simple_ret = np.diff(df['Close'].values) / df['Close'].values[:-1]
    sig  = riskmetrics(ret)
    lp   = lr[1:]

    horizons   = list(FIG10_HORIZONS)
    corr_vals  = []
    acorr_vals = []
    selected   = {}

    for h in horizons:
        X, y = prepare_horizon(lp, sig, h, simple_returns=simple_ret)
        P    = rolling_logit(
            X, y, h=h, upd_freq=UPD_FREQ, alignment=alignment,
            fit_intercept=FIT_INTERCEPT
        )

        # ── Forecast-realisation correlation ──
        corr, n_corr = _forecast_corr(P, y, h)
        corr_vals.append(corr)
        if h in (1, 5, 21, 63, 126, 250):
            selected[h] = (corr, n_corr)

        # ── Lag-1 sign autocorrelation (adjacent h-day returns) ──
        # I1[t] = sign(r_{t:t+h}),  I2[t] = sign(r_{t+h:t+2h})
        # Both defined for t = 0 … N-2h-1, giving N-2h pairs.
        n  = len(lp)
        if n < 2 * h + 10:
            acorr_vals.append(np.nan)
        else:
            I1 = (lp[h   : n - h] - lp[: n - 2*h] > 0).astype(float)
            I2 = (lp[2*h :]       - lp[h : n - h]  > 0).astype(float)
            if I1.std() < 1e-8 or I2.std() < 1e-8:
                acorr_vals.append(np.nan)
            else:
                acorr_vals.append(np.corrcoef(I1, I2)[0, 1])

    horizons   = np.array(horizons)
    corr_vals  = np.array(corr_vals)
    acorr_vals = np.array(acorr_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, corr_vals,  'steelblue', lw=2, label='Forecast–Realisation Correlation')
    ax.plot(horizons, acorr_vals, 'tomato',    lw=2, ls='--', label='Sign Autocorrelation (lag-1)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Horizon (trading days)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title(
        'Figure 10  —  Forecast Correlation & Sign Autocorrelation  '
        f'—  S&P 500 1963–2003\n'
        f'{_alignment_label(alignment)}; {_refit_label()}; {_logit_label()}; '
        f'{_return_target_label()}; corr: {_fig10_corr_label()}',
        fontsize=11)
    ax.legend(fontsize=11)
    ax.set_xlim(1, 250)
    plt.tight_layout()
    path = f'{OUTDIR}/figure10{_figure_suffix(alignment)}.png'
    fig.savefig(path, dpi=150)
    plt.close()
    if selected:
        parts = [
            f"h={h}: {selected[h][0]:.3f} (n={selected[h][1]})"
            for h in sorted(selected)
        ]
        print("    selected corr(P, y): " + ", ".join(parts))
    print(f"  Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating Figures 8–10  (empirical S&P 500) …")
    alignment = os.environ.get("CD_ROLLING_ALIGNMENT", "strict").strip().lower()
    if alignment not in VALID_ALIGNMENTS | {"both"}:
        raise ValueError(
            "CD_ROLLING_ALIGNMENT must be 'strict', 'paper_like', or 'both'"
        )
    selected_figures = _selected_figures()
    if UPD_FREQ == 1 and "10" in selected_figures:
        print("  Note: daily refit for Figure 10 fits roughly 2.25M logit models.")
    df = load_sp500()
    if "8" in selected_figures:
        figure8(df)
    alignments = ["strict", "paper_like"] if alignment == "both" else [alignment]
    for item in alignments:
        if "9" in selected_figures:
            figure9(df, alignment=item)
        if "10" in selected_figures:
            figure10(df, alignment=item)
    print("Done.")
