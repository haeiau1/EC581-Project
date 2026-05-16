"""
tables.py — replicate the paper's tables for USA (S&P 500).

Table 1a : Summary statistics of returns          (mean, std, skew, kurt, JB)
Table 1b : Summary statistics of log realized vol (same structure)
Table 2  : MSPE / sample-variance ratio for vol forecasts
Table 3  : Brier(Abs) mean & std by vol subperiod  (Low / Medium / High)
Table 4a : Relative Brier scores — full OOS sample
Table 4b : Relative Brier scores — low volatility subperiod
Table 4c : Relative Brier scores — medium volatility subperiod
Table 4d : Relative Brier scores — high volatility subperiod

Additional diagnostics (paper Sec. 2 & 3, S&P 500 replication):
Table MR  : Mean regression R_t = β₀+β₁·log σ_t+β₂·(log σ_t)² in starting
            estimation sample. Paper: "the quadratic term is significant
            for almost all series in the starting estimation sample."
Table ACF : ρ(k) and Ljung-Box p-values for log realized vol. Paper:
            "autocorrelations diminish [monthly→quarterly] but still
            indicate predictability."
Table OOB : Extended-model p̂ out-of-[0,1] counts. Paper: "this was
            inconsequential as all our predicted probabilities turn out
            to lie between 0 and 1."

In Tables 4*, 'Baseline' shows the actual Brier score; 'Nonpar' and 'Extended'
show ratios relative to the Baseline (< 1 = improvement, bolded in paper).
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from config import TAB_DIR, FREQ_LABEL


# ── Table 1 ───────────────────────────────────────────────────────────────────

def _desc(series: np.ndarray) -> dict:
    """
    Paper Tablo 1: reported 'Kurtosis' is the RAW (Pearson) kurtosis,
    so a normal series has kurtosis = 3.  scipy.stats.kurtosis returns
    excess kurtosis by default; we add 3 to align with the paper's column.
    """
    jb_stat, jb_p = stats.jarque_bera(series)
    return {
        "Mean":     round(float(series.mean()), 4),
        "Std.Dev":  round(float(series.std(ddof=1)), 4),
        "Skewness": round(float(stats.skew(series)), 4),
        "Kurtosis": round(float(stats.kurtosis(series)) + 3, 4),
        "JB p-val": round(float(jb_p), 4),
    }


def table1(period_data: dict) -> tuple:
    """Table 1a (returns) and Table 1b (log realized vol)."""
    rows_ret, rows_vol = [], []
    for h in [1, 2, 3]:
        df = period_data[h]
        rows_ret.append({"Freq": FREQ_LABEL[h], **_desc(df["Return"].values)})
        rows_vol.append({"Freq": FREQ_LABEL[h], **_desc(df["LogRealVol"].values)})

    t1a = pd.DataFrame(rows_ret).set_index("Freq")
    t1b = pd.DataFrame(rows_vol).set_index("Freq")

    t1a.to_csv(TAB_DIR / "table1a_return_stats.csv")
    t1b.to_csv(TAB_DIR / "table1b_vol_stats.csv")
    print("  Saved → table1a_return_stats.csv")
    print("  Saved → table1b_vol_stats.csv")
    return t1a, t1b


# ── Table 2 ───────────────────────────────────────────────────────────────────

def table2(period_data: dict, vol_forecasts: dict) -> pd.DataFrame:
    """
    MSPE of log-vol forecasts divided by the sample variance of
    actual log-vol over the OOS period.
    """
    rows = []
    for h in [1, 2, 3]:
        lv_actual = period_data[h]["LogRealVol"]
        fc        = vol_forecasts[h]
        act_oos   = lv_actual.loc[fc.index].values
        var_act   = float(np.nanvar(act_oos, ddof=1))

        row = {"Freq": FREQ_LABEL[h]}
        for crit in ["AIC", "SIC"]:
            fcast = fc[f"LogVolFcast_{crit}"].values
            mspe  = float(np.nanmean((fcast - act_oos) ** 2))
            row[f"{crit} Forecast"] = round(mspe / var_act, 4)
        rows.append(row)

    t2 = pd.DataFrame(rows).set_index("Freq")
    t2.to_csv(TAB_DIR / "table2_mspe_ratio.csv")
    print("  Saved → table2_mspe_ratio.csv")
    return t2


# ── Table 3 ───────────────────────────────────────────────────────────────────

def table3(eval_results: dict) -> pd.DataFrame:
    """
    Brier(Abs) mean and std-dev for Baseline, Nonpar_AIC, Extended_AIC
    broken out by vol subperiod (Low / Medium / High).
    """
    rows = []
    for h in [1, 2, 3]:
        ev = eval_results[h]
        for period in ["low", "medium", "high"]:
            row = {"Freq": FREQ_LABEL[h], "Subperiod": period.capitalize()}
            for name in ["Baseline", "Nonpar_AIC", "Extended_AIC"]:
                d = ev.get(period, {}).get(name, {})
                row[f"{name}_mean"] = round(d.get("mean_abs", np.nan), 4)
                row[f"{name}_std"]  = round(d.get("std_abs",  np.nan), 4)
            rows.append(row)

    t3 = pd.DataFrame(rows)
    t3.to_csv(TAB_DIR / "table3_brier_subperiod.csv", index=False)
    print("  Saved → table3_brier_subperiod.csv")
    return t3


# ── Tables 4a-4d ──────────────────────────────────────────────────────────────

def _rel_brier(eval_results: dict, period: str) -> pd.DataFrame:
    """
    Build one relative-Brier table for a given period (full/low/medium/high).

    Columns:
      Brier(Abs): Bsln (actual score), Npar/Ext (ratio to Bsln)
      Brier(Sq):  same structure
    """
    rows = []
    for h in [1, 2, 3]:
        ev = eval_results[h].get(period, {})

        bsln_abs = ev.get("Baseline", {}).get("brier_abs", np.nan)
        bsln_sq  = ev.get("Baseline", {}).get("brier_sq",  np.nan)

        def _ratio_abs(model):
            v = ev.get(model, {}).get("brier_abs", np.nan)
            return round(v / bsln_abs, 4) if not np.isnan(bsln_abs) else np.nan

        def _ratio_sq(model):
            v = ev.get(model, {}).get("brier_sq", np.nan)
            return round(v / bsln_sq, 4) if not np.isnan(bsln_sq) else np.nan

        rows.append({
            "Freq":        FREQ_LABEL[h],
            # Brier(Abs)
            "Bsln_Abs":    round(bsln_abs, 4),
            "Npar_Abs":    _ratio_abs("Nonpar_AIC"),
            "Ext_Abs":     _ratio_abs("Extended_AIC"),
            # Brier(Sq)
            "Bsln_Sq":     round(bsln_sq, 4),
            "Npar_Sq":     _ratio_sq("Nonpar_AIC"),
            "Ext_Sq":      _ratio_sq("Extended_AIC"),
        })

    return pd.DataFrame(rows).set_index("Freq")


def table4(eval_results: dict) -> dict:
    """Write Tables 4a–4d and return them in a dict."""
    mapping = [
        ("full",   "table4a_full_sample.csv"),
        ("low",    "table4b_low_vol.csv"),
        ("medium", "table4c_medium_vol.csv"),
        ("high",   "table4d_high_vol.csv"),
    ]
    tables = {}
    for period, fname in mapping:
        t = _rel_brier(eval_results, period)
        t.to_csv(TAB_DIR / fname)
        print(f"  Saved → {fname}")
        tables[period] = t
    return tables


# ── Mean-regression diagnostic ────────────────────────────────────────────────

def table_mean_regression(period_data: dict, init_windows: dict) -> pd.DataFrame:
    """
    In-sample (starting estimation sample, 1980:01–1993:12 in paper, here the
    matching window) OLS:
        R_t = β₀ + β₁·log(σ_t) + β₂·[log(σ_t)]²
    Paper Sec. 3 states the quadratic term is significant for almost all
    series in the starting sample — we test this for the S&P 500.
    """
    rows = []
    for h in [1, 2, 3]:
        df = period_data[h].iloc[:init_windows[h]]
        lv = df["LogRealVol"].values
        X  = sm.add_constant(np.column_stack([lv, lv ** 2]))
        y  = df["Return"].values
        res = sm.OLS(y, X).fit()
        rows.append({
            "Freq":   FREQ_LABEL[h],
            "N":      int(res.nobs),
            "β0":     round(float(res.params[0]), 5),
            "p(β0)":  round(float(res.pvalues[0]), 4),
            "β1":     round(float(res.params[1]), 5),
            "p(β1)":  round(float(res.pvalues[1]), 4),
            "β2":     round(float(res.params[2]), 5),
            "p(β2)":  round(float(res.pvalues[2]), 4),
            "R²":     round(float(res.rsquared), 4),
            "F p":    round(float(res.f_pvalue), 4),
        })
    t = pd.DataFrame(rows).set_index("Freq")
    t.to_csv(TAB_DIR / "table_mean_regression.csv")
    print("  Saved → table_mean_regression.csv")
    return t


# ── Volatility ACF / Ljung-Box diagnostic ─────────────────────────────────────

def table_volatility_acf(period_data: dict) -> pd.DataFrame:
    """
    Log realized volatility persistence: autocorrelations ρ(k) at k=1,5,10
    and Ljung-Box test p-values for the same lags.  Paper Sec. 2:
        "As we move from the monthly frequency to the quarterly frequency,
         the autocorrelations diminish, but still indicate predictability."
    """
    rows = []
    for h in [1, 2, 3]:
        lv = period_data[h]["LogRealVol"].dropna().values
        rho = {k: float(np.corrcoef(lv[:-k], lv[k:])[0, 1]) for k in (1, 5, 10)}
        lb  = acorr_ljungbox(lv, lags=[1, 5, 10], return_df=True)
        rows.append({
            "Freq":     FREQ_LABEL[h],
            "N":        int(len(lv)),
            "ρ(1)":     round(rho[1],  4),
            "ρ(5)":     round(rho[5],  4),
            "ρ(10)":    round(rho[10], 4),
            "LB(1) p":  round(float(lb["lb_pvalue"].iloc[0]), 4),
            "LB(5) p":  round(float(lb["lb_pvalue"].iloc[1]), 4),
            "LB(10) p": round(float(lb["lb_pvalue"].iloc[2]), 4),
        })
    t = pd.DataFrame(rows).set_index("Freq")
    t.to_csv(TAB_DIR / "table_volatility_acf.csv")
    print("  Saved → table_volatility_acf.csv")
    return t


# ── Extended-model out-of-[0,1] diagnostic ────────────────────────────────────

def table_extended_oob(forecast_results: dict) -> pd.DataFrame:
    """
    Number of OOS observations for which the Extended-model raw probability
    p̂ = 1 - Φ(-μ̂·x̂)·(β̂₀ + β̂₁·x̂) falls outside [0, 1] (i.e., where the
    clip [1e-8, 1-1e-8] in forecasting.py was actually triggered).
    Paper states this never happened in their international sample.
    """
    rows = []
    for h in [1, 2, 3]:
        oob = forecast_results[h].attrs.get("ext_oob", {})
        for crit in ("aic", "sic"):
            tot = oob.get(f"{crit}_total", 0)
            blw = oob.get(f"{crit}_below", 0)
            abv = oob.get(f"{crit}_above", 0)
            pct = (blw + abv) / tot * 100 if tot else 0.0
            rows.append({
                "Freq":     FREQ_LABEL[h],
                "Criterion": crit.upper(),
                "OOS_N":    tot,
                "p<0":      blw,
                "p>1":      abv,
                "OOB %":    round(pct, 3),
            })
    t = pd.DataFrame(rows).set_index(["Freq", "Criterion"])
    t.to_csv(TAB_DIR / "table_extended_oob.csv")
    print("  Saved → table_extended_oob.csv")
    return t
