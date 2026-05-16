"""
main.py — full replication pipeline.

Replicates Christoffersen, Diebold, Mariano, Tay & Tse (2007)
"Direction-of-Change Forecasts Based on Conditional Variance,
 Skewness and Kurtosis Dynamics: International Evidence"
for USA using S&P 500 (^GSPC) as a proxy for the MSCI USA index.

Run from the code_3/ directory:
  python main.py
"""
import sys
import numpy as np
import pandas as pd

import data_loader as dl
import volatility  as vol_mod
import forecasting as fc_mod
import evaluation  as ev_mod
import plots, tables

from config import OOS_FRACTION, FREQUENCIES, FREQ_NAME, MARKET

SEP = "=" * 62


def main():
    print(SEP)
    print("EC581 — Direction-of-Change Replication")
    print(f"Market  : {MARKET}")
    print(f"OOS frac: {OOS_FRACTION:.4f}  ({OOS_FRACTION*100:.2f}%)")
    print(SEP)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1] Downloading and aggregating data ...")
    daily_ret   = dl.download()
    period_data = dl.build_all(daily_ret)

    for h in FREQUENCIES:
        df = period_data[h]
        print(f"  {FREQ_NAME[h]:12s}: {df.index[0].date()} → "
              f"{df.index[-1].date()}  ({len(df)} periods)")

    # ── 2. Train / test split ─────────────────────────────────────────────────
    print("\n[2] Computing train/test split ...")
    init_windows = {}
    for h in FREQUENCIES:
        n         = len(period_data[h])
        n_oos     = round(OOS_FRACTION * n)
        init_w    = n - n_oos
        init_windows[h] = init_w
        oos_start = period_data[h].index[init_w].date()
        oos_end   = period_data[h].index[-1].date()
        print(f"  {FREQ_NAME[h]:12s}: N={n}, in-sample={init_w}, "
              f"OOS={n_oos}  ({oos_start} → {oos_end})")

    # ── 3. Recursive volatility forecasts ────────────────────────────────────
    print("\n[3] Recursive ARMA volatility forecasting ...")
    vol_forecasts = {}
    for h in FREQUENCIES:
        print(f"\n  [{FREQ_NAME[h]}]")
        vol_forecasts[h] = vol_mod.recursive_forecast(
            period_data[h]["LogRealVol"], init_windows[h]
        )

    # ── 4. Direction-of-change probability forecasts ──────────────────────────
    print("\n[4] Generating probability forecasts ...")
    forecast_results = {}
    for h in FREQUENCIES:
        print(f"  {FREQ_NAME[h]} ...")
        forecast_results[h] = fc_mod.generate(
            period_df   = period_data[h],
            vol_fc      = vol_forecasts[h],
            init_window = init_windows[h],
        )

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    print("\n[5] Computing Brier scores ...")
    eval_results = {}
    for h in FREQUENCIES:
        eval_results[h] = ev_mod.evaluate(forecast_results[h])
        ev = eval_results[h]["full"]
        b_base = ev["Baseline"]["brier_abs"]
        b_np   = ev["Nonpar_AIC"]["brier_abs"]
        b_ext  = ev["Extended_AIC"]["brier_abs"]
        print(f"  {FREQ_NAME[h]:12s}  Brier(Abs):"
              f"  Baseline={b_base:.4f}"
              f"  Nonpar={b_np:.4f} (×{b_np/b_base:.3f})"
              f"  Extended={b_ext:.4f} (×{b_ext/b_base:.3f})")

    # ── 6. Tables ─────────────────────────────────────────────────────────────
    print("\n[6] Producing tables ...")
    t1a, t1b = tables.table1(period_data)
    t2       = tables.table2(period_data, vol_forecasts)
    t3       = tables.table3(eval_results)
    t4       = tables.table4(eval_results)

    # Replication diagnostics (paper Sec. 2 & 3)
    t_mr   = tables.table_mean_regression(period_data, init_windows)
    t_acf  = tables.table_volatility_acf(period_data)
    t_oob  = tables.table_extended_oob(forecast_results)

    # Print key tables to console
    print("\n  — Table 1a: Return Statistics —")
    print(t1a.to_string())
    print("\n  — Table 2: MSPE / Variance Ratio —")
    print(t2.to_string())
    print("\n  — Table 4a: Relative Brier (Full Sample) —")
    print(t4["full"].to_string())
    print("\n  — Table 4b: Relative Brier (Low Volatility) —")
    print(t4["low"].to_string())

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print("\n  — Mean Regression Diagnostic (starting sample) —")
    print("    R_t = β₀ + β₁·log(σ_t) + β₂·[log(σ_t)]²")
    print(t_mr.to_string())
    sig = (t_mr["p(β2)"] < 0.05).sum()
    print(f"    → β₂ significant (p<0.05) at {sig}/{len(t_mr)} frequencies "
          f"(paper: 'almost all series').")

    print("\n  — Log Realized Volatility ACF / Ljung-Box —")
    print(t_acf.to_string())
    print(f"    → ρ(1) declines monthly→quarterly: "
          f"{t_acf['ρ(1)'].iloc[0]:.3f} → {t_acf['ρ(1)'].iloc[-1]:.3f}; "
          f"all LB(10) p-values: "
          f"{', '.join(f'{p:.4f}' for p in t_acf['LB(10) p'])}.")

    print("\n  — Extended Model: p̂ out-of-[0,1] count —")
    print(t_oob.to_string())
    total_oob = int(t_oob["p<0"].sum() + t_oob["p>1"].sum())
    if total_oob == 0:
        print("    → All Extended forecasts in [0,1] (matches paper).")
    else:
        print(f"    → {total_oob} forecasts outside [0,1] — "
              f"differs from paper's USA result.")

    # ── 7. Figures ────────────────────────────────────────────────────────────
    print("\n[7] Producing figures ...")
    plots.fig1_realized_volatility(period_data)
    plots.fig2_vol_forecasts(period_data, vol_forecasts)
    plots.fig3_predicted_probabilities(forecast_results)
    plots.fig4(forecast_results, "Pr_Nonpar_AIC",   "Nonparametric", "4a")
    plots.fig4(forecast_results, "Pr_Extended_AIC", "Extended",      "4b")

    print(f"\n{SEP}")
    print("DONE")
    print(f"  Figures → code_3/results/figures/  (5 files)")
    print(f"  Tables  → code_3/results/tables/   (8 files)")
    print(SEP)


if __name__ == "__main__":
    main()
