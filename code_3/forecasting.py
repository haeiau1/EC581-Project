"""
forecasting.py — three direction-of-change probability forecast models.

All three models are estimated recursively: at each out-of-sample step k
all parameters are re-estimated using the expanding window [0 : k].

MODEL 1 — Baseline (Eq 3)
  p_t = (1/k) Σ_{s<k} I(R_s > 0)
  Unconditional empirical fraction of positive returns.

MODEL 2 — Nonparametric (Eq 5)
  Step 1 — Mean regression (Eq 4):
    R_t = β₀ + β₁·log(σ_t) + β₂·[log(σ_t)]²    (actual σ_t in-sample)
  Step 2 — Standardised residuals:
    z_t = (R_t − μ̂_t) / σ_t
  Step 3 — Empirical CDF forecast:
    p_{k+1} = 1 − F̂(−μ̂_{k+1|k} / σ̂_{k+1|k})

MODEL 3 — Extended / Gram-Charlier (Eq 8)
  Uses same mean regression as Model 2, then:
  Step 3 — OLS without intercept (β reused with new meaning):
    (1 − I(R_t > 0)) = β₀·Φ(−μ̂_t·x_t) + β₁·Φ(−μ̂_t·x_t)·x_t
    where x_t = 1/σ_t
  Step 4 — Forecast:
    p_{k+1} = 1 − Φ(−μ̂_{k+1|k}·x̂_{k+1}) · (β̂₀ + β̂₁·x̂_{k+1})
    where x̂_{k+1} = 1/σ̂_{k+1|k}
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


# ── OLS helpers ────────────────────────────────────────────────────────────────

def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def _mean_regression(ret: np.ndarray, log_vol: np.ndarray) -> np.ndarray:
    """
    OLS: R_t = β₀ + β₁·log(σ_t) + β₂·[log(σ_t)]²
    Returns [β₀, β₁, β₂].
    """
    X = np.column_stack([np.ones(len(log_vol)), log_vol, log_vol ** 2])
    return _ols(X, ret)


def _eval_mu(lv: float, c: np.ndarray) -> float:
    """μ̂ = c[0] + c[1]·lv + c[2]·lv²"""
    return c[0] + c[1] * lv + c[2] * lv ** 2


# ── Main forecast function ─────────────────────────────────────────────────────

def generate(period_df: pd.DataFrame,
             vol_fc: pd.DataFrame,
             init_window: int) -> pd.DataFrame:
    """
    Generate out-of-sample probability forecasts for all three models.

    Parameters
    ----------
    period_df   : full-sample DataFrame with Return, RealVol, LogRealVol, Positive
    vol_fc      : out-of-sample ARMA forecasts (LogVolFcast_AIC, LogVolFcast_SIC)
    init_window : number of periods in the initial in-sample window

    Returns
    -------
    DataFrame indexed by vol_fc.index with columns:
      Pr_Baseline, Pr_Nonpar_AIC, Pr_Nonpar_SIC,
      Pr_Extended_AIC, Pr_Extended_SIC,
      Return, Positive, RealVol
    """
    R   = period_df["Return"].values
    LV  = period_df["LogRealVol"].values   # log σ_t (actual)
    SV  = period_df["RealVol"].values      # σ_t (actual)
    Z   = period_df["Positive"].values.astype(float)

    n_oos  = len(vol_fc)
    fc_aic = vol_fc["LogVolFcast_AIC"].values
    fc_sic = vol_fc["LogVolFcast_SIC"].values

    pr_base  = np.full(n_oos, np.nan)
    pr_np_a  = np.full(n_oos, np.nan)
    pr_np_s  = np.full(n_oos, np.nan)
    pr_ext_a = np.full(n_oos, np.nan)
    pr_ext_s = np.full(n_oos, np.nan)

    # Extended-model p̂ ∉ [0,1] sayacı (paper Sec. 3: "this was inconsequential
    # as all our predicted probabilities turn out to lie between 0 and 1").
    # Bu sayaç S&P 500 verisinde de aynı düzenliliğin geçerli olup olmadığını
    # ampirik olarak test eder.
    ext_oob = {"aic_below": 0, "aic_above": 0,
               "sic_below": 0, "sic_above": 0,
               "aic_total": 0, "sic_total": 0}

    for i in range(n_oos):
        k = init_window + i   # index of the period being forecast

        # In-sample data: periods 0 … k-1
        R_is  = R[:k]
        LV_is = LV[:k]
        SV_is = SV[:k]
        Z_is  = Z[:k]

        # ── Baseline ──────────────────────────────────────────────────────────
        pr_base[i] = float(Z_is.mean())

        # ── Mean regression (Eq 4) ────────────────────────────────────────────
        # R_t = β₀ + β₁·log(σ_t) + β₂·[log(σ_t)]²   (actual σ_t)
        c    = _mean_regression(R_is, LV_is)
        mu_is = c[0] + c[1] * LV_is + c[2] * LV_is ** 2   # μ̂_t for t < k

        # Standardised in-sample residuals z_t = (R_t − μ̂_t) / σ_t
        z_is = (R_is - mu_is) / SV_is

        # x_t = 1/σ_t (used for extended model)
        x_is = 1.0 / SV_is

        # Gram-Charlier regression regressors and response (computed once)
        y_gc   = 1.0 - Z_is                              # 1 − I(R_t > 0)

        # ── Nonparametric (Eq 5) + Extended (Eq 8) ───────────────────────────
        for lv_fc, out_np, out_ext, label in [
            (fc_aic[i], pr_np_a, pr_ext_a, "aic"),
            (fc_sic[i], pr_np_s, pr_ext_s, "sic"),
        ]:
            if np.isnan(lv_fc):
                continue

            sig_fc = np.exp(lv_fc)          # σ̂_{k+1|k}
            mu_fc  = _eval_mu(lv_fc, c)    # μ̂_{k+1|k}
            x_fc   = 1.0 / sig_fc          # x̂_{k+1}

            # — Nonparametric —
            # p = 1 − F̂(−μ̂_{k+1|k} / σ̂_{k+1|k})  →  zaten [0,1], clip yok.
            threshold = -mu_fc / sig_fc
            out_np[i] = float(1.0 - np.mean(z_is <= threshold))

            # — Extended / Gram-Charlier —
            # Regressors: Φ(−μ̂_t · x_t)  and  Φ(−μ̂_t · x_t) · x_t
            Phi_is  = norm.cdf(-mu_is * x_is)
            X_gc    = np.column_stack([Phi_is, Phi_is * x_is])
            try:
                b       = _ols(X_gc, y_gc)          # [β̂₀, β̂₁]
                Phi_fc  = norm.cdf(-mu_fc * x_fc)
                p_raw   = 1.0 - Phi_fc * (b[0] + b[1] * x_fc)

                # Out-of-range bookkeeping (clip stat) — paper [0,1] içinde
                # kaldığını gözlemliyor; biz S&P 500 için sayıyoruz.
                ext_oob[f"{label}_total"] += 1
                if p_raw < 0.0:
                    ext_oob[f"{label}_below"] += 1
                elif p_raw > 1.0:
                    ext_oob[f"{label}_above"] += 1

                out_ext[i] = float(np.clip(p_raw, 1e-8, 1 - 1e-8))
            except Exception:
                pass

    out = pd.DataFrame({
        "Pr_Baseline":     pr_base,
        "Pr_Nonpar_AIC":   pr_np_a,
        "Pr_Nonpar_SIC":   pr_np_s,
        "Pr_Extended_AIC": pr_ext_a,
        "Pr_Extended_SIC": pr_ext_s,
        "Return":   R[init_window: init_window + n_oos],
        "Positive": Z[init_window: init_window + n_oos],
        "RealVol":  SV[init_window: init_window + n_oos],
    }, index=vol_fc.index)
    out.attrs["ext_oob"] = ext_oob
    return out
