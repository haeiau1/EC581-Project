"""
evaluation.py — Brier score computation and volatility subperiod analysis.

Two scoring rules (paper Section 3):
  Brier(Abs) = (1/T) Σ |p_t − z_t|
  Brier(Sq)  = (1/T) Σ 2·(p_t − z_t)²
  where z_t = I(R_t > 0)

Volatility subperiods (out-of-sample realized vol):
  Low    : 1st – 33rd percentile
  Medium : 34th – 66th percentile
  High   : 67th – 100th percentile
"""
import numpy as np
import pandas as pd

# Model columns present in the forecast DataFrame
MODELS = {
    "Baseline":     "Pr_Baseline",
    "Nonpar_AIC":   "Pr_Nonpar_AIC",
    "Nonpar_SIC":   "Pr_Nonpar_SIC",
    "Extended_AIC": "Pr_Extended_AIC",
    "Extended_SIC": "Pr_Extended_SIC",
}


def _brier_abs(p: np.ndarray, z: np.ndarray) -> float:
    return float(np.nanmean(np.abs(p - z)))


def _brier_sq(p: np.ndarray, z: np.ndarray) -> float:
    return float(np.nanmean(2.0 * (p - z) ** 2))


def _vol_masks(rv: np.ndarray) -> dict:
    """Boolean masks for the three volatility subperiods."""
    p33 = np.nanpercentile(rv, 33)
    p66 = np.nanpercentile(rv, 66)
    return {
        "low":    rv <= p33,
        "medium": (rv > p33) & (rv <= p66),
        "high":   rv > p66,
    }


def evaluate(fdf: pd.DataFrame) -> dict:
    """
    Compute Brier(Abs), Brier(Sq), mean(|error|), std(|error|)
    for every model over the full OOS sample and each vol subperiod.

    Returns
    -------
    dict  result[period][model_name] = {brier_abs, brier_sq, mean_abs, std_abs}
    period ∈ {full, low, medium, high}
    """
    z  = fdf["Positive"].values.astype(float)
    rv = fdf["RealVol"].values

    masks = {"full": np.ones(len(z), dtype=bool)}
    masks.update(_vol_masks(rv))

    result = {}
    for period, mask in masks.items():
        result[period] = {}
        for name, col in MODELS.items():
            p   = fdf[col].values
            ok  = mask & ~np.isnan(p)
            if ok.sum() < 5:
                result[period][name] = dict(brier_abs=np.nan, brier_sq=np.nan,
                                            mean_abs=np.nan,  std_abs=np.nan)
                continue
            indiv = np.abs(p[ok] - z[ok])
            result[period][name] = {
                "brier_abs": _brier_abs(p[ok], z[ok]),
                "brier_sq":  _brier_sq(p[ok],  z[ok]),
                "mean_abs":  float(indiv.mean()),
                "std_abs":   float(indiv.std()),
            }

    return result
