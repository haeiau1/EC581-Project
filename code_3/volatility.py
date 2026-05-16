"""
volatility.py — recursive ARMA forecasting of log realized volatility.

At each out-of-sample period k the model is re-estimated on the expanding
window log_vol[0 : k] and a one-step-ahead forecast is generated.
Both AIC-selected and SIC-selected models are recorded.

Paper reference: Section 2 — "we also choose our models recursively:
at each period we select ARMA models for log-volatility by minimizing
either the AIC or the SIC."
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import ARMA_ORDERS

warnings.filterwarnings("ignore")


def _try_arma(series: np.ndarray, p: int, q: int) -> tuple:
    """
    Fit ARMA(p,q) with constant to series.
    Returns (aic, bic, one-step-ahead forecast) or (inf, inf, nan) on failure.
    """
    try:
        res = SARIMAX(
            series, order=(p, 0, q), trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, method="lbfgs", maxiter=200)
        fc_raw = res.forecast(steps=1)
        fcast  = float(fc_raw.iloc[0] if hasattr(fc_raw, "iloc") else fc_raw[0])
        return res.aic, res.bic, fcast
    except Exception:
        return np.inf, np.inf, np.nan


def _select_and_forecast(series: np.ndarray) -> tuple:
    """
    Try every order in ARMA_ORDERS, select by AIC and SIC separately.
    Returns (forecast_aic, forecast_sic).
    """
    best_aic = (np.inf, np.nan)
    best_sic = (np.inf, np.nan)

    for p, q in ARMA_ORDERS:
        aic, bic, fc = _try_arma(series, p, q)
        if aic < best_aic[0]:
            best_aic = (aic, fc)
        if bic < best_sic[0]:
            best_sic = (bic, fc)

    return best_aic[1], best_sic[1]


def recursive_forecast(log_vol: pd.Series, init_window: int) -> pd.DataFrame:
    """
    Expanding-window one-step-ahead forecasts of log realized volatility.

    For k = init_window, …, N-1:
      - Fit ARMA on log_vol[0 : k]   (in-sample = periods 0 … k-1)
      - Forecast log_vol[k]           (= σ_{k+1|k} in paper notation)

    Returns DataFrame indexed by the forecast dates (log_vol.index[init_window:])
    with columns: LogVolFcast_AIC, LogVolFcast_SIC.
    """
    vals  = log_vol.values
    n     = len(vals)
    n_oos = n - init_window

    rows, dates = [], []

    print(f"    Recursive ARMA: {n_oos} steps", flush=True)
    for k in range(init_window, n):
        step = k - init_window
        if step % 50 == 0:
            print(f"      step {step}/{n_oos}", flush=True)

        fa, fs = _select_and_forecast(vals[:k])
        rows.append({"LogVolFcast_AIC": fa, "LogVolFcast_SIC": fs})
        dates.append(log_vol.index[k])

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
