"""
data_loader.py — download ^GSPC and aggregate into h-month periods.

For each frequency h ∈ {1, 2, 3}:
  - Return     : sum of daily log-returns within the period
  - RealVol    : sqrt( sum of squared daily log-returns )   [realized vol, σ_t]
  - LogRealVol : log(RealVol)                               [log σ_t, modelled by ARMA]
  - Positive   : I(Return > 0)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from config import TICKER, START_DATE, FREQUENCIES


def download() -> pd.Series:
    """Download daily adjusted-close prices and return daily log-returns."""
    df = yf.download(TICKER, start=START_DATE, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return np.log(close / close.shift(1)).dropna()


def _period_key(date: pd.Timestamp, h: int) -> tuple:
    """
    Map a calendar date to a non-overlapping h-month period key.
      h=1 → (year, month)
      h=2 → (year, 0-based bi-month index)   Jan-Feb=0, Mar-Apr=1, …
      h=3 → (year, 0-based quarter index)    Q1=0, Q2=1, Q3=2, Q4=3
    """
    y, m = date.year, date.month
    if h == 1:
        return (y, m)
    if h == 2:
        return (y, (m - 1) // 2)
    return (y, (m - 1) // 3)


def build_periods(daily_ret: pd.Series, h: int) -> pd.DataFrame:
    """
    Aggregate daily log-returns into non-overlapping h-month periods.
    Periods with fewer than (h * 8) trading days are dropped (e.g. first/last
    incomplete period at data boundaries).
    """
    keys = [_period_key(d, h) for d in daily_ret.index]
    tmp  = pd.DataFrame({"ret": daily_ret.values, "key": keys},
                         index=daily_ret.index)

    rows = []
    for key, grp in tmp.groupby("key", sort=True):
        r = grp["ret"].values
        if len(r) < max(10, h * 8):          # skip incomplete boundary periods
            continue
        rv = float(np.sqrt((r ** 2).sum()))   # realized volatility σ_t
        rows.append({
            "Date":       grp.index[-1],      # end-of-period date
            "Return":     float(r.sum()),      # log return over the period
            "RealVol":    rv,
            "LogRealVol": float(np.log(rv)),
            "Positive":   int(r.sum() > 0),
        })

    return pd.DataFrame(rows).set_index("Date")


def build_all(daily_ret: pd.Series) -> dict:
    """Return {h: DataFrame} for h in FREQUENCIES."""
    return {h: build_periods(daily_ret, h) for h in FREQUENCIES}
