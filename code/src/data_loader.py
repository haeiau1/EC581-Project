"""
Data loading and preparation for the regime-based portfolio allocation project.

DataLoader handles:
1. Downloading price and macro data from Yahoo Finance / FRED
2. Persisting raw data to CSV so results are reproducible without internet
3. Reading from CSV and building the feature DataFrame used by the HMM
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from fredapi import Fred
from datetime import datetime

from src.config import (
    FRED_API_KEY, START_DATE, END_DATE,
    PRICE_TICKERS, FRED_SERIES, FEATURES,
    MARKET_DATA_CSV, FRED_DATA_CSV, DATA_DIR,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DataLoader:
    """Downloads, caches, and prepares market data for the HMM pipeline."""

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def download_and_save(self) -> None:
        """
        Download all raw data from Yahoo Finance and FRED, then save to CSV.
        Only call this once (or when you need to refresh the data).
        Results are saved to DATA_DIR/market_data.csv and fred_data.csv.
        """
        os.makedirs(DATA_DIR, exist_ok=True)

        logger.info("Downloading price data from Yahoo Finance...")
        prices_pd = self._fetch_prices()
        prices_pd.to_csv(MARKET_DATA_CSV, index=True)
        logger.info(f"  Saved price data → {MARKET_DATA_CSV}")

        logger.info("Downloading macro data from FRED...")
        fred_pd = self._fetch_fred()
        fred_pd.to_csv(FRED_DATA_CSV, index=True)
        logger.info(f"  Saved FRED data → {FRED_DATA_CSV}")

    def load(self) -> pl.DataFrame:
        """
        Load data from local CSV files and build the full feature DataFrame.

        Returns
        -------
        pl.DataFrame
            Columns: date, er_large, er_small, er_bond, r_large, r_small,
                     r_bond, rf_daily, vix, term_spread, hy_oas,
                     yield_10y, yield_2y, tbill_ann
            Rows: one per business day from START_DATE, nulls dropped.

        Raises
        ------
        FileNotFoundError
            If CSV files do not exist (run download_and_save() first).
        """
        if not os.path.exists(MARKET_DATA_CSV) or not os.path.exists(FRED_DATA_CSV):
            raise FileNotFoundError(
                f"Data files not found. Run download_and_save() first.\n"
                f"  Expected: {MARKET_DATA_CSV}\n"
                f"            {FRED_DATA_CSV}"
            )

        prices_pd = pd.read_csv(MARKET_DATA_CSV, index_col=0, parse_dates=True)
        fred_pd   = pd.read_csv(FRED_DATA_CSV,   index_col=0, parse_dates=True)

        return self._build_features(prices_pd, fred_pd)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _fetch_prices(self) -> pd.DataFrame:
        """Download adjusted close prices from Yahoo Finance."""
        tickers = list(PRICE_TICKERS.keys())
        raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        daily = close.resample("B").last().ffill()
        daily.index.name = "date"
        daily.columns = [PRICE_TICKERS.get(c, c.lower().replace("^", ""))
                         for c in daily.columns]
        return daily

    def _fetch_fred(self) -> pd.DataFrame:
        """Download macro series from FRED."""
        fred = Fred(api_key=FRED_API_KEY)
        frames = {}
        for series_id, col_name in FRED_SERIES.items():
            s = fred.get_series(series_id,
                                observation_start=START_DATE,
                                observation_end=END_DATE)
            frames[col_name] = s
        df_pd = pd.DataFrame(frames)
        df_pd.index = pd.to_datetime(df_pd.index)
        daily = df_pd.resample("B").ffill()
        daily.index.name = "date"
        return daily

    def _build_features(self, prices_pd: pd.DataFrame, fred_pd: pd.DataFrame) -> pl.DataFrame:
        """
        Join prices and FRED data, compute log returns and excess-return features.

        Features used by the HMM (FEATURES config):
          er_large = log(SPY_t / SPY_{t-1}) - rf_daily
          er_small = log(IWM_t / IWM_{t-1}) - rf_daily
          er_bond  = log(TLT_t / TLT_{t-1}) - rf_daily
        """
        prices = pl.from_pandas(prices_pd.reset_index())
        fred   = pl.from_pandas(fred_pd.reset_index())

        # Ensure date column is pl.Date
        prices = prices.with_columns(pl.col("date").cast(pl.Date))
        fred   = fred.with_columns(pl.col("date").cast(pl.Date))

        returns = (
            prices.sort("date")
            .with_columns([
                (pl.col("spy").log() - pl.col("spy").shift(1).log()).alias("r_large"),
                (pl.col("iwm").log() - pl.col("iwm").shift(1).log()).alias("r_small"),
                (pl.col("tlt").log() - pl.col("tlt").shift(1).log()).alias("r_bond"),
            ])
        )

        data = (
            returns
            .join(fred, on="date", how="left")
            .with_columns([
                (pl.col("tbill_ann") / 100.0 / 252.0).alias("rf_daily"),
                (pl.col("yield_10y") - pl.col("yield_2y")).alias("term_spread"),
            ])
            .with_columns([
                (pl.col("r_large") - pl.col("rf_daily")).alias("er_large"),
                (pl.col("r_small") - pl.col("rf_daily")).alias("er_small"),
                (pl.col("r_bond")  - pl.col("rf_daily")).alias("er_bond"),
            ])
            .drop_nulls(subset=FEATURES + ["rf_daily"])
            .select([
                "date", "er_large", "er_small", "er_bond",
                "r_large", "r_small", "r_bond", "rf_daily",
                "vix", "term_spread", "hy_oas",
                "yield_10y", "yield_2y", "tbill_ann",
            ])
        )

        return data
