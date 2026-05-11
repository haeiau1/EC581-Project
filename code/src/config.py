"""
Configuration constants for the regime-based portfolio allocation project.
All tunable parameters are defined here so changes propagate everywhere.
"""

import os
from datetime import datetime

# ── API keys ────────────────────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")

# ── Date range ───────────────────────────────────────────────────────────────
START_DATE = "2002-09-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
OOS_START  = "2019-01-01"   # out-of-sample period starts here

# ── HMM parameters (Guidolin & Timmermann 2007) ───────────────────────────
N_STATES  = 4       # crash, correction, moderate_growth, bull
MAX_ITER  = 300
TOL       = 1e-1
N_JOBS    = -1      # -1 = use all CPU cores for parallel HMM restarts

# ── Portfolio / backtest ──────────────────────────────────────────────────
GAMMA             = 5       # risk-aversion coefficient (G&T 2007 base case)
TRANSACTION_COST  = 0.0005  # 5 bps per unit of portfolio turnover
RETRAIN_FREQ      = 126     # business days between HMM refits (~6 months)

# ── Enhanced engine extras ────────────────────────────────────────────────
GAMMA_BULL        = 5       # γ for bull / moderate-growth regimes
GAMMA_DEFENSIVE   = 12      # γ for crash / correction regimes (more conservative)
PROB_EMA_HALFLIFE = 5       # EMA halflife (business days) for probability smoothing
WEIGHT_EMA_ALPHA  = 0.15    # daily snap-fraction toward target weights (0 = no snap)
REBALANCE_THRESH  = 0.02    # minimum weight drift to trigger a rebalance trade

# ── Asset tickers ─────────────────────────────────────────────────────────
PRICE_TICKERS = {
    "SPY":  "spy",
    "IWM":  "iwm",
    "TLT":  "tlt",
    "^VIX": "vix",
}

# ── FRED series ───────────────────────────────────────────────────────────
FRED_SERIES = {
    "DTB3":         "tbill_ann",
    "GS10":         "yield_10y",
    "GS2":          "yield_2y",
    "BAMLH0A0HYM2": "hy_oas",
}

# ── Feature columns fed into HMM ─────────────────────────────────────────
FEATURES = ["er_large", "er_small", "er_bond"]

# ── Canonical regime ordering (sorted by large-cap mean return, ascending)
REGIME_ORDER = ["crash", "correction", "bull", "moderate_growth"]

REGIME_COLORS = {
    "crash":           "#d62728",
    "correction":      "#ff7f0e",
    "bull":            "#2ca02c",
    "moderate_growth": "#1f77b4",
}

# ── Historical events to annotate on charts ────────────────────────────────
EVENTS = [
    ("2008-09-01", "GFC"),
    ("2020-03-01", "COVID"),
    ("2022-01-01", "Rate Hike"),
]

# ── File paths ────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
DATA_DIR    = os.path.join(_BASE, "data")
RESULTS_DIR = os.path.join(_BASE, "results")

MARKET_DATA_CSV = os.path.join(DATA_DIR, "market_data.csv")
FRED_DATA_CSV   = os.path.join(DATA_DIR, "fred_data.csv")
