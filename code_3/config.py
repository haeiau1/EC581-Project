"""
config.py — shared constants and paths for the replication.
"""
from pathlib import Path

BASE    = Path(__file__).parent
FIG_DIR = BASE / "results" / "figures"
TAB_DIR = BASE / "results" / "tables"

for _d in [FIG_DIR, TAB_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER     = "^GSPC"          # S&P 500 — proxy for MSCI USA
MARKET     = "USA (S&P 500)"
START_DATE = "1980-01-01"

# ── Train / test split ────────────────────────────────────────────────────────
# Paper: estimation 1980:01–1993:12 (168 months), OOS 1994:01–2004:06 (126 months)
# Out-of-sample fraction = 126 / 294 ≈ 0.4286
OOS_FRACTION = 126 / 294

# ── ARMA candidate orders for log-volatility model selection ──────────────────
ARMA_ORDERS = [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]

# ── Frequencies (in months) ───────────────────────────────────────────────────
FREQUENCIES = [1, 2, 3]
FREQ_LABEL  = {1: "1 mth",    2: "2 mth",    3: "3 mth"}
FREQ_NAME   = {1: "Monthly",  2: "Two-Month", 3: "Quarterly"}
