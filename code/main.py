"""
EC581 — Regime-Based Portfolio Allocation
Main pipeline orchestration.

Execution order:
  1. Load pre-downloaded data from CSV
  2. Fit 4-state Gaussian HMM
  3. Compute regime labels and filtered probabilities
  4. Walk-forward backtest (original + enhanced HMM strategies)
  5. Run same walk-forward weights through Backtrader for independent verification
  6. Compute benchmark strategies (equal-weight, buy-and-hold 60/40, static MV)
  7. Print performance comparison table
  8. Save all results to CSV and visualise

Prerequisites
-------------
Run download_data.py once before running this script.

Usage
-----
    python main.py
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Make sure the code/ directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    OOS_START, GAMMA, RESULTS_DIR,
    MARKET_DATA_CSV, FRED_DATA_CSV, FEATURES,
)
from src.data_loader      import DataLoader
from src.hmm_model        import HMMRegimeModel
from src.portfolio        import PortfolioOptimizer
from src.backtest         import WalkForwardEngine, EnhancedWalkForwardEngine, BacktraderEngine
from src.benchmarks       import BenchmarkRunner
from src.metrics          import PerformanceAnalyzer
from src.visualization    import Visualizer


# ─── Paths ────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("═" * 65)
    print("  EC581 — REGIME-BASED PORTFOLIO ALLOCATION")
    print("═" * 65)

    # ── STEP 1: Load data ─────────────────────────────────────────────────
    print("\n  STEP 1 — LOAD DATA")
    print("  " + "─" * 40)

    loader = DataLoader()
    data   = loader.load()

    X = data.select(FEATURES).to_numpy()
    dates_pd = pd.to_datetime(data["date"].to_pandas())

    print(f"  Observations : {X.shape[0]} business days")
    print(f"  Date range   : {dates_pd.min().date()} → {dates_pd.max().date()}")

    # ── STEP 2: Fit HMM ───────────────────────────────────────────────────
    print("\n  STEP 2 — FIT HMM")
    print("  " + "─" * 40)

    hmm = HMMRegimeModel()
    n_restarts = HMMRegimeModel.get_n_restarts(X.shape[0], has_prev_model=False)
    print(f"  Running {n_restarts} restarts...")
    hmm.fit(X, n_restarts=n_restarts)

    # ── STEP 3: Regime parameters and filtered probabilities ──────────────
    print("\n  STEP 3 — REGIME PARAMETERS")
    print("  " + "─" * 40)
    hmm.print_summary()

    results, state_probs = hmm.build_results(data, X)
    print("\n  REGIME FREQUENCIES")
    freq = (
        results.group_by("regime")
        .agg(pl.len().alias("days"))
        .with_columns((pl.col("days") / results.shape[0]).alias("pct"))
        .sort("days", descending=True)
    )
    for row in freq.iter_rows(named=True):
        print(f"    {row['regime']:<16} {row['days']:>4} days ({row['pct']:.1%})")

    # ── STEP 4: Portfolio weights ─────────────────────────────────────────
    print("\n  STEP 4 — PORTFOLIO WEIGHTS")
    print("  " + "─" * 40)

    opt = PortfolioOptimizer()
    regime_weights = opt.compute_regime_weights(hmm.model, hmm.canonical_idx,
                                               gamma=GAMMA, allow_short=False)
    print("  Per-regime weights (long-only):")
    for regime, w in regime_weights.items():
        print(f"    {regime:<18}: SPY {w[0]*100:>5.1f}% | IWM {w[1]*100:>5.1f}% | TLT {w[2]*100:>5.1f}%")

    # ── STEP 5: Walk-forward backtests ────────────────────────────────────
    print("\n  STEP 5 — WALK-FORWARD BACKTESTS")
    print("  " + "─" * 40)

    # Original HMM soft allocation
    engine_orig = WalkForwardEngine()
    print("  Running Original HMM Walk-Forward...")
    soft_results, retrain_dates = engine_orig.run(results, hmm.model, hmm.canonical_idx)
    print(f"    HMM refitted {len(retrain_dates)} times")

    # Enhanced HMM
    engine_enh = EnhancedWalkForwardEngine()
    print("  Running Enhanced HMM Walk-Forward...")
    enh_results, enh_retrain_dates = engine_enh.run(results, hmm.model, hmm.canonical_idx)
    print(f"    HMM refitted {len(enh_retrain_dates)} times")

    # ── STEP 6: Backtrader independent verification ───────────────────────
    print("\n  STEP 6 — BACKTRADER VERIFICATION")
    print("  " + "─" * 40)

    price_df = pd.read_csv(MARKET_DATA_CSV, index_col=0, parse_dates=True)
    price_df.index = pd.to_datetime(price_df.index)

    # Weights DataFrame for Backtrader (OOS window only)
    bt_weights = soft_results.select(["date", "w_spy", "w_iwm", "w_tlt"])

    bt_engine = BacktraderEngine()
    try:
        bt_out = bt_engine.run(bt_weights, price_df)
        print(f"  Backtrader final portfolio value : ${bt_out['final_value']:,.0f}")
        sharpe_bt = bt_out.get("sharpe", {})
        if sharpe_bt and "sharperatio" in sharpe_bt:
            print(f"  Backtrader Sharpe ratio         : {sharpe_bt['sharperatio']:.2f}")
    except Exception as e:
        logger.warning(f"Backtrader run failed (non-fatal): {e}")
        bt_out = None

    # ── STEP 7: Benchmarks ────────────────────────────────────────────────
    print("\n  STEP 7 — BENCHMARKS")
    print("  " + "─" * 40)

    bench  = BenchmarkRunner()
    ew_r   = bench.equal_weight(results)
    bh_r   = bench.buy_and_hold(results)
    smv_r  = bench.static_mv(results, hmm.model, hmm.canonical_idx, gamma=GAMMA)
    print("  Benchmarks computed.")

    # ── STEP 8: Performance comparison table ──────────────────────────────
    print("\n  STEP 8 — PERFORMANCE COMPARISON")
    print("  " + "─" * 40)

    strategies = {
        "Enhanced HMM":         enh_results,
        "Original HMM":         soft_results,
        "Static Mean-Variance": smv_r,
        "Buy & Hold 60/40":     bh_r,
        "Equal Weight 1/N":     ew_r,
    }

    oos_dates_list = soft_results["date"].to_list()
    oos_rf = results.filter(
        pl.col("date").cast(pl.Date).is_in(oos_dates_list)
    )["rf_daily"]

    analyzer = PerformanceAnalyzer()
    analyzer.print_comparison_table(strategies, oos_rf, gamma=GAMMA)

    # Regime-conditional analysis for the enhanced engine
    enh_with_rf = enh_results.join(
        results.select(["date", "rf_daily", "regime"]).with_columns(pl.col("date").cast(pl.Date)),
        on="date", how="left"
    )
    analyzer.regime_conditional_analysis(
        enh_with_rf,
        enh_with_rf["net_return"].to_numpy(),
        strategy_name="Enhanced HMM",
    )

    # ── STEP 9: Save results ──────────────────────────────────────────────
    print("\n  STEP 9 — SAVING RESULTS")
    print("  " + "─" * 40)

    _save_results(strategies)

    # ── STEP 10: Visualisations ───────────────────────────────────────────
    print("\n  STEP 10 — VISUALISATIONS")
    print("  " + "─" * 40)

    viz = Visualizer(save_dir=RESULTS_DIR)

    viz.plot_regimes(dates_pd, state_probs)

    oos_dates = soft_results["date"].to_list()
    returns_dict = {name: df["net_return"].to_numpy() for name, df in strategies.items()}

    viz.plot_equity_curves(returns_dict, oos_dates,
                           title="Equity Curves — Enhanced vs Benchmarks")
    viz.plot_drawdowns(returns_dict, oos_dates,
                       title="Drawdowns — Enhanced vs Benchmarks")
    viz.plot_weight_evolution(soft_results,
                              title="Original HMM — Weight Evolution")
    viz.plot_regime_performance(
        enh_results["net_return"].to_numpy(),
        enh_results["regime"].to_numpy(),
        enh_results["date"].to_list(),
        title="Enhanced HMM — Performance by Regime",
    )

    print("\n═" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("═" * 65)


def _save_results(strategies: dict) -> None:
    """Save each strategy's results to CSV and a combined parquet file."""
    for name, df in strategies.items():
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        csv_path  = os.path.join(RESULTS_DIR, f"{safe_name}_results.csv")
        df_pd     = df.to_pandas()
        df_pd.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")

    # Combined parquet — align columns across strategies before concatenating
    common_cols = ["date", "gross_return", "net_return", "turnover",
                   "w_spy", "w_iwm", "w_tlt", "w_cash"]
    combined = pl.concat([
        df.select([c for c in common_cols if c in df.columns])
          .with_columns(pl.lit(name).alias("strategy"))
        for name, df in strategies.items()
    ])
    combined.write_parquet(os.path.join(RESULTS_DIR, "all_strategies.parquet"))
    print(f"    Saved: {os.path.join(RESULTS_DIR, 'all_strategies.parquet')}")


if __name__ == "__main__":
    main()
