"""
Christoffersen & Diebold (2006) — Full Replication
Management Science 52(8), pp. 1273-1287

Run:
    cd code_2
    python run_all.py

Produces Figures 1–10 in code_2/results/
"""

import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEPARATOR = "=" * 62


def main():
    t0 = time.time()
    print(SEPARATOR)
    print("  Christoffersen & Diebold (2006) — Full Replication")
    print(SEPARATOR)

    # ── Figures 1–2  (Analytical Gaussian) ───────────────────────
    print("\n[1/3] Analytical Gaussian figures (Figures 1–2)")
    from figures_1_2 import figure1, figure2
    figure1()
    figure2()

    # ── Figures 3–7  (Heston Simulation) ─────────────────────────
    print("\n[2/3] Heston simulation figures (Figures 3–7)")
    print("      This may take several minutes …")
    from figures_3_7 import (figure3, figure4, figure5,
                              figure6, figure7)
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()

    # ── Figures 8–10 (S&P 500 Empirical) ─────────────────────────
    print("\n[3/3] S&P 500 empirical figures (Figures 8–10)")
    from figures_8_10 import load_sp500, figure8, figure9, figure10
    df = load_sp500()
    figure8(df)
    figure9(df)
    figure10(df)

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    results = os.path.join(HERE, "results")
    files   = sorted(f for f in os.listdir(results) if f.endswith('.png'))
    print(f"\n{SEPARATOR}")
    print(f"  Done in {elapsed:.0f}s  —  {len(files)} figures saved to results/")
    for f in files:
        print(f"    {f}")
    print(SEPARATOR)


if __name__ == '__main__':
    main()
