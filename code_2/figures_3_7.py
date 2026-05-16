"""
Christoffersen & Diebold (2006) — Figures 3–7
Heston stochastic-volatility simulation (Sections 5.1–5.2).

Benchmark:  μ=0.10, κ=2, θ=0.015, η=0.15, ρ=-0.50
            (daily mean ≈ 0.037%, daily σ ≈ 0.77%)
Simulation: Euler–Maruyama, 24 intraday steps/day (hourly)
            Paper uses 288 (5-min / 24 h), but results are qualitatively identical.
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from heston import (BENCHMARK, N_DAYS,
                    sign_prob_scalar, sign_prob_batch, sign_prob_grid,
                    simulate_variance, simulate_heston)

OUTDIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTDIR, exist_ok=True)

# Six display horizons (trading days)
H6     = [1, 5, 21, 63, 126, 250]
H6_LAB = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semiannual', 'Annual']

N_LONG = 20_000   # days for long correlation analysis
N_SIM  =    500   # days shown in Figure 3

# Horizons for Figures 4–6 (log-spaced 1 → 250 days)
TAU_DAYS  = np.unique(np.round(np.geomspace(1, 250, 55)).astype(int))
TAU_YEARS = TAU_DAYS / N_DAYS


# ── Helper: forecast-realisation correlation (eq. 10) ───────────────────────
def forecast_corr(v_path, tau_years, mu, kappa, theta, eta, rho, n_quad=250):
    """
    Corr(Iₜ, Pₜ) = Std(Pₜ) / √(P̄(1−P̄))  [eq. 10 in paper]
    Uses sign_prob_grid to compute P for every day in v_path.
    """
    P = sign_prob_grid(v_path, tau_years, mu, kappa, theta, eta, rho, n_quad=n_quad)
    Pbar = P.mean(axis=0)
    Pstd = P.std(axis=0)
    denom = np.sqrt(np.maximum(Pbar * (1.0 - Pbar), 1e-12))
    return Pstd / denom


# ── Helper: lag-1 sign autocorrelation ──────────────────────────────────────
def sign_autocorr(x_path, tau_days_arr):
    """
    Lag-1 autocorrelation of the non-overlapping h-day sign sequence
    (Section 4.1). Returns NaN when fewer than 10 pairs are available.
    """
    n    = len(x_path) - 1
    acor = np.full(len(tau_days_arr), np.nan)
    for j, h in enumerate(tau_days_arr.astype(int)):
        idx   = np.arange(0, n - h, h)          # start indices of each period
        if len(idx) < 10: continue
        signs = (x_path[idx + h] - x_path[idx] > 0).astype(float)
        if signs.std() < 1e-8: continue
        acor[j] = np.corrcoef(signs[:-1], signs[1:])[0, 1]
    return acor


# ── Figure 3 ────────────────────────────────────────────────────────────────
def figure3():
    """
    Figure 3: time series of conditional sign probabilities at 6 horizons.
    Sample path of 500 simulated days.
    """
    print("  Figure 3: simulating 500-day Heston path ...")
    p = BENCHMARK
    x, v = simulate_heston(N_SIM, **p, n_intraday=288, seed=7)

    probs = {}
    for h, lab in zip(H6, H6_LAB):
        tau = h / N_DAYS
        P   = np.array([sign_prob_scalar(v[i], tau, **p) for i in range(N_SIM)])
        probs[lab] = P
        print(f"    {lab}: mean P = {P.mean():.3f}")

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax, lab in zip(axes.flat, H6_LAB):
        P = probs[lab]
        ax.plot(np.arange(N_SIM), P, color='steelblue', lw=0.8)
        ax.axhline(P.mean(), color='k', lw=1.1, ls='--')
        ax.set_title(f'{lab} returns', fontsize=11)
        ax.set_xlabel('Time (days)', fontsize=9)
        ax.set_ylabel('Pr(R > 0)', fontsize=9)
        ax.set_ylim(0.3, 1.0)
        ax.set_xlim(0, N_SIM)

    fig.suptitle(
        'Figure 3  —  Conditional Sign Probabilities at Various Horizons\n'
        r'($\mu=0.10,\ \kappa=2,\ \theta=0.015,\ \eta=0.15,\ \rho=-0.50$)',
        fontsize=12, y=1.01)
    plt.tight_layout()
    path = f'{OUTDIR}/figure3.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ── Figure 4 ────────────────────────────────────────────────────────────────
def figure4():
    """
    Figure 4: forecast-realisation correlation vs horizon for μ=0.10,0.05,0.00.
    Variance path is independent of μ → reuse single simulation.
    """
    print(f"  Figure 4: computing forecast correlations (N={N_LONG} days) ...")
    p = BENCHMARK.copy()
    v = simulate_variance(N_LONG, p['kappa'], p['theta'], p['eta'], seed=123)

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = [(0.10, 'steelblue',  r'$\mu=0.10$'),
              (0.05, 'darkorange', r'$\mu=0.05$'),
              (0.00, 'dimgray',    r'$\mu=0.00$')]
    for mu, color, label in styles:
        p['mu'] = mu
        corr = forecast_corr(v, TAU_YEARS, **p)
        ax.plot(TAU_DAYS, corr, color=color, lw=2, label=label)
        print(f"    μ={mu:.2f}: peak corr = {corr.max():.4f} at "
              f"τ={TAU_DAYS[corr.argmax()]} days")

    ax.set_xlabel('Horizon (trading days)', fontsize=12)
    ax.set_ylabel('Corr(Sign Forecast, Realisation)', fontsize=12)
    ax.set_title('Figure 4  —  Forecast Correlation vs Horizon  (varying $\\mu$)', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(1, 250); ax.set_ylim(0, None)
    plt.tight_layout()
    path = f'{OUTDIR}/figure4.png'
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Figure 5 ────────────────────────────────────────────────────────────────
def figure5():
    """
    Figure 5: forecast-realisation correlation vs horizon for κ=2,5,10.
    Each κ requires a separate variance simulation.
    """
    print(f"  Figure 5: varying κ (N={N_LONG} days each) ...")
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = [(2,  'steelblue',  r'$\kappa=2$',  111),
              (5,  'darkorange', r'$\kappa=5$',  222),
              (10, 'dimgray',    r'$\kappa=10$', 333)]
    for kappa, color, label, seed in styles:
        p = BENCHMARK.copy(); p['kappa'] = kappa
        v    = simulate_variance(N_LONG, p['kappa'], p['theta'], p['eta'], seed=seed)
        corr = forecast_corr(v, TAU_YEARS, **p)
        ax.plot(TAU_DAYS, corr, color=color, lw=2, label=label)
        print(f"    κ={kappa}: peak corr = {corr.max():.4f} at "
              f"τ={TAU_DAYS[corr.argmax()]} days")

    ax.set_xlabel('Horizon (trading days)', fontsize=12)
    ax.set_ylabel('Corr(Sign Forecast, Realisation)', fontsize=12)
    ax.set_title('Figure 5  —  Forecast Correlation vs Horizon  (varying $\\kappa$)', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(1, 250); ax.set_ylim(0, None)
    plt.tight_layout()
    path = f'{OUTDIR}/figure5.png'
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Figure 6 ────────────────────────────────────────────────────────────────
def figure6():
    """
    Figure 6: forecast-realisation corr vs first sign autocorrelation.
    Benchmark parameters; both plotted against horizon.
    """
    print(f"  Figure 6: forecast corr + sign autocorr (N={N_LONG} days) ...")
    p = BENCHMARK.copy()

    v    = simulate_variance(N_LONG, p['kappa'], p['theta'], p['eta'], seed=789)
    corr = forecast_corr(v, TAU_YEARS, **p)

    print("    Simulating full Heston path for sign autocorrelation ...")
    x, _ = simulate_heston(N_LONG, **p, n_intraday=288, seed=789)
    acor = sign_autocorr(x, TAU_DAYS)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(TAU_DAYS, corr, 'steelblue', lw=2, label='Forecast–Realisation Correlation')
    ax.plot(TAU_DAYS, acor, 'tomato',    lw=2, ls='--', label='First Sign Autocorrelation')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Horizon (trading days)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title(
        'Figure 6  —  Forecast Correlation vs Sign Autocorrelation\n'
        r'(Benchmark $\mu=0.10$)', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(1, 250)
    plt.tight_layout()
    path = f'{OUTDIR}/figure6.png'
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Figure 7 ────────────────────────────────────────────────────────────────
def figure7():
    """
    Figure 7: numerical derivative dP/dσ(t) vs annualised information ratio
    μ/σ_{t+τ|t} in the Heston model. τ = 40 trading days.

    Expected-average-variance formula (paper footnote 18):
        σ²_{t+τ|t} = θ + [(1−exp(−κτ))/(κτ)] · (v(t) − θ)

    The V-shape emerges because dP/dσ < 0 when IR is high (more vol hurts
    predictability) but dP/dσ → 0 from below as IR → 0 (vol hardly matters
    when drift is negligible). Zero-crossing at IR ≈ √2 (Gaussian analogue).
    """
    print("  Figure 7: numerical responsiveness in Heston model ...")
    p    = BENCHMARK.copy()
    tau  = 40.0 / N_DAYS                   # 40 trading days in years
    kappa, theta, mu = p['kappa'], p['theta'], p['mu']

    # Weight factor for expected average variance (paper footnote 18)
    factor = (1.0 - np.exp(-kappa * tau)) / (kappa * tau)

    # Wide grid: v(t) from near-zero to very high variance
    v_vals = np.logspace(-6, np.log10(0.50), 400)

    IR_list, deriv_list = [], []
    for v in v_vals:
        sig_c  = np.sqrt(v)
        # Adaptive step: 0.1% of current σ, at least 1e-5
        h_sig  = max(0.001 * sig_c, 1e-5)

        if sig_c > 2.0 * h_sig:
            v_up   = (sig_c + h_sig) ** 2
            v_dn   = (sig_c - h_sig) ** 2
            P_up   = sign_prob_scalar(v_up, tau, **p)
            P_dn   = sign_prob_scalar(v_dn, tau, **p)
            dPdsig = (P_up - P_dn) / (2.0 * h_sig)
        else:
            # One-sided forward difference for very small σ
            v_up   = (sig_c + h_sig) ** 2
            P_c    = sign_prob_scalar(v,    tau, **p)
            P_up   = sign_prob_scalar(v_up, tau, **p)
            dPdsig = (P_up - P_c) / h_sig

        # Expected average variance → annualised std → IR
        ev2     = theta + factor * (v - theta)
        sig_ann = np.sqrt(max(ev2, 1e-10))
        IR_list.append(mu / sig_ann)
        deriv_list.append(dPdsig)

    IR    = np.array(IR_list)
    deriv = np.array(deriv_list)
    idx   = np.argsort(IR)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(IR[idx], deriv[idx], 'steelblue', lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'Annualised Information Ratio $\mu/\sigma_{t+\tau|t}$', fontsize=12)
    ax.set_ylabel(r'$d\Pr(R>0)/d\sigma(t)$', fontsize=12)
    ax.set_title(
        r'Figure 7  —  Responsiveness in Heston Model  ($\tau=40$ days)',
        fontsize=12)
    ax.set_xlim(0, 4.0)
    plt.tight_layout()
    path = f'{OUTDIR}/figure7.png'
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating Figures 3–7  (simulation-based) ...")
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    print("Done.")
