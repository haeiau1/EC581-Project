"""
Christoffersen & Diebold (2006) — Figures 1 & 2
Analytical Gaussian results (Section 2).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTDIR, exist_ok=True)


def figure1():
    """
    Figure 1: Two Gaussian densities with μ=10%, σ=5% and σ=15%.
    Shows: Pr(R>0|σ=5%)≈0.98, Pr(R>0|σ=15%)≈0.75.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    x   = np.linspace(-0.25, 0.45, 600)
    mu  = 0.10

    styles = [
        (0.05, 'steelblue', r'$\sigma=5\%$  →  Pr(R>0) ≈ 0.98'),
        (0.15, 'tomato',    r'$\sigma=15\%$ →  Pr(R>0) ≈ 0.75'),
    ]
    for sigma, color, label in styles:
        p   = norm.cdf(mu / sigma)
        pdf = norm.pdf(x, mu, sigma)
        ax.plot(x, pdf, color=color, lw=2, label=label)
        xf  = x[x >= 0]
        ax.fill_between(xf, norm.pdf(xf, mu, sigma), alpha=0.20, color=color)

    ax.axvline(0, color='k', lw=0.9, ls='--', alpha=0.6)
    ax.set_xlabel('Return', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        'Figure 1  —  Gaussian Return Densities  ($\\mu=10\\%$, varying $\\sigma$)',
        fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.25, 0.45)
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = f'{OUTDIR}/figure1.png'
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def figure2():
    """
    Figure 2: Responsiveness ℛ = dPr(R>0)/dσ vs information ratio μ/σ.
    Minimum (most negative) at μ/σ = √2 ≈ 1.41.
    """
    mu     = 0.10
    z_vals = np.linspace(0.02, 3.0, 500)       # information ratio z = μ/σ
    sigma  = mu / z_vals                         # implied σ
    # ℛ = φ(z)·(-μ/σ²)  where φ is N(0,1) PDF
    resp   = norm.pdf(z_vals) * (-mu / sigma**2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_vals, resp, 'steelblue', lw=2)
    ax.axvline(np.sqrt(2), color='k', lw=1.0, ls='--',
               label=r'$\mu/\sigma = \sqrt{2} \approx 1.41$  (min)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'Information Ratio $\mu/\sigma$', fontsize=12)
    ax.set_ylabel(r'Responsiveness $\mathcal{R} = d\Pr(R>0)/d\sigma$', fontsize=12)
    ax.set_title(
        'Figure 2  —  Responsiveness of Sign Probability to Volatility (Gaussian)',
        fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 3)
    plt.tight_layout()
    path = f'{OUTDIR}/figure2.png'
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


if __name__ == '__main__':
    print("Generating Figures 1 & 2 ...")
    figure1()
    figure2()
    print("Done.")
