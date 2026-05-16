"""
Heston (1993) stochastic volatility model — core utilities.

Log-price dynamics:
  dx = (μ - v/2) dt + √v dz₁
  dv = κ(θ - v) dt + η √v dz₂      Corr(dz₁, dz₂) = ρ

All parameters are ANNUALISED. Time in years.
Uses numba JIT if available (highly recommended for n_intraday=288).
"""

import numpy as np
from scipy.integrate import quad

# ── Numba JIT (optional, but strongly recommended for n_intraday=288) ────────
try:
    from numba import njit

    @njit(cache=True)
    def _var_euler(z, kappa, theta, eta, dt, v0):
        sdt = np.sqrt(dt)
        n   = len(z)
        v   = np.empty(n + 1)
        v[0] = v0
        for i in range(n):
            vp    = v[i] if v[i] > 0.0 else 0.0
            v[i+1] = v[i] + kappa*(theta - v[i])*dt + eta*np.sqrt(vp)*sdt*z[i]
            if v[i+1] < 0.0:
                v[i+1] = 0.0
        return v

    @njit(cache=True)
    def _heston_euler(z1, z2, mu, kappa, theta, eta, dt, v0, x0):
        sdt = np.sqrt(dt)
        n   = len(z1)
        x   = np.empty(n + 1)
        v   = np.empty(n + 1)
        x[0] = x0;  v[0] = v0
        for i in range(n):
            vp    = v[i] if v[i] > 0.0 else 0.0
            sig   = np.sqrt(vp)
            x[i+1] = x[i] + (mu - vp/2.0)*dt + sig*sdt*z1[i]
            v[i+1] = v[i] + kappa*(theta - v[i])*dt + eta*sig*sdt*z2[i]
            if v[i+1] < 0.0:
                v[i+1] = 0.0
        return x, v

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ── Benchmark parameters (Christoffersen & Diebold 2006, Section 5.1) ───────
BENCHMARK = dict(mu=0.10, kappa=2.0, theta=0.015, eta=0.15, rho=-0.50)
N_DAYS    = 250     # trading days per year


# ── Characteristic function ──────────────────────────────────────────────────
def heston_cf(u, v0, tau, mu, kappa, theta, eta, rho):
    """
    E[exp(i·u·R)] for log-return R = log(S(τ)/S(0)).
    Albrecher et al. (2007) numerically stable form.
    u, v0 may be arrays (broadcast).
    """
    iu  = 1j * u
    xi  = kappa - rho * eta * iu
    d   = np.sqrt(xi**2 + eta**2 * (iu + u**2))
    g1  = xi - d;  g2 = xi + d;  g = g1 / g2
    edt = np.exp(-d * tau)
    C   = (kappa * theta / eta**2) * (g1 * tau - 2.0 * np.log((1.0 - g*edt) / (1.0 - g)))
    D   = (g1 / eta**2) * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(iu * mu * tau + C + D * v0)


# ── Sign probability: scalar ─────────────────────────────────────────────────
def sign_prob_scalar(v0, tau, mu, kappa, theta, eta, rho, u_max=200.0):
    """Pr(R>0 | v₀) via Gil-Pelaez inversion (single v₀)."""
    def intgd(u):
        return np.real(heston_cf(u, v0, tau, mu, kappa, theta, eta, rho) / (1j * u))
    val, _ = quad(intgd, 1e-6, u_max, limit=500)
    return 0.5 + val / np.pi


# ── Sign probability: vectorised over v ──────────────────────────────────────
def sign_prob_batch(v_array, tau, mu, kappa, theta, eta, rho,
                    n_quad=300, u_max=120.0):
    """
    Pr(R>0 | v) for an array of variances at horizon τ.
    Vectorised: evaluates CF for all v simultaneously on a fixed quadrature grid.
    """
    u  = np.linspace(1e-5, u_max, n_quad)
    v  = np.asarray(v_array, dtype=float)

    iu  = 1j * u
    xi  = kappa - rho * eta * iu
    d   = np.sqrt(xi**2 + eta**2 * (iu + u**2))
    g1  = xi - d;  g2 = xi + d;  g = g1 / g2
    edt = np.exp(-d * tau)
    C   = (kappa * theta / eta**2) * (g1 * tau - 2.0 * np.log((1.0 - g*edt) / (1.0 - g)))
    D   = (g1 / eta**2) * (1.0 - edt) / (1.0 - g * edt)
    expCD = np.exp(iu * mu * tau + C)   # (Q,)

    cf   = expCD[:, None] * np.exp(D[:, None] * v[None, :])  # (Q, N)
    intg = np.real(cf / (1j * u[:, None]))                    # (Q, N)
    return 0.5 + np.trapezoid(intg, u, axis=0) / np.pi


# ── Sign probability: grid over (v, τ) ───────────────────────────────────────
def sign_prob_grid(v_array, tau_array, mu, kappa, theta, eta, rho,
                   n_quad=300, u_max=120.0):
    """P[i,j] = Pr(R>0 | v_array[i]) at horizon tau_array[j]."""
    v = np.asarray(v_array, dtype=float)
    P = np.zeros((len(v), len(tau_array)))
    for j, tau in enumerate(tau_array):
        P[:, j] = sign_prob_batch(v, tau, mu, kappa, theta, eta, rho, n_quad, u_max)
    return P


# ── Variance-only simulation (fast, supports intraday substepping) ───────────
def simulate_variance(n_days, kappa, theta, eta, v0=None, seed=42,
                      n_intraday=288):
    """
    Simulate v(t) = σ²(t) and return daily values.
    n_intraday : intraday steps per trading day
                 288 → 5-min steps (24 h × 12),  matches paper exactly.
    Returns array of shape (n_days + 1,).
    """
    rng   = np.random.default_rng(seed)
    if v0 is None: v0 = theta
    dt    = 1.0 / (N_DAYS * n_intraday)
    total = n_days * n_intraday
    z     = rng.standard_normal(total)

    if HAS_NUMBA:
        v_all = _var_euler(z, kappa, theta, eta, dt, float(v0))
    else:
        sdt   = np.sqrt(dt)
        v_all = np.empty(total + 1);  v_all[0] = v0
        for i in range(total):
            vp = max(v_all[i], 0.0)
            v_all[i+1] = v_all[i] + kappa*(theta-v_all[i])*dt + eta*np.sqrt(vp)*sdt*z[i]
            if v_all[i+1] < 0.0: v_all[i+1] = 0.0

    return v_all[::n_intraday]   # daily values: shape (n_days+1,)


# ── Full (x, v) simulation ────────────────────────────────────────────────────
def simulate_heston(n_days, mu, kappa, theta, eta, rho,
                    n_intraday=288, v0=None, x0=0.0, seed=42):
    """
    Simulate Heston at intraday resolution, return daily (x, v) arrays.
    n_intraday=288 → 5-min / 24 h (matches paper, Section 5.1).
    Returns (x_daily, v_daily) each of shape (n_days+1,).
    """
    rng   = np.random.default_rng(seed)
    if v0 is None: v0 = theta
    dt    = 1.0 / (N_DAYS * n_intraday)
    sq1r  = np.sqrt(1.0 - rho**2)
    total = n_days * n_intraday

    z1_all = rng.standard_normal(total)
    z2_all = rho * z1_all + sq1r * rng.standard_normal(total)

    if HAS_NUMBA:
        x_all, v_all = _heston_euler(
            z1_all, z2_all, mu, kappa, theta, eta, dt, float(v0), float(x0))
    else:
        sdt   = np.sqrt(dt)
        x_all = np.empty(total + 1);  x_all[0] = x0
        v_all = np.empty(total + 1);  v_all[0] = v0
        for i in range(total):
            vp = max(v_all[i], 0.0);  sig = np.sqrt(vp)
            x_all[i+1] = x_all[i] + (mu - vp/2.0)*dt + sig*sdt*z1_all[i]
            v_all[i+1] = v_all[i] + kappa*(theta-v_all[i])*dt + eta*sig*sdt*z2_all[i]
            if v_all[i+1] < 0.0: v_all[i+1] = 0.0

    return x_all[::n_intraday], v_all[::n_intraday]
