"""
Hidden Markov Model (HMM) regime detection.

HMMRegimeModel wraps hmmlearn's GaussianHMM with:
  - Parallelised EM restarts (joblib) for better log-likelihood optima
  - Numba-accelerated forward algorithm (JIT compiled if numba is available)
  - Online single-day filter update for intraday / live use
  - Adaptive restart count: fewer restarts as training set grows

References: Hamilton (1989); Ang & Bekaert (2002); Guidolin & Timmermann (2007)
"""

import logging
import warnings
import numpy as np
from scipy import linalg
from joblib import Parallel, delayed
from hmmlearn.hmm import GaussianHMM

from src.config import N_STATES, MAX_ITER, TOL, N_JOBS, REGIME_ORDER, REGIME_COLORS

try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Numba-accelerated forward loop ────────────────────────────────────────────

if _NUMBA:
    from numba import njit as _njit

    @_njit
    def _forward_loop_nb(fwd, log_em, transmat):
        n_samples, n_states = fwd.shape
        for t in range(1, n_samples):
            pred = fwd[t - 1] @ transmat
            for k in range(n_states):
                fwd[t, k] = pred[k] * np.exp(log_em[t, k])
            total = fwd[t].sum()
            if total > 0:
                fwd[t] /= total
        return fwd

    def _forward_pass(model, X):
        n = X.shape[0]
        log_em = model._compute_log_likelihood(X)
        fwd = np.zeros((n, model.n_components))
        fwd[0] = model.startprob_ * np.exp(log_em[0])
        fwd[0] /= fwd[0].sum()
        return _forward_loop_nb(fwd, log_em, model.transmat_)

else:
    def _forward_pass(model, X):
        n = X.shape[0]
        log_em = model._compute_log_likelihood(X)
        fwd = np.zeros((n, model.n_components))
        fwd[0] = model.startprob_ * np.exp(log_em[0])
        fwd[0] /= fwd[0].sum()
        for t in range(1, n):
            pred = fwd[t - 1] @ model.transmat_
            fwd[t] = pred * np.exp(log_em[t])
            s = fwd[t].sum()
            if s > 0:
                fwd[t] /= s
        return fwd


# ── Helper: fit one HMM restart ───────────────────────────────────────────────

def _fit_single(seed, X, n_states, max_iter, tol, prev_model=None):
    """Fit one HMM random restart; return (log_likelihood, model) or None on failure."""
    try:
        m = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=max_iter, tol=tol, random_state=seed, verbose=False)
        # Seed 0 warm-starts from previous model when one is provided
        if prev_model is not None and seed == 0:
            m.startprob_ = prev_model.startprob_.copy()
            m.transmat_  = prev_model.transmat_.copy()
            m.means_     = prev_model.means_.copy()
            m.covars_    = prev_model.covars_.copy()
        m.fit(X)
        return m.score(X), m
    except Exception:
        return None


class HMMRegimeModel:
    """
    4-state Gaussian HMM for identifying market regimes.

    Regimes (ordered by large-cap excess return, ascending):
        crash, correction, bull, moderate_growth
    """

    def __init__(self):
        self.model        = None   # best fitted GaussianHMM
        self.canonical_idx = None  # maps [crash, correction, bull, moderate_growth] → raw HMM state index
        self.raw_to_name   = None  # raw index → regime name
        self.lc_vols       = None  # annualised large-cap vol per raw state

    # ─── Fitting ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, n_restarts: int = 25,
            prev_model: GaussianHMM = None) -> GaussianHMM:
        """
        Fit HMM with parallelised random restarts and optional warm start.

        Parameters
        ----------
        X : np.ndarray, shape (n_days, 3)
            Daily excess returns [er_large, er_small, er_bond]
        n_restarts : int
            Number of EM restarts (seed 0 = warm start from prev_model if given)
        prev_model : GaussianHMM, optional
            Previously fitted model used to warm-start seed 0

        Returns
        -------
        GaussianHMM  — best model by log-likelihood
        """
        results = Parallel(n_jobs=N_JOBS)(
            delayed(_fit_single)(seed, X, N_STATES, MAX_ITER, TOL, prev_model)
            for seed in range(n_restarts)
        )
        results = [r for r in results if r is not None]
        if not results:
            raise RuntimeError("All HMM fits failed")

        scores = [r[0] for r in results]
        _, best = max(results, key=lambda r: r[0])

        logger.info(f"HMM fit: {len(results)}/{n_restarts} restarts OK, "
                    f"best log-lik={max(scores):.4f}, std={np.std(scores):.4f}")

        self.model = best
        self.raw_to_name, self.canonical_idx, self.lc_vols = self._label_states(best)
        return best

    @staticmethod
    def get_n_restarts(n_train_days: int, has_prev_model: bool) -> int:
        """Heuristic: fewer restarts as dataset grows and warm-start improves."""
        if not has_prev_model:
            return 50
        if n_train_days < 500:
            return 20
        if n_train_days < 1000:
            return 10
        if n_train_days < 2000:
            return 7
        return 5

    # ─── Regime labelling ─────────────────────────────────────────────────

    @staticmethod
    def _label_states(model: GaussianHMM):
        """
        Assign canonical regime names by sorting raw states on large-cap mean.

        Returns
        -------
        raw_to_name  : dict {raw_idx: regime_name}
        canonical_idx: list [crash_raw, correction_raw, bull_raw, moderate_growth_raw]
        lc_vols      : np.ndarray annualised volatilities per raw state
        """
        lc_means = model.means_[:, 0]
        lc_vols  = np.sqrt(np.array([c[0, 0] for c in model.covars_]) * 252)

        sorted_idx = sorted(range(N_STATES), key=lambda i: lc_means[i])
        crash_raw, correction_raw, moderate_raw, bull_raw = sorted_idx

        raw_to_name = {
            crash_raw:      "crash",
            correction_raw: "correction",
            moderate_raw:   "moderate_growth",
            bull_raw:       "bull",
        }
        canonical_idx = [crash_raw, correction_raw, bull_raw, moderate_raw]
        return raw_to_name, canonical_idx, lc_vols

    # ─── Filtered probabilities ───────────────────────────────────────────

    def filtered_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Compute filtered (forward) regime probabilities for the full dataset.

        Returns
        -------
        np.ndarray, shape (n_days, 4)
            Columns ordered as canonical_idx (crash, correction, bull, moderate_growth)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        raw_probs = _forward_pass(self.model, X)           # (n, 4) in raw state order
        return raw_probs[:, self.canonical_idx]             # reorder to canonical

    def update_filter_one_day(self, prev_filtered: np.ndarray,
                              new_returns: np.ndarray) -> np.ndarray:
        """
        O(k²) online filter update for a single new day of returns.
        Useful for live / intraday signal refreshes without a full forward pass.

        Parameters
        ----------
        prev_filtered : (k,) array  — yesterday's filtered probs (canonical order)
        new_returns   : (3,) array  — today's [er_large, er_small, er_bond]

        Returns
        -------
        (k,) array — today's filtered probs (canonical order)
        """
        # Convert from canonical back to raw order for HMM prediction
        k = self.model.n_components
        raw_prev = np.zeros(k)
        for canon_i, raw_i in enumerate(self.canonical_idx):
            raw_prev[raw_i] = prev_filtered[canon_i]

        predicted = raw_prev @ self.model.transmat_
        log_em    = self.model._compute_log_likelihood(new_returns.reshape(1, -1))
        updated   = predicted * np.exp(log_em[0])
        total     = updated.sum()
        updated   = updated / total if total > 0 else predicted

        # Return in canonical order
        return updated[self.canonical_idx]

    # ─── Build results DataFrame ──────────────────────────────────────────

    def build_results(self, data, X: np.ndarray):
        """
        Attach filtered probabilities and Viterbi regime labels to the data DataFrame.

        Parameters
        ----------
        data : pl.DataFrame  — full feature DataFrame from DataLoader
        X    : np.ndarray    — feature matrix aligned with data rows

        Returns
        -------
        results     : pl.DataFrame  with regime and prob_* columns added
        state_probs : np.ndarray, shape (n, 4) in canonical order
        """
        import polars as pl

        state_probs  = self.filtered_probs(X)   # canonical order
        viterbi_raw  = self.model.predict(X)
        viterbi_named = np.array([self.raw_to_name[s] for s in viterbi_raw])

        results = data.with_columns([
            pl.Series("regime",               viterbi_named.tolist()),
            pl.Series("prob_crash",           state_probs[:, 0].tolist()),
            pl.Series("prob_correction",      state_probs[:, 1].tolist()),
            pl.Series("prob_bull",            state_probs[:, 2].tolist()),
            pl.Series("prob_moderate_growth", state_probs[:, 3].tolist()),
            pl.Series("max_prob",             state_probs.max(axis=1).tolist()),
        ])

        return results, state_probs

    # ─── Console summary ─────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print regime parameter table and transition matrix to stdout."""
        model        = self.model
        canonical    = self.canonical_idx
        lc_vols      = self.lc_vols
        means_raw    = model.means_
        transmat_raw = model.transmat_

        eigvals, eigvecs = linalg.eig(transmat_raw.T)
        ss_vec   = eigvecs[:, np.argmax(eigvals.real)].real
        ss_probs = ss_vec / ss_vec.sum()

        def avg_dur(p):
            return 1.0 / (1.0 - p) if p < 1.0 else np.inf

        print("═" * 75)
        print("  REGIME PARAMETER ESTIMATES")
        print("  (means annualised ×252, vol annualised ×√252)")
        print("═" * 75)
        print(f"  {'Regime':<16} {'Mean LC':>9} {'Mean SC':>9} {'Mean BD':>9} "
              f"{'Vol LC':>9} {'Persist':>8} {'Avg Dur':>9} {'SS Prob':>8}")
        print("  " + "─" * 75)

        for name, raw_idx in zip(REGIME_ORDER, canonical):
            m_ann  = means_raw[raw_idx] * 252
            v_ann  = lc_vols[raw_idx]
            p_self = transmat_raw[raw_idx, raw_idx]
            dur    = avg_dur(p_self)
            ss     = ss_probs[raw_idx]
            print(f"  {name:<16} {m_ann[0]:>+9.2%} {m_ann[1]:>+9.2%} {m_ann[2]:>+9.2%} "
                  f"{v_ann:>9.2%} {p_self:>8.3f} {dur:>8.1f}d {ss:>8.1%}")

        print("\n  TRANSITION MATRIX  P(row → col)")
        print("  " + "─" * 70)
        print("  " + f"{'':18}" + "".join(f"{n:>14}" for n in REGIME_ORDER))
        for from_name, from_raw in zip(REGIME_ORDER, canonical):
            row = [transmat_raw[from_raw, to_raw] for to_raw in canonical]
            print(f"  {from_name:<18}" + "".join(f"{v:>14.3f}" for v in row))
