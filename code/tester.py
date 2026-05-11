"""
EC581 — Regime Prediction Accuracy Tester

Üç farklı açıdan HMM rejim tahminlerinin kalitesini test eder:

  TEST 1 — Model Kalitesi
      Log-likelihood, BIC, AIC ve farklı rejim sayıları (n=2..6)
      karşılaştırılarak 4-state seçiminin istatistiksel olarak doğru
      olup olmadığı gösterilir.

  TEST 2 — Ekonomik Geçerlilik (VIX Ground Truth)
      HMM'nin "crash / correction" dediği günler gerçekten yüksek
      korku (VIX > 25) günleriyle örtüşüyor mu?
      Confusion matrix, precision, recall ve F1 hesaplanır.

  TEST 3 — Walk-Forward Stabilite
      HMM farklı büyüklükte eğitim setleriyle yeniden eğitildiğinde
      aynı tarihlere verilen rejim etiketleri ne kadar tutarlı?
      Pairwise agreement heatmap ve zaman serisi grafikleri üretilir.

  TEST 4 — Rejim Koşullu İstatistikler
      Her rejimdeki getiri dağılımı istatistiksel açıdan birbirinden
      anlamlı şekilde ayrılıyor mu? ANOVA, pairwise t-test ve
      Jarque-Bera normallik testi uygulanır.

Kullanım:
    python code/tester.py
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.hmm_model   import HMMRegimeModel
from src.config      import FEATURES, N_STATES, RESULTS_DIR, REGIME_COLORS

sns.set_theme(style="whitegrid")

# Tüm test çıktıları bu klasöre gider
TESTER_DIR = os.path.join(RESULTS_DIR, "tester")


# ─── Yardımcı: tek HMM fit (paralel kullanım için) ────────────────────────────

def _fit_one(seed: int, X: np.ndarray, n_states: int):
    """Bir random seed ile HMM eğitir; (log_lik, model) veya None döner."""
    try:
        m = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=200, tol=1e-2, random_state=seed, verbose=False)
        m.fit(X)
        return m.score(X), m
    except Exception:
        return None


def _label_any_n(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Herhangi bir n_states için Viterbi etiketlerini büyük hisse
    ortalamasına göre sıralar: en kötü → 'crash', en iyi → 'bull'.
    """
    labels_raw = model.predict(X)
    lc_means   = model.means_[:, 0]
    sorted_idx = np.argsort(lc_means)   # küçükten büyüğe

    n = model.n_components
    # 4 state için canonical isimler; daha az/fazlası için basit sıra numarası
    if n == 4:
        name_map = {
            sorted_idx[0]: "crash",
            sorted_idx[1]: "correction",
            sorted_idx[2]: "moderate_growth",
            sorted_idx[3]: "bull",
        }
    else:
        name_map = {sorted_idx[i]: f"state_{i}" for i in range(n)}

    return np.array([name_map[s] for s in labels_raw])


# ══════════════════════════════════════════════════════════════════════════════
# Ana test sınıfı
# ══════════════════════════════════════════════════════════════════════════════

class RegimeTester:
    """
    HMM rejim tahminlerinin doğruluğunu ve kalitesini dört farklı
    boyuttan ölçen test paketi.
    """

    VIX_HIGH = 25   # Bu seviyenin üzeri "türbülanslı piyasa" kabul edilir
    VIX_LOW  = 15   # Bu seviyenin altı "sakin piyasa"

    def __init__(self, out_dir: str = TESTER_DIR):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # TEST 1 — Model Kalitesi: BIC, AIC, n-state karşılaştırması
    # ─────────────────────────────────────────────────────────────────────

    def test_model_quality(self, model: GaussianHMM, X: np.ndarray,
                           n_states_range=(2, 3, 4, 5, 6)) -> dict:
        """
        Mevcut modelin log-likelihood, BIC ve AIC değerlerini hesaplar.
        Farklı rejim sayıları (2–6) için aynı metrikleri karşılaştırarak
        n=4 seçiminin istatistiksel olarak gerekçeli olduğunu doğrular.

        BIC = -2 * log_lik + k * log(n)   (Bayesian Information Criterion)
        AIC = -2 * log_lik + 2 * k         (Akaike Information Criterion)
        k   = serbest parametre sayısı
        """
        print("\n" + "═" * 65)
        print("  TEST 1 — MODEL KALİTESİ (BIC / AIC / N-STATE KARŞILAŞTIRMA)")
        print("═" * 65)

        n, d = X.shape

        # Mevcut model metrikleri
        log_lik = model.score(X)
        k       = self._count_params(model.n_components, d)
        bic     = -2 * log_lik + k * np.log(n)
        aic     = -2 * log_lik + 2 * k

        print(f"\n  Mevcut model (n_states = {model.n_components})")
        print(f"  {'Log-likelihood':<22}: {log_lik:>12.2f}")
        print(f"  {'Serbest parametre (k)':<22}: {k:>12d}")
        print(f"  {'BIC':<22}: {bic:>12.2f}  (düşük = iyi)")
        print(f"  {'AIC':<22}: {aic:>12.2f}  (düşük = iyi)")

        # n_states karşılaştırması (5 paralel restart yeterli, sadece kalite testi)
        print(f"\n  n_states = {list(n_states_range)} için modeller eğitiliyor (5 restart)...")
        rows = []
        for n_s in n_states_range:
            fits = Parallel(n_jobs=-1)(
                delayed(_fit_one)(seed, X, n_s) for seed in range(5)
            )
            fits = [f for f in fits if f is not None]
            if not fits:
                continue

            best_ll, _ = max(fits, key=lambda r: r[0])
            k_s        = self._count_params(n_s, d)
            bic_s      = -2 * best_ll + k_s * np.log(n)
            aic_s      = -2 * best_ll + 2 * k_s
            rows.append({
                "n_states": n_s,
                "log_lik":  round(best_ll, 2),
                "k":        k_s,
                "BIC":      round(bic_s, 2),
                "AIC":      round(aic_s, 2),
                "mevcut":   " ← seçilen" if n_s == model.n_components else "",
            })

        df = pd.DataFrame(rows)
        best_bic_n = df.loc[df["BIC"].idxmin(), "n_states"]
        best_aic_n = df.loc[df["AIC"].idxmin(), "n_states"]

        print("\n  N-STATE KARŞILAŞTIRMA TABLOSU")
        print("  " + "─" * 62)
        print(f"  {'n_states':>8} {'Log-Lik':>12} {'k':>6} {'BIC':>12} {'AIC':>12}  ")
        print("  " + "─" * 62)
        for _, row in df.iterrows():
            flag = row["mevcut"]
            print(f"  {int(row['n_states']):>8} {row['log_lik']:>12.2f} {row['k']:>6} "
                  f"{row['BIC']:>12.2f} {row['AIC']:>12.2f} {flag}")
        print(f"\n  En düşük BIC → n_states = {best_bic_n}")
        print(f"  En düşük AIC → n_states = {best_aic_n}")

        if best_bic_n == model.n_components:
            print("  ✓ BIC, mevcut n=4 seçimini ONAYLIYOR")
        else:
            print(f"  ⚠ BIC, n={best_bic_n}'ü tercih ediyor — n=4 seçimi tartışılabilir")

        self._plot_bic_aic(df, model.n_components)

        return {
            "log_lik": log_lik, "bic": bic, "aic": aic,
            "comparison": df, "best_bic_n": int(best_bic_n), "best_aic_n": int(best_aic_n),
        }

    @staticmethod
    def _count_params(n_states: int, n_features: int) -> int:
        """
        Tam kovaryans matrisi (full covariance) olan Gaussian HMM
        için serbest parametre sayısı.
        """
        startprob = n_states - 1                                  # π
        transmat  = n_states * (n_states - 1)                    # A
        means     = n_states * n_features                         # μ
        covars    = n_states * n_features * (n_features + 1) // 2 # Σ (simetrik)
        return startprob + transmat + means + covars

    # ─────────────────────────────────────────────────────────────────────
    # TEST 2 — Ekonomik Geçerlilik: VIX Ground Truth
    # ─────────────────────────────────────────────────────────────────────

    def test_economic_validity(self, results_df: pl.DataFrame) -> dict:
        """
        HMM rejim etiketlerini VIX tabanlı sınıflandırmayla karşılaştırır.

        Ground truth kuralı:
            VIX > VIX_HIGH → "türbülanslı" (crash/correction beklenir)
            VIX ≤ VIX_LOW  → "sakin"       (bull/moderate_growth beklenir)

        Metrikler:
            Accuracy : (TP + TN) / toplam
            Precision: TP / (TP + FP)  — "defensive" dediğimizde ne kadar haklıyız?
            Recall   : TP / (TP + FN)  — gerçek kriz günlerini ne kadar yakalıyoruz?
            F1 Score : harmonik ortalama
        """
        print("\n" + "═" * 65)
        print("  TEST 2 — EKONOMİK GEÇERLİLİK (VIX GROUND TRUTH)")
        print("═" * 65)

        regimes = results_df["regime"].to_numpy()
        vix     = results_df["vix"].to_numpy().astype(float)

        # ── Her rejimde VIX istatistikleri ────────────────────────────
        print(f"\n  Rejim başına VIX dağılımı  (eşikler: düşük={self.VIX_LOW}, yüksek={self.VIX_HIGH})")
        print(f"  {'Rejim':<18} {'N':>5} {'Ort. VIX':>9} {'Medyan':>8} {'VIX>25 (%)':>11}")
        print("  " + "─" * 55)

        regime_vix_stats = {}
        for regime in ["crash", "correction", "moderate_growth", "bull"]:
            mask = regimes == regime
            v    = vix[mask & ~np.isnan(vix)]
            if len(v) == 0:
                continue
            pct_high = (v > self.VIX_HIGH).mean() * 100
            regime_vix_stats[regime] = {
                "n": len(v), "mean": v.mean(), "median": np.median(v), "pct_high": pct_high
            }
            print(f"  {regime:<18} {len(v):>5} {v.mean():>9.1f} {np.median(v):>8.1f} {pct_high:>10.1f}%")

        # ── Binary confusion matrix ────────────────────────────────────
        valid        = ~np.isnan(vix)
        hmm_def      = np.isin(regimes, ["crash", "correction"])  # HMM: kötü piyasa
        vix_turb     = vix > self.VIX_HIGH                        # VIX: kötü piyasa

        hmm_def_v    = hmm_def[valid]
        vix_turb_v   = vix_turb[valid]

        tp = ( hmm_def_v &  vix_turb_v).sum()  # ikisi de kötü    ✓
        tn = (~hmm_def_v & ~vix_turb_v).sum()  # ikisi de iyi     ✓
        fp = ( hmm_def_v & ~vix_turb_v).sum()  # HMM kötü, VIX iyi ✗
        fn = (~hmm_def_v &  vix_turb_v).sum()  # HMM iyi, VIX kötü ✗

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / (tp + tn + fp + fn)

        print(f"\n  BINARY SINIFLANDIRMA: HMM Defensif vs VIX > {self.VIX_HIGH}")
        print(f"  {'':22} {'VIX Türbülanslı':>16} {'VIX Sakin':>11}")
        print(f"  {'HMM Defensif (crash/corr)':22} {tp:>16,d} {fp:>11,d}")
        print(f"  {'HMM Büyüme (bull/mod)':22} {fn:>16,d} {tn:>11,d}")
        print(f"\n  Accuracy  : {accuracy:.1%}  — genel doğruluk oranı")
        print(f"  Precision : {precision:.1%}  — 'defensif' dediğimizde VIX gerçekten yüksek mi?")
        print(f"  Recall    : {recall:.1%}  — yüksek VIX günlerini HMM ne kadar yakalıyor?")
        print(f"  F1 Score  : {f1:.3f}")

        # Yorumlama rehberi
        print("\n  Yorum:")
        if recall > 0.6:
            print(f"  ✓ Recall {recall:.0%} — HMM, kriz günlerinin büyük bölümünü yakalıyor")
        else:
            print(f"  ⚠ Recall {recall:.0%} — HMM bazı kriz günlerini ıskalıyor (false negative)")
        if precision > 0.5:
            print(f"  ✓ Precision {precision:.0%} — 'defensif' sinyallerin çoğu gerçek yüksek volatiliteye karşılık geliyor")
        else:
            print(f"  ⚠ Precision {precision:.0%} — 'defensif' sinyallerin bir kısmı gereksiz (false positive)")

        self._plot_vix_by_regime(results_df)
        self._plot_confusion(tp, tn, fp, fn)

        return {
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "regime_vix_stats": regime_vix_stats,
        }

    # ─────────────────────────────────────────────────────────────────────
    # TEST 3 — Walk-Forward Stabilite
    # ─────────────────────────────────────────────────────────────────────

    def test_walkforward_stability(self, data: pl.DataFrame, X: np.ndarray,
                                   year_splits=None) -> dict:
        """
        HMM'yi farklı büyüklükte eğitim setleriyle yeniden eğiterek
        tarihsel rejim etiketlerinin ne kadar tutarlı kaldığını ölçer.

        Yöntem:
            Her split yılı için HMM, data[başlangıç:o_yıl_sonu] üzerinde eğitilir.
            İki farklı pencerede eğitilen modellerin, ortak olan tarihsel
            dönem için verdiği rejim etiketleri karşılaştırılır.

        Beklenti:
            Yüksek agreement (>70%) → model, geçmişe dönük olarak tutarlı
            rejimler üretiyor. Düşükse HMM, her yeniden eğitimde tarihi
            farklı yorumluyor (label instability).
        """
        print("\n" + "═" * 65)
        print("  TEST 3 — WALK-FORWARD STABİLİTE")
        print("═" * 65)

        if year_splits is None:
            year_splits = [2008, 2011, 2014, 2017, 2020, 2023]

        dates      = data["date"].to_list()
        date_years = np.array([
            d.year if hasattr(d, "year") else pd.Timestamp(d).year
            for d in dates
        ])

        # Her yıl için model eğit ve Viterbi etiketlerini kaydet
        print(f"\n  {len(year_splits)} farklı pencere için HMM eğitiliyor (10 restart her biri)...")
        windows = {}
        for year in year_splits:
            mask   = date_years <= year
            n_train = int(mask.sum())
            if n_train < 300:
                print(f"    {year}: yeterli veri yok ({n_train} gün), atlandı.")
                continue

            X_train = X[mask]
            fits    = Parallel(n_jobs=-1)(
                delayed(_fit_one)(seed, X_train, N_STATES) for seed in range(10)
            )
            fits = [f for f in fits if f is not None]
            if not fits:
                print(f"    {year}: tüm fitler başarısız, atlandı.")
                continue

            _, best_m = max(fits, key=lambda r: r[0])
            labels    = _label_any_n(best_m, X_train)

            windows[year] = {"mask": mask, "labels": labels, "n_train": n_train}
            print(f"    ≤{year}: {n_train:>5} gün  (log-lik={fits[0][0]:.0f} ile başlayan fit)")

        # Pairwise agreement matrisi
        years_done = sorted(windows.keys())
        n_w        = len(years_done)
        agree_mat  = np.zeros((n_w, n_w))

        for i, y1 in enumerate(years_done):
            for j, y2 in enumerate(years_done):
                if i == j:
                    agree_mat[i, j] = 1.0
                    continue

                mask1 = windows[y1]["mask"]
                mask2 = windows[y2]["mask"]

                # Hangi tarihler her ikisinin de eğitim setinde?
                overlap = mask1 & mask2

                # Bu tarihlerin mask1 ve mask2 içindeki index'leri
                # mask1'in True olan satırları arasında hangisi overlap?
                idx_in_1 = np.where(overlap[mask1])[0]
                idx_in_2 = np.where(overlap[mask2])[0]

                if len(idx_in_1) == 0 or len(idx_in_2) == 0:
                    agree_mat[i, j] = np.nan
                    continue

                l1 = windows[y1]["labels"][idx_in_1]
                l2 = windows[y2]["labels"][idx_in_2]

                agree_mat[i, j] = (l1 == l2).mean()

        # Sonuçları yazdır
        print("\n  ART ARDA PENCERELER ARASI UYUM ORANI")
        print("  " + "─" * 45)
        for i in range(len(years_done) - 1):
            y1, y2 = years_done[i], years_done[i + 1]
            rate   = agree_mat[i, i + 1]
            bar    = "█" * int(rate * 25)
            print(f"  ≤{y1} → ≤{y2} : {rate:.1%}  {bar}")

        valid_pairs = agree_mat[np.triu_indices(n_w, k=1)]
        valid_pairs = valid_pairs[~np.isnan(valid_pairs)]
        overall     = valid_pairs.mean() if len(valid_pairs) > 0 else 0.0

        print(f"\n  Tüm pencere çiftlerinde ortalama uyum : {overall:.1%}")

        if overall > 0.75:
            print("  ✓ Yüksek stabilite — model, geçmişi tutarlı şekilde yorumluyor")
        elif overall > 0.55:
            print("  ~ Orta stabilite — bazı tarihlerde etiketler değişiyor")
        else:
            print("  ⚠ Düşük stabilite — HMM, yeni veri gelince geçmişi önemli ölçüde yeniden yorumluyor")

        self._plot_stability_heatmap(agree_mat, years_done)
        self._plot_stability_timeseries(windows, years_done, data)

        return {
            "agreement_matrix": agree_mat,
            "years":            years_done,
            "overall":          overall,
            "windows":          windows,
        }

    # ─────────────────────────────────────────────────────────────────────
    # TEST 4 — Rejim Koşullu İstatistikler
    # ─────────────────────────────────────────────────────────────────────

    def test_regime_statistics(self, results_df: pl.DataFrame) -> dict:
        """
        Rejim koşullu getiri dağılımlarının istatistiksel testleri.

        t-test (tek örneklem): Her rejimde ortalama getiri sıfırdan farklı mı?
        ANOVA (F-test)        : Rejim ortalamaları birbirinden farklı mı?
        Pairwise t-test       : Hangi rejim çiftleri istatistiksel olarak ayrışıyor?
        Jarque-Bera           : Her rejimde getiriler normal dağılıyor mu?

        *** = p < 0.001   ** = p < 0.01   * = p < 0.05
        """
        print("\n" + "═" * 65)
        print("  TEST 4 — REJİM KOŞULLU İSTATİSTİKLER")
        print("═" * 65)

        regimes = results_df["regime"].to_numpy()
        r_large = results_df["r_large"].to_numpy()   # SPY log getirileri

        # ── Her rejim için temel istatistikler ve testler ─────────────
        print(f"\n  {'Rejim':<18} {'N':>5} {'Yıllık Ret':>11} {'Yıllık Vol':>11} "
              f"{'t-stat':>8} {'p-val':>8} {'H0:μ=0':>9} {'JB p':>8}")
        print("  " + "─" * 85)

        regime_returns = {}
        for regime in ["crash", "correction", "moderate_growth", "bull"]:
            mask = regimes == regime
            r    = r_large[mask]
            if len(r) < 10:
                continue

            ann_ret = r.mean() * 252
            ann_vol = r.std(ddof=1) * np.sqrt(252)

            # H0: rejim ortalaması = 0
            t_stat, p_val = stats.ttest_1samp(r, popmean=0)
            reject        = "REJECT" if p_val < 0.05 else "fail"

            # Normallik testi (Jarque-Bera)
            _, jb_p = stats.jarque_bera(r)

            regime_returns[regime] = r
            print(f"  {regime:<18} {len(r):>5} {ann_ret:>+11.2%} {ann_vol:>11.2%} "
                  f"{t_stat:>8.2f} {p_val:>8.4f} {reject:>9} {jb_p:>8.4f}")

        # ── ANOVA: rejim ortalamaları eşit mi? ───────────────────────
        groups  = list(regime_returns.values())
        f_stat, anova_p = stats.f_oneway(*groups)

        print(f"\n  ONE-WAY ANOVA (H0: tüm rejim ortalamaları eşit)")
        print(f"  F-istatistiği : {f_stat:.2f}")
        print(f"  p-değeri      : {anova_p:.2e}")
        print(f"  Sonuç         : {'REJECT — rejimler istatistiksel olarak ayrışıyor ✓' if anova_p < 0.05 else 'Fail — rejimler anlamlı şekilde ayrışmıyor ⚠'}")

        # ── Pairwise t-testleri ────────────────────────────────────────
        print(f"\n  PARÇALı t-TESTLER (p-değerleri, *** p<0.001 ** p<0.01 * p<0.05)")
        print("  " + "─" * 60)
        reg_list = list(regime_returns.keys())
        for i in range(len(reg_list)):
            for j in range(i + 1, len(reg_list)):
                r1, r2 = regime_returns[reg_list[i]], regime_returns[reg_list[j]]
                # Welch t-test (eşit varyans varsayımı yok)
                _, p = stats.ttest_ind(r1, r2, equal_var=False)
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {reg_list[i]:<18} vs {reg_list[j]:<18}: p = {p:.4f}  {stars}")

        self._plot_regime_distributions(regime_returns)

        return {
            "regime_returns": regime_returns,
            "f_stat":         f_stat,
            "anova_p":        anova_p,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Tüm testleri sırayla çalıştır
    # ─────────────────────────────────────────────────────────────────────

    def run_all(self, model: GaussianHMM, data: pl.DataFrame,
                X: np.ndarray, results_df: pl.DataFrame) -> dict:
        """Dört testi sırayla çalıştırır ve özet tablosu basar."""

        print("\n" + "█" * 65)
        print("  EC581 — REJİM TAHMİN DOĞRULUK TESTİ")
        print("█" * 65)

        q1 = self.test_model_quality(model, X)
        q2 = self.test_economic_validity(results_df)
        q3 = self.test_walkforward_stability(data, X)
        q4 = self.test_regime_statistics(results_df)

        # Özet
        print("\n" + "═" * 65)
        print("  ÖZET")
        print("═" * 65)
        bic_ok = "✓" if q1["best_bic_n"] == N_STATES else "⚠"
        print(f"  {bic_ok} BIC optimal n_states      : {q1['best_bic_n']}  (mevcut: {N_STATES})")
        print(f"  {'✓' if q2['accuracy'] > 0.60 else '⚠'} VIX accuracy              : {q2['accuracy']:.1%}")
        print(f"  {'✓' if q2['recall']   > 0.50 else '⚠'} VIX crash recall          : {q2['recall']:.1%}")
        print(f"  {'✓' if q3['overall']  > 0.70 else '⚠'} Walk-forward stabilite    : {q3['overall']:.1%}")
        print(f"  {'✓' if q4['anova_p']  < 0.05 else '⚠'} ANOVA (rejimler ayrışıyor?): p = {q4['anova_p']:.2e}")
        print(f"\n  Grafikler: {self.out_dir}")

        return {"model_quality": q1, "economic_validity": q2,
                "stability": q3, "statistics": q4}

    # ─────────────────────────────────────────────────────────────────────
    # Grafik fonksiyonları (private)
    # ─────────────────────────────────────────────────────────────────────

    def _save(self, name: str) -> None:
        path = os.path.join(self.out_dir, name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()

    def _plot_bic_aic(self, df: pd.DataFrame, current_n: int) -> None:
        """BIC ve AIC vs n_states eğri grafiği."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["n_states"], df["BIC"], "o-", color="#1f77b4", label="BIC")
        ax.plot(df["n_states"], df["AIC"], "s--", color="#ff7f0e", label="AIC")
        ax.axvline(current_n, color="red", linestyle=":", alpha=0.7,
                   label=f"Seçilen n={current_n}")
        ax.set_xlabel("Rejim Sayısı (n_states)")
        ax.set_ylabel("Bilgi Kriteri Değeri")
        ax.set_title("Model Seçimi: BIC / AIC vs n_states\n(düşük = daha iyi model)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save("test1_bic_aic.png")

    def _plot_vix_by_regime(self, results_df: pl.DataFrame) -> None:
        """Her rejimde VIX dağılımı (boxplot + strip)."""
        df = results_df.select(["regime", "vix"]).to_pandas().dropna()
        df["regime"] = pd.Categorical(
            df["regime"],
            categories=["crash", "correction", "moderate_growth", "bull"],
            ordered=True,
        )
        df = df.sort_values("regime")

        palette = {r: REGIME_COLORS.get(r, "#888") for r in df["regime"].unique()}

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=df, x="regime", y="vix", palette=palette, ax=ax,
                    width=0.5, flierprops={"marker": ".", "markersize": 2})
        ax.axhline(self.VIX_HIGH, color="red", linestyle="--", linewidth=1.2,
                   label=f"Yüksek eşik ({self.VIX_HIGH})")
        ax.axhline(self.VIX_LOW,  color="green", linestyle="--", linewidth=1.2,
                   label=f"Düşük eşik ({self.VIX_LOW})")
        ax.set_title("VIX Dağılımı — HMM Rejimine Göre\n(yüksek VIX = gerçek piyasa stresi)")
        ax.set_xlabel("Rejim")
        ax.set_ylabel("VIX Endeksi")
        ax.legend()
        plt.tight_layout()
        self._save("test2_vix_distribution.png")

    def _plot_confusion(self, tp: int, tn: int, fp: int, fn: int) -> None:
        """2x2 confusion matrix ısı haritası."""
        matrix = np.array([[tp, fp], [fn, tn]])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            matrix, annot=True, fmt=",d", cmap="Blues",
            xticklabels=[f"VIX > {self.VIX_HIGH}\n(Türbülanslı)", f"VIX ≤ {self.VIX_HIGH}\n(Sakin)"],
            yticklabels=["HMM: Defensif\n(crash/correction)", "HMM: Büyüme\n(bull/mod_growth)"],
            ax=ax,
        )
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        ax.set_title(
            f"Confusion Matrix — HMM vs VIX\n"
            f"Precision={prec:.1%}  Recall={rec:.1%}  F1={2*prec*rec/(prec+rec+1e-9):.2f}"
        )
        plt.tight_layout()
        self._save("test2_confusion_matrix.png")

    def _plot_stability_heatmap(self, agree: np.ndarray, years: list) -> None:
        """Pencereler arası uyum oranları ısı haritası."""
        fig, ax = plt.subplots(figsize=(8, 6))
        mask_nan = np.isnan(agree)
        sns.heatmap(agree, annot=True, fmt=".0%", cmap="RdYlGn",
                    xticklabels=[f"≤{y}" for y in years],
                    yticklabels=[f"≤{y}" for y in years],
                    vmin=0, vmax=1, ax=ax, mask=mask_nan)
        ax.set_title(
            "Walk-Forward Stabilite — Rejim Uyum Oranı\n"
            "(% aynı etiket, ortak eğitim dönemi için)"
        )
        plt.tight_layout()
        self._save("test3_stability_heatmap.png")

    def _plot_stability_timeseries(self, windows: dict, years: list,
                                   data: pl.DataFrame) -> None:
        """Her eğitim penceresi için rejim etiketlerini zaman serisinde gösterir."""
        dates = pd.to_datetime(data["date"].to_pandas())

        regime_order = ["crash", "correction", "moderate_growth", "bull"]
        regime_num   = {r: i for i, r in enumerate(regime_order)}
        color_list   = [REGIME_COLORS.get(r, "#888") for r in regime_order]

        valid_years = [y for y in years if y in windows]
        n_rows      = len(valid_years)
        if n_rows == 0:
            return

        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        for ax, year in zip(axes, valid_years):
            mask   = windows[year]["mask"]
            labels = windows[year]["labels"]
            d_sub  = dates[mask]
            nums   = np.array([regime_num.get(l, 0) for l in labels])
            colors = [color_list[n] for n in nums]

            ax.scatter(d_sub, nums, s=1.5, c=colors)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(["crash", "corr.", "mod.", "bull"], fontsize=7)
            ax.set_ylabel(f"≤{year}", fontsize=8)
            ax.set_ylim(-0.6, 3.6)
            ax.grid(axis="x", alpha=0.2)

        axes[0].set_title(
            "Farklı Eğitim Pencereleri — Rejim Etiket Karşılaştırması\n"
            "(aynı tarih her satırda aynı renk → yüksek stabilite)"
        )
        plt.tight_layout()
        self._save("test3_stability_timeseries.png")

    def _plot_regime_distributions(self, regime_returns: dict) -> None:
        """Rejim koşullu getiri dağılımları (violin + boxplot)."""
        order = ["crash", "correction", "moderate_growth", "bull"]
        data_rows = [
            {"Rejim": regime, "Yıllık Getiri (%)": r * 252 * 100}
            for regime in order
            if regime in regime_returns
            for r in regime_returns[regime]
        ]
        df = pd.DataFrame(data_rows)

        palette = {r: REGIME_COLORS.get(r, "#888") for r in order}

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=df, x="Rejim", y="Yıllık Getiri (%)",
            order=[o for o in order if o in df["Rejim"].unique()],
            palette=palette, ax=ax, inner="quartile", density_norm="width",
        )
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_title(
            "Rejim Koşullu Getiri Dağılımları\n"
            "(SPY log getirileri, yıllıklandırılmış, tam in-sample dönem)"
        )
        ax.set_ylabel("Yıllık Getiri (%)")
        plt.tight_layout()
        self._save("test4_regime_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# Çalıştırma giriş noktası
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 65)
    print("  EC581 — REJİM TAHMİN DOĞRULUK TESTİ BAŞLATILIYOR")
    print("═" * 65)

    # 1. Veri yükle
    print("\n  Veri CSV'den yükleniyor...")
    loader = DataLoader()
    data   = loader.load()
    X      = data.select(FEATURES).to_numpy()
    print(f"  {X.shape[0]} gün, {X.shape[1]} özellik")

    # 2. HMM eğit (ana model — 50 restart)
    print("\n  HMM eğitiliyor (50 restart)...")
    hmm = HMMRegimeModel()
    hmm.fit(X, n_restarts=50)
    results_df, _ = hmm.build_results(data, X)

    # 3. Testleri çalıştır
    tester = RegimeTester()
    tester.run_all(hmm.model, data, X, results_df)


if __name__ == "__main__":
    main()
