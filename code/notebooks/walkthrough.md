# Rejim Tabanlı Portföy Tahsisi ve Geriye Dönük Test: Kapsamlı Teknik Rehber

Bu doküman, projede uygulanan kantitatif portföy modelinin arkasındaki matematiksel temelleri, literatür dayanaklarını (academic justifications) ve bunların kod mimarisine nasıl döküldüğünü detaylandıran kapsamlı bir teknik rehberdir. Amacımız, HMM tabanlı rejim tespitini Mean-Variance optimizasyonu ile birleştirerek, gerçek dünya işlem maliyetlerini içeren dinamik bir alım-satım stratejisi oluşturmaktır.

---

## 1. Metodolojik Çerçeve ve Akademik Dayanaklar

Projedeki strateji tasarımı beş ana akademik sütuna dayanmaktadır.

### 1.1. Gizli Markov Modeli (HMM) ile Rejim Tespiti
Modelin temelini, piyasa getirilerinin (SPY, IWM, TLT) gözlemlenemeyen (hidden) rejimler tarafından yönlendirildiğini varsayan **Gizli Markov Modeli (HMM)** oluşturur *(Hamilton, 1989; Ang & Bekaert, 2002)*. 
* Projede toplam 4 adet rejim tanımlanmıştır: `Crash` (Çöküş), `Correction` (Düzeltme), `Moderate Growth` (Ilımlı Büyüme) ve `Bull` (Boğa).
* Model her bir gün ("t" günü) için varlıkların getiri vektörünü kullanarak, o gün her bir rejimde ("k" rejimi) olma olasılığını (p_tk) hesaplar (Filtered Probabilities).

### 1.2. Rejim Bazlı Mean-Variance Optimizasyonu
Klasik Markowitz modelinin aksine, yatırımcının getiri beklentileri ve risk algısı içinde bulunulan rejime göre değişir. **Guidolin & Timmermann (2007)**'nin yaklaşımı referans alınarak, her bir "k" rejimi için ayrı ayrı optimal portföy ağırlıkları hesaplanmıştır.

Optimizasyon problemi şu şekildedir:
**Maksimum Yapılacak Denklem:** w^T * µ_k - (γ / 2) * w^T * Σ_k * w

* **µ_k (Mu):** "k" rejimindeki beklenen getiri vektörü (HMM'den gelen `means_`)
* **Σ_k (Sigma):** "k" rejimindeki varyans-kovaryans matrisi (HMM'den gelen `covars_`)
* **γ (Gamma):** Riskten kaçınma katsayısı (Risk Aversion). Guidolin & Timmermann baz alınarak risk iştahı γ = 5 olarak sabitlenmiştir.

Kısa pozisyonlara (short selling) izin verilip verilmemesine göre ağırlıklar [0, Sonsuz) veya [-1.5, 1.5] (maksimum %150 brüt kaldıraç) aralığında kısıtlandırılmıştır.

### 1.3. Soft Allocation (Yumuşak Tahsis) ve Harmanlama
Portföy ağırlıkları, piyasanın sadece tek bir rejimde olduğuna eminmiş gibi "Hard" bir geçiş (Kritzman et al., 2012) yapmak yerine, HMM'in ürettiği günlük rejim olasılıkları kullanılarak yumuşatılmıştır *(Honda, 2003)*. 

Günlük portföy ağırlığı (w_blended), rejim ağırlıklarının (w_k) olasılıklarla (p_tk) çarpılmış halidir:
**w_blended = Toplam( p_tk * w_k )** *(k=1'den 4'e kadar)*

Bu sayede rejim geçişleri sırasındaki belirsizlik dönemlerinde portföy sert şoklardan korunur ve daha pürüzsüz bir getiri eğrisi (equity curve) elde edilir.

### 1.4. Yön Tahmini (Long / Short Positioning)
Modele çift yönlü işlem yapma esnekliği kazandırılmıştır. Bir varlığın o günkü ağırlığı sıfırın altına düştüğünde (veya %2 eşik değerin altında kaldığında) "Short", üstünde kaldığında "Long" pozisyon alınır. Özellikle "Crash" ihtimalinin yüksek olduğu günlerde riskli varlıklardan (Hisse) çıkarak Nakit veya Bono (TLT) tarafına geçilmesi literatürle *(Bulla et al., 2011; Hauptmann et al., 2014)* uyumlu bir korunma stratejisidir.

### 1.5. Backtesting Motoru ve İşlem Maliyetleri (Transaction Costs)
Stratejinin geçmiş performansı, gerçeğe en uygun şekilde **Walk-Forward (Genişleyen Pencere)** yöntemiyle test edilmiştir.
* Model Out-of-Sample (Örn. 2019) tarihinden itibaren test edilmeye başlar ve her 126 iş gününde (yaklaşık 6 ay) bir geçmiş tüm verilerle yeniden eğitilir (Retrain).
* **Nystrup et al. (2015)** baz alınarak, her gün portföy devir hızı (Turnover) hesaplanmış ve işlem maliyetleri (5 bps / %0.05) getiri üzerinden düşülmüştür:

**Turnover = 0.5 * Toplam( |w_i(bugün) - w_i(dün)| )**
**Net Getiri = Brüt Getiri - (Turnover * 0.0005)**

---

## 2. Kod Mimarisi ve Fonksiyonel Eşleşmeler

Yazdığımız Python kodları (şu an `RegimeAllocationFaster.ipynb` içinde gömülüdür), bu karmaşık matematiği modüler ve hızlı bir şekilde uygulayacak şekilde tasarlanmıştır.

### A. Portföy Optimizasyonu Sınıfları (Portfolio Weights)
* `compute_regime_weights()`: SciPy optimize kullanılarak Guidolin & Timmermann (2007) Mean-Variance formülünü çözer. Regülarizasyon için Ridge (1e-6) kullanılarak singüler kovaryans matrisi hataları önlenmiştir.
* `compute_daily_weights()`: HMM'den gelen 4 olasılığı (prob_crash vb.) matris çarpımıyla (dot product) optimal ağırlıklarla harmanlar (Soft Allocation).
* `determine_positions()`: Ağırlık eşiklerine (> 0.02) bakarak gün gün hangi varlıkta Long, Short veya Flat olduğumuzu sınıflandırır.

### B. Geriye Dönük Test Sınıfları (Backtest Engine & Benchmarks)
* `BacktestEngine.run_walkforward()`: Kronolojik olarak ilerler, 126 günde bir HMM modelini sıfırdan "fit" eder. Turnover hesaplar ve Net Getiriyi bulur.
* **Benchmarklar:** Modelin değer yarattığını kanıtlamak için 3 farklı stratejiyle yarışır:
  * `equal_weight_benchmark`: 1/N kuralı (SPY, IWM, TLT'ye eşit dağılım).
  * `buy_and_hold_benchmark`: 60/40 Geleneksel portföy (Sadece başlangıçta işlem maliyeti öder).
  * `static_mv_benchmark`: Yalnızca 2019 öncesi verilerle tek bir kere eğitilen, zaman içinde değişmeyen Statik Markowitz portföyü.

### C. Performans ve Görselleştirme Sınıfları (Analysis)
* `print_comparison_table()`: OOS periyodu için Yıllıklandırılmış Getiri, Volatilite, Sharpe Oranı, Calmar, Max Drawdown ve Power Utility (Fayda Fonksiyonu) metriklerini hesaplar.
* `plot_regime_performance()`: Soft allocation getiri eğrisinin (equity curve) arka planını, o anki aktif (dominant) rejimin rengiyle boyar (Crash=Kırmızı, Bull=Yeşil vb.). Böylece modelin krizlerde nasıl korunduğu görsel olarak kanıtlanır.
* `regime_conditional_analysis()`: Stratejinin spesifik olarak "Bull" veya "Crash" dönemlerindeki Sharpe oranlarını ayrıştırarak gösterir.

---

## 3. Çalıştırma Kılavuzu

Tüm bu modüler kodlar, hiçbir dış bağımlılık yaratmaması ve takım arkadaşlarının kolayca test edebilmesi için **tek bir `RegimeAllocationFaster.ipynb` dosyasına** entegre edilmiştir.

**Nasıl Çalıştırılır?**
1. Jupyter Notebook'u açın.
2. Üst menüden **"Run All" (Tümünü Çalıştır)** seçeneğine tıklayın.
3. Model sırasıyla:
   * Verileri çekecek ve ön hazırlığı yapacak,
   * HMM modelini paralelize edilmiş şekilde `hmmlearn` ile eğitecek,
   * Her bir rejim için optimal ağırlıkları hesaplayıp konsola basacak,
   * 2019 sonrası Out-of-Sample Walk-Forward backtesti çalıştıracak,
   * İşlem maliyetlerini kestikten sonra Performans Tablosunu yazdıracak,
   * En altta Getiri (Equity), Drawdown ve Rejim Overlay grafiklerini çizecektir.

Bu proje, akademik literatürün gücünü modern Python veri bilim aracıyla (Polars, SciPy, HMM) birleştirerek tam teşekküllü bir sistematik yatırım modeli oluşturmuştur.
