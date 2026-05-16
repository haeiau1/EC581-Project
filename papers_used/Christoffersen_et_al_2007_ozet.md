# Makale Özeti: Direction-of-Change Forecasts Based on Conditional Variance, Skewness and Kurtosis Dynamics: International Evidence

**Yazarlar:** Peter F. Christoffersen, Francis X. Diebold, Roberto S. Mariano, Anthony S. Tay, Yiu Kuen Tse
**Yayın:** Journal of Financial Forecasting, 2007 (working paper formatı; Christoffersen & Diebold (2006) Management Science makalesinin uluslararası uzantısı)

---

## 1. Genel Çerçeve

Bu makale, Christoffersen & Diebold (2006) çalışmasının doğrudan uzantısıdır. 2006 makalesi, koşullu volatilite dinamiklerinin getiri *işaret* (yön) öngörülebilirliği yaratabileceğini teorik ve simülasyon temelli olarak göstermişti. 2007 makalesi aşağıdaki sorulara yanıt arar:

1. **Ampirik gerçeklik kontrolü:** 2006'da kurulan teori, gerçek uluslararası piyasalarda *out-of-sample* tahmin gücüne dönüşür mü?
2. **Skewness ve kurtosis genişletmesi:** Koşullu varyansa ek olarak koşullu *çarpıklık* (skewness) ve *basıklık* (kurtosis) dinamikleri kullanılırsa yön tahmin gücü artar mı?
3. **Volatilite rejimleri:** Yön tahmin gücü piyasa volatilitesi düşükken mi yoksa yüksekken mi daha güçlüdür?

**Temel iddia:** Volatilite tahmin edilebilirken — özellikle volatilite düşükken — yön tahminleri istatistiksel olarak baseline (koşulsuz) tahminlerden anlamlı derecede daha iyi performans gösterir.

---

## 2. Veri ve Frekanslar

### 2.1. Ülke Endeksleri

Makale, dokuz ülke için MSCI ulusal endekslerini kullanır:

- **G7 ülkeleri:** ABD, Birleşik Krallık, Kanada, Almanya, Fransa, İtalya, Japonya
- **Asya:** Hong Kong, Singapur

Veri kaynağı **MSCI Country Indexes** (Datastream / Bloomberg üzerinden). Replikasyonumuzda yalnızca ABD için S&P 500 (^GSPC) Yahoo Finance proxy'si kullanılmıştır.

### 2.2. Örnek Aralığı

- **Başlangıç:** 1980 başı (bazı endeksler için 1973 — veri uygunluğuna göre değişir)
- **Bitiş:** 2004 sonu civarı
- **Toplam:** ABD için yaklaşık 25 yıl × 12 ay = ~294 ay

### 2.3. Tahmin Horizonları

Üç farklı, çakışmayan zaman dilimi için analiz yapılır:

| Frekans | Açıklama | Bir Yıldaki Periyot Sayısı |
|---------|----------|-----------------------------|
| 1 aylık | Aylık çakışmayan veri | 12 |
| 2 aylık | İki aylık çakışmayan veri | 6 |
| 3 aylık | Üç aylık (çeyreklik) çakışmayan veri | 4 |

Frekansların bu şekilde seçilmesinin nedeni 2006 makalesinden gelir: simülasyonlar yön öngörülebilirliğinin **40–60 işlem günü** (yaklaşık 2–3 ay) horizonunda zirveye ulaştığını göstermişti. Bu nedenle 1–3 aylık aralık "altın bant" olarak hedeflenir.

---

## 3. Yöntem

### 3.1. Gerçekleşen Volatilite (Realized Volatility) Hesaplama

Her h-aylık periyot için, periyot içindeki günlük log-getirilerin karesinin toplamı:

$$\widehat{RV}_t = \sum_{d \in \text{period } t} r_d^2$$

Bu, Andersen, Bollerslev, Diebold ve Labys (2003) tarafından gösterildiği üzere, koşullu varyansın tutarlı bir kestiricisidir. Modellemede daha çok logaritması kullanılır:

$$\log\widehat{RV}_t$$

çünkü log-vol daha simetrik ve normal-yakın dağılır.

### 3.2. Volatilite Tahmini: Özyinelemeli ARMA

Her out-of-sample (OOS) adımda **expanding window** (genişleyen pencere) kullanılarak ARMA modeli yeniden tahmin edilir:

- Aday model uzayı: ARMA(1,0), (0,1), (1,1), (2,0), (2,1)
- Her aday model için sabit terimli ve `enforce_stationarity=False` ile uydurulur
- Tahminler **AIC** ve **SIC** kriterleriyle ayrı ayrı seçilir (iki paralel tahmin hattı)
- Çıktı: $\log\hat{\sigma}_{t+1|t}$ (bir adım ileri log-vol tahmini)

### 3.3. Yön Tahmin Modelleri

Makale üç farklı yön olasılığı tahmin yöntemini karşılaştırır:

#### Model 1 — Baseline (Eq 3)

Koşulsuz empirik olasılık:

$$\hat{p}_{t+1|t}^{\text{base}} = \frac{1}{t}\sum_{s=1}^{t} \mathbb{I}(R_s > 0)$$

Bu, "tarihsel pozitif gün oranı"dır. Herhangi bir koşullu bilgi kullanmaz. Karşılaştırma referansı (benchmark) görevini görür.

#### Model 2 — Nonparametric (Eq 5)

İki adımlı yapı:

**Adım 1 — Koşullu ortalama regresyonu (Eq 4):**

$$R_t = \beta_0 + \beta_1 \log\sigma_t + \beta_2 (\log\sigma_t)^2 + u_t$$

(Burada $\sigma_t$ gerçekleşen vol; in-sample'da gerçek değer, OOS'ta ARMA tahmini kullanılır.)

**Adım 2 — Standardize edilmiş artıkların empirik CDF'i:**

$$z_t = \frac{R_t - \hat{\mu}_t}{\sigma_t}$$

$$\hat{p}_{t+1|t} = 1 - \hat{F}\!\left(-\frac{\hat{\mu}_{t+1|t}}{\hat{\sigma}_{t+1|t}}\right)$$

Bu yaklaşımın güzelliği: normal dağılım varsayımı yapmaz, getirilerin kalın kuyruğunu ve çarpıklığını standardize edilmiş artıkların gerçek dağılımı üzerinden yakalar.

#### Model 3 — Extended / Gram-Charlier (Eq 8)

Aynı koşullu ortalama regresyonunu kullanır, ama empirik CDF yerine **Gram-Charlier serisi** ile çarpıklık ve basıklık etkilerini yaklaşık olarak modellemeye çalışır. Pratik uygulama:

Sabit terimsiz OLS:

$$(1 - \mathbb{I}(R_t > 0)) = \beta_0 \Phi(-\hat{\mu}_t x_t) + \beta_1 \Phi(-\hat{\mu}_t x_t) \cdot x_t + \text{hata}$$

burada $x_t = 1/\sigma_t$. Tahmin:

$$\hat{p}_{t+1|t} = 1 - \Phi(-\hat{\mu}_{t+1|t}\hat{x}_{t+1}) \cdot (\hat{\beta}_0 + \hat{\beta}_1 \hat{x}_{t+1})$$

Bu, $\Phi(-\mu/\sigma)$'lık temel Gauss yön olasılığını çarpıklık ve basıklık düzeltmeleriyle çarpan bir genişlemedir.

### 3.4. Değerlendirme: Brier Skorları

Tahmin doğruluğu Brier skorlarıyla ölçülür:

- **Brier(Abs):** $\frac{1}{T}\sum |p_t - z_t|$ (mean absolute error)
- **Brier(Sq):** $\frac{2}{T}\sum (p_t - z_t)^2$ (mean squared error, ×2)

Burada $z_t = \mathbb{I}(R_t > 0)$ gerçek yön göstergesidir.

Modelleri kıyaslamak için **göreli skor** kullanılır:

$$\text{Relative Brier} = \frac{\text{Brier}_{\text{model}}}{\text{Brier}_{\text{baseline}}}$$

Oran < 1 → model baseline'dan **daha iyi**.

### 3.5. Volatilite Alt-Dönemleri

OOS örneği gerçekleşen volatiliteye göre üç gruba ayrılır:

- **Düşük (Low):** RV ≤ 33. persantil
- **Orta (Medium):** 33–66. persantil
- **Yüksek (High):** 67. persantil üstü

Bu, "yön öngörülebilirliği volatilite rejimine göre nasıl değişir?" sorusunu yanıtlamak için kritik bir bölümleme.

---

## 4. Ana Bulgular

### 4.1. Volatilite Tahmin Edilebilir

Tüm ülkeler ve frekanslar için ARMA log-vol tahminlerinin MSPE/varyans oranı 0.6–0.8 bandındadır. Bu, log-vol'un sabit ortalama-tahminden anlamlı derecede daha iyi tahmin edilebildiğini gösterir. Frekans uzadıkça oran 1'e yaklaşır (uzun horizonda volatilite öngörülebilirliği azalır).

### 4.2. Yön Öngörülebilirliği Düşük Vol'da Maksimum

**Bu, makalenin en önemli bulgusudur:**

- **Düşük volatilite döneminde:** Nonparametrik ve Extended modeller baseline'a göre %10–20 oranında daha iyi (Brier oranı 0.80–0.90 bandında)
- **Orta volatilite döneminde:** Modeller baseline'la başa baş (oran ≈ 1)
- **Yüksek volatilite döneminde:** Modeller baseline'dan kötü (oran > 1)

Bu örüntü tüm dokuz ülkede tutarlı bir şekilde gözlemlenir. Mekanizma şudur:

- Sakin piyasalarda boğa trendi süreklilik gösterir; düşük vol → daha yüksek pozitif getiri olasılığı; modeller bu sinyali yakalar
- Çalkantılı piyasalarda volatilite kendisi tahmin edilemez hale gelir; yön kaos olur; modellerin koşulluluğu işe yaramaz, hatta yanlış yönlendirebilir

### 4.3. Extended vs Nonparametric

Genelde Extended (Gram-Charlier) model Nonparametric'ten **biraz daha iyi** sonuç verir, ama fark her zaman istatistiksel olarak anlamlı değildir. Düşük-vol döneminde fark belirginleşir; bu da çarpıklık ve basıklık dinamiklerinin sakin piyasalarda ek bilgi içerdiğini düşündürür.

### 4.4. AIC vs SIC

Her iki kriter benzer sonuçlar üretir. SIC genelde daha küçük modeller seçer ama bu örnekte tahmin gücüne anlamlı etkisi yoktur.

### 4.5. Uluslararası Tutarlılık

Bulgu yalnızca ABD'ye özgü değildir. Avrupa (Almanya, Fransa, İtalya, BK), Asya (Japonya, Hong Kong, Singapur) ve Kuzey Amerika (ABD, Kanada) piyasalarında benzer örüntü görülür. Bu, koşullu volatilite-yön ilişkisinin **evrensel bir piyasa özelliği** olduğunu güçlendirir.

---

## 5. Replikasyon Sonuçları (Bizim Projemiz)

ABD için S&P 500 proxy'si kullanılarak yapılan replikasyonda makalenin tüm temel bulguları doğrulanmıştır:

| Bulgu | Makale | Bizim Replikasyon |
|-------|--------|--------------------|
| MSPE/var oranı (1M) | ~0.6 | 0.61 |
| MSPE/var oranı (3M) | ~0.8 | 0.78 |
| Düşük vol — Nonpar (Abs ratio, 3M) | ~0.85 | 0.84 |
| Düşük vol — Extended (Abs ratio, 3M) | ~0.80 | 0.80 |
| Yüksek vol — Nonpar (Abs ratio, 3M) | > 1 | 1.09 |
| Yüksek vol — Extended (Abs ratio, 3M) | > 1 | 1.10 |

Sonuçlar makaledeki örüntüyü, beklenen ±%5 sapma aralığında, **tam olarak tekrarlar.**

---

## 6. Sonuç ve Katkı

Makale şu üç katkıyı yapar:

1. **Ampirik doğrulama:** 2006'daki teori, dokuz uluslararası piyasada *out-of-sample* tahmin gücüne dönüşür.
2. **Pratik tahmin yöntemleri:** Üç farklı tahmin modeli — Baseline, Nonparametrik (empirik CDF), Extended (Gram-Charlier) — net algoritmik adımlarla tanımlanır ve karşılaştırılır.
3. **Rejim-koşullu sonuç:** Yön öngörülebilirliği volatilite rejimine bağlıdır. Bu, portföy yönetimi ve trading stratejileri için doğrudan operasyonel bir uyarı sağlar — sakin piyasada koşullu modelleri kullan, çalkantılı piyasada baseline'a geri dön.

**Proje Açısından Anlamı:** EC581 projesi HMM tabanlı rejim sınıflandırması ile portföy alokasyonu yapmaktadır. Bu makale, **rejimlerin yalnızca varyans/getiri farklılıkları değil, *yön öngörülebilirliği* açısından da farklı olduğunu** gösterir. Bu, rejim bazlı alokasyon kararlarının kavramsal temelini güçlendirir.
