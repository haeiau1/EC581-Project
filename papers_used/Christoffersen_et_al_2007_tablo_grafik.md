# Christoffersen, Diebold, Mariano, Tay & Tse (2007) — Tablo ve Şekil Açıklamaları

Makale: **Direction-of-Change Forecasts Based on Conditional Variance, Skewness and Kurtosis Dynamics: International Evidence**

Bu doküman makaledeki tüm tablo ve şekilleri tek tek, ne anlama geldiklerini ve hangi mesajı taşıdıklarını açıklar. Replikasyonumuzdaki ABD sonuçları da yanına eklenmiştir.

---

## Table 1a — Returns Summary Statistics

### Ne Gösteriyor?

Her ülke ve her frekans (1, 2, 3 aylık) için ham getirilerin tanımlayıcı istatistikleri:

- **Mean** — ortalama getiri (periyot başına)
- **Std.Dev** — standart sapma (volatilite proxy'si)
- **Skewness** — çarpıklık (sol/sağ kuyruk asimetrisi)
- **Excess Kurtosis** — fazla basıklık (normal dağılımdan sapma; kalın kuyruk göstergesi)
- **JB p-val** — Jarque-Bera normallik testi p-değeri

### Ana Mesaj

Tüm ülkelerde, tüm frekanslarda:

1. Getiri ortalaması pozitif ama küçük (aylık ~%0.5–1)
2. Standart sapma artan frekansla artar (kümülatif etki)
3. Çarpıklık **negatif** (sol kuyruk; çöküş riski)
4. Fazla basıklık **pozitif** (kalın kuyruk; aşırı olaylar)
5. JB p-değeri ≈ 0 → getiriler **normal dağılmaz**

### Replikasyon (ABD, S&P 500)

| Frekans | Mean | Std.Dev | Skewness | Ex.Kurt | JB p |
|---------|------|---------|----------|---------|------|
| 1 mth | 0.0076 | 0.0440 | −0.83 | 2.67 | 0.000 |
| 2 mth | 0.0152 | 0.0598 | −1.07 | 3.75 | 0.000 |
| 3 mth | 0.0228 | 0.0803 | −0.96 | 1.47 | 0.000 |

### Neden Önemli?

Bu tablo, makalenin neden Gauss-temelli (normal dağılım varsayan) modellerin yetersiz olabileceğini gösterdiği temel verisidir. Negatif çarpıklık ve yüksek basıklık, $\Phi(\mu/\sigma)$ formülünün gerçek dağılımı yakalamayacağını ima eder; Nonparametrik ve Extended modellerin motivasyonu burada doğar.

---

## Table 1b — Log Realized Volatility Summary Statistics

### Ne Gösteriyor?

Tablo 1a'nın paraleli — fakat ham getiri yerine **log gerçekleşen volatilite** $\log\widehat{RV}_t$ için.

### Ana Mesaj

1. Log-vol ortalaması negatif (vol < 1, log < 0)
2. Çarpıklık **pozitif** (sağ kuyruk; volatilite ara sıra patlar)
3. Basıklık ham getirilere göre **daha düşük** (log dönüşümü dağılımı yumuşatıyor)
4. JB hâlâ reddediliyor ama daha simetrik bir dağılım var

### Replikasyon (ABD, S&P 500)

| Frekans | Mean | Std.Dev | Skewness | Ex.Kurt |
|---------|------|---------|----------|---------|
| 1 mth | −3.24 | 0.454 | 0.70 | 1.45 |
| 2 mth | −2.86 | 0.422 | 0.86 | 1.73 |
| 3 mth | −2.64 | 0.414 | 0.86 | 1.43 |

### Neden Önemli?

Bu tablo, neden makalenin ARMA modellemesini **log-vol** üzerinde yaptığını açıklar. Ham vol pozitif olmak zorundadır ve oldukça çarpık dağılır; log dönüşümü modelleme için çok daha uygun bir seri üretir. ARMA gibi lineer Gauss-temelli modeller log-vol'da çok daha iyi çalışır.

---

## Table 2 — Out-of-Sample Volatility Forecast Accuracy (MSPE / Variance Ratio)

### Ne Gösteriyor?

Her ülke, her frekans ve her bilgi kriteri (AIC, SIC) için:

$$\text{Ratio} = \frac{\text{MSPE of log-vol forecast}}{\text{Sample variance of actual log-vol}}$$

### Yorumu

- **Oran < 1** → ARMA tahminleri sabit-ortalama (naive) tahminden daha iyi
- **Oran ≈ 1** → ARMA hiç katkı sağlamıyor
- **Oran > 1** → ARMA daha kötü (tahmin gürültü ekliyor)

### Ana Mesaj

Makale tüm ülkelerde, tüm frekanslarda oranların **0.6–0.85** bandında olduğunu raporlar. Yani:

- Log-vol **gerçekten tahmin edilebilir**
- Bu tahmin edilebilirlik kısa horizonda (1M) en güçlü, uzun horizonda (3M) zayıflıyor
- AIC ve SIC arasında küçük farklar var

### Replikasyon (ABD)

| Frekans | AIC | SIC |
|---------|-----|-----|
| 1 mth | 0.611 | 0.607 |
| 2 mth | 0.703 | 0.712 |
| 3 mth | 0.784 | 0.793 |

Bu sonuçlar makalenin G7 ortalama değerleriyle son derece uyumludur.

### Neden Önemli?

Tablo 2, makalenin tüm mantıksal zincirinin **ilk halkasıdır**. Eğer volatilite tahmin edilemezse, ondan türetilen yön tahminleri de yararsız olur. Bu tablo bu ön-koşulun sağlandığını gösterir.

---

## Table 3 — Brier(Abs) Scores by Volatility Subperiod

### Ne Gösteriyor?

Her ülke, frekans ve model (Baseline, Nonpar, Extended) için Brier(Abs) skorlarının:

- **Mean** — ortalama
- **Std** — standart sapma

değerleri, vol alt-dönemlerine bölünmüş halde (Low / Medium / High).

### Yorumu

- Skor **düşük** = tahmin **iyi** (0'a yakın)
- Skor **yüksek** = tahmin kötü (0.5'e yakın = rastgele tahminden farksız)

### Ana Mesaj

Tüm modellerde:

- **Düşük vol alt-döneminde:** Skorlar en düşük (en iyi tahminler)
- **Yüksek vol alt-döneminde:** Skorlar en yüksek (tahmin zorlaşıyor)

Ayrıca: düşük vol'da Nonpar/Extended skorları Baseline'dan **belirgin biçimde** daha düşük; yüksek vol'da ya eşit ya da daha yüksek.

### Replikasyon (ABD, 1 aylık)

| Subperiod | Baseline | Nonpar AIC | Extended AIC |
|-----------|----------|------------|--------------|
| Low | 0.404 | 0.359 | 0.358 |
| Medium | 0.469 | 0.476 | 0.479 |
| High | 0.517 | 0.538 | 0.537 |

Düşük vol'da Nonpar/Extended yaklaşık 0.045 daha düşük; yüksek vol'da yaklaşık 0.021 daha yüksek. Makaledeki örüntüyle aynı.

### Neden Önemli?

Tablo 3, makalenin asıl mesajını **mutlak skorlarla** sunar (oransal değil). Yön tahmin performansının vol rejiminden ne kadar etkilendiğini doğrudan rakamla gösterir.

---

## Table 4 — Relative Brier Scores (Tables 4a, 4b, 4c, 4d)

Her tablo aynı yapıya sahiptir: ülkeler × frekanslar × (Brier(Abs), Brier(Sq)) × (Baseline, Nonpar, Extended). Yalnızca veri alt-dönemi değişir:

| Tablo | Alt-Dönem | Açıklama |
|-------|-----------|----------|
| **4a** | Full Sample | Tüm OOS örneği |
| **4b** | Low Volatility | Yalnızca düşük-vol periyotları |
| **4c** | Medium Volatility | Yalnızca orta-vol periyotları |
| **4d** | High Volatility | Yalnızca yüksek-vol periyotları |

### Hücre Formatı

- **Bsln** sütunları: Baseline'ın **mutlak** skoru
- **Npar / Ext** sütunları: ilgili modelin Baseline'a göre **oranı** (relative ratio)

Yani:
- Oran < 1 → model baseline'ı yeniyor (kalın yazılır makalede)
- Oran > 1 → model baseline'ı kaybediyor

### Table 4a — Full Sample

#### Ana Mesaj

Tam örnekte iyileşmeler mütevazıdır (oranlar genellikle 0.95–1.02). Bazı ülke-frekans kombinasyonlarında istatistiksel anlamlı iyileşme var, bazılarında yok. Bu, "ortalama olarak" yön tahmininin çok keskin bir kazanım sağlamadığını ima eder.

#### Replikasyon (ABD)

| Frekans | Bsln_Abs | Npar_Abs | Ext_Abs | Bsln_Sq | Npar_Sq | Ext_Sq |
|---------|----------|----------|---------|---------|---------|--------|
| 1 mth | 0.464 | 0.989 | 0.990 | 0.456 | 1.035 | 1.041 |
| 2 mth | 0.441 | 0.991 | 0.986 | 0.424 | 1.042 | 1.048 |
| 3 mth | 0.422 | 0.982 | 0.972 | 0.405 | 1.061 | 1.078 |

### Table 4b — Low Volatility ← **Asıl Bulgu**

#### Ana Mesaj

Düşük vol alt-döneminde **çarpıcı iyileşme**. Oranlar genellikle 0.80–0.90 bandında, hatta bazı ülke-frekans kombinasyonlarında 0.75'e iniyor. Hem Brier(Abs) hem Brier(Sq) için aynı yön.

Bu, makalenin teorisini ampirik olarak doğrular:
- Düşük volatilite → koşullu olasılık 0.5'ten anlamlı şekilde sapar
- Bu sapma, kondisyonel modellerle yakalanabilir
- Baseline (sabit p) bu bilgiyi kullanamaz

#### Replikasyon (ABD)

| Frekans | Bsln_Abs | Npar_Abs | Ext_Abs | Bsln_Sq | Npar_Sq | Ext_Sq |
|---------|----------|----------|---------|---------|---------|--------|
| 1 mth | 0.404 | 0.889 | 0.886 | 0.337 | 0.861 | 0.867 |
| 2 mth | 0.375 | 0.895 | 0.869 | 0.294 | 0.897 | 0.863 |
| 3 mth | 0.351 | 0.838 | 0.804 | 0.264 | 0.813 | 0.791 |

3 aylık frekansta Extended modeli %20 daha iyi — makalenin G7 ortalamasıyla uyumlu.

### Table 4c — Medium Volatility

#### Ana Mesaj

Orta vol döneminde sonuçlar **karışıktır**. Oranlar 1'in çevresinde, bazen biraz altta bazen biraz üstte. Net bir kazanım yok. Bu "geçiş bölgesi"dir.

#### Replikasyon (ABD)

| Frekans | Bsln_Abs | Npar_Abs | Ext_Abs |
|---------|----------|----------|---------|
| 1 mth | 0.469 | 1.016 | 1.022 |
| 2 mth | 0.401 | 1.018 | 1.010 |
| 3 mth | 0.391 | 0.966 | 0.949 |

### Table 4d — High Volatility

#### Ana Mesaj

Yüksek vol döneminde modeller **baseline'ı kaybediyor**. Oranlar 1.02–1.15 bandında. Yorum:

- Yüksek vol genellikle kriz dönemlerine denk gelir
- Bu dönemlerde getiri dağılımı yapısal olarak değişir
- ARMA tabanlı vol tahmini eski yapıyı modeller; gerçek yapıyla uyuşmaz
- Quadratic mean regresyonu da kriz davranışını yakalayamaz
- Sonuç: koşullu modeller yanlış yönlendirir

#### Replikasyon (ABD)

| Frekans | Bsln_Abs | Npar_Abs | Ext_Abs |
|---------|----------|----------|---------|
| 1 mth | 0.517 | 1.042 | 1.040 |
| 2 mth | 0.542 | 1.036 | 1.046 |
| 3 mth | 0.522 | 1.091 | 1.103 |

### Neden Önemli? (Tablo 4 Bütünü)

Tablo 4'ün dört alt tablosu birlikte okunduğunda makalenin **anahtar pratik mesajı** çıkar:

> Yön tahmin modellerinin değeri rejime bağlıdır. Sakin piyasada onlara güven, çalkantılı piyasada baseline'a dön.

Bu, portföy ve risk yönetimi uygulamaları için doğrudan operasyonel bir kuraldır.

---

## Figure 1 — Realized Volatility Time Series and ACF

### Ne Gösteriyor?

2 satır × 3 sütun (3 frekans × 2 panel tipi):

- **Üst satır:** Log gerçekleşen volatilite zaman serileri
- **Alt satır:** Otokorelasyon fonksiyonu (ACF) up to lag 20

### Ana Mesaj

1. **Volatilite kümelenmesi:** Log-vol serileri hareketli ortalama etrafında dalgalanır; periyodik patlamalar (1987 Black Monday, 2000 dot-com, 2008 GFC, 2020 COVID) belirgin
2. **Yüksek persistans:** ACF lag 1'de yaklaşık 0.7–0.8, lag 5'te bile 0.4 civarı — volatilite uzun hafızalı
3. **Yavaş bozulma:** ACF üstel olmaktan çok hiperbolik bozulma sergiler (uzun hafıza işareti)

### Replikasyon Notu

Bizim ABD serisinde benzer örüntü görülür: COVID döneminde belirgin sıçrama, ardından yavaş bozulma. ACF yapısı makaledekine paralel.

### Neden Önemli?

Bu şekil ARMA modellemesinin **neden anlamlı** olduğunu görsel olarak gösterir. Yüksek otokorelasyonlar → güçlü tahmin sinyali → düşük MSPE oranı (Tablo 2'deki sonuçların temeli).

---

## Figure 2 — Volatility Forecasts vs Actual

### Ne Gösteriyor?

3 panel (her frekans için bir panel), her panel OOS dönemi boyunca:

- **Siyah çizgi:** Gerçek log-vol
- **Mavi kesikli:** AIC seçimli ARMA tahmini
- **Kırmızı noktalı:** SIC seçimli ARMA tahmini

### Ana Mesaj

1. AIC ve SIC tahminleri birbirine çok yakın
2. Tahminler gerçeği "yumuşatılmış" şekilde takip ediyor — ani sıçramaları kaçırıyor ama trendi yakalıyor
3. Kriz dönemlerinde (yüksek vol) tahmin hataları büyüyor — bu, Tablo 4d'deki yüksek-vol başarısızlığının görsel kanıtı

### Replikasyon Notu

ABD için bizim grafiğimiz makaledeki ABD grafiğine çok benzer: COVID 2020 sıçramasında belirgin tahmin hatası, 2010–2019 sakin döneminde çok düşük hata.

### Neden Önemli?

Bu şekil, **volatilite tahmin kalitesinin görsel sağlamasıdır**. Sayısal MSPE oranını (Tablo 2) somutlaştırır.

---

## Figure 3 — Predicted Sign Probabilities

### Ne Gösteriyor?

3 satır × 3 sütun = 9 panel:

- **Satırlar:** 1, 2, 3 aylık frekanslar
- **Sütunlar:** Baseline | Nonparametric | Extended

Her panelde OOS dönemi boyunca tahmin edilen $\hat{p}_{t+1|t}$ (pozitif getiri olasılığı). Nonpar ve Extended sütunlarında hem AIC (mavi) hem SIC (kırmızı noktalı) çizilir. Gri kesikli yatay çizgi 0.5 referansını gösterir.

### Ana Mesaj

1. **Baseline:** Neredeyse sabit — yalnızca yavaş yavaş (örneklem ortalaması güncellendikçe) küçük kayar. ~0.55–0.60 bandında.
2. **Nonparametric ve Extended:** Çok daha **değişken**. Volatilite yükseldikçe tahmin 0.5'e (hatta altına) iniyor; düştükçe 0.7–0.8'e yükseliyor.
3. **AIC ve SIC:** Görsel olarak çok yakın — pratik fark az.

### Replikasyon Notu

Bizim grafiğimizde aynı örüntü net görünür: 2008 ve 2020'de tahmin olasılığı belirgin şekilde aşağı iniyor; sakin dönemlerde 0.7+ seviyesine çıkıyor.

### Neden Önemli?

Bu şekil, kondisyonel modellerin **gerçek zamanlı uyaranlara** ne kadar duyarlı olduğunu gösterir. Baseline tepkisiz bir benchmark; Nonpar/Extended ise aktif bir sinyal üretiyor. Bu duyarlılığın **doğru** olup olmadığı Tablo 4 ile değerlendirilir.

---

## Figure 4 — Individual Brier Score Scatter (Düşük Volatilite)

İki alt-şekil: **4a** Nonparametric vs Baseline, **4b** Extended vs Baseline.

### Ne Gösteriyor?

Her şekilde 1×3 panel (her frekans için):

- **x ekseni:** Baseline modelin bireysel Brier(Abs) skoru
- **y ekseni:** Karşılaştırılan modelin (Nonpar veya Extended) bireysel Brier(Abs) skoru
- **Noktalar:** Sadece **düşük-vol** periyot tahminleri (1. – 33. persantil)
- **Kesikli çapraz:** 45° referans çizgisi
- **Gri yatay/dikey çizgiler:** 0.5 eşikleri (tahmin "yön doğru" mu sınırı)

### Çeyrekler

Şekil 4 paneli dört bölgeye ayrılır:

| Bölge | Anlam |
|-------|-------|
| Sol-alt (< 0.5, < 0.5) | Her iki model de doğru |
| Sağ-alt (> 0.5, < 0.5) | Baseline yanıldı, alternatif doğru |
| Sol-üst (< 0.5, > 0.5) | Baseline doğru, alternatif yanıldı |
| Sağ-üst (> 0.5, > 0.5) | Her iki model de yanıldı |

### Ana Mesaj

Eğer alternatif model baseline'dan daha iyi ise:
- Noktaların büyük çoğunluğu 45° çizgisinin **altında** olmalı (alternatif daha düşük skor → daha doğru tahmin)
- Sağ-alt çeyrekte (baseline yanıldı, alternatif doğru) yoğun nokta birikimi olmalı
- Sol-üst çeyrekte (alternatif yanıldı, baseline doğru) çok az nokta olmalı

Makaledeki düşük-vol dönemi için bu örüntü gerçekten gözlenir. Bu, Tablo 4b'deki oransal iyileşmenin **dağılım perspektifinden** kanıtıdır — ortalama değerde gizlenmiş bir avantaj değil, tahminlerin büyük kısmında sistematik bir iyileşme.

### Replikasyon Notu

ABD için bizim 4a ve 4b grafikleri tipik scatter örüntüsünü gösterir — düşük vol döneminde alternatif modeller baseline'ı çoğu noktada yener; özellikle Extended modeli 3 aylık frekansta belirgin avantaj sergiler.

### Neden Önemli?

Şekil 4, Tablo 4b'nin **görsel doğrulamasıdır**. Ortalama Brier oranı 0.85 olabilir; ama bu, "tüm tahminler %15 daha iyi" mi yoksa "yarısı çok daha iyi, yarısı eşit" mi? Scatter plot bu ayrımı yapar. Makaledeki desen, iyileşmenin **yaygın** (geniş tabanlı) olduğunu gösterir.

---

## Genel Değerlendirme

Makalenin tablo ve şekilleri birlikte okunduğunda mantık zinciri şudur:

```
Tablo 1a/1b → Veri özellikleri (getiriler non-normal, log-vol simetrik)
       ↓
Şekil 1 → Volatilite kümelenir, ACF yüksek (tahmin edilebilirlik koşulu)
       ↓
Tablo 2, Şekil 2 → ARMA log-vol tahminleri başarılı (MSPE ratio < 1)
       ↓
Şekil 3 → Bu tahminler değişken yön olasılığı üretir
       ↓
Tablo 3, Tablo 4 → Yön tahminleri düşük-vol'da baseline'ı yener, yüksek-vol'da kaybeder
       ↓
Şekil 4 → Bu kazanım dağılımsal olarak yaygın, tek tip outlier'lara bağlı değil
```

Bu zincir hem teorik (2006 makalesinden) hem ampirik (2007 makalesinin tabloları) olarak kurulur. Bizim ABD replikasyonumuz bu zincirin her halkasını **bağımsız olarak** doğrulamıştır.
