# Makale Özeti: Financial Asset Returns, Direction-of-Change Forecasting, and Volatility Dynamics

**Yazarlar:** Peter F. Christoffersen (McGill University) & Francis X. Diebold (University of Pennsylvania)  
**Yayın:** Management Science, Cilt 52, Sayı 8, Ağustos 2006, ss. 1273–1287  
**DOI:** 10.1287/mnsc.1060.0520

---

## 1. Genel Çerçeve ve Motivasyon

Bu makale, finansal ekonomi literatüründe sıklıkla tartışılan üç temel olguyu ele alır ve aralarındaki ilişkileri hem teorik hem de ampirik olarak inceler:

1. **Koşullu Ortalama Bağımsızlığı (Conditional Mean Independence):** Varlık getirilerinin koşullu ortalaması bilgi kümesine göre anlamlı biçimde değişmez; bu nedenle getirilerin kendisi büyük ölçüde tahmin edilemez. Fama (1970, 1991) ve geniş ampirik literatür bu görüşü destekler.

2. **İşaret Bağımlılığı (Sign Dependence):** Getiri *yönü* (pozitif mi, negatif mi?) istatistiksel olarak tahmin edilebilir olabilir. Getirilerin kendisini tahmin etmek yerine, yalnızca yönünü doğru tahmin etmek kârlı trading stratejileri üretebilir.

3. **Volatilite Bağımlılığı (Volatility Dependence):** Varlık getirilerinin koşullu varyansı (volatilitesi) zaman içinde değişir ve GARCH, stokastik volatilite modelleri gibi araçlarla öngörülebilir. Bu olgu literatürde son derece güçlü biçimde belgelenmiştir.

Makalenin temel iddiası şudur: Bu üç olgu birbirinden bağımsız değildir; bilhassa volatilite dinamikleri, getiri işaret öngörülebilirliğini doğrudan tetikler.

---

## 2. Temel Teorik Bulgular

### 2.1. Volatilite Dinamikleri → İşaret Dinamikleri

En basit durumda, bir varlığın getirisi koşullu olarak normal dağıldığı varsayılsın:

$$R_{t+1} \mid \Omega_t \sim \mathcal{N}(\mu, \sigma_{t+1|t}^2), \quad \mu > 0$$

Burada koşullu *ortalama* sabittir (yani ortalama bağımsızlığı varsayılır), ancak koşullu *varyans* zamanla değişmektedir. Pozitif getiri olasılığı:

$$\Pr_t(R_{t+1} > 0) = \Phi\left(\frac{\mu}{\sigma_{t+1|t}}\right)$$

Bu ifade zaman içinde değişir çünkü $\sigma_{t+1|t}$ değişmektedir. Sonuç şaşırtıcıdır: **koşullu ortalama sabit olsa bile, dağılımın simetrisi bozulmasa bile, getirinin işareti tahmin edilebilirdir.** Mekanizma şöyledir:

- Volatilite yükseldikçe, pozitif getiri olasılığı düşer (beklenen getiri pozitifken).
- Volatilite düştükçe, pozitif getiri olasılığı artar.

Bu ilişki, sıfırdan farklı bir beklenen getiri ($\mu \neq 0$) ve değişken volatilite ($\sigma$ sabit değil) olmak üzere iki koşulun birlikte var olması durumunda geçerlidir.

### 2.2. İşaret Bağımlılığı, Ortalama Bağımlılığı Gerektirmez

İşaret tahmin edilebilirliğinin kaynağı koşullu ortalamadaki değişkenlik olmak zorunda değildir. Volatilite dinamiklerinin varlığı, beklenen getiri sabit kalsa dahi işaret öngörülebilirliği yaratmak için yeterlidir. Bu bulgu, geleneksel yoruma karşı güçlü bir argümandır: ampirik çalışmalarda işaret bağımlılığı bulunduğunda, bunun mutlaka değişken risk primlerinden kaynaklandığı sonucuna varılmamalıdır.

### 2.3. Getiri Ayrıştırması

Makale şu çarpıcı ayrıştırmayı vurgular:

$$R_{t+1} = \text{sign}(R_{t+1}) \cdot |R_{t+1}|$$

Hem `sign(R)` (getiri işareti) hem de `|R|` (getirinin mutlak değeri) koşullu ortalama bağımlılığı gösterir ve dolayısıyla tahmin edilebilirdir. Ancak bunların çarpımı olan `R` kendisi koşullu ortalamadan bağımsız olabilir. Bu, Engle ve Kozicki (1993)'nin "ortak özellik" (common feature) kavramının doğrusal olmayan bir örneğidir.

---

## 3. İşaret Öngörülebilirliğinin Ölçülmesi

### 3.1. İşaret Tahminlerinin Volatiliteye Duyarlılığı

İşaret olasılığının volatiliteye göre türevi (duyarlılık, $\mathcal{R}$):

$$\mathcal{R} = \frac{d\Pr_t(R_{t+1}>0)}{d\sigma_{t+1|t}} = -f\!\left(\frac{\mu}{\sigma}\right)\cdot\frac{\mu}{\sigma^2}$$

Bu türev **her zaman negatiftir** (volatilite artınca pozitif getiri olasılığı düşer). Önemli bir nokta: duyarlılık, bilgi oranı $\mu/\sigma$'ya göre **monoton değildir**; $\mu/\sigma \approx \sqrt{2} = 1.41$ değerinde minimum (maksimum mutlak duyarlılık) elde edilir. Çok düşük ya da çok yüksek bilgi oranlarında duyarlılık sıfıra yaklaşır.

### 3.2. Tahmin-Gerçekleşme Korelasyonu

İşaret tahmini $P_{t+1|t}$ ile gerçekleşen işaret $I_{t+1}$ arasındaki kovaryans:

$$\text{Cov}(I_{t+1}, P_{t+1|t}) = \text{Var}(P_{t+1|t})$$

Bu da korelasyona dönüştürüldüğünde:

$$\text{Corr}(I_{t+1}, P_{t+1|t}) = \frac{\text{Std}(P_{t+1|t})}{\sqrt{P(1-P)}}$$

Sonuç: İşaret tahmininin gerçekleşmeyle korelasyonu, yalnızca **tahmin varyansına** bağlıdır. Tahmin varyansı ise volatilitenin volatilitesine (vol-of-vol) bağlıdır. Dolayısıyla **işaret öngörülebilirliği, vol-of-vol tarafından yönlendirilir.**

---

## 4. İşaret Öngörülebilirliğinin Tespiti: Geleneksel Yöntemlerin Yetersizliği

### 4.1. İşaret Otokorelasyonları

İşaret serisi $I_t$'nin kendi gecikmesiyle korelasyonu, optimal işaret tahmini $P_{t+1|t}$'nin gerçekleşmeyle korelasyonundan **her zaman daha düşüktür**:

$$\text{Corr}(I_{t+1}, I_t) \leq \text{Corr}(P_{t+1|t}, I_{t+1})$$

Bunun nedeni: $I_t$ (geçmiş işaret), $P_{t+1|t}$'yi (yarınki volatilite temelli olasılık) iyi öngöremez. İşaret otokorelasyonları bu nedenle işaret öngörülebilirliğini yakalamakta yetersiz kalır.

### 4.2. Koşu Testleri (Runs Tests)

Bir koşu (run), ardışık aynı değerlerin (0-1) dizisidir. Matematiksel gösterim:

$$N_{\text{runs}} = 1 + 2\hat{P}(1-\hat{P}) \cdot T \cdot (1 - \hat{\rho}_1)$$

Buradan görülür ki koşu testlerindeki bilgi, birinci derece işaret otokorelasyonuyla ($\hat{\rho}_1$) tam olarak örtüşmektedir. **Koşu testleri, otokorelasyon testlerine göre ek hiçbir bilgi içermez.** Her iki yöntem de volatilite dinamiklerinden kaynaklanan nonlineer işaret bağımlılığını tespit edemez.

### 4.3. Geleneksel Piyasa Zamanlama Testleri

Henriksson-Merton (1981) ve Breen vd. (1989) gibi klasik testler, olasılık tahminini yalnızca bir eşik (0.5) üzerinden işler:

$$I(R_{t+1} > 0) = a + b \cdot I(P_{t+1|t} > 0.5) + \varepsilon$$

**Kritik sorun:** Volatilite kaynaklı işaret bağımlılığında, beklenen getiri pozitif ($\mu > 0$) ve dağılım simetrikse, $P_{t+1|t}$ her zaman 0.5'in üzerinde kalır. Bu durumda $I(P_{t+1|t} > 0.5) = 1$ her zaman geçerlidir ve test hiçbir güce sahip olamaz. **Geleneksel piyasa zamanlama testleri ancak beklenen getiri değişkenliğinden kaynaklanan işaret bağımlılığını tespit edebilir; volatilite kaynaklı olanı tespit edemez.**

---

## 5. Simülasyon Deneyi: Farklı Horizonlarda İşaret Öngörülebilirliği

### 5.1. Heston (1993) Stokastik Volatilite Modeli

Simülasyon için sürekli zamanlı Heston modeli kullanılır:

$$dS(t) = \mu S\, dt + \sigma(t) S\, dz_1$$
$$d\sigma^2(t) = \kappa(\theta - \sigma^2(t))\, dt + \eta\, \sigma(t)\, dz_2$$
$$\text{Corr}(dz_1, dz_2) = \rho$$

Parametreler:
- $\mu = 0.10$ (yıllık beklenen getiri)
- $\kappa = 2$ (ortalamaya dönüş hızı)
- $\theta = 0.015$ (uzun dönem varyans)
- $\eta = 0.15$ (volatilitenin volatilitesi)
- $\rho = -0.50$ (kaldıraç etkisi)

Bu parametreler yaklaşık olarak günlük ortalama %0.037, günlük standart sapma %0.77 ve negatif çarpıklık üretiyor. İşaret olasılığı tahminleri, karakteristik fonksiyon ters çevirme (Fourier inversion) yöntemiyle hesaplanır.

### 5.2. Horizon Analizi: Temel Bulgu

Makale, farklı horizonlarda (günlük, haftalık, aylık, çeyreklik, altı aylık, yıllık) işaret öngörülebilirliğini inceler. Temel çatışma şöyledir:

| Horizon | Volatilite Öngörülebilirliği | Beklenen Getiri | İşaret Öngörülebilirliği |
|---------|------------------------------|-----------------|--------------------------|
| Çok kısa (günlük) | Çok yüksek | İhmal edilebilir | Düşük |
| Orta (2-3 ay) | Orta | Anlamlı | **Maksimum** |
| Çok uzun (yıllık) | İhmal edilebilir | Yüksek | Düşük |

**Ana sonuç:** İşaret öngörülebilirliği, yaklaşık **40-60 işlem günü (2-3 ay)** horizonunda en yüksek seviyeye ulaşır. Bu, volatilite öngörülebilirliğinin azalması ile kümülatif beklenen getirinin artması arasındaki optimal dengenin bulunduğu noktadır.

### 5.3. Volatilite Kalıcılığının Etkisi

Volatilite kalıcılığı azaldıkça ($\kappa$ arttıkça), tüm horizonlarda işaret öngörülebilirliği azalır. Yüksek volatilite kalıcılığı, işaret öngörülebilirliği için önemli bir koşuldur.

### 5.4. İşaret Otokorelasyonu vs. Tahmin-Gerçekleşme Korelasyonu

Simülasyonlar, tüm horizonlarda:

$$\text{Corr}(I_{t+1}, I_t) \ll \text{Corr}(P_{t+1|t}, I_{t+1})$$

olduğunu kanıtlar. İşaret otokorelasyonu son derece küçükken, volatilite temelli tahmin çok daha yüksek bir korelasyona sahiptir. Bu, **işaretlerin doğrusal araçlarla değil, volatilite modelleriyle tahmin edilmesi gerektiğini** gösterir.

---

## 6. Ampirik Uygulama: S&P 500 (1963–2003)

### 6.1. Veri ve Yöntem

- **Veri:** S&P 500 endeksi (CRSP), 1 Ocak 1963 – 31 Aralık 2003
- **Volatilite tahmini:** RiskMetrics yaklaşımı (EWMA, $\lambda = 0.94$)
- **Model:** Her horizon $h$ için ayrı bir logit modeli:

$$\Pr(R_{t+1:t+h} > 0) = \text{logit}\!\left(\frac{\mu_h}{\sigma_t}\right)$$

  5 yıllık kayan pencere ile yaklaşık 100.000 logit modeli tahmin edilmiştir.

### 6.2. Bulgular

- Koşullu işaret olasılıkları, tüm horizonlarda koşulsuz olasılıkların etrafında geniş ve kalıcı dalgalanmalar göstermektedir. Bu dalgalanmalar simülasyondakinden bile daha belirgindir (gerçek S&P 500 volatilitesinin Heston modelinden daha karmaşık olduğunu yansıtır).
- Tahmin-gerçekleşme korelasyonu ile işaret otokorelasyonu arasındaki **çan eğrisi (humped)** deseni teorik öngörülerle örtüşür: ikisi de önce artar, sonra azalır; tahmin korelasyonu otokorelasyonun sürekli üzerinde kalır.
- Hem korelasyon hem de otokorelasyon değerleri simülasyondan belirgin şekilde yüksektir; bu, S&P 500'deki gerçek volatilite dinamiklerinin Heston modelinin varsaydığından daha zengin olduğuna işaret eder.

---

## 7. Sonuç ve Katkı

### 7.1. Ana Katkılar

1. **Teorik katkı:** Volatilite dinamiklerinin, beklenen getiri sabit olsa dahi, getiri işaret öngörülebilirliğini matematiksel olarak nasıl ürettiğini kanıtlamıştır.

2. **Metodolojik katkı:** Geleneksel tespit araçlarının (otokorelasyon, koşu testleri, piyasa zamanlama testleri) volatilite kaynaklı işaret bağımlılığını neden yakalayamadığını göstermiştir. Doğru yaklaşım, *doğrusal olmayan, volatilite temelli* modellerdir.

3. **Ampirik katkı:** S&P 500 üzerinde gerçekleştirilen geniş ölçekli logit tahminleriyle teorik bulguları doğrulamıştır.

### 7.2. Gelecek Araştırma Yönleri

- İşaret tahminlerine dayalı trading stratejileri geliştirmek (örn. dijital opsiyonlar) ve risk-düzeltilmiş getirilerini ölçmek.
- Volatilite zamanlama stratejileriyle (Fleming vd. 2001, 2003) entegrasyon.
- Koşullu çarpıklık ve basıklık dinamiklerini (El Babsiri ve Zakoian 2001) modele dahil ederek "moment zamanlama" stratejileri.
- En yüksek doğrusal olmayan tahmin edilebilirliğe sahip fonksiyon $f(R)$'yi parametrik olmayan yollarla tahmin etmek.

---

## 8. EC581 Projesi ile İlişkisi

Bu makale, EC581 projesindeki **HMM tabanlı rejim belirleme ve portföy tahsisi** çerçevesiyle doğrudan ilgilidir:

- **Volatilite rejimleri ve işaret öngörülebilirliği:** Makale, yüksek volatilite rejimlerinde negatif getiri olasılığının arttığını ve bu bilginin portföy tahsisinde kullanılabileceğini kanıtlamaktadır. HMM modelleri tam da bu volatilite rejimlerini yakalamak için tasarlanmıştır.

- **Orta vadeli yeniden dengeleme:** İşaret öngörülebilirliğinin 2-3 aylık horizonlarda zirveye ulaşması, rejim tabanlı portföylerin yeniden dengeleme sıklığına (optimal rebalancing frequency) ilişkin teorik bir dayanak sunar.

- **Doğrusal olmayan bağımlılık:** Makalenin doğrusal araçların (otokorelasyon, runs testleri) yetersiz kalacağına dair vurgusu, HMM gibi nonlineer, rejim-anahtarlamalı modellerin tercih edilmesi gerektiğini desteklemektedir.

- **Volatilite kalıcılığı:** HMM modellerindeki rejim geçiş matrislerinin yüksek kalıcılık (düşük geçiş olasılıkları) göstermesi, bu makaledeki volatilite kalıcılığı argümanıyla örtüşür.

---

## Referanslar (Seçilmiş Temel Kaynaklar)

- Heston, S.L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6, 327–343.
- Bollerslev, T., Chou, R.Y., Kroner, K.F. (1992). ARCH modeling in finance. *Journal of Econometrics*, 52, 5–59.
- Henriksson, R.D., Merton, R.C. (1981). On market timing and investment performance II. *Journal of Business*, 54, 513–533.
- Breen, W., Glosten, L.R., Jagannathan, R. (1989). Economic significance of predictable variations in stock index returns. *Journal of Finance*, 44, 1177–1189.
- Fama, E.F. (1970). Efficient capital markets. *Journal of Finance*, 25, 383–417.
- Fleming, J., Kirby, C., Ostdiek, B. (2001). The economic value of volatility timing. *Journal of Finance*, 56, 329–352.
