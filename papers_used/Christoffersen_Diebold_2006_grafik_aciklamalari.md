# Christoffersen & Diebold (2006) Grafik Açıklamaları

Makale: **Financial Asset Returns, Direction-of-Change Forecasting, and Volatility Dynamics**  
Yazarlar: **Peter F. Christoffersen ve Francis X. Diebold**  
Amaç: Getirilerin seviyesini tahmin etmek zor olsa bile, getirinin yönünün volatilite dinamikleri nedeniyle tahmin edilebilir olabileceğini göstermek.

Bu doküman makaledeki grafiklerin ne gösterdiğini ve ana mesajlarını tek tek açıklar.

---

## Figure 1 — The Dependence of Sign Probability on Volatility

### Ne Gösteriyor?

Bu grafik iki normal getiri dağılımını karşılaştırır:

- Aynı beklenen getiri: `mu = 10%`
- Farklı volatilite:
  - Düşük volatilite: `sigma = 5%`
  - Yüksek volatilite: `sigma = 15%`

Grafikte sıfır getiri çizgisi dikey çizgiyle gösterilir. Sıfırın sağında kalan alan, pozitif getiri olasılığıdır.

### Ana Mesaj

Beklenen getiri aynı kalsa bile volatilite arttığında pozitif getiri olasılığı azalır.

Matematiksel olarak:

```text
Pr(R > 0) = Phi(mu / sigma)
```

`mu` sabitken `sigma` yükselirse `mu / sigma` düşer. Bu da pozitif getiri olasılığını düşürür.

### Neden Önemli?

Bu grafik makalenin temel sezgisini kurar:

> Getirinin ortalaması tahmin edilemiyor olsa bile, volatilite değiştiği için getirinin işareti tahmin edilebilir olabilir.

Yani yön tahmini mutlaka değişen beklenen getiriden gelmek zorunda değildir; volatilite değişimi de yön tahmini yaratabilir.

### Proje Açısından Anlamı

Volatilite tahmini yalnızca risk ölçümü değildir. Aynı zamanda:

```text
Gelecek getirinin pozitif olma olasılığı
```

hakkında bilgi taşıyabilir. Bizim projede EWMA veya GARCH volatilite tahminlerinin yön tahmini modeline girdi olmasının temel gerekçesi budur.

---

## Figure 2 — Responsiveness of Sign Probability to Volatility Movements

### Ne Gösteriyor?

Bu grafik pozitif getiri olasılığının volatiliteye duyarlılığını gösterir.

Dikey eksen:

```text
d Pr(R > 0) / d sigma
```

Yatay eksen:

```text
mu / sigma
```

yani bilgi oranı benzeri bir ölçüdür.

### Ana Mesaj

Volatilite arttığında pozitif getiri olasılığı azalır; bu nedenle türev negatiftir.

Ancak bu duyarlılık her yerde aynı değildir. En güçlü duyarlılık yaklaşık:

```text
mu / sigma = sqrt(2) ≈ 1.41
```

civarında ortaya çıkar.

### Neden Önemli?

Çok düşük bilgi oranında, beklenen getiri volatiliteye göre çok küçüktür. Bu durumda pozitif getiri olasılığı zaten yaklaşık `0.5` civarındadır ve volatilite değişimi fazla etki yaratmaz.

Çok yüksek bilgi oranında ise pozitif getiri olasılığı zaten `1`e yakındır. Bu durumda da volatilite değişimi fazla etki yaratmaz.

En güçlü yön tahmini etkisi ara bölgede ortaya çıkar.

### Proje Açısından Anlamı

Volatilite sinyalinin her dönemde aynı güçte çalışmasını beklememek gerekir. Bazı piyasa koşullarında volatilite yön tahmini için çok bilgilendirici olabilir; bazı koşullarda ise etkisi zayıf kalabilir.

Bu, bizim projede rejim katmanının neden faydalı olabileceğini destekler:

```text
Volatilite sinyali farklı rejimlerde farklı güçte çalışabilir.
```

---

## Figure 3 — Time Series of Conditional Sign Probabilities

### Ne Gösteriyor?

Bu grafik Heston stokastik volatilite modeliyle simüle edilen bir örnek yol üzerinde, farklı horizonlarda pozitif getiri olasılıklarını gösterir.

Horizonlar:

- Günlük
- Haftalık
- Aylık
- Çeyreklik
- Altı aylık
- Yıllık

Her panelde çizilen seri:

```text
Pr_t(R_{t+1:t+h} > 0)
```

yani `t` anındaki bilgiye göre gelecek `h` dönemlik getirinin pozitif olma olasılığıdır.

### Ana Mesaj

Pozitif getiri olasılığı zaman içinde sabit değildir. Volatilite değiştikçe yön olasılığı da değişir.

Fakat bu değişkenlik horizonlara göre farklıdır:

- Günlük horizonlarda olasılık çok az oynar.
- Orta vadelerde olasılık daha belirgin oynar.
- Çok uzun vadelerde oynaklık tekrar azalabilir.

### Neden Önemli?

Bu grafik makalenin horizon argümanını görsel olarak başlatır:

> Yön tahmini gücü çok kısa veya çok uzun vadede değil, daha çok orta vadede ortaya çıkabilir.

### Proje Açısından Anlamı

Bizim projede 1 ay, 2 ay, 3 ay ve 6 ay horizonlarının ayrı ayrı test edilmesi gerekir. Özellikle 40-60 işlem günü civarı kritik olabilir.

Bu grafik, “neden tek günlük al-sat sinyali yerine orta vadeli yön tahmini?” sorusunun ilk cevabıdır.

---

## Figure 4 — Correlation Between Sign Forecasts and Realizations Across Horizons

### Ne Gösteriyor?

Bu grafik farklı beklenen getiri parametreleri için, tahmin edilen yön olasılığı ile gerçekleşen yön arasındaki korelasyonu horizon bazında gösterir.

Karşılaştırılan beklenen getiri değerleri:

- `mu = 0.10`
- `mu = 0.05`
- `mu = 0.00`

Dikey eksen:

```text
Corr(sign forecast, realized sign)
```

Yatay eksen:

```text
Forecast horizon
```

### Ana Mesaj

Beklenen getiri sıfıra yaklaştıkça yön tahmin edilebilirliği zayıflar.

`mu = 0.10` için korelasyon en yüksektir. `mu = 0.05` için daha düşüktür. `mu = 0` için ise neredeyse yoktur.

Ayrıca korelasyon horizonla birlikte önce artar, sonra düşer. Yani şekil “humped”dır.

### Neden Önemli?

Volatilite tek başına yeterli değildir. Volatilite dinamiklerinin yön tahmini yaratabilmesi için beklenen getirinin sıfırdan farklı olması gerekir.

Makalenin teorik koşulu şudur:

```text
mu ≠ 0
sigma_t değişken
```

Bu iki koşul birlikte olduğunda sign predictability ortaya çıkar.

### Proje Açısından Anlamı

Model yalnızca volatiliteye bakmamalı; hangi varlıkta veya sektörde pozitif risk primi bulunduğu da önemlidir.

Bu yüzden sektör seçimi veya rejim bazlı beklenen getiri farkları projeye mantıklı şekilde eklenebilir.

---

## Figure 5 — Correlation Between Sign Forecasts and Realizations Across Horizons: Volatility Persistence

### Ne Gösteriyor?

Bu grafik volatilite kalıcılığı farklı olduğunda yön tahmin edilebilirliğinin nasıl değiştiğini gösterir.

Heston modelinde kalıcılık parametresi:

```text
kappa
```

ile temsil edilir.

Karşılaştırılan değerler:

- `kappa = 2`
- `kappa = 5`
- `kappa = 10`

Düşük `kappa`, volatilitenin daha kalıcı olduğu anlamına gelir. Yüksek `kappa`, volatilitenin daha hızlı ortalamaya döndüğü anlamına gelir.

### Ana Mesaj

Volatilite daha kalıcı olduğunda yön tahmin edilebilirliği daha güçlüdür.

Grafikte `kappa = 2` en yüksek korelasyonu üretir. `kappa = 10` ise en düşük korelasyonu üretir.

### Neden Önemli?

Yön tahmini volatilite dinamiklerinden geliyorsa, volatilitenin tahmin edilebilir olması gerekir.

Volatilite hızlıca ortalamaya dönüyorsa bugünkü volatilite gelecekteki volatilite hakkında daha az bilgi verir. Bu durumda yön tahmini gücü de azalır.

### Proje Açısından Anlamı

Rejim tahmini bu noktada önemlidir. Eğer HMM rejimleri yüksek kalıcılık gösteriyorsa, bu makalenin argümanıyla uyumludur.

Özellikle:

```text
yüksek volatilite rejimi kalıcı mı?
düşük volatilite rejimi kalıcı mı?
```

soruları yön tahmini başarısı için önemlidir.

---

## Figure 6 — Forecast Correlation and First Autocorrelation of Return Signs

### Ne Gösteriyor?

Bu grafik iki şeyi karşılaştırır:

1. Volatilite temelli yön tahmininin gerçekleşen yönle korelasyonu
2. Getiri işaretlerinin kendi birinci derece otokorelasyonu

Yani grafik şunu sorar:

```text
Geçmiş işaret mi daha iyi bilgi taşır,
yoksa volatilite temelli olasılık tahmini mi?
```

### Ana Mesaj

Volatilite temelli yön tahmini, basit sign autocorrelation ölçüsünden daha güçlüdür.

Makaledeki grafikte forecast-realization correlation çizgisi, sign autocorrelation çizgisinin üzerinde kalır.

### Neden Önemli?

Bu grafik, klasik doğrusal yöntemlerin yön tahmin edilebilirliğini yakalamakta yetersiz kalabileceğini gösterir.

Sign autocorrelation düşük olabilir, ama bu yön tahmini yok demek değildir. Çünkü yön tahmini doğrusal otokorelasyondan değil, volatilite dinamiklerinden doğabilir.

### Proje Açısından Anlamı

Projede yalnızca:

```text
geçmiş getiri pozitif miydi?
geçmiş işaret devam ediyor mu?
```

gibi basit momentum/sign autocorrelation testleri yeterli değildir.

Bunun yerine volatilite tahminini ve rejim bilgisini kullanmak daha tutarlı olur.

---

## Figure 7 — Responsiveness in the Heston Stochastic Volatility Model

### Ne Gösteriyor?

Bu grafik Figure 2’deki duyarlılık analizini daha gerçekçi bir stokastik volatilite ortamına taşır.

Dikey eksen:

```text
d Pr(R > 0) / d sigma(t)
```

Yatay eksen:

```text
mu / sigma_{t+tau|t}
```

Burada `sigma_{t+tau|t}`, forecast horizon boyunca beklenen ortalama volatiliteyi temsil eder.

### Ana Mesaj

Figure 2’deki sonuç Heston modelinde de geçerlidir:

- Çok düşük volatilite bölgesinde pozitif getiri olasılığı zaten yüksektir.
- Çok yüksek volatilite bölgesinde pozitif getiri olasılığı `0.5`e yaklaşır.
- En yüksek duyarlılık ara bölgede ortaya çıkar.

### Neden Önemli?

Makale sadece basit normal dağılım sezgisine dayanmıyor. Daha gerçekçi, zamana göre değişen volatilite modelinde de aynı temel mekanizma çalışıyor.

### Proje Açısından Anlamı

Volatilite-yön ilişkisi yalnızca teorik bir normal dağılım sonucu değildir. Rejim değişimi ve volatilite kalıcılığı olan daha gerçekçi ortamlarda da anlamlı olabilir.

Bu, HMM veya GARCH gibi modelleri projeye dahil etmeyi teorik olarak güçlendirir.

---

## Figure 8 — Daily RiskMetrics Volatility

### Ne Gösteriyor?

Bu grafik S&P 500 için günlük yıllıklandırılmış RiskMetrics volatilitesini gösterir.

RiskMetrics formülü:

```text
sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
lambda = 0.94
```

Grafik 1963-2003 dönemi için volatilitenin zaman içinde nasıl değiştiğini gösterir.

### Ana Mesaj

Volatilite çok belirgin şekilde zamanla değişir ve kalıcıdır.

Özellikle kriz dönemlerinde volatilite sert yükselir:

- 1987 crash
- 1990 civarı stres
- 1998-2002 dönemi

### Neden Önemli?

Makalenin ampirik uygulaması için temel girdi budur. Eğer volatilite zamanla güçlü şekilde değişmiyor olsaydı, volatilite tabanlı yön tahmini de anlamlı olmazdı.

### Proje Açısından Anlamı

Bu grafik bizim projede volatilite tahmini katmanının neden gerekli olduğunu açıklar.

RiskMetrics/EWMA basit ama etkili bir başlangıçtır. Sonrasında GARCH gibi modeller alternatif olarak denenebilir.

---

## Figure 9 — Conditional Probability Forecasts Across Horizons

### Ne Gösteriyor?

Bu grafik S&P 500 için farklı horizonlarda pozitif getiri olasılığı tahminlerini gösterir.

Horizonlar:

- Günlük
- Haftalık
- Aylık
- Çeyreklik
- Altı aylık
- Yıllık

Model:

```text
logit Pr(R_{t+1:t+h} > 0) = alpha_h + beta_h * (1 / sigma_t)
```

Her horizon için ayrı logit modeli tahmin edilir.

### Ana Mesaj

Koşullu pozitif getiri olasılıkları zaman içinde dalgalanır.

Bu dalgalanma özellikle orta ve uzun vadelerde daha belirgin hale gelir.

Grafik ayrıca koşulsuz pozitif getiri olasılıklarını yatay çizgiyle gösterir. Horizon uzadıkça koşulsuz pozitif getiri olasılığı artar.

### Neden Önemli?

Bu grafik makalenin ampirik kanıtlarından biridir:

> Volatilite tahmini kullanılarak gelecekteki getirinin pozitif olma olasılığı zaman içinde tahmin edilebilir şekilde değişmektedir.

Bu, getirinin seviyesini tahmin etmekten farklıdır. Burada amaç:

```text
getiri kaç olacak?
```

değil,

```text
getiri pozitif mi olacak?
```

sorusudur.

### Proje Açısından Anlamı

Bizim yön tahmini modelimizin doğrudan karşılığı bu grafiktir.

Projede her gün şu olasılık üretilebilir:

```text
p_t = Pr(R_{t+1:t+h} > 0)
```

Sonra bu olasılık portföy kararına dönüştürülebilir:

```text
p_t yüksekse riskli varlık ağırlığı artırılır
p_t düşükse nakit / tahvil / defansif pozisyon artırılır
```

---

## Figure 10 — Forecast Correlation and Return Sign Autocorrelation

### Ne Gösteriyor?

Bu grafik S&P 500 ampirik uygulamasında iki ölçüyü horizon bazında karşılaştırır:

1. Volatilite temelli yön tahmini ile gerçekleşen yön arasındaki korelasyon
2. Getiri işaretlerinin birinci derece otokorelasyonu

Dikey eksen korelasyon, yatay eksen horizon uzunluğudur.

### Ana Mesaj

Makaledeki sonuç şudur:

- Forecast-realization correlation orta vadelerde yükselir.
- Sign autocorrelation daha düşüktür.
- Her iki seri de humped pattern gösterir.
- Yön tahmini gücü özellikle orta vadede belirgindir.

Bu grafik makalenin en önemli ampirik sonucudur.

### Neden Önemli?

Figure 10, teorik ve simülasyon sonuçlarının S&P 500 verisinde de karşılık bulduğunu gösterir.

Yani makale yalnızca model içinde değil, gerçek piyasa verisinde de şunu savunur:

```text
volatilite dinamikleri yön tahmini için bilgi taşır.
```

Ayrıca basit sign autocorrelation testlerinin bu ilişkiyi tam yakalayamayacağını gösterir.

### Proje Açısından Anlamı

Bu grafik bizim proje için ana motivasyonlardan biridir.

Proje şu soruyu buradan alır:

```text
Volatilite tabanlı yön tahmini, rejim ve sektör bilgisiyle güçlendirilebilir mi?
```

Makale yalnızca yön olasılığını inceler. Bizim proje ise bunu bir sonraki adıma taşıyıp portföy/strateji kararına dönüştürmeyi hedefler.

---

## Genel Sonuç

Makaledeki grafikler birlikte şu hikayeyi kurar:

1. Volatilite arttığında pozitif getiri olasılığı değişir.
2. Bu ilişki basit normal dağılımda da, Heston gibi daha gerçekçi volatilite modellerinde de görülür.
3. Yön tahmin gücü horizonlara göre değişir ve genellikle orta vadede daha güçlüdür.
4. Basit sign autocorrelation testleri volatilite kaynaklı yön tahmin edilebilirliğini tam yakalayamaz.
5. S&P 500 uygulaması, teorik mekanizmanın gerçek veride de gözlenebileceğini gösterir.

Bu nedenle makalenin proje açısından ana mesajı şudur:

> Volatilite tahminleri yalnızca risk yönetimi için değil, orta vadeli yön tahmini için de kullanılabilir. Bu yön tahmini rejim ve sektör katmanlarıyla birleştirilerek trading stratejisine dönüştürülebilir.
