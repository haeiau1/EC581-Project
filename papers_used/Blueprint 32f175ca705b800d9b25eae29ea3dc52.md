# Blueprint

# Volatilite Tabanlı Yön Tahmini ile Makro Rejim ve Sektör Analizi İçeren Orta Vadeli Trading Stratejisi

## Projenin Amacı ve Motivasyonu

 İncelediğimiz makale de volatilite tahminlerinin gelecekteki getirilerin işaretini (pozitif veya negatif olması) tahmin etmede kullanılabileceğini göstermektedir. Makalenin ampirik sonuçlarına göre özellikle çok kısa veya çok uzun vadelerde değil, **orta vadelerde (yaklaşık 2–3 ay civarında)** bu tür bir yön tahmininin daha belirgin hale geldiği gözlemlenmektedir.

Bu proje söz konusu fikri temel alarak volatilite temelli yön tahminini daha geniş bir yatırım çerçevesi içine yerleştirmeyi amaçlamaktadır. Özellikle volatilite tahminlerinden elde edilen yön sinyallerinin makroekonomik rejim analizi ve sektör seçimi ile birlikte kullanılması durumunda daha güçlü bir yatırım stratejisi oluşturulup oluşturulamayacağı araştırılacaktır. Böylece çalışma hem akademik literatürdeki bulguları test etmeyi hem de pratikte uygulanabilir bir trading stratejisi geliştirmeyi hedeflemektedir.

---

## Araştırma Sorusu

Bu projenin temel araştırma sorusu şu şekilde ifade edilebilir:

**Volatilite tahminlerine dayalı yön tahmini, makroekonomik rejim ve sektör seçimi ile birlikte kullanıldığında orta vadeli yatırım performansını artırabilir mi?**

Bu ana soruya ek olarak aşağıdaki alt sorular da incelenecektir:

- Volatilite temelli yön tahmini farklı ekonomik rejimlerde farklı performans gösterir mi?
- Belirli sektörlerde volatiliteye dayalı yön tahmini daha güçlü olabilir mi?
- Hangi yatırım vadelerinde (örneğin 1 ay, 2 ay, 3 ay) tahmin gücü daha yüksek ortaya çıkmaktadır?

Bu sorulara verilecek cevaplar hem makalenin bulgularını doğrulamaya hem de daha kapsamlı bir yatırım çerçevesi oluşturulmasına katkı sağlayacaktır.

---

## Stratejinin Genel Yapısı

Önerilen yatırım stratejisi üç ana katmandan oluşmaktadır. Bu katmanlar sırasıyla makroekonomik ortamın belirlenmesi, sektör bazında fırsatların tespit edilmesi ve son olarak volatilite tabanlı yön tahmininin kullanılmasıdır. Bu yapı şu şekilde özetlenebilir:

```
Makro Rejim Tespiti
        ↓
Sektör Seçimi
        ↓
Volatilite Tabanlı Yön Tahmini
        ↓
Portföy Oluşturma
```

---

## Makroekonomik Rejim Tespiti

Başlangıç aşamasında üç farklı ekonomik rejim tanımlanacaktır:

1. **Genişleme / Risk-On dönemi**
2. **Geçiş veya nötr dönem**
3. **Stres / Risk-Off dönemi**

Ekonomik rejimi tespit etmek için kullanılabilecek yöntemler/alternatifler:

### Alternatif A: Ekonomik mantıkla kural bazlı rejim

Bu en sade versiyon.

- VIX düşükse
- yield curve çok bozulmamışsa
- kredi spreadleri sakin ise
- piyasa momentumu pozitifse

→ risk-on

Avantajı:

Sunumu kolay, anlaşılır, açıklaması güçlü.

Dezavantajı:

Rejimleri siz tanımladığınız için biraz “el yapımı” kalabilir.

### Alternatif B: Hidden Markov Model

Burada modele doğrudan “3 tane gizli rejim öğren” dersiniz. Girdi olarak VIX, term spread, credit spread, market return, realized volatility gibi seriler verirsiniz. Model, bunların arkasında yatan gizli durumları keşfeder.

Avantajı:

- veri kendi rejimini kısmen kendi çıkarır
- econometrics tarafı kuvvetlidir
- “latent state” fikri profesörlerce genelde beğenilir

Dezavantajı:

- yorumlama bazen zorlaşır
- hangi state’in ne olduğunu sonradan isimlendirmeniz gerekir

### Alternatif C: Clustering / regime discovery

Burada k-means, Gaussian Mixture gibi yöntemlerle gözlemleri kümelersiniz.

Avantajı:

- hızlı prototipleme
- basit deneme yapmak için iyi

Dezavantajı:

- zaman serisi doğasını tam kullanmaz
- “rejim” yerine bazen sadece “benzer gün kümeleri” yakalar

### Ben profesör gibi tavsiye verecek olsam

İlk prototip için:

- önce **kural bazlı**
- sonra **HMM**

Yani önce ekonomik sezgiyle çalışan bir baseline kurun. Sonra “şimdi bunu latent-state model ile geliştiriyoruz” deyin. Bu çok güçlü bir araştırma hikâyesi olur.

---

## Sektör Seçimi

Makroekonomik ortam belirlendikten sonra ikinci adım sektör bazında yatırım fırsatlarını tespit etmektir. Finansal piyasalarda farklı sektörlerin farklı ekonomik dönemlerde daha iyi performans göstermesi yaygın bir gözlemdir. Bu nedenle strateji yalnızca piyasa yönünü tahmin etmek yerine aynı zamanda hangi sektörlerin göreli olarak daha güçlü olduğunu belirlemeyi hedeflemektedir.

Alternatifler/Yöntemler:

## Alternatif A: Relative Strength yaklaşımı

Bu en doğal yaklaşım.

Her sektör için diyorsunuz ki:

- Sektör getirisi piyasa getirisinin üstünde mi?
- Son 4–12 haftada momentum nasıl?
- Volatiliteye göre düzeltilmiş performansı nasıl?

Mesela teknoloji sektörü piyasa yatayken bile güçlü olabilir. Bu durumda siz diyorsunuz:

“Makro ortam orta halli ama teknoloji sektörü relative olarak güçlü. Demek ki fırsat burada.”

Avantajı:

- anlaşılır
- backtesti kolay
- ETF’lerle uygulanabilir

Dezavantajı:

- bazen geç sinyal verir
- yapısal nedenleri açıklamak zor olabilir

## Alternatif B: Rejim-uyumluluk skoru

Burada her sektörün belli rejimlerde sistematik olarak daha iyi performans verip vermediğine bakarsınız.

Örneğin:

- risk-on → tech, industrials, consumer discretionary
- risk-off → utilities, healthcare, staples
- inflationary → energy, materials

Bu ilişkiyi veriyle ölçersiniz ve her rejimde sektör skorları çıkarırsınız.

Avantajı:

- ekonomik anlatısı daha güçlü
- “sector rotation” literatürüyle uyumlu

Dezavantajı:

- rejim tanımı kötüyse sektör katmanı da bozulur

## Alternatif C: Pure ranking model

Burada sektörleri bir ML ya da istatistiksel modelle sıralarsınız. Girdi olarak şunlar olabilir:

- momentum
- realized volatility
- earnings revisions
- valuation
- breadth
- macro exposure proxies

Avantajı:

- esnek
- güçlü olabilir

Dezavantajı:

- fazla feature → overfitting riski

---

## Volatilite Tabanlı Yön Tahmini

Stratejinin temel bileşeni makalede önerilen volatilite tabanlı yön tahmini yaklaşımıdır. Geleneksel finansal tahmin modelleri genellikle gelecekteki getirinin büyüklüğünü tahmin etmeye çalışırken bu çalışmada getirinin yalnızca pozitif mi yoksa negatif mi olacağı tahmin edilmektedir.

Bu amaçla belirli bir zaman ufku için getirinin pozitif olup olmadığını gösteren ikili bir değişken tanımlanacaktır. Daha sonra volatilite tahminleri ve diğer açıklayıcı değişkenler kullanılarak bu değişken için bir olasılık modeli kurulacaktır.

Volatilite tahmini için başlangıçta **RiskMetrics (EWMA)** yöntemi kullanılacaktır. Bunun yanında daha gelişmiş bir alternatif olarak **GARCH(1,1)** modeli de değerlendirilecektir. Elde edilen volatilite tahminleri lojistik regresyon gibi bir yöntem kullanılarak gelecekteki getirinin pozitif olma olasılığını tahmin etmek için kullanılacaktır.

---

## Portföy Oluşturma

Tahmin edilen olasılıklar yatırım kararına dönüştürülerek portföy oluşturulacaktır. Örneğin tahmin edilen pozitif getiri olasılığı belirli bir eşik değerin üzerindeyse long pozisyon açılabilir. Olasılık düşükse short pozisyon alınabilir veya pozisyon kapatılabilir.

Pozisyonlar haftalık olarak gözden geçirilecektir. Ancak stratejinin hedefi kısa vadeli trading değil, makalenin bulgularına uygun şekilde **orta vadeli yatırım fırsatlarını yakalamaktır**. Bu nedenle pozisyonların ortalama tutulma süresi birkaç hafta ile birkaç ay arasında değişebilir.

---

## Yatırım Aralığı

Stratejinin performansı farklı yatırım aralıkalrında test edilecektir. Özellikle 1 ay, 2 ay, 3 ay ve 6 ay gibi farklı horizonlar incelenerek volatilite tabanlı yön tahmininin hangi vadelerde daha güçlü olduğu analiz edilecektir. Makaledeki bulgular orta vadelerde, özellikle yaklaşık 2–3 ay civarında daha güçlü bir sinyal olabileceğini göstermektedir.

Bu nedenle proje aynı zamanda bu sonucun farklı veri setlerinde ve farklı strateji yapılarında da geçerli olup olmadığını incelemeyi amaçlamaktadır.

---

# **Backup Plan: Senaryo Bazlı Modüler Trading Stratejisi**

---

Ana strateji makroekonomik rejim tespiti, sektör bazlı analiz ve volatilite temelli yön tahminini bir araya getiren çok katmanlı bir yapı üzerine kurulmaktadır. Ancak bu yapı uygulama açısından veri gereksinimi, model karmaşıklığı ve entegrasyon zorlukları nedeniyle risk içermektedir. Bu nedenle proje kapsamında yalnızca basitleştirilmiş bir alternatif değil, aynı zamanda farklı bileşenlerin bağımsız olarak çalışabileceği **senaryo bazlı modüler bir backup plan** geliştirilmesi amaçlanmaktadır.

Bu yaklaşımda strateji tek bir modele bağımlı olmak yerine, farklı veri ve model bileşenlerinin çalışabilirliğine bağlı olarak alternatif yollar üretmektedir. Böylece ana stratejinin belirli kısımlarının başarısız olması durumunda dahi projenin akademik ve uygulamalı çıktıları korunmuş olur. Tüm senaryolarda ortak nokta, makalenin temel bulgularına dayanan **volatilite temelli yön tahmini çerçevesinin korunmasıdır**. Bunun üzerine ek olarak residual risk, uluslararası piyasa etkileri ve yüksek momentler (skewness ve kurtosis) gibi literatürde gösterilmiş ek bilgi kaynakları da uygun senaryolarda modele dahil edilmektedir.

---

## **Senaryo 1: Makro Rejim Başarılı, Sektör Analizi Başarısız**

Bu senaryoda makroekonomik rejim başarıyla tespit edilebilmiş, ancak sektör bazlı ayrıştırma yeterince güçlü sonuç vermemiştir. Bu durumda strateji makro seviyede kalacak şekilde yeniden yapılandırılır. Ekonomik rejim bilgisi korunur ve buna ek olarak ABD piyasasının küresel piyasalara olan yön verici etkisi modele dahil edilir. Literatürde ABD piyasasının diğer piyasalardaki yön tahmini üzerinde anlamlı bir etkisi olduğu gösterilmiştir ve bu bilgi direction-of-change modellerine entegre edilebilir.

Bu çerçevede model girdileri makro rejim değişkenleri, volatilite tahminleri ve ABD piyasasına ait gecikmeli getiriler veya yön sinyallerinden oluşacaktır. Volatilite tahminleri temel sinyal üretiminde kullanılmaya devam ederken, makro rejim ve ABD etkisi bu sinyalin koşullu olarak güçlendirilmesini sağlar. Böylece sektör ayrımı olmadan dahi piyasa yönü hakkında anlamlı tahminler üretilebilecek bir yapı elde edilir.

---

## **Senaryo 2: ABD Yerine Türkiye Piyasası ve Yerel Makro Rejim**

Bu senaryoda uluslararası veri kullanımı yerine yerel piyasa odaklı bir yaklaşım benimsenir. ABD piyasasına dayalı bilgi yerine Türkiye piyasasına ait makroekonomik göstergeler ve piyasa dinamikleri kullanılır. Bu yaklaşım özellikle veri erişimi veya model uyumu açısından ABD verisinin kullanılamadığı durumlarda alternatif bir çözüm sunar.

Model bu durumda Türkiye’ye özgü makro değişkenler (faiz oranları, enflasyon, CDS primi, döviz kuru gibi) ve volatilite tahminlerini kullanarak yön tahmini üretir. Bu yaklaşım literatürdeki genel volatilite–yön ilişkisini korurken, yerel piyasa dinamiklerinin modele entegre edilmesini sağlar. Böylece strateji hem farklı piyasalara uyarlanabilir hale gelir hem de projenin genellenebilirliği test edilmiş olur.

---

## **Senaryo 3: Makro ve Sektör Katmanları Başarısız**

Makroekonomik rejim tespiti ve sektör analizi yapılamadığı durumda strateji tamamen **istatistiksel yön tahmini modeli** olarak yeniden kurgulanır.

Bu modelde volatilite tahmini temel bileşen olmaya devam eder, ancak buna ek olarak residual risk (idiosyncratic volatility), skewness ve kurtosis gibi yüksek momentler modele dahil edilir. Literatürde yalnızca volatilitenin değil, aynı zamanda dağılımın asimetrisi ve kuyruk davranışlarının da yön tahmini açısından bilgi içerdiği gösterilmiştir  . Ayrıca residual risk bileşeninin sistematik riske göre daha güçlü bir tahmin gücü sağlayabileceği ortaya konmuştur.

Bu senaryoda model tamamen veri odaklı çalışır ve herhangi bir ekonomik yorum katmanı içermez. Buna rağmen, kullanılan değişkenlerin literatürde güçlü teorik temellere sahip olması nedeniyle akademik açıdan geçerli bir çalışma sunar. Bu yapı aynı zamanda ana stratejinin çekirdek bileşenini temsil eder.

---

## **Senaryo 4: Sektör Analizi Başarılı, Makro Rejim Başarısız**

Bu senaryoda sektör bazlı analiz başarılı bir şekilde uygulanabilmiş ancak makroekonomik rejim modeli yeterli performans göstermemiştir. Bu durumda strateji sektör filtresi üzerine inşa edilir.

İlk olarak sektör bazlı bir seçim yapılır ve yalnızca güçlü performans gösteren sektörler yatırım evrenine dahil edilir. Bu filtreleme işlemi tamamlandıktan sonra yön tahmini modeli devreye girer. Bu modelde yine volatilite, residual risk, skewness ve kurtosis gibi değişkenler kullanılarak her sektör için yön tahmini yapılır.

Bu yaklaşım, sektör rotasyonu fikrini volatilite tabanlı yön tahmini ile birleştirir. Makro bilgi olmadan dahi, sektör bazlı ayrışmanın sağladığı bilgi ile daha dar ve hedeflenmiş bir yatırım evreninde işlem yapılmasını sağlar.

---