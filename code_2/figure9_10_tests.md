# Figure 9-10 Test Notu

Bu not, `figures_8_10.py` icin yapilan hizalama testlerini ozetler.

## Test Edilen Degiskenler

- Logit cezasi: `penalty=None`, `L2 C=1`, `L2 C=10`
- Logit intercept: `fit_intercept=True` ve `fit_intercept=False`
- Figure 10 forecast-realization korelasyon orneklemi: overlapping gunluk
  originler ve non-overlapping horizon bloklari
- Hedef return tanimi: compound h-gunluk return ve gunluk simple returnlerin
  toplami
- Horizon seti: `20/21`, `60/63`, `125/126`, `250`
- Rolling pencere hizasi:
  - `strict`: forecast origin `T` aninda sadece gercekten bilinen etiketler kullanilir. Her horizon icin tahminler 5 yil dolduktan sonra baslar; `h>1` icin son `h` gunluk tamamlanmamis etiketler 5 yillik pencerenin icinden atilir.
  - `paper_like`: pencere `T-1`'de biter. Bu makaledeki Figure 10 davranisina daha cok benzer, ancak `h>1` icin gercek zamanli al-sat tahmini olarak kullanilirsa look-ahead riski tasir.

## Ana Bulgu

Logit cezasi sonuclari neredeyse degistirmedi. Figure 10 farkinin ana kaynagi rolling pencere hizasi.

Ilk `strict` uygulamada gereksiz muhafazakar bir pencere vardi: `h>1` icin ilk
tahmin `1250 + h - 1` gun sonra baslatiliyor ve egitim icin 1250 tamamlanmis
etiket geriye dogru toplaniyordu. Bu look-ahead yaratmiyordu, ama "5 yillik
rolling estimation window" ifadesine tam uymuyordu. Duzeltmeden sonra her
horizon icin tahminler 5 yil dolunca basliyor; pencerenin sonundaki
tamamlanmamis etiketler atiliyor.

Ikinci onemli fark logit intercept varsayimiydi. Kodun ilk halinde standart
logit aliskanligi ile `alpha + beta * (1/sigma_t)` kullanilmisti. Makaledeki
ampirik bolum ise once modeli `F(mu / sigma_t)` olarak kuruyor, sonra
`I_{t+h}`'yi `1/sigma_t` uzerine logit ile tahmin ettigini soyluyor. Bu formda
`beta` h-gunluk beklenen getiriyi temsil eder ve intercept yoktur. Bu nedenle
varsayilan model `logit(P) = beta * (1/sigma_t)` olarak degistirildi.

| Mode | h=1 | h=5 | h=21 | h=63 | h=126 | h=250 |
|---|---:|---:|---:|---:|---:|---:|
| `strict`, interceptli eski model | 0.002 | 0.008 | 0.011 | 0.078 | -0.013 | -0.088 |
| `strict`, no-intercept paper modeli | 0.008 | 0.021 | 0.032 | 0.093 | 0.069 | 0.010 |
| `paper_like` corr(P, y) | 0.002 | 0.009 | 0.027 | 0.142 | 0.188 | 0.219 |

Figure 10 icin makaledeki Figure 4 notu korelasyonun non-overlapping
horizonlarda hesaplandigini soyluyor; Figure 10 da Figure 6'nin ampirik
analogu olarak veriliyor. Bu nedenle non-overlap tanim test edildi. Ancak bu
tanim uzun horizonlarda gozlem sayisini cok dusurdugu ve grafigin genel seklini
bozdugu icin ana varsayim olarak kullanilmadi. Ana Figure 10 tekrar overlapping
gunluk origin korelasyonuna donduruldu.

| Figure 10 mode | h=1 | h=5 | h=21 | h=63 | h=126 | h=250 |
|---|---:|---:|---:|---:|---:|---:|
| compound target, non-overlap corr | 0.008 | 0.039 | -0.006 | 0.068 | 0.071 | 0.181 |
| sum-simple target, non-overlap corr | 0.008 | 0.034 | 0.008 | 0.062 | 0.063 | 0.215 |

Non-overlap sonucu ozellikle `h=250` icin yukari oynar, fakat `h=250` icin
yalnizca 34-36 gozlem kalir. Bu nedenle uzun-horizon degerleri partition
offset'ine hassastir ve ana grafik icin guvenilir bir duzeltme degildir.

## Yorum

`strict` mode akademik olarak daha temiz bir gercek zamanli out-of-sample testidir:
`T` gununde `y[j]` etiketi ancak `j+h <= T` ise bilinir. Bu nedenle egitim
penceresi sonundaki tamamlanmamis horizon etiketleri atilir; pencere daha eski
tarihlere dogru 1250 tamamlanmis etikete tamamlanmaz.

`paper_like` mode ise makaledeki grafige daha yakin davranir. Bunun nedeni,
modeli tahmin ederken `T-1`'e kadar olan satirlari kullanmasidir. Fakat
`y[T-1]`, `h>1` icin `T-1+h` tarihindeki fiyat bilgisine ihtiyac duyar. Bu
yuzden bu hizalama pratik al-sat sistemi icin dogrudan kullanilmaz; replika
amacli ayri tutulmalidir.

## Ek Testler

- Volatilite serisini bir gun ileri hizalamak veya gun sonu guncellenmis EWMA
  kullanmak uzun-horizon korelasyonlarini iyilestirmedi.
- 21 gunluk yeniden tahmin periyodunun baslangic offset'i sonucu aciklamadi.
- Figure 10 korelasyonunu tek bir non-overlapping partition uzerinden hesaplamak
  bazi uzun horizonlarda pozitif sonuc verebiliyor, fakat `h=250` icin yalnizca
  yaklasik 36 gozleme dayaniyor ve offset secimine hassas. Tum offset'ler
  ortalandiginda overlapping sonuca geri donuyor.
- Lokal projede CRSP `SPINDX` verisi yok. Stooq alternatifi API key/captcha
  istedigi icin bu oturumda SPINDX benzeri alternatif veriyle tam test
  yapilamadi.

## Uretilen Dosyalar

- `results/figure9.png`: strict real-time OOS
- `results/figure10.png`: strict real-time OOS
- `results/figure9_paper_like.png`: paper-like rolling window
- `results/figure10_paper_like.png`: paper-like rolling window

Script su sekilde calisir:

```bash
CD_ROLLING_ALIGNMENT=strict .venv/bin/python code_2/figures_8_10.py
CD_ROLLING_ALIGNMENT=paper_like .venv/bin/python code_2/figures_8_10.py
CD_ROLLING_ALIGNMENT=both .venv/bin/python code_2/figures_8_10.py
```

CRSP `SPINDX` gibi alternatif bir fiyat dosyasi varsa dogrudan kullanilabilir:

```bash
CD_SP500_CSV=/path/to/spindx.csv CD_ROLLING_ALIGNMENT=strict .venv/bin/python code_2/figures_8_10.py
```

CSV dosyasinda tarih kolonu `Date` veya `CALDT`; fiyat kolonu `Close`,
`SPINDX`, `Price` veya `Level` olabilir.

Refit sikligi `CD_UPD_FREQ` ile degistirilebilir:

```bash
# Gunluk refit, look-ahead'siz, sadece Figure 9
CD_UPD_FREQ=1 CD_ROLLING_ALIGNMENT=strict CD_FIGURES=9 .venv/bin/python code_2/figures_8_10.py

# Gunluk refit ile Figure 10 da mumkun, fakat yaklasik 2.25 milyon logit fit eder.
CD_UPD_FREQ=1 CD_ROLLING_ALIGNMENT=strict CD_FIGURES=10 .venv/bin/python code_2/figures_8_10.py
```

Interceptli eski modeli tekrar uretmek icin:

```bash
CD_LOGIT_INTERCEPT=1 CD_ROLLING_ALIGNMENT=strict .venv/bin/python code_2/figures_8_10.py
```

Figure 10 korelasyon orneklemi ve hedef return tanimi:

```bash
# Ana hesap: overlapping gunluk originler
CD_FIGURES=10 .venv/bin/python code_2/figures_8_10.py

# Tani amacli non-overlap hesap
CD_FIG10_CORR=nonoverlap CD_FIGURES=10 .venv/bin/python code_2/figures_8_10.py

# Explicit overlapping hesap
CD_FIG10_CORR=overlap CD_FIGURES=10 .venv/bin/python code_2/figures_8_10.py

# Gunluk simple return toplamlarini hedef olarak kullanma deneyi
CD_RETURN_TARGET=sum_simple CD_FIGURES=10 .venv/bin/python code_2/figures_8_10.py
```

`CD_UPD_FREQ=1` iken dosyalar normal aylik-refit grafiklerinin uzerine yazilmaz:

- `results/figure9_daily.png`
- `results/figure10_daily.png`

Gunluk refit ile ana horizonlarda strict/no-intercept korelasyonlar:

| Mode | h=1 | h=5 | h=21 | h=63 | h=126 | h=250 |
|---|---:|---:|---:|---:|---:|---:|
| `strict`, daily refit, no-intercept | 0.008 | 0.019 | 0.032 | 0.098 | 0.078 | 0.018 |
