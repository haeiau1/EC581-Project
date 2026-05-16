# S&P 500 Sector Index Data

## Source

**Provider:** Yahoo Finance (via `yfinance`)
**Frequency:** Daily (trading days only)
**Price:** Adjusted Close (splits and dividends adjusted)
**Download script:** `download_and_quality.py`

---

## Sector Index Definitions

The S&P 500 is divided into 11 sectors using the **GICS (Global Industry Classification Standard)**, jointly developed by S&P Dow Jones Indices and MSCI in 1999. Historical data before 1999 is backdated by S&P.

| File | Ticker | Full Name | Coverage | N Days | Description |
|------|--------|-----------|----------|--------|-------------|
| `IT.csv` | `^SP500-45` | S&P 500 Information Technology | 1993-05-04 → today | ~8,312 | Software, hardware, semiconductors, IT services (Apple, Microsoft, Nvidia, TSMC ADR) |
| `CommSvcs.csv` | `^SP500-50` | S&P 500 Communication Services | 1993-05-04 → today | ~8,313 | Telecom, media, internet platforms (Alphabet, Meta, Netflix, Disney) |
| `ConDisc.csv` | `^SP500-25` | S&P 500 Consumer Discretionary | 1993-05-04 → today | ~8,312 | Non-essential consumer goods and retail (Amazon, Tesla, McDonald's, Nike) |
| `ConStaples.csv` | `^SP500-30` | S&P 500 Consumer Staples | 1993-05-04 → today | ~8,313 | Essential everyday products (P&G, Coca-Cola, Walmart, PepsiCo) |
| `Financials.csv` | `^SP500-40` | S&P 500 Financials | 1993-05-04 → today | ~8,313 | Banks, insurers, asset managers (JPMorgan, Berkshire, Visa, Mastercard) |
| `HealthCare.csv` | `^SP500-35` | S&P 500 Health Care | 1993-05-04 → today | ~8,313 | Pharma, biotech, medical devices, managed care (UnitedHealth, J&J, Eli Lilly, Pfizer) |
| `Industrials.csv` | `^SP500-20` | S&P 500 Industrials | 1993-05-04 → today | ~8,313 | Aerospace, defense, machinery, transportation (GE, Caterpillar, Boeing, UPS) |
| `Energy.csv` | `^GSPE` | S&P 500 Energy | 1993-05-04 → today | ~8,316 | Oil, gas, refiners, pipeline companies (ExxonMobil, Chevron, ConocoPhillips) |
| `Materials.csv` | `^SP500-15` | S&P 500 Materials | 1993-05-04 → today | ~8,315 | Chemicals, metals, mining, paper (Linde, Sherwin-Williams, Newmont, Freeport) |
| `Utilities.csv` | `^SP500-55` | S&P 500 Utilities | 1993-05-04 → today | ~8,313 | Electric, gas, water utilities (NextEra, Duke Energy, Southern Company) |
| `RealEstate.csv` | `^SP500-60` | S&P 500 Real Estate | 2001-10-09 → today | ~6,190 | REITs and real estate services (Prologis, American Tower, Equinix). Added to GICS in 2016; Yahoo backdates to 2001. |

---

## Notes

**Energy ticker:** The standard `^SP500-10` ticker is unavailable on Yahoo Finance. `^GSPE` is the official S&P 500 Energy Index ticker and covers the identical constituent universe.

**Real Estate:** Separated from Financials as a standalone GICS sector in August 2016. Historical data before that date is reconstructed by S&P and Yahoo backdates it to October 2001 only.

**Communication Services:** Substantially reconstituted in September 2018 when large internet companies (Alphabet, Facebook/Meta) were moved from IT/Consumer Discretionary into this sector. Pre-2018 data reflects the old Telecommunications sector composition (AT&T, Verizon-heavy), so the character of the series changes structurally around that date.

**Adjusted Close:** All prices are adjusted for stock splits and dividend distributions. This ensures return calculations are not contaminated by mechanical price drops on ex-dividend dates.

---

## Quality Summary (from last run)

| Sector | First Date | Last Date | N Obs | Missing | Ann. Return | Ann. Vol | Skewness | Ex. Kurtosis |
|--------|-----------|-----------|-------|---------|-------------|----------|----------|--------------|
| IT | 1993-05-04 | 2026-05-15 | 8,312 | 0 | 17.2% | 27.1% | 0.25 | 6.42 |
| CommSvcs | 1993-05-04 | 2026-05-15 | 8,313 | 0 | 7.2% | 22.1% | 0.13 | 6.76 |
| ConDisc | 1993-05-04 | 2026-05-15 | 8,312 | 0 | 11.3% | 21.5% | -0.05 | 7.33 |
| ConStaples | 1993-05-04 | 2026-05-15 | 8,313 | 0 | 8.4% | 15.0% | -0.05 | 10.18 |
| Financials | 1993-05-04 | 2026-05-15 | 8,313 | 0 | 10.0% | 27.2% | 0.30 | 17.06 |
| HealthCare | 1993-05-04 | 2026-05-15 | 8,313 | 0 | 10.5% | 18.1% | -0.04 | 7.12 |
| Industrials | 1993-05-04 | 2026-05-15 | 8,313 | 0 | 10.3% | 20.2% | -0.19 | 8.11 |
| Energy | 1993-05-04 | 2026-05-15 | 8,316 | 0 | 10.1% | 26.4% | -0.19 | 12.55 |
| Materials | 1993-05-04 | 2026-05-15 | 8,315 | 0 | 8.4% | 22.4% | -0.09 | 7.56 |
| Utilities | 1993-05-04 | 2026-05-15 | 8,313 | 0 | 5.6% | 18.5% | 0.12 | 13.07 |
| RealEstate | 2001-10-09 | 2026-05-15 | 6,190 | 0 | 8.3% | 29.1% | 0.41 | 20.68 |

*All return/volatility figures are annualized. Kurtosis is excess (normal = 0).*

---

## File Structure

```
code_data/
├── download_and_quality.py      # download + quality check script
├── DATA.md                      # this file
├── data/
│   └── raw/                     # one CSV per sector
│       ├── IT.csv
│       ├── CommSvcs.csv
│       ├── ConDisc.csv
│       ├── ConStaples.csv
│       ├── Financials.csv
│       ├── HealthCare.csv
│       ├── Industrials.csv
│       ├── Energy.csv
│       ├── Materials.csv
│       ├── Utilities.csv
│       └── RealEstate.csv
└── results/
    └── quality_report/
        ├── quality_summary.csv  # machine-readable stats table
        ├── quality_report.txt   # human-readable report
        └── plots/
            ├── 01_price_levels.png       # log-price series per sector
            ├── 02_daily_returns.png      # daily return series per sector
            ├── 03_missing_heatmap.png    # annual data availability heatmap
            └── 04_return_distributions.png  # histogram + normal overlay
```
