# Differences Between Paper Figures and Local Replication

This note compares the figures generated in `code_2/results/figure1.png` through
`figure10.png` with the figures in Christoffersen and Diebold (2006),
*Financial Asset Returns, Direction-of-Change Forecasting, and Volatility
Dynamics*.

The comparison is visual and methodological. The paper does not publish the
original code, random seeds, CRSP data extract, or every implementation detail,
so some differences are expected even when the same broad method is used.

## Summary

| Figure | Match Level | Main Difference |
|---|---:|---|
| Figure 1 | High | Same Gaussian idea; styling, shading, and x-axis range differ. |
| Figure 2 | High | Same derivative shape; local x-axis stops at 3 instead of paper's 5. |
| Figure 3 | Medium/Low | Local Heston sample path is much smoother than the paper's path. |
| Figure 4 | High | Very similar shape and magnitude; small peak/location differences. |
| Figure 5 | High | Very similar ranking and shape; small peak/location differences. |
| Figure 6 | Low | Local sign autocorrelation is noisy and much larger/different than paper. |
| Figure 7 | High | Same broad derivative shape; local x-axis extends farther and right tail differs visually. |
| Figure 8 | Medium/High | Same 1987 spike and general volatility dynamics; data source and plotting scale differ. |
| Figure 9 | Medium | Same broad horizon pattern, but local long-horizon probabilities are more extreme. |
| Figure 10 | Low | Forecast-realisation correlation is materially different and turns negative locally. |

## Figure 1

**Paper:** Two Gaussian return densities with mean `mu = 10%`, standard
deviations `sigma = 5%` and `sigma = 15%`. The note emphasizes probabilities
around `0.98` and `0.75`.

**Local replication:** `code_2/results/figure1.png`.

**Differences:**

- The mathematical content matches.
- The local figure uses color and shaded right-tail areas; the paper uses black
  line art without colored shading.
- The paper x-axis runs approximately from `-0.3` to `0.5`; the local figure
  uses roughly `-0.25` to `0.45`.
- The local legend explicitly prints `Pr(R>0)` values; the paper puts this
  information in the figure note.

**Assessment:** No substantive methodological difference.

## Figure 2

**Paper:** Responsiveness of sign probability to volatility movements,
plotted against the information ratio `mu / sigma`. The curve reaches its most
negative value near `sqrt(2)`.

**Local replication:** `code_2/results/figure2.png`.

**Differences:**

- The derivative curve and minimum are consistent with the paper.
- The paper's x-axis extends to about `5`; the local plot stops at `3`.
- The local figure marks the minimum with a dashed vertical line and legend;
  the paper does not use the same styling.

**Assessment:** Substantively consistent. To look closer to the paper, extend
the x-axis to `5`.

## Figure 3

**Paper:** A 500-period Heston sample path of conditional sign probabilities
for daily, weekly, monthly, quarterly, semiannual, and annual horizons.

**Local replication:** `code_2/results/figure3.png`.

**Differences:**

- The local daily/weekly/monthly panels are much smoother than the paper's.
- The paper's monthly, quarterly, semiannual, and annual panels show visibly
  more high-frequency jagged movement.
- The paper's y-axis is approximately `0.5` to `0.9`; the local figure uses a
  wider range, `0.3` to `1.0`, which visually compresses fluctuations.
- The local simulation uses `n_intraday=24`; the paper states it uses a more
  refined Heston simulation setup. The source code comment says the paper uses
  `288` intraday steps.
- The specific sample path is seed-dependent. The paper does not provide the
  random seed, so the exact line cannot be reproduced from the article alone.

**Likely causes:**

- Different random seed/sample path.
- Fewer intraday simulation steps in the local code.
- Wider y-axis in the local plot.
- Possible difference in how the Heston conditional probability is evaluated
  at each simulated point.

**Assessment:** Qualitative horizon ordering is similar, but the figure is not
visually close to the paper.

## Figure 4

**Paper:** Forecast-realisation correlation across horizons for expected return
parameters `mu = 0.10`, `0.05`, and `0.00`.

**Local replication:** `code_2/results/figure4.png`.

**Differences:**

- The local curves have the same ordering as the paper: `mu = 0.10` highest,
  `mu = 0.05` middle, `mu = 0.00` lowest.
- The local peak for `mu = 0.10` is about `0.088` at roughly `81` trading days.
  The paper's peak appears close to `0.09` around the same intermediate horizon.
- Minor differences in smoothness and peak location remain.

**Likely causes:**

- The local code estimates the correlation from a simulated variance path
  (`N_LONG = 20,000` days), whereas the paper describes a large number of
  realizations and a quasi-analytic result.
- Different simulation seed and numerical integration settings.

**Assessment:** Close match.

## Figure 5

**Paper:** Forecast-realisation correlation across horizons for volatility
persistence parameters `kappa = 2`, `5`, and `10`.

**Local replication:** `code_2/results/figure5.png`.

**Differences:**

- The local ranking matches the paper: `kappa = 2` highest, `kappa = 5`
  middle, `kappa = 10` lowest.
- Magnitudes are close: local peak for `kappa = 2` is about `0.089`, similar
  to the paper's top curve.
- Small differences exist in exact peak locations and tail levels.

**Likely causes:**

- Finite simulated sample in local code.
- Different random seeds and numerical integration settings.

**Assessment:** Close match.

## Figure 6

**Paper:** Forecast-realisation correlation and first autocorrelation of return
signs across horizons under benchmark Heston parameters.

**Local replication:** `code_2/results/figure6.png`.

**Differences:**

- The local forecast-realisation correlation curve is close to Figure 4's
  benchmark curve, as expected.
- The local sign autocorrelation line is highly jagged and reaches much larger
  values than the paper's smoother autocorrelation line.
- The paper's autocorrelation curve is smooth, positive, and remains well below
  the forecast-realisation correlation curve for most horizons.
- In the local figure, the autocorrelation can become larger than the forecast
  correlation and even shows sharp jumps.

**Likely causes:**

- The local autocorrelation is computed from one finite simulated Heston path
  of `N_LONG = 20,000` days. For large horizons this leaves relatively few
  non-overlapping observations, making the estimate noisy.
- The paper appears to use a much larger simulation / quasi-analytic approach.
- The local autocorrelation convention may not match the paper's convention
  exactly. The code uses adjacent non-overlapping horizon returns:
  `sign(R[t:t+h])` versus `sign(R[t+h:t+2h])`.

**Assessment:** Not a close match. This is a substantive discrepancy and should
be fixed if Figure 6 is intended to visually replicate the paper. The likely
fix is to increase simulation length substantially and verify the exact
autocorrelation definition.

## Figure 7

**Paper:** Responsiveness of sign probability to volatility movements in the
Heston model, plotted against annualized information ratio.

**Local replication:** `code_2/results/figure7.png`.

**Differences:**

- The local curve has the same broad U-shape and minimum near information ratio
  around `1.5`.
- The paper's x-axis runs to about `2.5`; the local plot extends to `4.0`.
- The local right-hand side rises sharply near the end of the computed grid.
  This is consistent with the derivative approaching zero, but the wider local
  x-axis makes the visual shape look different.
- Styling and labels differ.

**Assessment:** Substantively consistent. To match the paper visually, set the
x-axis to about `0` to `2.5`.

## Figure 8

**Paper:** Daily annualized RiskMetrics volatility for CRSP S&P 500 `SPINDX`,
January 1, 1963 through December 31, 2003, using smoothing parameter `0.94`.

**Local replication:** `code_2/results/figure8.png`.

**Differences:**

- The local figure uses Yahoo Finance `^GSPC`, not CRSP `SPINDX`.
- The broad pattern matches: the 1987 crash volatility spike appears near
  `95%`, and late-1990s/early-2000s volatility is elevated.
- The exact daily path differs because Yahoo and CRSP index histories are not
  guaranteed to be identical.
- The local plot is larger, colored, and includes a wider visual time axis.

**Likely causes:**

- Different data source: Yahoo `^GSPC` versus CRSP `SPINDX`.
- Possible differences in missing trading days and historical data revisions.

**Assessment:** Broadly consistent, not exact.

## Figure 9

**Paper:** Conditional probability forecasts for positive S&P 500 returns at
six horizons, estimated using a logit model with RiskMetrics volatility
forecast and five-year rolling estimation windows.

**Local replication:** `code_2/results/figure9.png`.

**Differences:**

- The local daily and weekly panels are broadly similar to the paper.
- The local monthly, quarterly, semiannual, and annual panels are more extreme,
  with probabilities moving closer to `0` and `1` than in the paper.
- The paper's annual panel reaches high values near `1`, but the local annual
  panel spends more time extremely close to `1` and has sharper movements.
- The local quarterly and semiannual panels show deeper drops than the paper in
  some periods.
- The local figure begins after the initial five-year rolling window, as does
  the paper visually, but exact start dates and paths differ.

**Likely causes:**

- Data source mismatch: Yahoo `^GSPC` versus CRSP `SPINDX`.
- The local code uses strict no-lookahead rolling windows. The paper says
  forecasts are out-of-sample, but does not describe the exact treatment of
  overlapping `h`-day labels at the end of each rolling window.
- The local logit is unregularized (`penalty=None`), which can produce extreme
  fitted probabilities when overlapping long-horizon labels create near
  separation.
- Horizon convention may differ. The local code uses `[1, 5, 21, 63, 126, 250]`;
  the paper labels horizons as daily, weekly, monthly, quarterly, semiannual,
  and annual, but does not explicitly state whether those are `[1, 5, 20, 60,
  125, 250]` or `[1, 5, 21, 63, 126, 250]`.
- The paper states that `sigma_t` is a forecast of `h`-day return volatility.
  In the local logit, the predictor is `1 / sigma_t` based on annualized daily
  RiskMetrics volatility. Multiplying by `sqrt(h)` would not change a single
  horizon logit except by rescaling the slope, but any additional horizon-level
  convention in the paper could affect exact output.

**Assessment:** Same methodology at a high level, but not a close visual
replication for longer horizons.

## Figure 10

**Paper:** Forecast-realisation correlation and return sign autocorrelation
for S&P 500 across horizons.

**Local replication:** `code_2/results/figure10.png`.

**Differences:**

- This is the largest empirical discrepancy.
- In the paper, the forecast-realisation correlation is positive, rises with
  horizon, and remains high through long horizons.
- In the local figure, the forecast-realisation correlation peaks near the
  intermediate horizon but turns negative after roughly 90 trading days and is
  strongly negative at long horizons.
- The local sign autocorrelation line is closer in broad shape to the paper's
  autocorrelation line, but the exact level and timing differ.
- In the paper, forecast correlation stays above autocorrelation. In the local
  figure, the opposite happens for many horizons.

**Likely causes:**

- Data source mismatch: Yahoo `^GSPC` versus CRSP `SPINDX`.
- Strict no-lookahead rolling windows may differ from the paper's exact
  implementation.
- Long-horizon overlapping labels make logistic estimates highly sensitive to
  small changes in data and window definitions.
- The local forecast correlation uses daily overlapping out-of-sample
  forecasts and realizations. The paper's exact evaluation convention is not
  fully specified in the article text.
- The local autocorrelation calculation may not match the paper's exact
  first-order sign autocorrelation convention.

**Assessment:** Not a close match. This figure needs further investigation
before claiming empirical replication.

## Highest Priority Fixes

1. Obtain CRSP `SPINDX` data, if possible, instead of Yahoo `^GSPC`.
2. Add a `paper_like` mode for horizons `[1, 5, 20, 60, 125, 250]`.
3. Re-check whether Figure 9 and Figure 10 should use forecast-origin dates or
   realization dates for plotting/evaluation.
4. For Figure 6, increase the simulation length substantially and verify the
   exact autocorrelation convention.
5. For Figure 10, compute both versions of sign autocorrelation:
   daily-overlapping lag-1 and adjacent non-overlapping horizon-block
   autocorrelation. Compare both to the paper.
6. For Figure 9 and Figure 10, compare strict no-lookahead rolling estimation
   with a closer reconstruction of the paper's possible rolling-window
   convention, while clearly labeling any version that is not true
   out-of-sample.

## Bottom Line

The local code reproduces the theoretical and simulation figures reasonably
well for Figures 1, 2, 4, 5, and 7. Figure 8 is broadly consistent but uses a
different data source. Figures 3, 6, 9, and especially 10 show material visual
or methodological differences from the paper. The empirical replication should
therefore be described as an approximate reconstruction rather than an exact
replication unless the CRSP data and the paper's original implementation
details are recovered.
