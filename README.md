# Tail-Risk Validation

## Research question
How can historical loss VaR and expected shortfall be estimated from a trailing window and checked against subsequent synthetic observations?

## Methodology and formulas
For losses L=-r and confidence alpha, historical VaR is the linear empirical alpha-quantile of L. Expected shortfall is the sample mean of losses at or above that VaR. At forecast index t, the implementation uses only returns t-lookback through t-1. The Kupiec unconditional-coverage likelihood-ratio statistic compares the observed exception rate with 1-alpha; this package reports the statistic, not an asymptotic p-value.

## Assumptions and limitations
- The experiment uses a seeded Student-t(5) synthetic return process scaled to 1% standard deviation. It contains no real asset or portfolio data.
- The empirical tail mean includes observations equal to the interpolated VaR threshold; small samples and ties can make it conservative.
- VaR/ES are unconditional trailing-window estimates and can lag regime changes. The Kupiec statistic checks exception frequency only, not exception independence or tail severity.
- No transaction costs, portfolio aggregation, model-selection correction, or survivorship-bias adjustment is included.
- This is a statistical validation example, not a real portfolio risk forecast or investment performance claim.

## Reproducibility
With Python 3.11+, run python -m pip install -e ".[test]", then python -m pytest and tail-risk-demo --seed 127. The run writes a walk-forward plot and JSON summary. Data are generated locally and no credentials or APIs are needed.

