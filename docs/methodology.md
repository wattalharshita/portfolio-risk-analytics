# Methodology

Every method used in this project, why it was chosen, and what it cannot do. Written to be defensible without notes.

---

## Returns

**Simple return:** `r_t = (P_t − P_{t−1}) / P_{t−1}`
**Log return:** `ℓ_t = ln(P_t / P_{t−1})`

Log returns are used for **aggregation over time** because they are additive: the log return over a period is the sum of its daily log returns. Simple returns are used for **aggregation across assets** because a portfolio's simple return is the weight-weighted sum of its constituents' simple returns — this is *not* true of log returns, since the log of a sum is not the sum of logs.

**Adjusted close is used throughout.** Raw close produces wrong returns across corporate actions: a 2-for-1 split halves the quoted price overnight, which an unadjusted calculation records as a −50% return that no investor experienced. Adjusted close back-adjusts historical prices for splits and dividends so the return series reflects actual investor experience.

**Annualisation uses 252 trading days** — the approximate number of US trading days per year (365 days less weekends and roughly nine exchange holidays). Volatility scales with the square root of time under the i.i.d. assumption, so annualised volatility is `σ_daily × √252`. That i.i.d. assumption is itself questionable given volatility clustering, which is the argument for GARCH.

---

## Modern Portfolio Theory (Markowitz, 1952)

**The core idea.** An asset should be judged not in isolation but by its contribution to portfolio-level risk. Because assets are imperfectly correlated, a combination can achieve lower variance than the weighted average of its components. Diversification becomes a mathematical result rather than a slogan.

**Formulation.**
- Portfolio expected return: `μ_p = wᵀμ`
- Portfolio variance: `σ²_p = wᵀΣw`, where `Σ` is the covariance matrix

Minimising `wᵀΣw` subject to a target return `wᵀμ = μ*`, weights summing to one (`wᵀ1 = 1`), and long-only bounds (`0 ≤ w ≤ 1`), then sweeping `μ*`, traces the **efficient frontier** — the locus of portfolios for which no higher return is available at that level of risk.

**Implementation.** Solved with `scipy.optimize.minimize` using SLSQP (Sequential Least Squares Programming), which handles the equality constraints and box bounds directly. Implemented manually first, then cross-validated against `PyPortfolioOpt` — the two agree to within 0.0006 maximum absolute weight difference.

**Known limitations, all relevant to this project's result:**

| Limitation | Consequence here |
|---|---|
| Extreme sensitivity to expected-return inputs | Weights swung 67pp across rolling windows (see findings §5) |
| Assumes normally distributed returns | Mean excess kurtosis of 11.58 says otherwise (findings §3) |
| Variance as the risk measure penalises upside equally | Sortino exceeds Sharpe for every security (findings §1) |
| Historical data as a proxy for the future | The entire out-of-sample failure (findings §4) |
| Produces concentrated, unstable weights unless constrained | 100% single-asset allocations appeared in four of eight windows |

These are not caveats bolted on at the end. Sections 4 and 5 of `findings.md` are a direct empirical demonstration of the first and last of them.

---

## Sharpe and Sortino

**Sharpe ratio** `(R_p − R_f) / σ_p` — excess return per unit of total volatility. The industry default for risk-adjusted performance.

**Limitation:** it penalises upside and downside volatility identically. An asset that occasionally jumps +8% is scored as risky as one that occasionally drops −8%.

**Sortino ratio** `(R_p − R_f) / σ_downside`, where downside deviation uses only negative returns. Sortino exceeding Sharpe means the volatility being penalised is disproportionately favourable. In this universe the gap ranged from 0.31 (GOOG) to 0.55 (AMZN).

**Calmar ratio** `CAGR / |max drawdown|` — return per unit of worst-case historical loss. Useful because drawdown is the risk measure investors actually experience, as opposed to standard deviation which they do not perceive directly.

---

## Value at Risk and Conditional VaR

**VaR** answers: what loss threshold will not be exceeded with X% confidence over a given horizon? A 95% one-day VaR of 2% means losses should not exceed 2% on 95% of days.

Two estimation approaches are implemented and **compared**, because the gap between them is itself the finding:

- **Historical VaR** — the empirical quantile of observed returns. No distributional assumption, but limited to history actually observed.
- **Parametric VaR** — assumes normality: `−(μ + z_α σ)`. Smooth and extrapolable beyond observed history, but wrong in the tails when returns are not normal.

In this data, historical VaR exceeds parametric VaR for every security, by 13.3% on average at 99% confidence. That gap is fat-tail evidence.

**CVaR / Expected Shortfall** — the average loss *conditional on* breaching VaR. Answers "if things go badly, how badly?" rather than "where is the threshold?"

CVaR is a **coherent** risk measure and VaR is not — specifically, VaR violates subadditivity, meaning the VaR of a combined portfolio can exceed the sum of its parts' VaRs, which contradicts the whole premise of diversification. CVaR is also sensitive to the shape of the tail beyond the threshold, where VaR is blind to it. **Basel III moved market-risk capital from VaR to Expected Shortfall for exactly these reasons.**

---

## Monte Carlo simulation and Geometric Brownian Motion

Prices are modelled as `dS = μS dt + σS dW`, where `dW` is a Wiener increment. Discretised to daily steps, the log price evolves as `ln(S_{t+1}/S_t) ~ N((μ − σ²/2)dt, σ²dt)`. The `−σ²/2` correction is Itô's term — without it the simulated *mean* return exceeds the intended drift, because the exponential of a normal variable has a higher mean than the exponential of its mean.

Simulating 10,000 paths produces a distribution of outcomes rather than a point estimate, permitting probabilistic statements: "a 4.4% probability of a drawdown exceeding 20% over one year."

**Limitation.** GBM assumes constant volatility and log-normal returns. Real returns exhibit volatility clustering and fat tails, so GBM understates the frequency of extreme events. Given mean excess kurtosis of 11.58 in this data, the simulated tail figures are optimistic. This is precisely the gap GARCH addresses.

---

## Backtesting methodology

Testing a strategy on data it was not fitted to. The value is less the result than the control for known failure modes — which is what experienced interviewers probe.

| Failure mode | Control applied here |
|---|---|
| **Lookahead bias** | At each rebalance date `t`, weights are optimised on returns strictly before `t`. No future data reaches any weight. |
| **Transaction costs** | 10bps one-way charged on turnover at every rebalance, on by default. Frictionless backtests flatter turnover-heavy strategies — this one turned over 0.339 per rebalance and paid 1.22% cumulatively. |
| **Overfitting** | No parameter search. Lookback (252d) and rebalance frequency set once in `config.yaml`, never tuned against the result. |
| **Data snooping** | One strategy tested, one result reported — including that it lost. |
| **Survivorship bias** | **Not controlled.** The universe is securities that still exist and performed well. Results are biased upward. Stated openly rather than hidden. |

**Why the equal-weighted baseline is essential.** Optimisation results are meaningless without something to beat. An optimiser that cannot outperform naive 1/N allocation after costs has not earned its complexity — and this one did not. Reporting that is the point.

---

## Weight stability analysis

Re-running the optimiser on rolling historical windows and measuring how far the weights move. This is the diagnostic that converts "the backtest underperformed" into "here is the mechanism."

If the optimiser were recovering a stable property of these securities, weights would be broadly similar across windows. They were not: mean per-asset swing of 67 percentage points, with AAPL ranging the full 0–100%. The optimiser is fitting estimation noise in `μ`, which is both the input it is most sensitive to and the one estimated least precisely from historical data.

**This is the empirical justification for** constrained optimisation (cap concentration), shrinkage estimators such as Ledoit–Wolf (stabilise `Σ`), resampled efficiency (average over bootstrapped inputs), and Black–Litterman (blend market-implied equilibrium returns with explicit views rather than trusting raw historical means).

---

## Planned extensions

**Fama–French factor models (1993, extended 2015).** CAPM explains returns through market exposure alone. Fama and French showed size (SMB) and value (HML) systematically explain returns CAPM leaves unexplained, later adding profitability and investment. Regressing portfolio excess returns on these factors moves the analysis from *describing* performance to *explaining* it — a portfolio that beat the market may simply have loaded on small-cap value rather than exhibited skill. The typical mature finding is that alpha is not statistically distinguishable from zero. Factor data is freely available from the Kenneth French Data Library; the schema already carries a `factors` table.

**GARCH(1,1) (Bollerslev, 1986).** Volatility is not constant — it clusters, with calm following calm and turbulence following turbulence. GARCH models conditional variance as a function of past shocks and past variance: `σ²_t = ω + α ε²_{t−1} + β σ²_{t−1}`. Every static-volatility metric in this project inherits the assumption that risk is constant; recomputing VaR on GARCH-forecast volatility would show risk estimates changing materially with the volatility regime, and would widen the Monte Carlo tails toward something defensible.

---

# Extensions implemented

The sections above describe the core platform. What follows documents the layers added afterwards, in the same format: what the method is, why it was chosen, and what it cannot do.

## Allocation strategies (`src/strategies.py`)

Seven allocators share one signature and one test protocol. They are deliberately ordered by how much they rely on estimated inputs, because the hypothesis under test is that reliance on estimation drives out-of-sample degradation.

**Equal weighting (1/N).** The null hypothesis. DeMiguel, Garlappi and Uppal (2009) showed 1/N is remarkably hard to beat out-of-sample precisely because it estimates nothing. If nothing beats it after costs, nothing else here has earned its complexity.

**Minimum variance.** Drops the expected-return vector entirely and optimises on the covariance matrix alone. Justified by the fact that covariances are estimated far more precisely than means — the estimation error in `mu` scales with the square root of sample length, and returns are dominated by noise.

**Risk parity.** Equalises each asset's contribution to portfolio variance: `RC_i = w_i (Σw)_i / σ_p`, minimising the dispersion of contributions. Uses no return forecast. The intuition is that a capital-weighted portfolio is not a risk-weighted one — 1/N in capital can be 80% concentrated in risk.

**Hierarchical Risk Parity (Lopez de Prado, 2016).** Clusters assets by correlation distance, then allocates recursively down the resulting tree. Its central claim is that Markowitz's instability comes substantially from **inverting** the covariance matrix: inversion amplifies estimation error, and near-singular matrices (common when assets are correlated) amplify it catastrophically. HRP never inverts anything.

**Ledoit–Wolf shrinkage.** Pulls the sample covariance matrix toward a structured target with an analytically optimal shrinkage intensity. The sample covariance matrix of N assets estimated from T observations is badly conditioned when T is not much larger than N²; shrinkage trades a little bias for a large variance reduction.

**Why the shrinkage result matters.** Shrinkage barely improved out-of-sample performance here (Sharpe 1.436 vs 1.415, weight swing 15.3% vs 15.8%). That is not a failure of the method — it is a diagnostic. **Shrinkage repairs the covariance matrix; the factor regression shows the instability originates in the mean vector.** The correction has to match the diagnosis. Reporting a fix that did not work, and explaining precisely why, is more informative than reporting one that did.

## Factor attribution (`src/factors.py`)

**The question.** A backtest says a strategy earned 28% annualised. It does not say whether that was skill or exposure to a systematic risk factor any investor could buy cheaply. Regression separates the two: realised excess return decomposes into factor betas and an unexplained residual, and the residual's statistical significance can be tested.

**What was built, and what was not.** Fama–French SMB and HML require market capitalisation and book-to-market data, absent from a price-and-volume dataset. Rather than claim a Fama–French regression that was not run, three factors were constructed from the price cross-section of all 505 constituents:

- **MKT** — equal-weighted market excess return.
- **WML** — winners-minus-losers momentum (Jegadeesh & Titman 1993; Carhart 1997). Ranks on trailing 12-month return *skipping the most recent month*, the standard 12-1 formation that avoids the documented short-term reversal effect. Long top decile, short bottom, monthly rebalance.
- **VOL** — low-volatility minus high-volatility, ranked on trailing 60-day realised volatility. A price-based proxy for the betting-against-beta anomaly (Frazzini & Pedersen, 2014), and labelled as a proxy.

Reported alpha is therefore alpha **relative to these three factors only**. Confirming a genuinely unexplained residual would require the full factor set.

**Newey–West standard errors (5 lags)** correct for heteroskedasticity and autocorrelation in daily returns. Ordinary OLS standard errors would overstate significance materially here.

**Formation uses only prior information.** Momentum signals are lagged; no lookahead enters the factor construction.

## GARCH and VaR validation (`src/volatility.py`)

**Why Student-t innovations.** The kurtosis analysis established that returns are not normal. Fitting GARCH with normal innovations would reintroduce the exact assumption GARCH exists to relax. The fitted ν of 5.80 confirms heavy tails — a normal distribution corresponds to ν → ∞.

**Persistence and half-life.** α + β = 0.958 is the rate at which a volatility shock decays. The implied half-life, `ln(0.5) / ln(α+β)`, is 16.2 trading days. This is what "volatility clusters" means quantitatively: a shock today is still half-present three weeks later.

**Why validate rather than just fit.** Producing a time-varying volatility estimate is not the same as producing a better one. Two formal tests:

**Kupiec (1995) unconditional coverage.** Does the VaR model breach at the correct *rate*? A 99% VaR should be exceeded on about 1% of days. The likelihood-ratio statistic is χ² with 1 degree of freedom. Too many breaches means risk is understated; too few means capital is wasted.

**Christoffersen (1998) independence.** Are breaches *independent*, or do they cluster? This is the test most projects skip, and it is the more damning one. A model with a correct average breach rate but clustered breaches fails in runs — during exactly the stress episodes it exists to warn about. Both models tested here fail it, and saying so is the honest result.

**Limitation, stated.** The GARCH VaR is fitted in-sample, so the coverage test is a specification check rather than true out-of-sample forecast evaluation. A rolling refit — re-estimating parameters at each step using only prior data — is the stricter protocol and the natural next iteration.

## Regulatory reporting (`sql/reports/risk_report.sql`)

Modelled on the shape of an institutional risk pack: sector exposure with policy-limit breach flags, 99% VaR and Expected Shortfall per holding, sector risk aggregation, and a systemic-stress breach register. Assumptions are documented formally in `docs/risk_report_spec.md`, as a specification would be, so the report can be reviewed independently of the code producing it.

**Why Expected Shortfall sits beside VaR.** VaR is not a coherent risk measure — it violates subadditivity, so a combined portfolio's VaR can exceed the sum of its components', contradicting the premise of diversification. ES is coherent and is sensitive to the shape of the tail beyond the threshold, where VaR is blind. Basel III moved market-risk capital from VaR to Expected Shortfall on exactly these grounds.

**The breach register's design.** Thresholds are per-security empirical percentiles computed with no reference to market history. The discriminating variable is the *count* of simultaneously-breaching holdings: high counts indicate a common factor moving everything, while moderate counts concentrated in one sector grouping indicate a macro-factor event. Days where only utilities, REITs and telecoms breach are interest-rate events, not equity-market events. Separating systemic from idiosyncratic stress is what the register is for.
