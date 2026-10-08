# Findings

All figures reproducible via `scripts/run_analysis.py`, `run_horserace.py`, `run_volatility.py`, `run_factors.py`.

**Universe:** 32 securities, 11 GICS sectors · **Period:** 2013-02-11 → 2018-02-07, 1,258 trading days · **Benchmark:** equal-weighted index of all 505 S&P 500 constituents · **rf = 2%**

---

## 1. Diversification: correlation does the work sectors imply

Mean pairwise correlation **0.299** across 32 securities.

| | Pair | r | Note |
|---|---|---:|---|
| Highest | JPM–BAC | **0.833** | Two money-centre banks |
| Lowest | BAC–SO | **−0.053** | Bank vs regulated utility |

The universe was deliberately constructed with correlated pairs (two banks, two oil majors, two telecoms, two utilities, two staples) precisely so the correlation analysis would have something to say. It does: the deliberately-paired names sit at the top of the table, and the extremes span 0.89 of correlation range.

**Interpretation.** Diversification *within* a sector is largely illusory — holding JPM and BAC is closer to holding one position than two. Diversification *across* sectors is real: a bank and a utility are effectively uncorrelated in this sample. Counting holdings is not measuring diversification.

---

## 2. Fat tails: normal-distribution VaR understates by 16.4%

At 99% confidence, historical VaR exceeds parametric (normal) VaR across the universe by **16.4% on average**, with **mean excess kurtosis of 5.56** (normal = 0).

Every variance-based number elsewhere in this project — Sharpe, the covariance matrix, the efficient frontier, GBM Monte Carlo — inherits that optimism. Each should be read as a floor on risk. This is the empirical argument for Expected Shortfall over VaR, and the reason Basel III made that switch.

---

## 3. Headline: the optimiser won on Sharpe and lost on drawdown

### In-sample (fitted and evaluated on the same data)

| Portfolio | Return | Vol | Sharpe | Holdings > 0.5% |
|---|---:|---:|---:|---:|
| Equal-weighted | 14.52% | 11.35% | 1.103 | 32 |
| Min-volatility | 7.97% | 9.72% | 0.615 | 18 |
| **Max-Sharpe (unconstrained)** | **37.91%** | 15.95% | **2.251** | **6** |
| Max-Sharpe (≤10%/asset) | 27.23% | 12.81% | 1.969 | 13 |

In-sample Sharpe improvement: **+104.1%**. From 32 available securities the unconstrained optimiser holds **six**, with a largest position of 25%.

### Out-of-sample (walk-forward, 252-day lookback, monthly rebalance, 10bps costs)

| Strategy | Relies on | Cum | Ann | Vol | Sharpe | Sortino | MaxDD | **Calmar** | Turnover | Cost drag | Weight swing |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Equal-weighted (1/N) | nothing | 77.1% | 15.4% | 11.5% | 1.164 | 1.496 | −12.3% | 1.25 | 0.038 | 0.19% | 0.0% |
| Min-variance | covariance | 46.9% | 10.1% | 10.0% | 0.809 | 1.053 | −13.0% | 0.78 | 0.283 | 1.39% | 9.0% |
| Max-Sharpe (unconstrained) | mean + cov | **169.4%** | **28.2%** | 18.5% | 1.415 | 1.883 | **−25.0%** | 1.13 | 0.573 | 2.81% | 15.8% |
| Max-Sharpe (≤10% cap) | mean + cov, bounded | 94.7% | 18.2% | 13.0% | 1.247 | 1.628 | −14.2% | **1.28** | 0.410 | 2.01% | 9.4% |
| Max-Sharpe (Ledoit–Wolf) | mean + shrunk cov | 172.9% | 28.6% | 18.5% | **1.436** | **1.914** | −25.0% | 1.14 | 0.552 | 2.71% | 15.3% |
| Risk parity | covariance | 66.7% | 13.7% | 10.9% | 1.068 | 1.372 | −11.9% | 1.15 | 0.052 | 0.26% | 1.2% |
| Hierarchical Risk Parity | correlation only | 60.2% | 12.5% | 10.6% | 0.995 | 1.281 | **−11.2%** | 1.12 | 0.163 | 0.80% | 2.9% |

Benchmark cumulative return over the same window: **53.1%**.

**The result depends on which metric you pick.**

- On **Sharpe**, the optimiser wins: 1.415 vs 1.164, +0.251.
- On **maximum drawdown**, it loses badly: −25.0% vs −12.3%, more than double.
- On **Calmar** (return per unit of worst-case loss), it **loses**: 1.13 vs 1.25.
- On **turnover** it traded **15×** as much and paid **2.81%** in costs against 0.19%.

In-sample it beat the baseline by +104%. Out-of-sample by +21.6% on Sharpe and −9.6% on Calmar. The gap between those two numbers is the entire lesson.

**The capped optimiser produced the best drawdown-adjusted outcome of anything tested** — Calmar 1.28, ahead of both the unconstrained optimiser and the naive baseline. Constraints are not a handicap on optimisation; here they were the thing that made it work.

---

## 4. Why: the optimiser is a momentum strategy nobody designed

Three-factor regression (market, 12-1 momentum, low-volatility proxy), Newey–West standard errors, 5 lags:

| Strategy | α (ann) | t | p | **β MKT** | **β WML** | **β VOL** | R² | CAPM R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Equal-weighted (1/N) | 5.16% | 2.50 | 0.012 | 0.91 | **−0.02** | 0.10 | 0.882 | 0.870 |
| Min-variance | 2.53% | 0.87 | 0.387 | 0.79 | −0.05 | 0.30 | 0.690 | 0.545 |
| Max-Sharpe (unconstrained) | 14.02% | 2.28 | 0.022 | 0.96 | **+0.34** | −0.03 | 0.495 | 0.401 |
| Max-Sharpe (≤10% cap) | 6.30% | 1.98 | 0.047 | 0.91 | +0.20 | 0.08 | 0.745 | 0.641 |
| Max-Sharpe (Ledoit–Wolf) | 14.28% | 2.33 | 0.020 | 0.96 | **+0.34** | −0.04 | 0.498 | 0.404 |
| Risk parity | 4.06% | 1.92 | 0.054 | 0.89 | −0.02 | 0.15 | 0.865 | 0.835 |
| Hierarchical Risk Parity | 3.23% | 1.57 | 0.116 | 0.89 | −0.03 | 0.19 | 0.866 | 0.809 |

**This is the central result of the project.** The optimiser carries a momentum loading of **+0.34**; every risk-based allocator sits near **zero**. Nobody instructed it to buy momentum. **Optimising on trailing-window mean returns *is* buying recent winners, mechanically.**

Adding the momentum factor absorbs **2.44 percentage points** of the optimiser's CAPM alpha (16.46% → 14.02%) and lifts R² from 0.401 to 0.495. For equal weighting the same addition absorbs **0.06pp**. The optimiser's outperformance is substantially a factor exposure available to any investor, not skill.

Three consequences follow directly, and each matches the data:

1. **It outperformed because momentum paid** over 2014–2018, a persistently trending market.
2. **It doubled the drawdown because momentum crashes** — the factor's known failure mode.
3. **Ledoit–Wolf shrinkage barely helped** (Sharpe 1.436 vs 1.415; weight swing 15.3% vs 15.8%) **because the exposure originates in the mean vector, and shrinkage repairs the covariance matrix.** Applying the wrong correction to the right diagnosis changes almost nothing. This is the cleanest evidence in the project that the instability is a *mu* problem.

Only **4 of 7** strategies show alpha statistically distinguishable from zero at 5% — and even those are alpha relative to three price-based factors, not the full Fama–French set.

---

## 5. Instability predicts cost and drawdown almost deterministically

Mean per-asset weight swing across 12 rolling estimation windows, correlated across all seven allocators:

| Relationship | Correlation |
|---|---:|
| Weight swing ↔ turnover | **+0.987** |
| Weight swing ↔ maximum drawdown | **+0.888** |
| Weight swing ↔ Sharpe | +0.592 |

Ranked most to least stable: 1/N (0.0%) → risk parity (1.2%) → HRP (2.9%) → min-variance (9.0%) → capped max-Sharpe (9.4%) → Ledoit–Wolf (15.3%) → unconstrained max-Sharpe (15.8%).

**Interpretation.** An allocator whose answer changes substantially when the estimation window moves will trade more, pay more, and fall further. The +0.99 turnover correlation is nearly mechanical. The +0.89 drawdown correlation is the finding: **instability is not merely a cost problem, it is a risk problem.**

The positive Sharpe correlation (+0.59) is the honest complication — in this sample, instability also bought return. That is a statement about a trending bull market, not a general law, and it is exactly why Calmar tells a different story from Sharpe.

---

## 6. Constant-volatility VaR is statistically rejected

**Is clustering present?** Ljung-Box (20 lags) on the equal-weighted portfolio: raw returns show little serial correlation; **squared returns are strongly autocorrelated**. Volatility is forecastable even when direction is not — the statistical licence to fit GARCH.

**GARCH(1,1), Student-t innovations:**

| Parameter | Value | Meaning |
|---|---:|---|
| α (shock reaction) | 0.1918 | How sharply volatility responds to new shocks |
| β (persistence) | 0.7663 | How long elevated volatility lasts |
| α + β | **0.9580** | Total persistence — shocks decay slowly |
| Shock half-life | **16.2 days** | Time for a volatility shock to halve |
| ν (Student-t df) | 5.80 | Low df confirms fat tails; normal innovations would be wrong |

Conditional volatility ranges from **6.16%** (2017-06-13) to **39.80%** (2015-08-27) — a **6.5× span**. The static model reports **11.35%** on every one of the 1,258 days.

The model's highest-volatility months, identified without being told about market history: **Sep 2015** (21.6%), **Feb 2018** (20.5%), **Jan 2016** (19.5%), **Feb 2016** (18.1%), **Aug 2015** (17.0%).

**VaR validation at 99%:**

| Model | Avg VaR | Range | Breaches | Expected | Rate | Kupiec LR | p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Static (constant vol) | 1.61% | 1.0× | **29** | 12.6 | 2.31% | 15.818 | **0.0001** | **REJECTED** |
| GARCH(1,1) | 1.67% | **7.0×** | 17 | 12.6 | 1.35% | 1.413 | 0.2345 | **pass** |

The industry-standard constant-volatility model breached **2.3× more often than its own 99% claim**, and is rejected at any conventional significance level. GARCH brings the breach rate into line and passes formal coverage testing.

**Both fail Christoffersen independence** (static p = 0.0295, GARCH p = 0.0010). Breaches cluster: for the GARCH model, P(breach | breach yesterday) = 17.6% against P(breach | calm yesterday) = 1.1%. **Neither model is adequate during sustained stress.** A model with the right average breach rate but clustered breaches still fails precisely when it is relied upon. Reporting this is more useful than reporting only the model that passed.

---

## 7. The breach register reconstructed market history unaided

`sql/reports/risk_report.sql` R4 flags dates where ≥5 holdings simultaneously breached their own 99% VaR. Thresholds are per-security empirical percentiles with no reference to market events.

| Date | Breaching | Worst holding | What it was |
|---|---:|---:|---|
| 2015-08-24 | **26 of 32** | −6.89% | China devaluation / "Black Monday" |
| 2018-02-05 | 22 | −8.49% | Volatility spike |
| 2015-08-21 | 12 | −6.12% | Lead-up to Black Monday |
| 2016-06-24 | 12 | −7.41% | Brexit referendum result |
| 2016-09-09 | 9 | −5.21% | Rate scare — AMT, DUK, KO, NEE, PEP, SO, T, UNP, VZ |
| 2013-06-20 | 7 | −5.37% | Taper tantrum — AMT, JNJ, KO, NEE, PEP, PG, SPG |

**The count of simultaneous breaches is the discriminating variable.** High counts (26, 22) indicate systemic stress — a common factor moving everything. The 2016-09-09 and 2013-06-20 entries are different in kind: the breaching names are exclusively **utilities, REITs, telecoms and staples** — rate-sensitive sectors. Those are interest-rate events, not equity-market events, and the register separates them without being told the difference.

Distinguishing systemic from idiosyncratic stress is the register's entire purpose.

---

## 8. Forward risk — Monte Carlo

10,000 GBM paths, 252-day horizon, $100,000 initial, driven by **out-of-sample realised** return and volatility:

| | Optimised (μ=28.2%, σ=18.5%) | Baseline (μ=15.4%, σ=11.5%) |
|---|---:|---:|
| Median outcome | $130,434 | $115,942 |
| 95% VaR | 3.79% | 4.05% |
| 95% CVaR | 10.23% | 8.15% |
| 99% CVaR | 19.60% | 14.24% |
| P(loss) | 7.49% | 9.76% |
| P(drawdown > 20%) | **1.88%** | **0.21%** |

A methodological note that matters: driving this simulation with the **in-sample optimised** expected return (37.9%) would produce a distribution in which even the 5th percentile is a gain — a garbage-in artefact that compounds the very estimation error this project set out to measure. Using realised out-of-sample inputs is the only defensible choice.

**Caveat.** GBM assumes constant volatility. Section 6 measured a 6.5× swing in conditional volatility with Ljung-Box strongly rejecting constancy. **These tail figures are optimistic.** Re-simulating on GARCH-forecast volatility is the correct next step.

---

## 9. What this adds up to

The project's defensible claim is not that it built an optimiser. It is that it built seven allocators, tested them identically, found the answer depends on the metric, and could explain the mechanism:

1. Mean-variance optimisation is most sensitive to expected-return estimates.
2. Expected returns are the hardest quantity to estimate from historical data.
3. So the optimiser concentrates on whatever led the lookback window — which **is** a momentum bet (β_WML = +0.34, measured, not assumed).
4. Momentum paid over 2014–2018, so it won on Sharpe. Momentum crashes, so it lost on drawdown and Calmar.
5. Shrinking the covariance matrix did not help, because the problem is in the mean vector — **the correction has to match the diagnosis**.
6. Capping concentration did help, producing the best Calmar of anything tested.

A version of this reporting only the +104% in-sample improvement would have been wrong, and wrong in a way one question would expose.

---

## Limitations

- **Survivorship bias is uncontrolled** — the universe is S&P 500 membership as of 2018, back-tested from 2013. This is the single largest caveat and biases every return figure upward.
- **One period, one market.** 2013–2018 was an unusually strong, low-volatility bull market with persistent momentum. Which allocator "wins" is conditional on that regime.
- **Factors are price-based.** No size or value factor — alpha is relative to three factors only.
- **GARCH VaR is fitted in-sample**, so the coverage test is a specification check, not true out-of-sample forecast validation. A rolling refit is stricter.
- **Simple cost model** — flat bps on turnover, no slippage, market impact or spread.
- **Equal-weighted benchmark**, not cap-weighted like SPY.
- **GBM understates tails**, as section 6 demonstrates directly.
