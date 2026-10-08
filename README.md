# Portfolio Risk & Returns Analytics Platform

An end-to-end quantitative platform: ingests daily equity market data into a SQL warehouse, computes risk-adjusted performance metrics, optimises allocations, and stress-tests the result through walk-forward backtesting, GARCH volatility modelling, factor attribution and Monte Carlo simulation.

**32 securities · 11 GICS sectors · 1,258 trading days · 41,546 price rows · 7 allocation strategies · 37 unit tests**

---

## The finding

**Mean-variance optimisation beat equal weighting on Sharpe out-of-sample — and lost on drawdown. Which metric you choose decides whether it "worked".**

| Strategy | Sharpe | Max drawdown | Calmar | Turnover | Cost drag |
|---|---:|---:|---:|---:|---:|
| Max-Sharpe (unconstrained) | **1.415** | −25.0% | 1.13 | 0.573 | 2.81% |
| Max-Sharpe (10% cap) | 1.247 | −14.2% | **1.28** | 0.410 | 2.01% |
| Equal-weighted (1/N) | 1.164 | **−12.3%** | 1.25 | 0.038 | 0.19% |
| Hierarchical Risk Parity | 0.995 | −11.2% | 1.12 | 0.163 | 0.80% |

The optimiser earned a 0.25 higher Sharpe by taking **double the drawdown** and trading **15× as much**. On return per unit of worst-case loss it finished behind doing nothing.

### Why: the optimiser is a momentum strategy nobody designed

Factor regression against a momentum factor built from the full 505-security cross-section:

| Strategy | β momentum | α (annual) | t-stat | R² |
|---|---:|---:|---:|---:|
| Max-Sharpe (unconstrained) | **+0.34** | 14.0% | 2.28 | 0.50 |
| Max-Sharpe (10% cap) | +0.20 | 6.3% | 1.98 | 0.75 |
| Equal-weighted (1/N) | −0.02 | 5.2% | 2.50 | 0.88 |
| Hierarchical Risk Parity | −0.03 | 3.2% | 1.57 | 0.87 |

Nobody told the optimiser to buy momentum. **Optimising on trailing-window mean returns is mechanically buying recent winners.** That one fact explains everything else: it outperformed because momentum paid over 2014–2018, it doubled the drawdown because momentum crashes, and **Ledoit–Wolf shrinkage barely helped (Sharpe 1.436 vs 1.415) because the exposure lives in the mean vector while shrinkage repairs the covariance matrix.**

### Instability predicts cost and drawdown almost deterministically

Across all seven allocators, mean per-asset weight swing across rolling estimation windows correlates **+0.99 with turnover** and **+0.89 with maximum drawdown**. Constraining weights to 10% cut the swing from 15.8% to 9.4% and produced the **best Calmar of any strategy tested**.

### Constant-volatility VaR is statistically rejected

| VaR model | Breaches | Expected | Rate | Kupiec p | Verdict |
|---|---:|---:|---:|---:|---|
| Static (constant volatility) | 29 | 12.6 | 2.31% | **0.0001** | **REJECTED** |
| GARCH(1,1), Student-t | 17 | 12.6 | 1.35% | 0.2345 | pass |

The industry-standard constant-volatility model breached its 99% threshold **2.3× more often than it should**. Replacing it with GARCH(1,1) conditional volatility brought the breach rate into line and passed formal coverage testing. GARCH persistence is 0.958 with a 16-day shock half-life, and conditional volatility ranges **6.5×** — from 6.2% to 39.8% annualised — where the static model reports 11.4% every single day.

**Both models still fail the Christoffersen independence test.** Breaches cluster, so neither is adequate during sustained stress. Reporting that is more useful than reporting only the model that passed.

### The VaR breach register reconstructed market history on its own

Built purely from per-security percentile thresholds with no knowledge of market events, the register surfaced:

| Date | Holdings breaching | Event |
|---|---:|---|
| 2015-08-24 | **26 of 32** | China devaluation / "Black Monday" |
| 2018-02-05 | 22 | Volatility spike |
| 2016-06-24 | 12 | Brexit referendum result |
| 2016-09-09 | 9 | Rate scare — **utilities, REITs and telecoms only** |
| 2013-06-20 | 7 | Taper tantrum — **rate-sensitive names only** |

The count of simultaneous breaches separates systemic stress from sector-specific events. That distinction is the register's whole purpose.

---

## Live demo

*Deploy to Streamlit Community Cloud and paste the link here.*

```bash
streamlit run app.py
```

Nine tabs: Overview · Risk & Return · Optimisation · Strategy Race · Volatility · Attribution · Simulation · Risk Report · Findings. Every chart carries a stated takeaway.

---

## Quick start

```bash
git clone <repo-url> && cd portfolio-risk-analytics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/ingest.py --source sp500     # builds the warehouse (downloads source data once)
python -m pytest tests/ -v              # 37 tests
python scripts/run_analysis.py          # core metrics, optimisation, backtest
python scripts/run_horserace.py         # 7-strategy out-of-sample comparison
python scripts/run_volatility.py        # GARCH + Kupiec/Christoffersen VaR validation
python scripts/run_factors.py           # factor construction and attribution
streamlit run app.py                    # dashboard
```

---

## Data

| Source | What | When |
|---|---|---|
| `--source sp500` | **Default.** All 505 S&P 500 constituents, daily OHLCV, 2013-02-08 → 2018-02-07, 619,040 rows. Downloaded once from a public mirror, then cached. | Reproduces every number in this README |
| `--source yfinance` | Live Yahoo Finance pull for any universe and date range | To extend or bring up to date |
| `--source demo` | Four cached tickers | Minimal smoke test, no network |

**Benchmark.** An equal-weighted index constructed from all 505 constituents (~492 with data on a typical day), rebased to 100. This is *equal*-weighted, not cap-weighted like SPY — it tilts toward smaller constituents and was a tougher benchmark over this period. Stated rather than glossed.

**Data quality.** The ingestion pass scans all 505 securities for single-day moves above 40% and flagged 11 across 9 tickers — genuine unadjusted corporate actions including the eBay/PayPal and Baxter/Baxalta spinoffs (July 2015) and the Discovery split (August 2014). None fall inside the selected 32-security universe, and all are excluded from benchmark construction. Adjusted close is used throughout, because unadjusted close records a 2-for-1 split as a −50% return that no investor experienced.

---

## Architecture

```
portfolio-risk-analytics/
├── config.yaml                  # universe, sectors, dates, all parameters
├── app.py                       # 9-tab Streamlit dashboard
├── .github/workflows/ci.yml     # builds warehouse, runs tests, verifies every SQL query
├── data/
│   ├── raw/                     # cached source data
│   └── portfolio.db             # SQLite warehouse
├── sql/
│   ├── schema.sql               # hand-written DDL
│   └── reports/
│       ├── analytical_queries.sql   # 7 analytical queries
│       └── risk_report.sql          # 4 regulatory-style risk queries
├── src/
│   ├── db.py                    # connection + idempotent upserts
│   ├── ingest.py                # 3-source ETL with data-quality layer
│   ├── metrics.py               # returns, vol, Sharpe, Sortino, drawdown, VaR, CVaR
│   ├── optimise.py              # Markowitz + efficient frontier + library validation
│   ├── strategies.py            # 7 pluggable allocators
│   ├── backtest.py              # walk-forward engine + stability analysis
│   ├── simulate.py              # Monte Carlo / GBM
│   ├── volatility.py            # GARCH + Kupiec + Christoffersen
│   └── factors.py               # factor construction + attribution regression
├── scripts/                     # run_analysis, run_horserace, run_volatility,
│                                # run_factors, refresh_data.sh, run_report.sh
├── tests/                       # 37 tests
└── docs/                        # methodology.md, findings.md, risk_report_spec.md
```

### The SQL layer

Raw `sqlite3` with hand-written DDL rather than an ORM, deliberately — it keeps joins, window functions and aggregates visible as the thing being evaluated.

`analytical_queries.sql` — rolling 30-day volatility via moving window frames, sector aggregates with `GROUP BY ... HAVING`, best/worst performer per quarter using `RANK() OVER (PARTITION BY ...)`, pairwise co-movement via self-join, running-peak drawdown, tail-risk breach counts, ingestion audit trail.

`risk_report.sql` — sector exposure with policy-limit breach flags, 99% VaR and Expected Shortfall per holding, sector risk aggregation, and the systemic-stress breach register. Assumptions formally documented in `docs/risk_report_spec.md`.

All 11 queries execute in CI on every push.

---

## Methodology

**Optimisation.** Markowitz mean-variance implemented manually with `scipy.optimize` (SLSQP), then cross-validated against `PyPortfolioOpt` — the two agree to **0.0001** maximum absolute weight difference. Both implementations are kept.

**The seven allocators**, ordered by how much they rely on estimated inputs — because the hypothesis under test is that reliance on estimation drives out-of-sample degradation:

| Allocator | Relies on |
|---|---|
| Equal-weighted (1/N) | nothing |
| Hierarchical Risk Parity | correlation structure only |
| Risk parity | covariance only |
| Min-variance | covariance only |
| Max-Sharpe (capped) | mean + covariance, bounded |
| Max-Sharpe (Ledoit–Wolf) | mean + shrunk covariance |
| Max-Sharpe (unconstrained) | mean + covariance |

**Backtest bias controls.** Lookahead: at each rebalance date `t`, weights use only returns strictly before `t`. Transaction costs: 10bps one-way on turnover, on by default. Overfitting: no parameter search — lookback and rebalance frequency set once in `config.yaml` and never tuned against results. Data snooping: every strategy reported, including losers. **Survivorship bias: not controlled**, and it is the largest caveat here.

**Factor model.** Fama–French SMB and HML need market-cap and book-to-market data a price dataset does not contain. Rather than claim a regression that was not run, three factors are built from the price cross-section of all 505 constituents: market excess return, 12-1 month momentum (long top decile, short bottom, monthly rebalance), and a low-volatility proxy. Regressions use Newey–West standard errors (5 lags). Reported alpha is alpha *relative to these three factors only*.

**Volatility.** GARCH(1,1) with Student-t innovations — normal innovations would reintroduce the exact assumption GARCH exists to relax, given the measured excess kurtosis. Ljung-Box on squared returns establishes clustering before fitting. VaR models validated with Kupiec unconditional coverage and Christoffersen independence tests.

---

## Limitations

- **Survivorship bias is uncontrolled.** The universe is S&P 500 membership as of 2018, back-tested from 2013. Results are biased upward.
- **One period, one market.** 2013–2018 was an unusually strong, low-volatility bull market with persistent momentum. Which allocator "wins" is conditional on that regime.
- **Factors are price-based** — no size or value factor, so alpha is relative to three factors only.
- **GARCH VaR is fitted in-sample**, making the coverage test a specification check rather than true out-of-sample validation. A rolling refit is the stricter test.
- **Simple cost model** — flat bps on turnover, no slippage, market impact or spread.
- **GBM Monte Carlo understates tails**, as the volatility analysis demonstrates directly.

## What this is not

Not a trading system — no broker integration, no execution, no real money. Not a price-prediction model — it forecasts *risk*, which is tractable, rather than returns, which largely are not. Not a research contribution — it is a rigorous implementation of established methods with their limitations measured rather than asserted.

## Next layers

Rolling out-of-sample GARCH refit for true VaR forecast validation · Black–Litterman with explicit views · regime-switching volatility models · extension to the full 505-security universe (the pipeline is universe-agnostic; only `config.yaml` changes).
