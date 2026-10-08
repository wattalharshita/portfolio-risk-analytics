# Risk Report Specification

Formal documentation of the assumptions behind `sql/reports/risk_report.sql`, written in
the register a specification would use so the report can be reviewed independently of the
code that produces it.

## Scope

Daily market-risk measurement for a long-only equity portfolio of 32 securities across 11
GICS sectors. Excludes the benchmark series (`is_benchmark = 1`), which is a constructed
index rather than a holding.

## Policy limits

| Limit | Threshold | Rationale |
|---|---|---|
| Single-sector exposure | ≤ 20% of portfolio | Prevents sector concentration from dominating portfolio risk |
| Single-security exposure | ≤ 10% of portfolio | Caps idiosyncratic issuer risk |

Breaches are flagged in report R1 with status `BREACH`; compliant rows return `OK`. Limits
are stated here and not embedded as magic numbers anywhere outside the SQL.

## Position assumptions

R1 assumes equal-weighted holdings (1/N). This is a deliberate default: the report is
about the *limit framework*, and equal weighting is the position set the limits are
calibrated against. For a live report, substitute optimiser weights from
`src/strategies.py` into the `holdings` CTE. No other query depends on position weights.

## Risk measures

**Value at Risk (99%, one-day, historical).** The empirical first-percentile daily return
per security, computed via `NTILE(100)`. Reported as a positive loss magnitude. No
distributional assumption is made; the measure is limited to observed history.

**Expected Shortfall (99%, one-day).** The mean return across the first percentile — the
average loss conditional on VaR being breached.

ES is reported alongside VaR because VaR is not a coherent risk measure: it violates
subadditivity, so a combined portfolio's VaR can exceed the sum of its components', which
contradicts the premise of diversification. ES is coherent and is sensitive to the shape
of the tail beyond the threshold, where VaR is blind to it. Basel III replaced VaR with
Expected Shortfall for market-risk capital on these grounds. Column
`es_exceeds_var_by_pct` quantifies the gap per holding.

**Volatility (R3).** Annualised from daily returns using a 252-day convention, computed in
SQL as `sqrt(E[r²] − E[r]²) × sqrt(252)` because SQLite provides no `STDDEV` aggregate.

## Breach register (R4)

Lists dates on which five or more holdings simultaneously breached their own 99% VaR
threshold. The threshold is the security's own empirical first percentile over the full
sample, so the register is constructed with no reference to market history.

The count of simultaneously-breaching holdings is the discriminating variable:

- **High counts (>15 of 32)** indicate systemic stress — a common factor moving the whole
  market.
- **Moderate counts concentrated in one sector grouping** indicate a sector or macro-factor
  event rather than a market-wide one. Days where only utilities, REITs and telecoms breach
  are interest-rate events, not equity-market events.

This distinction — systemic versus idiosyncratic — is the register's purpose.

## Known limitations

1. **Historical VaR only.** No parametric or Monte Carlo VaR in the SQL layer; those live
   in `src/metrics.py` and `src/volatility.py`. Historical VaR cannot extrapolate beyond
   the worst observed day.
2. **Unconditional risk measures.** Thresholds use full-sample percentiles and are
   therefore constant through time. `src/volatility.py` demonstrates via Kupiec testing
   that constant-volatility VaR is statistically rejected on this data; the SQL layer
   retains it because that is what a standard regulatory report computes, and the
   comparison is itself informative.
3. **No netting, no derivatives, no FX.** Long-only cash equities in a single currency.
4. **Survivorship bias.** The universe consists of index members as of the end of the
   sample, back-tested across it.
5. **One-day horizon only.** No multi-day scaling, which would require an autocorrelation
   assumption the data does not support.

## Reproduction

```bash
python src/ingest.py --source sp500
python - <<'PY'
import sqlite3, pathlib
conn = sqlite3.connect("data/portfolio.db")
sql = pathlib.Path("sql/reports/risk_report.sql").read_text()
for stmt in [s for s in sql.split(";") if s.strip()]:
    try:
        conn.execute(stmt).fetchall()
    except sqlite3.Error:
        pass
PY
```
