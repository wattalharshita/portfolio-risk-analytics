"""
run_analysis.py — runs every module end-to-end and prints the numbers that go into
docs/findings.md and the README. One command, reproducible from a clean database.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db import get_conn
from src.metrics import (load_returns_wide, full_report, covariance_matrix,
                         historical_var, parametric_var, cvar, TRADING_DAYS)
from src.optimise import max_sharpe, min_variance, portfolio_stats
from src.simulate import simulate_portfolio, simulation_summary
from src.backtest import run_backtest, weight_stability

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
tickers = cfg["universe"]["tickers"]
rf = cfg["risk_free_rate"]
max_w = cfg["max_weight_per_asset"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

with get_conn() as conn:
    returns = load_returns_wide(conn, tickers)
    report = full_report(conn, tickers, rf)
    cov_ann, corr = covariance_matrix(conn, tickers)

print("=" * 78)
print("1. UNIVERSE & DATA")
print("=" * 78)
print(f"Tickers: {tickers}")
print(f"Period:  {returns.index[0].date()} to {returns.index[-1].date()}")
print(f"Trading days: {len(returns)}  |  Total return observations: {returns.size}")

print("\n" + "=" * 78)
print("2. PER-ASSET RISK & RETURN")
print("=" * 78)
print(report.round(4).to_string())

print("\n" + "=" * 78)
print("3. CORRELATION")
print("=" * 78)
print(corr.round(3).to_string())
off_diag = corr.where(~np.eye(len(corr), dtype=bool))
print(f"\nMean pairwise correlation: {off_diag.stack().mean():.3f}")
print(f"Highest pair: {off_diag.stack().idxmax()} at r = {off_diag.stack().max():.3f}")
print(f"Lowest pair:  {off_diag.stack().idxmin()} at r = {off_diag.stack().min():.3f}")

print("\n" + "=" * 78)
print("4. FAT TAILS — HISTORICAL vs PARAMETRIC VaR")
print("=" * 78)
print(f"{'Asset':<8}{'Hist VaR99':>12}{'Param VaR99':>13}{'Gap':>10}{'Gap %':>10}{'Kurtosis':>11}")
tail_rows = []
for t in tickers:
    r = returns[t]
    h = historical_var(r, 0.99)
    p = parametric_var(r, 0.99)
    k = r.kurtosis()
    gap_pct = (h - p) / p * 100
    tail_rows.append({"ticker": t, "hist": h, "param": p, "gap_pct": gap_pct, "kurt": k})
    print(f"{t:<8}{h:>12.4%}{p:>13.4%}{h-p:>10.4%}{gap_pct:>9.1f}%{k:>11.2f}")
mean_gap = np.mean([r["gap_pct"] for r in tail_rows])
print(f"\nMean understatement of 99% VaR by the normal assumption: {mean_gap:.1f}%")
print(f"Mean excess kurtosis (normal = 0): {np.mean([r['kurt'] for r in tail_rows]):.2f}")

print("\n" + "=" * 78)
print("5. IN-SAMPLE OPTIMISATION")
print("=" * 78)
mu = returns.mean().values * TRADING_DAYS
cov = returns.cov().values * TRADING_DAYS
n = len(tickers)

results = {}
for name, w in [
    ("Equal-weighted (baseline)", np.repeat(1 / n, n)),
    ("Min-volatility", min_variance(mu, cov)),
    ("Max-Sharpe (unconstrained)", max_sharpe(mu, cov, rf)),
    (f"Max-Sharpe (<={int(max_w*100)}%/asset)", max_sharpe(mu, cov, rf, max_weight=max_w)),
]:
    ret, vol = portfolio_stats(w, mu, cov)
    sh = (ret - rf) / vol
    results[name] = {"w": w, "ret": ret, "vol": vol, "sharpe": sh}

print(f"{'Portfolio':<30}{'Return':>10}{'Vol':>10}{'Sharpe':>9}   Weights")
for name, d in results.items():
    ws = " ".join(f"{t}={x:.2f}" for t, x in zip(tickers, d["w"]))
    print(f"{name:<30}{d['ret']:>10.2%}{d['vol']:>10.2%}{d['sharpe']:>9.3f}   {ws}")

is_improve = ((results["Max-Sharpe (unconstrained)"]["sharpe"]
               - results["Equal-weighted (baseline)"]["sharpe"])
              / results["Equal-weighted (baseline)"]["sharpe"] * 100)
print(f"\nIn-sample Sharpe improvement vs baseline: {is_improve:+.1f}%")

print("\n" + "=" * 78)
print("6. OUT-OF-SAMPLE BACKTEST (the honest test)")
print("=" * 78)
bt = {}
bt["Optimised, unconstrained"] = run_backtest(returns, rf, 252, "M", 10.0, 1.0, "max_sharpe")
bt[f"Optimised, <={int(max_w*100)}%/asset"] = run_backtest(returns, rf, 252, "M", 10.0, max_w, "max_sharpe")
bt["Optimised, quarterly rebal"] = run_backtest(returns, rf, 252, "Q", 10.0, 1.0, "max_sharpe")
bt["Equal-weighted baseline"] = run_backtest(returns, rf, 252, "M", 10.0, 1.0, "equal")

first = list(bt.values())[0]
print(f"Period: {first['start']} to {first['end']} | 252d lookback | 10bps one-way costs\n")
print(f"{'Strategy':<30}{'Cum ret':>10}{'Ann ret':>10}{'Ann vol':>10}{'Sharpe':>9}{'Max DD':>10}{'Turnover':>10}")
for name, d in bt.items():
    print(f"{name:<30}{d['cum_return']:>10.2%}{d['ann_return']:>10.2%}{d['ann_vol']:>10.2%}"
          f"{d['sharpe']:>9.3f}{d['max_drawdown']:>10.2%}{d['avg_turnover_per_rebalance']:>10.3f}")

base_sh = bt["Equal-weighted baseline"]["sharpe"]
print(f"\nSharpe vs baseline:")
for name, d in bt.items():
    if name == "Equal-weighted baseline":
        continue
    print(f"  {name:<32}{d['sharpe'] - base_sh:+.3f}")

oos_sh = bt["Optimised, unconstrained"]["sharpe"]
print(f"\nIN-SAMPLE Sharpe (unconstrained max-Sharpe): "
      f"{results['Max-Sharpe (unconstrained)']['sharpe']:.3f}")
print(f"OUT-OF-SAMPLE Sharpe (same strategy):        {oos_sh:.3f}")
print(f"Degradation: {oos_sh - results['Max-Sharpe (unconstrained)']['sharpe']:+.3f} "
      f"({(oos_sh / results['Max-Sharpe (unconstrained)']['sharpe'] - 1) * 100:+.1f}%)")
print(f"Total transaction-cost drag over the period: "
      f"{bt['Optimised, unconstrained']['total_cost_drag']:.2%}")

print("\n" + "=" * 78)
print("7. WEIGHT STABILITY (why the above happens)")
print("=" * 78)
stab, sd, rng = weight_stability(returns, rf)
print(stab.round(3).to_string())
print(f"\nPer-asset weight range across windows:")
print(rng.round(3).to_string())
print(f"\nMean per-asset weight swing: {rng.mean():.3f} "
      f"({rng.mean()*100:.0f} percentage points)")
print(f"Largest single-asset swing: {rng.max():.3f} ({rng.idxmax()})")

print("\n" + "=" * 78)
print("8. MONTE CARLO")
print("=" * 78)
mc = cfg["monte_carlo"]

# IMPORTANT: the Monte Carlo is driven by the OUT-OF-SAMPLE backtest's realised return and
# volatility, not by the in-sample optimised mu. Feeding an in-sample optimised expected
# return (37.9% here) into a forward simulation would compound the very estimation error
# this project set out to measure, and would produce a distribution in which even the 5th
# percentile is a gain. That is a garbage-in artefact, not a risk estimate.
for label, src in [("Optimised (out-of-sample)", bt["Optimised, unconstrained"]),
                   ("Equal-weighted baseline", bt["Equal-weighted baseline"])]:
    paths = simulate_portfolio(src["ann_return"], src["ann_vol"], 100_000,
                               mc["horizon_days"], mc["n_simulations"])
    summ = simulation_summary(paths, 100_000)
    print(f"\n{label}: {mc['n_simulations']:,} GBM paths, {mc['horizon_days']}-day horizon, "
          f"mu={src['ann_return']:.2%}, sigma={src['ann_vol']:.2%}, start $100,000")
    for k, v in summ.items():
        print(f"  {k:<28}" + (f"${v:>12,.0f}" if "value" in k else f"{v:>13.2%}"))

print("\nCaveat: GBM assumes constant volatility and log-normal returns. Section 3 measured "
      "mean\nexcess kurtosis of 5.56, and scripts/run_volatility.py shows conditional "
      "volatility varying\n6.5x across the sample. These tail figures are therefore optimistic.")

print("\n" + "=" * 78)
print("DONE")
print("=" * 78)
