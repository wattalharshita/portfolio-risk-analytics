"""
run_horserace.py — out-of-sample comparison of every allocator in src/strategies.py.

The hypothesis under test: if estimation error in the expected-return vector is what
degrades mean-variance optimisation out-of-sample, then allocators that rely less on
estimated inputs should degrade less. The strategies are ordered by how much they trust
their estimates, and the results are read against that ordering.

Identical treatment for every strategy: same universe, same dates, same 252-day lookback,
same monthly rebalancing, same 10bps cost model, no parameter tuning. The only thing that
varies is the allocation rule.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db import get_conn
from src.metrics import load_returns_wide, TRADING_DAYS
from src.strategies import STRATEGIES, ESTIMATION_RELIANCE
from src.backtest import run_strategy, weight_turnover_stability

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
tickers = cfg["universe"]["tickers"]
bench_ticker = cfg["universe"]["benchmark"]
rf = cfg["risk_free_rate"]
max_w = cfg["max_weight_per_asset"]
opt_cfg = cfg["optimiser"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

with get_conn() as conn:
    returns = load_returns_wide(conn, tickers)
    bench = load_returns_wide(conn, [bench_ticker])[bench_ticker]

print("=" * 112)
print("OUT-OF-SAMPLE STRATEGY HORSE RACE")
print("=" * 112)
print(f"Universe:   {len(tickers)} securities, "
      f"{len(set(cfg['universe']['sectors'].values()))} GICS sectors")
print(f"Period:     {returns.index[0].date()} to {returns.index[-1].date()} "
      f"({len(returns):,} trading days)")
print(f"Protocol:   {opt_cfg['lookback_days']}-day lookback, "
      f"{'monthly' if opt_cfg['rebalance'] == 'M' else 'quarterly'} rebalancing, "
      f"{opt_cfg['transaction_cost_bps']:.0f}bps one-way costs, rf = {rf:.1%}")
print(f"Benchmark:  {bench_ticker}  (equal-weighted index of all S&P 500 constituents)")

results, stability = {}, {}
for name, fn in STRATEGIES.items():
    mw = max_w if "capped" in name.lower() else 1.0
    results[name] = run_strategy(
        returns, fn, rf,
        lookback_days=opt_cfg["lookback_days"],
        rebalance=opt_cfg["rebalance"],
        cost_bps=opt_cfg["transaction_cost_bps"],
        max_weight=mw, benchmark=bench,
    )
    _, rng = weight_turnover_stability(returns, fn, rf, opt_cfg["lookback_days"],
                                       n_windows=12, max_weight=mw)
    stability[name] = rng.mean()
    print(f"  ran {name}")

# ----------------------------------------------------------------- performance
print("\n" + "=" * 112)
print("1. RISK-ADJUSTED PERFORMANCE (net of costs)")
print("=" * 112)
hdr = (f"{'Strategy':<30}{'Relies on':<30}{'Cum':>9}{'Ann':>8}{'Vol':>8}"
       f"{'Sharpe':>8}{'Sortino':>9}{'MaxDD':>9}{'Calmar':>8}")
print(hdr)
print("-" * 112)
for name, d in results.items():
    print(f"{name:<30}{ESTIMATION_RELIANCE[name]:<30}"
          f"{d['cum_return']:>8.1%}{d['ann_return']:>8.1%}{d['ann_vol']:>8.1%}"
          f"{d['sharpe']:>8.3f}{d['sortino']:>9.3f}{d['max_drawdown']:>9.1%}"
          f"{d['calmar']:>8.2f}")

base = results["Equal-weighted (1/N)"]
print(f"\n{'Strategy':<30}{'ΔSharpe vs 1/N':>16}{'ΔMaxDD vs 1/N':>16}")
print("-" * 62)
for name, d in results.items():
    if name == "Equal-weighted (1/N)":
        continue
    print(f"{name:<30}{d['sharpe'] - base['sharpe']:>+16.3f}"
          f"{d['max_drawdown'] - base['max_drawdown']:>+16.1%}")

# ----------------------------------------------------------- cost & complexity
print("\n" + "=" * 112)
print("2. COST, TURNOVER AND CONCENTRATION")
print("=" * 112)
print(f"{'Strategy':<30}{'AvgTurnover':>13}{'CostDrag':>11}{'Rebal':>8}"
      f"{'AvgHoldings':>13}{'AvgMaxWt':>10}{'WeightSwing':>13}")
print("-" * 112)
for name, d in results.items():
    print(f"{name:<30}{d['avg_turnover']:>13.3f}{d['total_cost_drag']:>10.2%}"
          f"{d['n_rebalances']:>8}{d['avg_n_holdings']:>13.1f}"
          f"{d['avg_max_weight']:>10.1%}{stability[name]:>13.1%}")
print("\nWeightSwing = mean per-asset (max − min) weight across 12 rolling windows.")
print("It measures how much the allocator's answer changes when the estimation window moves.")

# ------------------------------------------------------------ benchmark-relative
print("\n" + "=" * 112)
print("3. BENCHMARK-RELATIVE (vs equal-weighted S&P 500)")
print("=" * 112)
print(f"Benchmark cumulative return over the same window: {base['benchmark_cum']:.1%}")
print(f"\n{'Strategy':<30}{'Alpha (ann)':>13}{'Beta':>8}{'TrackErr':>11}{'InfoRatio':>12}")
print("-" * 76)
for name, d in results.items():
    print(f"{name:<30}{d['alpha_annual']:>13.2%}{d['beta']:>8.2f}"
          f"{d['tracking_error']:>11.2%}{d['information_ratio']:>12.3f}")

# ------------------------------------------------- the hypothesis, tested
print("\n" + "=" * 112)
print("4. DOES ESTIMATION RELIANCE PREDICT INSTABILITY?")
print("=" * 112)
sw = pd.Series(stability)
sh = pd.Series({k: v["sharpe"] for k, v in results.items()})
dd = pd.Series({k: abs(v["max_drawdown"]) for k, v in results.items()})
tw = pd.Series({k: v["avg_turnover"] for k, v in results.items()})

print(f"Correlation, weight-swing vs max drawdown:  {sw.corr(dd):+.3f}")
print(f"Correlation, weight-swing vs turnover:      {sw.corr(tw):+.3f}")
print(f"Correlation, weight-swing vs Sharpe:        {sw.corr(sh):+.3f}")
print("\nRanked by weight instability (most stable first):")
for name, v in sw.sort_values().items():
    print(f"  {name:<30}swing {v:>6.1%}   Sharpe {sh[name]:>6.3f}   "
          f"MaxDD {-dd[name]:>6.1%}   turnover {tw[name]:.3f}")

# ------------------------------------------------------------------ equity data
eq = pd.DataFrame({k: v["equity_curve"] for k, v in results.items()})
eq["Benchmark"] = (1 + bench.reindex(eq.index).fillna(0)).cumprod()
out = ROOT / "reports" / "horserace_equity.csv"
out.parent.mkdir(exist_ok=True)
eq.to_csv(out)
print(f"\nEquity curves written to {out.relative_to(ROOT)}")
print("=" * 112)
