"""
run_factors.py — factor attribution for every strategy in the horse race.

Answers the question the backtest cannot: was the optimiser's outperformance skill, or
was it exposure to a known systematic factor that any investor could have bought cheaply?
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.db import get_conn
from src.metrics import load_returns_wide, TRADING_DAYS
from src.strategies import STRATEGIES
from src.backtest import run_strategy
from src.factors import build_factors, factor_regression, capm_regression

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
tickers, rf = cfg["universe"]["tickers"], cfg["risk_free_rate"]
max_w, oc = cfg["max_weight_per_asset"], cfg["optimiser"]

print("=" * 104)
print("FACTOR ATTRIBUTION")
print("=" * 104)
print("Building factors from the full 505-security cross-section...")
factors = build_factors(rf_annual=rf)
print(f"Factors built: {len(factors):,} daily observations, "
      f"{factors.index[0].date()} to {factors.index[-1].date()}")

print(f"\n{'Factor':<8}{'Ann. return':>14}{'Ann. vol':>12}{'Sharpe':>10}   Description")
print("-" * 104)
desc = {"MKT": "Equal-weighted market excess return",
        "WML": "Winners minus losers (12-1 momentum, monthly)",
        "VOL": "Low-vol minus high-vol (60d realised, monthly)"}
for c in ["MKT", "WML", "VOL"]:
    a = factors[c].mean() * TRADING_DAYS
    v = factors[c].std() * np.sqrt(TRADING_DAYS)
    print(f"{c:<8}{a:>14.2%}{v:>12.2%}{a/v:>10.2f}   {desc[c]}")

print(f"\nFactor correlations:")
print(factors[["MKT", "WML", "VOL"]].corr().round(3).to_string())

with get_conn() as conn:
    returns = load_returns_wide(conn, tickers)

print("\n" + "=" * 104)
print("STRATEGY ATTRIBUTION — 3-factor model (MKT + WML + VOL), Newey-West SE")
print("=" * 104)
print(f"{'Strategy':<30}{'Alpha(ann)':>12}{'t-stat':>9}{'p':>8}{'Sig?':>7}"
      f"{'bMKT':>8}{'bWML':>8}{'bVOL':>8}{'R2':>8}")
print("-" * 104)

rows = {}
for name, fn in STRATEGIES.items():
    mw = max_w if "capped" in name.lower() else 1.0
    bt = run_strategy(returns, fn, rf, oc["lookback_days"], oc["rebalance"],
                      oc["transaction_cost_bps"], mw)
    reg = factor_regression(bt["net_returns"], factors, rf_annual=rf)
    capm = capm_regression(bt["net_returns"], factors, rf_annual=rf)
    rows[name] = (reg, capm, bt)
    sig = "YES" if reg["alpha_significant"] else "no"
    print(f"{name:<30}{reg['alpha_annual']:>12.2%}{reg['alpha_tstat']:>9.2f}"
          f"{reg['alpha_pvalue']:>8.3f}{sig:>7}"
          f"{reg['betas']['MKT']:>8.2f}{reg['betas']['WML']:>8.2f}"
          f"{reg['betas']['VOL']:>8.2f}{reg['r_squared']:>8.3f}")

print("\n" + "=" * 104)
print("CAPM vs 3-FACTOR — how much does adding WML and VOL explain?")
print("=" * 104)
print(f"{'Strategy':<30}{'CAPM alpha':>13}{'CAPM R2':>10}{'3F alpha':>12}{'3F R2':>9}"
      f"{'Alpha absorbed':>16}{'R2 gain':>10}")
print("-" * 104)
for name, (reg, capm, bt) in rows.items():
    absorbed = capm["alpha_annual"] - reg["alpha_annual"]
    print(f"{name:<30}{capm['alpha_annual']:>13.2%}{capm['r_squared']:>10.3f}"
          f"{reg['alpha_annual']:>12.2%}{reg['r_squared']:>9.3f}"
          f"{absorbed:>16.2%}{reg['r_squared']-capm['r_squared']:>10.3f}")

print("\n" + "=" * 104)
print("RETURN DECOMPOSITION — where did the return come from?")
print("=" * 104)
print(f"{'Strategy':<30}{'Total(ann)':>12}{'from MKT':>11}{'from WML':>11}"
      f"{'from VOL':>11}{'Alpha':>11}{'%explained':>12}")
print("-" * 104)
for name, (reg, capm, bt) in rows.items():
    c = reg["return_contribution_annual"]
    total = bt["ann_return"] - rf
    expl = reg["total_explained_annual"]
    pct = expl / total * 100 if total else np.nan
    print(f"{name:<30}{total:>12.2%}{c['MKT']:>11.2%}{c['WML']:>11.2%}"
          f"{c['VOL']:>11.2%}{reg['alpha_annual']:>11.2%}{pct:>11.0f}%")

sig_count = sum(1 for r, _, _ in rows.values() if r["alpha_significant"])
print(f"\nStrategies with alpha statistically distinguishable from zero at 5%: "
      f"{sig_count} of {len(rows)}")
print("=" * 104)
