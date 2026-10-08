"""
run_volatility.py — GARCH conditional volatility and formal VaR model validation.

Fits GARCH(1,1) to the equal-weighted portfolio, then backtests static (constant-
volatility) VaR against GARCH-based VaR using Kupiec coverage and Christoffersen
independence tests. The question is not "can I fit a GARCH model" but "does the
time-varying model produce a VaR that breaches at the right rate, without clustering".
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
from src.volatility import (volatility_clustering_test, fit_garch, forecast_volatility,
                            garch_var_series, static_var_series, backtest_var)

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
tickers = cfg["universe"]["tickers"]
conf = cfg["garch"]["var_confidence"]
p, q = cfg["garch"]["p"], cfg["garch"]["q"]

with get_conn() as conn:
    returns = load_returns_wide(conn, tickers)

port = returns.mean(axis=1)          # equal-weighted portfolio return series
port.name = "portfolio"

print("=" * 96)
print("GARCH VOLATILITY MODELLING & VaR MODEL VALIDATION")
print("=" * 96)
print(f"Series:  equal-weighted portfolio of {len(tickers)} securities")
print(f"Period:  {port.index[0].date()} to {port.index[-1].date()} ({len(port):,} days)")

# ------------------------------------------------------- 1. is GARCH justified?
print("\n" + "=" * 96)
print("1. IS VOLATILITY CLUSTERING PRESENT? (Ljung-Box)")
print("=" * 96)
lb = volatility_clustering_test(port, lags=20)
print(f"{'Series':<26}{'LB statistic':>15}{'p-value':>12}   Interpretation")
print("-" * 96)
print(f"{'Raw returns':<26}{lb['returns_stat']:>15.2f}{lb['returns_pvalue']:>12.4f}   "
      + ("autocorrelated" if lb["returns_pvalue"] < 0.05 else "no serial correlation"))
print(f"{'Squared returns':<26}{lb['squared_stat']:>15.2f}{lb['squared_pvalue']:>12.4f}   "
      + ("VOLATILITY CLUSTERS" if lb["squared_pvalue"] < 0.05 else "no clustering"))
print(f"\nReading: returns themselves are close to unpredictable, but squared returns "
      f"(variance proxy)\nare strongly autocorrelated. Volatility is forecastable even "
      f"when direction is not —\nwhich is the statistical licence to fit GARCH.")

# ------------------------------------------------------------- 2. fit the model
print("\n" + "=" * 96)
print(f"2. GARCH({p},{q}) FIT — Student-t innovations")
print("=" * 96)
g = fit_garch(port, p, q, dist="t")
print(f"{'omega (long-run variance)':<34}{g['omega']:>12.6f}")
print(f"{'alpha (shock reaction)':<34}{g['alpha']:>12.4f}")
print(f"{'beta  (volatility persistence)':<34}{g['beta']:>12.4f}")
print(f"{'alpha + beta (total persistence)':<34}{g['persistence']:>12.4f}")
print(f"{'nu (Student-t deg. freedom)':<34}{g['nu']:>12.2f}")
print(f"{'shock half-life (trading days)':<34}{g['half_life_days']:>12.1f}")
print(f"{'log-likelihood':<34}{g['loglik']:>12.1f}")
print(f"{'AIC / BIC':<34}{g['aic']:>12.1f} / {g['bic']:.1f}")

cv = g["conditional_vol_annual"]
static_vol = port.std() * np.sqrt(TRADING_DAYS)
print(f"\n{'Static (whole-sample) volatility':<40}{static_vol:>10.2%}")
print(f"{'GARCH conditional vol — minimum':<40}{cv.min():>10.2%}  ({cv.idxmin().date()})")
print(f"{'GARCH conditional vol — maximum':<40}{cv.max():>10.2%}  ({cv.idxmax().date()})")
print(f"{'Ratio, calmest to most turbulent day':<40}{cv.max()/cv.min():>10.1f}x")

print(f"\nReading: persistence of {g['persistence']:.3f} means a volatility shock decays "
      f"with a half-life of\n{g['half_life_days']:.0f} trading days. A single static "
      f"volatility number of {static_vol:.1%} is reporting the same\nrisk on the calmest "
      f"day ({cv.min():.1%}) as on the most violent ({cv.max():.1%}) — a "
      f"{cv.max()/cv.min():.0f}x difference.")

print("\nMost turbulent periods identified by the model:")
top = cv.nlargest(200)
for period, grp in top.groupby(top.index.to_period("M")):
    pass
monthly = cv.groupby(cv.index.to_period("M")).mean().nlargest(6)
for per, v in monthly.items():
    print(f"   {per}   mean conditional vol {v:.1%}")

print("\nForward forecast (annualised):")
fc = forecast_volatility(g, horizon=10)
print("   " + "  ".join(f"t+{i+1}: {v:.1%}" for i, v in enumerate(fc[:5])))
print(f"   t+10: {fc[-1]:.1%}   (converging toward the long-run level)")

# ---------------------------------------------------- 3. VaR model backtesting
print("\n" + "=" * 96)
print(f"3. VaR MODEL VALIDATION AT {conf:.0%} CONFIDENCE")
print("=" * 96)

static_v = static_var_series(port, conf)
garch_v = garch_var_series(port, conf, p, q, dist="t")

bt_static = backtest_var(port, static_v, conf, "Static (constant volatility)")
bt_garch = backtest_var(port, garch_v, conf, "GARCH(1,1) conditional")

print(f"{'Model':<32}{'AvgVaR':>9}{'MinVaR':>9}{'MaxVaR':>9}{'Range':>8}"
      f"{'Breaches':>10}{'Expected':>10}{'Rate':>8}")
print("-" * 96)
for bt in (bt_static, bt_garch):
    k = bt["kupiec"]
    rng = f"{bt['var_range_ratio']:.1f}x" if not np.isnan(bt["var_range_ratio"]) else "—"
    print(f"{bt['label']:<32}{bt['avg_var']:>9.2%}{bt['min_var']:>9.2%}{bt['max_var']:>9.2%}"
          f"{rng:>8}{k['breaches']:>10}{k['expected_breaches']:>10.1f}"
          f"{k['observed_rate']:>8.2%}")

print(f"\n{'Model':<32}{'Kupiec LR':>12}{'p-value':>10}{'Verdict':>14}"
      f"{'Christof. LR':>14}{'p-value':>10}{'Verdict':>14}")
print("-" * 106)
for bt in (bt_static, bt_garch):
    k, c = bt["kupiec"], bt["christoffersen"]
    kv = "REJECT" if k["reject_at_5pct"] else "pass"
    cv_ = ("REJECT" if c["reject_at_5pct"]
           else ("pass" if not np.isnan(c["p_value"]) else "n/a"))
    cl = f"{c['lr_statistic']:.3f}" if not np.isnan(c["lr_statistic"]) else "—"
    cp = f"{c['p_value']:.4f}" if not np.isnan(c["p_value"]) else "—"
    print(f"{bt['label']:<32}{k['lr_statistic']:>12.3f}{k['p_value']:>10.4f}{kv:>14}"
          f"{cl:>14}{cp:>10}{cv_:>14}")

print("\nKupiec H0: breach rate equals 1 − confidence. REJECT = the model is miscalibrated.")
print("Christoffersen H0: breaches are independent. REJECT = breaches cluster, so the "
      "model\nfails in runs — during exactly the stress episodes it exists to warn about.")

# ------------------------------------------------------------------- 4. verdict
print("\n" + "=" * 96)
print("4. VERDICT")
print("=" * 96)
ks, kg = bt_static["kupiec"], bt_garch["kupiec"]
cs, cg = bt_static["christoffersen"], bt_garch["christoffersen"]

print(f"Static VaR  : {ks['breaches']} breaches vs {ks['expected_breaches']:.1f} expected "
      f"({ks['observed_rate']:.2%} vs {ks['expected_rate']:.2%}), "
      f"Kupiec p={ks['p_value']:.4f}")
print(f"GARCH VaR   : {kg['breaches']} breaches vs {kg['expected_breaches']:.1f} expected "
      f"({kg['observed_rate']:.2%} vs {kg['expected_rate']:.2%}), "
      f"Kupiec p={kg['p_value']:.4f}")

if not np.isnan(cs["p_value"]):
    print(f"\nBreach clustering — static: P(breach | breach yesterday) = "
          f"{cs['p_breach_given_breach']:.1%} vs P(breach | calm yesterday) = "
          f"{cs['p_breach_given_calm']:.1%}")
if not np.isnan(cg["p_value"]):
    print(f"Breach clustering — GARCH : P(breach | breach yesterday) = "
          f"{cg['p_breach_given_breach']:.1%} vs P(breach | calm yesterday) = "
          f"{cg['p_breach_given_calm']:.1%}")

print(f"\nThe static model reports one number ({bt_static['avg_var']:.2%}) every day for "
      f"{len(port):,} days.\nThe GARCH model ranges from {bt_garch['min_var']:.2%} to "
      f"{bt_garch['max_var']:.2%} — a {bt_garch['var_range_ratio']:.1f}x span — "
      f"adapting capital\nto the volatility regime instead of holding it flat.")

out = ROOT / "reports" / "garch_conditional_vol.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame({
    "conditional_vol_annual": cv,
    "static_vol_annual": static_vol,
    "garch_var": garch_v.reindex(cv.index),
    "static_var": static_v.reindex(cv.index),
    "portfolio_return": port.reindex(cv.index),
}).to_csv(out)
print(f"\nConditional volatility series written to {out.relative_to(ROOT)}")
print("=" * 96)
