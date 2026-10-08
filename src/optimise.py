"""
optimise.py — Markowitz mean-variance optimisation.

Implemented manually with scipy.optimize first (so the maths is provable), then
cross-checked against PyPortfolioOpt (so "I can call a library" isn't the whole story).
Both are kept — see main() below, which prints both and their weight-difference.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import get_conn
from src.metrics import load_returns_wide, TRADING_DAYS


def portfolio_stats(weights, mu, cov):
    ret = weights @ mu
    vol = np.sqrt(weights @ cov @ weights)
    return ret, vol


def neg_sharpe(weights, mu, cov, rf):
    ret, vol = portfolio_stats(weights, mu, cov)
    return -(ret - rf) / vol


def min_variance(mu, cov, target_return=None, max_weight=1.0):
    n = len(mu)
    bounds = [(0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return is not None:
        constraints.append({"type": "eq", "fun": lambda w, mu=mu: w @ mu - target_return})
    x0 = np.repeat(1 / n, n)
    result = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints)
    return result.x


def max_sharpe(mu, cov, rf, max_weight=1.0):
    n = len(mu)
    bounds = [(0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.repeat(1 / n, n)
    result = minimize(neg_sharpe, x0, args=(mu, cov, rf), method="SLSQP",
                       bounds=bounds, constraints=constraints)
    return result.x


def efficient_frontier(mu, cov, n_points=40, max_weight=1.0):
    targets = np.linspace(mu.min(), mu.max(), n_points)
    frontier = []
    for tr in targets:
        w = min_variance(mu, cov, target_return=tr, max_weight=max_weight)
        ret, vol = portfolio_stats(w, mu, cov)
        frontier.append((ret, vol))
    return pd.DataFrame(frontier, columns=["return", "vol"])


def validate_with_pypfopt(mu, cov, rf, tickers):
    """Cross-check the manual max-Sharpe optimum against PyPortfolioOpt."""
    from pypfopt import EfficientFrontier
    mu_s = pd.Series(mu, index=tickers)
    cov_df = pd.DataFrame(cov, index=tickers, columns=tickers)
    ef = EfficientFrontier(mu_s, cov_df, weight_bounds=(0, 1))
    ef.max_sharpe(risk_free_rate=rf)
    w = ef.clean_weights()
    return np.array([w[t] for t in tickers])


def main():
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
    tickers = cfg["universe"]["tickers"]
    rf = cfg["risk_free_rate"]
    max_w = cfg["max_weight_per_asset"]

    with get_conn() as conn:
        rets = load_returns_wide(conn, tickers)

    mu = rets.mean().values * TRADING_DAYS
    cov = rets.cov().values * TRADING_DAYS
    n = len(tickers)

    eq_w = np.repeat(1 / n, n)
    eq_ret, eq_vol = portfolio_stats(eq_w, mu, cov)
    eq_sharpe = (eq_ret - rf) / eq_vol

    mv_w = min_variance(mu, cov)
    mv_ret, mv_vol = portfolio_stats(mv_w, mu, cov)
    mv_sharpe = (mv_ret - rf) / mv_vol

    ms_w = max_sharpe(mu, cov, rf)
    ms_ret, ms_vol = portfolio_stats(ms_w, mu, cov)
    ms_sharpe = (ms_ret - rf) / ms_vol

    ms_w_constrained = max_sharpe(mu, cov, rf, max_weight=max_w)
    msc_ret, msc_vol = portfolio_stats(ms_w_constrained, mu, cov)
    msc_sharpe = (msc_ret - rf) / msc_vol

    pypfopt_w = validate_with_pypfopt(mu, cov, rf, tickers)
    weight_diff = np.abs(ms_w - pypfopt_w).max()

    print(f"Universe: {tickers}\n")
    print(f"{'Portfolio':<28}{'Return':>10}{'Vol':>10}{'Sharpe':>10}")
    print(f"{'Equal-weighted (baseline)':<28}{eq_ret:>10.2%}{eq_vol:>10.2%}{eq_sharpe:>10.3f}")
    print(f"{'Min-volatility':<28}{mv_ret:>10.2%}{mv_vol:>10.2%}{mv_sharpe:>10.3f}")
    print(f"{'Max-Sharpe (unconstrained)':<28}{ms_ret:>10.2%}{ms_vol:>10.2%}{ms_sharpe:>10.3f}")
    print(f"{f'Max-Sharpe (<={max_w:.0%}/asset)':<28}{msc_ret:>10.2%}{msc_vol:>10.2%}{msc_sharpe:>10.3f}")

    print(f"\nMax-Sharpe weights (manual scipy): "
          + ", ".join(f"{t}={w:.3f}" for t, w in zip(tickers, ms_w)))
    print(f"Max-Sharpe weights (PyPortfolioOpt): "
          + ", ".join(f"{t}={w:.3f}" for t, w in zip(tickers, pypfopt_w)))
    print(f"Max abs weight difference (manual vs library): {weight_diff:.4f}")

    improvement = (ms_sharpe - eq_sharpe) / eq_sharpe * 100
    print(f"\nSharpe improvement, max-Sharpe vs equal-weighted baseline: {improvement:+.1f}%")

    return {
        "eq_w": eq_w, "eq_ret": eq_ret, "eq_vol": eq_vol, "eq_sharpe": eq_sharpe,
        "mv_w": mv_w, "mv_ret": mv_ret, "mv_vol": mv_vol, "mv_sharpe": mv_sharpe,
        "ms_w": ms_w, "ms_ret": ms_ret, "ms_vol": ms_vol, "ms_sharpe": ms_sharpe,
        "msc_w": ms_w_constrained, "msc_ret": msc_ret, "msc_vol": msc_vol, "msc_sharpe": msc_sharpe,
        "weight_diff_vs_library": weight_diff, "sharpe_improvement_pct": improvement,
        "mu": mu, "cov": cov, "tickers": tickers,
    }


if __name__ == "__main__":
    main()
