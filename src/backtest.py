"""
backtest.py — walk-forward out-of-sample backtest with periodic rebalancing.

Methodology, stated explicitly because this is what an interviewer probes:

  Lookahead bias      At each rebalance date t, weights are optimised using ONLY returns
                      strictly before t (the `lookback_days` window ending at t-1), then
                      held through the next period. No future data touches the weights.
  Transaction costs   Charged on turnover at each rebalance (default 10 bps per unit
                      traded, one-way). A frictionless backtest flatters turnover-heavy
                      strategies, so costs are on by default, not an optional extra.
  Survivorship bias   NOT controlled for. The universe is tickers that still exist today,
                      so results are biased upward. Stated as a limitation, not hidden.
  Overfitting         No parameter search. Lookback and rebalance frequency are set once
                      in config.yaml and not tuned against the backtest result.
  Data snooping       One strategy tested, one result reported — including when it loses.

The equal-weighted portfolio over the same dates, with the same cost model, is the
baseline. An optimiser that cannot beat equal-weighting after costs has not earned its
complexity, and this module is built to be able to say so.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import get_conn
from src.metrics import load_returns_wide, TRADING_DAYS, max_drawdown
from src.optimise import max_sharpe, portfolio_stats


def _rebalance_dates(index: pd.DatetimeIndex, start_i: int, freq: str):
    """Indices of the first trading day of each month/quarter at or after start_i."""
    dates = index[start_i:]
    if freq == "M":
        keys = dates.to_period("M")
    elif freq == "Q":
        keys = dates.to_period("Q")
    else:
        raise ValueError(f"unsupported rebalance frequency: {freq}")
    first_of_period = ~pd.Series(keys).duplicated().values
    return [start_i + i for i, is_first in enumerate(first_of_period) if is_first]


def run_backtest(returns: pd.DataFrame, rf: float, lookback_days: int = 252,
                 rebalance: str = "M", cost_bps: float = 10.0, max_weight: float = 1.0,
                 strategy: str = "max_sharpe"):
    """
    Walk-forward backtest. Returns a dict with the daily net-return series and summary stats.

    strategy: "max_sharpe" (re-optimised each rebalance) or "equal" (the baseline).
    """
    n_assets = returns.shape[1]
    index = returns.index
    start_i = lookback_days
    if start_i >= len(index):
        raise ValueError("lookback window is longer than the available history")

    rebal_idx = set(_rebalance_dates(index, start_i, rebalance))

    weights = np.repeat(1 / n_assets, n_assets)
    daily_net = []
    daily_gross = []
    turnover_log = []
    weight_history = []

    for i in range(start_i, len(index)):
        cost_today = 0.0
        if i in rebal_idx:
            window = returns.iloc[i - lookback_days:i]  # strictly before today
            if strategy == "max_sharpe":
                mu = window.mean().values * TRADING_DAYS
                cov = window.cov().values * TRADING_DAYS
                target = max_sharpe(mu, cov, rf, max_weight=max_weight)
            else:
                target = np.repeat(1 / n_assets, n_assets)
            turnover = np.abs(target - weights).sum()
            cost_today = turnover * (cost_bps / 10_000.0)
            turnover_log.append((index[i], turnover))
            weights = target
            weight_history.append((index[i], weights.copy()))

        gross = float(returns.iloc[i].values @ weights)
        daily_gross.append(gross)
        daily_net.append(gross - cost_today)

        # let weights drift with realised returns between rebalances
        grown = weights * (1 + returns.iloc[i].values)
        weights = grown / grown.sum()

    dates = index[start_i:]
    net = pd.Series(daily_net, index=dates)
    gross = pd.Series(daily_gross, index=dates)
    equity = (1 + net).cumprod()

    years = (dates[-1] - dates[0]).days / 365.25
    cum_return = equity.iloc[-1] - 1
    ann_return = (1 + cum_return) ** (1 / years) - 1
    ann_vol = net.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - rf) / ann_vol
    mdd = max_drawdown(equity)

    return {
        "net_returns": net,
        "gross_returns": gross,
        "equity_curve": equity,
        "cum_return": cum_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd["max_drawdown"],
        "drawdown_peak": mdd["peak_date"],
        "drawdown_trough": mdd["trough_date"],
        "total_cost_drag": float((gross - net).sum()),
        "avg_turnover_per_rebalance": float(np.mean([t for _, t in turnover_log])) if turnover_log else 0.0,
        "n_rebalances": len(turnover_log),
        "weight_history": weight_history,
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
    }



def run_strategy(returns: pd.DataFrame, allocator, rf: float, lookback_days: int = 252,
                 rebalance: str = "M", cost_bps: float = 10.0, max_weight: float = 1.0,
                 benchmark: pd.Series = None):
    """
    Generic walk-forward engine. `allocator(window, rf, max_weight) -> weights` is called
    at each rebalance date with ONLY the prior `lookback_days` of returns.

    Identical bias controls to run_backtest(): no lookahead, costs charged on turnover,
    no parameter tuning against the result.
    """
    n_assets = returns.shape[1]
    index = returns.index
    start_i = lookback_days
    if start_i >= len(index):
        raise ValueError("lookback window longer than available history")

    rebal_idx = set(_rebalance_dates(index, start_i, rebalance))
    weights = np.repeat(1 / n_assets, n_assets)
    daily_net, daily_gross, turnover_log, weight_history = [], [], [], []

    for i in range(start_i, len(index)):
        cost_today = 0.0
        if i in rebal_idx:
            window = returns.iloc[i - lookback_days:i]      # strictly before today
            target = allocator(window, rf, max_weight)
            turnover = float(np.abs(target - weights).sum())
            cost_today = turnover * (cost_bps / 10_000.0)
            turnover_log.append((index[i], turnover))
            weights = target
            weight_history.append((index[i], weights.copy()))

        gross = float(returns.iloc[i].values @ weights)
        daily_gross.append(gross)
        daily_net.append(gross - cost_today)
        grown = weights * (1 + returns.iloc[i].values)      # drift between rebalances
        weights = grown / grown.sum()

    dates = index[start_i:]
    net = pd.Series(daily_net, index=dates)
    gross = pd.Series(daily_gross, index=dates)
    equity = (1 + net).cumprod()

    years = (dates[-1] - dates[0]).days / 365.25
    cum_return = equity.iloc[-1] - 1
    ann_return = (1 + cum_return) ** (1 / years) - 1
    ann_vol = net.std() * np.sqrt(TRADING_DAYS)
    downside = net[net < 0].std() * np.sqrt(TRADING_DAYS)
    mdd = max_drawdown(equity)

    out = {
        "net_returns": net, "gross_returns": gross, "equity_curve": equity,
        "cum_return": cum_return, "ann_return": ann_return, "ann_vol": ann_vol,
        "sharpe": (ann_return - rf) / ann_vol,
        "sortino": (ann_return - rf) / downside if downside > 0 else np.nan,
        "max_drawdown": mdd["max_drawdown"],
        "calmar": ann_return / abs(mdd["max_drawdown"]) if mdd["max_drawdown"] else np.nan,
        "total_cost_drag": float((gross - net).sum()),
        "avg_turnover": float(np.mean([t for _, t in turnover_log])) if turnover_log else 0.0,
        "n_rebalances": len(turnover_log),
        "weight_history": weight_history,
        "avg_n_holdings": float(np.mean([(w > 0.005).sum() for _, w in weight_history])),
        "avg_max_weight": float(np.mean([w.max() for _, w in weight_history])),
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
    }

    if benchmark is not None:
        bench = benchmark.reindex(net.index).fillna(0.0)
        out["benchmark_cum"] = float((1 + bench).prod() - 1)
        active = net - bench
        out["tracking_error"] = float(active.std() * np.sqrt(TRADING_DAYS))
        out["information_ratio"] = (float(active.mean() * TRADING_DAYS) / out["tracking_error"]
                                    if out["tracking_error"] > 0 else np.nan)
        cov = np.cov(net.values, bench.values)
        out["beta"] = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan
        out["alpha_annual"] = float((net.mean() - out["beta"] * bench.mean()) * TRADING_DAYS)
    return out


def weight_turnover_stability(returns: pd.DataFrame, allocator, rf: float,
                              lookback_days: int = 252, n_windows: int = 12,
                              max_weight: float = 1.0):
    """Re-run an allocator on rolling windows; return the weight matrix and per-asset range."""
    usable = len(returns) - lookback_days
    step = max(usable // n_windows, 1)
    rows = []
    for k in range(n_windows):
        end = lookback_days + k * step
        if end > len(returns):
            break
        w = allocator(returns.iloc[end - lookback_days:end], rf, max_weight)
        rows.append(pd.Series(w, index=returns.columns,
                              name=str(returns.index[end - 1].date())))
    df = pd.DataFrame(rows)
    return df, (df.max() - df.min())


def weight_stability(returns: pd.DataFrame, rf: float, lookback_days: int = 252,
                     n_windows: int = 8, max_weight: float = 1.0):
    """
    Re-run the optimisation on rolling historical windows and measure how much the
    weights move. Demonstrates MPT's estimation-error sensitivity rather than treating
    the optimiser's output as truth.
    """
    usable = len(returns) - lookback_days
    step = usable // n_windows
    rows = []
    for k in range(n_windows):
        end = lookback_days + k * step
        window = returns.iloc[end - lookback_days:end]
        mu = window.mean().values * TRADING_DAYS
        cov = window.cov().values * TRADING_DAYS
        w = max_sharpe(mu, cov, rf, max_weight=max_weight)
        rows.append(pd.Series(w, index=returns.columns, name=str(returns.index[end - 1].date())))
    df = pd.DataFrame(rows)
    return df, df.std(), (df.max() - df.min())


def main():
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
    tickers = cfg["universe"]["tickers"]
    rf = cfg["risk_free_rate"]

    with get_conn() as conn:
        returns = load_returns_wide(conn, tickers)

    opt = run_backtest(returns, rf, lookback_days=252, rebalance="M", cost_bps=10.0)
    base = run_backtest(returns, rf, lookback_days=252, rebalance="M", cost_bps=10.0,
                        strategy="equal")

    print(f"Out-of-sample backtest: {opt['start']} to {opt['end']} "
          f"(252-day lookback, monthly rebalancing, 10bps costs)\n")
    print(f"{'Strategy':<26}{'Cum ret':>10}{'Ann ret':>10}{'Ann vol':>10}{'Sharpe':>9}{'Max DD':>10}")
    print(f"{'Optimised (max-Sharpe)':<26}{opt['cum_return']:>10.2%}{opt['ann_return']:>10.2%}"
          f"{opt['ann_vol']:>10.2%}{opt['sharpe']:>9.3f}{opt['max_drawdown']:>10.2%}")
    print(f"{'Equal-weighted baseline':<26}{base['cum_return']:>10.2%}{base['ann_return']:>10.2%}"
          f"{base['ann_vol']:>10.2%}{base['sharpe']:>9.3f}{base['max_drawdown']:>10.2%}")

    print(f"\nRebalances: {opt['n_rebalances']}  |  "
          f"avg turnover/rebalance: {opt['avg_turnover_per_rebalance']:.3f}  |  "
          f"total cost drag: {opt['total_cost_drag']:.4%}")
    print(f"Sharpe difference (optimised - baseline): {opt['sharpe'] - base['sharpe']:+.3f}")

    stab, sd, rng = weight_stability(returns, rf)
    print("\nWeight stability across rolling 1-year windows (max-Sharpe re-optimised):")
    print(stab.round(3).to_string())
    print("\nPer-asset weight range (max - min) across windows:")
    print(rng.round(3).to_string())

    return {"optimised": opt, "baseline": base, "stability": stab, "stability_range": rng}


if __name__ == "__main__":
    main()
