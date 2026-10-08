"""
metrics.py — return and risk metrics computed from the database.

Log returns are used for time-aggregation (they're additive across days: sum of daily
log returns = log return over the period). Simple returns are used for cross-asset
aggregation (a portfolio's simple return is the weight-weighted sum of its constituents'
simple returns; log returns are not).
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import get_conn

TRADING_DAYS = 252


def load_returns_wide(conn, tickers):
    """Wide DataFrame of simple returns, index=date, columns=ticker."""
    frames = []
    for t in tickers:
        df = pd.read_sql_query(
            """SELECT r.date, r.simple_return FROM returns r
               JOIN securities s ON s.security_id = r.security_id
               WHERE s.ticker = ? ORDER BY r.date""",
            conn, params=(t,), parse_dates=["date"],
        ).set_index("date")
        df.columns = [t]
        frames.append(df)
    return pd.concat(frames, axis=1).dropna()


def cagr(prices: pd.Series) -> float:
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1


def annualised_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float) -> float:
    excess = returns.mean() * TRADING_DAYS - rf
    return excess / annualised_vol(returns)


def sortino_ratio(returns: pd.Series, rf: float) -> float:
    downside = returns[returns < 0]
    downside_dev = downside.std() * np.sqrt(TRADING_DAYS)
    excess = returns.mean() * TRADING_DAYS - rf
    return excess / downside_dev if downside_dev > 0 else np.nan


def max_drawdown(prices: pd.Series):
    cum = prices / prices.iloc[0]
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    mdd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = cum.loc[:trough_date].idxmax()
    recovery = cum.loc[trough_date:][cum.loc[trough_date:] >= running_max.loc[trough_date]]
    recovery_date = recovery.index[0] if len(recovery) else None
    return {
        "max_drawdown": mdd,
        "peak_date": str(peak_date.date()),
        "trough_date": str(trough_date.date()),
        "duration_days": (trough_date - peak_date).days,
        "recovery_date": str(recovery_date.date()) if recovery_date is not None else "not recovered in sample",
    }


def calmar_ratio(prices: pd.Series) -> float:
    mdd = max_drawdown(prices)["max_drawdown"]
    return cagr(prices) / abs(mdd) if mdd != 0 else np.nan


def historical_var(returns: pd.Series, conf: float) -> float:
    return -np.percentile(returns, (1 - conf) * 100)


def parametric_var(returns: pd.Series, conf: float) -> float:
    from scipy.stats import norm
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - conf)
    return -(mu + z * sigma)


def cvar(returns: pd.Series, conf: float) -> float:
    threshold = np.percentile(returns, (1 - conf) * 100)
    tail = returns[returns <= threshold]
    return -tail.mean()


def full_report(conn, tickers, rf: float, conf_levels=(0.95, 0.99)):
    rets = load_returns_wide(conn, tickers)
    rows = []
    for t in tickers:
        prices = pd.read_sql_query(
            """SELECT p.date, p.adj_close FROM prices p
               JOIN securities s ON s.security_id = p.security_id
               WHERE s.ticker = ? ORDER BY p.date""",
            conn, params=(t,), parse_dates=["date"],
        ).set_index("date")["adj_close"]
        r = rets[t]
        mdd = max_drawdown(prices)
        row = {
            "ticker": t,
            "cagr": cagr(prices),
            "ann_vol": annualised_vol(r),
            "sharpe": sharpe_ratio(r, rf),
            "sortino": sortino_ratio(r, rf),
            "max_drawdown": mdd["max_drawdown"],
            "drawdown_duration_days": mdd["duration_days"],
            "calmar": calmar_ratio(prices),
        }
        for c in conf_levels:
            row[f"hist_var_{int(c*100)}"] = historical_var(r, c)
            row[f"param_var_{int(c*100)}"] = parametric_var(r, c)
            row[f"cvar_{int(c*100)}"] = cvar(r, c)
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


def covariance_matrix(conn, tickers):
    rets = load_returns_wide(conn, tickers)
    return rets.cov() * TRADING_DAYS, rets.corr()


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
    tickers = cfg["universe"]["tickers"]
    with get_conn() as conn:
        report = full_report(conn, tickers, cfg["risk_free_rate"])
        pd.set_option("display.width", 160)
        pd.set_option("display.max_columns", 20)
        print(report.round(4).to_string())
        cov, corr = covariance_matrix(conn, tickers)
        print("\nAnnualised correlation matrix:")
        print(corr.round(3).to_string())
