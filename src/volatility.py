"""
volatility.py — GARCH conditional volatility and formal VaR backtesting.

Two things happen here, and they are connected.

**The problem.** Every risk number produced elsewhere in this project — Sharpe, the
covariance matrix, the efficient frontier, historical and parametric VaR, the Monte Carlo
simulation — assumes volatility is a single constant estimated over the whole sample.
Financial volatility is not constant. It clusters: calm periods follow calm periods,
turbulent periods follow turbulent periods. A risk model that ignores this reports the
same risk on the quietest day of the sample as on the most violent.

**The fix.** GARCH(1,1) models conditional variance as a function of yesterday's shock
and yesterday's variance:

    sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

alpha measures how sharply volatility reacts to new shocks; beta how long it persists.
alpha + beta is the persistence of the process — close to 1 means shocks decay slowly.

**The test.** Producing a time-varying volatility estimate is not the same as producing a
*better* one. So both the static and the GARCH-based VaR are backtested formally:

  Kupiec unconditional coverage test  — does the VaR model breach at the right RATE?
                                        A 99% VaR should be exceeded on ~1% of days.
                                        Too many breaches understates risk; too few
                                        wastes capital.
  Christoffersen independence test    — are the breaches INDEPENDENT, or do they cluster?
                                        A model with the right breach rate but clustered
                                        breaches is still broken: it fails exactly when
                                        it matters, during stress.

This pairing is what a risk function actually does, and it is what separates "I fitted a
GARCH model" from "I demonstrated whether the GARCH model was worth fitting."
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import TRADING_DAYS


# ---------------------------------------------------------------- diagnostics
def volatility_clustering_test(returns: pd.Series, lags: int = 20):
    """
    Ljung-Box test on SQUARED returns. Returns are close to unpredictable, but squared
    returns (a proxy for variance) are strongly autocorrelated if volatility clusters.
    A rejected null is the statistical justification for fitting GARCH at all.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    r = returns.dropna()
    lb_ret = acorr_ljungbox(r, lags=[lags], return_df=True)
    lb_sq = acorr_ljungbox(r ** 2, lags=[lags], return_df=True)
    return {
        "lags": lags,
        "returns_stat": float(lb_ret["lb_stat"].iloc[0]),
        "returns_pvalue": float(lb_ret["lb_pvalue"].iloc[0]),
        "squared_stat": float(lb_sq["lb_stat"].iloc[0]),
        "squared_pvalue": float(lb_sq["lb_pvalue"].iloc[0]),
        "clustering_detected": bool(lb_sq["lb_pvalue"].iloc[0] < 0.05),
    }


# --------------------------------------------------------------------- fitting
def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "t"):
    """
    Fit GARCH(p,q) to percentage returns. Student-t innovations by default, because the
    kurtosis analysis in metrics.py already established that returns are not normal —
    forcing normal innovations would reintroduce the exact assumption GARCH is here to
    relax.
    """
    from arch import arch_model
    r = returns.dropna() * 100.0          # arch expects percentage returns
    model = arch_model(r, vol="GARCH", p=p, q=q, dist=dist, mean="Constant")
    res = model.fit(disp="off", show_warning=False)

    params = res.params
    alpha = float(params.get("alpha[1]", np.nan))
    beta = float(params.get("beta[1]", np.nan))
    cond_vol = (res.conditional_volatility / 100.0) * np.sqrt(TRADING_DAYS)
    cond_vol.index = r.index

    return {
        "result": res,
        "omega": float(params.get("omega", np.nan)),
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "nu": float(params.get("nu", np.nan)),      # Student-t degrees of freedom
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "conditional_vol_annual": cond_vol,
        "half_life_days": (np.log(0.5) / np.log(alpha + beta)
                           if 0 < alpha + beta < 1 else np.nan),
    }


def forecast_volatility(fitted, horizon: int = 10):
    """Forward conditional volatility forecast, annualised."""
    f = fitted["result"].forecast(horizon=horizon, reindex=False)
    daily_var = f.variance.values[-1] / (100.0 ** 2)
    return np.sqrt(daily_var) * np.sqrt(TRADING_DAYS)


# ------------------------------------------------------------------ VaR series
def garch_var_series(returns: pd.Series, conf: float = 0.99,
                     p: int = 1, q: int = 1, dist: str = "t") -> pd.Series:
    """
    One-day-ahead VaR from in-sample GARCH conditional volatility, scaled by the fitted
    innovation distribution's quantile. Positive numbers denote loss magnitude.

    Note on honesty: this is an in-sample fit, so the coverage test below is a
    specification check rather than a true out-of-sample forecast evaluation. A rolling
    refit would be the stricter version — flagged as a limitation rather than glossed.
    """
    fitted = fit_garch(returns, p, q, dist)
    res = fitted["result"]
    daily_vol = res.conditional_volatility / 100.0
    mu = float(res.params.get("mu", 0.0)) / 100.0

    if dist == "t" and not np.isnan(fitted["nu"]):
        nu = fitted["nu"]
        # standardised Student-t quantile (unit variance)
        qtl = stats.t.ppf(1 - conf, df=nu) / np.sqrt(nu / (nu - 2))
    else:
        qtl = stats.norm.ppf(1 - conf)

    var = -(mu + qtl * daily_vol.values)
    return pd.Series(var, index=returns.dropna().index)


def static_var_series(returns: pd.Series, conf: float = 0.99) -> pd.Series:
    """Constant parametric VaR from the full-sample mean and standard deviation."""
    r = returns.dropna()
    v = -(r.mean() + stats.norm.ppf(1 - conf) * r.std())
    return pd.Series(np.repeat(v, len(r)), index=r.index)


# ------------------------------------------------------------- VaR backtesting
def kupiec_test(breaches: np.ndarray, conf: float):
    """
    Kupiec (1995) proportion-of-failures test of unconditional coverage.
    H0: the true breach probability equals 1 - conf. Likelihood-ratio statistic is
    chi-square with 1 degree of freedom.
    """
    n = len(breaches)
    x = int(breaches.sum())
    p = 1 - conf
    if x == 0:
        lr = -2 * (n * np.log(1 - p))
    else:
        pi = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - pi) - x * np.log(pi))
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return {
        "observations": n, "breaches": x,
        "observed_rate": x / n, "expected_rate": p,
        "expected_breaches": n * p,
        "lr_statistic": float(lr), "p_value": float(pval),
        "reject_at_5pct": bool(pval < 0.05),
    }


def christoffersen_independence(breaches: np.ndarray):
    """
    Christoffersen (1998) independence test. H0: a breach today is independent of a breach
    yesterday. Rejection means breaches CLUSTER — the model fails in runs, during stress,
    which is precisely when it is relied upon.
    """
    b = breaches.astype(int)
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(b[:-1], b[1:]):
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1

    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return {"lr_statistic": np.nan, "p_value": np.nan, "reject_at_5pct": False,
                "transitions": (n00, n01, n10, n11),
                "note": "insufficient breach transitions to test"}

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(v):
        return np.log(v) if v > 0 else 0.0

    ll_null = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    ll_alt = (n00 * safe_log(1 - pi01) + n01 * safe_log(pi01)
              + n10 * safe_log(1 - pi11) + n11 * safe_log(pi11))
    lr = -2 * (ll_null - ll_alt)
    pval = 1 - stats.chi2.cdf(lr, df=1)
    return {
        "lr_statistic": float(lr), "p_value": float(pval),
        "reject_at_5pct": bool(pval < 0.05),
        "transitions": (n00, n01, n10, n11),
        "p_breach_given_breach": pi11,
        "p_breach_given_calm": pi01,
        "note": "",
    }


def backtest_var(returns: pd.Series, var_series: pd.Series, conf: float, label: str):
    """Align a VaR series with realised returns, count breaches, run both tests."""
    idx = var_series.index.intersection(returns.index)
    r, v = returns.loc[idx], var_series.loc[idx]
    breaches = (r < -v).values

    kup = kupiec_test(breaches, conf)
    chr_ = christoffersen_independence(breaches)
    return {
        "label": label, "confidence": conf,
        "avg_var": float(v.mean()), "min_var": float(v.min()), "max_var": float(v.max()),
        "var_range_ratio": float(v.max() / v.min()) if v.min() > 0 else np.nan,
        "kupiec": kup, "christoffersen": chr_,
        "breach_dates": list(pd.Series(idx)[breaches].dt.strftime("%Y-%m-%d")),
    }
