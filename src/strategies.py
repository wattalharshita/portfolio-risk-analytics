"""
strategies.py — pluggable portfolio allocators.

Every allocator has the same signature:

    allocator(window: pd.DataFrame, rf: float, max_weight: float) -> np.ndarray

`window` is a DataFrame of daily simple returns covering ONLY the lookback period
strictly before the rebalance date. Each returns a long-only weight vector summing to 1.

Why this module exists. The backtest on the 4-security universe showed unconstrained
mean-variance optimisation losing to naive equal weighting out-of-sample, and traced the
cause to estimation error in the expected-return vector. Each allocator below is a
specific, named response to that diagnosis:

  equal_weight        The null hypothesis. If nothing beats 1/N after costs, nothing here
                      has earned its complexity.
  min_variance        Drops mu entirely. Only the covariance matrix is estimated, and
                      covariances are estimated far more precisely than means.
  max_sharpe          Raw Markowitz. The thing under test.
  max_sharpe_capped   Same, with a per-asset cap. Limits how far estimation error can
                      push the portfolio into a single name.
  max_sharpe_shrunk   Ledoit-Wolf shrinkage applied to the covariance matrix, pulling the
                      sample estimate toward a structured target. Addresses instability in
                      Sigma, though not in mu.
  risk_parity         Equalises each asset's contribution to portfolio variance. Uses no
                      return forecast at all.
  hrp                 Hierarchical Risk Parity (Lopez de Prado, 2016). Clusters assets by
                      correlation and allocates recursively down the tree, avoiding the
                      covariance-matrix inversion that amplifies estimation error in
                      Markowitz.

The ordering is deliberate: it runs from "trusts the estimates completely" to "trusts
them barely at all." If estimation error is genuinely the problem, performance should
improve as reliance on estimated inputs decreases.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


# ------------------------------------------------------------------- utilities
def _annualise(window: pd.DataFrame):
    mu = window.mean().values * TRADING_DAYS
    cov = window.cov().values * TRADING_DAYS
    return mu, cov


def _normalise(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.repeat(1 / len(w), len(w))


# ----------------------------------------------------------------- allocators
def equal_weight(window, rf=0.0, max_weight=1.0):
    n = window.shape[1]
    return np.repeat(1 / n, n)


def min_variance(window, rf=0.0, max_weight=1.0):
    _, cov = _annualise(window)
    n = cov.shape[0]
    res = minimize(lambda w: w @ cov @ w, np.repeat(1 / n, n), method="SLSQP",
                   bounds=[(0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return _normalise(res.x)


def max_sharpe(window, rf=0.02, max_weight=1.0):
    mu, cov = _annualise(window)
    n = len(mu)

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / vol if vol > 0 else 0.0

    res = minimize(neg_sharpe, np.repeat(1 / n, n), method="SLSQP",
                   bounds=[(0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return _normalise(res.x)


def max_sharpe_capped(window, rf=0.02, max_weight=0.10):
    return max_sharpe(window, rf, max_weight)


def max_sharpe_shrunk(window, rf=0.02, max_weight=1.0):
    """Ledoit-Wolf shrinkage on the covariance matrix before optimising."""
    from sklearn.covariance import LedoitWolf
    mu = window.mean().values * TRADING_DAYS
    lw = LedoitWolf().fit(window.values)
    cov = lw.covariance_ * TRADING_DAYS
    n = len(mu)

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / vol if vol > 0 else 0.0

    res = minimize(neg_sharpe, np.repeat(1 / n, n), method="SLSQP",
                   bounds=[(0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return _normalise(res.x)


def risk_parity(window, rf=0.0, max_weight=1.0):
    """
    Equal risk contribution: each asset contributes the same share of portfolio variance.
    Minimises the dispersion of risk contributions RC_i = w_i * (Sigma w)_i / sigma_p.
    """
    _, cov = _annualise(window)
    n = cov.shape[0]

    def objective(w):
        vol = np.sqrt(w @ cov @ w)
        if vol <= 0:
            return 1e6
        rc = w * (cov @ w) / vol
        return float(((rc - rc.mean()) ** 2).sum())

    res = minimize(objective, np.repeat(1 / n, n), method="SLSQP",
                   bounds=[(1e-6, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 400, "ftol": 1e-12})
    return _normalise(res.x)


def hrp(window, rf=0.0, max_weight=1.0):
    """Hierarchical Risk Parity (Lopez de Prado, 2016) via PyPortfolioOpt."""
    from pypfopt import HRPOpt
    h = HRPOpt(window)
    h.optimize()
    w = h.clean_weights()
    return _normalise(np.array([w[c] for c in window.columns]))


# --------------------------------------------------------------------- registry
STRATEGIES = {
    "Equal-weighted (1/N)":        equal_weight,
    "Min-variance":                min_variance,
    "Max-Sharpe (unconstrained)":  max_sharpe,
    "Max-Sharpe (capped)":         max_sharpe_capped,
    "Max-Sharpe (Ledoit-Wolf)":    max_sharpe_shrunk,
    "Risk parity":                 risk_parity,
    "Hierarchical Risk Parity":    hrp,
}

# How much each strategy relies on estimated inputs — used to order results and to test
# the hypothesis that reliance on estimation drives out-of-sample degradation.
ESTIMATION_RELIANCE = {
    "Equal-weighted (1/N)":        "none",
    "Min-variance":                "covariance only",
    "Max-Sharpe (unconstrained)":  "mean + covariance",
    "Max-Sharpe (capped)":         "mean + covariance (bounded)",
    "Max-Sharpe (Ledoit-Wolf)":    "mean + shrunk covariance",
    "Risk parity":                 "covariance only",
    "Hierarchical Risk Parity":    "correlation structure only",
}
