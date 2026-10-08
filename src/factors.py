"""
factors.py — factor construction and return attribution.

**What this answers.** The backtest tells you a strategy earned 28% annualised. It does
not tell you *why*. A portfolio that beat the market may simply have loaded on a known
systematic risk factor rather than exhibited any skill. Factor regression separates the
two: it decomposes realised excess return into exposures to systematic factors (beta) and
an unexplained residual (alpha), and reports whether that alpha is statistically
distinguishable from zero.

**An honest note on what these factors are.** Fama-French SMB and HML require market
capitalisation and book-to-market data, which are not present in a price-and-volume
dataset. Rather than claim a Fama-French regression that was not run, the factors below
are constructed from the price cross-section of all 505 S&P 500 constituents:

  MKT  Market excess return — the equal-weighted index of all constituents, minus rf.
  WML  Winners-minus-losers momentum (Jegadeesh & Titman, 1993; Carhart, 1997). Ranks
       securities on trailing 12-month return skipping the most recent month (the standard
       12-1 formation, which avoids the well-documented short-term reversal effect), then
       goes long the top decile and short the bottom decile, rebalanced monthly.
  VOL  Low-volatility minus high-volatility. Long the lowest-volatility decile, short the
       highest, ranked on trailing 60-day realised volatility. A price-based proxy for the
       low-beta / betting-against-beta anomaly (Frazzini & Pedersen, 2014).

MKT and WML are standard and independently documented. VOL is a proxy, and is labelled as
one. What this cannot capture is size and value, so the alpha reported below is alpha
*relative to these three factors only* — a genuinely unexplained residual would need the
full factor set to confirm. That limitation is stated rather than hidden.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import TRADING_DAYS

ROOT = Path(__file__).resolve().parent.parent
SP500_FILE = ROOT / "data" / "raw" / "all_stocks_5yr.csv"
FACTOR_CACHE = ROOT / "data" / "factors.csv"


# ------------------------------------------------------------ factor construction
def _wide_returns_all() -> pd.DataFrame:
    """Daily simple returns for every S&P 500 constituent in the source dataset."""
    raw = pd.read_csv(SP500_FILE, parse_dates=["date"])
    px = raw.pivot(index="date", columns="Name", values="close").sort_index()
    rets = px.pct_change()
    # neutralise the unadjusted corporate actions the ingestion QC pass identified
    return rets.where(rets.abs() <= 0.40)


def build_factors(rf_annual: float = 0.02, decile: float = 0.10,
                  use_cache: bool = True) -> pd.DataFrame:
    """
    Construct MKT, WML and VOL daily factor returns from the full cross-section.

    Long-short legs are equal-weighted within each decile and rebalanced monthly (WML)
    or monthly on trailing 60-day volatility (VOL). Formation always uses information
    strictly prior to the holding period — no lookahead.
    """
    # Constructing factors requires the full 505-security source file (~30MB), which is
    # not committed. The precomputed cache lets a fresh clone or a hosted deployment run
    # the attribution layer without it. Delete data/factors.csv to force a rebuild.
    if use_cache and FACTOR_CACHE.exists():
        f = pd.read_csv(FACTOR_CACHE, index_col=0, parse_dates=True)
        f["RF"] = rf_annual / TRADING_DAYS      # respect the caller's risk-free rate
        return f
    if not SP500_FILE.exists():
        raise FileNotFoundError(
            f"Neither the factor cache ({FACTOR_CACHE.name}) nor the source dataset "
            f"({SP500_FILE.name}) is present. Run: python src/ingest.py --source sp500"
        )

    rets = _wide_returns_all()
    rf_daily = rf_annual / TRADING_DAYS

    mkt = rets.mean(axis=1) - rf_daily
    mkt.name = "MKT"

    # --- momentum: 12-1 month formation, monthly rebalance ------------------
    px = (1 + rets.fillna(0)).cumprod()
    form_end = px.shift(21)                    # skip most recent month
    form_start = px.shift(252)                 # 12 months back
    mom_signal = (form_end / form_start) - 1

    # --- low volatility: trailing 60-day realised vol ------------------------
    vol_signal = rets.rolling(60).std()

    month_key = rets.index.to_period("M")
    is_rebal = ~pd.Series(month_key, index=rets.index).duplicated()

    wml, volf = [], []
    mom_long = mom_short = vol_long = vol_short = None

    for dt in rets.index:
        if is_rebal.loc[dt]:
            m = mom_signal.loc[dt].dropna()
            if len(m) > 50:
                k = max(int(len(m) * decile), 5)
                ranked = m.sort_values()
                mom_short, mom_long = ranked.index[:k], ranked.index[-k:]
            v = vol_signal.loc[dt].dropna()
            if len(v) > 50:
                k = max(int(len(v) * decile), 5)
                ranked = v.sort_values()
                vol_long, vol_short = ranked.index[:k], ranked.index[-k:]

        row = rets.loc[dt]
        wml.append(row[mom_long].mean() - row[mom_short].mean()
                   if mom_long is not None else np.nan)
        volf.append(row[vol_long].mean() - row[vol_short].mean()
                    if vol_long is not None else np.nan)

    f = pd.DataFrame({
        "MKT": mkt,
        "WML": pd.Series(wml, index=rets.index),
        "VOL": pd.Series(volf, index=rets.index),
    })
    f["RF"] = rf_daily
    return f.dropna()


# --------------------------------------------------------------------- regression
def factor_regression(portfolio_returns: pd.Series, factors: pd.DataFrame,
                      factor_cols=("MKT", "WML", "VOL"), rf_annual: float = 0.02):
    """
    OLS of portfolio EXCESS return on the factor set, with Newey-West standard errors
    (5 lags) to correct for heteroskedasticity and autocorrelation in daily data.
    Ordinary OLS standard errors would overstate significance here.
    """
    import statsmodels.api as sm

    idx = portfolio_returns.index.intersection(factors.index)
    y = portfolio_returns.loc[idx] - factors.loc[idx, "RF"]
    X = sm.add_constant(factors.loc[idx, list(factor_cols)])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    alpha_daily = model.params["const"]
    out = {
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "alpha_daily": float(alpha_daily),
        "alpha_annual": float(alpha_daily * TRADING_DAYS),
        "alpha_tstat": float(model.tvalues["const"]),
        "alpha_pvalue": float(model.pvalues["const"]),
        "alpha_significant": bool(model.pvalues["const"] < 0.05),
        "betas": {c: float(model.params[c]) for c in factor_cols},
        "tstats": {c: float(model.tvalues[c]) for c in factor_cols},
        "pvalues": {c: float(model.pvalues[c]) for c in factor_cols},
        "model": model,
    }

    # variance decomposition: share of explained variation attributable to each factor
    contrib = {}
    for c in factor_cols:
        contrib[c] = float(model.params[c] * factors.loc[idx, c].mean() * TRADING_DAYS)
    out["return_contribution_annual"] = contrib
    out["total_explained_annual"] = float(sum(contrib.values()))
    return out


def capm_regression(portfolio_returns: pd.Series, factors: pd.DataFrame,
                    rf_annual: float = 0.02):
    """Single-factor CAPM, for comparison against the multi-factor model."""
    return factor_regression(portfolio_returns, factors, ("MKT",), rf_annual)
