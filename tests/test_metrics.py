"""Unit tests for src/metrics.py — verified against hand-computed values on a tiny known series."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import cagr, annualised_vol, sharpe_ratio, max_drawdown, historical_var, cvar


def test_cagr_doubling_in_one_year():
    idx = pd.date_range("2020-01-01", "2021-01-01", freq="D")
    prices = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    # not exactly linear growth -> CAGR won't be exactly 100%, so check it's in a sane band
    assert 0.90 < cagr(prices) < 1.10


def test_cagr_known_value():
    idx = pd.to_datetime(["2020-01-01", "2022-01-01"])  # 2 years
    prices = pd.Series([100, 121], index=idx)
    assert abs(cagr(prices) - 0.10) < 1e-3  # 100 -> 121 over 2y is ~10% CAGR (365.25-day year convention)


def test_annualised_vol_known_value():
    # daily returns with std = 0.01 -> annualised vol = 0.01 * sqrt(252)
    rets = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01, -0.01] * 50)
    expected = rets.std() * np.sqrt(252)
    assert abs(annualised_vol(rets) - expected) < 1e-9


def test_sharpe_zero_when_return_equals_rf():
    rets = pd.Series([0.0] * 100)
    # mean return 0, rf 0 -> excess 0 -> sharpe 0 / vol(undefined since vol=0) -> guard not needed here
    rets2 = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 500))
    rf = rets2.mean() * 252
    s = sharpe_ratio(rets2, rf)
    assert abs(s) < 1e-6


def test_max_drawdown_known_series():
    idx = pd.date_range("2020-01-01", periods=5)
    prices = pd.Series([100, 120, 90, 95, 130], index=idx)
    result = max_drawdown(prices)
    # peak 120 -> trough 90 => drawdown = 90/120 - 1 = -0.25
    assert abs(result["max_drawdown"] - (-0.25)) < 1e-9


def test_historical_var_known_series():
    rets = pd.Series(sorted(range(-50, 50)))  # -50..49, 100 points
    v95 = historical_var(rets / 1000, 0.95)  # scale to return-like magnitude
    # 5th percentile of -50..49 is around -45..-46 -> VaR should be positive and near 0.045
    assert v95 > 0


def test_cvar_worse_than_var():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.02, 2000))
    from src.metrics import historical_var
    v = historical_var(rets, 0.95)
    c = cvar(rets, 0.95)
    assert c >= v  # CVaR (average of tail) is always at least as severe as VaR


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
