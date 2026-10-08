"""Unit tests for allocators, VaR backtesting, and factor construction."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.strategies import STRATEGIES, equal_weight, min_variance, risk_parity
from src.volatility import kupiec_test, christoffersen_independence


@pytest.fixture
def window():
    rng = np.random.default_rng(42)
    cols = list("ABCDE")
    return pd.DataFrame(rng.normal(0.0005, 0.012, (300, 5)),
                        index=pd.date_range("2020-01-01", periods=300, freq="B"),
                        columns=cols)


# ------------------------------------------------------------------ allocators
@pytest.mark.parametrize("name", list(STRATEGIES))
def test_weights_sum_to_one(window, name):
    w = STRATEGIES[name](window, 0.02, 1.0)
    assert abs(w.sum() - 1.0) < 1e-6, f"{name} weights sum to {w.sum()}"


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_weights_are_long_only(window, name):
    w = STRATEGIES[name](window, 0.02, 1.0)
    assert (w >= -1e-9).all(), f"{name} produced a negative weight"


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_weight_cap_is_respected(window, name):
    cap = 0.30
    w = STRATEGIES[name](window, 0.02, cap)
    if name in ("Max-Sharpe (capped)", "Min-variance", "Max-Sharpe (unconstrained)",
                "Max-Sharpe (Ledoit-Wolf)", "Risk parity"):
        assert w.max() <= cap + 1e-6, f"{name} breached cap: max weight {w.max()}"


def test_equal_weight_is_exactly_uniform(window):
    w = equal_weight(window)
    assert np.allclose(w, 1 / window.shape[1])


def test_min_variance_beats_equal_weight_on_variance(window):
    cov = window.cov().values * 252
    w_mv, w_eq = min_variance(window), equal_weight(window)
    assert w_mv @ cov @ w_mv <= w_eq @ cov @ w_eq + 1e-9


def test_risk_parity_equalises_risk_contributions(window):
    cov = window.cov().values * 252
    w = risk_parity(window)
    vol = np.sqrt(w @ cov @ w)
    rc = w * (cov @ w) / vol
    # each asset's share of total risk should be close to 1/n
    assert rc.std() / rc.mean() < 0.05


def test_uncorrelated_equal_assets_give_near_equal_minvar():
    """With identical, independent assets, min-variance must approach equal weighting."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, (2000, 4)), columns=list("WXYZ"))
    w = min_variance(df)
    assert np.allclose(w, 0.25, atol=0.03)


# --------------------------------------------------------------- VaR backtests
def test_kupiec_passes_a_correctly_calibrated_model():
    rng = np.random.default_rng(1)
    breaches = rng.random(2000) < 0.01      # exactly the 99% VaR breach rate
    r = kupiec_test(breaches, 0.99)
    assert not r["reject_at_5pct"]


def test_kupiec_rejects_an_undercalibrated_model():
    breaches = np.zeros(1000, dtype=bool)
    breaches[:60] = True                    # 6% breaches against a 1% target
    r = kupiec_test(breaches, 0.99)
    assert r["reject_at_5pct"]
    assert r["observed_rate"] == pytest.approx(0.06)


def test_kupiec_expected_breach_count():
    breaches = np.zeros(1000, dtype=bool)
    r = kupiec_test(breaches, 0.95)
    assert r["expected_breaches"] == pytest.approx(50.0)


def test_christoffersen_detects_clustering():
    """20 breaches in one consecutive run is maximally clustered."""
    b = np.zeros(1000, dtype=bool)
    b[500:520] = True
    r = christoffersen_independence(b)
    assert r["reject_at_5pct"]
    assert r["p_breach_given_breach"] > r["p_breach_given_calm"]


def test_christoffersen_accepts_independent_breaches():
    rng = np.random.default_rng(7)
    b = rng.random(3000) < 0.02             # i.i.d. by construction
    r = christoffersen_independence(b)
    assert not r["reject_at_5pct"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
