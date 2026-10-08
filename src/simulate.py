"""
simulate.py — Monte Carlo simulation of the optimised portfolio via Geometric Brownian Motion.

dS = mu*S*dt + sigma*S*dW, simulated at daily steps and aggregated to a portfolio path.
GBM assumes constant volatility and log-normal returns, which understates fat-tail risk —
noted in docs/findings.md as a documented limitation, not silently glossed over.
"""
import numpy as np
import pandas as pd


def simulate_portfolio(mu_annual: float, vol_annual: float, start_value: float,
                        horizon_days: int, n_sims: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    mu_d = mu_annual * dt
    sigma_d = vol_annual * np.sqrt(dt)
    shocks = rng.normal(mu_d - 0.5 * sigma_d**2, sigma_d, size=(n_sims, horizon_days))
    log_paths = np.cumsum(shocks, axis=1)
    paths = start_value * np.exp(log_paths)
    paths = np.hstack([np.full((n_sims, 1), start_value), paths])
    return paths


def simulation_summary(paths: np.ndarray, start_value: float, conf_levels=(0.95, 0.99)):
    end_values = paths[:, -1]
    total_returns = end_values / start_value - 1
    out = {
        "mean_end_value": end_values.mean(),
        "median_end_value": np.median(end_values),
        "mean_total_return": total_returns.mean(),
    }
    for c in conf_levels:
        var = -np.percentile(total_returns, (1 - c) * 100)
        tail = total_returns[total_returns <= -var]
        cvar = -tail.mean() if len(tail) else np.nan
        out[f"sim_var_{int(c*100)}"] = var
        out[f"sim_cvar_{int(c*100)}"] = cvar
    out["prob_loss"] = (total_returns < 0).mean()
    out["prob_drawdown_gt_20pct"] = ((paths.min(axis=1) / start_value - 1) < -0.20).mean()
    return out


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from src.optimise import main as run_optimise

    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
    opt = run_optimise()
    mc_cfg = cfg["monte_carlo"]

    paths = simulate_portfolio(
        mu_annual=opt["ms_ret"], vol_annual=opt["ms_vol"], start_value=100_000,
        horizon_days=mc_cfg["horizon_days"], n_sims=mc_cfg["n_simulations"],
    )
    summary = simulation_summary(paths, start_value=100_000)

    print(f"\nMonte Carlo — {mc_cfg['n_simulations']:,} paths, "
          f"{mc_cfg['horizon_days']}-day horizon, max-Sharpe portfolio "
          f"(mu={opt['ms_ret']:.2%}, sigma={opt['ms_vol']:.2%})\n")
    for k, v in summary.items():
        if "value" in k:
            print(f"{k:<28}{v:>14,.0f}")
        else:
            print(f"{k:<28}{v:>14.2%}")
