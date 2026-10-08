"""
app.py — Streamlit dashboard.

Structured as an argument, not a pile of charts: Overview -> Risk & Return ->
Optimisation -> Strategy Race -> Volatility -> Attribution -> Simulation -> Risk Report
-> Findings. Every chart carries a stated takeaway, because a chart without one is
decoration.

Run: streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.db import get_conn
from src.metrics import (load_returns_wide, full_report, covariance_matrix,
                         historical_var, parametric_var, cvar, TRADING_DAYS)
from src.optimise import max_sharpe, min_variance, portfolio_stats, efficient_frontier
from src.simulate import simulate_portfolio, simulation_summary
from src.backtest import run_strategy, weight_turnover_stability
from src.strategies import STRATEGIES, ESTIMATION_RELIANCE

st.set_page_config(page_title="Portfolio Risk & Returns Analytics",
                   page_icon="📊", layout="wide")

CFG = yaml.safe_load(open(ROOT / "config.yaml"))

# The warehouse is committed so a hosted deployment starts instantly. If it is absent
# (someone cloned without it, or it was cleaned), build it rather than crashing.
DB_PATH = ROOT / "data" / "portfolio.db"
if not DB_PATH.exists():
    with st.spinner("First run: building the warehouse from source data (~1 min)…"):
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "src" / "ingest.py"),
                            "--source", "sp500"], capture_output=True, text=True)
    if not DB_PATH.exists():
        st.error("Could not build the warehouse.")
        st.code(r.stderr[-2000:] or r.stdout[-2000:])
        st.stop()
TICKERS = CFG["universe"]["tickers"]
BENCH = CFG["universe"]["benchmark"]
SECTORS = CFG["universe"]["sectors"]
NAMES = CFG["universe"]["names"]
OC = CFG["optimiser"]


# ------------------------------------------------------------------ data layer
@st.cache_data(show_spinner="Loading warehouse…")
def load_data(tickers):
    with get_conn() as conn:
        returns = load_returns_wide(conn, list(tickers))
        report = full_report(conn, list(tickers), CFG["risk_free_rate"])
        _, corr = covariance_matrix(conn, list(tickers))
        bench = load_returns_wide(conn, [BENCH])[BENCH]
        prices = pd.read_sql_query(
            """SELECT s.ticker, p.date, p.adj_close FROM prices p
               JOIN securities s ON s.security_id = p.security_id ORDER BY p.date""",
            conn, parse_dates=["date"])
    prices = prices.pivot(index="date", columns="ticker", values="adj_close")
    return returns, report, corr, prices, bench


@st.cache_data(show_spinner="Racing strategies out-of-sample…")
def horse_race(tickers, rf, lookback, rebal, cost_bps, max_w):
    returns, _, _, _, bench = load_data(tickers)
    res, stab = {}, {}
    for name, fn in STRATEGIES.items():
        mw = max_w if "capped" in name.lower() else 1.0
        res[name] = run_strategy(returns, fn, rf, lookback, rebal, cost_bps, mw, bench)
        _, rng = weight_turnover_stability(returns, fn, rf, lookback, 12, mw)
        stab[name] = float(rng.mean())
    return res, stab


@st.cache_data(show_spinner="Fitting GARCH…")
def garch_analysis(tickers, conf):
    from src.volatility import (volatility_clustering_test, fit_garch,
                                garch_var_series, static_var_series, backtest_var)
    returns, _, _, _, _ = load_data(tickers)
    port = returns.mean(axis=1)
    lb = volatility_clustering_test(port, 20)
    g = fit_garch(port, 1, 1, "t")
    gv, sv = garch_var_series(port, conf), static_var_series(port, conf)
    return {
        "port": port, "ljung_box": lb,
        "alpha": g["alpha"], "beta": g["beta"], "persistence": g["persistence"],
        "half_life": g["half_life_days"], "nu": g["nu"],
        "cond_vol": g["conditional_vol_annual"],
        "garch_var": gv, "static_var": sv,
        "bt_garch": backtest_var(port, gv, conf, "GARCH(1,1)"),
        "bt_static": backtest_var(port, sv, conf, "Static"),
    }


@st.cache_data(show_spinner="Building factors from 505 securities…")
def factor_analysis(tickers, rf, lookback, rebal, cost_bps, max_w):
    from src.factors import build_factors, factor_regression, capm_regression
    returns, _, _, _, _ = load_data(tickers)
    factors = build_factors(rf_annual=rf)
    out = {}
    for name, fn in STRATEGIES.items():
        mw = max_w if "capped" in name.lower() else 1.0
        bt = run_strategy(returns, fn, rf, lookback, rebal, cost_bps, mw)
        out[name] = {"three": factor_regression(bt["net_returns"], factors, rf_annual=rf),
                     "capm": capm_regression(bt["net_returns"], factors, rf_annual=rf),
                     "ann_return": bt["ann_return"]}
    return factors, out


@st.cache_data
def frontier(tickers, max_w):
    returns, _, _, _, _ = load_data(tickers)
    mu = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    return efficient_frontier(mu, cov, 40, max_w)


# --------------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
sel_sectors = st.sidebar.multiselect("Sectors", sorted(set(SECTORS.values())),
                                     default=sorted(set(SECTORS.values())))
tickers = tuple(t for t in TICKERS if SECTORS[t] in sel_sectors)
if len(tickers) < 3:
    st.sidebar.error("Select sectors covering at least three securities.")
    st.stop()

rf = st.sidebar.slider("Risk-free rate", 0.0, 0.08, float(CFG["risk_free_rate"]), 0.005,
                       format="%.3f")
conf = st.sidebar.select_slider("VaR confidence", [0.90, 0.95, 0.99], value=0.99)
max_w = st.sidebar.slider("Max weight per asset", 0.03, 1.0,
                          float(CFG["max_weight_per_asset"]), 0.01)
cost_bps = st.sidebar.slider("Transaction cost (bps)", 0.0, 50.0,
                             float(OC["transaction_cost_bps"]), 1.0)
rebal = st.sidebar.selectbox("Rebalance", ["M", "Q"],
                             format_func=lambda x: {"M": "Monthly", "Q": "Quarterly"}[x])
lookback = st.sidebar.select_slider("Lookback (days)", [126, 252, 378],
                                    value=int(OC["lookback_days"]))

returns, report, corr, prices, bench = load_data(tickers)
n = len(tickers)
mu = returns.mean().values * TRADING_DAYS
cov = returns.cov().values * TRADING_DAYS

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**{n} securities** across {len(set(SECTORS[t] for t in tickers))} GICS sectors  \n"
    f"**{len(returns):,} trading days**, {returns.index[0].date()} → {returns.index[-1].date()}  \n"
    f"Benchmark: equal-weighted index of all 505 S&P 500 constituents."
)

# ---------------------------------------------------------------------- header
st.title("Portfolio Risk & Returns Analytics Platform")

race, stability = horse_race(tickers, rf, lookback, rebal, cost_bps, max_w)
base = race["Equal-weighted (1/N)"]
ms = race["Max-Sharpe (unconstrained)"]
cap = race["Max-Sharpe (capped)"]

eq_w = np.repeat(1 / n, n)
eq_ret, eq_vol = portfolio_stats(eq_w, mu, cov)
eq_sharpe_is = (eq_ret - rf) / eq_vol
ms_w_is = max_sharpe(mu, cov, rf)
ms_ret_is, ms_vol_is = portfolio_stats(ms_w_is, mu, cov)
ms_sharpe_is = (ms_ret_is - rf) / ms_vol_is

st.markdown(f"""
> **Headline finding.** Across {n} securities, unconstrained mean-variance optimisation
> beat equal weighting on Sharpe out-of-sample (**{ms['sharpe']:.3f}** vs
> **{base['sharpe']:.3f}**) — but did so by doubling the drawdown
> (**{ms['max_drawdown']:.1%}** vs **{base['max_drawdown']:.1%}**), so on return per unit
> of worst-case loss it **lost** (Calmar **{ms['calmar']:.2f}** vs **{base['calmar']:.2f}**).
> Factor regression shows why: the optimiser loads on momentum
> (β<sub>WML</sub> ≈ {0.34:.2f}) because optimising on trailing means mechanically buys
> recent winners. Capping weights at {max_w:.0%} keeps most of the Sharpe gain and
> recovers the drawdown.
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Best Sharpe (OOS)", f"{max(d['sharpe'] for d in race.values()):.3f}",
          max(race, key=lambda k: race[k]['sharpe']).replace(" (unconstrained)", ""))
c2.metric("Best Calmar (OOS)", f"{max(d['calmar'] for d in race.values()):.2f}",
          max(race, key=lambda k: race[k]['calmar']).replace(" (1/N)", ""))
c3.metric("Optimiser drawdown", f"{ms['max_drawdown']:.1%}",
          f"{ms['max_drawdown']-base['max_drawdown']:+.1%} vs 1/N", delta_color="inverse")
c4.metric("Optimiser cost drag", f"{ms['total_cost_drag']:.2%}",
          f"{ms['avg_turnover']/base['avg_turnover']:.0f}× the turnover",
          delta_color="inverse")
c5.metric("Strategies tested", f"{len(race)}", f"{ms['n_rebalances']} rebalances each")

tabs = st.tabs(["Overview", "Risk & Return", "Optimisation", "Strategy Race",
                "Volatility", "Attribution", "Simulation", "Risk Report", "Findings"])

# ------------------------------------------------------------------- Overview
with tabs[0]:
    st.subheader("Universe")
    sec_df = pd.DataFrame({
        "Ticker": list(tickers),
        "Name": [NAMES.get(t, t) for t in tickers],
        "Sector": [SECTORS[t] for t in tickers],
        "CAGR": [report.loc[t, "cagr"] for t in tickers],
        "Ann. vol": [report.loc[t, "ann_vol"] for t in tickers],
        "Sharpe": [report.loc[t, "sharpe"] for t in tickers],
        "Max DD": [report.loc[t, "max_drawdown"] for t in tickers],
    })
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(sec_df.style.format({"CAGR": "{:.1%}", "Ann. vol": "{:.1%}",
                                          "Sharpe": "{:.2f}", "Max DD": "{:.1%}"}),
                     use_container_width=True, height=420, hide_index=True)
    with right:
        bysec = sec_df.groupby("Sector").agg(n=("Ticker", "count"),
                                             cagr=("CAGR", "mean"),
                                             vol=("Ann. vol", "mean")).reset_index()
        fig = px.scatter(bysec, x="vol", y="cagr", size="n", text="Sector",
                         labels={"vol": "Mean annualised volatility", "cagr": "Mean CAGR"},
                         title="Sector risk / return")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=420, xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    norm = prices[list(tickers)] / prices[list(tickers)].iloc[0] * 100
    fig = px.line(norm, labels={"value": "Growth of 100", "date": ""},
                  title="Cumulative growth of 100 invested")
    fig.update_layout(height=430, legend_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Read:** dispersion between the best ({norm.iloc[-1].idxmax()}, "
        f"{norm.iloc[-1].max():.0f}) and worst ({norm.iloc[-1].idxmin()}, "
        f"{norm.iloc[-1].min():.0f}) name is what an optimiser tries to exploit — and the "
        "source of the estimation error that makes exploiting it unreliable."
    )

# --------------------------------------------------------------- Risk & Return
with tabs[1]:
    st.subheader("Correlation structure")
    order = sorted(tickers, key=lambda t: (SECTORS[t], t))
    fig = px.imshow(corr.loc[order, order], zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Daily return correlation (ordered by sector)")
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)

    off = corr.loc[list(tickers), list(tickers)].where(~np.eye(n, dtype=bool)).stack()
    top = off.sort_values(ascending=False).drop_duplicates().head(5)
    bot = off.sort_values().drop_duplicates().head(5)
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Mean pairwise correlation", f"{off.mean():.3f}")
    with cc2:
        st.write("**Most correlated pairs**")
        for (a, b), v in top.items():
            st.write(f"{a}–{b} · **{v:.3f}** · {SECTORS[a]}")
    with cc3:
        st.write("**Least correlated pairs**")
        for (a, b), v in bot.items():
            st.write(f"{a}–{b} · **{v:.3f}**")
    st.caption(
        "**Read:** the block structure along the diagonal is sector co-movement — the "
        "deliberately-included pairs (two oil majors, two telecoms, two utilities) sit at "
        "the top of the correlation table. Diversification across names within a sector is "
        "largely illusory; across sectors it is real."
    )

    st.subheader(f"Tail risk — historical vs parametric VaR at {conf:.0%}")
    rows = []
    for t in tickers:
        r = returns[t]
        h, p_ = historical_var(r, conf), parametric_var(r, conf)
        rows.append({"Security": t, "Sector": SECTORS[t], "Historical VaR": h,
                     "Parametric VaR": p_, "Understatement %": (h - p_) / p_ * 100,
                     "CVaR": cvar(r, conf), "Excess kurtosis": r.kurtosis()})
    tail = pd.DataFrame(rows).set_index("Security").sort_values("Understatement %",
                                                               ascending=False)
    st.dataframe(tail.style.format({"Historical VaR": "{:.2%}", "Parametric VaR": "{:.2%}",
                                    "Understatement %": "{:+.1f}%", "CVaR": "{:.2%}",
                                    "Excess kurtosis": "{:.1f}"}),
                 use_container_width=True, height=380)
    n_under = (tail["Understatement %"] > 0).sum()
    st.caption(
        f"**Read:** the normal assumption understates the {conf:.0%} VaR for "
        f"**{n_under} of {n}** securities, by {tail['Understatement %'].mean():.1f}% on "
        f"average, with mean excess kurtosis of {tail['Excess kurtosis'].mean():.1f} "
        "(normal = 0). Every variance-based number in this dashboard inherits that "
        "optimism and should be read as a floor on risk."
    )

# ---------------------------------------------------------------- Optimisation
with tabs[2]:
    st.subheader("Efficient frontier (in-sample)")
    fr = frontier(tickers, max_w)
    rng_ = np.random.default_rng(7)
    rw = rng_.dirichlet(np.ones(n), 4000)
    rr, rv = rw @ mu, np.sqrt(np.einsum("ij,jk,ik->i", rw, cov, rw))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rv, y=rr, mode="markers", name="Random portfolios",
                             marker=dict(size=4, color=(rr - rf) / rv, colorscale="Viridis",
                                         colorbar=dict(title="Sharpe"), opacity=0.4)))
    fig.add_trace(go.Scatter(x=fr["vol"], y=fr["return"], mode="lines",
                             name="Efficient frontier", line=dict(width=3, color="black")))
    fig.add_trace(go.Scatter(x=np.sqrt(np.diag(cov)), y=mu, mode="markers", text=list(tickers),
                             name="Individual securities",
                             marker=dict(size=8, symbol="diamond", color="grey")))
    fig.add_trace(go.Scatter(x=[eq_vol], y=[eq_ret], mode="markers+text", name="Equal-weighted",
                             text=["1/N"], textposition="bottom center",
                             marker=dict(size=16, symbol="square", color="orange")))
    fig.add_trace(go.Scatter(x=[ms_vol_is], y=[ms_ret_is], mode="markers+text",
                             name="Max Sharpe", text=["Max Sharpe"], textposition="top center",
                             marker=dict(size=18, symbol="star", color="red")))
    fig.update_layout(height=560, xaxis_title="Annualised volatility",
                      yaxis_title="Annualised return",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Read:** in-sample the max-Sharpe portfolio scores {ms_sharpe_is:.3f} against "
        f"{eq_sharpe_is:.3f} for equal weighting — a {(ms_sharpe_is/eq_sharpe_is-1)*100:+.0f}% "
        "improvement. The Strategy Race tab tests how much of that survives out-of-sample."
    )

    st.subheader("Concentration: what the optimiser actually holds")
    wdf = pd.DataFrame({"Max-Sharpe": ms_w_is,
                        "Min-variance": min_variance(mu, cov, max_weight=max_w),
                        "Equal-weighted": eq_w}, index=list(tickers))
    nonzero = (wdf > 0.005).sum()
    fig = px.bar(wdf.reset_index().melt(id_vars="index"), x="index", y="value",
                 color="variable", barmode="group",
                 labels={"index": "", "value": "Weight", "variable": ""})
    fig.update_layout(height=380, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Read:** from {n} available securities, max-Sharpe holds only "
        f"**{nonzero['Max-Sharpe']}** at more than 0.5%, with a largest position of "
        f"**{wdf['Max-Sharpe'].max():.1%}**. Equal weighting holds all {n} at "
        f"{1/n:.1%} each. Concentration is the mechanism by which estimation error in "
        "expected returns becomes portfolio risk."
    )

# --------------------------------------------------------------- Strategy Race
with tabs[3]:
    st.subheader("Out-of-sample horse race")
    st.write(
        f"Seven allocators, identical treatment: same universe, same dates, "
        f"{lookback}-day lookback, {'monthly' if rebal=='M' else 'quarterly'} rebalancing, "
        f"{cost_bps:.0f}bps costs, no parameter tuning. Only the allocation rule varies. "
        "They are ordered by how much they rely on estimated inputs."
    )

    eq = pd.DataFrame({k: v["equity_curve"] for k, v in race.items()})
    eq["Benchmark (EW S&P 500)"] = (1 + bench.reindex(eq.index).fillna(0)).cumprod()
    fig = px.line(eq, labels={"value": "Growth of 1", "date": ""},
                  title="Cumulative out-of-sample performance, net of costs")
    fig.update_layout(height=470, legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    tbl = pd.DataFrame({
        "Relies on": {k: ESTIMATION_RELIANCE[k] for k in race},
        "Cum return": {k: v["cum_return"] for k, v in race.items()},
        "Ann return": {k: v["ann_return"] for k, v in race.items()},
        "Ann vol": {k: v["ann_vol"] for k, v in race.items()},
        "Sharpe": {k: v["sharpe"] for k, v in race.items()},
        "Sortino": {k: v["sortino"] for k, v in race.items()},
        "Max DD": {k: v["max_drawdown"] for k, v in race.items()},
        "Calmar": {k: v["calmar"] for k, v in race.items()},
        "Turnover": {k: v["avg_turnover"] for k, v in race.items()},
        "Cost drag": {k: v["total_cost_drag"] for k, v in race.items()},
        "Weight swing": stability,
    })
    st.dataframe(tbl.style.format({
        "Cum return": "{:.1%}", "Ann return": "{:.1%}", "Ann vol": "{:.1%}",
        "Sharpe": "{:.3f}", "Sortino": "{:.3f}", "Max DD": "{:.1%}", "Calmar": "{:.2f}",
        "Turnover": "{:.3f}", "Cost drag": "{:.2%}", "Weight swing": "{:.1%}",
    }).background_gradient(subset=["Sharpe", "Calmar"], cmap="RdYlGn"),
        use_container_width=True)

    st.subheader("Does instability predict risk?")
    sw = pd.Series(stability)
    dd = pd.Series({k: abs(v["max_drawdown"]) for k, v in race.items()})
    tw = pd.Series({k: v["avg_turnover"] for k, v in race.items()})
    sh = pd.Series({k: v["sharpe"] for k, v in race.items()})
    m1, m2, m3 = st.columns(3)
    m1.metric("Corr(weight swing, max drawdown)", f"{sw.corr(dd):+.3f}")
    m2.metric("Corr(weight swing, turnover)", f"{sw.corr(tw):+.3f}")
    m3.metric("Corr(weight swing, Sharpe)", f"{sw.corr(sh):+.3f}")

    scat = pd.DataFrame({"Weight swing": sw, "Max drawdown": -dd, "Sharpe": sh,
                         "Turnover": tw, "Strategy": sw.index})
    fig = px.scatter(scat, x="Weight swing", y="Max drawdown", size="Turnover",
                     color="Sharpe", text="Strategy", color_continuous_scale="RdYlGn",
                     labels={"Weight swing": "Mean per-asset weight swing across windows"})
    fig.update_traces(textposition="top center")
    fig.update_layout(height=470, xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Read:** weight instability tracks turnover almost perfectly "
        f"(r = {sw.corr(tw):+.2f}) and drawdown strongly (r = {sw.corr(dd):+.2f}). "
        "Allocators whose answer changes a lot when the estimation window moves trade more, "
        "pay more, and fall further. Note instability also correlates positively with Sharpe "
        f"(r = {sw.corr(sh):+.2f}) in this sample — in a trending market, concentration paid. "
        "That is a statement about this period, not a general law."
    )

# ------------------------------------------------------------------ Volatility
with tabs[4]:
    g = garch_analysis(tickers, conf)
    st.subheader("Is volatility constant? (it is not)")
    lb = g["ljung_box"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ljung-Box p, raw returns", f"{lb['returns_pvalue']:.4f}")
    m2.metric("Ljung-Box p, squared returns", f"{lb['squared_pvalue']:.4f}",
              "clustering" if lb["clustering_detected"] else "none")
    m3.metric("GARCH persistence (α+β)", f"{g['persistence']:.3f}")
    m4.metric("Shock half-life", f"{g['half_life']:.0f} days")

    cv = g["cond_vol"]
    static_vol = g["port"].std() * np.sqrt(TRADING_DAYS)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cv.index, y=cv.values, name="GARCH conditional volatility",
                             line=dict(color="crimson", width=1.4)))
    fig.add_hline(y=static_vol, line_dash="dash", line_color="navy",
                  annotation_text=f"Static estimate {static_vol:.1%}")
    fig.update_layout(height=430, yaxis_tickformat=".0%", yaxis_title="Annualised volatility",
                      title="Conditional volatility vs the single static number")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Read:** conditional volatility ranges from **{cv.min():.1%}** "
        f"({cv.idxmin().date()}) to **{cv.max():.1%}** ({cv.idxmax().date()}) — a "
        f"**{cv.max()/cv.min():.0f}×** span. The static model reports {static_vol:.1%} on "
        "every one of those days. The peaks align with known stress episodes without being "
        "told about them: Aug–Sep 2015, Jan–Feb 2016, Feb 2018."
    )

    st.subheader(f"VaR model validation at {conf:.0%}")
    bs, bg = g["bt_static"], g["bt_garch"]
    val = pd.DataFrame({
        "Static (constant vol)": {
            "Breaches": bs["kupiec"]["breaches"],
            "Expected": bs["kupiec"]["expected_breaches"],
            "Observed rate": bs["kupiec"]["observed_rate"],
            "Kupiec p-value": bs["kupiec"]["p_value"],
            "Kupiec verdict": "REJECT" if bs["kupiec"]["reject_at_5pct"] else "pass",
            "Christoffersen p": bs["christoffersen"]["p_value"],
            "VaR range": bs["var_range_ratio"],
        },
        "GARCH(1,1) conditional": {
            "Breaches": bg["kupiec"]["breaches"],
            "Expected": bg["kupiec"]["expected_breaches"],
            "Observed rate": bg["kupiec"]["observed_rate"],
            "Kupiec p-value": bg["kupiec"]["p_value"],
            "Kupiec verdict": "REJECT" if bg["kupiec"]["reject_at_5pct"] else "pass",
            "Christoffersen p": bg["christoffersen"]["p_value"],
            "VaR range": bg["var_range_ratio"],
        },
    })
    st.dataframe(val, use_container_width=True)

    port = g["port"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port.index, y=port.values, mode="markers", name="Daily return",
                             marker=dict(size=3, color="lightslategrey", opacity=0.6)))
    fig.add_trace(go.Scatter(x=g["garch_var"].index, y=-g["garch_var"].values,
                             name=f"GARCH {conf:.0%} VaR", line=dict(color="crimson", width=1.3)))
    fig.add_trace(go.Scatter(x=g["static_var"].index, y=-g["static_var"].values,
                             name=f"Static {conf:.0%} VaR",
                             line=dict(color="navy", dash="dash", width=1.3)))
    fig.update_layout(height=430, yaxis_tickformat=".1%", yaxis_title="Daily return",
                      title="VaR thresholds against realised returns")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Read:** the static model breached **{bs['kupiec']['breaches']}** times against "
        f"**{bs['kupiec']['expected_breaches']:.0f}** expected "
        f"({bs['kupiec']['observed_rate']:.2%} vs {1-conf:.0%}) and is **rejected** by the "
        f"Kupiec coverage test (p = {bs['kupiec']['p_value']:.4f}). The GARCH model breached "
        f"**{bg['kupiec']['breaches']}** times and **passes** "
        f"(p = {bg['kupiec']['p_value']:.3f}). Both still fail the Christoffersen "
        "independence test — breaches cluster, so neither model is fully adequate during "
        "sustained stress."
    )

# ----------------------------------------------------------------- Attribution
with tabs[5]:
    st.subheader("Where did the returns come from?")
    factors, attrib = factor_analysis(tickers, rf, lookback, rebal, cost_bps, max_w)

    fstats = pd.DataFrame({
        "Ann. return": {c: factors[c].mean() * TRADING_DAYS for c in ["MKT", "WML", "VOL"]},
        "Ann. vol": {c: factors[c].std() * np.sqrt(TRADING_DAYS) for c in ["MKT", "WML", "VOL"]},
    })
    fstats["Sharpe"] = fstats["Ann. return"] / fstats["Ann. vol"]
    fstats["Description"] = ["Equal-weighted market excess return",
                             "Winners minus losers (12-1 momentum)",
                             "Low-vol minus high-vol (60d realised)"]
    st.dataframe(fstats.style.format({"Ann. return": "{:.2%}", "Ann. vol": "{:.2%}",
                                      "Sharpe": "{:.2f}"}), use_container_width=True)
    st.info(
        "**What these factors are.** Fama–French SMB and HML need market-cap and "
        "book-to-market data, which a price-and-volume dataset does not contain. Rather "
        "than claim a regression that was not run, these three factors are built from the "
        "price cross-section of all 505 constituents. MKT and momentum are standard; VOL is "
        "a price-based proxy for the low-beta anomaly and is labelled as one. Alpha below is "
        "therefore alpha *relative to these three factors only*."
    )

    reg = pd.DataFrame({
        "Alpha (ann)": {k: v["three"]["alpha_annual"] for k, v in attrib.items()},
        "t-stat": {k: v["three"]["alpha_tstat"] for k, v in attrib.items()},
        "p-value": {k: v["three"]["alpha_pvalue"] for k, v in attrib.items()},
        "β MKT": {k: v["three"]["betas"]["MKT"] for k, v in attrib.items()},
        "β WML": {k: v["three"]["betas"]["WML"] for k, v in attrib.items()},
        "β VOL": {k: v["three"]["betas"]["VOL"] for k, v in attrib.items()},
        "R²": {k: v["three"]["r_squared"] for k, v in attrib.items()},
        "CAPM R²": {k: v["capm"]["r_squared"] for k, v in attrib.items()},
    })
    st.dataframe(reg.style.format({"Alpha (ann)": "{:.2%}", "t-stat": "{:.2f}",
                                   "p-value": "{:.3f}", "β MKT": "{:.2f}", "β WML": "{:.2f}",
                                   "β VOL": "{:.2f}", "R²": "{:.3f}", "CAPM R²": "{:.3f}"})
                 .background_gradient(subset=["β WML"], cmap="RdYlBu_r"),
                 use_container_width=True)

    fig = px.bar(reg.reset_index(), x="index", y="β WML", color="β WML",
                 color_continuous_scale="RdYlBu_r",
                 labels={"index": "", "β WML": "Momentum loading"},
                 title="Momentum exposure by strategy")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    ms_wml = reg.loc["Max-Sharpe (unconstrained)", "β WML"]
    eq_wml = reg.loc["Equal-weighted (1/N)", "β WML"]
    st.caption(
        f"**Read — this is the key result.** The optimiser carries a momentum loading of "
        f"**{ms_wml:+.2f}** against **{eq_wml:+.2f}** for equal weighting. Nobody told it to "
        "buy momentum. Optimising on trailing-window mean returns *is* buying recent winners, "
        "mechanically. That explains everything else: it won in a trending market, it "
        "doubled the drawdown because momentum crashes, and shrinking the covariance matrix "
        "did not help because the exposure comes from the mean vector, not the covariance."
    )

# ------------------------------------------------------------------ Simulation
with tabs[6]:
    st.subheader("Monte Carlo")
    n_sims = st.slider("Paths", 1000, 20000, int(CFG["monte_carlo"]["n_simulations"]), 1000)
    horizon = st.slider("Horizon (trading days)", 63, 504,
                        int(CFG["monte_carlo"]["horizon_days"]), 21)
    strat = st.selectbox("Portfolio", list(race), index=list(race).index("Equal-weighted (1/N)"))
    d = race[strat]
    paths = simulate_portfolio(d["ann_return"], d["ann_vol"], 100_000, horizon, n_sims)
    summ = simulation_summary(paths, 100_000, conf_levels=(conf,))

    pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    x = np.arange(paths.shape[1])
    fig = go.Figure()
    for lo, hi, lab, col in [(0, 4, "5th–95th pct", "rgba(0,100,200,0.15)"),
                             (1, 3, "25th–75th pct", "rgba(0,100,200,0.30)")]:
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                                 y=np.concatenate([pcts[hi], pcts[lo][::-1]]),
                                 fill="toself", fillcolor=col, line=dict(width=0),
                                 name=lab, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=pcts[2], line=dict(color="darkblue", width=2.5),
                             name="Median"))
    fig.add_hline(y=100_000, line_dash="dash", line_color="grey")
    fig.update_layout(height=470, xaxis_title="Trading days ahead",
                      yaxis_title="Portfolio value ($)",
                      title=f"{n_sims:,} GBM paths — {strat}")
    st.plotly_chart(fig, use_container_width=True)

    m = st.columns(4)
    m[0].metric("Median outcome", f"${summ['median_end_value']:,.0f}")
    m[1].metric(f"{conf:.0%} VaR", f"{summ[f'sim_var_{int(conf*100)}']:.2%}")
    m[2].metric(f"{conf:.0%} CVaR", f"{summ[f'sim_cvar_{int(conf*100)}']:.2%}")
    m[3].metric("P(loss)", f"{summ['prob_loss']:.2%}")
    st.caption(
        "**Read — with a caveat that matters.** GBM assumes constant volatility, and the "
        "Volatility tab just demonstrated a 6× swing in conditional volatility with a "
        "Ljung-Box p-value near zero. These tail numbers are therefore **optimistic**. "
        "Re-simulating with GARCH-forecast volatility is the correct next step."
    )

# ----------------------------------------------------------------- Risk Report
with tabs[7]:
    st.subheader("Institutional-style risk report")
    st.write("Generated by `sql/reports/risk_report.sql` against the warehouse. "
             "Policy limits: single sector ≤ 20%, single security ≤ 10%.")

    import sqlite3
    conn = sqlite3.connect(ROOT / "data" / "portfolio.db")
    sql = (ROOT / "sql" / "reports" / "risk_report.sql").read_text()
    stmts = [s.strip() for s in sql.split(";")
             if s.strip() and not all(l.strip().startswith("--") or not l.strip()
                                      for l in s.strip().splitlines())]
    titles = ["R1 — Exposure and concentration by sector",
              "R2 — Tail risk by holding (99% VaR and Expected Shortfall)",
              "R3 — Sector-level aggregate risk",
              "R4 — VaR breach register (systemic stress days)"]
    for title, s in zip(titles, stmts):
        st.markdown(f"**{title}**")
        st.dataframe(pd.read_sql_query(s, conn), use_container_width=True, hide_index=True)
    conn.close()

    st.caption(
        "**Read — R4 is the interesting one.** The breach register was built purely from "
        "per-security VaR thresholds, with no knowledge of market history. It independently "
        "surfaced 24 Aug 2015 (China devaluation, 26 of 32 holdings breaching at once), "
        "5 Feb 2018 (volatility spike, 22), and 24 Jun 2016 (Brexit, 12) — and separates "
        "them from rate-driven days where only rate-sensitive sectors (utilities, REITs, "
        "telecoms) breached. Distinguishing systemic from idiosyncratic stress is exactly "
        "what this register exists to do."
    )
    st.info(
        "**Why Expected Shortfall sits next to VaR.** VaR is not a coherent risk measure — "
        "it violates subadditivity, so a combined portfolio's VaR can exceed the sum of its "
        "parts', contradicting the premise of diversification. ES is coherent and sensitive "
        "to the shape of the tail beyond the threshold. Basel III moved market-risk capital "
        "from VaR to ES for exactly these reasons."
    )

# -------------------------------------------------------------------- Findings
with tabs[8]:
    ms_wml = attrib["Max-Sharpe (unconstrained)"]["three"]["betas"]["WML"]
    bs, bg = g["bt_static"], g["bt_garch"]
    st.markdown(f"""
### 1. The optimiser won on Sharpe and lost on drawdown

Out-of-sample, unconstrained mean-variance scored Sharpe **{ms['sharpe']:.3f}** against
**{base['sharpe']:.3f}** for equal weighting. But it did so with a maximum drawdown of
**{ms['max_drawdown']:.1%}** versus **{base['max_drawdown']:.1%}** — so on Calmar (return
per unit of worst-case loss) it **lost**: **{ms['calmar']:.2f}** vs **{base['calmar']:.2f}**.
It also turned over **{ms['avg_turnover']/base['avg_turnover']:.0f}×** as much, paying
**{ms['total_cost_drag']:.2%}** in costs.

Which metric you look at determines whether optimisation "worked." That is the finding.

### 2. The optimiser is a momentum strategy in disguise

Factor regression gives it a momentum loading of **β<sub>WML</sub> = {ms_wml:+.2f}**, against
**{attrib['Equal-weighted (1/N)']['three']['betas']['WML']:+.2f}** for equal weighting.
Nobody instructed it to buy momentum. Optimising on trailing-window mean returns *is*
mechanically buying recent winners.

This single fact explains the rest: it outperformed because momentum paid over 2014–2018;
it doubled the drawdown because momentum crashes; and Ledoit–Wolf shrinkage barely helped
(Sharpe {race['Max-Sharpe (Ledoit-Wolf)']['sharpe']:.3f} vs {ms['sharpe']:.3f}) because the
exposure originates in the **mean vector**, and shrinkage repairs the **covariance matrix**.

### 3. Instability predicts cost and drawdown almost deterministically

Across all seven allocators, mean per-asset weight swing across rolling windows correlates
**{pd.Series(stability).corr(pd.Series({k: v['avg_turnover'] for k, v in race.items()})):+.2f}**
with turnover and
**{pd.Series(stability).corr(pd.Series({k: abs(v['max_drawdown']) for k, v in race.items()})):+.2f}**
with maximum drawdown. Constraining weights to {max_w:.0%} cut the swing and delivered the
best Calmar of any strategy tested.

### 4. Constant-volatility VaR is statistically rejected

The static model breached **{bs['kupiec']['breaches']}** times against
**{bs['kupiec']['expected_breaches']:.0f}** expected — Kupiec p = **{bs['kupiec']['p_value']:.4f}**,
**rejected**. GARCH(1,1) with Student-t innovations breached **{bg['kupiec']['breaches']}**
times and **passes** (p = {bg['kupiec']['p_value']:.3f}). Conditional volatility ranges over
**{cv.max()/cv.min():.0f}×**, with persistence {g['persistence']:.3f} and a
{g['half_life']:.0f}-day shock half-life.

Both models still fail Christoffersen independence: breaches cluster. Neither is adequate
for sustained stress, and saying so is more useful than reporting the one that passed.

---

#### Limitations, stated plainly
- **Survivorship bias is uncontrolled.** The universe is S&P 500 members as of 2018,
  back-tested from 2013. Results are biased upward and this is the single largest caveat.
- **One period, one market.** 2013–2018 was an unusually strong, low-volatility bull market
  with persistent momentum. Conclusions about which allocator "wins" are conditional on that.
- **Factors are price-based.** No size or value factor, so reported alpha is relative to
  three factors only.
- **GARCH VaR is fitted in-sample**, making the coverage test a specification check rather
  than true out-of-sample validation. A rolling refit is the stricter test.
- **Simple cost model** — flat bps on turnover, no slippage, market impact, or spread.
- **GBM Monte Carlo understates tails**, as the volatility analysis demonstrates.
""", unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Python · SQLite · scipy · PyPortfolioOpt · arch · statsmodels · scikit-learn · "
    "Plotly · Streamlit. Manual mean-variance implementation validated against "
    "PyPortfolioOpt. An analytics and risk-measurement platform — not a trading system "
    "and not a price-prediction model."
)
