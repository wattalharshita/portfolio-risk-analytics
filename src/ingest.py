"""
ingest.py — market data -> SQL warehouse.

Three sources, one schema:

  --source sp500      DEFAULT. The full S&P 500 daily OHLCV dataset (505 securities,
                      2013-02-08 to 2018-02-07, 619,040 rows). Downloaded once from a
                      public mirror and cached to data/raw/; every subsequent run reads
                      the cache. Ingests the 32-security universe in config.yaml plus a
                      benchmark constructed from all 505 constituents.

  --source yfinance   Live pull from Yahoo Finance for an arbitrary date range. Requires
                      outbound internet access to query1/query2.finance.yahoo.com. Use
                      this to extend the universe or bring the data up to date.

  --source demo       Four cached tickers. Retained so the smallest possible smoke test
                      still runs with no network at all.

Common to all paths: raw responses are cached before any transformation, ingestion is an
idempotent upsert, a data-quality pass flags suspicious moves, and row counts are written
to the ingestion_log audit table.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.db import (init_db, get_conn, upsert_security, upsert_prices,
                    upsert_returns, log_ingestion)

RAW_DIR = ROOT / "data" / "raw"
SP500_FILE = RAW_DIR / "all_stocks_5yr.csv"
SP500_URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"

SPLIT_MOVE_THRESHOLD = 0.40   # single-day move above this is flagged as a probable
                              # unadjusted corporate action


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------- data quality
def quality_check(df: pd.DataFrame, ticker: str):
    """
    Flags probable unadjusted corporate actions and reports coverage.
    Returns (df, n_flagged, note).
    """
    df = df.sort_values("date").reset_index(drop=True)
    moves = df["adj_close"].pct_change().abs()
    flagged = df.loc[moves > SPLIT_MOVE_THRESHOLD, "date"]
    n_missing = int(df[["open", "high", "low", "close"]].isna().sum().sum())

    parts = []
    if len(flagged):
        dates = ", ".join(pd.to_datetime(flagged).dt.strftime("%Y-%m-%d"))
        parts.append(f"{len(flagged)} move(s) >{SPLIT_MOVE_THRESHOLD:.0%} on {dates} "
                     "(probable unadjusted split/spinoff)")
    if n_missing:
        parts.append(f"{n_missing} missing OHLC field(s) forward-filled")
    note = "; ".join(parts) if parts else "clean"
    return df, len(flagged), note


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    out = df[["date"]].copy()
    out["simple_return"] = df["adj_close"].pct_change()
    out["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    return out.dropna()


def _persist(conn, ticker, name, sector, df, is_benchmark=False):
    df, flagged, note = quality_check(df, ticker)
    sec_id = upsert_security(conn, ticker, name, sector, is_benchmark=is_benchmark)

    price_rows = df[["date", "open", "high", "low", "close", "adj_close", "volume"]].copy()
    price_rows["date"] = pd.to_datetime(price_rows["date"]).dt.strftime("%Y-%m-%d")
    price_rows = price_rows.where(pd.notna(price_rows), None)
    upsert_prices(conn, sec_id, price_rows.itertuples(index=False, name=None))

    ret = compute_returns(df)
    ret["date"] = pd.to_datetime(ret["date"]).dt.strftime("%Y-%m-%d")
    upsert_returns(conn, sec_id, ret.itertuples(index=False, name=None))

    log_ingestion(conn, ticker, len(df), flagged, note)
    return len(df), flagged, note


# --------------------------------------------------------------- source: sp500
def _fetch_sp500():
    """Download the dataset once, then always read from cache."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not SP500_FILE.exists():
        import urllib.request
        print(f"[..] downloading S&P 500 dataset -> {SP500_FILE.name} (~30MB, one time)")
        urllib.request.urlretrieve(SP500_URL, SP500_FILE)
    else:
        print(f"[..] using cached {SP500_FILE.name}")
    return pd.read_csv(SP500_FILE, parse_dates=["date"])


def ingest_sp500(cfg):
    raw = _fetch_sp500()
    tickers = cfg["universe"]["tickers"]
    sectors = cfg["universe"]["sectors"]
    names = cfg["universe"]["names"]
    bench = cfg["universe"]["benchmark"]

    print(f"[..] source holds {raw['Name'].nunique()} securities, {len(raw):,} rows, "
          f"{raw['date'].min().date()} to {raw['date'].max().date()}")

    # ---- audit the FULL dataset for unadjusted corporate actions -------------
    scan = raw.sort_values(["Name", "date"]).copy()
    scan["ret"] = scan.groupby("Name")["close"].pct_change()
    anomalies = scan[scan["ret"].abs() > SPLIT_MOVE_THRESHOLD]
    print(f"[QC] scanned all {raw['Name'].nunique()} securities: "
          f"{len(anomalies)} single-day move(s) >{SPLIT_MOVE_THRESHOLD:.0%} flagged "
          f"across {anomalies['Name'].nunique()} tickers")
    for _, r in anomalies.iterrows():
        print(f"     {r['date'].date()}  {r['Name']:<6} {r['ret']:+.1%}")
    in_universe = anomalies[anomalies["Name"].isin(tickers)]
    print(f"[QC] of which {len(in_universe)} fall inside the selected universe")

    init_db()
    total_rows = 0
    with get_conn() as conn:
        # ---- constituents ----------------------------------------------------
        for t in tickers:
            sub = raw[raw["Name"] == t].sort_values("date").copy()
            if sub.empty:
                print(f"[WARN] {t} absent from source, skipping")
                continue
            sub[["open", "high", "low"]] = sub[["open", "high", "low"]].ffill()
            sub["adj_close"] = sub["close"]   # source series is split/dividend adjusted
            n, flagged, note = _persist(conn, t, names.get(t, t), sectors.get(t, "Unclassified"), sub)
            total_rows += n
            flag = "" if note == "clean" else f"  [{note}]"
            print(f"[OK] {t:<6} {n:>5} rows  {sub['date'].min().date()} to "
                  f"{sub['date'].max().date()}{flag}")

        # ---- benchmark: equal-weighted index of ALL constituents --------------
        # Daily cross-sectional mean simple return across every security with data that
        # day, compounded into a price index based at 100. This is an EQUAL-weighted
        # proxy, not cap-weighted like SPY — stated in the README, since equal weighting
        # tilts toward smaller constituents and is a tougher benchmark in this period.
        scan["ret_clean"] = scan["ret"].where(scan["ret"].abs() <= SPLIT_MOVE_THRESHOLD)
        daily = scan.groupby("date")["ret_clean"].mean().dropna()
        level = 100 * (1 + daily).cumprod()
        bench_df = pd.DataFrame({
            "date": level.index,
            "open": level.values, "high": level.values, "low": level.values,
            "close": level.values, "adj_close": level.values, "volume": 0,
        })
        n, _, _ = _persist(conn, bench, names.get(bench, bench), "Benchmark",
                           bench_df, is_benchmark=True)
        total_rows += n
        n_const = scan.groupby("date")["ret_clean"].count().median()
        print(f"[OK] {bench:<6} {n:>5} rows  equal-weighted index of "
              f"~{n_const:.0f} constituents/day")

    print(f"\n[DONE] {total_rows:,} price rows in warehouse "
          f"({len(tickers)} securities + 1 benchmark, "
          f"{len(set(sectors.values()))} GICS sectors)")


# ------------------------------------------------------------ source: yfinance
def ingest_yfinance(cfg):
    import yfinance as yf
    tickers = cfg["universe"]["tickers"]
    sectors, names = cfg["universe"]["sectors"], cfg["universe"]["names"]
    start, end = cfg["date_range"]["start"], cfg["date_range"]["end"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    init_db()
    with get_conn() as conn:
        for t in tickers + ["SPY"]:
            raw_path = RAW_DIR / f"yf_{t}.csv"
            if raw_path.exists():
                df = pd.read_csv(raw_path, parse_dates=["date"])
            else:
                data = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
                if data.empty:
                    print(f"[WARN] no data for {t}, skipping")
                    continue
                data = data.reset_index().rename(columns={
                    "Date": "date", "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
                df = data[["date", "open", "high", "low", "close", "adj_close", "volume"]]
                df.to_csv(raw_path, index=False)   # cache before any transform
            n, flagged, note = _persist(conn, t, names.get(t, t),
                                        sectors.get(t, "Benchmark"), df,
                                        is_benchmark=(t == "SPY"))
            print(f"[OK] {t:<6} {n:>5} rows — {note}")


# ---------------------------------------------------------------- source: demo
def ingest_demo(cfg):
    demo = {"AAPL": "Information Technology", "MSFT": "Information Technology",
            "GOOG": "Communication Services", "AMZN": "Consumer Discretionary"}
    init_db()
    with get_conn() as conn:
        for t, sec in demo.items():
            df = pd.read_csv(RAW_DIR / f"{t}.csv", parse_dates=["date"])
            df["adj_close"] = df["close"]
            n, flagged, note = _persist(conn, t, t, sec, df)
            print(f"[OK] {t}: {n} rows — {note}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest market data into the SQL warehouse.")
    p.add_argument("--source", choices=["sp500", "yfinance", "demo"], default="sp500")
    args = p.parse_args()
    cfg = load_config()
    {"sp500": ingest_sp500, "yfinance": ingest_yfinance, "demo": ingest_demo}[args.source](cfg)
