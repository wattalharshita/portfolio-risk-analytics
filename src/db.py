"""
db.py — SQLite connection and query helpers.

Kept deliberately thin: raw sqlite3 + parameterised SQL rather than an ORM,
so the SQL itself (joins, window functions, aggregates) stays visible and
is the thing being evaluated, not an abstraction over it.
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "portfolio.db"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


@contextmanager
def get_conn(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text())


def upsert_security(conn, ticker: str, name: str, sector: str, is_benchmark: bool = False):
    conn.execute(
        """
        INSERT INTO securities (ticker, name, sector, exchange, is_benchmark)
        VALUES (?, ?, ?, 'NASDAQ/NYSE', ?)
        ON CONFLICT(ticker) DO UPDATE SET
            name=excluded.name, sector=excluded.sector, is_benchmark=excluded.is_benchmark
        """,
        (ticker, name, sector, int(is_benchmark)),
    )
    row = conn.execute("SELECT security_id FROM securities WHERE ticker = ?", (ticker,)).fetchone()
    return row[0]


def upsert_prices(conn, security_id: int, rows):
    """rows: iterable of (date, open, high, low, close, adj_close, volume). Idempotent upsert."""
    conn.executemany(
        """
        INSERT INTO prices (security_id, date, open, high, low, close, adj_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(security_id, date) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, adj_close=excluded.adj_close, volume=excluded.volume
        """,
        [(security_id, *r) for r in rows],
    )


def upsert_returns(conn, security_id: int, rows):
    """rows: iterable of (date, simple_return, log_return)."""
    conn.executemany(
        """
        INSERT INTO returns (security_id, date, simple_return, log_return)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(security_id, date) DO UPDATE SET
            simple_return=excluded.simple_return, log_return=excluded.log_return
        """,
        [(security_id, *r) for r in rows],
    )


def log_ingestion(conn, ticker: str, rows_ingested: int, rows_rejected: int, reason: str):
    conn.execute(
        "INSERT INTO ingestion_log (ticker, rows_ingested, rows_rejected, reason, run_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (ticker, rows_ingested, rows_rejected, reason),
    )


def prices_df(conn, ticker: str = None):
    import pandas as pd
    q = """
        SELECT s.ticker, p.date, p.open, p.high, p.low, p.close, p.adj_close, p.volume
        FROM prices p JOIN securities s ON s.security_id = p.security_id
    """
    params = ()
    if ticker:
        q += " WHERE s.ticker = ?"
        params = (ticker,)
    q += " ORDER BY p.date"
    return pd.read_sql_query(q, conn, params=params, parse_dates=["date"])
