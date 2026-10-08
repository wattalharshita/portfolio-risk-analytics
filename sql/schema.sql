-- Portfolio Risk & Returns Analytics Platform — schema
-- Raw DDL, written by hand (not ORM-generated), so joins/indexes are deliberate choices.

CREATE TABLE IF NOT EXISTS securities (
    security_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT NOT NULL UNIQUE,
    name          TEXT,
    sector        TEXT,
    exchange      TEXT,
    currency      TEXT DEFAULT 'USD',
    is_benchmark  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS prices (
    security_id   INTEGER NOT NULL REFERENCES securities(security_id),
    date          TEXT NOT NULL,           -- ISO date, YYYY-MM-DD
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    adj_close     REAL,
    volume        INTEGER,
    PRIMARY KEY (security_id, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_security_date ON prices(security_id, date);

CREATE TABLE IF NOT EXISTS returns (
    security_id     INTEGER NOT NULL REFERENCES securities(security_id),
    date            TEXT NOT NULL,
    simple_return   REAL,
    log_return      REAL,
    PRIMARY KEY (security_id, date)
);
CREATE INDEX IF NOT EXISTS idx_returns_security_date ON returns(security_id, date);

CREATE TABLE IF NOT EXISTS ingestion_log (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT,
    rows_ingested INTEGER,
    rows_rejected INTEGER,
    reason        TEXT,
    run_at        TEXT
);

-- factors table (Fama-French) — schema present for the M6a extension; unused in the core run.
CREATE TABLE IF NOT EXISTS factors (
    date    TEXT PRIMARY KEY,
    mkt_rf  REAL,
    smb     REAL,
    hml     REAL,
    rf      REAL
);
