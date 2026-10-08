-- ============================================================================
-- Analytical query library — portfolio.db
-- These are the reporting queries, not SELECT *. Each is the kind of thing a
-- risk/analytics team actually asks the warehouse for.
-- Run: sqlite3 data/portfolio.db < sql/reports/analytical_queries.sql
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Q1. Rolling 30-day annualised volatility per security (WINDOW FUNCTION)
-- Uses a moving standard deviation over a 30-row frame, annualised by sqrt(252).
-- SQLite has no STDDEV, so it is computed from the moving mean of squares:
--   var = E[r^2] - (E[r])^2
-- ----------------------------------------------------------------------------
WITH rolling AS (
    SELECT
        s.ticker,
        r.date,
        r.simple_return,
        AVG(r.simple_return) OVER w                       AS mean_r,
        AVG(r.simple_return * r.simple_return) OVER w     AS mean_r2,
        COUNT(*) OVER w                                   AS n_obs
    FROM returns r
    JOIN securities s ON s.security_id = r.security_id
    WINDOW w AS (
        PARTITION BY s.ticker
        ORDER BY r.date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    )
)
SELECT
    ticker,
    date,
    ROUND(SQRT(MAX(mean_r2 - mean_r * mean_r, 0)) * SQRT(252) * 100, 2) AS rolling_30d_vol_pct
FROM rolling
WHERE n_obs = 30
ORDER BY ticker, date
LIMIT 20;


-- ----------------------------------------------------------------------------
-- Q2. Sector-level aggregate performance (GROUP BY + HAVING)
-- Only sectors with enough history to be meaningful are reported.
-- ----------------------------------------------------------------------------
SELECT
    s.sector,
    COUNT(DISTINCT s.ticker)                          AS n_securities,
    COUNT(r.date)                                     AS n_observations,
    ROUND(AVG(r.simple_return) * 252 * 100, 2)        AS avg_annualised_return_pct,
    ROUND(MIN(r.simple_return) * 100, 2)              AS worst_day_pct,
    ROUND(MAX(r.simple_return) * 100, 2)              AS best_day_pct
FROM returns r
JOIN securities s ON s.security_id = r.security_id
GROUP BY s.sector
HAVING COUNT(r.date) > 500
ORDER BY avg_annualised_return_pct DESC;


-- ----------------------------------------------------------------------------
-- Q3. Best and worst performer per quarter (RANK() OVER PARTITION BY)
-- Quarterly return is built from the first and last adjusted close in the quarter,
-- then securities are ranked within each quarter.
-- ----------------------------------------------------------------------------
WITH quarterly AS (
    SELECT
        s.ticker,
        STRFTIME('%Y', p.date) || '-Q' ||
            CAST((CAST(STRFTIME('%m', p.date) AS INTEGER) + 2) / 3 AS TEXT) AS quarter,
        FIRST_VALUE(p.adj_close) OVER (
            PARTITION BY s.ticker, STRFTIME('%Y', p.date),
                         (CAST(STRFTIME('%m', p.date) AS INTEGER) + 2) / 3
            ORDER BY p.date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS q_open,
        LAST_VALUE(p.adj_close) OVER (
            PARTITION BY s.ticker, STRFTIME('%Y', p.date),
                         (CAST(STRFTIME('%m', p.date) AS INTEGER) + 2) / 3
            ORDER BY p.date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS q_close
    FROM prices p
    JOIN securities s ON s.security_id = p.security_id
),
q_returns AS (
    SELECT DISTINCT
        ticker,
        quarter,
        ROUND((q_close / q_open - 1) * 100, 2) AS quarter_return_pct
    FROM quarterly
),
ranked AS (
    SELECT
        quarter,
        ticker,
        quarter_return_pct,
        RANK() OVER (PARTITION BY quarter ORDER BY quarter_return_pct DESC) AS rank_best,
        RANK() OVER (PARTITION BY quarter ORDER BY quarter_return_pct ASC)  AS rank_worst
    FROM q_returns
)
SELECT
    quarter,
    MAX(CASE WHEN rank_best  = 1 THEN ticker END)             AS best_performer,
    MAX(CASE WHEN rank_best  = 1 THEN quarter_return_pct END) AS best_return_pct,
    MAX(CASE WHEN rank_worst = 1 THEN ticker END)             AS worst_performer,
    MAX(CASE WHEN rank_worst = 1 THEN quarter_return_pct END) AS worst_return_pct
FROM ranked
WHERE rank_best = 1 OR rank_worst = 1
GROUP BY quarter
ORDER BY quarter;


-- ----------------------------------------------------------------------------
-- Q4. Co-movement pairs (SELF-JOIN)
-- Pairs each security's daily return with every other security's return on the
-- same date, and measures how often they move in the same direction. This is the
-- input a correlation calculation is built on, expressed relationally.
-- ----------------------------------------------------------------------------
SELECT
    sa.ticker AS ticker_a,
    sb.ticker AS ticker_b,
    COUNT(*)  AS shared_days,
    ROUND(
        SUM(CASE WHEN (ra.simple_return > 0 AND rb.simple_return > 0)
                   OR (ra.simple_return < 0 AND rb.simple_return < 0)
                 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    ) AS same_direction_pct,
    ROUND(AVG(ra.simple_return * rb.simple_return) * 252 * 10000, 2) AS ann_cross_product_bps
FROM returns ra
JOIN returns rb ON ra.date = rb.date AND ra.security_id < rb.security_id
JOIN securities sa ON sa.security_id = ra.security_id
JOIN securities sb ON sb.security_id = rb.security_id
GROUP BY sa.ticker, sb.ticker
ORDER BY same_direction_pct DESC;


-- ----------------------------------------------------------------------------
-- Q5. Drawdown from running peak per security (WINDOW FUNCTION, running MAX)
-- Surfaces the worst peak-to-trough decline each security experienced.
-- ----------------------------------------------------------------------------
WITH with_peak AS (
    SELECT
        s.ticker,
        p.date,
        p.adj_close,
        MAX(p.adj_close) OVER (
            PARTITION BY s.ticker ORDER BY p.date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_peak
    FROM prices p
    JOIN securities s ON s.security_id = p.security_id
)
SELECT
    ticker,
    ROUND(MIN(adj_close / running_peak - 1) * 100, 2) AS max_drawdown_pct,
    COUNT(*)                                          AS trading_days
FROM with_peak
GROUP BY ticker
ORDER BY max_drawdown_pct ASC;


-- ----------------------------------------------------------------------------
-- Q6. Tail-risk day count — how often did each security breach a -3% day?
-- The kind of exception-count a risk function reports for VaR backtesting.
-- ----------------------------------------------------------------------------
SELECT
    s.ticker,
    COUNT(*)                                               AS total_days,
    SUM(CASE WHEN r.simple_return <= -0.03 THEN 1 ELSE 0 END) AS days_below_minus_3pct,
    ROUND(SUM(CASE WHEN r.simple_return <= -0.03 THEN 1 ELSE 0 END) * 100.0
          / COUNT(*), 2)                                   AS breach_rate_pct
FROM returns r
JOIN securities s ON s.security_id = r.security_id
GROUP BY s.ticker
ORDER BY breach_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- Q7. Data-quality / ingestion audit trail
-- ----------------------------------------------------------------------------
SELECT ticker, rows_ingested, rows_rejected, reason, run_at
FROM ingestion_log
ORDER BY run_at DESC, ticker;
