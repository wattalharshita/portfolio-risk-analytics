-- ============================================================================
-- Institutional-style risk report queries
-- Modelled on the shape of a regulatory / ALM risk pack: exposure by sector,
-- concentration limits with breach flags, and tail risk per holding.
--
-- Assumptions are documented in docs/risk_report_spec.md, as a specification would.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- R1. EXPOSURE AND CONCENTRATION BY SECTOR, with limit breach flags
-- Policy limits (see spec): single sector <= 20% of portfolio,
--                           single security <= 10% of portfolio.
-- Equal-weighted holdings assumed. Swap in optimiser weights for a live report.
-- ----------------------------------------------------------------------------
WITH holdings AS (
    SELECT s.security_id, s.ticker, s.sector,
           1.0 / (SELECT COUNT(*) FROM securities WHERE is_benchmark = 0) AS weight
    FROM securities s
    WHERE s.is_benchmark = 0
),
sector_exposure AS (
    SELECT sector,
           COUNT(*)      AS n_holdings,
           SUM(weight)   AS sector_weight,
           MAX(weight)   AS largest_single_name
    FROM holdings
    GROUP BY sector
)
SELECT
    sector,
    n_holdings,
    ROUND(sector_weight * 100, 2)        AS sector_exposure_pct,
    ROUND(largest_single_name * 100, 2)  AS largest_name_pct,
    CASE WHEN sector_weight > 0.20 THEN 'BREACH' ELSE 'OK' END        AS sector_limit_20pct,
    CASE WHEN largest_single_name > 0.10 THEN 'BREACH' ELSE 'OK' END  AS name_limit_10pct
FROM sector_exposure
ORDER BY sector_weight DESC;


-- ----------------------------------------------------------------------------
-- R2. TAIL RISK BY HOLDING — 99% historical VaR and Expected Shortfall
-- Basel III replaced VaR with Expected Shortfall for market-risk capital because
-- ES is coherent (subadditive) and sensitive to the shape of the tail beyond the
-- threshold, where VaR is blind to it. Both are reported for comparison.
-- ----------------------------------------------------------------------------
WITH ranked AS (
    SELECT s.ticker, s.sector, r.simple_return,
           NTILE(100) OVER (PARTITION BY s.ticker ORDER BY r.simple_return) AS pctile
    FROM returns r
    JOIN securities s ON s.security_id = r.security_id
    WHERE s.is_benchmark = 0
),
var_level AS (
    SELECT ticker, sector, MAX(simple_return) AS var_99_threshold
    FROM ranked
    WHERE pctile = 1
    GROUP BY ticker, sector
),
es AS (
    SELECT r.ticker, AVG(r.simple_return) AS es_99
    FROM ranked r
    WHERE r.pctile = 1
    GROUP BY r.ticker
)
SELECT
    v.ticker,
    v.sector,
    ROUND(-v.var_99_threshold * 100, 3) AS var_99_pct,
    ROUND(-e.es_99 * 100, 3)            AS expected_shortfall_99_pct,
    ROUND((e.es_99 / NULLIF(v.var_99_threshold, 0) - 1) * 100, 1) AS es_exceeds_var_by_pct
FROM var_level v
JOIN es e ON e.ticker = v.ticker
ORDER BY var_99_pct DESC;


-- ----------------------------------------------------------------------------
-- R3. SECTOR-LEVEL AGGREGATE RISK CONTRIBUTION
-- Which sectors carry the portfolio's risk, as opposed to its capital.
-- ----------------------------------------------------------------------------
SELECT
    s.sector,
    COUNT(DISTINCT s.ticker)                                   AS n_holdings,
    ROUND(AVG(r.simple_return) * 252 * 100, 2)                 AS ann_return_pct,
    ROUND(
        SQRT(MAX(AVG(r.simple_return * r.simple_return)
                 - AVG(r.simple_return) * AVG(r.simple_return), 0)) * SQRT(252) * 100, 2
    )                                                          AS ann_volatility_pct,
    ROUND(MIN(r.simple_return) * 100, 2)                       AS worst_day_pct,
    SUM(CASE WHEN r.simple_return <= -0.05 THEN 1 ELSE 0 END)  AS days_below_minus_5pct
FROM returns r
JOIN securities s ON s.security_id = r.security_id
WHERE s.is_benchmark = 0
GROUP BY s.sector
ORDER BY ann_volatility_pct DESC;


-- ----------------------------------------------------------------------------
-- R4. VaR BREACH REGISTER — days where a holding exceeded its own 99% VaR
-- The exception log a risk function reviews. Clustered dates indicate a market
-- stress episode rather than idiosyncratic events.
-- ----------------------------------------------------------------------------
WITH pctiles AS (
    SELECT s.ticker, r.date, r.simple_return,
           NTILE(100) OVER (PARTITION BY s.ticker ORDER BY r.simple_return) AS pctile
    FROM returns r
    JOIN securities s ON s.security_id = r.security_id
    WHERE s.is_benchmark = 0
),
thresholds AS (
    SELECT ticker, MAX(simple_return) AS var_99_threshold
    FROM pctiles
    WHERE pctile = 1
    GROUP BY ticker
)
SELECT
    p.date,
    COUNT(*)                                AS n_holdings_breaching,
    ROUND(MIN(p.simple_return) * 100, 2)    AS worst_holding_return_pct,
    GROUP_CONCAT(p.ticker)                  AS breaching_tickers
FROM pctiles p
JOIN thresholds t ON t.ticker = p.ticker
WHERE p.simple_return < t.var_99_threshold
GROUP BY p.date
HAVING COUNT(*) >= 5
ORDER BY n_holdings_breaching DESC, p.date
LIMIT 15;
