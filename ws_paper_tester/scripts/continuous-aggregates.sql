-- ============================================
-- Continuous Aggregates for Multi-Timeframe Data
-- TimescaleDB Auto-Rollup System
-- ============================================
-- This script creates continuous aggregates that automatically
-- compute higher timeframe candles from 1-minute base data.
--
-- Run this script AFTER initial data load to avoid slow refresh.
-- Usage: psql -U trading -d kraken_data < continuous-aggregates.sql

-- ============================================
-- 5-MINUTE CANDLES (from 1m)
-- ============================================
DROP MATERIALIZED VIEW IF EXISTS candles_5m CASCADE;

CREATE MATERIALIZED VIEW candles_5m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('5 minutes', timestamp) AS timestamp,
    5::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol, time_bucket('5 minutes', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_5m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- ============================================
-- 15-MINUTE CANDLES (from 1m)
-- ============================================
DROP MATERIALIZED VIEW IF EXISTS candles_15m CASCADE;

CREATE MATERIALIZED VIEW candles_15m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('15 minutes', timestamp) AS timestamp,
    15::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol, time_bucket('15 minutes', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_15m',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE
);

-- ============================================
-- 30-MINUTE CANDLES (from 1m)
-- ============================================
DROP MATERIALIZED VIEW IF EXISTS candles_30m CASCADE;

CREATE MATERIALIZED VIEW candles_30m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('30 minutes', timestamp) AS timestamp,
    30::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol, time_bucket('30 minutes', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_30m',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '30 minutes',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE
);

-- ============================================
-- 1-HOUR CANDLES (from 1m)
-- ============================================
DROP MATERIALIZED VIEW IF EXISTS candles_1h CASCADE;

CREATE MATERIALIZED VIEW candles_1h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', timestamp) AS timestamp,
    60::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol, time_bucket('1 hour', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_1h',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ============================================
-- 4-HOUR CANDLES (from 1h)
-- ============================================
DROP MATERIALIZED VIEW IF EXISTS candles_4h CASCADE;

CREATE MATERIALIZED VIEW candles_4h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('4 hours', timestamp) AS timestamp,
    240::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles_1h
GROUP BY symbol, time_bucket('4 hours', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_4h',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '4 hours',
    schedule_interval => INTERVAL '4 hours',
    if_not_exists => TRUE
);

-- ============================================
-- 12-HOUR CANDLES (from 1h)
-- ============================================
DROP MATERIALIZED VIEW IF EXISTS candles_12h CASCADE;

CREATE MATERIALIZED VIEW candles_12h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('12 hours', timestamp) AS timestamp,
    720::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles_1h
GROUP BY symbol, time_bucket('12 hours', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_12h',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '12 hours',
    schedule_interval => INTERVAL '12 hours',
    if_not_exists => TRUE
);

-- ============================================
-- DAILY CANDLES (from 1h)
-- ============================================
DROP MATERIALIZED VIEW IF EXISTS candles_1d CASCADE;

CREATE MATERIALIZED VIEW candles_1d
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 day', timestamp) AS timestamp,
    1440::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles_1h
GROUP BY symbol, time_bucket('1 day', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_1d',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ============================================
-- WEEKLY CANDLES (from 1d)
-- ============================================
DROP MATERIALIZED VIEW IF EXISTS candles_1w CASCADE;

CREATE MATERIALIZED VIEW candles_1w
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 week', timestamp) AS timestamp,
    10080::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles_1d
GROUP BY symbol, time_bucket('1 week', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_1w',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '1 week',
    schedule_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- ============================================
-- REFRESH ALL CONTINUOUS AGGREGATES
-- ============================================
-- Call this after initial data load to populate aggregates
-- Note: This can take a while with large datasets

-- Manually refresh all aggregates for initial load
-- Uncomment and run these after data import:

-- CALL refresh_continuous_aggregate('candles_5m', NULL, NULL);
-- CALL refresh_continuous_aggregate('candles_15m', NULL, NULL);
-- CALL refresh_continuous_aggregate('candles_30m', NULL, NULL);
-- CALL refresh_continuous_aggregate('candles_1h', NULL, NULL);
-- CALL refresh_continuous_aggregate('candles_4h', NULL, NULL);
-- CALL refresh_continuous_aggregate('candles_12h', NULL, NULL);
-- CALL refresh_continuous_aggregate('candles_1d', NULL, NULL);
-- CALL refresh_continuous_aggregate('candles_1w', NULL, NULL);

-- ============================================
-- RETENTION POLICIES (Optional)
-- ============================================
-- Uncomment to enable automatic data retention

-- Keep raw trades for 90 days (can rebuild candles from them)
-- SELECT add_retention_policy('trades', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep 1-minute candles for 1 year
-- SELECT add_retention_policy('candles', INTERVAL '365 days', if_not_exists => TRUE);

-- ============================================
-- VERIFICATION QUERIES
-- ============================================
-- Check continuous aggregate status

SELECT
    view_name,
    format('%I.%I', materialization_hypertable_schema, materialization_hypertable_name) AS materialization_hypertable
FROM timescaledb_information.continuous_aggregates
ORDER BY view_name;

-- Check compression status
SELECT
    hypertable_name,
    compression_enabled
FROM timescaledb_information.hypertables
ORDER BY hypertable_name;
