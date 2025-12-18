-- ============================================
-- Historical Data System - Database Schema
-- TimescaleDB (PostgreSQL 15+)
-- ============================================
-- This script runs automatically on first container startup
-- via Docker entrypoint initdb.d

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================
-- TRADES TABLE (Highest Granularity)
-- ============================================
-- Stores individual trade ticks from Kraken
-- Used for building candles and detailed analysis

CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(20, 10) NOT NULL,
    volume DECIMAL(20, 10) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(10),
    misc VARCHAR(50),
    PRIMARY KEY (timestamp, symbol, id)
);

-- Convert to hypertable with daily chunks
SELECT create_hypertable('trades', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Enable compression (after 7 days)
ALTER TABLE trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);
SELECT add_compression_policy('trades', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================
-- CANDLES TABLE (Base 1-minute data)
-- ============================================
-- Stores OHLCV candle data at 1-minute intervals
-- Higher timeframes computed via continuous aggregates

CREATE TABLE IF NOT EXISTS candles (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    interval_minutes SMALLINT NOT NULL,
    open DECIMAL(20, 10) NOT NULL,
    high DECIMAL(20, 10) NOT NULL,
    low DECIMAL(20, 10) NOT NULL,
    close DECIMAL(20, 10) NOT NULL,
    volume DECIMAL(20, 10) NOT NULL,
    quote_volume DECIMAL(20, 10),
    trade_count INTEGER,
    vwap DECIMAL(20, 10),
    PRIMARY KEY (timestamp, symbol, interval_minutes)
);

-- Convert to hypertable with weekly chunks
SELECT create_hypertable('candles', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Enable compression (after 30 days)
ALTER TABLE candles SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, interval_minutes',
    timescaledb.compress_orderby = 'timestamp DESC'
);
SELECT add_compression_policy('candles', INTERVAL '30 days', if_not_exists => TRUE);

-- ============================================
-- DATA SYNC STATUS TABLE (For gap detection)
-- ============================================
-- Tracks the sync state for each symbol/data_type combination
-- Used by the gap filler to detect and fill missing data

CREATE TABLE IF NOT EXISTS data_sync_status (
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(20) NOT NULL,  -- 'trades', 'candles_1m', etc.
    oldest_timestamp TIMESTAMPTZ,
    newest_timestamp TIMESTAMPTZ,
    last_sync_at TIMESTAMPTZ DEFAULT NOW(),
    last_kraken_since BIGINT,  -- Kraken 'since' parameter for continuation
    total_records BIGINT DEFAULT 0,
    PRIMARY KEY (symbol, data_type)
);

-- ============================================
-- INDEXES
-- ============================================
-- Optimized indexes for common query patterns

CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_ts
    ON candles (symbol, interval_minutes, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts
    ON trades (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_external_indicator_name_ts
    ON external_indicators (indicator_name, timestamp DESC);

-- Composite index for range queries on 1-minute candles
CREATE INDEX IF NOT EXISTS idx_candles_range
    ON candles (symbol, interval_minutes, timestamp)
    WHERE interval_minutes = 1;

-- ============================================
-- GRANTS
-- ============================================
-- Grant privileges to the trading user

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading;

-- ============================================
-- COMMENTS
-- ============================================
-- Document the tables

COMMENT ON TABLE trades IS 'Individual trade ticks from Kraken exchange';
COMMENT ON TABLE candles IS 'OHLCV candle data at various intervals';
COMMENT ON TABLE data_sync_status IS 'Sync state for gap detection and resumption';

-- Note: Continuous aggregates should be created after initial data load
-- Run the continuous-aggregates.sql script separately after schema creation
