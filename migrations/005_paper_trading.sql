-- ============================================================================
-- Phase 6: Paper Trading Database Migration
-- ============================================================================
-- Creates tables for paper trading data isolation:
-- - Paper trading sessions
-- - Paper orders (separate from live orders)
-- - Paper positions (separate from live positions)
-- - Paper trade history
-- - Paper portfolio snapshots
--
-- Run: psql -h localhost -U triplegain -d triplegain -f migrations/005_paper_trading.sql
-- ============================================================================

-- ============================================================================
-- Paper Trading Sessions
-- ============================================================================
-- Tracks paper trading sessions for portfolio management.

CREATE TABLE IF NOT EXISTS paper_sessions (
    id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    initial_balances JSONB NOT NULL,
    current_balances JSONB NOT NULL,
    realized_pnl DECIMAL(20, 10) DEFAULT 0,
    total_fees_paid DECIMAL(20, 10) DEFAULT 0,
    trade_count INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    notes TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'ended'))
);

CREATE INDEX IF NOT EXISTS idx_paper_sessions_status
    ON paper_sessions(status);
CREATE INDEX IF NOT EXISTS idx_paper_sessions_created_at
    ON paper_sessions(created_at DESC);

-- ============================================================================
-- Paper Orders
-- ============================================================================
-- Stores paper trading orders separately from live orders.

CREATE TABLE IF NOT EXISTS paper_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(100) REFERENCES paper_sessions(id),
    external_id VARCHAR(50),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(30) NOT NULL
        CHECK (order_type IN ('market', 'limit', 'stop-loss', 'take-profit', 'stop-loss-limit', 'take-profit-limit')),
    size DECIMAL(20, 10) NOT NULL,
    price DECIMAL(20, 10),
    stop_price DECIMAL(20, 10),
    filled_size DECIMAL(20, 10) DEFAULT 0,
    filled_price DECIMAL(20, 10),
    fee_amount DECIMAL(20, 10) DEFAULT 0,
    fee_currency VARCHAR(10),
    leverage INT DEFAULT 1,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'open', 'partially_filled', 'filled', 'cancelled', 'expired', 'error')),
    parent_order_id UUID,
    stop_loss_order_id UUID,
    take_profit_order_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_paper_orders_session_id
    ON paper_orders(session_id);
CREATE INDEX IF NOT EXISTS idx_paper_orders_symbol
    ON paper_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_paper_orders_status
    ON paper_orders(status);
CREATE INDEX IF NOT EXISTS idx_paper_orders_created_at
    ON paper_orders(created_at DESC);

-- ============================================================================
-- Paper Positions
-- ============================================================================
-- Stores paper trading positions separately from live positions.

CREATE TABLE IF NOT EXISTS paper_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(100) REFERENCES paper_sessions(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    size DECIMAL(20, 10) NOT NULL,
    entry_price DECIMAL(20, 10) NOT NULL,
    leverage INT NOT NULL DEFAULT 1,
    status VARCHAR(20) NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'closing', 'closed', 'liquidated')),
    order_id UUID,
    stop_loss DECIMAL(20, 10),
    take_profit DECIMAL(20, 10),
    stop_loss_order_id UUID,
    take_profit_order_id UUID,
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    exit_price DECIMAL(20, 10),
    realized_pnl DECIMAL(20, 10) DEFAULT 0,
    total_fees DECIMAL(20, 10) DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_paper_positions_session_id
    ON paper_positions(session_id);
CREATE INDEX IF NOT EXISTS idx_paper_positions_symbol
    ON paper_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_paper_positions_status
    ON paper_positions(status);
CREATE INDEX IF NOT EXISTS idx_paper_positions_opened_at
    ON paper_positions(opened_at DESC);

-- ============================================================================
-- Paper Trade History
-- ============================================================================
-- Detailed record of each paper trade execution.

CREATE TABLE IF NOT EXISTS paper_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(100) REFERENCES paper_sessions(id),
    order_id UUID REFERENCES paper_orders(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    size DECIMAL(20, 10) NOT NULL,
    price DECIMAL(20, 10) NOT NULL,
    value DECIMAL(20, 10) NOT NULL,
    fee DECIMAL(20, 10) NOT NULL,
    fee_currency VARCHAR(10) NOT NULL,
    balance_after JSONB,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_session_id
    ON paper_trades(session_id);
CREATE INDEX IF NOT EXISTS idx_paper_trades_timestamp
    ON paper_trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol
    ON paper_trades(symbol);

-- ============================================================================
-- Paper Position Snapshots (TimescaleDB Hypertable)
-- ============================================================================
-- Time-series data for paper position P&L tracking.

CREATE TABLE IF NOT EXISTS paper_position_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    session_id VARCHAR(100) NOT NULL,
    position_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    current_price DECIMAL(20, 10),
    unrealized_pnl DECIMAL(20, 10),
    unrealized_pnl_pct DECIMAL(10, 4),

    PRIMARY KEY (timestamp, session_id, position_id)
);

-- Convert to hypertable if TimescaleDB is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('paper_position_snapshots', 'timestamp',
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_paper_position_snapshots_session
    ON paper_position_snapshots(session_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_paper_position_snapshots_position
    ON paper_position_snapshots(position_id, timestamp DESC);

-- ============================================================================
-- Paper Portfolio Snapshots (TimescaleDB Hypertable)
-- ============================================================================
-- Time-series data for paper portfolio equity tracking.

CREATE TABLE IF NOT EXISTS paper_portfolio_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    session_id VARCHAR(100) NOT NULL,
    total_equity_usd DECIMAL(20, 10),
    balances JSONB,
    unrealized_pnl DECIMAL(20, 10),
    realized_pnl DECIMAL(20, 10),
    trade_count INT,

    PRIMARY KEY (timestamp, session_id)
);

-- Convert to hypertable if TimescaleDB is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('paper_portfolio_snapshots', 'timestamp',
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_paper_portfolio_snapshots_session
    ON paper_portfolio_snapshots(session_id, timestamp DESC);

-- ============================================================================
-- Data Retention Policies (TimescaleDB)
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Keep paper position snapshots for 30 days (shorter than live)
        PERFORM add_retention_policy('paper_position_snapshots', INTERVAL '30 days',
            if_not_exists => TRUE);

        -- Keep paper portfolio snapshots for 30 days
        PERFORM add_retention_policy('paper_portfolio_snapshots', INTERVAL '30 days',
            if_not_exists => TRUE);

        -- Add compression after 3 days
        ALTER TABLE paper_position_snapshots SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'session_id, position_id'
        );
        PERFORM add_compression_policy('paper_position_snapshots', INTERVAL '3 days',
            if_not_exists => TRUE);

        ALTER TABLE paper_portfolio_snapshots SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'session_id'
        );
        PERFORM add_compression_policy('paper_portfolio_snapshots', INTERVAL '3 days',
            if_not_exists => TRUE);
    END IF;
END $$;

-- ============================================================================
-- Triggers for updated_at timestamps
-- ============================================================================

CREATE TRIGGER update_paper_orders_updated_at
    BEFORE UPDATE ON paper_orders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_paper_positions_updated_at
    BEFORE UPDATE ON paper_positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Cleanup Functions for Paper Trading
-- ============================================================================

-- Function to clean old ended paper sessions
CREATE OR REPLACE FUNCTION cleanup_old_paper_sessions(retention_days INT DEFAULT 30)
RETURNS INT AS $$
DECLARE
    deleted_count INT;
    session_rec RECORD;
BEGIN
    deleted_count := 0;

    -- Find and delete old ended sessions and their related data
    FOR session_rec IN
        SELECT id FROM paper_sessions
        WHERE status = 'ended'
        AND ended_at < NOW() - (retention_days || ' days')::INTERVAL
    LOOP
        -- Delete related trades
        DELETE FROM paper_trades WHERE session_id = session_rec.id;
        -- Delete related positions
        DELETE FROM paper_positions WHERE session_id = session_rec.id;
        -- Delete related orders
        DELETE FROM paper_orders WHERE session_id = session_rec.id;
        -- Delete the session
        DELETE FROM paper_sessions WHERE id = session_rec.id;
        deleted_count := deleted_count + 1;
    END LOOP;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Helper Views
-- ============================================================================

-- View for paper session performance summary
CREATE OR REPLACE VIEW paper_session_performance AS
SELECT
    s.id AS session_id,
    s.created_at,
    s.ended_at,
    s.status,
    s.trade_count,
    s.winning_trades,
    s.losing_trades,
    CASE
        WHEN s.trade_count > 0
        THEN ROUND(s.winning_trades::DECIMAL / s.trade_count * 100, 2)
        ELSE 0
    END AS win_rate_pct,
    s.realized_pnl,
    s.total_fees_paid,
    s.realized_pnl - s.total_fees_paid AS net_pnl,
    s.initial_balances,
    s.current_balances
FROM paper_sessions s;

-- ============================================================================
-- Migration complete
-- ============================================================================
-- Verify tables:
-- \dt paper_sessions paper_orders paper_positions paper_trades paper_position_snapshots paper_portfolio_snapshots
