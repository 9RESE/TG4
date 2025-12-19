-- ============================================================================
-- Phase 3: Orchestration Database Migration
-- ============================================================================
-- Creates tables for:
-- - Order status tracking
-- - Position tracking and snapshots
-- - Hodl bags management
-- - Coordinator state persistence
--
-- Run: psql -h localhost -U triplegain -d triplegain -f migrations/003_phase3_orchestration.sql

-- ============================================================================
-- Order Status Log
-- ============================================================================
-- Tracks order lifecycle events for audit and debugging.

CREATE TABLE IF NOT EXISTS order_status_log (
    id SERIAL PRIMARY KEY,
    order_id UUID NOT NULL,
    external_id VARCHAR(50),
    status VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    details JSONB,

    -- Indexes
    CONSTRAINT order_status_log_status_check
        CHECK (status IN ('pending', 'open', 'partially_filled', 'filled', 'cancelled', 'expired', 'error'))
);

CREATE INDEX IF NOT EXISTS idx_order_status_log_order_id
    ON order_status_log(order_id);
CREATE INDEX IF NOT EXISTS idx_order_status_log_timestamp
    ON order_status_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_order_status_log_external_id
    ON order_status_log(external_id) WHERE external_id IS NOT NULL;

-- ============================================================================
-- Positions Table
-- ============================================================================
-- Tracks open and closed trading positions.

CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    size DECIMAL(20, 10) NOT NULL,
    entry_price DECIMAL(20, 10) NOT NULL,
    leverage INT NOT NULL DEFAULT 1,
    status VARCHAR(20) NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'closing', 'closed', 'liquidated')),
    order_id VARCHAR(50),
    stop_loss DECIMAL(20, 10),
    take_profit DECIMAL(20, 10),
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    exit_price DECIMAL(20, 10),
    realized_pnl DECIMAL(20, 10) DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at DESC);

-- ============================================================================
-- Position Snapshots (TimescaleDB Hypertable)
-- ============================================================================
-- Time-series data for position P&L tracking.

CREATE TABLE IF NOT EXISTS position_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    position_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    current_price DECIMAL(20, 10),
    unrealized_pnl DECIMAL(20, 10),
    unrealized_pnl_pct DECIMAL(10, 4),

    PRIMARY KEY (timestamp, position_id)
);

-- Convert to hypertable if TimescaleDB is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('position_snapshots', 'timestamp',
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_position_snapshots_position_id
    ON position_snapshots(position_id, timestamp DESC);

-- ============================================================================
-- Hodl Bags Table
-- ============================================================================
-- Tracks hodl bag allocations (excluded from trading).

CREATE TABLE IF NOT EXISTS hodl_bags (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL DEFAULT 'default',
    asset VARCHAR(10) NOT NULL,
    amount DECIMAL(20, 10) NOT NULL DEFAULT 0,
    last_addition TIMESTAMPTZ,
    total_additions DECIMAL(20, 10) DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(account_id, asset)
);

-- Insert default hodl bag entries
INSERT INTO hodl_bags (account_id, asset, amount)
VALUES
    ('default', 'BTC', 0),
    ('default', 'XRP', 0),
    ('default', 'USDT', 0)
ON CONFLICT (account_id, asset) DO NOTHING;

-- ============================================================================
-- Hodl Bag History
-- ============================================================================
-- Tracks additions to hodl bags over time.

CREATE TABLE IF NOT EXISTS hodl_bag_history (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL DEFAULT 'default',
    asset VARCHAR(10) NOT NULL,
    amount_added DECIMAL(20, 10) NOT NULL,
    source_trade_id UUID,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hodl_bag_history_timestamp
    ON hodl_bag_history(timestamp DESC);

-- ============================================================================
-- Coordinator State
-- ============================================================================
-- Persists coordinator state for recovery.

CREATE TABLE IF NOT EXISTS coordinator_state (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'current',
    state VARCHAR(20) NOT NULL DEFAULT 'running',
    last_task_runs JSONB DEFAULT '{}',
    statistics JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Initialize coordinator state
INSERT INTO coordinator_state (id, state)
VALUES ('current', 'running')
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- Rebalancing History
-- ============================================================================
-- Tracks portfolio rebalancing events.

CREATE TABLE IF NOT EXISTS rebalancing_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger_reason VARCHAR(100),
    before_allocation JSONB,
    after_allocation JSONB,
    trades JSONB,
    execution_strategy VARCHAR(50),
    total_volume_usd DECIMAL(20, 10),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'executing', 'completed', 'failed', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS idx_rebalancing_history_timestamp
    ON rebalancing_history(timestamp DESC);

-- ============================================================================
-- Conflict Resolution Log
-- ============================================================================
-- Tracks coordinator conflict resolutions.

CREATE TABLE IF NOT EXISTS conflict_resolution_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    conflict_type VARCHAR(50) NOT NULL,
    agents_involved TEXT[],
    signal_details JSONB,
    resolution_action VARCHAR(20) NOT NULL,
    resolution_reasoning TEXT,
    llm_model_used VARCHAR(50),
    latency_ms INT
);

CREATE INDEX IF NOT EXISTS idx_conflict_resolution_log_timestamp
    ON conflict_resolution_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_conflict_resolution_log_type
    ON conflict_resolution_log(conflict_type);

-- ============================================================================
-- Execution Events Log
-- ============================================================================
-- Tracks all execution events for monitoring.

CREATE TABLE IF NOT EXISTS execution_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    order_id UUID,
    position_id UUID,
    symbol VARCHAR(20),
    details JSONB,
    source VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_execution_events_timestamp
    ON execution_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_execution_events_type
    ON execution_events(event_type);
CREATE INDEX IF NOT EXISTS idx_execution_events_order_id
    ON execution_events(order_id) WHERE order_id IS NOT NULL;

-- ============================================================================
-- Data Retention Policies (TimescaleDB)
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Keep position snapshots for 90 days
        PERFORM add_retention_policy('position_snapshots', INTERVAL '90 days',
            if_not_exists => TRUE);

        -- Add compression after 7 days
        ALTER TABLE position_snapshots SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'position_id'
        );
        PERFORM add_compression_policy('position_snapshots', INTERVAL '7 days',
            if_not_exists => TRUE);
    END IF;
END $$;

-- ============================================================================
-- Cleanup Functions
-- ============================================================================

-- Function to clean old order status logs
CREATE OR REPLACE FUNCTION cleanup_old_order_logs(retention_days INT DEFAULT 90)
RETURNS INT AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM order_status_log
    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean old execution events
CREATE OR REPLACE FUNCTION cleanup_old_execution_events(retention_days INT DEFAULT 90)
RETURNS INT AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM execution_events
    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Trigger for updated_at timestamps
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_hodl_bags_updated_at
    BEFORE UPDATE ON hodl_bags
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_coordinator_state_updated_at
    BEFORE UPDATE ON coordinator_state
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Grants (adjust as needed for your setup)
-- ============================================================================

-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO triplegain;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO triplegain;

-- ============================================================================
-- Migration complete
-- ============================================================================
-- Verify tables:
-- \dt order_status_log positions position_snapshots hodl_bags coordinator_state
