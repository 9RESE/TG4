-- ============================================================================
-- Phase 8: Hodl Bag System Database Migration
-- ============================================================================
-- Creates tables for hodl bag profit accumulation system:
-- - hodl_bags: Current hodl bag balances per asset
-- - hodl_transactions: Transaction history for accumulations/withdrawals
-- - hodl_pending: Pending accumulations waiting for threshold
-- - hodl_bag_snapshots: Time-series snapshots for value tracking
--
-- Run: psql -h localhost -U triplegain -d triplegain -f migrations/009_hodl_bags.sql
-- ============================================================================

-- ============================================================================
-- Hodl Bags - Current Holdings
-- ============================================================================
-- One row per asset (BTC, XRP, USDT) tracking current hodl balance.

CREATE TABLE IF NOT EXISTS hodl_bags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset VARCHAR(10) NOT NULL,  -- BTC, XRP, USDT
    balance DECIMAL(20, 10) NOT NULL DEFAULT 0
        CHECK (balance >= 0),
    cost_basis_usd DECIMAL(20, 2) NOT NULL DEFAULT 0
        CHECK (cost_basis_usd >= 0),
    first_accumulation TIMESTAMPTZ,
    last_accumulation TIMESTAMPTZ,
    last_valuation_usd DECIMAL(20, 2),
    last_valuation_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(asset)
);

-- Initialize with zero balances for each asset
INSERT INTO hodl_bags (asset, balance, cost_basis_usd)
VALUES ('BTC', 0, 0), ('XRP', 0, 0), ('USDT', 0, 0)
ON CONFLICT (asset) DO NOTHING;

-- Index for quick asset lookups
CREATE INDEX IF NOT EXISTS idx_hodl_bags_asset
    ON hodl_bags (asset);

COMMENT ON TABLE hodl_bags IS 'Current hodl bag balances per asset - excluded from trading/rebalancing';
COMMENT ON COLUMN hodl_bags.balance IS 'Current asset balance in hodl bag';
COMMENT ON COLUMN hodl_bags.cost_basis_usd IS 'Total USD cost basis for this asset';
COMMENT ON COLUMN hodl_bags.first_accumulation IS 'Timestamp of first accumulation';
COMMENT ON COLUMN hodl_bags.last_accumulation IS 'Timestamp of most recent accumulation';


-- ============================================================================
-- Hodl Transactions - Accumulation/Withdrawal History
-- ============================================================================
-- All hodl bag transactions for audit trail.

CREATE TABLE IF NOT EXISTS hodl_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset VARCHAR(10) NOT NULL,
    transaction_type VARCHAR(20) NOT NULL CHECK (
        transaction_type IN ('accumulation', 'withdrawal', 'adjustment')
    ),
    amount DECIMAL(20, 10) NOT NULL,  -- Asset amount (positive for accumulation, negative for withdrawal)
    price_usd DECIMAL(20, 10) NOT NULL,  -- Price at execution
    value_usd DECIMAL(20, 2) NOT NULL,  -- USD value of transaction
    source_trade_id UUID,  -- Reference to trade/position (null for withdrawals)
    order_id VARCHAR(50),  -- Exchange order ID (null for USDT/paper)
    fee_usd DECIMAL(20, 4) DEFAULT 0,  -- Trading fee if any
    notes TEXT,
    is_paper BOOLEAN DEFAULT FALSE,  -- True for paper trading simulations
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_hodl_transactions_asset_ts
    ON hodl_transactions (asset, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_hodl_transactions_source
    ON hodl_transactions (source_trade_id)
    WHERE source_trade_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_hodl_transactions_type
    ON hodl_transactions (transaction_type, timestamp DESC);

COMMENT ON TABLE hodl_transactions IS 'Hodl bag transaction history for audit trail';
COMMENT ON COLUMN hodl_transactions.transaction_type IS 'Type: accumulation, withdrawal, or adjustment';
COMMENT ON COLUMN hodl_transactions.source_trade_id IS 'UUID of the profitable trade that triggered this accumulation';
COMMENT ON COLUMN hodl_transactions.is_paper IS 'True if this was a paper trading simulation';


-- ============================================================================
-- Hodl Pending - Pending Accumulations
-- ============================================================================
-- Tracks pending USD amounts waiting for threshold before purchase.

CREATE TABLE IF NOT EXISTS hodl_pending (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset VARCHAR(10) NOT NULL,
    amount_usd DECIMAL(20, 2) NOT NULL CHECK (amount_usd > 0),
    source_trade_id UUID NOT NULL,
    source_profit_usd DECIMAL(20, 2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    executed_at TIMESTAMPTZ,
    execution_transaction_id UUID REFERENCES hodl_transactions(id),
    is_paper BOOLEAN DEFAULT FALSE
);

-- Index for finding unexecuted pending by asset
CREATE INDEX IF NOT EXISTS idx_hodl_pending_asset_unexecuted
    ON hodl_pending (asset, created_at)
    WHERE executed_at IS NULL;

-- Index for paper trading queries
CREATE INDEX IF NOT EXISTS idx_hodl_pending_paper
    ON hodl_pending (is_paper, asset)
    WHERE executed_at IS NULL;

COMMENT ON TABLE hodl_pending IS 'Pending hodl accumulations waiting for threshold';
COMMENT ON COLUMN hodl_pending.amount_usd IS 'USD amount pending for this asset';
COMMENT ON COLUMN hodl_pending.executed_at IS 'When this pending was executed (null if still pending)';


-- ============================================================================
-- Hodl Bag Snapshots - Time-Series Value Tracking
-- ============================================================================
-- Daily snapshots for tracking hodl bag value over time.

CREATE TABLE IF NOT EXISTS hodl_bag_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    asset VARCHAR(10) NOT NULL,
    balance DECIMAL(20, 10) NOT NULL,
    price_usd DECIMAL(20, 10) NOT NULL,
    value_usd DECIMAL(20, 2) NOT NULL,
    cost_basis_usd DECIMAL(20, 2) NOT NULL,
    unrealized_pnl_usd DECIMAL(20, 2) NOT NULL,
    unrealized_pnl_pct DECIMAL(10, 4) NOT NULL,
    PRIMARY KEY (timestamp, asset)
);

-- Enable hypertable for time-series queries (if TimescaleDB extension is available)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('hodl_bag_snapshots', 'timestamp',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE);
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        -- TimescaleDB not available, use regular table
        RAISE NOTICE 'TimescaleDB not available, using regular table for hodl_bag_snapshots';
END $$;

-- Indexes for snapshot queries
CREATE INDEX IF NOT EXISTS idx_hodl_snapshots_asset_ts
    ON hodl_bag_snapshots (asset, timestamp DESC);

COMMENT ON TABLE hodl_bag_snapshots IS 'Daily snapshots of hodl bag value for time-series analysis';
COMMENT ON COLUMN hodl_bag_snapshots.unrealized_pnl_pct IS 'Percentage P&L = (value - cost_basis) / cost_basis * 100';


-- ============================================================================
-- Helper Views
-- ============================================================================

-- Latest hodl bag status per asset
CREATE OR REPLACE VIEW latest_hodl_bags AS
SELECT
    asset,
    balance,
    cost_basis_usd,
    last_valuation_usd AS current_value_usd,
    CASE
        WHEN cost_basis_usd > 0
        THEN (last_valuation_usd - cost_basis_usd)
        ELSE 0
    END AS unrealized_pnl_usd,
    CASE
        WHEN cost_basis_usd > 0
        THEN ROUND(((last_valuation_usd - cost_basis_usd) / cost_basis_usd * 100)::NUMERIC, 2)
        ELSE 0
    END AS unrealized_pnl_pct,
    first_accumulation,
    last_accumulation,
    last_valuation_timestamp,
    updated_at
FROM hodl_bags
WHERE balance > 0
ORDER BY asset;

COMMENT ON VIEW latest_hodl_bags IS 'Current hodl bag status with unrealized P&L';


-- Pending accumulation totals per asset
CREATE OR REPLACE VIEW hodl_pending_totals AS
SELECT
    asset,
    SUM(amount_usd) AS pending_usd,
    COUNT(*) AS pending_count,
    MIN(created_at) AS oldest_pending,
    MAX(created_at) AS newest_pending
FROM hodl_pending
WHERE executed_at IS NULL
GROUP BY asset;

COMMENT ON VIEW hodl_pending_totals IS 'Total pending accumulations per asset';


-- Hodl bag performance summary
CREATE OR REPLACE VIEW hodl_performance_summary AS
SELECT
    h.asset,
    h.balance,
    h.cost_basis_usd,
    h.last_valuation_usd AS current_value_usd,
    COALESCE(h.last_valuation_usd, 0) - h.cost_basis_usd AS unrealized_pnl_usd,
    CASE
        WHEN h.cost_basis_usd > 0
        THEN ROUND(((COALESCE(h.last_valuation_usd, 0) - h.cost_basis_usd) / h.cost_basis_usd * 100)::NUMERIC, 2)
        ELSE 0
    END AS unrealized_pnl_pct,
    COALESCE(p.pending_usd, 0) AS pending_usd,
    t.total_accumulations,
    t.total_accumulated_usd,
    h.first_accumulation,
    h.last_accumulation
FROM hodl_bags h
LEFT JOIN hodl_pending_totals p ON h.asset = p.asset
LEFT JOIN (
    SELECT
        asset,
        COUNT(*) AS total_accumulations,
        SUM(value_usd) AS total_accumulated_usd
    FROM hodl_transactions
    WHERE transaction_type = 'accumulation'
    GROUP BY asset
) t ON h.asset = t.asset
ORDER BY h.asset;

COMMENT ON VIEW hodl_performance_summary IS 'Complete hodl bag performance with pending and historical data';


-- ============================================================================
-- Trigger for updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION update_hodl_bags_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS hodl_bags_updated_at_trigger ON hodl_bags;
CREATE TRIGGER hodl_bags_updated_at_trigger
    BEFORE UPDATE ON hodl_bags
    FOR EACH ROW
    EXECUTE FUNCTION update_hodl_bags_updated_at();


-- ============================================================================
-- Cleanup Function
-- ============================================================================
-- Function to clean up old snapshot data (run periodically).

CREATE OR REPLACE FUNCTION cleanup_old_hodl_snapshots(retention_days INTEGER DEFAULT 365)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    cutoff_date TIMESTAMPTZ;
BEGIN
    cutoff_date := NOW() - (retention_days || ' days')::INTERVAL;

    DELETE FROM hodl_bag_snapshots
    WHERE timestamp < cutoff_date;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_hodl_snapshots IS 'Removes hodl bag snapshots older than specified retention period';


-- ============================================================================
-- Grant Permissions
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE ON hodl_bags TO trading;
        GRANT SELECT, INSERT, UPDATE, DELETE ON hodl_transactions TO trading;
        GRANT SELECT, INSERT, UPDATE, DELETE ON hodl_pending TO trading;
        GRANT SELECT, INSERT, UPDATE, DELETE ON hodl_bag_snapshots TO trading;
        GRANT SELECT ON latest_hodl_bags TO trading;
        GRANT SELECT ON hodl_pending_totals TO trading;
        GRANT SELECT ON hodl_performance_summary TO trading;
        GRANT EXECUTE ON FUNCTION cleanup_old_hodl_snapshots TO trading;
    END IF;
END $$;


-- ============================================================================
-- Migration Complete
-- ============================================================================
-- Phase 8: Hodl Bag System tables created successfully.
--
-- Tables:
--   - hodl_bags: Current hodl bag balances per asset
--   - hodl_transactions: All accumulation/withdrawal transactions
--   - hodl_pending: Pending accumulations waiting for threshold
--   - hodl_bag_snapshots: Daily value snapshots for time-series
--
-- Views:
--   - latest_hodl_bags: Current status with unrealized P&L
--   - hodl_pending_totals: Total pending per asset
--   - hodl_performance_summary: Complete performance metrics
--
-- Functions:
--   - cleanup_old_hodl_snapshots(days): Remove old snapshots
-- ============================================================================
