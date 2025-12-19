-- ============================================================================
-- Migration: 003_risk_state_and_indexes.sql
-- Description: Add risk_state persistence table and additional indexes
-- Created: 2025-12-18
-- Phase: 2 - Code Review Fixes
-- ============================================================================

-- ============================================================================
-- RISK STATE TABLE
-- Purpose: Persist risk management state across restarts
-- Single row table with 'current' as the ID
-- Note: This replaces the less detailed risk_state from 001_agent_tables.sql
-- ============================================================================

-- Drop the old risk_state table if it exists (migration from 001)
-- The new schema uses JSONB for flexibility
DROP TABLE IF EXISTS risk_state CASCADE;

CREATE TABLE IF NOT EXISTS risk_state (
    id VARCHAR(20) PRIMARY KEY DEFAULT 'current',
    state_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE risk_state IS 'Persists risk management state (single row) for recovery after restart';
COMMENT ON COLUMN risk_state.state_data IS 'JSON containing full RiskState: equity, drawdown, circuit breakers, cooldowns';

-- Create trigger for updated_at
DROP TRIGGER IF EXISTS update_risk_state_updated_at ON risk_state;
CREATE TRIGGER update_risk_state_updated_at
    BEFORE UPDATE ON risk_state
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ADDITIONAL INDEXES FOR MODEL COMPARISONS
-- Purpose: Optimize queries for model performance analysis
-- ============================================================================

-- Index on model_name for per-model queries
CREATE INDEX IF NOT EXISTS idx_model_comparisons_model_name
    ON model_comparisons (model_name, timestamp DESC);

-- Index on was_consensus for accuracy analysis
CREATE INDEX IF NOT EXISTS idx_model_comparisons_was_consensus
    ON model_comparisons (was_consensus, timestamp DESC)
    WHERE was_consensus IS NOT NULL;

-- Index on action for action distribution analysis
CREATE INDEX IF NOT EXISTS idx_model_comparisons_action
    ON model_comparisons (action, timestamp DESC);

-- Index on cost_usd for cost analysis
CREATE INDEX IF NOT EXISTS idx_model_comparisons_cost
    ON model_comparisons (timestamp DESC, cost_usd)
    WHERE cost_usd > 0;

-- ============================================================================
-- ADDITIONAL INDEXES FOR AGENT OUTPUTS
-- Purpose: Optimize queries for agent performance analysis
-- ============================================================================

-- Composite index for agent stats queries
CREATE INDEX IF NOT EXISTS idx_agent_outputs_stats
    ON agent_outputs (agent_name, symbol, confidence, timestamp DESC);

-- Index for latency analysis
CREATE INDEX IF NOT EXISTS idx_agent_outputs_latency
    ON agent_outputs (agent_name, latency_ms, timestamp DESC);

-- Index for high confidence outputs
CREATE INDEX IF NOT EXISTS idx_agent_outputs_high_conf
    ON agent_outputs (agent_name, symbol, timestamp DESC)
    WHERE confidence >= 0.7;

-- ============================================================================
-- RISK STATE HISTORY TABLE (Optional)
-- Purpose: Track risk state changes over time for audit
-- ============================================================================

CREATE TABLE IF NOT EXISTS risk_state_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,  -- 'update', 'reset_daily', 'reset_weekly', 'circuit_breaker', 'cooldown'
    state_before JSONB,
    state_after JSONB NOT NULL,
    triggered_by VARCHAR(100),  -- What triggered the change
    notes TEXT
);

-- Convert to hypertable for efficient time-series storage
SELECT create_hypertable('risk_state_history', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Index for querying by event type
CREATE INDEX IF NOT EXISTS idx_risk_state_history_event
    ON risk_state_history (event_type, timestamp DESC);

-- Retention policy: Keep 30 days of history
SELECT add_retention_policy('risk_state_history', INTERVAL '30 days',
    if_not_exists => TRUE);

COMMENT ON TABLE risk_state_history IS 'Audit log of risk state changes for compliance and debugging';

-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
DECLARE
    tables_exist INTEGER;
    indexes_exist INTEGER;
BEGIN
    SELECT COUNT(*) INTO tables_exist
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN ('risk_state', 'risk_state_history');

    SELECT COUNT(*) INTO indexes_exist
    FROM pg_indexes
    WHERE schemaname = 'public'
    AND indexname LIKE 'idx_model_comparisons_%';

    IF tables_exist >= 2 AND indexes_exist >= 4 THEN
        RAISE NOTICE 'Migration 003_risk_state_and_indexes completed successfully.';
        RAISE NOTICE 'Created: risk_state, risk_state_history tables, additional indexes.';
    ELSE
        RAISE WARNING 'Migration 003 may be incomplete. Tables: %, Indexes: %', tables_exist, indexes_exist;
    END IF;
END $$;
