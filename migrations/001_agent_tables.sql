-- ============================================================================
-- Migration: 001_agent_tables.sql
-- Description: Create tables for TripleGain Phase 1 - Agent outputs and trading
-- Created: 2025-12-18
-- ============================================================================

-- Ensure pgcrypto extension for UUID generation
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- AGENT OUTPUTS TABLE
-- Purpose: Store all agent outputs for audit, learning, and evaluation
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_name VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    output_type VARCHAR(50) NOT NULL,
    output_data JSONB NOT NULL,
    model_used VARCHAR(50),
    prompt_hash VARCHAR(64),  -- For caching/deduplication
    latency_ms INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost_usd DECIMAL(10, 6),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_agent_outputs_agent_ts
    ON agent_outputs (agent_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_outputs_symbol_ts
    ON agent_outputs (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_outputs_prompt_hash
    ON agent_outputs (prompt_hash, timestamp DESC);

COMMENT ON TABLE agent_outputs IS 'Stores all LLM agent outputs for audit and evaluation';


-- ============================================================================
-- TRADING DECISIONS TABLE
-- Purpose: Audit trail for all trading decisions (approved, modified, rejected)
-- ============================================================================

CREATE TABLE IF NOT EXISTS trading_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    decision_type VARCHAR(20) NOT NULL CHECK (
        decision_type IN ('signal', 'execution', 'modification', 'rebalance')
    ),
    action VARCHAR(10) NOT NULL CHECK (
        action IN ('BUY', 'SELL', 'HOLD', 'CLOSE', 'REBALANCE')
    ),
    confidence DECIMAL(4, 3),
    parameters JSONB,
    agent_inputs JSONB,  -- References to agent_outputs UUIDs
    risk_evaluation JSONB,
    final_status VARCHAR(20) NOT NULL CHECK (
        final_status IN ('approved', 'modified', 'rejected', 'pending', 'executed')
    ),
    rejection_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trading_decisions_symbol_ts
    ON trading_decisions (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_decisions_status
    ON trading_decisions (final_status, timestamp DESC);

COMMENT ON TABLE trading_decisions IS 'Audit trail for all trading decisions';


-- ============================================================================
-- TRADE EXECUTIONS TABLE
-- Purpose: Track executed trades and their outcomes
-- ============================================================================

CREATE TABLE IF NOT EXISTS trade_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID REFERENCES trading_decisions(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(5) NOT NULL CHECK (side IN ('long', 'short')),
    size DECIMAL(20, 10) NOT NULL,
    size_usd DECIMAL(20, 2) NOT NULL,
    entry_price DECIMAL(20, 10) NOT NULL,
    leverage INTEGER DEFAULT 1 CHECK (leverage BETWEEN 1 AND 5),
    stop_loss DECIMAL(20, 10),
    take_profit DECIMAL(20, 10),
    status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (
        status IN ('open', 'closed', 'partially_closed', 'cancelled')
    ),
    exit_price DECIMAL(20, 10),
    exit_timestamp TIMESTAMPTZ,
    realized_pnl DECIMAL(20, 10),
    realized_pnl_pct DECIMAL(10, 4),
    fees_usd DECIMAL(20, 6),
    exit_reason VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trade_executions_status
    ON trade_executions (status, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trade_executions_symbol
    ON trade_executions (symbol, status, timestamp DESC);

COMMENT ON TABLE trade_executions IS 'Tracks all executed trades and their outcomes';


-- ============================================================================
-- PORTFOLIO SNAPSHOTS TABLE
-- Purpose: Track portfolio state over time for performance analysis
-- ============================================================================

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    total_equity_usd DECIMAL(20, 2) NOT NULL,
    available_margin_usd DECIMAL(20, 2) NOT NULL,
    used_margin_usd DECIMAL(20, 2) NOT NULL,

    -- Balances
    btc_balance DECIMAL(20, 10) DEFAULT 0,
    xrp_balance DECIMAL(20, 10) DEFAULT 0,
    usdt_balance DECIMAL(20, 2) DEFAULT 0,

    -- Hodl bags (separate from trading balances)
    btc_hodl DECIMAL(20, 10) DEFAULT 0,
    xrp_hodl DECIMAL(20, 10) DEFAULT 0,
    usdt_hodl DECIMAL(20, 2) DEFAULT 0,

    -- Allocation percentages
    allocation_btc_pct DECIMAL(5, 2),
    allocation_xrp_pct DECIMAL(5, 2),
    allocation_usdt_pct DECIMAL(5, 2),

    -- Performance metrics
    unrealized_pnl DECIMAL(20, 2),
    daily_pnl DECIMAL(20, 2),
    daily_pnl_pct DECIMAL(10, 4),
    peak_equity_usd DECIMAL(20, 2),
    drawdown_pct DECIMAL(10, 4),

    -- Risk state
    open_positions_count INTEGER DEFAULT 0,
    total_exposure_pct DECIMAL(10, 4),

    PRIMARY KEY (timestamp)
);

-- Convert to hypertable for efficient time-series queries
SELECT create_hypertable('portfolio_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

COMMENT ON TABLE portfolio_snapshots IS 'Time-series portfolio state for performance analysis';


-- ============================================================================
-- RISK STATE TABLE
-- Purpose: Track risk-related state (circuit breakers, cooldowns)
-- ============================================================================

CREATE TABLE IF NOT EXISTS risk_state (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Loss tracking
    daily_loss_pct DECIMAL(10, 4) DEFAULT 0,
    weekly_loss_pct DECIMAL(10, 4) DEFAULT 0,
    max_drawdown_pct DECIMAL(10, 4) DEFAULT 0,
    peak_equity_usd DECIMAL(20, 2),

    -- Trade tracking
    consecutive_losses INTEGER DEFAULT 0,
    trades_today INTEGER DEFAULT 0,

    -- Circuit breaker state
    circuit_breakers_active JSONB DEFAULT '[]'::jsonb,
    trading_halted BOOLEAN DEFAULT FALSE,
    halt_reason TEXT,
    halt_until TIMESTAMPTZ,

    -- Cooldowns
    active_cooldowns JSONB DEFAULT '{}'::jsonb,

    -- Metadata
    last_trade_timestamp TIMESTAMPTZ,
    daily_reset_timestamp TIMESTAMPTZ,
    weekly_reset_timestamp TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_risk_state_timestamp
    ON risk_state (timestamp DESC);

COMMENT ON TABLE risk_state IS 'Current risk state including circuit breakers and cooldowns';


-- ============================================================================
-- EXTERNAL DATA CACHE TABLE
-- Purpose: Cache external API responses (news, sentiment indicators)
-- ============================================================================

CREATE TABLE IF NOT EXISTS external_data_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    data JSONB NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    UNIQUE (source, data_type, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_external_data_source_ts
    ON external_data_cache (source, data_type, timestamp DESC);

COMMENT ON TABLE external_data_cache IS 'Cache for external API data (news, sentiment)';


-- ============================================================================
-- INDICATOR CACHE TABLE
-- Purpose: Cache computed indicators to avoid recalculation
-- ============================================================================

CREATE TABLE IF NOT EXISTS indicator_cache (
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DECIMAL(30, 10),
    metadata JSONB,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, indicator_name, timestamp)
);

-- Convert to hypertable for efficient time-series queries
SELECT create_hypertable('indicator_cache', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

COMMENT ON TABLE indicator_cache IS 'Cache for computed technical indicators';


-- ============================================================================
-- HELPER FUNCTION: Update timestamp trigger
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to trade_executions
DROP TRIGGER IF EXISTS update_trade_executions_updated_at ON trade_executions;
CREATE TRIGGER update_trade_executions_updated_at
    BEFORE UPDATE ON trade_executions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN (
        'agent_outputs',
        'trading_decisions',
        'trade_executions',
        'portfolio_snapshots',
        'risk_state',
        'external_data_cache',
        'indicator_cache'
    );

    IF table_count = 7 THEN
        RAISE NOTICE 'Migration 001_agent_tables completed successfully. All 7 tables created.';
    ELSE
        RAISE EXCEPTION 'Migration failed. Expected 7 tables, found %', table_count;
    END IF;
END $$;
