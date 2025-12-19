-- ============================================================================
-- Migration: 002_model_comparisons.sql
-- Description: Create model_comparisons table for 6-model A/B testing
-- Created: 2025-12-18
-- Phase: 2 - Core Agents
-- ============================================================================

-- ============================================================================
-- MODEL COMPARISONS TABLE
-- Purpose: Track individual model decisions for A/B testing and comparison
-- Each row represents one model's decision for a trading opportunity
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,

    -- Model identification
    model_name VARCHAR(50) NOT NULL,  -- e.g., 'qwen', 'gpt4', 'grok', 'deepseek', 'sonnet', 'opus'
    provider VARCHAR(30) NOT NULL,     -- e.g., 'ollama', 'openai', 'anthropic', 'deepseek', 'xai'

    -- Decision details
    action VARCHAR(20) NOT NULL CHECK (
        action IN ('BUY', 'SELL', 'HOLD', 'CLOSE_LONG', 'CLOSE_SHORT')
    ),
    confidence DECIMAL(4, 3) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    entry_price DECIMAL(20, 10),
    stop_loss DECIMAL(20, 10),
    take_profit DECIMAL(20, 10),
    position_size_pct DECIMAL(5, 2),
    reasoning TEXT,

    -- Performance metrics
    latency_ms INTEGER NOT NULL DEFAULT 0,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    cost_usd DECIMAL(10, 6) NOT NULL DEFAULT 0,

    -- Comparison context
    consensus_action VARCHAR(20),      -- The final consensus action taken
    was_consensus BOOLEAN,             -- Did this model agree with consensus?

    -- Outcome tracking (updated after trade resolution)
    -- These are updated later to evaluate model accuracy
    price_at_decision DECIMAL(20, 10),
    price_after_1h DECIMAL(20, 10),
    price_after_4h DECIMAL(20, 10),
    price_after_24h DECIMAL(20, 10),
    was_correct BOOLEAN,               -- Updated after outcome known
    outcome_pnl_pct DECIMAL(10, 4),    -- % P&L if model's decision was followed

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable for efficient time-series queries
SELECT create_hypertable('model_comparisons', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Indexes for common query patterns
-- Query by model to analyze model performance
CREATE INDEX IF NOT EXISTS idx_model_comparisons_model_ts
    ON model_comparisons (model_name, timestamp DESC);

-- Query by symbol to see all models' decisions for a symbol
CREATE INDEX IF NOT EXISTS idx_model_comparisons_symbol_ts
    ON model_comparisons (symbol, timestamp DESC);

-- Query consensus agreement for accuracy analysis
CREATE INDEX IF NOT EXISTS idx_model_comparisons_consensus
    ON model_comparisons (was_consensus, timestamp DESC);

-- Query for outcome analysis
CREATE INDEX IF NOT EXISTS idx_model_comparisons_outcome
    ON model_comparisons (was_correct, model_name, timestamp DESC);

-- Composite index for model performance by symbol
CREATE INDEX IF NOT EXISTS idx_model_comparisons_perf
    ON model_comparisons (model_name, symbol, was_correct, timestamp DESC);

COMMENT ON TABLE model_comparisons IS 'Tracks individual model decisions for 6-model A/B testing';
COMMENT ON COLUMN model_comparisons.was_consensus IS 'True if model agreed with final consensus action';
COMMENT ON COLUMN model_comparisons.was_correct IS 'Updated after trade outcome known - based on price movement';
COMMENT ON COLUMN model_comparisons.outcome_pnl_pct IS 'Hypothetical P&L if this model''s recommendation was followed';


-- ============================================================================
-- MODEL PERFORMANCE SUMMARY VIEW
-- Purpose: Easy access to model performance metrics
-- ============================================================================

CREATE OR REPLACE VIEW model_performance_summary AS
SELECT
    model_name,
    provider,
    COUNT(*) as total_decisions,
    COUNT(CASE WHEN was_consensus THEN 1 END) as consensus_matches,
    ROUND(100.0 * COUNT(CASE WHEN was_consensus THEN 1 END) / NULLIF(COUNT(*), 0), 2) as consensus_rate,
    COUNT(CASE WHEN was_correct THEN 1 END) as correct_decisions,
    ROUND(100.0 * COUNT(CASE WHEN was_correct THEN 1 END) / NULLIF(COUNT(CASE WHEN was_correct IS NOT NULL THEN 1 END), 0), 2) as accuracy_pct,
    ROUND(AVG(outcome_pnl_pct)::numeric, 4) as avg_outcome_pnl_pct,
    ROUND(AVG(latency_ms)::numeric, 0) as avg_latency_ms,
    ROUND(SUM(cost_usd)::numeric, 4) as total_cost_usd,
    ROUND(AVG(cost_usd)::numeric, 6) as avg_cost_per_decision,
    MIN(timestamp) as first_decision,
    MAX(timestamp) as last_decision
FROM model_comparisons
GROUP BY model_name, provider
ORDER BY accuracy_pct DESC NULLS LAST;

COMMENT ON VIEW model_performance_summary IS 'Aggregated model performance metrics for A/B comparison';


-- ============================================================================
-- MODEL PERFORMANCE BY SYMBOL VIEW
-- Purpose: Model performance broken down by trading symbol
-- ============================================================================

CREATE OR REPLACE VIEW model_performance_by_symbol AS
SELECT
    model_name,
    symbol,
    COUNT(*) as total_decisions,
    ROUND(100.0 * COUNT(CASE WHEN was_consensus THEN 1 END) / NULLIF(COUNT(*), 0), 2) as consensus_rate,
    ROUND(100.0 * COUNT(CASE WHEN was_correct THEN 1 END) / NULLIF(COUNT(CASE WHEN was_correct IS NOT NULL THEN 1 END), 0), 2) as accuracy_pct,
    ROUND(AVG(outcome_pnl_pct)::numeric, 4) as avg_outcome_pnl_pct,
    ROUND(AVG(confidence)::numeric, 3) as avg_confidence
FROM model_comparisons
GROUP BY model_name, symbol
ORDER BY model_name, symbol;

COMMENT ON VIEW model_performance_by_symbol IS 'Model performance metrics by trading symbol';


-- ============================================================================
-- TRIGGER: Update timestamp
-- ============================================================================

DROP TRIGGER IF EXISTS update_model_comparisons_updated_at ON model_comparisons;
CREATE TRIGGER update_model_comparisons_updated_at
    BEFORE UPDATE ON model_comparisons
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================================================
-- RETENTION POLICY
-- Purpose: Keep model comparison data for 90 days (same as agent_outputs)
-- ============================================================================

SELECT add_retention_policy('model_comparisons', INTERVAL '90 days',
    if_not_exists => TRUE);

COMMENT ON COLUMN model_comparisons.timestamp IS 'Retention: 90 days';


-- ============================================================================
-- COMPRESSION POLICY
-- Purpose: Compress old data to save storage
-- ============================================================================

ALTER TABLE model_comparisons SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'model_name,symbol'
);

SELECT add_compression_policy('model_comparisons', INTERVAL '7 days',
    if_not_exists => TRUE);


-- ============================================================================
-- FUNCTION: Update outcome after trade resolution
-- Purpose: Called after price data available to calculate if decision was correct
-- ============================================================================

CREATE OR REPLACE FUNCTION update_model_comparison_outcome(
    p_comparison_id UUID,
    p_price_1h DECIMAL,
    p_price_4h DECIMAL,
    p_price_24h DECIMAL
) RETURNS BOOLEAN AS $$
DECLARE
    v_action VARCHAR(20);
    v_price_at_decision DECIMAL;
    v_was_correct BOOLEAN;
    v_pnl_pct DECIMAL;
BEGIN
    -- Get the original decision
    SELECT action, price_at_decision
    INTO v_action, v_price_at_decision
    FROM model_comparisons
    WHERE id = p_comparison_id;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    -- Determine if correct (simplified: BUY correct if price went up, SELL if down)
    IF v_action = 'BUY' THEN
        v_was_correct := p_price_4h > v_price_at_decision;
        v_pnl_pct := ((p_price_4h - v_price_at_decision) / v_price_at_decision) * 100;
    ELSIF v_action = 'SELL' THEN
        v_was_correct := p_price_4h < v_price_at_decision;
        v_pnl_pct := ((v_price_at_decision - p_price_4h) / v_price_at_decision) * 100;
    ELSE
        -- HOLD/CLOSE actions - consider correct if price stayed within 2%
        v_was_correct := ABS((p_price_4h - v_price_at_decision) / v_price_at_decision) < 0.02;
        v_pnl_pct := 0;
    END IF;

    -- Update the record
    UPDATE model_comparisons
    SET
        price_after_1h = p_price_1h,
        price_after_4h = p_price_4h,
        price_after_24h = p_price_24h,
        was_correct = v_was_correct,
        outcome_pnl_pct = v_pnl_pct,
        updated_at = NOW()
    WHERE id = p_comparison_id;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_model_comparison_outcome IS 'Updates model comparison with outcome after price data available';


-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
DECLARE
    table_exists BOOLEAN;
    views_exist INTEGER;
BEGIN
    SELECT EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = 'model_comparisons'
    ) INTO table_exists;

    SELECT COUNT(*) INTO views_exist
    FROM information_schema.views
    WHERE table_schema = 'public'
    AND table_name IN ('model_performance_summary', 'model_performance_by_symbol');

    IF table_exists AND views_exist = 2 THEN
        RAISE NOTICE 'Migration 002_model_comparisons completed successfully.';
        RAISE NOTICE 'Created: model_comparisons table, 2 performance views, 1 outcome function.';
    ELSE
        RAISE EXCEPTION 'Migration 002_model_comparisons failed.';
    END IF;
END $$;
