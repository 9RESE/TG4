-- ============================================================================
-- Phase 7: Sentiment Analysis Database Migration
-- ============================================================================
-- Creates tables for sentiment analysis data:
-- - sentiment_outputs: Main sentiment analysis outputs
-- - sentiment_provider_responses: Individual provider (Grok/GPT) responses
--
-- Run: psql -h localhost -U triplegain -d triplegain -f migrations/007_sentiment_analysis.sql
-- ============================================================================

-- ============================================================================
-- Sentiment Analysis Outputs
-- ============================================================================
-- Stores aggregated sentiment analysis from the dual-model system.

CREATE TABLE IF NOT EXISTS sentiment_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,

    -- Sentiment values
    bias VARCHAR(20) NOT NULL
        CHECK (bias IN ('very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish')),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    social_score DECIMAL(5, 4) CHECK (social_score >= -1 AND social_score <= 1),
    news_score DECIMAL(5, 4) CHECK (news_score >= -1 AND news_score <= 1),
    overall_score DECIMAL(5, 4) CHECK (overall_score >= -1 AND overall_score <= 1),
    fear_greed VARCHAR(20)
        CHECK (fear_greed IN ('extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed')),

    -- Provider details
    grok_available BOOLEAN DEFAULT FALSE,
    gpt_available BOOLEAN DEFAULT FALSE,

    -- Extended data (JSON for flexibility)
    key_events JSONB,
    market_narratives JSONB,
    reasoning TEXT,

    -- Metadata
    total_latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sentiment_outputs_symbol_ts
    ON sentiment_outputs (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_outputs_bias
    ON sentiment_outputs (bias, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_outputs_created_at
    ON sentiment_outputs (created_at DESC);

COMMENT ON TABLE sentiment_outputs IS 'Aggregated sentiment analysis outputs from Grok + GPT dual-model system';
COMMENT ON COLUMN sentiment_outputs.bias IS 'Final sentiment bias after aggregation';
COMMENT ON COLUMN sentiment_outputs.social_score IS 'Weighted social media sentiment score (-1 to 1)';
COMMENT ON COLUMN sentiment_outputs.news_score IS 'Weighted news sentiment score (-1 to 1)';
COMMENT ON COLUMN sentiment_outputs.overall_score IS 'Combined overall sentiment score (-1 to 1)';
COMMENT ON COLUMN sentiment_outputs.fear_greed IS 'Fear/Greed market assessment';
COMMENT ON COLUMN sentiment_outputs.providers_agreed IS 'Whether Grok and GPT agreed on bias';


-- ============================================================================
-- Sentiment Provider Responses
-- ============================================================================
-- Tracks individual provider responses for debugging and analysis.

CREATE TABLE IF NOT EXISTS sentiment_provider_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sentiment_output_id UUID REFERENCES sentiment_outputs(id) ON DELETE CASCADE,
    provider VARCHAR(20) NOT NULL CHECK (provider IN ('grok', 'gpt')),
    model VARCHAR(50) NOT NULL,

    -- Response data
    raw_response JSONB,
    parsed_bias VARCHAR(20)
        CHECK (parsed_bias IN ('very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish')),
    parsed_score DECIMAL(5, 4) CHECK (parsed_score >= -1 AND parsed_score <= 1),
    parsed_confidence DECIMAL(5, 4) CHECK (parsed_confidence >= 0 AND parsed_confidence <= 1),

    -- Performance
    latency_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,

    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_provider_responses_output
    ON sentiment_provider_responses (sentiment_output_id);

CREATE INDEX IF NOT EXISTS idx_provider_responses_provider
    ON sentiment_provider_responses (provider, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_provider_responses_success
    ON sentiment_provider_responses (success, timestamp DESC);

COMMENT ON TABLE sentiment_provider_responses IS 'Individual responses from Grok and GPT for debugging and performance analysis';
COMMENT ON COLUMN sentiment_provider_responses.raw_response IS 'Full JSON response from the LLM provider';
COMMENT ON COLUMN sentiment_provider_responses.success IS 'Whether the provider call succeeded';


-- ============================================================================
-- Sentiment Cache View
-- ============================================================================
-- View for quick access to latest sentiment per symbol.

CREATE OR REPLACE VIEW latest_sentiment AS
SELECT DISTINCT ON (symbol)
    id,
    timestamp,
    symbol,
    bias,
    confidence,
    social_score,
    news_score,
    overall_score,
    fear_greed,
    key_events,
    market_narratives,
    grok_available,
    gpt_available,
    reasoning
FROM sentiment_outputs
ORDER BY symbol, timestamp DESC;

COMMENT ON VIEW latest_sentiment IS 'Latest sentiment analysis for each symbol';


-- ============================================================================
-- Sentiment Metrics View
-- ============================================================================
-- View for sentiment statistics and provider performance.

CREATE OR REPLACE VIEW sentiment_provider_stats AS
SELECT
    provider,
    model,
    COUNT(*) AS total_calls,
    COUNT(*) FILTER (WHERE success) AS successful_calls,
    ROUND(100.0 * COUNT(*) FILTER (WHERE success) / COUNT(*), 2) AS success_rate_pct,
    ROUND(AVG(latency_ms), 0) AS avg_latency_ms,
    ROUND(AVG(parsed_confidence), 4) AS avg_confidence,
    MIN(timestamp) AS first_call,
    MAX(timestamp) AS last_call
FROM sentiment_provider_responses
GROUP BY provider, model;

COMMENT ON VIEW sentiment_provider_stats IS 'Performance statistics for each sentiment provider/model';


-- ============================================================================
-- Cleanup Function
-- ============================================================================
-- Function to clean up old sentiment data (run periodically).

CREATE OR REPLACE FUNCTION cleanup_old_sentiment_data(retention_days INTEGER DEFAULT 30)
RETURNS TABLE(outputs_deleted INTEGER, responses_deleted INTEGER) AS $$
DECLARE
    outputs_count INTEGER;
    responses_count INTEGER;
    cutoff_date TIMESTAMPTZ;
BEGIN
    cutoff_date := NOW() - (retention_days || ' days')::INTERVAL;

    -- Delete old provider responses first (foreign key constraint)
    DELETE FROM sentiment_provider_responses
    WHERE timestamp < cutoff_date;
    GET DIAGNOSTICS responses_count = ROW_COUNT;

    -- Delete old sentiment outputs
    DELETE FROM sentiment_outputs
    WHERE timestamp < cutoff_date;
    GET DIAGNOSTICS outputs_count = ROW_COUNT;

    RETURN QUERY SELECT outputs_count, responses_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_sentiment_data IS 'Removes sentiment data older than specified retention period';


-- ============================================================================
-- Grant Permissions
-- ============================================================================
-- Grant permissions if running as superuser

DO $$
BEGIN
    -- Grant permissions to trading user if it exists
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE ON sentiment_outputs TO trading;
        GRANT SELECT, INSERT, UPDATE, DELETE ON sentiment_provider_responses TO trading;
        GRANT SELECT ON latest_sentiment TO trading;
        GRANT SELECT ON sentiment_provider_stats TO trading;
        GRANT EXECUTE ON FUNCTION cleanup_old_sentiment_data TO trading;
    END IF;
END $$;


-- ============================================================================
-- Migration Complete
-- ============================================================================
-- Phase 7: Sentiment Analysis tables created successfully.
--
-- Tables:
--   - sentiment_outputs: Main sentiment aggregation storage
--   - sentiment_provider_responses: Individual provider responses
--
-- Views:
--   - latest_sentiment: Quick access to latest sentiment per symbol
--   - sentiment_provider_stats: Provider performance metrics
--
-- Functions:
--   - cleanup_old_sentiment_data(days): Remove old data
-- ============================================================================
