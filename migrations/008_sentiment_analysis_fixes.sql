-- ============================================================================
-- Phase 7 Fix: Sentiment Analysis Schema Updates
-- ============================================================================
-- Adds missing columns and fixes cleanup function per deep review findings.
--
-- Fixes:
-- 1. Add social_analysis and news_analysis TEXT columns to sentiment_outputs
-- 2. Update cleanup function to use ID-based deletion for consistency
-- 3. Update latest_sentiment view to include new columns
--
-- Run: psql -h localhost -U triplegain -d triplegain -f migrations/008_sentiment_analysis_fixes.sql
-- ============================================================================

-- ============================================================================
-- Add Missing Columns to sentiment_outputs
-- ============================================================================
-- These columns store the detailed analysis text from each provider
-- for trading decision context.

ALTER TABLE sentiment_outputs
    ADD COLUMN IF NOT EXISTS social_analysis TEXT,
    ADD COLUMN IF NOT EXISTS news_analysis TEXT;

COMMENT ON COLUMN sentiment_outputs.social_analysis IS 'Grok provider Twitter/X sentiment analysis reasoning';
COMMENT ON COLUMN sentiment_outputs.news_analysis IS 'GPT provider news sentiment analysis reasoning';


-- ============================================================================
-- Update Latest Sentiment View
-- ============================================================================
-- Include new columns in the view for API access.

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
    social_analysis,
    news_analysis,
    fear_greed,
    key_events,
    market_narratives,
    grok_available,
    gpt_available,
    reasoning
FROM sentiment_outputs
ORDER BY symbol, timestamp DESC;

COMMENT ON VIEW latest_sentiment IS 'Latest sentiment analysis for each symbol (includes provider analysis)';


-- ============================================================================
-- Fix Cleanup Function
-- ============================================================================
-- Original function deleted by timestamp which could cause FK issues.
-- New version deletes child records by parent ID reference.

CREATE OR REPLACE FUNCTION cleanup_old_sentiment_data(retention_days INTEGER DEFAULT 30)
RETURNS TABLE(outputs_deleted INTEGER, responses_deleted INTEGER) AS $$
DECLARE
    outputs_count INTEGER;
    responses_count INTEGER;
    cutoff_date TIMESTAMPTZ;
BEGIN
    cutoff_date := NOW() - (retention_days || ' days')::INTERVAL;

    -- First, identify outputs to delete
    -- Then delete child provider responses by ID reference (not timestamp)
    -- This prevents orphaned records when timestamps differ slightly
    DELETE FROM sentiment_provider_responses
    WHERE sentiment_output_id IN (
        SELECT id FROM sentiment_outputs WHERE timestamp < cutoff_date
    );
    GET DIAGNOSTICS responses_count = ROW_COUNT;

    -- Now delete the parent sentiment outputs
    DELETE FROM sentiment_outputs
    WHERE timestamp < cutoff_date;
    GET DIAGNOSTICS outputs_count = ROW_COUNT;

    RETURN QUERY SELECT outputs_count, responses_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_sentiment_data IS 'Removes sentiment data older than specified retention period (FK-safe)';


-- ============================================================================
-- Grant Permissions
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE ON sentiment_outputs TO trading;
    END IF;
END $$;


-- ============================================================================
-- Migration Complete
-- ============================================================================
-- Phase 7 Fix: Schema updates applied.
--
-- Changes:
--   - Added social_analysis TEXT column to sentiment_outputs
--   - Added news_analysis TEXT column to sentiment_outputs
--   - Updated latest_sentiment view with new columns
--   - Fixed cleanup_old_sentiment_data to use ID-based FK deletion
-- ============================================================================
