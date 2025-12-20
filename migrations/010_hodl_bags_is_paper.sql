-- ============================================================================
-- Phase 8: Add is_paper Column to hodl_bags Table
-- ============================================================================
-- M5 Fix: Adds is_paper column to hodl_bags table to distinguish paper trading
-- hodl bags from live trading hodl bags.
--
-- Run: psql -h localhost -U triplegain -d triplegain -f migrations/010_hodl_bags_is_paper.sql
-- ============================================================================

-- Add is_paper column to hodl_bags table
ALTER TABLE hodl_bags ADD COLUMN IF NOT EXISTS is_paper BOOLEAN DEFAULT FALSE;

-- Create index for paper trading queries
CREATE INDEX IF NOT EXISTS idx_hodl_bags_is_paper
    ON hodl_bags (is_paper);

-- Update existing rows to mark as non-paper (live) by default
UPDATE hodl_bags SET is_paper = FALSE WHERE is_paper IS NULL;

-- Add comment explaining the column
COMMENT ON COLUMN hodl_bags.is_paper IS 'True for paper trading simulations, False for live trading';

-- Update the views to include is_paper filter

-- Latest hodl bags with paper filter
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
    is_paper,
    updated_at
FROM hodl_bags
WHERE balance > 0
ORDER BY is_paper, asset;

COMMENT ON VIEW latest_hodl_bags IS 'Current hodl bag status with unrealized P&L, includes paper mode flag';

-- Hodl performance summary with paper filter
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
    h.last_accumulation,
    h.is_paper
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
ORDER BY h.is_paper, h.asset;

COMMENT ON VIEW hodl_performance_summary IS 'Complete hodl bag performance with pending and historical data, includes paper mode flag';

-- ============================================================================
-- Migration Complete
-- ============================================================================
-- M5 Fix: Added is_paper column to hodl_bags table.
--
-- Changes:
--   - Added is_paper BOOLEAN column to hodl_bags
--   - Created index for paper trading queries
--   - Updated views to include is_paper column
-- ============================================================================
