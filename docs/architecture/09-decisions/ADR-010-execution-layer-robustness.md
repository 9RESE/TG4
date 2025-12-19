# ADR-010: Execution Layer Robustness Fixes

## Status
Accepted

## Date
2025-12-19

## Context

During the Phase 3C code review of the execution layer (OrderExecutionManager and PositionTracker), 17 issues were identified ranging from critical bugs to low-priority improvements. The most severe issues could result in:

1. **P0 Critical**: Stop-loss orders placed incorrectly on Kraken, potentially failing to protect positions
2. **P0 Critical**: Market order sizes calculated incorrectly, leading to wrong position sizes
3. **P1 High**: Partial fills not tracked, causing position/order state inconsistencies
4. **P1 High**: No synchronization between local and exchange position state

These issues needed immediate resolution before any production deployment.

## Decision

We implemented 17 fixes across the execution layer:

### Critical Fixes (P0)
- **F01**: Fixed Kraken stop-loss parameter - trigger price now goes in `price` field (not `price2`) for STOP_LOSS orders
- **F02**: Added `_get_current_price()` to fetch market price for proper USD-to-base-currency conversion

### High Priority Fixes (P1)
- **F03**: Added `_handle_partial_fill()` method with incremental position updates
- **F04**: `_handle_order_fill()` now publishes `RISK_ALERTS` on contingent order failures
- **F05**: Added `sync_with_exchange()` to PositionTracker for state reconciliation
- **F06**: Transaction-like error handling in fill processing with proper alerting
- **F07**: Made `enable_trailing_stop_for_position()` async with proper locking

### Medium Priority Fixes (P2)
- **F08**: `modify_position()` accepts optional `order_manager` to update exchange orders
- **F09**: Added separate `_trigger_check_loop()` running every 5s for faster SL/TP detection
- **F10**: Added fee tracking fields (`fee_amount`, `fee_currency`, `total_fees`)
- **F11**: Failed orders now persisted before returning error for audit trail
- **F12**: Added `cancel_orphan_orders()` for cleanup when positions close
- **F13**: Fixed inconsistent case sensitivity in error message checking

### Low Priority Fixes (P3)
- **F14**: Added order ID references to Position dataclass (`stop_loss_order_id`, `take_profit_order_id`)
- **F15**: Fixed race condition in `get_order()` with nested lock acquisition
- **F16**: Implemented OCO (one-cancels-other) with `handle_oco_fill()`
- **F17**: Added 32 new tests, improved coverage from 47% to 63%

## Consequences

### Positive
- Stop-loss orders will now execute correctly on Kraken
- Market orders will have correct position sizes
- Partial fills properly tracked and positions created
- Local state stays synchronized with exchange
- Better audit trail for failed orders
- Faster SL/TP trigger detection (5s vs 60s)
- OCO behavior prevents double execution

### Negative
- Increased complexity in order/position lifecycle
- Additional API calls for price lookups on market orders
- Nested locks in `get_order()` could impact performance under high contention

### Neutral
- Test coverage still below 85% target (at 63%)
- Further testing recommended for exchange communication paths

## Alternatives Considered

1. **Polling-only position sync**: Rejected because it doesn't detect unknown positions
2. **Single fast loop for everything**: Rejected because snapshots don't need 5s frequency
3. **Native Kraken OCO orders**: Not available for all order types, so implemented locally

## Related
- ADR-009: Agent Robustness Fixes
- Phase 3C Review: `docs/development/reviews/full/review-4/findings/phase-3c-findings.md`
- Implementation Plan: `docs/development/TripleGain-implementation-plan/`
