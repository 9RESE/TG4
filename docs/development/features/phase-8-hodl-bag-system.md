# Phase 8: Hodl Bag System

**Status**: COMPLETE
**Completion Date**: 2025-12-20
**Version**: v0.6.2
**Test Coverage**: 118 tests (55 manager, 39 API, 24 integration)

---

## Overview

Phase 8 implements the Hodl Bag System - an automated profit allocation mechanism that converts 10% of realized trading profits into long-term holdings. These "hodl bags" are completely separate from active trading capital and are never sold by the system.

## Components

### 1. HodlBagManager

**Location**: `triplegain/src/execution/hodl_bag.py`
**Lines**: 600+

Core manager for hodl bag operations.

**Features**:
- Profit allocation calculation (10% of realized profits)
- Asset split (33.33% USDT, 33.33% XRP, 33.33% BTC)
- Per-asset purchase threshold tracking
- Accumulation execution (market orders)
- Paper trading simulation
- Daily accumulation limits

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `process_trade_profit()` | Main entry point - processes profitable trade |
| `execute_accumulation()` | Executes pending purchase when threshold reached |
| `force_accumulation()` | Manual override for below-threshold execution |
| `get_hodl_state()` | Returns current balances and P&L |
| `get_pending()` | Returns pending accumulation amounts |
| `calculate_metrics()` | Computes performance metrics |
| `create_daily_snapshot()` | Creates daily value snapshots |
| `_record_slippage()` | Tracks slippage events (L5) |
| `_wait_for_fill_with_slippage()` | Waits for order fill with slippage validation |

### 2. API Routes

**Location**: `triplegain/src/api/routes_hodl.py`

7 REST endpoints for hodl bag monitoring and control.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/hodl/status` | GET | Current hodl bag states with P&L |
| `/api/v1/hodl/pending` | GET | Pending accumulations per asset |
| `/api/v1/hodl/thresholds` | GET | Per-asset purchase thresholds |
| `/api/v1/hodl/history` | GET | Transaction history |
| `/api/v1/hodl/metrics` | GET | Performance metrics |
| `/api/v1/hodl/force-accumulation` | POST | Force pending purchase (ADMIN) |
| `/api/v1/hodl/snapshots` | GET | Historical value snapshots |

### 3. Database Schema

**Location**: `migrations/009_hodl_bags.sql`

Four dedicated tables for hodl bag tracking.

| Table | Purpose |
|-------|---------|
| `hodl_bags` | Current balance and cost basis per asset |
| `hodl_transactions` | All accumulations/withdrawals for audit |
| `hodl_pending` | Pending amounts waiting for threshold |
| `hodl_bag_snapshots` | Time-series value tracking (hypertable) |

**Views**:
- `latest_hodl_bags` - Current status with unrealized P&L
- `hodl_pending_totals` - Total pending per asset
- `hodl_performance_summary` - Complete performance metrics

### 4. Configuration

**Location**: `config/hodl.yaml`

```yaml
hodl_bags:
  enabled: true
  allocation_pct: 10           # 10% of realized profits

  split:
    usdt_pct: 33.34            # 1/3 to USDT (stable reserve)
    xrp_pct: 33.33             # 1/3 to XRP
    btc_pct: 33.33             # 1/3 to BTC

  min_accumulation:            # Per-asset purchase thresholds
    usdt: 1                    # $1 (no purchase needed)
    xrp: 25                    # $25 minimum for XRP
    btc: 15                    # $15 minimum for BTC

  execution:
    order_type: market
    max_retries: 3             # Retry failed purchases
    retry_delay_seconds: 30
    max_slippage_pct: 0.5      # L5: Slippage alert threshold

  limits:
    max_single_accumulation_usd: 1000
    daily_accumulation_limit_usd: 5000
    min_profit_to_allocate_usd: 1.0
```

---

## Integration Points

### Position Tracker Integration

```python
# triplegain/src/execution/position_tracker.py

class PositionTracker:
    def __init__(self, ..., hodl_manager=None):
        self.hodl_manager = hodl_manager

    async def close_position(self, ...):
        # ... close position logic ...

        # Phase 8: Notify hodl bag manager
        if self.hodl_manager and position.realized_pnl > 0:
            await self.hodl_manager.process_trade_profit(
                trade_id=position.id,
                profit_usd=position.realized_pnl,
                source_symbol=position.symbol,
            )
```

### Portfolio Rebalance Exclusion

The `PortfolioRebalanceAgent` already excludes hodl balances from allocation calculations via `_get_hodl_bags()` which queries the `hodl_bags` table directly.

---

## Flow Diagram

```
Profitable Trade Closes ($100 profit)
            │
            ▼
    PositionTracker
            │
            ▼
    HodlBagManager.process_trade_profit()
            │
            ▼
    Calculate Allocation: $100 × 10% = $10.00
            │
            ▼
    Split Across Assets:
    ├── USDT: $3.34 (33.34%)
    ├── XRP:  $3.33 (33.33%)
    └── BTC:  $3.33 (33.33%)
            │
            ▼
    Record as Pending
            │
            ▼
    Check Thresholds:
    ├── USDT $3.34 >= $1.00  → Execute immediately
    ├── XRP  $3.33 < $25.00  → Keep pending
    └── BTC  $3.33 < $15.00  → Keep pending
            │
            ▼
    After ~8 Profitable Trades:
    ├── XRP pending reaches $25 → Market buy XRP
    └── BTC pending reaches $15 → Market buy BTC
```

---

## Data Classes

### HodlBagState

```python
@dataclass
class HodlBagState:
    asset: str                              # BTC, XRP, USDT
    balance: Decimal                        # Current asset balance
    cost_basis_usd: Decimal                 # Total USD invested
    current_value_usd: Optional[Decimal]    # Current USD value
    unrealized_pnl_usd: Optional[Decimal]   # Value - cost basis
    unrealized_pnl_pct: Optional[Decimal]   # Percentage P&L
    pending_usd: Decimal                    # Pending accumulation
    first_accumulation: Optional[datetime]  # First purchase date
    last_accumulation: Optional[datetime]   # Most recent purchase
    accumulation_count: int                 # Number of purchases
```

### HodlAllocation

```python
@dataclass
class HodlAllocation:
    trade_id: str                    # Source trade ID
    profit_usd: Decimal              # Original profit
    total_allocation_usd: Decimal    # 10% of profit
    usdt_amount_usd: Decimal         # USDT portion
    xrp_amount_usd: Decimal          # XRP portion
    btc_amount_usd: Decimal          # BTC portion
    timestamp: datetime              # Allocation time
```

### HodlTransaction

```python
@dataclass
class HodlTransaction:
    id: str
    timestamp: datetime
    asset: str
    transaction_type: TransactionType  # accumulation, withdrawal, adjustment
    amount: Decimal                    # Asset amount
    price_usd: Decimal                 # Price at execution
    value_usd: Decimal                 # USD value
    source_trade_id: Optional[str]     # Originating trade
    order_id: Optional[str]            # Exchange order ID
    fee_usd: Decimal                   # Trading fee
    is_paper: bool                     # Paper trading flag
```

---

## Safety Features

### Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| Max Single Accumulation | $1,000 | Prevents large single allocations |
| Daily Accumulation Limit | $5,000 | Prevents runaway accumulation |
| Minimum Profit | $1.00 | Avoids micro-allocations |
| Max Slippage | 0.5% | Alerts on excessive slippage (L5) |

### Slippage Protection (L5)

The system tracks slippage on all hodl purchases:

- **Configuration**: `max_slippage_pct` in `hodl.yaml` (default 0.5%)
- **Tracking**: Every purchase records expected vs actual price
- **Alerting**: Warnings logged when slippage exceeds threshold
- **Statistics**: Available via `get_stats()` API response

```python
# Slippage statistics in get_stats() response
{
    "slippage": {
        "max_slippage_pct": 0.5,      # Configured threshold
        "total_events": 10,            # Total tracked purchases
        "max_observed_pct": 0.25,      # Worst slippage seen
        "warnings": 0                  # Threshold breach count
    }
}
```

### Paper Trading

- Full simulation with fallback prices
- Separate `is_paper` flag in all records
- Same API and behavior as live mode
- No real orders placed in paper mode
- Slippage tracked as 0% (simulated instant fill)

---

## Testing

### Unit Tests

**Location**: `triplegain/tests/unit/execution/test_hodl_bag.py` (55 tests)

| Category | Tests |
|----------|-------|
| HodlThresholds | 5 tests (defaults, custom, get, to_dict) |
| HodlBagState | 3 tests (creation, P&L, serialization) |
| HodlAllocation | 2 tests (creation, serialization) |
| HodlTransaction | 2 tests (creation, serialization) |
| HodlBagManagerInit | 5 tests (config loading, splits, limits) |
| ProfitAllocation | 4 tests (basic, large, small, fractional) |
| ProcessTradeProfit | 7 tests (basic, disabled, zero, negative, cap) |
| AccumulationExecution | 6 tests (USDT, XRP, BTC, threshold, force) |
| DailyLimits | 2 tests (enforcement, reset) |
| StateAndMetrics | 5 tests (state, pending, metrics, stats) |
| PriceSource | 2 tests (fallback, custom) |
| Integration | 2 tests (full flow, threshold triggers) |
| SlippageProtection | 10 tests (L5: config, tracking, warnings, stats) |

### API Tests

**Location**: `triplegain/tests/unit/api/test_routes_hodl.py` (39 tests)

| Category | Tests |
|----------|-------|
| HodlStatus | 4 tests |
| HodlThresholds | 1 test |
| RouterCreation | 3 tests |
| ForceAccumulation | 2 tests |
| Metrics | 2 tests |
| TransactionHistory | 2 tests |
| Snapshots | 2 tests |
| RetryLogic | 3 tests |
| ThreadSafety | 2 tests |
| AdminOperations | 3 tests |
| EdgeCases | 3 tests |
| ErrorResponses | 2 tests (L6) |
| EmptyPendingCases | 2 tests (L6) |
| ZeroBalanceCases | 1 test (L6) |
| ConcurrentRequests | 3 tests (L6) |
| ForceAccumulationEdgeCases | 3 tests (L6) |
| SlippageStatistics | 1 test (L5) |

### Integration Tests

**Location**: `triplegain/tests/integration/test_hodl_integration.py` (24 tests)

| Category | Tests |
|----------|-------|
| ProfitFlowIntegration | 4 tests (allocation, threshold execution, force, daily limit) |
| CoordinatorIntegration | 2 tests (lifecycle, stats display) |
| PositionTrackerIntegration | 3 tests (profit allocation, loss handling, minimum) |
| MessageBusIntegration | 2 tests (allocation events, execution events) |
| StateSerialization | 2 tests (state, metrics) |
| ConcurrentOperations | 2 tests (profit processing, state access) |
| RetryLogicIntegration | 2 tests (config applied, paper mode) |
| SnapshotIntegration | 2 tests (creation, retrieval) |
| PriceCacheIntegration | 1 test (concurrent access) |
| ConfigurationIntegration | 3 tests (allocation %, thresholds, split) |
| DisabledMode | 1 test (no allocation when disabled) |

---

## Related Documentation

- [ADR-014: Hodl Bag System Architecture](../../architecture/09-decisions/ADR-014-hodl-bag-system.md)
- [Phase 8 Implementation Plan](../TripleGain-implementation-plan/08-phase-8-hodl-bag-system.md)
- [Phase 8 Deep Review v2](../reviews/phase-8/deep-review-v2-2025-12-20.md)
- [CHANGELOG v0.6.2](../../../CHANGELOG.md)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.6.0 | 2025-12-20 | Initial implementation |
| v0.6.1 | 2025-12-20 | Coordinator integration, retry logic, daily snapshots |
| v0.6.2 | 2025-12-20 | Integration tests (M4), slippage protection (L5), API tests (L6) |
