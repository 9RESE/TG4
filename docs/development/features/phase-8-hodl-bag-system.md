# Phase 8: Hodl Bag System

**Status**: COMPLETE
**Completion Date**: 2025-12-20
**Test Coverage**: 56 tests (45 manager, 11 API)

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

### Paper Trading

- Full simulation with fallback prices
- Separate `is_paper` flag in all records
- Same API and behavior as live mode
- No real orders placed in paper mode

---

## Testing

### Unit Tests

**Location**: `triplegain/tests/unit/execution/test_hodl_bag.py`

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

### API Tests

**Location**: `triplegain/tests/unit/api/test_routes_hodl.py`

| Category | Tests |
|----------|-------|
| RouterCreation | 3 tests |
| ForceAccumulation | 2 tests |
| Metrics | 2 tests |
| TransactionHistory | 2 tests |
| HodlThresholds | 1 test |
| HodlStatus | 1 test |

---

## Related Documentation

- [ADR-014: Hodl Bag System Architecture](../../architecture/09-decisions/ADR-014-hodl-bag-system.md)
- [Phase 8 Implementation Plan](../TripleGain-implementation-plan/08-phase-8-hodl-bag-system.md)
- [CHANGELOG v0.6.0](../../../CHANGELOG.md)
