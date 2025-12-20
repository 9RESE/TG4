# Phase 8: Hodl Bag System

**Phase Status**: Ready to Start
**Dependencies**: Phase 3 (Execution Manager, Position Tracker), Phase 6 (Paper Trading)
**Deliverable**: Automated profit allocation system for long-term BTC/XRP accumulation

---

## Overview

The Hodl Bag System automatically allocates a percentage of trading profits to long-term "hodl bag" positions in BTC and XRP. These positions are **excluded from rebalancing and trading**, representing a separate wealth accumulation strategy built on trading success.

### Why This Phase Matters

| Benefit | Description |
|---------|-------------|
| Wealth Building | Converts trading profits into long-term holdings |
| Compounding | Hodl bags grow with both accumulation and appreciation |
| Risk Separation | Isolates long-term holdings from trading volatility |
| Tax Efficiency | Long-term holdings may qualify for lower tax rates |
| Psychological Edge | Tangible growth visible beyond trading P&L |

### Core Concept

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HODL BAG PHILOSOPHY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Trading Profits ──────→ 10% ──────→ Hodl Bags ──────→ Never Sold          │
│                           │                                                  │
│                           ▼                                                  │
│                    ┌─────────────┐                                          │
│                    │  50% BTC    │  Long-term growth asset                  │
│                    │  50% XRP    │  Long-term growth asset                  │
│                    └─────────────┘                                          │
│                                                                             │
│  Rules:                                                                     │
│  • Only profitable trades contribute                                        │
│  • Never included in rebalancing                                           │
│  • Never sold by trading system                                            │
│  • Manual withdrawal only with explicit user action                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.1 Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HODL BAG ACCUMULATION FLOW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PROFIT REALIZED (Trade closes with gain)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Trade Result:                                                        │   │
│  │   Symbol: BTC/USDT                                                   │   │
│  │   Realized P&L: +$85.00                                             │   │
│  │                                                                      │   │
│  │ Hodl Allocation: 10% of profit                                      │   │
│  │   To Hodl: $8.50                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. ALLOCATION CALCULATION                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Default Split: 50% to BTC hodl, 50% to XRP hodl                     │   │
│  │                                                                      │   │
│  │ This example ($8.50 from BTC trade):                                │   │
│  │   BTC hodl: +$4.25 worth                                            │   │
│  │   XRP hodl: +$4.25 worth                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. EXECUTION (Batched when threshold reached)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ If accumulated hodl amount >= minimum threshold ($10):              │   │
│  │   → Execute purchase                                                 │   │
│  │   → Transfer to hodl balance (excluded from trading)                │   │
│  │   → Update hodl_bags table                                          │   │
│  │                                                                      │   │
│  │ Else:                                                                │   │
│  │   → Add to pending hodl accumulation                                │   │
│  │   → Execute when threshold reached                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  4. HODL BAG STATE (Example after 1 year)                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ BTC Hodl Bag:                                                        │   │
│  │   Balance: 0.0155 BTC                                               │   │
│  │   Value: $698.25                                                    │   │
│  │   Cost Basis: $612.40                                               │   │
│  │   Unrealized Gain: +$85.85 (+14.0%)                                │   │
│  │   First Accumulation: 2025-01-15                                    │   │
│  │   Last Accumulation: 2025-12-18                                     │   │
│  │                                                                      │   │
│  │ XRP Hodl Bag:                                                        │   │
│  │   Balance: 850 XRP                                                  │   │
│  │   Value: $510.00                                                    │   │
│  │   Cost Basis: $442.00                                               │   │
│  │   Unrealized Gain: +$68.00 (+15.4%)                                │   │
│  │   First Accumulation: 2025-01-15                                    │   │
│  │   Last Accumulation: 2025-12-18                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Integration Points

| Integration | Direction | Description |
|-------------|-----------|-------------|
| Position Tracker | Trigger | Notifies on trade close with profit |
| Order Manager | Execute | Places accumulation buy orders |
| Portfolio Rebalance | Exclude | Hodl balances excluded from allocation |
| Database | Persist | Stores all hodl transactions |
| Dashboard (Phase 10) | Display | Shows hodl bag status and history |

---

## 8.2 Data Structures

### Database Schema

```sql
-- Hodl bag holdings (one row per asset)
CREATE TABLE hodl_bags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset VARCHAR(10) NOT NULL,  -- BTC, XRP
    balance DECIMAL(20, 10) NOT NULL DEFAULT 0,
    cost_basis_usd DECIMAL(20, 2) NOT NULL DEFAULT 0,
    first_accumulation TIMESTAMPTZ,
    last_accumulation TIMESTAMPTZ,
    last_valuation_usd DECIMAL(20, 2),
    last_valuation_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(asset)
);

-- Initialize with zero balances
INSERT INTO hodl_bags (asset, balance, cost_basis_usd)
VALUES ('BTC', 0, 0), ('XRP', 0, 0)
ON CONFLICT (asset) DO NOTHING;

-- Hodl bag transactions (all accumulations/withdrawals)
CREATE TABLE hodl_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset VARCHAR(10) NOT NULL,
    transaction_type VARCHAR(20) NOT NULL CHECK (
        transaction_type IN ('accumulation', 'withdrawal', 'adjustment')
    ),
    amount DECIMAL(20, 10) NOT NULL,  -- Asset amount
    price_usd DECIMAL(20, 10) NOT NULL,  -- Price at execution
    value_usd DECIMAL(20, 2) NOT NULL,  -- USD value
    source_trade_id UUID,  -- Reference to trade_executions (null for withdrawals)
    order_id VARCHAR(50),  -- Exchange order ID
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hodl_transactions_asset
    ON hodl_transactions (asset, timestamp DESC);

CREATE INDEX idx_hodl_transactions_source
    ON hodl_transactions (source_trade_id) WHERE source_trade_id IS NOT NULL;

-- Pending hodl accumulation (not yet executed)
CREATE TABLE hodl_pending (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset VARCHAR(10) NOT NULL,
    amount_usd DECIMAL(20, 2) NOT NULL,
    source_trade_id UUID NOT NULL,
    source_profit_usd DECIMAL(20, 2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    executed_at TIMESTAMPTZ,
    execution_transaction_id UUID REFERENCES hodl_transactions(id)
);

CREATE INDEX idx_hodl_pending_asset_unexecuted
    ON hodl_pending (asset, created_at)
    WHERE executed_at IS NULL;

-- Daily hodl bag snapshots for tracking value over time
CREATE TABLE hodl_bag_snapshots (
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

-- Enable hypertable for time-series queries
SELECT create_hypertable('hodl_bag_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);
```

### Data Classes

| Class | Purpose | Fields |
|-------|---------|--------|
| `HodlBagState` | Current bag status | asset, balance, cost_basis, current_value, unrealized_pnl, dates |
| `HodlAllocation` | Allocation from profit | btc_amount_usd, xrp_amount_usd, total_amount_usd |
| `HodlTransaction` | Single transaction record | asset, type, amount, price, value, source |
| `HodlPending` | Pending accumulation | asset, amount_usd, source_trade_id |

---

## 8.3 Component Details

### 8.3.1 HodlBagManager

**File**: `triplegain/src/execution/hodl_bag.py`

**Responsibilities**:
- Process trade profits for hodl allocation
- Track pending accumulation amounts
- Execute purchases when threshold reached
- Maintain hodl bag state
- Exclude hodl balances from trading capital

**Key Methods**:

| Method | Purpose | Parameters | Returns |
|--------|---------|------------|---------|
| `process_trade_profit()` | Handle closed profitable trade | trade_id, profit_usd, symbol | Optional[HodlAllocation] |
| `get_hodl_state()` | Get current bag states | - | dict[str, HodlBagState] |
| `get_pending()` | Get pending accumulations | asset (optional) | dict[str, Decimal] |
| `execute_accumulation()` | Execute pending purchase | asset | Optional[Decimal] |
| `force_accumulation()` | Manual trigger | asset | bool |
| `get_transaction_history()` | Get transaction log | asset, limit | list[HodlTransaction] |
| `calculate_metrics()` | Calculate performance | - | dict |

**Internal Methods**:

| Method | Purpose |
|--------|---------|
| `_calculate_allocation()` | Determine BTC/XRP split |
| `_record_pending()` | Store pending to database |
| `_update_hodl_bag()` | Update bag after execution |
| `_mark_pending_executed()` | Mark pending as done |
| `_get_current_prices()` | Fetch current asset prices |
| `_to_kraken_symbol()` | Convert symbol format |
| `_wait_for_fill()` | Wait for order execution |

### 8.3.2 Integration with Position Tracker

**File Modification**: `triplegain/src/execution/position_tracker.py`

Add profit notification:

```python
async def close_position(self, position_id: str, close_price: Decimal) -> ClosedPosition:
    """Close position and notify hodl bag manager of profit."""
    closed = await self._execute_close(position_id, close_price)

    if closed.realized_pnl > 0:
        # Notify hodl bag manager
        await self.hodl_manager.process_trade_profit(
            trade_id=closed.trade_id,
            profit_usd=closed.realized_pnl,
            source_symbol=closed.symbol
        )

    return closed
```

### 8.3.3 Integration with Portfolio Rebalance

**File Modification**: `triplegain/src/agents/portfolio_rebalance.py`

Exclude hodl bags from available balance:

```python
async def check_allocation(self) -> PortfolioAllocation:
    """Check allocation excluding hodl bags."""
    balances = await self.kraken.get_balances()
    hodl_bags = await self.hodl_manager.get_hodl_state()

    # Exclude hodl amounts from available
    available_btc = Decimal(balances.get("XXBT", 0)) - hodl_bags["BTC"].balance
    available_xrp = Decimal(balances.get("XXRP", 0)) - hodl_bags["XRP"].balance

    # Continue with available amounts only
    ...
```

---

## 8.4 Configuration

**File**: `config/hodl.yaml`

```yaml
hodl_bags:
  enabled: true

  # Allocation from profits
  allocation_pct: 10  # 10% of realized profits to hodl bags

  # Minimum accumulation before purchase
  min_accumulation_usd: 10

  # Assets to accumulate
  assets:
    - BTC
    - XRP

  # Split strategy
  split:
    btc_pct: 50
    xrp_pct: 50

  # Execution settings
  execution:
    order_type: market  # Immediate execution for simplicity
    retry_on_failure: true
    max_retries: 3
    retry_delay_seconds: 30

  # Paper trading mode
  paper_trading:
    enabled: true  # Use simulated execution in paper mode

  # Snapshot settings
  snapshots:
    enabled: true
    interval_hours: 24  # Daily snapshots

  # Safety limits
  limits:
    max_single_accumulation_usd: 1000  # Cap single purchase
    daily_accumulation_limit_usd: 5000  # Daily limit
```

---

## 8.5 Execution Modes

### Paper Trading Mode

When paper trading is enabled, accumulations are simulated:

| Aspect | Behavior |
|--------|----------|
| Balance Tracking | Updated in database only |
| Order Execution | Simulated with current price |
| Cost Basis | Calculated from simulated fills |
| Reporting | Full metrics available |

### Live Trading Mode

| Aspect | Behavior |
|--------|----------|
| Balance Tracking | Real exchange balances |
| Order Execution | Actual Kraken orders |
| Cost Basis | Real fill prices |
| Reporting | Real performance |

---

## 8.6 API Endpoints

### New Routes

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| GET | `/api/v1/hodl/status` | Current hodl bag states | dict[asset, HodlBagState] |
| GET | `/api/v1/hodl/pending` | Pending accumulations | dict[asset, Decimal] |
| GET | `/api/v1/hodl/history` | Transaction history | list[HodlTransaction] |
| GET | `/api/v1/hodl/metrics` | Performance metrics | HodlMetrics |
| POST | `/api/v1/hodl/force-accumulation` | Force pending purchase | ExecutionResult |
| GET | `/api/v1/hodl/snapshots` | Historical snapshots | list[HodlBagSnapshot] |

### Route Implementation

**File**: `triplegain/src/api/routes_hodl.py`

---

## 8.7 Metrics and Reporting

### Tracked Metrics

| Metric | Description | Calculation |
|--------|-------------|-------------|
| Total Accumulated | Sum of all accumulations | SUM(value_usd) from transactions |
| Current Value | Present value of holdings | balance × current_price |
| Unrealized P&L | Gain/loss on holdings | current_value - cost_basis |
| Average Cost | Weighted average purchase price | cost_basis / balance |
| Accumulation Count | Number of purchases | COUNT(transactions) |
| Avg Accumulation | Average purchase size | AVG(value_usd) |
| Time Held | Duration of hodl | first_accumulation to now |

### Dashboard Display Data

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HODL BAG STATUS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BTC HODL BAG                                                               │
│  ───────────────────────────────────────────────────────────────────────── │
│  Balance:         0.0155 BTC                                               │
│  Current Value:   $698.25                                                  │
│  Cost Basis:      $612.40                                                  │
│  Unrealized P&L:  +$85.85 (+14.0%)                                        │
│  Avg Cost:        $39,509.68 /BTC                                          │
│  Accumulations:   45 purchases                                             │
│  First Buy:       2025-01-15                                               │
│  Last Buy:        2025-12-18                                               │
│                                                                             │
│  XRP HODL BAG                                                               │
│  ───────────────────────────────────────────────────────────────────────── │
│  Balance:         850 XRP                                                  │
│  Current Value:   $510.00                                                  │
│  Cost Basis:      $442.00                                                  │
│  Unrealized P&L:  +$68.00 (+15.4%)                                        │
│  Avg Cost:        $0.52 /XRP                                               │
│  Accumulations:   45 purchases                                             │
│  First Buy:       2025-01-15                                               │
│  Last Buy:        2025-12-18                                               │
│                                                                             │
│  PENDING ACCUMULATION                                                       │
│  ───────────────────────────────────────────────────────────────────────── │
│  BTC Pending:     $6.25 (threshold: $10.00)                                │
│  XRP Pending:     $6.25 (threshold: $10.00)                                │
│                                                                             │
│  TOTAL HODL VALUE:  $1,208.25                                              │
│  TOTAL UNREALIZED:  +$153.85 (+14.6%)                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.8 Test Requirements

### Unit Tests

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_profit_allocation` | Correct percentage allocated | 10% of profit |
| `test_split_calculation` | 50/50 BTC/XRP split | Equal amounts |
| `test_threshold_batching` | Below threshold accumulated | Not executed |
| `test_threshold_execution` | At threshold executed | Order placed |
| `test_state_tracking` | Balances updated correctly | Accurate state |
| `test_cost_basis_update` | Cost basis calculated | Weighted average |
| `test_pending_tracking` | Pending amounts tracked | Correct totals |
| `test_pending_clear` | Pending cleared on execution | Zero after |
| `test_zero_profit` | No allocation on loss | No pending created |
| `test_negative_profit` | Loss handling | No action taken |
| `test_paper_mode` | Paper trading simulation | State updated |

### Integration Tests

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_position_close_triggers` | Profit triggers allocation | HodlManager notified |
| `test_portfolio_excludes_hodl` | Rebalancing excludes | Correct available |
| `test_database_persistence` | Data persists | Survives restart |
| `test_api_endpoints` | REST endpoints work | Valid responses |
| `test_snapshot_creation` | Daily snapshots created | Data recorded |

---

## 8.9 Edge Cases

### Handled Scenarios

| Scenario | Behavior |
|----------|----------|
| Exchange API failure during purchase | Retry with backoff, mark pending as failed |
| Price moves significantly during execution | Use limit orders with slippage protection |
| Partial fill | Track partial accumulation, retry remainder |
| Zero balance initialization | Create empty hodl bag records on startup |
| Multiple profitable trades rapidly | Batch accumulations together |
| Insufficient trading balance | Skip accumulation, preserve trading capital |

### Manual Intervention

| Action | Description | API Endpoint |
|--------|-------------|--------------|
| Force Accumulation | Execute pending immediately | POST /api/v1/hodl/force-accumulation |
| Adjust Pending | Modify pending amount | PUT /api/v1/hodl/pending |
| Withdrawal | Manual hodl withdrawal | POST /api/v1/hodl/withdraw (future) |

---

## 8.10 Deliverables Checklist

- [ ] `triplegain/src/execution/hodl_bag.py` - Core manager implementation
- [ ] `triplegain/src/api/routes_hodl.py` - API endpoints
- [ ] `config/hodl.yaml` - Configuration file
- [ ] `migrations/008_hodl_bags.sql` - Database migration
- [ ] `triplegain/tests/unit/execution/test_hodl_bag.py` - Unit tests
- [ ] `triplegain/tests/integration/test_hodl_integration.py` - Integration tests
- [ ] Update `position_tracker.py` with profit notification
- [ ] Update `portfolio_rebalance.py` to exclude hodl
- [ ] Update paper trading to simulate hodl accumulation

---

## 8.11 Acceptance Criteria

### Functional Requirements

| Requirement | Test Method | Acceptance |
|-------------|-------------|------------|
| 10% profit allocation | Unit test | Exact 10% |
| 50/50 BTC/XRP split | Unit test | Equal amounts |
| Threshold batching | Unit test | Executes at $10 |
| Portfolio exclusion | Integration | Hodl excluded |
| Persistent tracking | Integration | Data survives restart |

### Non-Functional Requirements

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| Accumulation latency | < 5s | From profit to pending |
| Execution latency | < 30s | From threshold to fill |
| Data accuracy | 100% | Audit transactions |
| API response time | < 100ms | Endpoint monitoring |

---

## References

- Design: [03-risk-management-rules-engine.md](../TripleGain-master-design/03-risk-management-rules-engine.md)
- Existing: [position_tracker.py](../../../triplegain/src/execution/position_tracker.py)
- Existing: [portfolio_rebalance.py](../../../triplegain/src/agents/portfolio_rebalance.py)
- Phase 3: [03-phase-3-orchestration.md](./03-phase-3-orchestration.md)

---

*Phase 8 Implementation Plan v1.0 - December 2025*
