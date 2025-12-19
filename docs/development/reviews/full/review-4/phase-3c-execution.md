# Review Phase 3C: Execution Layer

**Status**: Ready for Review
**Estimated Context**: ~2,500 tokens (code) + review
**Priority**: Critical - Real money execution
**Output**: `findings/phase-3c-findings.md`

---

## Files to Review

| File | Lines | Purpose |
|------|-------|---------|
| `triplegain/src/execution/order_manager.py` | ~600 | Order lifecycle management |
| `triplegain/src/execution/position_tracker.py` | ~400 | Position state tracking |

**Total**: ~1,000 lines

---

## Pre-Review: Load Files

```bash
# Read these files before starting review
cat triplegain/src/execution/order_manager.py
cat triplegain/src/execution/position_tracker.py

# Also review config
cat config/execution.yaml
```

---

## Review Checklist

### 1. Order Manager (`order_manager.py`)

#### Order Lifecycle
- [ ] States defined:
  - [ ] PENDING
  - [ ] OPEN
  - [ ] PARTIALLY_FILLED
  - [ ] FILLED
  - [ ] CANCELLED
  - [ ] EXPIRED
  - [ ] ERROR
- [ ] State transitions correct
- [ ] State machine enforced

#### Order Types
- [ ] MARKET orders supported
- [ ] LIMIT orders supported
- [ ] STOP_LOSS orders supported
- [ ] TAKE_PROFIT orders supported

#### Order Creation
- [ ] Symbol conversion to Kraken format
- [ ] Order type mapping correct
- [ ] Size formatting correct (Decimal → string)
- [ ] Price formatting correct
- [ ] Leverage parameter handled

#### Kraken API Integration
- [ ] add_order() called correctly
- [ ] query_orders() for status
- [ ] cancel_order() for cancellation
- [ ] Error response handling
- [ ] Rate limiting respected

#### Order Monitoring
- [ ] Polling interval appropriate (5s)
- [ ] Status updates detected
- [ ] Fill detection correct
- [ ] Partial fills handled
- [ ] Expiration/cancellation detected

#### Contingent Orders
- [ ] Stop loss placed after fill
- [ ] Take profit placed after fill
- [ ] OCO (one-cancels-other) if supported
- [ ] Contingent order failure handling

#### Error Handling
- [ ] API errors handled gracefully
- [ ] Network timeout handling
- [ ] Invalid response handling
- [ ] Order rejection handling
- [ ] Insufficient balance handling

#### Execution Events
- [ ] Events published to message bus
- [ ] Event format correct
- [ ] Order ID included
- [ ] Fill details included

---

### 2. Position Tracker (`position_tracker.py`)

#### Position Data
- [ ] Position dataclass complete:
  - [ ] id
  - [ ] symbol
  - [ ] side (long/short)
  - [ ] size
  - [ ] entry_price
  - [ ] entry_time
  - [ ] leverage
  - [ ] stop_loss_order_id
  - [ ] take_profit_order_id
  - [ ] unrealized_pnl
  - [ ] unrealized_pnl_pct

#### Position Creation
- [ ] Created on order fill
- [ ] Entry price from fill (not order)
- [ ] Size from fill (handles partial)
- [ ] Leverage recorded
- [ ] Stop/TP order IDs linked

#### P&L Calculation
- [ ] Unrealized P&L formula correct:
  ```python
  # Long position
  unrealized_pnl = (current_price - entry_price) * size

  # Short position
  unrealized_pnl = (entry_price - current_price) * size
  ```
- [ ] P&L percentage correct:
  ```python
  pnl_pct = unrealized_pnl / (entry_price * size) * 100
  ```
- [ ] Leverage factored into P&L display
- [ ] Decimal precision maintained

#### Position Closing
- [ ] Triggered by SL/TP fill
- [ ] Manual close supported
- [ ] Realized P&L calculated
- [ ] Position removed from active list
- [ ] Trade execution record updated

#### Position Modification
- [ ] Stop loss modification
- [ ] Take profit modification
- [ ] Old orders cancelled before new
- [ ] Modification logged

#### Exchange Sync
- [ ] Sync with Kraken positions
- [ ] Detect manual changes
- [ ] Handle discrepancies
- [ ] Periodic sync interval

#### Persistence
- [ ] Positions stored in database
- [ ] Recovered on restart
- [ ] Consistency maintained

---

## Critical Questions

1. **Order Race Conditions**: What if order fills during status check?
2. **Partial Fill Handling**: Is position tracking correct for partial fills?
3. **Orphan Orders**: What happens to SL/TP if position closed manually?
4. **Position Limits**: Is max positions (6) enforced?
5. **Exchange Sync**: What if exchange shows different position?
6. **Fee Tracking**: Are trading fees recorded?

---

## Order Execution Safety

### Pre-Execution Checks
- [ ] Risk validation complete
- [ ] Sufficient balance verified
- [ ] Not exceeding position limits
- [ ] Market hours check (if applicable)
- [ ] Circuit breakers clear

### Execution Safety
- [ ] No duplicate orders
- [ ] Order acknowledgment received
- [ ] Transaction ID recorded
- [ ] Execution timestamp recorded

### Post-Execution Verification
- [ ] Order status confirmed
- [ ] Fill price recorded
- [ ] Position created
- [ ] Stop/TP orders placed
- [ ] Portfolio updated

---

## Kraken API Mapping

### Symbol Conversion
| Internal | Kraken |
|----------|--------|
| BTC/USDT | XBTUSDT or XXBTZUSD |
| XRP/USDT | XRPUSDT |
| XRP/BTC | XXRPXXBT |

- [ ] Conversion function correct
- [ ] Edge cases handled
- [ ] Error on unknown symbol

### Order Type Mapping
| Internal | Kraken |
|----------|--------|
| MARKET | market |
| LIMIT | limit |
| STOP_LOSS | stop-loss |
| TAKE_PROFIT | take-profit |

- [ ] Mapping correct
- [ ] Kraken order flags set correctly

### Order Response Handling
```json
{
  "error": [],
  "result": {
    "descr": { "order": "..." },
    "txid": ["ORDER-ID-HERE"]
  }
}
```

- [ ] Error array checked
- [ ] txid extracted correctly
- [ ] Description logged

---

## Position P&L Verification

### Long Position Example
```
Entry: 45,000 USDT
Size: 0.1 BTC
Current: 46,000 USDT
Leverage: 2x

Unrealized P&L = (46,000 - 45,000) * 0.1 = 100 USDT
P&L % = 100 / (45,000 * 0.1) * 100 = 2.22%
With leverage display: 2.22% * 2 = 4.44% on margin
```

### Short Position Example
```
Entry: 45,000 USDT
Size: 0.1 BTC
Current: 44,000 USDT
Leverage: 2x

Unrealized P&L = (45,000 - 44,000) * 0.1 = 100 USDT
P&L % = 100 / (45,000 * 0.1) * 100 = 2.22%
```

- [ ] Both scenarios calculate correctly
- [ ] Negative P&L handled
- [ ] Large positions don't overflow

---

## Error Handling Matrix

| Error | Expected Behavior | Recovery |
|-------|-------------------|----------|
| Insufficient balance | Reject order, log | User notification |
| Invalid symbol | Reject order, log | Raise exception |
| Rate limited | Backoff, retry | Automatic |
| Order rejected by exchange | Log, alert | Manual review |
| Network timeout | Retry with backoff | 3 attempts max |
| Position not found | Log error | Sync with exchange |
| Partial fill timeout | Monitor until complete | Alert if stale |
| SL/TP placement fail | Alert, manual | Critical |

---

## Concurrency Review

- [ ] Order operations are atomic
- [ ] Position updates are atomic
- [ ] No race between fill and close
- [ ] Lock on position modification
- [ ] Thread-safe order tracking

---

## Fee Tracking

- [ ] Trading fees recorded
- [ ] Fee deducted from P&L
- [ ] Fee currency handled (base vs quote)
- [ ] Fee summary available

---

## Test Coverage Check

```bash
pytest --cov=triplegain/src/execution \
       --cov-report=term-missing \
       triplegain/tests/unit/execution/
```

Expected tests:
- [ ] Order creation (all types)
- [ ] Order status monitoring
- [ ] Position creation on fill
- [ ] P&L calculation (long/short)
- [ ] Position closing
- [ ] Contingent order placement
- [ ] Error handling
- [ ] Exchange sync

---

## Design Conformance

### Implementation Plan 3.4 (Order Execution Manager)
- [ ] Order states match spec
- [ ] Order flow matches design
- [ ] Position tracking matches spec

### Master Design
- [ ] Integrates with risk engine
- [ ] Publishes execution events
- [ ] Supports portfolio rebalancing trades

---

## Findings Template

```markdown
## Finding: [Title]

**File**: `triplegain/src/execution/filename.py:123`
**Priority**: P0/P1/P2/P3
**Category**: Security/Logic/Performance/Quality

### Description
[What was found]

### Current Code
```python
# current implementation
```

### Recommended Fix
```python
# recommended fix
```

### Financial Impact
[What could go wrong with real money]
```

---

## Review Completion

After completing this phase:

1. [ ] Order manager logic verified
2. [ ] Position tracker logic verified
3. [ ] P&L calculations verified
4. [ ] Error handling verified
5. [ ] Kraken integration verified
6. [ ] Findings documented
7. [ ] Ready for Phase 4

---

*Phase 3C Review Plan v1.0 - EXECUTION CRITICAL*
