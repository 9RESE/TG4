# Recommendations: Momentum Scalping Strategy

**Review Date:** 2025-12-14

---

## Priority Definitions

| Priority | Timeframe | Description |
|----------|-----------|-------------|
| **P0** | Immediate | Block trading until resolved |
| **P1** | 1 week | High-impact improvement |
| **P2** | 1 month | Moderate improvement |
| **P3** | Backlog | Low-priority enhancement |

## Effort Definitions

| Effort | Description |
|--------|-------------|
| **LOW** | Config change or < 1 hour code |
| **MEDIUM** | 1-4 hours, single module |
| **HIGH** | 4+ hours, multiple modules |

---

## Recommendations Summary

| ID | Priority | Effort | Finding | Action |
|----|----------|--------|---------|--------|
| REC-001 | P0 | LOW | XRP/BTC correlation | Pause XRP/BTC trading |
| REC-002 | P1 | LOW | ADX threshold | Raise to 30 for BTC |
| REC-003 | P1 | LOW | RSI period | Backtest with 8-9 |
| REC-004 | P1 | HIGH | Guide v2.0 | Create or clarify |
| REC-005 | P2 | MEDIUM | Trailing stops | Implement ATR-based |
| REC-006 | P2 | LOW | DST handling | Document behavior |
| REC-007 | P2 | MEDIUM | Trade flow | Add imbalance filter |
| REC-008 | P2 | LOW | Correlation lookback | Increase to 100 |
| REC-009 | P3 | LOW | Momentum exit | Add breakeven option |
| REC-010 | P3 | LOW | Logging | Use structured logging |

---

## Detailed Recommendations

### REC-001: Pause XRP/BTC Trading (P0, LOW)

**Finding:** CRIT-001
**Affected:** XRP/BTC pair
**Current Risk:** HIGH

**Recommendation:**
Pause XRP/BTC trading until correlation stabilizes above 0.60.

**Implementation Options:**

**Option A: Configuration Change (Recommended)**
```python
# In config.py
'correlation_pause_threshold': 0.60,  # Raise from 0.50
```

**Option B: Disable Pair Temporarily**
```python
# In config.py
SYMBOLS = ["XRP/USDT", "BTC/USDT"]  # Remove XRP/BTC
```

**Option C: Reduce Position Size to Minimum**
```python
# In SYMBOL_CONFIGS
'XRP/BTC': {
    'position_size_usd': 5.0,  # Reduce from 15.0
}
```

**Monitoring:**
- Check XRP-BTC correlation daily
- Resume when 7-day average correlation > 0.65
- Log correlation values for trend analysis

**Success Criteria:**
- No XRP/BTC trades executed while correlation < 0.60
- Correlation logged in indicators

---

### REC-002: Raise ADX Threshold for BTC (P1, LOW)

**Finding:** HIGH-001
**Affected:** BTC/USDT pair
**Location:** `config.py:211`

**Recommendation:**
Raise ADX threshold from 25 to 30 for BTC/USDT to avoid entries during strong trends.

**Implementation:**
```python
# In config.py
'adx_strong_trend_threshold': 30,  # Raise from 25
```

**Alternative: Per-Symbol ADX Threshold**
```python
# In SYMBOL_CONFIGS (if implemented)
'BTC/USDT': {
    'adx_threshold': 30,  # BTC-specific
}
```

**Rationale:**
- 2024 market data shows BTC ADX > 30 during major moves
- Current BTC ADX ~24.81 is borderline
- Crypto volatility warrants higher threshold than traditional 25

**Success Criteria:**
- BTC/USDT entries rejected when ADX > 30
- Reduced false entries during BTC rallies

---

### REC-003: Evaluate RSI Period for XRP/USDT (P1, LOW)

**Finding:** HIGH-002
**Affected:** XRP/USDT pair
**Location:** `config.py:238`

**Recommendation:**
Backtest RSI period 8 or 9 for XRP/USDT to reduce noise while maintaining responsiveness.

**Backtest Plan:**
1. Run paper trading with RSI period 7 (current)
2. Run paper trading with RSI period 8
3. Run paper trading with RSI period 9
4. Compare: Signal count, win rate, profit factor

**Implementation (after backtest):**
```python
# In SYMBOL_CONFIGS
'XRP/USDT': {
    'rsi_period': 8,  # If backtest supports
}
```

**Success Criteria:**
- Backtest shows improved win rate with period 8/9
- Signal quality improves without significant signal reduction

---

### REC-004: Create Strategy Development Guide v2.0 (P1, HIGH)

**Finding:** CRIT-002
**Affected:** Documentation

**Recommendation:**
Create Strategy Development Guide v2.0 with the following sections:

**Proposed Section Structure:**
- Section 15: Volatility Regime Classification
- Section 16: Circuit Breaker Protection
- Section 17: Signal Rejection Tracking
- Section 18: Trade Flow Confirmation
- Section 19: Multi-Timeframe Analysis
- Section 20: Correlation Management
- Section 21: Dynamic Parameter Adjustment
- Section 22: Per-Symbol Configuration (SYMBOL_CONFIGS)
- Section 23: Indicator Logging Requirements
- Section 24: Correlation Monitoring

**Alternative:**
Clarify that v1.0 is the current standard and update review scope accordingly.

**Success Criteria:**
- Guide v2.0 published or scope clarified
- All strategies can be reviewed against documented requirements

---

### REC-005: Implement ATR-Based Trailing Stop (P2, MEDIUM)

**Finding:** MED-002
**Affected:** `exits.py`

**Recommendation:**
Implement trailing stop using ATR (already calculated in `indicators.py:407-438`).

**Proposed Logic:**
```python
def check_trailing_stop_exit(state, symbol, current_price, config):
    """Trail stop at entry + (profit / 2) once TP is 50% achieved."""
    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    entry_price = pos_entry['entry_price']
    highest = pos_entry.get('highest_price', entry_price)

    # For long: trail at highest - ATR * multiplier
    atr = state.get('indicators', {}).get('atr')
    if atr and pos_entry['side'] == 'long':
        trail_price = highest - (atr * config.get('trail_atr_mult', 1.5))
        if current_price <= trail_price and highest > entry_price:
            return Signal(action='sell', ...)
```

**Configuration:**
```python
'use_trailing_stop': True,
'trail_atr_mult': 1.5,
'trail_activation_pct': 0.4,  # Activate after 0.4% profit
```

**Success Criteria:**
- Trailing stops preserve profits on extended moves
- Configurable and can be disabled

---

### REC-006: Document DST Handling (P2, LOW)

**Finding:** MED-004
**Affected:** `regimes.py`, documentation

**Recommendation:**
Add documentation explaining DST behavior and how to adjust session boundaries.

**Proposed Documentation:**
```markdown
## Session Boundaries and DST

Session boundaries are defined in UTC. During Daylight Saving Time:
- US markets: Adjust boundaries by 1 hour
- European markets: Adjust boundaries by 1 hour

### Winter (Standard Time)
- US_EUROPE_OVERLAP: 14:00-17:00 UTC

### Summer (DST)
- US_EUROPE_OVERLAP: 13:00-16:00 UTC

Configure via:
session_boundaries:
  overlap_start: 13  # Summer
  overlap_end: 16    # Summer
```

**Success Criteria:**
- DST behavior documented
- Users know how to adjust for DST

---

### REC-007: Add Trade Flow Confirmation (P2, MEDIUM)

**Finding:** HIGH-003
**Affected:** `signal.py`

**Recommendation:**
Add trade imbalance confirmation before entries.

**Proposed Logic:**
```python
# In signal.py, before entry signal
if config.get('use_trade_flow_confirmation', True):
    imbalance = data.get_trade_imbalance(symbol, n_trades=50)
    state['indicators']['trade_imbalance'] = imbalance

    # For long entries, require buy-side imbalance
    if momentum_signal['long_signal'] and imbalance < 0.1:
        track_rejection(state, RejectionReason.TRADE_FLOW_MISALIGNMENT, symbol)
        return None
```

**Configuration:**
```python
'use_trade_flow_confirmation': True,
'trade_imbalance_threshold': 0.1,  # Require >10% buy imbalance for longs
```

**New Rejection Reason:**
```python
TRADE_FLOW_MISALIGNMENT = "trade_flow_misalignment"
```

**Success Criteria:**
- Entry signals confirmed by trade flow direction
- Rejection tracking for trade flow misalignment

---

### REC-008: Increase Correlation Lookback (P2, LOW)

**Finding:** MED-001
**Affected:** `config.py:199`

**Recommendation:**
Increase correlation lookback from 50 to 100 candles (5m) for more stable reading.

**Implementation:**
```python
# In config.py
'correlation_lookback': 100,  # ~8.3 hours instead of ~4 hours
```

**Trade-off:**
- More stable correlation reading
- Slower to detect rapid changes
- May miss flash decoupling events

**Success Criteria:**
- Correlation readings more stable
- Fewer false correlation warnings

---

### REC-009: Add Breakeven Momentum Exit Option (P3, LOW)

**Finding:** MED-003
**Affected:** `exits.py:204-271`

**Recommendation:**
Add configurable option to exit at breakeven on RSI extreme.

**Implementation:**
```python
# In exits.py:check_momentum_exhaustion_exit
if config.get('exit_breakeven_on_momentum_exhaustion', False):
    if pnl_pct >= -0.1:  # Within 0.1% of breakeven
        # Allow exit on RSI extreme
        pass
else:
    if pnl_pct <= 0:
        return None  # Current behavior
```

**Configuration:**
```python
'exit_breakeven_on_momentum_exhaustion': False,  # Off by default
```

**Success Criteria:**
- Optional feature for risk-averse traders
- Default behavior unchanged

---

### REC-010: Use Structured Logging (P3, LOW)

**Finding:** LOW-002
**Affected:** `lifecycle.py:49`

**Recommendation:**
Replace print statements with structured logging.

**Implementation:**
```python
import logging
logger = logging.getLogger('momentum_scalping')

# Instead of:
print(f"[momentum_scalping] Config warning: {error}")

# Use:
logger.warning("Config warning", extra={'warning': error})
```

**Success Criteria:**
- All logs in consistent JSON format
- Compatible with log aggregation tools

---

## Implementation Roadmap

### Phase 1: Immediate (This Week)
- REC-001: Pause XRP/BTC (config change)
- REC-002: Raise ADX threshold (config change)

### Phase 2: Short-term (1 Month)
- REC-003: Backtest RSI periods
- REC-005: Implement trailing stops
- REC-006: Document DST handling
- REC-008: Increase correlation lookback

### Phase 3: Medium-term (Quarter)
- REC-004: Create Guide v2.0
- REC-007: Add trade flow confirmation

### Phase 4: Backlog
- REC-009: Breakeven momentum exit
- REC-010: Structured logging

---

## Monitoring After Implementation

| Recommendation | Metric to Monitor |
|----------------|-------------------|
| REC-001 | XRP/BTC trade count (should be 0) |
| REC-002 | BTC/USDT ADX rejection rate |
| REC-003 | XRP/USDT win rate, signal count |
| REC-005 | Trailing stop exit count, avg profit |
| REC-007 | Trade flow rejection rate |
| REC-008 | Correlation warning frequency |

---

*Next: [Research References](./research-references.md)*
