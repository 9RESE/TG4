# Mean Reversion Strategy v4.2.0 - Correlation Risk Management

**Release Date:** 2025-12-14
**Previous Version:** 4.1.0
**Status:** Paper Testing Ready

---

## Overview

Version 4.2.0 of the Mean Reversion strategy implements correlation risk management recommendations from deep review v6.0. The primary focus is addressing the dramatic XRP-BTC correlation decline (from ~80% to ~40%) by tightening warning thresholds and adding automatic trading pause functionality for XRP/BTC when correlation drops to critically low levels.

## Key Finding: XRP-BTC Correlation Collapse

The v6.0 review identified a critical market condition change:

| Metric | Previous | Current (Dec 2025) |
|--------|----------|-------------------|
| XRP-BTC Correlation | ~80% | ~40% |
| Decoupling Status | Partial | Confirmed |
| Impact | Normal ratio trading | Extended reversion periods |

**Implication:** Traditional ratio trading assumptions are weakened. XRP is now trading on "its own fundamentals" with significant institutional adoption (Bitwise ETF).

## Changes from v4.1.0

### REC-001: Adjusted Correlation Thresholds

**Problem:** With XRP-BTC correlation at ~40% (down from ~80%), the original 0.5 warning threshold was regularly being breached, making warnings less meaningful.

**Solution:** Tightened warning threshold and added automatic pause mechanism.

**Configuration Changes:**
```python
# Before (v4.1.0)
'correlation_warn_threshold': 0.5,   # Log warning below this

# After (v4.2.0)
'correlation_warn_threshold': 0.4,   # Tightened from 0.5
'correlation_pause_threshold': 0.25, # NEW: Pause XRP/BTC trading below this
'correlation_pause_enabled': True,   # NEW: Enable automatic pause feature
```

**Rationale:**
- 0.4 warn threshold aligns with current ~40% correlation level
- 0.25 pause threshold triggers if correlation drops further during market stress
- Automatic pause prevents trading in conditions unsuitable for ratio mean reversion

### New Function: Correlation Pause Check

Added `_should_pause_for_low_correlation()` function:

```python
def _should_pause_for_low_correlation(
    symbol: str,
    state: Dict[str, Any],
    config: Dict[str, Any]
) -> bool:
    """
    Check if XRP/BTC trading should pause due to low correlation.

    REC-001 (v4.2.0): With XRP-BTC correlation at ~40% (down from ~80%),
    pause ratio trading when correlation drops below critical threshold.
    """
    # Only applies to XRP/BTC ratio trading
    if symbol != 'XRP/BTC':
        return False

    if not config.get('correlation_pause_enabled', True):
        return False

    return state.get('correlation_below_pause_threshold', False)
```

### New Rejection Reason: LOW_CORRELATION

Added new rejection tracking for correlation-based pauses:

```python
class RejectionReason(Enum):
    # ... existing reasons ...
    LOW_CORRELATION = "low_correlation"  # REC-001 (v4.2.0) - XRP/BTC correlation pause
```

**Tracking:** Low correlation rejections are tracked per-symbol in `rejection_counts` and `rejection_counts_by_symbol`.

## New Indicators (v4.2.0)

| Indicator | Type | Description |
|-----------|------|-------------|
| correlation_warn_threshold | float | Warning threshold (0.4) |
| correlation_pause_threshold | float | Pause threshold (0.25) |
| correlation_pause_enabled | bool | Whether pause feature is active |

## Configuration Reference

### New Parameters (v4.2.0)

```python
# Correlation Pause - REC-001
'correlation_pause_threshold': 0.25,  # Pause XRP/BTC below this
'correlation_pause_enabled': True,    # Enable automatic pause
```

### Modified Parameters (v4.2.0)

```python
# Correlation Warning - REC-001 (modified value)
'correlation_warn_threshold': 0.4,    # Was 0.5
```

### Complete Correlation Config (v4.2.0)

```python
# XRP/BTC Correlation Monitoring
'use_correlation_monitoring': True,   # Enable tracking
'correlation_lookback': 50,           # Candles for calculation
'correlation_warn_threshold': 0.4,    # Warning level (tightened)
'correlation_pause_threshold': 0.25,  # Pause trading level (NEW)
'correlation_pause_enabled': True,    # Enable pause feature (NEW)
```

**Total CONFIG Parameters:** 44 (was 42 in v4.1.0)

## Strategy Development Guide v2.0 Compliance

| Requirement | v4.1.0 | v4.2.0 |
|-------------|--------|--------|
| Section 24: Correlation Monitoring | PASS | PASS (enhanced) |
| Low correlation pause | - | PASS |
| Dynamic threshold adjustment | - | PASS (warn threshold) |
| Rejection reason tracking | 11 | 12 (+LOW_CORRELATION) |

**Compliance Score:** 96% (unchanged, already high)

## Deferred Recommendations (Future Work)

The v6.0 review identified additional recommendations deferred to future versions:

| ID | Recommendation | Priority | Effort | Reason for Deferral |
|----|----------------|----------|--------|---------------------|
| REC-003 | Add ADX trend strength filter | LOW | MEDIUM | Existing trend filter effective |
| REC-004 | Add band walk detection | LOW | MEDIUM | Existing BB + RSI sufficient |
| REC-006 | Investigate asymmetric MR | LOW | HIGH | Research agenda item |
| REC-007 | Backtest XRP vs BTC MR | MEDIUM | HIGH | Data collection in progress |

**Note:** REC-002 (Consider pausing BTC/USDT) is an operational recommendation. BTC/USDT already has reduced position size ($25) per v4.1.0 REC-001.

## Startup Logging (v4.2.0)

New parameters logged on strategy start:

```
[mean_reversion] v4.2.0 started
[mean_reversion] Symbols: ['XRP/USDT', 'BTC/USDT', 'XRP/BTC']
[mean_reversion] Features: VolatilityRegimes=True, CircuitBreaker=True, ...
[mean_reversion] v4.1 Params: DecayStart=15.0min, TrendConfirm=3 periods, ...
[mean_reversion] v4.2 Params: CorrelationWarn=0.4, CorrelationPause=0.25, PauseEnabled=True
```

## Related Files

### Modified
- `ws_paper_tester/strategies/mean_reversion.py` - v4.2.0 implementation

### Created
- `ws_paper_tester/docs/development/features/mean_reversion/mean-reversion-v4.2.md` - This document

### Review Documents
- `ws_paper_tester/docs/development/review/mean_reversion/mean-reversion-deep-review-v6.0.md` - Review that drove this release

## Version History

- **4.2.0** (2025-12-14): Correlation risk management per v6.0 deep review
  - REC-001: Tightened correlation_warn_threshold (0.5 -> 0.4)
  - REC-001: Added correlation_pause_threshold (0.25) for automatic XRP/BTC pause
  - REC-001: Added correlation_pause_enabled config flag
  - New rejection reason: LOW_CORRELATION
  - Deferred: REC-003 ADX filter, REC-004 band walk detection (LOW priority)
- **4.1.0** (2025-12-14): Risk adjustments per v5.0 review
  - Reduced BTC/USDT position size ($50 -> $25)
  - Added fee profitability check
  - Added SCOPE AND LIMITATIONS documentation
- **4.0.0** (2025-12-14): Optimization per v4.0 deep review
- **3.0.0** (2025-12-14): Major enhancement per v3.1 review
- **2.0.0** (2025-12-14): Major refactor per v1.0 review
- **1.0.1** (2025-12-13): Fixed RSI edge case (LOW-007)
- **1.0.0**: Initial implementation

## Future Enhancements (Deferred from v6.0 Review)

| Feature | Priority | Notes |
|---------|----------|-------|
| ADX Trend Filter | LOW | 14-period ADX, reject when > 25 AND trending |
| Band Walk Detection | LOW | Track consecutive BB touches, pause after 4+ |
| Asymmetric Mean Reversion | LOW | Research: negative moves revert faster |
| Correlation-Based Deviation | LOW | Wider threshold when correlation low |

---

**Document Version:** 1.0
**Last Updated:** 2025-12-14
