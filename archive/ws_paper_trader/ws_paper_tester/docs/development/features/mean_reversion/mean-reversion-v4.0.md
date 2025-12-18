# Mean Reversion Strategy v4.0.0 - Deep Review Optimization

**Release Date:** 2025-12-14
**Previous Version:** 3.0.0
**Status:** Paper Testing Ready

---

## Overview

Version 4.0.0 of the Mean Reversion strategy implements all recommendations from the deep review v4.0 analysis. This release focuses on research-backed parameter optimization, including disabling trailing stops by default (research shows fixed TP is better for mean reversion), extending position decay timing for crypto markets, adding trend confirmation periods to reduce false positives, and introducing XRP/BTC correlation monitoring.

## Changes from v3.0.0

### REC-001: Trailing Stops Reconfigured

**Problem:** Research suggests trailing stops are suboptimal for mean reversion strategies. Fixed take profit targets align better with mean reversion's reversion-to-mean assumption.

**Solution:** Disabled trailing stops by default with wider parameters when enabled.

**Configuration Changes:**
```python
# Before (v3.0.0)
'use_trailing_stop': True,
'trailing_activation_pct': 0.3,
'trailing_distance_pct': 0.2,

# After (v4.0.0)
'use_trailing_stop': False,        # Disabled by default
'trailing_activation_pct': 0.4,    # Higher activation threshold
'trailing_distance_pct': 0.3,      # Wider trail distance
```

**Rationale:**
- Mean reversion assumes price returns to mean - fixed TP matches this expectation
- Trailing stops designed for trend-following may exit prematurely during MR consolidation
- If enabled, wider parameters (0.4%/0.3%) reduce premature exits

### REC-002: Extended Position Decay Timing

**Problem:** Original 3-minute decay start was too aggressive for crypto markets where mean reversion can take multiple candles to complete.

**Solution:** Extended decay timing with gentler multipliers.

**Configuration Changes:**
```python
# Before (v3.0.0)
'decay_start_minutes': 3.0,
'decay_interval_minutes': 1.0,
'decay_multipliers': [1.0, 0.75, 0.5, 0.25],

# After (v4.0.0)
'decay_start_minutes': 15.0,       # 5x longer before decay
'decay_interval_minutes': 5.0,      # 5x longer intervals
'decay_multipliers': [1.0, 0.85, 0.7, 0.5],  # Gentler reduction
```

**New Decay Schedule:**
| Position Age | TP Multiplier | XRP/USDT TP |
|-------------|---------------|-------------|
| 0-15 min | 100% | 0.50% |
| 15-20 min | 85% | 0.425% |
| 20-25 min | 70% | 0.35% |
| 25+ min | 50% | 0.25% |

**Rationale:**
- Crypto volatility requires more time for mean reversion completion
- 5-minute candles need multiple periods to show reversion
- Gentler multipliers preserve more profit potential

### REC-003: Trend Confirmation Period

**Problem:** Single-tick trend detection caused false positives in choppy/ranging markets, rejecting valid mean reversion signals.

**Solution:** Added confirmation period requiring N consecutive trending evaluations before rejection.

**New Configuration:**
```python
'trend_confirmation_periods': 3,   # Require 3 consecutive trending evals
```

**Behavior:**
1. Track consecutive trending evaluations per symbol
2. Increment counter when slope exceeds threshold
3. Reset counter when slope below threshold
4. Only reject signals when count >= trend_confirmation_periods

**Updated Function Signature:**
```python
def _is_trending(
    candles: List,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str
) -> Tuple[bool, float, int]:
    """
    Returns:
        - is_confirmed_trending: Only True after N consecutive trending periods
        - slope_pct: Current slope percentage
        - consecutive_count: Number of consecutive trending evaluations
    """
```

**New State Tracking:**
```python
state['trend_confirmation_counts'] = {
    'XRP/USDT': 2,   # 2 consecutive trending evals
    'BTC/USDT': 0,   # Not trending
    'XRP/BTC': 1,    # 1 consecutive trending eval
}
```

### REC-005: XRP/BTC Correlation Monitoring

**Problem:** XRP correlation with BTC declining (research shows ~24.86% over 90 days), affecting ratio trading mean reversion timing.

**Solution:** Added rolling Pearson correlation coefficient calculation for XRP/BTC analysis.

**New Configuration:**
```python
'use_correlation_monitoring': True,   # Enable correlation tracking
'correlation_lookback': 50,           # Candles for calculation
'correlation_warn_threshold': 0.5,    # Log warning below this
```

**New Functions:**
```python
def _calculate_correlation(
    xrp_candles: List,
    btc_candles: List,
    lookback: int = 50
) -> Optional[float]:
    """Calculate rolling Pearson correlation between XRP and BTC returns."""

def _get_xrp_btc_correlation(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[float]:
    """Get correlation and update state tracking."""
```

**Correlation Tracking:**
- Uses 5-minute candle close prices for stability
- Calculates returns-based Pearson correlation
- Maintains bounded history (last 100 values)
- Logs to indicators for analysis

**Use Cases:**
- Monitor correlation trends over time
- Potential future use: adaptive deviation threshold based on correlation
- Research foundation for correlation regime detection

## New Indicators (v4.0.0)

| Indicator | Type | Description |
|-----------|------|-------------|
| trend_confirmation_count | int | Consecutive trending evaluations |
| trend_confirmation_required | int | Required confirmations (config) |
| xrp_btc_correlation | float | Rolling correlation (-1 to +1), XRP/BTC only |

## Configuration Reference

### New Parameters (v4.0.0)

```python
# Trend Confirmation - REC-003
'trend_confirmation_periods': 3,

# Correlation Monitoring - REC-005
'use_correlation_monitoring': True,
'correlation_lookback': 50,
'correlation_warn_threshold': 0.5,
```

### Modified Parameters (v4.0.0)

```python
# Trailing Stops - REC-001 (modified defaults)
'use_trailing_stop': False,       # Was True
'trailing_activation_pct': 0.4,   # Was 0.3
'trailing_distance_pct': 0.3,     # Was 0.2

# Position Decay - REC-002 (modified values)
'decay_start_minutes': 15.0,      # Was 3.0
'decay_interval_minutes': 5.0,    # Was 1.0
'decay_multipliers': [1.0, 0.85, 0.7, 0.5],  # Was [1.0, 0.75, 0.5, 0.25]
```

### Complete CONFIG (v4.0.0)

Total parameters: **42** (was 39 in v3.0.0)

## Strategy Development Guide Compliance

| Requirement | v3.0.0 | v4.0.0 |
|-------------|--------|--------|
| `STRATEGY_NAME` | PASS | PASS |
| `STRATEGY_VERSION` | `"3.0.0"` | `"4.0.0"` |
| `SYMBOLS` | 3 | 3 |
| `CONFIG` | 39 params | 42 params |
| `generate_signal()` | PASS | PASS |
| `on_start()` | PASS | PASS (extended logging) |
| `on_fill()` | PASS | PASS |
| `on_stop()` | PASS | PASS |
| Per-pair tracking | PASS | PASS |
| Config validation | PASS | PASS |
| Indicator logging | PASS | PASS (extended) |
| Rejection tracking | PASS | PASS |
| Trend filter | PASS | PASS (with confirmation) |
| Trailing stops | PASS | PASS (disabled default) |
| Position decay | PASS | PASS (extended timing) |
| Correlation monitoring | - | PASS |

**Compliance Score:** 100%

## Test Coverage

New tests added in `tests/test_strategies.py`:

| Test | Description |
|------|-------------|
| `test_trend_slope_calculation` | Linear regression slope accuracy |
| `test_trend_confirmation_period` | Consecutive trending evaluation tracking |
| `test_correlation_calculation` | Pearson correlation for pos/neg correlation |
| `test_position_decay_timing_v40` | Extended decay timing verification |
| `test_trailing_stop_v40_parameters` | Updated trailing stop parameters |
| `test_trailing_stop_disabled_by_default` | Default disabled verification |
| `test_config_updated_v40_parameters` | All v4.0 config changes |
| `test_version_is_4_0_0` | Version string verification |
| `test_indicators_include_v40_fields` | New indicator fields present |
| `test_on_start_logs_v40_params` | Startup logging verification |

**Total Tests:** 153 passing (10 new for v4.0.0)

## Related Files

### Modified
- `ws_paper_tester/strategies/mean_reversion.py` - v4.0.0 implementation (+213 lines)
- `ws_paper_tester/tests/test_strategies.py` - v4.0.0 tests (+431 lines)

### Created
- `ws_paper_tester/docs/development/features/mean_reversion/mean-reversion-v4.0.md` - This document

### Review Documents
- `ws_paper_tester/docs/development/review/mean_reversion/mean-reversion-deep-review-v4.0.md` - Review that drove this release

## Version History

- **4.0.0** (2025-12-14): Optimization per v4.0 deep review
  - REC-001: Trailing stops disabled by default (research: fixed TP better)
  - REC-002: Extended position decay timing (15 min start, 5 min intervals)
  - REC-003: Added trend confirmation period (reduce false positives)
  - REC-005: Added XRP/BTC correlation monitoring
  - Gentler decay multipliers: [1.0, 0.85, 0.7, 0.5]
  - Wider trailing parameters if enabled: 0.4%/0.3%
- **3.0.0** (2025-12-14): Major enhancement per v3.1 review
  - REC-001: Added XRP/BTC ratio trading pair
  - REC-002: Fixed hardcoded max_losses in on_fill
  - REC-003: Added wider stop-loss option (XRP/BTC 0.8%)
  - REC-004: Added optional trend filter
  - REC-006: Added trailing stops
  - REC-007: Added position decay
- **2.0.0** (2025-12-14): Major refactor per v1.0 review
- **1.0.1** (2025-12-13): Fixed RSI edge case (LOW-007)
- **1.0.0**: Initial implementation

## Future Enhancements (Deferred)

| Feature | Priority | Notes |
|---------|----------|-------|
| Adaptive Correlation Threshold | LOW | Adjust deviation based on correlation |
| ATR-Based Dynamic Stops | LOW | Use volatility for adaptive stops |
| Session Time Awareness | LOW | Adjust behavior by trading session |
| Correlation Regime Detection | LOW | Different behavior for high/low correlation |

---

**Document Version:** 1.0
**Last Updated:** 2025-12-14
