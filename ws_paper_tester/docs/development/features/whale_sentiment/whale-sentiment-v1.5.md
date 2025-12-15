# Whale Sentiment Strategy v1.5.0

**Implementation Date:** December 15, 2025
**Status:** Deep Review v5.0 Implementation Complete
**Research References:** See deep-review-v5.0.md Section 7

## Overview

The Whale Sentiment Strategy combines institutional activity detection (via volume spike analysis) with price deviation sentiment indicators to identify contrarian trading opportunities. The strategy operates on the principle that extreme market fear or greed, particularly when coupled with large-holder activity, often precedes price reversals.

**Key Changes in v1.5.0:**
- Legacy RSI validation code removed from validation.py (REC-034)
- Extended fear thresholds reduced for practical utility (REC-035)
- Deprecation timeline added to detect_rsi_divergence stub (REC-036)
- Extreme zone state persistence across restarts (REC-037)

## Version 1.5.0 Changes (Deep Review v5.0)

| REC ID | Priority | Change | Rationale |
|--------|----------|--------|-----------|
| REC-034 | MEDIUM | Removed legacy RSI validation code | Clean code - RSI removed in v1.3.0 |
| REC-035 | MEDIUM | Reduced extended fear thresholds (72h/168h) | Original 168h/336h rarely triggered |
| REC-036 | LOW | Added deprecation timeline to divergence stub | Clear v2.0.0 removal schedule |
| REC-037 | LOW | Extreme zone state persistence | Accurate tracking across restarts |

### Compliance Score Maintained

| Version | Compliance | Notes |
|---------|------------|-------|
| v1.4.0 | 100% | All 9 requirements met |
| v1.5.0 | 100% | All 9 requirements maintained |

## REC-034: RSI Validation Code Removal

### What Changed

Removed legacy RSI validation code from `validation.py`:

**Before (v1.4.0):**
```python
# validate_config() validated RSI parameters:
# - rsi_period, rsi_extreme_fear, rsi_fear, rsi_greed, rsi_extreme_greed
# validate_symbol_configs() validated per-symbol RSI thresholds
# validate_config_overrides() accepted RSI keys in known_keys
```

**After (v1.5.0):**
- RSI validation completely removed from `validate_config()`
- RSI threshold validation removed from `validate_symbol_configs()`
- RSI keys removed from `known_keys` in `validate_config_overrides()`
- Candle buffer calculation no longer references `rsi_period`

### Rationale

RSI was completely removed from the strategy in v1.3.0 (REC-021) based on academic evidence of RSI ineffectiveness in crypto markets. The validation code remained as technical debt. This cleanup aligns with clean code principles.

## REC-035: Extended Fear Threshold Reduction

### What Changed

Reduced extended fear detection thresholds for practical utility:

| Parameter | v1.4.0 | v1.5.0 | Rationale |
|-----------|--------|--------|-----------|
| `extended_fear_threshold_hours` | 168h (7 days) | 72h (3 days) | Size reduction triggers more frequently |
| `extended_fear_pause_hours` | 336h (14 days) | 168h (7 days) | Entry pause triggers more frequently |

### Files Changed

- `config.py:362-366`: Updated default values with REC-035 documentation
- `regimes.py:157-159`: Updated default values in docstring comments

### Rationale

The original 7-day/14-day thresholds were chosen conservatively but analysis showed:
- 7-day sustained extreme fear is uncommon even in crypto
- 14-day extreme fear would require unprecedented market conditions
- Reducing to 3-day/7-day provides more practical protection

## REC-036: Deprecation Timeline for Divergence Stub

### What Changed

Added explicit deprecation timeline and Python warning to `detect_rsi_divergence()`:

```python
def detect_rsi_divergence(...) -> Dict[str, Any]:
    """
    REC-032: RSI divergence REMOVED (v1.4.0).
    REC-036: Scheduled for removal in v2.0.0 (v1.5.0).

    .. deprecated:: 1.4.0
        RSI-based indicators removed from strategy per REC-021.
        This function is a stub retained for backwards compatibility.
        **Will be removed in v2.0.0** - Update any code that calls this function.
    ...
    """
    # REC-036: Issue deprecation warning
    warnings.warn(
        "detect_rsi_divergence is deprecated and will be removed in v2.0.0. "
        "RSI-based indicators are no longer used in this strategy (REC-021).",
        DeprecationWarning,
        stacklevel=2
    )
    return {'bullish_divergence': False, 'bearish_divergence': False, 'divergence_type': 'none'}
```

### Rationale

Provides clear communication to any code that may call this deprecated function:
- Docstring documents removal timeline
- Python DeprecationWarning alerts at runtime
- Allows time for dependent code to update before v2.0.0

## REC-037: Extreme Zone State Persistence

### What Changed

Added state persistence for extreme zone tracking across strategy restarts:

**New Functions in `persistence.py`:**

| Function | Description |
|----------|-------------|
| `get_state_file_path()` | Returns path for state persistence file |
| `save_extreme_zone_state()` | Saves extreme zone tracking to disk |
| `load_extreme_zone_state()` | Loads persisted extreme zone state |
| `delete_extreme_zone_state()` | Cleans up state file when exiting extreme zone |

**Changes to `regimes.py`:**

The `check_extended_fear_period()` function now:
1. Loads persisted state on first call if available
2. Saves state when entering an extreme zone
3. Periodically saves state (every hour) while in extreme zone
4. Deletes persisted state when exiting extreme zone

### State File Format

```json
{
  "saved_at": "2025-12-15T10:30:00+00:00",
  "extreme_zone_start": "2025-12-12T08:15:00+00:00",
  "extreme_zone_type": "EXTREME_FEAR"
}
```

**Location:** `data/candles/whale_sentiment_state.json`

### Validation

- State is validated on load (age check based on `extended_fear_pause_hours * 1.5`)
- Corrupted files are handled gracefully (starts fresh)
- State is cleaned up when exiting extreme zone

### Rationale

Previously, if the strategy restarted while in an extreme sentiment zone, the `extreme_zone_start` would be lost and tracking would reset to 0 hours. This could allow entries that should be blocked by extended fear protection. Persistence ensures accurate duration tracking across restarts.

## Module Structure

```
strategies/whale_sentiment/
    __init__.py          # Public API exports
    config.py            # Metadata, CONFIG, SYMBOL_CONFIGS, enums
    indicators.py        # Volume spike, fear/greed, ATR, composite
    signal.py            # generate_signal, _evaluate_symbol
    regimes.py           # Sentiment + volatility regime classification
    risk.py              # Circuit breaker, position limits, correlation
    exits.py             # Exit signal logic
    lifecycle.py         # on_start, on_fill, on_stop
    validation.py        # Config validation (RSI validation removed)
    persistence.py       # Candle data + extreme zone state persistence
```

## Configuration Changes (v1.5.0)

### Extended Fear Period Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `extended_fear_threshold_hours` | 72 | **REC-035:** 3 days in extreme = reduce size |
| `extended_fear_pause_hours` | 168 | **REC-035:** 7 days in extreme = pause entries |
| `extended_fear_size_reduction` | 0.70 | 30% size reduction when extended |

## Compliance Checklist (v1.5.0)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Section 15: Volatility Regime | Yes | EXTREME regime pauses trading |
| Section 16: Circuit Breaker | Yes | 2 losses, 45 min cooldown |
| Section 17: Signal Rejection Tracking | Yes | 19 rejection reasons |
| Section 18: Trade Flow Confirmation | Yes | Contrarian-aware |
| Section 22: Per-Symbol Configuration | Yes | 3 pairs configured |
| Section 24: Correlation Monitoring | Yes | Real-time blocking |
| R:R Ratio >= 1:1 | Yes | All pairs 2:1 |
| USD-Based Sizing | Yes | All signals |
| Indicator Logging | Yes | All code paths |
| Section 26: Scope Documentation | Yes | Added in v1.4.0 |

**Overall Compliance: 100%** (All requirements maintained)

## Future Improvements

### Deferred:
1. **REC-024:** Backtest-validated confidence weights (requires 6-12 months historical data)

### Next Steps:
2. Remove `detect_rsi_divergence` stub in v2.0.0
3. External whale data integration (Whale Alert API)
4. Social sentiment API integration
5. On-chain metrics integration
6. Adaptive thresholds based on market conditions
7. XRP/BTC re-enablement monitoring (golden cross printed)

---

**Document Version:** 1.5.0
**Author:** Deep Review v5.0 Implementation
**Platform Version:** WebSocket Paper Tester v1.5.0+
**Review Reference:** deep-review-v5.0.md
