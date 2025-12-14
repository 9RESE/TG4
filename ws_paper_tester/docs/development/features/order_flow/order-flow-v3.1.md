# Order Flow Strategy v3.1.0 - Asymmetric Thresholds and Bug Fixes

**Release Date:** 2025-12-13
**Previous Version:** 3.0.0
**Status:** Production Ready

---

## Overview

Version 3.1.0 of the Order Flow strategy implements fixes and enhancements from the v3.0.0 deep review, including asymmetric buy/sell thresholds, improved symbol configuration validation, and critical bug fixes for undefined variables.

## Changes from v3.0.0

### Bug Fixes

#### FIX-001: Undefined `base_threshold` Variable
**Location:** `strategies/order_flow.py:641`

**Problem:** When asymmetric thresholds were enabled (default), the `base_threshold` variable was only defined in the `else` block but referenced in indicator logging.

**Fix:** Removed the intermediate `base_threshold` variable. Now uses `base_buy_threshold` and `base_sell_threshold` directly.

**Before:**
```python
if use_asymmetric:
    base_buy_threshold = ...
    base_sell_threshold = ...
else:
    base_threshold = ...  # Only defined here
    base_buy_threshold = base_threshold
    base_sell_threshold = base_threshold

# Later: base_threshold referenced but undefined when use_asymmetric=True
```

**After:**
```python
if use_asymmetric:
    base_buy_threshold = ...
    base_sell_threshold = ...
else:
    base_buy_threshold = ...
    base_sell_threshold = base_buy_threshold  # Directly assigned
```

#### FIX-002: VWAP Reversion Using Wrong Threshold
**Location:** `strategies/order_flow.py:836`

**Problem:** The VWAP mean reversion sell check used `effective_threshold` (undefined) instead of `effective_sell_threshold`.

**Fix:** Changed to use `effective_sell_threshold` for sell signals.

### Enhancements

#### HIGH-NEW-001: Symbol Config Validation (Already in v3.0.0)
The `_validate_config()` function now validates SYMBOL_CONFIGS entries:
- Positive value checks for per-symbol settings
- Imbalance threshold bounds (0.1-0.8)
- Per-symbol R:R ratio warnings

#### HIGH-NEW-002: Asymmetric Buy/Sell Thresholds (Already in v3.0.0)
Research shows crypto selling pressure has larger price impact than buying pressure of equal magnitude. The strategy now uses separate thresholds:

```python
CONFIG = {
    'use_asymmetric_thresholds': True,
    'buy_imbalance_threshold': 0.30,   # Higher for buys
    'sell_imbalance_threshold': 0.25,  # Lower for sells (more impactful)
}

SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'buy_imbalance_threshold': 0.30,
        'sell_imbalance_threshold': 0.25,
    },
    'BTC/USDT': {
        'buy_imbalance_threshold': 0.25,  # Lower for high liquidity
        'sell_imbalance_threshold': 0.20,
    },
}
```

### Indicator Logging Updates

Updated indicator logging to show separate buy/sell thresholds:

**Old:**
```python
'base_threshold': 0.30,
'effective_threshold': 0.30,
```

**New:**
```python
'base_buy_threshold': 0.30,
'base_sell_threshold': 0.25,
'effective_buy_threshold': 0.30,
'effective_sell_threshold': 0.25,
```

## Configuration Reference

### Default CONFIG (v3.1.0)

```python
CONFIG = {
    # Asymmetric thresholds (HIGH-NEW-002)
    'imbalance_threshold': 0.3,        # Fallback when asymmetric disabled
    'buy_imbalance_threshold': 0.30,   # Buy signal threshold
    'sell_imbalance_threshold': 0.25,  # Sell signal threshold (lower)
    'use_asymmetric_thresholds': True,

    # Order flow parameters
    'volume_spike_mult': 2.0,
    'lookback_trades': 50,

    # Position sizing
    'position_size_usd': 25.0,
    'max_position_usd': 100.0,
    'min_trade_size_usd': 5.0,

    # Risk management (2:1 R:R)
    'take_profit_pct': 1.0,
    'stop_loss_pct': 0.5,

    # Cooldowns
    'cooldown_trades': 10,
    'cooldown_seconds': 5.0,

    # Volatility adjustment
    'base_volatility_pct': 0.5,
    'volatility_lookback': 20,
    'volatility_threshold_mult': 1.5,

    # Trade flow confirmation
    'use_trade_flow_confirmation': True,
    'trade_flow_threshold': 0.15,

    # Fee-aware profitability
    'fee_rate': 0.001,
    'min_profit_after_fees_pct': 0.05,
    'use_fee_check': True,

    # Advanced features
    'use_micro_price': True,
    'use_position_decay': True,
    'max_position_age_seconds': 300,
    'position_decay_tp_multiplier': 0.5,
    'use_trailing_stop': False,
    'trailing_stop_activation': 0.3,
    'trailing_stop_distance': 0.2,
    'use_circuit_breaker': True,
    'max_consecutive_losses': 3,
    'circuit_breaker_minutes': 15,
}
```

### Per-Symbol Configurations

| Parameter | XRP/USDT | BTC/USDT |
|-----------|----------|----------|
| `buy_imbalance_threshold` | 0.30 | 0.25 |
| `sell_imbalance_threshold` | 0.25 | 0.20 |
| `position_size_usd` | $25 | $50 |
| `volume_spike_mult` | 2.0 | 1.8 |
| `take_profit_pct` | 1.0% | 0.8% |
| `stop_loss_pct` | 0.5% | 0.4% |

## Related Files

### Modified
- `ws_paper_tester/strategies/order_flow.py` - v3.1.0 with fixes
- `ws_paper_tester/ws_tester/types.py` - CRIT-NEW-001 fix (trade array slicing)

### Review Documents
- `docs/development/review/order_flow/order-flow-strategy-review-v3.0.md` - Review that identified these issues
- `docs/development/review/order_flow/order-flow-strategy-review-v2.2.md` - Previous review

## Strategy Development Guide Compliance

| Requirement | Status |
|-------------|--------|
| `STRATEGY_NAME` | PASS (`"order_flow"`) |
| `STRATEGY_VERSION` | PASS (`"3.1.0"`) |
| `SYMBOLS` | PASS (`["XRP/USDT", "BTC/USDT"]`) |
| `CONFIG` | PASS |
| `generate_signal()` | PASS |
| `on_start()` | PASS |
| `on_fill()` | PASS |
| `on_stop()` | PASS |
| Per-pair PnL tracking | PASS |
| Config validation | PASS |
| Indicator logging | PASS |

## Version History

- **3.1.0** (2025-12-13): Bug fixes and asymmetric threshold implementation
- **3.0.0** (2025-12-13): Major refactor with fee-aware trading, circuit breaker, trailing stops
- **2.2.0**: Removed XRP/BTC (moved to dedicated strategies)
- **2.1.0**: Added XRP/BTC pair support
- **2.0.0**: Major refactor with volatility adjustment
- **1.0.1**: Position awareness fix
- **1.0.0**: Initial implementation

---

**Document Version:** 1.0
**Last Updated:** 2025-12-13
