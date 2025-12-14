# Market Making Strategy v1.5.0

**Release Date:** 2025-12-13
**Previous Version:** 1.4.0
**Status:** Production Ready

---

## Overview

Version 1.5.0 implements all recommendations from the deep strategy review (`market-making-strategy-review-v1.4.md`). This release focuses on improving profitability, code quality, and robustness.

## New Features

### MM-E03: Fee-Aware Profitability Check

Before entering trades, the strategy now verifies that the expected profit exceeds trading fees.

**Configuration:**
```yaml
fee_rate: 0.001              # 0.1% per trade (0.2% round-trip)
min_profit_after_fees_pct: 0.05  # Minimum 0.05% profit required
use_fee_check: true          # Enable/disable the check
```

**How it works:**
- Calculates expected profit from spread capture (spread / 2)
- Subtracts round-trip fees (fee_rate * 2)
- Only enters trade if net profit >= min_profit_after_fees_pct

**Example:**
- Spread: 0.5%, Fee rate: 0.1%
- Expected capture: 0.25%, Round-trip fees: 0.2%
- Net profit: 0.05% - APPROVED

### MM-E01: Micro-Price Calculation

Implements volume-weighted micro-price for better price discovery.

**Configuration:**
```yaml
use_micro_price: true   # Enable micro-price (default: on)
```

**Formula:**
```
micro_price = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
```

**Benefits:**
- More accurate fair value estimation
- Reduces adverse selection
- Better aligned with actual market sentiment

### MM-E02: Optimal Spread Calculation (A-S Style)

Implements Avellaneda-Stoikov optimal spread formula.

**Configuration:**
```yaml
use_optimal_spread: false  # Enable A-S optimal spread
kappa: 1.5                 # Market liquidity parameter
```

**Formula:**
```
optimal_spread = γ * σ² * T + (2/γ) * ln(1 + γ/κ)
```

Where:
- γ = gamma (risk aversion)
- σ = volatility
- T = time horizon
- κ = kappa (liquidity parameter)

### MM-E04: Time-Based Position Decay

Closes stale positions with reduced take-profit requirements.

**Configuration:**
```yaml
use_position_decay: true       # Enable position decay
max_position_age_seconds: 300  # 5 minutes max age
position_decay_tp_multiplier: 0.5  # 50% of original TP
```

**How it works:**
1. Tracks position entry time
2. If position age > max_position_age_seconds
3. Reduce TP requirement by decay multiplier
4. Exit if current profit >= adjusted TP

**Example:**
- Original TP: 0.5%, Position age: 6 minutes
- Adjusted TP: 0.25%
- If profit >= 0.25%, exit position

## Improvements

### MM-009: R:R Ratios Adjusted to 1:1

Updated risk/reward ratios for consistent win-rate requirements.

| Pair | Old TP | Old SL | New TP | New SL | R:R |
|------|--------|--------|--------|--------|-----|
| XRP/USDT | 0.4% | 0.5% | 0.5% | 0.5% | 1:1 |
| XRP/BTC | 0.3% | 0.4% | 0.4% | 0.4% | 1:1 |
| BTC/USDT | 0.35% | 0.35% | 0.35% | 0.35% | 1:1 |

### MM-010: Code Refactoring

The `_evaluate_symbol` function was refactored from 333 lines into smaller, testable functions:

- `_calculate_effective_thresholds()` - Volatility-adjusted thresholds
- `_check_trailing_stop_exit()` - Trailing stop logic
- `_check_position_decay_exit()` - Position decay logic
- `_build_entry_signal()` - Signal construction

### MM-011: Configurable Fallback Prices

Removed hardcoded fallback prices. Now configurable:

```yaml
fallback_xrp_usdt: 2.50  # Used when XRP/USDT price unavailable
```

## Configuration Reference

### New v1.5.0 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fee_rate` | float | 0.001 | Fee per trade (0.1%) |
| `min_profit_after_fees_pct` | float | 0.05 | Min profit after fees |
| `use_fee_check` | bool | true | Enable fee profitability check |
| `use_micro_price` | bool | true | Use volume-weighted micro-price |
| `use_optimal_spread` | bool | false | Enable A-S optimal spread |
| `kappa` | float | 1.5 | Market liquidity parameter |
| `fallback_xrp_usdt` | float | 2.50 | Fallback XRP/USDT price |
| `use_position_decay` | bool | true | Enable position decay |
| `max_position_age_seconds` | float | 300 | Max position age |
| `position_decay_tp_multiplier` | float | 0.5 | TP reduction for stale positions |

## Indicator Logging

New fields in `state['indicators']`:

```python
{
    'micro_price': 2.3505,           # Volume-weighted micro-price
    'optimal_spread': 0.0823,        # A-S optimal spread (if enabled)
    'is_fee_profitable': True,       # Fee profitability check result
    'expected_profit_pct': 0.0756,   # Expected profit after fees
}
```

## Testing

New test cases in `tests/test_strategies.py`:

- `TestMarketMakingV15Features::test_micro_price_calculation`
- `TestMarketMakingV15Features::test_optimal_spread_calculation`
- `TestMarketMakingV15Features::test_fee_profitability_check`
- `TestMarketMakingV15Features::test_position_decay_check`
- `TestMarketMakingV15Features::test_configurable_fallback_price`
- `TestMarketMakingV15Features::test_build_entry_signal`
- `TestMarketMakingV15Features::test_position_entry_tracking_with_timestamp`
- `TestMarketMakingV15Features::test_indicators_include_v15_fields`
- `TestMarketMakingV15Features::test_rr_ratios_updated`

All 26 tests pass.

## Migration Guide

### From v1.4.0

No breaking changes. New features are enabled by default:
- `use_fee_check: true`
- `use_micro_price: true`
- `use_position_decay: true`

Optional features disabled by default:
- `use_optimal_spread: false`

### Recommended Configuration Changes

For improved profitability, consider:

```yaml
# Enable all v1.5.0 features
use_fee_check: true
use_micro_price: true
use_position_decay: true
use_optimal_spread: true  # For volatile markets

# Adjust fee rate for your exchange
fee_rate: 0.0006  # If you have maker rebates
```

## Performance Impact

| Feature | Latency Impact | Memory Impact |
|---------|----------------|---------------|
| Micro-price | +0.01ms | None |
| Fee check | +0.01ms | None |
| Optimal spread | +0.02ms | None |
| Position decay | +0.01ms | +8 bytes/position |

## References

- [Avellaneda-Stoikov Paper](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf)
- [Hummingbot A-S Implementation](https://hummingbot.org/strategies/avellaneda-market-making/)
- [Strategy Development Guide](../strategy-development-guide.md)
- [Deep Review Document](../market-making-strategy-review-v1.4.md)

---

**Document Version:** 1.0
**Last Updated:** 2025-12-13
