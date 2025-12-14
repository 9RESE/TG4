# Market Making Strategy v1.3.0

**Version:** 1.3.0
**Date:** 2025-12-13
**Status:** Production Ready
**Pairs:** XRP/USDT, BTC/USDT, XRP/BTC

## Overview

The Market Making strategy v1.3.0 provides liquidity by placing orders on both sides of the spread, capturing the bid-ask spread while managing inventory risk. This version includes major improvements based on a deep strategy review addressing 8 identified issues.

## Key Features

### 1. Volatility-Adjusted Spreads (MM-002)

Dynamic threshold adjustment based on recent price volatility:

```python
volatility = _calculate_volatility(candles, lookback=20)
vol_multiplier = min(volatility / base_vol, 1.5)
effective_threshold = imbalance_threshold * vol_multiplier
```

**Benefits:**
- Reduces over-trading in volatile markets
- Prevents noise-driven signals
- Adapts to changing market conditions

### 2. Signal Cooldown (MM-003)

Time-based cooldown between signals to prevent rapid-fire trading:

| Symbol | Cooldown |
|--------|----------|
| XRP/USDT | 5 seconds |
| BTC/USDT | 3 seconds |
| XRP/BTC | 10 seconds |

### 3. Trade Flow Confirmation (MM-007)

Validates orderbook imbalance with trade tape before signaling:

```python
def is_trade_flow_aligned(direction):
    if direction == 'buy':
        return trade_flow > trade_flow_threshold
    elif direction == 'sell':
        return trade_flow < -trade_flow_threshold
```

### 4. Improved Risk-Reward Ratios (MM-004)

| Symbol | Take Profit | Stop Loss | R:R Ratio |
|--------|-------------|-----------|-----------|
| XRP/USDT | 0.4% | 0.5% | 0.8:1 |
| BTC/USDT | 0.35% | 0.35% | 1:1 |
| XRP/BTC | 0.3% | 0.4% | 0.75:1 |

### 5. XRP/BTC Size Unit Fix (MM-001)

Converts XRP size to USD equivalent for Signal compatibility:

```python
xrp_usdt_price = data.prices.get('XRP/USDT', 2.35)
base_size = base_size_xrp * xrp_usdt_price
```

## Configuration

### Global Config

```python
CONFIG = {
    # Spread parameters
    'min_spread_pct': 0.1,
    'position_size_usd': 20,
    'max_inventory': 100,
    'inventory_skew': 0.5,
    'imbalance_threshold': 0.1,

    # Risk management
    'take_profit_pct': 0.4,
    'stop_loss_pct': 0.5,

    # Signal control
    'cooldown_seconds': 5.0,

    # Volatility adjustment
    'base_volatility_pct': 0.5,
    'volatility_lookback': 20,
    'volatility_threshold_mult': 1.5,

    # Trade flow confirmation
    'use_trade_flow': True,
    'trade_flow_threshold': 0.15,
}
```

### Per-Symbol Config

```python
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'min_spread_pct': 0.05,
        'position_size_usd': 20,
        'cooldown_seconds': 5.0,
    },
    'BTC/USDT': {
        'min_spread_pct': 0.03,
        'position_size_usd': 50,
        'take_profit_pct': 0.35,
        'stop_loss_pct': 0.35,
        'cooldown_seconds': 3.0,
    },
    'XRP/BTC': {
        'position_size_xrp': 25,
        'max_inventory_xrp': 150,
        'cooldown_seconds': 10.0,
    },
}
```

## Enhanced Indicators

The strategy now logs comprehensive indicators including volatility metrics:

```python
state['indicators'] = {
    'symbol': 'XRP/USDT',
    'spread_pct': 0.08,
    'effective_min_spread': 0.075,
    'best_bid': 2.3450,
    'best_ask': 2.3470,
    'mid': 2.3460,
    'inventory': 45.5,
    'max_inventory': 100,
    'imbalance': 0.25,
    'effective_threshold': 0.12,
    'volatility_pct': 0.65,
    'vol_multiplier': 1.3,
    'trade_flow': 0.18,
    'trade_flow_aligned': True,
}
```

## Signal Flow

```
generate_signal()
├── Global cooldown check
└── For each symbol:
    ├── Get orderbook and price
    ├── Calculate volatility
    ├── Apply dynamic thresholds
    ├── Get trade flow imbalance
    ├── Check inventory
    ├── Evaluate conditions:
    │   ├── Long + sell pressure → Sell to reduce
    │   ├── Short + buy pressure → Buy to cover
    │   ├── Buy opportunity (imbalance > threshold)
    │   │   └── Check trade flow alignment
    │   └── Sell opportunity (imbalance < -threshold)
    │       └── Check trade flow alignment
    └── Return Signal with entry-based SL/TP
```

## Issues Fixed

| ID | Issue | Resolution |
|----|-------|------------|
| MM-001 | XRP/BTC size units | Convert XRP to USD |
| MM-002 | Static spreads | Volatility adjustment |
| MM-003 | No cooldown | Per-symbol cooldown |
| MM-004 | Poor R:R | Improved ratios |
| MM-005 | on_fill units | Use value field |
| MM-006 | SL/TP on mid | Use entry price |
| MM-007 | No confirmation | Trade flow check |
| MM-008 | Missing metrics | Enhanced logging |

## Testing

All 126 tests pass including strategy-specific tests:

```bash
pytest tests/test_strategies.py -v
# 9 passed
```

## References

- [Deep Strategy Review](../market-making-strategy-review-v1.2.md)
- [Strategy Development Guide](../strategy-development-guide.md)
- [hftbacktest - Market Making with OBI](https://hftbacktest.readthedocs.io)
- [DWF Labs - Market Making Strategies](https://www.dwf-labs.com)
