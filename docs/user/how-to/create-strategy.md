# How to Create a Strategy for WebSocket Paper Tester

This guide explains how to create a new trading strategy for the WebSocket Paper Tester.

## Quick Start

Create a new Python file in `ws_paper_tester/strategies/`:

```python
# strategies/my_strategy.py

from typing import Optional
from ws_tester.types import DataSnapshot, Signal

# Required: Strategy metadata
STRATEGY_NAME = "my_strategy"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["XRP/USD"]

# Optional: Configuration with defaults
CONFIG = {
    'threshold': 0.5,
    'position_size_usd': 20,
}

def generate_signal(data: DataSnapshot, config: dict, state: dict) -> Optional[Signal]:
    """Generate trading signal from market data."""
    symbol = "XRP/USD"
    price = data.prices.get(symbol)

    if not price:
        return None

    # Your strategy logic here
    if should_buy(data, config, state):
        return Signal(
            action='buy',
            symbol=symbol,
            size=config['position_size_usd'],
            price=price,
            reason="My buy reason"
        )

    return None
```

That's it! The strategy will be automatically discovered and loaded when you run `ws_tester.py`.

## Strategy Interface

### Required Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `STRATEGY_NAME` | `str` | Unique identifier for the strategy |
| `generate_signal` | `function` | Main signal generation function |

### Optional Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `STRATEGY_VERSION` | `str` | "0.0.0" | Version string |
| `SYMBOLS` | `List[str]` | `[]` | Symbols this strategy trades |
| `CONFIG` | `dict` | `{}` | Default configuration |
| `on_fill` | `function` | None | Called when order is filled |
| `on_start` | `function` | None | Called when strategy starts |
| `on_stop` | `function` | None | Called when strategy stops |

## The generate_signal Function

```python
def generate_signal(
    data: DataSnapshot,    # Immutable market data
    config: dict,          # Strategy configuration
    state: dict            # Mutable strategy state
) -> Optional[Signal]:
    """
    Returns:
        Signal if action needed, None for no action
    """
```

### DataSnapshot Contents

```python
data.timestamp          # Current time
data.prices             # {'XRP/USD': 2.35, 'BTC/USD': 104500}
data.candles_1m         # {'XRP/USD': (Candle, ...)}
data.candles_5m         # 5-minute candles
data.orderbooks         # {'XRP/USD': OrderbookSnapshot}
data.trades             # {'XRP/USD': (Trade, ...)}

# Computed properties
data.spreads            # Bid-ask spreads
data.mids               # Mid prices
data.get_vwap('XRP/USD', 50)        # VWAP from last 50 trades
data.get_trade_imbalance('XRP/USD') # Buy/sell volume imbalance
```

### Creating a Signal

```python
Signal(
    action='buy',           # 'buy', 'sell', 'short', 'cover'
    symbol='XRP/USD',       # Symbol to trade
    size=100.0,             # Size in USD
    price=2.35,             # Reference price (for logging)
    reason="Buy signal",    # Human-readable explanation

    # Optional
    order_type='market',    # 'market' or 'limit'
    limit_price=2.34,       # For limit orders
    stop_loss=2.30,         # Auto stop-loss price
    take_profit=2.50,       # Auto take-profit price
    metadata={'key': 'value'}  # Custom data
)
```

## Using Strategy State

The `state` dict persists between calls:

```python
def generate_signal(data, config, state):
    # Initialize state on first call
    if 'position' not in state:
        state['position'] = 0
        state['last_signal_time'] = None

    # Use state for decisions
    if state['position'] > config['max_position']:
        return None  # Don't buy more

    # Store indicators for logging
    state['indicators'] = {
        'rsi': calculated_rsi,
        'sma': calculated_sma,
    }
```

## Callback Functions

### on_fill

Called when an order is filled:

```python
def on_fill(fill: dict, state: dict) -> None:
    """
    fill contains:
        side: 'buy' or 'sell'
        size: float (base asset amount)
        price: float
        fee: float
        pnl: float (realized P&L, if closing position)
    """
    if fill['side'] == 'buy':
        state['position'] = state.get('position', 0) + fill['size']
    else:
        state['position'] = state.get('position', 0) - fill['size']
```

### on_start

Called when strategy starts:

```python
def on_start(config: dict, state: dict) -> None:
    state['position'] = 0
    state['indicators'] = {}
    print(f"Strategy started with config: {config}")
```

### on_stop

Called when strategy stops:

```python
def on_stop(state: dict) -> None:
    print(f"Strategy stopped. Final position: {state.get('position', 0)}")
```

## Example: Simple Moving Average Strategy

```python
# strategies/sma_crossover.py

from typing import Optional
from ws_tester.types import DataSnapshot, Signal

STRATEGY_NAME = "sma_crossover"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["XRP/USD"]

CONFIG = {
    'fast_period': 5,
    'slow_period': 20,
    'position_size_usd': 25,
}

def calculate_sma(candles, period):
    if len(candles) < period:
        return None
    closes = [c.close for c in candles[-period:]]
    return sum(closes) / len(closes)

def generate_signal(data: DataSnapshot, config: dict, state: dict) -> Optional[Signal]:
    symbol = "XRP/USD"
    candles = data.candles_5m.get(symbol, ())

    if len(candles) < config['slow_period']:
        return None

    fast_sma = calculate_sma(list(candles), config['fast_period'])
    slow_sma = calculate_sma(list(candles), config['slow_period'])

    if not fast_sma or not slow_sma:
        return None

    # Store indicators
    state['indicators'] = {
        'fast_sma': fast_sma,
        'slow_sma': slow_sma,
    }

    price = data.prices.get(symbol)
    prev_fast = state.get('prev_fast_sma')
    prev_slow = state.get('prev_slow_sma')

    # Update state for next iteration
    state['prev_fast_sma'] = fast_sma
    state['prev_slow_sma'] = slow_sma

    if not prev_fast or not prev_slow:
        return None

    # Golden cross: fast crosses above slow
    if prev_fast <= prev_slow and fast_sma > slow_sma:
        return Signal(
            action='buy',
            symbol=symbol,
            size=config['position_size_usd'],
            price=price,
            reason=f"Golden cross: fast={fast_sma:.4f} > slow={slow_sma:.4f}",
            stop_loss=price * 0.98,
            take_profit=price * 1.03,
        )

    # Death cross: fast crosses below slow
    if prev_fast >= prev_slow and fast_sma < slow_sma:
        return Signal(
            action='sell',
            symbol=symbol,
            size=config['position_size_usd'],
            price=price,
            reason=f"Death cross: fast={fast_sma:.4f} < slow={slow_sma:.4f}",
        )

    return None

def on_start(config: dict, state: dict) -> None:
    state['prev_fast_sma'] = None
    state['prev_slow_sma'] = None
```

## Best Practices

1. **Return None for no action** - Don't return signals unless you want to trade
2. **Store indicators in state** - They'll be logged automatically
3. **Use stop-loss and take-profit** - Protect against large losses
4. **Handle missing data gracefully** - Check for None/empty before accessing
5. **Keep strategies simple** - One strategy, one idea

## Modular Strategy Structure

For complex strategies (500+ lines), consider splitting into a package:

```
strategies/my_strategy/
├── __init__.py      # Re-exports public interface
├── config.py        # CONFIG, SYMBOL_CONFIGS, validation
├── indicators.py    # Technical indicator calculations
├── signals.py       # Signal generation logic
└── lifecycle.py     # on_start, on_fill, on_stop callbacks
```

The `__init__.py` must export the required interface:

```python
# strategies/my_strategy/__init__.py

from .config import STRATEGY_NAME, STRATEGY_VERSION, SYMBOLS, CONFIG
from .signals import generate_signal
from .lifecycle import on_start, on_fill, on_stop

__all__ = [
    'STRATEGY_NAME', 'STRATEGY_VERSION', 'SYMBOLS', 'CONFIG',
    'generate_signal', 'on_start', 'on_fill', 'on_stop',
]
```

See [mean_reversion](../../../ws_paper_tester/strategies/mean_reversion/) and [ratio_trading](../../../ws_paper_tester/strategies/ratio_trading/) for examples.

## Testing Your Strategy

Run the paper tester with simulated data:

```bash
cd ws_paper_tester
python ws_tester.py --simulated --duration 5
```

Check the strategy logs in `logs/strategies/your_strategy_*.jsonl`.

---
*Last updated: 2025-12-14*
