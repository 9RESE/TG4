# Strategy Development Guide

**Version:** 2.0
**Target Platform:** WebSocket Paper Tester v1.15.0+
**Supported Pairs:** XRP/USDT, BTC/USDT, XRP/BTC (Kraken)

This guide provides comprehensive instructions for developing trading strategies that integrate seamlessly with the WebSocket Paper Tester platform.

---

## Table of Contents

1. [Quick Start Template](#1-quick-start-template)
2. [Strategy Module Contract](#2-strategy-module-contract)
3. [Signal Generation](#3-signal-generation)
4. [Stop Loss & Take Profit](#4-stop-loss--take-profit)
5. [Position Management](#5-position-management)
6. [State Management](#6-state-management)
7. [Logging Requirements](#7-logging-requirements)
8. [Data Access Patterns](#8-data-access-patterns)
9. [Configuration Best Practices](#9-configuration-best-practices)
10. [Testing Your Strategy](#10-testing-your-strategy)
11. [Common Pitfalls](#11-common-pitfalls)
12. [Performance Considerations](#12-performance-considerations)

---

## 1. Quick Start Template

Create a new file in `strategies/` directory (e.g., `strategies/my_strategy.py`):

```python
"""
My Trading Strategy

Brief description of what this strategy does.
"""
from typing import Optional, Dict, Any
from ws_tester.types import DataSnapshot, Signal

# =============================================================================
# REQUIRED: Strategy Metadata
# =============================================================================
STRATEGY_NAME = "my_strategy"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["XRP/USDT"]  # or ["XRP/USDT", "BTC/USDT"]

# =============================================================================
# REQUIRED: Default Configuration
# =============================================================================
CONFIG = {
    # Position sizing
    'position_size_usd': 20.0,      # Trade size in USD
    'max_position_usd': 50.0,       # Maximum position exposure

    # Risk management
    'stop_loss_pct': 0.5,           # Stop loss percentage
    'take_profit_pct': 0.4,         # Take profit percentage

    # Strategy-specific parameters
    'signal_threshold': 0.1,        # Your custom parameters
}

# =============================================================================
# REQUIRED: Main Signal Generation Function
# =============================================================================
def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """
    Generate trading signal based on market data.

    Called every tick (default: 100ms).

    Args:
        data: Immutable market data snapshot
        config: Strategy configuration (CONFIG + overrides)
        state: Mutable strategy state dict

    Returns:
        Signal if trading opportunity found, None otherwise
    """
    # Initialize state on first call
    if 'initialized' not in state:
        state['initialized'] = True
        state['position'] = 0.0
        state['indicators'] = {}

    for symbol in SYMBOLS:
        price = data.prices.get(symbol)
        if not price:
            continue

        # Calculate your indicators
        # ...

        # Store indicators for logging
        state['indicators'] = {
            'price': price,
            # Add your indicators here
        }

        # Check entry conditions
        if _should_buy(data, config, state, symbol):
            return Signal(
                action='buy',
                symbol=symbol,
                size=config['position_size_usd'],
                price=price,
                reason="Entry signal description",
                stop_loss=price * (1 - config['stop_loss_pct'] / 100),
                take_profit=price * (1 + config['take_profit_pct'] / 100),
            )

        # Check exit conditions
        if _should_sell(data, config, state, symbol):
            return Signal(
                action='sell',
                symbol=symbol,
                size=abs(state['position']),
                price=price,
                reason="Exit signal description",
            )

    return None

# =============================================================================
# OPTIONAL: Lifecycle Callbacks
# =============================================================================
def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Called when strategy starts. Initialize state here."""
    state['position'] = 0.0
    state['entry_price'] = 0.0
    state['trade_count'] = 0

def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Called when a trade is executed. Update position tracking."""
    if fill['side'] == 'buy':
        state['position'] += fill['value']
        state['entry_price'] = fill['price']
    elif fill['side'] == 'sell':
        state['position'] -= fill['value']
        if abs(state['position']) < 0.01:
            state['position'] = 0.0
            state['entry_price'] = 0.0
    state['trade_count'] += 1

def on_stop(state: Dict[str, Any]) -> None:
    """Called when strategy stops. Cleanup if needed."""
    pass

# =============================================================================
# Helper Functions (Private)
# =============================================================================
def _should_buy(data, config, state, symbol) -> bool:
    """Your buy logic here."""
    return False

def _should_sell(data, config, state, symbol) -> bool:
    """Your sell logic here."""
    return False
```

---

## 2. Strategy Module Contract

### Required Components

| Component | Type | Purpose |
|-----------|------|---------|
| `STRATEGY_NAME` | `str` | Unique identifier (lowercase, underscores) |
| `STRATEGY_VERSION` | `str` | Semantic version (e.g., "1.0.0") |
| `SYMBOLS` | `List[str]` | Trading pairs (e.g., `["XRP/USDT"]`) |
| `CONFIG` | `Dict[str, Any]` | Default configuration values |
| `generate_signal()` | `function` | Main signal generation logic |

### Optional Components

| Component | Type | Purpose |
|-----------|------|---------|
| `on_start()` | `function` | Called when strategy loads |
| `on_fill()` | `function` | Called after each fill execution |
| `on_stop()` | `function` | Called when strategy stops |

### Naming Conventions

```python
# Strategy name: lowercase with underscores
STRATEGY_NAME = "mean_reversion"      # Good
STRATEGY_NAME = "MeanReversion"       # Bad
STRATEGY_NAME = "mean-reversion"      # Bad

# Version: semantic versioning
STRATEGY_VERSION = "1.0.0"            # Good
STRATEGY_VERSION = "v1.0"             # Bad
STRATEGY_VERSION = "1"                # Bad
```

---

## 3. Signal Generation

### Signal Structure

```python
from ws_tester.types import Signal

Signal(
    # REQUIRED
    action: str,        # 'buy', 'sell', 'short', 'cover'
    symbol: str,        # 'XRP/USDT' or 'BTC/USDT'
    size: float,        # Size in USD (not base asset!)
    price: float,       # Reference price for logging
    reason: str,        # Human-readable explanation

    # OPTIONAL
    order_type: str = 'market',           # Only 'market' supported
    limit_price: Optional[float] = None,  # Reserved for future
    stop_loss: Optional[float] = None,    # Auto stop-loss price
    take_profit: Optional[float] = None,  # Auto take-profit price
    metadata: Optional[dict] = None,      # Strategy-specific data
)
```

### Action Types

| Action | Direction | Use Case |
|--------|-----------|----------|
| `'buy'` | Open/add long | Enter long position or add to existing |
| `'sell'` | Close long | Exit long position (partial or full) |
| `'short'` | Open/add short | Enter short position or add to existing |
| `'cover'` | Close short | Exit short position (partial or full) |

### Signal Examples

**Entry Signal (Long):**
```python
Signal(
    action='buy',
    symbol='XRP/USDT',
    size=20.0,  # $20 USD
    price=2.35,
    reason="RSI oversold (28.5), price below SMA",
    stop_loss=2.30,      # 2.1% below entry
    take_profit=2.40,    # 2.1% above entry
)
```

**Exit Signal (Close Long):**
```python
Signal(
    action='sell',
    symbol='XRP/USDT',
    size=20.0,  # Close $20 worth
    price=2.40,
    reason="Take profit target reached",
)
```

**Short Entry:**
```python
Signal(
    action='short',
    symbol='XRP/USDT',
    size=20.0,
    price=2.50,
    reason="Price rejected at resistance",
    stop_loss=2.55,      # Stop ABOVE entry for shorts
    take_profit=2.45,    # TP BELOW entry for shorts
)
```

### Leveraged Positions

The executor supports leveraged positions for both longs (up to 1.5x default) and shorts (up to 2x default). Leverage is applied automatically based on configuration.

**How Leverage Works:**

| Position | No Leverage | With Leverage |
|----------|-------------|---------------|
| Long | Limited to available USDT | Can exceed USDT (borrowed funds) |
| Short | Limited to equity * leverage | Same |

**Margin Calls:**
When equity drops below 25% of total position value, the executor automatically liquidates all positions. This applies to both leveraged longs and shorts.

```python
# The executor will cap your position to max leverage
Signal(
    action='buy',
    symbol='XRP/USDT',
    size=200.0,  # Requested $200 on $100 equity with 1.5x leverage
    price=2.35,
    reason="High conviction entry",
)
# Actual fill will be capped to ~$150 (1.5x of equity)
```

**Check for Margin Call Exits:**
```python
def on_fill(fill: dict, state: dict) -> None:
    """Handle margin call liquidations."""
    if fill.get('trigger') == 'margin_call':
        # Position was force-liquidated
        state['margin_calls'] = state.get('margin_calls', 0) + 1
```

### Reason Field Best Practices

The `reason` field is logged and displayed in the dashboard. Make it informative:

```python
# Good reasons (informative, include key values)
reason="RSI oversold (28.5), price 2.1% below SMA"
reason="Orderbook imbalance 0.35, spread 0.15%"
reason="Stop loss triggered at 2.30"
reason="Mean reversion: deviation -1.2 std"

# Bad reasons (vague, no context)
reason="Buy signal"
reason="Entry"
reason="Selling"
```

---

## 4. Stop Loss & Take Profit

### How Stop Loss Works

The executor automatically checks ALL positions every tick:

```
Price drops to stop_loss level
       ↓
PaperExecutor.check_stops()
       ↓
Generates automatic 'sell'/'cover' Signal
       ↓
Position closed at market price
       ↓
Logged as "stop_loss triggered"
```

### Stop Loss Rules

| Position Type | Stop Loss | Trigger Condition |
|---------------|-----------|-------------------|
| Long (`'buy'`) | Below entry | `current_price <= stop_loss` |
| Short (`'short'`) | Above entry | `current_price >= stop_loss` |

### Take Profit Rules

| Position Type | Take Profit | Trigger Condition |
|---------------|-------------|-------------------|
| Long (`'buy'`) | Above entry | `current_price >= take_profit` |
| Short (`'short'`) | Below entry | `current_price <= take_profit` |

### Calculating Stop Levels

```python
# For LONG positions
entry_price = 2.35
stop_loss_pct = 0.5   # 0.5%
take_profit_pct = 0.4 # 0.4%

stop_loss = entry_price * (1 - stop_loss_pct / 100)    # 2.3383
take_profit = entry_price * (1 + take_profit_pct / 100) # 2.3594

# For SHORT positions (inverted!)
stop_loss = entry_price * (1 + stop_loss_pct / 100)    # 2.3618
take_profit = entry_price * (1 - take_profit_pct / 100) # 2.3406
```

### Risk-Reward Ratio

Always consider your risk-reward ratio:

```python
# Example: 0.5% stop, 0.4% take profit = 0.8:1 R:R
# This means you risk $1 to make $0.80 - need >55.6% win rate to profit

# Better: 0.5% stop, 1.0% take profit = 2:1 R:R
# Risk $1 to make $2 - need only >33.3% win rate to profit

CONFIG = {
    'stop_loss_pct': 0.5,
    'take_profit_pct': 1.0,  # 2:1 R:R ratio
}
```

### Dynamic Stops

You can adjust stops based on market conditions:

```python
def generate_signal(data, config, state):
    symbol = 'XRP/USDT'
    ob = data.orderbooks.get(symbol)
    price = data.prices.get(symbol)

    if ob and price:
        # Wider stops in volatile markets (wide spread)
        volatility_multiplier = 1 + (ob.spread_pct * 10)

        stop_loss = price * (1 - config['stop_loss_pct'] / 100 * volatility_multiplier)
        take_profit = price * (1 + config['take_profit_pct'] / 100)

        return Signal(
            action='buy',
            symbol=symbol,
            size=config['position_size_usd'],
            price=price,
            reason=f"Entry with dynamic stop (vol={volatility_multiplier:.2f})",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
```

### Trailing Stops (Manual Implementation)

The platform tracks `highest_price` and `lowest_price` on positions. You can implement trailing stops:

```python
def generate_signal(data, config, state):
    # Check if we have position and should trail stop
    if state.get('position', 0) > 0:  # Long position
        price = data.prices.get('XRP/USDT')
        entry = state.get('entry_price', 0)
        highest = state.get('highest_price', entry)

        # Update highest price tracking
        if price > highest:
            state['highest_price'] = price
            highest = price

        # Trail stop: 0.5% below highest price
        trailing_stop = highest * (1 - 0.005)

        if price <= trailing_stop:
            return Signal(
                action='sell',
                symbol='XRP/USDT',
                size=state['position'],
                price=price,
                reason=f"Trailing stop hit (high={highest:.4f}, stop={trailing_stop:.4f})",
            )

    return None
```

---

## 5. Position Management

### Position Lifecycle

```
Signal(action='buy')
        ↓
PaperExecutor.execute()
        ↓
Position created in portfolio
        ↓
on_fill() callback
        ↓
Monitor via state tracking
        ↓
Signal(action='sell') OR auto stop/TP
        ↓
Position closed, P&L realized
```

### Position Sizing

The `size` in Signal is always in **USD**, not base asset:

```python
# Buying $20 worth of XRP at $2.35
Signal(
    action='buy',
    symbol='XRP/USDT',
    size=20.0,  # USD amount
    price=2.35,
    ...
)

# Executor calculates:
# base_size = 20.0 / 2.35 = 8.51 XRP
# cost = 8.51 * 2.35 * 1.001 (fee) = $20.10 USDT deducted
```

### Position Limits

Always respect maximum position limits:

```python
def generate_signal(data, config, state):
    current_position = state.get('position', 0)
    max_position = config.get('max_position_usd', 50)
    position_size = config.get('position_size_usd', 20)

    # Check if adding to position would exceed limit
    if current_position + position_size > max_position:
        return None  # Skip signal

    # Or reduce size to stay within limit
    available = max_position - current_position
    actual_size = min(position_size, available)

    if actual_size < 5:  # Minimum trade size
        return None

    return Signal(
        action='buy',
        symbol='XRP/USDT',
        size=actual_size,
        ...
    )
```

### Partial Position Closes

You can close positions partially:

```python
def generate_signal(data, config, state):
    position = state.get('position', 0)

    if position > 0:
        price = data.prices.get('XRP/USDT')
        entry = state.get('entry_price')

        # Close half at 0.3% profit
        if (price - entry) / entry > 0.003:
            return Signal(
                action='sell',
                symbol='XRP/USDT',
                size=position / 2,  # Half position
                price=price,
                reason="Partial take profit at 0.3%",
            )

    return None
```

### Tracking Position State

Use `on_fill()` to accurately track position:

```python
def on_fill(fill: dict, state: dict) -> None:
    """Update state after fill execution."""
    side = fill['side']
    value = fill['value']  # USD value
    price = fill['price']

    if side == 'buy':
        # Add to long position
        old_pos = state.get('position', 0)
        old_cost = state.get('total_cost', 0)

        state['position'] = old_pos + value
        state['total_cost'] = old_cost + value

        # Calculate average entry price
        if state['position'] > 0:
            state['avg_entry'] = state['total_cost'] / state['position'] * price

    elif side == 'sell':
        # Close long position
        state['position'] = max(0, state.get('position', 0) - value)

        if state['position'] < 0.01:  # Closed
            state['position'] = 0
            state['total_cost'] = 0
            state['avg_entry'] = 0

    # Similar for short/cover
```

---

## 6. State Management

### State Dictionary

Each strategy has an isolated `state` dict that persists across ticks:

```python
state = {
    # Position tracking
    'position': 0.0,           # Current position in USD
    'entry_price': 0.0,        # Average entry price
    'trade_count': 0,          # Number of trades

    # Indicator cache
    'indicators': {},          # Logged each tick
    'sma_values': [],          # Rolling calculations
    'last_signal_time': None,  # Cooldown tracking

    # Custom state
    'my_custom_data': {},
}
```

### Initialization Pattern

```python
def generate_signal(data, config, state):
    # Lazy initialization (recommended)
    if 'initialized' not in state:
        state['initialized'] = True
        state['position'] = 0.0
        state['indicators'] = {}
        state['candle_buffer'] = []

    # ... rest of logic

# OR use on_start (called once when strategy loads)
def on_start(config, state):
    state['position'] = 0.0
    state['indicators'] = {}
    state['candle_buffer'] = []
```

### Indicator State for Logging

The `state['indicators']` dict is automatically logged:

```python
def generate_signal(data, config, state):
    symbol = 'XRP/USDT'
    price = data.prices.get(symbol)

    # Calculate indicators
    sma = calculate_sma(data.candles_1m.get(symbol, ()), 20)
    rsi = calculate_rsi(data.candles_1m.get(symbol, ()), 14)

    # Store for logging (IMPORTANT!)
    state['indicators'] = {
        'price': price,
        'sma_20': sma,
        'rsi_14': rsi,
        'deviation_pct': (price - sma) / sma * 100 if sma else 0,
    }

    # Use indicators for signal logic
    if rsi < 30 and price < sma:
        return Signal(...)
```

### State Cleanup

Avoid unbounded state growth:

```python
def generate_signal(data, config, state):
    # Bad: Unbounded list
    state.setdefault('all_prices', []).append(price)  # Memory leak!

    # Good: Bounded buffer
    buffer = state.setdefault('price_buffer', [])
    buffer.append(price)
    if len(buffer) > 100:  # Keep last 100
        state['price_buffer'] = buffer[-100:]
```

---

## 7. Logging Requirements

### What Gets Logged Automatically

The platform logs these automatically:

| Log Type | Location | Contents |
|----------|----------|----------|
| System | `logs/system/` | Strategy load, WebSocket events |
| Strategy | `logs/strategies/{name}_*.jsonl` | Signals, indicators, latency |
| Trades | `logs/trades/fills_*.jsonl` | Fills, P&L, portfolio state |
| Aggregated | `logs/aggregated/unified_*.jsonl` | End-to-end trace |

### Signal Logging (Automatic)

Every `generate_signal()` call is logged:

```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "strategy": "my_strategy",
  "event": "signal_generated",
  "correlation_id": "abc12345-000001",
  "signal": {
    "action": "buy",
    "symbol": "XRP/USDT",
    "size": 20.0,
    "price": 2.35,
    "reason": "RSI oversold (28.5)",
    "stop_loss": 2.30,
    "take_profit": 2.40
  },
  "indicators": {
    "price": 2.35,
    "rsi": 28.5,
    "sma_20": 2.38
  },
  "latency_us": 156
}
```

### Required: Populate Indicators

**Always populate `state['indicators']`** for debugging:

```python
def generate_signal(data, config, state):
    # Calculate all indicators FIRST
    price = data.prices.get('XRP/USDT')
    ob = data.orderbooks.get('XRP/USDT')

    indicators = {
        'price': price,
        'spread_pct': ob.spread_pct if ob else None,
        'imbalance': ob.imbalance if ob else None,
        # Add ALL relevant indicators
    }

    # Store BEFORE returning
    state['indicators'] = indicators

    # Now generate signal using indicators
    if indicators['imbalance'] and indicators['imbalance'] > 0.3:
        return Signal(...)

    return None
```

### Fill Logging (Automatic)

When signals execute, fills are logged:

```json
{
  "timestamp": "2024-01-15T10:30:00.234567",
  "event": "fill",
  "fill": {
    "fill_id": "abc12345-000001-0",
    "side": "buy",
    "symbol": "XRP/USDT",
    "size": 8.51,
    "price": 2.35,
    "fee": 0.02,
    "pnl": 0.0,
    "signal_reason": "RSI oversold (28.5)"
  },
  "portfolio_after": {
    "usdt": 79.98,
    "equity": 99.98,
    "positions": {...}
  }
}
```

### Custom Logging via Metadata

Use `metadata` for strategy-specific logging:

```python
Signal(
    action='buy',
    symbol='XRP/USDT',
    size=20.0,
    price=2.35,
    reason="Grid level 3 triggered",
    metadata={
        'grid_level': 3,
        'grid_price': 2.35,
        'total_grid_fills': 5,
    }
)
```

---

## 8. Data Access Patterns

### DataSnapshot Structure

```python
@dataclass(frozen=True)
class DataSnapshot:
    timestamp: datetime
    prices: Dict[str, float]                    # Current prices
    candles_1m: Dict[str, Tuple[Candle, ...]]   # 1-minute candles
    candles_5m: Dict[str, Tuple[Candle, ...]]   # 5-minute candles
    orderbooks: Dict[str, OrderbookSnapshot]    # Top 10 levels
    trades: Dict[str, Tuple[Trade, ...]]        # Last 100 trades
```

### Accessing Price Data

```python
def generate_signal(data, config, state):
    # Current price
    xrp_price = data.prices.get('XRP/USDT')
    btc_price = data.prices.get('BTC/USDT')

    # Safety check
    if not xrp_price:
        return None

    # Mid prices (calculated property)
    mids = data.mids  # {'XRP/USDT': 2.35, 'BTC/USDT': 45000.0}
```

### Accessing Candle Data

```python
def generate_signal(data, config, state):
    # Get 1-minute candles for XRP
    candles_1m = data.candles_1m.get('XRP/USDT', ())

    if len(candles_1m) < 20:  # Need enough data
        return None

    # Candles are immutable tuples, newest last
    latest = candles_1m[-1]

    # Candle properties
    close_prices = [c.close for c in candles_1m]
    volumes = [c.volume for c in candles_1m]

    # Calculate SMA
    sma_20 = sum(close_prices[-20:]) / 20

    # Candle attributes
    # c.timestamp, c.open, c.high, c.low, c.close, c.volume
    # c.body_size, c.range, c.is_bullish
```

### Accessing Orderbook Data

```python
def generate_signal(data, config, state):
    ob = data.orderbooks.get('XRP/USDT')

    if not ob or not ob.best_bid:
        return None

    # Key properties
    best_bid = ob.best_bid      # (price, size)
    best_ask = ob.best_ask      # (price, size)
    spread = ob.spread          # Absolute spread
    spread_pct = ob.spread_pct  # Percentage spread
    mid = ob.mid                # Mid price

    # Depth analysis
    bid_depth = ob.bid_depth    # Total bid volume
    ask_depth = ob.ask_depth    # Total ask volume
    imbalance = ob.imbalance    # -1 to +1 (positive = more bids)

    # Full orderbook (top 10 levels)
    for price, size in ob.bids:
        print(f"Bid: {price} x {size}")
    for price, size in ob.asks:
        print(f"Ask: {price} x {size}")
```

### Accessing Trade Tape

```python
def generate_signal(data, config, state):
    trades = data.trades.get('XRP/USDT', ())

    if len(trades) < 10:
        return None

    # Recent trades (newest last)
    recent = trades[-10:]

    # Analyze trade flow
    buy_volume = sum(t.value for t in recent if t.side == 'buy')
    sell_volume = sum(t.value for t in recent if t.side == 'sell')

    # Built-in VWAP calculation
    vwap = data.get_vwap('XRP/USDT', n_trades=50)

    # Built-in imbalance calculation
    trade_imbalance = data.get_trade_imbalance('XRP/USDT', n_trades=50)
```

---

## 9. Configuration Best Practices

### Config Structure

```python
CONFIG = {
    # === Position Sizing ===
    'position_size_usd': 20.0,      # Base trade size
    'max_position_usd': 50.0,       # Maximum exposure
    'min_trade_size': 5.0,          # Minimum viable trade

    # === Risk Management ===
    'stop_loss_pct': 0.5,           # Stop loss %
    'take_profit_pct': 0.4,         # Take profit %
    'max_daily_loss_pct': 5.0,      # Daily loss limit

    # === Signal Generation ===
    'lookback_periods': 20,         # Indicator lookback
    'signal_threshold': 0.1,        # Entry threshold
    'cooldown_seconds': 60,         # Min time between trades

    # === Strategy-Specific ===
    'custom_param_1': 0.5,
    'custom_param_2': 10,
}
```

### Accessing Configuration

```python
def generate_signal(data, config, state):
    # Direct access
    position_size = config['position_size_usd']

    # Safe access with defaults
    cooldown = config.get('cooldown_seconds', 60)
    threshold = config.get('signal_threshold', 0.1)
```

### Config Overrides (config.yaml)

```yaml
strategy_overrides:
  my_strategy:
    position_size_usd: 30.0    # Override from 20 to 30
    stop_loss_pct: 0.3         # Tighter stop
    custom_param_1: 0.7        # Adjusted threshold
```

### Runtime Validation

```python
def generate_signal(data, config, state):
    # Validate config on first run
    if 'config_validated' not in state:
        assert config.get('position_size_usd', 0) > 0, "position_size_usd must be positive"
        assert config.get('stop_loss_pct', 0) > 0, "stop_loss_pct must be positive"
        assert config.get('stop_loss_pct', 0) < 10, "stop_loss_pct too large"
        state['config_validated'] = True
```

---

## 10. Testing Your Strategy

### Manual Testing

```bash
# Run paper tester with your strategy
cd ws_paper_tester
python ws_tester.py --duration 5 --symbols XRP/USDT

# Watch logs
tail -f logs/strategies/my_strategy_*.jsonl | jq
tail -f logs/trades/fills_*.jsonl | jq
```

### Unit Testing Pattern

```python
# tests/test_my_strategy.py
import pytest
from datetime import datetime
from strategies.my_strategy import generate_signal, CONFIG, on_fill
from ws_tester.types import DataSnapshot, Candle, OrderbookSnapshot

@pytest.fixture
def sample_data():
    """Create sample DataSnapshot for testing."""
    return DataSnapshot(
        timestamp=datetime.now(),
        prices={'XRP/USDT': 2.35},
        candles_1m={'XRP/USDT': tuple(
            Candle(datetime.now(), 2.30, 2.36, 2.29, 2.35, 1000)
            for _ in range(20)
        )},
        candles_5m={'XRP/USDT': ()},
        orderbooks={'XRP/USDT': OrderbookSnapshot(
            bids=((2.34, 1000), (2.33, 2000)),
            asks=((2.36, 1000), (2.37, 2000)),
        )},
        trades={'XRP/USDT': ()},
    )

def test_no_signal_without_data():
    """Strategy returns None with empty data."""
    empty_data = DataSnapshot(
        timestamp=datetime.now(),
        prices={},
        candles_1m={},
        candles_5m={},
        orderbooks={},
        trades={},
    )
    state = {}
    signal = generate_signal(empty_data, CONFIG, state)
    assert signal is None

def test_buy_signal_generation(sample_data):
    """Strategy generates buy signal under right conditions."""
    state = {}
    # Setup state to trigger buy
    # ...
    signal = generate_signal(sample_data, CONFIG, state)

    if signal:
        assert signal.action in ('buy', 'sell', 'short', 'cover')
        assert signal.size > 0
        assert signal.symbol == 'XRP/USDT'

def test_on_fill_updates_state():
    """on_fill correctly updates position tracking."""
    state = {'position': 0}
    fill = {
        'side': 'buy',
        'value': 20.0,
        'price': 2.35,
    }
    on_fill(fill, state)
    assert state['position'] == 20.0
```

### Run Tests

```bash
cd ws_paper_tester
pytest tests/ -v
pytest tests/test_my_strategy.py -v
```

---

## 11. Common Pitfalls

### 1. Returning Signal on Every Tick

```python
# BAD: Generates buy signal every 100ms
def generate_signal(data, config, state):
    return Signal(action='buy', ...)  # Trades constantly!

# GOOD: Check conditions first
def generate_signal(data, config, state):
    if not _should_enter(data, config, state):
        return None
    return Signal(action='buy', ...)
```

### 2. Not Checking Position Before Entry

```python
# BAD: Can exceed max position
def generate_signal(data, config, state):
    if _buy_condition(data):
        return Signal(action='buy', size=20, ...)

# GOOD: Check position limits
def generate_signal(data, config, state):
    if state.get('position', 0) >= config['max_position_usd']:
        return None
    if _buy_condition(data):
        return Signal(action='buy', size=20, ...)
```

### 3. Stop Loss on Wrong Side

```python
# BAD: Stop loss above entry for long (will trigger immediately!)
Signal(
    action='buy',
    price=2.35,
    stop_loss=2.40,  # WRONG! This is above entry
)

# GOOD: Stop loss below entry for long
Signal(
    action='buy',
    price=2.35,
    stop_loss=2.30,  # Correct: below entry
)
```

### 4. Unbounded State Growth

```python
# BAD: Memory leak
state.setdefault('history', []).append(data.prices)

# GOOD: Bounded buffer
history = state.setdefault('history', [])
history.append(data.prices)
state['history'] = history[-100:]  # Keep only last 100
```

### 5. Missing Data Checks

```python
# BAD: Will crash on missing data
def generate_signal(data, config, state):
    price = data.prices['XRP/USDT']  # KeyError if missing!
    ob = data.orderbooks['XRP/USDT']
    spread = ob.spread_pct  # AttributeError if ob is None!

# GOOD: Safe access
def generate_signal(data, config, state):
    price = data.prices.get('XRP/USDT')
    if not price:
        return None

    ob = data.orderbooks.get('XRP/USDT')
    if not ob or not ob.best_bid:
        return None

    spread = ob.spread_pct
```

### 6. Forgetting to Update on_fill

```python
# BAD: State doesn't reflect actual position
# (on_fill not implemented or incorrect)

# GOOD: Always track fills
def on_fill(fill, state):
    if fill['side'] == 'buy':
        state['position'] = state.get('position', 0) + fill['value']
    elif fill['side'] == 'sell':
        state['position'] = max(0, state.get('position', 0) - fill['value'])
```

### 7. Size Confusion (USD vs Base)

```python
# BAD: Thinking size is in XRP
Signal(
    action='buy',
    symbol='XRP/USDT',
    size=10,  # This is $10 USD, not 10 XRP!
)

# GOOD: Document and think in USD
Signal(
    action='buy',
    symbol='XRP/USDT',
    size=20.0,  # $20 USD worth of XRP
)
```

---

## 12. Performance Considerations

### Signal Generation Latency

Target: **< 1000 microseconds (1ms)** per signal generation.

```python
def generate_signal(data, config, state):
    # FAST: Simple lookups
    price = data.prices.get('XRP/USDT')
    ob = data.orderbooks.get('XRP/USDT')

    # MODERATE: Built-in calculations
    vwap = data.get_vwap('XRP/USDT', 50)  # Pre-optimized

    # SLOW: Complex calculations - cache them!
    if 'sma_cache' not in state or state.get('last_candle_count') != len(candles):
        state['sma_cache'] = _calculate_sma(candles, 20)
        state['last_candle_count'] = len(candles)
```

### Caching Expensive Calculations

```python
def generate_signal(data, config, state):
    candles = data.candles_1m.get('XRP/USDT', ())

    # Only recalculate when new candle arrives
    current_count = len(candles)
    if state.get('_candle_count') != current_count:
        state['_candle_count'] = current_count
        state['_sma'] = _calculate_sma(candles, 20)
        state['_rsi'] = _calculate_rsi(candles, 14)

    # Use cached values
    sma = state['_sma']
    rsi = state['_rsi']
```

### Memory-Efficient Indicators

```python
# BAD: Keep all data
state['all_closes'] = [c.close for c in candles]  # Grows unbounded

# GOOD: Keep only what you need
closes = [c.close for c in candles[-50:]]  # Last 50 only
sma = sum(closes[-20:]) / 20
```

### Minimize Signal Object Creation

```python
# BAD: Creating Signal even when not trading
def generate_signal(data, config, state):
    signal = Signal(action='buy', ...)  # Created even if not needed
    if not should_trade:
        return None
    return signal

# GOOD: Only create when needed
def generate_signal(data, config, state):
    if not should_trade:
        return None
    return Signal(action='buy', ...)  # Only created when trading
```

---

## Appendix A: Complete Example Strategy

See `strategies/mean_reversion.py` for a complete, production-ready example implementing:
- RSI and SMA indicators
- Dynamic stop/take-profit levels
- Position tracking with on_fill
- Proper state management
- Comprehensive indicator logging

---

## Appendix B: Signal Quick Reference

| Scenario | Action | Stop Loss | Take Profit |
|----------|--------|-----------|-------------|
| Enter long | `'buy'` | Below entry | Above entry |
| Exit long | `'sell'` | N/A or new stop | N/A |
| Enter short | `'short'` | Above entry | Below entry |
| Exit short | `'cover'` | N/A or new stop | N/A |
| Add to long | `'buy'` | Optional update | Optional update |
| Partial close | `'sell'` (fractional) | N/A | N/A |

---

## Appendix C: DataSnapshot Quick Reference

```python
data.prices['XRP/USDT']           # float: Current price
data.mids                        # dict: Mid prices from orderbook
data.spreads                     # dict: Spread percentages

data.candles_1m['XRP/USDT']       # tuple[Candle]: 1-min candles (newest last)
data.candles_5m['XRP/USDT']       # tuple[Candle]: 5-min candles

data.orderbooks['XRP/USDT'].best_bid    # (price, size)
data.orderbooks['XRP/USDT'].best_ask    # (price, size)
data.orderbooks['XRP/USDT'].spread_pct  # float: Spread %
data.orderbooks['XRP/USDT'].imbalance   # float: -1 to +1

data.trades['XRP/USDT']           # tuple[Trade]: Recent trades
data.get_vwap('XRP/USDT', 50)     # float: VWAP over last N trades
data.get_trade_imbalance('XRP/USDT', 50)  # float: Buy-sell imbalance
```

---

**Document Version:** 2.0
**Last Updated:** 2025-12-15
**Platform Version:** WebSocket Paper Tester v1.15.0+
