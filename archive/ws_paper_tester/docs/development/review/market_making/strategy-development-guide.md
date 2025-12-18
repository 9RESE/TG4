# Strategy Development Guide

**Version:** 2.0
**Target Platform:** WebSocket Paper Tester v1.4.0+
**Supported Pairs:** XRP/USDT, BTC/USDT, XRP/BTC (Kraken)
**Last Updated:** 2025-12-14

This guide provides comprehensive instructions for developing trading strategies that integrate seamlessly with the WebSocket Paper Tester platform.

> **Version 2.0 Note:** This guide has been significantly enhanced based on lessons learned from 15+ strategy review cycles across order_flow, mean_reversion, ratio_trading, and market_making strategies. New sections cover volatility regime classification, circuit breakers, signal rejection tracking, and other patterns that emerged as essential during development.

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
13. [Per-Pair PnL Tracking](#13-per-pair-pnl-tracking) *(v1.4.0+)*
14. [Advanced Features](#14-advanced-features) *(v1.4.0+)*

### **Version 2.0: Lessons Learned from Production Strategies**

15. [Volatility Regime Classification](#15-volatility-regime-classification) *(v2.0)*
16. [Circuit Breaker Protection](#16-circuit-breaker-protection) *(v2.0)*
17. [Signal Rejection Tracking](#17-signal-rejection-tracking) *(v2.0)*
18. [Trade Flow Confirmation](#18-trade-flow-confirmation) *(v2.0)*
19. [Trend Filtering](#19-trend-filtering) *(v2.0)*
20. [Session & Time-of-Day Awareness](#20-session--time-of-day-awareness) *(v2.0)*
21. [Position Decay](#21-position-decay) *(v2.0)*
22. [Per-Symbol Configuration (SYMBOL_CONFIGS)](#22-per-symbol-configuration) *(v2.0)*
23. [Fee Profitability Checks](#23-fee-profitability-checks) *(v2.0)*
24. [Correlation Monitoring](#24-correlation-monitoring) *(v2.0)*
25. [Research-Backed Parameters](#25-research-backed-parameters) *(v2.0)*
26. [Strategy Scope Documentation](#26-strategy-scope-documentation) *(v2.0)*

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

## 13. Per-Pair PnL Tracking

*Added in v1.4.0*

The platform now tracks P&L and trade metrics per trading pair automatically.

### Automatic Portfolio Tracking

The `StrategyPortfolio` class tracks these metrics per symbol:

```python
portfolio.pnl_by_symbol      # {'XRP/USDT': 12.50, 'BTC/USDT': -3.25}
portfolio.trades_by_symbol   # {'XRP/USDT': 15, 'BTC/USDT': 8}
portfolio.wins_by_symbol     # {'XRP/USDT': 10, 'BTC/USDT': 3}
portfolio.losses_by_symbol   # {'XRP/USDT': 5, 'BTC/USDT': 5}
```

### Strategy-Level Tracking

In your strategy, track per-pair metrics in state:

```python
def on_fill(fill: dict, state: dict) -> None:
    symbol = fill.get('symbol')
    pnl = fill.get('pnl', 0)

    # Initialize tracking dicts
    if 'pnl_by_symbol' not in state:
        state['pnl_by_symbol'] = {}
    if 'trades_by_symbol' not in state:
        state['trades_by_symbol'] = {}

    # Update per-pair metrics
    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl
    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1
```

### Including Per-Pair Stats in Indicators

Add per-pair P&L to your indicator logging:

```python
state['indicators'] = {
    'symbol': symbol,
    'price': price,
    # ... other indicators ...

    # Per-pair metrics (v1.4.0+)
    'pnl_symbol': state.get('pnl_by_symbol', {}).get(symbol, 0),
    'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
}
```

### Portfolio Snapshot Method

Use the logger's new method for detailed portfolio state:

```python
# In ws_tester main loop (done automatically)
logger.log_portfolio_snapshot(
    strategy='my_strategy',
    portfolio=portfolio.to_dict(prices),
    prices=prices,
    symbol_stats=portfolio.get_all_symbol_stats(),
)
```

### Console Output

Fills now show cumulative per-pair P&L:

```
[FILL] [market_making] BUY XRP/USDT @ 2.350000 P&L: +$1.25 [XRP/USDT total: +$8.50]
```

---

## 14. Advanced Features

*Added in v1.4.0*

### Configuration Validation

Validate your config parameters on startup:

```python
def _validate_config(config: dict) -> list:
    """Validate configuration. Returns list of error messages."""
    errors = []

    # Required positive values
    required = ['position_size_usd', 'stop_loss_pct', 'take_profit_pct']
    for key in required:
        val = config.get(key)
        if val is None:
            errors.append(f"Missing required config: {key}")
        elif val <= 0:
            errors.append(f"{key} must be positive, got {val}")

    # Check R:R ratio
    sl = config.get('stop_loss_pct', 0.5)
    tp = config.get('take_profit_pct', 0.4)
    if sl > 0 and tp > 0:
        rr = tp / sl
        if rr < 0.5:
            errors.append(f"Warning: Poor R:R ratio ({rr:.2f}:1)")

    return errors

def on_start(config: dict, state: dict) -> None:
    errors = _validate_config(config)
    if errors:
        for e in errors:
            print(f"[my_strategy] Config warning: {e}")
    state['config_validated'] = True
```

### Avellaneda-Stoikov Reservation Price

For advanced inventory-aware quote adjustment:

```python
def _calculate_reservation_price(
    mid_price: float,
    inventory: float,
    max_inventory: float,
    gamma: float,          # Risk aversion (0.01-1.0)
    volatility_pct: float  # Current volatility
) -> float:
    """
    Reservation price = mid * (1 - q * γ * σ² * 100)

    - Positive inventory → lower price (favor selling)
    - Negative inventory → higher price (favor buying)
    """
    if max_inventory <= 0:
        return mid_price

    q = inventory / max_inventory
    sigma_sq = (volatility_pct / 100) ** 2

    return mid_price * (1 - q * gamma * sigma_sq * 100)
```

Enable in config:
```python
CONFIG = {
    'use_reservation_price': False,  # Enable A-S model
    'gamma': 0.1,                    # Risk aversion
}
```

### Trailing Stops

Implement trailing stops for profit protection:

```python
def _calculate_trailing_stop(
    entry_price: float,
    highest_price: float,  # Track in on_fill
    side: str,             # 'long' or 'short'
    activation_pct: float, # e.g., 0.2 (activate at 0.2% profit)
    trail_distance_pct: float,  # e.g., 0.15 (trail 0.15% from high)
) -> float:
    """Returns trailing stop price or None if not activated."""
    if side == 'long':
        profit_pct = (highest_price - entry_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)
    elif side == 'short':
        profit_pct = (entry_price - highest_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 + trail_distance_pct / 100)
    return None
```

Track position entries for trailing stops:

```python
def on_fill(fill: dict, state: dict) -> None:
    symbol = fill.get('symbol')
    side = fill.get('side')
    price = fill.get('price')

    if 'position_entries' not in state:
        state['position_entries'] = {}

    if side == 'buy':
        state['position_entries'][symbol] = {
            'entry_price': price,
            'highest_price': price,
            'side': 'long',
        }
    elif side == 'sell':
        if symbol in state['position_entries']:
            del state['position_entries'][symbol]
```

Config options:
```python
CONFIG = {
    'use_trailing_stop': False,
    'trailing_stop_activation': 0.2,  # Activate at 0.2% profit
    'trailing_stop_distance': 0.15,   # Trail 0.15% from high
}
```

---

## 15. Volatility Regime Classification

*Added in v2.0 - Learned from order_flow v3.0+, mean_reversion v2.0+*

### Why Volatility Regimes Matter

Cryptocurrency markets exhibit extreme volatility variations. Fixed parameters fail because:
- **Low volatility**: Thresholds too wide, missing opportunities
- **High volatility**: Thresholds too tight, generating false signals
- **Extreme volatility**: Market conditions unsuitable for trading

### Standard Regime Classification

```python
from enum import Enum

class VolatilityRegime(Enum):
    LOW = "low"           # < 0.3% volatility
    MEDIUM = "medium"     # 0.3% - 0.8%
    HIGH = "high"         # 0.8% - 1.5%
    EXTREME = "extreme"   # > 1.5%

def _get_volatility_regime(volatility_pct: float, config: dict) -> VolatilityRegime:
    """Classify current volatility into regime."""
    low_thresh = config.get('regime_low_threshold', 0.3)
    medium_thresh = config.get('regime_medium_threshold', 0.8)
    high_thresh = config.get('regime_high_threshold', 1.5)

    if volatility_pct < low_thresh:
        return VolatilityRegime.LOW
    elif volatility_pct < medium_thresh:
        return VolatilityRegime.MEDIUM
    elif volatility_pct < high_thresh:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.EXTREME
```

### Regime Adjustment Multipliers

| Regime | Threshold Multiplier | Size Multiplier | Notes |
|--------|---------------------|-----------------|-------|
| LOW | 0.8 - 0.9 | 1.0 | Tighter thresholds, normal size |
| MEDIUM | 1.0 | 1.0 | Baseline parameters |
| HIGH | 1.2 - 1.3 | 0.7 - 0.8 | Wider thresholds, reduced size |
| EXTREME | N/A | 0.0 | **Pause trading** |

### Configuration Pattern

```python
CONFIG = {
    # Volatility regime settings
    'use_volatility_regimes': True,
    'base_volatility_pct': 0.5,        # Baseline volatility
    'volatility_lookback': 20,          # Candles for calculation

    # Regime thresholds
    'regime_low_threshold': 0.3,
    'regime_medium_threshold': 0.8,
    'regime_high_threshold': 1.5,
    'regime_extreme_pause': True,       # Pause in EXTREME

    # Regime multipliers
    'regime_low_threshold_mult': 0.9,
    'regime_high_threshold_mult': 1.3,
    'regime_high_size_mult': 0.8,
    'regime_extreme_size_mult': 0.0,    # No trading
}
```

### Key Lesson Learned

> **From order_flow v3.1 review**: "The strategy uses fixed thresholds regardless of market volatility. Research indicates mean reversion needs wider thresholds in high volatility and tighter in low volatility."

---

## 16. Circuit Breaker Protection

*Added in v2.0 - Learned from all strategy reviews*

### Why Circuit Breakers Are Essential

Without circuit breakers, strategies can:
- Continue losing during adverse market conditions
- Compound losses during regime changes
- Fail to recognize when assumptions are invalid

### Standard Implementation

```python
def _check_circuit_breaker(state: dict, config: dict) -> bool:
    """Returns True if circuit breaker is active (should NOT trade)."""
    if not config.get('use_circuit_breaker', True):
        return False

    max_losses = config.get('max_consecutive_losses', 3)
    cooldown_minutes = config.get('circuit_breaker_minutes', 15)

    consecutive_losses = state.get('consecutive_losses', 0)

    # Check if in cooldown period
    cb_triggered_time = state.get('circuit_breaker_triggered_time')
    if cb_triggered_time:
        elapsed = (datetime.now() - cb_triggered_time).total_seconds() / 60
        if elapsed < cooldown_minutes:
            return True  # Still in cooldown
        else:
            # Cooldown complete, reset
            state['circuit_breaker_triggered_time'] = None
            state['consecutive_losses'] = 0
            return False

    # Check if should trigger
    if consecutive_losses >= max_losses:
        state['circuit_breaker_triggered_time'] = datetime.now()
        return True

    return False
```

### Tracking in on_fill()

```python
def on_fill(fill: dict, state: dict) -> None:
    pnl = fill.get('pnl', 0)

    if pnl < 0:
        state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1
    elif pnl > 0:
        state['consecutive_losses'] = 0  # Reset on win
```

### Configuration Pattern

```python
CONFIG = {
    'use_circuit_breaker': True,
    'max_consecutive_losses': 3,
    'circuit_breaker_minutes': 15,
}
```

### Key Lesson Learned

> **From mean_reversion v1.0 review**: "No protection against consecutive losses. Can continue losing during adverse conditions. May experience significant drawdown."

---

## 17. Signal Rejection Tracking

*Added in v2.0 - Learned from mean_reversion v2.0+, order_flow v3.0+*

### Why Track Rejections

Tracking why signals are NOT generated is as important as tracking signals:
- Identifies parameter tuning opportunities
- Reveals market condition patterns
- Enables strategy optimization

### Standard Rejection Categories

```python
from enum import Enum

class RejectionReason(Enum):
    NO_SIGNAL_CONDITIONS = "no_signal_conditions"
    TIME_COOLDOWN = "time_cooldown"
    TRADE_COOLDOWN = "trade_cooldown"
    WARMING_UP = "warming_up"
    TRADE_FLOW_NOT_ALIGNED = "trade_flow_not_aligned"
    REGIME_PAUSE = "regime_pause"
    CIRCUIT_BREAKER = "circuit_breaker"
    MAX_POSITION = "max_position"
    INSUFFICIENT_SIZE = "insufficient_size"
    NO_PRICE_DATA = "no_price_data"
    TRENDING_MARKET = "trending_market"     # For mean reversion
    HIGH_VPIN = "high_vpin"                 # For order flow
    SPREAD_TOO_WIDE = "spread_too_wide"     # For market making
    LOW_CORRELATION = "low_correlation"      # For ratio trading
```

### Tracking Implementation

```python
def _track_rejection(state: dict, reason: RejectionReason, symbol: str) -> None:
    """Track why a signal was rejected."""
    if not state.get('track_rejections', True):
        return

    if 'rejection_counts' not in state:
        state['rejection_counts'] = {}

    key = reason.value
    state['rejection_counts'][key] = state['rejection_counts'].get(key, 0) + 1
```

### Logging in on_stop()

```python
def on_stop(state: dict) -> None:
    """Log rejection statistics in session summary."""
    rejection_counts = state.get('rejection_counts', {})
    if rejection_counts:
        print(f"\n[{STRATEGY_NAME}] Signal Rejection Summary:")
        for reason, count in sorted(rejection_counts.items(),
                                    key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
```

### Key Lesson Learned

> **From order_flow v2.2 review**: "Cannot debug why signals aren't generated. No visibility into market conditions. Production monitoring impossible."

---

## 18. Trade Flow Confirmation

*Added in v2.0 - Learned from order_flow v2.2+, mean_reversion v2.0+*

### Why Confirm with Trade Flow

Technical indicator signals can be invalidated by actual market microstructure. Trade flow confirmation:
- Validates that the market "agrees" with your signal
- Reduces false signals in momentum strategies
- Confirms mean reversion opportunities

### Implementation Pattern

```python
def _is_trade_flow_aligned(
    data: DataSnapshot,
    direction: str,       # 'buy' or 'sell'
    symbol: str,
    config: dict
) -> bool:
    """Check if trade flow supports signal direction."""
    if not config.get('use_trade_flow_confirmation', True):
        return True  # Skip check if disabled

    threshold = config.get('trade_flow_threshold', 0.10)
    lookback = config.get('trade_flow_lookback', 50)

    trade_flow = data.get_trade_imbalance(symbol, lookback)
    if trade_flow is None:
        return True  # No data, allow signal

    if direction == 'buy':
        return trade_flow > threshold  # More buy pressure
    elif direction == 'sell':
        return trade_flow < -threshold  # More sell pressure

    return True
```

### Configuration Pattern

```python
CONFIG = {
    'use_trade_flow_confirmation': True,
    'trade_flow_threshold': 0.10,
    'trade_flow_lookback': 50,
}
```

### Strategy-Specific Considerations

| Strategy Type | Trade Flow Usage | Notes |
|---------------|------------------|-------|
| Momentum (order_flow) | Confirm direction | Flow should match signal |
| Mean Reversion | Confirm reversal starting | Flow should be turning |
| Market Making | Optional | Check before adding to position |
| Ratio Trading | Cross-check both assets | Ensure both moving as expected |

---

## 19. Trend Filtering

*Added in v2.0 - Learned from mean_reversion v3.0+*

### Why Filter Trends

Mean reversion strategies perform poorly in trending markets. Trend filtering:
- Blocks signals during strong directional moves
- Allows "band walks" to continue without losses
- Focuses on ranging market opportunities

### Linear Regression Slope Method

```python
def _calculate_trend_slope(candles: list, period: int) -> float:
    """Calculate trend slope using linear regression."""
    if len(candles) < period:
        return 0.0

    recent = candles[-period:]
    closes = [c.close for c in recent]

    # Simple linear regression
    n = len(closes)
    x_mean = (n - 1) / 2
    y_mean = sum(closes) / n

    numerator = sum((i - x_mean) * (closes[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    slope = numerator / denominator

    # Convert to percentage
    return (slope / y_mean) * 100

def _is_trending(slope: float, threshold: float) -> bool:
    """Determine if market is trending based on slope."""
    return abs(slope) > threshold
```

### Trend Confirmation Period

```python
# Lesson: Single evaluation can flip trend status incorrectly
# Use confirmation period to avoid false positives in choppy markets

def _check_trend_with_confirmation(
    current_slope: float,
    state: dict,
    config: dict
) -> bool:
    """Returns True if market is confirmed trending."""
    threshold = config.get('trend_slope_threshold', 0.05)
    confirmation_periods = config.get('trend_confirmation_periods', 3)

    is_trending_now = abs(current_slope) > threshold

    if is_trending_now:
        state['trend_confirmation_count'] = state.get('trend_confirmation_count', 0) + 1
    else:
        state['trend_confirmation_count'] = 0

    return state['trend_confirmation_count'] >= confirmation_periods
```

### Key Lesson Learned

> **From mean_reversion v4.0 review**: "The trend filter uses linear regression slope with 0.05% threshold. May trigger in choppy markets. Added confirmation period (3 consecutive trending evaluations) to prevent false positives."

---

## 20. Session & Time-of-Day Awareness

*Added in v2.0 - Learned from order_flow v4.0*

### Why Session Awareness Matters

Cryptocurrency markets show distinct patterns by trading session:
- **Asian Session**: Lower liquidity, higher volatility
- **European Session**: Increasing volume, moderate volatility
- **US Session**: Highest volume, often directional moves
- **Overlap Periods**: Highest activity, best opportunities

### Session Classification

```python
from enum import Enum

class TradingSession(Enum):
    ASIA = "asia"
    EUROPE = "europe"
    US = "us"
    US_EUROPE_OVERLAP = "us_europe_overlap"
    OFF_HOURS = "off_hours"

def _get_trading_session(hour_utc: int) -> TradingSession:
    """Classify current hour into trading session."""
    # Note: These are approximations; adjust for DST
    if 0 <= hour_utc < 8:
        return TradingSession.ASIA
    elif 8 <= hour_utc < 14:
        return TradingSession.EUROPE
    elif 14 <= hour_utc < 17:
        return TradingSession.US_EUROPE_OVERLAP
    elif 17 <= hour_utc < 22:
        return TradingSession.US
    else:
        return TradingSession.OFF_HOURS
```

### Session Adjustment Multipliers

| Session | Threshold Mult | Size Mult | Notes |
|---------|---------------|-----------|-------|
| ASIA | 1.2 | 0.8 | Lower liquidity, be conservative |
| EUROPE | 1.0 | 1.0 | Baseline |
| US_EUROPE_OVERLAP | 0.85 | 1.1 | High activity, lower thresholds |
| US | 1.0 | 1.0 | Baseline |
| OFF_HOURS | 1.3 | 0.6 | Very conservative |

### Configuration Pattern

```python
CONFIG = {
    'use_session_awareness': True,
    'session_asia_threshold_mult': 1.2,
    'session_asia_size_mult': 0.8,
    'session_overlap_threshold_mult': 0.85,
    'session_overlap_size_mult': 1.1,
}
```

---

## 21. Position Decay

*Added in v2.0 - Learned from mean_reversion v3.0+, order_flow v4.0*

### Why Position Decay

Positions that haven't hit take profit may still be profitable:
- Time erodes edge in momentum strategies
- Holding costs (opportunity cost) accumulate
- Earlier exit with smaller profit often beats waiting

### Timing Considerations

> **Key Lesson from mean_reversion v4.0 review**: "The 3-minute decay start with aggressive multipliers may force premature exits. With 5-minute candles used for signals, decay begins before even one new candle completes."

**Recommended timing for different strategies:**

| Strategy Type | Decay Start | Decay Interval | Notes |
|---------------|-------------|----------------|-------|
| Scalping | 5-10 min | 2-3 min | Fast decay acceptable |
| Mean Reversion | 15-30 min | 5 min | Allow time for reversion |
| Momentum | 10-15 min | 3-5 min | Balance edge vs time |

### Implementation Pattern

```python
def _get_decayed_take_profit(
    entry_time: datetime,
    original_tp_pct: float,
    config: dict
) -> float:
    """Calculate decayed take profit based on position age."""
    if not config.get('use_position_decay', True):
        return original_tp_pct

    decay_start = config.get('decay_start_minutes', 15.0)
    decay_interval = config.get('decay_interval_minutes', 5.0)
    decay_multipliers = config.get('decay_multipliers', [1.0, 0.85, 0.70, 0.50])

    age_minutes = (datetime.now() - entry_time).total_seconds() / 60

    if age_minutes < decay_start:
        return original_tp_pct

    # Calculate decay stage
    stages_elapsed = int((age_minutes - decay_start) / decay_interval)
    stage_index = min(stages_elapsed, len(decay_multipliers) - 1)

    multiplier = decay_multipliers[stage_index]
    return original_tp_pct * multiplier
```

### Recommended Configuration

```python
CONFIG = {
    'use_position_decay': True,
    'decay_start_minutes': 15.0,        # Conservative start
    'decay_interval_minutes': 5.0,      # Gentle progression
    'decay_multipliers': [1.0, 0.85, 0.70, 0.50],  # Gradual reduction
}
```

---

## 22. Per-Symbol Configuration (SYMBOL_CONFIGS)

*Added in v2.0 - Learned from all multi-symbol strategies*

### Why Per-Symbol Configuration

Different trading pairs have different characteristics:
- **BTC/USDT**: Lower volatility, tighter thresholds
- **XRP/USDT**: Higher volatility, wider thresholds
- **XRP/BTC**: Ratio behavior, different sizing

### Standard Pattern

```python
SYMBOLS = ["XRP/USDT", "BTC/USDT", "XRP/BTC"]

# Global defaults
CONFIG = {
    'position_size_usd': 20.0,
    'stop_loss_pct': 0.5,
    'take_profit_pct': 0.5,
    'cooldown_seconds': 10.0,
    # ... other defaults
}

# Per-symbol overrides
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'deviation_threshold': 0.5,
        'position_size_usd': 20.0,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'cooldown_seconds': 10.0,
    },
    'BTC/USDT': {
        'deviation_threshold': 0.3,     # Tighter for lower volatility
        'position_size_usd': 50.0,      # Larger for BTC liquidity
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'cooldown_seconds': 5.0,        # Faster for liquid BTC
    },
    'XRP/BTC': {
        'deviation_threshold': 1.0,     # Wider for ratio volatility
        'position_size_usd': 15.0,      # Smaller for lower liquidity
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'cooldown_seconds': 20.0,       # Slower for ratio trades
    },
}
```

### Merging Configuration

```python
def _get_symbol_config(symbol: str, base_config: dict) -> dict:
    """Get merged config for specific symbol."""
    symbol_overrides = SYMBOL_CONFIGS.get(symbol, {})
    merged = {**base_config, **symbol_overrides}
    return merged
```

### Usage in generate_signal()

```python
def generate_signal(data, config, state):
    for symbol in SYMBOLS:
        # Get symbol-specific config
        sym_config = _get_symbol_config(symbol, config)

        # Use sym_config instead of config for all parameters
        threshold = sym_config['deviation_threshold']
        position_size = sym_config['position_size_usd']
        # ...
```

---

## 23. Fee Profitability Checks

*Added in v2.0 - Learned from order_flow v2.2+, market_making v1.4+*

### Why Check Fee Profitability

With typical crypto fees (0.1% maker/taker), round-trip costs are ~0.2%:
- A 0.3% take profit leaves only 0.1% profit after fees
- In tight spread conditions, trades may be unprofitable

### Implementation Pattern

```python
def _check_fee_profitability(
    expected_profit_pct: float,
    fee_rate: float,
    min_profit_pct: float = 0.05
) -> tuple[bool, float]:
    """
    Check if trade is profitable after fees.

    Returns:
        (is_profitable, net_profit_pct)
    """
    round_trip_fee_pct = fee_rate * 2 * 100  # Both entry and exit
    net_profit_pct = expected_profit_pct - round_trip_fee_pct

    return net_profit_pct >= min_profit_pct, net_profit_pct
```

### Configuration Pattern

```python
CONFIG = {
    'check_fee_profitability': True,
    'estimated_fee_rate': 0.001,        # 0.1% per side
    'min_net_profit_pct': 0.05,         # Minimum after fees
}
```

### Key Lesson Learned

> **From order_flow v2.2 review**: "The strategy doesn't calculate whether trades are profitable after fees. With 0.1% maker/taker fees, a 0.2% round-trip cost can significantly impact profitability."

---

## 24. Correlation Monitoring

*Added in v2.0 - Learned from ratio_trading v3.0+, order_flow v4.0*

### Why Monitor Correlation

For multi-asset strategies:
- **Ratio trading**: Requires assets to maintain equilibrium relationship
- **Cross-pair exposure**: Avoid overexposure to correlated assets
- **Market regime detection**: Correlation spikes during stress

### Correlation Warning for Ratio Trading

```python
def _calculate_rolling_correlation(
    prices_a: list,
    prices_b: list,
    window: int = 20
) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(prices_a) < window or len(prices_b) < window:
        return 1.0  # Assume correlated if insufficient data

    a = prices_a[-window:]
    b = prices_b[-window:]

    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)

    covariance = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(window)) / window
    std_a = (sum((x - mean_a) ** 2 for x in a) / window) ** 0.5
    std_b = (sum((x - mean_b) ** 2 for x in b) / window) ** 0.5

    if std_a == 0 or std_b == 0:
        return 1.0

    return covariance / (std_a * std_b)

def _check_correlation_pause(
    correlation: float,
    config: dict
) -> tuple[bool, bool]:
    """
    Check if correlation is problematic.

    Returns:
        (should_pause, should_warn)
    """
    warning_threshold = config.get('correlation_warning_threshold', 0.5)
    pause_threshold = config.get('correlation_pause_threshold', 0.3)
    pause_enabled = config.get('correlation_pause_enabled', True)

    should_warn = correlation < warning_threshold
    should_pause = pause_enabled and correlation < pause_threshold

    return should_pause, should_warn
```

### Cross-Pair Exposure Management

```python
def _check_correlation_exposure(
    current_positions: dict,
    new_signal_direction: str,
    max_same_direction_exposure: float = 150.0
) -> float:
    """
    Reduce position size if multiple correlated pairs in same direction.

    Returns adjusted position size multiplier (0.0 to 1.0).
    """
    total_long = sum(v for v in current_positions.values() if v > 0)
    total_short = sum(abs(v) for v in current_positions.values() if v < 0)

    if new_signal_direction == 'buy' and total_long > max_same_direction_exposure * 0.5:
        return 0.75  # Reduce by 25%
    elif new_signal_direction == 'sell' and total_short > max_same_direction_exposure * 0.5:
        return 0.75

    return 1.0
```

### Key Lesson Learned

> **From ratio_trading v5.0 review**: "XRP/BTC Correlation at Historical Lows - The correlation between XRP and BTC has dropped to ~0.40, making it the altcoin with the highest degree of independence. This fundamentally challenges the viability of pairs trading."

---

## 25. Research-Backed Parameters

*Added in v2.0 - Learned from all deep strategy reviews*

### Why Research Matters

Academic research provides optimized parameter starting points:
- Reduces trial-and-error optimization time
- Provides theoretical justification for decisions
- Establishes baseline performance expectations

### Key Research-Backed Parameters

#### Bollinger Bands for Crypto

| Parameter | Common Default | Research Optimized | Notes |
|-----------|---------------|-------------------|-------|
| Period | 20 | 20 | Standard across assets |
| Std Dev | 2.0 | 2.5-3.0 for crypto | Wider for volatile markets |

> **Source**: Research suggests crypto markets benefit from wider bands to avoid false signals.

#### Z-Score Thresholds (Pairs Trading)

| Parameter | Common Default | Research Optimized | Notes |
|-----------|---------------|-------------------|-------|
| Entry Threshold | 2.0 std | 1.42 std | Lower entry for more signals |
| Exit Threshold | 1.0 std | 0.37 std | Exit closer to mean |

> **Source**: ArXiv paper 2412.12555v1 optimization study.

#### RSI Settings

| Market Condition | Oversold | Overbought | Notes |
|------------------|----------|------------|-------|
| Standard | 30 | 70 | Traditional settings |
| High Volatility | 25 | 75 | More extreme for crypto |
| Conservative | 35 | 65 | Fewer signals, higher quality |

#### Risk-Reward Ratios

| Strategy Type | Minimum R:R | Recommended R:R | Win Rate Required |
|---------------|-------------|-----------------|-------------------|
| Mean Reversion | 1:1 | 1:1 | 50% for breakeven |
| Momentum | 1.5:1 | 2:1 | 33% for breakeven |
| Scalping | 1:1 | 1:1 | 50% for breakeven |

### Key Lesson Learned

> **From mean_reversion v4.0 review**: "Research indicates trailing stops are designed for trend-following strategies, not mean reversion. Mean reversion anticipates price returning to a specific level, making fixed take profit more appropriate."

---

## 26. Strategy Scope Documentation

*Added in v2.0 - Learned from ratio_trading reviews*

### Why Document Strategy Scope

Each strategy is designed for specific:
- Market conditions (trending vs ranging)
- Asset pairs (crypto/crypto vs crypto/stablecoin)
- Timeframes (scalping vs swing)

### Documentation Template

Include at the top of your strategy file:

```python
"""
Strategy Name: Your Strategy Name
Version: X.Y.Z

SCOPE AND LIMITATIONS:
- Asset Types: [List applicable asset types]
- Market Conditions: [Ranging/Trending/Both]
- Timeframe: [Scalping/Intraday/Swing]
- NOT Suitable For: [Explicitly list exclusions]

THEORETICAL BASIS:
- [Brief description of the trading theory]
- [Key academic references]

KEY ASSUMPTIONS:
- [List assumptions that must hold for strategy to work]
- [Conditions that would invalidate the strategy]
"""
```

### Example: Ratio Trading Scope

```python
"""
Ratio Trading Strategy v3.0.0

SCOPE AND LIMITATIONS:
- Asset Types: Crypto-to-crypto pairs ONLY (e.g., XRP/BTC)
- Market Conditions: Mean-reverting ratio relationships
- NOT Suitable For: USDT-denominated pairs (XRP/USDT, BTC/USDT)

THEORETICAL BASIS:
- Pairs trading / statistical arbitrage
- Requires cointegrated relationship between assets
- Trades relative value, not absolute direction

KEY ASSUMPTIONS:
- XRP and BTC maintain cointegrated relationship
- Correlation remains above 0.5
- Ratio reverts to mean within reasonable timeframe

WHEN TO PAUSE:
- Correlation drops below 0.4
- Major regulatory news affecting one asset
- Extreme market volatility (VPIN > 0.7)
"""
```

### Key Lesson Learned

> **From ratio_trading v1.0 review**: "Fundamental Design Mismatch - The user requested analysis for XRP/USDT, BTC/USDT, and XRP/BTC pairs. However, ratio trading is fundamentally incompatible with USDT pairs. USDT is a stable quote currency - there is no 'ratio' to mean-revert."

---

## Appendix D: Strategy Development Checklist

*Added in v2.0*

Use this checklist before marking a strategy as production-ready:

### Required Components ✓

- [ ] `STRATEGY_NAME` (lowercase with underscores)
- [ ] `STRATEGY_VERSION` (semantic versioning)
- [ ] `SYMBOLS` list
- [ ] `CONFIG` with all parameters
- [ ] `generate_signal()` function
- [ ] `on_start()` callback
- [ ] `on_fill()` callback
- [ ] `on_stop()` callback

### Risk Management ✓

- [ ] R:R ratio >= 1:1
- [ ] Circuit breaker protection
- [ ] Position limits enforced
- [ ] Cooldown mechanisms
- [ ] Fee profitability check

### Volatility Handling ✓

- [ ] Volatility regime classification
- [ ] EXTREME regime pause
- [ ] Dynamic threshold adjustments
- [ ] Dynamic position sizing

### Logging & Debugging ✓

- [ ] Indicators always populated (including early returns)
- [ ] Signal rejection tracking
- [ ] Per-pair PnL tracking
- [ ] Configuration validation on startup

### Research Alignment ✓

- [ ] Strategy scope documented
- [ ] Parameters research-backed
- [ ] Theoretical basis documented
- [ ] Limitations explicitly stated

---

## Appendix E: Common Patterns Reference

*Added in v2.0*

### Pattern: Always Populate Indicators

```python
def generate_signal(data, config, state):
    symbol = 'XRP/USDT'
    price = data.prices.get(symbol)

    # ALWAYS set base indicators first
    state['indicators'] = {
        'symbol': symbol,
        'price': price,
        'status': 'evaluating',
    }

    if not price:
        state['indicators']['status'] = 'no_price'
        return None

    # Continue with logic, updating indicators as you go
    state['indicators']['status'] = 'active'
    # ...
```

### Pattern: Config Validation

```python
def on_start(config, state):
    errors = _validate_config(config)
    if errors:
        for e in errors:
            print(f"[{STRATEGY_NAME}] Warning: {e}")

    # Log enabled features
    features = []
    if config.get('use_volatility_regimes'):
        features.append('volatility_regimes')
    if config.get('use_circuit_breaker'):
        features.append('circuit_breaker')
    print(f"[{STRATEGY_NAME}] Enabled features: {features}")
```

### Pattern: Comprehensive on_stop()

```python
def on_stop(state):
    print(f"\n{'='*50}")
    print(f"[{STRATEGY_NAME}] Session Summary")
    print(f"{'='*50}")

    # Per-pair stats
    for symbol in SYMBOLS:
        pnl = state.get('pnl_by_symbol', {}).get(symbol, 0)
        trades = state.get('trades_by_symbol', {}).get(symbol, 0)
        print(f"  {symbol}: PnL ${pnl:.2f} ({trades} trades)")

    # Rejection analysis
    rejections = state.get('rejection_counts', {})
    if rejections:
        print(f"\nRejection Analysis:")
        for reason, count in sorted(rejections.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Circuit breaker activations
    cb_count = state.get('circuit_breaker_activations', 0)
    if cb_count:
        print(f"\nCircuit breaker activated: {cb_count} times")
```

---

**Document Version:** 2.0
**Last Updated:** 2025-12-14
**Platform Version:** WebSocket Paper Tester v1.4.0+

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-12 | Initial guide |
| 1.1 | 2025-12-13 | Added v1.4.0+ features |
| 2.0 | 2025-12-14 | Major update with lessons learned from 15+ strategy review cycles |
