# Signal Reference

The `Signal` class represents a trading instruction from a strategy.

## Definition

```python
@dataclass
class Signal:
    action: str
    symbol: str
    size: float
    price: float
    reason: str
    order_type: str = 'market'
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
```

## Fields

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `action` | `str` | Trade action: `'buy'`, `'sell'`, `'short'`, `'cover'` |
| `symbol` | `str` | Trading pair, e.g., `'XRP/USDT'` |
| `size` | `float` | Position size in **USD** (not base asset) |
| `price` | `float` | Reference price for logging/slippage calculation |
| `reason` | `str` | Human-readable explanation for the trade |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `order_type` | `str` | `'market'` | Order type (only `'market'` supported) |
| `limit_price` | `float` | `None` | Reserved for future limit order support |
| `stop_loss` | `float` | `None` | Auto stop-loss price level |
| `take_profit` | `float` | `None` | Auto take-profit price level |
| `metadata` | `dict` | `None` | Strategy-specific data for logging |

## Action Types

| Action | Direction | Use Case |
|--------|-----------|----------|
| `'buy'` | Open/add long | Enter or add to long position |
| `'sell'` | Close long | Exit long position (partial or full) |
| `'short'` | Open/add short | Enter or add to short position |
| `'cover'` | Close short | Exit short position (partial or full) |

## Stop-Loss / Take-Profit Rules

### Long Positions (`'buy'`)
- `stop_loss`: Must be **below** entry price
- `take_profit`: Must be **above** entry price

### Short Positions (`'short'`)
- `stop_loss`: Must be **above** entry price
- `take_profit`: Must be **below** entry price

## Examples

### Basic Buy Signal
```python
Signal(
    action='buy',
    symbol='XRP/USDT',
    size=20.0,
    price=2.35,
    reason='RSI oversold (28.5)'
)
```

### Buy with Risk Management
```python
Signal(
    action='buy',
    symbol='XRP/USDT',
    size=20.0,
    price=2.35,
    reason='RSI oversold (28.5)',
    stop_loss=2.30,      # 2.1% below entry
    take_profit=2.42,    # 3.0% above entry
)
```

### Short with Metadata
```python
Signal(
    action='short',
    symbol='BTC/USDT',
    size=50.0,
    price=45000.0,
    reason='Resistance rejection',
    stop_loss=45500.0,   # Above entry for shorts
    take_profit=44000.0, # Below entry for shorts
    metadata={
        'resistance_level': 45200,
        'volume_spike': True,
    }
)
```

### Exit Signal
```python
Signal(
    action='sell',
    symbol='XRP/USDT',
    size=20.0,
    price=2.40,
    reason='Take profit target reached'
)
```

## Usage in Strategies

```python
def generate_signal(data, config, state) -> Optional[Signal]:
    price = data.prices.get('XRP/USDT')
    if not price:
        return None

    # Your trading logic here
    if should_buy(data, state):
        return Signal(
            action='buy',
            symbol='XRP/USDT',
            size=config['position_size_usd'],
            price=price,
            reason='Entry condition met',
            stop_loss=price * 0.98,
            take_profit=price * 1.02,
        )

    return None
```

## Validation

The executor validates signals before execution:

1. `action` must be one of: `'buy'`, `'sell'`, `'short'`, `'cover'`
2. `symbol` must be in configured symbols list
3. `size` must be positive
4. `price` must be positive
5. `stop_loss` direction validated against action (below for longs, above for shorts)

Invalid signals are logged and rejected.
