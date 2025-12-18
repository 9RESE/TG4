# C4 Code Diagram

## Key Classes and Modules

This level shows the internal structure of key components.

### WebSocket Client

```python
class KrakenWebSocket:
    """
    Manages WebSocket connection to Kraken exchange.
    """
    - connect()
    - disconnect()
    - subscribe(pairs: List[str])
    - on_message(handler: Callable)
    - reconnect_with_backoff()
```

### LLM Client

```python
class LLMClient:
    """
    Multi-provider LLM interface.
    """
    - providers: Dict[str, Provider]
    - active_provider: str
    - query(prompt: str) -> Decision
    - switch_provider(name: str)
    - get_performance_metrics()
```

### Risk Manager

```python
class RiskManager:
    """
    Enforces trading risk limits.
    """
    - max_position_size: float
    - max_drawdown: float
    - validate_trade(signal: Signal) -> bool
    - calculate_position_size(signal: Signal) -> float
    - check_drawdown() -> bool
    - emergency_halt()
```

### Signal Generator

```python
class Signal:
    """
    Trade signal from LLM.
    """
    - pair: str
    - action: Literal["BUY", "SELL", "HOLD"]
    - confidence: float
    - entry_price: float
    - stop_loss: float
    - take_profit: float
    - reasoning: str
```

## Module Structure

```
ws_paper_tester/
├── __init__.py
├── websocket/
│   ├── client.py         # KrakenWebSocket
│   └── handlers.py       # Message handlers
├── strategy/
│   ├── llm_client.py     # LLMClient
│   ├── prompt_builder.py # Prompt templates
│   └── signals.py        # Signal, SignalGenerator
├── execution/
│   ├── risk_manager.py   # RiskManager
│   ├── order_executor.py # OrderExecutor
│   └── positions.py      # PositionTracker
├── data/
│   ├── indicators.py     # Technical indicators
│   └── storage.py        # TimescaleDB interface
└── config/
    └── settings.py       # Configuration
```
