# WebSocket Paper Trading Tester - Design Document

**Version:** 2.0
**Date:** 2025-12-13
**Status:** IMPLEMENTED
**Author:** Architecture Review
**Implementation:** `/ws_paper_tester/`

---

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Code Sharing** | Standalone | Clean architecture, no legacy dependencies |
| **State Persistence** | No | Fresh start each run, logs capture everything |
| **Production Path** | Yes | Design for eventual live trading |
| **Dashboard** | Yes | Real-time web UI for monitoring |
| **Portfolio Model** | Isolated per strategy | Each strategy: $100 USDT starting capital |

---

## Executive Summary

This document defines a **dedicated WebSocket paper trading system** built from scratch as a **standalone project**. Designed with production potential, it features:

1. **Easy strategy addition** - Plugin architecture with minimal boilerplate
2. **Isolated portfolios** - Each strategy gets $100 USDT, tracked independently
3. **Detailed logging** - Comprehensive system + strategy logs
4. **Real-time dashboard** - Web UI showing live performance
5. **WebSocket-native** - Built around real-time data from the ground up
6. **Production-ready architecture** - Can evolve into main trading system

### Why Start Fresh?

| Current System (unified_trader.py) | Proposed System (ws_tester.py) |
|-----------------------------------|-------------------------------|
| REST-first, WebSocket bolted on | WebSocket-native from day 1 |
| 1700+ lines, complex modes | Target: <800 lines core |
| 30+ strategies entangled | Clean plugin system |
| Shared portfolio complexity | Isolated $100 per strategy |
| Hard to add new strategies | `strategies/` folder auto-discovery |
| No real-time visibility | Live dashboard included |

---

## Design Principles

### 1. Simplicity Over Features
- One job: test strategies with live WebSocket data
- No dual portfolios, no regime weighting, no experiment presets
- Add complexity only when proven necessary

### 2. Strategy as Plugin
- Drop a file in `strategies/`, it's automatically discovered
- Minimal interface: `generate_signal(data) -> Signal`
- No base class inheritance required (duck typing)

### 3. Logging as First-Class Citizen
- Every decision logged with full context
- Queryable structured logs (JSON Lines)
- Real-time dashboard potential

### 4. Separation of Concerns
- Data layer: WebSocket → normalized data
- Strategy layer: data → signals
- Execution layer: signals → paper fills
- Logging layer: everything → structured logs

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WebSocket Paper Tester                               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        Data Layer (Async)                               │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐ │ │
│  │  │ KrakenWS    │   │ Orderbook   │   │ OHLC        │   │ Trade      │ │ │
│  │  │ Client      │──►│ Manager     │   │ Builder     │   │ Tape       │ │ │
│  │  └─────────────┘   └─────────────┘   └─────────────┘   └────────────┘ │ │
│  │                            │                │                │         │ │
│  │                            └────────────────┴────────────────┘         │ │
│  │                                          │                              │ │
│  │                                          ▼                              │ │
│  │                              ┌─────────────────────┐                    │ │
│  │                              │   DataSnapshot      │                    │ │
│  │                              │   (immutable view)  │                    │ │
│  │                              └─────────────────────┘                    │ │
│  └──────────────────────────────────────┬─────────────────────────────────┘ │
│                                         │                                    │
│  ┌──────────────────────────────────────┴─────────────────────────────────┐ │
│  │                      Strategy Layer (Sync)                              │ │
│  │                                                                          │ │
│  │   strategies/                                                            │ │
│  │   ├── market_making.py      ──┐                                         │ │
│  │   ├── order_flow.py           │  Auto-discovered                        │ │
│  │   ├── triangular_arb.py       ├─►  at startup                           │ │
│  │   ├── mean_reversion.py       │                                         │ │
│  │   └── my_new_strategy.py    ──┘                                         │ │
│  │                                                                          │ │
│  │   Each strategy receives DataSnapshot, returns Signal                   │ │
│  └──────────────────────────────────────┬─────────────────────────────────┘ │
│                                         │                                    │
│  ┌──────────────────────────────────────┴─────────────────────────────────┐ │
│  │                      Execution Layer                                    │ │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐  │ │
│  │  │ Paper Executor  │   │ Position        │   │ Risk Manager        │  │ │
│  │  │ (simulated      │◄──│ Tracker         │◄──│ (per-strategy       │  │ │
│  │  │  fills)         │   │                 │   │  limits)            │  │ │
│  │  └─────────────────┘   └─────────────────┘   └─────────────────────┘  │ │
│  └──────────────────────────────────────┬─────────────────────────────────┘ │
│                                         │                                    │
│  ┌──────────────────────────────────────┴─────────────────────────────────┐ │
│  │                       Logging Layer                                     │ │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐  │ │
│  │  │ System Log      │   │ Strategy Log    │   │ Trade Log           │  │ │
│  │  │ (lifecycle,     │   │ (signals,       │   │ (fills, P&L,        │  │ │
│  │  │  data events)   │   │  decisions)     │   │  positions)         │  │ │
│  │  └─────────────────┘   └─────────────────┘   └─────────────────────┘  │ │
│  │                                │                                       │ │
│  │                                ▼                                       │ │
│  │                    ┌─────────────────────────┐                         │ │
│  │                    │  Unified Log Aggregator │                         │ │
│  │                    │  (correlates all logs)  │                         │ │
│  │                    └─────────────────────────┘                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Data Layer

#### DataSnapshot (Immutable View)

Every strategy tick receives an immutable snapshot of current market state:

```python
@dataclass(frozen=True)
class DataSnapshot:
    """Immutable market data snapshot passed to strategies."""
    timestamp: datetime

    # Current prices
    prices: Dict[str, float]              # {'XRP/USD': 2.35, 'BTC/USD': 104500}

    # OHLC candles (last N)
    candles_1m: Dict[str, List[Candle]]   # {'XRP/USD': [Candle, ...]}
    candles_5m: Dict[str, List[Candle]]

    # Orderbook state
    orderbooks: Dict[str, OrderbookSnapshot]

    # Recent trades
    trades: Dict[str, List[Trade]]        # Last 100 trades per symbol

    # Computed indicators (optional, lazy)
    @cached_property
    def spread(self) -> Dict[str, float]:
        return {sym: ob.spread for sym, ob in self.orderbooks.items()}

    @cached_property
    def vwap(self) -> Dict[str, float]:
        # Calculate from trades
        ...

@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass(frozen=True)
class OrderbookSnapshot:
    bids: Tuple[Tuple[float, float], ...]  # ((price, size), ...)
    asks: Tuple[Tuple[float, float], ...]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

@dataclass(frozen=True)
class Trade:
    timestamp: datetime
    price: float
    size: float
    side: str  # 'buy' or 'sell'
```

#### Why Immutable?

1. **Thread safety** - No locks needed, strategies can't corrupt data
2. **Reproducibility** - Same snapshot = same signal (testable)
3. **Logging** - Can serialize entire snapshot for replay

---

### 2. Strategy Layer

#### Minimal Strategy Interface

```python
# strategies/my_strategy.py

from dataclasses import dataclass
from typing import Optional
from ws_tester.types import DataSnapshot, Signal

# Strategy metadata (required)
STRATEGY_NAME = "my_momentum_strategy"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["XRP/USD", "BTC/USD"]  # Which symbols this strategy trades

# Optional config with defaults
CONFIG = {
    'lookback': 5,
    'threshold': 0.002,
    'position_size': 100,  # USD
}

def generate_signal(data: DataSnapshot, config: dict, state: dict) -> Optional[Signal]:
    """
    Generate trading signal from market data.

    Args:
        data: Immutable market snapshot
        config: Strategy configuration (can be overridden at runtime)
        state: Mutable strategy state (persisted between calls)

    Returns:
        Signal if action needed, None for hold
    """
    symbol = "XRP/USD"
    candles = data.candles_1m.get(symbol, [])

    if len(candles) < config['lookback']:
        return None

    # Simple momentum: price vs N-bar average
    recent_closes = [c.close for c in candles[-config['lookback']:]]
    avg = sum(recent_closes) / len(recent_closes)
    current = data.prices.get(symbol, 0)

    if current > avg * (1 + config['threshold']):
        return Signal(
            action='buy',
            symbol=symbol,
            size=config['position_size'],
            price=current,
            reason=f"Price {current:.4f} > avg {avg:.4f} + threshold"
        )
    elif current < avg * (1 - config['threshold']):
        return Signal(
            action='sell',
            symbol=symbol,
            size=config['position_size'],
            price=current,
            reason=f"Price {current:.4f} < avg {avg:.4f} - threshold"
        )

    return None


def on_fill(fill: dict, state: dict) -> None:
    """
    Optional callback when an order is filled.
    Update strategy state as needed.
    """
    state['last_fill'] = fill
    state['position'] = state.get('position', 0) + (
        fill['size'] if fill['side'] == 'buy' else -fill['size']
    )
```

#### Signal Type

```python
@dataclass
class Signal:
    action: str              # 'buy', 'sell', 'short', 'cover'
    symbol: str              # 'XRP/USD'
    size: float              # Position size in USD or base asset
    price: float             # Reference price (for logging)
    reason: str              # Human-readable explanation

    # Optional
    order_type: str = 'market'      # 'market' or 'limit'
    limit_price: float = None       # For limit orders
    stop_loss: float = None         # Auto stop-loss price
    take_profit: float = None       # Auto take-profit price
    metadata: dict = None           # Strategy-specific data
```

#### Auto-Discovery

```python
# ws_tester/strategy_loader.py

import importlib
import sys
from pathlib import Path
from typing import Dict, Any

def discover_strategies(strategies_dir: str = "strategies") -> Dict[str, Any]:
    """
    Auto-discover strategies from directory.

    Each .py file in strategies/ with STRATEGY_NAME is loaded.
    """
    strategies = {}
    strategies_path = Path(strategies_dir)

    if not strategies_path.exists():
        return strategies

    # Add to Python path
    sys.path.insert(0, str(strategies_path.parent))

    for file in strategies_path.glob("*.py"):
        if file.name.startswith("_"):
            continue

        module_name = f"{strategies_dir}.{file.stem}"
        try:
            module = importlib.import_module(module_name)

            # Check for required attributes
            if hasattr(module, 'STRATEGY_NAME') and hasattr(module, 'generate_signal'):
                strategies[module.STRATEGY_NAME] = {
                    'module': module,
                    'name': module.STRATEGY_NAME,
                    'version': getattr(module, 'STRATEGY_VERSION', '0.0.0'),
                    'symbols': getattr(module, 'SYMBOLS', []),
                    'config': getattr(module, 'CONFIG', {}),
                    'generate_signal': module.generate_signal,
                    'on_fill': getattr(module, 'on_fill', None),
                    'state': {},  # Mutable state
                }
                print(f"  + Loaded: {module.STRATEGY_NAME} v{strategies[module.STRATEGY_NAME]['version']}")
        except Exception as e:
            print(f"  ! Failed to load {file.name}: {e}")

    return strategies
```

---

### 3. Execution Layer

#### Isolated Portfolio Model

Each strategy operates with its own **isolated $100 USDT portfolio**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Isolated Portfolio Architecture                      │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ market_making    │  │ order_flow       │  │ triangular_arb   │       │
│  │ ────────────────│  │ ────────────────│  │ ────────────────│       │
│  │ USDT: $100.00   │  │ USDT: $100.00   │  │ USDT: $100.00   │       │
│  │ XRP:  0.00      │  │ XRP:  0.00      │  │ XRP:  0.00      │       │
│  │ BTC:  0.00      │  │ BTC:  0.00      │  │ BTC:  0.00      │       │
│  │ ────────────────│  │ ────────────────│  │ ────────────────│       │
│  │ Trades: 0       │  │ Trades: 0       │  │ Trades: 0       │       │
│  │ P&L: $0.00      │  │ P&L: $0.00      │  │ P&L: $0.00      │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Aggregate Dashboard View                       │   │
│  │  Total Capital: $300.00 | Combined P&L: $0.00 | Win Rate: 0%     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- Clear strategy performance comparison (apples to apples)
- One bad strategy can't drain capital from others
- Easy to identify which strategies are profitable
- Simple to add/remove strategies without rebalancing

#### Strategy Portfolio

```python
# ws_tester/portfolio.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import uuid

STARTING_CAPITAL = 100.0  # $100 USDT per strategy

@dataclass
class Position:
    symbol: str
    side: str           # 'long' or 'short'
    size: float         # Base asset amount
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    highest_price: float = 0.0   # For trailing stop
    lowest_price: float = float('inf')

@dataclass
class Fill:
    fill_id: str
    timestamp: datetime
    symbol: str
    side: str
    size: float
    price: float
    fee: float
    signal_reason: str
    pnl: float = 0.0     # Realized P&L for this fill

@dataclass
class StrategyPortfolio:
    """Isolated portfolio for a single strategy."""
    strategy_name: str
    starting_capital: float = STARTING_CAPITAL

    # Balances
    usdt: float = field(default=STARTING_CAPITAL)
    assets: Dict[str, float] = field(default_factory=dict)  # {'XRP': 0, 'BTC': 0}

    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)  # symbol -> Position

    # History
    fills: List[Fill] = field(default_factory=list)

    # Metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = STARTING_CAPITAL

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity (USDT + position value)."""
        equity = self.usdt
        for asset, amount in self.assets.items():
            if amount > 0:
                price = prices.get(f"{asset}/USD", 0)
                equity += amount * price
        return equity

    def update_drawdown(self, prices: Dict[str, float]):
        """Track max drawdown."""
        equity = self.get_equity(prices)
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def get_win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def get_roi(self, prices: Dict[str, float]) -> float:
        """Calculate return on investment percentage."""
        equity = self.get_equity(prices)
        return ((equity - self.starting_capital) / self.starting_capital) * 100

    def to_dict(self, prices: Dict[str, float]) -> dict:
        """Serialize for dashboard/logging."""
        return {
            'strategy': self.strategy_name,
            'usdt': self.usdt,
            'assets': dict(self.assets),
            'equity': self.get_equity(prices),
            'pnl': self.total_pnl,
            'roi_pct': self.get_roi(prices),
            'trades': self.total_trades,
            'win_rate': self.get_win_rate(),
            'max_drawdown_pct': self.max_drawdown * 100,
            'open_positions': len(self.positions)
        }


class PortfolioManager:
    """Manages isolated portfolios for all strategies."""

    FEE_RATE = 0.001  # 0.1%

    def __init__(self, strategy_names: List[str], starting_capital: float = STARTING_CAPITAL):
        self.starting_capital = starting_capital
        self.portfolios: Dict[str, StrategyPortfolio] = {}

        for name in strategy_names:
            self.portfolios[name] = StrategyPortfolio(
                strategy_name=name,
                starting_capital=starting_capital,
                usdt=starting_capital
            )

    def add_strategy(self, name: str):
        """Add a new strategy portfolio at runtime."""
        if name not in self.portfolios:
            self.portfolios[name] = StrategyPortfolio(
                strategy_name=name,
                starting_capital=self.starting_capital,
                usdt=self.starting_capital
            )

    def get_portfolio(self, strategy: str) -> StrategyPortfolio:
        """Get portfolio for a strategy."""
        return self.portfolios.get(strategy)

    def get_aggregate(self, prices: Dict[str, float]) -> dict:
        """Get aggregate stats across all strategies."""
        total_equity = 0
        total_pnl = 0
        total_trades = 0
        total_wins = 0

        for p in self.portfolios.values():
            total_equity += p.get_equity(prices)
            total_pnl += p.total_pnl
            total_trades += p.total_trades
            total_wins += p.winning_trades

        return {
            'total_strategies': len(self.portfolios),
            'total_capital': len(self.portfolios) * self.starting_capital,
            'total_equity': total_equity,
            'total_pnl': total_pnl,
            'total_roi_pct': (total_pnl / (len(self.portfolios) * self.starting_capital)) * 100 if self.portfolios else 0,
            'total_trades': total_trades,
            'win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0
        }

    def get_leaderboard(self, prices: Dict[str, float]) -> List[dict]:
        """Get strategies ranked by P&L."""
        stats = [p.to_dict(prices) for p in self.portfolios.values()]
        return sorted(stats, key=lambda x: x['pnl'], reverse=True)

#### Paper Executor

```python
# ws_tester/executor.py

class PaperExecutor:
    """
    Executes signals against isolated strategy portfolios.

    Features:
    - Slippage simulation based on orderbook
    - Fee calculation (0.1%)
    - Per-strategy position tracking
    - Auto stop-loss / take-profit
    """

    FEE_RATE = 0.001  # 0.1%

    def __init__(self, portfolio_manager: PortfolioManager):
        self.portfolio_manager = portfolio_manager

    def execute(self, signal: Signal, strategy: str, data: DataSnapshot) -> Optional[Fill]:
        """Execute signal against strategy's isolated portfolio."""
        portfolio = self.portfolio_manager.get_portfolio(strategy)
        if not portfolio:
            return None

        ob = data.orderbooks.get(signal.symbol)
        if not ob:
            return None

        # Simulate slippage from orderbook
        if signal.action in ['buy', 'cover']:
            execution_price = ob.best_ask * 1.0005  # 0.05% slippage
        else:
            execution_price = ob.best_bid * 0.9995

        # Calculate size in base asset
        if signal.size <= 0:
            return None
        base_size = signal.size / execution_price
        base_asset = signal.symbol.split('/')[0]

        # Execute based on action
        pnl = 0.0
        if signal.action == 'buy':
            cost = base_size * execution_price * (1 + self.FEE_RATE)
            if portfolio.usdt < cost:
                # Reduce size to available balance
                base_size = (portfolio.usdt / execution_price) * (1 - self.FEE_RATE)
                cost = portfolio.usdt
                if base_size <= 0:
                    return None

            portfolio.usdt -= cost
            portfolio.assets[base_asset] = portfolio.assets.get(base_asset, 0) + base_size

            # Track position
            portfolio.positions[signal.symbol] = Position(
                symbol=signal.symbol,
                side='long',
                size=base_size,
                entry_price=execution_price,
                entry_time=data.timestamp,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )

        elif signal.action == 'sell':
            available = portfolio.assets.get(base_asset, 0)
            if available <= 0:
                return None

            base_size = min(base_size, available)
            proceeds = base_size * execution_price * (1 - self.FEE_RATE)

            # Calculate P&L if closing position
            if signal.symbol in portfolio.positions:
                pos = portfolio.positions[signal.symbol]
                pnl = (execution_price - pos.entry_price) * base_size
                portfolio.total_pnl += pnl
                if pnl > 0:
                    portfolio.winning_trades += 1
                else:
                    portfolio.losing_trades += 1
                del portfolio.positions[signal.symbol]

            portfolio.assets[base_asset] -= base_size
            portfolio.usdt += proceeds

        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4())[:8],
            timestamp=data.timestamp,
            symbol=signal.symbol,
            side=signal.action,
            size=base_size,
            price=execution_price,
            fee=base_size * execution_price * self.FEE_RATE,
            signal_reason=signal.reason,
            pnl=pnl
        )

        portfolio.fills.append(fill)
        portfolio.total_trades += 1
        portfolio.update_drawdown(data.prices)

        return fill

    def check_stops(self, data: DataSnapshot) -> List[Tuple[str, Signal]]:
        """Check all strategy positions for stop-loss / take-profit."""
        signals = []

        for strategy, portfolio in self.portfolio_manager.portfolios.items():
            for symbol, pos in list(portfolio.positions.items()):
                price = data.prices.get(symbol, 0)
                if not price:
                    continue

                # Update tracking prices
                pos.highest_price = max(pos.highest_price, price)
                pos.lowest_price = min(pos.lowest_price, price)

                trigger = None
                if pos.side == 'long':
                    if pos.stop_loss and price <= pos.stop_loss:
                        trigger = 'stop_loss'
                    elif pos.take_profit and price >= pos.take_profit:
                        trigger = 'take_profit'

                if trigger:
                    signals.append((strategy, Signal(
                        action='sell',
                        symbol=symbol,
                        size=pos.size * price,
                        price=price,
                        reason=f"{trigger} triggered at {price:.4f}",
                        metadata={'trigger': trigger}
                    )))

        return signals
```

---

### 4. Logging Layer

#### Log Structure

Three separate log streams that can be correlated:

```
logs/
├── system/
│   └── ws_tester_20251213_143022.jsonl     # System events
├── strategies/
│   ├── market_making_20251213_143022.jsonl # Per-strategy decisions
│   ├── order_flow_20251213_143022.jsonl
│   └── triangular_arb_20251213_143022.jsonl
├── trades/
│   └── fills_20251213_143022.jsonl          # All executions
└── aggregated/
    └── unified_20251213_143022.jsonl        # Correlated view
```

#### Log Schemas

```python
# System Log Entry
{
    "timestamp": "2025-12-13T14:30:22.123456",
    "event": "data_update",           # or: ws_connect, ws_disconnect, strategy_loaded, etc.
    "level": "INFO",
    "details": {
        "symbols": ["XRP/USD", "BTC/USD"],
        "prices": {"XRP/USD": 2.3542, "BTC/USD": 104523.50},
        "spread": {"XRP/USD": 0.0003, "BTC/USD": 1.50}
    }
}

# Strategy Log Entry
{
    "timestamp": "2025-12-13T14:30:22.125000",
    "strategy": "market_making",
    "event": "signal_generated",      # or: signal_rejected, no_signal, error
    "correlation_id": "abc123",       # Links to trade log
    "data_snapshot_hash": "sha256:...",  # Can reproduce exact conditions
    "signal": {
        "action": "buy",
        "symbol": "XRP/USD",
        "size": 100,
        "price": 2.3542,
        "reason": "Spread capture opportunity"
    },
    "indicators": {
        "spread": 0.0003,
        "inventory": -50,
        "fair_value": 2.3541
    },
    "latency_us": 125                 # Microseconds to generate signal
}

# Trade Log Entry
{
    "timestamp": "2025-12-13T14:30:22.130000",
    "event": "fill",
    "correlation_id": "abc123",
    "fill": {
        "fill_id": "f1234567",
        "symbol": "XRP/USD",
        "side": "buy",
        "size": 42.48,
        "price": 2.3545,
        "fee": 0.10,
        "slippage": 0.0003
    },
    "strategy": "market_making",
    "portfolio_after": {
        "USD": 9900.00,
        "XRP": 42.48
    },
    "position": {
        "symbol": "XRP/USD",
        "side": "long",
        "size": 42.48,
        "entry_price": 2.3545
    }
}

# Aggregated Log Entry (combines all)
{
    "timestamp": "2025-12-13T14:30:22.130000",
    "correlation_id": "abc123",
    "sequence": 12345,
    "data": {
        "prices": {"XRP/USD": 2.3542},
        "spread": {"XRP/USD": 0.0003}
    },
    "strategy": "market_making",
    "signal": {
        "action": "buy",
        "symbol": "XRP/USD",
        "reason": "Spread capture"
    },
    "execution": {
        "filled": true,
        "price": 2.3545,
        "slippage": 0.0003
    },
    "portfolio": {
        "equity": 9999.90,
        "positions": 1,
        "pnl_session": -0.10
    }
}
```

#### Logger Implementation

```python
# ws_tester/logger.py

import json
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
import threading
from queue import Queue

@dataclass
class LogConfig:
    base_dir: str = "logs"
    compress: bool = False           # Gzip old logs
    max_file_size_mb: int = 100      # Rotate at 100MB
    buffer_size: int = 100           # Flush every N entries
    enable_aggregated: bool = True   # Write unified log

class TesterLogger:
    """
    Structured logging for WebSocket paper tester.

    Features:
    - Separate streams for system, strategies, trades
    - Correlation IDs to link related events
    - Buffered async writing
    - Optional aggregated view
    """

    def __init__(self, session_id: str, config: LogConfig = None):
        self.session_id = session_id
        self.config = config or LogConfig()
        self.sequence = 0
        self._lock = threading.Lock()
        self._write_queue = Queue()

        # Create log directories
        self.log_dir = Path(self.config.base_dir)
        self.system_dir = self.log_dir / "system"
        self.strategy_dir = self.log_dir / "strategies"
        self.trades_dir = self.log_dir / "trades"
        self.aggregated_dir = self.log_dir / "aggregated"

        for d in [self.system_dir, self.strategy_dir, self.trades_dir, self.aggregated_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Open file handles
        self._files: Dict[str, Any] = {}
        self._open_files()

        # Start async writer thread
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _open_files(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._files['system'] = open(self.system_dir / f"ws_tester_{ts}.jsonl", 'a')
        self._files['trades'] = open(self.trades_dir / f"fills_{ts}.jsonl", 'a')
        if self.config.enable_aggregated:
            self._files['aggregated'] = open(self.aggregated_dir / f"unified_{ts}.jsonl", 'a')

    def _get_strategy_file(self, strategy: str):
        if strategy not in self._files:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._files[strategy] = open(
                self.strategy_dir / f"{strategy}_{ts}.jsonl", 'a'
            )
        return self._files[strategy]

    def _writer_loop(self):
        """Background thread that writes log entries."""
        buffer = []
        while True:
            try:
                entry = self._write_queue.get(timeout=1.0)
                buffer.append(entry)

                if len(buffer) >= self.config.buffer_size:
                    self._flush_buffer(buffer)
                    buffer = []
            except:
                if buffer:
                    self._flush_buffer(buffer)
                    buffer = []

    def _flush_buffer(self, buffer):
        for stream, data in buffer:
            try:
                f = self._files.get(stream) or self._get_strategy_file(stream)
                f.write(json.dumps(data) + '\n')
                f.flush()
            except Exception as e:
                print(f"Log write error: {e}")

    def _generate_correlation_id(self) -> str:
        with self._lock:
            self.sequence += 1
            return f"{self.session_id[:8]}-{self.sequence:06d}"

    def log_system(self, event: str, level: str = "INFO", details: Dict = None):
        """Log system event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "level": level,
            "details": details or {}
        }
        self._write_queue.put(('system', entry))

    def log_signal(self, strategy: str, signal: Optional[Any],
                   data_hash: str, indicators: Dict, latency_us: int) -> str:
        """Log strategy signal. Returns correlation_id."""
        correlation_id = self._generate_correlation_id()

        entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "event": "signal_generated" if signal else "no_signal",
            "correlation_id": correlation_id,
            "data_snapshot_hash": data_hash,
            "signal": asdict(signal) if signal else None,
            "indicators": indicators,
            "latency_us": latency_us
        }
        self._write_queue.put((strategy, entry))

        return correlation_id

    def log_fill(self, fill: Any, correlation_id: str, portfolio: Dict, position: Dict = None):
        """Log trade execution."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "fill",
            "correlation_id": correlation_id,
            "fill": asdict(fill) if hasattr(fill, '__dataclass_fields__') else fill,
            "strategy": fill.strategy if hasattr(fill, 'strategy') else 'unknown',
            "portfolio_after": portfolio,
            "position": position
        }
        self._write_queue.put(('trades', entry))

    def log_aggregated(self, correlation_id: str, data: Dict, strategy: str,
                       signal: Any, execution: Dict, portfolio: Dict):
        """Log unified aggregated entry."""
        if not self.config.enable_aggregated:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": correlation_id,
            "sequence": self.sequence,
            "data": data,
            "strategy": strategy,
            "signal": asdict(signal) if signal and hasattr(signal, '__dataclass_fields__') else signal,
            "execution": execution,
            "portfolio": portfolio
        }
        self._write_queue.put(('aggregated', entry))

    def close(self):
        """Flush and close all log files."""
        for f in self._files.values():
            try:
                f.flush()
                f.close()
            except:
                pass
```

---

## Main Application

### ws_tester.py

```python
#!/usr/bin/env python3
"""
WebSocket Paper Trading Tester
Lightweight, strategy-focused paper trading with live WebSocket data.

Usage:
    python ws_tester.py                    # Run with default strategies
    python ws_tester.py --duration 60      # Run for 60 minutes
    python ws_tester.py --strategies mm,of # Only market_making and order_flow
    python ws_tester.py --config test.yaml # Custom config
"""

import asyncio
import argparse
import time
import hashlib
from datetime import datetime
from pathlib import Path

from ws_tester.data_layer import KrakenWSClient, DataManager
from ws_tester.strategy_loader import discover_strategies
from ws_tester.paper_executor import PaperExecutor
from ws_tester.logger import TesterLogger, LogConfig
from ws_tester.types import DataSnapshot, Signal


class WebSocketPaperTester:
    """Main application coordinator."""

    def __init__(self,
                 symbols: list = None,
                 strategies_dir: str = "strategies",
                 log_config: LogConfig = None):

        self.symbols = symbols or ['XRP/USD', 'BTC/USD', 'XRP/BTC']
        self.session_id = f"wst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize components
        self.logger = TesterLogger(self.session_id, log_config)
        self.data_manager = DataManager(self.symbols)
        self.executor = PaperExecutor()

        # Load strategies
        self.strategies = discover_strategies(strategies_dir)
        self.logger.log_system("strategies_loaded", details={
            "count": len(self.strategies),
            "names": list(self.strategies.keys())
        })

        # Stats
        self.tick_count = 0
        self.signal_count = 0
        self.fill_count = 0

    async def run(self, duration_minutes: int = 60, interval_ms: int = 100):
        """Main async run loop."""

        print(f"\n{'='*60}")
        print("WebSocket Paper Trading Tester")
        print(f"{'='*60}")
        print(f"Session: {self.session_id}")
        print(f"Symbols: {self.symbols}")
        print(f"Strategies: {list(self.strategies.keys())}")
        print(f"Duration: {duration_minutes} min | Interval: {interval_ms}ms")
        print(f"{'='*60}\n")

        self.logger.log_system("session_start", details={
            "session_id": self.session_id,
            "symbols": self.symbols,
            "strategies": list(self.strategies.keys()),
            "duration_minutes": duration_minutes
        })

        # Connect to WebSocket
        ws_client = KrakenWSClient(self.symbols)
        await ws_client.connect()
        await ws_client.subscribe(['trade', 'ticker', 'book'])

        self.logger.log_system("ws_connected", details={"url": ws_client.url})

        # Start data collection task
        data_task = asyncio.create_task(
            ws_client.run_forever(self.data_manager.on_message)
        )

        # Main trading loop
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_report = start_time

        try:
            while time.time() < end_time:
                self.tick_count += 1

                # Get immutable data snapshot
                snapshot = self.data_manager.get_snapshot()
                if not snapshot:
                    await asyncio.sleep(interval_ms / 1000)
                    continue

                # Hash for reproducibility
                data_hash = hashlib.sha256(
                    str(snapshot.prices).encode()
                ).hexdigest()[:16]

                # Check stop-loss / take-profit
                stop_signals = self.executor.check_stops(snapshot)
                for sig in stop_signals:
                    fill = self.executor.execute(sig, snapshot)
                    if fill:
                        self.fill_count += 1
                        self.logger.log_fill(
                            fill,
                            f"auto-{self.tick_count}",
                            {"USD": self.executor.balances.get('USD', 0)},
                            None
                        )

                # Run each strategy
                for name, strat in self.strategies.items():
                    start_us = time.perf_counter_ns() // 1000

                    try:
                        signal = strat['generate_signal'](
                            snapshot,
                            strat['config'],
                            strat['state']
                        )
                    except Exception as e:
                        self.logger.log_system("strategy_error", "ERROR", {
                            "strategy": name,
                            "error": str(e)
                        })
                        continue

                    latency_us = (time.perf_counter_ns() // 1000) - start_us

                    # Log signal (or no-signal)
                    correlation_id = self.logger.log_signal(
                        name, signal, data_hash,
                        indicators=strat['state'].get('indicators', {}),
                        latency_us=latency_us
                    )

                    if signal:
                        self.signal_count += 1
                        signal.metadata = signal.metadata or {}
                        signal.metadata['strategy'] = name

                        # Execute
                        fill = self.executor.execute(signal, snapshot)

                        if fill:
                            self.fill_count += 1

                            # Notify strategy
                            if strat['on_fill']:
                                strat['on_fill'](fill.__dict__, strat['state'])

                            # Log fill
                            self.logger.log_fill(
                                fill,
                                correlation_id,
                                {
                                    "USD": self.executor.balances.get('USD', 0),
                                    "equity": self.executor.get_equity(snapshot.prices)
                                },
                                self.executor.positions.get(signal.symbol)
                            )

                            print(f"[{name}] {signal.action.upper()} {signal.symbol} "
                                  f"@ {fill.price:.4f} | {signal.reason}")

                # Periodic status report
                if time.time() - last_report > 30:
                    equity = self.executor.get_equity(snapshot.prices)
                    print(f"\n--- Status @ {datetime.now().strftime('%H:%M:%S')} ---")
                    print(f"Ticks: {self.tick_count} | Signals: {self.signal_count} | Fills: {self.fill_count}")
                    print(f"Equity: ${equity:,.2f} | Positions: {len(self.executor.positions)}")
                    print(f"Prices: {snapshot.prices}")
                    print()
                    last_report = time.time()

                await asyncio.sleep(interval_ms / 1000)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            # Cleanup
            data_task.cancel()
            await ws_client.close()

            # Final report
            final_equity = self.executor.get_equity(snapshot.prices if snapshot else {})

            print(f"\n{'='*60}")
            print("SESSION COMPLETE")
            print(f"{'='*60}")
            print(f"Duration: {(time.time() - start_time) / 60:.1f} minutes")
            print(f"Ticks: {self.tick_count}")
            print(f"Signals: {self.signal_count}")
            print(f"Fills: {self.fill_count}")
            print(f"Final Equity: ${final_equity:,.2f}")
            print(f"P&L: ${final_equity - 10000:+,.2f}")
            print(f"\nStrategy P&L:")
            for strat, pnl in self.executor.strategy_pnl.items():
                print(f"  {strat}: ${pnl:+,.2f}")
            print(f"{'='*60}\n")

            self.logger.log_system("session_end", details={
                "duration_minutes": (time.time() - start_time) / 60,
                "ticks": self.tick_count,
                "signals": self.signal_count,
                "fills": self.fill_count,
                "final_equity": final_equity,
                "strategy_pnl": self.executor.strategy_pnl
            })

            self.logger.close()


def main():
    parser = argparse.ArgumentParser(description="WebSocket Paper Trading Tester")
    parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
    parser.add_argument('--interval', type=int, default=100, help='Loop interval in ms')
    parser.add_argument('--symbols', type=str, default='XRP/USD,BTC/USD,XRP/BTC',
                       help='Comma-separated symbols')
    parser.add_argument('--strategies-dir', type=str, default='strategies',
                       help='Directory containing strategy files')

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]

    tester = WebSocketPaperTester(
        symbols=symbols,
        strategies_dir=args.strategies_dir
    )

    asyncio.run(tester.run(
        duration_minutes=args.duration,
        interval_ms=args.interval
    ))


if __name__ == "__main__":
    main()
```

---

## File Structure

```
ws_paper_tester/                    # New standalone project
├── ws_tester.py                    # Main entry point
├── requirements.txt                # Dependencies
├── config.yaml                     # Default configuration
│
├── ws_tester/                      # Core library
│   ├── __init__.py
│   ├── types.py                    # DataSnapshot, Signal, Candle, etc.
│   ├── data_layer.py               # KrakenWSClient, DataManager
│   ├── strategy_loader.py          # Auto-discovery
│   ├── paper_executor.py           # Simulated execution
│   ├── logger.py                   # Structured logging
│   └── utils.py                    # Helpers
│
├── strategies/                     # Drop-in strategies
│   ├── __init__.py
│   ├── market_making.py
│   ├── order_flow.py
│   ├── triangular_arb.py
│   ├── momentum_breakout.py
│   └── mean_reversion.py
│
├── logs/                           # Generated logs
│   ├── system/
│   ├── strategies/
│   ├── trades/
│   └── aggregated/
│
└── tests/                          # Unit tests
    ├── test_data_layer.py
    ├── test_executor.py
    └── test_strategies.py
```

---

## Migration Path

### Rewriting Existing Strategies

When migrating from `unified_trader.py` strategies:

| Old Pattern | New Pattern |
|-------------|-------------|
| Class inheriting `BaseStrategy` | Simple module with `generate_signal()` |
| `__init__(self, config)` | Module-level `CONFIG = {...}` |
| `self.config['param']` | `config['param']` (passed in) |
| Instance state in `self.x` | State dict passed in `state['x']` |
| Complex `validate_signal()` | Return `None` for no signal |

**Example Migration:**

```python
# OLD: src/strategies/mean_reversion_vwap/strategy.py
class MeanReversionVWAP(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.rsi_oversold = config.get('rsi_oversold', 35)

    def generate_signals(self, data):
        df = data.get('XRP/USDT')
        # ... calculate RSI, VWAP ...
        if rsi < self.rsi_oversold:
            return {'action': 'buy', 'symbol': 'XRP/USDT', ...}
        return {'action': 'hold'}

# NEW: strategies/mean_reversion.py
STRATEGY_NAME = "mean_reversion"
STRATEGY_VERSION = "2.0.0"
SYMBOLS = ["XRP/USD"]

CONFIG = {
    'rsi_oversold': 35,
    'rsi_overbought': 65,
    'vwap_dev_threshold': 0.003,
}

def generate_signal(data: DataSnapshot, config: dict, state: dict) -> Optional[Signal]:
    candles = data.candles_5m.get("XRP/USD", [])
    if len(candles) < 20:
        return None

    # Calculate RSI
    closes = [c.close for c in candles[-20:]]
    rsi = calculate_rsi(closes)

    # Store for logging
    state['indicators'] = {'rsi': rsi}

    if rsi < config['rsi_oversold']:
        return Signal(
            action='buy',
            symbol='XRP/USD',
            size=100,
            price=data.prices['XRP/USD'],
            reason=f"RSI oversold: {rsi:.1f}"
        )

    return None
```

---

## Comparison: unified_trader.py vs ws_tester.py

| Feature | unified_trader.py | ws_tester.py |
|---------|-------------------|--------------|
| **Lines of code** | ~1700 (orchestrator) + ~650 (trader) | ~500 target |
| **Data source** | REST primary, WS optional | WebSocket native |
| **Strategy interface** | BaseStrategy class inheritance | Simple function |
| **Adding strategies** | Register in StrategyRegistry, create directory | Drop file in `strategies/` |
| **Portfolio modes** | Dual (70/30), isolated, shared | Single unified |
| **Regime detection** | Yes (chop, trend_up, etc.) | No (strategy handles it) |
| **Weighted voting** | Yes | No (each strategy independent) |
| **Logging** | StrategyLoggerManager | Dedicated TesterLogger |
| **Use case** | Production paper/live trading | Rapid strategy development |

---

## When to Use Which

### Use unified_trader.py when:
- Running production paper trading
- Need dual portfolio allocation
- Want regime-based strategy weighting
- Need isolated portfolios per strategy

### Use ws_tester.py when:
- Developing new strategies
- Testing HFT concepts
- Want minimal boilerplate
- Need detailed replay logs
- Rapid iteration on ideas

---

## Next Steps

1. **Decide**: Is this a good direction? Or extend unified_trader.py instead?
2. **Prototype**: Build minimal ws_tester.py with one strategy
3. **Logging first**: Get logging infrastructure right before strategies
4. **Migrate incrementally**: Port one strategy at a time

---

## Real-Time Dashboard

### Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Dashboard Architecture                               │
│                                                                              │
│  ┌─────────────────┐     WebSocket      ┌─────────────────────────────────┐ │
│  │  ws_tester.py   │ ◄───────────────► │  Dashboard Server (FastAPI)     │ │
│  │  (Trading Loop) │    /ws/updates     │  - REST API for history         │ │
│  └────────┬────────┘                    │  - WebSocket for live updates   │ │
│           │                             └───────────────┬─────────────────┘ │
│           │ Publishes                                   │                    │
│           ▼                                             ▼                    │
│  ┌─────────────────┐                    ┌─────────────────────────────────┐ │
│  │  Event Bus      │                    │  Web UI (HTML/JS)               │ │
│  │  (asyncio Queue)│                    │  - Strategy Leaderboard         │ │
│  └─────────────────┘                    │  - Live Price Charts            │ │
│                                         │  - Trade Feed                   │ │
│                                         │  - Per-Strategy Details         │ │
│                                         └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dashboard Components

#### 1. Strategy Leaderboard (Main View)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  WEBSOCKET PAPER TESTER - Live Dashboard                    Session: 2h 15m  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  AGGREGATE: $312.45 (+$12.45 / +4.15%)  │  Strategies: 3  │  Trades: 47     │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  #   Strategy          Equity    P&L       ROI     Trades  Win%   Drawdown  │
│  ─────────────────────────────────────────────────────────────────────────── │
│  1   order_flow        $108.23   +$8.23   +8.23%    15    66.7%    2.1%     │
│  2   triangular_arb    $104.12   +$4.12   +4.12%    22    54.5%    1.8%     │
│  3   market_making     $100.10   +$0.10   +0.10%    10    50.0%    3.2%     │
├──────────────────────────────────────────────────────────────────────────────┤
│  PRICES: XRP/USD $2.3542 (+0.5%)  │  BTC/USD $104,523 (-0.2%)               │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 2. Strategy Detail View

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STRATEGY: order_flow                                        [Back to List]  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │ PORTFOLIO                   │  │ PERFORMANCE CHART                   │   │
│  │ ───────────────────────     │  │                              ___    │   │
│  │ USDT:  $58.23              │  │               ___/\__/\    _/       │   │
│  │ XRP:   21.25 ($50.00)      │  │     ____/\__/           \_/         │   │
│  │ ───────────────────────     │  │ $100 ─────────────────────────────  │   │
│  │ Total: $108.23 (+8.23%)    │  │ 0h        1h         2h             │   │
│  └─────────────────────────────┘  └─────────────────────────────────────┘   │
│                                                                              │
│  OPEN POSITIONS                                                              │
│  ─────────────────────────────────────────────────────────────────────────── │
│  XRP/USD  LONG  21.25 @ $2.3510  │  Current: $2.3542  │  P&L: +$0.68 (+1.4%) │
│                                                                              │
│  RECENT TRADES                                                               │
│  ─────────────────────────────────────────────────────────────────────────── │
│  14:32:15  BUY   XRP/USD   21.25 @ $2.3510   "Buy pressure imbalance 2.3x"  │
│  14:28:03  SELL  XRP/USD   18.50 @ $2.3580   "Take profit +1.2%"  +$0.52    │
│  14:25:41  BUY   XRP/USD   18.50 @ $2.3298   "Volume spike with imbalance"  │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 3. Live Trade Feed

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  LIVE TRADE FEED                                              [Auto-scroll] │
├──────────────────────────────────────────────────────────────────────────────┤
│ 14:32:15  order_flow      BUY   XRP/USD  21.25 @ $2.3510   $50.00           │
│ 14:31:02  triangular_arb  SELL  BTC/USD  0.001 @ $104,520  +$0.12           │
│ 14:30:58  triangular_arb  BUY   XRP/USD  45.00 @ $2.3508   $105.78          │
│ 14:30:45  market_making   SELL  XRP/USD  10.00 @ $2.3515   +$0.03           │
│ 14:30:12  market_making   BUY   XRP/USD  10.00 @ $2.3512   $23.51           │
│ ...                                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Dashboard Server Implementation

```python
# ws_tester/dashboard/server.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
from typing import List, Dict
from datetime import datetime

app = FastAPI(title="WS Paper Tester Dashboard")

# Store connected dashboard clients
dashboard_clients: List[WebSocket] = []

# Latest state (updated by trading loop)
latest_state: Dict = {
    "timestamp": None,
    "prices": {},
    "strategies": [],
    "aggregate": {},
    "recent_trades": []
}


class DashboardPublisher:
    """Publishes updates from trading loop to dashboard clients."""

    def __init__(self):
        self.queue = asyncio.Queue()

    async def publish(self, event_type: str, data: dict):
        """Queue an update for broadcast."""
        await self.queue.put({
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })

    async def broadcast_loop(self):
        """Continuously broadcast queued updates."""
        while True:
            event = await self.queue.get()
            message = json.dumps(event)

            # Update latest state
            if event["type"] == "state_update":
                latest_state.update(event["data"])
            elif event["type"] == "trade":
                latest_state["recent_trades"].insert(0, event["data"])
                latest_state["recent_trades"] = latest_state["recent_trades"][:100]

            # Broadcast to all connected clients
            disconnected = []
            for client in dashboard_clients:
                try:
                    await client.send_text(message)
                except:
                    disconnected.append(client)

            for client in disconnected:
                dashboard_clients.remove(client)


publisher = DashboardPublisher()


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live dashboard updates."""
    await websocket.accept()
    dashboard_clients.append(websocket)

    # Send current state on connect
    await websocket.send_text(json.dumps({
        "type": "initial_state",
        "data": latest_state
    }))

    try:
        while True:
            # Keep connection alive, handle client messages if needed
            data = await websocket.receive_text()
            # Could handle dashboard commands here (pause, filter, etc.)
    except WebSocketDisconnect:
        dashboard_clients.remove(websocket)


@app.get("/api/strategies")
async def get_strategies():
    """Get current strategy stats."""
    return latest_state.get("strategies", [])


@app.get("/api/strategy/{name}")
async def get_strategy_detail(name: str):
    """Get detailed stats for a specific strategy."""
    strategies = latest_state.get("strategies", [])
    for s in strategies:
        if s["strategy"] == name:
            return s
    return {"error": "Strategy not found"}


@app.get("/api/trades")
async def get_recent_trades(limit: int = 100):
    """Get recent trades across all strategies."""
    return latest_state.get("recent_trades", [])[:limit]


@app.get("/api/aggregate")
async def get_aggregate():
    """Get aggregate stats across all strategies."""
    return latest_state.get("aggregate", {})


@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Serve dashboard HTML."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WS Paper Tester Dashboard</title>
        <style>
            body { font-family: 'Courier New', monospace; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }
            .header { background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
            .aggregate { display: flex; gap: 30px; font-size: 1.2em; }
            .leaderboard { background: #16213e; padding: 15px; border-radius: 8px; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
            th { color: #888; }
            .positive { color: #4ade80; }
            .negative { color: #f87171; }
            .prices { margin-top: 15px; padding-top: 15px; border-top: 1px solid #333; }
            .trade-feed { background: #16213e; padding: 15px; border-radius: 8px; margin-top: 20px; max-height: 300px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>WEBSOCKET PAPER TESTER - Live Dashboard</h1>
            <div class="aggregate" id="aggregate">Loading...</div>
        </div>
        <div class="leaderboard">
            <h2>Strategy Leaderboard</h2>
            <table id="strategies">
                <thead>
                    <tr><th>#</th><th>Strategy</th><th>Equity</th><th>P&L</th><th>ROI</th><th>Trades</th><th>Win%</th><th>Drawdown</th></tr>
                </thead>
                <tbody></tbody>
            </table>
            <div class="prices" id="prices"></div>
        </div>
        <div class="trade-feed">
            <h2>Live Trade Feed</h2>
            <div id="trades"></div>
        </div>
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws/live`);
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'initial_state' || msg.type === 'state_update') {
                    updateDashboard(msg.data);
                } else if (msg.type === 'trade') {
                    addTrade(msg.data);
                }
            };
            function updateDashboard(state) {
                // Update aggregate
                const agg = state.aggregate || {};
                document.getElementById('aggregate').innerHTML = `
                    Total: $${(agg.total_equity || 0).toFixed(2)} |
                    P&L: <span class="${agg.total_pnl >= 0 ? 'positive' : 'negative'}">$${(agg.total_pnl || 0).toFixed(2)}</span> |
                    Strategies: ${agg.total_strategies || 0} |
                    Trades: ${agg.total_trades || 0}
                `;
                // Update leaderboard
                const tbody = document.querySelector('#strategies tbody');
                tbody.innerHTML = (state.strategies || []).map((s, i) => `
                    <tr>
                        <td>${i+1}</td>
                        <td>${s.strategy}</td>
                        <td>$${s.equity.toFixed(2)}</td>
                        <td class="${s.pnl >= 0 ? 'positive' : 'negative'}">$${s.pnl.toFixed(2)}</td>
                        <td class="${s.roi_pct >= 0 ? 'positive' : 'negative'}">${s.roi_pct.toFixed(2)}%</td>
                        <td>${s.trades}</td>
                        <td>${s.win_rate.toFixed(1)}%</td>
                        <td>${s.max_drawdown_pct.toFixed(1)}%</td>
                    </tr>
                `).join('');
                // Update prices
                const prices = state.prices || {};
                document.getElementById('prices').innerHTML = Object.entries(prices)
                    .map(([sym, price]) => `${sym}: $${price.toFixed(4)}`).join(' | ');
            }
            function addTrade(trade) {
                const div = document.getElementById('trades');
                const html = `<div>${trade.timestamp} | ${trade.strategy} | ${trade.side} ${trade.symbol} @ $${trade.price.toFixed(4)}</div>`;
                div.innerHTML = html + div.innerHTML;
            }
        </script>
    </body>
    </html>
    """


# Mount static files if needed
# app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
```

### Integration with Trading Loop

```python
# In ws_tester.py main loop

async def run(self, ...):
    # Start dashboard server in background
    import uvicorn
    from ws_tester.dashboard.server import app, publisher

    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="warning")
    server = uvicorn.Server(config)

    # Run server in background task
    dashboard_task = asyncio.create_task(server.serve())
    broadcast_task = asyncio.create_task(publisher.broadcast_loop())

    print(f"\nDashboard: http://localhost:8080\n")

    # Main trading loop
    while running:
        # ... execute strategies ...

        # Publish state update every second
        if loop_count % 10 == 0:  # Every 10 ticks at 100ms = 1 second
            await publisher.publish("state_update", {
                "prices": dict(snapshot.prices),
                "strategies": self.portfolio_manager.get_leaderboard(snapshot.prices),
                "aggregate": self.portfolio_manager.get_aggregate(snapshot.prices)
            })

        # Publish trades immediately
        if fill:
            await publisher.publish("trade", {
                "timestamp": fill.timestamp.isoformat(),
                "strategy": strategy_name,
                "symbol": fill.symbol,
                "side": fill.side,
                "size": fill.size,
                "price": fill.price,
                "pnl": fill.pnl
            })

        await asyncio.sleep(interval_ms / 1000)
```

---

## Production Path

### Phase 1: Paper Trading (Current)
- All execution simulated
- Focus on strategy development
- Prove concepts work

### Phase 2: Shadow Mode
- Real orders prepared but not sent
- Compare simulated vs what would have happened
- Validate execution logic

### Phase 3: Limited Live
- Small real capital ($100 per strategy)
- Same isolated portfolio model
- Circuit breakers mandatory

### Phase 4: Full Production
- Increased capital allocation
- Proven strategies only
- Full monitoring and alerting

### Production Requirements Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Thread-safe portfolio | Planned | Add RLock to StrategyPortfolio |
| Connection recovery | Planned | WebSocket auto-reconnect |
| Order reconciliation | Not Started | Match fills with exchange |
| Rate limiting | Planned | Per-exchange limits |
| Circuit breakers | Planned | Max loss per strategy/session |
| Audit logging | Planned | Immutable trade log |
| Alerting | Not Started | Discord/Telegram notifications |
| Backup execution | Not Started | Fallback exchange |

---

## Implementation Plan

### Phase 1: Core Infrastructure (Priority)
1. `ws_tester/types.py` - DataSnapshot, Signal, Candle types
2. `ws_tester/portfolio.py` - StrategyPortfolio, PortfolioManager
3. `ws_tester/executor.py` - PaperExecutor with isolated portfolios
4. `ws_tester/data_layer.py` - KrakenWSClient, DataManager

### Phase 2: Strategy System
5. `ws_tester/strategy_loader.py` - Auto-discovery
6. `strategies/` - 2-3 example strategies
7. Integration testing

### Phase 3: Logging & Dashboard
8. `ws_tester/logger.py` - Structured logging
9. `ws_tester/dashboard/` - FastAPI + WebSocket dashboard
10. End-to-end testing

### Phase 4: Polish
11. CLI arguments and config
12. Documentation
13. 24-48h paper trading test

---

*Document updated 2025-12-13 - APPROVED for implementation*
