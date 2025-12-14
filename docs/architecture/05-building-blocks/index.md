# Building Block View

## System Overview

The trading platform consists of two main subsystems:

```
grok-4_1/
├── src/                      # Main Trading Platform
│   ├── unified_trader.py     # Production paper/live trading
│   ├── strategies/           # 30+ trading strategies
│   └── ...
│
└── ws_paper_tester/          # WebSocket Paper Tester
    ├── ws_tester.py          # Lightweight strategy testing
    ├── ws_tester/            # Core library
    └── strategies/           # Drop-in test strategies
```

## Main Trading Platform (src/)

### Core Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| UnifiedTrader | `src/unified_trader.py` | Main orchestrator, manages trading lifecycle |
| StrategyRegistry | `src/strategy_registry.py` | Strategy discovery and configuration |
| StrategyPortfolio | `src/strategy_portfolio.py` | Portfolio management, dual allocation |
| RiskManager | `src/risk_manager.py` | Position sizing, exposure limits |
| Executor | `src/executor.py` | Order execution, Kraken API |
| DataFetcher | `src/data_fetcher.py` | Market data retrieval |

### Strategy Categories

| Category | Count | Examples |
|----------|-------|----------|
| Momentum | 5 | EMA9 Scalper, Intraday Scalper |
| Mean Reversion | 4 | VWAP, RSI-based |
| Trend Following | 3 | MA Trend, Ichimoku Cloud |
| Arbitrage | 2 | Triangular, Pair Trading |
| Accumulation | 3 | DCA, TWAP, Grid |
| ML-Enhanced | 2 | XRP Momentum LSTM |

---

## WebSocket Paper Tester (ws_paper_tester/)

Lightweight, WebSocket-native system for rapid strategy development.

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    WebSocket Paper Tester                        │
│                                                                  │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐  │
│  │ KrakenWSClient │   │  DataManager   │   │ DataSnapshot   │  │
│  │ (Connection)   │──►│ (Aggregation)  │──►│ (Immutable)    │  │
│  └────────────────┘   └────────────────┘   └───────┬────────┘  │
│                                                     │           │
│  ┌────────────────────────────────────────────────▼─────────┐  │
│  │                   Strategy Layer (Modular Packages)       │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │  │
│  │  │market_making/│ │ order_flow/  │ │mean_reversion/│ ... │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘      │  │
│  │         │                │                │               │  │
│  │         └────────────────┴────────────────┘               │  │
│  │                          │ Signal                         │  │
│  └──────────────────────────┼────────────────────────────────┘  │
│                             ▼                                    │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐  │
│  │ PaperExecutor  │◄──│PortfolioMgr   │   │  TesterLogger  │  │
│  │ (Simulation)   │   │(Isolated $100) │   │ (JSON Lines)   │  │
│  └────────────────┘   └────────────────┘   └────────────────┘  │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Dashboard (FastAPI + WebSocket)              │  │
│  │  - Strategy Leaderboard    - Live Trade Feed             │  │
│  │  - Real-time Prices        - Aggregate Stats             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Building Blocks

#### Data Layer

| Component | File | Responsibility |
|-----------|------|----------------|
| KrakenWSClient | `data_layer.py` | WebSocket connection, message handling |
| DataManager | `data_layer.py` | Candle building, orderbook maintenance |
| SimulatedDataManager | `data_layer.py` | Mock data for testing |

#### Types (Immutable)

| Type | File | Description |
|------|------|-------------|
| DataSnapshot | `types.py` | Complete market state at a point in time |
| Candle | `types.py` | OHLCV candle data |
| Trade | `types.py` | Single trade from tape |
| OrderbookSnapshot | `types.py` | Orderbook state |
| Signal | `types.py` | Trading signal from strategy |
| Position | `types.py` | Open position |
| Fill | `types.py` | Executed trade |

#### Strategy Layer

| Component | File | Responsibility |
|-----------|------|----------------|
| StrategyLoader | `strategy_loader.py` | Auto-discovery, module loading |
| StrategyWrapper | `strategy_loader.py` | Wraps strategy modules with state |

**Strategy Package Structure:**

All strategies are now modular packages with consistent structure:

```
strategies/
├── mean_reversion/     # v4.3.0 - Counter-trend strategy
├── ratio_trading/      # v4.3.1 - Statistical arbitrage
├── order_flow/         # v4.1.1 - Momentum/order imbalance
├── market_making/      # v1.5.0 - Liquidity provision
├── mean_reversion.py   # Backward-compatible shim
├── ratio_trading.py    # Backward-compatible shim
├── order_flow.py       # Backward-compatible shim
└── market_making.py    # Backward-compatible shim
```

Each package typically contains:
- `__init__.py` - Public exports (STRATEGY_NAME, generate_signal, etc.)
- `config.py` - Configuration and enums
- `signals.py` - Signal generation logic
- `lifecycle.py` - on_start, on_fill, on_stop callbacks
- Additional modules as needed (indicators, risk, etc.)

#### Execution Layer

| Component | File | Responsibility |
|-----------|------|----------------|
| PortfolioManager | `portfolio.py` | Manages isolated strategy portfolios |
| StrategyPortfolio | `portfolio.py` | Single strategy's $100 portfolio |
| PaperExecutor | `executor.py` | Simulates order execution |

#### Infrastructure

| Component | File | Responsibility |
|-----------|------|----------------|
| TesterLogger | `logger.py` | Structured logging to JSON Lines |
| DashboardServer | `dashboard/server.py` | FastAPI real-time dashboard |
| DashboardPublisher | `dashboard/server.py` | WebSocket broadcast to clients |

---

## Cross-Cutting Concerns

### Logging

Both systems use structured logging:

| System | Format | Location |
|--------|--------|----------|
| Main Platform | Text + JSON | `logs/` |
| WS Paper Tester | JSON Lines | `ws_paper_tester/logs/` |

### Configuration

| System | Config Files |
|--------|-------------|
| Main Platform | `config/*.yaml`, `strategies_config/*.yaml` |
| WS Paper Tester | `config.yaml`, Strategy `CONFIG` dicts |

### Testing

| System | Test Location | Framework |
|--------|---------------|-----------|
| Main Platform | `tests/` | pytest |
| WS Paper Tester | `ws_paper_tester/tests/` | pytest |

---
*Last updated: 2025-12-14 - Strategy modular package refactoring*
