# 5. Building Block View

## 5.1 Level 1: System Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           External Systems                               │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Kraken WS  │  │ Kraken REST │  │Alternative.me│  │    CoinGecko    │ │
│  │   API v2    │  │     API     │  │  (F&G Index) │  │   (BTC Dom)     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
│         │                │                │                   │          │
└─────────┼────────────────┼────────────────┼───────────────────┼──────────┘
          │                │                │                   │
          ▼                ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      WebSocket Paper Tester                              │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         ws_tester.py (Main)                          ││
│  │  - Orchestrates all components                                       ││
│  │  - Main event loop                                                   ││
│  │  - Configuration loading                                             ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           TimescaleDB                                    │
│                    (Historical Data Storage)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## 5.2 Level 2: Container View

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      WebSocket Paper Tester                              │
│                                                                          │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────────┐│
│  │  Data Layer   │    │Strategy Engine│    │      Execution Layer      ││
│  │               │    │               │    │                           ││
│  │ ┌───────────┐ │    │ ┌───────────┐ │    │ ┌───────────┐ ┌─────────┐││
│  │ │KrakenWS   │ │───▶│ │ Strategy  │ │───▶│ │  Paper    │ │Portfolio│││
│  │ │Client     │ │    │ │ Loader    │ │    │ │ Executor  │ │Manager  │││
│  │ └───────────┘ │    │ └───────────┘ │    │ └───────────┘ └─────────┘││
│  │ ┌───────────┐ │    │ ┌───────────┐ │    │                           ││
│  │ │  Data     │ │    │ │ Regime    │ │    └───────────────────────────┘│
│  │ │ Manager   │ │    │ │ Detector  │ │                                 │
│  │ └───────────┘ │    │ └───────────┘ │    ┌───────────────────────────┐│
│  │ ┌───────────┐ │    │ ┌───────────┐ │    │     Dashboard Layer       ││
│  │ │Simulated  │ │    │ │Indicators │ │    │                           ││
│  │ │DataManager│ │    │ │ Library   │ │    │ ┌───────────┐ ┌─────────┐││
│  │ └───────────┘ │    │ └───────────┘ │    │ │ FastAPI   │ │WebSocket│││
│  └───────────────┘    └───────────────┘    │ │ Server    │ │Broadcast│││
│                                             │ └───────────┘ └─────────┘││
│  ┌───────────────┐    ┌───────────────┐    └───────────────────────────┘│
│  │ Historical    │    │   Logging     │                                 │
│  │ Data System   │    │   System      │                                 │
│  │               │    │               │                                 │
│  │ ┌───────────┐ │    │ ┌───────────┐ │                                 │
│  │ │ Provider  │ │    │ │Structured │ │                                 │
│  │ │           │ │    │ │Logger     │ │                                 │
│  │ └───────────┘ │    │ └───────────┘ │                                 │
│  │ ┌───────────┐ │    └───────────────┘                                 │
│  │ │Gap Filler │ │                                                      │
│  │ └───────────┘ │                                                      │
│  └───────────────┘                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## 5.3 Component Details

### 5.3.1 Data Layer (`ws_tester/data_layer.py`)

| Component | Responsibility |
|-----------|----------------|
| `KrakenWSClient` | WebSocket connection to Kraken v2 API |
| `DataManager` | Aggregates market data into `DataSnapshot` |
| `SimulatedDataManager` | Generates synthetic data for testing |

**Key Data Structures:**
- `DataSnapshot`: Immutable snapshot of market state (prices, candles, orderbook, trades)
- `Candle`: OHLCV data for a time period
- `OrderbookSnapshot`: Top 10 bid/ask levels

### 5.3.2 Strategy Engine (`ws_tester/strategy_loader.py`)

| Component | Responsibility |
|-----------|----------------|
| `StrategyLoader` | Discovers and loads strategy modules |
| `StrategyWrapper` | Wraps strategy functions with error handling |
| Hash verification | Security: validates strategy file integrity |

**Strategy Interface:**
```python
# Required
STRATEGY_NAME: str
STRATEGY_VERSION: str
SYMBOLS: List[str]
CONFIG: Dict[str, Any]
def generate_signal(data, config, state) -> Optional[Signal]

# Optional
def on_start(config, state) -> None
def on_fill(fill, state) -> None
def on_stop(state) -> None
```

### 5.3.3 Execution Layer (`ws_tester/executor.py`, `ws_tester/portfolio.py`)

| Component | Responsibility |
|-----------|----------------|
| `PaperExecutor` | Simulates order execution with fees/slippage |
| `PortfolioManager` | Manages per-strategy isolated portfolios |
| `StrategyPortfolio` | Tracks balances, positions, P&L |

**Key Features:**
- Leveraged longs (up to 1.5x) and shorts (up to 2x)
- Margin call liquidation at 25% maintenance margin
- Stop-loss and take-profit automation
- Per-symbol P&L tracking

### 5.3.4 Indicator Library (`ws_tester/indicators/`)

| Module | Functions |
|--------|-----------|
| `moving_averages.py` | SMA, EMA |
| `oscillators.py` | RSI, ADX, MACD |
| `volatility.py` | ATR, Bollinger Bands, Z-score |
| `correlation.py` | Rolling correlation |
| `volume.py` | Volume ratio, VPIN |
| `flow.py` | Trade flow analysis |
| `trend.py` | Trend slope, trailing stops |

### 5.3.5 Regime Detection (`ws_tester/regime/`)

| Component | Responsibility |
|-----------|----------------|
| `RegimeDetector` | Main orchestrator with hysteresis |
| `CompositeScorer` | Weighted indicator scoring |
| `MTFAnalyzer` | Multi-timeframe confluence |
| `ExternalDataFetcher` | Fear & Greed, BTC Dominance |
| `ParameterRouter` | Strategy parameter adjustments |

**Regimes:** STRONG_BULL, BULL, SIDEWAYS, BEAR, STRONG_BEAR

### 5.3.6 Historical Data (`ws_tester/data/`)

| Component | Responsibility |
|-----------|----------------|
| `HistoricalDataProvider` | Query API for candles |
| `WebSocketDBWriter` | Real-time data persistence |
| `GapFiller` | Detects and fills data gaps |
| `BulkCSVImporter` | Import historical CSV files |

**Storage:** TimescaleDB with hypertables, continuous aggregates, 90%+ compression

### 5.3.7 Dashboard (`ws_tester/dashboard/`)

| Component | Responsibility |
|-----------|----------------|
| `FastAPI` app | REST API endpoints |
| WebSocket broadcast | Real-time updates to clients |
| HTML/JS frontend | Trading dashboard UI |

## 5.4 Data Flow

```
Kraken WS ──▶ DataManager ──▶ DataSnapshot ──▶ Strategy.generate_signal()
                                                        │
                                                        ▼
                                                     Signal
                                                        │
                                                        ▼
                                              PaperExecutor.execute()
                                                        │
                                                        ▼
                                                      Fill
                                                        │
                              ┌──────────────┬──────────┴──────────┬──────────────┐
                              ▼              ▼                     ▼              ▼
                         Portfolio      on_fill()              Logger       Dashboard
                          Update       callback               (JSONL)       (WebSocket)
```
