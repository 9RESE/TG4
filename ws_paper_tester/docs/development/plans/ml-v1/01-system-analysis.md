# System Analysis - Current Architecture

**Document Version**: 1.0
**Created**: 2025-12-16
**Status**: Research Complete

---

## Overview

This document provides a comprehensive analysis of the TG4 trading bot architecture, identifying components relevant to ML integration.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TG4 Trading Bot                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Data Layer  │───▶│  Strategies  │───▶│  Execution & Portfolio   │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│         │                   │                        │                   │
│         ▼                   ▼                        ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ TimescaleDB  │    │ Signal Gen   │    │   Paper/Live Executor    │   │
│  │  + WebSocket │    │ + Indicators │    │   + Position Tracking    │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Analysis

### 1. Data Layer (`ws_tester/data_layer.py`)

**Purpose**: WebSocket data management and candle accumulation

**Key Classes**:
- `KrakenWSClient`: WebSocket connection to Kraken v2 API
- `DataManager`: Candle accumulation and snapshot generation

**ML Integration Points**:
| Component | Integration Opportunity |
|-----------|------------------------|
| `DataManager.candles_1m` | Raw features for ML models |
| `DataManager.candles_5m` | Multi-timeframe features |
| `DataManager.orderbooks` | Order flow features |
| `DataManager.trades` | Trade-level features |

**Data Flow**:
```
WebSocket → DataManager → DataSnapshot → Strategies
                │
                ▼
         [ML Feature Extraction Point]
```

### 2. Strategy System (`strategies/`)

**Architecture**: Plugin-based auto-discovery

**Strategy Interface**:
```python
# Required exports per strategy module
STRATEGY_NAME: str                    # Unique identifier
STRATEGY_VERSION: str                 # Semantic version
SYMBOLS: List[str]                    # Trading pairs
CONFIG: Dict[str, Any]                # Strategy parameters

# Required function
def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """Generate trading signal from market data"""
    pass
```

**Current Strategies**:

| Strategy | Style | Timeframe | ML Potential |
|----------|-------|-----------|--------------|
| EMA-9 Trend Flip | Trend Following | 1H | High - clear flip labels |
| Mean Reversion | Mean Reversion | 5m | High - deviation metrics |
| Grid RSI Reversion | Grid Trading | 5m | Medium - grid parameters |
| Momentum Scalping | Scalping | 1m | High - multi-filter signals |
| Order Flow | Microstructure | Tick | High - flow features |
| Market Making | Market Making | Tick | Low - latency critical |
| Ratio Trading | Pairs | 5m | Medium - spread metrics |
| Wavetrend | Oscillator | 5m | Medium - oscillator signals |
| Whale Sentiment | Contrarian | 5m | High - volume spikes |

### 3. Signal Generation Pipeline

**Typical Signal Flow**:
```
1. Data Validation
   └── Check warmup (min candles available)

2. Indicator Calculation
   └── EMA, RSI, ATR, BB, VPIN, etc.

3. Filter Evaluation
   ├── Regime filter (volatility)
   ├── Trend filter (ADX)
   ├── Volume filter (spike detection)
   ├── Correlation filter (XRP/BTC)
   └── Position limits

4. Signal Creation
   └── Buy/Sell with stops & targets

5. Cooldown Check
   └── Time since last signal
```

**ML Integration**:
- **Input**: Steps 1-3 produce features
- **Output**: Step 4 decision is the label
- **Feedback**: Fill results provide reward signal

### 4. Execution System (`executor.py`)

**Paper Execution Model**:
- Slippage: 0.05% default
- Fees: 0.1% default
- Leverage: 1.5x long, 2.0x short

**Position Tracking**:
```python
@dataclass
class Position:
    symbol: str
    side: str           # 'long' or 'short'
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    max_hold_until: datetime
```

### 5. Portfolio System (`portfolio.py`)

**Isolated Portfolios**: Each strategy gets independent capital

**Tracking Metrics**:
- `total_trades`, `winning_trades`, `losing_trades`
- `total_pnl`, `max_drawdown`
- `pnl_by_symbol`, `trades_by_symbol`

**ML Value**: Portfolio metrics provide reward signals for RL

### 6. Regime Detection (`regime/`)

**Multi-Factor Regime Classification**:

| Factor | Weight | Description |
|--------|--------|-------------|
| ADX | 25% | Trend strength |
| Choppiness | 20% | Range detection |
| MA Alignment | 20% | Trend direction |
| RSI Momentum | 15% | Overbought/oversold |
| Volume | 10% | Activity level |
| Sentiment | 10% | Fear & Greed |

**Output**: `RegimeSnapshot` with bull/bear/sideways classification

**ML Value**: Pre-computed regime features for model input

### 7. Logging System (`logger.py`)

**JSONL Event Streams**:
- `logs/system/` - System events
- `logs/strategies/` - Per-strategy signals
- `logs/trades/` - Fills and P&L
- `logs/aggregated/` - Complete audit trail

**ML Training Data Source**:
```json
{
  "timestamp": "2025-12-16T10:30:45.123Z",
  "event_type": "signal",
  "strategy": "ema9_trend_flip",
  "signal": {
    "action": "buy",
    "price": 104500.0,
    "stop_loss": 103445.0,
    "take_profit": 106609.0
  },
  "indicators": {
    "ema_9": 104200.0,
    "atr": 500.0,
    "consecutive_count": 3
  },
  "data_hash": "a1b2c3d4e5f6"
}
```

## ML Integration Architecture

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ML-Enhanced TG4                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐   │
│  │  Data Layer  │───▶│              ML Pipeline                      │   │
│  └──────────────┘    │  ┌────────────┐  ┌────────────┐  ┌────────┐  │   │
│         │            │  │  Feature   │  │   Model    │  │ Signal │  │   │
│         │            │  │ Extraction │─▶│ Inference  │─▶│ Fusion │  │   │
│         │            │  └────────────┘  └────────────┘  └────────┘  │   │
│         │            └──────────────────────────────────────────────┘   │
│         │                                      │                         │
│         ▼                                      ▼                         │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │  Strategies  │───▶│ ML Signals   │───▶│ Execution & Portfolio   │   │
│  │  (Existing)  │    │ (New)        │    └─────────────────────────┘   │
│  └──────────────┘    └──────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Integration Points

| Layer | Integration Type | Implementation |
|-------|------------------|----------------|
| Data Layer | Feature extraction | Add `MLFeatureExtractor` class |
| Strategies | ML signal generator | New `ml_signal/` strategy |
| Execution | Position sizing | ML-based size calculator |
| Portfolio | Reward calculation | RL environment wrapper |
| Logging | Training data | Export to parquet/CSV |

## Key Files Reference

| File | Lines | Purpose | ML Relevance |
|------|-------|---------|--------------|
| `data_layer.py` | ~600 | Data management | Feature source |
| `strategy_loader.py` | ~200 | Strategy loading | Model loading |
| `executor.py` | ~400 | Trade execution | Action execution |
| `portfolio.py` | ~500 | Portfolio tracking | Reward signal |
| `logger.py` | ~400 | Event logging | Training data |
| `paper_tester.py` | ~500 | Main loop | Integration point |

## Recommendations

### High Priority

1. **Create Feature Extraction Layer**
   - Add `ws_paper_tester/ml/features.py`
   - Extract indicators in consistent format
   - Support multi-timeframe features

2. **Add ML Strategy Type**
   - New strategy that calls ML models
   - Same interface as existing strategies
   - Supports model hot-swapping

3. **Build Training Data Pipeline**
   - Export logs to training format
   - Handle time alignment
   - Prevent look-ahead bias

### Medium Priority

4. **RL Environment Wrapper**
   - Wrap `PaperExecutor` as Gymnasium env
   - Define action/observation spaces
   - Implement reward shaping

5. **Model Registry**
   - Version control for trained models
   - A/B testing support
   - Rollback capability

---

**Next Document**: [Data Analysis](./02-data-analysis.md)
