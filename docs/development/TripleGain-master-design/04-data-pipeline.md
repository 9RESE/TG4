# TripleGain Data Pipeline

**Document Version**: 1.0
**Status**: Design Phase

---

## 1. Data Pipeline Overview

### 1.1 End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           TRIPLEGAIN DATA PIPELINE                                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          DATA SOURCES                                             │   │
│  │                                                                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │   Kraken    │  │   Kraken    │  │   News      │  │  On-Chain   │             │   │
│  │  │  WebSocket  │  │  REST API   │  │   APIs      │  │  Analytics  │             │   │
│  │  │  (Trades,   │  │  (Balance,  │  │(CryptoPanic,│  │  (Future)   │             │   │
│  │  │  Order Book)│  │  Orders)    │  │ F&G Index)  │  │             │             │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │   │
│  └─────────┼────────────────┼────────────────┼────────────────┼────────────────────────┘   │
│            │                │                │                │                            │
│            v                v                v                v                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                       DATA INGESTION LAYER                                        │   │
│  │                                                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                    WebSocket Handlers                                    │    │   │
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            │    │   │
│  │  │  │  Trade    │  │  OHLC     │  │  Order    │  │  Ticker   │            │    │   │
│  │  │  │  Handler  │  │  Handler  │  │  Book     │  │  Handler  │            │    │   │
│  │  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘            │    │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                       │                                          │   │
│  │                                       v                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                    Message Queue (In-Memory)                             │    │   │
│  │  │                    • Buffering for burst handling                       │    │   │
│  │  │                    • Deduplication                                       │    │   │
│  │  │                    • Priority ordering                                   │    │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                                  │
│                                       v                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                       DATA STORAGE LAYER                                          │   │
│  │                                                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                    TimescaleDB                                           │    │   │
│  │  │  ┌───────────────────────────────────────────────────────────────────┐  │    │   │
│  │  │  │ trades (hypertable)                  │ Retention: 90 days         │  │    │   │
│  │  │  │ candles (hypertable)                 │ Retention: 365 days        │  │    │   │
│  │  │  │ candles_5m, 15m, 30m, 1h, 4h, 1d, 1w │ Continuous Aggregates      │  │    │   │
│  │  │  │ order_book_snapshots                 │ Retention: 7 days          │  │    │   │
│  │  │  │ agent_outputs                        │ Indefinite                  │  │    │   │
│  │  │  │ trade_history                        │ Indefinite                  │  │    │   │
│  │  │  └───────────────────────────────────────────────────────────────────┘  │    │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                    Redis Cache                                           │    │   │
│  │  │  • Latest market data (TTL: 60s)                                        │    │   │
│  │  │  • Agent outputs (TTL: varies)                                          │    │   │
│  │  │  • Session state                                                         │    │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                                  │
│                                       v                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                       FEATURE ENGINEERING LAYER                                   │   │
│  │                                                                                   │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐     │   │
│  │  │  Indicator    │  │  Order Book   │  │  Volume       │  │  Multi-TF     │     │   │
│  │  │  Calculator   │  │  Features     │  │  Features     │  │  Aggregator   │     │   │
│  │  │  (RSI, MACD,  │  │  (Imbalance,  │  │  (VWAP, OBV,  │  │  (MTF state)  │     │   │
│  │  │  EMA, ATR)    │  │  Depth, Spread)│  │  Volume MA)   │  │               │     │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                                  │
│                                       v                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                       PROMPT ASSEMBLY LAYER                                       │   │
│  │                                                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                    Market Snapshot Builder                               │    │   │
│  │  │  • Aggregates all features into structured snapshot                     │    │   │
│  │  │  • Formats for LLM consumption                                          │    │   │
│  │  │  • Handles missing data gracefully                                      │    │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                       │                                          │   │
│  │                                       v                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                    Prompt Builder                                        │    │   │
│  │  │  • Combines system prompt + context + market data + query               │    │   │
│  │  │  • Agent-specific templates                                              │    │   │
│  │  │  • Token budget management                                               │    │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                                  │
│                                       v                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                       LLM AGENTS                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Source Specifications

### 2.1 Kraken WebSocket v2

| Channel | Data Type | Update Frequency | Primary Use |
|---------|-----------|------------------|-------------|
| `trade` | Trade ticks | Per trade | Volume analysis, VWAP |
| `ohlc` | OHLCV candles | Per candle close | Price data, indicators |
| `book` | Order book | Per change | Depth analysis, liquidity |
| `ticker` | Top of book | Per tick | Current price, spread |

**Connection Configuration**:
```yaml
kraken_websocket:
  url: "wss://ws.kraken.com/v2"
  auth_url: "wss://ws-auth.kraken.com/v2"

  subscriptions:
    - channel: "trade"
      symbols: ["XRP/USD", "BTC/USD", "XRP/BTC"]

    - channel: "ohlc"
      symbols: ["XRP/USD", "BTC/USD", "XRP/BTC"]
      intervals: [1, 5, 15, 60, 240, 1440]  # 1m, 5m, 15m, 1h, 4h, 1d

    - channel: "book"
      symbols: ["XRP/USD", "BTC/USD"]
      depth: 10

    - channel: "ticker"
      symbols: ["XRP/USD", "BTC/USD", "XRP/BTC"]

  reconnection:
    max_retries: 10
    base_delay_seconds: 1
    max_delay_seconds: 60
    backoff_multiplier: 2

  heartbeat:
    interval_seconds: 30
    timeout_seconds: 10
```

### 2.2 Symbol Mapping

```python
# Internal to Kraken API mapping
SYMBOL_MAP = {
    "BTC/USDT": "XBT/USDT",   # Kraken uses XBT for Bitcoin
    "BTC/USD": "XBT/USD",
    "XRP/USDT": "XRP/USDT",
    "XRP/USD": "XRP/USD",
    "XRP/BTC": "XRP/XBT"
}

# Database symbol format (normalized)
DB_SYMBOLS = ["BTC/USDT", "XRP/USDT", "XRP/BTC"]

# Trading pairs
TRADING_PAIRS = {
    "BTC/USDT": {
        "base": "BTC",
        "quote": "USDT",
        "min_order_size": 0.0001,
        "price_precision": 1,
        "size_precision": 8
    },
    "XRP/USDT": {
        "base": "XRP",
        "quote": "USDT",
        "min_order_size": 10,
        "price_precision": 5,
        "size_precision": 2
    },
    "XRP/BTC": {
        "base": "XRP",
        "quote": "BTC",
        "min_order_size": 10,
        "price_precision": 8,
        "size_precision": 2
    }
}
```

### 2.3 External Data Sources

| Source | Data Type | Update Frequency | Integration Status | LLM Provider |
|--------|-----------|------------------|-------------------|--------------|
| **Web Search (Grok)** | Real-time news & sentiment | Every 30 minutes | Primary | Grok |
| **Web Search (GPT)** | Aggregated news | Every 30 minutes | Secondary | GPT |
| **CryptoPanic API** | News aggregation | Every 5 minutes | Planned | - |
| **Alternative.me** | Fear & Greed Index | Every 8 hours | Planned | - |
| **Glassnode** | On-chain metrics | Hourly | Future | - |
| **Twitter/X API** | Social sentiment | Every 30 minutes | Future | Grok |

**Note**: Grok and GPT are the designated Sentiment Analysis agents with native web search capabilities. They fetch and analyze news/sentiment data every 30 minutes.

---

## 3. TimescaleDB Schema

### 3.1 Core Tables

```sql
-- Trades table (highest granularity)
CREATE TABLE trades (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(20, 10) NOT NULL,
    volume DECIMAL(20, 10) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(10),
    misc VARCHAR(50),
    PRIMARY KEY (timestamp, symbol, id)
);

-- Convert to hypertable with daily chunks
SELECT create_hypertable('trades', 'timestamp',
    chunk_time_interval => INTERVAL '1 day');

-- Candles table (base 1-minute)
CREATE TABLE candles (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    interval_minutes SMALLINT NOT NULL,
    open DECIMAL(20, 10) NOT NULL,
    high DECIMAL(20, 10) NOT NULL,
    low DECIMAL(20, 10) NOT NULL,
    close DECIMAL(20, 10) NOT NULL,
    volume DECIMAL(20, 10) NOT NULL,
    quote_volume DECIMAL(20, 10),
    trade_count INTEGER,
    vwap DECIMAL(20, 10),
    PRIMARY KEY (timestamp, symbol, interval_minutes)
);

SELECT create_hypertable('candles', 'timestamp',
    chunk_time_interval => INTERVAL '7 days');
```

### 3.2 New Tables for TripleGain

```sql
-- Agent outputs table (for learning and audit)
CREATE TABLE agent_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_name VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    output_type VARCHAR(50) NOT NULL,
    output_data JSONB NOT NULL,
    model_used VARCHAR(50),
    latency_ms INTEGER,
    tokens_used INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agent_outputs_agent_ts
    ON agent_outputs (agent_name, timestamp DESC);

CREATE INDEX idx_agent_outputs_symbol_ts
    ON agent_outputs (symbol, timestamp DESC);

-- Trading decisions table
CREATE TABLE trading_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    decision_type VARCHAR(20) NOT NULL,  -- 'signal', 'execution', 'modification'
    action VARCHAR(10) NOT NULL,          -- 'BUY', 'SELL', 'HOLD', 'CLOSE'
    confidence DECIMAL(4, 3),
    parameters JSONB,
    agent_inputs JSONB,                   -- References to agent_outputs
    risk_evaluation JSONB,
    final_status VARCHAR(20),             -- 'approved', 'modified', 'rejected'
    execution_result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trading_decisions_symbol_ts
    ON trading_decisions (symbol, timestamp DESC);

-- Trade executions table
CREATE TABLE trade_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID REFERENCES trading_decisions(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    size DECIMAL(20, 10) NOT NULL,
    entry_price DECIMAL(20, 10) NOT NULL,
    leverage INTEGER DEFAULT 1,
    stop_loss DECIMAL(20, 10),
    take_profit DECIMAL(20, 10),
    status VARCHAR(20) NOT NULL,          -- 'open', 'closed', 'partially_closed'
    exit_price DECIMAL(20, 10),
    exit_timestamp TIMESTAMPTZ,
    realized_pnl DECIMAL(20, 10),
    fees DECIMAL(20, 10),
    exit_reason VARCHAR(50),              -- 'stop_loss', 'take_profit', 'manual', 'signal'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trade_executions_status
    ON trade_executions (status, timestamp DESC);

-- Portfolio snapshots table
CREATE TABLE portfolio_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    total_equity_usd DECIMAL(20, 10) NOT NULL,
    available_margin_usd DECIMAL(20, 10) NOT NULL,
    btc_balance DECIMAL(20, 10),
    xrp_balance DECIMAL(20, 10),
    usdt_balance DECIMAL(20, 10),
    btc_hodl DECIMAL(20, 10) DEFAULT 0,
    xrp_hodl DECIMAL(20, 10) DEFAULT 0,
    allocation_btc_pct DECIMAL(5, 2),
    allocation_xrp_pct DECIMAL(5, 2),
    allocation_usdt_pct DECIMAL(5, 2),
    unrealized_pnl DECIMAL(20, 10),
    daily_pnl DECIMAL(20, 10),
    max_drawdown_pct DECIMAL(5, 2),
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('portfolio_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 day');

-- External data cache table
CREATE TABLE external_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL,          -- 'cryptopanic', 'fear_greed', 'glassnode'
    data_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    data JSONB NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX idx_external_data_source_ts
    ON external_data (source, data_type, timestamp DESC);
```

### 3.3 Continuous Aggregates (Existing)

The system leverages existing continuous aggregates:
- `candles_5m` - 5-minute candles
- `candles_15m` - 15-minute candles
- `candles_30m` - 30-minute candles
- `candles_1h` - Hourly candles
- `candles_4h` - 4-hour candles
- `candles_12h` - 12-hour candles
- `candles_1d` - Daily candles
- `candles_1w` - Weekly candles

---

## 4. Feature Engineering

### 4.1 Technical Indicators

```python
INDICATOR_CONFIG = {
    # Trend Indicators
    "ema": {
        "periods": [9, 21, 50, 200],
        "source": "close"
    },
    "sma": {
        "periods": [20, 50, 200],
        "source": "close"
    },
    "adx": {
        "period": 14
    },
    "supertrend": {
        "period": 10,
        "multiplier": 3.0
    },

    # Momentum Indicators
    "rsi": {
        "period": 14,
        "source": "close"
    },
    "stoch_rsi": {
        "rsi_period": 14,
        "stoch_period": 14,
        "k_period": 3,
        "d_period": 3
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    "roc": {
        "period": 10
    },

    # Volatility Indicators
    "atr": {
        "period": 14
    },
    "bollinger_bands": {
        "period": 20,
        "std_dev": 2.0
    },
    "keltner_channels": {
        "ema_period": 20,
        "atr_period": 10,
        "multiplier": 2.0
    },

    # Volume Indicators
    "obv": {},
    "vwap": {
        "anchor": "session"  # Reset daily
    },
    "volume_sma": {
        "period": 20
    },

    # Pattern Detection
    "choppiness": {
        "period": 14
    },
    "squeeze": {
        "bb_period": 20,
        "bb_std": 2.0,
        "kc_period": 20,
        "kc_mult": 1.5
    }
}
```

### 4.2 Order Book Features

```python
def calculate_order_book_features(order_book: dict) -> dict:
    """
    Extract trading-relevant features from order book snapshot.
    """
    bids = order_book["bids"]  # [[price, size], ...]
    asks = order_book["asks"]

    # Calculate depths
    bid_depth = sum(float(b[1]) for b in bids[:10])
    ask_depth = sum(float(a[1]) for a in asks[:10])
    total_depth = bid_depth + ask_depth

    # Imbalance
    imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

    # Spread
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = best_ask - best_bid
    spread_bps = (spread / best_bid) * 10000

    # Weighted mid price
    weighted_mid = (best_bid * ask_depth + best_ask * bid_depth) / total_depth

    # Depth at levels
    depth_levels = {}
    for pct in [0.5, 1.0, 2.0]:  # % from mid
        mid = (best_bid + best_ask) / 2
        bid_level = mid * (1 - pct/100)
        ask_level = mid * (1 + pct/100)

        bid_depth_at_level = sum(
            float(b[1]) for b in bids if float(b[0]) >= bid_level
        )
        ask_depth_at_level = sum(
            float(a[1]) for a in asks if float(a[0]) <= ask_level
        )

        depth_levels[f"depth_{pct}pct"] = {
            "bid": bid_depth_at_level,
            "ask": ask_depth_at_level
        }

    return {
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance": imbalance,
        "spread_bps": spread_bps,
        "weighted_mid": weighted_mid,
        "depth_levels": depth_levels,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### 4.3 Multi-Timeframe Aggregation

```python
def aggregate_mtf_state(candles_by_tf: dict) -> dict:
    """
    Aggregate indicator state across multiple timeframes.
    """
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    mtf_state = {}

    # Trend alignment
    trend_scores = []
    for tf in timeframes:
        candles = candles_by_tf.get(tf, [])
        if not candles:
            continue

        # Calculate trend score for this timeframe
        ema_fast = calculate_ema(candles, 9)
        ema_slow = calculate_ema(candles, 21)
        current_close = candles[-1]["close"]

        if current_close > ema_fast > ema_slow:
            trend_scores.append(1)  # Bullish
        elif current_close < ema_fast < ema_slow:
            trend_scores.append(-1)  # Bearish
        else:
            trend_scores.append(0)  # Neutral

    # Aggregate trend
    avg_trend = sum(trend_scores) / len(trend_scores) if trend_scores else 0
    mtf_state["trend_alignment"] = {
        "score": avg_trend,
        "aligned_bullish": sum(1 for t in trend_scores if t > 0),
        "aligned_bearish": sum(1 for t in trend_scores if t < 0),
        "total_timeframes": len(trend_scores)
    }

    # RSI divergence detection
    rsi_values = {}
    for tf in ["1h", "4h", "1d"]:
        candles = candles_by_tf.get(tf, [])
        if candles:
            rsi_values[tf] = calculate_rsi(candles, 14)

    mtf_state["rsi"] = rsi_values

    # Volatility comparison
    atr_values = {}
    for tf in ["1h", "4h", "1d"]:
        candles = candles_by_tf.get(tf, [])
        if candles:
            atr_values[tf] = calculate_atr(candles, 14)

    mtf_state["atr"] = atr_values

    return mtf_state
```

---

## 5. Market Snapshot Builder

### 5.1 Snapshot Structure

```python
@dataclass
class MarketSnapshot:
    """
    Complete market state snapshot for LLM consumption.
    """
    timestamp: datetime
    symbol: str

    # Price data
    current_price: float
    candles: dict  # By timeframe

    # Technical indicators
    indicators: dict

    # Order book
    order_book_features: dict

    # Multi-timeframe state
    mtf_state: dict

    # Volume analysis
    volume_analysis: dict

    # Regime classification
    regime_hint: str  # From previous regime detection

    def to_prompt_format(self) -> str:
        """
        Format snapshot for LLM prompt injection.
        """
        return json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "current_price": self.current_price,
            "candles_summary": self._summarize_candles(),
            "indicators": self.indicators,
            "order_book": self.order_book_features,
            "mtf_analysis": self.mtf_state,
            "volume": self.volume_analysis,
            "regime_hint": self.regime_hint
        }, indent=2)

    def _summarize_candles(self) -> dict:
        """
        Create compact candle summary for prompt.
        """
        summary = {}
        for tf, candles in self.candles.items():
            if candles:
                recent = candles[-5:]  # Last 5 candles
                summary[tf] = {
                    "count": len(candles),
                    "recent_5": [
                        {
                            "o": round(c["open"], 2),
                            "h": round(c["high"], 2),
                            "l": round(c["low"], 2),
                            "c": round(c["close"], 2),
                            "v": round(c["volume"], 2)
                        }
                        for c in recent
                    ],
                    "period_high": max(c["high"] for c in candles),
                    "period_low": min(c["low"] for c in candles)
                }
        return summary
```

### 5.2 Snapshot Builder

```python
class MarketSnapshotBuilder:
    """
    Builds complete market snapshots for agent consumption.
    """

    def __init__(self, db_pool, redis_client, indicator_calculator):
        self.db = db_pool
        self.redis = redis_client
        self.indicators = indicator_calculator

    async def build_snapshot(self, symbol: str) -> MarketSnapshot:
        """
        Build complete market snapshot for a symbol.
        """
        timestamp = datetime.utcnow()

        # Fetch candles for all timeframes
        candles = await self._fetch_candles_all_timeframes(symbol)

        # Get current price from ticker
        current_price = await self._get_current_price(symbol)

        # Calculate indicators
        indicators = self._calculate_all_indicators(candles)

        # Get order book features
        order_book = await self._get_order_book_features(symbol)

        # Build MTF state
        mtf_state = aggregate_mtf_state(candles)

        # Volume analysis
        volume_analysis = self._analyze_volume(candles)

        # Get cached regime hint
        regime_hint = await self._get_cached_regime(symbol)

        return MarketSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            current_price=current_price,
            candles=candles,
            indicators=indicators,
            order_book_features=order_book,
            mtf_state=mtf_state,
            volume_analysis=volume_analysis,
            regime_hint=regime_hint
        )

    async def _fetch_candles_all_timeframes(self, symbol: str) -> dict:
        """
        Fetch candles from all timeframes.
        """
        timeframe_config = {
            "1m": {"lookback": 100, "table": "candles", "interval": 1},
            "5m": {"lookback": 60, "table": "candles_5m", "interval": 5},
            "15m": {"lookback": 48, "table": "candles_15m", "interval": 15},
            "1h": {"lookback": 48, "table": "candles_1h", "interval": 60},
            "4h": {"lookback": 30, "table": "candles_4h", "interval": 240},
            "1d": {"lookback": 30, "table": "candles_1d", "interval": 1440}
        }

        candles = {}
        for tf, config in timeframe_config.items():
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {config['table']}
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """
            rows = await self.db.fetch(query, symbol, config["lookback"])
            candles[tf] = [dict(row) for row in reversed(rows)]

        return candles
```

---

## 6. Prompt Assembly

### 6.1 Token Budget Management

```python
TOKEN_BUDGETS = {
    "tier1_local": {
        "total": 8192,  # Qwen 2.5 7B context
        "system_prompt": 2000,
        "context": 1000,
        "market_data": 3500,
        "query": 500,
        "buffer": 1192
    },
    "tier2_api": {
        "total": 128000,  # Most API models
        "system_prompt": 4000,
        "context": 3000,
        "market_data": 8000,
        "query": 1000,
        "buffer": 112000  # For response
    }
}


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (conservative).
    """
    # ~4 characters per token for English text
    # JSON tends to be more token-dense
    return len(text) // 3


def truncate_to_budget(content: str, max_tokens: int) -> str:
    """
    Truncate content to fit within token budget.
    """
    estimated_tokens = estimate_tokens(content)
    if estimated_tokens <= max_tokens:
        return content

    # Calculate truncation ratio
    ratio = max_tokens / estimated_tokens
    target_chars = int(len(content) * ratio * 0.9)  # 10% safety margin

    # Truncate and add indicator
    return content[:target_chars] + "\n... [TRUNCATED]"
```

### 6.2 Prompt Builder

```python
class PromptBuilder:
    """
    Builds complete prompts for LLM agents.
    """

    def __init__(self, templates: dict, tier: str):
        self.templates = templates
        self.tier = tier
        self.budget = TOKEN_BUDGETS[tier]

    def build_prompt(
        self,
        agent: str,
        market_snapshot: MarketSnapshot,
        portfolio_context: dict,
        query: str
    ) -> dict:
        """
        Build complete prompt with budget management.
        """
        # Get system prompt for agent
        system_prompt = self.templates[agent]["system_prompt"]
        system_tokens = estimate_tokens(system_prompt)

        if system_tokens > self.budget["system_prompt"]:
            system_prompt = truncate_to_budget(
                system_prompt,
                self.budget["system_prompt"]
            )

        # Build context
        context = self._build_context(portfolio_context)
        context_tokens = estimate_tokens(context)

        if context_tokens > self.budget["context"]:
            context = truncate_to_budget(context, self.budget["context"])

        # Format market data
        market_data = market_snapshot.to_prompt_format()
        market_tokens = estimate_tokens(market_data)

        if market_tokens > self.budget["market_data"]:
            # Reduce candle history
            market_data = self._reduce_market_data(
                market_snapshot,
                self.budget["market_data"]
            )

        # Assemble user message
        user_message = f"""
PORTFOLIO CONTEXT:
{context}

MARKET DATA:
{market_data}

TASK:
{query}
"""

        return {
            "system": system_prompt,
            "user": user_message,
            "estimated_input_tokens": (
                system_tokens + context_tokens +
                market_tokens + estimate_tokens(query)
            )
        }

    def _build_context(self, portfolio: dict) -> str:
        """
        Format portfolio context for prompt.
        """
        return json.dumps({
            "equity_usd": portfolio.get("equity"),
            "available_margin_usd": portfolio.get("available_margin"),
            "positions": portfolio.get("positions", []),
            "allocation": portfolio.get("allocation"),
            "daily_pnl_pct": portfolio.get("daily_pnl_pct"),
            "consecutive_losses": portfolio.get("consecutive_losses", 0),
            "win_rate_7d": portfolio.get("win_rate_7d")
        }, indent=2)

    def _reduce_market_data(
        self,
        snapshot: MarketSnapshot,
        target_tokens: int
    ) -> str:
        """
        Progressively reduce market data to fit budget.
        """
        # Strategy: reduce candle history, then reduce timeframes
        reduced = {
            "timestamp": snapshot.timestamp.isoformat(),
            "symbol": snapshot.symbol,
            "current_price": snapshot.current_price,
            "indicators": snapshot.indicators,  # Keep indicators
            "order_book": {
                k: v for k, v in snapshot.order_book_features.items()
                if k in ["imbalance", "spread_bps"]
            },
            "regime_hint": snapshot.regime_hint
        }

        # Add minimal candle data
        reduced["candles"] = {}
        for tf in ["1h", "4h", "1d"]:  # Only key timeframes
            if tf in snapshot.candles:
                candles = snapshot.candles[tf][-5:]  # Last 5 only
                reduced["candles"][tf] = [
                    {"c": round(c["close"], 2), "v": round(c["volume"], 0)}
                    for c in candles
                ]

        return json.dumps(reduced, indent=2)
```

---

## 7. Data Quality & Monitoring

### 7.1 Data Quality Checks

```python
DATA_QUALITY_CHECKS = {
    "candle_freshness": {
        "max_age_seconds": 120,  # 2 minutes for 1m candles
        "action_on_stale": "use_cached"
    },
    "price_sanity": {
        "max_change_pct": 10,  # Flag if price moves >10% in 1 minute
        "action_on_invalid": "reject"
    },
    "volume_sanity": {
        "min_volume": 0,
        "max_volume_ratio": 100,  # vs 20-period average
        "action_on_invalid": "flag_only"
    },
    "indicator_sanity": {
        "rsi_range": [0, 100],
        "atr_positive": True,
        "action_on_invalid": "recalculate"
    },
    "order_book_depth": {
        "min_depth_usd": 1000,
        "action_on_insufficient": "reduce_position_size"
    }
}


async def validate_market_data(snapshot: MarketSnapshot) -> dict:
    """
    Validate market data quality before agent consumption.
    """
    issues = []
    warnings = []

    # Check freshness
    age_seconds = (datetime.utcnow() - snapshot.timestamp).total_seconds()
    if age_seconds > DATA_QUALITY_CHECKS["candle_freshness"]["max_age_seconds"]:
        issues.append(f"Data stale: {age_seconds:.0f}s old")

    # Check price sanity
    if snapshot.candles.get("1m"):
        recent = snapshot.candles["1m"][-2:]
        if len(recent) == 2:
            change_pct = abs(
                (recent[-1]["close"] - recent[-2]["close"]) /
                recent[-2]["close"] * 100
            )
            if change_pct > DATA_QUALITY_CHECKS["price_sanity"]["max_change_pct"]:
                warnings.append(f"Large price move: {change_pct:.1f}%")

    # Check indicator sanity
    rsi = snapshot.indicators.get("rsi_14")
    if rsi is not None:
        if not 0 <= rsi <= 100:
            issues.append(f"Invalid RSI value: {rsi}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### 7.2 Pipeline Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `data_latency_ms` | Time from exchange to database | > 1000ms |
| `candle_gap_count` | Missing candles detected | > 0 |
| `ws_reconnect_count` | WebSocket reconnections/hour | > 5 |
| `indicator_calc_time_ms` | Time to compute indicators | > 500ms |
| `snapshot_build_time_ms` | Time to build complete snapshot | > 1000ms |
| `prompt_token_count` | Tokens in assembled prompt | > 80% budget |
| `data_quality_issues` | Quality check failures | > 0 critical |

---

## 8. Data Flow Timing

### 8.1 Latency Budget

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION LATENCY BUDGET (Target: <500ms)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Exchange → WebSocket Handler:          10ms                                │
│  WebSocket Handler → Database:          20ms                                │
│  Database Query (candles):              30ms                                │
│  Indicator Calculation:                 50ms                                │
│  Order Book Feature Extraction:         10ms                                │
│  Market Snapshot Assembly:              30ms                                │
│  Prompt Building:                       20ms                                │
│  LLM Inference (Tier 1 Local):         200ms                               │
│  Output Parsing & Validation:           20ms                                │
│  Risk Validation:                       10ms                                │
│  ─────────────────────────────────────────────                              │
│  TOTAL:                                400ms (100ms buffer)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Update Frequencies

| Data Type | Update Frequency | Trigger |
|-----------|------------------|---------|
| 1m Candles | Every minute | Candle close |
| Indicators | Every minute | After candle update |
| Order Book | Every 5 seconds | WebSocket update |
| Portfolio State | Every 10 seconds | Position/balance change |
| Market Snapshot | On demand | Agent request |
| Regime Detection | Every 5 minutes | Scheduled |
| Sentiment Analysis | Every 15 minutes | Scheduled |
| Strategic Decisions | Every hour | Scheduled |

---

*Document Version 1.0 - December 2025*
