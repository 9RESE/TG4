# Phase 1: Foundation

**Phase Status**: Ready for Implementation
**Dependencies**: Existing TimescaleDB, Ollama, Kraken Collectors
**Deliverable**: Working data→prompt pipeline

---

## Overview

Phase 1 establishes the foundation layer that all agents will use. This phase builds upon existing infrastructure (TimescaleDB with 5-9 years of data, existing collectors) and adds the components needed to transform raw market data into LLM-ready prompts.

### Components

| Component | Description | Depends On |
|-----------|-------------|------------|
| 1.1 Data Pipeline Extensions | New tables for agent outputs | Existing TimescaleDB |
| 1.2 Indicator Library | Technical indicator calculations | 1.1 |
| 1.3 Market Snapshot Builder | Aggregates data for agents | 1.1, 1.2 |
| 1.4 Prompt Template System | Builds LLM prompts | 1.3 |

---

## 1.1 Data Pipeline Extensions

### Purpose

Extend existing TimescaleDB schema with tables for agent outputs, trading decisions, and system state.

### Database Schema Additions

```sql
-- ============================================================================
-- AGENT OUTPUTS TABLE
-- Purpose: Store all agent outputs for audit, learning, and evaluation
-- ============================================================================

CREATE TABLE agent_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_name VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    output_type VARCHAR(50) NOT NULL,
    output_data JSONB NOT NULL,
    model_used VARCHAR(50),
    prompt_hash VARCHAR(64),  -- For caching/deduplication
    latency_ms INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost_usd DECIMAL(10, 6),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX idx_agent_outputs_agent_ts
    ON agent_outputs (agent_name, timestamp DESC);
CREATE INDEX idx_agent_outputs_symbol_ts
    ON agent_outputs (symbol, timestamp DESC);
CREATE INDEX idx_agent_outputs_prompt_hash
    ON agent_outputs (prompt_hash, timestamp DESC);

-- Partitioning for efficient time-based queries (optional for hypertable)
-- SELECT create_hypertable('agent_outputs', 'timestamp',
--     chunk_time_interval => INTERVAL '1 day',
--     if_not_exists => TRUE);


-- ============================================================================
-- TRADING DECISIONS TABLE
-- Purpose: Audit trail for all trading decisions (approved, modified, rejected)
-- ============================================================================

CREATE TABLE trading_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    decision_type VARCHAR(20) NOT NULL CHECK (
        decision_type IN ('signal', 'execution', 'modification', 'rebalance')
    ),
    action VARCHAR(10) NOT NULL CHECK (
        action IN ('BUY', 'SELL', 'HOLD', 'CLOSE', 'REBALANCE')
    ),
    confidence DECIMAL(4, 3),
    parameters JSONB,
    agent_inputs JSONB,  -- References to agent_outputs UUIDs
    risk_evaluation JSONB,
    final_status VARCHAR(20) NOT NULL CHECK (
        final_status IN ('approved', 'modified', 'rejected', 'pending', 'executed')
    ),
    rejection_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trading_decisions_symbol_ts
    ON trading_decisions (symbol, timestamp DESC);
CREATE INDEX idx_trading_decisions_status
    ON trading_decisions (final_status, timestamp DESC);


-- ============================================================================
-- TRADE EXECUTIONS TABLE
-- Purpose: Track executed trades and their outcomes
-- ============================================================================

CREATE TABLE trade_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID REFERENCES trading_decisions(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('long', 'short')),
    size DECIMAL(20, 10) NOT NULL,
    size_usd DECIMAL(20, 2) NOT NULL,
    entry_price DECIMAL(20, 10) NOT NULL,
    leverage INTEGER DEFAULT 1 CHECK (leverage BETWEEN 1 AND 5),
    stop_loss DECIMAL(20, 10),
    take_profit DECIMAL(20, 10),
    status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (
        status IN ('open', 'closed', 'partially_closed', 'cancelled')
    ),
    exit_price DECIMAL(20, 10),
    exit_timestamp TIMESTAMPTZ,
    realized_pnl DECIMAL(20, 10),
    realized_pnl_pct DECIMAL(10, 4),
    fees_usd DECIMAL(20, 6),
    exit_reason VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trade_executions_status
    ON trade_executions (status, timestamp DESC);
CREATE INDEX idx_trade_executions_symbol
    ON trade_executions (symbol, status, timestamp DESC);


-- ============================================================================
-- PORTFOLIO SNAPSHOTS TABLE
-- Purpose: Track portfolio state over time for performance analysis
-- ============================================================================

CREATE TABLE portfolio_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    total_equity_usd DECIMAL(20, 2) NOT NULL,
    available_margin_usd DECIMAL(20, 2) NOT NULL,
    used_margin_usd DECIMAL(20, 2) NOT NULL,

    -- Balances
    btc_balance DECIMAL(20, 10) DEFAULT 0,
    xrp_balance DECIMAL(20, 10) DEFAULT 0,
    usdt_balance DECIMAL(20, 2) DEFAULT 0,

    -- Hodl bags (separate from trading balances)
    btc_hodl DECIMAL(20, 10) DEFAULT 0,
    xrp_hodl DECIMAL(20, 10) DEFAULT 0,
    usdt_hodl DECIMAL(20, 2) DEFAULT 0,

    -- Allocation percentages
    allocation_btc_pct DECIMAL(5, 2),
    allocation_xrp_pct DECIMAL(5, 2),
    allocation_usdt_pct DECIMAL(5, 2),

    -- Performance metrics
    unrealized_pnl DECIMAL(20, 2),
    daily_pnl DECIMAL(20, 2),
    daily_pnl_pct DECIMAL(10, 4),
    peak_equity_usd DECIMAL(20, 2),
    drawdown_pct DECIMAL(10, 4),

    -- Risk state
    open_positions_count INTEGER DEFAULT 0,
    total_exposure_pct DECIMAL(10, 4),

    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('portfolio_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);


-- ============================================================================
-- RISK STATE TABLE
-- Purpose: Track risk-related state (circuit breakers, cooldowns)
-- ============================================================================

CREATE TABLE risk_state (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Loss tracking
    daily_loss_pct DECIMAL(10, 4) DEFAULT 0,
    weekly_loss_pct DECIMAL(10, 4) DEFAULT 0,
    max_drawdown_pct DECIMAL(10, 4) DEFAULT 0,
    peak_equity_usd DECIMAL(20, 2),

    -- Trade tracking
    consecutive_losses INTEGER DEFAULT 0,
    trades_today INTEGER DEFAULT 0,

    -- Circuit breaker state
    circuit_breakers_active JSONB DEFAULT '[]'::jsonb,
    trading_halted BOOLEAN DEFAULT FALSE,
    halt_reason TEXT,
    halt_until TIMESTAMPTZ,

    -- Cooldowns
    active_cooldowns JSONB DEFAULT '{}'::jsonb,

    -- Metadata
    last_trade_timestamp TIMESTAMPTZ,
    daily_reset_timestamp TIMESTAMPTZ,
    weekly_reset_timestamp TIMESTAMPTZ
);


-- ============================================================================
-- EXTERNAL DATA CACHE TABLE
-- Purpose: Cache external API responses (news, sentiment indicators)
-- ============================================================================

CREATE TABLE external_data_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    data JSONB NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    UNIQUE (source, data_type, timestamp)
);

CREATE INDEX idx_external_data_source_ts
    ON external_data_cache (source, data_type, timestamp DESC);


-- ============================================================================
-- INDICATOR CACHE TABLE
-- Purpose: Cache computed indicators to avoid recalculation
-- ============================================================================

CREATE TABLE indicator_cache (
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DECIMAL(30, 10),
    metadata JSONB,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, indicator_name, timestamp)
);

SELECT create_hypertable('indicator_cache', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);
```

### Configuration Schema

```yaml
# config/database.yaml

database:
  connection:
    host: ${DATABASE_HOST:-localhost}
    port: ${DATABASE_PORT:-5432}
    database: ${DATABASE_NAME:-triplegain}
    user: ${DATABASE_USER:-postgres}
    password: ${DATABASE_PASSWORD}
    min_connections: 5
    max_connections: 20

  retention:
    agent_outputs_days: 90
    trading_decisions_days: indefinite
    trade_executions_days: indefinite
    portfolio_snapshots_days: indefinite
    indicator_cache_days: 7
    external_data_cache_days: 30

  maintenance:
    vacuum_schedule: "0 3 * * *"  # 3 AM daily
    analyze_schedule: "0 4 * * 0"  # 4 AM Sunday
```

### Input/Output Contract

**Input**: Raw market data from existing hypertables
**Output**: Structured tables ready for agent consumption

| Operation | Input | Output |
|-----------|-------|--------|
| `store_agent_output()` | AgentOutput object | UUID of stored record |
| `get_agent_outputs()` | agent_name, time_range | List of AgentOutput |
| `store_trading_decision()` | TradingDecision object | UUID of stored record |

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Schema creation | All tables created without errors | Tables exist |
| Index performance | Query on indexed columns | < 10ms for recent data |
| Constraint validation | Invalid data rejected | Constraint errors raised |
| Retention policy | Old data cleaned up | Count before/after matches policy |

---

## 1.2 Indicator Library

### Purpose

Centralized library for calculating all technical indicators used by agents. Indicators are pre-computed (not calculated by LLMs) to ensure accuracy and performance.

### Indicator Specifications

#### Trend Indicators

| Indicator | Parameters | Calculation | Reference |
|-----------|------------|-------------|-----------|
| EMA | periods: [9, 21, 50, 200] | Exponential Moving Average | Design 04-data-pipeline.md |
| SMA | periods: [20, 50, 200] | Simple Moving Average | Design 04-data-pipeline.md |
| ADX | period: 14 | Average Directional Index | Design 01-multi-agent.md |
| Supertrend | period: 10, multiplier: 3.0 | Supertrend | Design 04-data-pipeline.md |

#### Momentum Indicators

| Indicator | Parameters | Calculation |
|-----------|------------|-------------|
| RSI | period: 14 | Relative Strength Index |
| Stochastic RSI | rsi_period: 14, stoch_period: 14, k: 3, d: 3 | Stochastic of RSI |
| MACD | fast: 12, slow: 26, signal: 9 | Moving Average Convergence Divergence |
| ROC | period: 10 | Rate of Change |

#### Volatility Indicators

| Indicator | Parameters | Calculation |
|-----------|------------|-------------|
| ATR | period: 14 | Average True Range |
| Bollinger Bands | period: 20, std_dev: 2.0 | Price channels |
| Keltner Channels | ema: 20, atr: 10, mult: 2.0 | ATR-based channels |

#### Volume Indicators

| Indicator | Parameters | Calculation |
|-----------|------------|-------------|
| OBV | - | On-Balance Volume |
| VWAP | anchor: session | Volume Weighted Average Price |
| Volume SMA | period: 20 | Volume Moving Average |

#### Pattern Detection

| Indicator | Parameters | Calculation |
|-----------|------------|-------------|
| Choppiness | period: 14 | Choppiness Index |
| Squeeze | bb: 20/2.0, kc: 20/1.5 | BB inside KC detection |

### Configuration Schema

```yaml
# config/indicators.yaml

indicators:
  # Trend Indicators
  ema:
    periods: [9, 21, 50, 200]
    source: close

  sma:
    periods: [20, 50, 200]
    source: close

  adx:
    period: 14

  supertrend:
    period: 10
    multiplier: 3.0

  # Momentum Indicators
  rsi:
    period: 14
    source: close

  stochastic_rsi:
    rsi_period: 14
    stoch_period: 14
    k_period: 3
    d_period: 3

  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9

  roc:
    period: 10

  # Volatility Indicators
  atr:
    period: 14

  bollinger_bands:
    period: 20
    std_dev: 2.0

  keltner_channels:
    ema_period: 20
    atr_period: 10
    multiplier: 2.0

  # Volume Indicators
  obv: {}

  vwap:
    anchor: session  # Reset daily

  volume_sma:
    period: 20

  # Pattern Detection
  choppiness:
    period: 14

  squeeze:
    bb_period: 20
    bb_std: 2.0
    kc_period: 20
    kc_mult: 1.5

# Timeframes to calculate for
timeframes:
  - 1m
  - 5m
  - 15m
  - 1h
  - 4h
  - 1d

# Symbols
symbols:
  - BTC/USDT
  - XRP/USDT
  - XRP/BTC
```

### Interface Definition

```python
# src/data/indicator_library.py

from dataclasses import dataclass
from typing import Optional
from decimal import Decimal

@dataclass
class IndicatorResult:
    """Result from indicator calculation."""
    name: str
    timestamp: datetime
    value: Decimal | dict
    metadata: Optional[dict] = None


class IndicatorLibrary:
    """
    Central library for technical indicator calculations.

    All calculations use numpy for performance.
    Results can be cached to indicator_cache table.
    """

    def __init__(self, config: dict, db_pool):
        self.config = config
        self.db = db_pool
        self._cache_enabled = True

    async def calculate_all(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict]
    ) -> dict[str, IndicatorResult]:
        """
        Calculate all configured indicators for a symbol/timeframe.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1h")
            candles: List of OHLCV candles (oldest first)

        Returns:
            Dictionary of indicator_name -> IndicatorResult
        """
        pass

    async def calculate_single(
        self,
        indicator_name: str,
        candles: list[dict],
        params: Optional[dict] = None
    ) -> IndicatorResult:
        """Calculate a single indicator with optional custom params."""
        pass

    # Individual indicator methods
    def calculate_ema(self, closes: list, period: int) -> list: ...
    def calculate_sma(self, closes: list, period: int) -> list: ...
    def calculate_rsi(self, closes: list, period: int) -> list: ...
    def calculate_macd(self, closes: list, fast: int, slow: int, signal: int) -> dict: ...
    def calculate_atr(self, highs: list, lows: list, closes: list, period: int) -> list: ...
    def calculate_bollinger_bands(self, closes: list, period: int, std_dev: float) -> dict: ...
    def calculate_adx(self, highs: list, lows: list, closes: list, period: int) -> list: ...
    def calculate_obv(self, closes: list, volumes: list) -> list: ...
    def calculate_vwap(self, highs: list, lows: list, closes: list, volumes: list) -> list: ...
    def calculate_choppiness(self, highs: list, lows: list, closes: list, period: int) -> list: ...
    def detect_squeeze(self, closes: list, bb_config: dict, kc_config: dict) -> bool: ...
```

### Output Format

```json
{
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "timestamp": "2025-12-18T10:00:00Z",
  "indicators": {
    "ema_9": 45123.45,
    "ema_21": 44890.12,
    "ema_50": 44500.00,
    "ema_200": 42000.00,
    "rsi_14": 62.5,
    "macd": {
      "line": 150.2,
      "signal": 120.5,
      "histogram": 29.7
    },
    "atr_14": 1250.0,
    "adx_14": 28.5,
    "bollinger_bands": {
      "upper": 46500.0,
      "middle": 45000.0,
      "lower": 43500.0,
      "width": 0.067,
      "position": 0.65
    },
    "obv": 125000000,
    "vwap": 45050.0,
    "choppiness_14": 48.2,
    "squeeze_detected": false
  }
}
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| EMA accuracy | Compare with known values | < 0.001% deviation |
| RSI bounds | All values 0-100 | No out-of-range values |
| ATR positivity | All ATR values positive | No negative values |
| Performance | Calculate all indicators | < 50ms for 1000 candles |
| Cache hit | Same input returns cached | < 5ms on cache hit |

---

## 1.3 Market Snapshot Builder

### Purpose

Aggregates all market data, indicators, and context into a single structured snapshot that can be converted to an LLM prompt.

### Snapshot Structure

```python
# src/data/market_snapshot.py

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

@dataclass
class CandleSummary:
    """Compact candle representation for prompts."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass
class OrderBookFeatures:
    """Extracted order book features."""
    bid_depth_usd: Decimal
    ask_depth_usd: Decimal
    imbalance: Decimal  # -1 to 1 (negative = more asks)
    spread_bps: Decimal
    weighted_mid: Decimal


@dataclass
class MultiTimeframeState:
    """Aggregated state across timeframes."""
    trend_alignment_score: Decimal  # -1 to 1
    aligned_bullish_count: int
    aligned_bearish_count: int
    total_timeframes: int
    rsi_by_timeframe: dict[str, Decimal]
    atr_by_timeframe: dict[str, Decimal]


@dataclass
class MarketSnapshot:
    """
    Complete market state snapshot for LLM consumption.

    This is the primary data structure passed to agents.
    """
    # Identification
    timestamp: datetime
    symbol: str

    # Current price
    current_price: Decimal
    price_24h_ago: Optional[Decimal] = None
    price_change_24h_pct: Optional[Decimal] = None

    # Candles by timeframe
    candles: dict[str, list[CandleSummary]] = field(default_factory=dict)

    # Pre-computed indicators
    indicators: dict[str, any] = field(default_factory=dict)

    # Order book
    order_book: Optional[OrderBookFeatures] = None

    # Multi-timeframe analysis
    mtf_state: Optional[MultiTimeframeState] = None

    # Volume analysis
    volume_24h: Optional[Decimal] = None
    volume_vs_avg: Optional[Decimal] = None  # Ratio vs 20-period avg

    # Cached regime (from previous detection)
    regime_hint: Optional[str] = None
    regime_confidence: Optional[Decimal] = None

    # Data quality
    data_age_seconds: int = 0
    missing_data_flags: list[str] = field(default_factory=list)

    def to_prompt_format(self, token_budget: int = 4000) -> str:
        """
        Convert snapshot to LLM-friendly JSON string.

        Automatically truncates to fit within token budget.
        """
        pass

    def to_compact_format(self) -> str:
        """
        Minimal format for Tier 1 (local) LLM prompts.

        Includes only essential data to minimize latency.
        """
        pass
```

### Builder Configuration

```yaml
# config/snapshot.yaml

snapshot_builder:
  # Candle lookback per timeframe
  candle_lookback:
    1m: 60
    5m: 48
    15m: 32
    1h: 48
    4h: 30
    1d: 30

  # Which indicators to include
  include_indicators:
    - rsi_14
    - macd
    - ema_9
    - ema_21
    - ema_50
    - ema_200
    - atr_14
    - adx_14
    - bollinger_bands
    - obv
    - vwap
    - choppiness_14

  # Order book settings
  order_book:
    enabled: true
    depth_levels: 10

  # Data quality thresholds
  data_quality:
    max_age_seconds: 60
    min_candles_required: 20

  # Token budget management
  token_budgets:
    tier1_local: 3500
    tier2_api: 8000
```

### Interface Definition

```python
class MarketSnapshotBuilder:
    """
    Builds complete market snapshots for agent consumption.
    """

    def __init__(
        self,
        db_pool,
        indicator_library: IndicatorLibrary,
        config: dict
    ):
        self.db = db_pool
        self.indicators = indicator_library
        self.config = config

    async def build_snapshot(
        self,
        symbol: str,
        include_order_book: bool = True
    ) -> MarketSnapshot:
        """
        Build complete market snapshot for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            include_order_book: Whether to fetch order book data

        Returns:
            MarketSnapshot with all data populated
        """
        pass

    async def build_multi_symbol_snapshot(
        self,
        symbols: list[str]
    ) -> dict[str, MarketSnapshot]:
        """
        Build snapshots for multiple symbols in parallel.
        """
        pass

    async def _fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> list[CandleSummary]:
        """Fetch candles from TimescaleDB."""
        pass

    async def _fetch_order_book(
        self,
        symbol: str
    ) -> Optional[OrderBookFeatures]:
        """Fetch and process order book data."""
        pass

    async def _calculate_mtf_state(
        self,
        candles_by_tf: dict[str, list]
    ) -> MultiTimeframeState:
        """Calculate multi-timeframe alignment state."""
        pass

    def _validate_data_quality(
        self,
        snapshot: MarketSnapshot
    ) -> list[str]:
        """Validate snapshot data quality, return list of issues."""
        pass
```

### Output Format (Full)

```json
{
  "timestamp": "2025-12-18T10:30:00Z",
  "symbol": "BTC/USDT",
  "current_price": 45250.50,
  "price_change_24h_pct": 2.35,

  "candles": {
    "1h": [
      {"t": "2025-12-18T09:00:00Z", "o": 45000, "h": 45300, "l": 44950, "c": 45200, "v": 125.5},
      {"t": "2025-12-18T10:00:00Z", "o": 45200, "h": 45350, "l": 45100, "c": 45250, "v": 98.2}
    ],
    "4h": [...],
    "1d": [...]
  },

  "indicators": {
    "rsi_14": 62.5,
    "macd": {"line": 150.2, "signal": 120.5, "histogram": 29.7},
    "ema_9": 45123.45,
    "ema_21": 44890.12,
    "atr_14": 1250.0,
    "adx_14": 28.5,
    "bollinger_bands": {"upper": 46500, "middle": 45000, "lower": 43500, "position": 0.65}
  },

  "order_book": {
    "bid_depth_usd": 2500000,
    "ask_depth_usd": 2200000,
    "imbalance": 0.12,
    "spread_bps": 3.5
  },

  "mtf_state": {
    "trend_alignment_score": 0.67,
    "aligned_bullish": 4,
    "aligned_bearish": 1,
    "total_timeframes": 5
  },

  "volume_vs_avg": 1.35,
  "regime_hint": "trending_bull",
  "data_age_seconds": 15
}
```

### Output Format (Compact - for Tier 1)

```json
{
  "ts": "2025-12-18T10:30:00Z",
  "sym": "BTC/USDT",
  "px": 45250.5,
  "rsi": 62.5,
  "macd_h": 29.7,
  "atr": 1250,
  "adx": 28.5,
  "bb_pos": 0.65,
  "trend": 0.67,
  "regime": "bull"
}
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Build latency | Full snapshot build time | < 200ms |
| Compact latency | Compact format build time | < 50ms |
| Token estimation | Prompt fits in budget | Within 10% of budget |
| Missing data handling | Build with partial data | No exceptions, flags set |
| Concurrent builds | 3 symbols in parallel | < 300ms total |

---

## 1.4 Prompt Template System

### Purpose

Manages prompt templates for each agent, assembles prompts from snapshots and context, and handles token budget management.

### Template Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROMPT STRUCTURE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ SYSTEM PROMPT (Fixed per agent)                                      │   │
│  │  - Role definition                                                   │   │
│  │  - Capabilities and constraints                                      │   │
│  │  - Output format specification                                       │   │
│  │  - Rules and guidelines                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ CONTEXT INJECTION (Dynamic)                                          │   │
│  │  - Portfolio state                                                   │   │
│  │  - Active positions                                                  │   │
│  │  - Risk state                                                        │   │
│  │  - Recent performance                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ MARKET DATA (Dynamic)                                                │   │
│  │  - From MarketSnapshot                                               │   │
│  │  - Adjusted for token budget                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ QUERY (Task-specific)                                                │   │
│  │  - Specific question or analysis request                             │   │
│  │  - Required output fields                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### System Prompt Templates

Templates stored in `config/prompts/` directory:

```
config/prompts/
├── technical_analysis.txt
├── regime_detection.txt
├── sentiment_analysis.txt
├── trading_decision.txt
├── portfolio_rebalance.txt
└── coordinator.txt
```

### Configuration Schema

```yaml
# config/prompts.yaml

prompts:
  # Token budget by LLM tier
  token_budgets:
    tier1_local:
      total: 8192
      system_prompt: 1500
      context: 800
      market_data: 3000
      query: 400
      buffer: 2492  # For response

    tier2_api:
      total: 128000
      system_prompt: 3000
      context: 2000
      market_data: 6000
      query: 1000
      buffer: 116000

  # Template locations
  templates_dir: config/prompts/

  # Agent-specific settings
  agents:
    technical_analysis:
      template: technical_analysis.txt
      tier: tier1_local
      max_candles_per_tf: 10

    regime_detection:
      template: regime_detection.txt
      tier: tier1_local
      max_candles_per_tf: 5

    sentiment_analysis:
      template: sentiment_analysis.txt
      tier: tier2_api
      include_news: true

    trading_decision:
      template: trading_decision.txt
      tier: tier2_api
      include_agent_outputs: true

    portfolio_rebalance:
      template: portfolio_rebalance.txt
      tier: tier2_api

    coordinator:
      template: coordinator.txt
      tier: tier2_api
```

### Interface Definition

```python
# src/llm/prompt_builder.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class AssembledPrompt:
    """Complete prompt ready for LLM."""
    system_prompt: str
    user_message: str
    estimated_tokens: int
    agent_name: str
    tier: str


@dataclass
class PortfolioContext:
    """Portfolio state for context injection."""
    total_equity_usd: Decimal
    available_margin_usd: Decimal
    positions: list[dict]
    allocation: dict[str, Decimal]
    daily_pnl_pct: Decimal
    drawdown_pct: Decimal
    consecutive_losses: int
    win_rate_7d: Decimal


class PromptBuilder:
    """
    Builds prompts for LLM agents.

    Handles template loading, context injection, token management.
    """

    def __init__(self, config: dict):
        self.config = config
        self._templates: dict[str, str] = {}
        self._load_templates()

    def build_prompt(
        self,
        agent_name: str,
        snapshot: MarketSnapshot,
        portfolio_context: Optional[PortfolioContext] = None,
        additional_context: Optional[dict] = None,
        query: Optional[str] = None
    ) -> AssembledPrompt:
        """
        Build complete prompt for an agent.

        Args:
            agent_name: Name of the target agent
            snapshot: Market data snapshot
            portfolio_context: Portfolio state (optional)
            additional_context: Extra context (e.g., other agent outputs)
            query: Specific query/task (uses default if not provided)

        Returns:
            AssembledPrompt ready for LLM
        """
        pass

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses ~3.5 characters per token (conservative for JSON).
        """
        return len(text) // 3

    def truncate_to_budget(
        self,
        content: str,
        max_tokens: int
    ) -> str:
        """Truncate content to fit token budget."""
        pass

    def _load_templates(self) -> None:
        """Load all prompt templates from disk."""
        pass

    def _format_portfolio_context(
        self,
        context: PortfolioContext
    ) -> str:
        """Format portfolio context for injection."""
        pass

    def _format_market_data(
        self,
        snapshot: MarketSnapshot,
        tier: str
    ) -> str:
        """Format market data, adjusting detail for tier."""
        pass
```

### Sample System Prompt (Technical Analysis)

```
# config/prompts/technical_analysis.txt

You are an expert quantitative technical analyst for cryptocurrency markets.

ROLE:
You analyze market data using technical indicators to identify trading opportunities.
You DO NOT make final trading decisions - you provide analysis to other agents.

CAPABILITIES:
- Interpret technical indicators (RSI, MACD, EMA, ATR, etc.)
- Identify chart patterns and price action signals
- Determine trend direction and strength
- Identify key support and resistance levels
- Assess volatility conditions

ANALYSIS GUIDELINES:
1. Always consider multiple timeframes (1m, 5m, 1h, 4h, 1d)
2. Confirm signals across different indicator categories
3. Note confluence of signals at key levels
4. Be explicit about confidence levels (0.0-1.0)
5. Identify invalidation conditions for your analysis

INDICATORS PROVIDED (pre-computed, do NOT calculate):
- RSI (14-period)
- MACD (12/26/9)
- EMA (9, 21, 50, 200)
- ATR (14-period)
- ADX (14-period)
- Bollinger Bands (20, 2.0)

OUTPUT FORMAT (JSON only, no other text):
{
  "timestamp": "ISO8601",
  "symbol": "SYMBOL",
  "trend": {
    "direction": "bullish|bearish|neutral",
    "strength": 0.0-1.0,
    "timeframe_alignment": ["list of aligned timeframes"]
  },
  "momentum": {
    "score": -1.0 to 1.0,
    "rsi_signal": "oversold|neutral|overbought",
    "macd_signal": "bullish_cross|bearish_cross|bullish|bearish|neutral"
  },
  "key_levels": {
    "resistance": [price levels],
    "support": [price levels],
    "current_position": "near_support|mid_range|near_resistance"
  },
  "signals": {
    "primary": "description of main signal",
    "secondary": ["additional observations"],
    "warnings": ["any concerns"]
  },
  "bias": "long|short|neutral",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation (max 100 words)"
}

CONSTRAINTS:
- Never recommend specific entry/exit prices (that's the Trading Agent's job)
- Always provide confidence scores between 0.0 and 1.0
- Flag when indicators are conflicting
- Output ONLY valid JSON, no markdown or explanations
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Template loading | All templates load without error | No exceptions |
| Token estimation | Accurate within 10% | Compare to tiktoken |
| Budget compliance | Generated prompts fit budget | All prompts under limit |
| Schema validation | Output matches expected schema | JSON schema validates |
| Context injection | Portfolio data correctly inserted | All fields present |

---

## Phase 1 Acceptance Criteria

### Functional Requirements

| Requirement | Test Method | Acceptance |
|-------------|-------------|------------|
| Database schema deployed | Migration script runs | All tables exist |
| Indicators calculate correctly | Unit tests vs known values | < 0.01% deviation |
| Snapshot builds in < 500ms | Performance test | 95th percentile < 500ms |
| Prompts fit token budget | Token counter | 100% compliance |
| Data flows end-to-end | Integration test | Snapshot→Prompt works |

### Non-Functional Requirements

| Requirement | Acceptance |
|-------------|------------|
| Code coverage | > 80% for new code |
| Documentation | All public methods documented |
| Type hints | All functions typed |
| Linting | Passes ruff/pylint |

### Deliverables Checklist

- [ ] Database migration scripts in `migrations/`
- [ ] `src/data/indicator_library.py` with all indicators
- [ ] `src/data/market_snapshot.py` with builder
- [ ] `src/llm/prompt_builder.py` with templates
- [ ] Configuration files in `config/`
- [ ] Unit tests in `tests/unit/`
- [ ] Integration tests in `tests/integration/`
- [ ] Documentation updated

---

## API Endpoints (Phase 1)

Phase 1 includes basic API endpoints for testing:

```yaml
# API Routes for Phase 1

endpoints:
  # Health check
  - path: /health
    method: GET
    description: System health status

  # Indicators
  - path: /api/v1/indicators/{symbol}/{timeframe}
    method: GET
    description: Get calculated indicators
    response: IndicatorResult

  # Market snapshot
  - path: /api/v1/snapshot/{symbol}
    method: GET
    description: Get market snapshot
    response: MarketSnapshot

  # Prompt preview (debug)
  - path: /api/v1/debug/prompt/{agent}
    method: GET
    description: Preview assembled prompt for agent
    response: AssembledPrompt
```

---

## References

- Design: [04-data-pipeline.md](../TripleGain-master-design/04-data-pipeline.md)
- Design: [02-llm-integration-system.md](../TripleGain-master-design/02-llm-integration-system.md)
- Design: [01-multi-agent-architecture.md](../TripleGain-master-design/01-multi-agent-architecture.md)

---

*Phase 1 Implementation Plan v1.0 - December 2025*
