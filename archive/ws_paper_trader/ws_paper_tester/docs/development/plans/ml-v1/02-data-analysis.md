# Data Analysis - Historical Data for ML Training

**Document Version**: 1.0
**Created**: 2025-12-16
**Status**: Research Complete

---

## Overview

This document analyzes the historical data available in TimescaleDB for ML model training, including schema, volumes, and data quality considerations.

## Database Architecture

### Storage Backend: TimescaleDB

**Configuration**:
- PostgreSQL 15+ with TimescaleDB extension
- Docker container with persistent volumes
- Port: 5433 (external) -> 5432 (internal)

**Performance Tuning**:
```
shared_buffers = 2GB
effective_cache_size = 6GB
maintenance_work_mem = 512MB
work_mem = 64MB
max_parallel_workers = 8
```

## Schema Definition

### Core Tables

#### `trades` - Individual Trade Ticks
```sql
CREATE TABLE trades (
    symbol          VARCHAR(20) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    price           DECIMAL(20, 10) NOT NULL,
    volume          DECIMAL(20, 10) NOT NULL,
    side            VARCHAR(4) NOT NULL,  -- 'buy' or 'sell'
    order_type      VARCHAR(10),
    misc            VARCHAR(50),
    PRIMARY KEY (symbol, timestamp)
);

-- Hypertable with daily partitioning
SELECT create_hypertable('trades', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Compression after 7 days
ALTER TABLE trades SET (timescaledb.compress);
SELECT add_compression_policy('trades', INTERVAL '7 days');

-- Retention: 90 days
SELECT add_retention_policy('trades', INTERVAL '90 days');
```

#### `candles` - OHLCV Candles (1-minute base)
```sql
CREATE TABLE candles (
    symbol              VARCHAR(20) NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL,
    interval_minutes    INTEGER NOT NULL DEFAULT 1,
    open                DECIMAL(20, 10) NOT NULL,
    high                DECIMAL(20, 10) NOT NULL,
    low                 DECIMAL(20, 10) NOT NULL,
    close               DECIMAL(20, 10) NOT NULL,
    volume              DECIMAL(20, 10) NOT NULL,
    quote_volume        DECIMAL(20, 10),
    trade_count         INTEGER,
    vwap                DECIMAL(20, 10),
    PRIMARY KEY (symbol, timestamp, interval_minutes)
);

-- Hypertable with weekly partitioning
SELECT create_hypertable('candles', 'timestamp', chunk_time_interval => INTERVAL '7 days');

-- Compression after 30 days
SELECT add_compression_policy('candles', INTERVAL '30 days');

-- Retention: 365 days
SELECT add_retention_policy('candles', INTERVAL '365 days');
```

### Continuous Aggregates (Auto-Computed)

TimescaleDB automatically computes higher timeframes from 1-minute base data:

```sql
-- Example: 5-minute continuous aggregate
CREATE MATERIALIZED VIEW candles_5m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('5 minutes', timestamp) AS timestamp,
    5 AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(close * volume) / NULLIF(sum(volume), 0) AS vwap
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol, time_bucket('5 minutes', timestamp);
```

**Available Aggregates**:

| Interval | View Name | Refresh Rate | Source |
|----------|-----------|--------------|--------|
| 5m | `candles_5m` | 5 minutes | 1m base |
| 15m | `candles_15m` | 15 minutes | 1m base |
| 30m | `candles_30m` | 30 minutes | 1m base |
| 1h | `candles_1h` | 1 hour | 1m base |
| 4h | `candles_4h` | 4 hours | 1h agg |
| 12h | `candles_12h` | 12 hours | 1h agg |
| 1d | `candles_1d` | Daily | 1h agg |
| 1w | `candles_1w` | Weekly | 1d agg |

### Supporting Tables

#### `data_sync_status` - Gap Detection
```sql
CREATE TABLE data_sync_status (
    symbol              VARCHAR(20) NOT NULL,
    data_type           VARCHAR(20) NOT NULL,  -- 'trades' or 'candles'
    oldest_timestamp    TIMESTAMPTZ,
    newest_timestamp    TIMESTAMPTZ,
    last_sync_at        TIMESTAMPTZ DEFAULT NOW(),
    total_records       BIGINT,
    PRIMARY KEY (symbol, data_type)
);
```

#### `external_indicators` - Market Data
```sql
CREATE TABLE external_indicators (
    timestamp       TIMESTAMPTZ NOT NULL,
    indicator_name  VARCHAR(50) NOT NULL,  -- 'fear_greed', 'btc_dominance', etc.
    value           DECIMAL(10, 4),
    source          VARCHAR(50),
    PRIMARY KEY (timestamp, indicator_name)
);
```

#### `backtest_runs` - Results Storage
```sql
CREATE TABLE backtest_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name   VARCHAR(100) NOT NULL,
    symbols         VARCHAR(20)[] NOT NULL,
    start_date      TIMESTAMPTZ NOT NULL,
    end_date        TIMESTAMPTZ NOT NULL,
    metrics         JSONB NOT NULL,
    trades          JSONB,
    equity_curve    JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

## Data Volumes

### Estimated Storage Requirements

| Symbol | Period | 1m Candles | Storage (Raw) | Storage (Compressed) |
|--------|--------|------------|---------------|----------------------|
| XRP/USDT | 1 year | 525,600 | ~2.5 GB | ~500 MB |
| BTC/USDT | 1 year | 525,600 | ~2.5 GB | ~500 MB |
| XRP/BTC | 1 year | 525,600 | ~2.5 GB | ~500 MB |
| **Total** | **1 year** | **1,576,800** | **~7.5 GB** | **~1.5 GB** |

**Higher Timeframes** (minimal additional storage due to aggregation):
- 5m: ~105K records/year/symbol
- 1h: ~8.7K records/year/symbol
- 1d: ~365 records/year/symbol

### Data Growth Rate

- **1-minute candles**: 1,440 per day per symbol
- **Trades**: Variable (10K-500K per day depending on market activity)
- **Monthly growth**: ~45MB per symbol (compressed)

## Data Quality

### Validation Rules (Implemented)

```python
# In historical_backfill.py (lines 239-306)
def validate_trade(trade: dict) -> bool:
    """Validate trade data before storage"""
    # Price validation
    if trade['price'] <= 0 or math.isnan(trade['price']):
        return False

    # Volume validation
    if trade['volume'] <= 0 or math.isnan(trade['volume']):
        return False

    # Timestamp validation
    try:
        datetime.fromisoformat(trade['timestamp'])
    except ValueError:
        return False

    return True
```

### Gap Detection & Filling

**Gap Filler Strategy** (`gap_filler.py`):

| Gap Size | Strategy | Source | Rate Limit |
|----------|----------|--------|------------|
| < 12 hours | OHLC API | `api.kraken.com/0/public/OHLC` | Fast |
| >= 12 hours | Trades API | `api.kraken.com/0/public/Trades` | 1.1s/req |

**Auto-Detection**:
```python
# Real-time catch-up detection
if batch_size < 100 and time_to_now < 60:
    # Caught up to real-time, stop backfill
    break
```

### Data Integrity Checks

1. **Timestamp Ordering**: Candles sorted ascending by time
2. **OHLC Consistency**: high >= open, close and low <= open, close
3. **Volume Positivity**: volume > 0
4. **No Duplicates**: Primary key prevents duplicate timestamps
5. **Continuous Coverage**: Gap detection tracks missing periods

## Data Access Patterns

### HistoricalDataProvider API

**File**: `data/historical_provider.py`

```python
class HistoricalDataProvider:
    """Async interface to historical data"""

    async def get_candles(
        self,
        symbol: str,
        interval_minutes: int = 1,
        start: datetime = None,
        end: datetime = None,
        limit: int = None
    ) -> List[Candle]:
        """Fetch candles for a time range"""
        pass

    async def get_latest_candles(
        self,
        symbol: str,
        interval_minutes: int = 1,
        count: int = 400
    ) -> List[Candle]:
        """Fetch most recent N candles (for warmup)"""
        pass

    async def get_multi_timeframe_candles(
        self,
        symbol: str,
        end_time: datetime,
        intervals: List[int] = [1, 5, 15, 60, 240],
        lookback: int = 200
    ) -> Dict[int, List[Candle]]:
        """Fetch aligned multi-timeframe data"""
        pass

    async def replay_candles(
        self,
        symbol: str,
        interval_minutes: int,
        start: datetime,
        end: datetime,
        speed: float = 1.0
    ) -> AsyncGenerator[Candle, None]:
        """Stream candles for backtesting"""
        pass

    async def get_data_range(
        self,
        symbol: str
    ) -> Tuple[datetime, datetime, int]:
        """Get oldest, newest timestamps and count"""
        pass
```

### Connection Pooling

```python
# Pool configuration
pool = await asyncpg.create_pool(
    DATABASE_URL,
    min_size=2,
    max_size=10,
    command_timeout=300.0  # 5 min for large queries
)
```

## ML Training Data Preparation

### Feature Dataset Schema

For ML training, data should be exported in this format:

```python
@dataclass
class MLTrainingRow:
    # Identifiers
    timestamp: datetime
    symbol: str

    # Price features (current candle)
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Derived features (computed)
    returns_1: float      # 1-bar return
    returns_5: float      # 5-bar return
    returns_20: float     # 20-bar return
    volatility: float     # 20-bar std dev

    # Technical indicators
    ema_9: float
    ema_21: float
    ema_50: float
    rsi_14: float
    atr_14: float
    adx_14: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float

    # Multi-timeframe features
    ema_9_5m: float
    ema_9_1h: float
    trend_5m: int         # 1=up, -1=down, 0=neutral
    trend_1h: int

    # Regime features
    volatility_regime: str  # LOW, MEDIUM, HIGH, EXTREME
    market_regime: str      # BULL, BEAR, SIDEWAYS

    # Labels (for supervised learning)
    signal_type: str        # BUY, SELL, HOLD
    future_return_5: float  # Return over next 5 bars
    future_return_20: float # Return over next 20 bars
    future_direction: int   # 1=up, -1=down
```

### Data Export Pipeline

```python
async def export_training_data(
    provider: HistoricalDataProvider,
    symbol: str,
    start: datetime,
    end: datetime,
    output_path: str
) -> None:
    """Export historical data with computed features for ML training"""

    # Fetch raw candles
    candles = await provider.get_candles(symbol, 1, start, end)

    # Convert to DataFrame
    df = pd.DataFrame([c.to_dict() for c in candles])

    # Compute indicators using pandas-ta
    df['ema_9'] = ta.ema(df['close'], length=9)
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Compute returns
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_20'] = df['close'].pct_change(20)

    # Compute labels (future returns)
    df['future_return_5'] = df['close'].pct_change(5).shift(-5)
    df['future_return_20'] = df['close'].pct_change(20).shift(-20)
    df['future_direction'] = (df['future_return_5'] > 0).astype(int) * 2 - 1

    # Drop warmup rows (NaN from indicators)
    df = df.dropna()

    # Export to parquet (efficient for ML)
    df.to_parquet(output_path, compression='snappy')
```

### Preventing Look-Ahead Bias

**Critical Considerations**:

1. **Normalization**: Use `EncoderNormalizer` that scales per-sequence, not globally
2. **Feature Calculation**: All indicators must use only past data
3. **Labels**: Future returns calculated separately, not during training
4. **Train/Test Split**: Always split by time, not random
5. **Validation**: Use walk-forward validation

```python
# Correct: Time-based split
train_end = datetime(2024, 6, 30)
test_start = datetime(2024, 7, 1)

train_data = df[df['timestamp'] < train_end]
test_data = df[df['timestamp'] >= test_start]

# WRONG: Random split (causes look-ahead bias)
# train_data, test_data = train_test_split(df, test_size=0.2)
```

## Recommendations

### For ML Training

1. **Primary Dataset**: Use 1-minute candles for maximum granularity
2. **Multi-Timeframe**: Fetch 5m, 15m, 1h aggregates for context
3. **Sequence Length**: 60-200 bars depending on model type
4. **Normalization**: Per-sequence scaling to prevent look-ahead
5. **Storage Format**: Parquet with snappy compression

### Data Pipeline

```
TimescaleDB → HistoricalDataProvider → Feature Engineering → Parquet
                                              │
                                              ▼
                                    PyTorch DataLoader → Model Training
```

### Volume Recommendations

| Model Type | Training Data | Validation | Test |
|------------|---------------|------------|------|
| Signal Classifier | 6 months | 2 months | 2 months |
| Price Predictor | 12 months | 3 months | 3 months |
| RL Agent | 12 months | 3 months | Live paper |

---

**Next Document**: [Feature Engineering](./03-feature-engineering.md)
