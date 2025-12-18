# Understanding the Kraken Historical Data System

This document explains the architecture, design decisions, and trade-offs in the Kraken Historical Data System.

## Why This System Exists

### The Problem

Cryptocurrency trading strategies require historical data for:

1. **Backtesting**: Testing strategies against historical price movements
2. **Indicator Warmup**: Initializing technical indicators (e.g., a 200-period moving average needs 200 historical candles)
3. **Pattern Analysis**: Identifying market patterns and regime changes
4. **Real-time Trading**: Combining historical context with live data

Traditional approaches have limitations:
- **API calls on demand**: Slow, rate-limited, and unreliable during strategy execution
- **File-based storage**: Difficult to query efficiently, no automatic timeframe aggregation
- **Generic databases**: Not optimized for time-series data, poor query performance

### The Solution

The Kraken Historical Data System provides:
- **Persistent local storage** with fast query performance
- **Automatic timeframe aggregation** via TimescaleDB continuous aggregates
- **Gap detection and filling** to ensure data completeness
- **Real-time data integration** for seamless live trading
- **Efficient data compression** for long-term storage

## Architecture Deep Dive

### Why TimescaleDB?

TimescaleDB extends PostgreSQL with time-series superpowers:

| Feature | Benefit |
|---------|---------|
| **Hypertables** | Automatic partitioning by time |
| **Continuous Aggregates** | Auto-computed higher timeframes |
| **Compression** | 90%+ storage reduction for old data |
| **Retention Policies** | Automatic data expiration |
| **Standard SQL** | Use familiar tools and queries |

### Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION                               │
│                                                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│  │ WebSocket   │   │ REST API    │   │ CSV Files   │            │
│  │ (Real-time) │   │ (Backfill)  │   │ (Initial)   │            │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘            │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    WRITE LAYER                               │ │
│  │  • DatabaseWriter (buffered, async)                         │ │
│  │  • Buffer overflow protection                               │ │
│  │  • COPY for bulk inserts                                    │ │
│  │  • Upsert for conflict handling                             │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                             │                                    │
└─────────────────────────────┼────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      TIMESCALEDB                                  │
│                                                                   │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │     trades      │────│  candles (1m base data)             │ │
│  │   (raw ticks)   │    │                                     │ │
│  └─────────────────┘    └──────────────┬──────────────────────┘ │
│                                        │                         │
│         ┌──────────────────────────────┼───────────────────┐    │
│         │         CONTINUOUS AGGREGATES                     │    │
│         │                              ▼                    │    │
│         │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │    │
│         │  │  5m  │ │ 15m  │ │  1h  │ │  4h  │ │  1d  │   │    │
│         │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘   │    │
│         └──────────────────────────────────────────────────┘    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      DATA CONSUMPTION                             │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    READ LAYER                                │ │
│  │  • HistoricalDataProvider                                   │ │
│  │  • Automatic view routing by interval                       │ │
│  │  • Connection pooling                                        │ │
│  │  • Async/await support                                       │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Backtester  │   │ Strategies  │   │  Analysis   │           │
│  │             │   │ (warmup)    │   │   Tools     │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Design Decisions Explained

### REC-001: Buffer Overflow Protection

**Problem**: High-volume trading pairs can generate thousands of trades per second. If the database can't keep up, unbounded memory growth could crash the application.

**Solution**: Hard limits on buffer sizes (10,000 trades, 1,000 candles). When exceeded, oldest records are dropped and logged.

**Trade-off**: Some data loss during extreme load vs. system stability.

### REC-002: Connection State Checking

**Problem**: Database operations on a closed connection cause cryptic errors.

**Solution**: All read operations check `self.pool` before executing, raising `RuntimeError` with a helpful message.

**Pattern**:
```python
def _ensure_connected(self):
    if not self.pool:
        raise RuntimeError(
            "HistoricalDataProvider not connected. "
            "Call await provider.connect() first."
        )
```

### REC-003: Trade Data Validation

**Problem**: Kraken occasionally returns malformed data (zero prices, invalid timestamps).

**Solution**: Validate every trade before storage:
- Reject prices <= 0
- Reject volumes <= 0
- Skip invalid timestamps
- Log rejected records for debugging

### REC-004: No Default Credentials

**Problem**: Default passwords are a security risk, especially in Docker environments.

**Solution**: All components require explicit `DATABASE_URL` or `DB_PASSWORD` configuration. Startup fails with helpful error messages if credentials are missing.

### REC-005: Centralized Pair Mappings

**Problem**: Different Kraken endpoints use different symbol naming:
- WebSocket: `XRP/USDT`
- REST API: `XRPUSDT`
- CSV files: `XRPUSDT` or `XRPUSD`

**Solution**: Central `types.py` module with all mappings. Components import from this single source of truth.

### REC-006: Two Candle Types

**Problem**: Database operations need efficiency, but domain logic needs rich types.

**Solution**:
1. **`Candle`** (in `historical_provider.py`): Lightweight, includes `from_row()` for fast DB conversion
2. **`HistoricalCandle`** (in `types.py`): Full domain type with computed properties

**Usage**:
- Use `Candle` for database queries and strategy warmup
- Use `HistoricalCandle` for internal data representation

### REC-009: Retention Policies

**Problem**: Unbounded data growth consumes storage and degrades query performance.

**Solution**: Automatic retention policies:
- Trades: 90 days (high volume, can be recomputed if needed)
- Candles: 365 days (lower volume, valuable for backtesting)

**Trade-off**: Limited historical depth vs. predictable storage costs.

### REC-010: Performance Optimization

**Problem**: Pandas `iterrows()` is notoriously slow due to type checking overhead.

**Solution**: Use `itertuples()` which returns namedtuples without type conversion:
```python
# Slow (~100x slower)
for index, row in df.iterrows():
    process(row['column'])

# Fast
for row in df.itertuples(index=False):
    process(row.column)
```

## Understanding Continuous Aggregates

### How They Work

Continuous aggregates are like materialized views that automatically update:

```sql
CREATE MATERIALIZED VIEW candles_5m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('5 minutes', timestamp) AS timestamp,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol, time_bucket('5 minutes', timestamp)
WITH NO DATA;
```

### Aggregation Hierarchy

```
1m (base) ──┬── 5m
            ├── 15m
            └── 30m
                 │
1h ──────────────┬── 4h
                 ├── 12h
                 └── 1d
                      │
                      └── 1w
```

Higher timeframes (4h, 12h, 1d, 1w) aggregate from 1h rather than 1m for efficiency.

### Refresh Policies

Each aggregate has a refresh policy:
- **Start offset**: How far back to look for changes
- **End offset**: Buffer for incomplete data
- **Schedule interval**: How often to refresh

Example for 5m:
```sql
SELECT add_continuous_aggregate_policy('candles_5m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes'
);
```

## Gap Filling Strategy

### Detection

On startup, the `GapFiller` compares:
1. `newest_timestamp` in `data_sync_status`
2. Current time

Any gap > 2 minutes triggers gap filling.

### API Selection

| Gap Size | API Used | Reason |
|----------|----------|--------|
| < 12 hours | OHLC | Fast, returns pre-built candles (720 max) |
| >= 12 hours | Trades | Complete data, builds candles from ticks |

### Concurrent Filling

Gaps for multiple symbols fill concurrently with a semaphore:
```python
semaphore = asyncio.Semaphore(max_concurrent)

async def fill_with_limit(gap):
    async with semaphore:
        await self.fill_gap(gap)

await asyncio.gather(*[fill_with_limit(gap) for gap in gaps])
```

## Buffered Writing Strategy

### Why Buffer?

Database round-trips are expensive. Buffering reduces them:

| Approach | DB Calls | Latency |
|----------|----------|---------|
| Write each trade immediately | 1000/sec | High |
| Buffer 100 trades, flush | 10/sec | Low |

### Buffer Behavior

```
┌─────────────────────────────────────────────────────────────┐
│                     DatabaseWriter                           │
│                                                              │
│  Trade arrives ─────────────────────────────────┐           │
│                                                  ▼           │
│                              ┌─────────────────────┐        │
│                              │   Trade Buffer      │        │
│                              │   (deque, max 100)  │        │
│                              └──────────┬──────────┘        │
│                                         │                    │
│              ┌──────────────────────────┼──────────────────┐│
│              │                          │                   ││
│              ▼                          ▼                   ▼│
│       Buffer Full?              Timer (5 sec)?      Shutdown?│
│              │                          │                   ││
│              └──────────────────────────┼───────────────────┘│
│                                         │                    │
│                                         ▼                    │
│                              ┌─────────────────────┐        │
│                              │      FLUSH          │        │
│                              │  COPY to database   │        │
│                              └─────────────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Flush Triggers

1. **Buffer size**: Flush when buffer reaches threshold (100 trades, 10 candles)
2. **Timer**: Periodic flush even if buffer not full (every 5 seconds)
3. **Shutdown**: Final flush on graceful shutdown

### Error Recovery

On flush failure:
1. Log the error
2. Re-add records to buffer (up to max size)
3. Retry on next flush cycle

## Data Validation Philosophy

### Fail Fast, Fail Safe

The system validates data at ingestion boundaries:

```
External Data ───▶ [Validation] ───▶ Internal Processing ───▶ Database
                       │
                       ├── Invalid price? REJECT
                       ├── Invalid volume? REJECT
                       ├── Invalid timestamp? REJECT
                       └── Valid ✓
```

### Why Not Rely on Database Constraints?

1. **Performance**: Check constraints in code before network round-trip
2. **Flexibility**: Different validation rules per source (API vs CSV)
3. **Debugging**: Log exactly why records were rejected

## Connection Pooling Strategy

### Why Pool Connections?

Creating database connections is expensive (~100ms). Pooling amortizes this cost:

```
┌─────────────────────────────────────────────────────────────┐
│                      Connection Pool                         │
│                                                              │
│   ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐       │
│   │ conn1 │ │ conn2 │ │ conn3 │ │ conn4 │ │ conn5 │       │
│   └───┬───┘ └───────┘ └───────┘ └───────┘ └───────┘       │
│       │                                                      │
│       │ acquire()                                            │
│       ▼                                                      │
│   Task A uses conn1 ─────────────────────────────▶ release() │
│       │                                                      │
│       └──────────────────────────────────────────▶ Pool      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Pool Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_size` | 2 | Always keep 2 connections ready |
| `max_size` | 10 | Limit concurrent database operations |

Higher `max_size` allows more concurrency but consumes more database connections.

## When to Use Each Component

| Scenario | Component | Method |
|----------|-----------|--------|
| Initial data load from CSV | `BulkCSVImporter` | `import_directory()` |
| Fetch complete history | `KrakenTradesBackfill` | `backfill_symbol()` |
| Fill gaps after downtime | `GapFiller` | `fill_all_gaps()` |
| Persist live WebSocket data | `DatabaseWriter` | `write_trade()` |
| Query for backtesting | `HistoricalDataProvider` | `get_candles()` |
| Warm up indicators | `HistoricalDataProvider` | `get_warmup_data()` |

## Performance Characteristics

### Query Performance

| Query Type | Expected Time | Notes |
|------------|---------------|-------|
| 1000 candles by range | < 50ms | Uses index |
| Latest 200 candles | < 20ms | Descending index scan |
| Data range query | < 10ms | Aggregate query |
| Multi-timeframe (5 intervals) | < 100ms | Parallel queries |

### Write Performance

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Buffered trade writes | 10,000+/sec | COPY protocol |
| Candle upserts | 1,000+/sec | Conflict handling overhead |
| Bulk CSV import | 100,000+/sec | Batch inserts |

### Storage Efficiency

| Data Type | Raw Size | Compressed | Ratio |
|-----------|----------|------------|-------|
| Trades | 100 GB | ~10 GB | 90% |
| Candles | 10 GB | ~2 GB | 80% |

Compression kicks in after 7 days (trades) or 30 days (candles).
