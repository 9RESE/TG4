# Kraken Historical Data System - Building Block

## Overview

The Kraken Historical Data System (`data/kraken_db/`) is a comprehensive subsystem for storing, retrieving, and managing historical cryptocurrency market data from the Kraken exchange. It provides the foundational data layer for backtesting, strategy warmup, and real-time trading operations.

## Purpose

- **Data Persistence**: Store trades and OHLCV candles in TimescaleDB
- **Data Acquisition**: Fetch historical data from Kraken REST API
- **Gap Management**: Detect and fill data gaps automatically
- **Query Interface**: Provide efficient data access for strategies
- **Real-time Integration**: Persist WebSocket data as it arrives

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kraken Historical Data System                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   Data Sources   │    │   Data Consumers │                   │
│  │                  │    │                  │                   │
│  │  • Kraken WS     │    │  • Backtester    │                   │
│  │  • Kraken REST   │    │  • Strategies    │                   │
│  │  • CSV Files     │    │  • Analysis      │                   │
│  └────────┬─────────┘    └────────▲─────────┘                   │
│           │                       │                              │
│           ▼                       │                              │
│  ┌────────────────────────────────┴──────────────────────────┐  │
│  │                    Python Components                       │  │
│  │                                                            │  │
│  │  ┌─────────────────┐  ┌──────────────────┐                │  │
│  │  │ DatabaseWriter  │  │ HistoricalData   │                │  │
│  │  │ (Real-time)     │  │ Provider         │                │  │
│  │  └────────┬────────┘  └────────▲─────────┘                │  │
│  │           │                    │                           │  │
│  │  ┌────────┴────────┐  ┌───────┴──────────┐                │  │
│  │  │ Backfill        │  │ GapFiller        │                │  │
│  │  │ (Historical)    │  │ (Startup)        │                │  │
│  │  └────────┬────────┘  └────────┬─────────┘                │  │
│  │           │                    │                           │  │
│  │  ┌────────┴────────┐           │                          │  │
│  │  │ BulkCSVImporter │           │                          │  │
│  │  │ (Initial Load)  │           │                          │  │
│  │  └────────┬────────┘           │                          │  │
│  │           │                    │                           │  │
│  └───────────┴────────────────────┴───────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                      TimescaleDB                           │  │
│  │                                                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │   trades    │  │   candles   │  │ external_       │   │  │
│  │  │ (hypertable)│  │ (hypertable)│  │ indicators      │   │  │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────────────┘   │  │
│  │         │                │                                │  │
│  │         │     ┌──────────┴────────────────┐              │  │
│  │         │     │   Continuous Aggregates   │              │  │
│  │         │     │  5m, 15m, 30m, 1h, 4h,   │              │  │
│  │         │     │  12h, 1d, 1w             │              │  │
│  │         │     └───────────────────────────┘              │  │
│  │         │                                                 │  │
│  │  ┌──────┴──────────────────────────────────────────┐     │  │
│  │  │              data_sync_status                    │     │  │
│  │  │  (Gap detection & resumption tracking)          │     │  │
│  │  └─────────────────────────────────────────────────┘     │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. types.py - Data Types

**Purpose**: Defines all data structures used throughout the system.

| Type | Description | Mutable |
|------|-------------|---------|
| `HistoricalTrade` | Individual trade tick | No (frozen) |
| `HistoricalCandle` | Full domain candle with computed properties | No (frozen) |
| `Candle` | Lightweight candle for database queries | No (frozen) |
| `DataGap` | Represents missing data period | No (frozen) |
| `TradeRecord` | Mutable buffer for trade inserts | Yes |
| `CandleRecord` | Mutable buffer for candle inserts | Yes |
| `DataSyncStatus` | Sync state for resumption | No (frozen) |

**Centralized Pair Mappings** (REC-005):
- `PAIR_MAP`: Internal symbol → Kraken API pair
- `REVERSE_PAIR_MAP`: Kraken API pair → Internal symbol
- `CSV_SYMBOL_MAP`: CSV filename → Internal symbol
- `DEFAULT_SYMBOLS`: Default symbols for operations

### 2. websocket_db_writer.py - Real-time Persistence

**Purpose**: Persist WebSocket data to database with buffering.

**Classes**:
- `DatabaseWriter`: Async buffered writer with connection pooling
- `WebSocketDBIntegration`: Integration layer for KrakenWSClient

**Key Features**:
- Buffer overflow protection (REC-001)
- Automatic flush on buffer size or time interval
- COPY protocol for efficient bulk inserts
- Graceful error handling with retry logic
- Statistics tracking (trades/candles written, errors, overflows)

**Configuration**:
```python
DatabaseWriter(
    db_url="postgresql://...",
    trade_buffer_size=100,        # Flush after N trades
    trade_flush_interval=5.0,     # Max seconds between flushes
    candle_flush_interval=1.0,
    pool_min_size=2,
    pool_max_size=10
)
```

### 3. historical_backfill.py - Historical Data Fetching

**Purpose**: Fetch complete trade history from Kraken REST API.

**Class**: `KrakenTradesBackfill`

**Key Features**:
- Pagination via 'since' parameter (nanosecond timestamps)
- Rate limiting (1.1 seconds between requests)
- Auto-stop when caught up to real-time
- Builds 1-minute candles from trades
- Data validation (REC-003): Rejects invalid prices/volumes

**API Endpoints Used**:
- `GET /0/public/Trades` - Trade history with pagination

### 4. historical_provider.py - Query Interface

**Purpose**: Query historical data for backtesting and strategy warmup.

**Class**: `HistoricalDataProvider`

**Key Features**:
- Automatic routing to continuous aggregates by interval
- Multi-timeframe data queries
- Candle replay for backtesting with speed control
- Connection state checking (REC-002)

**Interval Routing**:
| Interval | View/Table |
|----------|------------|
| 1m | candles |
| 5m | candles_5m |
| 15m | candles_15m |
| 30m | candles_30m |
| 1h | candles_1h |
| 4h | candles_4h |
| 12h | candles_12h |
| 1d | candles_1d |
| 1w | candles_1w |

### 5. gap_filler.py - Data Gap Management

**Purpose**: Detect and fill data gaps on program startup.

**Class**: `GapFiller`

**Strategy**:
1. Query `data_sync_status` for each symbol
2. Identify gaps between `newest_timestamp` and now
3. Use OHLC API for small gaps (< 12 hours)
4. Use Trades API for large gaps (>= 12 hours)
5. Refresh continuous aggregates after filling

**Key Features**:
- Concurrent gap filling with semaphore
- Automatic API selection based on gap size
- Builds candles from trades for large gaps

### 6. bulk_csv_importer.py - Initial Data Load

**Purpose**: Import Kraken historical CSV files for initial setup.

**Class**: `BulkCSVImporter`

**CSV Format**: `timestamp, open, high, low, close, volume, trades`

**Key Features**:
- Batch inserts (10,000 records per batch)
- Conflict handling (upsert on duplicate)
- Performance optimization (REC-010): Uses `itertuples()` instead of `iterrows()`

## Database Schema

### Tables

#### trades
```sql
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
-- Hypertable: daily chunks
-- Compression: after 7 days
-- Retention: 90 days
```

#### candles
```sql
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
-- Hypertable: weekly chunks
-- Compression: after 30 days
-- Retention: 365 days
```

#### data_sync_status
```sql
CREATE TABLE data_sync_status (
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(20) NOT NULL,
    oldest_timestamp TIMESTAMPTZ,
    newest_timestamp TIMESTAMPTZ,
    last_sync_at TIMESTAMPTZ DEFAULT NOW(),
    last_kraken_since BIGINT,
    total_records BIGINT DEFAULT 0,
    PRIMARY KEY (symbol, data_type)
);
```

### Continuous Aggregates

Higher timeframes are automatically computed from 1-minute data:

| Aggregate | Source | Refresh Interval |
|-----------|--------|------------------|
| candles_5m | candles (1m) | 5 minutes |
| candles_15m | candles (1m) | 15 minutes |
| candles_30m | candles (1m) | 30 minutes |
| candles_1h | candles (1m) | 1 hour |
| candles_4h | candles_1h | 4 hours |
| candles_12h | candles_1h | 12 hours |
| candles_1d | candles_1h | 1 day |
| candles_1w | candles_1d | 1 week |

## Design Decisions

### REC-001: Buffer Overflow Protection
Trade and candle buffers have maximum sizes to prevent memory exhaustion under high load. Oldest records are dropped when buffers overflow.

### REC-002: Connection State Checking
All database operations check connection state before executing, raising `RuntimeError` if not connected.

### REC-003: Trade Data Validation
Invalid trades (zero/negative prices or volumes) are rejected during import to maintain data integrity.

### REC-004: No Default Credentials
All components require explicit `DATABASE_URL` configuration. No default passwords are used.

### REC-005: Centralized Pair Mappings
All symbol/pair mappings are centralized in `types.py` to ensure consistency across components.

### REC-006: Two Candle Types
- `Candle`: Lightweight type optimized for database queries
- `HistoricalCandle`: Full domain type with additional computed properties

### REC-009: Retention Policies
Automatic data retention to prevent unbounded storage growth:
- Trades: 90 days
- Candles: 365 days

### REC-010: Performance Optimization
Uses `itertuples()` instead of `iterrows()` for ~100x speedup in CSV processing.

## Dependencies

### Required
- `asyncpg`: Async PostgreSQL driver
- `aiohttp`: Async HTTP client (for API calls)

### Optional
- `pandas`: For CSV import functionality

## Docker Deployment

The system uses Docker Compose with TimescaleDB:

```yaml
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    ports:
      - "5433:5432"
    environment:
      POSTGRES_USER: trading
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: kraken_data
```

PostgreSQL is tuned for time-series workloads:
- `shared_buffers=2GB`
- `effective_cache_size=6GB`
- `work_mem=64MB`
- `max_parallel_workers_per_gather=4`

## Quality Attributes

| Attribute | Implementation |
|-----------|---------------|
| **Performance** | Buffered writes, COPY protocol, connection pooling |
| **Reliability** | Retry logic, graceful error handling, data validation |
| **Scalability** | TimescaleDB hypertables, automatic compression |
| **Maintainability** | Frozen dataclasses, centralized configuration |
| **Security** | No default credentials, explicit configuration required |

## Current Data Holdings (As of 2025-12-17)

### Summary

| Metric | Value |
|--------|-------|
| Database Size | 845 MB |
| Symbols Tracked | 3 (XRP/USDT, BTC/USDT, XRP/BTC) |
| Historical Coverage | **5-9 years** (via continuous aggregates) |
| Base 1m Candle Retention | 365 days |
| Trade Retention | 90 days |
| Continuous Aggregates | 8 timeframes with full history |

### Historical Data in Aggregates

| Symbol | Daily Candles | Coverage |
|--------|---------------|----------|
| XRP/BTC | 3,355 | 2016-07-19 → Present (9 years) |
| BTC/USDT | 2,190 | 2019-12-19 → Present (5 years) |
| XRP/USDT | 2,057 | 2020-04-30 → Present (5 years) |

### Aggregate Holdings

| Timeframe | Records |
|-----------|---------|
| 5-minute | 1,551,789 |
| 1-hour | 178,494 |
| 1-day | 7,602 |
| 1-week | 1,090 |

### Base Table Coverage (Subject to Retention)

| Symbol | 1m Candles | Trade Count | Trade History |
|--------|------------|-------------|---------------|
| BTC/USDT | 343,519 | 538,692 | 90 days |
| XRP/USDT | 219,965 | 247,116 | 90 days |
| XRP/BTC | 219,739 | 138,604 | 90 days |

### Known Data Quality Issues

1. **1m Candle Gaps**: ~65% gap rate in recent data (needs gap filler)
2. **Historical 1m Data**: Not available (use 5m+ aggregates for backtesting)
3. **Trade Data Limited**: 90-day retention - tick-level analysis limited to recent data

### Data Not Currently Collected

| Data Type | API Endpoint | Priority |
|-----------|-------------|----------|
| Order Book Depth | `/0/public/Depth` | High |
| Ticker Snapshots | `/0/public/Ticker` | Medium |
| Spread History | `/0/public/Spread` | Medium |
| Trade History (own) | `/0/private/TradesHistory` | High |
| Ledger Entries | `/0/private/Ledgers` | Medium |

See [Kraken Data Gap Analysis](../../api/kraken-data-gap-analysis.md) for detailed recommendations.

## Related Components

- **ws_paper_tester**: WebSocket client that produces trade data
- **Backtester**: Consumes historical data for strategy testing
- **Strategy Framework**: Uses warmup data for indicator initialization
