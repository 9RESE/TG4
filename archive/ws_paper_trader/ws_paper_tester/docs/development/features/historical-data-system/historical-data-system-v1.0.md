# Historical Data System v1.0

**Version:** 1.0.0
**Date:** 2025-12-15
**Status:** Implemented
**Author:** Trading Bot Team

---

## Overview

The Historical Data System provides persistent storage and retrieval of historical market data using TimescaleDB (PostgreSQL with time-series optimization). This enables comprehensive backtesting, strategy warmup, and multi-timeframe analysis.

### Key Features

- **TimescaleDB Storage**: Hypertables with automatic time-based partitioning
- **Continuous Aggregates**: Auto-compute higher timeframes (5m, 15m, 1h, 4h, 1d, 1w)
- **90%+ Compression**: Columnar compression for historical data
- **Gap Filler**: Automatic detection and filling of data gaps on startup
- **Real-Time Persistence**: WebSocket data writer for live data storage
- **Historical Provider**: Query API for backtesting and strategy warmup

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Historical Data System                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    TimescaleDB (PostgreSQL 15)               │   │
│  │  - Hypertables for time-series optimization                  │   │
│  │  - Continuous aggregates for automatic rollups               │   │
│  │  - Native compression (90%+ reduction)                       │   │
│  │  - Parallel query execution                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│              ┌───────────────┼───────────────┐                      │
│              │               │               │                      │
│              ▼               ▼               ▼                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Trades     │  │   Candles    │  │  External    │              │
│  │  Hypertable  │  │  Hypertable  │  │  Indicators  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Types (`data/types.py`)

| Type | Description |
|------|-------------|
| `HistoricalTrade` | Individual trade tick with price, volume, side |
| `HistoricalCandle` | OHLCV candle with computed properties |
| `DataGap` | Gap description for detection/filling |
| `TradeRecord` | Mutable record for database insertion |
| `CandleRecord` | Mutable record for database insertion |

### 2. DatabaseWriter (`data/websocket_db_writer.py`)

Buffered async writer for real-time data persistence:

- Configurable buffer sizes (default: 100 trades)
- Automatic flush on buffer size or time interval
- COPY protocol for efficient bulk inserts
- Conflict handling with upsert

### 3. HistoricalDataProvider (`data/historical_provider.py`)

Query API for historical data:

| Method | Description |
|--------|-------------|
| `get_candles()` | Query candles in time range |
| `get_latest_candles()` | Get N most recent candles |
| `replay_candles()` | Stream candles for backtesting |
| `get_warmup_data()` | Get indicator warmup data |
| `get_multi_timeframe_candles()` | Get MTF aligned data |

### 4. GapFiller (`data/gap_filler.py`)

Automatic gap detection and filling:

- Runs on startup before WebSocket connection
- Small gaps (< 12h): Uses OHLC REST API
- Large gaps (>= 12h): Uses Trades REST API
- Updates sync status after filling
- Refreshes continuous aggregates

### 5. BulkCSVImporter (`data/bulk_csv_importer.py`)

Import Kraken historical CSV files:

- Parses Kraken OHLCVT format
- Maps symbol names to our format
- Batch inserts with conflict handling
- Only imports 1-minute data (others via aggregates)

### 6. HistoricalBackfill (`data/historical_backfill.py`)

Fetch complete trade history from Kraken:

- Paginated fetching with rate limiting
- Builds candles from raw trades
- Resume capability via sync status

## Database Schema

### Tables

| Table | Description | Partitioning |
|-------|-------------|--------------|
| `trades` | Individual trade ticks | Daily chunks |
| `candles` | Base 1-minute candles | Weekly chunks |
| `external_indicators` | External data (Fear & Greed, etc.) | Monthly chunks |
| `data_sync_status` | Sync state for gap detection | None |
| `backtest_runs` | Backtest results | None |

### Continuous Aggregates

| View | Source | Interval |
|------|--------|----------|
| `candles_5m` | candles (1m) | 5 minutes |
| `candles_15m` | candles (1m) | 15 minutes |
| `candles_30m` | candles (1m) | 30 minutes |
| `candles_1h` | candles (1m) | 1 hour |
| `candles_4h` | candles_1h | 4 hours |
| `candles_12h` | candles_1h | 12 hours |
| `candles_1d` | candles_1h | 1 day |
| `candles_1w` | candles_1d | 1 week |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql://trading:changeme@localhost:5432/kraken_data` |
| `DB_PASSWORD` | Database password | `changeme` |
| `KRAKEN_API_KEY` | Kraken API key (optional) | None |
| `KRAKEN_API_SECRET` | Kraken API secret (optional) | None |

### Docker Compose

The system includes a `docker-compose.yml` with:

- **TimescaleDB**: PostgreSQL 15 with TimescaleDB extension
- **PgAdmin** (optional): Database management UI

## Files Created

```
ws_paper_tester/
├── data/
│   ├── __init__.py              # Module exports
│   ├── types.py                 # Data types
│   ├── websocket_db_writer.py   # Real-time writer
│   ├── historical_provider.py   # Query API
│   ├── gap_filler.py            # Gap detection/filling
│   ├── bulk_csv_importer.py     # CSV import
│   └── historical_backfill.py   # API backfill
├── scripts/
│   ├── init-db.sql              # Database schema
│   └── continuous-aggregates.sql # Multi-timeframe views
├── docker-compose.yml           # TimescaleDB deployment
├── .env.example                 # Environment template
├── main_with_historical.py      # Extended entry point
└── tests/
    └── test_historical_data.py  # Unit tests
```

## Related Documentation

- [ADR-001: Historical Data Storage](/docs/development/plans/historical-data-system/ADR-001-historical-data-storage.md)
- [Historical Data System Design](/docs/development/plans/historical-data-system/historical-data-system.md)
- [User Guide: Operating with Historical Data](/docs/user/how-to/operate-historical-data.md)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-15 | Initial implementation |
