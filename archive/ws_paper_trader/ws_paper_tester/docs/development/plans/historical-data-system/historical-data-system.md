# Historical Data System for Backtesting

**Version:** 1.0.0
**Date:** 2025-12-15
**Status:** Planned
**Author:** Trading Bot Team

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Database Architecture](#database-architecture)
4. [Data Types and Timeframes](#data-types-and-timeframes)
5. [Schema Design](#schema-design)
6. [Data Ingestion Pipeline](#data-ingestion-pipeline)
7. [WebSocket Real-Time Population](#websocket-real-time-population)
8. [Gap Filler System](#gap-filler-system)
9. [Integration with ws_paper_tester](#integration-with-ws_paper_tester)
10. [Docker Deployment](#docker-deployment)
11. [Data Retention and Maintenance](#data-retention-and-maintenance)
12. [API Reference](#api-reference)

---

## Overview

### Purpose

This document outlines the design and implementation plan for a PostgreSQL-based historical data system to support backtesting of trading strategies in the `ws_paper_tester` framework.

### Goals

- Store complete historical trade and OHLCV data for all traded symbols
- Enable accurate backtesting against years of historical data
- Maintain real-time synchronization with live WebSocket data
- Automatically detect and fill data gaps on startup
- Support multi-timeframe analysis with efficient aggregation

### Current Limitations

The `ws_paper_tester` currently operates with:
- In-memory candle storage (100 candles per timeframe per symbol)
- No persistent historical data
- Backtesting limited to real-time paper trading simulation
- No parameter optimization or walk-forward testing capability

---

## Data Sources

### Kraken API Endpoints

| Source | Endpoint | Data Available | Rate Limit | Use Case |
|--------|----------|---------------|------------|----------|
| **OHLC REST** | `/0/public/OHLC` | 720 most recent candles | 1 req/sec | Recent data only |
| **Trades REST** | `/0/public/Trades` | Entire history since market inception | 1 req/sec | Historical backfill |
| **WebSocket v2** | `wss://ws.kraken.com/v2` | Real-time trades, OHLC, orderbook | Unlimited | Live data sync |
| **CSV Downloads** | support.kraken.com | OHLCVT to Q3 2024 | N/A | Bulk initial import |

### Kraken API Authentication

```python
# Authentication recommended for higher rate limits
# Free account required (no trading/deposits needed)
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')
```

### Third-Party Sources (Optional)

| Source | URL | Data Type | Cost |
|--------|-----|-----------|------|
| CryptoDataDownload | cryptodatadownload.com/data/kraken/ | Daily OHLCV CSV | Free |
| Kaiko | kaiko.com | Tick data | Paid |

---

## Database Architecture

### Technology Stack

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

### Why TimescaleDB?

1. **Time-Series Optimized**: Built for exactly this use case
2. **Hypertables**: Automatic partitioning by time chunks
3. **Continuous Aggregates**: Auto-compute higher timeframes from 1-minute data
4. **Compression**: 90-95% storage reduction with columnar compression
5. **PostgreSQL Compatible**: Full SQL support, asyncpg compatibility

---

## Data Types and Timeframes

### Symbols

| Symbol | Kraken Pair | Description | Priority |
|--------|-------------|-------------|----------|
| `XRP/USDT` | `XRPUSDT` | Primary trading pair | High |
| `BTC/USDT` | `BTCUSDT` | Reference/correlation pair | High |
| `XRP/BTC` | `XRPXBT` | Ratio trading pair | High |
| `ETH/USDT` | `ETHUSDT` | Expansion pair | Medium |
| `SOL/USDT` | `SOLUSDT` | Expansion pair | Low |

### Timeframes

| Interval | Minutes | Storage | Aggregation Source | Use Case |
|----------|---------|---------|-------------------|----------|
| **1m** | 1 | Primary | WebSocket/Trades API | Base data, indicator calculation |
| **5m** | 5 | Continuous Aggregate | 1m candles | WaveTrend, short-term signals |
| **15m** | 15 | Continuous Aggregate | 1m candles | MTF trend confirmation |
| **30m** | 30 | Continuous Aggregate | 1m candles | Swing analysis |
| **1h** | 60 | Continuous Aggregate | 1m candles | Regime detection, hourly trends |
| **4h** | 240 | Continuous Aggregate | 1m candles | Major trend analysis |
| **12h** | 720 | Continuous Aggregate | 1m candles | Daily bias |
| **1d** | 1440 | Continuous Aggregate | 1m candles | Long-term regime, correlations |
| **1w** | 10080 | Continuous Aggregate | 1d candles | Weekly analysis |

### Data Types

#### Trade Data (Highest Granularity)

```python
@dataclass(frozen=True)
class HistoricalTrade:
    """Individual trade tick from Kraken."""
    id: int                    # Unique trade ID
    symbol: str                # 'XRP/USDT'
    timestamp: datetime        # Nanosecond precision
    price: Decimal             # Execution price
    volume: Decimal            # Trade volume
    side: Literal['buy', 'sell']  # Taker side
    order_type: str            # 'market', 'limit'
    misc: str                  # Miscellaneous flags
```

#### OHLCV Candle Data

```python
@dataclass(frozen=True)
class HistoricalCandle:
    """OHLCV candle with additional metrics."""
    symbol: str                # 'XRP/USDT'
    timestamp: datetime        # Candle open time (UTC)
    interval_minutes: int      # 1, 5, 15, 60, etc.
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal            # Base asset volume
    quote_volume: Decimal      # Quote asset volume (USDT)
    trade_count: int           # Number of trades in candle
    vwap: Decimal              # Volume-weighted average price

    # Computed fields for strategy use
    @property
    def typical_price(self) -> Decimal:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> Decimal:
        return self.high - self.low
```

#### External Indicator Data

```python
@dataclass(frozen=True)
class ExternalIndicator:
    """External market indicators."""
    timestamp: datetime
    indicator_name: str        # 'fear_greed', 'btc_dominance', etc.
    value: Decimal
    source: str                # 'alternative.me', 'coingecko'
```

### Estimated Data Volumes

| Symbol | Data Since | Est. Trades | Trade Storage | 1m Candles | Candle Storage |
|--------|-----------|-------------|---------------|------------|----------------|
| XRP/USDT | 2019-01 | ~50M | ~5 GB raw | ~3.1M | ~300 MB |
| BTC/USDT | 2017-01 | ~200M | ~20 GB raw | ~4.2M | ~500 MB |
| XRP/BTC | 2018-01 | ~30M | ~3 GB raw | ~3.6M | ~200 MB |
| **Total** | - | ~280M | ~28 GB | ~11M | ~1 GB |

**With TimescaleDB Compression:** ~3 GB total (90% reduction)

---

## Schema Design

### Core Tables

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================
-- TRADES TABLE (Highest Granularity)
-- ============================================
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
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Enable compression (after 7 days)
ALTER TABLE trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);
SELECT add_compression_policy('trades', INTERVAL '7 days');

-- ============================================
-- CANDLES TABLE (Base 1-minute data)
-- ============================================
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

-- Convert to hypertable with weekly chunks
SELECT create_hypertable('candles', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Enable compression (after 30 days)
ALTER TABLE candles SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, interval_minutes',
    timescaledb.compress_orderby = 'timestamp DESC'
);
SELECT add_compression_policy('candles', INTERVAL '30 days');

-- ============================================
-- EXTERNAL INDICATORS TABLE
-- ============================================
CREATE TABLE external_indicators (
    timestamp TIMESTAMPTZ NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    value DECIMAL(20, 10) NOT NULL,
    source VARCHAR(50),
    PRIMARY KEY (timestamp, indicator_name)
);

SELECT create_hypertable('external_indicators', 'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================
-- DATA SYNC STATUS TABLE (For gap detection)
-- ============================================
CREATE TABLE data_sync_status (
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(20) NOT NULL,  -- 'trades', 'candles_1m', etc.
    oldest_timestamp TIMESTAMPTZ,
    newest_timestamp TIMESTAMPTZ,
    last_sync_at TIMESTAMPTZ DEFAULT NOW(),
    last_kraken_since BIGINT,  -- Kraken 'since' parameter for continuation
    total_records BIGINT DEFAULT 0,
    PRIMARY KEY (symbol, data_type)
);

-- ============================================
-- BACKTEST RUNS TABLE (Results tracking)
-- ============================================
CREATE TABLE backtest_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name VARCHAR(100) NOT NULL,
    symbols TEXT[] NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    parameters JSONB NOT NULL,
    metrics JSONB NOT NULL,
    trades JSONB,
    equity_curve JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_backtest_strategy ON backtest_runs (strategy_name, created_at DESC);
```

### Continuous Aggregates (Auto-Rollup)

```sql
-- ============================================
-- 5-MINUTE CANDLES (from 1m)
-- ============================================
CREATE MATERIALIZED VIEW candles_5m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('5 minutes', timestamp) AS timestamp,
    5::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol, time_bucket('5 minutes', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_5m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes'
);

-- ============================================
-- 15-MINUTE CANDLES (from 1m)
-- ============================================
CREATE MATERIALIZED VIEW candles_15m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('15 minutes', timestamp) AS timestamp,
    15::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol, time_bucket('15 minutes', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_15m',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- ============================================
-- 30-MINUTE CANDLES (from 1m)
-- ============================================
CREATE MATERIALIZED VIEW candles_30m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('30 minutes', timestamp) AS timestamp,
    30::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol, time_bucket('30 minutes', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_30m',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '30 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- ============================================
-- 1-HOUR CANDLES (from 1m)
-- ============================================
CREATE MATERIALIZED VIEW candles_1h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', timestamp) AS timestamp,
    60::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles
WHERE interval_minutes = 1
GROUP BY symbol, time_bucket('1 hour', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_1h',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- ============================================
-- 4-HOUR CANDLES (from 1h)
-- ============================================
CREATE MATERIALIZED VIEW candles_4h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('4 hours', timestamp) AS timestamp,
    240::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles_1h
GROUP BY symbol, time_bucket('4 hours', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_4h',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '4 hours',
    schedule_interval => INTERVAL '4 hours'
);

-- ============================================
-- 12-HOUR CANDLES (from 1h)
-- ============================================
CREATE MATERIALIZED VIEW candles_12h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('12 hours', timestamp) AS timestamp,
    720::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles_1h
GROUP BY symbol, time_bucket('12 hours', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_12h',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '12 hours',
    schedule_interval => INTERVAL '12 hours'
);

-- ============================================
-- DAILY CANDLES (from 1h)
-- ============================================
CREATE MATERIALIZED VIEW candles_1d
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 day', timestamp) AS timestamp,
    1440::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles_1h
GROUP BY symbol, time_bucket('1 day', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_1d',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day'
);

-- ============================================
-- WEEKLY CANDLES (from 1d)
-- ============================================
CREATE MATERIALIZED VIEW candles_1w
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 week', timestamp) AS timestamp,
    10080::smallint AS interval_minutes,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trade_count) AS trade_count,
    sum(volume * vwap) / nullif(sum(volume), 0) AS vwap
FROM candles_1d
GROUP BY symbol, time_bucket('1 week', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_1w',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '1 week',
    schedule_interval => INTERVAL '1 week'
);
```

### Indexes

```sql
-- Optimized indexes for common query patterns
CREATE INDEX idx_candles_symbol_interval_ts
    ON candles (symbol, interval_minutes, timestamp DESC);

CREATE INDEX idx_trades_symbol_ts
    ON trades (symbol, timestamp DESC);

CREATE INDEX idx_external_indicator_name_ts
    ON external_indicators (indicator_name, timestamp DESC);

-- Composite index for range queries
CREATE INDEX idx_candles_range
    ON candles (symbol, interval_minutes, timestamp)
    WHERE interval_minutes = 1;
```

---

## Data Ingestion Pipeline

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Data Ingestion Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: BULK IMPORT                    PHASE 2: HISTORICAL BACKFILL       │
│  ┌─────────────────────┐                 ┌─────────────────────────┐        │
│  │ Kraken CSV Files    │                 │ Kraken Trades REST API  │        │
│  │ (Q3 2024 and prior) │                 │ (fills gaps to present) │        │
│  └──────────┬──────────┘                 └───────────┬─────────────┘        │
│             │                                        │                       │
│             ▼                                        ▼                       │
│  ┌─────────────────────┐                 ┌─────────────────────────┐        │
│  │ CSV Importer        │                 │ Async Trade Fetcher     │        │
│  │ (pandas + psycopg)  │                 │ (aiohttp + asyncpg)     │        │
│  └──────────┬──────────┘                 └───────────┬─────────────┘        │
│             │                                        │                       │
│             └────────────────┬───────────────────────┘                       │
│                              │                                               │
│                              ▼                                               │
│                   ┌─────────────────────┐                                    │
│                   │    TimescaleDB      │                                    │
│                   │    (PostgreSQL)     │                                    │
│                   └──────────┬──────────┘                                    │
│                              │                                               │
│  PHASE 3: REAL-TIME SYNC    │                                               │
│  ┌─────────────────────┐    │                                               │
│  │ Kraken WebSocket    │────┘                                               │
│  │ (continuous stream) │                                                    │
│  └─────────────────────┘                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Bulk CSV Import

```python
"""
bulk_csv_importer.py - Import Kraken historical CSV files
"""
import pandas as pd
import asyncio
import asyncpg
from pathlib import Path
from datetime import datetime
from typing import List
import logging

logger = logging.getLogger(__name__)

class BulkCSVImporter:
    """Import Kraken historical CSV files into TimescaleDB."""

    # Kraken CSV column mapping
    CSV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']

    # Symbol mapping (Kraken CSV naming to our format)
    SYMBOL_MAP = {
        'XRPUSDT': 'XRP/USDT',
        'XBTUSDT': 'BTC/USDT',
        'BTCUSDT': 'BTC/USDT',
        'XRPXBT': 'XRP/BTC',
        'XRPBTC': 'XRP/BTC',
        'ETHUSDT': 'ETH/USDT',
    }

    # Interval mapping (filename suffix to minutes)
    INTERVAL_MAP = {
        '1': 1,
        '5': 5,
        '15': 15,
        '30': 30,
        '60': 60,
        '240': 240,
        '720': 720,
        '1440': 1440,
    }

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool: asyncpg.Pool = None

    async def connect(self):
        """Establish database connection pool."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=10,
            command_timeout=300
        )
        logger.info("Connected to TimescaleDB")

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()

    async def import_csv_file(
        self,
        filepath: Path,
        symbol: str,
        interval_minutes: int
    ) -> int:
        """
        Import a single CSV file into the candles table.

        Args:
            filepath: Path to CSV file
            symbol: Trading pair symbol (e.g., 'XRP/USDT')
            interval_minutes: Candle interval in minutes

        Returns:
            Number of rows imported
        """
        logger.info(f"Importing {filepath} for {symbol} ({interval_minutes}m)")

        # Read CSV file
        df = pd.read_csv(
            filepath,
            names=self.CSV_COLUMNS,
            header=None
        )

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

        # Calculate VWAP (approximation from OHLC)
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3

        # Add symbol and interval
        df['symbol'] = symbol
        df['interval_minutes'] = interval_minutes

        # Prepare records for bulk insert
        records = [
            (
                row['symbol'],
                row['timestamp'],
                row['interval_minutes'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                None,  # quote_volume (not in CSV)
                int(row['trades']) if pd.notna(row['trades']) else None,
                float(row['vwap'])
            )
            for _, row in df.iterrows()
        ]

        # Bulk insert with conflict handling
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO candles
                    (symbol, timestamp, interval_minutes, open, high, low, close,
                     volume, quote_volume, trade_count, vwap)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (timestamp, symbol, interval_minutes)
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    trade_count = EXCLUDED.trade_count,
                    vwap = EXCLUDED.vwap
                """,
                records
            )

            # Update sync status
            if records:
                oldest = min(r[1] for r in records)
                newest = max(r[1] for r in records)
                await conn.execute(
                    """
                    INSERT INTO data_sync_status
                        (symbol, data_type, oldest_timestamp, newest_timestamp, total_records)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (symbol, data_type) DO UPDATE SET
                        oldest_timestamp = LEAST(data_sync_status.oldest_timestamp, EXCLUDED.oldest_timestamp),
                        newest_timestamp = GREATEST(data_sync_status.newest_timestamp, EXCLUDED.newest_timestamp),
                        total_records = data_sync_status.total_records + EXCLUDED.total_records,
                        last_sync_at = NOW()
                    """,
                    symbol, f'candles_{interval_minutes}m', oldest, newest, len(records)
                )

        logger.info(f"Imported {len(records)} candles from {filepath}")
        return len(records)

    async def import_directory(self, directory: Path) -> dict:
        """
        Import all CSV files from a directory.

        Expected structure:
            directory/
                XRPUSDT_1.csv
                XRPUSDT_5.csv
                XRPUSDT_60.csv
                BTCUSDT_1.csv
                ...

        Returns:
            Dictionary of {symbol: {interval: count}}
        """
        results = {}
        csv_files = list(directory.glob('*.csv'))

        logger.info(f"Found {len(csv_files)} CSV files in {directory}")

        for filepath in csv_files:
            # Parse filename: XRPUSDT_1.csv -> symbol=XRP/USDT, interval=1
            parts = filepath.stem.split('_')
            if len(parts) != 2:
                logger.warning(f"Skipping unrecognized file: {filepath}")
                continue

            pair_code, interval_str = parts

            # Map to our symbol format
            symbol = self.SYMBOL_MAP.get(pair_code.upper())
            if not symbol:
                logger.warning(f"Unknown pair code: {pair_code}")
                continue

            # Map interval
            interval = self.INTERVAL_MAP.get(interval_str)
            if not interval:
                logger.warning(f"Unknown interval: {interval_str}")
                continue

            # Only import 1-minute data (others computed via continuous aggregates)
            if interval != 1:
                logger.info(f"Skipping {filepath} (only 1m data needed)")
                continue

            try:
                count = await self.import_csv_file(filepath, symbol, interval)

                if symbol not in results:
                    results[symbol] = {}
                results[symbol][interval] = count

            except Exception as e:
                logger.error(f"Failed to import {filepath}: {e}")

        return results


async def main():
    """Run bulk CSV import."""
    import os

    db_url = os.getenv('DATABASE_URL', 'postgresql://trading:password@localhost:5432/kraken_data')
    csv_dir = Path('./data/kraken_csv')

    importer = BulkCSVImporter(db_url)

    try:
        await importer.connect()
        results = await importer.import_directory(csv_dir)

        print("\nImport Summary:")
        for symbol, intervals in results.items():
            for interval, count in intervals.items():
                print(f"  {symbol} {interval}m: {count:,} candles")

    finally:
        await importer.close()


if __name__ == '__main__':
    asyncio.run(main())
```

### Phase 2: Historical Trades Backfill

```python
"""
historical_backfill.py - Fetch complete trade history from Kraken API
"""
import asyncio
import aiohttp
import asyncpg
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncIterator, List, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)

class KrakenTradesBackfill:
    """Fetch and store complete trade history from Kraken REST API."""

    BASE_URL = 'https://api.kraken.com'

    # Rate limiting: 1 request per second for public endpoints
    RATE_LIMIT_DELAY = 1.1

    # Kraken pair names
    PAIR_MAP = {
        'XRP/USDT': 'XRPUSDT',
        'BTC/USDT': 'XBTUSDT',
        'XRP/BTC': 'XRPXBT',
    }

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool: asyncpg.Pool = None
        self.session: aiohttp.ClientSession = None

    async def connect(self):
        """Initialize connections."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=10
        )
        self.session = aiohttp.ClientSession()
        logger.info("Connected to database and initialized HTTP session")

    async def close(self):
        """Close connections."""
        if self.session:
            await self.session.close()
        if self.pool:
            await self.pool.close()

    async def fetch_trades_page(
        self,
        pair: str,
        since: int = 0
    ) -> Tuple[List[dict], int]:
        """
        Fetch a page of trades from Kraken API.

        Args:
            pair: Kraken pair name (e.g., 'XRPUSDT')
            since: Starting timestamp (nanoseconds)

        Returns:
            Tuple of (trades list, last timestamp for pagination)
        """
        url = f'{self.BASE_URL}/0/public/Trades'
        params = {'pair': pair, 'since': since}

        async with self.session.get(url, params=params) as response:
            data = await response.json()

            if data.get('error'):
                raise Exception(f"Kraken API error: {data['error']}")

            result = data['result']

            # Get trades (key is the pair name)
            trades_key = list(result.keys())[0]
            if trades_key == 'last':
                trades_key = list(result.keys())[1] if len(result) > 1 else None

            trades = result.get(trades_key, [])
            last = int(result.get('last', 0))

            return trades, last

    async def fetch_all_trades(
        self,
        symbol: str,
        start_since: int = 0
    ) -> AsyncIterator[List[dict]]:
        """
        Generator that fetches all trades for a symbol.

        Args:
            symbol: Our symbol format (e.g., 'XRP/USDT')
            start_since: Starting timestamp (0 for beginning)

        Yields:
            Batches of trade records
        """
        pair = self.PAIR_MAP.get(symbol)
        if not pair:
            raise ValueError(f"Unknown symbol: {symbol}")

        since = start_since
        total_fetched = 0

        while True:
            try:
                trades, last = await self.fetch_trades_page(pair, since)

                if not trades:
                    logger.info(f"{symbol}: No more trades after {since}")
                    break

                total_fetched += len(trades)
                logger.info(
                    f"{symbol}: Fetched {len(trades)} trades "
                    f"(total: {total_fetched:,}, since: {since})"
                )

                yield trades

                # Update since for next page
                since = last

                # Rate limiting
                await asyncio.sleep(self.RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"Error fetching trades: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def store_trades(self, symbol: str, trades: List[list]):
        """
        Store trades in database.

        Kraken trade format: [price, volume, time, side, type, misc]
        """
        if not trades:
            return

        records = []
        for trade in trades:
            price, volume, timestamp, side, order_type, misc = trade
            records.append((
                symbol,
                datetime.fromtimestamp(float(timestamp), tz=timezone.utc),
                Decimal(str(price)),
                Decimal(str(volume)),
                'buy' if side == 'b' else 'sell',
                order_type,
                misc
            ))

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO trades
                    (symbol, timestamp, price, volume, side, order_type, misc)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT DO NOTHING
                """,
                records
            )

    async def build_candles_from_trades(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ):
        """Build 1-minute candles from stored trades."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO candles (symbol, timestamp, interval_minutes, open, high, low, close, volume, trade_count, vwap)
                SELECT
                    symbol,
                    time_bucket('1 minute', timestamp) AS timestamp,
                    1 AS interval_minutes,
                    first(price, timestamp) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price, timestamp) AS close,
                    sum(volume) AS volume,
                    count(*) AS trade_count,
                    sum(price * volume) / sum(volume) AS vwap
                FROM trades
                WHERE symbol = $1
                  AND timestamp >= $2
                  AND timestamp < $3
                GROUP BY symbol, time_bucket('1 minute', timestamp)
                ON CONFLICT (timestamp, symbol, interval_minutes) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    trade_count = EXCLUDED.trade_count,
                    vwap = EXCLUDED.vwap
                """,
                symbol, start_time, end_time
            )

    async def backfill_symbol(self, symbol: str, since: int = 0) -> int:
        """
        Backfill complete trade history for a symbol.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            since: Starting timestamp (0 for complete history)

        Returns:
            Total trades imported
        """
        logger.info(f"Starting backfill for {symbol} from {since}")

        total_trades = 0
        batch_count = 0
        last_timestamp = None

        async for trades in self.fetch_all_trades(symbol, since):
            await self.store_trades(symbol, trades)
            total_trades += len(trades)
            batch_count += 1

            # Track progress
            if trades:
                last_timestamp = datetime.fromtimestamp(
                    float(trades[-1][2]),
                    tz=timezone.utc
                )

            # Build candles every 100 batches
            if batch_count % 100 == 0 and last_timestamp:
                logger.info(f"{symbol}: Building candles up to {last_timestamp}")
                # This will be handled by continuous aggregates

        logger.info(f"{symbol}: Backfill complete. Total trades: {total_trades:,}")
        return total_trades

    async def get_resume_point(self, symbol: str) -> int:
        """Get the point to resume backfill from."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT last_kraken_since
                FROM data_sync_status
                WHERE symbol = $1 AND data_type = 'trades'
                """,
                symbol
            )
            return row['last_kraken_since'] if row else 0


async def main():
    """Run historical backfill."""
    import os

    db_url = os.getenv('DATABASE_URL', 'postgresql://trading:password@localhost:5432/kraken_data')
    symbols = ['XRP/USDT', 'BTC/USDT', 'XRP/BTC']

    backfill = KrakenTradesBackfill(db_url)

    try:
        await backfill.connect()

        for symbol in symbols:
            # Resume from last position if available
            since = await backfill.get_resume_point(symbol)
            await backfill.backfill_symbol(symbol, since)

    finally:
        await backfill.close()


if __name__ == '__main__':
    asyncio.run(main())
```

---

## WebSocket Real-Time Population

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WebSocket Real-Time Data Flow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐                                                     │
│  │  Kraken WebSocket  │                                                     │
│  │  wss://ws.kraken   │                                                     │
│  │  .com/v2           │                                                     │
│  └─────────┬──────────┘                                                     │
│            │                                                                 │
│            │ subscribe: trade, ohlc                                         │
│            ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    KrakenWSClient (Extended)                        │    │
│  │                                                                     │    │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │    │
│  │  │ Trade Handler   │    │ OHLC Handler    │    │ Ticker Handler │  │    │
│  │  │                 │    │                 │    │                │  │    │
│  │  │ on_trade()      │    │ on_ohlc()       │    │ on_ticker()    │  │    │
│  │  └────────┬────────┘    └────────┬────────┘    └────────────────┘  │    │
│  │           │                      │                                  │    │
│  └───────────┼──────────────────────┼──────────────────────────────────┘    │
│              │                      │                                        │
│              ▼                      ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DatabaseWriter (Async)                           │    │
│  │                                                                     │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │                     Write Buffer                             │   │    │
│  │  │  Trades: [...]  (flush every 100 trades or 5 seconds)       │   │    │
│  │  │  Candles: [...]  (flush every complete candle)              │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │                              │                                      │    │
│  │                              ▼                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │              Batch Insert (asyncpg)                          │   │    │
│  │  │  - COPY for bulk trades                                      │   │    │
│  │  │  - Upsert for candles (handle late updates)                 │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
"""
websocket_db_writer.py - Write WebSocket data to TimescaleDB in real-time
"""
import asyncio
import asyncpg
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Trade record for database insertion."""
    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Decimal
    side: str


@dataclass
class CandleRecord:
    """Candle record for database insertion."""
    symbol: str
    timestamp: datetime
    interval_minutes: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trade_count: int
    vwap: Optional[Decimal] = None


class DatabaseWriter:
    """
    Asynchronous database writer with buffering for efficient batch inserts.

    Features:
    - Buffered writes to reduce database round-trips
    - Automatic flush on buffer size or time interval
    - Connection pooling for concurrent writes
    - Graceful error handling with retry logic
    """

    def __init__(
        self,
        db_url: str,
        trade_buffer_size: int = 100,
        trade_flush_interval: float = 5.0,
        candle_flush_interval: float = 1.0
    ):
        self.db_url = db_url
        self.trade_buffer_size = trade_buffer_size
        self.trade_flush_interval = trade_flush_interval
        self.candle_flush_interval = candle_flush_interval

        self.pool: asyncpg.Pool = None
        self.trade_buffer: deque[TradeRecord] = deque()
        self.candle_buffer: deque[CandleRecord] = deque()

        self._running = False
        self._flush_task: asyncio.Task = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Initialize database connection and start flush task."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )

        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

        logger.info("DatabaseWriter started")

    async def stop(self):
        """Stop writer and flush remaining data."""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_trades()
        await self._flush_candles()

        if self.pool:
            await self.pool.close()

        logger.info("DatabaseWriter stopped")

    async def write_trade(self, trade: TradeRecord):
        """
        Buffer a trade for batch insertion.

        Flushes immediately if buffer is full.
        """
        async with self._lock:
            self.trade_buffer.append(trade)

            if len(self.trade_buffer) >= self.trade_buffer_size:
                await self._flush_trades()

    async def write_candle(self, candle: CandleRecord):
        """
        Buffer a candle for insertion.

        Candles are upserted to handle updates to the current candle.
        """
        async with self._lock:
            self.candle_buffer.append(candle)

    async def _periodic_flush(self):
        """Periodic flush task."""
        while self._running:
            try:
                await asyncio.sleep(self.trade_flush_interval)

                async with self._lock:
                    if self.trade_buffer:
                        await self._flush_trades()
                    if self.candle_buffer:
                        await self._flush_candles()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic flush error: {e}")

    async def _flush_trades(self):
        """Flush trade buffer to database."""
        if not self.trade_buffer:
            return

        trades = list(self.trade_buffer)
        self.trade_buffer.clear()

        try:
            async with self.pool.acquire() as conn:
                # Use COPY for efficient bulk insert
                await conn.copy_records_to_table(
                    'trades',
                    records=[
                        (t.symbol, t.timestamp, t.price, t.volume, t.side, None, None)
                        for t in trades
                    ],
                    columns=['symbol', 'timestamp', 'price', 'volume', 'side', 'order_type', 'misc']
                )

            logger.debug(f"Flushed {len(trades)} trades to database")

        except Exception as e:
            logger.error(f"Failed to flush trades: {e}")
            # Re-add to buffer for retry
            self.trade_buffer.extendleft(reversed(trades))

    async def _flush_candles(self):
        """Flush candle buffer to database."""
        if not self.candle_buffer:
            return

        candles = list(self.candle_buffer)
        self.candle_buffer.clear()

        try:
            async with self.pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO candles
                        (symbol, timestamp, interval_minutes, open, high, low, close,
                         volume, trade_count, vwap)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (timestamp, symbol, interval_minutes)
                    DO UPDATE SET
                        high = GREATEST(candles.high, EXCLUDED.high),
                        low = LEAST(candles.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        trade_count = EXCLUDED.trade_count,
                        vwap = EXCLUDED.vwap
                    """,
                    [
                        (c.symbol, c.timestamp, c.interval_minutes, c.open, c.high,
                         c.low, c.close, c.volume, c.trade_count, c.vwap)
                        for c in candles
                    ]
                )

            logger.debug(f"Flushed {len(candles)} candles to database")

        except Exception as e:
            logger.error(f"Failed to flush candles: {e}")
            self.candle_buffer.extendleft(reversed(candles))

    async def update_sync_status(self, symbol: str, data_type: str, timestamp: datetime):
        """Update the sync status for gap detection."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO data_sync_status (symbol, data_type, newest_timestamp, last_sync_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (symbol, data_type) DO UPDATE SET
                    newest_timestamp = GREATEST(data_sync_status.newest_timestamp, EXCLUDED.newest_timestamp),
                    last_sync_at = NOW()
                """,
                symbol, data_type, timestamp
            )


class WebSocketDBIntegration:
    """
    Integration layer between Kraken WebSocket client and database writer.

    Hooks into the existing ws_paper_tester WebSocket client to persist data.
    """

    def __init__(self, db_writer: DatabaseWriter):
        self.db_writer = db_writer

        # Track current candles for proper OHLC handling
        self._current_candles: dict[tuple[str, int], CandleRecord] = {}

    async def on_trade(self, symbol: str, trade_data: dict):
        """
        Handle incoming trade from WebSocket.

        Called by KrakenWSClient on each trade message.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            trade_data: Trade data from Kraken WebSocket
                {
                    'price': '0.5234',
                    'qty': '100.5',
                    'timestamp': '2024-01-15T10:30:45.123456Z',
                    'side': 'buy'
                }
        """
        trade = TradeRecord(
            symbol=symbol,
            timestamp=datetime.fromisoformat(trade_data['timestamp'].replace('Z', '+00:00')),
            price=Decimal(trade_data['price']),
            volume=Decimal(trade_data['qty']),
            side=trade_data['side']
        )

        await self.db_writer.write_trade(trade)

    async def on_ohlc(self, symbol: str, ohlc_data: dict, interval: int = 1):
        """
        Handle incoming OHLC candle from WebSocket.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            ohlc_data: OHLC data from Kraken WebSocket
                {
                    'timestamp': '2024-01-15T10:30:00Z',
                    'open': '0.5200',
                    'high': '0.5250',
                    'low': '0.5190',
                    'close': '0.5234',
                    'volume': '50000.5',
                    'trades': 150,
                    'vwap': '0.5220'
                }
            interval: Candle interval in minutes
        """
        candle = CandleRecord(
            symbol=symbol,
            timestamp=datetime.fromisoformat(ohlc_data['timestamp'].replace('Z', '+00:00')),
            interval_minutes=interval,
            open=Decimal(ohlc_data['open']),
            high=Decimal(ohlc_data['high']),
            low=Decimal(ohlc_data['low']),
            close=Decimal(ohlc_data['close']),
            volume=Decimal(ohlc_data['volume']),
            trade_count=int(ohlc_data.get('trades', 0)),
            vwap=Decimal(ohlc_data['vwap']) if ohlc_data.get('vwap') else None
        )

        # Track for duplicate detection
        key = (symbol, interval)
        prev_candle = self._current_candles.get(key)

        # Only write if this is a new candle or an update to current
        if prev_candle is None or prev_candle.timestamp != candle.timestamp:
            # Previous candle is complete, flush it
            if prev_candle is not None:
                await self.db_writer.write_candle(prev_candle)

            self._current_candles[key] = candle
        else:
            # Update current candle
            self._current_candles[key] = candle

    async def flush_current_candles(self):
        """Flush all current candles (call on shutdown)."""
        for candle in self._current_candles.values():
            await self.db_writer.write_candle(candle)
        self._current_candles.clear()


# Integration with existing KrakenWSClient
def integrate_db_writer(ws_client, db_writer: DatabaseWriter):
    """
    Integrate database writer with existing WebSocket client.

    Usage:
        db_writer = DatabaseWriter(db_url)
        await db_writer.start()

        ws_client = KrakenWSClient(...)
        integration = integrate_db_writer(ws_client, db_writer)

        await ws_client.connect()
    """
    integration = WebSocketDBIntegration(db_writer)

    # Store original handlers
    original_on_trade = getattr(ws_client, 'on_trade', None)
    original_on_ohlc = getattr(ws_client, 'on_ohlc', None)

    async def wrapped_on_trade(symbol: str, trade_data: dict):
        # Write to database
        await integration.on_trade(symbol, trade_data)
        # Call original handler
        if original_on_trade:
            await original_on_trade(symbol, trade_data)

    async def wrapped_on_ohlc(symbol: str, ohlc_data: dict, interval: int = 1):
        # Write to database
        await integration.on_ohlc(symbol, ohlc_data, interval)
        # Call original handler
        if original_on_ohlc:
            await original_on_ohlc(symbol, ohlc_data, interval)

    # Monkey-patch handlers
    ws_client.on_trade = wrapped_on_trade
    ws_client.on_ohlc = wrapped_on_ohlc

    return integration
```

---

## Gap Filler System

### Overview

The gap filler runs on program startup to detect and fill any missing data between the last stored record and the current time.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Gap Filler System Flow                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STARTUP                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     1. Gap Detection                                │    │
│  │                                                                     │    │
│  │  For each (symbol, interval):                                       │    │
│  │    - Query data_sync_status.newest_timestamp                        │    │
│  │    - Compare to NOW()                                               │    │
│  │    - Calculate gap duration                                         │    │
│  │                                                                     │    │
│  │  Gaps found:                                                        │    │
│  │    XRP/USDT 1m: 2024-12-14 23:45:00 → 2024-12-15 08:30:00 (8.75h)  │    │
│  │    BTC/USDT 1m: 2024-12-14 23:50:00 → 2024-12-15 08:30:00 (8.67h)  │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     2. Gap Classification                           │    │
│  │                                                                     │    │
│  │  Small gap (< 12 hours):   Use OHLC REST API (fast)                │    │
│  │  Large gap (>= 12 hours):  Use Trades REST API (complete)          │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     3. Parallel Fill                                │    │
│  │                                                                     │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │    │
│  │  │ XRP/USDT     │    │ BTC/USDT     │    │ XRP/BTC      │          │    │
│  │  │ Gap Filler   │    │ Gap Filler   │    │ Gap Filler   │          │    │
│  │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │    │
│  │         │                   │                   │                   │    │
│  │         └───────────────────┼───────────────────┘                   │    │
│  │                             │                                       │    │
│  │                             ▼                                       │    │
│  │                   ┌─────────────────┐                              │    │
│  │                   │  TimescaleDB    │                              │    │
│  │                   └─────────────────┘                              │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     4. Validation                                   │    │
│  │                                                                     │    │
│  │  - Verify no gaps remain                                           │    │
│  │  - Update data_sync_status                                         │    │
│  │  - Refresh continuous aggregates                                   │    │
│  │  - Log completion summary                                          │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│                    CONTINUE TO WEBSOCKET SYNC                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
"""
gap_filler.py - Detect and fill data gaps on startup
"""
import asyncio
import aiohttp
import asyncpg
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataGap:
    """Represents a gap in historical data."""
    symbol: str
    data_type: str
    start_time: datetime
    end_time: datetime
    duration: timedelta

    @property
    def is_small(self) -> bool:
        """Small gaps can use OHLC API (720 candles max = 12 hours for 1m)."""
        return self.duration < timedelta(hours=12)

    @property
    def candles_needed(self) -> int:
        """Estimate number of 1-minute candles needed."""
        return int(self.duration.total_seconds() / 60)


class GapFiller:
    """
    Detect and fill gaps in historical data on startup.

    Strategy:
    1. Query data_sync_status for each symbol
    2. Identify gaps between newest_timestamp and now
    3. Use OHLC API for small gaps (< 12 hours)
    4. Use Trades API for large gaps (>= 12 hours)
    5. Update sync status after filling
    """

    KRAKEN_BASE_URL = 'https://api.kraken.com'

    SYMBOLS = ['XRP/USDT', 'BTC/USDT', 'XRP/BTC']

    PAIR_MAP = {
        'XRP/USDT': 'XRPUSDT',
        'BTC/USDT': 'XBTUSDT',
        'XRP/BTC': 'XRPXBT',
    }

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool: asyncpg.Pool = None
        self.session: aiohttp.ClientSession = None

    async def start(self):
        """Initialize connections."""
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
        self.session = aiohttp.ClientSession()
        logger.info("GapFiller initialized")

    async def stop(self):
        """Close connections."""
        if self.session:
            await self.session.close()
        if self.pool:
            await self.pool.close()

    async def detect_gaps(self) -> List[DataGap]:
        """
        Detect gaps for all symbols.

        Returns:
            List of DataGap objects describing missing data
        """
        gaps = []
        now = datetime.now(timezone.utc)

        async with self.pool.acquire() as conn:
            for symbol in self.SYMBOLS:
                # Check 1-minute candles
                row = await conn.fetchrow(
                    """
                    SELECT newest_timestamp
                    FROM data_sync_status
                    WHERE symbol = $1 AND data_type = 'candles_1m'
                    """,
                    symbol
                )

                if row and row['newest_timestamp']:
                    last_timestamp = row['newest_timestamp']
                    gap_duration = now - last_timestamp

                    # Only report gaps > 2 minutes (allow for processing delay)
                    if gap_duration > timedelta(minutes=2):
                        gaps.append(DataGap(
                            symbol=symbol,
                            data_type='candles_1m',
                            start_time=last_timestamp,
                            end_time=now,
                            duration=gap_duration
                        ))
                        logger.info(
                            f"Gap detected: {symbol} 1m candles "
                            f"from {last_timestamp} to {now} ({gap_duration})"
                        )
                else:
                    # No data at all - need full backfill
                    # Start from 30 days ago as minimum
                    start = now - timedelta(days=30)
                    gaps.append(DataGap(
                        symbol=symbol,
                        data_type='candles_1m',
                        start_time=start,
                        end_time=now,
                        duration=timedelta(days=30)
                    ))
                    logger.warning(f"No data found for {symbol} - need full backfill")

        return gaps

    async def fill_gap_ohlc(self, gap: DataGap) -> int:
        """
        Fill a small gap using OHLC REST API.

        Args:
            gap: DataGap to fill

        Returns:
            Number of candles inserted
        """
        pair = self.PAIR_MAP[gap.symbol]
        since = int(gap.start_time.timestamp())

        url = f'{self.KRAKEN_BASE_URL}/0/public/OHLC'
        params = {
            'pair': pair,
            'interval': 1,  # 1-minute candles
            'since': since
        }

        async with self.session.get(url, params=params) as response:
            data = await response.json()

            if data.get('error'):
                logger.error(f"Kraken OHLC API error: {data['error']}")
                return 0

            result = data['result']
            pair_key = [k for k in result.keys() if k != 'last'][0]
            candles = result.get(pair_key, [])

        if not candles:
            logger.info(f"No OHLC data returned for {gap.symbol}")
            return 0

        # Convert and insert candles
        records = []
        for c in candles:
            timestamp, open_, high, low, close, vwap, volume, count = c[:8]

            # Filter to gap period
            candle_time = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
            if candle_time < gap.start_time or candle_time >= gap.end_time:
                continue

            records.append((
                gap.symbol,
                candle_time,
                1,  # interval_minutes
                Decimal(str(open_)),
                Decimal(str(high)),
                Decimal(str(low)),
                Decimal(str(close)),
                Decimal(str(volume)),
                None,  # quote_volume
                int(count),
                Decimal(str(vwap))
            ))

        if records:
            async with self.pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO candles
                        (symbol, timestamp, interval_minutes, open, high, low, close,
                         volume, quote_volume, trade_count, vwap)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (timestamp, symbol, interval_minutes) DO UPDATE SET
                        high = GREATEST(candles.high, EXCLUDED.high),
                        low = LEAST(candles.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        trade_count = EXCLUDED.trade_count,
                        vwap = EXCLUDED.vwap
                    """,
                    records
                )

        logger.info(f"Filled {len(records)} candles for {gap.symbol} via OHLC API")
        return len(records)

    async def fill_gap_trades(self, gap: DataGap) -> int:
        """
        Fill a large gap using Trades REST API.

        This fetches raw trades and builds candles from them.

        Args:
            gap: DataGap to fill

        Returns:
            Number of trades processed
        """
        pair = self.PAIR_MAP[gap.symbol]
        since = int(gap.start_time.timestamp() * 1_000_000_000)  # Nanoseconds

        total_trades = 0

        while True:
            url = f'{self.KRAKEN_BASE_URL}/0/public/Trades'
            params = {'pair': pair, 'since': since}

            async with self.session.get(url, params=params) as response:
                data = await response.json()

                if data.get('error'):
                    logger.error(f"Kraken Trades API error: {data['error']}")
                    break

                result = data['result']
                pair_key = [k for k in result.keys() if k != 'last'][0]
                trades = result.get(pair_key, [])
                last = result.get('last', 0)

            if not trades:
                break

            # Check if we've passed the gap end time
            last_trade_time = datetime.fromtimestamp(float(trades[-1][2]), tz=timezone.utc)
            if last_trade_time >= gap.end_time:
                # Filter trades to gap period
                trades = [
                    t for t in trades
                    if datetime.fromtimestamp(float(t[2]), tz=timezone.utc) < gap.end_time
                ]

                if trades:
                    await self._store_trades_and_build_candles(gap.symbol, trades)
                    total_trades += len(trades)
                break

            await self._store_trades_and_build_candles(gap.symbol, trades)
            total_trades += len(trades)

            since = int(last)
            await asyncio.sleep(1.1)  # Rate limiting

        logger.info(f"Filled gap for {gap.symbol} with {total_trades} trades")
        return total_trades

    async def _store_trades_and_build_candles(self, symbol: str, trades: list):
        """Store trades and build 1-minute candles."""
        if not trades:
            return

        # Store raw trades
        trade_records = [
            (
                symbol,
                datetime.fromtimestamp(float(t[2]), tz=timezone.utc),
                Decimal(str(t[0])),
                Decimal(str(t[1])),
                'buy' if t[3] == 'b' else 'sell',
                t[4],
                t[5]
            )
            for t in trades
        ]

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO trades (symbol, timestamp, price, volume, side, order_type, misc)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT DO NOTHING
                """,
                trade_records
            )

            # Build candles from these trades
            min_time = min(t[1] for t in trade_records)
            max_time = max(t[1] for t in trade_records)

            await conn.execute(
                """
                INSERT INTO candles (symbol, timestamp, interval_minutes, open, high, low, close, volume, trade_count, vwap)
                SELECT
                    symbol,
                    time_bucket('1 minute', timestamp) AS timestamp,
                    1 AS interval_minutes,
                    first(price, timestamp) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price, timestamp) AS close,
                    sum(volume) AS volume,
                    count(*) AS trade_count,
                    sum(price * volume) / nullif(sum(volume), 0) AS vwap
                FROM trades
                WHERE symbol = $1
                  AND timestamp >= $2
                  AND timestamp <= $3
                GROUP BY symbol, time_bucket('1 minute', timestamp)
                ON CONFLICT (timestamp, symbol, interval_minutes) DO UPDATE SET
                    high = GREATEST(candles.high, EXCLUDED.high),
                    low = LEAST(candles.low, EXCLUDED.low),
                    close = EXCLUDED.close,
                    volume = candles.volume + EXCLUDED.volume,
                    trade_count = candles.trade_count + EXCLUDED.trade_count
                """,
                symbol, min_time, max_time
            )

    async def update_sync_status(self, symbol: str, newest_time: datetime):
        """Update sync status after filling."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO data_sync_status (symbol, data_type, newest_timestamp, last_sync_at)
                VALUES ($1, 'candles_1m', $2, NOW())
                ON CONFLICT (symbol, data_type) DO UPDATE SET
                    newest_timestamp = GREATEST(data_sync_status.newest_timestamp, EXCLUDED.newest_timestamp),
                    last_sync_at = NOW()
                """,
                symbol, newest_time
            )

    async def refresh_continuous_aggregates(self):
        """Refresh all continuous aggregates after gap fill."""
        aggregates = [
            'candles_5m', 'candles_15m', 'candles_30m',
            'candles_1h', 'candles_4h', 'candles_12h',
            'candles_1d', 'candles_1w'
        ]

        async with self.pool.acquire() as conn:
            for agg in aggregates:
                try:
                    await conn.execute(f"CALL refresh_continuous_aggregate('{agg}', NULL, NULL)")
                    logger.info(f"Refreshed continuous aggregate: {agg}")
                except Exception as e:
                    logger.warning(f"Failed to refresh {agg}: {e}")

    async def fill_all_gaps(self) -> dict:
        """
        Main entry point: detect and fill all gaps.

        Returns:
            Summary of gap filling results
        """
        results = {
            'gaps_detected': 0,
            'gaps_filled': 0,
            'candles_inserted': 0,
            'trades_processed': 0,
            'errors': []
        }

        gaps = await self.detect_gaps()
        results['gaps_detected'] = len(gaps)

        if not gaps:
            logger.info("No gaps detected - data is up to date")
            return results

        logger.info(f"Detected {len(gaps)} gaps to fill")

        # Fill gaps in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(3)

        async def fill_with_limit(gap: DataGap):
            async with semaphore:
                try:
                    if gap.is_small:
                        count = await self.fill_gap_ohlc(gap)
                        results['candles_inserted'] += count
                    else:
                        count = await self.fill_gap_trades(gap)
                        results['trades_processed'] += count

                    await self.update_sync_status(gap.symbol, gap.end_time)
                    results['gaps_filled'] += 1

                except Exception as e:
                    error_msg = f"Failed to fill gap for {gap.symbol}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)

        await asyncio.gather(*[fill_with_limit(gap) for gap in gaps])

        # Refresh continuous aggregates
        await self.refresh_continuous_aggregates()

        logger.info(
            f"Gap fill complete: {results['gaps_filled']}/{results['gaps_detected']} gaps filled, "
            f"{results['candles_inserted']} candles, {results['trades_processed']} trades"
        )

        return results


async def run_gap_filler(db_url: str) -> dict:
    """
    Convenience function to run gap filler.

    Usage:
        results = await run_gap_filler(db_url)
    """
    filler = GapFiller(db_url)

    try:
        await filler.start()
        return await filler.fill_all_gaps()
    finally:
        await filler.stop()
```

---

## Integration with ws_paper_tester

### Startup Sequence

```python
"""
main_with_historical.py - Extended main.py with historical data support
"""
import asyncio
import os
import logging
from datetime import datetime

from ws_paper_tester.core.tester import WsPaperTester
from ws_paper_tester.data.historical_provider import HistoricalDataProvider
from ws_paper_tester.data.gap_filler import run_gap_filler
from ws_paper_tester.data.websocket_db_writer import DatabaseWriter, integrate_db_writer

logger = logging.getLogger(__name__)


async def main():
    """
    Main entry point with historical data integration.

    Startup sequence:
    1. Initialize database connection
    2. Run gap filler to sync historical data
    3. Start WebSocket connection with DB writer
    4. Run paper tester with historical data support
    """
    db_url = os.getenv(
        'DATABASE_URL',
        'postgresql://trading:password@localhost:5432/kraken_data'
    )

    # =========================================
    # PHASE 1: Gap Filler (Startup Sync)
    # =========================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Running gap filler...")
    logger.info("=" * 60)

    gap_results = await run_gap_filler(db_url)

    if gap_results['errors']:
        logger.warning(f"Gap filler completed with errors: {gap_results['errors']}")
    else:
        logger.info(f"Gap filler complete: {gap_results}")

    # =========================================
    # PHASE 2: Initialize Database Writer
    # =========================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Starting database writer...")
    logger.info("=" * 60)

    db_writer = DatabaseWriter(db_url)
    await db_writer.start()

    # =========================================
    # PHASE 3: Initialize Historical Provider
    # =========================================
    historical_provider = HistoricalDataProvider(db_url)
    await historical_provider.connect()

    # =========================================
    # PHASE 4: Start Paper Tester
    # =========================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Starting ws_paper_tester...")
    logger.info("=" * 60)

    try:
        tester = WsPaperTester(
            config_path='config.yaml',
            historical_provider=historical_provider  # New parameter
        )

        # Integrate DB writer with WebSocket client
        integration = integrate_db_writer(tester.ws_client, db_writer)

        # Run tester
        await tester.run()

    finally:
        # Cleanup
        await integration.flush_current_candles()
        await db_writer.stop()
        await historical_provider.close()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
```

### Historical Data Provider

```python
"""
historical_provider.py - Query historical data for backtesting and strategy warmup
"""
import asyncio
import asyncpg
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Optional, AsyncIterator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candle:
    """Candle data structure compatible with ws_paper_tester."""
    symbol: str
    timestamp: datetime
    interval_minutes: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trade_count: int
    vwap: Optional[Decimal]

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> 'Candle':
        return cls(
            symbol=row['symbol'],
            timestamp=row['timestamp'],
            interval_minutes=row['interval_minutes'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            trade_count=row['trade_count'] or 0,
            vwap=row['vwap']
        )


class HistoricalDataProvider:
    """
    Provides historical candle data for:
    - Strategy warmup (loading indicator history on startup)
    - Backtesting (replaying historical data through strategies)
    - Analysis (querying historical patterns)
    """

    # Mapping of interval minutes to continuous aggregate views
    INTERVAL_VIEWS = {
        1: 'candles',
        5: 'candles_5m',
        15: 'candles_15m',
        30: 'candles_30m',
        60: 'candles_1h',
        240: 'candles_4h',
        720: 'candles_12h',
        1440: 'candles_1d',
        10080: 'candles_1w',
    }

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool: asyncpg.Pool = None

    async def connect(self):
        """Establish database connection."""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=10
        )
        logger.info("HistoricalDataProvider connected")

    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()

    def _get_view_for_interval(self, interval_minutes: int) -> str:
        """Get the appropriate view/table for the interval."""
        if interval_minutes in self.INTERVAL_VIEWS:
            return self.INTERVAL_VIEWS[interval_minutes]

        # For non-standard intervals, use base candles table
        return 'candles'

    async def get_candles(
        self,
        symbol: str,
        interval_minutes: int,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """
        Query historical candles.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            interval_minutes: Candle interval (1, 5, 15, 60, etc.)
            start: Start time (inclusive)
            end: End time (exclusive)
            limit: Maximum number of candles to return

        Returns:
            List of Candle objects, sorted by timestamp ascending
        """
        view = self._get_view_for_interval(interval_minutes)

        query = f"""
            SELECT symbol, timestamp, interval_minutes,
                   open, high, low, close, volume, trade_count, vwap
            FROM {view}
            WHERE symbol = $1
              AND timestamp >= $2
              AND timestamp < $3
        """

        if view == 'candles':
            query += " AND interval_minutes = $4"
            params = [symbol, start, end, interval_minutes]
        else:
            params = [symbol, start, end]

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [Candle.from_row(row) for row in rows]

    async def get_latest_candles(
        self,
        symbol: str,
        interval_minutes: int,
        count: int
    ) -> List[Candle]:
        """
        Get the N most recent candles.

        Args:
            symbol: Trading pair
            interval_minutes: Candle interval
            count: Number of candles to retrieve

        Returns:
            List of Candle objects, sorted by timestamp ascending
        """
        view = self._get_view_for_interval(interval_minutes)

        if view == 'candles':
            query = f"""
                SELECT symbol, timestamp, interval_minutes,
                       open, high, low, close, volume, trade_count, vwap
                FROM {view}
                WHERE symbol = $1 AND interval_minutes = $2
                ORDER BY timestamp DESC
                LIMIT $3
            """
            params = [symbol, interval_minutes, count]
        else:
            query = f"""
                SELECT symbol, timestamp, {interval_minutes} as interval_minutes,
                       open, high, low, close, volume, trade_count, vwap
                FROM {view}
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """
            params = [symbol, count]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        # Reverse to ascending order
        return [Candle.from_row(row) for row in reversed(rows)]

    async def replay_candles(
        self,
        symbol: str,
        interval_minutes: int,
        start: datetime,
        end: datetime,
        speed: float = 1.0
    ) -> AsyncIterator[Candle]:
        """
        Replay historical candles as a stream for backtesting.

        Args:
            symbol: Trading pair
            interval_minutes: Candle interval
            start: Replay start time
            end: Replay end time
            speed: Replay speed multiplier (1.0 = real-time, 0 = instant)

        Yields:
            Candle objects in chronological order
        """
        candles = await self.get_candles(symbol, interval_minutes, start, end)

        for i, candle in enumerate(candles):
            yield candle

            # Simulate time delay between candles
            if speed > 0 and i < len(candles) - 1:
                next_candle = candles[i + 1]
                time_diff = (next_candle.timestamp - candle.timestamp).total_seconds()
                await asyncio.sleep(time_diff / speed)

    async def get_warmup_data(
        self,
        symbol: str,
        interval_minutes: int,
        warmup_periods: int
    ) -> List[Candle]:
        """
        Get historical data for strategy warmup.

        This provides enough historical candles to initialize indicators
        (e.g., 200 candles for a 200-period moving average).

        Args:
            symbol: Trading pair
            interval_minutes: Candle interval
            warmup_periods: Number of periods needed for warmup

        Returns:
            List of Candle objects
        """
        return await self.get_latest_candles(symbol, interval_minutes, warmup_periods)

    async def get_data_range(self, symbol: str) -> dict:
        """
        Get the available data range for a symbol.

        Returns:
            Dict with 'oldest' and 'newest' timestamps
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest,
                    COUNT(*) as total_candles
                FROM candles
                WHERE symbol = $1 AND interval_minutes = 1
                """,
                symbol
            )

        return {
            'oldest': row['oldest'],
            'newest': row['newest'],
            'total_candles': row['total_candles']
        }

    async def get_multi_timeframe_candles(
        self,
        symbol: str,
        end_time: datetime,
        intervals: List[int] = [1, 5, 15, 60, 240]
    ) -> dict[int, List[Candle]]:
        """
        Get candles for multiple timeframes at once.

        Useful for MTF analysis where you need aligned data across timeframes.

        Args:
            symbol: Trading pair
            end_time: End time for all timeframes
            intervals: List of intervals to fetch

        Returns:
            Dict mapping interval -> list of candles
        """
        result = {}

        # Determine lookback for each interval (100 candles)
        tasks = []
        for interval in intervals:
            lookback = timedelta(minutes=interval * 100)
            start_time = end_time - lookback
            tasks.append(self.get_candles(symbol, interval, start_time, end_time))

        candle_lists = await asyncio.gather(*tasks)

        for interval, candles in zip(intervals, candle_lists):
            result[interval] = candles

        return result
```

---

## Docker Deployment

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # =============================================
  # TimescaleDB (PostgreSQL with time-series)
  # =============================================
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    container_name: kraken_timescaledb
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: trading
      POSTGRES_PASSWORD: ${DB_PASSWORD:-changeme}
      POSTGRES_DB: kraken_data
      TIMESCALEDB_TELEMETRY: "off"
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/01-init.sql:ro
    command: >
      postgres
        -c shared_preload_libraries=timescaledb
        -c timescaledb.max_background_workers=8
        -c max_parallel_workers_per_gather=4
        -c max_worker_processes=16
        -c shared_buffers=2GB
        -c effective_cache_size=6GB
        -c maintenance_work_mem=512MB
        -c work_mem=64MB
        -c wal_buffers=64MB
        -c checkpoint_completion_target=0.9
        -c random_page_cost=1.1
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading -d kraken_data"]
      interval: 10s
      timeout: 5s
      retries: 5

  # =============================================
  # PgAdmin (Optional - Database Management UI)
  # =============================================
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: kraken_pgadmin
    restart: unless-stopped
    ports:
      - "5050:80"
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@local.dev
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD:-admin}
      PGADMIN_CONFIG_SERVER_MODE: "False"
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    depends_on:
      timescaledb:
        condition: service_healthy

  # =============================================
  # Data Ingester Service (Optional)
  # =============================================
  ingester:
    build:
      context: .
      dockerfile: Dockerfile.ingester
    container_name: kraken_ingester
    restart: unless-stopped
    environment:
      DATABASE_URL: postgresql://trading:${DB_PASSWORD:-changeme}@timescaledb:5432/kraken_data
      KRAKEN_API_KEY: ${KRAKEN_API_KEY:-}
      KRAKEN_API_SECRET: ${KRAKEN_API_SECRET:-}
      SYMBOLS: "XRP/USDT,BTC/USDT,XRP/BTC"
    depends_on:
      timescaledb:
        condition: service_healthy
    profiles:
      - ingester

volumes:
  timescaledb_data:
    driver: local
  pgadmin_data:
    driver: local

networks:
  default:
    name: kraken_network
```

### Environment File

```bash
# .env
DB_PASSWORD=your_secure_password_here
PGADMIN_PASSWORD=admin_password_here
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_API_SECRET=your_kraken_api_secret
```

### Database Initialization Script

```sql
-- scripts/init-db.sql
-- This runs automatically on first container startup

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create tables (abbreviated - full schema in Schema Design section)
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(20, 10) NOT NULL,
    volume DECIMAL(20, 10) NOT NULL,
    side VARCHAR(4) NOT NULL,
    order_type VARCHAR(10),
    misc VARCHAR(50),
    PRIMARY KEY (timestamp, symbol, id)
);

SELECT create_hypertable('trades', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS candles (
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
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS data_sync_status (
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(20) NOT NULL,
    oldest_timestamp TIMESTAMPTZ,
    newest_timestamp TIMESTAMPTZ,
    last_sync_at TIMESTAMPTZ DEFAULT NOW(),
    last_kraken_since BIGINT,
    total_records BIGINT DEFAULT 0,
    PRIMARY KEY (symbol, data_type)
);

CREATE TABLE IF NOT EXISTS external_indicators (
    timestamp TIMESTAMPTZ NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    value DECIMAL(20, 10) NOT NULL,
    source VARCHAR(50),
    PRIMARY KEY (timestamp, indicator_name)
);

SELECT create_hypertable('external_indicators', 'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_ts
    ON candles (symbol, interval_minutes, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts
    ON trades (symbol, timestamp DESC);

-- Note: Continuous aggregates should be created after initial data load
-- Run the continuous aggregate creation scripts separately

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading;
```

### Startup Commands

```bash
# Start TimescaleDB
docker-compose up -d timescaledb

# Wait for healthy status
docker-compose ps

# Optional: Start PgAdmin for database management
docker-compose up -d pgadmin

# Connect to database
docker exec -it kraken_timescaledb psql -U trading -d kraken_data

# Run schema migrations (after first startup)
docker exec -i kraken_timescaledb psql -U trading -d kraken_data < scripts/continuous-aggregates.sql
```

---

## Data Retention and Maintenance

### Retention Policies

```sql
-- Automatic data retention policies

-- Keep raw trades for 90 days (can rebuild candles from them)
SELECT add_retention_policy('trades', INTERVAL '90 days');

-- Keep 1-minute candles for 1 year
-- (Older data available via continuous aggregates)
SELECT add_retention_policy('candles', INTERVAL '365 days');

-- Continuous aggregates retained longer:
-- - 5m, 15m, 30m: 2 years
-- - 1h, 4h: 5 years
-- - 12h, 1d, 1w: Forever (no retention policy)
```

### Compression Schedule

```sql
-- Compression runs automatically via policies set in schema
-- Manual compression for testing:

-- Compress chunks older than 7 days
SELECT compress_chunk(c)
FROM show_chunks('trades', older_than => INTERVAL '7 days') c;

SELECT compress_chunk(c)
FROM show_chunks('candles', older_than => INTERVAL '30 days') c;

-- View compression stats
SELECT
    hypertable_name,
    chunk_name,
    before_compression_table_bytes,
    after_compression_table_bytes,
    (1 - after_compression_table_bytes::float / before_compression_table_bytes) * 100 as compression_ratio
FROM chunk_compression_stats('candles')
ORDER BY chunk_name DESC
LIMIT 10;
```

### Backup Strategy

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backups/kraken_data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/kraken_data_${TIMESTAMP}.sql.gz"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Dump database with compression
docker exec kraken_timescaledb pg_dump \
    -U trading \
    -d kraken_data \
    --no-owner \
    --no-privileges \
    | gzip > ${BACKUP_FILE}

# Keep only last 7 days of backups
find ${BACKUP_DIR} -name "*.sql.gz" -mtime +7 -delete

echo "Backup created: ${BACKUP_FILE}"
```

---

## API Reference

### HistoricalDataProvider Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `get_candles()` | Query candles in time range | symbol, interval, start, end, limit |
| `get_latest_candles()` | Get N most recent candles | symbol, interval, count |
| `replay_candles()` | Stream candles for backtesting | symbol, interval, start, end, speed |
| `get_warmup_data()` | Get indicator warmup data | symbol, interval, periods |
| `get_data_range()` | Get available data range | symbol |
| `get_multi_timeframe_candles()` | Get MTF aligned data | symbol, end_time, intervals |

### DatabaseWriter Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `start()` | Initialize writer | - |
| `stop()` | Flush and close | - |
| `write_trade()` | Buffer trade for insert | TradeRecord |
| `write_candle()` | Buffer candle for upsert | CandleRecord |
| `update_sync_status()` | Update sync timestamp | symbol, data_type, timestamp |

### GapFiller Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `detect_gaps()` | Find all data gaps | - |
| `fill_gap_ohlc()` | Fill small gap via OHLC API | DataGap |
| `fill_gap_trades()` | Fill large gap via Trades API | DataGap |
| `fill_all_gaps()` | Main entry: detect and fill all | - |
| `refresh_continuous_aggregates()` | Refresh all aggregates | - |

---

## Appendix: Quick Start Commands

```bash
# 1. Start database
docker-compose up -d timescaledb
docker-compose logs -f timescaledb  # Wait for "database system is ready"

# 2. Apply schema
docker exec -i kraken_timescaledb psql -U trading -d kraken_data < scripts/schema.sql
docker exec -i kraken_timescaledb psql -U trading -d kraken_data < scripts/continuous-aggregates.sql

# 3. Download Kraken CSV files
mkdir -p data/kraken_csv
# Download from: https://support.kraken.com/hc/en-us/articles/360047124832
# Place XRPUSDT_1.csv, BTCUSDT_1.csv, etc. in data/kraken_csv/

# 4. Run bulk import
python -m ws_paper_tester.data.bulk_csv_importer

# 5. Run historical backfill (optional - for complete trade history)
python -m ws_paper_tester.data.historical_backfill

# 6. Start ws_paper_tester with historical data
python -m ws_paper_tester.main_with_historical
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-15 | Initial design document |

---

*Document generated for grok-4_1 trading bot project*
