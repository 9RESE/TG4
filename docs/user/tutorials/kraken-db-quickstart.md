# Tutorial: Getting Started with Kraken Historical Data System

This tutorial walks you through setting up and using the Kraken Historical Data System for your first time.

## What You'll Learn

- How to start the TimescaleDB database
- How to import historical data
- How to query data for backtesting

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ with pip
- Basic familiarity with command line

## Step 1: Set Up the Database

### 1.1 Configure Environment

Navigate to the kraken_db directory and create your environment file:

```bash
cd data/kraken_db
cp .env.example .env
```

Edit `.env` and set a secure password:

```bash
# Generate a secure password
openssl rand -base64 32

# Edit .env file
DB_PASSWORD=your_generated_password_here
DATABASE_URL=postgresql://trading:${DB_PASSWORD}@localhost:5433/kraken_data
```

### 1.2 Start TimescaleDB

```bash
docker compose up -d timescaledb
```

Wait for the database to be ready:

```bash
docker compose logs -f timescaledb
# Look for: "database system is ready to accept connections"
```

### 1.3 Verify Database Setup

```bash
docker exec -it kraken_timescaledb psql -U trading -d kraken_data -c "\dt"
```

You should see tables: `trades`, `candles`, `external_indicators`, `data_sync_status`, `backtest_runs`.

## Step 2: Install Python Dependencies

```bash
pip install asyncpg aiohttp pandas
```

## Step 3: Import Historical Data

You have three options for importing data:

### Option A: Backfill from Kraken API (Recommended)

This fetches complete trade history from Kraken's public API:

```bash
# Set the database URL
export DATABASE_URL="postgresql://trading:your_password@localhost:5433/kraken_data"

# Run backfill for XRP/USDT
python -m data.kraken_db.historical_backfill --symbols XRP/USDT
```

**Note**: This can take several hours for complete history. The script will automatically stop when it catches up to real-time.

### Option B: Import from CSV Files

If you have Kraken CSV files (download from [Kraken Support](https://support.kraken.com/hc/en-us/articles/360047124832)):

```bash
# Place CSV files in data/kraken_csv/
# Format: XRPUSDT_1.csv, BTCUSDT_1.csv, etc.

python -m data.kraken_db.bulk_csv_importer --dir ./data/kraken_csv
```

### Option C: Resume Interrupted Backfill

If a backfill was interrupted, resume from where it stopped:

```bash
python -m data.kraken_db.historical_backfill --symbols XRP/USDT --resume
```

## Step 4: Create Continuous Aggregates

After importing data, create the higher timeframe aggregates:

```bash
docker exec -it kraken_timescaledb psql -U trading -d kraken_data \
  -f /docker-entrypoint-initdb.d/continuous-aggregates.sql
```

Or manually refresh them:

```bash
docker exec -it kraken_timescaledb psql -U trading -d kraken_data << 'EOF'
CALL refresh_continuous_aggregate('candles_5m', NULL, NULL);
CALL refresh_continuous_aggregate('candles_15m', NULL, NULL);
CALL refresh_continuous_aggregate('candles_1h', NULL, NULL);
CALL refresh_continuous_aggregate('candles_4h', NULL, NULL);
CALL refresh_continuous_aggregate('candles_1d', NULL, NULL);
EOF
```

## Step 5: Query Historical Data

### 5.1 Basic Python Usage

```python
import asyncio
import os
from datetime import datetime, timezone, timedelta
from data.kraken_db import HistoricalDataProvider

async def main():
    db_url = os.getenv('DATABASE_URL')
    provider = HistoricalDataProvider(db_url)

    try:
        await provider.connect()

        # Health check
        health = await provider.health_check()
        print(f"Connected: {health['connected']}")
        print(f"Symbols: {health['symbols']}")
        print(f"Total candles: {health['total_candles']:,}")

        # Get warmup data (last 200 5-minute candles)
        candles = await provider.get_warmup_data('XRP/USDT', 5, 200)
        print(f"\nLoaded {len(candles)} candles for warmup")

        if candles:
            print(f"First: {candles[0].timestamp}")
            print(f"Last: {candles[-1].timestamp}")
            print(f"Last close: {candles[-1].close}")

    finally:
        await provider.close()

asyncio.run(main())
```

### 5.2 Get Specific Time Range

```python
from datetime import datetime, timezone, timedelta

# Last 24 hours of 1-minute candles
end = datetime.now(timezone.utc)
start = end - timedelta(hours=24)

candles = await provider.get_candles('XRP/USDT', 1, start, end)
```

### 5.3 Multi-Timeframe Data

```python
# Get aligned data across multiple timeframes
end = datetime.now(timezone.utc)
mtf_data = await provider.get_multi_timeframe_candles(
    'XRP/USDT',
    end_time=end,
    intervals=[1, 5, 15, 60, 240],  # 1m, 5m, 15m, 1h, 4h
    lookback_candles=100
)

for interval, candles in mtf_data.items():
    print(f"{interval}m: {len(candles)} candles")
```

## Step 6: Set Up Real-time Data Collection

Integrate with your WebSocket client to persist live data:

```python
from data.kraken_db import DatabaseWriter, integrate_db_writer

async def run_with_db():
    db_writer = DatabaseWriter(os.getenv('DATABASE_URL'))
    await db_writer.start()

    # Assuming you have a KrakenWSClient
    ws_client = KrakenWSClient(...)
    integration = integrate_db_writer(ws_client, db_writer)

    try:
        await ws_client.connect()
        # Run your trading logic...
        await asyncio.sleep(3600)  # Run for 1 hour
    finally:
        await integration.flush_current_candles()
        await db_writer.stop()
```

## Step 7: Run Gap Filler on Startup

Fill any gaps that occurred while your system was offline:

```python
from data.kraken_db import run_gap_filler

results = await run_gap_filler(os.getenv('DATABASE_URL'))
print(f"Gaps detected: {results['gaps_detected']}")
print(f"Gaps filled: {results['gaps_filled']}")
```

## Verification

Check your data is working correctly:

```bash
# Count candles per symbol
docker exec -it kraken_timescaledb psql -U trading -d kraken_data << 'EOF'
SELECT symbol, interval_minutes, COUNT(*) as count
FROM candles
GROUP BY symbol, interval_minutes
ORDER BY symbol, interval_minutes;
EOF

# Check sync status
docker exec -it kraken_timescaledb psql -U trading -d kraken_data << 'EOF'
SELECT symbol, data_type, oldest_timestamp, newest_timestamp, total_records
FROM data_sync_status
ORDER BY symbol, data_type;
EOF
```

## Next Steps

- Read the [How-To Guides](../how-to/kraken-db-operations.md) for specific operations
- Check the [API Reference](../reference/kraken-db-api.md) for detailed documentation
- Learn about [Database Architecture](../../architecture/05-building-blocks/kraken-db.md)

## Troubleshooting

### Database Connection Refused

Ensure Docker is running and the container is healthy:
```bash
docker compose ps
docker compose logs timescaledb
```

### Rate Limiting During Backfill

The backfill script handles rate limiting automatically. If you see many retries, increase the delay:
```bash
python -m data.kraken_db.historical_backfill --symbols XRP/USDT --rate-limit 2.0
```

### Missing asyncpg Module

Install the required dependency:
```bash
pip install asyncpg
```
