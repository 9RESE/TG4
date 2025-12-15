# How To: Operate the Historical Data System

This guide covers the complete setup and operation of the ws_paper_tester Historical Data System.

---

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ with pip
- At least 4GB RAM (8GB recommended)
- 50GB+ disk space for historical data

## Quick Start

### 1. Start the Database

```bash
cd ws_paper_tester

# Copy environment template
cp .env.example .env

# Edit .env to set your passwords
nano .env

# Start TimescaleDB
docker-compose up -d timescaledb

# Wait for database to be ready (watch for "database system is ready")
docker-compose logs -f timescaledb
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Apply Database Schema

```bash
# The init-db.sql script runs automatically on first startup
# Verify it's working:
docker exec -it kraken_timescaledb psql -U trading -d kraken_data -c "\\dt"

# You should see: trades, candles, data_sync_status, external_indicators, backtest_runs
```

### 4. Create Continuous Aggregates

```bash
# Apply continuous aggregates for multi-timeframe data
docker exec -i kraken_timescaledb psql -U trading -d kraken_data < scripts/continuous-aggregates.sql
```

### 5. Run the Paper Tester

```bash
# Run with historical data support
python main_with_historical.py

# Or skip gap filling for faster startup
python main_with_historical.py --skip-gap-fill
```

---

## Importing Historical Data

### Option A: Download Kraken CSV Files (Recommended)

Kraken provides free historical OHLCV data:

1. Download CSV files from [Kraken Support](https://support.kraken.com/hc/en-us/articles/360047124832)
2. Place files in `data/kraken_csv/` directory
3. Run the importer:

```bash
python -m data.bulk_csv_importer --dir ./data/kraken_csv
```

### Option B: Fetch from Kraken API

Fetch complete trade history (slower, but comprehensive):

```bash
# Fetch all history for specific symbols
python -m data.historical_backfill --symbols XRP/USDT BTC/USDT

# Resume from last sync point
python -m data.historical_backfill --symbols XRP/USDT --resume
```

### Option C: Let Gap Filler Handle It

The gap filler runs automatically on startup and will:
- Detect missing data since last run
- Fetch using OHLC API for gaps < 12 hours
- Fetch using Trades API for larger gaps

---

## Database Operations

### Connect to Database

```bash
# Using psql directly
docker exec -it kraken_timescaledb psql -U trading -d kraken_data

# Or start PgAdmin (optional)
docker-compose --profile tools up -d pgadmin
# Access at http://localhost:5050 (admin@local.dev / admin)
```

### Check Data Status

```sql
-- View sync status
SELECT symbol, data_type, oldest_timestamp, newest_timestamp, total_records
FROM data_sync_status;

-- Count candles by symbol
SELECT symbol, interval_minutes, COUNT(*) as count
FROM candles
GROUP BY symbol, interval_minutes
ORDER BY symbol, interval_minutes;

-- Check compression stats
SELECT hypertable_name,
       compression_enabled,
       before_compression_table_bytes,
       after_compression_table_bytes
FROM timescaledb_information.hypertables;
```

### Manually Refresh Aggregates

```sql
-- Refresh a specific aggregate
CALL refresh_continuous_aggregate('candles_1h', NULL, NULL);

-- Refresh all aggregates
CALL refresh_continuous_aggregate('candles_5m', NULL, NULL);
CALL refresh_continuous_aggregate('candles_15m', NULL, NULL);
CALL refresh_continuous_aggregate('candles_30m', NULL, NULL);
CALL refresh_continuous_aggregate('candles_1h', NULL, NULL);
CALL refresh_continuous_aggregate('candles_4h', NULL, NULL);
CALL refresh_continuous_aggregate('candles_12h', NULL, NULL);
CALL refresh_continuous_aggregate('candles_1d', NULL, NULL);
CALL refresh_continuous_aggregate('candles_1w', NULL, NULL);
```

### Backup and Restore

```bash
# Backup
docker exec kraken_timescaledb pg_dump -U trading -d kraken_data | gzip > backup.sql.gz

# Restore
gunzip -c backup.sql.gz | docker exec -i kraken_timescaledb psql -U trading -d kraken_data
```

---

## Running the Paper Tester

### Standard Mode (with Historical Data)

```bash
python main_with_historical.py
```

This will:
1. Run gap filler to sync any missing data
2. Start database writer for real-time persistence
3. Initialize historical data provider
4. Start WebSocket paper trading

### Command Line Options

| Option | Description |
|--------|-------------|
| `--skip-gap-fill` | Skip gap filling on startup |
| `--duration 60` | Run for 60 minutes then stop |
| `--simulated` | Use simulated data instead of live |
| `--no-dashboard` | Disable web dashboard |
| `--symbols XRP/USDT BTC/USDT` | Trade specific symbols |
| `--db-url postgresql://...` | Custom database URL |
| `--log-level DEBUG` | Set logging level |

### Examples

```bash
# Run for 2 hours with specific symbols
python main_with_historical.py --duration 120 --symbols XRP/USDT BTC/USDT

# Quick test with simulated data (no database required)
python main_with_historical.py --simulated --skip-gap-fill

# Debug mode
python main_with_historical.py --log-level DEBUG
```

---

## Monitoring

### Check Database Health

```sql
-- View active connections
SELECT count(*) FROM pg_stat_activity;

-- Check hypertable sizes
SELECT hypertable_name,
       pg_size_pretty(total_bytes) AS total_size
FROM hypertable_detailed_size('candles');

-- View chunk information
SELECT chunk_schema, chunk_name,
       pg_size_pretty(total_bytes) AS size
FROM chunks_detailed_size('candles')
ORDER BY range_start DESC
LIMIT 10;
```

### View Logs

```bash
# Database logs
docker-compose logs -f timescaledb

# Paper tester logs
tail -f logs/session_*.log
```

---

## Troubleshooting

### Database Won't Start

```bash
# Check container status
docker-compose ps

# View full logs
docker-compose logs timescaledb

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d timescaledb
```

### Gap Filler Times Out

```bash
# Skip gap filling and run manually later
python main_with_historical.py --skip-gap-fill

# Run gap filler separately
python -m data.gap_filler --symbols XRP/USDT
```

### Out of Disk Space

```sql
-- Compress old chunks manually
SELECT compress_chunk(c)
FROM show_chunks('trades', older_than => INTERVAL '7 days') c;

-- Add retention policy (keeps last 90 days of trades)
SELECT add_retention_policy('trades', INTERVAL '90 days');
```

### High Memory Usage

Edit `docker-compose.yml` to reduce PostgreSQL memory settings:

```yaml
command: >
  postgres
    -c shared_buffers=512MB
    -c effective_cache_size=1GB
    -c work_mem=32MB
```

---

## Stopping the System

```bash
# Stop paper tester (Ctrl+C or wait for duration)

# Stop database
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

---

## Related Documentation

- [Feature Documentation](/docs/development/features/historical-data-system/historical-data-system-v1.0.md)
- [Configuration Guide](/docs/user/how-to/configure-paper-tester.md)
