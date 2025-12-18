# How-To: Kraken Database Operations

Practical guides for common Kraken Historical Data System operations.

## Data Import Operations

### How to Backfill Historical Trades

Fetch complete trade history from Kraken API:

```bash
export DATABASE_URL="postgresql://trading:password@localhost:5433/kraken_data"

# Backfill single symbol
python -m data.kraken_db.historical_backfill --symbols XRP/USDT

# Backfill multiple symbols
python -m data.kraken_db.historical_backfill --symbols XRP/USDT BTC/USDT ETH/USDT

# Resume interrupted backfill
python -m data.kraken_db.historical_backfill --symbols XRP/USDT --resume

# Skip candle building (just import trades)
python -m data.kraken_db.historical_backfill --symbols XRP/USDT --no-candles

# Continuous polling (don't auto-stop at real-time)
python -m data.kraken_db.historical_backfill --symbols XRP/USDT --no-auto-stop
```

### How to Import CSV Files

Import Kraken historical CSV files:

```bash
# Import only 1-minute candles (recommended - others computed automatically)
python -m data.kraken_db.bulk_csv_importer --dir ./data/kraken_csv

# Import all intervals
python -m data.kraken_db.bulk_csv_importer --dir ./data/kraken_csv --all

# Custom batch size
python -m data.kraken_db.bulk_csv_importer --dir ./data/kraken_csv --batch-size 5000
```

**Expected CSV format**: `XRPUSDT_1.csv` (symbol_interval.csv)
**CSV columns**: timestamp, open, high, low, close, volume, trades

### How to Fill Data Gaps

Detect and fill gaps in historical data:

```bash
# Fill gaps for default symbols (XRP/USDT, BTC/USDT, XRP/BTC)
python -m data.kraken_db.gap_filler

# Fill gaps for specific symbols
python -m data.kraken_db.gap_filler --symbols XRP/USDT ETH/USDT
```

**Programmatic usage**:

```python
from data.kraken_db import run_gap_filler

results = await run_gap_filler(
    db_url="postgresql://trading:password@localhost:5433/kraken_data",
    symbols=['XRP/USDT', 'BTC/USDT']
)

print(f"Gaps detected: {results['gaps_detected']}")
print(f"Gaps filled: {results['gaps_filled']}")
print(f"Candles inserted: {results['candles_inserted']}")
print(f"Trades processed: {results['trades_processed']}")
```

## Data Query Operations

### How to Get Candles for a Time Range

```python
from datetime import datetime, timezone, timedelta
from data.kraken_db import HistoricalDataProvider

provider = HistoricalDataProvider(db_url)
await provider.connect()

# Get 1-hour candles for the last 7 days
end = datetime.now(timezone.utc)
start = end - timedelta(days=7)
candles = await provider.get_candles('XRP/USDT', 60, start, end)

# Get with limit
candles = await provider.get_candles('XRP/USDT', 60, start, end, limit=100)
```

### How to Get Latest N Candles

```python
# Get last 200 5-minute candles
candles = await provider.get_latest_candles('XRP/USDT', 5, 200)

# First candle is oldest, last is newest
oldest = candles[0]
newest = candles[-1]
```

### How to Get Warmup Data for Indicators

```python
# Get enough historical data to initialize a 200-period moving average
warmup_data = await provider.get_warmup_data('XRP/USDT', 5, 200)

# Use with your indicator
for candle in warmup_data:
    indicator.update(candle.close)
```

### How to Get Multi-Timeframe Data

```python
# Get aligned data across timeframes
mtf_data = await provider.get_multi_timeframe_candles(
    symbol='XRP/USDT',
    end_time=datetime.now(timezone.utc),
    intervals=[1, 5, 15, 60, 240],
    lookback_candles=100
)

for interval, candles in mtf_data.items():
    print(f"{interval}m: {len(candles)} candles, latest close: {candles[-1].close}")
```

### How to Replay Historical Candles

```python
# Replay candles in chronological order
async for candle in provider.replay_candles('XRP/USDT', 1, start, end, speed=0):
    await strategy.on_candle(candle)

# speed=0: instant replay
# speed=1.0: real-time speed
# speed=10.0: 10x faster than real-time
```

## Real-Time Data Operations

### How to Persist WebSocket Data

```python
from data.kraken_db import DatabaseWriter, integrate_db_writer

# Create writer with custom settings
db_writer = DatabaseWriter(
    db_url=os.getenv('DATABASE_URL'),
    trade_buffer_size=100,
    trade_flush_interval=5.0,
    candle_flush_interval=1.0
)
await db_writer.start()

# Integrate with your WebSocket client
integration = integrate_db_writer(ws_client, db_writer)

# Run your logic...
await some_trading_logic()

# Clean shutdown
await integration.flush_current_candles()
await db_writer.stop()
```

### How to Write Trades Directly

```python
from data.kraken_db import DatabaseWriter, TradeRecord
from datetime import datetime, timezone
from decimal import Decimal

trade = TradeRecord(
    symbol='XRP/USDT',
    timestamp=datetime.now(timezone.utc),
    price=Decimal('0.5234'),
    volume=Decimal('1000.5'),
    side='buy'
)

await db_writer.write_trade(trade)
```

### How to Write Candles Directly

```python
from data.kraken_db import CandleRecord
from decimal import Decimal

candle = CandleRecord(
    symbol='XRP/USDT',
    timestamp=datetime.now(timezone.utc),
    interval_minutes=1,
    open=Decimal('0.5200'),
    high=Decimal('0.5250'),
    low=Decimal('0.5190'),
    close=Decimal('0.5234'),
    volume=Decimal('50000.5'),
    trade_count=150,
    vwap=Decimal('0.5220')
)

await db_writer.write_candle(candle)
```

## Database Management Operations

### How to Check Database Health

```python
health = await provider.health_check()
print(f"Connected: {health['connected']}")
print(f"Symbols: {health['symbols']}")
print(f"Total candles: {health['total_candles']:,}")
print(f"Oldest data: {health['oldest_data']}")
print(f"Newest data: {health['newest_data']}")
```

### How to Get Data Range for a Symbol

```python
data_range = await provider.get_data_range('XRP/USDT')
print(f"Oldest: {data_range['oldest']}")
print(f"Newest: {data_range['newest']}")
print(f"Total candles: {data_range['total_candles']}")
```

### How to Get Available Symbols

```python
symbols = await provider.get_symbols()
print(f"Available symbols: {symbols}")
```

### How to Check Sync Status

```python
status = await provider.get_sync_status('XRP/USDT')
if status:
    print(f"Oldest: {status['oldest']}")
    print(f"Newest: {status['newest']}")
    print(f"Last sync: {status['last_sync']}")
    print(f"Total records: {status['total_records']}")
```

### How to Refresh Continuous Aggregates

```sql
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

### How to Check Compression Status

```sql
SELECT
    hypertable_name,
    compression_enabled,
    before_compression_total_bytes,
    after_compression_total_bytes,
    (1 - (after_compression_total_bytes::float / before_compression_total_bytes::float)) * 100 as compression_ratio
FROM timescaledb_information.compression_status
WHERE before_compression_total_bytes IS NOT NULL
ORDER BY hypertable_name;
```

## Docker Operations

### How to Start/Stop Database

```bash
# Start database
docker compose up -d timescaledb

# Stop database
docker compose down

# Stop and remove volumes (data loss!)
docker compose down -v
```

### How to Start with PgAdmin

```bash
docker compose --profile tools up -d
# Access PgAdmin at http://localhost:5050
# Email: admin@local.dev
# Password: (from PGADMIN_PASSWORD in .env)
```

### How to Connect via psql

```bash
docker exec -it kraken_timescaledb psql -U trading -d kraken_data
```

### How to Backup Database

```bash
docker exec -it kraken_timescaledb pg_dump -U trading kraken_data > backup.sql
```

### How to Restore Database

```bash
cat backup.sql | docker exec -i kraken_timescaledb psql -U trading kraken_data
```

## Performance Tuning

### How to Monitor Buffer Statistics

```python
stats = db_writer.get_stats()
print(f"Trades written: {stats['trades_written']}")
print(f"Candles written: {stats['candles_written']}")
print(f"Flush count: {stats['flush_count']}")
print(f"Error count: {stats['error_count']}")
print(f"Overflow count: {stats['overflow_count']}")
print(f"Trade buffer size: {stats['trade_buffer_size']}")
print(f"Candle buffer size: {stats['candle_buffer_size']}")
```

### How to Tune Buffer Sizes

```python
# For high-volume trading pairs
db_writer = DatabaseWriter(
    db_url,
    trade_buffer_size=500,       # Larger buffer
    trade_flush_interval=2.0,    # More frequent flushes
    pool_max_size=20             # More connections
)
```

### How to Manually Compress Old Data

```sql
-- Compress chunks older than 7 days
SELECT compress_chunk(chunk_schema || '.' || chunk_name)
FROM timescaledb_information.chunks
WHERE hypertable_name = 'trades'
  AND range_start < NOW() - INTERVAL '7 days'
  AND is_compressed = FALSE;
```
