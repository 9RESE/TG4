# Getting Started Guide

Complete setup and operation guide for the TG4 WebSocket Paper Trading Platform.

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.10+**: `python --version`
- **Docker & Docker Compose**: `docker --version && docker compose version`
- **Git**: `git --version`
- **4GB+ RAM** (8GB recommended for database)
- **50GB+ disk space** for historical data

---

## Quick Start (5 Minutes)

### Option A: Simple Mode (No Database)

Test strategies with simulated data - no database required:

```bash
# Clone and setup
git clone https://github.com/9RESE/TG4.git
cd TG4/ws_paper_tester

# Install dependencies
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

pip install -r requirements.txt

# Run with simulated data
python ws_tester.py --simulated
```

Open http://localhost:8787 to view the dashboard.

### Option B: Full Setup (With Database)

For persistent historical data and backtesting:

```bash
cd ws_paper_tester

# 1. Configure environment
cp .env.example .env
nano .env  # Set DB_PASSWORD (generate with: openssl rand -base64 32)

# 2. Start TimescaleDB
docker compose up -d timescaledb

# 3. Wait for healthy status
docker compose ps  # Should show "healthy"

# 4. Apply continuous aggregates (after first data)
docker exec -i kraken_timescaledb psql -U trading -d kraken_data < scripts/continuous-aggregates.sql

# 5. Run with historical data
python main_with_historical.py
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/9RESE/TG4.git
cd TG4
```

### 2. Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate      # Linux/macOS
# OR
venv\Scripts\activate         # Windows PowerShell
# OR
source venv/Scripts/activate  # Windows Git Bash

# Install dependencies
pip install -r ws_paper_tester/requirements.txt
```

### 3. Database Setup (Optional)

For historical data persistence:

```bash
cd ws_paper_tester

# Copy environment template
cp .env.example .env

# Edit and set your password
nano .env
```

Required `.env` values:

```bash
# Generate secure password
DB_PASSWORD=$(openssl rand -base64 32)

# Full connection URL
DATABASE_URL=postgresql://trading:${DB_PASSWORD}@localhost:5433/kraken_data
```

Start TimescaleDB:

```bash
# Start database container
docker compose up -d timescaledb

# Verify it's running and healthy
docker compose ps

# View logs if needed
docker compose logs -f timescaledb
```

---

## Running the Paper Tester

### Basic Commands

```bash
cd ws_paper_tester

# Simulated data (no API, no database)
python ws_tester.py --simulated

# Live Kraken data (no database)
python ws_tester.py

# With historical data support
python main_with_historical.py

# Skip gap filling for faster startup
python main_with_historical.py --skip-gap-fill

# Run for specific duration (minutes)
python main_with_historical.py --duration 60

# Specific symbols only
python main_with_historical.py --symbols XRP/USDT BTC/USDT

# Debug mode
python main_with_historical.py --log-level DEBUG
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--simulated` | Use simulated data | Live data |
| `--duration N` | Run for N minutes | Unlimited |
| `--symbols S1 S2` | Trade specific symbols | All configured |
| `--skip-gap-fill` | Skip historical gap filling | Fill gaps |
| `--no-dashboard` | Disable web dashboard | Enabled |
| `--log-level LEVEL` | Set log level (DEBUG/INFO/WARNING) | INFO |
| `--db-url URL` | Custom database URL | From .env |

---

## Database Operations

### Starting Services

```bash
cd ws_paper_tester

# TimescaleDB only
docker compose up -d timescaledb

# With PgAdmin web UI (http://localhost:5050)
docker compose --profile tools up -d
```

### Connecting to Database

```bash
# Direct psql connection
docker exec -it kraken_timescaledb psql -U trading -d kraken_data

# Or via connection string
psql "postgresql://trading:YOUR_PASSWORD@localhost:5433/kraken_data"
```

### Common SQL Commands

```sql
-- View sync status
SELECT symbol, data_type, oldest_timestamp, newest_timestamp, total_records
FROM data_sync_status;

-- Count candles by symbol and interval
SELECT symbol, interval_minutes, COUNT(*) as count
FROM candles
GROUP BY symbol, interval_minutes
ORDER BY symbol, interval_minutes;

-- Check database size
SELECT pg_size_pretty(pg_database_size('kraken_data'));

-- View compression stats
SELECT hypertable_name,
       pg_size_pretty(before_compression_table_bytes) as before,
       pg_size_pretty(after_compression_table_bytes) as after
FROM timescaledb_information.compression_settings;
```

### Stopping Services

```bash
# Stop containers (preserve data)
docker compose down

# Stop and remove data (WARNING: deletes all data)
docker compose down -v
```

---

## Importing Historical Data

### Option 1: Kraken CSV Files (Fastest)

Download free historical data from [Kraken](https://support.kraken.com/hc/en-us/articles/360047124832):

```bash
# Place CSV files in data/kraken_csv/
mkdir -p data/kraken_csv
# Download and extract Kraken OHLC files here

# Run importer
python -m data.bulk_csv_importer --dir ./data/kraken_csv
```

### Option 2: REST API Backfill

Fetch complete trade history from Kraken API:

```bash
# Fetch all available history
python -m data.historical_backfill --symbols XRP/USDT BTC/USDT

# Resume interrupted backfill
python -m data.historical_backfill --symbols XRP/USDT --resume
```

### Option 3: Automatic Gap Filling

The gap filler runs automatically on startup with `main_with_historical.py`:

- Small gaps (< 12 hours): Uses OHLC REST API (fast)
- Large gaps (>= 12 hours): Uses Trades REST API (complete)

Run manually:

```bash
python -m data.gap_filler --symbols XRP/USDT BTC/USDT
```

---

## Dashboard

Access the real-time dashboard at **http://localhost:8787**

### Features

- Live portfolio values per strategy
- Trade execution log
- Strategy signals and rejections
- Real-time price updates

### Configuration

Edit `config.yaml`:

```yaml
dashboard:
  enabled: true
  host: 127.0.0.1
  port: 8787
```

---

## Configuration

### Main Configuration File

`ws_paper_tester/config.yaml`:

```yaml
# General settings
general:
  duration_minutes: 60        # How long to run
  interval_ms: 100           # Main loop interval
  starting_capital: 100.0    # USDT per strategy

# Starting assets (in addition to USDT)
starting_assets:
  XRP: 500.0
  BTC: 0.0

# Symbols to trade
symbols:
  - XRP/USDT
  - BTC/USDT
  - XRP/BTC

# Data source
data:
  source: kraken              # 'kraken' or 'simulated'
  ws_url: wss://ws.kraken.com/v2

# Execution settings
execution:
  fee_rate: 0.001            # 0.1%
  slippage_rate: 0.0005      # 0.05%
```

### Strategy Overrides

Override default strategy parameters:

```yaml
strategy_overrides:
  market_making:
    min_spread_pct: 0.05
    position_size_usd: 20

  mean_reversion:
    lookback_candles: 20
    deviation_threshold: 0.5
```

---

## Monitoring

### Logs

```bash
# View paper tester logs
tail -f ws_paper_tester/logs/session_*.log

# Database logs
docker compose logs -f timescaledb
```

### Health Checks

```bash
# Database health
docker compose ps  # Should show "healthy"

# Check database connectivity
docker exec -it kraken_timescaledb pg_isready -U trading -d kraken_data
```

---

## Troubleshooting

### Database Won't Start

```bash
# Check container logs
docker compose logs timescaledb

# Verify .env file exists and has DB_PASSWORD
cat .env | grep DB_PASSWORD

# Reset database (WARNING: deletes data)
docker compose down -v
docker compose up -d timescaledb
```

### Connection Refused Errors

```bash
# Verify container is running
docker compose ps

# Check port binding
docker compose port timescaledb 5432
# Should show: 0.0.0.0:5433

# Test connection
psql "postgresql://trading:YOUR_PASSWORD@localhost:5433/kraken_data"
```

### Gap Filler Timeout

```bash
# Skip gap filling and run manually later
python main_with_historical.py --skip-gap-fill

# Run gap filler separately with specific symbols
python -m data.gap_filler --symbols XRP/USDT
```

### Out of Memory

Edit `docker-compose.yml` to reduce PostgreSQL memory:

```yaml
command: >
  postgres
    -c shared_buffers=512MB
    -c effective_cache_size=1GB
    -c work_mem=32MB
```

---

## Next Steps

1. **Explore strategies**: Check `ws_paper_tester/strategies/` for available strategies
2. **View documentation**: See [Historical Data Guide](../../../ws_paper_tester/docs/user/how-to/operate-historical-data.md)
3. **Create custom strategy**: Add a new `.py` file to `strategies/` directory
4. **Run backtests**: Use the historical data provider API

---

## Support

- **Documentation**: [docs/index.md](../../index.md)
- **Issues**: [GitHub Issues](https://github.com/9RESE/TG4/issues)

---

*Last updated: 2025-12-15*
