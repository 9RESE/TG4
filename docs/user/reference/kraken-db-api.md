# Kraken Historical Data System - API Reference

Complete API reference for the Kraken Historical Data System.

## Module: data.kraken_db

### Exports

```python
from data.kraken_db import (
    # Types
    HistoricalTrade,
    HistoricalCandle,
    DataGap,
    TradeRecord,
    CandleRecord,
    Candle,
    # Pair mappings
    PAIR_MAP,
    REVERSE_PAIR_MAP,
    CSV_SYMBOL_MAP,
    DEFAULT_SYMBOLS,
    # Core classes
    HistoricalDataProvider,
    DatabaseWriter,
    WebSocketDBIntegration,
    GapFiller,
    KrakenTradesBackfill,
    BulkCSVImporter,
    # Utility functions
    integrate_db_writer,
    run_gap_filler,
)
```

---

## Types

### HistoricalTrade

Individual trade tick from Kraken. Immutable (frozen dataclass).

```python
@dataclass(frozen=True)
class HistoricalTrade:
    id: int                           # Unique trade ID
    symbol: str                       # 'XRP/USDT'
    timestamp: datetime               # Nanosecond precision
    price: Decimal                    # Execution price
    volume: Decimal                   # Trade volume in base asset
    side: Literal['buy', 'sell']      # Taker side
    order_type: str                   # 'market', 'limit'
    misc: str                         # Miscellaneous flags

    @property
    def value(self) -> Decimal:
        """Trade value in quote currency."""
```

### HistoricalCandle

Full domain candle type with additional metrics. Immutable.

```python
@dataclass(frozen=True)
class HistoricalCandle:
    symbol: str
    timestamp: datetime               # Candle open time (UTC)
    interval_minutes: int             # 1, 5, 15, 60, etc.
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal                   # Base asset volume
    quote_volume: Optional[Decimal]   # Quote asset volume
    trade_count: int
    vwap: Optional[Decimal]           # Volume-weighted average price

    @property
    def typical_price(self) -> Decimal
    @property
    def range(self) -> Decimal
    @property
    def body_size(self) -> Decimal
    @property
    def is_bullish(self) -> bool
    @property
    def upper_wick(self) -> Decimal
    @property
    def lower_wick(self) -> Decimal
```

### Candle

Lightweight candle type optimized for database queries. Immutable.

```python
@dataclass(frozen=True)
class Candle:
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
    def from_row(cls, row: asyncpg.Record) -> 'Candle'

    @property
    def typical_price(self) -> Decimal
    @property
    def range(self) -> Decimal
    @property
    def body_size(self) -> Decimal
    @property
    def is_bullish(self) -> bool

    def to_dict(self) -> dict
```

### DataGap

Represents a gap in historical data. Immutable.

```python
@dataclass(frozen=True)
class DataGap:
    symbol: str
    data_type: str                    # 'trades', 'candles_1m'
    start_time: datetime
    end_time: datetime
    duration: timedelta

    @property
    def is_small(self) -> bool        # <12 hours, can use OHLC API
    @property
    def candles_needed(self) -> int   # Estimated 1m candles
    @property
    def hours(self) -> float
```

### TradeRecord

Mutable buffer for trade inserts.

```python
@dataclass
class TradeRecord:
    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Decimal
    side: str

    def to_tuple(self) -> tuple
```

### CandleRecord

Mutable buffer for candle inserts.

```python
@dataclass
class CandleRecord:
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

    def to_tuple(self) -> tuple
```

---

## Pair Mappings

### PAIR_MAP

Internal symbol to Kraken API pair name.

```python
PAIR_MAP = {
    'XRP/USDT': 'XRPUSDT',
    'BTC/USDT': 'XBTUSDT',
    'XRP/BTC': 'XRPXBT',
    'ETH/USDT': 'ETHUSDT',
    'SOL/USDT': 'SOLUSDT',
    'ETH/BTC': 'ETHXBT',
    'LTC/USDT': 'LTCUSDT',
    'DOT/USDT': 'DOTUSDT',
    'ADA/USDT': 'ADAUSDT',
    'LINK/USDT': 'LINKUSDT',
}
```

### REVERSE_PAIR_MAP

Kraken API pair name to internal symbol.

```python
REVERSE_PAIR_MAP = {v: k for k, v in PAIR_MAP.items()}
```

### CSV_SYMBOL_MAP

CSV filename variations to internal symbol.

```python
CSV_SYMBOL_MAP = {
    'XRPUSDT': 'XRP/USDT',
    'XBTUSDT': 'BTC/USDT',
    'BTCUSDT': 'BTC/USDT',   # Alternative naming
    # ...
}
```

### DEFAULT_SYMBOLS

Default symbols for gap filling and monitoring.

```python
DEFAULT_SYMBOLS = ['XRP/USDT', 'BTC/USDT', 'XRP/BTC']
```

---

## Classes

### HistoricalDataProvider

Query historical data for backtesting and strategy warmup.

```python
class HistoricalDataProvider:
    INTERVAL_VIEWS = {
        1: 'candles', 5: 'candles_5m', 15: 'candles_15m',
        30: 'candles_30m', 60: 'candles_1h', 240: 'candles_4h',
        720: 'candles_12h', 1440: 'candles_1d', 10080: 'candles_1w',
    }

    def __init__(
        self,
        db_url: str,
        pool_min_size: int = 2,
        pool_max_size: int = 10
    )

    async def connect(self) -> None
    async def close(self) -> None

    async def get_candles(
        self,
        symbol: str,
        interval_minutes: int,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None
    ) -> List[Candle]

    async def get_latest_candles(
        self,
        symbol: str,
        interval_minutes: int,
        count: int
    ) -> List[Candle]

    async def replay_candles(
        self,
        symbol: str,
        interval_minutes: int,
        start: datetime,
        end: datetime,
        speed: float = 1.0
    ) -> AsyncIterator[Candle]

    async def get_warmup_data(
        self,
        symbol: str,
        interval_minutes: int,
        warmup_periods: int
    ) -> List[Candle]

    async def get_data_range(
        self,
        symbol: str
    ) -> dict  # {'oldest', 'newest', 'total_candles'}

    async def get_multi_timeframe_candles(
        self,
        symbol: str,
        end_time: datetime,
        intervals: List[int] = None,
        lookback_candles: int = 100
    ) -> dict[int, List[Candle]]

    async def get_symbols(self) -> List[str]

    async def get_sync_status(
        self,
        symbol: str
    ) -> Optional[dict]

    async def health_check(self) -> dict
```

### DatabaseWriter

Asynchronous database writer with buffering.

```python
class DatabaseWriter:
    MAX_TRADE_BUFFER_SIZE = 10000
    MAX_CANDLE_BUFFER_SIZE = 1000

    def __init__(
        self,
        db_url: str,
        trade_buffer_size: int = 100,
        trade_flush_interval: float = 5.0,
        candle_flush_interval: float = 1.0,
        pool_min_size: int = 2,
        pool_max_size: int = 10
    )

    async def start(self) -> None
    async def stop(self) -> None

    async def write_trade(self, trade: TradeRecord) -> None
    async def write_candle(self, candle: CandleRecord) -> None

    async def update_sync_status(
        self,
        symbol: str,
        data_type: str,
        timestamp: datetime
    ) -> None

    def get_stats(self) -> dict
```

**get_stats() return value**:
```python
{
    'trades_written': int,
    'candles_written': int,
    'flush_count': int,
    'error_count': int,
    'overflow_count': int,
    'trade_buffer_size': int,
    'candle_buffer_size': int,
}
```

### WebSocketDBIntegration

Integration layer for KrakenWSClient.

```python
class WebSocketDBIntegration:
    def __init__(self, db_writer: DatabaseWriter)

    async def on_trade(
        self,
        symbol: str,
        trade_data: dict  # {'price', 'qty', 'timestamp', 'side'}
    ) -> None

    async def on_ohlc(
        self,
        symbol: str,
        ohlc_data: dict,  # {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades', 'vwap'}
        interval: int = 1
    ) -> None

    async def flush_current_candles(self) -> None
```

### GapFiller

Detect and fill data gaps on startup.

```python
class GapFiller:
    def __init__(
        self,
        db_url: str,
        symbols: Optional[List[str]] = None,
        rate_limit_delay: float = 1.1,
        max_retries: int = 3
    )

    async def start(self) -> None
    async def stop(self) -> None

    async def detect_gaps(
        self,
        min_gap_minutes: int = 2
    ) -> List[DataGap]

    async def fill_gap_ohlc(self, gap: DataGap) -> int
    async def fill_gap_trades(self, gap: DataGap) -> int

    async def update_sync_status(
        self,
        symbol: str,
        newest_time: datetime
    ) -> None

    async def refresh_continuous_aggregates(self) -> None

    async def fill_all_gaps(
        self,
        max_concurrent: int = 3
    ) -> dict
```

**fill_all_gaps() return value**:
```python
{
    'gaps_detected': int,
    'gaps_filled': int,
    'candles_inserted': int,
    'trades_processed': int,
    'errors': List[str],
}
```

### KrakenTradesBackfill

Fetch complete trade history from Kraken REST API.

```python
class KrakenTradesBackfill:
    BASE_URL = 'https://api.kraken.com'
    RATE_LIMIT_DELAY = 1.1

    def __init__(
        self,
        db_url: str,
        rate_limit_delay: float = 1.1,
        max_retries: int = 3
    )

    async def connect(self) -> None
    async def close(self) -> None

    async def fetch_trades_page(
        self,
        pair: str,
        since: int = 0
    ) -> Tuple[List[list], int]

    async def fetch_all_trades(
        self,
        symbol: str,
        start_since: int = 0,
        end_timestamp: Optional[datetime] = None,
        auto_stop_realtime: bool = True
    ) -> AsyncIterator[List[list]]

    async def store_trades(
        self,
        symbol: str,
        trades: List[list]
    ) -> None

    async def build_candles_from_trades(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> None

    async def backfill_symbol(
        self,
        symbol: str,
        since: int = 0,
        end_timestamp: Optional[datetime] = None,
        build_candles: bool = True,
        auto_stop_realtime: bool = True
    ) -> int  # Total trades imported

    async def get_resume_point(
        self,
        symbol: str
    ) -> int  # Nanosecond timestamp
```

### BulkCSVImporter

Import Kraken historical CSV files.

```python
class BulkCSVImporter:
    CSV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
    INTERVAL_MAP = {'1': 1, '5': 5, '15': 15, '30': 30, '60': 60, ...}

    def __init__(
        self,
        db_url: str,
        batch_size: int = 10000
    )

    async def connect(self) -> None
    async def close(self) -> None

    async def import_csv_file(
        self,
        filepath: Path,
        symbol: str,
        interval_minutes: int
    ) -> int  # Rows imported

    async def import_directory(
        self,
        directory: Path,
        only_1m: bool = True
    ) -> dict  # {symbol: {interval: count}}
```

---

## Functions

### integrate_db_writer

Integrate database writer with existing WebSocket client.

```python
def integrate_db_writer(
    ws_client,
    db_writer: DatabaseWriter
) -> WebSocketDBIntegration
```

**Usage**:
```python
db_writer = DatabaseWriter(db_url)
await db_writer.start()

ws_client = KrakenWSClient(...)
integration = integrate_db_writer(ws_client, db_writer)

await ws_client.connect()
# ... trading logic ...

await integration.flush_current_candles()
await db_writer.stop()
```

### run_gap_filler

Convenience function to run gap filler.

```python
async def run_gap_filler(
    db_url: str,
    symbols: Optional[List[str]] = None
) -> dict
```

**Return value**:
```python
{
    'gaps_detected': int,
    'gaps_filled': int,
    'candles_inserted': int,
    'trades_processed': int,
    'errors': List[str],
}
```

---

## CLI Commands

### historical_backfill

```bash
python -m data.kraken_db.historical_backfill \
    --symbols XRP/USDT BTC/USDT \
    --db-url postgresql://trading:pass@localhost:5433/kraken_data \
    --resume \
    --no-candles \
    --no-auto-stop
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--symbols` | Trading pairs to backfill | XRP/USDT BTC/USDT XRP/BTC |
| `--db-url` | PostgreSQL connection URL | $DATABASE_URL |
| `--resume` | Resume from last sync point | False |
| `--no-candles` | Skip candle building | False |
| `--no-auto-stop` | Disable auto-stop at real-time | False |

### gap_filler

```bash
python -m data.kraken_db.gap_filler \
    --symbols XRP/USDT \
    --db-url postgresql://trading:pass@localhost:5433/kraken_data
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--symbols` | Trading pairs to check | DEFAULT_SYMBOLS |
| `--db-url` | PostgreSQL connection URL | $DATABASE_URL |

### bulk_csv_importer

```bash
python -m data.kraken_db.bulk_csv_importer \
    --dir ./data/kraken_csv \
    --db-url postgresql://trading:pass@localhost:5433/kraken_data \
    --all \
    --batch-size 5000
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--dir` | Directory with CSV files | ./data/kraken_csv |
| `--db-url` | PostgreSQL connection URL | $DATABASE_URL |
| `--all` | Import all intervals | False (only 1m) |
| `--batch-size` | Records per batch insert | 10000 |

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection URL | Yes |
| `DB_PASSWORD` | Database password (for Docker) | Yes (Docker) |
| `PGADMIN_PASSWORD` | PgAdmin password | No |
| `KRAKEN_API_KEY` | Kraken API key | No |
| `KRAKEN_API_SECRET` | Kraken API secret | No |

---

## Database Schema

### trades table

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | BIGSERIAL | No | Auto-increment ID |
| symbol | VARCHAR(20) | No | Trading pair |
| timestamp | TIMESTAMPTZ | No | Trade time |
| price | DECIMAL(20,10) | No | Execution price |
| volume | DECIMAL(20,10) | No | Trade volume |
| side | VARCHAR(4) | No | 'buy' or 'sell' |
| order_type | VARCHAR(10) | Yes | 'market', 'limit' |
| misc | VARCHAR(50) | Yes | Miscellaneous flags |

**Primary Key**: (timestamp, symbol, id)

### candles table

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| symbol | VARCHAR(20) | No | Trading pair |
| timestamp | TIMESTAMPTZ | No | Candle open time |
| interval_minutes | SMALLINT | No | 1, 5, 15, 60, etc. |
| open | DECIMAL(20,10) | No | Open price |
| high | DECIMAL(20,10) | No | High price |
| low | DECIMAL(20,10) | No | Low price |
| close | DECIMAL(20,10) | No | Close price |
| volume | DECIMAL(20,10) | No | Base volume |
| quote_volume | DECIMAL(20,10) | Yes | Quote volume |
| trade_count | INTEGER | Yes | Number of trades |
| vwap | DECIMAL(20,10) | Yes | Volume-weighted avg |

**Primary Key**: (timestamp, symbol, interval_minutes)

### data_sync_status table

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| symbol | VARCHAR(20) | No | Trading pair |
| data_type | VARCHAR(20) | No | 'trades', 'candles_1m' |
| oldest_timestamp | TIMESTAMPTZ | Yes | Oldest record |
| newest_timestamp | TIMESTAMPTZ | Yes | Newest record |
| last_sync_at | TIMESTAMPTZ | No | Last sync time |
| last_kraken_since | BIGINT | Yes | Kraken 'since' param |
| total_records | BIGINT | No | Total record count |

**Primary Key**: (symbol, data_type)

---

## Continuous Aggregates

Higher timeframe candles are automatically computed from 1-minute base data.

| View | Interval | Source | Refresh Schedule |
|------|----------|--------|------------------|
| candles_5m | 5 minutes | candles (1m) | Every 5 minutes |
| candles_15m | 15 minutes | candles (1m) | Every 15 minutes |
| candles_30m | 30 minutes | candles (1m) | Every 30 minutes |
| candles_1h | 1 hour | candles (1m) | Every 1 hour |
| candles_4h | 4 hours | candles_1h | Every 4 hours |
| candles_12h | 12 hours | candles_1h | Every 12 hours |
| candles_1d | 1 day | candles_1h | Daily |
| candles_1w | 1 week | candles_1d | Weekly |

All aggregates have the same column structure as the base `candles` table.

---

## TimescaleDB Features

### Hypertables

| Table | Chunk Interval | Compression After |
|-------|----------------|-------------------|
| trades | 1 day | 7 days |
| candles | 7 days | 30 days |

### Retention Policies

| Table | Retention Period |
|-------|-----------------|
| trades | 90 days |
| candles | 365 days |

### Compression Settings

Trades are compressed by `symbol`, ordered by `timestamp DESC`.
Candles are compressed by `symbol, interval_minutes`, ordered by `timestamp DESC`.
