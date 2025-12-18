# C4 Component Diagram: Kraken Historical Data System

This document provides C4 model component diagrams for the Kraken Historical Data System.

## Context Diagram (Level 1)

Shows the Kraken Historical Data System in the context of the overall trading system.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRADING SYSTEM CONTEXT                              │
│                                                                              │
│                                                                              │
│        ┌─────────────────┐                     ┌─────────────────┐          │
│        │                 │                     │                 │          │
│        │  Kraken         │    WebSocket API    │  Trading        │          │
│        │  Exchange       │◀───────────────────▶│  Strategies     │          │
│        │                 │    REST API         │                 │          │
│        │  [External]     │                     │  [System]       │          │
│        └────────┬────────┘                     └────────▲────────┘          │
│                 │                                       │                    │
│                 │                                       │                    │
│                 │ Market Data                           │ Historical Data    │
│                 │                                       │                    │
│                 ▼                                       │                    │
│        ┌────────────────────────────────────────────────┴────────┐          │
│        │                                                          │          │
│        │            KRAKEN HISTORICAL DATA SYSTEM                 │          │
│        │                                                          │          │
│        │  Stores and provides access to historical market data    │          │
│        │  for backtesting, strategy warmup, and analysis         │          │
│        │                                                          │          │
│        │  [Software System]                                       │          │
│        └──────────────────────────────────────────────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Container Diagram (Level 2)

Shows the containers within the Kraken Historical Data System.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KRAKEN HISTORICAL DATA SYSTEM                             │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                      │   │
│   │                      PYTHON APPLICATION                              │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│   │   │   Ingestion  │  │   Query      │  │   Gap        │              │   │
│   │   │   Layer      │  │   Layer      │  │   Management │              │   │
│   │   │              │  │              │  │              │              │   │
│   │   │ • WebSocket  │  │ • Provider   │  │ • GapFiller  │              │   │
│   │   │ • Backfill   │  │ • Warmup     │  │ • Detect     │              │   │
│   │   │ • CSV Import │  │ • Replay     │  │ • Fill       │              │   │
│   │   └──────┬───────┘  └──────▲───────┘  └──────┬───────┘              │   │
│   │          │                 │                 │                       │   │
│   └──────────┼─────────────────┼─────────────────┼───────────────────────┘   │
│              │                 │                 │                           │
│              │    asyncpg      │    asyncpg      │    asyncpg                │
│              │                 │                 │                           │
│              ▼                 │                 ▼                           │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                                                                       │  │
│   │                           TIMESCALEDB                                 │  │
│   │                                                                       │  │
│   │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                │  │
│   │   │   trades    │   │   candles   │   │ sync_status │                │  │
│   │   │ (hypertable)│   │ (hypertable)│   │   (table)   │                │  │
│   │   └──────┬──────┘   └──────┬──────┘   └─────────────┘                │  │
│   │          │                 │                                          │  │
│   │          │      ┌──────────┴────────────────────┐                    │  │
│   │          │      │    Continuous Aggregates      │                    │  │
│   │          │      │  5m, 15m, 30m, 1h, 4h, 1d, 1w │                    │  │
│   │          │      └───────────────────────────────┘                    │  │
│   │                                                                       │  │
│   │   [Container: PostgreSQL 15 + TimescaleDB]                           │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                          PGADMIN (Optional)                           │  │
│   │   Database management and visualization                              │  │
│   │   [Container: Web Application]                                       │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Diagram (Level 3)

Detailed view of the Python application components.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PYTHON APPLICATION                                   │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        DATA TYPES (types.py)                           │  │
│  │                                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │HistoricalTrade│ │HistoricalCandle│ │   DataGap    │                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  │                                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │ TradeRecord  │  │ CandleRecord │  │   PAIR_MAP   │                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  │                                                                         │  │
│  │  [Component: Immutable data structures and centralized mappings]       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│              ┌─────────────────────┼─────────────────────┐                  │
│              │                     │                     │                  │
│              ▼                     ▼                     ▼                  │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐   │
│  │                     │ │                     │ │                     │   │
│  │  INGESTION LAYER    │ │   QUERY LAYER       │ │  GAP MANAGEMENT     │   │
│  │                     │ │                     │ │                     │   │
│  ├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤   │
│  │                     │ │                     │ │                     │   │
│  │  ┌───────────────┐  │ │  ┌───────────────┐  │ │  ┌───────────────┐  │   │
│  │  │DatabaseWriter │  │ │  │HistoricalData │  │ │  │  GapFiller    │  │   │
│  │  │               │  │ │  │Provider       │  │ │  │               │  │   │
│  │  │ • Buffering   │  │ │  │               │  │ │  │ • Detection   │  │   │
│  │  │ • COPY insert │  │ │  │ • get_candles │  │ │  │ • OHLC fill   │  │   │
│  │  │ • Statistics  │  │ │  │ • warmup      │  │ │  │ • Trades fill │  │   │
│  │  └───────┬───────┘  │ │  │ • replay      │  │ │  │ • Refresh     │  │   │
│  │          │          │ │  │ • MTF data    │  │ │  │   aggregates  │  │   │
│  │  ┌───────┴───────┐  │ │  └───────────────┘  │ │  └───────────────┘  │   │
│  │  │WebSocketDB    │  │ │                     │ │                     │   │
│  │  │Integration    │  │ │  [Component]        │ │  [Component]        │   │
│  │  │               │  │ │                     │ │                     │   │
│  │  │ • on_trade    │  │ │                     │ │                     │   │
│  │  │ • on_ohlc     │  │ │                     │ │                     │   │
│  │  └───────────────┘  │ │                     │ │                     │   │
│  │                     │ │                     │ │                     │   │
│  │  [Component]        │ │                     │ │                     │   │
│  └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘   │
│             │                       │                       │               │
│  ┌──────────┴───────────────────────┴───────────────────────┴──────────┐   │
│  │                                                                      │   │
│  │                     HISTORICAL BACKFILL LAYER                        │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                   KrakenTradesBackfill                       │    │   │
│  │  │                                                              │    │   │
│  │  │  • fetch_trades_page() - Single API call                    │    │   │
│  │  │  • fetch_all_trades() - Paginated generator                 │    │   │
│  │  │  • store_trades() - Validate and persist                    │    │   │
│  │  │  • build_candles_from_trades() - Aggregate to 1m            │    │   │
│  │  │  • backfill_symbol() - Full backfill orchestration          │    │   │
│  │  │  • get_resume_point() - Resume interrupted backfill         │    │   │
│  │  │                                                              │    │   │
│  │  │  [Component: REST API Client]                               │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                   BulkCSVImporter                            │    │   │
│  │  │                                                              │    │   │
│  │  │  • import_csv_file() - Single file import                   │    │   │
│  │  │  • import_directory() - Batch import                        │    │   │
│  │  │                                                              │    │   │
│  │  │  [Component: CSV Parser]                                    │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Code Diagram (Level 4)

Detailed class diagram for key components.

### DatabaseWriter Class

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DatabaseWriter                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ CLASS CONSTANTS                                                              │
│   MAX_TRADE_BUFFER_SIZE: int = 10000                                        │
│   MAX_CANDLE_BUFFER_SIZE: int = 1000                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ ATTRIBUTES                                                                   │
│   db_url: str                                                                │
│   trade_buffer_size: int = 100                                               │
│   trade_flush_interval: float = 5.0                                          │
│   candle_flush_interval: float = 1.0                                         │
│   pool_min_size: int = 2                                                     │
│   pool_max_size: int = 10                                                    │
│   pool: Optional[asyncpg.Pool]                                               │
│   trade_buffer: deque[TradeRecord]                                           │
│   candle_buffer: deque[CandleRecord]                                         │
│   _running: bool                                                             │
│   _flush_task: Optional[asyncio.Task]                                        │
│   _lock: asyncio.Lock                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ STATISTICS                                                                   │
│   _trades_written: int                                                       │
│   _candles_written: int                                                      │
│   _flush_count: int                                                          │
│   _error_count: int                                                          │
│   _overflow_count: int                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ PUBLIC METHODS                                                               │
│   async start() -> None                                                      │
│   async stop() -> None                                                       │
│   async write_trade(trade: TradeRecord) -> None                              │
│   async write_candle(candle: CandleRecord) -> None                           │
│   async update_sync_status(symbol, data_type, timestamp) -> None             │
│   get_stats() -> dict                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ PRIVATE METHODS                                                              │
│   async _periodic_flush() -> None                                            │
│   async _flush_trades() -> None                                              │
│   async _flush_candles() -> None                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### HistoricalDataProvider Class

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HistoricalDataProvider                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ CLASS CONSTANTS                                                              │
│   INTERVAL_VIEWS: dict[int, str]                                             │
│     {1: 'candles', 5: 'candles_5m', 15: 'candles_15m', ...}                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ ATTRIBUTES                                                                   │
│   db_url: str                                                                │
│   pool_min_size: int = 2                                                     │
│   pool_max_size: int = 10                                                    │
│   pool: Optional[asyncpg.Pool]                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ CONNECTION METHODS                                                           │
│   async connect() -> None                                                    │
│   async close() -> None                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ QUERY METHODS                                                                │
│   async get_candles(symbol, interval, start, end, limit) -> List[Candle]    │
│   async get_latest_candles(symbol, interval, count) -> List[Candle]         │
│   async replay_candles(symbol, interval, start, end, speed) -> AsyncIterator│
│   async get_warmup_data(symbol, interval, periods) -> List[Candle]          │
│   async get_data_range(symbol) -> dict                                       │
│   async get_multi_timeframe_candles(symbol, end, intervals) -> dict         │
│   async get_symbols() -> List[str]                                           │
│   async get_sync_status(symbol) -> Optional[dict]                            │
│   async health_check() -> dict                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ PRIVATE METHODS                                                              │
│   _get_view_for_interval(interval_minutes: int) -> str                       │
│   _ensure_connected() -> None                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GapFiller Class

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GapFiller                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ CLASS CONSTANTS                                                              │
│   KRAKEN_BASE_URL: str = 'https://api.kraken.com'                           │
│   DEFAULT_SYMBOLS: List[str]                                                 │
│   PAIR_MAP: dict[str, str]                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ ATTRIBUTES                                                                   │
│   db_url: str                                                                │
│   symbols: List[str]                                                         │
│   rate_limit_delay: float = 1.1                                              │
│   max_retries: int = 3                                                       │
│   pool: Optional[asyncpg.Pool]                                               │
│   session: Optional[aiohttp.ClientSession]                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ LIFECYCLE METHODS                                                            │
│   async start() -> None                                                      │
│   async stop() -> None                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ GAP OPERATIONS                                                               │
│   async detect_gaps(min_gap_minutes: int = 2) -> List[DataGap]              │
│   async fill_gap_ohlc(gap: DataGap) -> int                                  │
│   async fill_gap_trades(gap: DataGap) -> int                                │
│   async fill_all_gaps(max_concurrent: int = 3) -> dict                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ HELPER METHODS                                                               │
│   async _store_trades_and_build_candles(symbol, trades) -> None             │
│   async update_sync_status(symbol, newest_time) -> None                     │
│   async refresh_continuous_aggregates() -> None                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Sequences

### Real-time Data Flow

```
┌─────────┐     ┌───────────────┐     ┌─────────────────┐     ┌────────────┐
│ Kraken  │     │ KrakenWSClient│     │ WebSocketDB     │     │ Database   │
│   WS    │     │               │     │ Integration     │     │ Writer     │
└────┬────┘     └───────┬───────┘     └────────┬────────┘     └─────┬──────┘
     │                  │                      │                    │
     │  Trade Message   │                      │                    │
     │─────────────────▶│                      │                    │
     │                  │                      │                    │
     │                  │  on_trade callback   │                    │
     │                  │─────────────────────▶│                    │
     │                  │                      │                    │
     │                  │                      │  write_trade()     │
     │                  │                      │───────────────────▶│
     │                  │                      │                    │
     │                  │                      │                    │ add to buffer
     │                  │                      │                    │──────┐
     │                  │                      │                    │      │
     │                  │                      │                    │◀─────┘
     │                  │                      │                    │
     │                  │                      │      (on flush)    │
     │                  │                      │                    │
     │                  │                      │                    │ ┌──────────┐
     │                  │                      │                    │ │TimescaleDB│
     │                  │                      │                    │─▶│ (COPY)   │
     │                  │                      │                    │ └──────────┘
```

### Backfill Data Flow

```
┌─────────┐     ┌───────────────────┐     ┌────────────┐
│ Kraken  │     │ KrakenTrades      │     │ Timescale  │
│  API    │     │ Backfill          │     │    DB      │
└────┬────┘     └─────────┬─────────┘     └─────┬──────┘
     │                    │                     │
     │                    │  fetch_trades_page  │
     │◀───────────────────│                     │
     │                    │                     │
     │  trades + since    │                     │
     │───────────────────▶│                     │
     │                    │                     │
     │                    │  store_trades       │
     │                    │────────────────────▶│
     │                    │                     │
     │                    │  build_candles      │
     │                    │────────────────────▶│
     │                    │                     │
     │                    │  (loop until        │
     │◀───────────────────│   caught up)        │
     │                    │                     │
```

### Gap Fill Data Flow

```
┌──────────┐     ┌───────────┐     ┌─────────┐     ┌────────────┐
│ Startup  │     │ GapFiller │     │ Kraken  │     │ Timescale  │
│          │     │           │     │   API   │     │    DB      │
└────┬─────┘     └─────┬─────┘     └────┬────┘     └─────┬──────┘
     │                 │                │                │
     │  run_gap_filler │                │                │
     │────────────────▶│                │                │
     │                 │                │                │
     │                 │  detect_gaps   │                │
     │                 │───────────────────────────────▶│
     │                 │                │                │
     │                 │◀───────────────────────────────│
     │                 │  gaps list                      │
     │                 │                │                │
     │                 │  (for each gap)│                │
     │                 │                │                │
     │                 │  is_small?     │                │
     │                 │────┐           │                │
     │                 │    │ yes       │                │
     │                 │◀───┘           │                │
     │                 │                │                │
     │                 │  OHLC API      │                │
     │                 │───────────────▶│                │
     │                 │  candles       │                │
     │                 │◀───────────────│                │
     │                 │                │                │
     │                 │  insert candles                 │
     │                 │───────────────────────────────▶│
     │                 │                │                │
     │  results        │                │                │
     │◀────────────────│                │                │
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Application | Python 3.10+ | Core language |
| Async I/O | asyncio | Non-blocking operations |
| Database Driver | asyncpg | PostgreSQL async client |
| HTTP Client | aiohttp | REST API calls |
| Data Processing | pandas | CSV processing |
| Database | TimescaleDB | Time-series storage |
| Container | Docker | Deployment |
| Orchestration | Docker Compose | Multi-container setup |

## Deployment View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DOCKER NETWORK                                    │
│                           (kraken_network)                                   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                      │   │
│   │  ┌─────────────────┐              ┌─────────────────┐               │   │
│   │  │                 │              │                 │               │   │
│   │  │  TimescaleDB    │◀────────────▶│    PgAdmin      │               │   │
│   │  │                 │   port 5432   │    (optional)   │               │   │
│   │  │  Port: 5433     │              │  Port: 5050     │               │   │
│   │  │  (exposed)      │              │  (exposed)      │               │   │
│   │  │                 │              │                 │               │   │
│   │  │  Volumes:       │              │  Volumes:       │               │   │
│   │  │  - data         │              │  - config       │               │   │
│   │  │  - init-db.sql  │              │                 │               │   │
│   │  │                 │              │                 │               │   │
│   │  └─────────────────┘              └─────────────────┘               │   │
│   │           ▲                                                          │   │
│   │           │                                                          │   │
│   └───────────┼──────────────────────────────────────────────────────────┘   │
│               │                                                              │
│               │ asyncpg (port 5433)                                          │
│               │                                                              │
│   ┌───────────┴──────────────────────────────────────────────────────────┐   │
│   │                                                                       │   │
│   │                     PYTHON APPLICATION                                │   │
│   │                     (Host Machine)                                    │   │
│   │                                                                       │   │
│   │   • DatabaseWriter                                                    │   │
│   │   • HistoricalDataProvider                                           │   │
│   │   • GapFiller                                                         │   │
│   │   • KrakenTradesBackfill                                             │   │
│   │   • BulkCSVImporter                                                  │   │
│   │                                                                       │   │
│   └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
