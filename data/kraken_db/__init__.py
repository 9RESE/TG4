"""
Kraken Historical Data System.

This module provides persistent storage and retrieval of historical market data
using TimescaleDB (PostgreSQL with time-series optimization).

Components:
- types: Data types for trades, candles, and gap detection
- bulk_csv_importer: Import historical CSV files from Kraken
- historical_backfill: Fetch complete trade history from Kraken API
- gap_filler: Detect and fill data gaps on startup
- websocket_db_writer: Persist real-time WebSocket data
- historical_provider: Query historical data for backtesting
- order_book_collector: Collect order book depth snapshots
- private_data_collector: Collect account trade/ledger history

Usage:
    from data.kraken_db import (
        HistoricalDataProvider,
        DatabaseWriter,
        GapFiller,
        run_gap_filler,
        OrderBookCollector,
        PrivateDataCollector,
    )
"""

from .types import (
    HistoricalTrade,
    HistoricalCandle,
    DataGap,
    TradeRecord,
    CandleRecord,
    # REC-005: Centralized pair mappings
    PAIR_MAP,
    REVERSE_PAIR_MAP,
    CSV_SYMBOL_MAP,
    DEFAULT_SYMBOLS,
)
from .historical_provider import HistoricalDataProvider, Candle
from .websocket_db_writer import DatabaseWriter, WebSocketDBIntegration, integrate_db_writer
from .gap_filler import GapFiller, run_gap_filler
from .historical_backfill import KrakenTradesBackfill
from .bulk_csv_importer import BulkCSVImporter
from .order_book_collector import OrderBookCollector, run_order_book_collector
from .private_data_collector import PrivateDataCollector, run_private_data_sync

__all__ = [
    # Types
    'HistoricalTrade',
    'HistoricalCandle',
    'DataGap',
    'TradeRecord',
    'CandleRecord',
    'Candle',
    # REC-005: Centralized pair mappings
    'PAIR_MAP',
    'REVERSE_PAIR_MAP',
    'CSV_SYMBOL_MAP',
    'DEFAULT_SYMBOLS',
    # Core classes
    'HistoricalDataProvider',
    'DatabaseWriter',
    'WebSocketDBIntegration',
    'GapFiller',
    'KrakenTradesBackfill',
    'BulkCSVImporter',
    # New collectors
    'OrderBookCollector',
    'PrivateDataCollector',
    # Utility functions
    'integrate_db_writer',
    'run_gap_filler',
    'run_order_book_collector',
    'run_private_data_sync',
]
