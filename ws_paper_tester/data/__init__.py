"""
Historical Data System for ws_paper_tester.

This module provides persistent storage and retrieval of historical market data
using TimescaleDB (PostgreSQL with time-series optimization).

Components:
- types: Data types for trades, candles, and external indicators
- database: Database connection and session management
- bulk_csv_importer: Import historical CSV files from Kraken
- historical_backfill: Fetch complete trade history from Kraken API
- gap_filler: Detect and fill data gaps on startup
- websocket_db_writer: Persist real-time WebSocket data
- historical_provider: Query historical data for backtesting

Usage:
    from ws_paper_tester.data import (
        HistoricalDataProvider,
        DatabaseWriter,
        GapFiller,
        run_gap_filler
    )
"""

from .types import (
    HistoricalTrade,
    HistoricalCandle,
    ExternalIndicator,
    DataGap,
    TradeRecord,
    CandleRecord,
)
from .historical_provider import HistoricalDataProvider, Candle
from .websocket_db_writer import DatabaseWriter, WebSocketDBIntegration, integrate_db_writer
from .gap_filler import GapFiller, run_gap_filler

__all__ = [
    # Types
    'HistoricalTrade',
    'HistoricalCandle',
    'ExternalIndicator',
    'DataGap',
    'TradeRecord',
    'CandleRecord',
    'Candle',
    # Core classes
    'HistoricalDataProvider',
    'DatabaseWriter',
    'WebSocketDBIntegration',
    'GapFiller',
    # Utility functions
    'integrate_db_writer',
    'run_gap_filler',
]
