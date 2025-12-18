"""
Data types for the Historical Data System.

All types are immutable (frozen dataclasses) for thread safety and reproducibility.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal, Optional


# =============================================================================
# Symbol/Pair Mappings (REC-005: Centralized location)
# =============================================================================

# Mapping from our internal symbol format to Kraken API pair names
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

# Reverse mapping from Kraken API pair names to our internal format
REVERSE_PAIR_MAP = {v: k for k, v in PAIR_MAP.items()}

# Additional mappings for CSV import (Kraken CSV naming variations)
CSV_SYMBOL_MAP = {
    'XRPUSDT': 'XRP/USDT',
    'XBTUSDT': 'BTC/USDT',
    'BTCUSDT': 'BTC/USDT',  # Alternative naming
    'XRPXBT': 'XRP/BTC',
    'XRPBTC': 'XRP/BTC',    # Alternative naming
    'ETHUSDT': 'ETH/USDT',
    'SOLUSDT': 'SOL/USDT',
    'ETHXBT': 'ETH/BTC',
    'LTCUSDT': 'LTC/USDT',
    'DOTUSDT': 'DOT/USDT',
    'ADAUSDT': 'ADA/USDT',
    'LINKUSDT': 'LINK/USDT',
}

# Default symbols for gap filling and monitoring
DEFAULT_SYMBOLS = ['XRP/USDT', 'BTC/USDT', 'XRP/BTC']


@dataclass(frozen=True)
class HistoricalTrade:
    """
    Individual trade tick from Kraken.

    Represents the highest granularity data available - every executed trade.
    Used for building candles and detailed analysis.
    """
    id: int                    # Unique trade ID
    symbol: str                # 'XRP/USDT'
    timestamp: datetime        # Nanosecond precision
    price: Decimal             # Execution price
    volume: Decimal            # Trade volume in base asset
    side: Literal['buy', 'sell']  # Taker side
    order_type: str            # 'market', 'limit'
    misc: str                  # Miscellaneous flags

    @property
    def value(self) -> Decimal:
        """Trade value in quote currency."""
        return self.price * self.volume


@dataclass(frozen=True)
class HistoricalCandle:
    """
    Full domain candle type with additional metrics.

    REC-006: Design rationale - There are two Candle types in the system:

    1. `Candle` (in historical_provider.py) - Lightweight database-optimized type:
       - Used by HistoricalDataProvider for query results
       - Includes `from_row()` for efficient asyncpg Record conversion
       - For most use cases (strategy warmup, backtesting)

    2. `HistoricalCandle` (this class) - Full domain type:
       - Includes additional fields (quote_volume)
       - Additional computed properties (upper_wick, lower_wick)
       - Used for internal data representation and storage
       - Stores candle data at various intervals (1m, 5m, 15m, etc.)

    Higher timeframes are computed via continuous aggregates in TimescaleDB.
    """
    symbol: str                # 'XRP/USDT'
    timestamp: datetime        # Candle open time (UTC)
    interval_minutes: int      # 1, 5, 15, 60, etc.
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal            # Base asset volume
    quote_volume: Optional[Decimal] = None  # Quote asset volume (USDT)
    trade_count: int = 0       # Number of trades in candle
    vwap: Optional[Decimal] = None  # Volume-weighted average price

    @property
    def typical_price(self) -> Decimal:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> Decimal:
        """Candle range (high - low)."""
        return self.high - self.low

    @property
    def body_size(self) -> Decimal:
        """Absolute size of candle body."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open

    @property
    def upper_wick(self) -> Decimal:
        """Upper wick size."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> Decimal:
        """Lower wick size."""
        return min(self.open, self.close) - self.low


@dataclass(frozen=True)
class DataGap:
    """
    Represents a gap in historical data.

    Used by the GapFiller to identify and fill missing data periods.
    """
    symbol: str
    data_type: str             # 'trades', 'candles_1m', etc.
    start_time: datetime       # Gap start (last known data)
    end_time: datetime         # Gap end (now or first known after gap)
    duration: timedelta

    @property
    def is_small(self) -> bool:
        """
        Small gaps can use OHLC API (720 candles max = 12 hours for 1m).

        Large gaps require the Trades API for complete data.
        """
        return self.duration < timedelta(hours=12)

    @property
    def candles_needed(self) -> int:
        """Estimate number of 1-minute candles needed to fill the gap."""
        return int(self.duration.total_seconds() / 60)

    @property
    def hours(self) -> float:
        """Gap duration in hours."""
        return self.duration.total_seconds() / 3600


@dataclass
class TradeRecord:
    """
    Trade record for database insertion.

    Mutable dataclass used as a buffer for batch inserts.
    """
    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Decimal
    side: str

    def to_tuple(self) -> tuple:
        """Convert to tuple for database insertion."""
        return (
            self.symbol,
            self.timestamp,
            self.price,
            self.volume,
            self.side,
            None,  # order_type
            None,  # misc
        )


@dataclass
class CandleRecord:
    """
    Candle record for database insertion.

    Mutable dataclass used as a buffer for batch inserts.
    """
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

    def to_tuple(self) -> tuple:
        """Convert to tuple for database insertion."""
        return (
            self.symbol,
            self.timestamp,
            self.interval_minutes,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            None,  # quote_volume
            self.trade_count,
            self.vwap,
        )


@dataclass(frozen=True)
class DataSyncStatus:
    """
    Sync status for gap detection and resumption.
    """
    symbol: str
    data_type: str             # 'trades', 'candles_1m', etc.
    oldest_timestamp: Optional[datetime]
    newest_timestamp: Optional[datetime]
    last_sync_at: datetime
    last_kraken_since: Optional[int]  # Kraken 'since' parameter for continuation
    total_records: int
