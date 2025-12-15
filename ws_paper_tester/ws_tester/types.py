"""
Core types for WebSocket Paper Trading Tester.
All data structures are immutable for thread safety and reproducibility.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from functools import cached_property

# Type-only import to avoid circular dependency
if TYPE_CHECKING:
    from ws_tester.regime.types import RegimeSnapshot


@dataclass(frozen=True)
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body_size(self) -> float:
        """Absolute size of candle body."""
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """High to low range."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open


@dataclass(frozen=True)
class Trade:
    """Single trade from the tape."""
    timestamp: datetime
    price: float
    size: float
    side: str  # 'buy' or 'sell'

    @property
    def value(self) -> float:
        """Trade value in quote currency."""
        return self.price * self.size


@dataclass(frozen=True)
class OrderbookSnapshot:
    """Snapshot of orderbook state."""
    bids: Tuple[Tuple[float, float], ...]  # ((price, size), ...)
    asks: Tuple[Tuple[float, float], ...]

    @property
    def best_bid(self) -> float:
        """Best bid price."""
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Best ask price."""
        return self.asks[0][0] if self.asks else 0.0

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.best_ask - self.best_bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        mid = self.mid
        return (self.spread / mid * 100) if mid > 0 else 0.0

    @property
    def mid(self) -> float:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask or 0.0

    @property
    def bid_depth(self) -> float:
        """Total bid side depth."""
        return sum(size for _, size in self.bids)

    @property
    def ask_depth(self) -> float:
        """Total ask side depth."""
        return sum(size for _, size in self.asks)

    @property
    def imbalance(self) -> float:
        """Order book imbalance: positive = more bids, negative = more asks."""
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return 0.0
        return (self.bid_depth - self.ask_depth) / total


@dataclass(frozen=True)
class DataSnapshot:
    """
    Immutable market data snapshot passed to strategies.

    Thread-safe: All fields are immutable (frozen dataclass with tuple containers).

    Note: Not truly hashable due to Dict fields. For replay/logging purposes,
    use the snapshot's timestamp and a content hash of serialized data.
    Consider using DataSnapshot.to_hash() for unique identification.
    """
    timestamp: datetime

    # Current prices
    prices: Dict[str, float]  # {'XRP/USD': 2.35, 'BTC/USD': 104500}

    # OHLC candles (last N)
    candles_1m: Dict[str, Tuple[Candle, ...]]  # {'XRP/USD': (Candle, ...)}
    candles_5m: Dict[str, Tuple[Candle, ...]]

    # Orderbook state
    orderbooks: Dict[str, OrderbookSnapshot]

    # Recent trades
    trades: Dict[str, Tuple[Trade, ...]]  # Last 100 trades per symbol

    # Market regime (optional, populated by RegimeDetector)
    regime: Optional['RegimeSnapshot'] = None

    @cached_property
    def spreads(self) -> Dict[str, float]:
        """Get spreads for all symbols."""
        return {sym: ob.spread for sym, ob in self.orderbooks.items()}

    @cached_property
    def mids(self) -> Dict[str, float]:
        """Get mid prices for all symbols."""
        return {sym: ob.mid for sym, ob in self.orderbooks.items()}

    def get_vwap(self, symbol: str, n_trades: int = 50) -> Optional[float]:
        """Calculate VWAP from recent trades."""
        trades = self.trades.get(symbol, ())
        if not trades:
            return None

        # FIX: Use negative index to get NEWEST trades (CRIT-NEW-001)
        trades = trades[-n_trades:]
        total_value = sum(t.price * t.size for t in trades)
        total_volume = sum(t.size for t in trades)

        return total_value / total_volume if total_volume > 0 else None

    def get_trade_imbalance(self, symbol: str, n_trades: int = 50) -> float:
        """Calculate buy/sell imbalance from recent trades."""
        trades = self.trades.get(symbol, ())
        if not trades:
            return 0.0

        # FIX: Use negative index to get NEWEST trades (CRIT-NEW-001)
        trades = trades[-n_trades:]
        buy_volume = sum(t.size for t in trades if t.side == 'buy')
        sell_volume = sum(t.size for t in trades if t.side == 'sell')
        total = buy_volume + sell_volume

        return (buy_volume - sell_volume) / total if total > 0 else 0.0


@dataclass
class Signal:
    """
    Trading signal generated by a strategy.

    MED-002: Metadata is copied on init to prevent shared mutable state.
    """
    action: str  # 'buy', 'sell', 'short', 'cover'
    symbol: str  # 'XRP/USD'
    size: float  # Position size in USD or base asset
    price: float  # Reference price (for logging)
    reason: str  # Human-readable explanation

    # Optional fields
    order_type: str = 'market'  # 'market' or 'limit'
    limit_price: Optional[float] = None  # For limit orders
    stop_loss: Optional[float] = None  # Auto stop-loss price
    take_profit: Optional[float] = None  # Auto take-profit price
    metadata: Optional[dict] = field(default_factory=dict)  # Strategy-specific data

    def __post_init__(self):
        """Validate signal data and ensure metadata is a copy."""
        valid_actions = {'buy', 'sell', 'short', 'cover'}
        if self.action not in valid_actions:
            raise ValueError(f"Invalid action '{self.action}'. Must be one of {valid_actions}")

        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")

        if self.order_type == 'limit' and self.limit_price is None:
            raise ValueError("Limit orders require limit_price")

        # Ensure metadata is a copy to prevent shared mutable state
        # Note: Signal is not frozen, so we use regular assignment
        if self.metadata is None:
            self.metadata = {}
        elif not isinstance(self.metadata, dict):
            self.metadata = {}
        else:
            self.metadata = dict(self.metadata)


@dataclass(frozen=True)
class Position:
    """Open position in a symbol."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float  # Base asset amount
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = float('inf')
    entry_fee: float = 0.0  # Fee paid at entry for complete P&L calculation

    @property
    def notional(self) -> float:
        """Position notional value at entry."""
        return self.size * self.entry_price

    def unrealized_pnl(self, current_price: float, exit_fee: float = 0.0) -> float:
        """
        Calculate unrealized P&L including entry and estimated exit fees.

        Args:
            current_price: Current market price
            exit_fee: Estimated exit fee (default 0)

        Returns:
            Net unrealized P&L after fees
        """
        if self.side == 'long':
            gross_pnl = (current_price - self.entry_price) * self.size
        else:  # short
            gross_pnl = (self.entry_price - current_price) * self.size

        return gross_pnl - self.entry_fee - exit_fee

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L as percentage (excludes fees for simplicity)."""
        if self.side == 'long':
            gross_pnl = (current_price - self.entry_price) * self.size
        else:
            gross_pnl = (self.entry_price - current_price) * self.size
        return (gross_pnl / self.notional) * 100 if self.notional > 0 else 0.0


@dataclass
class Fill:
    """Executed trade fill."""
    fill_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy', 'sell'
    size: float  # Base asset size
    price: float
    fee: float
    signal_reason: str
    pnl: float = 0.0  # Realized P&L for this fill
    strategy: str = ''  # Strategy that generated this fill

    @property
    def value(self) -> float:
        """Total fill value in quote currency."""
        return self.size * self.price

    @property
    def cost(self) -> float:
        """Total cost including fee."""
        return self.value + self.fee
