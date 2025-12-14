"""
Portfolio management for WebSocket Paper Trading Tester.
Each strategy operates with its own isolated portfolio.

Thread Safety:
- StrategyPortfolio uses RLock for all balance/position modifications
- PortfolioManager uses RLock for portfolio-level operations
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from .types import Position, Fill


STARTING_CAPITAL = 100.0  # $100 USDT per strategy
MAX_FILLS_HISTORY = 1000  # Maximum number of fills to keep in history


@dataclass
class StrategyPortfolio:
    """
    Isolated portfolio for a single strategy.
    Thread-safe: uses RLock for all balance/position modifications.

    v1.1: Added per-pair PnL and metrics tracking
    """
    strategy_name: str
    starting_capital: float = STARTING_CAPITAL

    # Balances
    usdt: float = field(default=STARTING_CAPITAL)
    assets: Dict[str, float] = field(default_factory=dict)  # {'XRP': 0, 'BTC': 0}

    # Positions - symbol -> Position
    positions: Dict[str, Position] = field(default_factory=dict)

    # History (bounded to prevent memory growth)
    fills: List[Fill] = field(default_factory=list)
    _max_fills: int = field(default=MAX_FILLS_HISTORY, repr=False)

    # Metrics - Global
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = field(default=None)  # Will be set in __post_init__

    # Metrics - Per-pair (v1.1)
    pnl_by_symbol: Dict[str, float] = field(default_factory=dict)
    trades_by_symbol: Dict[str, int] = field(default_factory=dict)
    wins_by_symbol: Dict[str, int] = field(default_factory=dict)
    losses_by_symbol: Dict[str, int] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_trade_at: Optional[datetime] = None

    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self):
        """Initialize with starting capital and set peak equity correctly."""
        # Fix MED-004: peak_equity defaults to starting_capital, not module constant
        if self.peak_equity is None:
            self.peak_equity = self.starting_capital
        if self.usdt == STARTING_CAPITAL and self.starting_capital != STARTING_CAPITAL:
            self.usdt = self.starting_capital

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity (USDT + position value). Thread-safe."""
        with self._lock:
            equity = self.usdt
            for asset, amount in self.assets.items():
                if amount > 0:
                    # Try different symbol formats
                    price = prices.get(f"{asset}/USD", 0)
                    if not price:
                        price = prices.get(f"{asset}/USDT", 0)
                    equity += amount * price
            return equity

    def update_drawdown(self, prices: Dict[str, float]):
        """Track max drawdown. Thread-safe."""
        with self._lock:
            equity = self.get_equity(prices)
            if equity > self.peak_equity:
                self.peak_equity = equity
            if self.peak_equity > 0:
                drawdown = (self.peak_equity - equity) / self.peak_equity
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown

    def get_win_rate(self) -> float:
        """Calculate win rate percentage. Thread-safe."""
        with self._lock:
            if self.total_trades == 0:
                return 0.0
            return (self.winning_trades / self.total_trades) * 100

    def get_roi(self, prices: Dict[str, float]) -> float:
        """Calculate return on investment percentage. Thread-safe."""
        with self._lock:
            # Fix MED-003: Guard against division by zero
            if self.starting_capital <= 0:
                return 0.0
            equity = self.get_equity(prices)
            return ((equity - self.starting_capital) / self.starting_capital) * 100

    def add_fill(self, fill: Fill):
        """Add a fill to history. Thread-safe. Bounded to prevent memory growth."""
        with self._lock:
            self.fills.append(fill)
            # LOW-008: Bound the fills list to prevent memory growth
            if len(self.fills) > self._max_fills:
                self.fills = self.fills[-self._max_fills:]

    def record_trade_result(self, symbol: str, pnl: float):
        """
        Record trade result for per-pair metrics. Thread-safe.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            pnl: Realized P&L for this trade
        """
        with self._lock:
            # Update per-symbol PnL
            self.pnl_by_symbol[symbol] = self.pnl_by_symbol.get(symbol, 0.0) + pnl

            # Update per-symbol trade counts
            self.trades_by_symbol[symbol] = self.trades_by_symbol.get(symbol, 0) + 1

            # Update per-symbol win/loss
            if pnl > 0:
                self.wins_by_symbol[symbol] = self.wins_by_symbol.get(symbol, 0) + 1
            elif pnl < 0:
                self.losses_by_symbol[symbol] = self.losses_by_symbol.get(symbol, 0) + 1

    def get_symbol_stats(self, symbol: str) -> dict:
        """Get statistics for a specific trading pair. Thread-safe."""
        with self._lock:
            trades = self.trades_by_symbol.get(symbol, 0)
            wins = self.wins_by_symbol.get(symbol, 0)
            losses = self.losses_by_symbol.get(symbol, 0)
            pnl = self.pnl_by_symbol.get(symbol, 0.0)

            win_rate = (wins / trades * 100) if trades > 0 else 0.0

            # Calculate avg P&L for this symbol
            symbol_fills = [f for f in self.fills if f.symbol == symbol and f.pnl != 0]
            avg_pnl = (sum(f.pnl for f in symbol_fills) / len(symbol_fills)) if symbol_fills else 0.0

            return {
                'symbol': symbol,
                'trades': trades,
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 1),
                'pnl': round(pnl, 4),
                'avg_pnl': round(avg_pnl, 4),
            }

    def get_all_symbol_stats(self) -> Dict[str, dict]:
        """Get statistics for all traded symbols. Thread-safe."""
        with self._lock:
            all_symbols = set(self.pnl_by_symbol.keys()) | set(self.trades_by_symbol.keys())
            return {symbol: self.get_symbol_stats(symbol) for symbol in all_symbols}

    def get_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss). Thread-safe."""
        with self._lock:
            gross_profit = sum(f.pnl for f in self.fills if f.pnl > 0)
            gross_loss = abs(sum(f.pnl for f in self.fills if f.pnl < 0))
            return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def get_avg_trade_pnl(self) -> float:
        """Calculate average P&L per trade. Thread-safe."""
        with self._lock:
            if self.total_trades == 0:
                return 0.0
            return self.total_pnl / self.total_trades

    def get_sharpe_ratio(self, prices: Dict[str, float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from fills (simplified). Thread-safe."""
        with self._lock:
            if len(self.fills) < 2:
                return 0.0

            # Guard against zero starting capital
            if self.starting_capital <= 0:
                return 0.0

            returns = [f.pnl / self.starting_capital for f in self.fills if f.pnl != 0]
            if not returns:
                return 0.0

            import statistics
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0.0

            if std_return == 0:
                return 0.0

            return (mean_return - risk_free_rate) / std_return

    def to_dict(self, prices: Dict[str, float]) -> dict:
        """Serialize for dashboard/logging. Thread-safe."""
        with self._lock:
            # Calculate asset values in USD
            asset_values = {}
            for asset, amount in self.assets.items():
                if amount != 0:
                    price = prices.get(f"{asset}/USDT", 0) or prices.get(f"{asset}/USD", 0)
                    asset_values[asset] = {
                        'amount': round(amount, 8),
                        'price': round(price, 8) if price else 0,
                        'value_usd': round(amount * price, 4) if price else 0,
                    }

            return {
                'strategy': self.strategy_name,
                'usdt': round(self.usdt, 4),
                'assets': {k: round(v, 8) for k, v in self.assets.items()},
                'asset_values': asset_values,
                'equity': round(self.get_equity(prices), 4),
                'pnl': round(self.total_pnl, 4),
                'pnl_by_symbol': {k: round(v, 4) for k, v in self.pnl_by_symbol.items()},
                'roi_pct': round(self.get_roi(prices), 2),
                'trades': self.total_trades,
                'trades_by_symbol': dict(self.trades_by_symbol),
                'win_rate': round(self.get_win_rate(), 1),
                'max_drawdown_pct': round(self.max_drawdown * 100, 2),
                'open_positions': len(self.positions),
                'profit_factor': round(self.get_profit_factor(), 2),
                'avg_trade_pnl': round(self.get_avg_trade_pnl(), 4),
                'symbol_stats': self.get_all_symbol_stats(),
            }


class PortfolioManager:
    """
    Manages isolated portfolios for all strategies.
    Thread-safe: uses RLock for portfolio-level operations.
    """

    FEE_RATE = 0.001  # 0.1%

    def __init__(
        self,
        strategy_names: List[str] = None,
        starting_capital: float = STARTING_CAPITAL,
        starting_assets: Dict[str, float] = None
    ):
        self.starting_capital = starting_capital
        self.starting_assets = starting_assets or {}
        self.portfolios: Dict[str, StrategyPortfolio] = {}
        self._lock = threading.RLock()

        if strategy_names:
            for name in strategy_names:
                portfolio = StrategyPortfolio(
                    strategy_name=name,
                    starting_capital=starting_capital,
                    usdt=starting_capital
                )
                # Add starting assets to portfolio
                if self.starting_assets:
                    portfolio.assets = dict(self.starting_assets)
                self.portfolios[name] = portfolio

    def add_strategy(self, name: str) -> StrategyPortfolio:
        """Add a new strategy portfolio at runtime. Thread-safe."""
        with self._lock:
            if name not in self.portfolios:
                portfolio = StrategyPortfolio(
                    strategy_name=name,
                    starting_capital=self.starting_capital,
                    usdt=self.starting_capital
                )
                # Add starting assets to portfolio
                if self.starting_assets:
                    portfolio.assets = dict(self.starting_assets)
                self.portfolios[name] = portfolio
            return self.portfolios[name]

    def get_portfolio(self, strategy: str) -> Optional[StrategyPortfolio]:
        """Get portfolio for a strategy. Thread-safe."""
        with self._lock:
            return self.portfolios.get(strategy)

    def get_aggregate(self, prices: Dict[str, float]) -> dict:
        """Get aggregate stats across all strategies. Thread-safe."""
        with self._lock:
            if not self.portfolios:
                return {
                    'total_strategies': 0,
                    'total_capital': 0,
                    'total_equity': 0,
                    'total_pnl': 0,
                    'total_roi_pct': 0,
                    'total_trades': 0,
                    'win_rate': 0,
                }

            total_equity = 0
            total_pnl = 0
            total_trades = 0
            total_wins = 0

            for p in self.portfolios.values():
                total_equity += p.get_equity(prices)
                total_pnl += p.total_pnl
                total_trades += p.total_trades
                total_wins += p.winning_trades

            total_capital = len(self.portfolios) * self.starting_capital

            return {
                'total_strategies': len(self.portfolios),
                'total_capital': round(total_capital, 2),
                'total_equity': round(total_equity, 4),
                'total_pnl': round(total_pnl, 4),
                'total_roi_pct': round((total_pnl / total_capital) * 100, 2) if total_capital > 0 else 0,
                'total_trades': total_trades,
                'win_rate': round((total_wins / total_trades * 100), 1) if total_trades > 0 else 0,
            }

    def get_leaderboard(self, prices: Dict[str, float]) -> List[dict]:
        """Get strategies ranked by P&L. Thread-safe."""
        with self._lock:
            stats = [p.to_dict(prices) for p in self.portfolios.values()]
            return sorted(stats, key=lambda x: x['pnl'], reverse=True)

    def get_all_fills(self, limit: int = 100) -> List[dict]:
        """Get recent fills across all strategies. Thread-safe."""
        with self._lock:
            all_fills = []
            for portfolio in self.portfolios.values():
                for fill in portfolio.fills:
                    fill_dict = {
                        'fill_id': fill.fill_id,
                        'timestamp': fill.timestamp.isoformat() if isinstance(fill.timestamp, datetime) else fill.timestamp,
                        'strategy': portfolio.strategy_name,
                        'symbol': fill.symbol,
                        'side': fill.side,
                        'size': round(fill.size, 8),
                        'price': round(fill.price, 6),
                        'fee': round(fill.fee, 6),
                        'pnl': round(fill.pnl, 4),
                        'reason': fill.signal_reason,
                    }
                    all_fills.append(fill_dict)

            # Sort by timestamp descending
            all_fills.sort(key=lambda x: x['timestamp'], reverse=True)
            return all_fills[:limit]

    def reset_strategy(self, strategy: str):
        """Reset a strategy's portfolio to starting capital and assets. Thread-safe."""
        with self._lock:
            if strategy in self.portfolios:
                portfolio = StrategyPortfolio(
                    strategy_name=strategy,
                    starting_capital=self.starting_capital,
                    usdt=self.starting_capital
                )
                # Reset with starting assets
                if self.starting_assets:
                    portfolio.assets = dict(self.starting_assets)
                self.portfolios[strategy] = portfolio

    def reset_all(self):
        """Reset all portfolios to starting capital. Thread-safe."""
        with self._lock:
            for name in list(self.portfolios.keys()):
                self.reset_strategy(name)
