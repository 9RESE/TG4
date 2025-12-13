"""
Portfolio management for WebSocket Paper Trading Tester.
Each strategy operates with its own isolated portfolio.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from .types import Position, Fill


STARTING_CAPITAL = 100.0  # $100 USDT per strategy


@dataclass
class StrategyPortfolio:
    """Isolated portfolio for a single strategy."""
    strategy_name: str
    starting_capital: float = STARTING_CAPITAL

    # Balances
    usdt: float = field(default=STARTING_CAPITAL)
    assets: Dict[str, float] = field(default_factory=dict)  # {'XRP': 0, 'BTC': 0}

    # Positions - symbol -> Position
    positions: Dict[str, Position] = field(default_factory=dict)

    # History
    fills: List[Fill] = field(default_factory=list)

    # Metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = STARTING_CAPITAL

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_trade_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize with starting capital."""
        if self.usdt == STARTING_CAPITAL:
            self.usdt = self.starting_capital
            self.peak_equity = self.starting_capital

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity (USDT + position value)."""
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
        """Track max drawdown."""
        equity = self.get_equity(prices)
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

    def get_win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def get_roi(self, prices: Dict[str, float]) -> float:
        """Calculate return on investment percentage."""
        equity = self.get_equity(prices)
        return ((equity - self.starting_capital) / self.starting_capital) * 100

    def get_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(f.pnl for f in self.fills if f.pnl > 0)
        gross_loss = abs(sum(f.pnl for f in self.fills if f.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def get_avg_trade_pnl(self) -> float:
        """Calculate average P&L per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    def get_sharpe_ratio(self, prices: Dict[str, float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from fills (simplified)."""
        if len(self.fills) < 2:
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
        """Serialize for dashboard/logging."""
        return {
            'strategy': self.strategy_name,
            'usdt': round(self.usdt, 4),
            'assets': {k: round(v, 8) for k, v in self.assets.items()},
            'equity': round(self.get_equity(prices), 4),
            'pnl': round(self.total_pnl, 4),
            'roi_pct': round(self.get_roi(prices), 2),
            'trades': self.total_trades,
            'win_rate': round(self.get_win_rate(), 1),
            'max_drawdown_pct': round(self.max_drawdown * 100, 2),
            'open_positions': len(self.positions),
            'profit_factor': round(self.get_profit_factor(), 2),
            'avg_trade_pnl': round(self.get_avg_trade_pnl(), 4),
        }


class PortfolioManager:
    """Manages isolated portfolios for all strategies."""

    FEE_RATE = 0.001  # 0.1%

    def __init__(self, strategy_names: List[str] = None, starting_capital: float = STARTING_CAPITAL):
        self.starting_capital = starting_capital
        self.portfolios: Dict[str, StrategyPortfolio] = {}

        if strategy_names:
            for name in strategy_names:
                self.portfolios[name] = StrategyPortfolio(
                    strategy_name=name,
                    starting_capital=starting_capital,
                    usdt=starting_capital
                )

    def add_strategy(self, name: str) -> StrategyPortfolio:
        """Add a new strategy portfolio at runtime."""
        if name not in self.portfolios:
            self.portfolios[name] = StrategyPortfolio(
                strategy_name=name,
                starting_capital=self.starting_capital,
                usdt=self.starting_capital
            )
        return self.portfolios[name]

    def get_portfolio(self, strategy: str) -> Optional[StrategyPortfolio]:
        """Get portfolio for a strategy."""
        return self.portfolios.get(strategy)

    def get_aggregate(self, prices: Dict[str, float]) -> dict:
        """Get aggregate stats across all strategies."""
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
        """Get strategies ranked by P&L."""
        stats = [p.to_dict(prices) for p in self.portfolios.values()]
        return sorted(stats, key=lambda x: x['pnl'], reverse=True)

    def get_all_fills(self, limit: int = 100) -> List[dict]:
        """Get recent fills across all strategies."""
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
        """Reset a strategy's portfolio to starting capital."""
        if strategy in self.portfolios:
            self.portfolios[strategy] = StrategyPortfolio(
                strategy_name=strategy,
                starting_capital=self.starting_capital,
                usdt=self.starting_capital
            )

    def reset_all(self):
        """Reset all portfolios to starting capital."""
        for name in list(self.portfolios.keys()):
            self.reset_strategy(name)
