"""
Per-Strategy Isolated Portfolios
Phase 32: Each strategy gets its own $1000 USDT portfolio for independent performance tracking.

This module provides:
- StrategyPortfolio: Isolated balance and position tracking per strategy
- StrategyPortfolioManager: Manages all strategy portfolios and leaderboard
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class StrategyMetrics:
    """Per-strategy performance metrics."""
    starting_balance: float = 1000.0
    current_balance: float = 1000.0
    peak_balance: float = 1000.0  # High water mark
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    roi_pct: float = 0.0
    total_fees: float = 0.0


@dataclass
class StrategyPosition:
    """Position within a strategy portfolio."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    size: float  # In base asset units
    cost_basis: float  # USDT cost
    leverage: int
    entry_time: datetime
    peak_price: float = 0.0  # For trailing stops
    lowest_price: float = float('inf')  # For short trailing stops


@dataclass
class StrategyRanking:
    """Ranking entry for leaderboard."""
    rank: int
    strategy_name: str
    roi_pct: float
    total_pnl: float
    realized_pnl: float  # P&L from closed trades
    unrealized_pnl: float  # P&L from open positions
    win_rate: float
    trade_count: int
    closed_trades: int  # Number of closed trades (for accurate win rate)
    max_drawdown_pct: float
    status: str  # 'profitable', 'losing', 'inactive'


class StrategyPortfolio:
    """
    Isolated portfolio for a single strategy.
    Each strategy gets $1000 USDT starting capital.
    Strategies cannot borrow from each other.
    """

    def __init__(self, strategy_name: str, starting_balance: float = 1000.0,
                 fee_rate: float = 0.001):
        self.strategy_name = strategy_name
        self.starting_balance = starting_balance
        self.usdt_balance = starting_balance
        self.fee_rate = fee_rate
        self.positions: Dict[str, StrategyPosition] = {}  # symbol -> position
        self.trade_history: List[Dict] = []
        self.metrics = StrategyMetrics(
            starting_balance=starting_balance,
            current_balance=starting_balance,
            peak_balance=starting_balance
        )
        self.created_at = datetime.now()

    def get_available_usdt(self) -> float:
        """Get available USDT (not locked in positions)."""
        return self.usdt_balance

    def get_equity(self, prices: Dict[str, float]) -> float:
        """
        Calculate total equity including unrealized P&L.

        Args:
            prices: Dict of asset -> price (e.g., {'XRP': 2.50, 'BTC': 100000})
        """
        equity = self.usdt_balance

        for symbol, pos in self.positions.items():
            # Extract base asset from symbol (e.g., 'XRP' from 'XRP/USDT')
            base = symbol.split('/')[0] if '/' in symbol else symbol
            current_price = prices.get(base, prices.get(symbol, pos.entry_price))

            if pos.side == 'long':
                # Long position value
                position_value = pos.size * current_price
                equity += position_value
            else:
                # Short position: collateral + unrealized P&L
                unrealized_pnl = (pos.entry_price - current_price) * pos.size
                equity += pos.cost_basis + unrealized_pnl

        return equity

    def can_trade(self, required_usdt: float) -> bool:
        """Check if strategy has enough capital for trade."""
        return self.usdt_balance >= required_usdt

    def execute_buy(self, symbol: str, usdt_amount: float,
                    price: float, leverage: int = 1) -> Dict[str, Any]:
        """
        Execute a buy order within this strategy's portfolio.

        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            usdt_amount: Amount of USDT to spend
            price: Current price
            leverage: Leverage multiplier (1 for spot)

        Returns:
            Dict with execution result
        """
        if usdt_amount > self.usdt_balance:
            return {
                'executed': False,
                'reason': f'Insufficient balance: need ${usdt_amount:.2f}, have ${self.usdt_balance:.2f}'
            }

        # Apply fee
        fee = usdt_amount * self.fee_rate
        net_amount = usdt_amount - fee

        # Calculate position size
        size = (net_amount * leverage) / price

        # Deduct from balance
        self.usdt_balance -= usdt_amount
        self.metrics.total_fees += fee

        # Create or add to position
        if symbol in self.positions and self.positions[symbol].side == 'long':
            # Average into existing long position
            old_pos = self.positions[symbol]
            total_size = old_pos.size + size
            avg_price = (old_pos.entry_price * old_pos.size + price * size) / total_size
            old_pos.entry_price = avg_price
            old_pos.size = total_size
            old_pos.cost_basis += usdt_amount
            old_pos.peak_price = max(old_pos.peak_price, price)
        else:
            self.positions[symbol] = StrategyPosition(
                symbol=symbol,
                side='long',
                entry_price=price,
                size=size,
                cost_basis=usdt_amount,
                leverage=leverage,
                entry_time=datetime.now(),
                peak_price=price
            )

        self.metrics.trade_count += 1
        self._update_equity_metrics()

        return {
            'executed': True,
            'action': 'buy',
            'symbol': symbol,
            'size': size,
            'price': price,
            'cost': usdt_amount,
            'fee': fee,
            'leverage': leverage,
            'balance_after': self.usdt_balance
        }

    def execute_sell(self, symbol: str, price: float,
                     sell_pct: float = 1.0) -> Dict[str, Any]:
        """
        Execute a sell order (close long position).

        Args:
            symbol: Trading pair
            price: Current price
            sell_pct: Fraction of position to sell (1.0 = 100%)

        Returns:
            Dict with execution result including P&L
        """
        if symbol not in self.positions:
            return {'executed': False, 'reason': f'No position in {symbol}'}

        pos = self.positions[symbol]
        if pos.side != 'long':
            return {'executed': False, 'reason': 'Position is short, use cover'}

        # Calculate sell amount
        sell_size = pos.size * sell_pct
        gross_proceeds = sell_size * price

        # Apply fee
        fee = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - fee

        # Calculate P&L
        cost_portion = pos.cost_basis * sell_pct
        pnl = net_proceeds - cost_portion

        # Update balance
        self.usdt_balance += net_proceeds
        self.metrics.total_fees += fee

        entry_time = pos.entry_time

        # Update or close position
        if sell_pct >= 0.99:
            del self.positions[symbol]
        else:
            pos.size -= sell_size
            pos.cost_basis -= cost_portion

        # Update metrics
        self._record_trade_close(pnl, entry_time, price, 'sell')

        return {
            'executed': True,
            'action': 'sell',
            'symbol': symbol,
            'size': sell_size,
            'price': price,
            'proceeds': net_proceeds,
            'fee': fee,
            'pnl': pnl,
            'pnl_pct': (pnl / cost_portion * 100) if cost_portion > 0 else 0,
            'balance_after': self.usdt_balance
        }

    def execute_short(self, symbol: str, usdt_amount: float,
                      price: float, leverage: int = 1) -> Dict[str, Any]:
        """
        Execute a short order.

        Args:
            symbol: Trading pair
            usdt_amount: Collateral amount in USDT
            price: Current price
            leverage: Leverage multiplier

        Returns:
            Dict with execution result
        """
        if usdt_amount > self.usdt_balance:
            return {
                'executed': False,
                'reason': f'Insufficient balance: need ${usdt_amount:.2f}, have ${self.usdt_balance:.2f}'
            }

        # Apply fee
        fee = usdt_amount * self.fee_rate
        net_collateral = usdt_amount - fee

        # Calculate position size (leveraged)
        size = (net_collateral * leverage) / price

        # Lock collateral
        self.usdt_balance -= usdt_amount
        self.metrics.total_fees += fee

        # Create or add to short position
        if symbol in self.positions and self.positions[symbol].side == 'short':
            old_pos = self.positions[symbol]
            total_size = old_pos.size + size
            avg_price = (old_pos.entry_price * old_pos.size + price * size) / total_size
            old_pos.entry_price = avg_price
            old_pos.size = total_size
            old_pos.cost_basis += usdt_amount
            old_pos.lowest_price = min(old_pos.lowest_price, price)
        else:
            self.positions[symbol] = StrategyPosition(
                symbol=symbol,
                side='short',
                entry_price=price,
                size=size,
                cost_basis=usdt_amount,
                leverage=leverage,
                entry_time=datetime.now(),
                lowest_price=price
            )

        self.metrics.trade_count += 1
        self._update_equity_metrics()

        return {
            'executed': True,
            'action': 'short',
            'symbol': symbol,
            'size': size,
            'price': price,
            'collateral': usdt_amount,
            'fee': fee,
            'leverage': leverage,
            'balance_after': self.usdt_balance
        }

    def execute_cover(self, symbol: str, price: float,
                      cover_pct: float = 1.0) -> Dict[str, Any]:
        """
        Execute a cover order (close short position).

        Args:
            symbol: Trading pair
            price: Current price
            cover_pct: Fraction of position to cover (1.0 = 100%)

        Returns:
            Dict with execution result including P&L
        """
        if symbol not in self.positions:
            return {'executed': False, 'reason': f'No position in {symbol}'}

        pos = self.positions[symbol]
        if pos.side != 'short':
            return {'executed': False, 'reason': 'Position is long, use sell'}

        # Calculate cover amount
        cover_size = pos.size * cover_pct
        collateral_portion = pos.cost_basis * cover_pct

        # Calculate P&L (profit when price drops)
        pnl = (pos.entry_price - price) * cover_size

        # Apply fee on the trade value
        trade_value = cover_size * price
        fee = trade_value * self.fee_rate
        net_pnl = pnl - fee

        # Return collateral + P&L
        self.usdt_balance += collateral_portion + net_pnl
        self.metrics.total_fees += fee

        entry_time = pos.entry_time

        # Update or close position
        if cover_pct >= 0.99:
            del self.positions[symbol]
        else:
            pos.size -= cover_size
            pos.cost_basis -= collateral_portion

        # Update metrics
        self._record_trade_close(net_pnl, entry_time, price, 'cover')

        return {
            'executed': True,
            'action': 'cover',
            'symbol': symbol,
            'size': cover_size,
            'price': price,
            'entry_price': pos.entry_price,
            'fee': fee,
            'pnl': net_pnl,
            'pnl_pct': (net_pnl / collateral_portion * 100) if collateral_portion > 0 else 0,
            'balance_after': self.usdt_balance
        }

    def _record_trade_close(self, pnl: float, entry_time: datetime,
                            exit_price: float, action: str):
        """Record closed trade for metrics."""
        self.metrics.realized_pnl += pnl

        if pnl > 0:
            self.metrics.win_count += 1
            self.metrics.largest_win = max(self.metrics.largest_win, pnl)
        elif pnl < 0:
            self.metrics.loss_count += 1
            self.metrics.largest_loss = min(self.metrics.largest_loss, pnl)

        # Update equity-based metrics
        self._update_equity_metrics()

        # Record in history
        self.trade_history.append({
            'entry_time': entry_time.isoformat(),
            'exit_time': datetime.now().isoformat(),
            'action': action,
            'exit_price': exit_price,
            'pnl': pnl,
            'balance_after': self.usdt_balance
        })

    def _update_equity_metrics(self):
        """Update drawdown, ROI, peak balance."""
        # Use balance as proxy (full equity requires prices)
        equity = self.usdt_balance

        # Update peak (high water mark)
        if equity > self.metrics.peak_balance:
            self.metrics.peak_balance = equity

        # Calculate drawdown
        if self.metrics.peak_balance > 0:
            self.metrics.current_drawdown = (
                (self.metrics.peak_balance - equity) / self.metrics.peak_balance
            )
            self.metrics.max_drawdown = max(
                self.metrics.max_drawdown,
                self.metrics.current_drawdown
            )

        # ROI
        self.metrics.roi_pct = (
            (equity - self.starting_balance) / self.starting_balance * 100
        )
        self.metrics.current_balance = equity

    def update_position_prices(self, prices: Dict[str, float]):
        """Update peak/lowest prices for trailing stops."""
        for symbol, pos in self.positions.items():
            base = symbol.split('/')[0] if '/' in symbol else symbol
            current_price = prices.get(base, prices.get(symbol))
            if current_price:
                if pos.side == 'long':
                    pos.peak_price = max(pos.peak_price, current_price)
                else:
                    pos.lowest_price = min(pos.lowest_price, current_price)

    def get_performance_summary(self, prices: Dict[str, float] = None) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        prices = prices or {}
        equity = self.get_equity(prices)
        total_trades = self.metrics.win_count + self.metrics.loss_count
        win_rate = (self.metrics.win_count / total_trades * 100) if total_trades > 0 else 0.0

        return {
            'strategy_name': self.strategy_name,
            'starting_balance': self.starting_balance,
            'current_balance': self.usdt_balance,
            'equity': equity,
            'realized_pnl': self.metrics.realized_pnl,
            'total_pnl': equity - self.starting_balance,
            'roi_pct': (equity - self.starting_balance) / self.starting_balance * 100,
            'trade_count': self.metrics.trade_count,
            'closed_trades': total_trades,
            'win_count': self.metrics.win_count,
            'loss_count': self.metrics.loss_count,
            'win_rate': win_rate,
            'largest_win': self.metrics.largest_win,
            'largest_loss': self.metrics.largest_loss,
            'max_drawdown_pct': self.metrics.max_drawdown * 100,
            'current_drawdown_pct': self.metrics.current_drawdown * 100,
            'total_fees': self.metrics.total_fees,
            'open_positions': len(self.positions),
            'position_symbols': list(self.positions.keys())
        }


class StrategyPortfolioManager:
    """
    Manages isolated portfolios for all strategies.
    Provides aggregate stats and leaderboard functionality.
    """

    def __init__(self,
                 strategy_names: List[str],
                 per_strategy_capital: float = 1000.0,
                 fee_rate: float = 0.001):
        self.per_strategy_capital = per_strategy_capital
        self.fee_rate = fee_rate
        self.portfolios: Dict[str, StrategyPortfolio] = {}

        # Initialize portfolio for each strategy
        for name in strategy_names:
            self.portfolios[name] = StrategyPortfolio(
                name,
                starting_balance=per_strategy_capital,
                fee_rate=fee_rate
            )

        self.total_capital = len(strategy_names) * per_strategy_capital
        self.created_at = datetime.now()

    def get_portfolio(self, strategy_name: str) -> StrategyPortfolio:
        """Get portfolio for a specific strategy."""
        if strategy_name not in self.portfolios:
            # Create on demand with default capital
            self.portfolios[strategy_name] = StrategyPortfolio(
                strategy_name,
                starting_balance=self.per_strategy_capital,
                fee_rate=self.fee_rate
            )
            self.total_capital += self.per_strategy_capital
        return self.portfolios[strategy_name]

    def get_leaderboard(self, prices: Dict[str, float] = None) -> List[StrategyRanking]:
        """Get strategy rankings sorted by ROI."""
        prices = prices or {}
        rankings = []

        for name, portfolio in self.portfolios.items():
            summary = portfolio.get_performance_summary(prices)

            # Calculate unrealized P&L (total minus realized)
            realized = summary['realized_pnl']
            unrealized = summary['total_pnl'] - realized

            # Determine status based on realized P&L (not unrealized)
            if summary['closed_trades'] == 0:
                status = 'inactive'
            elif realized > 0:
                status = 'profitable'
            else:
                status = 'losing'

            rankings.append(StrategyRanking(
                rank=0,  # Will be set after sorting
                strategy_name=name,
                roi_pct=summary['roi_pct'],
                total_pnl=summary['total_pnl'],
                realized_pnl=realized,
                unrealized_pnl=unrealized,
                win_rate=summary['win_rate'],
                trade_count=summary['trade_count'],
                closed_trades=summary['closed_trades'],
                max_drawdown_pct=summary['max_drawdown_pct'],
                status=status
            ))

        # Sort by ROI descending
        rankings.sort(key=lambda x: x.roi_pct, reverse=True)

        # Assign ranks
        for i, r in enumerate(rankings):
            r.rank = i + 1

        return rankings

    def get_aggregate_stats(self, prices: Dict[str, float] = None) -> Dict[str, Any]:
        """Get aggregated stats across all strategies."""
        prices = prices or {}
        total_equity = 0.0
        total_realized_pnl = 0.0
        total_trades = 0
        total_wins = 0
        total_fees = 0.0
        profitable_count = 0
        active_count = 0

        for portfolio in self.portfolios.values():
            summary = portfolio.get_performance_summary(prices)
            total_equity += summary['equity']
            total_realized_pnl += summary['realized_pnl']
            total_trades += summary['trade_count']
            total_wins += summary['win_count']
            total_fees += summary['total_fees']

            if summary['closed_trades'] > 0:
                active_count += 1
                if summary['total_pnl'] > 0:
                    profitable_count += 1

        return {
            'total_capital_allocated': self.total_capital,
            'total_equity': total_equity,
            'total_pnl': total_equity - self.total_capital,
            'total_roi_pct': ((total_equity - self.total_capital) / self.total_capital * 100)
                             if self.total_capital > 0 else 0,
            'realized_pnl': total_realized_pnl,
            'total_trades': total_trades,
            'overall_win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
            'total_fees': total_fees,
            'strategies_count': len(self.portfolios),
            'active_strategies': active_count,
            'profitable_strategies': profitable_count,
            'losing_strategies': active_count - profitable_count,
            'inactive_strategies': len(self.portfolios) - active_count
        }

    def print_leaderboard(self, prices: Dict[str, float] = None, top_n: int = None):
        """Print formatted leaderboard with realized vs unrealized P&L."""
        rankings = self.get_leaderboard(prices)
        agg = self.get_aggregate_stats(prices)

        if top_n is None:
            top_n = len(rankings)

        print("\n" + "=" * 100)
        print("STRATEGY LEADERBOARD")
        print("=" * 100)
        print(f"{'Rank':<5} {'Strategy':<24} {'ROI%':>7} {'Realized':>10} {'Unrealized':>10} {'Win%':>6} {'Closed':>6} {'DD%':>6}")
        print("-" * 100)

        for r in rankings[:top_n]:
            if r.status == 'profitable':
                status_icon = "+"
            elif r.status == 'losing':
                status_icon = "-"
            else:
                status_icon = " "

            # Format P&L with sign
            realized_str = f"${r.realized_pnl:+.2f}" if r.realized_pnl != 0 else "$0.00"
            unrealized_str = f"${r.unrealized_pnl:+.2f}" if r.unrealized_pnl != 0 else "$0.00"

            print(f"{r.rank:<5} {status_icon}{r.strategy_name:<23} "
                  f"{r.roi_pct:>6.2f}% {realized_str:>10} {unrealized_str:>10} "
                  f"{r.win_rate:>5.1f}% {r.closed_trades:>6} {r.max_drawdown_pct:>5.2f}%")

        if len(rankings) > top_n:
            print(f"  ... and {len(rankings) - top_n} more strategies")

        # Calculate aggregate realized/unrealized
        total_realized = sum(r.realized_pnl for r in rankings)
        total_unrealized = sum(r.unrealized_pnl for r in rankings)

        print("=" * 100)
        print(f"Aggregate: ${agg['total_equity']:,.2f} equity | "
              f"Realized: ${total_realized:+,.2f} | Unrealized: ${total_unrealized:+,.2f} | "
              f"{agg['profitable_strategies']}/{agg['active_strategies']} profitable")
        print("=" * 100)

    def update_all_position_prices(self, prices: Dict[str, float]):
        """Update position prices across all portfolios."""
        for portfolio in self.portfolios.values():
            portfolio.update_position_prices(prices)

    def get_all_open_positions(self) -> Dict[str, List[Dict]]:
        """Get all open positions across all strategies."""
        all_positions = {}
        for name, portfolio in self.portfolios.items():
            if portfolio.positions:
                all_positions[name] = [
                    {
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'entry_price': pos.entry_price,
                        'size': pos.size,
                        'cost_basis': pos.cost_basis,
                        'leverage': pos.leverage,
                        'entry_time': pos.entry_time.isoformat()
                    }
                    for pos in portfolio.positions.values()
                ]
        return all_positions
