"""
Paper trading executor for WebSocket Paper Tester.
Simulates order execution against isolated strategy portfolios.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Tuple

from .types import Signal, DataSnapshot, Fill, Position
from .portfolio import PortfolioManager, StrategyPortfolio


class PaperExecutor:
    """
    Executes signals against isolated strategy portfolios.

    Features:
    - Slippage simulation based on orderbook
    - Fee calculation (configurable, default 0.1%)
    - Per-strategy position tracking
    - Auto stop-loss / take-profit
    - Configurable short selling leverage limit
    - Configurable slippage rate
    """

    DEFAULT_FEE_RATE = 0.001  # 0.1%
    DEFAULT_SLIPPAGE_RATE = 0.0005  # 0.05% slippage
    DEFAULT_MAX_SHORT_LEVERAGE = 2.0  # 2x leverage for shorts

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        max_short_leverage: float = None,
        slippage_rate: float = None,
        fee_rate: float = None,
    ):
        self.portfolio_manager = portfolio_manager
        # Configurable execution parameters
        self.max_short_leverage = max_short_leverage if max_short_leverage is not None else self.DEFAULT_MAX_SHORT_LEVERAGE
        self.slippage_rate = slippage_rate if slippage_rate is not None else self.DEFAULT_SLIPPAGE_RATE
        self.fee_rate = fee_rate if fee_rate is not None else self.DEFAULT_FEE_RATE

    def execute(
        self,
        signal: Signal,
        strategy: str,
        data: DataSnapshot
    ) -> Optional[Fill]:
        """Execute signal against strategy's isolated portfolio."""
        portfolio = self.portfolio_manager.get_portfolio(strategy)
        if not portfolio:
            return None

        # Get orderbook for the symbol
        ob = data.orderbooks.get(signal.symbol)

        # Calculate execution price with slippage
        if signal.action in ['buy', 'cover']:
            if ob and ob.best_ask > 0:
                execution_price = ob.best_ask * (1 + self.slippage_rate)
            else:
                execution_price = signal.price * (1 + self.slippage_rate)
        else:  # sell, short
            if ob and ob.best_bid > 0:
                execution_price = ob.best_bid * (1 - self.slippage_rate)
            else:
                execution_price = signal.price * (1 - self.slippage_rate)

        # Calculate size in base asset
        if signal.size <= 0:
            return None

        base_asset = signal.symbol.split('/')[0]
        base_size = signal.size / execution_price

        # Execute based on action
        pnl = 0.0

        if signal.action == 'buy':
            fill = self._execute_buy(
                portfolio, signal, base_asset, base_size,
                execution_price, data.timestamp
            )
        elif signal.action == 'sell':
            fill = self._execute_sell(
                portfolio, signal, base_asset, base_size,
                execution_price, data.timestamp
            )
        elif signal.action == 'short':
            fill = self._execute_short(
                portfolio, signal, base_asset, base_size,
                execution_price, data.timestamp
            )
        elif signal.action == 'cover':
            fill = self._execute_cover(
                portfolio, signal, base_asset, base_size,
                execution_price, data.timestamp
            )
        else:
            return None

        if fill:
            portfolio.add_fill(fill)
            portfolio.total_trades += 1
            portfolio.last_trade_at = data.timestamp
            portfolio.update_drawdown(data.prices)

        return fill

    def _execute_buy(
        self,
        portfolio: StrategyPortfolio,
        signal: Signal,
        base_asset: str,
        base_size: float,
        execution_price: float,
        timestamp: datetime
    ) -> Optional[Fill]:
        """Execute a buy order."""
        cost = base_size * execution_price * (1 + self.fee_rate)

        if portfolio.usdt < cost:
            # Reduce size to available balance
            available_for_trade = portfolio.usdt / (1 + self.fee_rate)
            base_size = available_for_trade / execution_price
            cost = portfolio.usdt
            if base_size <= 0:
                return None

        fee = base_size * execution_price * self.fee_rate

        portfolio.usdt -= cost
        portfolio.assets[base_asset] = portfolio.assets.get(base_asset, 0) + base_size

        # Track position
        existing_pos = portfolio.positions.get(signal.symbol)
        if existing_pos and existing_pos.side == 'long':
            # Add to existing position (average in)
            total_size = existing_pos.size + base_size
            avg_price = (
                (existing_pos.entry_price * existing_pos.size) +
                (execution_price * base_size)
            ) / total_size
            total_entry_fee = existing_pos.entry_fee + fee

            new_pos = Position(
                symbol=signal.symbol,
                side='long',
                size=total_size,
                entry_price=avg_price,
                entry_time=existing_pos.entry_time,
                stop_loss=signal.stop_loss or existing_pos.stop_loss,
                take_profit=signal.take_profit or existing_pos.take_profit,
                highest_price=max(existing_pos.highest_price, execution_price),
                lowest_price=min(existing_pos.lowest_price, execution_price),
                entry_fee=total_entry_fee
            )
        else:
            new_pos = Position(
                symbol=signal.symbol,
                side='long',
                size=base_size,
                entry_price=execution_price,
                entry_time=timestamp,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                highest_price=execution_price,
                lowest_price=execution_price,
                entry_fee=fee
            )

        portfolio.positions[signal.symbol] = new_pos

        return Fill(
            fill_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            symbol=signal.symbol,
            side='buy',
            size=base_size,
            price=execution_price,
            fee=fee,
            signal_reason=signal.reason,
            pnl=0.0,
            strategy=portfolio.strategy_name
        )

    def _execute_sell(
        self,
        portfolio: StrategyPortfolio,
        signal: Signal,
        base_asset: str,
        base_size: float,
        execution_price: float,
        timestamp: datetime
    ) -> Optional[Fill]:
        """Execute a sell order."""
        available = portfolio.assets.get(base_asset, 0)
        if available <= 0:
            return None

        base_size = min(base_size, available)
        proceeds = base_size * execution_price * (1 - self.fee_rate)
        fee = base_size * execution_price * self.fee_rate

        # Calculate P&L if closing position
        pnl = 0.0
        if signal.symbol in portfolio.positions:
            pos = portfolio.positions[signal.symbol]
            if pos.side == 'long':
                # Include proportional entry fee in P&L calculation
                proportional_entry_fee = pos.entry_fee * (base_size / pos.size) if pos.size > 0 else 0
                pnl = (execution_price - pos.entry_price) * base_size - fee - proportional_entry_fee
                portfolio.total_pnl += pnl

                if pnl > 0:
                    portfolio.winning_trades += 1
                else:
                    portfolio.losing_trades += 1

                # Remove or reduce position
                if base_size >= pos.size:
                    del portfolio.positions[signal.symbol]
                else:
                    remaining_size = pos.size - base_size
                    remaining_entry_fee = pos.entry_fee * (remaining_size / pos.size) if pos.size > 0 else 0
                    portfolio.positions[signal.symbol] = Position(
                        symbol=pos.symbol,
                        side=pos.side,
                        size=remaining_size,
                        entry_price=pos.entry_price,
                        entry_time=pos.entry_time,
                        stop_loss=pos.stop_loss,
                        take_profit=pos.take_profit,
                        highest_price=pos.highest_price,
                        lowest_price=pos.lowest_price,
                        entry_fee=remaining_entry_fee
                    )

        portfolio.assets[base_asset] -= base_size
        if portfolio.assets[base_asset] < 1e-10:
            portfolio.assets[base_asset] = 0
        portfolio.usdt += proceeds

        return Fill(
            fill_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            symbol=signal.symbol,
            side='sell',
            size=base_size,
            price=execution_price,
            fee=fee,
            signal_reason=signal.reason,
            pnl=pnl,
            strategy=portfolio.strategy_name
        )

    def _execute_short(
        self,
        portfolio: StrategyPortfolio,
        signal: Signal,
        base_asset: str,
        base_size: float,
        execution_price: float,
        timestamp: datetime
    ) -> Optional[Fill]:
        """Execute a short sell order (borrow and sell)."""
        # MED-006: Configurable short selling leverage limit
        # For paper trading, we allow shorting up to portfolio equity * leverage
        equity = portfolio.usdt
        max_short_value = equity * self.max_short_leverage

        short_value = base_size * execution_price
        if short_value > max_short_value:
            base_size = max_short_value / execution_price

        fee = base_size * execution_price * self.fee_rate

        # Credit USDT for short sale
        proceeds = base_size * execution_price * (1 - self.fee_rate)
        portfolio.usdt += proceeds

        # Track negative position
        portfolio.assets[base_asset] = portfolio.assets.get(base_asset, 0) - base_size

        # Track short position
        existing_pos = portfolio.positions.get(signal.symbol)
        if existing_pos and existing_pos.side == 'short':
            total_size = existing_pos.size + base_size
            avg_price = (
                (existing_pos.entry_price * existing_pos.size) +
                (execution_price * base_size)
            ) / total_size
            total_entry_fee = existing_pos.entry_fee + fee

            new_pos = Position(
                symbol=signal.symbol,
                side='short',
                size=total_size,
                entry_price=avg_price,
                entry_time=existing_pos.entry_time,
                stop_loss=signal.stop_loss or existing_pos.stop_loss,
                take_profit=signal.take_profit or existing_pos.take_profit,
                highest_price=max(existing_pos.highest_price, execution_price),
                lowest_price=min(existing_pos.lowest_price, execution_price),
                entry_fee=total_entry_fee
            )
        else:
            new_pos = Position(
                symbol=signal.symbol,
                side='short',
                size=base_size,
                entry_price=execution_price,
                entry_time=timestamp,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                highest_price=execution_price,
                lowest_price=execution_price,
                entry_fee=fee
            )

        portfolio.positions[signal.symbol] = new_pos

        return Fill(
            fill_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            symbol=signal.symbol,
            side='short',
            size=base_size,
            price=execution_price,
            fee=fee,
            signal_reason=signal.reason,
            pnl=0.0,
            strategy=portfolio.strategy_name
        )

    def _execute_cover(
        self,
        portfolio: StrategyPortfolio,
        signal: Signal,
        base_asset: str,
        base_size: float,
        execution_price: float,
        timestamp: datetime
    ) -> Optional[Fill]:
        """Execute a cover order (buy to close short)."""
        # Check if we have a short position
        if signal.symbol not in portfolio.positions:
            return None

        pos = portfolio.positions[signal.symbol]
        if pos.side != 'short':
            return None

        base_size = min(base_size, pos.size)
        cost = base_size * execution_price * (1 + self.fee_rate)
        fee = base_size * execution_price * self.fee_rate

        # Check if we have enough USDT to cover
        if portfolio.usdt < cost:
            base_size = (portfolio.usdt / (1 + self.fee_rate)) / execution_price
            cost = portfolio.usdt
            if base_size <= 0:
                return None

        # Calculate P&L (profit if price went down) including proportional entry fee
        proportional_entry_fee = pos.entry_fee * (base_size / pos.size) if pos.size > 0 else 0
        pnl = (pos.entry_price - execution_price) * base_size - fee - proportional_entry_fee
        portfolio.total_pnl += pnl

        if pnl > 0:
            portfolio.winning_trades += 1
        else:
            portfolio.losing_trades += 1

        portfolio.usdt -= cost
        portfolio.assets[base_asset] = portfolio.assets.get(base_asset, 0) + base_size

        # Remove or reduce position
        if base_size >= pos.size:
            del portfolio.positions[signal.symbol]
        else:
            remaining_size = pos.size - base_size
            remaining_entry_fee = pos.entry_fee * (remaining_size / pos.size) if pos.size > 0 else 0
            portfolio.positions[signal.symbol] = Position(
                symbol=pos.symbol,
                side=pos.side,
                size=remaining_size,
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                highest_price=pos.highest_price,
                lowest_price=pos.lowest_price,
                entry_fee=remaining_entry_fee
            )

        return Fill(
            fill_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            symbol=signal.symbol,
            side='cover',
            size=base_size,
            price=execution_price,
            fee=fee,
            signal_reason=signal.reason,
            pnl=pnl,
            strategy=portfolio.strategy_name
        )

    def check_stops(self, data: DataSnapshot) -> List[Tuple[str, Signal]]:
        """Check all strategy positions for stop-loss / take-profit."""
        signals = []

        for strategy, portfolio in self.portfolio_manager.portfolios.items():
            for symbol, pos in list(portfolio.positions.items()):
                price = data.prices.get(symbol, 0)
                if not price:
                    continue

                trigger = None
                trigger_action = None

                if pos.side == 'long':
                    if pos.stop_loss and price <= pos.stop_loss:
                        trigger = 'stop_loss'
                        trigger_action = 'sell'
                    elif pos.take_profit and price >= pos.take_profit:
                        trigger = 'take_profit'
                        trigger_action = 'sell'

                elif pos.side == 'short':
                    if pos.stop_loss and price >= pos.stop_loss:
                        trigger = 'stop_loss'
                        trigger_action = 'cover'
                    elif pos.take_profit and price <= pos.take_profit:
                        trigger = 'take_profit'
                        trigger_action = 'cover'

                if trigger and trigger_action:
                    signals.append((strategy, Signal(
                        action=trigger_action,
                        symbol=symbol,
                        size=pos.size * price,  # Close full position
                        price=price,
                        reason=f"{trigger} triggered at {price:.6f}",
                        metadata={'trigger': trigger, 'position_side': pos.side}
                    )))

        return signals

    def update_position_tracking(self, data: DataSnapshot):
        """Update highest/lowest prices for trailing stops."""
        for portfolio in self.portfolio_manager.portfolios.values():
            for symbol, pos in list(portfolio.positions.items()):
                price = data.prices.get(symbol, 0)
                if not price:
                    continue

                # Create new position with updated tracking
                new_pos = Position(
                    symbol=pos.symbol,
                    side=pos.side,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    entry_time=pos.entry_time,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    highest_price=max(pos.highest_price, price),
                    lowest_price=min(pos.lowest_price, price),
                    entry_fee=pos.entry_fee
                )
                portfolio.positions[symbol] = new_pos
