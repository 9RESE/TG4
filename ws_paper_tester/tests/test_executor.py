"""Tests for paper executor."""

import pytest
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.types import DataSnapshot, Signal, OrderbookSnapshot
from ws_tester.portfolio import PortfolioManager, STARTING_CAPITAL
from ws_tester.executor import PaperExecutor


def create_test_snapshot(price: float = 2.35) -> DataSnapshot:
    """Create a test data snapshot."""
    ob = OrderbookSnapshot(
        bids=((price - 0.001, 100.0), (price - 0.002, 200.0)),
        asks=((price + 0.001, 100.0), (price + 0.002, 200.0))
    )
    return DataSnapshot(
        timestamp=datetime.now(),
        prices={'XRP/USD': price},
        candles_1m={},
        candles_5m={},
        orderbooks={'XRP/USD': ob},
        trades={}
    )


class TestPaperExecutor:
    def test_executor_creation(self):
        pm = PortfolioManager(['test_strategy'])
        executor = PaperExecutor(pm)
        assert executor is not None

    def test_buy_execution(self):
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm)
        snapshot = create_test_snapshot(2.35)

        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=50.0,  # $50 USD
            price=2.35,
            reason='Test buy'
        )

        fill = executor.execute(signal, 'test', snapshot)

        assert fill is not None
        assert fill.side == 'buy'
        assert fill.symbol == 'XRP/USD'

        # Check portfolio updated
        portfolio = pm.get_portfolio('test')
        assert portfolio.usdt < STARTING_CAPITAL
        assert portfolio.assets.get('XRP', 0) > 0

    def test_sell_execution(self):
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm)
        snapshot = create_test_snapshot(2.35)

        # First buy
        buy_signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=50.0,
            price=2.35,
            reason='Test buy'
        )
        executor.execute(buy_signal, 'test', snapshot)

        # Then sell
        sell_signal = Signal(
            action='sell',
            symbol='XRP/USD',
            size=50.0,
            price=2.35,
            reason='Test sell'
        )
        fill = executor.execute(sell_signal, 'test', snapshot)

        assert fill is not None
        assert fill.side == 'sell'

    def test_insufficient_balance(self):
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm)
        snapshot = create_test_snapshot(2.35)

        # Try to buy more than available
        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=200.0,  # More than $100 starting capital
            price=2.35,
            reason='Test big buy'
        )

        fill = executor.execute(signal, 'test', snapshot)

        # Should execute with reduced size
        assert fill is not None
        portfolio = pm.get_portfolio('test')
        assert portfolio.usdt < 1.0  # Almost depleted

    def test_cannot_sell_without_position(self):
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm)
        snapshot = create_test_snapshot(2.35)

        signal = Signal(
            action='sell',
            symbol='XRP/USD',
            size=50.0,
            price=2.35,
            reason='Test sell without position'
        )

        fill = executor.execute(signal, 'test', snapshot)
        assert fill is None

    def test_position_tracking(self):
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm)
        snapshot = create_test_snapshot(2.35)

        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=50.0,
            price=2.35,
            reason='Test buy',
            stop_loss=2.30,
            take_profit=2.50
        )

        executor.execute(signal, 'test', snapshot)

        portfolio = pm.get_portfolio('test')
        assert 'XRP/USD' in portfolio.positions
        position = portfolio.positions['XRP/USD']
        assert position.side == 'long'
        assert position.stop_loss == 2.30
        assert position.take_profit == 2.50

    def test_stop_loss_trigger(self):
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm)
        snapshot = create_test_snapshot(2.35)

        # Buy with stop loss
        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=50.0,
            price=2.35,
            reason='Test buy',
            stop_loss=2.30
        )
        executor.execute(signal, 'test', snapshot)

        # Price drops below stop loss
        snapshot_stopped = create_test_snapshot(2.29)
        stop_signals = executor.check_stops(snapshot_stopped)

        assert len(stop_signals) == 1
        strategy_name, stop_signal = stop_signals[0]
        assert strategy_name == 'test'
        assert stop_signal.action == 'sell'
        assert 'stop_loss' in stop_signal.reason

    def test_take_profit_trigger(self):
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm)
        snapshot = create_test_snapshot(2.35)

        # Buy with take profit
        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=50.0,
            price=2.35,
            reason='Test buy',
            take_profit=2.40
        )
        executor.execute(signal, 'test', snapshot)

        # Price rises above take profit
        snapshot_profit = create_test_snapshot(2.41)
        stop_signals = executor.check_stops(snapshot_profit)

        assert len(stop_signals) == 1
        strategy_name, tp_signal = stop_signals[0]
        assert strategy_name == 'test'
        assert tp_signal.action == 'sell'
        assert 'take_profit' in tp_signal.reason

    def test_pnl_calculation(self):
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm)
        snapshot_entry = create_test_snapshot(2.35)
        snapshot_exit = create_test_snapshot(2.40)

        # Buy
        buy_signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=50.0,
            price=2.35,
            reason='Test buy'
        )
        executor.execute(buy_signal, 'test', snapshot_entry)

        # Sell at higher price
        sell_signal = Signal(
            action='sell',
            symbol='XRP/USD',
            size=50.0,
            price=2.40,
            reason='Test sell'
        )
        fill = executor.execute(sell_signal, 'test', snapshot_exit)

        assert fill is not None
        # Should have positive P&L
        portfolio = pm.get_portfolio('test')
        assert portfolio.total_pnl > 0


class TestMultipleStrategies:
    def test_isolated_portfolios(self):
        pm = PortfolioManager(['strategy_a', 'strategy_b'])
        executor = PaperExecutor(pm)
        snapshot = create_test_snapshot(2.35)

        # Strategy A buys
        signal_a = Signal(
            action='buy',
            symbol='XRP/USD',
            size=50.0,
            price=2.35,
            reason='Strategy A buy'
        )
        executor.execute(signal_a, 'strategy_a', snapshot)

        # Check portfolios are isolated
        portfolio_a = pm.get_portfolio('strategy_a')
        portfolio_b = pm.get_portfolio('strategy_b')

        assert portfolio_a.usdt < STARTING_CAPITAL
        assert portfolio_b.usdt == STARTING_CAPITAL  # Unchanged
        assert portfolio_a.assets.get('XRP', 0) > 0
        assert portfolio_b.assets.get('XRP', 0) == 0
