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


class TestLeveragedPositions:
    """Tests for leveraged long and short positions."""

    def test_leveraged_long_execution(self):
        """Test that longs can exceed cash balance with leverage."""
        pm = PortfolioManager(['test'])
        # 1.5x leverage means $100 cash can buy $150 worth
        executor = PaperExecutor(pm, max_long_leverage=1.5)
        snapshot = create_test_snapshot(2.35)

        # Try to buy $120 worth (exceeds $100 cash, but within 1.5x leverage)
        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=120.0,
            price=2.35,
            reason='Leveraged buy'
        )

        fill = executor.execute(signal, 'test', snapshot)

        assert fill is not None
        portfolio = pm.get_portfolio('test')
        # USDT should be negative (borrowed)
        assert portfolio.usdt < 0
        # Should have significant XRP holdings
        assert portfolio.assets.get('XRP', 0) > 40  # ~$120 / $2.35 = ~51 XRP

    def test_leveraged_long_capped_at_max(self):
        """Test that leverage is capped at max_long_leverage."""
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm, max_long_leverage=1.5)
        snapshot = create_test_snapshot(2.35)

        # Try to buy $200 worth (exceeds 1.5x of $100)
        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=200.0,
            price=2.35,
            reason='Over-leveraged buy'
        )

        fill = executor.execute(signal, 'test', snapshot)

        assert fill is not None
        # Should be capped at ~$150 (1.5x of $100)
        portfolio = pm.get_portfolio('test')
        total_value = portfolio.assets.get('XRP', 0) * 2.35
        assert total_value < 160  # Should be around $150, accounting for fees

    def test_no_leverage_when_set_to_one(self):
        """Test that setting leverage to 1.0 disables leveraged longs."""
        pm = PortfolioManager(['test'])
        # Disable fees to test pure leverage cap
        executor = PaperExecutor(pm, max_long_leverage=1.0, fee_rate=0.0, slippage_rate=0.0)
        snapshot = create_test_snapshot(2.35)

        # Try to buy $120 worth
        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=120.0,
            price=2.35,
            reason='Should be limited to cash'
        )

        fill = executor.execute(signal, 'test', snapshot)

        assert fill is not None
        portfolio = pm.get_portfolio('test')
        # With 1x leverage and no fees, should use exactly all cash
        # Position value should be capped at ~$100 (1x of equity)
        position_value = portfolio.assets.get('XRP', 0) * 2.35
        assert position_value <= 105  # Allow small tolerance
        assert portfolio.usdt >= -1  # Near zero with no fees

    def test_margin_call_liquidation(self):
        """Test that margin call triggers liquidation."""
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm, max_long_leverage=2.0, fee_rate=0.0, slippage_rate=0.0)
        snapshot_entry = create_test_snapshot(2.35)

        # Buy with 2x leverage ($200 worth on $100 equity)
        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=180.0,  # ~1.8x leverage
            price=2.35,
            reason='Leveraged buy'
        )

        executor.execute(signal, 'test', snapshot_entry)

        portfolio = pm.get_portfolio('test')
        assert portfolio.usdt < 0  # Leveraged

        # Price drops significantly (50% - should trigger margin call)
        # With 25% maintenance margin and 1.8x leverage, a ~45% drop should liquidate
        snapshot_crash = create_test_snapshot(1.30)
        margin_signals = executor.check_stops(snapshot_crash)

        # Should have a margin call liquidation signal
        assert len(margin_signals) >= 1
        margin_signal = margin_signals[0][1]
        assert 'MARGIN CALL' in margin_signal.reason

    def test_short_leverage_unchanged(self):
        """Test that short leverage still works as before."""
        pm = PortfolioManager(['test'])
        executor = PaperExecutor(pm, max_short_leverage=2.0)
        snapshot = create_test_snapshot(2.35)

        # Short $150 worth (1.5x of $100 equity)
        signal = Signal(
            action='short',
            symbol='XRP/USD',
            size=150.0,
            price=2.35,
            reason='Leveraged short'
        )

        fill = executor.execute(signal, 'test', snapshot)

        assert fill is not None
        assert fill.side == 'short'
        portfolio = pm.get_portfolio('test')
        # Should have negative XRP (borrowed and sold)
        assert portfolio.assets.get('XRP', 0) < 0
        # Should have more USDT (from short sale proceeds)
        assert portfolio.usdt > STARTING_CAPITAL


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
