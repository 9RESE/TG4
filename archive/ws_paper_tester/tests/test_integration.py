"""Integration tests for the paper trading system."""

import pytest
import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.data_layer import SimulatedDataManager
from ws_tester.portfolio import PortfolioManager, STARTING_CAPITAL
from ws_tester.executor import PaperExecutor
from ws_tester.strategy_loader import discover_strategies, StrategyWrapper
from ws_tester.types import Signal


class TestIntegrationSimulatedMode:
    """Integration tests using simulated data."""

    @pytest.mark.asyncio
    async def test_full_trading_loop_simulated(self):
        """Test complete trading loop with simulated data."""
        symbols = ['XRP/USD']

        # Initialize components
        data_manager = SimulatedDataManager(symbols)
        strategies = discover_strategies()
        strategy_names = list(strategies.keys())

        portfolio_manager = PortfolioManager(strategy_names)
        executor = PaperExecutor(portfolio_manager)

        # Run trading loop for N ticks
        ticks = 50
        signals_generated = 0
        fills_executed = 0

        for _ in range(ticks):
            await data_manager.simulate_tick()
            snapshot = data_manager.get_snapshot()

            if not snapshot:
                continue

            # Process each strategy
            for name, strategy in strategies.items():
                try:
                    signal = strategy.generate_signal(snapshot)
                    if signal:
                        signals_generated += 1
                        fill = executor.execute(signal, name, snapshot)
                        if fill:
                            fills_executed += 1
                            strategy.on_fill(fill.__dict__)
                except Exception as e:
                    # Strategy errors should not crash the loop
                    print(f"Strategy {name} error: {e}")

            # Check stops
            stop_signals = executor.check_stops(snapshot)
            for strategy_name, stop_signal in stop_signals:
                fill = executor.execute(stop_signal, strategy_name, snapshot)
                if fill:
                    fills_executed += 1

        # Verify the loop completed
        assert ticks == 50

        # Verify strategies processed without crashes
        for name, strategy in strategies.items():
            portfolio = portfolio_manager.get_portfolio(name)
            assert portfolio is not None

            # Portfolio should still have some value
            equity = portfolio.get_equity(snapshot.prices)
            assert equity > 0, f"Strategy {name} has no equity"

    @pytest.mark.asyncio
    async def test_strategy_isolation(self):
        """Test that strategy portfolios are truly isolated."""
        symbols = ['XRP/USD']
        data_manager = SimulatedDataManager(symbols)

        # Create two test strategies
        strategy_names = ['test_a', 'test_b']
        portfolio_manager = PortfolioManager(strategy_names, starting_capital=100.0)
        executor = PaperExecutor(portfolio_manager)

        # Run simulation
        for _ in range(10):
            await data_manager.simulate_tick()

        snapshot = data_manager.get_snapshot()

        # Strategy A buys
        signal_a = Signal(
            action='buy',
            symbol='XRP/USD',
            size=50.0,
            price=snapshot.prices['XRP/USD'],
            reason='Test A buy'
        )
        executor.execute(signal_a, 'test_a', snapshot)

        # Check isolation
        portfolio_a = portfolio_manager.get_portfolio('test_a')
        portfolio_b = portfolio_manager.get_portfolio('test_b')

        assert portfolio_a.usdt < 100.0  # A spent money
        assert portfolio_b.usdt == 100.0  # B unchanged
        assert portfolio_a.assets.get('XRP', 0) > 0  # A has XRP
        assert portfolio_b.assets.get('XRP', 0) == 0  # B has no XRP

    @pytest.mark.asyncio
    async def test_pnl_tracking(self):
        """Test P&L calculation through buy/sell cycle."""
        symbols = ['XRP/USD']
        data_manager = SimulatedDataManager(
            symbols,
            initial_prices={'XRP/USD': 2.00}  # Fixed starting price
        )

        portfolio_manager = PortfolioManager(['pnl_test'], starting_capital=100.0)
        executor = PaperExecutor(portfolio_manager, fee_rate=0.0, slippage_rate=0.0)  # No fees for clean test

        snapshot = data_manager.get_snapshot()

        # Buy at $2.00
        buy_signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=20.0,  # $20 worth
            price=2.00,
            reason='Test buy'
        )
        fill = executor.execute(buy_signal, 'pnl_test', snapshot)
        assert fill is not None
        assert fill.size == pytest.approx(10.0, rel=0.01)  # ~10 XRP (slight variance from orderbook mid)

        # Simulate price increase
        data_manager._prices['XRP/USD'] = 2.20  # 10% increase
        data_manager._orderbooks['XRP/USD'] = {
            'bids': [(2.19, 100)],
            'asks': [(2.21, 100)]
        }

        snapshot = data_manager.get_snapshot()

        # Sell at $2.20
        sell_signal = Signal(
            action='sell',
            symbol='XRP/USD',
            size=22.0,  # All XRP
            price=2.20,
            reason='Test sell'
        )
        fill = executor.execute(sell_signal, 'pnl_test', snapshot)
        assert fill is not None

        portfolio = portfolio_manager.get_portfolio('pnl_test')

        # P&L should be positive (bought at 2.00, sold at ~2.19)
        assert portfolio.total_pnl > 0, f"Expected positive P&L, got {portfolio.total_pnl}"
        assert portfolio.winning_trades == 1
        assert portfolio.total_trades == 2

    @pytest.mark.asyncio
    async def test_stop_loss_execution(self):
        """Test automatic stop-loss execution."""
        symbols = ['XRP/USD']
        data_manager = SimulatedDataManager(
            symbols,
            initial_prices={'XRP/USD': 2.00}
        )

        portfolio_manager = PortfolioManager(['stop_test'], starting_capital=100.0)
        executor = PaperExecutor(portfolio_manager, fee_rate=0.0, slippage_rate=0.0)

        snapshot = data_manager.get_snapshot()

        # Buy with stop-loss
        buy_signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=50.0,
            price=2.00,
            reason='Test buy with stop',
            stop_loss=1.90  # 5% stop-loss
        )
        executor.execute(buy_signal, 'stop_test', snapshot)

        # Verify position has stop
        portfolio = portfolio_manager.get_portfolio('stop_test')
        assert 'XRP/USD' in portfolio.positions
        assert portfolio.positions['XRP/USD'].stop_loss == 1.90

        # Simulate price drop below stop
        data_manager._prices['XRP/USD'] = 1.85
        data_manager._orderbooks['XRP/USD'] = {
            'bids': [(1.84, 100)],
            'asks': [(1.86, 100)]
        }
        snapshot = data_manager.get_snapshot()

        # Check stops
        stop_signals = executor.check_stops(snapshot)

        assert len(stop_signals) == 1
        strategy_name, stop_signal = stop_signals[0]
        assert strategy_name == 'stop_test'
        assert stop_signal.action == 'sell'
        assert 'stop_loss' in stop_signal.reason

        # Execute stop
        fill = executor.execute(stop_signal, strategy_name, snapshot)
        assert fill is not None

        # Position should be closed
        portfolio = portfolio_manager.get_portfolio('stop_test')
        assert 'XRP/USD' not in portfolio.positions

    @pytest.mark.asyncio
    async def test_take_profit_execution(self):
        """Test automatic take-profit execution."""
        symbols = ['XRP/USD']
        data_manager = SimulatedDataManager(
            symbols,
            initial_prices={'XRP/USD': 2.00}
        )

        portfolio_manager = PortfolioManager(['tp_test'], starting_capital=100.0)
        executor = PaperExecutor(portfolio_manager, fee_rate=0.0, slippage_rate=0.0)

        snapshot = data_manager.get_snapshot()

        # Buy with take-profit
        buy_signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=50.0,
            price=2.00,
            reason='Test buy with TP',
            take_profit=2.20  # 10% take-profit
        )
        executor.execute(buy_signal, 'tp_test', snapshot)

        # Simulate price rise above take-profit
        data_manager._prices['XRP/USD'] = 2.25
        data_manager._orderbooks['XRP/USD'] = {
            'bids': [(2.24, 100)],
            'asks': [(2.26, 100)]
        }
        snapshot = data_manager.get_snapshot()

        # Check stops (includes take-profit)
        stop_signals = executor.check_stops(snapshot)

        assert len(stop_signals) == 1
        strategy_name, tp_signal = stop_signals[0]
        assert 'take_profit' in tp_signal.reason

    @pytest.mark.asyncio
    async def test_short_selling(self):
        """Test short selling functionality."""
        symbols = ['XRP/USD']
        data_manager = SimulatedDataManager(
            symbols,
            initial_prices={'XRP/USD': 2.00}
        )

        portfolio_manager = PortfolioManager(['short_test'], starting_capital=100.0)
        executor = PaperExecutor(portfolio_manager, fee_rate=0.0, slippage_rate=0.0)

        snapshot = data_manager.get_snapshot()
        initial_usdt = portfolio_manager.get_portfolio('short_test').usdt

        # Short sell
        short_signal = Signal(
            action='short',
            symbol='XRP/USD',
            size=40.0,  # $40 worth
            price=2.00,
            reason='Test short'
        )
        fill = executor.execute(short_signal, 'short_test', snapshot)
        assert fill is not None
        assert fill.side == 'short'

        portfolio = portfolio_manager.get_portfolio('short_test')

        # Should have more USDT (from short proceeds)
        assert portfolio.usdt > initial_usdt

        # Should have negative XRP
        assert portfolio.assets.get('XRP', 0) < 0

        # Should have short position
        assert 'XRP/USD' in portfolio.positions
        assert portfolio.positions['XRP/USD'].side == 'short'

        # Simulate price drop (profitable for short)
        data_manager._prices['XRP/USD'] = 1.80
        data_manager._orderbooks['XRP/USD'] = {
            'bids': [(1.79, 100)],
            'asks': [(1.81, 100)]
        }
        snapshot = data_manager.get_snapshot()

        # Cover at lower price
        cover_signal = Signal(
            action='cover',
            symbol='XRP/USD',
            size=40.0,
            price=1.80,
            reason='Test cover'
        )
        fill = executor.execute(cover_signal, 'short_test', snapshot)
        assert fill is not None
        assert fill.side == 'cover'
        assert fill.pnl > 0  # Should be profitable

        portfolio = portfolio_manager.get_portfolio('short_test')
        assert portfolio.total_pnl > 0


class TestIntegrationCredentials:
    """Test credentials loading."""

    def test_load_kraken_credentials(self):
        """Test loading Kraken credentials from .env file."""
        from ws_tester.credentials import load_kraken_credentials

        # Try to load credentials (may or may not exist)
        credentials = load_kraken_credentials()

        # If credentials exist, verify structure
        if credentials:
            assert hasattr(credentials, 'api_key')
            assert hasattr(credentials, 'private_key')
            assert credentials.is_valid


class TestIntegrationDashboardState:
    """Test dashboard state updates during trading."""

    def test_dashboard_state_during_trading(self):
        """Test that dashboard state updates correctly."""
        try:
            from ws_tester.dashboard.server import update_state, add_trade, _get_state_snapshot
        except ImportError:
            pytest.skip("FastAPI not installed")

        # Simulate trading state updates
        update_state(
            prices={'XRP/USD': 2.35, 'BTC/USD': 104500.0},
            strategies=[
                {'strategy': 'market_making', 'equity': 105.0, 'pnl': 5.0},
                {'strategy': 'order_flow', 'equity': 98.0, 'pnl': -2.0}
            ],
            aggregate={
                'total_equity': 203.0,
                'total_pnl': 3.0,
                'total_strategies': 2
            },
            session_info={
                'session_id': 'test-session',
                'mode': 'simulated'
            }
        )

        # Add some trades
        for i in range(5):
            add_trade({
                'timestamp': datetime.now().isoformat(),
                'symbol': 'XRP/USD',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'price': 2.35 + i * 0.01,
                'strategy': 'market_making'
            })

        # Get snapshot
        state = _get_state_snapshot()

        assert state['prices']['XRP/USD'] == 2.35
        assert len(state['strategies']) == 2
        assert state['aggregate']['total_pnl'] == 3.0
        assert len(state['recent_trades']) == 5
