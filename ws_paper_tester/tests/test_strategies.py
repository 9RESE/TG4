"""Tests for strategy loader and strategies."""

import pytest
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.types import DataSnapshot, OrderbookSnapshot, Candle, Trade
from ws_tester.strategy_loader import discover_strategies, StrategyWrapper, get_all_symbols


def create_rich_snapshot(price: float = 2.35) -> DataSnapshot:
    """Create a test snapshot with candles, trades, and orderbook."""
    now = datetime.now()

    # Create candles
    candles = []
    for i in range(25):
        offset = (25 - i) * 0.001
        candles.append(Candle(
            timestamp=now,
            open=price - offset,
            high=price - offset + 0.005,
            low=price - offset - 0.005,
            close=price - offset + 0.002,
            volume=100.0
        ))

    # Create trades
    trades = []
    for i in range(60):
        trades.append(Trade(
            timestamp=now,
            price=price + (i % 2 - 0.5) * 0.002,
            size=50.0 + i,
            side='buy' if i % 3 != 0 else 'sell'
        ))

    # Create orderbook
    ob = OrderbookSnapshot(
        bids=tuple((price - 0.001 - i*0.001, 100.0 + i*10) for i in range(10)),
        asks=tuple((price + 0.001 + i*0.001, 100.0 + i*10) for i in range(10))
    )

    return DataSnapshot(
        timestamp=now,
        prices={'XRP/USDT': price, 'BTC/USDT': 104500.0},
        candles_1m={'XRP/USDT': tuple(candles)},
        candles_5m={'XRP/USDT': tuple(candles)},
        orderbooks={'XRP/USDT': ob, 'BTC/USDT': ob},
        trades={'XRP/USDT': tuple(trades), 'BTC/USDT': tuple(trades)}
    )


class TestStrategyLoader:
    def test_discover_strategies(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))

        assert len(strategies) >= 3  # market_making, order_flow, mean_reversion
        assert 'market_making' in strategies
        assert 'order_flow' in strategies
        assert 'mean_reversion' in strategies

    def test_strategy_wrapper(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))

        mm = strategies.get('market_making')
        assert mm is not None
        assert isinstance(mm, StrategyWrapper)
        assert mm.name == 'market_making'
        assert mm.version.startswith('1.')  # Allow minor/patch version bumps
        assert len(mm.symbols) > 0

    def test_get_all_symbols(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        symbols = get_all_symbols(strategies)

        assert 'XRP/USDT' in symbols
        assert len(symbols) >= 1


class TestMarketMakingStrategy:
    def test_generate_signal(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mm = strategies.get('market_making')

        assert mm is not None

        snapshot = create_rich_snapshot(2.35)
        mm.on_start()

        # May or may not generate signal depending on conditions
        signal = mm.generate_signal(snapshot)

        # Check state was updated
        assert 'indicators' in mm.state

    def test_on_fill_updates_inventory(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mm = strategies.get('market_making')

        mm.on_start()
        assert mm.state.get('inventory', 0) == 0

        # Simulate a fill
        fill = {
            'side': 'buy',
            'size': 42.5,
            'price': 2.35
        }
        mm.on_fill(fill)

        # Inventory should increase
        assert mm.state.get('inventory', 0) > 0


class TestOrderFlowStrategy:
    def test_generate_signal(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        of = strategies.get('order_flow')

        assert of is not None

        snapshot = create_rich_snapshot(2.35)
        of.on_start()

        signal = of.generate_signal(snapshot)

        # Check state was updated
        assert 'indicators' in of.state or of.state.get('total_trades_seen', 0) > 0


class TestMeanReversionStrategy:
    def test_generate_signal(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mr = strategies.get('mean_reversion')

        assert mr is not None

        snapshot = create_rich_snapshot(2.35)
        mr.on_start()

        signal = mr.generate_signal(snapshot)

        # Check indicators were calculated
        assert 'indicators' in mr.state
        indicators = mr.state['indicators']
        assert 'sma' in indicators
        assert 'rsi' in indicators


class TestMarketMakingV14Features:
    """Tests for v1.4.0 enhancements."""

    def test_config_validation(self):
        """Test that config validation catches invalid parameters."""
        from strategies.market_making import _validate_config

        # Valid config
        valid_config = {
            'position_size_usd': 20,
            'max_inventory': 100,
            'stop_loss_pct': 0.5,
            'take_profit_pct': 0.4,
            'cooldown_seconds': 5.0,
            'gamma': 0.1,
            'inventory_skew': 0.5,
        }
        errors = _validate_config(valid_config)
        assert len([e for e in errors if 'Warning' not in e]) == 0

        # Invalid gamma
        invalid_gamma = valid_config.copy()
        invalid_gamma['gamma'] = 5.0  # Out of bounds
        errors = _validate_config(invalid_gamma)
        assert any('gamma' in e for e in errors)

        # Missing required value
        missing_config = {'gamma': 0.1}  # Missing required fields
        errors = _validate_config(missing_config)
        assert any('Missing' in e or 'must be positive' in e for e in errors)

    def test_reservation_price_calculation(self):
        """Test Avellaneda-Stoikov reservation price model."""
        from strategies.market_making import _calculate_reservation_price

        mid_price = 100.0
        max_inv = 100.0
        gamma = 0.1
        volatility = 0.5

        # No inventory - should return mid price
        result = _calculate_reservation_price(mid_price, 0, max_inv, gamma, volatility)
        assert result == mid_price

        # Positive inventory (long) - should lower reservation price
        result_long = _calculate_reservation_price(mid_price, 50, max_inv, gamma, volatility)
        assert result_long < mid_price

        # Negative inventory (short) - should raise reservation price
        result_short = _calculate_reservation_price(mid_price, -50, max_inv, gamma, volatility)
        assert result_short > mid_price

    def test_trailing_stop_calculation(self):
        """Test trailing stop calculation."""
        from strategies.market_making import _calculate_trailing_stop

        entry_price = 100.0
        activation_pct = 0.2  # Activate at 0.2% profit
        trail_distance = 0.15  # Trail at 0.15%

        # Not yet activated (high price not high enough)
        result = _calculate_trailing_stop(entry_price, 100.1, 'long', activation_pct, trail_distance)
        assert result is None

        # Activated (high price 0.5% above entry)
        highest = 100.5  # 0.5% above entry
        result = _calculate_trailing_stop(entry_price, highest, 'long', activation_pct, trail_distance)
        assert result is not None
        assert result < highest
        assert result > entry_price  # Should still be above entry

        # Short position - activated
        lowest = 99.5  # 0.5% below entry
        result_short = _calculate_trailing_stop(entry_price, lowest, 'short', activation_pct, trail_distance)
        assert result_short is not None
        assert result_short > lowest
        assert result_short < entry_price

    def test_on_fill_tracks_per_pair_metrics(self):
        """Test that on_fill tracks per-pair PnL and trade counts."""
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mm = strategies.get('market_making')

        mm.on_start()

        # Initial state should have empty per-pair metrics
        assert mm.state.get('pnl_by_symbol', {}) == {}
        assert mm.state.get('trades_by_symbol', {}) == {}

        # Simulate fills with PnL
        fill1 = {
            'symbol': 'XRP/USDT',
            'side': 'buy',
            'size': 42.5,
            'price': 2.35,
            'pnl': 0,  # Entry has no PnL
        }
        mm.on_fill(fill1)

        # Closing fill with PnL
        fill2 = {
            'symbol': 'XRP/USDT',
            'side': 'sell',
            'size': 42.5,
            'price': 2.40,
            'pnl': 2.125,  # Profit
        }
        mm.on_fill(fill2)

        # Check per-pair metrics
        assert mm.state['trades_by_symbol'].get('XRP/USDT', 0) == 2
        assert mm.state['pnl_by_symbol'].get('XRP/USDT', 0) == 2.125

    def test_position_entry_tracking_for_trailing_stops(self):
        """Test position entry tracking for trailing stops."""
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mm = strategies.get('market_making')

        mm.on_start()

        # Simulate buy fill
        fill = {
            'symbol': 'XRP/USDT',
            'side': 'buy',
            'size': 100.0,
            'price': 2.35,
            'pnl': 0,
        }
        mm.on_fill(fill)

        # Check position entry was tracked
        assert 'position_entries' in mm.state
        assert 'XRP/USDT' in mm.state['position_entries']
        pos = mm.state['position_entries']['XRP/USDT']
        assert pos['entry_price'] == 2.35
        assert pos['side'] == 'long'

        # Simulate sell to close
        close_fill = {
            'symbol': 'XRP/USDT',
            'side': 'sell',
            'size': 100.0,
            'price': 2.40,
            'pnl': 5.0,
        }
        mm.on_fill(close_fill)

        # Position should be cleared
        assert 'XRP/USDT' not in mm.state['position_entries']


class TestStrategyValidation:
    def test_all_strategies_have_required_attrs(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))

        for name, strategy in strategies.items():
            assert strategy.name, f"{name} missing name"
            assert strategy.version, f"{name} missing version"
            assert callable(strategy._generate_signal), f"{name} generate_signal not callable"
            assert len(strategy.symbols) > 0, f"{name} has no symbols"

    def test_strategies_dont_crash_on_empty_data(self):
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))

        # Empty snapshot
        snapshot = DataSnapshot(
            timestamp=datetime.now(),
            prices={},
            candles_1m={},
            candles_5m={},
            orderbooks={},
            trades={}
        )

        for name, strategy in strategies.items():
            strategy.on_start()
            try:
                signal = strategy.generate_signal(snapshot)
                # Should return None, not crash
                assert signal is None or hasattr(signal, 'action')
            except Exception as e:
                pytest.fail(f"{name} crashed on empty data: {e}")


class TestPortfolioPerPairTracking:
    """Tests for portfolio per-pair PnL tracking."""

    def test_record_trade_result(self):
        """Test per-pair trade result recording."""
        from ws_tester.portfolio import StrategyPortfolio

        portfolio = StrategyPortfolio(strategy_name="test")

        # Record winning trade
        portfolio.record_trade_result('XRP/USDT', 5.0)
        assert portfolio.pnl_by_symbol.get('XRP/USDT') == 5.0
        assert portfolio.trades_by_symbol.get('XRP/USDT') == 1
        assert portfolio.wins_by_symbol.get('XRP/USDT') == 1

        # Record losing trade
        portfolio.record_trade_result('XRP/USDT', -2.0)
        assert portfolio.pnl_by_symbol.get('XRP/USDT') == 3.0
        assert portfolio.trades_by_symbol.get('XRP/USDT') == 2
        assert portfolio.losses_by_symbol.get('XRP/USDT') == 1

        # Record trade on different symbol
        portfolio.record_trade_result('BTC/USDT', 10.0)
        assert portfolio.pnl_by_symbol.get('BTC/USDT') == 10.0
        assert portfolio.trades_by_symbol.get('BTC/USDT') == 1

    def test_get_symbol_stats(self):
        """Test per-symbol statistics."""
        from ws_tester.portfolio import StrategyPortfolio

        portfolio = StrategyPortfolio(strategy_name="test")

        # Record some trades
        portfolio.record_trade_result('XRP/USDT', 5.0)
        portfolio.record_trade_result('XRP/USDT', 3.0)
        portfolio.record_trade_result('XRP/USDT', -2.0)

        stats = portfolio.get_symbol_stats('XRP/USDT')

        assert stats['symbol'] == 'XRP/USDT'
        assert stats['trades'] == 3
        assert stats['wins'] == 2
        assert stats['losses'] == 1
        assert stats['pnl'] == 6.0  # 5 + 3 - 2
        assert stats['win_rate'] == pytest.approx(66.7, rel=0.01)

    def test_to_dict_includes_per_pair(self):
        """Test that to_dict includes per-pair data."""
        from ws_tester.portfolio import StrategyPortfolio

        portfolio = StrategyPortfolio(strategy_name="test")
        portfolio.record_trade_result('XRP/USDT', 5.0)
        portfolio.record_trade_result('BTC/USDT', -2.0)

        prices = {'XRP/USDT': 2.35, 'BTC/USDT': 100000.0}
        result = portfolio.to_dict(prices)

        assert 'pnl_by_symbol' in result
        assert 'XRP/USDT' in result['pnl_by_symbol']
        assert 'BTC/USDT' in result['pnl_by_symbol']
        assert 'symbol_stats' in result
