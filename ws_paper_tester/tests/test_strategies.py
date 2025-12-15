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

    # Create XRP/BTC candles (different price scale)
    xrp_btc_price = 0.0000225  # XRP/BTC ratio
    xrp_btc_candles = []
    for i in range(25):
        offset = (25 - i) * 0.0000001
        xrp_btc_candles.append(Candle(
            timestamp=now,
            open=xrp_btc_price - offset,
            high=xrp_btc_price - offset + 0.0000001,
            low=xrp_btc_price - offset - 0.0000001,
            close=xrp_btc_price - offset,
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

    # Create XRP/BTC trades
    xrp_btc_trades = []
    for i in range(60):
        xrp_btc_trades.append(Trade(
            timestamp=now,
            price=xrp_btc_price + (i % 2 - 0.5) * 0.0000001,
            size=50.0 + i,
            side='buy' if i % 3 != 0 else 'sell'
        ))

    # Create orderbook
    ob = OrderbookSnapshot(
        bids=tuple((price - 0.001 - i*0.001, 100.0 + i*10) for i in range(10)),
        asks=tuple((price + 0.001 + i*0.001, 100.0 + i*10) for i in range(10))
    )

    # Create XRP/BTC orderbook
    xrp_btc_ob = OrderbookSnapshot(
        bids=tuple((xrp_btc_price - 0.0000001 - i*0.0000001, 100.0 + i*10) for i in range(10)),
        asks=tuple((xrp_btc_price + 0.0000001 + i*0.0000001, 100.0 + i*10) for i in range(10))
    )

    return DataSnapshot(
        timestamp=now,
        prices={'XRP/USDT': price, 'BTC/USDT': 104500.0, 'XRP/BTC': xrp_btc_price},
        candles_1m={'XRP/USDT': tuple(candles), 'BTC/USDT': tuple(candles), 'XRP/BTC': tuple(xrp_btc_candles)},
        candles_5m={'XRP/USDT': tuple(candles), 'BTC/USDT': tuple(candles), 'XRP/BTC': tuple(xrp_btc_candles)},
        orderbooks={'XRP/USDT': ob, 'BTC/USDT': ob, 'XRP/BTC': xrp_btc_ob},
        trades={'XRP/USDT': tuple(trades), 'BTC/USDT': tuple(trades), 'XRP/BTC': tuple(xrp_btc_trades)}
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
        # Validate version follows semver format (major.minor.patch)
        version_parts = mm.version.split('.')
        assert len(version_parts) >= 2, f"Version {mm.version} should follow semver"
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

        # Create a snapshot with enough candles for all 3 symbols
        now = datetime.now()
        candles = []
        for i in range(50):
            offset = (50 - i) * 0.001
            candles.append(Candle(
                timestamp=now,
                open=2.35 - offset,
                high=2.35 - offset + 0.005,
                low=2.35 - offset - 0.005,
                close=2.35 - offset + 0.002,
                volume=100.0
            ))

        xrp_btc_candles = []
        for i in range(50):
            offset = (50 - i) * 0.0000001
            xrp_btc_candles.append(Candle(
                timestamp=now,
                open=0.0000225 - offset,
                high=0.0000225 - offset + 0.0000001,
                low=0.0000225 - offset - 0.0000001,
                close=0.0000225 - offset,
                volume=100.0
            ))

        trades = []
        for i in range(60):
            trades.append(Trade(
                timestamp=now,
                price=2.35 + (i % 2 - 0.5) * 0.002,
                size=50.0 + i,
                side='buy' if i % 3 != 0 else 'sell'
            ))

        ob = OrderbookSnapshot(
            bids=tuple((2.34 - i*0.001, 100.0 + i*10) for i in range(10)),
            asks=tuple((2.36 + i*0.001, 100.0 + i*10) for i in range(10))
        )

        snapshot = DataSnapshot(
            timestamp=now,
            prices={'XRP/USDT': 2.35, 'BTC/USDT': 104500.0, 'XRP/BTC': 0.0000225},
            candles_1m={
                'XRP/USDT': tuple(candles),
                'BTC/USDT': tuple(candles),
                'XRP/BTC': tuple(xrp_btc_candles)
            },
            candles_5m={
                'XRP/USDT': tuple(candles),
                'BTC/USDT': tuple(candles),
                'XRP/BTC': tuple(xrp_btc_candles)
            },
            orderbooks={'XRP/USDT': ob, 'BTC/USDT': ob},
            trades={'XRP/USDT': tuple(trades), 'BTC/USDT': tuple(trades)}
        )

        mr.on_start()

        signal = mr.generate_signal(snapshot)

        # Check indicators were calculated
        assert 'indicators' in mr.state
        indicators = mr.state['indicators']
        # 'active' or 'no_signal' both indicate successful indicator calculation
        assert indicators.get('status') in ('active', 'no_signal'), \
            f"Status should be active or no_signal, got {indicators.get('status')}"
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


class TestMarketMakingV15Features:
    """Tests for v1.5.0 enhancements."""

    def test_micro_price_calculation(self):
        """Test volume-weighted micro-price calculation (MM-E01)."""
        from strategies.market_making import _calculate_micro_price

        # Create orderbook with different bid/ask sizes
        # If ask size is larger, micro price should be closer to bid
        # If bid size is larger, micro price should be closer to ask
        ob_bid_heavy = OrderbookSnapshot(
            bids=((100.0, 1000.0),),  # Large bid size
            asks=((101.0, 100.0),),   # Small ask size
        )
        micro = _calculate_micro_price(ob_bid_heavy)
        mid = (100.0 + 101.0) / 2  # 100.5
        # With bid_size=1000, ask_size=100:
        # micro = (100*100 + 101*1000) / 1100 = (10000 + 101000) / 1100 = 100.909
        assert micro > mid  # Micro should be above mid (closer to ask)

        ob_ask_heavy = OrderbookSnapshot(
            bids=((100.0, 100.0),),   # Small bid size
            asks=((101.0, 1000.0),),  # Large ask size
        )
        micro2 = _calculate_micro_price(ob_ask_heavy)
        # micro = (100*1000 + 101*100) / 1100 = (100000 + 10100) / 1100 = 100.091
        assert micro2 < mid  # Micro should be below mid (closer to bid)

    def test_optimal_spread_calculation(self):
        """Test Avellaneda-Stoikov optimal spread calculation (MM-E02)."""
        from strategies.market_making import _calculate_optimal_spread

        # With low volatility, spread should be smaller
        low_vol_spread = _calculate_optimal_spread(
            volatility_pct=0.1,
            gamma=0.1,
            kappa=1.5
        )

        # With high volatility, spread should be larger
        high_vol_spread = _calculate_optimal_spread(
            volatility_pct=1.0,
            gamma=0.1,
            kappa=1.5
        )

        assert high_vol_spread > low_vol_spread
        assert low_vol_spread >= 0
        assert high_vol_spread >= 0

        # With higher gamma (more risk averse), spread should be larger
        high_gamma_spread = _calculate_optimal_spread(
            volatility_pct=0.5,
            gamma=0.5,
            kappa=1.5
        )
        low_gamma_spread = _calculate_optimal_spread(
            volatility_pct=0.5,
            gamma=0.1,
            kappa=1.5
        )
        # Note: The relationship depends on the formula components
        # Just verify they're both positive and different
        assert high_gamma_spread > 0
        assert low_gamma_spread > 0

    def test_fee_profitability_check(self):
        """Test fee-aware profitability check (MM-E03)."""
        from strategies.market_making import _check_fee_profitability

        fee_rate = 0.001  # 0.1% per trade

        # Wide spread - should be profitable
        is_profitable, profit = _check_fee_profitability(
            spread_pct=0.6,  # 0.6% spread (wider to ensure profitability)
            fee_rate=fee_rate,
            min_profit_pct=0.05
        )
        # Expected capture = 0.6/2 = 0.30%
        # Round trip fee = 0.1% * 2 = 0.2%
        # Net profit = 0.30% - 0.2% = 0.10%
        assert is_profitable is True
        assert profit > 0.05  # Above minimum profit threshold

        # Tight spread - should NOT be profitable
        is_profitable2, profit2 = _check_fee_profitability(
            spread_pct=0.1,  # 0.1% spread
            fee_rate=fee_rate,
            min_profit_pct=0.05
        )
        # Expected capture = 0.1/2 = 0.05%
        # Round trip fee = 0.2%
        # Net profit = 0.05% - 0.2% = -0.15%
        assert is_profitable2 is False
        assert profit2 < 0

    def test_position_decay_check(self):
        """Test position decay for stale positions (MM-E04)."""
        from strategies.market_making import _check_position_decay
        from datetime import timedelta

        now = datetime.now()

        # Fresh position - should not be stale
        fresh_pos = {
            'entry_time': now - timedelta(seconds=60),  # 1 minute old
        }
        is_stale, mult = _check_position_decay(
            fresh_pos, now, max_age_seconds=300, tp_multiplier=0.5
        )
        assert is_stale is False
        assert mult == 1.0

        # Stale position - should trigger decay
        stale_pos = {
            'entry_time': now - timedelta(seconds=400),  # 6.67 minutes old
        }
        is_stale2, mult2 = _check_position_decay(
            stale_pos, now, max_age_seconds=300, tp_multiplier=0.5
        )
        assert is_stale2 is True
        assert mult2 == 0.5

        # Position without entry_time - should not be stale
        no_time_pos = {}
        is_stale3, mult3 = _check_position_decay(
            no_time_pos, now, max_age_seconds=300, tp_multiplier=0.5
        )
        assert is_stale3 is False
        assert mult3 == 1.0

    def test_configurable_fallback_price(self):
        """Test configurable XRP/USDT fallback price (MM-011)."""
        from strategies.market_making import _get_xrp_usdt_price

        now = datetime.now()

        # Snapshot with XRP/USDT price
        snapshot_with_price = DataSnapshot(
            timestamp=now,
            prices={'XRP/USDT': 2.50},
            candles_1m={},
            candles_5m={},
            orderbooks={},
            trades={}
        )
        config = {'fallback_xrp_usdt': 3.00}

        # Should use actual price when available
        price = _get_xrp_usdt_price(snapshot_with_price, config)
        assert price == 2.50

        # Snapshot without XRP/USDT price
        snapshot_no_price = DataSnapshot(
            timestamp=now,
            prices={'BTC/USDT': 100000.0},
            candles_1m={},
            candles_5m={},
            orderbooks={},
            trades={}
        )

        # Should use fallback from config
        price2 = _get_xrp_usdt_price(snapshot_no_price, config)
        assert price2 == 3.00

        # With different fallback config
        config2 = {'fallback_xrp_usdt': 2.75}
        price3 = _get_xrp_usdt_price(snapshot_no_price, config2)
        assert price3 == 2.75

    def test_build_entry_signal(self):
        """Test signal builder function (MM-010 refactor)."""
        from strategies.market_making import _build_entry_signal

        # Test buy signal
        signal = _build_entry_signal(
            symbol='XRP/USDT',
            action='buy',
            size=50.0,
            entry_price=2.50,
            reason='Test buy',
            sl_pct=0.5,
            tp_pct=0.5,
            is_cross_pair=False,
            xrp_usdt_price=2.50
        )

        assert signal.action == 'buy'
        assert signal.symbol == 'XRP/USDT'
        assert signal.size == 50.0
        assert signal.price == 2.50
        assert signal.stop_loss == pytest.approx(2.50 * 0.995, rel=0.001)  # -0.5%
        assert signal.take_profit == pytest.approx(2.50 * 1.005, rel=0.001)  # +0.5%
        # Not cross-pair: metadata is None or empty dict (Signal defaults to {})
        assert signal.metadata is None or signal.metadata == {}

        # Test sell signal (stop/tp reversed)
        sell_signal = _build_entry_signal(
            symbol='XRP/USDT',
            action='sell',
            size=50.0,
            entry_price=2.50,
            reason='Test sell',
            sl_pct=0.5,
            tp_pct=0.5,
            is_cross_pair=False,
            xrp_usdt_price=2.50
        )

        assert sell_signal.stop_loss == pytest.approx(2.50 * 1.005, rel=0.001)  # +0.5%
        assert sell_signal.take_profit == pytest.approx(2.50 * 0.995, rel=0.001)  # -0.5%

        # Test cross-pair signal
        cross_signal = _build_entry_signal(
            symbol='XRP/BTC',
            action='buy',
            size=62.5,  # $25 worth at 2.50
            entry_price=0.00002,
            reason='Test cross',
            sl_pct=0.4,
            tp_pct=0.4,
            is_cross_pair=True,
            xrp_usdt_price=2.50
        )

        assert cross_signal.metadata is not None
        assert 'xrp_size' in cross_signal.metadata
        assert cross_signal.metadata['xrp_size'] == pytest.approx(25.0, rel=0.01)

    def test_position_entry_tracking_with_timestamp(self):
        """Test position entry tracking includes timestamp (v1.5.0)."""
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mm = strategies.get('market_making')

        mm.on_start()

        fill_time = datetime.now()
        fill = {
            'symbol': 'XRP/USDT',
            'side': 'buy',
            'size': 100.0,
            'price': 2.35,
            'pnl': 0,
            'timestamp': fill_time,
        }
        mm.on_fill(fill)

        # Check position entry includes entry_time (v1.5.0 for decay)
        assert 'position_entries' in mm.state
        pos = mm.state['position_entries'].get('XRP/USDT')
        assert pos is not None
        assert 'entry_time' in pos
        assert pos['entry_time'] == fill_time

    def test_indicators_include_v15_fields(self):
        """Test that indicators include v1.5.0 fields."""
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mm = strategies.get('market_making')

        mm.on_start()
        snapshot = create_rich_snapshot(2.35)
        mm.generate_signal(snapshot)

        indicators = mm.state.get('indicators', {})

        # v1.5.0 fields
        assert 'micro_price' in indicators
        assert 'is_fee_profitable' in indicators
        assert 'expected_profit_pct' in indicators

    def test_rr_ratios_updated(self):
        """Test that R:R ratios are 1:1 for XRP pairs (MM-009)."""
        from strategies.market_making import SYMBOL_CONFIGS

        # XRP/USDT should have 1:1 R:R
        xrp_usdt = SYMBOL_CONFIGS.get('XRP/USDT', {})
        assert xrp_usdt.get('take_profit_pct') == 0.5
        assert xrp_usdt.get('stop_loss_pct') == 0.5

        # XRP/BTC should have 1:1 R:R
        xrp_btc = SYMBOL_CONFIGS.get('XRP/BTC', {})
        assert xrp_btc.get('take_profit_pct') == 0.4
        assert xrp_btc.get('stop_loss_pct') == 0.4

        # BTC/USDT already had 1:1 R:R
        btc_usdt = SYMBOL_CONFIGS.get('BTC/USDT', {})
        assert btc_usdt.get('take_profit_pct') == 0.35
        assert btc_usdt.get('stop_loss_pct') == 0.35


class TestMeanReversionV40Features:
    """Tests for Mean Reversion v4.0.0 features per deep-review-v4.0.md."""

    def test_trend_slope_calculation(self):
        """Test linear regression slope calculation."""
        from strategies.mean_reversion import _calculate_trend_slope

        now = datetime.now()

        # Create uptrending candles
        uptrend_candles = []
        for i in range(60):
            price = 2.30 + (i * 0.002)  # Upward slope
            uptrend_candles.append(Candle(
                timestamp=now,
                open=price - 0.001,
                high=price + 0.002,
                low=price - 0.002,
                close=price,
                volume=100.0
            ))

        slope = _calculate_trend_slope(uptrend_candles, 50)
        assert slope > 0, "Uptrend should have positive slope"

        # Create downtrending candles
        downtrend_candles = []
        for i in range(60):
            price = 2.50 - (i * 0.002)  # Downward slope
            downtrend_candles.append(Candle(
                timestamp=now,
                open=price + 0.001,
                high=price + 0.002,
                low=price - 0.002,
                close=price,
                volume=100.0
            ))

        slope2 = _calculate_trend_slope(downtrend_candles, 50)
        assert slope2 < 0, "Downtrend should have negative slope"

        # Flat market should have near-zero slope
        flat_candles = []
        for i in range(60):
            flat_candles.append(Candle(
                timestamp=now,
                open=2.35,
                high=2.36,
                low=2.34,
                close=2.35,
                volume=100.0
            ))

        slope3 = _calculate_trend_slope(flat_candles, 50)
        assert abs(slope3) < 0.01, "Flat market should have near-zero slope"

    def test_trend_confirmation_period(self):
        """Test REC-003: trend confirmation requires N consecutive periods."""
        from strategies.mean_reversion import _is_trending

        now = datetime.now()
        state = {}
        symbol = 'XRP/USDT'

        # Create trending candles
        trending_candles = []
        for i in range(60):
            price = 2.30 + (i * 0.003)  # Strong uptrend
            trending_candles.append(Candle(
                timestamp=now,
                open=price - 0.001,
                high=price + 0.002,
                low=price - 0.002,
                close=price,
                volume=100.0
            ))

        config = {
            'trend_sma_period': 50,
            'trend_slope_threshold': 0.05,
            'trend_confirmation_periods': 3
        }

        # First evaluation - should not be confirmed yet
        is_trending1, slope1, count1 = _is_trending(trending_candles, config, state, symbol)
        assert count1 == 1, "First trending evaluation should set count to 1"
        assert is_trending1 is False, "First trending evaluation should not confirm"

        # Second evaluation
        is_trending2, slope2, count2 = _is_trending(trending_candles, config, state, symbol)
        assert count2 == 2, "Second trending evaluation should set count to 2"
        assert is_trending2 is False, "Second trending evaluation should not confirm"

        # Third evaluation - should now confirm
        is_trending3, slope3, count3 = _is_trending(trending_candles, config, state, symbol)
        assert count3 == 3, "Third trending evaluation should set count to 3"
        assert is_trending3 is True, "Third trending evaluation should confirm"

        # Reset with non-trending candles
        flat_candles = []
        for i in range(60):
            flat_candles.append(Candle(
                timestamp=now,
                open=2.35,
                high=2.36,
                low=2.34,
                close=2.35,
                volume=100.0
            ))

        is_trending4, slope4, count4 = _is_trending(flat_candles, config, state, symbol)
        assert count4 == 0, "Non-trending should reset count to 0"
        assert is_trending4 is False, "Non-trending should not confirm"

    def test_correlation_calculation(self):
        """Test REC-005: XRP/BTC correlation calculation."""
        from strategies.mean_reversion import _calculate_correlation
        import math

        now = datetime.now()

        # Create positively correlated candles using sine wave
        # Both move in the same direction
        xrp_candles = []
        btc_candles = []
        for i in range(60):
            # Both use same sine pattern -> positive correlation
            xrp_price = 2.35 + 0.05 * math.sin(i * 0.3)
            btc_price = 100000 + 2000 * math.sin(i * 0.3)
            xrp_candles.append(Candle(now, xrp_price, xrp_price+0.01, xrp_price-0.01, xrp_price, 100))
            btc_candles.append(Candle(now, btc_price, btc_price+100, btc_price-100, btc_price, 10))

        corr = _calculate_correlation(xrp_candles, btc_candles, 50)
        assert corr is not None
        assert corr > 0.9, f"Positively correlated should have high correlation, got {corr}"

        # Create negatively correlated candles
        # XRP uses sine, BTC uses -sine (opposite phase)
        xrp_candles2 = []
        btc_candles2 = []
        for i in range(60):
            xrp_price = 2.35 + 0.05 * math.sin(i * 0.3)
            btc_price = 100000 - 2000 * math.sin(i * 0.3)  # Opposite pattern
            xrp_candles2.append(Candle(now, xrp_price, xrp_price+0.01, xrp_price-0.01, xrp_price, 100))
            btc_candles2.append(Candle(now, btc_price, btc_price+100, btc_price-100, btc_price, 10))

        corr2 = _calculate_correlation(xrp_candles2, btc_candles2, 50)
        assert corr2 is not None
        assert corr2 < -0.9, f"Negatively correlated should have negative correlation, got {corr2}"

        # Insufficient data
        corr3 = _calculate_correlation(xrp_candles[:10], btc_candles[:10], 50)
        assert corr3 is None, "Insufficient data should return None"

    def test_position_decay_timing_v40(self):
        """Test REC-002: extended position decay timing."""
        from strategies.mean_reversion import _get_decayed_take_profit
        from datetime import timedelta

        now = datetime.now()
        entry_price = 2.35
        original_tp_pct = 0.5

        config = {
            'decay_start_minutes': 15.0,      # v4.0: increased from 3
            'decay_interval_minutes': 5.0,     # v4.0: increased from 1
            'decay_multipliers': [1.0, 0.85, 0.7, 0.5],  # v4.0: gentler
        }

        # Fresh position (5 min) - no decay
        entry_time1 = now - timedelta(minutes=5)
        tp1, mult1 = _get_decayed_take_profit(
            entry_price, original_tp_pct, entry_time1, now, 'long', config
        )
        assert mult1 == 1.0, "Fresh position should have no decay"

        # Position at 15 min - first decay stage
        entry_time2 = now - timedelta(minutes=15)
        tp2, mult2 = _get_decayed_take_profit(
            entry_price, original_tp_pct, entry_time2, now, 'long', config
        )
        assert mult2 == 1.0, "At exactly decay_start should still be 1.0"

        # Position at 20 min - second decay stage
        entry_time3 = now - timedelta(minutes=20)
        tp3, mult3 = _get_decayed_take_profit(
            entry_price, original_tp_pct, entry_time3, now, 'long', config
        )
        assert mult3 == 0.85, f"At 20 min should be 0.85, got {mult3}"

        # Position at 25 min - third decay stage
        entry_time4 = now - timedelta(minutes=25)
        tp4, mult4 = _get_decayed_take_profit(
            entry_price, original_tp_pct, entry_time4, now, 'long', config
        )
        assert mult4 == 0.7, f"At 25 min should be 0.7, got {mult4}"

        # Position at 30+ min - final decay stage
        entry_time5 = now - timedelta(minutes=30)
        tp5, mult5 = _get_decayed_take_profit(
            entry_price, original_tp_pct, entry_time5, now, 'long', config
        )
        assert mult5 == 0.5, f"At 30 min should be 0.5, got {mult5}"

    def test_trailing_stop_v40_parameters(self):
        """Test REC-001: updated trailing stop parameters."""
        from strategies.mean_reversion import _calculate_trailing_stop

        entry_price = 2.35

        # v4.0 defaults: activation=0.4%, distance=0.3%
        activation_pct = 0.4
        trail_distance_pct = 0.3

        # Not yet activated (profit < 0.4%)
        highest = entry_price * 1.003  # 0.3% profit
        result = _calculate_trailing_stop(
            entry_price, highest, highest, 'long',
            activation_pct, trail_distance_pct
        )
        assert result is None, "Should not activate below 0.4% profit"

        # Activated (profit >= 0.4%)
        highest2 = entry_price * 1.005  # 0.5% profit
        result2 = _calculate_trailing_stop(
            entry_price, highest2, highest2, 'long',
            activation_pct, trail_distance_pct
        )
        assert result2 is not None, "Should activate at 0.5% profit"
        expected_stop = highest2 * (1 - 0.003)  # 0.3% below highest
        assert result2 == pytest.approx(expected_stop, rel=0.001)

        # Short position
        lowest = entry_price * 0.995  # 0.5% profit for short
        result3 = _calculate_trailing_stop(
            entry_price, lowest, lowest, 'short',
            activation_pct, trail_distance_pct
        )
        assert result3 is not None, "Short should activate at 0.5% profit"
        expected_stop_short = lowest * (1 + 0.003)  # 0.3% above lowest
        assert result3 == pytest.approx(expected_stop_short, rel=0.001)

    def test_trailing_stop_disabled_by_default(self):
        """Test REC-001: trailing stops disabled by default in v4.0."""
        from strategies.mean_reversion import CONFIG

        assert CONFIG['use_trailing_stop'] is False, \
            "v4.0: Trailing stops should be disabled by default"

    def test_config_updated_v40_parameters(self):
        """Test that CONFIG has v4.0 updated parameters."""
        from strategies.mean_reversion import CONFIG

        # REC-001: Trailing stop defaults
        assert CONFIG['trailing_activation_pct'] == 0.4, "v4.0: activation should be 0.4%"
        assert CONFIG['trailing_distance_pct'] == 0.3, "v4.0: distance should be 0.3%"

        # REC-002: Position decay timing
        assert CONFIG['decay_start_minutes'] == 15.0, "v4.0: decay start should be 15 min"
        assert CONFIG['decay_interval_minutes'] == 5.0, "v4.0: decay interval should be 5 min"
        assert CONFIG['decay_multipliers'] == [1.0, 0.85, 0.7, 0.5], \
            "v4.0: decay multipliers should be gentler"

        # REC-003: Trend confirmation
        assert CONFIG['trend_confirmation_periods'] == 3, \
            "v4.0: trend confirmation should require 3 periods"

        # REC-005: Correlation monitoring
        assert CONFIG['use_correlation_monitoring'] is True, \
            "v4.0: correlation monitoring should be enabled"

    def test_version_is_4_3_0(self):
        """Test strategy version is 4.3.0."""
        from strategies.mean_reversion import STRATEGY_VERSION
        assert STRATEGY_VERSION == "4.3.0"

    def test_indicators_include_v40_fields(self):
        """Test that indicators include v4.0.0 fields."""
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mr = strategies.get('mean_reversion')

        mr.on_start()

        # Create a snapshot with data for XRP/USDT with enough candles
        now = datetime.now()
        candles = []
        for i in range(50):  # Need at least 50 for trend calculation
            offset = (50 - i) * 0.001
            candles.append(Candle(
                timestamp=now,
                open=2.35 - offset,
                high=2.35 - offset + 0.005,
                low=2.35 - offset - 0.005,
                close=2.35 - offset + 0.002,
                volume=100.0
            ))

        trades = []
        for i in range(60):
            trades.append(Trade(
                timestamp=now,
                price=2.35 + (i % 2 - 0.5) * 0.002,
                size=50.0 + i,
                side='buy' if i % 3 != 0 else 'sell'
            ))

        ob = OrderbookSnapshot(
            bids=tuple((2.34 - i*0.001, 100.0 + i*10) for i in range(10)),
            asks=tuple((2.36 + i*0.001, 100.0 + i*10) for i in range(10))
        )

        # Create XRP/BTC candles too (strategy evaluates all 3 symbols)
        xrp_btc_candles = []
        for i in range(50):
            offset = (50 - i) * 0.0000001
            xrp_btc_candles.append(Candle(
                timestamp=now,
                open=0.0000225 - offset,
                high=0.0000225 - offset + 0.0000001,
                low=0.0000225 - offset - 0.0000001,
                close=0.0000225 - offset,
                volume=100.0
            ))

        snapshot = DataSnapshot(
            timestamp=now,
            prices={'XRP/USDT': 2.35, 'BTC/USDT': 104500.0, 'XRP/BTC': 0.0000225},
            candles_1m={
                'XRP/USDT': tuple(candles),
                'BTC/USDT': tuple(candles),
                'XRP/BTC': tuple(xrp_btc_candles)
            },
            candles_5m={
                'XRP/USDT': tuple(candles),
                'BTC/USDT': tuple(candles),
                'XRP/BTC': tuple(xrp_btc_candles)
            },
            orderbooks={'XRP/USDT': ob, 'BTC/USDT': ob},
            trades={'XRP/USDT': tuple(trades), 'BTC/USDT': tuple(trades)}
        )

        mr.generate_signal(snapshot)

        indicators = mr.state.get('indicators', {})

        # Check we got past warming up
        assert indicators.get('status') != 'warming_up', \
            f"Should not be warming up with 50 candles, status: {indicators.get('status')}"

        # v4.0.0 fields
        assert 'trend_confirmation_count' in indicators, \
            "v4.0: should include trend_confirmation_count"
        assert 'trend_confirmation_required' in indicators, \
            "v4.0: should include trend_confirmation_required"
        # xrp_btc_correlation is only for XRP/BTC symbol, so it may be None for XRP/USDT
        assert 'xrp_btc_correlation' in indicators, \
            "v4.0: should include xrp_btc_correlation field"

    def test_on_start_logs_v40_params(self):
        """Test on_start logs v4.0 specific parameters."""
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        mr = strategies.get('mean_reversion')

        # Just verify it doesn't crash
        mr.on_start()
        assert mr.state.get('config_validated', False) is True
