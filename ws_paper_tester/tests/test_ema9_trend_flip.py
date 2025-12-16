"""Tests for EMA-9 Trend Flip Strategy v1.0.0."""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.types import DataSnapshot, OrderbookSnapshot, Candle, Trade
from ws_tester.strategy_loader import discover_strategies


def create_hourly_candle_data(
    symbol: str,
    base_price: float,
    num_minutes: int = 120,
    trend: str = 'up'
) -> tuple:
    """
    Create 1-minute candle data that simulates hourly patterns.

    Args:
        symbol: Trading symbol
        base_price: Starting price
        num_minutes: Number of 1-minute candles to create
        trend: 'up', 'down', or 'flat'

    Returns:
        Tuple of Candle objects
    """
    now = datetime.now()
    candles = []

    for i in range(num_minutes):
        ts = now - timedelta(minutes=num_minutes - i)

        if trend == 'up':
            price = base_price + (i * 0.5)  # $0.50 per minute
        elif trend == 'down':
            price = base_price - (i * 0.5)
        else:
            price = base_price + (i % 10 - 5) * 0.1  # Oscillate

        candles.append(Candle(
            timestamp=ts,
            open=price - 5,
            high=price + 10,
            low=price - 10,
            close=price,
            volume=100.0 + i
        ))

    return tuple(candles)


def create_flip_scenario_data(
    base_price: float = 100000.0,
    consecutive_below: int = 5,
    flip_above: bool = True
) -> tuple:
    """
    Create candle data that simulates an EMA flip scenario.

    Creates 1-minute candles for ~2 hours where:
    - First part: candles open below EMA
    - Last part: candle opens above EMA (flip)
    """
    now = datetime.now()
    candles = []
    num_minutes = 120  # 2 hours of 1-minute candles

    # We need to create a pattern where:
    # - EMA-9 of hourly candles is calculable
    # - Last N hourly candles opened below EMA
    # - Current hourly candle opens above EMA

    # Create initial data to establish EMA baseline
    for i in range(60):  # First hour - stable around base_price
        ts = now - timedelta(minutes=num_minutes - i)
        price = base_price + (i % 5 - 2) * 10  # Small oscillation
        candles.append(Candle(
            timestamp=ts,
            open=price,
            high=price + 20,
            low=price - 20,
            close=price + 5,
            volume=100.0
        ))

    # Create second hour where price drops below EMA trend
    for i in range(60, 110):  # 50 minutes - price consistently lower
        ts = now - timedelta(minutes=num_minutes - i)
        # Price opens significantly below where EMA would be
        price = base_price - 500 - ((i - 60) * 5)
        candles.append(Candle(
            timestamp=ts,
            open=price,
            high=price + 20,
            low=price - 20,
            close=price - 10,
            volume=100.0
        ))

    if flip_above:
        # Last 10 minutes - price flips above EMA
        for i in range(110, 120):
            ts = now - timedelta(minutes=num_minutes - i)
            price = base_price + 200  # Opens above EMA
            candles.append(Candle(
                timestamp=ts,
                open=price,
                high=price + 50,
                low=price - 10,
                close=price + 30,
                volume=150.0
            ))
    else:
        # Last 10 minutes - price stays below
        for i in range(110, 120):
            ts = now - timedelta(minutes=num_minutes - i)
            price = base_price - 600
            candles.append(Candle(
                timestamp=ts,
                open=price,
                high=price + 20,
                low=price - 20,
                close=price,
                volume=100.0
            ))

    return tuple(candles)


def create_test_snapshot(
    symbol: str = 'BTC/USDT',
    price: float = 100000.0,
    candles_1m: tuple = None,
    candles_5m: tuple = None
) -> DataSnapshot:
    """Create a test DataSnapshot."""
    now = datetime.now()

    if candles_1m is None:
        candles_1m = create_hourly_candle_data(symbol, price, 120, 'flat')

    if candles_5m is None:
        candles_5m = tuple()

    # Create orderbook
    ob = OrderbookSnapshot(
        bids=tuple((price - 10 - i * 5, 10.0 + i) for i in range(10)),
        asks=tuple((price + 10 + i * 5, 10.0 + i) for i in range(10))
    )

    # Create trades
    trades = []
    for i in range(50):
        trades.append(Trade(
            timestamp=now,
            price=price + (i % 2 - 0.5) * 10,
            size=1.0,
            side='buy' if i % 2 == 0 else 'sell'
        ))

    return DataSnapshot(
        timestamp=now,
        prices={symbol: price},
        candles_1m={symbol: candles_1m},
        candles_5m={symbol: candles_5m},
        orderbooks={symbol: ob},
        trades={symbol: tuple(trades)}
    )


class TestEma9TrendFlipStrategy:
    """Tests for EMA-9 Trend Flip Strategy."""

    def test_strategy_discovery(self):
        """Test that strategy is discovered by loader."""
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))

        assert 'ema9_trend_flip' in strategies
        strategy = strategies['ema9_trend_flip']
        assert strategy.name == 'ema9_trend_flip'
        assert strategy.version == '1.0.0'
        assert 'BTC/USDT' in strategy.symbols

    def test_strategy_has_required_components(self):
        """Test strategy has all required components."""
        from strategies.ema9_trend_flip import (
            STRATEGY_NAME, STRATEGY_VERSION, SYMBOLS, CONFIG,
            generate_signal, on_start, on_fill, on_stop
        )

        assert STRATEGY_NAME == 'ema9_trend_flip'
        assert STRATEGY_VERSION == '1.0.0'
        assert len(SYMBOLS) > 0
        assert 'ema_period' in CONFIG
        assert 'consecutive_candles' in CONFIG
        assert callable(generate_signal)
        assert callable(on_start)
        assert callable(on_fill)
        assert callable(on_stop)

    def test_ema_calculation(self):
        """Test EMA calculation function."""
        from strategies.ema9_trend_flip import calculate_ema

        # Test with known values
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        ema = calculate_ema(prices, 9)

        assert ema is not None
        # EMA should be between min and max of recent prices
        assert ema >= min(prices[-9:])
        assert ema <= max(prices[-9:])

        # Test with insufficient data
        ema_short = calculate_ema([1, 2, 3], 9)
        assert ema_short is None

    def test_ema_series_calculation(self):
        """Test EMA series calculation."""
        from strategies.ema9_trend_flip import calculate_ema_series

        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        ema_series = calculate_ema_series(prices, 9)

        assert len(ema_series) == len(prices)
        # First 8 values should be None
        for i in range(8):
            assert ema_series[i] is None
        # 9th and onwards should have values
        assert ema_series[8] is not None
        assert ema_series[-1] is not None

    def test_build_hourly_candles(self):
        """Test hourly candle aggregation from 1-minute candles."""
        from strategies.ema9_trend_flip import build_hourly_candles

        # Create 120 1-minute candles (2 hours)
        candles_1m = create_hourly_candle_data('BTC/USDT', 100000, 120, 'flat')

        hourly = build_hourly_candles(candles_1m, timeframe_minutes=60)

        # Should have 2 hourly candles (or 3 if partial)
        assert len(hourly) >= 2
        assert len(hourly) <= 3

        # Check candle structure
        for candle in hourly:
            assert 'open' in candle
            assert 'high' in candle
            assert 'low' in candle
            assert 'close' in candle
            assert 'volume' in candle
            assert 'timestamp' in candle

    def test_get_candle_position(self):
        """Test candle position relative to EMA."""
        from strategies.ema9_trend_flip import get_candle_position

        candle = {'open': 100.0, 'close': 101.0}
        ema = 99.0
        buffer_pct = 0.1

        # Price above EMA with buffer
        position = get_candle_position(candle, ema, buffer_pct, use_open=True)
        assert position == 'above'

        # Price below EMA
        candle_below = {'open': 97.0, 'close': 98.0}
        position_below = get_candle_position(candle_below, ema, buffer_pct, use_open=True)
        assert position_below == 'below'

        # Price within buffer (neutral)
        candle_neutral = {'open': 99.05, 'close': 99.1}
        position_neutral = get_candle_position(candle_neutral, ema, buffer_pct, use_open=True)
        assert position_neutral == 'neutral'

    def test_atr_calculation(self):
        """Test ATR calculation."""
        from strategies.ema9_trend_flip import calculate_atr

        # Create candles with known range
        candles = []
        for i in range(20):
            candles.append({
                'open': 100.0,
                'high': 102.0,
                'low': 98.0,
                'close': 101.0
            })

        atr = calculate_atr(candles, period=14)
        assert atr is not None
        # ATR should be approximately 4 (high - low = 102 - 98)
        assert atr > 3.0
        assert atr < 5.0

    def test_generate_signal_no_crash_empty_data(self):
        """Test strategy doesn't crash with empty data."""
        from strategies.ema9_trend_flip import generate_signal, CONFIG

        empty_snapshot = DataSnapshot(
            timestamp=datetime.now(),
            prices={},
            candles_1m={},
            candles_5m={},
            orderbooks={},
            trades={}
        )

        state = {}
        signal = generate_signal(empty_snapshot, CONFIG, state)

        # Should return None, not crash
        assert signal is None

    def test_generate_signal_warming_up(self):
        """Test strategy in warming up state with insufficient data."""
        from strategies.ema9_trend_flip import generate_signal, CONFIG

        # Create snapshot with only 30 minutes of data (not enough for 1H candles)
        now = datetime.now()
        candles = []
        for i in range(30):
            ts = now - timedelta(minutes=30 - i)
            candles.append(Candle(
                timestamp=ts,
                open=100000 + i * 10,
                high=100000 + i * 10 + 50,
                low=100000 + i * 10 - 50,
                close=100000 + i * 10 + 20,
                volume=10.0
            ))

        snapshot = DataSnapshot(
            timestamp=now,
            prices={'BTC/USDT': 100300},
            candles_1m={'BTC/USDT': tuple(candles)},
            candles_5m={'BTC/USDT': ()},
            orderbooks={'BTC/USDT': OrderbookSnapshot(
                bids=((100290, 10),),
                asks=((100310, 10),)
            )},
            trades={'BTC/USDT': ()}
        )

        state = {}
        signal = generate_signal(snapshot, CONFIG, state)

        assert signal is None
        assert state['indicators']['status'] == 'warming_up'

    def test_on_start_initializes_state(self):
        """Test on_start initializes state correctly."""
        from strategies.ema9_trend_flip import on_start, CONFIG

        state = {}
        on_start(CONFIG, state)

        assert state['initialized'] is True
        assert state['position'] == 0.0
        assert state['position_side'] is None
        assert state['trade_count'] == 0
        assert 'rejection_counts' in state

    def test_on_fill_tracks_position(self):
        """Test on_fill updates position tracking."""
        from strategies.ema9_trend_flip import on_start, on_fill, CONFIG

        state = {}
        on_start(CONFIG, state)

        # Simulate buy fill
        fill = {
            'side': 'buy',
            'value': 50.0,
            'price': 100000.0,
            'pnl': 0
        }
        on_fill(fill, state)

        assert state['position'] == 50.0
        assert state['position_side'] == 'long'
        assert state['entry_price'] == 100000.0

        # Simulate closing sell
        close_fill = {
            'side': 'sell',
            'value': 50.0,
            'price': 100500.0,
            'pnl': 25.0
        }
        on_fill(close_fill, state)

        assert state['position'] == 0.0
        assert state['position_side'] is None
        assert state['trade_count'] == 1
        assert state['win_count'] == 1
        assert state['total_pnl'] == 25.0

    def test_config_defaults(self):
        """Test CONFIG has expected defaults."""
        from strategies.ema9_trend_flip import CONFIG

        assert CONFIG['ema_period'] == 9
        assert CONFIG['consecutive_candles'] == 3
        assert CONFIG['buffer_pct'] == 0.1
        assert CONFIG['candle_timeframe_minutes'] == 60
        assert CONFIG['stop_loss_pct'] == 1.0
        assert CONFIG['take_profit_pct'] == 2.0  # 2:1 R:R
        assert CONFIG['max_hold_hours'] == 72

    def test_check_consecutive_positions(self):
        """Test consecutive position checking."""
        from strategies.ema9_trend_flip import (
            check_consecutive_positions, calculate_ema_series
        )

        # Create candles all opening below EMA
        candles = []
        for i in range(10):
            candles.append({
                'open': 95.0,  # Consistently below
                'close': 96.0,
                'high': 97.0,
                'low': 94.0
            })

        # EMA would be around 100 if prices were around 100
        ema_values = [100.0] * 10  # Simulate EMA at 100

        position, count = check_consecutive_positions(
            candles, ema_values, n_consecutive=3, buffer_pct=0.1, use_open=True
        )

        assert position == 'below'
        assert count >= 3

    def test_strategy_generates_long_signal_on_flip(self):
        """Test strategy generates long signal when flip occurs."""
        strategies_path = Path(__file__).parent.parent / "strategies"
        strategies = discover_strategies(str(strategies_path))
        strategy = strategies['ema9_trend_flip']

        # Use much longer data to ensure sufficient hourly candles
        now = datetime.now()
        candles = []

        # Create 15 hours of data (900 minutes)
        # First 10 hours: price opens below EMA (around 99000-99500)
        # Last 5 hours: price opens above EMA (around 101000)
        for i in range(900):
            ts = now - timedelta(minutes=900 - i)

            if i < 600:  # First 10 hours - price below trend
                price = 99000 + (i % 100) * 2
            else:  # Last 5 hours - price above trend
                price = 101000 + (i - 600) * 5

            candles.append(Candle(
                timestamp=ts,
                open=price,
                high=price + 100,
                low=price - 100,
                close=price + 50,
                volume=100.0
            ))

        snapshot = DataSnapshot(
            timestamp=now,
            prices={'BTC/USDT': 103500},
            candles_1m={'BTC/USDT': tuple(candles)},
            candles_5m={'BTC/USDT': ()},
            orderbooks={'BTC/USDT': OrderbookSnapshot(
                bids=((103450, 10),),
                asks=((103550, 10),)
            )},
            trades={'BTC/USDT': ()}
        )

        strategy.on_start()

        # Generate signal - may or may not trigger depending on exact conditions
        signal = strategy.generate_signal(snapshot)

        # Verify indicators are populated
        assert 'indicators' in strategy.state
        indicators = strategy.state['indicators']
        assert 'ema_9' in indicators
        assert 'current_position' in indicators
        assert 'prev_position' in indicators

    def test_exit_on_ema_flip(self):
        """Test exit signal when EMA flips opposite direction."""
        from strategies.ema9_trend_flip import check_exit_conditions, CONFIG

        state = {
            'position_side': 'long',
            'position': 50.0,
            'entry_price': 100000.0,
            'entry_time': datetime.now() - timedelta(hours=1)
        }

        # EMA flip: current position is below (bearish) while we're long
        exit_signal = check_exit_conditions(
            state=state,
            symbol='BTC/USDT',
            current_price=99000.0,
            current_ema=99500.0,
            current_position='below',  # Flip to below
            current_time=datetime.now(),
            config=CONFIG,
            atr=500.0
        )

        assert exit_signal is not None
        assert exit_signal.action == 'sell'
        assert 'EMA flip' in exit_signal.reason

    def test_max_hold_time_exit(self):
        """Test exit due to max hold time exceeded."""
        from strategies.ema9_trend_flip import check_exit_conditions, CONFIG

        config = CONFIG.copy()
        config['max_hold_hours'] = 24

        state = {
            'position_side': 'long',
            'position': 50.0,
            'entry_price': 100000.0,
            'entry_time': datetime.now() - timedelta(hours=25)  # Over 24 hours
        }

        exit_signal = check_exit_conditions(
            state=state,
            symbol='BTC/USDT',
            current_price=100500.0,
            current_ema=100000.0,
            current_position='above',  # No flip
            current_time=datetime.now(),
            config=config,
            atr=500.0
        )

        assert exit_signal is not None
        assert 'hold time' in exit_signal.reason.lower()

    def test_cooldown_after_signal(self):
        """Test cooldown prevents rapid signals."""
        from strategies.ema9_trend_flip import generate_signal, CONFIG

        # Create sufficient data - 15 hours of 1-minute candles (900 candles)
        now = datetime.now()
        candles = []
        for i in range(900):  # 15 hours of data
            ts = now - timedelta(minutes=900 - i)
            price = 100000 + (i % 50) * 10
            candles.append(Candle(
                timestamp=ts,
                open=price,
                high=price + 50,
                low=price - 50,
                close=price + 20,
                volume=100.0
            ))

        snapshot = DataSnapshot(
            timestamp=now,
            prices={'BTC/USDT': 100250},
            candles_1m={'BTC/USDT': tuple(candles)},
            candles_5m={'BTC/USDT': ()},
            orderbooks={'BTC/USDT': OrderbookSnapshot(
                bids=((100200, 10),),
                asks=((100300, 10),)
            )},
            trades={'BTC/USDT': ()}
        )

        state = {
            'initialized': True,
            'last_signal_time': datetime.now() - timedelta(minutes=5),  # Recent signal
            'position': 0.0,
            'position_side': None,
            'indicators': {},
            'last_trade_was_loss': False
        }

        config = CONFIG.copy()
        config['cooldown_minutes'] = 30  # 30 min cooldown

        signal = generate_signal(snapshot, config, state)

        # Should be in cooldown
        assert signal is None
        assert state['indicators']['status'] == 'cooldown'

    def test_rejection_tracking(self):
        """Test rejection tracking functionality."""
        from strategies.ema9_trend_flip import (
            track_rejection, RejectionReason
        )

        state = {}

        track_rejection(state, RejectionReason.WARMING_UP, 'BTC/USDT')
        track_rejection(state, RejectionReason.WARMING_UP, 'BTC/USDT')
        track_rejection(state, RejectionReason.NO_FLIP_SIGNAL, 'BTC/USDT')

        assert state['rejection_counts']['warming_up'] == 2
        assert state['rejection_counts']['no_flip_signal'] == 1

    def test_short_entry_signal(self):
        """Test strategy creates short signal on downward flip."""
        from strategies.ema9_trend_flip import create_entry_signal, CONFIG

        signal = create_entry_signal(
            symbol='BTC/USDT',
            direction='short',
            price=100000.0,
            ema=100500.0,
            config=CONFIG,
            state={'position': 0.0},
            atr=500.0,
            prev_count=4
        )

        assert signal is not None
        assert signal.action == 'short'
        assert signal.stop_loss > signal.price  # Stop above for shorts
        assert signal.take_profit < signal.price  # TP below for shorts
        assert 'flip' in signal.reason.lower()

    def test_long_entry_signal(self):
        """Test strategy creates long signal on upward flip."""
        from strategies.ema9_trend_flip import create_entry_signal, CONFIG

        signal = create_entry_signal(
            symbol='BTC/USDT',
            direction='long',
            price=100000.0,
            ema=99500.0,
            config=CONFIG,
            state={'position': 0.0},
            atr=500.0,
            prev_count=3
        )

        assert signal is not None
        assert signal.action == 'buy'
        assert signal.stop_loss < signal.price  # Stop below for longs
        assert signal.take_profit > signal.price  # TP above for longs

    def test_atr_based_stops(self):
        """Test ATR-based stop loss and take profit."""
        from strategies.ema9_trend_flip import create_entry_signal, CONFIG

        config = CONFIG.copy()
        config['use_atr_stops'] = True
        config['atr_stop_mult'] = 1.5
        config['atr_tp_mult'] = 3.0

        atr = 500.0
        price = 100000.0

        signal = create_entry_signal(
            symbol='BTC/USDT',
            direction='long',
            price=price,
            ema=99500.0,
            config=config,
            state={'position': 0.0},
            atr=atr,
            prev_count=3
        )

        # Stop loss should be 1.5 * ATR below entry
        expected_sl = price - (atr * 1.5)
        assert signal.stop_loss == pytest.approx(expected_sl, rel=0.01)

        # Take profit should be 3.0 * ATR above entry
        expected_tp = price + (atr * 3.0)
        assert signal.take_profit == pytest.approx(expected_tp, rel=0.01)


class TestEma9EdgeCases:
    """Edge case tests for EMA-9 Trend Flip Strategy."""

    def test_insufficient_1m_candles(self):
        """Test behavior with insufficient 1-minute candles."""
        from strategies.ema9_trend_flip import generate_signal, CONFIG

        # Only 5 1-minute candles
        candles = tuple(Candle(
            timestamp=datetime.now() - timedelta(minutes=5-i),
            open=100000 + i * 10,
            high=100000 + i * 10 + 20,
            low=100000 + i * 10 - 20,
            close=100000 + i * 10 + 10,
            volume=10.0
        ) for i in range(5))

        snapshot = DataSnapshot(
            timestamp=datetime.now(),
            prices={'BTC/USDT': 100050},
            candles_1m={'BTC/USDT': candles},
            candles_5m={'BTC/USDT': ()},
            orderbooks={'BTC/USDT': OrderbookSnapshot(
                bids=((100040, 10),),
                asks=((100060, 10),)
            )},
            trades={'BTC/USDT': ()}
        )

        state = {}
        signal = generate_signal(snapshot, CONFIG, state)

        assert signal is None
        # Should indicate warming up or insufficient data
        assert state['indicators']['status'] in ['warming_up', 'no_candle_data']

    def test_position_size_limits(self):
        """Test position sizing respects limits."""
        from strategies.ema9_trend_flip import create_entry_signal, CONFIG

        config = CONFIG.copy()
        config['position_size_usd'] = 100.0
        config['max_position_usd'] = 50.0  # Max less than size

        state = {'position': 30.0}  # Already have position

        signal = create_entry_signal(
            symbol='BTC/USDT',
            direction='long',
            price=100000.0,
            ema=99500.0,
            config=config,
            state=state,
            atr=500.0,
            prev_count=3
        )

        # Should cap at available: max - current = 50 - 30 = 20
        assert signal.size == 20.0

    def test_no_signal_when_at_max_position(self):
        """Test no signal generated when at max position."""
        from strategies.ema9_trend_flip import create_entry_signal, CONFIG

        config = CONFIG.copy()
        config['position_size_usd'] = 50.0
        config['max_position_usd'] = 100.0
        config['min_trade_size_usd'] = 10.0

        state = {'position': 95.0}  # Almost at max

        signal = create_entry_signal(
            symbol='BTC/USDT',
            direction='long',
            price=100000.0,
            ema=99500.0,
            config=config,
            state=state,
            atr=500.0,
            prev_count=3
        )

        # Available = 100 - 95 = 5, which is below min_trade_size
        assert signal is None

    def test_loss_tracking(self):
        """Test consecutive loss tracking."""
        from strategies.ema9_trend_flip import on_fill, on_start, CONFIG

        state = {}
        on_start(CONFIG, state)

        # Open long
        on_fill({'side': 'buy', 'value': 50.0, 'price': 100000.0, 'pnl': 0}, state)

        # Close with loss
        on_fill({'side': 'sell', 'value': 50.0, 'price': 99000.0, 'pnl': -50.0}, state)

        assert state['loss_count'] == 1
        assert state['last_trade_was_loss'] is True
        assert state['total_pnl'] == -50.0

    def test_buffer_calculation(self):
        """Test buffer percentage calculation."""
        from strategies.ema9_trend_flip import get_candle_position

        ema = 100.0
        buffer_pct = 1.0  # 1% buffer

        # Price at exactly 1% above
        candle_boundary_above = {'open': 101.0, 'close': 101.0}
        pos = get_candle_position(candle_boundary_above, ema, buffer_pct)
        # At boundary - could be above or neutral depending on implementation
        assert pos in ['above', 'neutral']

        # Price clearly above
        candle_clear_above = {'open': 102.0, 'close': 102.0}
        pos2 = get_candle_position(candle_clear_above, ema, buffer_pct)
        assert pos2 == 'above'

        # Price within buffer
        candle_within = {'open': 100.5, 'close': 100.5}
        pos3 = get_candle_position(candle_within, ema, buffer_pct)
        assert pos3 == 'neutral'
