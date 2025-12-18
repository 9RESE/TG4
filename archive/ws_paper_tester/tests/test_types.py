"""Tests for core types."""

import pytest
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.types import (
    Candle, Trade, OrderbookSnapshot, DataSnapshot, Signal, Position, Fill
)


class TestCandle:
    def test_candle_creation(self):
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000.0
        )
        assert candle.open == 100.0
        assert candle.close == 105.0

    def test_candle_properties(self):
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000.0
        )
        assert candle.body_size == 5.0
        assert candle.range == 15.0
        assert candle.is_bullish is True

    def test_bearish_candle(self):
        candle = Candle(
            timestamp=datetime.now(),
            open=105.0,
            high=110.0,
            low=95.0,
            close=100.0,
            volume=1000.0
        )
        assert candle.is_bullish is False


class TestOrderbookSnapshot:
    def test_orderbook_creation(self):
        ob = OrderbookSnapshot(
            bids=((100.0, 10.0), (99.0, 20.0)),
            asks=((101.0, 15.0), (102.0, 25.0))
        )
        assert ob.best_bid == 100.0
        assert ob.best_ask == 101.0

    def test_spread_calculation(self):
        ob = OrderbookSnapshot(
            bids=((100.0, 10.0),),
            asks=((101.0, 15.0),)
        )
        assert ob.spread == 1.0
        assert ob.mid == 100.5

    def test_empty_orderbook(self):
        ob = OrderbookSnapshot(bids=(), asks=())
        assert ob.best_bid == 0.0
        assert ob.best_ask == 0.0
        assert ob.mid == 0.0

    def test_imbalance(self):
        # More bids than asks
        ob = OrderbookSnapshot(
            bids=((100.0, 100.0),),
            asks=((101.0, 50.0),)
        )
        assert ob.imbalance > 0  # Positive = more bids


class TestSignal:
    def test_signal_creation(self):
        signal = Signal(
            action='buy',
            symbol='XRP/USD',
            size=100.0,
            price=2.35,
            reason='Test signal'
        )
        assert signal.action == 'buy'
        assert signal.symbol == 'XRP/USD'

    def test_invalid_action(self):
        with pytest.raises(ValueError):
            Signal(
                action='invalid',
                symbol='XRP/USD',
                size=100.0,
                price=2.35,
                reason='Test'
            )

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            Signal(
                action='buy',
                symbol='XRP/USD',
                size=-10.0,
                price=2.35,
                reason='Test'
            )

    def test_limit_order_requires_price(self):
        with pytest.raises(ValueError):
            Signal(
                action='buy',
                symbol='XRP/USD',
                size=100.0,
                price=2.35,
                reason='Test',
                order_type='limit'
            )


class TestPosition:
    def test_position_creation(self):
        pos = Position(
            symbol='XRP/USD',
            side='long',
            size=100.0,
            entry_price=2.35,
            entry_time=datetime.now()
        )
        assert pos.notional == 235.0

    def test_unrealized_pnl_long(self):
        pos = Position(
            symbol='XRP/USD',
            side='long',
            size=100.0,
            entry_price=2.35,
            entry_time=datetime.now()
        )
        # Price went up
        pnl = pos.unrealized_pnl(2.40)
        assert abs(pnl - 5.0) < 0.001  # (2.40 - 2.35) * 100

    def test_unrealized_pnl_short(self):
        pos = Position(
            symbol='XRP/USD',
            side='short',
            size=100.0,
            entry_price=2.35,
            entry_time=datetime.now()
        )
        # Price went down (profit for short)
        pnl = pos.unrealized_pnl(2.30)
        assert abs(pnl - 5.0) < 0.001  # (2.35 - 2.30) * 100


class TestDataSnapshot:
    def test_snapshot_creation(self):
        snapshot = DataSnapshot(
            timestamp=datetime.now(),
            prices={'XRP/USD': 2.35, 'BTC/USD': 104500.0},
            candles_1m={},
            candles_5m={},
            orderbooks={},
            trades={}
        )
        assert snapshot.prices['XRP/USD'] == 2.35

    def test_snapshot_vwap(self):
        trades = (
            Trade(datetime.now(), 2.35, 100.0, 'buy'),
            Trade(datetime.now(), 2.36, 200.0, 'buy'),
        )
        snapshot = DataSnapshot(
            timestamp=datetime.now(),
            prices={'XRP/USD': 2.35},
            candles_1m={},
            candles_5m={},
            orderbooks={},
            trades={'XRP/USD': trades}
        )
        vwap = snapshot.get_vwap('XRP/USD')
        # VWAP = (2.35*100 + 2.36*200) / 300 = 707/300 = 2.3567
        assert vwap is not None
        assert abs(vwap - 2.3567) < 0.001

    def test_trade_imbalance(self):
        trades = (
            Trade(datetime.now(), 2.35, 100.0, 'buy'),
            Trade(datetime.now(), 2.36, 50.0, 'sell'),
        )
        snapshot = DataSnapshot(
            timestamp=datetime.now(),
            prices={'XRP/USD': 2.35},
            candles_1m={},
            candles_5m={},
            orderbooks={},
            trades={'XRP/USD': trades}
        )
        imbalance = snapshot.get_trade_imbalance('XRP/USD')
        # (100 - 50) / 150 = 0.333
        assert abs(imbalance - 0.333) < 0.01
