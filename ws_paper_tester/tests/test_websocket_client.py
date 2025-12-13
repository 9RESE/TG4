"""Tests for WebSocket client and data manager."""

import pytest
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.data_layer import KrakenWSClient, DataManager, SimulatedDataManager


class TestKrakenWSClient:
    """Test WebSocket client functionality."""

    def test_client_creation(self):
        """Test client instantiation."""
        client = KrakenWSClient(['XRP/USD', 'BTC/USD'])

        assert client.symbols == ['XRP/USD', 'BTC/USD']
        assert client._kraken_symbols == ['XRP/USD', 'BTC/USD']
        assert client.ws is None
        assert client._running is False

    def test_symbol_conversion(self):
        """Test symbol format conversion."""
        client = KrakenWSClient(['XRP/USD'])

        # Test to Kraken format
        assert client._to_kraken_symbol('XRP/USD') == 'XRP/USD'

        # Test from Kraken format
        assert client._from_kraken_symbol('XRP/USD') == 'XRP/USD'

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure handling."""
        client = KrakenWSClient(['XRP/USD'])

        # Mock websockets.connect to fail
        with patch('ws_tester.data_layer.websockets.connect', side_effect=Exception("Connection failed")):
            result = await client.connect()
            assert result is False
            assert client._running is False

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        client = KrakenWSClient(['XRP/USD'])

        mock_ws = MagicMock()
        mock_ws.closed = False

        # Create an async mock for the connect coroutine
        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch('ws_tester.data_layer.websockets.connect', side_effect=mock_connect):
            result = await client.connect()
            assert result is True
            assert client._running is True
            assert client.ws is not None

    @pytest.mark.asyncio
    async def test_subscribe(self):
        """Test channel subscription."""
        client = KrakenWSClient(['XRP/USD'])

        mock_ws = AsyncMock()
        client.ws = mock_ws

        await client.subscribe(['trade', 'ticker', 'book'])

        # Should have sent 3 subscription messages
        assert mock_ws.send.call_count == 3

        # Verify trade subscription
        calls = mock_ws.send.call_args_list
        trade_msg = json.loads(calls[0][0][0])
        assert trade_msg['method'] == 'subscribe'
        assert trade_msg['params']['channel'] == 'trade'
        assert trade_msg['params']['symbol'] == ['XRP/USD']

    @pytest.mark.asyncio
    async def test_close(self):
        """Test connection close."""
        client = KrakenWSClient(['XRP/USD'])

        mock_ws = AsyncMock()
        client.ws = mock_ws
        client._running = True

        await client.close()

        assert client._running is False
        mock_ws.close.assert_called_once()


class TestDataManagerMessageHandling:
    """Test DataManager message processing."""

    @pytest.mark.asyncio
    async def test_handle_trade_message(self):
        """Test processing trade messages."""
        dm = DataManager(['XRP/USD'])

        trade_msg = {
            'channel': 'trade',
            'data': [{
                'symbol': 'XRP/USD',
                'price': '2.35',
                'qty': '100',
                'side': 'buy',
                'timestamp': '2024-01-01T12:00:00Z'
            }]
        }

        await dm.on_message(trade_msg)

        assert dm._prices['XRP/USD'] == 2.35
        assert len(dm._trades['XRP/USD']) == 1
        trade = dm._trades['XRP/USD'][0]
        assert trade.price == 2.35
        assert trade.size == 100.0
        assert trade.side == 'buy'

    @pytest.mark.asyncio
    async def test_handle_ticker_message(self):
        """Test processing ticker messages."""
        dm = DataManager(['XRP/USD'])

        ticker_msg = {
            'channel': 'ticker',
            'data': [{
                'symbol': 'XRP/USD',
                'last': '2.40'
            }]
        }

        await dm.on_message(ticker_msg)

        assert dm._prices['XRP/USD'] == 2.40

    @pytest.mark.asyncio
    async def test_handle_book_message(self):
        """Test processing orderbook messages."""
        dm = DataManager(['XRP/USD'])

        book_msg = {
            'channel': 'book',
            'data': [{
                'symbol': 'XRP/USD',
                'bids': [
                    {'price': '2.34', 'qty': '100'},
                    {'price': '2.33', 'qty': '200'}
                ],
                'asks': [
                    {'price': '2.36', 'qty': '100'},
                    {'price': '2.37', 'qty': '200'}
                ]
            }]
        }

        await dm.on_message(book_msg)

        assert 'XRP/USD' in dm._orderbooks
        ob = dm._orderbooks['XRP/USD']
        assert len(ob['bids']) == 2
        assert len(ob['asks']) == 2
        assert ob['bids'][0] == (2.34, 100.0)  # Highest bid first
        assert ob['asks'][0] == (2.36, 100.0)  # Lowest ask first

        # Mid price should be set (use approx for floating point)
        assert dm._prices['XRP/USD'] == pytest.approx(2.35, rel=1e-9)

    @pytest.mark.asyncio
    async def test_handle_ohlc_message(self):
        """Test processing OHLC candle messages."""
        dm = DataManager(['XRP/USD'])

        ohlc_msg = {
            'channel': 'ohlc',
            'data': [{
                'symbol': 'XRP/USD',
                'interval': 1,
                'timestamp': '2024-01-01T12:00:00Z',
                'open': '2.30',
                'high': '2.40',
                'low': '2.29',
                'close': '2.35',
                'volume': '10000'
            }]
        }

        await dm.on_message(ohlc_msg)

        assert len(dm._candles_1m['XRP/USD']) == 1
        candle = dm._candles_1m['XRP/USD'][0]
        assert candle.open == 2.30
        assert candle.high == 2.40
        assert candle.low == 2.29
        assert candle.close == 2.35
        assert candle.volume == 10000.0

    @pytest.mark.asyncio
    async def test_ignore_unknown_symbol(self):
        """Test that unknown symbols are ignored."""
        dm = DataManager(['XRP/USD'])

        trade_msg = {
            'channel': 'trade',
            'data': [{
                'symbol': 'ETH/USD',
                'price': '3000',
                'qty': '1',
                'side': 'buy',
                'timestamp': '2024-01-01T12:00:00Z'
            }]
        }

        await dm.on_message(trade_msg)

        assert 'ETH/USD' not in dm._prices

    @pytest.mark.asyncio
    async def test_handle_invalid_message(self):
        """Test handling invalid messages."""
        dm = DataManager(['XRP/USD'])

        # Non-dict message
        await dm.on_message("invalid")
        await dm.on_message(None)
        await dm.on_message([1, 2, 3])

        # Should not crash, just ignore
        assert len(dm._prices) == 0


class TestDataManagerCandleBuilding:
    """Test candle building from trades."""

    @pytest.mark.asyncio
    async def test_candle_building_from_trades(self):
        """Test that candles are built from trades."""
        dm = DataManager(['XRP/USD'])

        now = datetime.now().replace(second=0, microsecond=0)

        # Simulate trades within same minute
        for i, price in enumerate([2.30, 2.35, 2.28, 2.32]):
            trade_msg = {
                'channel': 'trade',
                'data': [{
                    'symbol': 'XRP/USD',
                    'price': str(price),
                    'qty': '100',
                    'side': 'buy',
                    'timestamp': now.isoformat() + 'Z'
                }]
            }
            await dm.on_message(trade_msg)

        # Check building candle
        assert 'XRP/USD' in dm._current_candle_1m
        candle = dm._current_candle_1m['XRP/USD']
        assert candle['open'] == 2.30
        assert candle['high'] == 2.35
        assert candle['low'] == 2.28
        assert candle['close'] == 2.32
        assert candle['volume'] == 400.0

    @pytest.mark.asyncio
    async def test_candle_rollover(self):
        """Test candle rollover at minute boundary."""
        dm = DataManager(['XRP/USD'])

        # First minute
        minute1 = datetime(2024, 1, 1, 12, 0, 0)
        dm._update_building_candle('XRP/USD', 2.30, 100, minute1)
        dm._update_building_candle('XRP/USD', 2.35, 100, minute1)

        # Second minute - should trigger rollover
        minute2 = datetime(2024, 1, 1, 12, 1, 0)
        dm._update_building_candle('XRP/USD', 2.40, 100, minute2)

        # Old candle should be saved
        assert len(dm._candles_1m['XRP/USD']) == 1
        saved_candle = dm._candles_1m['XRP/USD'][0]
        assert saved_candle.open == 2.30
        assert saved_candle.high == 2.35
        assert saved_candle.close == 2.35

        # New candle building
        assert dm._current_candle_1m['XRP/USD']['open'] == 2.40


class TestDataManagerThreadSafety:
    """Test thread safety of DataManager."""

    @pytest.mark.asyncio
    async def test_async_snapshot(self):
        """Test async snapshot method."""
        dm = DataManager(['XRP/USD'])

        # Add some data
        await dm.on_message({
            'channel': 'trade',
            'data': [{
                'symbol': 'XRP/USD',
                'price': '2.35',
                'qty': '100',
                'side': 'buy',
                'timestamp': datetime.now().isoformat() + 'Z'
            }]
        })

        # Get async snapshot
        snapshot = await dm.get_snapshot_async()

        assert snapshot is not None
        assert snapshot.prices['XRP/USD'] == 2.35

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent message handling and snapshot access."""
        dm = DataManager(['XRP/USD'])

        errors = []

        async def send_messages():
            try:
                for i in range(50):
                    await dm.on_message({
                        'channel': 'trade',
                        'data': [{
                            'symbol': 'XRP/USD',
                            'price': str(2.30 + i * 0.01),
                            'qty': '100',
                            'side': 'buy',
                            'timestamp': datetime.now().isoformat() + 'Z'
                        }]
                    })
                    await asyncio.sleep(0.001)
            except Exception as e:
                errors.append(e)

        async def get_snapshots():
            try:
                for _ in range(50):
                    snapshot = await dm.get_snapshot_async()
                    if snapshot:
                        # Verify snapshot is valid
                        assert isinstance(snapshot.prices, dict)
                    await asyncio.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Run concurrently
        await asyncio.gather(
            send_messages(),
            get_snapshots(),
            get_snapshots()
        )

        assert len(errors) == 0, f"Errors: {errors}"


class TestSimulatedDataManager:
    """Test simulated data manager."""

    def test_simulated_creation(self):
        """Test simulated manager initialization."""
        sim = SimulatedDataManager(['XRP/USD', 'BTC/USD'])

        assert sim._prices['XRP/USD'] == 2.35
        assert sim._prices['BTC/USD'] == 104500.0

        # Should have orderbooks
        assert 'XRP/USD' in sim._orderbooks
        assert len(sim._orderbooks['XRP/USD']['bids']) == 10

    def test_custom_initial_prices(self):
        """Test custom initial prices."""
        sim = SimulatedDataManager(
            ['XRP/USD'],
            initial_prices={'XRP/USD': 3.00}
        )

        assert sim._prices['XRP/USD'] == 3.00

    @pytest.mark.asyncio
    async def test_simulate_tick(self):
        """Test price simulation."""
        sim = SimulatedDataManager(['XRP/USD'])

        initial_price = sim._prices['XRP/USD']

        # Run some ticks
        for _ in range(10):
            await sim.simulate_tick()

        # Price should have changed
        assert sim._prices['XRP/USD'] != initial_price

        # Should have trades
        assert len(sim._trades['XRP/USD']) > 0

    @pytest.mark.asyncio
    async def test_simulate_generates_orderbook(self):
        """Test that simulation updates orderbook."""
        sim = SimulatedDataManager(['XRP/USD'])

        await sim.simulate_tick()

        assert 'XRP/USD' in sim._orderbooks
        ob = sim._orderbooks['XRP/USD']
        assert len(ob['bids']) > 0
        assert len(ob['asks']) > 0

        # Best bid should be less than best ask
        assert ob['bids'][0][0] < ob['asks'][0][0]

    @pytest.mark.asyncio
    async def test_snapshot_after_simulation(self):
        """Test snapshot creation after simulation."""
        sim = SimulatedDataManager(['XRP/USD'])

        # Run simulation
        for _ in range(5):
            await sim.simulate_tick()

        # Get snapshot
        snapshot = sim.get_snapshot()

        assert snapshot is not None
        assert 'XRP/USD' in snapshot.prices
        assert 'XRP/USD' in snapshot.orderbooks
        assert len(snapshot.trades.get('XRP/USD', ())) > 0

    def test_staleness_check(self):
        """Test data staleness detection."""
        sim = SimulatedDataManager(['XRP/USD'])

        # Initially stale (no updates)
        sim._last_update = 0
        assert sim.is_stale() is True

    @pytest.mark.asyncio
    async def test_not_stale_after_tick(self):
        """Test not stale after tick."""
        sim = SimulatedDataManager(['XRP/USD'])

        await sim.simulate_tick()

        assert sim.is_stale(max_age_seconds=1.0) is False
