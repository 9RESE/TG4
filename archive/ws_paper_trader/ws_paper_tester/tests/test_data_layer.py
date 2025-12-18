"""Tests for data layer (HIGH-005)."""

import pytest
import asyncio
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.data_layer import DataManager, SimulatedDataManager
from ws_tester.types import Trade, Candle


class TestDataManager:
    """Tests for DataManager class."""

    def test_data_manager_creation(self):
        """Test DataManager initialization."""
        symbols = ['XRP/USD', 'BTC/USD']
        dm = DataManager(symbols)

        assert dm.symbols == symbols
        assert dm._prices == {}
        assert 'XRP/USD' in dm._trades
        assert 'BTC/USD' in dm._trades

    def test_get_snapshot_returns_none_when_empty(self):
        """Test that get_snapshot returns None when no data."""
        dm = DataManager(['XRP/USD'])
        snapshot = dm.get_snapshot()
        assert snapshot is None

    def test_get_price(self):
        """Test price retrieval."""
        dm = DataManager(['XRP/USD'])
        assert dm.get_price('XRP/USD') == 0.0  # No price yet

        dm._prices['XRP/USD'] = 2.35
        assert dm.get_price('XRP/USD') == 2.35

    @pytest.mark.asyncio
    async def test_handle_trade_message(self):
        """Test processing trade messages."""
        dm = DataManager(['XRP/USD'])

        # Simulate a trade message
        trade_msg = {
            'channel': 'trade',
            'data': [{
                'symbol': 'XRP/USD',
                'price': '2.35',
                'qty': '100',
                'side': 'buy',
                'timestamp': datetime.now().isoformat() + 'Z'
            }]
        }

        await dm.on_message(trade_msg)

        assert dm._prices.get('XRP/USD') == 2.35
        assert len(dm._trades['XRP/USD']) == 1

    @pytest.mark.asyncio
    async def test_handle_ticker_message(self):
        """Test processing ticker messages."""
        dm = DataManager(['XRP/USD'])

        ticker_msg = {
            'channel': 'ticker',
            'data': [{
                'symbol': 'XRP/USD',
                'last': '2.35'
            }]
        }

        await dm.on_message(ticker_msg)
        assert dm._prices.get('XRP/USD') == 2.35

    @pytest.mark.asyncio
    async def test_handle_book_message(self):
        """Test processing orderbook messages."""
        dm = DataManager(['XRP/USD'])

        book_msg = {
            'channel': 'book',
            'data': [{
                'symbol': 'XRP/USD',
                'bids': [{'price': '2.34', 'qty': '100'}],
                'asks': [{'price': '2.36', 'qty': '150'}]
            }]
        }

        await dm.on_message(book_msg)
        assert 'XRP/USD' in dm._orderbooks
        assert dm._orderbooks['XRP/USD']['bids'][0] == (2.34, 100.0)
        assert dm._orderbooks['XRP/USD']['asks'][0] == (2.36, 150.0)

    def test_get_snapshot_with_data(self):
        """Test snapshot creation with data."""
        dm = DataManager(['XRP/USD'])
        dm._prices['XRP/USD'] = 2.35
        dm._orderbooks['XRP/USD'] = {
            'bids': [(2.34, 100.0)],
            'asks': [(2.36, 150.0)]
        }

        snapshot = dm.get_snapshot()

        assert snapshot is not None
        assert snapshot.prices['XRP/USD'] == 2.35
        assert 'XRP/USD' in snapshot.orderbooks

    @pytest.mark.asyncio
    async def test_candle_building(self):
        """Test that candles are built from trades."""
        dm = DataManager(['XRP/USD'])

        # Simulate multiple trades
        for price in [2.35, 2.36, 2.34, 2.37]:
            trade_msg = {
                'channel': 'trade',
                'data': [{
                    'symbol': 'XRP/USD',
                    'price': str(price),
                    'qty': '100',
                    'side': 'buy',
                    'timestamp': datetime.now().isoformat() + 'Z'
                }]
            }
            await dm.on_message(trade_msg)

        # Check that building candle is being updated
        assert 'XRP/USD' in dm._current_candle_1m


class TestSimulatedDataManager:
    """Tests for SimulatedDataManager class."""

    def test_simulated_creation(self):
        """Test SimulatedDataManager initialization."""
        symbols = ['XRP/USD', 'BTC/USD']
        sdm = SimulatedDataManager(symbols)

        assert sdm.symbols == symbols
        # Check that initial prices are set
        assert 'XRP/USD' in sdm._prices
        assert 'BTC/USD' in sdm._prices

    @pytest.mark.asyncio
    async def test_simulate_tick(self):
        """Test simulated data generation."""
        sdm = SimulatedDataManager(['XRP/USD'])
        initial_price = sdm._prices.get('XRP/USD')

        await sdm.simulate_tick()

        # Price should have changed slightly
        new_price = sdm._prices.get('XRP/USD')
        assert new_price is not None
        # Trades should have been added
        assert len(sdm._trades['XRP/USD']) > 0

    @pytest.mark.asyncio
    async def test_simulate_generates_orderbook(self):
        """Test that simulation generates orderbook."""
        sdm = SimulatedDataManager(['XRP/USD'])
        await sdm.simulate_tick()

        assert 'XRP/USD' in sdm._orderbooks
        orderbook = sdm._orderbooks['XRP/USD']
        assert len(orderbook.get('bids', [])) > 0
        assert len(orderbook.get('asks', [])) > 0

    @pytest.mark.asyncio
    async def test_snapshot_after_simulation(self):
        """Test snapshot after simulation."""
        sdm = SimulatedDataManager(['XRP/USD'])
        await sdm.simulate_tick()

        snapshot = sdm.get_snapshot()

        assert snapshot is not None
        assert 'XRP/USD' in snapshot.prices
        assert 'XRP/USD' in snapshot.orderbooks


class TestDataManagerThreadSafety:
    """Tests for thread safety features (MED-007)."""

    @pytest.mark.asyncio
    async def test_async_snapshot(self):
        """Test async snapshot creation."""
        dm = DataManager(['XRP/USD'])
        dm._prices['XRP/USD'] = 2.35

        snapshot = await dm.get_snapshot_async()

        assert snapshot is not None
        assert snapshot.prices['XRP/USD'] == 2.35

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test that concurrent operations don't cause issues."""
        dm = DataManager(['XRP/USD'])

        async def add_trades():
            for i in range(10):
                trade_msg = {
                    'channel': 'trade',
                    'data': [{
                        'symbol': 'XRP/USD',
                        'price': str(2.35 + i * 0.01),
                        'qty': '100',
                        'side': 'buy',
                        'timestamp': datetime.now().isoformat() + 'Z'
                    }]
                }
                await dm.on_message(trade_msg)
                await asyncio.sleep(0.001)

        async def read_snapshots():
            for _ in range(10):
                snapshot = await dm.get_snapshot_async()
                await asyncio.sleep(0.001)

        # Run concurrently - should not raise any exceptions
        await asyncio.gather(add_trades(), read_snapshots())

        # Verify data integrity
        assert len(dm._trades['XRP/USD']) == 10
