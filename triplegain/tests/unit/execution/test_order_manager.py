"""
Unit tests for OrderExecutionManager - Order lifecycle management.

Tests cover:
- Order creation and serialization
- Order placement with retry logic
- Order monitoring
- Contingent orders (stop-loss, take-profit)
- Order cancellation
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from triplegain.src.execution.order_manager import (
    Order,
    OrderExecutionManager,
    OrderStatus,
    OrderType,
    OrderSide,
    ExecutionResult,
    SYMBOL_MAP,
)


class TestOrder:
    """Tests for Order dataclass."""

    def test_order_creation(self):
        """Test basic order creation."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("0.1"),
            price=Decimal("45000"),
        )
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.size == Decimal("0.1")
        assert order.price == Decimal("45000")
        assert order.status == OrderStatus.PENDING

    def test_order_with_stop_price(self):
        """Test order with stop price."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            size=Decimal("0.1"),
            stop_price=Decimal("44000"),
        )
        assert order.stop_price == Decimal("44000")

    def test_order_with_leverage(self):
        """Test order with leverage."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
            leverage=5,
        )
        assert order.leverage == 5

    def test_order_to_dict(self):
        """Test order serialization to dictionary."""
        order = Order(
            id="test-123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("0.1"),
            price=Decimal("45000"),
            status=OrderStatus.OPEN,
            external_id="kraken-123",
        )
        d = order.to_dict()
        assert d["id"] == "test-123"
        assert d["symbol"] == "BTC/USDT"
        assert d["side"] == "buy"
        assert d["order_type"] == "limit"
        assert d["size"] == "0.1"
        assert d["price"] == "45000"
        assert d["status"] == "open"
        assert d["external_id"] == "kraken-123"

    def test_order_timestamps(self):
        """Test order timestamps."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
        )
        assert order.created_at is not None
        assert order.updated_at is not None


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        expected = ["pending", "open", "partially_filled", "filled", "cancelled", "expired", "error"]
        status_values = [s.value for s in OrderStatus]
        for expected_status in expected:
            assert expected_status in status_values


class TestOrderType:
    """Tests for OrderType enum."""

    def test_all_types_exist(self):
        """Test all expected types exist."""
        expected = ["market", "limit", "stop-loss", "take-profit"]
        type_values = [t.value for t in OrderType]
        for expected_type in expected:
            assert expected_type in type_values


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_result(self):
        """Test successful execution result."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.1"),
        )
        result = ExecutionResult(
            success=True,
            order=order,
            position_id="pos-123",
            execution_time_ms=150,
        )
        assert result.success is True
        assert result.order is not None
        assert result.position_id == "pos-123"

    def test_failure_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            error_message="Insufficient margin",
            execution_time_ms=50,
        )
        assert result.success is False
        assert result.error_message == "Insufficient margin"

    def test_result_to_dict(self):
        """Test execution result serialization."""
        result = ExecutionResult(
            success=True,
            position_id="pos-123",
            execution_time_ms=100,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["position_id"] == "pos-123"
        assert d["execution_time_ms"] == 100


class TestSymbolMapping:
    """Tests for symbol mapping."""

    def test_btc_usdt_mapping(self):
        """Test BTC/USDT mapping."""
        assert SYMBOL_MAP["BTC/USDT"] == "XBTUSDT"

    def test_xrp_usdt_mapping(self):
        """Test XRP/USDT mapping."""
        assert SYMBOL_MAP["XRP/USDT"] == "XRPUSDT"


@pytest.fixture
def mock_kraken_client():
    """Create a mock Kraken client."""
    client = AsyncMock()
    client.add_order = AsyncMock(return_value={
        "result": {"txid": ["OTEST-12345"]},
    })
    client.cancel_order = AsyncMock(return_value={"result": {"count": 1}})
    client.query_orders = AsyncMock(return_value={
        "result": {"OTEST-12345": {"status": "closed", "vol_exec": "0.1", "price": "45000"}},
    })
    client.open_orders = AsyncMock(return_value={"result": {"open": {}}})
    return client


@pytest.fixture
def mock_message_bus():
    """Create a mock message bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock(return_value=1)
    return bus


@pytest.fixture
def order_manager_config():
    """Create order manager configuration."""
    return {
        "orders": {
            "default_type": "limit",
            "time_in_force": "GTC",
            "max_retry_count": 3,
            "retry_delay_seconds": 1,
        },
        "kraken": {
            "rate_limit": {
                "calls_per_minute": 60,
                "order_calls_per_minute": 30,
            },
        },
    }


class TestOrderExecutionManager:
    """Tests for OrderExecutionManager."""

    @pytest.fixture
    def order_manager(self, mock_kraken_client, mock_message_bus, order_manager_config):
        """Create an order manager instance for testing."""
        return OrderExecutionManager(
            kraken_client=mock_kraken_client,
            message_bus=mock_message_bus,
            position_tracker=None,
            db_pool=None,
            config=order_manager_config,
        )

    def test_manager_creation(self, order_manager):
        """Test order manager creation."""
        assert order_manager._default_order_type == "limit"
        assert order_manager._max_retries == 3

    @pytest.mark.asyncio
    async def test_execute_trade_success(self, order_manager, mock_kraken_client):
        """Test successful trade execution."""
        from triplegain.src.risk.rules_engine import TradeProposal

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000,
            entry_price=45000,
            stop_loss=44000,
            take_profit=47000,
            leverage=1,
            confidence=0.8,
        )

        result = await order_manager.execute_trade(proposal)

        assert result.success is True
        assert result.order is not None
        assert result.order.status == OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_execute_trade_no_kraken_mock_mode(self):
        """Test trade execution in mock mode (no Kraken client)."""
        manager = OrderExecutionManager(
            kraken_client=None,
            message_bus=None,
            config={"orders": {"max_retry_count": 1}},
        )

        from triplegain.src.risk.rules_engine import TradeProposal

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000,
            entry_price=45000,
            leverage=1,
            confidence=0.8,
        )

        result = await manager.execute_trade(proposal)
        assert result.success is True
        assert "mock_" in result.order.external_id

    @pytest.mark.asyncio
    async def test_place_order_retry_on_timeout(self, order_manager, mock_kraken_client):
        """Test order placement retries on timeout."""
        mock_kraken_client.add_order.side_effect = [
            asyncio.TimeoutError(),
            {"result": {"txid": ["OTEST-12345"]}},
        ]

        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("0.1"),
            price=Decimal("45000"),
        )

        success = await order_manager._place_order(order)
        assert success is True
        assert mock_kraken_client.add_order.call_count == 2

    @pytest.mark.asyncio
    async def test_place_order_no_retry_on_invalid_error(self, order_manager, mock_kraken_client):
        """Test order placement does not retry on invalid errors."""
        mock_kraken_client.add_order.return_value = {
            "error": ["Invalid order parameters"],
        }

        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("0.1"),
            price=Decimal("45000"),
        )

        success = await order_manager._place_order(order)
        assert success is False
        assert order.status == OrderStatus.ERROR
        # Should not retry on Invalid errors
        assert mock_kraken_client.add_order.call_count == 1

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, order_manager, mock_kraken_client):
        """Test successful order cancellation."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("0.1"),
            price=Decimal("45000"),
            status=OrderStatus.OPEN,
            external_id="OTEST-12345",
        )
        order_manager._open_orders[order.id] = order

        success = await order_manager.cancel_order(order.id)
        assert success is True
        assert order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, order_manager):
        """Test cancelling non-existent order."""
        success = await order_manager.cancel_order("nonexistent-id")
        assert success is False

    @pytest.mark.asyncio
    async def test_cancel_order_already_filled(self, order_manager):
        """Test cancelling already filled order."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("0.1"),
            price=Decimal("45000"),
            status=OrderStatus.FILLED,
            external_id="OTEST-12345",
        )
        order_manager._open_orders[order.id] = order

        success = await order_manager.cancel_order(order.id)
        assert success is False

    @pytest.mark.asyncio
    async def test_get_open_orders(self, order_manager):
        """Test getting open orders."""
        order1 = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("0.1"),
        )
        order2 = Order(
            id=str(uuid.uuid4()),
            symbol="XRP/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            size=Decimal("100"),
        )
        order_manager._open_orders[order1.id] = order1
        order_manager._open_orders[order2.id] = order2

        # All orders
        orders = await order_manager.get_open_orders()
        assert len(orders) == 2

        # Filter by symbol
        btc_orders = await order_manager.get_open_orders("BTC/USDT")
        assert len(btc_orders) == 1
        assert btc_orders[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_get_order(self, order_manager):
        """Test getting a specific order."""
        order = Order(
            id="test-order-id",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("0.1"),
        )
        order_manager._open_orders[order.id] = order

        found = await order_manager.get_order("test-order-id")
        assert found is not None
        assert found.id == "test-order-id"

        not_found = await order_manager.get_order("nonexistent")
        assert not_found is None

    def test_to_kraken_symbol(self, order_manager):
        """Test symbol conversion to Kraken format."""
        assert order_manager._to_kraken_symbol("BTC/USDT") == "XBTUSDT"
        assert order_manager._to_kraken_symbol("XRP/USDT") == "XRPUSDT"
        assert order_manager._to_kraken_symbol("UNKNOWN") == "UNKNOWN"

    def test_from_kraken_symbol(self, order_manager):
        """Test symbol conversion from Kraken format."""
        assert order_manager._from_kraken_symbol("XBTUSDT") == "BTC/USDT"
        assert order_manager._from_kraken_symbol("XRPUSDT") == "XRP/USDT"

    @pytest.mark.asyncio
    async def test_calculate_size(self, order_manager):
        """Test size calculation from USD amount."""
        from triplegain.src.risk.rules_engine import TradeProposal

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=4500,
            entry_price=45000,
            leverage=1,
            confidence=0.8,
        )

        size = await order_manager._calculate_size(proposal)
        assert size == Decimal("0.1")  # 4500 / 45000 = 0.1

    def test_get_stats(self, order_manager):
        """Test getting execution statistics."""
        stats = order_manager.get_stats()
        assert "total_orders_placed" in stats
        assert "total_orders_filled" in stats
        assert "total_orders_cancelled" in stats
        assert "total_errors" in stats
        assert "open_orders_count" in stats

    @pytest.mark.asyncio
    async def test_sync_with_exchange(self, order_manager, mock_kraken_client):
        """Test syncing with exchange."""
        mock_kraken_client.open_orders.return_value = {
            "result": {"open": {"OTEST-1": {}, "OTEST-2": {}}}
        }

        synced = await order_manager.sync_with_exchange()
        assert synced == 2


class TestOrderManagerWithPositionTracker:
    """Tests for OrderExecutionManager with PositionTracker."""

    @pytest.mark.asyncio
    async def test_handle_order_fill_creates_position(self):
        """Test order fill creates a position."""
        position_tracker = AsyncMock()
        position_tracker.open_position = AsyncMock(return_value=MagicMock(id="pos-123"))

        manager = OrderExecutionManager(
            kraken_client=None,
            message_bus=AsyncMock(),
            position_tracker=position_tracker,
            config={"orders": {}},
        )

        from triplegain.src.risk.rules_engine import TradeProposal

        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("0.1"),
            filled_size=Decimal("0.1"),
            filled_price=Decimal("45000"),
        )
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=4500,
            entry_price=45000,
            stop_loss=44000,
            take_profit=47000,
        )

        await manager._handle_order_fill(order, proposal)

        position_tracker.open_position.assert_called_once()
