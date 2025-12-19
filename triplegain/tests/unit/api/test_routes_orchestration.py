"""
Unit tests for orchestration API routes.

Tests validate:
- Coordinator endpoints
- Portfolio endpoints
- Position endpoints
- Order endpoints
- Statistics endpoints
- Error handling and edge cases
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None
    FastAPI = None

# Skip all tests if FastAPI not available
pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")


# =============================================================================
# Authentication Helpers
# =============================================================================

def add_auth_override(app):
    """Add authentication override to app for testing."""
    from datetime import datetime, timezone
    from triplegain.src.api.security import get_current_user, User, UserRole

    async def override_get_current_user():
        return User(
            user_id="test-user-123",
            role=UserRole.ADMIN,  # Use ADMIN to pass all role checks
            api_key_hash="test-hash",
            created_at=datetime.now(timezone.utc),
        )

    app.dependency_overrides[get_current_user] = override_get_current_user
    return app


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_coordinator():
    """Create mock coordinator agent."""
    mock = MagicMock()
    mock.get_status.return_value = {
        "state": "running",
        "scheduled_tasks": [
            {"name": "technical_analysis", "enabled": True, "interval_seconds": 60},
            {"name": "regime_detection", "enabled": True, "interval_seconds": 300},
        ],
        "statistics": {
            "total_cycles": 100,
            "total_conflicts": 5,
            "total_signals_processed": 50,
        },
    }
    mock.pause = AsyncMock()
    mock.resume = AsyncMock()
    mock.force_run_task = AsyncMock(return_value=True)
    mock.enable_task.return_value = True
    mock.disable_task.return_value = True
    return mock


@pytest.fixture
def mock_portfolio_agent():
    """Create mock portfolio agent."""
    mock = MagicMock()
    mock.target_btc_pct = Decimal("33.33")
    mock.target_xrp_pct = Decimal("33.33")
    mock.target_usdt_pct = Decimal("33.34")
    mock.threshold_pct = Decimal("5.0")

    # Mock allocation
    allocation = MagicMock()
    allocation.total_equity_usd = Decimal("30000")
    allocation.btc_value_usd = Decimal("10000")
    allocation.xrp_value_usd = Decimal("10000")
    allocation.usdt_value_usd = Decimal("10000")
    allocation.btc_pct = Decimal("33.33")
    allocation.xrp_pct = Decimal("33.33")
    allocation.usdt_pct = Decimal("33.34")
    allocation.max_deviation_pct = Decimal("0.01")
    allocation.to_dict.return_value = {
        "total_equity_usd": 30000.0,
        "btc_value_usd": 10000.0,
        "xrp_value_usd": 10000.0,
        "usdt_value_usd": 10000.0,
        "btc_pct": 33.33,
        "xrp_pct": 33.33,
        "usdt_pct": 33.34,
        "max_deviation_pct": 0.01,
    }

    mock.check_allocation = AsyncMock(return_value=allocation)

    # Mock output
    output = MagicMock()
    output.action = "no_action"
    output.execution_strategy = "limit_orders"
    output.trades = []
    output.current_allocation = allocation
    output.reasoning = "Portfolio balanced"

    mock.process = AsyncMock(return_value=output)
    return mock


# Valid test UUIDs (reused across tests)
TEST_POSITION_UUID = "550e8400-e29b-41d4-a716-446655440001"
TEST_ORDER_UUID = "550e8400-e29b-41d4-a716-446655440002"
# UUID that doesn't exist in mock data (for "not found" tests)
NONEXISTENT_UUID = "550e8400-e29b-41d4-a716-446655449999"


@pytest.fixture
def mock_position_tracker():
    """Create mock position tracker."""
    mock = MagicMock()

    # Mock position with valid UUID
    position = MagicMock()
    position.id = TEST_POSITION_UUID
    position.symbol = "BTC/USDT"
    position.side = "long"
    position.entry_price = Decimal("45000")
    position.size = Decimal("0.1")
    position.current_price = Decimal("46000")
    position.unrealized_pnl = Decimal("100")
    position.status = "open"
    position.to_dict.return_value = {
        "id": TEST_POSITION_UUID,
        "symbol": "BTC/USDT",
        "side": "long",
        "entry_price": "45000",
        "size": "0.1",
        "current_price": "46000",
        "unrealized_pnl": "100",
        "status": "open",
    }

    mock.get_open_positions = AsyncMock(return_value=[position])
    mock.get_closed_positions = AsyncMock(return_value=[])
    mock.get_position = AsyncMock(return_value=position)
    mock.close_position = AsyncMock(return_value=position)
    mock.modify_position = AsyncMock(return_value=position)
    mock.get_total_exposure = AsyncMock(return_value={"total_exposure_usd": 4500.0})
    mock.get_total_unrealized_pnl = AsyncMock(return_value={"unrealized_pnl_usd": 100.0})
    mock.get_stats.return_value = {"total_positions": 10, "open_positions": 1}
    return mock


@pytest.fixture
def mock_order_manager():
    """Create mock order manager."""
    mock = MagicMock()

    # Mock order with valid UUID
    order = MagicMock()
    order.id = TEST_ORDER_UUID
    order.symbol = "BTC/USDT"
    order.side = "buy"
    order.order_type = "limit"
    order.size = Decimal("0.1")
    order.price = Decimal("45000")
    order.status = "open"
    order.to_dict.return_value = {
        "id": TEST_ORDER_UUID,
        "symbol": "BTC/USDT",
        "side": "buy",
        "order_type": "limit",
        "size": "0.1",
        "price": "45000",
        "status": "open",
    }

    mock.get_open_orders = AsyncMock(return_value=[order])
    mock.get_order = AsyncMock(return_value=order)
    mock.cancel_order = AsyncMock(return_value=True)
    mock.sync_with_exchange = AsyncMock(return_value=5)
    mock.get_stats.return_value = {"total_orders_placed": 50, "total_orders_filled": 45}
    return mock


@pytest.fixture
def mock_message_bus():
    """Create mock message bus."""
    mock = MagicMock()
    mock.get_stats.return_value = {"messages_published": 1000, "active_subscribers": 5}
    return mock


@pytest.fixture
def app(mock_coordinator, mock_portfolio_agent, mock_position_tracker, mock_order_manager, mock_message_bus):
    """Create test FastAPI app with orchestration router."""
    from triplegain.src.api.routes_orchestration import create_orchestration_router

    app = FastAPI()
    router = create_orchestration_router(
        coordinator=mock_coordinator,
        portfolio_agent=mock_portfolio_agent,
        position_tracker=mock_position_tracker,
        order_manager=mock_order_manager,
        message_bus=mock_message_bus,
    )
    app.include_router(router)

    # Add authentication override for testing
    add_auth_override(app)

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


# =============================================================================
# Coordinator Endpoint Tests
# =============================================================================

class TestCoordinatorEndpoints:
    """Tests for coordinator endpoints."""

    def test_get_coordinator_status(self, client):
        """Test getting coordinator status."""
        response = client.get("/api/v1/coordinator/status")
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "running"
        assert "scheduled_tasks" in data
        assert "statistics" in data

    def test_pause_coordinator(self, client, mock_coordinator):
        """Test pausing coordinator."""
        response = client.post("/api/v1/coordinator/pause")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"
        mock_coordinator.pause.assert_called_once()

    def test_resume_coordinator(self, client, mock_coordinator):
        """Test resuming coordinator."""
        response = client.post("/api/v1/coordinator/resume")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        mock_coordinator.resume.assert_called_once()

    def test_force_run_task(self, client, mock_coordinator):
        """Test force running a task."""
        response = client.post("/api/v1/coordinator/task/technical_analysis/run")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        mock_coordinator.force_run_task.assert_called_once()

    def test_force_run_task_not_found(self, client, mock_coordinator):
        """Test force running task that fails."""
        mock_coordinator.force_run_task = AsyncMock(return_value=False)
        # Use valid task name - coordinator returns False indicating task not scheduled
        response = client.post("/api/v1/coordinator/task/regime_detection/run")
        assert response.status_code == 404

    def test_enable_task(self, client, mock_coordinator):
        """Test enabling a task."""
        # Use valid task name from VALID_COORDINATOR_TASKS
        response = client.post("/api/v1/coordinator/task/regime_detection/enable")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "enabled"

    def test_enable_task_not_found(self, client, mock_coordinator):
        """Test enabling task that doesn't exist in scheduler."""
        mock_coordinator.enable_task.return_value = False
        # Use valid task name - coordinator returns False indicating task not found
        response = client.post("/api/v1/coordinator/task/portfolio_rebalance/enable")
        assert response.status_code == 404

    def test_disable_task(self, client, mock_coordinator):
        """Test disabling a task."""
        response = client.post("/api/v1/coordinator/task/technical_analysis/disable")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "disabled"

    def test_disable_task_not_found(self, client, mock_coordinator):
        """Test disabling task that doesn't exist in scheduler."""
        mock_coordinator.disable_task.return_value = False
        # Use valid task name - coordinator returns False indicating task not found
        response = client.post("/api/v1/coordinator/task/risk_check/disable")
        assert response.status_code == 404


# =============================================================================
# Portfolio Endpoint Tests
# =============================================================================

class TestPortfolioEndpoints:
    """Tests for portfolio endpoints."""

    def test_get_portfolio_allocation(self, client, mock_portfolio_agent):
        """Test getting portfolio allocation."""
        response = client.get("/api/v1/portfolio/allocation")
        assert response.status_code == 200
        data = response.json()
        assert "allocation" in data
        assert "target" in data
        assert "threshold_pct" in data
        assert "needs_rebalancing" in data
        assert data["target"]["btc_pct"] == 33.33

    def test_force_rebalance(self, client, mock_portfolio_agent):
        """Test forcing portfolio rebalance."""
        # First get a confirmation token
        confirm_response = client.get("/api/v1/portfolio/rebalance/confirm")
        assert confirm_response.status_code == 200
        token = confirm_response.json()["confirmation_token"]

        # Now rebalance with the token
        response = client.post(
            "/api/v1/portfolio/rebalance",
            json={"execution_strategy": "immediate", "confirmation_token": token}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_action"
        assert "trades" in data
        mock_portfolio_agent.process.assert_called_once_with(force=True)

    def test_force_rebalance_default_strategy(self, client, mock_portfolio_agent):
        """Test forcing rebalance with default strategy."""
        # First get a confirmation token
        confirm_response = client.get("/api/v1/portfolio/rebalance/confirm")
        assert confirm_response.status_code == 200
        token = confirm_response.json()["confirmation_token"]

        response = client.post(
            "/api/v1/portfolio/rebalance",
            json={"confirmation_token": token}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_action"


# =============================================================================
# Position Endpoint Tests
# =============================================================================

class TestPositionEndpoints:
    """Tests for position endpoints."""

    def test_get_positions_open(self, client, mock_position_tracker):
        """Test getting open positions."""
        response = client.get("/api/v1/positions?status=open")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["positions"]) == 1
        assert data["positions"][0]["id"] == "550e8400-e29b-41d4-a716-446655440001"

    def test_get_positions_closed(self, client, mock_position_tracker):
        """Test getting closed positions."""
        response = client.get("/api/v1/positions?status=closed")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0

    def test_get_positions_all(self, client, mock_position_tracker):
        """Test getting all positions."""
        response = client.get("/api/v1/positions?status=all")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1

    def test_get_positions_by_symbol(self, client, mock_position_tracker):
        """Test getting positions filtered by symbol."""
        response = client.get("/api/v1/positions?symbol=BTC/USDT")
        assert response.status_code == 200
        mock_position_tracker.get_open_positions.assert_called_with("BTC/USDT")

    def test_get_position(self, client, mock_position_tracker):
        """Test getting specific position."""
        response = client.get("/api/v1/positions/550e8400-e29b-41d4-a716-446655440001")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "550e8400-e29b-41d4-a716-446655440001"

    def test_get_position_not_found(self, client, mock_position_tracker):
        """Test getting non-existent position."""
        mock_position_tracker.get_position.return_value = None
        response = client.get(f"/api/v1/positions/{NONEXISTENT_UUID}")
        assert response.status_code == 404

    def test_close_position(self, client, mock_position_tracker):
        """Test closing a position."""
        # First get a confirmation token
        confirm_response = client.get(f"/api/v1/positions/{TEST_POSITION_UUID}/confirm")
        assert confirm_response.status_code == 200
        token = confirm_response.json()["confirmation_token"]

        # Now close with the token
        response = client.post(
            f"/api/v1/positions/{TEST_POSITION_UUID}/close",
            json={"exit_price": 46000.0, "reason": "take_profit", "confirmation_token": token}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "closed"
        mock_position_tracker.close_position.assert_called_once()

    def test_close_position_not_found(self, client, mock_position_tracker):
        """Test closing non-existent position - fails at confirmation step."""
        # First try to get a confirmation token - this should fail for nonexistent position
        mock_position_tracker.get_position.return_value = None
        response = client.get(f"/api/v1/positions/{NONEXISTENT_UUID}/confirm")
        assert response.status_code == 404

    def test_modify_position(self, client, mock_position_tracker):
        """Test modifying position stop-loss/take-profit."""
        response = client.patch(
            "/api/v1/positions/550e8400-e29b-41d4-a716-446655440001",
            json={"stop_loss": 44000.0, "take_profit": 48000.0}
        )
        assert response.status_code == 200
        mock_position_tracker.modify_position.assert_called_once()

    def test_modify_position_not_found(self, client, mock_position_tracker):
        """Test modifying non-existent position."""
        mock_position_tracker.modify_position.return_value = None
        response = client.patch(
            f"/api/v1/positions/{NONEXISTENT_UUID}",
            json={"stop_loss": 44000.0}
        )
        assert response.status_code == 404

    def test_get_exposure(self, client, mock_position_tracker):
        """Test getting portfolio exposure."""
        response = client.get("/api/v1/positions/exposure")
        assert response.status_code == 200
        data = response.json()
        assert "total_exposure_usd" in data
        assert "unrealized_pnl_usd" in data


# =============================================================================
# Order Endpoint Tests
# =============================================================================

class TestOrderEndpoints:
    """Tests for order endpoints."""

    def test_get_orders(self, client, mock_order_manager):
        """Test getting open orders."""
        response = client.get("/api/v1/orders")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["orders"]) == 1

    def test_get_orders_by_symbol(self, client, mock_order_manager):
        """Test getting orders filtered by symbol."""
        response = client.get("/api/v1/orders?symbol=BTC/USDT")
        assert response.status_code == 200
        mock_order_manager.get_open_orders.assert_called_with("BTC/USDT")

    def test_get_order(self, client, mock_order_manager):
        """Test getting specific order."""
        response = client.get("/api/v1/orders/550e8400-e29b-41d4-a716-446655440002")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "550e8400-e29b-41d4-a716-446655440002"

    def test_get_order_not_found(self, client, mock_order_manager):
        """Test getting non-existent order."""
        mock_order_manager.get_order.return_value = None
        response = client.get(f"/api/v1/orders/{NONEXISTENT_UUID}")
        assert response.status_code == 404

    def test_cancel_order(self, client, mock_order_manager):
        """Test cancelling an order."""
        # First get a confirmation token
        confirm_response = client.get(f"/api/v1/orders/{TEST_ORDER_UUID}/confirm")
        assert confirm_response.status_code == 200
        token = confirm_response.json()["confirmation_token"]

        # Now cancel with the token
        response = client.post(
            f"/api/v1/orders/{TEST_ORDER_UUID}/cancel",
            json={"confirmation_token": token, "reason": "test_cancel"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    def test_cancel_order_failed(self, client, mock_order_manager):
        """Test cancelling order that fails."""
        # First get a confirmation token
        confirm_response = client.get(f"/api/v1/orders/{TEST_ORDER_UUID}/confirm")
        assert confirm_response.status_code == 200
        token = confirm_response.json()["confirmation_token"]

        # Make the cancel operation fail
        mock_order_manager.cancel_order.return_value = False
        response = client.post(
            f"/api/v1/orders/{TEST_ORDER_UUID}/cancel",
            json={"confirmation_token": token, "reason": "test_cancel"}
        )
        assert response.status_code == 400

    def test_sync_orders(self, client, mock_order_manager):
        """Test syncing orders with exchange."""
        response = client.post("/api/v1/orders/sync")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "synced"
        assert data["orders_synced"] == 5


# =============================================================================
# Statistics Endpoint Tests
# =============================================================================

class TestStatisticsEndpoints:
    """Tests for statistics endpoints."""

    def test_get_execution_stats(self, client):
        """Test getting execution statistics."""
        response = client.get("/api/v1/stats/execution")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "orders" in data
        assert "positions" in data
        assert "message_bus" in data


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_coordinator_not_initialized(self):
        """Test endpoints when coordinator not initialized."""
        from triplegain.src.api.routes_orchestration import create_orchestration_router

        app = FastAPI()
        router = create_orchestration_router(coordinator=None)
        app.include_router(router)
        add_auth_override(app)
        client = TestClient(app)

        response = client.get("/api/v1/coordinator/status")
        assert response.status_code == 503

    def test_portfolio_agent_not_initialized(self):
        """Test endpoints when portfolio agent not initialized."""
        from triplegain.src.api.routes_orchestration import create_orchestration_router

        app = FastAPI()
        router = create_orchestration_router(portfolio_agent=None)
        app.include_router(router)
        add_auth_override(app)
        client = TestClient(app)

        response = client.get("/api/v1/portfolio/allocation")
        assert response.status_code == 503

    def test_position_tracker_not_initialized(self):
        """Test endpoints when position tracker not initialized."""
        from triplegain.src.api.routes_orchestration import create_orchestration_router

        app = FastAPI()
        router = create_orchestration_router(position_tracker=None)
        app.include_router(router)
        add_auth_override(app)
        client = TestClient(app)

        response = client.get("/api/v1/positions")
        assert response.status_code == 503

    def test_order_manager_not_initialized(self):
        """Test endpoints when order manager not initialized."""
        from triplegain.src.api.routes_orchestration import create_orchestration_router

        app = FastAPI()
        router = create_orchestration_router(order_manager=None)
        app.include_router(router)
        add_auth_override(app)
        client = TestClient(app)

        response = client.get("/api/v1/orders")
        assert response.status_code == 503

    def test_coordinator_exception(self, client, mock_coordinator):
        """Test error handling when coordinator raises exception."""
        mock_coordinator.get_status.side_effect = Exception("Internal error")
        response = client.get("/api/v1/coordinator/status")
        assert response.status_code == 500

    def test_portfolio_exception(self, client, mock_portfolio_agent):
        """Test error handling when portfolio agent raises exception."""
        mock_portfolio_agent.check_allocation.side_effect = Exception("Allocation error")
        response = client.get("/api/v1/portfolio/allocation")
        assert response.status_code == 500

    def test_position_exception(self, client, mock_position_tracker):
        """Test error handling when position tracker raises exception."""
        mock_position_tracker.get_open_positions.side_effect = Exception("Position error")
        response = client.get("/api/v1/positions")
        assert response.status_code == 500

    def test_order_exception(self, client, mock_order_manager):
        """Test error handling when order manager raises exception."""
        mock_order_manager.get_open_orders.side_effect = Exception("Order error")
        response = client.get("/api/v1/orders")
        assert response.status_code == 500


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_stats(self):
        """Test stats with no components initialized."""
        from triplegain.src.api.routes_orchestration import create_orchestration_router

        app = FastAPI()
        router = create_orchestration_router()
        app.include_router(router)
        add_auth_override(app)
        client = TestClient(app)

        response = client.get("/api/v1/stats/execution")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data

    def test_force_run_task_with_symbol(self, client, mock_coordinator):
        """Test force running task with custom symbol."""
        response = client.post("/api/v1/coordinator/task/technical_analysis/run?symbol=XRP/USDT")
        assert response.status_code == 200
        mock_coordinator.force_run_task.assert_called_with("technical_analysis", "XRP/USDT")

    def test_modify_position_partial(self, client, mock_position_tracker):
        """Test modifying only stop-loss."""
        response = client.patch(
            "/api/v1/positions/550e8400-e29b-41d4-a716-446655440001",
            json={"stop_loss": 44000.0}
        )
        assert response.status_code == 200

    def test_rebalance_with_trades(self, client, mock_portfolio_agent):
        """Test rebalancing that returns trades."""
        # Mock a rebalance that returns trades
        trade_mock = MagicMock()
        trade_mock.to_dict.return_value = {"symbol": "BTC/USDT", "action": "sell", "amount_usd": 500}

        output = MagicMock()
        output.action = "rebalance"
        output.execution_strategy = "limit_orders"
        output.trades = [trade_mock]
        output.current_allocation = mock_portfolio_agent.check_allocation.return_value
        output.reasoning = "Deviation exceeds threshold"

        mock_portfolio_agent.process.return_value = output

        # First get a confirmation token
        confirm_response = client.get("/api/v1/portfolio/rebalance/confirm")
        assert confirm_response.status_code == 200
        token = confirm_response.json()["confirmation_token"]

        response = client.post(
            "/api/v1/portfolio/rebalance",
            json={"confirmation_token": token}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rebalance"
        assert len(data["trades"]) == 1
