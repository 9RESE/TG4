"""Tests for dashboard server."""

import pytest
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test client
try:
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDashboardAPI:
    """Test REST API endpoints."""

    def setup_method(self):
        """Reset state before each test."""
        from ws_tester.dashboard.server import latest_state, _state_lock
        with _state_lock:
            latest_state.clear()
            latest_state.update({
                "timestamp": None,
                "prices": {},
                "strategies": [],
                "aggregate": {},
                "recent_trades": [],
                "session_info": {}
            })

    def test_get_strategies_empty(self):
        """Test strategies endpoint with no strategies."""
        from ws_tester.dashboard.server import app
        client = TestClient(app)

        response = client.get("/api/strategies")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_strategies_with_data(self):
        """Test strategies endpoint with data."""
        from ws_tester.dashboard.server import app, update_state

        # Add test data
        update_state(
            prices={"XRP/USD": 2.35},
            strategies=[
                {"strategy": "test_strategy", "equity": 105.0, "pnl": 5.0}
            ],
            aggregate={"total_equity": 105.0}
        )

        client = TestClient(app)
        response = client.get("/api/strategies")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["strategy"] == "test_strategy"
        assert data[0]["equity"] == 105.0

    def test_get_strategy_detail(self):
        """Test individual strategy endpoint."""
        from ws_tester.dashboard.server import app, update_state

        update_state(
            prices={},
            strategies=[
                {"strategy": "strategy_a", "equity": 100.0},
                {"strategy": "strategy_b", "equity": 110.0}
            ],
            aggregate={}
        )

        client = TestClient(app)

        # Get existing strategy
        response = client.get("/api/strategy/strategy_b")
        assert response.status_code == 200
        assert response.json()["strategy"] == "strategy_b"

        # Get non-existent strategy
        response = client.get("/api/strategy/nonexistent")
        assert response.status_code == 200
        assert "error" in response.json()

    def test_get_trades(self):
        """Test trades endpoint."""
        from ws_tester.dashboard.server import app, add_trade

        # Add test trades
        for i in range(10):
            add_trade({
                "timestamp": datetime.now().isoformat(),
                "symbol": "XRP/USD",
                "side": "buy" if i % 2 == 0 else "sell",
                "price": 2.35 + i * 0.01,
                "strategy": "test"
            })

        client = TestClient(app)

        # Get all trades
        response = client.get("/api/trades")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 10

        # Get limited trades
        response = client.get("/api/trades?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_get_aggregate(self):
        """Test aggregate endpoint."""
        from ws_tester.dashboard.server import app, update_state

        update_state(
            prices={"XRP/USD": 2.35},
            strategies=[],
            aggregate={
                "total_equity": 300.0,
                "total_pnl": 15.0,
                "total_strategies": 3,
                "win_rate": 60.0
            }
        )

        client = TestClient(app)
        response = client.get("/api/aggregate")

        assert response.status_code == 200
        data = response.json()
        assert data["total_equity"] == 300.0
        assert data["total_pnl"] == 15.0

    def test_get_prices(self):
        """Test prices endpoint."""
        from ws_tester.dashboard.server import app, update_state

        update_state(
            prices={"XRP/USD": 2.35, "BTC/USD": 104500.0},
            strategies=[],
            aggregate={}
        )

        client = TestClient(app)
        response = client.get("/api/prices")

        assert response.status_code == 200
        data = response.json()
        assert data["XRP/USD"] == 2.35
        assert data["BTC/USD"] == 104500.0

    def test_get_session(self):
        """Test session info endpoint."""
        from ws_tester.dashboard.server import app, update_state

        update_state(
            prices={},
            strategies=[],
            aggregate={},
            session_info={
                "session_id": "test-123",
                "start_time": datetime.now().isoformat(),
                "mode": "simulated"
            }
        )

        client = TestClient(app)
        response = client.get("/api/session")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-123"
        assert data["mode"] == "simulated"

    def test_dashboard_html(self):
        """Test dashboard HTML page."""
        from ws_tester.dashboard.server import app

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "WEBSOCKET PAPER TESTER" in response.text


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDashboardState:
    """Test state management functions."""

    def setup_method(self):
        """Reset state before each test."""
        from ws_tester.dashboard.server import latest_state, _state_lock
        with _state_lock:
            latest_state.clear()
            latest_state.update({
                "timestamp": None,
                "prices": {},
                "strategies": [],
                "aggregate": {},
                "recent_trades": [],
                "session_info": {}
            })

    def test_update_state_thread_safety(self):
        """Test that update_state is thread-safe."""
        from ws_tester.dashboard.server import update_state, _get_state_snapshot
        import threading

        errors = []

        def update_loop():
            try:
                for i in range(100):
                    update_state(
                        prices={"XRP/USD": 2.35 + i * 0.01},
                        strategies=[{"strategy": f"test_{i}", "equity": 100 + i}],
                        aggregate={"total": 100 + i}
                    )
            except Exception as e:
                errors.append(e)

        def read_loop():
            try:
                for _ in range(100):
                    snapshot = _get_state_snapshot()
                    # Verify snapshot is a valid dict
                    assert isinstance(snapshot, dict)
            except Exception as e:
                errors.append(e)

        # Run concurrent updates and reads
        threads = []
        for _ in range(3):
            t1 = threading.Thread(target=update_loop)
            t2 = threading.Thread(target=read_loop)
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

    def test_add_trade_bounded(self):
        """Test that trades list is bounded to 100."""
        from ws_tester.dashboard.server import add_trade, latest_state, _state_lock

        # Add more than 100 trades
        for i in range(150):
            add_trade({
                "timestamp": datetime.now().isoformat(),
                "id": i
            })

        with _state_lock:
            assert len(latest_state["recent_trades"]) == 100
            # Most recent should be first
            assert latest_state["recent_trades"][0]["id"] == 149

    def test_trade_xss_sanitization(self):
        """Test that trade data is sanitized for XSS."""
        from ws_tester.dashboard.server import add_trade, latest_state, _state_lock

        # Add trade with XSS attempt
        add_trade({
            "strategy": "<script>alert('xss')</script>",
            "reason": "<img src=x onerror=alert('xss')>"
        })

        with _state_lock:
            trade = latest_state["recent_trades"][0]
            assert "<script>" not in trade["strategy"]
            assert "<img" not in trade["reason"]
            assert "&lt;script&gt;" in trade["strategy"]

    def test_get_connected_client_count(self):
        """Test client count function."""
        from ws_tester.dashboard.server import get_connected_client_count

        # Should be 0 initially
        count = get_connected_client_count()
        assert count == 0


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDashboardWebSocket:
    """Test WebSocket functionality."""

    def setup_method(self):
        """Reset state before each test."""
        from ws_tester.dashboard.server import latest_state, dashboard_clients, _state_lock, _clients_lock
        with _state_lock:
            latest_state.clear()
            latest_state.update({
                "timestamp": None,
                "prices": {},
                "strategies": [],
                "aggregate": {},
                "recent_trades": [],
                "session_info": {}
            })
        with _clients_lock:
            dashboard_clients.clear()

    def test_websocket_connection(self):
        """Test WebSocket connection and initial state."""
        from ws_tester.dashboard.server import app, update_state

        # Set up initial state
        update_state(
            prices={"XRP/USD": 2.35},
            strategies=[{"strategy": "test", "equity": 100}],
            aggregate={"total": 100}
        )

        client = TestClient(app)

        with client.websocket_connect("/ws/live") as websocket:
            # Should receive initial state
            data = websocket.receive_json()
            assert data["type"] == "initial_state"
            assert "prices" in data["data"]
            assert data["data"]["prices"]["XRP/USD"] == 2.35

    def test_websocket_client_tracking(self):
        """Test that connected clients are tracked."""
        from ws_tester.dashboard.server import app, get_connected_client_count

        client = TestClient(app)

        # Before connection
        assert get_connected_client_count() == 0

        with client.websocket_connect("/ws/live") as websocket:
            # Receive initial message
            websocket.receive_json()
            # Client should be tracked
            assert get_connected_client_count() >= 1

        # After disconnect (may take a moment)
        # Note: TestClient may not properly trigger disconnect
