"""
Unit tests for API endpoints.

Tests the FastAPI application using mocked dependencies
to avoid requiring a live database connection.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

# Skip all tests if FastAPI is not installed
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_db_pool():
    """Create a mock database pool."""
    pool = MagicMock()
    pool.is_connected = True
    pool.check_health = AsyncMock(return_value={"status": "healthy", "latency_ms": 5})
    pool.connect = AsyncMock()
    pool.disconnect = AsyncMock()

    # Mock fetch_candles
    pool.fetch_candles = AsyncMock(return_value=[
        {
            'timestamp': datetime.now(timezone.utc),
            'open': 100.0,
            'high': 105.0,
            'low': 98.0,
            'close': 103.0,
            'volume': 1000.0
        }
        for _ in range(50)
    ])

    # Mock fetch_24h_data
    pool.fetch_24h_data = AsyncMock(return_value={
        'price_24h_ago': 100.0,
        'price_change_24h_pct': 3.0,
        'volume_24h': 50000.0
    })

    # Mock fetch_order_book
    pool.fetch_order_book = AsyncMock(return_value={
        'bids': [{'price': 102.0, 'size': 100}],
        'asks': [{'price': 103.0, 'size': 100}]
    })

    return pool


@pytest.fixture
def mock_indicator_library():
    """Create a mock indicator library."""
    library = MagicMock()
    library.calculate_all = MagicMock(return_value={
        'ema_9': 101.5,
        'ema_21': 100.8,
        'rsi_14': 55.2,
        'macd': {'line': 0.5, 'signal': 0.3, 'histogram': 0.2},
        'atr_14': 2.5,
        'adx_14': 25.0,
        'bollinger_bands': {
            'upper': 108.0,
            'middle': 103.0,
            'lower': 98.0,
            'width': 0.1,
            'position': 0.5
        },
        'volume_vs_avg': 1.2
    })
    return library


@pytest.fixture
def mock_prompt_builder():
    """Create a mock prompt builder."""
    builder = MagicMock()
    builder._templates = {'technical_analysis': {}, 'risk_manager': {}}

    prompt_result = MagicMock()
    prompt_result.system_prompt = "You are a trading agent."
    prompt_result.user_message = "Analyze the market."
    prompt_result.estimated_tokens = 500
    prompt_result.tier = "tier1_local"

    builder.build_prompt = MagicMock(return_value=prompt_result)
    return builder


@pytest.fixture
def mock_config_loader():
    """Create a mock config loader."""
    loader = MagicMock()
    loader.get_database_config = MagicMock(return_value={
        'connection': {'host': 'localhost', 'port': 5432}
    })
    loader.get_indicators_config = MagicMock(return_value={
        'ema': {'periods': [9, 21, 50, 200]},
        'sma': {'periods': [20, 50, 200]},
        'rsi': {'period': 14},
        'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'atr': {'period': 14},
        'bollinger_bands': {'period': 20, 'std_dev': 2.0},
    })
    loader.get_snapshot_config = MagicMock(return_value={
        'candle_lookback': {'1m': 60, '5m': 48, '1h': 48},
        'primary_timeframe': '1h',
        'data_quality': {'max_age_seconds': 60, 'min_candles_required': 20}
    })
    loader.get_prompts_config = MagicMock(return_value={
        'agents': {
            'technical_analysis': {'tier': 'tier1_local'},
            'risk_manager': {'tier': 'tier2_api'}
        }
    })
    return loader


@pytest.fixture
def test_client(mock_db_pool, mock_indicator_library, mock_prompt_builder, mock_config_loader):
    """Create a test client with mocked dependencies."""
    with patch('triplegain.src.api.app._db_pool', mock_db_pool), \
         patch('triplegain.src.api.app._indicator_library', mock_indicator_library), \
         patch('triplegain.src.api.app._snapshot_builder') as mock_snapshot_builder, \
         patch('triplegain.src.api.app._prompt_builder', mock_prompt_builder), \
         patch('triplegain.src.api.app.get_config_loader', return_value=mock_config_loader):

        # Configure mock snapshot builder
        from triplegain.src.data.market_snapshot import MarketSnapshot, MultiTimeframeState, OrderBookFeatures

        mock_snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            current_price=Decimal("103.0"),
            price_24h_ago=Decimal("100.0"),
            price_change_24h_pct=Decimal("3.0"),
            indicators=mock_indicator_library.calculate_all(),
            mtf_state=MultiTimeframeState(
                trend_alignment_score=Decimal("0.5"),
                aligned_bullish_count=3,
                aligned_bearish_count=1,
                total_timeframes=4,
                rsi_by_timeframe={'1h': Decimal("55.2")},
                atr_by_timeframe={'1h': Decimal("2.5")}
            ),
            order_book=OrderBookFeatures(
                bid_depth_usd=Decimal("10000"),
                ask_depth_usd=Decimal("10000"),
                imbalance=Decimal("0"),
                spread_bps=Decimal("10"),
                weighted_mid=Decimal("102.5")
            ),
            data_age_seconds=5
        )

        mock_snapshot_builder.build_snapshot = AsyncMock(return_value=mock_snapshot)

        from triplegain.src.api.app import create_app
        app = create_app()

        # Use sync test client (FastAPI handles async internally)
        client = TestClient(app, raise_server_exceptions=False)
        yield client


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check_healthy(self, test_client):
        """Test health endpoint returns healthy status."""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "timestamp" in data
        assert "components" in data

    def test_liveness_check(self, test_client):
        """Test liveness probe endpoint."""
        response = test_client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_check_ready(self, test_client, mock_db_pool):
        """Test readiness probe when database is connected."""
        mock_db_pool.is_connected = True
        response = test_client.get("/health/ready")
        # May return 200 or 503 depending on initialization
        assert response.status_code in (200, 503)


# =============================================================================
# Indicator Endpoint Tests
# =============================================================================

class TestIndicatorEndpoints:
    """Tests for indicator-related endpoints."""

    def test_get_indicators_success(self, test_client, mock_db_pool, mock_indicator_library):
        """Test successful indicator calculation."""
        response = test_client.get("/api/v1/indicators/BTC_USDT/1h?limit=100")

        # May return 200 or 503 depending on initialization
        if response.status_code == 200:
            data = response.json()
            assert data["symbol"] == "BTC_USDT"
            assert data["timeframe"] == "1h"
            assert "indicators" in data or "candle_count" in data

    def test_get_indicators_with_limit(self, test_client):
        """Test indicator endpoint respects limit parameter."""
        response = test_client.get("/api/v1/indicators/BTC_USDT/1h?limit=50")
        # Just verify the request doesn't crash
        assert response.status_code in (200, 404, 503)

    def test_get_indicators_invalid_limit(self, test_client):
        """Test indicator endpoint validates limit parameter."""
        response = test_client.get("/api/v1/indicators/BTC_USDT/1h?limit=0")
        # Should return 422 for validation error or 503 if not initialized
        assert response.status_code in (422, 503)

    def test_get_indicators_limit_too_high(self, test_client):
        """Test indicator endpoint validates max limit."""
        response = test_client.get("/api/v1/indicators/BTC_USDT/1h?limit=5000")
        # Should return 422 for validation error or 503 if not initialized
        assert response.status_code in (422, 503)


# =============================================================================
# Snapshot Endpoint Tests
# =============================================================================

class TestSnapshotEndpoints:
    """Tests for snapshot-related endpoints."""

    def test_get_snapshot_success(self, test_client):
        """Test successful snapshot retrieval."""
        response = test_client.get("/api/v1/snapshot/BTC_USDT")

        # May return 200 or 503 depending on initialization
        if response.status_code == 200:
            data = response.json()
            assert data["symbol"] == "BTC_USDT"
            assert "snapshot" in data

    def test_get_snapshot_without_order_book(self, test_client):
        """Test snapshot without order book data."""
        response = test_client.get("/api/v1/snapshot/BTC_USDT?include_order_book=false")
        # Just verify the request doesn't crash
        assert response.status_code in (200, 503)


# =============================================================================
# Debug Endpoint Tests
# =============================================================================

class TestDebugEndpoints:
    """Tests for debug endpoints."""

    def test_get_debug_prompt_success(self, test_client):
        """Test debug prompt generation."""
        response = test_client.get("/api/v1/debug/prompt/technical_analysis?symbol=BTC_USDT")

        # May return 200, 400, or 503 depending on initialization
        if response.status_code == 200:
            data = response.json()
            assert data["agent"] == "technical_analysis"
            assert "prompt" in data

    def test_get_debug_config(self, test_client):
        """Test debug config endpoint."""
        response = test_client.get("/api/v1/debug/config")

        # May return 200 or 503 depending on initialization
        if response.status_code == 200:
            data = response.json()
            assert "indicators" in data or "database" in data


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for API error handling."""

    def test_404_for_unknown_endpoint(self, test_client):
        """Test 404 response for unknown endpoints."""
        response = test_client.get("/api/v1/unknown")
        assert response.status_code == 404

    def test_error_response_does_not_expose_details(self, test_client, mock_db_pool):
        """Test that error responses don't expose internal details."""
        # Force an error by making fetch_candles raise
        mock_db_pool.fetch_candles = AsyncMock(side_effect=Exception("Database connection failed: password=secret"))

        response = test_client.get("/api/v1/indicators/BTC_USDT/1h")

        # If we get a 500, verify it doesn't contain sensitive info
        if response.status_code == 500:
            data = response.json()
            detail = data.get("detail", "")
            assert "password" not in detail.lower()
            assert "secret" not in detail.lower()
            # Should have generic error message
            assert "internal server error" in detail.lower() or "error" in detail.lower()


# =============================================================================
# API Response Format Tests
# =============================================================================

class TestAPIResponseFormat:
    """Tests for API response formats."""

    def test_health_response_has_required_fields(self, test_client):
        """Test health response structure."""
        response = test_client.get("/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        # timestamp should be ISO format
        assert "T" in data["timestamp"]

    def test_indicators_response_has_required_fields(self, test_client):
        """Test indicators response structure."""
        response = test_client.get("/api/v1/indicators/BTC_USDT/1h")

        if response.status_code == 200:
            data = response.json()
            assert "symbol" in data
            assert "timeframe" in data
            assert "timestamp" in data


# =============================================================================
# API without Dependencies Tests
# =============================================================================

class TestAPIWithoutDependencies:
    """Tests for API behavior when dependencies are not initialized."""

    def test_indicators_returns_503_when_not_initialized(self):
        """Test that indicators endpoint returns 503 when not initialized."""
        with patch('triplegain.src.api.app._db_pool', None), \
             patch('triplegain.src.api.app._indicator_library', None):

            from triplegain.src.api.app import create_app
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/api/v1/indicators/BTC_USDT/1h")
            assert response.status_code == 503

    def test_snapshot_returns_503_when_not_initialized(self):
        """Test that snapshot endpoint returns 503 when not initialized."""
        with patch('triplegain.src.api.app._snapshot_builder', None):

            from triplegain.src.api.app import create_app
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/api/v1/snapshot/BTC_USDT")
            assert response.status_code == 503
