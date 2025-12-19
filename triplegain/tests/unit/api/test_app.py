"""
Unit tests for FastAPI app.

Tests validate:
- Health endpoints
- Indicator endpoints
- Snapshot endpoints
- Debug endpoints
- Lifespan management
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

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    mock = AsyncMock()
    mock.is_connected = True
    mock.check_health.return_value = {'status': 'healthy', 'latency_ms': 5}
    mock.fetch_candles.return_value = [
        {'open': 45000.0, 'high': 45500.0, 'low': 44500.0, 'close': 45200.0, 'volume': 100.0}
        for _ in range(100)
    ]
    return mock


@pytest.fixture
def mock_indicator_library():
    """Create mock indicator library."""
    mock = MagicMock()
    mock.calculate_all.return_value = {
        'rsi': 55.0,
        'macd': {'macd': 0.5, 'signal': 0.3, 'histogram': 0.2},
        'ema_20': 45000.0,
        'sma_50': 44500.0,
    }
    return mock


@pytest.fixture
def mock_snapshot_builder():
    """Create mock snapshot builder."""
    mock = AsyncMock()

    snapshot = MagicMock()
    snapshot.symbol = "BTC/USDT"
    snapshot.timestamp = datetime.now(timezone.utc)
    snapshot.current_price = Decimal("45000.00")
    snapshot.price_24h_ago = Decimal("44000.00")
    snapshot.price_change_24h_pct = Decimal("2.27")
    snapshot.data_age_seconds = 5
    snapshot.missing_data_flags = []
    snapshot.indicators = {'rsi': 55.0}
    snapshot.mtf_state = MagicMock()
    snapshot.mtf_state.to_dict.return_value = {'1h': 'bullish', '4h': 'neutral'}
    snapshot.order_book = MagicMock()
    snapshot.order_book.to_dict.return_value = {'bid': 44999.0, 'ask': 45001.0}

    mock.build_snapshot.return_value = snapshot
    return mock


@pytest.fixture
def mock_prompt_builder():
    """Create mock prompt builder."""
    mock = MagicMock()
    mock._templates = {'technical_analysis': {}, 'regime_detection': {}}

    prompt = MagicMock()
    prompt.system_prompt = "You are a trading assistant."
    prompt.user_message = "Analyze BTC/USDT."
    prompt.estimated_tokens = 500
    prompt.tier = "local"

    mock.build_prompt.return_value = prompt
    return mock


@pytest.fixture
def mock_config_loader():
    """Create mock config loader."""
    mock = MagicMock()
    mock.get_database_config.return_value = {'host': 'localhost', 'port': 5432}
    mock.get_indicators_config.return_value = {'rsi': {'period': 14}, 'ema': {'periods': [20, 50]}}
    mock.get_snapshot_config.return_value = {'candle_lookback': {'1h': 100, '4h': 50}}
    mock.get_prompts_config.return_value = {'template_dir': 'templates'}
    return mock


@pytest.fixture
def app_with_mocks(
    mock_db_pool,
    mock_indicator_library,
    mock_snapshot_builder,
    mock_prompt_builder,
):
    """Create app with mocked dependencies."""
    from triplegain.src.api import app as app_module

    # Patch global state
    with patch.object(app_module, '_db_pool', mock_db_pool):
        with patch.object(app_module, '_indicator_library', mock_indicator_library):
            with patch.object(app_module, '_snapshot_builder', mock_snapshot_builder):
                with patch.object(app_module, '_prompt_builder', mock_prompt_builder):
                    app = FastAPI()

                    # Register routes manually
                    app_module.register_health_routes(app)
                    app_module.register_indicator_routes(app)
                    app_module.register_snapshot_routes(app)
                    # Use debug_mode=True for testing (no auth required)
                    app_module.register_debug_routes(app, debug_mode=True)

                    yield app


@pytest.fixture
def client(app_with_mocks):
    """Create test client."""
    return TestClient(app_with_mocks)


@pytest.fixture
def app_not_initialized():
    """Create app with uninitialized components."""
    from triplegain.src.api import app as app_module

    with patch.object(app_module, '_db_pool', None):
        with patch.object(app_module, '_indicator_library', None):
            with patch.object(app_module, '_snapshot_builder', None):
                with patch.object(app_module, '_prompt_builder', None):
                    app = FastAPI()
                    app_module.register_health_routes(app)
                    app_module.register_indicator_routes(app)
                    app_module.register_snapshot_routes(app)
                    # Use debug_mode=True for testing (no auth required)
                    app_module.register_debug_routes(app, debug_mode=True)
                    yield app


@pytest.fixture
def client_not_initialized(app_not_initialized):
    """Create client for uninitialized app."""
    return TestClient(app_not_initialized)


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check_healthy(self, client):
        """Test health check when all components healthy."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'components' in data
        assert data['components']['database']['status'] == 'healthy'

    def test_health_check_not_initialized(self, client_not_initialized):
        """Test health check when components not initialized."""
        response = client_not_initialized.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'degraded'
        assert data['components']['database']['status'] == 'not_initialized'

    def test_health_check_degraded_db(
        self,
        mock_indicator_library,
        mock_snapshot_builder,
        mock_prompt_builder,
    ):
        """Test health check with degraded database."""
        from triplegain.src.api import app as app_module

        mock_db = AsyncMock()
        mock_db.check_health.return_value = {'status': 'unhealthy', 'error': 'Connection lost'}

        with patch.object(app_module, '_db_pool', mock_db):
            with patch.object(app_module, '_indicator_library', mock_indicator_library):
                with patch.object(app_module, '_snapshot_builder', mock_snapshot_builder):
                    with patch.object(app_module, '_prompt_builder', mock_prompt_builder):
                        app = FastAPI()
                        app_module.register_health_routes(app)
                        client = TestClient(app)

                        response = client.get("/health")

                        assert response.status_code == 200
                        data = response.json()
                        assert data['status'] == 'degraded'

    def test_liveness_check(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'alive'

    def test_readiness_check_ready(self, client):
        """Test readiness probe when ready."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ready'

    def test_readiness_check_not_ready(self, client_not_initialized):
        """Test readiness probe when not ready."""
        response = client_not_initialized.get("/health/ready")

        assert response.status_code == 503


# =============================================================================
# Indicator Endpoint Tests
# =============================================================================

class TestIndicatorEndpoints:
    """Test indicator endpoints."""

    def test_get_indicators_success(self, client):
        """Test getting indicators successfully."""
        response = client.get("/api/v1/indicators/BTC_USDT/1h")

        assert response.status_code == 200
        data = response.json()
        # Symbol is normalized to BTC/USDT format (Finding 21 fix)
        assert data['symbol'] == 'BTC/USDT'
        assert data['timeframe'] == '1h'
        assert 'indicators' in data
        assert 'candle_count' in data

    def test_get_indicators_with_limit(self, client):
        """Test getting indicators with custom limit."""
        response = client.get("/api/v1/indicators/BTC_USDT/1h?limit=50")

        assert response.status_code == 200

    def test_get_indicators_not_initialized(self, client_not_initialized):
        """Test indicators when service not initialized."""
        response = client_not_initialized.get("/api/v1/indicators/BTC_USDT/1h")

        assert response.status_code == 503

    def test_get_indicators_no_data(
        self,
        mock_indicator_library,
        mock_snapshot_builder,
        mock_prompt_builder,
    ):
        """Test indicators when no data found.

        Finding 22 fixed: Exception handling order now correctly re-raises HTTPException.
        """
        from triplegain.src.api import app as app_module

        mock_db = MagicMock()
        mock_db.is_connected = True
        mock_db.fetch_candles = AsyncMock(return_value=[])  # No data

        with patch.object(app_module, '_db_pool', mock_db):
            with patch.object(app_module, '_indicator_library', mock_indicator_library):
                with patch.object(app_module, '_snapshot_builder', mock_snapshot_builder):
                    with patch.object(app_module, '_prompt_builder', mock_prompt_builder):
                        app = FastAPI()
                        app_module.register_indicator_routes(app)
                        client = TestClient(app)

                        response = client.get("/api/v1/indicators/XYZ_USDT/1h")

                        # Now correctly returns 404 (Finding 22 fix)
                        assert response.status_code == 404

    def test_get_indicators_invalid_limit(self, client):
        """Test indicators with invalid limit."""
        response = client.get("/api/v1/indicators/BTC_USDT/1h?limit=9999")

        assert response.status_code == 422  # Validation error


# =============================================================================
# Snapshot Endpoint Tests
# =============================================================================

class TestSnapshotEndpoints:
    """Test snapshot endpoints."""

    def test_get_snapshot_success(self, client):
        """Test getting snapshot successfully."""
        response = client.get("/api/v1/snapshot/BTC_USDT")

        assert response.status_code == 200
        data = response.json()
        # Symbol is normalized to BTC/USDT format (Finding 21 fix)
        assert data['symbol'] == 'BTC/USDT'
        assert 'snapshot' in data
        assert 'indicators' in data
        assert 'mtf_state' in data
        assert 'order_book' in data

    def test_get_snapshot_without_order_book(self, client):
        """Test getting snapshot without order book."""
        response = client.get("/api/v1/snapshot/BTC_USDT?include_order_book=false")

        assert response.status_code == 200

    def test_get_snapshot_not_initialized(self, client_not_initialized):
        """Test snapshot when service not initialized."""
        response = client_not_initialized.get("/api/v1/snapshot/BTC_USDT")

        assert response.status_code == 503


# =============================================================================
# Debug Endpoint Tests
# =============================================================================

class TestDebugEndpoints:
    """Test debug endpoints."""

    def test_get_debug_prompt(self, client):
        """Test getting debug prompt."""
        response = client.get("/api/v1/debug/prompt/technical_analysis?symbol=BTC_USDT")

        assert response.status_code == 200
        data = response.json()
        assert data['agent'] == 'technical_analysis'
        # Symbol is normalized to BTC/USDT format (Finding 21 fix)
        assert data['symbol'] == 'BTC/USDT'
        assert 'prompt' in data
        assert 'system_prompt' in data['prompt']
        assert 'user_message' in data['prompt']

    def test_get_debug_prompt_default_symbol(self, client):
        """Test debug prompt with default symbol."""
        response = client.get("/api/v1/debug/prompt/regime_detection")

        assert response.status_code == 200

    def test_get_debug_prompt_not_initialized(self, client_not_initialized):
        """Test debug prompt when not initialized."""
        response = client_not_initialized.get("/api/v1/debug/prompt/technical_analysis")

        assert response.status_code == 503

    def test_get_debug_prompt_invalid_agent(
        self,
        mock_db_pool,
        mock_indicator_library,
        mock_snapshot_builder,
    ):
        """Test debug prompt with invalid agent name."""
        from triplegain.src.api import app as app_module

        mock_prompt = MagicMock()
        mock_prompt._templates = {}
        mock_prompt.build_prompt.side_effect = ValueError("Unknown agent: invalid_agent")

        with patch.object(app_module, '_db_pool', mock_db_pool):
            with patch.object(app_module, '_indicator_library', mock_indicator_library):
                with patch.object(app_module, '_snapshot_builder', mock_snapshot_builder):
                    with patch.object(app_module, '_prompt_builder', mock_prompt):
                        app = FastAPI()
                        # Use debug_mode=True for testing (no auth required)
                        app_module.register_debug_routes(app, debug_mode=True)
                        client = TestClient(app)

                        response = client.get("/api/v1/debug/prompt/invalid_agent")

                        assert response.status_code == 400

    def test_get_config(self, client, mock_config_loader):
        """Test getting config."""
        with patch('triplegain.src.api.app.get_config_loader', return_value=mock_config_loader):
            response = client.get("/api/v1/debug/config")

            assert response.status_code == 200
            data = response.json()
            assert 'indicators' in data
            assert 'snapshot' in data
            assert 'database' in data


# =============================================================================
# App Creation Tests
# =============================================================================

class TestAppCreation:
    """Test app creation and configuration."""

    def test_create_app(self):
        """Test app creation."""
        from triplegain.src.api.app import create_app

        app = create_app()

        assert app is not None
        assert app.title == "TripleGain API"
        assert app.version == "1.0.0"

    def test_get_app_singleton(self):
        """Test get_app returns singleton."""
        from triplegain.src.api import app as app_module

        # Reset singleton
        app_module._app = None

        app1 = app_module.get_app()
        app2 = app_module.get_app()

        assert app1 is app2

    def test_fastapi_not_available(self):
        """Test error when FastAPI not available."""
        from triplegain.src.api import app as app_module

        with patch.object(app_module, 'FASTAPI_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="FastAPI is not installed"):
                app_module.create_app()


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_health_prompt_builder_templates(
        self,
        mock_db_pool,
        mock_indicator_library,
        mock_snapshot_builder,
    ):
        """Test health check shows template count."""
        from triplegain.src.api import app as app_module

        mock_prompt = MagicMock()
        mock_prompt._templates = {'ta': {}, 'regime': {}, 'trading': {}}

        with patch.object(app_module, '_db_pool', mock_db_pool):
            with patch.object(app_module, '_indicator_library', mock_indicator_library):
                with patch.object(app_module, '_snapshot_builder', mock_snapshot_builder):
                    with patch.object(app_module, '_prompt_builder', mock_prompt):
                        app = FastAPI()
                        app_module.register_health_routes(app)
                        client = TestClient(app)

                        response = client.get("/health")

                        data = response.json()
                        assert data['components']['prompt_builder']['templates_loaded'] == 3

    def test_indicator_calculation_error(
        self,
        mock_db_pool,
        mock_snapshot_builder,
        mock_prompt_builder,
    ):
        """Test indicator calculation error handling."""
        from triplegain.src.api import app as app_module

        mock_lib = MagicMock()
        mock_lib.calculate_all.side_effect = Exception("Calculation error")

        with patch.object(app_module, '_db_pool', mock_db_pool):
            with patch.object(app_module, '_indicator_library', mock_lib):
                with patch.object(app_module, '_snapshot_builder', mock_snapshot_builder):
                    with patch.object(app_module, '_prompt_builder', mock_prompt_builder):
                        app = FastAPI()
                        app_module.register_indicator_routes(app)
                        client = TestClient(app)

                        response = client.get("/api/v1/indicators/BTC_USDT/1h")

                        assert response.status_code == 500

    def test_snapshot_build_error(
        self,
        mock_db_pool,
        mock_indicator_library,
        mock_prompt_builder,
    ):
        """Test snapshot build error handling."""
        from triplegain.src.api import app as app_module

        mock_snapshot = AsyncMock()
        mock_snapshot.build_snapshot.side_effect = Exception("Build error")

        with patch.object(app_module, '_db_pool', mock_db_pool):
            with patch.object(app_module, '_indicator_library', mock_indicator_library):
                with patch.object(app_module, '_snapshot_builder', mock_snapshot):
                    with patch.object(app_module, '_prompt_builder', mock_prompt_builder):
                        app = FastAPI()
                        app_module.register_snapshot_routes(app)
                        client = TestClient(app)

                        response = client.get("/api/v1/snapshot/BTC_USDT")

                        assert response.status_code == 500
