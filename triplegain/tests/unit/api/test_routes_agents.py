"""
Unit tests for agent API routes.

Tests validate:
- TA agent endpoints
- Regime detection endpoints
- Trading decision endpoints
- Risk management endpoints
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
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_indicator_library():
    """Create mock indicator library."""
    mock = MagicMock()
    mock.calculate_all.return_value = {
        'rsi': 55.0,
        'macd': 0.5,
        'ema_20': 45000.0,
    }
    return mock


@pytest.fixture
def mock_snapshot_builder():
    """Create mock snapshot builder."""
    mock = AsyncMock()

    # Create mock snapshot
    snapshot = MagicMock()
    snapshot.symbol = "BTC/USDT"
    snapshot.timestamp = datetime.now(timezone.utc)
    snapshot.current_price = Decimal("45000.00")
    snapshot.indicators = {'rsi': 55.0}
    snapshot.to_dict.return_value = {'symbol': 'BTC/USDT', 'price': 45000.0}

    mock.build_snapshot.return_value = snapshot
    return mock


@pytest.fixture
def mock_prompt_builder():
    """Create mock prompt builder."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    mock = AsyncMock()
    mock.fetch_candles.return_value = [
        {'open': 45000, 'high': 45500, 'low': 44500, 'close': 45200, 'volume': 100}
    ]
    return mock


@pytest.fixture
def mock_ta_agent():
    """Create mock TA agent."""
    mock = MagicMock()

    # Mock output
    output = MagicMock()
    output.to_dict.return_value = {
        'trend_direction': 'bullish',
        'trend_strength': 0.72,
        'momentum_score': 0.65,
        'bias': 'long',
        'confidence': 0.70,
    }

    # Use AsyncMock for async methods, MagicMock for sync methods
    mock.process = AsyncMock(return_value=output)
    mock.get_latest_output = AsyncMock(return_value=None)
    mock.get_stats = MagicMock(return_value={'total_requests': 5, 'avg_latency_ms': 150})

    return mock


@pytest.fixture
def mock_ta_agent_with_cache(mock_ta_agent):
    """Create mock TA agent with cached output."""
    cached_output = MagicMock()
    cached_output.to_dict.return_value = {
        'trend_direction': 'bearish',
        'trend_strength': 0.55,
        'bias': 'short',
        'confidence': 0.60,
    }
    mock_ta_agent.get_latest_output.return_value = cached_output
    return mock_ta_agent


@pytest.fixture
def mock_regime_agent():
    """Create mock regime agent."""
    mock = MagicMock()

    output = MagicMock()
    output.to_dict.return_value = {
        'regime': 'trending_bull',
        'volatility': 'normal',
        'confidence': 0.75,
    }
    output.get_regime_parameters.return_value = {
        'position_size_multiplier': 1.0,
        'stop_loss_multiplier': 1.0,
    }

    mock.process = AsyncMock(return_value=output)
    mock.get_latest_output = AsyncMock(return_value=None)
    mock.get_stats = MagicMock(return_value={'total_requests': 3})

    return mock


@pytest.fixture
def mock_trading_agent():
    """Create mock trading decision agent."""
    mock = MagicMock()

    # Mock model decision
    model_decision = MagicMock()
    model_decision.model_name = 'gpt-4-turbo'
    model_decision.provider = 'openai'
    model_decision.action = 'BUY'
    model_decision.confidence = 0.75
    model_decision.latency_ms = 500
    model_decision.cost_usd = 0.01
    model_decision.error = None

    output = MagicMock()
    output.action = 'BUY'
    output.confidence = 0.72
    output.consensus_strength = 0.83
    output.entry_price = 45000.0
    output.stop_loss = 44000.0
    output.take_profit = 47000.0
    output.votes = {'BUY': 5, 'HOLD': 1}
    output.agreeing_models = 5
    output.total_models = 6
    output.model_decisions = [model_decision]
    output.regime = 'trending_bull'
    output.ta_bias = 'long'
    output.total_cost_usd = 0.05
    output.tokens_used = 1500
    output.latency_ms = 2500

    mock.process = AsyncMock(return_value=output)
    mock.get_stats = MagicMock(return_value={'total_requests': 2})

    return mock


@pytest.fixture
def mock_risk_engine():
    """Create mock risk engine."""
    mock = MagicMock()

    # Mock validation result
    result = MagicMock()
    result.status = MagicMock()
    result.status.value = 'approved'
    result.is_approved.return_value = True
    result.validation_time_ms = 5
    result.rejections = []
    result.warnings = []
    result.modifications = []
    result.modified_proposal = None
    result.risk_per_trade_pct = 1.5
    result.portfolio_exposure_pct = 15.0

    mock.validate_trade.return_value = result

    # Mock state
    state = MagicMock()
    state.peak_equity = Decimal('10000')
    state.current_equity = Decimal('9800')
    state.available_margin = Decimal('8000')
    state.daily_pnl_pct = -2.0
    state.weekly_pnl_pct = 1.5
    state.current_drawdown_pct = 2.0
    state.max_drawdown_pct = 5.0
    state.consecutive_losses = 1
    state.consecutive_wins = 0
    state.open_positions = 2
    state.total_exposure_pct = 20.0
    state.trading_halted = False
    state.halt_reason = None
    state.halt_until = None
    state.triggered_breakers = []
    state.in_cooldown = False
    state.cooldown_until = None
    state.cooldown_reason = None

    mock.get_state.return_value = state
    mock.get_max_allowed_leverage.return_value = 3
    mock.manual_reset.return_value = True

    return mock


@pytest.fixture
def app_with_agents(
    mock_indicator_library,
    mock_snapshot_builder,
    mock_prompt_builder,
    mock_db_pool,
    mock_ta_agent,
    mock_regime_agent,
    mock_trading_agent,
    mock_risk_engine,
):
    """Create FastAPI app with agent router."""
    from triplegain.src.api.routes_agents import create_agent_router

    app = FastAPI()
    router = create_agent_router(
        indicator_library=mock_indicator_library,
        snapshot_builder=mock_snapshot_builder,
        prompt_builder=mock_prompt_builder,
        db_pool=mock_db_pool,
        ta_agent=mock_ta_agent,
        regime_agent=mock_regime_agent,
        trading_agent=mock_trading_agent,
        risk_engine=mock_risk_engine,
    )
    app.include_router(router)
    return app


@pytest.fixture
def client(app_with_agents):
    """Create test client."""
    return TestClient(app_with_agents)


@pytest.fixture
def app_minimal():
    """Create app with no agents (for testing 503 errors)."""
    from triplegain.src.api.routes_agents import create_agent_router

    app = FastAPI()
    router = create_agent_router(
        indicator_library=MagicMock(),
        snapshot_builder=MagicMock(),
        prompt_builder=MagicMock(),
        db_pool=MagicMock(),
        ta_agent=None,
        regime_agent=None,
        trading_agent=None,
        risk_engine=None,
    )
    app.include_router(router)
    return app


@pytest.fixture
def client_minimal(app_minimal):
    """Create test client with minimal app."""
    return TestClient(app_minimal)


# =============================================================================
# TA Agent Endpoint Tests
# =============================================================================

class TestTAEndpoints:
    """Test Technical Analysis endpoints."""

    def test_get_ta_analysis_fresh(self, client):
        """Test getting fresh TA analysis."""
        response = client.get("/api/v1/agents/ta/BTC_USDT")

        assert response.status_code == 200
        data = response.json()
        # Symbol is normalized to standard format (BTC/USDT instead of BTC_USDT)
        assert data['symbol'] == 'BTC/USDT'
        assert data['cached'] is False
        assert 'output' in data
        assert 'stats' in data

    def test_get_ta_analysis_cached(
        self,
        mock_indicator_library,
        mock_snapshot_builder,
        mock_prompt_builder,
        mock_db_pool,
        mock_ta_agent_with_cache,
    ):
        """Test getting cached TA analysis."""
        from triplegain.src.api.routes_agents import create_agent_router

        app = FastAPI()
        router = create_agent_router(
            indicator_library=mock_indicator_library,
            snapshot_builder=mock_snapshot_builder,
            prompt_builder=mock_prompt_builder,
            db_pool=mock_db_pool,
            ta_agent=mock_ta_agent_with_cache,
        )
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/api/v1/agents/ta/BTC_USDT?max_age_seconds=120")

        assert response.status_code == 200
        data = response.json()
        assert data['cached'] is True

    def test_get_ta_analysis_no_agent(self, client_minimal):
        """Test TA endpoint when agent not initialized."""
        response = client_minimal.get("/api/v1/agents/ta/BTC_USDT")

        assert response.status_code == 503
        assert 'not initialized' in response.json()['detail']

    def test_post_ta_analysis(self, client):
        """Test triggering fresh TA analysis."""
        response = client.post("/api/v1/agents/ta/BTC_USDT/run")

        assert response.status_code == 200
        data = response.json()
        # Symbol is normalized to standard format (BTC/USDT instead of BTC_USDT)
        assert data['symbol'] == 'BTC/USDT'
        assert data['fresh'] is True

    def test_post_ta_analysis_no_agent(self, client_minimal):
        """Test POST TA endpoint when agent not initialized."""
        response = client_minimal.post("/api/v1/agents/ta/BTC_USDT/run")

        assert response.status_code == 503

    def test_get_ta_analysis_error(
        self,
        mock_indicator_library,
        mock_snapshot_builder,
        mock_prompt_builder,
        mock_db_pool,
    ):
        """Test TA endpoint error handling."""
        from triplegain.src.api.routes_agents import create_agent_router

        # Create agent that raises exception
        mock_ta = MagicMock()
        mock_ta.get_latest_output = AsyncMock(return_value=None)
        mock_ta.process = AsyncMock(side_effect=Exception("LLM error"))

        app = FastAPI()
        router = create_agent_router(
            indicator_library=mock_indicator_library,
            snapshot_builder=mock_snapshot_builder,
            prompt_builder=mock_prompt_builder,
            db_pool=mock_db_pool,
            ta_agent=mock_ta,
        )
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/api/v1/agents/ta/BTC_USDT")

        assert response.status_code == 500
        # Error messages are now generic for security (no details exposed)
        assert 'Internal server error' in response.json()['detail']


# =============================================================================
# Regime Detection Endpoint Tests
# =============================================================================

class TestRegimeEndpoints:
    """Test Regime Detection endpoints."""

    def test_get_regime(self, client):
        """Test getting regime detection."""
        response = client.get("/api/v1/agents/regime/BTC_USDT")

        assert response.status_code == 200
        data = response.json()
        # Symbol is normalized to standard format (BTC/USDT instead of BTC_USDT)
        assert data['symbol'] == 'BTC/USDT'
        assert data['cached'] is False
        assert 'output' in data
        assert 'parameters' in data

    def test_get_regime_no_agent(self, client_minimal):
        """Test regime endpoint when agent not initialized."""
        response = client_minimal.get("/api/v1/agents/regime/BTC_USDT")

        assert response.status_code == 503

    def test_post_regime_detection(self, client):
        """Test triggering fresh regime detection."""
        response = client.post("/api/v1/agents/regime/BTC_USDT/run")

        assert response.status_code == 200
        data = response.json()
        assert data['fresh'] is True
        assert 'parameters' in data

    def test_post_regime_no_agent(self, client_minimal):
        """Test POST regime endpoint when agent not initialized."""
        response = client_minimal.post("/api/v1/agents/regime/BTC_USDT/run")

        assert response.status_code == 503


# =============================================================================
# Trading Decision Endpoint Tests
# =============================================================================

class TestTradingDecisionEndpoints:
    """Test Trading Decision endpoints."""

    def test_run_trading_decision(self, client):
        """Test running trading decision."""
        response = client.post(
            "/api/v1/agents/trading/BTC_USDT/run",
            json={"use_ta": True, "use_regime": True, "force_refresh": False}
        )

        assert response.status_code == 200
        data = response.json()
        # Symbol is normalized to standard format (BTC/USDT instead of BTC_USDT)
        assert data['symbol'] == 'BTC/USDT'
        assert 'consensus' in data
        assert data['consensus']['action'] == 'BUY'
        assert 'model_results' in data
        assert 'cost' in data

    def test_run_trading_decision_default_params(self, client):
        """Test trading decision with default parameters."""
        response = client.post("/api/v1/agents/trading/BTC_USDT/run")

        assert response.status_code == 200
        data = response.json()
        assert 'consensus' in data

    def test_run_trading_decision_no_agent(self, client_minimal):
        """Test trading decision when agent not initialized."""
        response = client_minimal.post("/api/v1/agents/trading/BTC_USDT/run")

        assert response.status_code == 503

    def test_run_trading_decision_force_refresh(self, client):
        """Test trading decision with force refresh."""
        response = client.post(
            "/api/v1/agents/trading/BTC_USDT/run",
            json={"use_ta": True, "use_regime": True, "force_refresh": True}
        )

        assert response.status_code == 200


# =============================================================================
# Risk Management Endpoint Tests
# =============================================================================

class TestRiskEndpoints:
    """Test Risk Management endpoints."""

    def test_validate_trade_approved(self, client):
        """Test trade validation - approved."""
        response = client.post(
            "/api/v1/risk/validate",
            json={
                "symbol": "BTC/USDT",
                "side": "buy",
                "size_usd": 500.0,
                "entry_price": 45000.0,
                "stop_loss": 44000.0,
                "take_profit": 47000.0,
                "leverage": 2,
                "confidence": 0.75,
                "regime": "trending_bull"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'approved'
        assert data['approved'] is True
        assert 'risk_metrics' in data

    def test_validate_trade_no_engine(self, client_minimal):
        """Test trade validation when engine not initialized."""
        response = client_minimal.post(
            "/api/v1/risk/validate",
            json={
                "symbol": "BTC/USDT",
                "side": "buy",
                "size_usd": 500.0,
                "entry_price": 45000.0,
            }
        )

        assert response.status_code == 503

    def test_validate_trade_invalid_input(self, client):
        """Test trade validation with invalid input."""
        response = client.post(
            "/api/v1/risk/validate",
            json={
                "symbol": "BTC/USDT",
                "side": "buy",
                "size_usd": -100.0,  # Invalid negative size
                "entry_price": 45000.0,
            }
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_validate_trade_invalid_leverage(self, client):
        """Test trade validation with invalid leverage."""
        response = client.post(
            "/api/v1/risk/validate",
            json={
                "symbol": "BTC/USDT",
                "side": "buy",
                "size_usd": 500.0,
                "entry_price": 45000.0,
                "leverage": 10,  # Exceeds max of 5
            }
        )

        assert response.status_code == 422

    def test_get_risk_state(self, client):
        """Test getting risk state."""
        response = client.get("/api/v1/risk/state")

        assert response.status_code == 200
        data = response.json()
        assert 'equity' in data
        assert 'pnl' in data
        assert 'drawdown' in data
        assert 'circuit_breakers' in data
        assert 'cooldowns' in data

    def test_get_risk_state_no_engine(self, client_minimal):
        """Test risk state when engine not initialized."""
        response = client_minimal.get("/api/v1/risk/state")

        assert response.status_code == 503

    def test_reset_risk_state(self, client):
        """Test resetting risk state."""
        response = client.post("/api/v1/risk/reset")

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'reset_complete'

    def test_reset_risk_state_with_admin(self, client):
        """Test resetting risk state with admin override."""
        response = client.post("/api/v1/risk/reset?admin_override=true")

        assert response.status_code == 200

    def test_reset_risk_state_no_engine(self, client_minimal):
        """Test reset when engine not initialized."""
        response = client_minimal.post("/api/v1/risk/reset")

        assert response.status_code == 503

    def test_validate_trade_rejected(
        self,
        mock_indicator_library,
        mock_snapshot_builder,
        mock_prompt_builder,
        mock_db_pool,
    ):
        """Test trade validation - rejected."""
        from triplegain.src.api.routes_agents import create_agent_router

        # Create engine that rejects trades
        mock_engine = MagicMock()
        result = MagicMock()
        result.status = MagicMock()
        result.status.value = 'rejected'
        result.is_approved.return_value = False
        result.validation_time_ms = 3
        result.rejections = ['Daily loss limit exceeded']
        result.warnings = []
        result.modifications = []
        result.modified_proposal = None
        result.risk_per_trade_pct = 5.0
        result.portfolio_exposure_pct = 100.0

        mock_engine.validate_trade.return_value = result

        app = FastAPI()
        router = create_agent_router(
            indicator_library=mock_indicator_library,
            snapshot_builder=mock_snapshot_builder,
            prompt_builder=mock_prompt_builder,
            db_pool=mock_db_pool,
            risk_engine=mock_engine,
        )
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/api/v1/risk/validate",
            json={
                "symbol": "BTC/USDT",
                "side": "buy",
                "size_usd": 5000.0,
                "entry_price": 45000.0,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'rejected'
        assert data['approved'] is False
        assert 'rejections' in data


# =============================================================================
# Agent Stats Endpoint Tests
# =============================================================================

class TestAgentStatsEndpoints:
    """Test agent stats endpoints."""

    def test_get_agent_stats(self, client):
        """Test getting agent stats."""
        response = client.get("/api/v1/agents/stats")

        assert response.status_code == 200
        data = response.json()
        assert 'timestamp' in data
        assert 'agents' in data
        assert 'technical_analysis' in data['agents']
        assert 'regime_detection' in data['agents']
        assert 'trading_decision' in data['agents']

    def test_get_agent_stats_no_agents(self, client_minimal):
        """Test agent stats when no agents initialized."""
        response = client_minimal.get("/api/v1/agents/stats")

        assert response.status_code == 200
        data = response.json()
        assert data['agents'] == {}


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_special_characters_in_symbol(self, client):
        """Test handling of special characters in symbol."""
        # URL-encoded slash
        response = client.get("/api/v1/agents/ta/BTC%2FUSDT")
        # Should either work or return appropriate error
        assert response.status_code in [200, 404, 500]

    def test_empty_symbol(self, client):
        """Test empty symbol handling."""
        # This should return 404 as route won't match
        response = client.get("/api/v1/agents/ta/")
        assert response.status_code in [404, 307]

    def test_max_age_boundary(self, client):
        """Test max_age_seconds boundary values."""
        # Min value
        response = client.get("/api/v1/agents/ta/BTC_USDT?max_age_seconds=0")
        assert response.status_code == 200

        # Max value
        response = client.get("/api/v1/agents/ta/BTC_USDT?max_age_seconds=300")
        assert response.status_code == 200

        # Over max
        response = client.get("/api/v1/agents/ta/BTC_USDT?max_age_seconds=999")
        assert response.status_code == 422  # Validation error

    def test_reset_fails_without_admin(
        self,
        mock_indicator_library,
        mock_snapshot_builder,
        mock_prompt_builder,
        mock_db_pool,
    ):
        """Test reset fails when admin override required."""
        from triplegain.src.api.routes_agents import create_agent_router

        mock_engine = MagicMock()
        mock_engine.manual_reset.return_value = False  # Requires admin

        app = FastAPI()
        router = create_agent_router(
            indicator_library=mock_indicator_library,
            snapshot_builder=mock_snapshot_builder,
            prompt_builder=mock_prompt_builder,
            db_pool=mock_db_pool,
            risk_engine=mock_engine,
        )
        app.include_router(router)
        client = TestClient(app)

        response = client.post("/api/v1/risk/reset?admin_override=false")

        assert response.status_code == 403
        assert 'admin_override' in response.json()['detail']
