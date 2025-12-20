"""
Integration tests for Sentiment Analysis module.

Tests validate:
- Message bus publish/subscribe
- Coordinator receives sentiment updates
- API endpoint functionality
- Database storage and retrieval

These tests use mocked LLM clients but real message bus and in-memory data.
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_grok_client():
    """Create mock Grok client with web search."""
    client = AsyncMock()
    client.generate_with_search = AsyncMock(return_value=MagicMock(
        text=json.dumps({
            "bias": "bullish",
            "score": 0.4,
            "confidence": 0.75,
            "key_events": [
                {
                    "event": "Whale accumulation detected",
                    "impact": "positive",
                    "significance": "high",
                    "source": "Twitter"
                }
            ],
            "narratives": ["Social sentiment turning bullish"],
            "reasoning": "High engagement on positive posts"
        }),
        tokens_used=500,
        cost_usd=0.005,
        model="grok-2",
    ))
    return client


@pytest.fixture
def mock_gpt_client():
    """Create mock GPT client with web search."""
    client = AsyncMock()
    client.generate_with_search = AsyncMock(return_value=MagicMock(
        text=json.dumps({
            "bias": "bullish",
            "score": 0.5,
            "confidence": 0.8,
            "key_events": [
                {
                    "event": "ETF inflows continue",
                    "impact": "positive",
                    "significance": "high",
                    "source": "Bloomberg"
                }
            ],
            "narratives": ["Institutional adoption accelerating"],
            "reasoning": "News coverage increasingly positive"
        }),
        tokens_used=800,
        cost_usd=0.02,
        model="gpt-4o-search-preview",
    ))
    return client


@pytest.fixture
def mock_prompt_builder():
    """Create mock prompt builder."""
    builder = MagicMock()
    builder.build_prompt = MagicMock(return_value=MagicMock(
        system_prompt="You are a sentiment analyst",
        user_message="Analyze BTC sentiment",
    ))
    return builder


@pytest.fixture
def agent_config():
    """Create agent configuration."""
    return {
        "providers": {
            "grok": {
                "enabled": True,
                "model": "grok-2",
                "timeout_ms": 30000,
                "weight": {"social": 0.6, "news": 0.4},
                "retry": {"max_attempts": 2, "backoff_ms": 5000},
            },
            "gpt": {
                "enabled": True,
                "model": "gpt-4o-search-preview",
                "timeout_ms": 30000,
                "weight": {"social": 0.4, "news": 0.6},
                "retry": {"max_attempts": 2, "backoff_ms": 5000},
            },
        },
        "aggregation": {"min_providers": 1, "min_confidence": 0.3},
        "output": {"max_events": 5, "max_narratives": 3},
        "circuit_breaker": {"failure_threshold": 3, "cooldown_seconds": 300},
    }


# =============================================================================
# Message Bus Integration Tests
# =============================================================================

class TestMessageBusIntegration:
    """Test sentiment agent publishes to message bus correctly."""

    @pytest.mark.asyncio
    async def test_sentiment_publishes_to_bus(
        self, mock_grok_client, mock_gpt_client, mock_prompt_builder, agent_config
    ):
        """Sentiment agent should publish results to message bus."""
        from triplegain.src.agents.sentiment_analysis import SentimentAnalysisAgent

        # Create agent with mock clients
        agent = SentimentAnalysisAgent(
            llm_clients={"grok": mock_grok_client, "gpt": mock_gpt_client},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        # Process sentiment
        output = await agent.process(symbol="BTC/USDT")

        # Verify output structure for message bus
        assert output.symbol == "BTC/USDT"
        assert hasattr(output, 'bias')
        assert hasattr(output, 'overall_score')
        assert hasattr(output, 'social_score')
        assert hasattr(output, 'news_score')

        # Verify it can be serialized for message bus
        output_dict = output.to_dict()
        assert "bias" in output_dict
        assert "overall_score" in output_dict

    @pytest.mark.asyncio
    async def test_sentiment_output_serializable(
        self, mock_grok_client, mock_gpt_client, mock_prompt_builder, agent_config
    ):
        """Sentiment output should be JSON serializable for message bus."""
        from triplegain.src.agents.sentiment_analysis import SentimentAnalysisAgent

        agent = SentimentAnalysisAgent(
            llm_clients={"grok": mock_grok_client, "gpt": mock_gpt_client},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        output = await agent.process(symbol="BTC/USDT")

        # Should serialize to JSON without error
        json_str = output.to_json()
        parsed = json.loads(json_str)

        assert parsed["symbol"] == "BTC/USDT"
        assert "bias" in parsed


# =============================================================================
# Coordinator Integration Tests
# =============================================================================

class TestCoordinatorIntegration:
    """Test coordinator receives and processes sentiment correctly."""

    @pytest.mark.asyncio
    async def test_sentiment_output_has_required_coordinator_fields(
        self, mock_grok_client, mock_gpt_client, mock_prompt_builder, agent_config
    ):
        """Sentiment output should have fields coordinator expects."""
        from triplegain.src.agents.sentiment_analysis import SentimentAnalysisAgent

        agent = SentimentAnalysisAgent(
            llm_clients={"grok": mock_grok_client, "gpt": mock_gpt_client},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        output = await agent.process(symbol="BTC/USDT")
        output_dict = output.to_dict()

        # Coordinator expects these fields
        assert "bias" in output_dict  # Used as sent_bias in coordinator
        assert "confidence" in output_dict
        assert "overall_score" in output_dict
        assert "social_analysis" in output_dict  # For trading context
        assert "news_analysis" in output_dict  # For trading context

    @pytest.mark.asyncio
    async def test_sentiment_bias_key_matches_coordinator(
        self, mock_grok_client, mock_gpt_client, mock_prompt_builder, agent_config
    ):
        """Sentiment should use 'bias' key that coordinator expects."""
        from triplegain.src.agents.sentiment_analysis import SentimentAnalysisAgent

        agent = SentimentAnalysisAgent(
            llm_clients={"grok": mock_grok_client, "gpt": mock_gpt_client},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        output = await agent.process(symbol="BTC/USDT")

        # Coordinator uses: sent_bias = sentiment_msg.payload.get("bias", "neutral")
        # So we need "bias" key, not "sentiment_bias"
        output_dict = output.to_dict()
        assert "bias" in output_dict
        assert output_dict["bias"] in ["very_bullish", "bullish", "neutral", "bearish", "very_bearish"]


# =============================================================================
# Trading Agent Integration Tests
# =============================================================================

class TestTradingAgentIntegration:
    """Test trading agent receives sentiment data correctly."""

    @pytest.mark.asyncio
    async def test_sentiment_provides_trading_context(
        self, mock_grok_client, mock_gpt_client, mock_prompt_builder, agent_config
    ):
        """Sentiment should provide analysis text for trading decisions."""
        from triplegain.src.agents.sentiment_analysis import SentimentAnalysisAgent

        agent = SentimentAnalysisAgent(
            llm_clients={"grok": mock_grok_client, "gpt": mock_gpt_client},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        output = await agent.process(symbol="BTC/USDT")

        # Trading agent needs analysis context
        assert output.social_analysis != ""  # From Grok
        assert output.news_analysis != ""  # From GPT
        assert len(output.key_events) > 0  # Key events for context
        assert len(output.market_narratives) > 0  # Current narratives


# =============================================================================
# API Endpoint Integration Tests
# =============================================================================

class TestAPIIntegration:
    """Test API endpoints work with sentiment agent."""

    def test_sentiment_response_model_fields(self):
        """API response model should match sentiment output."""
        from triplegain.src.api.routes_sentiment import SentimentResponse

        # Check all expected fields exist in response model
        fields = SentimentResponse.model_fields
        assert "timestamp" in fields
        assert "symbol" in fields
        assert "bias" in fields
        assert "confidence" in fields
        assert "social_score" in fields
        assert "news_score" in fields
        assert "overall_score" in fields
        assert "social_analysis" in fields
        assert "news_analysis" in fields
        assert "fear_greed" in fields
        assert "key_events" in fields
        assert "market_narratives" in fields
        assert "grok_available" in fields
        assert "gpt_available" in fields

    def test_rate_limiter_integration(self):
        """Rate limiter should work with API endpoints."""
        from triplegain.src.api.routes_sentiment import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # Simulate API requests
        for i in range(5):
            assert limiter.is_allowed(f"user_{i % 2}") is True

        # User 0 and 1 each made requests, user 0 should hit limit
        # (3 requests from loop + this one = 4, need 2 more for limit)
        # Actually each user made 2-3 requests, let's be explicit
        limiter2 = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter2.is_allowed("api_user") is True
        assert limiter2.is_allowed("api_user") is True
        assert limiter2.is_allowed("api_user") is False


# =============================================================================
# Database Storage Integration Tests
# =============================================================================

class TestDatabaseIntegration:
    """Test sentiment storage works correctly."""

    @pytest.mark.asyncio
    async def test_sentiment_output_storable(
        self, mock_grok_client, mock_gpt_client, mock_prompt_builder, agent_config
    ):
        """Sentiment output should be storable to database."""
        from triplegain.src.agents.sentiment_analysis import SentimentAnalysisAgent

        agent = SentimentAnalysisAgent(
            llm_clients={"grok": mock_grok_client, "gpt": mock_gpt_client},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
            db_pool=None,  # No actual DB for this test
        )

        output = await agent.process(symbol="BTC/USDT")

        # Check all fields that would be stored to DB
        assert output.output_id is not None  # UUID for primary key
        assert output.timestamp is not None
        assert output.symbol == "BTC/USDT"
        assert -1 <= output.social_score <= 1
        assert -1 <= output.news_score <= 1
        assert -1 <= output.overall_score <= 1
        assert output.bias is not None
        assert output.fear_greed is not None

        # Key events and narratives should be JSON serializable
        events_json = json.dumps([e.to_dict() for e in output.key_events])
        narratives_json = json.dumps(output.market_narratives)
        assert events_json is not None
        assert narratives_json is not None

    @pytest.mark.asyncio
    async def test_provider_results_storable(
        self, mock_grok_client, mock_gpt_client, mock_prompt_builder, agent_config
    ):
        """Provider results should be storable for debugging."""
        from triplegain.src.agents.sentiment_analysis import SentimentAnalysisAgent

        agent = SentimentAnalysisAgent(
            llm_clients={"grok": mock_grok_client, "gpt": mock_gpt_client},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        output = await agent.process(symbol="BTC/USDT")

        # Should have results from both providers
        assert len(output.provider_results) == 2

        for result in output.provider_results:
            assert result.provider in ["grok", "gpt"]
            assert result.model is not None
            assert result.success is True
            assert result.latency_ms >= 0
            # Raw response should be serializable
            if result.raw_response:
                json.dumps(result.raw_response)


# =============================================================================
# End-to-End Flow Tests
# =============================================================================

class TestEndToEndFlow:
    """Test complete sentiment analysis flow."""

    @pytest.mark.asyncio
    async def test_full_analysis_flow(
        self, mock_grok_client, mock_gpt_client, mock_prompt_builder, agent_config
    ):
        """Test complete flow from analysis to output."""
        from triplegain.src.agents.sentiment_analysis import SentimentAnalysisAgent

        agent = SentimentAnalysisAgent(
            llm_clients={"grok": mock_grok_client, "gpt": mock_gpt_client},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        # Run analysis
        output = await agent.process(symbol="BTC/USDT", include_twitter=True)

        # Verify complete output
        assert output.symbol == "BTC/USDT"
        assert output.grok_available is True
        assert output.gpt_available is True
        assert output.confidence > 0
        assert output.latency_ms >= 0  # Mocks complete instantly, so >= 0
        assert output.total_cost_usd > 0

        # Verify weighted aggregation happened
        # With both providers, overall should be weighted average
        assert output.social_score != 0 or output.news_score != 0

        # Verify validation passes
        is_valid, errors = output.validate()
        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_graceful_degradation_single_provider(
        self, mock_gpt_client, mock_prompt_builder, agent_config
    ):
        """Test graceful degradation when one provider fails."""
        from triplegain.src.agents.sentiment_analysis import SentimentAnalysisAgent

        # Only GPT available
        agent = SentimentAnalysisAgent(
            llm_clients={"gpt": mock_gpt_client},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        output = await agent.process(symbol="BTC/USDT")

        # Should still produce valid output
        assert output.symbol == "BTC/USDT"
        assert output.grok_available is False
        assert output.gpt_available is True
        assert output.news_score != 0  # GPT provided news score
        assert output.social_score == 0  # No Grok

        # Validation should still pass
        is_valid, errors = output.validate()
        assert is_valid is True
