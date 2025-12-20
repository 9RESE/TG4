"""
Unit tests for the Sentiment Analysis Agent.

Tests validate:
- SentimentOutput validation
- SentimentBias and FearGreedLevel enums
- KeyEvent and ProviderResult dataclasses
- Response parsing and normalization
- Provider aggregation logic
- Error handling and fallbacks
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from triplegain.src.agents.sentiment_analysis import (
    SentimentAnalysisAgent,
    SentimentOutput,
    SentimentBias,
    FearGreedLevel,
    KeyEvent,
    ProviderResult,
    EventImpact,
    EventSignificance,
    SENTIMENT_OUTPUT_SCHEMA,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sentiment_output() -> SentimentOutput:
    """Create a valid SentimentOutput for testing."""
    return SentimentOutput(
        agent_name="sentiment_analysis",
        timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDT",
        confidence=0.75,
        reasoning="Strong bullish sentiment from both news and social media",
        bias=SentimentBias.BULLISH,
        social_score=0.4,
        news_score=0.5,
        overall_score=0.45,
        social_analysis="Twitter showing strong bullish momentum with whale accumulation",
        news_analysis="Recent ETF approval driving positive news coverage",
        fear_greed=FearGreedLevel.GREED,
        key_events=[
            KeyEvent(
                event="ETF approval",
                impact=EventImpact.POSITIVE,
                significance=EventSignificance.HIGH,
                source="SEC",
            )
        ],
        market_narratives=["Institutional adoption increasing"],
        grok_available=True,
        gpt_available=True,
    )


@pytest.fixture
def key_event() -> KeyEvent:
    """Create a valid KeyEvent for testing."""
    return KeyEvent(
        event="Major exchange listing",
        impact=EventImpact.POSITIVE,
        significance=EventSignificance.HIGH,
        source="CoinDesk",
    )


@pytest.fixture
def provider_result_grok() -> ProviderResult:
    """Create a successful Grok provider result."""
    return ProviderResult(
        provider="grok",
        model="grok-2",
        bias=SentimentBias.BULLISH,
        score=0.5,
        confidence=0.8,
        events=[
            KeyEvent(
                event="Whale accumulation spotted",
                impact=EventImpact.POSITIVE,
                significance=EventSignificance.MEDIUM,
                source="Twitter",
            )
        ],
        narratives=["Social sentiment turning bullish"],
        reasoning="High engagement on positive news",
        latency_ms=1500,
        tokens_used=500,
        cost_usd=0.005,
        success=True,
    )


@pytest.fixture
def provider_result_gpt() -> ProviderResult:
    """Create a successful GPT provider result."""
    return ProviderResult(
        provider="gpt",
        model="gpt-4-turbo",
        bias=SentimentBias.BULLISH,
        score=0.4,
        confidence=0.7,
        events=[
            KeyEvent(
                event="Regulatory clarity expected",
                impact=EventImpact.POSITIVE,
                significance=EventSignificance.HIGH,
                source="Bloomberg",
            )
        ],
        narratives=["Macro environment improving"],
        reasoning="News coverage increasingly positive",
        latency_ms=2000,
        tokens_used=800,
        cost_usd=0.02,
        success=True,
    )


@pytest.fixture
def mock_llm_clients():
    """Create mock LLM clients for testing."""
    grok_client = AsyncMock()
    grok_client.generate_with_search = AsyncMock(return_value=MagicMock(
        text=json.dumps({
            "bias": "bullish",
            "score": 0.5,
            "confidence": 0.8,
            "key_events": [{"event": "Test event", "impact": "positive", "significance": "high", "source": "test"}],
            "narratives": ["Test narrative"],
            "reasoning": "Test reasoning"
        }),
        tokens_used=500,
        cost_usd=0.005,
    ))

    gpt_client = AsyncMock()
    gpt_client.generate = AsyncMock(return_value=MagicMock(
        text=json.dumps({
            "bias": "bullish",
            "score": 0.4,
            "confidence": 0.7,
            "key_events": [{"event": "News event", "impact": "positive", "significance": "medium", "source": "news"}],
            "narratives": ["News narrative"],
            "reasoning": "News reasoning"
        }),
        tokens_used=800,
        cost_usd=0.02,
    ))

    return {"grok": grok_client, "gpt": gpt_client}


@pytest.fixture
def mock_prompt_builder():
    """Create a mock PromptBuilder."""
    builder = MagicMock()
    builder.build_prompt = MagicMock(return_value=MagicMock(
        system_prompt="You are a sentiment analyst",
        user_message="Analyze BTC sentiment",
    ))
    return builder


@pytest.fixture
def agent_config():
    """Create agent configuration for testing."""
    return {
        "providers": {
            "grok": {
                "enabled": True,
                "model": "grok-2",
                "timeout_ms": 30000,
                "weight": {"social": 0.6, "news": 0.4},
            },
            "gpt": {
                "enabled": True,
                "model": "gpt-4-turbo",
                "timeout_ms": 30000,
                "weight": {"social": 0.4, "news": 0.6},
            },
        },
        "aggregation": {
            "min_providers": 1,
            "confidence_boost_on_agreement": 0.1,
            "min_confidence": 0.3,
        },
        "output": {
            "max_events": 5,
            "max_narratives": 3,
        },
    }


# =============================================================================
# SentimentBias Enum Tests
# =============================================================================

class TestSentimentBias:
    """Test SentimentBias enum and conversions."""

    def test_from_score_very_bullish(self):
        """Score >= 0.6 should be VERY_BULLISH."""
        assert SentimentBias.from_score(0.8) == SentimentBias.VERY_BULLISH
        assert SentimentBias.from_score(0.6) == SentimentBias.VERY_BULLISH
        assert SentimentBias.from_score(1.0) == SentimentBias.VERY_BULLISH

    def test_from_score_bullish(self):
        """Score 0.2 to 0.6 should be BULLISH."""
        assert SentimentBias.from_score(0.4) == SentimentBias.BULLISH
        assert SentimentBias.from_score(0.2) == SentimentBias.BULLISH
        assert SentimentBias.from_score(0.59) == SentimentBias.BULLISH

    def test_from_score_neutral(self):
        """Score -0.2 to 0.2 should be NEUTRAL."""
        assert SentimentBias.from_score(0.0) == SentimentBias.NEUTRAL
        assert SentimentBias.from_score(0.1) == SentimentBias.NEUTRAL
        assert SentimentBias.from_score(-0.1) == SentimentBias.NEUTRAL
        assert SentimentBias.from_score(0.19) == SentimentBias.NEUTRAL

    def test_from_score_bearish(self):
        """Score -0.6 to -0.2 (exclusive) should be BEARISH."""
        assert SentimentBias.from_score(-0.4) == SentimentBias.BEARISH
        assert SentimentBias.from_score(-0.21) == SentimentBias.BEARISH
        assert SentimentBias.from_score(-0.59) == SentimentBias.BEARISH

    def test_from_score_very_bearish(self):
        """Score < -0.6 should be VERY_BEARISH."""
        assert SentimentBias.from_score(-0.8) == SentimentBias.VERY_BEARISH
        assert SentimentBias.from_score(-1.0) == SentimentBias.VERY_BEARISH


class TestFearGreedLevel:
    """Test FearGreedLevel enum and conversions."""

    def test_from_score_extreme_greed(self):
        """Score > 0.6 should be EXTREME_GREED."""
        assert FearGreedLevel.from_score(0.8) == FearGreedLevel.EXTREME_GREED
        assert FearGreedLevel.from_score(1.0) == FearGreedLevel.EXTREME_GREED

    def test_from_score_greed(self):
        """Score 0.2 to 0.6 should be GREED."""
        assert FearGreedLevel.from_score(0.4) == FearGreedLevel.GREED
        assert FearGreedLevel.from_score(0.6) == FearGreedLevel.GREED

    def test_from_score_neutral(self):
        """Score -0.2 to 0.2 should be NEUTRAL."""
        assert FearGreedLevel.from_score(0.0) == FearGreedLevel.NEUTRAL
        assert FearGreedLevel.from_score(0.2) == FearGreedLevel.NEUTRAL

    def test_from_score_fear(self):
        """Score -0.6 to -0.2 (exclusive) should be FEAR."""
        assert FearGreedLevel.from_score(-0.4) == FearGreedLevel.FEAR
        assert FearGreedLevel.from_score(-0.59) == FearGreedLevel.FEAR

    def test_from_score_extreme_fear(self):
        """Score < -0.6 should be EXTREME_FEAR."""
        assert FearGreedLevel.from_score(-0.8) == FearGreedLevel.EXTREME_FEAR
        assert FearGreedLevel.from_score(-1.0) == FearGreedLevel.EXTREME_FEAR


# =============================================================================
# KeyEvent Tests
# =============================================================================

class TestKeyEvent:
    """Test KeyEvent dataclass."""

    def test_to_dict(self, key_event):
        """KeyEvent should serialize to dict correctly."""
        result = key_event.to_dict()

        assert result["event"] == "Major exchange listing"
        assert result["impact"] == "positive"
        assert result["significance"] == "high"
        assert result["source"] == "CoinDesk"

    def test_from_dict(self):
        """KeyEvent should deserialize from dict correctly."""
        data = {
            "event": "Regulatory approval",
            "impact": "positive",
            "significance": "high",
            "source": "Reuters",
        }

        event = KeyEvent.from_dict(data)

        assert event.event == "Regulatory approval"
        assert event.impact == EventImpact.POSITIVE
        assert event.significance == EventSignificance.HIGH
        assert event.source == "Reuters"

    def test_from_dict_defaults(self):
        """KeyEvent should handle missing fields with defaults."""
        data = {}

        event = KeyEvent.from_dict(data)

        assert event.event == "Unknown event"
        assert event.impact == EventImpact.NEUTRAL
        assert event.significance == EventSignificance.MEDIUM
        assert event.source == "unknown"


# =============================================================================
# ProviderResult Tests
# =============================================================================

class TestProviderResult:
    """Test ProviderResult dataclass."""

    def test_to_dict(self, provider_result_grok):
        """ProviderResult should serialize correctly."""
        result = provider_result_grok.to_dict()

        assert result["provider"] == "grok"
        assert result["model"] == "grok-2"
        assert result["bias"] == "bullish"
        assert result["score"] == 0.5
        assert result["confidence"] == 0.8
        assert result["success"] is True
        assert result["error"] is None
        assert len(result["events"]) == 1
        assert len(result["narratives"]) == 1

    def test_failed_result(self):
        """Failed provider result should capture error."""
        result = ProviderResult(
            provider="grok",
            model="grok-2",
            bias=SentimentBias.NEUTRAL,
            score=0.0,
            confidence=0.0,
            success=False,
            error="Connection timeout",
        )

        assert result.success is False
        assert result.error == "Connection timeout"


# =============================================================================
# SentimentOutput Validation Tests
# =============================================================================

class TestSentimentOutputValidation:
    """Test SentimentOutput validation."""

    def test_valid_sentiment_output(self, sentiment_output):
        """Valid SentimentOutput should pass validation."""
        is_valid, errors = sentiment_output.validate()

        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_social_score(self):
        """Social score out of range should fail validation."""
        output = SentimentOutput(
            agent_name="sentiment_analysis",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.5,
            reasoning="Test reasoning here",
            social_score=1.5,  # Invalid - out of range
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("social_score" in e for e in errors)

    def test_invalid_news_score(self):
        """News score out of range should fail validation."""
        output = SentimentOutput(
            agent_name="sentiment_analysis",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.5,
            reasoning="Test reasoning here",
            news_score=-1.5,  # Invalid - out of range
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("news_score" in e for e in errors)

    def test_too_many_events(self):
        """More than 5 key events should fail validation."""
        events = [
            KeyEvent(event=f"Event {i}", impact=EventImpact.NEUTRAL,
                     significance=EventSignificance.LOW, source="test")
            for i in range(6)
        ]

        output = SentimentOutput(
            agent_name="sentiment_analysis",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.5,
            reasoning="Test reasoning here",
            key_events=events,
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("key_events" in e for e in errors)

    def test_too_many_narratives(self):
        """More than 3 narratives should fail validation."""
        output = SentimentOutput(
            agent_name="sentiment_analysis",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.5,
            reasoning="Test reasoning here",
            market_narratives=["n1", "n2", "n3", "n4"],  # Too many
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("narratives" in e for e in errors)


# =============================================================================
# SentimentOutput Serialization Tests
# =============================================================================

class TestSentimentOutputSerialization:
    """Test SentimentOutput serialization."""

    def test_to_dict(self, sentiment_output):
        """SentimentOutput should serialize to dict correctly."""
        result = sentiment_output.to_dict()

        assert result["agent_name"] == "sentiment_analysis"
        assert result["symbol"] == "BTC/USDT"
        assert result["bias"] == "bullish"
        assert result["social_score"] == 0.4
        assert result["news_score"] == 0.5
        assert result["overall_score"] == 0.45
        assert result["social_analysis"] == "Twitter showing strong bullish momentum with whale accumulation"
        assert result["news_analysis"] == "Recent ETF approval driving positive news coverage"
        assert result["fear_greed"] == "greed"
        assert result["grok_available"] is True
        assert result["gpt_available"] is True
        assert len(result["key_events"]) == 1

    def test_to_json(self, sentiment_output):
        """SentimentOutput should serialize to JSON correctly."""
        json_str = sentiment_output.to_json()
        parsed = json.loads(json_str)

        assert parsed["bias"] == "bullish"
        assert parsed["symbol"] == "BTC/USDT"


# =============================================================================
# SentimentAnalysisAgent Tests
# =============================================================================

class TestSentimentAnalysisAgent:
    """Test SentimentAnalysisAgent functionality."""

    def test_agent_initialization(self, mock_llm_clients, mock_prompt_builder, agent_config):
        """Agent should initialize correctly."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        assert agent.agent_name == "sentiment_analysis"
        assert agent.grok_enabled is True
        assert agent.gpt_enabled is True
        assert agent.grok_model == "grok-2"
        assert agent.gpt_model == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_process_with_both_providers(
        self, mock_llm_clients, mock_prompt_builder, agent_config
    ):
        """Agent should query both providers and aggregate results."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        output = await agent.process(symbol="BTC/USDT")

        assert output.symbol == "BTC/USDT"
        assert output.grok_available is True
        assert output.gpt_available is True
        assert output.bias == SentimentBias.BULLISH
        assert -1 <= output.overall_score <= 1

    @pytest.mark.asyncio
    async def test_process_with_single_provider(
        self, mock_prompt_builder, agent_config
    ):
        """Agent should handle single provider failure gracefully."""
        # Only GPT available
        gpt_client = AsyncMock()
        gpt_client.generate = AsyncMock(return_value=MagicMock(
            text=json.dumps({
                "bias": "neutral",
                "score": 0.0,
                "confidence": 0.5,
                "key_events": [],
                "narratives": [],
                "reasoning": "No strong signals"
            }),
            tokens_used=500,
            cost_usd=0.01,
        ))

        agent = SentimentAnalysisAgent(
            llm_clients={"gpt": gpt_client},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        output = await agent.process(symbol="BTC/USDT")

        assert output.symbol == "BTC/USDT"
        assert output.grok_available is False
        assert output.gpt_available is True

    @pytest.mark.asyncio
    async def test_process_with_no_providers(
        self, mock_prompt_builder, agent_config
    ):
        """Agent should return neutral when no providers available."""
        agent = SentimentAnalysisAgent(
            llm_clients={},
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        output = await agent.process(symbol="BTC/USDT")

        assert output.symbol == "BTC/USDT"
        assert output.bias == SentimentBias.NEUTRAL
        assert output.confidence == agent_config["aggregation"]["min_confidence"]


# =============================================================================
# Response Parsing Tests
# =============================================================================

class TestResponseParsing:
    """Test response parsing and normalization."""

    def test_parse_valid_response(self, mock_llm_clients, mock_prompt_builder, agent_config):
        """Valid JSON response should be parsed correctly."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        response = json.dumps({
            "bias": "bullish",
            "score": 0.5,
            "confidence": 0.8,
            "key_events": [
                {"event": "Test", "impact": "positive", "significance": "high", "source": "test"}
            ],
            "narratives": ["Narrative 1"],
            "reasoning": "Good reasoning",
        })

        parsed = agent._parse_provider_response(response, "test")

        assert parsed["bias"] == "bullish"
        assert parsed["score"] == 0.5
        assert parsed["confidence"] == 0.8
        assert len(parsed["key_events"]) == 1

    def test_parse_invalid_json(self, mock_llm_clients, mock_prompt_builder, agent_config):
        """Invalid JSON should return fallback response."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        parsed = agent._parse_provider_response("not valid json", "test")

        assert parsed["bias"] == "neutral"
        assert parsed["score"] == 0.0
        assert parsed["confidence"] == 0.3

    def test_normalize_out_of_range_score(
        self, mock_llm_clients, mock_prompt_builder, agent_config
    ):
        """Out of range scores should be clamped."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        parsed = agent._normalize_parsed_output({
            "bias": "bullish",
            "score": 1.5,  # Out of range
            "confidence": 1.2,  # Out of range
        })

        assert parsed["score"] == 1.0
        assert parsed["confidence"] == 1.0

    def test_normalize_invalid_bias(
        self, mock_llm_clients, mock_prompt_builder, agent_config
    ):
        """Invalid bias should default to neutral."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        parsed = agent._normalize_parsed_output({
            "bias": "invalid_bias",
            "score": 0.0,
        })

        assert parsed["bias"] == "neutral"


# =============================================================================
# Aggregation Tests
# =============================================================================

class TestAggregation:
    """Test result aggregation logic."""

    def test_aggregate_dual_providers(
        self, mock_llm_clients, mock_prompt_builder, agent_config,
        provider_result_grok, provider_result_gpt
    ):
        """Both providers should contribute their respective scores."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        output = agent._aggregate_results(
            "BTC/USDT",
            [provider_result_grok, provider_result_gpt]
        )

        # Grok provides social sentiment, GPT provides news sentiment
        assert output.grok_available is True
        assert output.gpt_available is True
        # Social score comes from Grok
        assert output.social_score == provider_result_grok.score
        # News score comes from GPT
        assert output.news_score == provider_result_gpt.score
        # Analysis reasoning comes from each provider
        assert output.social_analysis == provider_result_grok.reasoning
        assert output.news_analysis == provider_result_gpt.reasoning
        # Overall is weighted average
        assert output.overall_score != 0.0
        assert output.confidence > 0.0

    def test_aggregate_mixed_sentiment_sources(
        self, mock_llm_clients, mock_prompt_builder, agent_config
    ):
        """Social and news sentiment can differ - they measure different things."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        grok_result = ProviderResult(
            provider="grok",
            model="grok-2",
            bias=SentimentBias.BULLISH,
            score=0.6,  # Social sentiment is bullish
            confidence=0.8,
            success=True,
        )
        gpt_result = ProviderResult(
            provider="gpt",
            model="gpt-4-turbo",
            bias=SentimentBias.BEARISH,  # News sentiment is bearish
            score=-0.4,
            confidence=0.7,
            success=True,
        )

        output = agent._aggregate_results(
            "BTC/USDT",
            [grok_result, gpt_result]
        )

        # Both sources are valid and contribute their own scores
        assert output.social_score == 0.6  # From Grok
        assert output.news_score == -0.4   # From GPT
        # Overall is weighted combination
        assert -1.0 <= output.overall_score <= 1.0

    def test_aggregate_failed_provider(
        self, mock_llm_clients, mock_prompt_builder, agent_config
    ):
        """Failed provider should be excluded from aggregation."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        grok_result = ProviderResult(
            provider="grok",
            model="grok-2",
            bias=SentimentBias.NEUTRAL,
            score=0.0,
            confidence=0.0,
            success=False,
            error="Connection failed",
        )
        gpt_result = ProviderResult(
            provider="gpt",
            model="gpt-4-turbo",
            bias=SentimentBias.BULLISH,
            score=0.5,
            confidence=0.8,
            success=True,
        )

        output = agent._aggregate_results(
            "BTC/USDT",
            [grok_result, gpt_result]
        )

        assert output.grok_available is False
        assert output.gpt_available is True
        assert output.bias == SentimentBias.BULLISH


# =============================================================================
# Event Deduplication Tests
# =============================================================================

class TestEventDeduplication:
    """Test event deduplication logic."""

    def test_deduplicate_similar_events(
        self, mock_llm_clients, mock_prompt_builder, agent_config
    ):
        """Similar events should be deduplicated."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        events = [
            KeyEvent("ETF approved by SEC", EventImpact.POSITIVE, EventSignificance.HIGH, "SEC"),
            KeyEvent("ETF approved by SEC", EventImpact.POSITIVE, EventSignificance.MEDIUM, "CoinDesk"),
            KeyEvent("New partnership announced", EventImpact.POSITIVE, EventSignificance.MEDIUM, "Twitter"),
        ]

        deduped = agent._deduplicate_events(events)

        assert len(deduped) == 2  # One duplicate removed

    def test_prioritize_high_significance(
        self, mock_llm_clients, mock_prompt_builder, agent_config
    ):
        """High significance events should be prioritized."""
        agent = SentimentAnalysisAgent(
            llm_clients=mock_llm_clients,
            prompt_builder=mock_prompt_builder,
            config=agent_config,
        )

        events = [
            KeyEvent("Low event", EventImpact.NEUTRAL, EventSignificance.LOW, "src"),
            KeyEvent("High event", EventImpact.POSITIVE, EventSignificance.HIGH, "src"),
            KeyEvent("Medium event", EventImpact.POSITIVE, EventSignificance.MEDIUM, "src"),
        ]

        deduped = agent._deduplicate_events(events)

        # High significance should come first
        assert deduped[0].significance == EventSignificance.HIGH


# =============================================================================
# Schema Tests
# =============================================================================

class TestSentimentOutputSchema:
    """Test the JSON schema for sentiment output."""

    def test_schema_structure(self):
        """Schema should have correct structure."""
        assert SENTIMENT_OUTPUT_SCHEMA["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert "timestamp" in SENTIMENT_OUTPUT_SCHEMA["properties"]
        assert "symbol" in SENTIMENT_OUTPUT_SCHEMA["properties"]
        assert "bias" in SENTIMENT_OUTPUT_SCHEMA["properties"]

    def test_bias_enum_values(self):
        """Schema should list valid bias values."""
        bias_schema = SENTIMENT_OUTPUT_SCHEMA["properties"]["bias"]
        valid_biases = ["very_bullish", "bullish", "neutral", "bearish", "very_bearish"]
        assert bias_schema["enum"] == valid_biases
