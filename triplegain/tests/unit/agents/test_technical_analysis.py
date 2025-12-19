"""
Unit tests for the Technical Analysis Agent.

Tests validate:
- TAOutput validation
- Response parsing
- Fallback handling
- Output structure
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from triplegain.src.agents.technical_analysis import (
    TechnicalAnalysisAgent,
    TAOutput,
    TA_OUTPUT_SCHEMA,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def ta_output() -> TAOutput:
    """Create a valid TAOutput for testing."""
    return TAOutput(
        agent_name="technical_analysis",
        timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDT",
        confidence=0.75,
        reasoning="Strong bullish signals with EMA alignment",
        trend_direction="bullish",
        trend_strength=0.8,
        timeframe_alignment=["1h", "4h", "1d"],
        momentum_score=0.6,
        rsi_signal="neutral",
        macd_signal="bullish",
        resistance_levels=[46000.0, 47000.0],
        support_levels=[44000.0, 43000.0],
        current_position="mid_range",
        primary_signal="EMA 9 crossed above EMA 21",
        secondary_signals=["RSI rising", "Volume increasing"],
        warnings=["Near resistance"],
        bias="long",
    )


@pytest.fixture
def mock_snapshot():
    """Create a mock MarketSnapshot."""
    snapshot = MagicMock()
    snapshot.symbol = "BTC/USDT"
    snapshot.current_price = Decimal("45000")
    snapshot.indicators = {
        'rsi_14': 55.0,
        'macd': {'histogram': 100.0},
        'ema_9': 45100.0,
        'ema_21': 44900.0,
    }
    return snapshot


# =============================================================================
# TAOutput Validation Tests
# =============================================================================

class TestTAOutputValidation:
    """Test TAOutput validation."""

    def test_valid_ta_output(self, ta_output):
        """Valid TAOutput should pass validation."""
        is_valid, errors = ta_output.validate()

        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_trend_direction(self):
        """Invalid trend direction should fail validation."""
        output = TAOutput(
            agent_name="technical_analysis",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            trend_direction="sideways",  # Invalid
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("trend_direction" in e for e in errors)

    def test_invalid_trend_strength(self):
        """Trend strength out of range should fail validation."""
        output = TAOutput(
            agent_name="technical_analysis",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            trend_direction="bullish",
            trend_strength=1.5,  # Invalid
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Trend strength" in e for e in errors)

    def test_invalid_momentum_score(self):
        """Momentum score out of range should fail validation."""
        output = TAOutput(
            agent_name="technical_analysis",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            momentum_score=-1.5,  # Invalid (should be -1 to 1)
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Momentum score" in e for e in errors)

    def test_invalid_bias(self):
        """Invalid bias should fail validation."""
        output = TAOutput(
            agent_name="technical_analysis",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            bias="sideways",  # Invalid
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("bias" in e for e in errors)


# =============================================================================
# Response Parsing Tests
# =============================================================================

class TestResponseParsing:
    """Test LLM response parsing."""

    @pytest.fixture
    def ta_agent(self):
        """Create a TA agent for testing parsing."""
        mock_llm = MagicMock()
        mock_prompt_builder = MagicMock()
        config = {'model': 'test-model', 'timeout_ms': 5000, 'retry_count': 1}

        agent = TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, config)
        return agent

    def test_parse_valid_json_response(self, ta_agent, mock_snapshot):
        """Parse valid JSON response correctly."""
        response = '''
        {
            "trend": {"direction": "bullish", "strength": 0.8},
            "momentum": {"score": 0.5, "rsi_signal": "neutral", "macd_signal": "bullish"},
            "key_levels": {"resistance": [46000], "support": [44000], "current_position": "mid_range"},
            "signals": {"primary": "EMA cross", "secondary": ["RSI up"], "warnings": []},
            "bias": "long",
            "confidence": 0.75,
            "reasoning": "Strong trend"
        }
        '''

        parsed = ta_agent._parse_response(response, mock_snapshot)

        assert parsed['trend']['direction'] == 'bullish'
        assert parsed['trend']['strength'] == 0.8
        assert parsed['bias'] == 'long'
        assert parsed['confidence'] == 0.75

    def test_parse_json_with_extra_text(self, ta_agent, mock_snapshot):
        """Parse JSON embedded in text response."""
        response = '''
        Here is my analysis:

        {
            "trend": {"direction": "bearish", "strength": 0.6},
            "momentum": {"score": -0.3},
            "bias": "short",
            "confidence": 0.65,
            "reasoning": "Downtrend confirmed"
        }

        Let me know if you need more details.
        '''

        parsed = ta_agent._parse_response(response, mock_snapshot)

        assert parsed['trend']['direction'] == 'bearish'
        assert parsed['bias'] == 'short'

    def test_normalize_invalid_direction(self, ta_agent, mock_snapshot):
        """Invalid direction should be normalized to neutral."""
        parsed = {'trend': {'direction': 'sideways', 'strength': 0.5}}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['trend']['direction'] == 'neutral'

    def test_normalize_clamps_strength(self, ta_agent, mock_snapshot):
        """Trend strength should be clamped to [0, 1]."""
        parsed = {'trend': {'direction': 'bullish', 'strength': 1.5}}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['trend']['strength'] == 1.0

    def test_normalize_clamps_confidence(self, ta_agent, mock_snapshot):
        """Confidence should be clamped to [0, 1]."""
        parsed = {'confidence': 2.0}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['confidence'] == 1.0

    def test_fallback_from_indicators(self, ta_agent, mock_snapshot):
        """Fallback parsing should use indicators."""
        # When JSON parsing fails, agent should create output from indicators
        mock_snapshot.indicators = {
            'rsi_14': 75.0,  # Overbought
            'macd': {'histogram': 100.0},  # Positive
        }

        parsed = ta_agent._create_output_from_indicators(mock_snapshot)

        assert parsed['confidence'] == 0.4  # Lower for fallback
        assert 'LLM parsing failed' in str(parsed['signals']['warnings'])


# =============================================================================
# Output Schema Tests
# =============================================================================

class TestOutputSchema:
    """Test output schema definition."""

    def test_schema_has_required_fields(self):
        """Schema should define all required fields."""
        required = TA_OUTPUT_SCHEMA['required']

        assert 'timestamp' in required
        assert 'symbol' in required
        assert 'trend' in required
        assert 'momentum' in required
        assert 'bias' in required
        assert 'confidence' in required

    def test_trend_direction_enum(self):
        """Trend direction should be enum with valid values."""
        trend_props = TA_OUTPUT_SCHEMA['properties']['trend']['properties']
        direction_enum = trend_props['direction']['enum']

        assert 'bullish' in direction_enum
        assert 'bearish' in direction_enum
        assert 'neutral' in direction_enum

    def test_bias_enum(self):
        """Bias should be enum with valid values."""
        bias_enum = TA_OUTPUT_SCHEMA['properties']['bias']['enum']

        assert 'long' in bias_enum
        assert 'short' in bias_enum
        assert 'neutral' in bias_enum


# =============================================================================
# Fallback Output Tests
# =============================================================================

class TestFallbackOutput:
    """Test fallback output creation."""

    @pytest.fixture
    def ta_agent(self):
        """Create a TA agent for testing."""
        mock_llm = MagicMock()
        mock_prompt_builder = MagicMock()
        config = {'model': 'test-model'}

        return TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, config)

    def test_fallback_output_has_zero_confidence(self, ta_agent, mock_snapshot):
        """Fallback output should have zero confidence."""
        output = ta_agent._create_fallback_output(mock_snapshot, "Test error")

        assert output.confidence == 0.0
        assert "LLM error" in output.reasoning
        assert output.bias == "neutral"

    def test_fallback_output_has_warnings(self, ta_agent, mock_snapshot):
        """Fallback output should include warnings."""
        output = ta_agent._create_fallback_output(mock_snapshot, "Connection timeout")

        assert "LLM call failed" in output.warnings


# =============================================================================
# Process Method Tests (with mocked LLM)
# =============================================================================

class TestProcessMethod:
    """Test the process method with mocked LLM calls."""

    @pytest.fixture
    def mock_llm_response(self):
        """Create a standard mock LLM response."""
        response = MagicMock()
        response.text = '''
        {
            "trend": {"direction": "bullish", "strength": 0.75},
            "momentum": {"score": 0.5, "rsi_signal": "neutral", "macd_signal": "bullish"},
            "key_levels": {"resistance": [46000], "support": [44000], "current_position": "mid_range"},
            "signals": {"primary": "EMA cross up", "secondary": ["RSI rising"], "warnings": []},
            "bias": "long",
            "confidence": 0.8,
            "reasoning": "Strong bullish momentum"
        }
        '''
        response.tokens_used = 150
        response.cost_usd = 0.0  # Local LLM
        return response

    @pytest.fixture
    def ta_agent_with_mock(self, mock_llm_response):
        """Create a TA agent with mocked LLM client."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=mock_llm_response)

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "You are a technical analysis assistant."
        mock_prompt.user_message = "Analyze BTC/USDT."
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {'model': 'qwen2.5:7b', 'timeout_ms': 5000, 'retry_count': 2}

        return TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, config)

    @pytest.mark.asyncio
    async def test_process_returns_valid_output(self, ta_agent_with_mock, mock_snapshot):
        """Process should return valid TAOutput."""
        output = await ta_agent_with_mock.process(mock_snapshot)

        assert output is not None
        assert isinstance(output, TAOutput)
        assert output.symbol == "BTC/USDT"
        assert output.trend_direction == "bullish"
        assert output.bias == "long"
        assert output.confidence == 0.8

    @pytest.mark.asyncio
    async def test_process_stores_output(self, ta_agent_with_mock, mock_snapshot):
        """Process should store output in cache."""
        output = await ta_agent_with_mock.process(mock_snapshot)

        assert ta_agent_with_mock._last_output == output

    @pytest.mark.asyncio
    async def test_process_handles_llm_error(self, mock_snapshot):
        """Process should return fallback on LLM error."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("Connection refused"))

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "System"
        mock_prompt.user_message = "User"
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {'model': 'test', 'retry_count': 1}

        agent = TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, config)
        output = await agent.process(mock_snapshot)

        assert output.confidence == 0.0
        assert output.bias == "neutral"
        assert "LLM error" in output.reasoning

    @pytest.mark.asyncio
    async def test_process_retries_on_failure(self, mock_snapshot):
        """Process should retry on LLM failure."""
        response = MagicMock()
        response.text = '{"trend": {"direction": "bullish"}, "bias": "long", "confidence": 0.7, "reasoning": "test"}'
        response.tokens_used = 100

        mock_llm = MagicMock()
        # Fail first time, succeed second time
        mock_llm.generate = AsyncMock(side_effect=[Exception("Timeout"), response])

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "System"
        mock_prompt.user_message = "User"
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {'model': 'test', 'retry_count': 2}

        agent = TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, config)
        output = await agent.process(mock_snapshot)

        # Should have retried and succeeded
        assert output.bias == "long"
        assert mock_llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_process_with_parse_error(self, mock_snapshot):
        """Process should handle JSON parse errors."""
        response = MagicMock()
        response.text = "This is not valid JSON at all"
        response.tokens_used = 50

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=response)

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "System"
        mock_prompt.user_message = "User"
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {'model': 'test', 'retry_count': 0}

        agent = TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, config)
        output = await agent.process(mock_snapshot)

        # Should use indicator-based fallback
        assert output is not None
        assert output.symbol == "BTC/USDT"
        assert output.confidence == 0.4  # Fallback confidence


# =============================================================================
# Additional Parsing Edge Cases
# =============================================================================

class TestParsingEdgeCases:
    """Test edge cases in response parsing."""

    @pytest.fixture
    def ta_agent(self):
        """Create a TA agent for testing parsing."""
        mock_llm = MagicMock()
        mock_prompt_builder = MagicMock()
        config = {'model': 'test-model'}
        return TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, config)

    def test_normalize_missing_trend(self, ta_agent, mock_snapshot):
        """Missing trend should get defaults."""
        parsed = {'bias': 'long', 'confidence': 0.7}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['trend']['direction'] == 'neutral'
        assert normalized['trend']['strength'] == 0.5

    def test_normalize_missing_momentum(self, ta_agent, mock_snapshot):
        """Missing momentum should get defaults."""
        parsed = {'trend': {'direction': 'bullish'}}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['momentum']['score'] == 0.0

    def test_normalize_invalid_rsi_signal(self, ta_agent, mock_snapshot):
        """Invalid RSI signal should be normalized."""
        parsed = {'momentum': {'rsi_signal': 'invalid'}}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['momentum']['rsi_signal'] == 'neutral'

    def test_normalize_invalid_macd_signal(self, ta_agent, mock_snapshot):
        """Invalid MACD signal should be normalized."""
        parsed = {'momentum': {'macd_signal': 'invalid'}}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['momentum']['macd_signal'] == 'neutral'

    def test_normalize_truncates_long_reasoning(self, ta_agent, mock_snapshot):
        """Long reasoning should be truncated."""
        parsed = {'reasoning': 'x' * 600}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert len(normalized['reasoning']) <= 500
        assert normalized['reasoning'].endswith('...')

    def test_normalize_adds_default_reasoning(self, ta_agent, mock_snapshot):
        """Missing reasoning should get default."""
        parsed = {}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert 'BTC/USDT' in normalized['reasoning']

    def test_normalize_negative_strength(self, ta_agent, mock_snapshot):
        """Negative strength should be clamped to 0."""
        parsed = {'trend': {'strength': -0.5}}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['trend']['strength'] == 0.0

    def test_normalize_momentum_score_clamping(self, ta_agent, mock_snapshot):
        """Momentum score should be clamped to [-1, 1]."""
        parsed = {'momentum': {'score': 2.0}}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['momentum']['score'] == 1.0

        parsed = {'momentum': {'score': -2.0}}
        normalized = ta_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['momentum']['score'] == -1.0


# =============================================================================
# Indicator-Based Fallback Tests
# =============================================================================

class TestIndicatorFallback:
    """Test indicator-based fallback logic."""

    @pytest.fixture
    def ta_agent(self):
        """Create a TA agent for testing."""
        mock_llm = MagicMock()
        mock_prompt_builder = MagicMock()
        config = {'model': 'test-model'}
        return TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, config)

    def test_long_bias_from_indicators(self, ta_agent):
        """RSI > 60 and positive MACD should give long bias."""
        snapshot = MagicMock()
        snapshot.indicators = {
            'rsi_14': 65.0,
            'macd': {'histogram': 100.0},
        }

        parsed = ta_agent._create_output_from_indicators(snapshot)

        assert parsed['bias'] == 'long'
        assert parsed['trend']['direction'] == 'bullish'

    def test_short_bias_from_indicators(self, ta_agent):
        """RSI < 40 and negative MACD should give short bias."""
        snapshot = MagicMock()
        snapshot.indicators = {
            'rsi_14': 35.0,
            'macd': {'histogram': -100.0},
        }

        parsed = ta_agent._create_output_from_indicators(snapshot)

        assert parsed['bias'] == 'short'
        assert parsed['trend']['direction'] == 'bearish'

    def test_neutral_bias_from_indicators(self, ta_agent):
        """Mixed signals should give neutral bias."""
        snapshot = MagicMock()
        snapshot.indicators = {
            'rsi_14': 50.0,
            'macd': {'histogram': 0.0},
        }

        parsed = ta_agent._create_output_from_indicators(snapshot)

        assert parsed['bias'] == 'neutral'

    def test_oversold_rsi_signal(self, ta_agent):
        """RSI < 30 should give oversold signal."""
        snapshot = MagicMock()
        snapshot.indicators = {'rsi_14': 25.0}

        parsed = ta_agent._create_output_from_indicators(snapshot)

        assert parsed['momentum']['rsi_signal'] == 'oversold'

    def test_overbought_rsi_signal(self, ta_agent):
        """RSI > 70 should give overbought signal."""
        snapshot = MagicMock()
        snapshot.indicators = {'rsi_14': 75.0}

        parsed = ta_agent._create_output_from_indicators(snapshot)

        assert parsed['momentum']['rsi_signal'] == 'overbought'

    def test_missing_indicators(self, ta_agent):
        """Missing indicators should give defaults."""
        snapshot = MagicMock()
        snapshot.indicators = {}

        parsed = ta_agent._create_output_from_indicators(snapshot)

        assert parsed['bias'] == 'neutral'
        assert parsed['confidence'] == 0.4


# =============================================================================
# Agent Configuration Tests
# =============================================================================

class TestAgentConfiguration:
    """Test agent configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        mock_llm = MagicMock()
        mock_prompt_builder = MagicMock()
        config = {}

        agent = TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, config)

        assert agent.model == 'qwen2.5:7b'
        assert agent.timeout_ms == 5000
        assert agent.retry_count == 2

    def test_custom_config(self):
        """Test custom configuration values."""
        mock_llm = MagicMock()
        mock_prompt_builder = MagicMock()
        config = {
            'model': 'custom-model',
            'timeout_ms': 10000,
            'retry_count': 5,
        }

        agent = TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, config)

        assert agent.model == 'custom-model'
        assert agent.timeout_ms == 10000
        assert agent.retry_count == 5

    def test_get_output_schema(self):
        """get_output_schema should return the TA schema."""
        mock_llm = MagicMock()
        mock_prompt_builder = MagicMock()
        agent = TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, {})

        schema = agent.get_output_schema()

        assert schema == TA_OUTPUT_SCHEMA

    def test_get_analysis_query(self):
        """get_analysis_query should return a query string."""
        mock_llm = MagicMock()
        mock_prompt_builder = MagicMock()
        agent = TechnicalAnalysisAgent(mock_llm, mock_prompt_builder, {})

        query = agent._get_analysis_query()

        assert 'JSON' in query
        assert 'trend' in query
        assert 'momentum' in query
        assert 'bias' in query
