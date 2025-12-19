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
