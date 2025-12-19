"""
Unit tests for the Regime Detection Agent.

Tests validate:
- RegimeOutput validation
- Response parsing
- Normalization
- Indicator-based fallback
- Regime parameters
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

from triplegain.src.agents.regime_detection import (
    RegimeDetectionAgent,
    RegimeOutput,
    REGIME_OUTPUT_SCHEMA,
    REGIME_PARAMETERS,
    VALID_REGIMES,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def regime_output() -> RegimeOutput:
    """Create a valid RegimeOutput for testing."""
    return RegimeOutput(
        agent_name="regime_detection",
        timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDT",
        confidence=0.75,
        reasoning="Strong trending market with high ADX",
        regime="trending_bull",
        volatility="normal",
        trend_strength=0.8,
        volume_profile="increasing",
        choppiness=30.0,
        adx_value=35.0,
        position_size_multiplier=1.0,
        stop_loss_multiplier=1.2,
        take_profit_multiplier=2.0,
        entry_strictness="normal",
    )


@pytest.fixture
def mock_snapshot():
    """Create a mock MarketSnapshot."""
    snapshot = MagicMock()
    snapshot.symbol = "BTC/USDT"
    snapshot.current_price = Decimal("45000")
    snapshot.indicators = {
        'adx_14': 35.0,
        'atr_14': 900.0,  # 2% of price
        'choppiness_14': 40.0,
        'rsi_14': 55.0,
    }
    return snapshot


@pytest.fixture
def regime_agent():
    """Create a Regime Detection agent for testing."""
    mock_llm = MagicMock()
    mock_prompt_builder = MagicMock()
    config = {'model': 'qwen2.5:7b', 'timeout_ms': 5000, 'retry_count': 1}

    return RegimeDetectionAgent(mock_llm, mock_prompt_builder, config)


# =============================================================================
# RegimeOutput Validation Tests
# =============================================================================

class TestRegimeOutputValidation:
    """Test RegimeOutput validation."""

    def test_valid_regime_output(self, regime_output):
        """Valid RegimeOutput should pass validation."""
        is_valid, errors = regime_output.validate()

        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_regime(self):
        """Invalid regime should fail validation."""
        output = RegimeOutput(
            agent_name="regime_detection",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            regime="invalid_regime",  # Invalid
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Invalid regime" in e for e in errors)

    def test_invalid_volatility(self):
        """Invalid volatility should fail validation."""
        output = RegimeOutput(
            agent_name="regime_detection",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            regime="trending_bull",
            volatility="very_high",  # Invalid
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Invalid volatility" in e for e in errors)

    def test_invalid_trend_strength(self):
        """Trend strength out of range should fail validation."""
        output = RegimeOutput(
            agent_name="regime_detection",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            regime="trending_bull",
            trend_strength=1.5,  # Invalid - must be 0-1
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Trend strength" in e for e in errors)

    def test_invalid_position_multiplier(self):
        """Position multiplier out of range should fail validation."""
        output = RegimeOutput(
            agent_name="regime_detection",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            regime="trending_bull",
            position_size_multiplier=2.0,  # Invalid - max 1.5
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Position size multiplier" in e for e in errors)

    def test_invalid_stop_loss_multiplier(self):
        """Stop loss multiplier out of range should fail validation."""
        output = RegimeOutput(
            agent_name="regime_detection",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            regime="trending_bull",
            stop_loss_multiplier=3.0,  # Invalid - max 2.0
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Stop loss multiplier" in e for e in errors)


# =============================================================================
# Response Parsing Tests
# =============================================================================

class TestResponseParsing:
    """Test LLM response parsing."""

    def test_parse_valid_json_response(self, regime_agent, mock_snapshot):
        """Parse valid JSON response correctly."""
        response = '''
        {
            "regime": "trending_bull",
            "confidence": 0.8,
            "characteristics": {
                "volatility": "normal",
                "trend_strength": 0.7,
                "volume_profile": "increasing",
                "choppiness": 35,
                "adx_value": 32
            },
            "recommended_adjustments": {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.2,
                "take_profit_multiplier": 2.0,
                "entry_strictness": "normal"
            },
            "reasoning": "Strong uptrend with good momentum"
        }
        '''

        parsed = regime_agent._parse_response(response, mock_snapshot)

        assert parsed['regime'] == 'trending_bull'
        assert parsed['confidence'] == 0.8
        assert parsed['characteristics']['volatility'] == 'normal'
        assert parsed['characteristics']['trend_strength'] == 0.7

    def test_parse_json_with_extra_text(self, regime_agent, mock_snapshot):
        """Parse JSON embedded in text response."""
        response = '''
        Based on my analysis:

        {
            "regime": "ranging",
            "confidence": 0.65,
            "characteristics": {
                "volatility": "low",
                "trend_strength": 0.3
            },
            "reasoning": "No clear trend"
        }

        This indicates a consolidation phase.
        '''

        parsed = regime_agent._parse_response(response, mock_snapshot)

        assert parsed['regime'] == 'ranging'
        assert parsed['confidence'] == 0.65

    def test_normalize_invalid_regime(self, regime_agent, mock_snapshot):
        """Invalid regime should be normalized to ranging."""
        parsed = {'regime': 'sideways', 'confidence': 0.5}
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['regime'] == 'ranging'

    def test_normalize_clamps_confidence(self, regime_agent, mock_snapshot):
        """Confidence should be clamped to [0, 1]."""
        parsed = {'regime': 'trending_bull', 'confidence': 1.5}
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['confidence'] == 1.0

    def test_normalize_clamps_trend_strength(self, regime_agent, mock_snapshot):
        """Trend strength should be clamped to [0, 1]."""
        parsed = {
            'regime': 'trending_bull',
            'confidence': 0.5,
            'characteristics': {'trend_strength': 2.0}
        }
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['characteristics']['trend_strength'] == 1.0

    def test_normalize_invalid_volatility(self, regime_agent, mock_snapshot):
        """Invalid volatility should be normalized to normal."""
        parsed = {
            'regime': 'trending_bull',
            'confidence': 0.5,
            'characteristics': {'volatility': 'very_high'}
        }
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['characteristics']['volatility'] == 'normal'

    def test_normalize_clamps_choppiness(self, regime_agent, mock_snapshot):
        """Choppiness should be clamped to [0, 100]."""
        parsed = {
            'regime': 'choppy',
            'confidence': 0.5,
            'characteristics': {'choppiness': 150}
        }
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['characteristics']['choppiness'] == 100.0

    def test_normalize_clamps_position_multiplier(self, regime_agent, mock_snapshot):
        """Position multiplier should be clamped."""
        parsed = {
            'regime': 'trending_bull',
            'confidence': 0.5,
            'recommended_adjustments': {'position_size_multiplier': 3.0}
        }
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['recommended_adjustments']['position_size_multiplier'] == 1.5

    def test_normalize_truncates_long_reasoning(self, regime_agent, mock_snapshot):
        """Long reasoning should be truncated."""
        parsed = {
            'regime': 'trending_bull',
            'confidence': 0.5,
            'reasoning': 'x' * 400  # Too long
        }
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert len(normalized['reasoning']) <= 300
        assert normalized['reasoning'].endswith('...')


# =============================================================================
# Indicator-Based Output Tests
# =============================================================================

class TestIndicatorBasedOutput:
    """Test output creation from indicators."""

    def test_trending_bull_from_indicators(self, regime_agent, mock_snapshot):
        """High ADX with RSI > 50 should yield trending_bull."""
        mock_snapshot.indicators = {
            'adx_14': 35.0,
            'rsi_14': 60.0,
            'atr_14': 900.0,
            'choppiness_14': 40.0,
        }

        output = regime_agent._create_output_from_indicators(mock_snapshot)

        assert output['regime'] == 'trending_bull'
        assert output['confidence'] == 0.4

    def test_trending_bear_from_indicators(self, regime_agent, mock_snapshot):
        """High ADX with RSI < 50 should yield trending_bear."""
        mock_snapshot.indicators = {
            'adx_14': 35.0,
            'rsi_14': 40.0,
            'atr_14': 900.0,
            'choppiness_14': 40.0,
        }

        output = regime_agent._create_output_from_indicators(mock_snapshot)

        assert output['regime'] == 'trending_bear'

    def test_choppy_from_indicators(self, regime_agent, mock_snapshot):
        """Low ADX with high choppiness should yield choppy."""
        mock_snapshot.indicators = {
            'adx_14': 18.0,
            'rsi_14': 50.0,
            'atr_14': 900.0,
            'choppiness_14': 70.0,
        }

        output = regime_agent._create_output_from_indicators(mock_snapshot)

        assert output['regime'] == 'choppy'

    def test_ranging_from_indicators(self, regime_agent, mock_snapshot):
        """Low ADX without high choppiness should yield ranging."""
        mock_snapshot.indicators = {
            'adx_14': 18.0,
            'rsi_14': 50.0,
            'atr_14': 900.0,
            'choppiness_14': 45.0,
        }

        output = regime_agent._create_output_from_indicators(mock_snapshot)

        assert output['regime'] == 'ranging'

    def test_volatility_low(self, regime_agent, mock_snapshot):
        """Low ATR/price ratio should yield low volatility."""
        mock_snapshot.indicators = {
            'adx_14': 25.0,
            'rsi_14': 50.0,
            'atr_14': 400.0,  # ~0.9% of price
            'choppiness_14': 50.0,
        }

        output = regime_agent._create_output_from_indicators(mock_snapshot)

        assert output['characteristics']['volatility'] == 'low'

    def test_volatility_high(self, regime_agent, mock_snapshot):
        """High ATR/price ratio should yield high volatility."""
        mock_snapshot.indicators = {
            'adx_14': 25.0,
            'rsi_14': 50.0,
            'atr_14': 1800.0,  # ~4% of price
            'choppiness_14': 50.0,
        }

        output = regime_agent._create_output_from_indicators(mock_snapshot)

        assert output['characteristics']['volatility'] == 'high'

    def test_volatility_extreme(self, regime_agent, mock_snapshot):
        """Very high ATR/price ratio should yield extreme volatility."""
        mock_snapshot.indicators = {
            'adx_14': 25.0,
            'rsi_14': 50.0,
            'atr_14': 3000.0,  # ~6.7% of price
            'choppiness_14': 50.0,
        }

        output = regime_agent._create_output_from_indicators(mock_snapshot)

        assert output['characteristics']['volatility'] == 'extreme'


# =============================================================================
# Fallback Output Tests
# =============================================================================

class TestFallbackOutput:
    """Test fallback output creation."""

    def test_fallback_output_is_conservative(self, regime_agent, mock_snapshot):
        """Fallback output should be very conservative."""
        output = regime_agent._create_fallback_output(mock_snapshot, "Connection error")

        assert output.confidence == 0.0
        assert output.regime == 'choppy'
        assert output.position_size_multiplier == 0.25
        assert output.entry_strictness == 'very_strict'
        assert "LLM error" in output.reasoning

    def test_fallback_output_includes_error(self, regime_agent, mock_snapshot):
        """Fallback output should include error message."""
        output = regime_agent._create_fallback_output(mock_snapshot, "Timeout exceeded")

        assert "Timeout exceeded" in output.reasoning


# =============================================================================
# Regime Parameters Tests
# =============================================================================

class TestRegimeParameters:
    """Test regime parameter configuration."""

    def test_all_regimes_have_parameters(self):
        """All valid regimes should have parameters defined."""
        for regime in VALID_REGIMES:
            assert regime in REGIME_PARAMETERS

    def test_trending_bull_parameters(self):
        """Trending bull should have aggressive parameters."""
        params = REGIME_PARAMETERS['trending_bull']

        assert params['position_size_multiplier'] == 1.0
        assert params['max_leverage'] == 5
        assert params['entry_strictness'] == 'normal'

    def test_choppy_parameters(self):
        """Choppy regime should have very conservative parameters."""
        params = REGIME_PARAMETERS['choppy']

        assert params['position_size_multiplier'] == 0.25
        assert params['max_leverage'] == 1
        assert params['entry_strictness'] == 'very_strict'

    def test_volatile_parameters_are_conservative(self):
        """Volatile regimes should have reduced position sizes."""
        for regime in ['volatile_bull', 'volatile_bear']:
            params = REGIME_PARAMETERS[regime]
            assert params['position_size_multiplier'] == 0.5
            assert params['max_leverage'] == 2
            assert params['entry_strictness'] == 'strict'

    def test_get_regime_parameters_method(self, regime_agent):
        """get_regime_parameters should return correct parameters."""
        params = regime_agent.get_regime_parameters('trending_bull')

        assert params['position_size_multiplier'] == 1.0
        assert params['max_leverage'] == 5

    def test_get_regime_parameters_unknown(self, regime_agent):
        """Unknown regime should return ranging parameters."""
        params = regime_agent.get_regime_parameters('unknown_regime')

        assert params == REGIME_PARAMETERS['ranging']

    def test_regime_output_get_parameters(self, regime_output):
        """RegimeOutput.get_regime_parameters should work."""
        params = regime_output.get_regime_parameters()

        assert params['max_leverage'] == 5


# =============================================================================
# Output Schema Tests
# =============================================================================

class TestOutputSchema:
    """Test output schema definition."""

    def test_schema_has_required_fields(self):
        """Schema should define all required fields."""
        required = REGIME_OUTPUT_SCHEMA['required']

        assert 'timestamp' in required
        assert 'symbol' in required
        assert 'regime' in required
        assert 'confidence' in required
        assert 'characteristics' in required

    def test_regime_enum(self):
        """Regime should be enum with all valid values."""
        regime_enum = REGIME_OUTPUT_SCHEMA['properties']['regime']['enum']

        for regime in VALID_REGIMES:
            assert regime in regime_enum

    def test_characteristics_required_fields(self):
        """Characteristics should have required fields."""
        chars_required = REGIME_OUTPUT_SCHEMA['properties']['characteristics']['required']

        assert 'volatility' in chars_required
        assert 'trend_strength' in chars_required
        assert 'volume_profile' in chars_required


# =============================================================================
# Process Method Tests (with mocked LLM)
# =============================================================================

class TestProcessMethod:
    """Test the process method with mocked LLM calls."""

    from unittest.mock import AsyncMock

    @pytest.fixture
    def mock_llm_response(self):
        """Create a standard mock LLM response."""
        from unittest.mock import MagicMock
        response = MagicMock()
        response.text = '''
        {
            "regime": "trending_bull",
            "confidence": 0.85,
            "characteristics": {
                "volatility": "normal",
                "trend_strength": 0.75,
                "volume_profile": "increasing",
                "choppiness": 35.0,
                "adx_value": 32.0
            },
            "recommended_adjustments": {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.2,
                "take_profit_multiplier": 2.0,
                "entry_strictness": "normal"
            },
            "reasoning": "Strong uptrend confirmed"
        }
        '''
        response.tokens_used = 120
        response.cost_usd = 0.0
        return response

    @pytest.fixture
    def regime_agent_with_mock(self, mock_llm_response):
        """Create regime agent with mocked LLM."""
        from unittest.mock import MagicMock, AsyncMock as AM

        mock_llm = MagicMock()
        mock_llm.generate = AM(return_value=mock_llm_response)

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "You are a regime detection assistant."
        mock_prompt.user_message = "Classify the current market regime."
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {'model': 'qwen2.5:7b', 'timeout_ms': 5000, 'retry_count': 2}

        return RegimeDetectionAgent(mock_llm, mock_prompt_builder, config)

    @pytest.mark.asyncio
    async def test_process_returns_valid_output(self, regime_agent_with_mock, mock_snapshot):
        """Process should return valid RegimeOutput."""
        output = await regime_agent_with_mock.process(mock_snapshot)

        assert output is not None
        assert isinstance(output, RegimeOutput)
        assert output.symbol == "BTC/USDT"
        assert output.regime == "trending_bull"
        assert output.confidence == 0.85

    @pytest.mark.asyncio
    async def test_process_applies_regime_parameters(self, regime_agent_with_mock, mock_snapshot):
        """Process should apply regime parameters to output."""
        output = await regime_agent_with_mock.process(mock_snapshot)

        assert output.position_size_multiplier == 1.0
        assert output.entry_strictness == "normal"

    @pytest.mark.asyncio
    async def test_process_stores_output(self, regime_agent_with_mock, mock_snapshot):
        """Process should store output in cache."""
        output = await regime_agent_with_mock.process(mock_snapshot)

        assert regime_agent_with_mock._last_output == output

    @pytest.mark.asyncio
    async def test_process_handles_llm_error(self, mock_snapshot):
        """Process should return fallback on LLM error."""
        from unittest.mock import MagicMock, AsyncMock as AM

        mock_llm = MagicMock()
        mock_llm.generate = AM(side_effect=Exception("Connection refused"))

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "System"
        mock_prompt.user_message = "User"
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {'model': 'test', 'retry_count': 1}

        agent = RegimeDetectionAgent(mock_llm, mock_prompt_builder, config)
        output = await agent.process(mock_snapshot)

        assert output.confidence == 0.0
        assert output.regime == "choppy"  # Conservative fallback
        assert "LLM error" in output.reasoning

    @pytest.mark.asyncio
    async def test_process_retries_on_failure(self, mock_snapshot):
        """Process should retry on LLM failure."""
        from unittest.mock import MagicMock, AsyncMock as AM

        response = MagicMock()
        response.text = '{"regime": "ranging", "confidence": 0.7, "reasoning": "test"}'
        response.tokens_used = 100

        mock_llm = MagicMock()
        mock_llm.generate = AM(side_effect=[Exception("Timeout"), response])

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "System"
        mock_prompt.user_message = "User"
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {'model': 'test', 'retry_count': 2}

        agent = RegimeDetectionAgent(mock_llm, mock_prompt_builder, config)
        output = await agent.process(mock_snapshot)

        assert output.regime == "ranging"
        assert mock_llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_process_with_parse_error(self, mock_snapshot):
        """Process should handle JSON parse errors."""
        from unittest.mock import MagicMock, AsyncMock as AM

        response = MagicMock()
        response.text = "This is not valid JSON at all"
        response.tokens_used = 50

        mock_llm = MagicMock()
        mock_llm.generate = AM(return_value=response)

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "System"
        mock_prompt.user_message = "User"
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {'model': 'test', 'retry_count': 0}

        agent = RegimeDetectionAgent(mock_llm, mock_prompt_builder, config)
        output = await agent.process(mock_snapshot)

        # Should use indicator-based fallback
        assert output is not None
        assert output.symbol == "BTC/USDT"
        assert output.confidence == 0.4  # Fallback confidence


# =============================================================================
# Additional Normalization Tests
# =============================================================================

class TestNormalizationEdgeCases:
    """Test edge cases in normalization."""

    def test_normalize_missing_characteristics(self, regime_agent, mock_snapshot):
        """Missing characteristics should get defaults."""
        parsed = {'regime': 'trending_bull', 'confidence': 0.7}
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert 'characteristics' in normalized
        assert normalized['characteristics']['volatility'] == 'normal'
        assert normalized['characteristics']['trend_strength'] == 0.5

    def test_normalize_missing_adjustments(self, regime_agent, mock_snapshot):
        """Missing adjustments should get defaults from regime."""
        parsed = {'regime': 'trending_bull', 'confidence': 0.7, 'characteristics': {}}
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert 'recommended_adjustments' in normalized
        assert normalized['recommended_adjustments']['position_size_multiplier'] == 1.0

    def test_normalize_clamps_stop_loss_multiplier(self, regime_agent, mock_snapshot):
        """Stop loss multiplier should be clamped."""
        parsed = {
            'regime': 'trending_bull',
            'confidence': 0.5,
            'recommended_adjustments': {'stop_loss_multiplier': 5.0}
        }
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['recommended_adjustments']['stop_loss_multiplier'] == 2.0

    def test_normalize_clamps_take_profit_multiplier(self, regime_agent, mock_snapshot):
        """Take profit multiplier should be clamped."""
        parsed = {
            'regime': 'trending_bull',
            'confidence': 0.5,
            'recommended_adjustments': {'take_profit_multiplier': 10.0}
        }
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['recommended_adjustments']['take_profit_multiplier'] == 3.0  # Max is 3.0

    def test_normalize_invalid_entry_strictness(self, regime_agent, mock_snapshot):
        """Invalid entry strictness should be normalized."""
        parsed = {
            'regime': 'trending_bull',
            'confidence': 0.5,
            'recommended_adjustments': {'entry_strictness': 'invalid'}
        }
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['recommended_adjustments']['entry_strictness'] == 'normal'

    def test_normalize_invalid_volume_profile(self, regime_agent, mock_snapshot):
        """Invalid volume profile should be normalized."""
        parsed = {
            'regime': 'trending_bull',
            'confidence': 0.5,
            'characteristics': {'volume_profile': 'invalid'}
        }
        normalized = regime_agent._normalize_parsed_output(parsed, mock_snapshot)

        assert normalized['characteristics']['volume_profile'] == 'stable'  # Default is stable


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

        agent = RegimeDetectionAgent(mock_llm, mock_prompt_builder, config)

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

        agent = RegimeDetectionAgent(mock_llm, mock_prompt_builder, config)

        assert agent.model == 'custom-model'
        assert agent.timeout_ms == 10000
        assert agent.retry_count == 5

    def test_get_output_schema(self, regime_agent):
        """get_output_schema should return the regime schema."""
        schema = regime_agent.get_output_schema()

        assert schema == REGIME_OUTPUT_SCHEMA

# =============================================================================
# Indicator Fallback Edge Cases
# =============================================================================

class TestIndicatorFallbackEdgeCases:
    """Test edge cases in indicator-based fallback."""

    def test_missing_all_indicators(self, regime_agent):
        """Missing indicators should give defaults."""
        snapshot = MagicMock()
        snapshot.indicators = {}
        snapshot.current_price = Decimal("45000")

        output = regime_agent._create_output_from_indicators(snapshot)

        assert output['regime'] == 'ranging'
        assert output['confidence'] == 0.4

    def test_high_volatility_regime(self, regime_agent, mock_snapshot):
        """Very high volatility should give high_volatility regime."""
        mock_snapshot.indicators = {
            'adx_14': 40.0,  # Strong trend
            'rsi_14': 60.0,  # Bullish
            'atr_14': 2700.0,  # 6% volatility
            'choppiness_14': 35.0,
        }

        output = regime_agent._create_output_from_indicators(mock_snapshot)

        # Should get volatile_bull due to high ATR and bullish RSI
        assert 'volatile' in output['regime'] or output['characteristics']['volatility'] == 'extreme'
