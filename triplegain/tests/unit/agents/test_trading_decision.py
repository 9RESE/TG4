"""
Unit tests for the Trading Decision Agent.

Tests validate:
- TradingDecisionOutput validation
- ModelDecision dataclass
- ConsensusResult calculation
- Decision parsing
- Consensus building
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock

from triplegain.src.agents.trading_decision import (
    TradingDecisionAgent,
    TradingDecisionOutput,
    ModelDecision,
    ConsensusResult,
    VALID_ACTIONS,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def trading_output() -> TradingDecisionOutput:
    """Create a valid TradingDecisionOutput for testing."""
    return TradingDecisionOutput(
        agent_name="trading_decision",
        timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDT",
        confidence=0.75,
        reasoning="Strong consensus for BUY with 5/6 models agreeing",
        action="BUY",
        consensus_strength=0.83,
        entry_price=Decimal("45000.0"),  # P3-02: Use Decimal for prices
        stop_loss=Decimal("44100.0"),
        take_profit=Decimal("48000.0"),
        position_size_pct=10.0,
        votes={"BUY": 5, "HOLD": 1},
        agreeing_models=5,
        total_models=6,
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
    }
    return snapshot


@pytest.fixture
def trading_agent():
    """Create a Trading Decision agent for testing."""
    mock_clients = {
        'ollama': MagicMock(),
        'openai': MagicMock(),
        'anthropic': MagicMock(),
        'deepseek': MagicMock(),
        'xai': MagicMock(),
    }
    mock_prompt_builder = MagicMock()
    config = {
        'models': {
            'qwen': {'provider': 'ollama', 'model': 'qwen2.5:7b'},
            'gpt4': {'provider': 'openai', 'model': 'gpt-4-turbo'},
            'grok': {'provider': 'xai', 'model': 'grok-2'},
            'deepseek': {'provider': 'deepseek', 'model': 'deepseek-chat'},
            'sonnet': {'provider': 'anthropic', 'model': 'claude-3-5-sonnet'},
            'opus': {'provider': 'anthropic', 'model': 'claude-3-opus'},
        },
        'min_consensus': 0.5,
        'high_consensus': 0.67,
        'timeout_seconds': 30,
        # Set min_quorum to 1 for unit tests that test core consensus logic with few models
        # Production default is 4 (see P1-01 in phase-2b-findings.md)
        'min_quorum': 1,
    }

    return TradingDecisionAgent(mock_clients, mock_prompt_builder, config)


# =============================================================================
# TradingDecisionOutput Validation Tests
# =============================================================================

class TestTradingDecisionOutputValidation:
    """Test TradingDecisionOutput validation."""

    def test_valid_output(self, trading_output):
        """Valid TradingDecisionOutput should pass validation."""
        is_valid, errors = trading_output.validate()

        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_action(self):
        """Invalid action should fail validation."""
        output = TradingDecisionOutput(
            agent_name="trading_decision",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            action="INVALID_ACTION",
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Invalid action" in e for e in errors)

    def test_invalid_consensus_strength(self):
        """Consensus strength out of range should fail validation."""
        output = TradingDecisionOutput(
            agent_name="trading_decision",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            action="HOLD",
            consensus_strength=1.5,  # Invalid
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Consensus strength" in e for e in errors)

    def test_buy_requires_entry_price(self):
        """BUY action requires entry price."""
        output = TradingDecisionOutput(
            agent_name="trading_decision",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            action="BUY",
            entry_price=None,  # Missing
            stop_loss=44100.0,
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Entry price required" in e for e in errors)

    def test_buy_requires_stop_loss(self):
        """BUY action requires stop loss."""
        output = TradingDecisionOutput(
            agent_name="trading_decision",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            action="BUY",
            entry_price=45000.0,
            stop_loss=None,  # Missing
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Stop loss required" in e for e in errors)

    def test_sell_requires_entry_and_stop(self):
        """SELL action requires entry price and stop loss."""
        output = TradingDecisionOutput(
            agent_name="trading_decision",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            action="SELL",
            entry_price=None,
            stop_loss=None,
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("Entry price required" in e for e in errors)
        assert any("Stop loss required" in e for e in errors)

    def test_hold_doesnt_require_prices(self):
        """HOLD action doesn't require entry/stop prices."""
        output = TradingDecisionOutput(
            agent_name="trading_decision",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            action="HOLD",
        )

        is_valid, errors = output.validate()

        assert is_valid is True


# =============================================================================
# ModelDecision Tests
# =============================================================================

class TestModelDecision:
    """Test ModelDecision dataclass."""

    def test_valid_decision(self):
        """Valid ModelDecision should be valid."""
        decision = ModelDecision(
            model_name="gpt4",
            provider="openai",
            action="BUY",
            confidence=0.8,
            entry_price=45000.0,
            stop_loss=44100.0,
            reasoning="Strong uptrend",
        )

        assert decision.is_valid() is True

    def test_invalid_action(self):
        """Invalid action should make decision invalid."""
        decision = ModelDecision(
            model_name="gpt4",
            provider="openai",
            action="INVALID",
            confidence=0.8,
        )

        assert decision.is_valid() is False

    def test_error_makes_invalid(self):
        """Error should make decision invalid."""
        decision = ModelDecision(
            model_name="gpt4",
            provider="openai",
            action="BUY",
            confidence=0.8,
            error="Connection timeout",
        )

        assert decision.is_valid() is False


# =============================================================================
# ConsensusResult Tests
# =============================================================================

class TestConsensusResult:
    """Test ConsensusResult dataclass."""

    def test_to_dict(self):
        """to_dict should serialize properly."""
        decisions = [
            ModelDecision(
                model_name="gpt4",
                provider="openai",
                action="BUY",
                confidence=0.8,
                reasoning="Uptrend confirmed",
            ),
        ]

        consensus = ConsensusResult(
            final_action="BUY",
            final_confidence=0.8,
            consensus_strength=0.83,
            votes={"BUY": 5, "HOLD": 1},
            total_models=6,
            agreeing_models=5,
            model_decisions=decisions,
            total_cost_usd=0.05,
            total_tokens=1500,
        )

        result = consensus.to_dict()

        assert result['final_action'] == "BUY"
        assert result['consensus_strength'] == 0.83
        assert result['votes'] == {"BUY": 5, "HOLD": 1}
        assert len(result['model_decisions']) == 1


# =============================================================================
# Decision Parsing Tests
# =============================================================================

class TestDecisionParsing:
    """Test LLM response parsing for decisions."""

    def test_parse_valid_json(self, trading_agent):
        """Parse valid JSON response correctly."""
        response = '''
        {
            "action": "BUY",
            "confidence": 0.85,
            "entry_price": 45000,
            "stop_loss": 44100,
            "take_profit": 48000,
            "position_size_pct": 10,
            "reasoning": "Strong uptrend with momentum"
        }
        '''

        parsed = trading_agent._parse_decision(response, "test")

        assert parsed['action'] == 'BUY'
        assert parsed['confidence'] == 0.85
        assert parsed['entry_price'] == 45000
        assert parsed['stop_loss'] == 44100

    def test_parse_json_with_text(self, trading_agent):
        """Parse JSON embedded in text."""
        response = '''
        Based on analysis:

        {
            "action": "SELL",
            "confidence": 0.75,
            "reasoning": "Bearish divergence"
        }

        This is my recommendation.
        '''

        parsed = trading_agent._parse_decision(response, "test")

        assert parsed['action'] == 'SELL'
        assert parsed['confidence'] == 0.75

    def test_normalize_invalid_action(self, trading_agent):
        """Invalid action should be normalized to HOLD."""
        parsed = {'action': 'WAIT', 'confidence': 0.5}
        normalized = trading_agent._normalize_decision(parsed)

        assert normalized['action'] == 'HOLD'

    def test_normalize_clamps_confidence(self, trading_agent):
        """Confidence should be clamped to [0, 1]."""
        parsed = {'action': 'BUY', 'confidence': 1.5}
        normalized = trading_agent._normalize_decision(parsed)

        assert normalized['confidence'] == 1.0

    def test_normalize_handles_invalid_prices(self, trading_agent):
        """Invalid prices should become None."""
        parsed = {
            'action': 'BUY',
            'confidence': 0.5,
            'entry_price': 'invalid',
            'stop_loss': None,
        }
        normalized = trading_agent._normalize_decision(parsed)

        assert normalized['entry_price'] is None
        assert normalized['stop_loss'] is None

    def test_normalize_truncates_reasoning(self, trading_agent):
        """Long reasoning should be truncated."""
        parsed = {
            'action': 'BUY',
            'confidence': 0.5,
            'reasoning': 'x' * 400
        }
        normalized = trading_agent._normalize_decision(parsed)

        assert len(normalized['reasoning']) <= 300
        assert normalized['reasoning'].endswith('...')

    def test_extract_buy_from_text(self, trading_agent):
        """Extract BUY action from unstructured text."""
        response = "I recommend to BUY at current levels."

        parsed = trading_agent._extract_decision_from_text(response)

        assert parsed['action'] == 'BUY'
        assert parsed['confidence'] == 0.4

    def test_extract_sell_from_text(self, trading_agent):
        """Extract SELL action from unstructured text."""
        response = "We should SELL here due to resistance."

        parsed = trading_agent._extract_decision_from_text(response)

        assert parsed['action'] == 'SELL'

    def test_extract_hold_from_text(self, trading_agent):
        """Default to HOLD when unclear."""
        response = "The market is uncertain, stay cautious."

        parsed = trading_agent._extract_decision_from_text(response)

        assert parsed['action'] == 'HOLD'


# =============================================================================
# Consensus Calculation Tests
# =============================================================================

class TestConsensusCalculation:
    """Test consensus calculation from model decisions."""

    def test_unanimous_consensus(self, trading_agent):
        """Unanimous agreement should have high consensus strength."""
        decisions = [
            ModelDecision(model_name=f"model_{i}", provider="test", action="BUY", confidence=0.8)
            for i in range(6)
        ]

        consensus = trading_agent._calculate_consensus(decisions)

        assert consensus.final_action == "BUY"
        assert consensus.consensus_strength == 1.0
        assert consensus.agreeing_models == 6

    def test_majority_consensus(self, trading_agent):
        """Clear majority (>50%) should win."""
        # 4/6 = 67% consensus - clear majority
        decisions = [
            ModelDecision(model_name="m1", provider="test", action="BUY", confidence=0.8),
            ModelDecision(model_name="m2", provider="test", action="BUY", confidence=0.7),
            ModelDecision(model_name="m3", provider="test", action="BUY", confidence=0.75),
            ModelDecision(model_name="m4", provider="test", action="BUY", confidence=0.65),
            ModelDecision(model_name="m5", provider="test", action="SELL", confidence=0.8),
            ModelDecision(model_name="m6", provider="test", action="HOLD", confidence=0.5),
        ]

        consensus = trading_agent._calculate_consensus(decisions)

        assert consensus.final_action == "BUY"
        assert consensus.agreeing_models == 4
        assert consensus.votes["BUY"] == 4

    def test_weak_consensus_forces_hold(self, trading_agent):
        """Weak consensus (<=50%) should force HOLD to prevent risky trades."""
        # 3/6 = 50% consensus - too weak, should force HOLD
        decisions = [
            ModelDecision(model_name="m1", provider="test", action="BUY", confidence=0.8),
            ModelDecision(model_name="m2", provider="test", action="BUY", confidence=0.7),
            ModelDecision(model_name="m3", provider="test", action="BUY", confidence=0.75),
            ModelDecision(model_name="m4", provider="test", action="SELL", confidence=0.8),
            ModelDecision(model_name="m5", provider="test", action="HOLD", confidence=0.6),
            ModelDecision(model_name="m6", provider="test", action="HOLD", confidence=0.5),
        ]

        consensus = trading_agent._calculate_consensus(decisions)

        # CRITICAL: Weak consensus forces HOLD - prevents acting on uncertain signals
        assert consensus.final_action == "HOLD"
        assert consensus.votes["BUY"] == 3

    def test_no_valid_decisions(self, trading_agent):
        """No valid decisions should return HOLD."""
        decisions = [
            ModelDecision(model_name="m1", provider="test", action="BUY", confidence=0.8, error="Error"),
            ModelDecision(model_name="m2", provider="test", action="INVALID", confidence=0.7),
        ]

        consensus = trading_agent._calculate_consensus(decisions)

        assert consensus.final_action == "HOLD"
        assert consensus.consensus_strength == 0.0
        assert consensus.final_confidence == 0.0

    def test_average_confidence_from_agreeing(self, trading_agent):
        """Average confidence should be from agreeing models only."""
        decisions = [
            ModelDecision(model_name="m1", provider="test", action="BUY", confidence=0.8),
            ModelDecision(model_name="m2", provider="test", action="BUY", confidence=0.6),
            ModelDecision(model_name="m3", provider="test", action="SELL", confidence=0.9),  # Ignored
        ]

        consensus = trading_agent._calculate_consensus(decisions)

        assert consensus.final_action == "BUY"
        assert consensus.final_confidence == 0.7  # (0.8 + 0.6) / 2

    def test_average_prices_from_agreeing(self, trading_agent):
        """Average trade parameters from agreeing models."""
        decisions = [
            ModelDecision(
                model_name="m1", provider="test", action="BUY", confidence=0.8,
                entry_price=Decimal("45000"), stop_loss=Decimal("44000"), take_profit=Decimal("47000")
            ),
            ModelDecision(
                model_name="m2", provider="test", action="BUY", confidence=0.7,
                entry_price=Decimal("45100"), stop_loss=Decimal("44200"), take_profit=Decimal("47200")
            ),
            ModelDecision(model_name="m3", provider="test", action="SELL", confidence=0.9),
        ]

        consensus = trading_agent._calculate_consensus(decisions)

        assert consensus.avg_entry_price == Decimal("45050")  # (45000 + 45100) / 2
        assert consensus.avg_stop_loss == Decimal("44100")  # (44000 + 44200) / 2

    def test_total_cost_tracking(self, trading_agent):
        """Total cost should sum all model costs."""
        decisions = [
            ModelDecision(
                model_name="m1", provider="test", action="BUY", confidence=0.8,
                cost_usd=0.01, tokens_used=500, latency_ms=100
            ),
            ModelDecision(
                model_name="m2", provider="test", action="BUY", confidence=0.7,
                cost_usd=0.02, tokens_used=700, latency_ms=200
            ),
            ModelDecision(
                model_name="m3", provider="test", action="BUY", confidence=0.6,
                cost_usd=0.03, tokens_used=300, latency_ms=150
            ),
        ]

        consensus = trading_agent._calculate_consensus(decisions)

        assert consensus.total_cost_usd == 0.06
        assert consensus.total_tokens == 1500
        assert consensus.total_latency_ms == 200  # Max, since parallel


# =============================================================================
# Consensus Reasoning Tests
# =============================================================================

class TestConsensusReasoning:
    """Test consensus reasoning builder."""

    def test_builds_readable_reasoning(self, trading_agent):
        """Should build readable consensus reasoning."""
        decisions = [
            ModelDecision(
                model_name="gpt4", provider="openai", action="BUY",
                confidence=0.8, reasoning="Strong uptrend"
            ),
        ]

        consensus = ConsensusResult(
            final_action="BUY",
            final_confidence=0.8,
            consensus_strength=0.83,
            votes={"BUY": 5, "HOLD": 1},
            total_models=6,
            agreeing_models=5,
            model_decisions=decisions,
        )

        reasoning = trading_agent._build_consensus_reasoning(consensus)

        assert "BUY" in reasoning
        assert "5/6" in reasoning
        assert "83%" in reasoning


# =============================================================================
# Valid Actions Tests
# =============================================================================

class TestValidActions:
    """Test valid actions list."""

    def test_all_expected_actions_present(self):
        """All expected actions should be in VALID_ACTIONS."""
        assert "BUY" in VALID_ACTIONS
        assert "SELL" in VALID_ACTIONS
        assert "HOLD" in VALID_ACTIONS
        assert "CLOSE_LONG" in VALID_ACTIONS
        assert "CLOSE_SHORT" in VALID_ACTIONS


# =============================================================================
# Agent Stats Tests
# =============================================================================

class TestAgentStats:
    """Test agent statistics tracking."""

    def test_get_stats_initial(self, trading_agent):
        """Initial stats should be zero."""
        stats = trading_agent.get_stats()

        assert stats['total_invocations'] == 0
        assert stats['total_cost_usd'] == 0.0
        assert stats['avg_cost_per_decision'] == 0
        assert len(stats['models_configured']) == 6

    def test_stats_show_configured_models(self, trading_agent):
        """Stats should list configured models."""
        stats = trading_agent.get_stats()

        assert 'qwen' in stats['models_configured']
        assert 'gpt4' in stats['models_configured']
        assert 'sonnet' in stats['models_configured']


# =============================================================================
# Process Method Tests (with mocked LLMs)
# =============================================================================

class TestProcessMethod:
    """Test the process method with mocked LLM calls."""

    @pytest.fixture
    def mock_llm_response(self):
        """Create a standard mock LLM response."""
        response = MagicMock()
        response.text = '''
        {
            "action": "BUY",
            "confidence": 0.8,
            "entry_price": 45000,
            "stop_loss": 44100,
            "take_profit": 48000,
            "reasoning": "Strong bullish trend"
        }
        '''
        response.tokens_used = 200
        response.cost_usd = 0.01
        response.latency_ms = 500
        return response

    @pytest.fixture
    def trading_agent_with_mocks(self, mock_llm_response):
        """Create a trading agent with mocked LLM clients."""
        from unittest.mock import AsyncMock

        # Create async mocks for generate method
        mock_clients = {}
        for provider in ['ollama', 'openai', 'anthropic', 'deepseek', 'xai']:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=mock_llm_response)
            mock_clients[provider] = mock_client

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "You are a trading assistant."
        mock_prompt.user_message = "Analyze BTC/USDT."
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {
            'models': {
                'qwen': {'provider': 'ollama', 'model': 'qwen2.5:7b'},
                'gpt4': {'provider': 'openai', 'model': 'gpt-4-turbo'},
                'grok': {'provider': 'xai', 'model': 'grok-2'},
                'deepseek': {'provider': 'deepseek', 'model': 'deepseek-chat'},
                'sonnet': {'provider': 'anthropic', 'model': 'claude-3-5-sonnet'},
                'opus': {'provider': 'anthropic', 'model': 'claude-3-opus'},
            },
            'min_consensus': 0.5,
            'high_consensus': 0.67,
            'timeout_seconds': 30,
        }

        return TradingDecisionAgent(mock_clients, mock_prompt_builder, config)

    @pytest.mark.asyncio
    async def test_process_builds_consensus(self, trading_agent_with_mocks, mock_snapshot):
        """Process should return consensus decision."""
        output = await trading_agent_with_mocks.process(mock_snapshot)

        assert output is not None
        assert output.action == "BUY"
        assert output.symbol == "BTC/USDT"
        assert output.consensus_strength > 0
        assert output.agreeing_models > 0

    @pytest.mark.asyncio
    async def test_process_tracks_costs(self, trading_agent_with_mocks, mock_snapshot):
        """Process should track costs from all models."""
        output = await trading_agent_with_mocks.process(mock_snapshot)

        # With 6 models at 0.01 each
        assert output.latency_ms >= 0
        assert output.tokens_used > 0

    @pytest.mark.asyncio
    async def test_process_handles_mixed_responses(self, mock_snapshot):
        """Process should handle models returning different actions."""
        from unittest.mock import AsyncMock

        responses = {
            'ollama': '{"action": "BUY", "confidence": 0.8}',
            'openai': '{"action": "BUY", "confidence": 0.75}',
            'anthropic': '{"action": "SELL", "confidence": 0.7}',
            'deepseek': '{"action": "BUY", "confidence": 0.85}',
            'xai': '{"action": "HOLD", "confidence": 0.6}',
        }

        mock_clients = {}
        for provider, resp_text in responses.items():
            resp = MagicMock()
            resp.text = resp_text
            resp.tokens_used = 150
            resp.cost_usd = 0.01
            resp.latency_ms = 100  # Add latency_ms to fix comparison
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=resp)
            mock_clients[provider] = mock_client

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "You are a trading assistant."
        mock_prompt.user_message = "Analyze."
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {
            'models': {
                'qwen': {'provider': 'ollama', 'model': 'qwen2.5:7b'},
                'gpt4': {'provider': 'openai', 'model': 'gpt-4-turbo'},
                'grok': {'provider': 'xai', 'model': 'grok-2'},
                'deepseek': {'provider': 'deepseek', 'model': 'deepseek-chat'},
                'sonnet': {'provider': 'anthropic', 'model': 'claude-3-5-sonnet'},
            },
            'min_consensus': 0.5,
        }

        agent = TradingDecisionAgent(mock_clients, mock_prompt_builder, config)
        output = await agent.process(mock_snapshot)

        # BUY should win with 3/5 models
        assert output.action == "BUY"
        assert output.votes["BUY"] == 3

    @pytest.mark.asyncio
    async def test_process_handles_llm_errors(self, mock_snapshot):
        """Process should handle LLM errors gracefully."""
        from unittest.mock import AsyncMock

        mock_clients = {}
        for provider in ['ollama', 'openai', 'anthropic', 'deepseek', 'xai']:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(side_effect=Exception("LLM Error"))
            mock_clients[provider] = mock_client

        mock_prompt_builder = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = "System"
        mock_prompt.user_message = "User"
        mock_prompt_builder.build_prompt.return_value = mock_prompt

        config = {
            'models': {
                'qwen': {'provider': 'ollama', 'model': 'qwen2.5:7b'},
            },
            'min_consensus': 0.5,
        }

        agent = TradingDecisionAgent(mock_clients, mock_prompt_builder, config)
        output = await agent.process(mock_snapshot)

        # Should return HOLD with no consensus
        assert output.action == "HOLD"
        assert output.consensus_strength == 0.0


# =============================================================================
# Output Schema Tests
# =============================================================================

class TestOutputSchema:
    """Test output schema method."""

    def test_get_output_schema(self, trading_agent):
        """get_output_schema should return valid JSON schema."""
        schema = trading_agent.get_output_schema()

        assert isinstance(schema, dict)
        assert 'type' in schema
        assert 'properties' in schema
        assert 'action' in schema['properties']
        assert 'confidence' in schema['properties']


# =============================================================================
# TradingDecisionOutput Serialization Tests
# =============================================================================

class TestTradingDecisionOutputSerialization:
    """Test TradingDecisionOutput serialization."""

    def test_to_dict(self, trading_output):
        """to_dict should serialize all fields."""
        result = trading_output.to_dict()

        assert result['action'] == 'BUY'
        assert result['consensus_strength'] == 0.83
        assert result['votes'] == {'BUY': 5, 'HOLD': 1}
        assert result['entry_price'] == 45000.0
        assert result['stop_loss'] == 44100.0

    def test_to_json(self, trading_output):
        """to_json should produce valid JSON."""
        import json

        json_str = trading_output.to_json()
        parsed = json.loads(json_str)

        assert parsed['action'] == 'BUY'
        assert 'timestamp' in parsed


# =============================================================================
# Database Methods Tests
# =============================================================================

class TestDatabaseMethods:
    """Test database persistence methods."""

    @pytest.fixture
    def trading_output_with_decisions(self):
        """Create trading output with model decisions."""
        return TradingDecisionOutput(
            agent_name="trading_decision",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            action="BUY",
            confidence=0.85,
            consensus_strength=0.83,
            votes={"BUY": 5, "HOLD": 1},
            total_models=6,
            agreeing_models=5,
            reasoning="Strong consensus for BUY",
            model_decisions=[
                ModelDecision(
                    model_name="gpt4",
                    provider="openai",
                    action="BUY",
                    confidence=0.85,
                    entry_price=45000.0,
                    stop_loss=44100.0,
                    take_profit=48000.0,
                    reasoning="Bullish trend",
                    latency_ms=150,
                    tokens_used=500,
                    cost_usd=0.02
                ),
            ],
            entry_price=45000.0,
            stop_loss=44100.0,
            take_profit=48000.0,
            position_size_pct=10.0,
            total_cost_usd=0.12,
        )

    @pytest.mark.asyncio
    async def test_store_model_comparisons_no_db(
        self, trading_agent, trading_output_with_decisions, mock_snapshot
    ):
        """Store should handle no database gracefully."""
        trading_agent.db = None

        # Should not raise
        await trading_agent._store_model_comparisons(
            trading_output_with_decisions, mock_snapshot
        )

    @pytest.mark.asyncio
    async def test_store_model_comparisons_with_db(
        self, trading_agent, trading_output_with_decisions, mock_snapshot
    ):
        """Store should insert records to database."""
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=None)
        trading_agent.db = mock_db

        await trading_agent._store_model_comparisons(
            trading_output_with_decisions, mock_snapshot
        )

        # Should have called execute once per model decision
        assert mock_db.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_store_model_comparisons_db_error(
        self, trading_agent, trading_output_with_decisions, mock_snapshot
    ):
        """Store should handle database errors gracefully."""
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(side_effect=Exception("DB error"))
        trading_agent.db = mock_db

        # Should not raise
        await trading_agent._store_model_comparisons(
            trading_output_with_decisions, mock_snapshot
        )

    @pytest.mark.asyncio
    async def test_update_comparison_outcomes_no_db(self, trading_agent):
        """Update should return 0 when no database."""
        trading_agent.db = None

        result = await trading_agent.update_comparison_outcomes(
            symbol="BTC/USDT",
            timestamp_from=datetime.now(timezone.utc) - timedelta(hours=4),
            timestamp_to=datetime.now(timezone.utc),
            price_1h=45100.0,
            price_4h=45500.0,
            price_24h=46000.0,
        )

        assert result == 0

    @pytest.mark.asyncio
    async def test_update_comparison_outcomes_no_rows(self, trading_agent):
        """Update should return 0 when no matching rows."""
        mock_db = MagicMock()
        mock_db.fetch = AsyncMock(return_value=[])
        trading_agent.db = mock_db

        result = await trading_agent.update_comparison_outcomes(
            symbol="BTC/USDT",
            timestamp_from=datetime.now(timezone.utc) - timedelta(hours=4),
            timestamp_to=datetime.now(timezone.utc),
            price_1h=45100.0,
            price_4h=45500.0,
            price_24h=46000.0,
        )

        assert result == 0

    @pytest.mark.asyncio
    async def test_update_comparison_outcomes_buy_correct(self, trading_agent):
        """Update should calculate BUY as correct when price went up."""
        mock_db = MagicMock()
        mock_db.fetch = AsyncMock(return_value=[
            {'id': 1, 'action': 'BUY', 'price_at_decision': 45000.0}
        ])
        mock_db.execute = AsyncMock(return_value=None)
        trading_agent.db = mock_db

        result = await trading_agent.update_comparison_outcomes(
            symbol="BTC/USDT",
            timestamp_from=datetime.now(timezone.utc) - timedelta(hours=4),
            timestamp_to=datetime.now(timezone.utc),
            price_1h=45100.0,
            price_4h=46000.0,  # Price went up (correct for BUY)
            price_24h=47000.0,
        )

        assert result == 1
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_comparison_outcomes_sell_correct(self, trading_agent):
        """Update should calculate SELL as correct when price went down."""
        mock_db = MagicMock()
        mock_db.fetch = AsyncMock(return_value=[
            {'id': 1, 'action': 'SELL', 'price_at_decision': 45000.0}
        ])
        mock_db.execute = AsyncMock(return_value=None)
        trading_agent.db = mock_db

        result = await trading_agent.update_comparison_outcomes(
            symbol="BTC/USDT",
            timestamp_from=datetime.now(timezone.utc) - timedelta(hours=4),
            timestamp_to=datetime.now(timezone.utc),
            price_1h=44800.0,
            price_4h=44000.0,  # Price went down (correct for SELL)
            price_24h=43000.0,
        )

        assert result == 1

    @pytest.mark.asyncio
    async def test_update_comparison_outcomes_hold_correct(self, trading_agent):
        """Update should calculate HOLD as correct when price stayed flat."""
        mock_db = MagicMock()
        mock_db.fetch = AsyncMock(return_value=[
            {'id': 1, 'action': 'HOLD', 'price_at_decision': 45000.0}
        ])
        mock_db.execute = AsyncMock(return_value=None)
        trading_agent.db = mock_db

        result = await trading_agent.update_comparison_outcomes(
            symbol="BTC/USDT",
            timestamp_from=datetime.now(timezone.utc) - timedelta(hours=4),
            timestamp_to=datetime.now(timezone.utc),
            price_1h=45050.0,
            price_4h=45100.0,  # Price within 2% (correct for HOLD)
            price_24h=45200.0,
        )

        assert result == 1

    @pytest.mark.asyncio
    async def test_update_comparison_outcomes_db_error(self, trading_agent):
        """Update should handle database errors gracefully."""
        mock_db = MagicMock()
        mock_db.fetch = AsyncMock(side_effect=Exception("DB error"))
        trading_agent.db = mock_db

        result = await trading_agent.update_comparison_outcomes(
            symbol="BTC/USDT",
            timestamp_from=datetime.now(timezone.utc) - timedelta(hours=4),
            timestamp_to=datetime.now(timezone.utc),
            price_1h=45100.0,
            price_4h=46000.0,
            price_24h=47000.0,
        )

        assert result == 0

    @pytest.mark.asyncio
    async def test_update_multiple_records(self, trading_agent):
        """Update should process multiple records."""
        mock_db = MagicMock()
        mock_db.fetch = AsyncMock(return_value=[
            {'id': 1, 'action': 'BUY', 'price_at_decision': 45000.0},
            {'id': 2, 'action': 'SELL', 'price_at_decision': 45000.0},
            {'id': 3, 'action': 'HOLD', 'price_at_decision': 45000.0},
        ])
        mock_db.execute = AsyncMock(return_value=None)
        trading_agent.db = mock_db

        result = await trading_agent.update_comparison_outcomes(
            symbol="BTC/USDT",
            timestamp_from=datetime.now(timezone.utc) - timedelta(hours=4),
            timestamp_to=datetime.now(timezone.utc),
            price_1h=45100.0,
            price_4h=46000.0,
            price_24h=47000.0,
        )

        assert result == 3
        assert mock_db.execute.call_count == 3
