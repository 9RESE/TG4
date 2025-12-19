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
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

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
        entry_price=45000.0,
        stop_loss=44100.0,
        take_profit=48000.0,
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
        """Majority should win."""
        decisions = [
            ModelDecision(model_name="m1", provider="test", action="BUY", confidence=0.8),
            ModelDecision(model_name="m2", provider="test", action="BUY", confidence=0.7),
            ModelDecision(model_name="m3", provider="test", action="BUY", confidence=0.75),
            ModelDecision(model_name="m4", provider="test", action="SELL", confidence=0.8),
            ModelDecision(model_name="m5", provider="test", action="HOLD", confidence=0.6),
            ModelDecision(model_name="m6", provider="test", action="HOLD", confidence=0.5),
        ]

        consensus = trading_agent._calculate_consensus(decisions)

        assert consensus.final_action == "BUY"
        assert consensus.agreeing_models == 3
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
                entry_price=45000, stop_loss=44000, take_profit=47000
            ),
            ModelDecision(
                model_name="m2", provider="test", action="BUY", confidence=0.7,
                entry_price=45100, stop_loss=44200, take_profit=47200
            ),
            ModelDecision(model_name="m3", provider="test", action="SELL", confidence=0.9),
        ]

        consensus = trading_agent._calculate_consensus(decisions)

        assert consensus.avg_entry_price == 45050  # (45000 + 45100) / 2
        assert consensus.avg_stop_loss == 44100  # (44000 + 44200) / 2

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
