"""
Unit tests for the Base Agent class.

Tests validate:
- AgentOutput dataclass
- Output validation
- Output serialization
"""

import pytest
from datetime import datetime, timezone

from triplegain.src.agents.base_agent import AgentOutput


# =============================================================================
# AgentOutput Tests
# =============================================================================

class TestAgentOutput:
    """Test AgentOutput dataclass."""

    def test_create_valid_output(self):
        """Create a valid AgentOutput."""
        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test analysis reasoning",
            latency_ms=100,
            tokens_used=500,
            model_used="test-model",
        )

        assert output.agent_name == "test_agent"
        assert output.confidence == 0.75
        assert output.output_id is not None

    def test_validation_valid_output(self):
        """Valid output should pass validation."""
        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test analysis with sufficient reasoning text",
        )

        is_valid, errors = output.validate()

        assert is_valid is True
        assert len(errors) == 0

    def test_validation_invalid_confidence_above_1(self):
        """Confidence above 1 should fail validation."""
        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=1.5,  # Invalid
            reasoning="Test analysis reasoning",
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("not in [0, 1]" in e for e in errors)

    def test_validation_invalid_confidence_negative(self):
        """Negative confidence should fail validation."""
        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=-0.5,  # Invalid
            reasoning="Test analysis reasoning",
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("not in [0, 1]" in e for e in errors)

    def test_validation_short_reasoning(self):
        """Reasoning too short should fail validation."""
        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="short",  # Too short
        )

        is_valid, errors = output.validate()

        assert is_valid is False
        assert any("too short" in e for e in errors)

    def test_to_dict(self):
        """to_dict should serialize properly."""
        timestamp = datetime.now(timezone.utc)
        output = AgentOutput(
            agent_name="test_agent",
            timestamp=timestamp,
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test analysis reasoning",
            latency_ms=100,
            tokens_used=500,
            model_used="test-model",
        )

        result = output.to_dict()

        assert result['agent_name'] == "test_agent"
        assert result['timestamp'] == timestamp.isoformat()
        assert result['confidence'] == 0.75
        assert result['latency_ms'] == 100
        assert 'output_id' in result

    def test_to_json(self):
        """to_json should produce valid JSON."""
        import json

        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test analysis reasoning",
        )

        json_str = output.to_json()
        parsed = json.loads(json_str)

        assert parsed['agent_name'] == "test_agent"
        assert parsed['confidence'] == 0.75


# =============================================================================
# BaseAgent Tests
# =============================================================================

class TestBaseAgentAttributes:
    """Test BaseAgent initialization and attributes."""

    def test_last_output_initially_none(self):
        """last_output should be None before process() is called."""
        from unittest.mock import MagicMock
        from triplegain.src.agents.base_agent import BaseAgent

        # Create a concrete subclass for testing
        class TestAgent(BaseAgent):
            agent_name = "test"

            async def process(self, snapshot, portfolio_context=None, **kwargs):
                pass

            def get_output_schema(self):
                return {}

        agent = TestAgent(
            llm_client=MagicMock(),
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        assert agent.last_output is None
        assert agent._last_output is None

    def test_stats_initialized_to_zero(self):
        """Performance stats should start at zero."""
        from unittest.mock import MagicMock
        from triplegain.src.agents.base_agent import BaseAgent

        class TestAgent(BaseAgent):
            agent_name = "test"

            async def process(self, snapshot, portfolio_context=None, **kwargs):
                pass

            def get_output_schema(self):
                return {}

        agent = TestAgent(
            llm_client=MagicMock(),
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        stats = agent.get_stats()

        assert stats['total_invocations'] == 0
        assert stats['total_latency_ms'] == 0
        assert stats['total_tokens'] == 0
        assert stats['average_latency_ms'] == 0

    def test_cache_initialized_empty(self):
        """Cache should be empty on init."""
        from unittest.mock import MagicMock
        from triplegain.src.agents.base_agent import BaseAgent

        class TestAgent(BaseAgent):
            agent_name = "test"

            async def process(self, snapshot, portfolio_context=None, **kwargs):
                pass

            def get_output_schema(self):
                return {}

        agent = TestAgent(
            llm_client=MagicMock(),
            prompt_builder=MagicMock(),
            config={'cache_ttl_seconds': 600},
            db_pool=None,
        )

        assert agent._cache == {}
        assert agent._cache_ttl_seconds == 600
