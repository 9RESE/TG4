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
