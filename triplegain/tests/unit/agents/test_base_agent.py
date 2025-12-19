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


# =============================================================================
# BaseAgent Async Method Tests
# =============================================================================

class TestBaseAgentAsyncMethods:
    """Test BaseAgent async methods."""

    @pytest.fixture
    def test_agent_class(self):
        """Create a concrete test agent class."""
        from triplegain.src.agents.base_agent import BaseAgent

        class TestAgent(BaseAgent):
            agent_name = "test_agent"
            model = "test-model"

            async def process(self, snapshot, portfolio_context=None, **kwargs):
                pass

            def get_output_schema(self):
                return {}

        return TestAgent

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        from unittest.mock import AsyncMock, MagicMock
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_db_pool(self):
        """Create mock database pool."""
        from unittest.mock import AsyncMock, MagicMock
        mock = MagicMock()
        mock.execute = AsyncMock()
        mock.fetchrow = AsyncMock(return_value=None)
        return mock

    @pytest.mark.asyncio
    async def test_cache_output(self, test_agent_class, mock_llm_client):
        """Test caching an output."""
        from unittest.mock import MagicMock
        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning for cache",
        )

        await agent.cache_output(output)

        assert "BTC/USDT" in agent._cache
        assert agent._cache["BTC/USDT"][0] == output

    @pytest.mark.asyncio
    async def test_clear_cache_specific_symbol(self, test_agent_class, mock_llm_client):
        """Test clearing cache for specific symbol."""
        from unittest.mock import MagicMock
        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        output1 = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning for BTC",
        )
        output2 = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="ETH/USDT",
            confidence=0.65,
            reasoning="Test reasoning for ETH",
        )

        await agent.cache_output(output1)
        await agent.cache_output(output2)

        await agent.clear_cache("BTC/USDT")

        assert "BTC/USDT" not in agent._cache
        assert "ETH/USDT" in agent._cache

    @pytest.mark.asyncio
    async def test_clear_cache_all(self, test_agent_class, mock_llm_client):
        """Test clearing all cache."""
        from unittest.mock import MagicMock
        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
        )

        await agent.cache_output(output)
        await agent.clear_cache()

        assert agent._cache == {}

    @pytest.mark.asyncio
    async def test_get_latest_output_from_cache(self, test_agent_class, mock_llm_client):
        """Test getting output from cache."""
        from unittest.mock import MagicMock
        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={'cache_ttl_seconds': 300},
            db_pool=None,
        )

        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
        )

        await agent.cache_output(output)
        result = await agent.get_latest_output("BTC/USDT", max_age_seconds=300)

        assert result == output

    @pytest.mark.asyncio
    async def test_get_latest_output_expired(self, test_agent_class, mock_llm_client):
        """Test expired cache entry is removed."""
        from unittest.mock import MagicMock
        from datetime import timedelta

        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={'cache_ttl_seconds': 300},
            db_pool=None,
        )

        # Manually insert expired entry
        old_output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
        )
        old_time = datetime.now(timezone.utc) - timedelta(seconds=600)
        agent._cache["BTC/USDT"] = (old_output, old_time)

        result = await agent.get_latest_output("BTC/USDT", max_age_seconds=300)

        assert result is None
        assert "BTC/USDT" not in agent._cache

    @pytest.mark.asyncio
    async def test_get_latest_output_not_found(self, test_agent_class, mock_llm_client):
        """Test getting output when none exists."""
        from unittest.mock import MagicMock
        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        result = await agent.get_latest_output("BTC/USDT", max_age_seconds=300)

        assert result is None

    @pytest.mark.asyncio
    async def test_store_output_without_db(self, test_agent_class, mock_llm_client):
        """Test store_output without database still updates cache."""
        from unittest.mock import MagicMock
        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
        )

        await agent.store_output(output)

        assert "BTC/USDT" in agent._cache

    @pytest.mark.asyncio
    async def test_store_output_with_db(self, test_agent_class, mock_llm_client, mock_db_pool):
        """Test store_output with database."""
        from unittest.mock import MagicMock
        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=mock_db_pool,
        )

        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
        )

        await agent.store_output(output)

        mock_db_pool.execute.assert_called_once()
        assert "BTC/USDT" in agent._cache

    @pytest.mark.asyncio
    async def test_store_output_db_error(self, test_agent_class, mock_llm_client):
        """Test store_output handles database errors gracefully."""
        from unittest.mock import MagicMock, AsyncMock

        mock_db = MagicMock()
        mock_db.execute = AsyncMock(side_effect=Exception("DB error"))

        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=mock_db,
        )

        output = AgentOutput(
            agent_name="test_agent",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
        )

        # Should not raise exception
        await agent.store_output(output)

        # Cache should still be updated
        assert "BTC/USDT" in agent._cache

    @pytest.mark.asyncio
    async def test_call_llm_success(self, test_agent_class, mock_llm_client):
        """Test _call_llm success."""
        from unittest.mock import MagicMock

        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = '{"action": "BUY"}'
        mock_response.tokens_used = 100
        mock_llm_client.generate.return_value = mock_response

        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        text, latency_ms, tokens = await agent._call_llm("system prompt", "user message")

        assert text == '{"action": "BUY"}'
        assert tokens == 100
        assert latency_ms >= 0
        assert agent._total_invocations == 1
        assert agent._total_tokens == 100

    @pytest.mark.asyncio
    async def test_call_llm_error(self, test_agent_class, mock_llm_client):
        """Test _call_llm error handling."""
        from unittest.mock import MagicMock

        mock_llm_client.generate.side_effect = Exception("LLM error")

        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        with pytest.raises(Exception, match="LLM error"):
            await agent._call_llm("system prompt", "user message")

    @pytest.mark.asyncio
    async def test_get_latest_output_from_db(self, test_agent_class, mock_llm_client):
        """Test getting output from database when not in cache."""
        from unittest.mock import MagicMock, AsyncMock
        import json

        output_data = {
            "agent_name": "test_agent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "BTC/USDT",
            "confidence": 0.75,
            "reasoning": "DB stored reasoning",
            "latency_ms": 100,
            "tokens_used": 500,
            "model_used": "test-model",
            "output_id": "test-id-123"
        }

        mock_db = MagicMock()
        mock_db.fetchrow = AsyncMock(return_value={
            'output_data': json.dumps(output_data),
            'timestamp': datetime.now(timezone.utc)
        })

        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=mock_db,
        )

        result = await agent.get_latest_output("BTC/USDT", max_age_seconds=300)

        # Should have queried database
        mock_db.fetchrow.assert_called_once()
        # Should have cached the result
        assert "BTC/USDT" in agent._cache

    @pytest.mark.asyncio
    async def test_get_latest_output_db_error(self, test_agent_class, mock_llm_client):
        """Test get_latest_output handles database errors."""
        from unittest.mock import MagicMock, AsyncMock

        mock_db = MagicMock()
        mock_db.fetchrow = AsyncMock(side_effect=Exception("DB error"))

        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=mock_db,
        )

        # Should not raise, just return None
        result = await agent.get_latest_output("BTC/USDT", max_age_seconds=300)
        assert result is None

    def test_parse_stored_output_success(self, test_agent_class, mock_llm_client):
        """Test parsing stored JSON output."""
        from unittest.mock import MagicMock
        import json

        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        output_data = {
            "agent_name": "test_agent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "BTC/USDT",
            "confidence": 0.75,
            "reasoning": "Parsed reasoning",
            "latency_ms": 100,
            "tokens_used": 500,
            "model_used": "test-model",
            "output_id": "test-id"
        }

        result = agent._parse_stored_output(json.dumps(output_data))

        assert result is not None
        assert result.agent_name == "test_agent"
        assert result.confidence == 0.75

    def test_parse_stored_output_invalid_json(self, test_agent_class, mock_llm_client):
        """Test parsing invalid JSON returns None."""
        from unittest.mock import MagicMock

        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        result = agent._parse_stored_output("not valid json")

        assert result is None

    def test_stats_accumulate_after_llm_calls(self, test_agent_class, mock_llm_client):
        """Test that stats accumulate correctly."""
        from unittest.mock import MagicMock

        agent = test_agent_class(
            llm_client=mock_llm_client,
            prompt_builder=MagicMock(),
            config={},
            db_pool=None,
        )

        # Simulate stats accumulation
        agent._total_invocations = 3
        agent._total_latency_ms = 450
        agent._total_tokens = 300

        stats = agent.get_stats()

        assert stats['total_invocations'] == 3
        assert stats['average_latency_ms'] == 150
        assert stats['average_tokens'] == 100


# =============================================================================
# AgentOutput Decimal Handling Tests
# =============================================================================

class TestAgentOutputDecimalHandling:
    """Test AgentOutput with Decimal values."""

    def test_to_dict_with_decimal(self):
        """to_dict should convert Decimal to float."""
        from decimal import Decimal

        # Create a subclass that has a Decimal field
        from dataclasses import dataclass

        @dataclass
        class CustomOutput(AgentOutput):
            price: Decimal = Decimal('0')

        output = CustomOutput(
            agent_name="test",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            confidence=0.75,
            reasoning="Test reasoning here",
            price=Decimal('45000.50'),
        )

        result = output.to_dict()

        assert result['price'] == 45000.50
        assert isinstance(result['price'], float)
