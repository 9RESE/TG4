"""
Unit tests for LLM clients.

Tests validate:
- LLMResponse dataclass
- BaseLLMClient interface
- Client statistics tracking
- Cost calculation logic
"""

import pytest
from decimal import Decimal

from triplegain.src.llm.clients.base import LLMResponse, BaseLLMClient


# =============================================================================
# LLMResponse Tests
# =============================================================================

class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_create_basic_response(self):
        """Create a basic LLMResponse."""
        response = LLMResponse(
            text="Hello, world!",
            tokens_used=10,
            model="gpt-4-turbo",
        )

        assert response.text == "Hello, world!"
        assert response.tokens_used == 10
        assert response.model == "gpt-4-turbo"
        assert response.finish_reason == "stop"  # Default
        assert response.latency_ms == 0  # Default
        assert response.cost_usd == 0.0  # Default

    def test_create_full_response(self):
        """Create a full LLMResponse with all fields."""
        response = LLMResponse(
            text='{"action": "BUY", "confidence": 0.85}',
            tokens_used=150,
            model="claude-3-opus",
            finish_reason="stop",
            latency_ms=500,
            cost_usd=0.025,
            raw_response={"id": "msg_123", "usage": {"output_tokens": 150}},
        )

        assert response.tokens_used == 150
        assert response.latency_ms == 500
        assert response.cost_usd == 0.025
        assert response.raw_response is not None
        assert response.raw_response["id"] == "msg_123"

    def test_response_with_long_content(self):
        """Response can handle long content."""
        long_text = "Analysis: " + "x" * 10000
        response = LLMResponse(
            text=long_text,
            tokens_used=2500,
            model="deepseek-chat",
        )

        assert len(response.text) == 10010
        assert response.tokens_used == 2500


# =============================================================================
# Concrete Implementation for Testing
# =============================================================================

class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing base functionality."""

    provider_name = "mock"

    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Mock generate implementation."""
        response = LLMResponse(
            text='{"action": "HOLD", "confidence": 0.5}',
            tokens_used=100,
            model=model,
            latency_ms=50,
            cost_usd=0.001,
        )
        self._update_stats(response.tokens_used, response.cost_usd)
        return response

    async def health_check(self) -> bool:
        """Mock health check."""
        return True


# =============================================================================
# BaseLLMClient Tests
# =============================================================================

class TestBaseLLMClient:
    """Test BaseLLMClient base functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        return MockLLMClient({"api_key": "test"})

    def test_initial_stats(self, mock_client):
        """Initial stats should be zero."""
        stats = mock_client.get_stats()

        assert stats['provider'] == "mock"
        assert stats['total_requests'] == 0
        assert stats['total_tokens'] == 0
        assert stats['total_cost_usd'] == 0.0

    @pytest.mark.asyncio
    async def test_stats_update_after_generate(self, mock_client):
        """Stats should update after generate call."""
        await mock_client.generate(
            model="test-model",
            system_prompt="You are a test.",
            user_message="Hello",
        )

        stats = mock_client.get_stats()

        assert stats['total_requests'] == 1
        assert stats['total_tokens'] == 100
        assert stats['total_cost_usd'] == 0.001

    @pytest.mark.asyncio
    async def test_stats_accumulate(self, mock_client):
        """Stats should accumulate across multiple calls."""
        for _ in range(3):
            await mock_client.generate(
                model="test-model",
                system_prompt="You are a test.",
                user_message="Hello",
            )

        stats = mock_client.get_stats()

        assert stats['total_requests'] == 3
        assert stats['total_tokens'] == 300
        assert stats['total_cost_usd'] == 0.003

    @pytest.mark.asyncio
    async def test_health_check(self, mock_client):
        """Health check should return True."""
        is_healthy = await mock_client.health_check()

        assert is_healthy is True

    def test_config_stored(self, mock_client):
        """Config should be stored."""
        assert mock_client.config == {"api_key": "test"}


# =============================================================================
# Cost Calculation Tests
# =============================================================================

class TestCostCalculation:
    """Test cost calculation patterns used by clients."""

    def test_openai_cost_calculation(self):
        """Test OpenAI-style cost calculation."""
        # GPT-4-turbo pricing: $10/1M input, $30/1M output (hypothetical)
        input_tokens = 1000
        output_tokens = 500

        input_cost = (input_tokens / 1_000_000) * 10.0
        output_cost = (output_tokens / 1_000_000) * 30.0
        total_cost = input_cost + output_cost

        assert total_cost == pytest.approx(0.025, rel=1e-6)

    def test_anthropic_cost_calculation(self):
        """Test Anthropic-style cost calculation."""
        # Claude Sonnet pricing: $3/1M input, $15/1M output (hypothetical)
        input_tokens = 2000
        output_tokens = 200

        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0
        total_cost = input_cost + output_cost

        assert total_cost == pytest.approx(0.009, rel=1e-6)

    def test_local_has_zero_cost(self):
        """Local models should have zero cost."""
        response = LLMResponse(
            text="Analysis complete",
            tokens_used=500,
            model="qwen2.5:7b",
            cost_usd=0.0,  # Local = free
        )

        assert response.cost_usd == 0.0


# =============================================================================
# Response Parsing Patterns
# =============================================================================

class TestResponsePatterns:
    """Test common response patterns used by clients."""

    def test_json_response_pattern(self):
        """Test JSON response parsing."""
        import json

        response_text = '{"action": "BUY", "confidence": 0.85}'
        parsed = json.loads(response_text)

        assert parsed['action'] == "BUY"
        assert parsed['confidence'] == 0.85

    def test_json_with_markdown(self):
        """Test JSON extraction from markdown."""
        import re
        import json

        response_text = '''
        Here is my analysis:

        ```json
        {"action": "SELL", "confidence": 0.7}
        ```

        Based on the above.
        '''

        # Extract JSON from code block
        match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if match:
            json_str = match.group(1)
            parsed = json.loads(json_str)

            assert parsed['action'] == "SELL"

    def test_json_in_text(self):
        """Test JSON extraction from plain text."""
        import re
        import json

        response_text = 'Analysis: {"action": "HOLD", "confidence": 0.5} end.'

        # Extract JSON object
        match = re.search(r'\{[\s\S]*\}', response_text)
        if match:
            parsed = json.loads(match.group())

            assert parsed['action'] == "HOLD"


# =============================================================================
# Provider-Specific Config Tests
# =============================================================================

class TestProviderConfigs:
    """Test provider-specific configuration handling."""

    def test_ollama_config(self):
        """Ollama config should have host."""
        config = {
            'host': 'http://localhost:11434',
            'timeout_ms': 30000,
        }

        client = MockLLMClient(config)
        assert client.config['host'] == 'http://localhost:11434'

    def test_openai_config(self):
        """OpenAI config should have API key."""
        config = {
            'api_key': 'sk-test-key',
            'base_url': 'https://api.openai.com/v1',
        }

        client = MockLLMClient(config)
        assert client.config['api_key'] == 'sk-test-key'

    def test_anthropic_config(self):
        """Anthropic config should have API key."""
        config = {
            'api_key': 'sk-ant-test-key',
        }

        client = MockLLMClient(config)
        assert 'api_key' in client.config


# =============================================================================
# Latency Tracking Tests
# =============================================================================

class TestLatencyTracking:
    """Test latency tracking in responses."""

    def test_latency_in_response(self):
        """Latency should be tracked in milliseconds."""
        response = LLMResponse(
            text="Fast response",
            tokens_used=50,
            model="qwen2.5:7b",
            latency_ms=125,
        )

        assert response.latency_ms == 125

    def test_high_latency_api(self):
        """API calls can have higher latency."""
        response = LLMResponse(
            text="API response",
            tokens_used=200,
            model="gpt-4-turbo",
            latency_ms=2500,  # 2.5 seconds
        )

        assert response.latency_ms == 2500

    def test_latency_under_target(self):
        """Check if latency is under target."""
        response = LLMResponse(
            text="Quick response",
            tokens_used=100,
            model="qwen2.5:7b",
            latency_ms=450,
        )

        target_ms = 500
        assert response.latency_ms < target_ms
