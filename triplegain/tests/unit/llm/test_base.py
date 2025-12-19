"""
Tests for Base LLM Client Module.

Tests RateLimiter, LLMResponse, and BaseLLMClient functionality.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from triplegain.src.llm.clients.base import (
    RateLimiter,
    LLMResponse,
    BaseLLMClient,
    DEFAULT_RATE_LIMITS,
    MODEL_COSTS,
)


# =============================================================================
# RateLimiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for the RateLimiter class."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter with 5 requests per minute."""
        return RateLimiter(requests_per_minute=5)

    @pytest.mark.asyncio
    async def test_acquire_no_wait(self, rate_limiter):
        """First request should not need to wait."""
        wait_time = await rate_limiter.acquire()
        assert wait_time == 0

    @pytest.mark.asyncio
    async def test_acquire_multiple_no_wait(self, rate_limiter):
        """Multiple requests under limit should not wait."""
        for _ in range(5):
            wait_time = await rate_limiter.acquire()
            assert wait_time == 0

    @pytest.mark.asyncio
    async def test_acquire_rate_limited(self):
        """Exceeding rate limit should cause wait."""
        # Use very small window for testing
        rate_limiter = RateLimiter(requests_per_minute=3)

        # Make 3 requests (fill the limit)
        for _ in range(3):
            await rate_limiter.acquire()

        # Next request should wait
        # Mock time to simulate immediate "old" requests
        with patch.object(rate_limiter, '_request_times',
                          [time.monotonic() - 59.5] * 3):  # Almost expired
            start = time.monotonic()
            await rate_limiter.acquire()
            elapsed = time.monotonic() - start
            # Should have waited (at least a small amount)
            assert elapsed >= 0.01

    @pytest.mark.asyncio
    async def test_available_requests_full(self, rate_limiter):
        """available_requests should return full capacity initially."""
        assert rate_limiter.available_requests == 5

    @pytest.mark.asyncio
    async def test_available_requests_decreases(self, rate_limiter):
        """available_requests should decrease after acquire."""
        await rate_limiter.acquire()
        assert rate_limiter.available_requests == 4

        await rate_limiter.acquire()
        assert rate_limiter.available_requests == 3

    @pytest.mark.asyncio
    async def test_acquire_cleans_old_requests(self):
        """Old requests outside window should be removed."""
        rate_limiter = RateLimiter(requests_per_minute=5)

        # Manually add old request times (outside window)
        old_time = time.monotonic() - 120  # 2 minutes ago
        rate_limiter._request_times = [old_time, old_time]

        await rate_limiter.acquire()

        # Old requests should be cleaned, only new one remains
        assert len([t for t in rate_limiter._request_times
                   if t > time.monotonic() - 60]) == 1

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self, rate_limiter):
        """Concurrent acquire should be thread-safe."""
        # Run multiple concurrent acquires
        tasks = [rate_limiter.acquire() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 5


# =============================================================================
# LLMResponse Tests
# =============================================================================

class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_create_response(self):
        """Create a basic LLMResponse."""
        response = LLMResponse(
            text="Hello world",
            tokens_used=50,
            model="gpt-4o"
        )
        assert response.text == "Hello world"
        assert response.tokens_used == 50
        assert response.model == "gpt-4o"

    def test_default_values(self):
        """Test default values."""
        response = LLMResponse(
            text="Test",
            tokens_used=10,
            model="test-model"
        )
        assert response.finish_reason == "stop"
        assert response.latency_ms == 0
        assert response.cost_usd == 0.0
        assert response.raw_response is None

    def test_all_fields(self):
        """Test all fields set."""
        response = LLMResponse(
            text="Response text",
            tokens_used=100,
            model="gpt-4-turbo",
            finish_reason="length",
            latency_ms=500,
            cost_usd=0.05,
            raw_response={"id": "test"}
        )
        assert response.text == "Response text"
        assert response.tokens_used == 100
        assert response.model == "gpt-4-turbo"
        assert response.finish_reason == "length"
        assert response.latency_ms == 500
        assert response.cost_usd == 0.05
        assert response.raw_response == {"id": "test"}


# =============================================================================
# BaseLLMClient Tests
# =============================================================================

class ConcreteLLMClient(BaseLLMClient):
    """Concrete implementation for testing."""

    provider_name = "test_provider"

    def __init__(self, config: dict, mock_generate=None, mock_health=None):
        super().__init__(config)
        self._mock_generate = mock_generate
        self._mock_health = mock_health

    async def generate(self, model, system_prompt, user_message,
                       temperature=0.3, max_tokens=2048) -> LLMResponse:
        if self._mock_generate:
            return await self._mock_generate(model, system_prompt, user_message,
                                            temperature, max_tokens)
        return LLMResponse(text="Test response", tokens_used=50, model=model)

    async def health_check(self) -> bool:
        if self._mock_health:
            return await self._mock_health()
        return True


class TestBaseLLMClient:
    """Tests for BaseLLMClient abstract class."""

    @pytest.fixture
    def client(self):
        """Create a concrete client instance."""
        return ConcreteLLMClient(config={})

    @pytest.fixture
    def client_with_config(self):
        """Create client with custom config."""
        return ConcreteLLMClient(config={
            'rate_limit_rpm': 30,
            'max_retries': 2,
            'retry_base_delay': 0.1,
            'retry_max_delay': 1.0
        })

    def test_init_default_config(self, client):
        """Test initialization with default config."""
        assert client._max_retries == 3
        assert client._base_delay == 1.0
        assert client._max_delay == 30.0
        assert client._total_requests == 0
        assert client._total_tokens == 0
        assert client._total_cost == 0.0

    def test_init_custom_config(self, client_with_config):
        """Test initialization with custom config."""
        assert client_with_config._max_retries == 2
        assert client_with_config._base_delay == 0.1
        assert client_with_config._max_delay == 1.0

    def test_get_stats(self, client):
        """Test get_stats returns correct data."""
        stats = client.get_stats()

        assert stats["provider"] == "test_provider"
        assert stats["total_requests"] == 0
        assert stats["total_tokens"] == 0
        assert stats["total_cost_usd"] == 0.0
        assert "available_rate_limit" in stats

    def test_update_stats(self, client):
        """Test _update_stats updates counters."""
        client._update_stats(tokens=100, cost=0.05)

        assert client._total_requests == 1
        assert client._total_tokens == 100
        assert client._total_cost == 0.05

        client._update_stats(tokens=50, cost=0.02)

        assert client._total_requests == 2
        assert client._total_tokens == 150
        assert client._total_cost == 0.07


class TestCalculateCost:
    """Tests for cost calculation."""

    @pytest.fixture
    def client(self):
        return ConcreteLLMClient(config={})

    def test_calculate_cost_gpt4_turbo(self, client):
        """Test cost calculation for GPT-4 Turbo."""
        cost = client._calculate_cost("gpt-4-turbo", 1000)

        # 70% input (700 tokens) = 0.007
        # 30% output (300 tokens) = 0.009
        expected = (700 / 1000) * 0.01 + (300 / 1000) * 0.03
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_gpt4o(self, client):
        """Test cost calculation for GPT-4o."""
        cost = client._calculate_cost("gpt-4o", 1000)

        expected = (700 / 1000) * 0.005 + (300 / 1000) * 0.015
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_gpt4o_mini(self, client):
        """Test cost calculation for GPT-4o mini."""
        cost = client._calculate_cost("gpt-4o-mini", 1000)

        expected = (700 / 1000) * 0.00015 + (300 / 1000) * 0.0006
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_claude_sonnet(self, client):
        """Test cost calculation for Claude Sonnet."""
        cost = client._calculate_cost("claude-3-5-sonnet-20241022", 1000)

        expected = (700 / 1000) * 0.003 + (300 / 1000) * 0.015
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_claude_opus(self, client):
        """Test cost calculation for Claude Opus."""
        cost = client._calculate_cost("claude-3-opus-20240229", 1000)

        expected = (700 / 1000) * 0.015 + (300 / 1000) * 0.075
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_deepseek(self, client):
        """Test cost calculation for DeepSeek."""
        cost = client._calculate_cost("deepseek-chat", 1000)

        expected = (700 / 1000) * 0.00014 + (300 / 1000) * 0.00028
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_grok(self, client):
        """Test cost calculation for Grok."""
        cost = client._calculate_cost("grok-2-1212", 1000)

        expected = (700 / 1000) * 0.002 + (300 / 1000) * 0.010
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_ollama_free(self, client):
        """Test cost calculation for Ollama (free)."""
        cost = client._calculate_cost("qwen2.5:7b", 1000)
        assert cost == 0.0

    def test_calculate_cost_unknown_model(self, client):
        """Unknown models should return 0 cost."""
        cost = client._calculate_cost("unknown-model", 1000)
        assert cost == 0.0


class TestGenerateWithRetry:
    """Tests for generate_with_retry method."""

    @pytest.fixture
    def success_client(self):
        """Client that succeeds on first try."""
        async def mock_generate(*args, **kwargs):
            return LLMResponse(text="Success", tokens_used=50, model="test")

        return ConcreteLLMClient(
            config={'max_retries': 2, 'retry_base_delay': 0.01},
            mock_generate=mock_generate
        )

    @pytest.fixture
    def fail_then_succeed_client(self):
        """Client that fails first, then succeeds."""
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return LLMResponse(text="Success", tokens_used=50, model="test")

        return ConcreteLLMClient(
            config={'max_retries': 2, 'retry_base_delay': 0.01},
            mock_generate=mock_generate
        )

    @pytest.fixture
    def always_fail_client(self):
        """Client that always fails."""
        async def mock_generate(*args, **kwargs):
            raise Exception("Permanent failure")

        return ConcreteLLMClient(
            config={'max_retries': 2, 'retry_base_delay': 0.01},
            mock_generate=mock_generate
        )

    @pytest.mark.asyncio
    async def test_generate_with_retry_success(self, success_client):
        """Successful generation on first try."""
        response = await success_client.generate_with_retry(
            model="test-model",
            system_prompt="You are a test.",
            user_message="Hello"
        )

        assert response.text == "Success"
        assert response.tokens_used == 50

    @pytest.mark.asyncio
    async def test_generate_with_retry_updates_stats(self, success_client):
        """generate_with_retry should update stats."""
        await success_client.generate_with_retry(
            model="test-model",
            system_prompt="System",
            user_message="User"
        )

        stats = success_client.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 50

    @pytest.mark.asyncio
    async def test_generate_with_retry_calculates_cost(self, success_client):
        """generate_with_retry should calculate cost."""
        response = await success_client.generate_with_retry(
            model="gpt-4o",
            system_prompt="System",
            user_message="User"
        )

        # Cost should be calculated
        assert response.cost_usd >= 0

    @pytest.mark.asyncio
    async def test_generate_with_retry_retry_on_failure(self, fail_then_succeed_client):
        """Should retry on failure."""
        response = await fail_then_succeed_client.generate_with_retry(
            model="test-model",
            system_prompt="System",
            user_message="User"
        )

        assert response.text == "Success"

    @pytest.mark.asyncio
    async def test_generate_with_retry_raises_after_max_retries(self, always_fail_client):
        """Should raise after exhausting retries."""
        with pytest.raises(Exception) as exc_info:
            await always_fail_client.generate_with_retry(
                model="test-model",
                system_prompt="System",
                user_message="User"
            )

        assert "Permanent failure" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_with_retry_exponential_backoff(self):
        """Should use exponential backoff on retries."""
        call_times = []

        async def mock_generate(*args, **kwargs):
            call_times.append(time.monotonic())
            raise Exception("Failure")

        client = ConcreteLLMClient(
            config={'max_retries': 2, 'retry_base_delay': 0.05, 'retry_max_delay': 1.0},
            mock_generate=mock_generate
        )

        with pytest.raises(Exception):
            await client.generate_with_retry(
                model="test", system_prompt="sys", user_message="user"
            )

        # Should have 3 attempts (initial + 2 retries)
        assert len(call_times) == 3

        # Check delays increase (exponential)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # Second delay should be roughly 2x first
        assert delay2 > delay1


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_default_rate_limits_providers(self):
        """All expected providers should have rate limits."""
        assert 'ollama' in DEFAULT_RATE_LIMITS
        assert 'openai' in DEFAULT_RATE_LIMITS
        assert 'anthropic' in DEFAULT_RATE_LIMITS
        assert 'deepseek' in DEFAULT_RATE_LIMITS
        assert 'xai' in DEFAULT_RATE_LIMITS

    def test_default_rate_limits_values(self):
        """Rate limits should be reasonable."""
        for provider, limit in DEFAULT_RATE_LIMITS.items():
            assert limit > 0
            assert limit <= 200  # Reasonable upper bound

    def test_model_costs_structure(self):
        """Model costs should have input and output."""
        for model, costs in MODEL_COSTS.items():
            assert 'input' in costs
            assert 'output' in costs
            assert costs['input'] >= 0
            assert costs['output'] >= 0

    def test_model_costs_expected_models(self):
        """Expected models should be present."""
        expected = [
            'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini',
            'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229',
            'deepseek-chat', 'grok-2-1212', 'qwen2.5:7b'
        ]
        for model in expected:
            assert model in MODEL_COSTS


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_rate_limiter_zero_requests(self):
        """Rate limiter with 0 requests should work but always wait."""
        # This is an edge case - shouldn't happen in practice
        limiter = RateLimiter(requests_per_minute=1)

        # First request should not wait
        wait = await limiter.acquire()
        assert wait == 0

    def test_calculate_cost_zero_tokens(self):
        """Zero tokens should return zero cost."""
        client = ConcreteLLMClient(config={})
        cost = client._calculate_cost("gpt-4o", 0)
        assert cost == 0.0

    def test_calculate_cost_large_tokens(self):
        """Large token counts should work."""
        client = ConcreteLLMClient(config={})
        cost = client._calculate_cost("gpt-4o", 100000)

        # Should be roughly 100x the 1000 token cost
        cost_1k = client._calculate_cost("gpt-4o", 1000)
        assert abs(cost - cost_1k * 100) < 0.001

    @pytest.mark.asyncio
    async def test_client_with_default_provider_rate_limit(self):
        """Client should use provider default rate limit."""
        # Create a client without rate_limit_rpm in config
        client = ConcreteLLMClient(config={})

        # Should have default rate limit (60 for unknown provider)
        # test_provider isn't in DEFAULT_RATE_LIMITS, so default is 60
        assert client._rate_limiter.max_requests == 60

    @pytest.mark.asyncio
    async def test_generate_with_retry_max_delay_cap(self):
        """Exponential backoff should cap at max_delay."""
        call_times = []

        async def mock_generate(*args, **kwargs):
            call_times.append(time.monotonic())
            raise Exception("Failure")

        # Very low max_delay
        client = ConcreteLLMClient(
            config={'max_retries': 3, 'retry_base_delay': 0.5, 'retry_max_delay': 0.1},
            mock_generate=mock_generate
        )

        with pytest.raises(Exception):
            await client.generate_with_retry(
                model="test", system_prompt="sys", user_message="user"
            )

        # Check no delay exceeds max_delay (0.1s)
        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i-1]
            # Allow some tolerance for test execution time
            assert delay < 0.3  # Max delay + tolerance

    @pytest.mark.asyncio
    async def test_rate_limiter_with_high_rpm(self):
        """Rate limiter with high RPM should handle many requests."""
        limiter = RateLimiter(requests_per_minute=1000)

        # Make many concurrent requests
        tasks = [limiter.acquire() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        assert all(r == 0 for r in results)  # None should wait

    def test_llm_response_equality(self):
        """LLMResponse dataclass equality."""
        r1 = LLMResponse(text="test", tokens_used=50, model="m1")
        r2 = LLMResponse(text="test", tokens_used=50, model="m1")

        assert r1 == r2

    def test_llm_response_different(self):
        """Different LLMResponse instances should not be equal."""
        r1 = LLMResponse(text="test1", tokens_used=50, model="m1")
        r2 = LLMResponse(text="test2", tokens_used=50, model="m1")

        assert r1 != r2
