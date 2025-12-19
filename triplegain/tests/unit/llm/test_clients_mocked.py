"""
Mocked unit tests for LLM clients.

Uses aioresponses to mock aiohttp HTTP requests.
Tests validate:
- Successful API responses
- Error handling (API errors, timeouts, connection errors)
- Health check endpoints
- Cost calculation
- Token tracking
"""

import asyncio
import pytest
from aioresponses import aioresponses
import aiohttp

from triplegain.src.llm.clients.ollama import OllamaClient
from triplegain.src.llm.clients.openai_client import OpenAIClient
from triplegain.src.llm.clients.anthropic_client import AnthropicClient
from triplegain.src.llm.clients.deepseek_client import DeepSeekClient
from triplegain.src.llm.clients.xai_client import XAIClient


# =============================================================================
# Ollama Client Tests
# =============================================================================

class TestOllamaClient:
    """Test OllamaClient with mocked HTTP responses."""

    @pytest.fixture
    def client(self):
        """Create Ollama client for testing."""
        return OllamaClient({
            'base_url': 'http://localhost:11434',
            'timeout_seconds': 30,
            'default_model': 'qwen2.5:7b',
        })

    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        """Test successful generation."""
        with aioresponses() as mocked:
            mocked.post(
                'http://localhost:11434/api/generate',
                payload={
                    'response': '{"action": "BUY", "confidence": 0.85}',
                    'done': True,
                    'done_reason': 'stop',
                    'eval_count': 50,
                    'prompt_eval_count': 100,
                }
            )

            response = await client.generate(
                model='qwen2.5:7b',
                system_prompt='You are a trading assistant.',
                user_message='Analyze BTC/USDT',
            )

            assert response.text == '{"action": "BUY", "confidence": 0.85}'
            assert response.tokens_used == 150
            assert response.model == 'qwen2.5:7b'
            assert response.finish_reason == 'stop'
            assert response.cost_usd == 0.0  # Local = free
            assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_uses_default_model(self, client):
        """Test that default model is used when none specified."""
        with aioresponses() as mocked:
            mocked.post(
                'http://localhost:11434/api/generate',
                payload={
                    'response': 'test response',
                    'done': True,
                    'eval_count': 10,
                    'prompt_eval_count': 20,
                }
            )

            response = await client.generate(
                model=None,  # Should use default
                system_prompt='System',
                user_message='User',
            )

            assert response.model == 'qwen2.5:7b'

    @pytest.mark.asyncio
    async def test_generate_api_error(self, client):
        """Test API error handling."""
        with aioresponses() as mocked:
            mocked.post(
                'http://localhost:11434/api/generate',
                status=500,
                body='Internal Server Error',
            )

            with pytest.raises(RuntimeError, match='Ollama API error'):
                await client.generate(
                    model='qwen2.5:7b',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_timeout(self, client):
        """Test timeout handling."""
        with aioresponses() as mocked:
            mocked.post(
                'http://localhost:11434/api/generate',
                exception=asyncio.TimeoutError(),
            )

            with pytest.raises(RuntimeError, match='timed out'):
                await client.generate(
                    model='qwen2.5:7b',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, client):
        """Test connection error handling."""
        with aioresponses() as mocked:
            mocked.post(
                'http://localhost:11434/api/generate',
                exception=aiohttp.ClientError('Connection refused'),
            )

            with pytest.raises(RuntimeError, match='connection failed'):
                await client.generate(
                    model='qwen2.5:7b',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Test health check when healthy."""
        with aioresponses() as mocked:
            mocked.get(
                'http://localhost:11434/api/tags',
                payload={'models': [{'name': 'qwen2.5:7b'}]},
            )

            is_healthy = await client.health_check()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, client):
        """Test health check when unhealthy."""
        with aioresponses() as mocked:
            mocked.get(
                'http://localhost:11434/api/tags',
                status=503,
            )

            is_healthy = await client.health_check()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, client):
        """Test health check with connection error."""
        with aioresponses() as mocked:
            mocked.get(
                'http://localhost:11434/api/tags',
                exception=aiohttp.ClientError('Connection refused'),
            )

            is_healthy = await client.health_check()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        """Test listing available models."""
        with aioresponses() as mocked:
            mocked.get(
                'http://localhost:11434/api/tags',
                payload={
                    'models': [
                        {'name': 'qwen2.5:7b', 'size': 4000000000},
                        {'name': 'llama3:8b', 'size': 5000000000},
                    ]
                },
            )

            models = await client.list_models()
            assert len(models) == 2
            assert models[0]['name'] == 'qwen2.5:7b'

    @pytest.mark.asyncio
    async def test_list_models_error(self, client):
        """Test list models with error."""
        with aioresponses() as mocked:
            mocked.get(
                'http://localhost:11434/api/tags',
                status=500,
            )

            models = await client.list_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_is_model_available(self, client):
        """Test checking if model is available."""
        with aioresponses() as mocked:
            mocked.get(
                'http://localhost:11434/api/tags',
                payload={'models': [{'name': 'qwen2.5:7b'}]},
            )

            is_available = await client.is_model_available('qwen2.5:7b')
            assert is_available is True

    @pytest.mark.asyncio
    async def test_is_model_not_available(self, client):
        """Test checking model that doesn't exist."""
        with aioresponses() as mocked:
            mocked.get(
                'http://localhost:11434/api/tags',
                payload={'models': [{'name': 'llama3:8b'}]},
            )

            is_available = await client.is_model_available('qwen2.5:7b')
            assert is_available is False

    @pytest.mark.asyncio
    async def test_generate_with_chat(self, client):
        """Test chat-style API."""
        with aioresponses() as mocked:
            mocked.post(
                'http://localhost:11434/api/chat',
                payload={
                    'message': {'content': 'Hello!'},
                    'done': True,
                    'done_reason': 'stop',
                    'eval_count': 5,
                    'prompt_eval_count': 10,
                }
            )

            response = await client.generate_with_chat(
                model='qwen2.5:7b',
                messages=[{'role': 'user', 'content': 'Hi'}],
            )

            assert response.text == 'Hello!'
            assert response.tokens_used == 15

    @pytest.mark.asyncio
    async def test_generate_with_chat_timeout(self, client):
        """Test chat API timeout."""
        with aioresponses() as mocked:
            mocked.post(
                'http://localhost:11434/api/chat',
                exception=asyncio.TimeoutError(),
            )

            with pytest.raises(RuntimeError, match='timed out'):
                await client.generate_with_chat(
                    model='qwen2.5:7b',
                    messages=[{'role': 'user', 'content': 'Hi'}],
                )

    @pytest.mark.asyncio
    async def test_generate_with_chat_connection_error(self, client):
        """Test chat API connection error."""
        with aioresponses() as mocked:
            mocked.post(
                'http://localhost:11434/api/chat',
                exception=aiohttp.ClientError('Connection refused'),
            )

            with pytest.raises(RuntimeError, match='connection failed'):
                await client.generate_with_chat(
                    model='qwen2.5:7b',
                    messages=[{'role': 'user', 'content': 'Hi'}],
                )

    @pytest.mark.asyncio
    async def test_stats_tracking(self, client):
        """Test that stats are tracked correctly."""
        with aioresponses() as mocked:
            mocked.post(
                'http://localhost:11434/api/generate',
                payload={
                    'response': 'response 1',
                    'done': True,
                    'eval_count': 50,
                    'prompt_eval_count': 100,
                }
            )
            mocked.post(
                'http://localhost:11434/api/generate',
                payload={
                    'response': 'response 2',
                    'done': True,
                    'eval_count': 30,
                    'prompt_eval_count': 70,
                }
            )

            await client.generate('qwen2.5:7b', 'sys', 'user1')
            await client.generate('qwen2.5:7b', 'sys', 'user2')

            stats = client.get_stats()
            assert stats['total_requests'] == 2
            assert stats['total_tokens'] == 250  # 150 + 100
            assert stats['total_cost_usd'] == 0.0  # Local = free


# =============================================================================
# OpenAI Client Tests
# =============================================================================

class TestOpenAIClient:
    """Test OpenAIClient with mocked HTTP responses."""

    @pytest.fixture
    def client(self):
        """Create OpenAI client for testing."""
        return OpenAIClient({
            'api_key': 'sk-test-key',
            'base_url': 'https://api.openai.com/v1',
            'default_model': 'gpt-4-turbo',
            'timeout_seconds': 60,
        })

    @pytest.fixture
    def client_no_key(self):
        """Create OpenAI client without API key."""
        return OpenAIClient({})

    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        """Test successful generation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.openai.com/v1/chat/completions',
                payload={
                    'choices': [{
                        'message': {'content': '{"action": "SELL", "confidence": 0.72}'},
                        'finish_reason': 'stop',
                    }],
                    'usage': {
                        'prompt_tokens': 100,
                        'completion_tokens': 50,
                    }
                }
            )

            response = await client.generate(
                model='gpt-4-turbo',
                system_prompt='You are a trading assistant.',
                user_message='Analyze BTC/USDT',
            )

            assert response.text == '{"action": "SELL", "confidence": 0.72}'
            assert response.tokens_used == 150
            assert response.model == 'gpt-4-turbo'
            assert response.finish_reason == 'stop'
            assert response.cost_usd > 0  # API has cost
            assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_cost_calculation(self, client):
        """Test correct cost calculation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.openai.com/v1/chat/completions',
                payload={
                    'choices': [{'message': {'content': 'test'}, 'finish_reason': 'stop'}],
                    'usage': {
                        'prompt_tokens': 1000,
                        'completion_tokens': 500,
                    }
                }
            )

            response = await client.generate(
                model='gpt-4-turbo',
                system_prompt='System',
                user_message='User',
            )

            # GPT-4-turbo: $10/1M input, $30/1M output
            expected_cost = (1000 / 1_000_000) * 10.0 + (500 / 1_000_000) * 30.0
            assert response.cost_usd == pytest.approx(expected_cost, rel=1e-6)

    @pytest.mark.asyncio
    async def test_generate_no_api_key(self, client_no_key):
        """Test error when no API key configured."""
        with pytest.raises(RuntimeError, match='API key not configured'):
            await client_no_key.generate(
                model='gpt-4-turbo',
                system_prompt='System',
                user_message='User',
            )

    @pytest.mark.asyncio
    async def test_generate_api_error(self, client):
        """Test API error handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.openai.com/v1/chat/completions',
                status=429,
                payload={'error': {'message': 'Rate limit exceeded'}},
            )

            with pytest.raises(RuntimeError, match='OpenAI:.*Rate limit'):
                await client.generate(
                    model='gpt-4-turbo',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_timeout(self, client):
        """Test timeout handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.openai.com/v1/chat/completions',
                exception=asyncio.TimeoutError(),
            )

            with pytest.raises(RuntimeError, match='timed out'):
                await client.generate(
                    model='gpt-4-turbo',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, client):
        """Test connection error handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.openai.com/v1/chat/completions',
                exception=aiohttp.ClientError('Connection refused'),
            )

            with pytest.raises(RuntimeError, match='connection error'):
                await client.generate(
                    model='gpt-4-turbo',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Test health check when healthy."""
        with aioresponses() as mocked:
            mocked.get(
                'https://api.openai.com/v1/models',
                payload={'data': [{'id': 'gpt-4-turbo'}]},
            )

            is_healthy = await client.health_check()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_no_api_key(self, client_no_key):
        """Test health check without API key."""
        is_healthy = await client_no_key.health_check()
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_api_error(self, client):
        """Test health check with API error."""
        with aioresponses() as mocked:
            mocked.get(
                'https://api.openai.com/v1/models',
                status=401,
            )

            is_healthy = await client.health_check()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_gpt4o_mini_pricing(self, client):
        """Test GPT-4o-mini cost calculation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.openai.com/v1/chat/completions',
                payload={
                    'choices': [{'message': {'content': 'test'}, 'finish_reason': 'stop'}],
                    'usage': {
                        'prompt_tokens': 1000,
                        'completion_tokens': 500,
                    }
                }
            )

            response = await client.generate(
                model='gpt-4o-mini',
                system_prompt='System',
                user_message='User',
            )

            # GPT-4o-mini: $0.15/1M input, $0.60/1M output
            expected_cost = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
            assert response.cost_usd == pytest.approx(expected_cost, rel=1e-6)


# =============================================================================
# Anthropic Client Tests
# =============================================================================

class TestAnthropicClient:
    """Test AnthropicClient with mocked HTTP responses."""

    @pytest.fixture
    def client(self):
        """Create Anthropic client for testing."""
        return AnthropicClient({
            'api_key': 'sk-ant-test-key',
            'base_url': 'https://api.anthropic.com/v1',
            'default_model': 'claude-3-5-sonnet-20241022',
            'timeout_seconds': 60,
        })

    @pytest.fixture
    def client_no_key(self):
        """Create Anthropic client without API key."""
        return AnthropicClient({})

    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        """Test successful generation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.anthropic.com/v1/messages',
                payload={
                    'content': [{'text': '{"action": "HOLD", "confidence": 0.65}'}],
                    'stop_reason': 'end_turn',
                    'usage': {
                        'input_tokens': 120,
                        'output_tokens': 80,
                    }
                }
            )

            response = await client.generate(
                model='claude-3-5-sonnet-20241022',
                system_prompt='You are a trading assistant.',
                user_message='Analyze BTC/USDT',
            )

            assert response.text == '{"action": "HOLD", "confidence": 0.65}'
            assert response.tokens_used == 200
            assert response.model == 'claude-3-5-sonnet-20241022'
            assert response.finish_reason == 'end_turn'
            assert response.cost_usd > 0
            assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_cost_calculation(self, client):
        """Test correct cost calculation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.anthropic.com/v1/messages',
                payload={
                    'content': [{'text': 'test'}],
                    'stop_reason': 'end_turn',
                    'usage': {
                        'input_tokens': 2000,
                        'output_tokens': 200,
                    }
                }
            )

            response = await client.generate(
                model='claude-3-5-sonnet-20241022',
                system_prompt='System',
                user_message='User',
            )

            # Claude Sonnet: $3/1M input, $15/1M output
            expected_cost = (2000 / 1_000_000) * 3.0 + (200 / 1_000_000) * 15.0
            assert response.cost_usd == pytest.approx(expected_cost, rel=1e-6)

    @pytest.mark.asyncio
    async def test_generate_no_api_key(self, client_no_key):
        """Test error when no API key configured."""
        with pytest.raises(RuntimeError, match='API key not configured'):
            await client_no_key.generate(
                model='claude-3-5-sonnet-20241022',
                system_prompt='System',
                user_message='User',
            )

    @pytest.mark.asyncio
    async def test_generate_api_error(self, client):
        """Test API error handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.anthropic.com/v1/messages',
                status=400,
                payload={'error': {'message': 'Invalid request'}},
            )

            with pytest.raises(RuntimeError, match='Anthropic:.*Invalid'):
                await client.generate(
                    model='claude-3-5-sonnet-20241022',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_timeout(self, client):
        """Test timeout handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.anthropic.com/v1/messages',
                exception=asyncio.TimeoutError(),
            )

            with pytest.raises(RuntimeError, match='timed out'):
                await client.generate(
                    model='claude-3-5-sonnet-20241022',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, client):
        """Test connection error handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.anthropic.com/v1/messages',
                exception=aiohttp.ClientError('Connection refused'),
            )

            with pytest.raises(RuntimeError, match='connection error'):
                await client.generate(
                    model='claude-3-5-sonnet-20241022',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Test health check when healthy."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.anthropic.com/v1/messages',
                payload={
                    'content': [{'text': 'hi'}],
                    'stop_reason': 'end_turn',
                    'usage': {'input_tokens': 1, 'output_tokens': 1},
                },
            )

            is_healthy = await client.health_check()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_no_api_key(self, client_no_key):
        """Test health check without API key."""
        is_healthy = await client_no_key.health_check()
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_api_error(self, client):
        """Test health check with API error."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.anthropic.com/v1/messages',
                status=401,
            )

            is_healthy = await client.health_check()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_opus_pricing(self, client):
        """Test Claude Opus cost calculation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.anthropic.com/v1/messages',
                payload={
                    'content': [{'text': 'test'}],
                    'stop_reason': 'end_turn',
                    'usage': {
                        'input_tokens': 1000,
                        'output_tokens': 500,
                    }
                }
            )

            response = await client.generate(
                model='claude-3-opus-20240229',
                system_prompt='System',
                user_message='User',
            )

            # Claude Opus: $15/1M input, $75/1M output
            expected_cost = (1000 / 1_000_000) * 15.0 + (500 / 1_000_000) * 75.0
            assert response.cost_usd == pytest.approx(expected_cost, rel=1e-6)

    @pytest.mark.asyncio
    async def test_empty_content_handling(self, client):
        """Test handling of empty content response."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.anthropic.com/v1/messages',
                payload={
                    'content': [],
                    'stop_reason': 'end_turn',
                    'usage': {
                        'input_tokens': 50,
                        'output_tokens': 0,
                    }
                }
            )

            response = await client.generate(
                model='claude-3-5-sonnet-20241022',
                system_prompt='System',
                user_message='User',
            )

            assert response.text == ''


# =============================================================================
# DeepSeek Client Tests
# =============================================================================

class TestDeepSeekClient:
    """Test DeepSeekClient with mocked HTTP responses."""

    @pytest.fixture
    def client(self):
        """Create DeepSeek client for testing."""
        return DeepSeekClient({
            'api_key': 'sk-deepseek-test-key',
            'base_url': 'https://api.deepseek.com/v1',
            'default_model': 'deepseek-chat',
            'timeout_seconds': 60,
        })

    @pytest.fixture
    def client_no_key(self):
        """Create DeepSeek client without API key."""
        return DeepSeekClient({})

    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        """Test successful generation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.deepseek.com/v1/chat/completions',
                payload={
                    'choices': [{
                        'message': {'content': '{"action": "BUY", "confidence": 0.88}'},
                        'finish_reason': 'stop',
                    }],
                    'usage': {
                        'prompt_tokens': 80,
                        'completion_tokens': 40,
                    }
                }
            )

            response = await client.generate(
                model='deepseek-chat',
                system_prompt='You are a trading assistant.',
                user_message='Analyze XRP/USDT',
            )

            assert response.text == '{"action": "BUY", "confidence": 0.88}'
            assert response.tokens_used == 120
            assert response.model == 'deepseek-chat'
            assert response.finish_reason == 'stop'
            assert response.cost_usd > 0
            assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_cost_calculation(self, client):
        """Test correct cost calculation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.deepseek.com/v1/chat/completions',
                payload={
                    'choices': [{'message': {'content': 'test'}, 'finish_reason': 'stop'}],
                    'usage': {
                        'prompt_tokens': 10000,
                        'completion_tokens': 5000,
                    }
                }
            )

            response = await client.generate(
                model='deepseek-chat',
                system_prompt='System',
                user_message='User',
            )

            # DeepSeek: $0.14/1M input, $0.28/1M output
            expected_cost = (10000 / 1_000_000) * 0.14 + (5000 / 1_000_000) * 0.28
            assert response.cost_usd == pytest.approx(expected_cost, rel=1e-6)

    @pytest.mark.asyncio
    async def test_generate_no_api_key(self, client_no_key):
        """Test error when no API key configured."""
        with pytest.raises(RuntimeError, match='API key not configured'):
            await client_no_key.generate(
                model='deepseek-chat',
                system_prompt='System',
                user_message='User',
            )

    @pytest.mark.asyncio
    async def test_generate_api_error(self, client):
        """Test API error handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.deepseek.com/v1/chat/completions',
                status=500,
                payload={'error': {'message': 'Server error'}},
            )

            with pytest.raises(RuntimeError, match='DeepSeek:.*Server'):
                await client.generate(
                    model='deepseek-chat',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_timeout(self, client):
        """Test timeout handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.deepseek.com/v1/chat/completions',
                exception=asyncio.TimeoutError(),
            )

            with pytest.raises(RuntimeError, match='timed out'):
                await client.generate(
                    model='deepseek-chat',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, client):
        """Test connection error handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.deepseek.com/v1/chat/completions',
                exception=aiohttp.ClientError('Connection refused'),
            )

            with pytest.raises(RuntimeError, match='connection error'):
                await client.generate(
                    model='deepseek-chat',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Test health check when healthy."""
        with aioresponses() as mocked:
            mocked.get(
                'https://api.deepseek.com/v1/models',
                payload={'data': [{'id': 'deepseek-chat'}]},
            )

            is_healthy = await client.health_check()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_no_api_key(self, client_no_key):
        """Test health check without API key."""
        is_healthy = await client_no_key.health_check()
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_api_error(self, client):
        """Test health check with API error."""
        with aioresponses() as mocked:
            mocked.get(
                'https://api.deepseek.com/v1/models',
                status=401,
            )

            is_healthy = await client.health_check()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_deepseek_reasoner_pricing(self, client):
        """Test DeepSeek Reasoner cost calculation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.deepseek.com/v1/chat/completions',
                payload={
                    'choices': [{'message': {'content': 'test'}, 'finish_reason': 'stop'}],
                    'usage': {
                        'prompt_tokens': 1000,
                        'completion_tokens': 500,
                    }
                }
            )

            response = await client.generate(
                model='deepseek-reasoner',
                system_prompt='System',
                user_message='User',
            )

            # DeepSeek Reasoner: $0.55/1M input, $2.19/1M output
            expected_cost = (1000 / 1_000_000) * 0.55 + (500 / 1_000_000) * 2.19
            assert response.cost_usd == pytest.approx(expected_cost, rel=1e-6)


# =============================================================================
# xAI Client Tests
# =============================================================================

class TestXAIClient:
    """Test XAIClient with mocked HTTP responses."""

    @pytest.fixture
    def client(self):
        """Create xAI client for testing."""
        return XAIClient({
            'api_key': 'xai-test-key',
            'base_url': 'https://api.x.ai/v1',
            'default_model': 'grok-2-1212',
            'timeout_seconds': 60,
        })

    @pytest.fixture
    def client_no_key(self):
        """Create xAI client without API key."""
        return XAIClient({})

    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        """Test successful generation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.x.ai/v1/chat/completions',
                payload={
                    'choices': [{
                        'message': {'content': '{"action": "SELL", "confidence": 0.78}'},
                        'finish_reason': 'stop',
                    }],
                    'usage': {
                        'prompt_tokens': 90,
                        'completion_tokens': 60,
                    }
                }
            )

            response = await client.generate(
                model='grok-2-1212',
                system_prompt='You are a trading assistant.',
                user_message='Analyze BTC/USDT sentiment',
            )

            assert response.text == '{"action": "SELL", "confidence": 0.78}'
            assert response.tokens_used == 150
            assert response.model == 'grok-2-1212'
            assert response.finish_reason == 'stop'
            assert response.cost_usd > 0
            assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_cost_calculation(self, client):
        """Test correct cost calculation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.x.ai/v1/chat/completions',
                payload={
                    'choices': [{'message': {'content': 'test'}, 'finish_reason': 'stop'}],
                    'usage': {
                        'prompt_tokens': 1000,
                        'completion_tokens': 500,
                    }
                }
            )

            response = await client.generate(
                model='grok-2',
                system_prompt='System',
                user_message='User',
            )

            # Grok-2: $2/1M input, $10/1M output
            expected_cost = (1000 / 1_000_000) * 2.0 + (500 / 1_000_000) * 10.0
            assert response.cost_usd == pytest.approx(expected_cost, rel=1e-6)

    @pytest.mark.asyncio
    async def test_generate_no_api_key(self, client_no_key):
        """Test error when no API key configured."""
        with pytest.raises(RuntimeError, match='API key not configured'):
            await client_no_key.generate(
                model='grok-2',
                system_prompt='System',
                user_message='User',
            )

    @pytest.mark.asyncio
    async def test_generate_api_error(self, client):
        """Test API error handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.x.ai/v1/chat/completions',
                status=403,
                payload={'error': {'message': 'Forbidden'}},
            )

            with pytest.raises(RuntimeError, match='xAI:.*Forbidden'):
                await client.generate(
                    model='grok-2',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_timeout(self, client):
        """Test timeout handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.x.ai/v1/chat/completions',
                exception=asyncio.TimeoutError(),
            )

            with pytest.raises(RuntimeError, match='timed out'):
                await client.generate(
                    model='grok-2',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, client):
        """Test connection error handling."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.x.ai/v1/chat/completions',
                exception=aiohttp.ClientError('Connection refused'),
            )

            with pytest.raises(RuntimeError, match='connection error'):
                await client.generate(
                    model='grok-2',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Test health check when healthy."""
        with aioresponses() as mocked:
            mocked.get(
                'https://api.x.ai/v1/models',
                payload={'data': [{'id': 'grok-2'}]},
            )

            is_healthy = await client.health_check()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_no_api_key(self, client_no_key):
        """Test health check without API key."""
        is_healthy = await client_no_key.health_check()
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_api_error(self, client):
        """Test health check with API error."""
        with aioresponses() as mocked:
            mocked.get(
                'https://api.x.ai/v1/models',
                status=500,
            )

            is_healthy = await client.health_check()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_generate_with_search_success(self, client):
        """Test generate with search enabled."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.x.ai/v1/chat/completions',
                payload={
                    'choices': [{
                        'message': {'content': 'BTC sentiment is bullish based on Twitter'},
                        'finish_reason': 'stop',
                    }],
                    'usage': {
                        'prompt_tokens': 100,
                        'completion_tokens': 50,
                    }
                }
            )

            response = await client.generate_with_search(
                model='grok-2',
                system_prompt='Analyze crypto sentiment',
                user_message='What is the current sentiment on BTC?',
                search_enabled=True,
            )

            assert 'bullish' in response.text
            assert response.tokens_used == 150

    @pytest.mark.asyncio
    async def test_generate_with_search_no_api_key(self, client_no_key):
        """Test generate with search without API key."""
        with pytest.raises(RuntimeError, match='API key not configured'):
            await client_no_key.generate_with_search(
                model='grok-2',
                system_prompt='System',
                user_message='User',
            )

    @pytest.mark.asyncio
    async def test_generate_with_search_timeout(self, client):
        """Test generate with search timeout."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.x.ai/v1/chat/completions',
                exception=asyncio.TimeoutError(),
            )

            with pytest.raises(RuntimeError, match='timed out'):
                await client.generate_with_search(
                    model='grok-2',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_generate_with_search_connection_error(self, client):
        """Test generate with search connection error."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.x.ai/v1/chat/completions',
                exception=aiohttp.ClientError('Connection refused'),
            )

            with pytest.raises(RuntimeError, match='connection error'):
                await client.generate_with_search(
                    model='grok-2',
                    system_prompt='System',
                    user_message='User',
                )

    @pytest.mark.asyncio
    async def test_grok_beta_pricing(self, client):
        """Test Grok Beta cost calculation."""
        with aioresponses() as mocked:
            mocked.post(
                'https://api.x.ai/v1/chat/completions',
                payload={
                    'choices': [{'message': {'content': 'test'}, 'finish_reason': 'stop'}],
                    'usage': {
                        'prompt_tokens': 1000,
                        'completion_tokens': 500,
                    }
                }
            )

            response = await client.generate(
                model='grok-beta',
                system_prompt='System',
                user_message='User',
            )

            # Grok Beta: $5/1M input, $15/1M output
            expected_cost = (1000 / 1_000_000) * 5.0 + (500 / 1_000_000) * 15.0
            assert response.cost_usd == pytest.approx(expected_cost, rel=1e-6)


# =============================================================================
# Cross-Client Tests
# =============================================================================

class TestCrossClientBehavior:
    """Test consistent behavior across all clients."""

    @pytest.mark.asyncio
    async def test_all_clients_have_provider_name(self):
        """All clients should have a provider name."""
        clients = [
            OllamaClient({'base_url': 'http://localhost:11434'}),
            OpenAIClient({'api_key': 'test'}),
            AnthropicClient({'api_key': 'test'}),
            DeepSeekClient({'api_key': 'test'}),
            XAIClient({'api_key': 'test'}),
        ]

        expected_names = ['ollama', 'openai', 'anthropic', 'deepseek', 'xai']

        for client, expected_name in zip(clients, expected_names):
            assert client.provider_name == expected_name

    @pytest.mark.asyncio
    async def test_all_clients_track_stats(self):
        """All clients should track stats correctly."""
        clients = [
            OllamaClient({'base_url': 'http://localhost:11434'}),
            OpenAIClient({'api_key': 'test'}),
            AnthropicClient({'api_key': 'test'}),
            DeepSeekClient({'api_key': 'test'}),
            XAIClient({'api_key': 'test'}),
        ]

        for client in clients:
            stats = client.get_stats()
            assert 'total_requests' in stats
            assert 'total_tokens' in stats
            assert 'total_cost_usd' in stats
            assert stats['total_requests'] == 0

    @pytest.mark.asyncio
    async def test_all_clients_return_llm_response(self):
        """All clients should return LLMResponse objects."""
        from triplegain.src.llm.clients.base import LLMResponse

        # Test Ollama
        client = OllamaClient({'base_url': 'http://localhost:11434'})
        with aioresponses() as mocked:
            mocked.post(
                'http://localhost:11434/api/generate',
                payload={'response': 'test', 'done': True, 'eval_count': 10, 'prompt_eval_count': 20}
            )
            response = await client.generate('model', 'sys', 'user')
            assert isinstance(response, LLMResponse)

        # Test OpenAI
        client = OpenAIClient({'api_key': 'test'})
        with aioresponses() as mocked:
            mocked.post(
                'https://api.openai.com/v1/chat/completions',
                payload={
                    'choices': [{'message': {'content': 'test'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5}
                }
            )
            response = await client.generate('model', 'sys', 'user')
            assert isinstance(response, LLMResponse)
