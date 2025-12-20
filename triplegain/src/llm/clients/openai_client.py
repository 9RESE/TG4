"""
OpenAI Client - GPT model integration.

Used for A/B testing trading decisions with GPT-4-turbo.
"""

import asyncio
import logging
import os
import time
from typing import Optional

import aiohttp

from .base import (
    BaseLLMClient,
    LLMResponse,
    sanitize_error_message,
    get_user_agent,
    create_ssl_context,
)

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (as of Dec 2024)
# Note: These are used locally for cost calculation with actual tokens
OPENAI_PRICING = {
    'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
    'gpt-4-turbo-preview': {'input': 10.00, 'output': 30.00},
    'gpt-4o': {'input': 2.50, 'output': 10.00},
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    'gpt-4': {'input': 30.00, 'output': 60.00},
    # Web search enabled models (higher cost due to search queries)
    'gpt-4o-search-preview': {'input': 2.50, 'output': 10.00},
    'gpt-4o-mini-search-preview': {'input': 0.15, 'output': 0.60},
}

# Required fields in OpenAI response (2A-13)
OPENAI_REQUIRED_FIELDS = ['id', 'choices']


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client for GPT models.

    Supports all GPT-4 and GPT-4o variants.
    Features:
    - Connection pooling for performance (2A-02)
    - JSON response mode (2A-06)
    - Rate limit header parsing (2A-07)
    - Response schema validation (2A-13)
    """

    provider_name = "openai"

    def __init__(self, config: dict):
        """
        Initialize OpenAI client.

        Args:
            config: Configuration with:
                - api_key: OpenAI API key (or OPENAI_API_KEY env var)
                - default_model: Default model to use
                - timeout_seconds: Request timeout
                - json_mode: Enable JSON response mode (default: True)
        """
        super().__init__(config)
        self.api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not configured")

        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.default_model = config.get('default_model', 'gpt-4-turbo')
        self.timeout = aiohttp.ClientTimeout(
            total=config.get('timeout_seconds', 60)
        )
        self.json_mode = config.get('json_mode', True)  # 2A-06

    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self.api_key:
            raise RuntimeError("OpenAI API key not configured")

        model = model or self.default_model
        start_time = time.perf_counter()

        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
        }

        # Enable JSON response mode (2A-06)
        if self.json_mode:
            payload['response_format'] = {'type': 'json_object'}

        headers = self._get_headers(self.api_key, auth_type="bearer")

        try:
            session = await self._get_session(self.timeout)
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                # Parse rate limit headers (2A-07)
                self._parse_rate_limit_headers(dict(response.headers))

                if response.status != 200:
                    error = await response.json()
                    # Sanitize error message (2A-03)
                    raise RuntimeError(
                        sanitize_error_message(error, "OpenAI")
                    )

                data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Validate response schema (2A-13)
            self._validate_response_schema(data, OPENAI_REQUIRED_FIELDS, "OpenAI")

            # Extract response
            choice = data['choices'][0]
            text = choice['message']['content']
            finish_reason = choice.get('finish_reason', 'stop')

            # Token usage
            usage = data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost using actual tokens (2A-05)
            pricing = OPENAI_PRICING.get(model, {'input': 10.0, 'output': 30.0})
            cost = (
                (prompt_tokens / 1_000_000) * pricing['input'] +
                (completion_tokens / 1_000_000) * pricing['output']
            )

            self._update_stats(total_tokens, cost)

            logger.debug(
                f"OpenAI generate: model={model}, tokens={total_tokens}, "
                f"cost=${cost:.4f}, latency={latency_ms}ms"
            )

            return LLMResponse(
                text=text,
                tokens_used=total_tokens,
                model=model,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                cost_usd=cost,
                input_tokens=prompt_tokens,  # 2A-05
                output_tokens=completion_tokens,  # 2A-05
                raw_response=data,
            )

        except asyncio.TimeoutError:
            raise RuntimeError("OpenAI request timed out")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"OpenAI connection error: {e}")

    async def generate_with_search(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        search_context_size: str = "medium",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response with web search enabled.

        Uses OpenAI's web search capability via gpt-4o-search-preview or
        gpt-4o-mini-search-preview models. These models can search the web
        for real-time information.

        Args:
            model: Model to use (should be a search-enabled model)
            system_prompt: System prompt
            user_message: User message
            search_context_size: Search depth - "low" (fast), "medium", or "high" (thorough)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with web-informed answer

        Note:
            If a non-search model is specified, automatically uses gpt-4o-search-preview.
        """
        if not self.api_key:
            raise RuntimeError("OpenAI API key not configured")

        # Ensure we use a search-enabled model
        search_models = ['gpt-4o-search-preview', 'gpt-4o-mini-search-preview']
        if model not in search_models:
            logger.debug(f"Switching from {model} to gpt-4o-search-preview for web search")
            model = 'gpt-4o-search-preview'

        start_time = time.perf_counter()

        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'web_search_options': {
                'search_context_size': search_context_size,
            },
        }

        # Note: JSON mode is NOT compatible with web search models
        # Do not include response_format for search models

        headers = self._get_headers(self.api_key, auth_type="bearer")

        try:
            session = await self._get_session(self.timeout)
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                # Parse rate limit headers (2A-07)
                self._parse_rate_limit_headers(dict(response.headers))

                if response.status != 200:
                    error = await response.json()
                    raise RuntimeError(
                        sanitize_error_message(error, "OpenAI")
                    )

                data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Validate response schema (2A-13)
            self._validate_response_schema(data, OPENAI_REQUIRED_FIELDS, "OpenAI")

            # Extract response
            choice = data['choices'][0]
            text = choice['message']['content']
            finish_reason = choice.get('finish_reason', 'stop')

            # Token usage
            usage = data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost using actual tokens
            pricing = OPENAI_PRICING.get(model, {'input': 2.5, 'output': 10.0})
            cost = (
                (prompt_tokens / 1_000_000) * pricing['input'] +
                (completion_tokens / 1_000_000) * pricing['output']
            )

            self._update_stats(total_tokens, cost)

            logger.debug(
                f"OpenAI generate_with_search: model={model}, tokens={total_tokens}, "
                f"cost=${cost:.4f}, latency={latency_ms}ms, search={search_context_size}"
            )

            return LLMResponse(
                text=text,
                tokens_used=total_tokens,
                model=model,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                cost_usd=cost,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                raw_response=data,
            )

        except asyncio.TimeoutError:
            raise RuntimeError("OpenAI web search request timed out")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"OpenAI connection error: {e}")

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self.api_key:
            return False

        try:
            session = await self._get_session(aiohttp.ClientTimeout(total=10))
            headers = self._get_headers(self.api_key, auth_type="bearer")
            async with session.get(
                f"{self.base_url}/models",
                headers=headers
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
