"""
xAI Client - Grok model integration.

Used for A/B testing trading decisions and sentiment analysis.
Grok has access to real-time X (Twitter) data.
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
XAI_PRICING = {
    'grok-2': {'input': 2.00, 'output': 10.00},
    'grok-2-1212': {'input': 2.00, 'output': 10.00},
    'grok-2-latest': {'input': 2.00, 'output': 10.00},
    'grok-beta': {'input': 5.00, 'output': 15.00},
}

# Required fields in xAI response (2A-13)
XAI_REQUIRED_FIELDS = ['id', 'choices']


class XAIClient(BaseLLMClient):
    """
    xAI API client for Grok models.

    Grok has access to real-time X data for sentiment analysis.
    Features:
    - Connection pooling for performance (2A-02)
    - JSON response mode (2A-06)
    - Rate limit header parsing (2A-07)
    - Response schema validation (2A-13)
    """

    provider_name = "xai"

    def __init__(self, config: dict):
        """
        Initialize xAI client.

        Args:
            config: Configuration with:
                - api_key: xAI API key (or XAI_API_KEY env var)
                - default_model: Default model to use
                - timeout_seconds: Request timeout
                - json_mode: Enable JSON response mode (default: True)
        """
        super().__init__(config)
        self.api_key = config.get('api_key') or os.environ.get('XAI_API_KEY')
        if not self.api_key:
            logger.warning("xAI API key not configured")

        self.base_url = config.get('base_url', 'https://api.x.ai/v1')
        self.default_model = config.get('default_model', 'grok-2-1212')
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
        """Generate response using xAI API."""
        if not self.api_key:
            raise RuntimeError("xAI API key not configured")

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
        # xAI uses OpenAI-compatible API
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
                        sanitize_error_message(error, "xAI")
                    )

                data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Validate response schema (2A-13)
            self._validate_response_schema(data, XAI_REQUIRED_FIELDS, "xAI")

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
            pricing = XAI_PRICING.get(model, {'input': 2.0, 'output': 10.0})
            cost = (
                (prompt_tokens / 1_000_000) * pricing['input'] +
                (completion_tokens / 1_000_000) * pricing['output']
            )

            self._update_stats(total_tokens, cost)

            logger.debug(
                f"xAI generate: model={model}, tokens={total_tokens}, "
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
            raise RuntimeError("xAI request timed out")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"xAI connection error: {e}")

    async def health_check(self) -> bool:
        """Check if xAI API is accessible."""
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
            logger.warning(f"xAI health check failed: {e}")
            return False

    async def generate_with_search(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        search_enabled: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate response with optional X/web search grounding.

        WARNING (2A-12): Search functionality is not yet fully implemented.
        The search_enabled parameter is accepted but may not activate
        actual search grounding - this depends on xAI's API capabilities.

        See: https://docs.x.ai/api for current search capabilities.

        Args:
            model: Model name
            system_prompt: System prompt
            user_message: User message
            search_enabled: Enable real-time search for grounding (NOTE: may not be functional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLMResponse with generated text
        """
        if not self.api_key:
            raise RuntimeError("xAI API key not configured")

        # Log warning about search functionality (2A-12)
        if search_enabled:
            logger.warning(
                "generate_with_search: search_enabled=True but search functionality "
                "may not be active. Consult xAI API docs for current search support. "
                "This method currently behaves similarly to generate()."
            )

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

        # Note: xAI search features may have specific API params
        # TODO: Update when xAI documents explicit search control API

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
                        sanitize_error_message(error, "xAI")
                    )

                data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Validate response schema (2A-13)
            self._validate_response_schema(data, XAI_REQUIRED_FIELDS, "xAI")

            choice = data['choices'][0]
            text = choice['message']['content']

            usage = data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens

            pricing = XAI_PRICING.get(model, {'input': 2.0, 'output': 10.0})
            cost = (
                (prompt_tokens / 1_000_000) * pricing['input'] +
                (completion_tokens / 1_000_000) * pricing['output']
            )

            self._update_stats(total_tokens, cost)

            return LLMResponse(
                text=text,
                tokens_used=total_tokens,
                model=model,
                finish_reason=choice.get('finish_reason', 'stop'),
                latency_ms=latency_ms,
                cost_usd=cost,
                input_tokens=prompt_tokens,  # 2A-05
                output_tokens=completion_tokens,  # 2A-05
                raw_response=data,
            )

        except asyncio.TimeoutError:
            raise RuntimeError("xAI request timed out")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"xAI connection error: {e}")
