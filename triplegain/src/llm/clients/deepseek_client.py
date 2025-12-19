"""
DeepSeek Client - DeepSeek V3 model integration.

Used for A/B testing trading decisions.
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
DEEPSEEK_PRICING = {
    'deepseek-chat': {'input': 0.14, 'output': 0.28},
    'deepseek-coder': {'input': 0.14, 'output': 0.28},
    'deepseek-reasoner': {'input': 0.55, 'output': 2.19},
}

# Required fields in DeepSeek response (2A-13)
DEEPSEEK_REQUIRED_FIELDS = ['id', 'choices']


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek API client.

    Supports DeepSeek V3 (deepseek-chat) and DeepSeek Coder.
    Features:
    - Connection pooling for performance (2A-02)
    - JSON response mode (2A-06)
    - Rate limit header parsing (2A-07)
    - Response schema validation (2A-13)
    """

    provider_name = "deepseek"

    def __init__(self, config: dict):
        """
        Initialize DeepSeek client.

        Args:
            config: Configuration with:
                - api_key: DeepSeek API key (or DEEPSEEK_API_KEY env var)
                - default_model: Default model to use
                - timeout_seconds: Request timeout
                - json_mode: Enable JSON response mode (default: True)
        """
        super().__init__(config)
        self.api_key = config.get('api_key') or os.environ.get('DEEPSEEK_API_KEY')
        if not self.api_key:
            logger.warning("DeepSeek API key not configured")

        self.base_url = config.get('base_url', 'https://api.deepseek.com/v1')
        self.default_model = config.get('default_model', 'deepseek-chat')
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
        """Generate response using DeepSeek API."""
        if not self.api_key:
            raise RuntimeError("DeepSeek API key not configured")

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
        # DeepSeek uses OpenAI-compatible API
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
                        sanitize_error_message(error, "DeepSeek")
                    )

                data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Validate response schema (2A-13)
            self._validate_response_schema(data, DEEPSEEK_REQUIRED_FIELDS, "DeepSeek")

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
            pricing = DEEPSEEK_PRICING.get(model, {'input': 0.14, 'output': 0.28})
            cost = (
                (prompt_tokens / 1_000_000) * pricing['input'] +
                (completion_tokens / 1_000_000) * pricing['output']
            )

            self._update_stats(total_tokens, cost)

            logger.debug(
                f"DeepSeek generate: model={model}, tokens={total_tokens}, "
                f"cost=${cost:.6f}, latency={latency_ms}ms"
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
            raise RuntimeError("DeepSeek request timed out")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"DeepSeek connection error: {e}")

    async def health_check(self) -> bool:
        """Check if DeepSeek API is accessible."""
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
            logger.warning(f"DeepSeek health check failed: {e}")
            return False
