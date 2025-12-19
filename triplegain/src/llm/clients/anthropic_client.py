"""
Anthropic Client - Claude model integration.

Used for A/B testing trading decisions with Claude Sonnet and Opus.
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
ANTHROPIC_PRICING = {
    'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
    'claude-3-opus-20240229': {'input': 15.00, 'output': 75.00},
    'claude-3-sonnet-20240229': {'input': 3.00, 'output': 15.00},
    'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
}

# Required fields in Anthropic response (2A-13)
ANTHROPIC_REQUIRED_FIELDS = ['id', 'content', 'stop_reason']


class AnthropicClient(BaseLLMClient):
    """
    Anthropic API client for Claude models.

    Supports Claude 3 Opus, Sonnet, and Haiku.
    Features:
    - Connection pooling for performance (2A-02)
    - Empty content warning (2A-11)
    - Rate limit header parsing (2A-07)
    - Response schema validation (2A-13)
    """

    provider_name = "anthropic"

    def __init__(self, config: dict):
        """
        Initialize Anthropic client.

        Args:
            config: Configuration with:
                - api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
                - default_model: Default model to use
                - timeout_seconds: Request timeout
        """
        super().__init__(config)
        self.api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning("Anthropic API key not configured")

        self.base_url = config.get('base_url', 'https://api.anthropic.com/v1')
        self.default_model = config.get('default_model', 'claude-3-5-sonnet-20241022')
        self.timeout = aiohttp.ClientTimeout(
            total=config.get('timeout_seconds', 60)
        )
        self.api_version = config.get('api_version', '2023-06-01')

    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        if not self.api_key:
            raise RuntimeError("Anthropic API key not configured")

        model = model or self.default_model
        start_time = time.perf_counter()

        # For JSON responses, add instruction to system prompt (2A-06)
        # Note: Anthropic doesn't have a native JSON mode like OpenAI
        json_system_prompt = system_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON, no markdown or explanation."

        payload = {
            'model': model,
            'system': json_system_prompt,
            'messages': [
                {'role': 'user', 'content': user_message},
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
        }

        headers = self._get_headers(self.api_key, auth_type="x-api-key")
        headers['anthropic-version'] = self.api_version

        try:
            session = await self._get_session(self.timeout)
            async with session.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=headers
            ) as response:
                # Parse rate limit headers (2A-07)
                self._parse_rate_limit_headers(dict(response.headers))

                if response.status != 200:
                    error = await response.json()
                    # Sanitize error message (2A-03)
                    raise RuntimeError(
                        sanitize_error_message(error, "Anthropic")
                    )

                data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Validate response schema (2A-13)
            # Note: Anthropic may return empty content on some stop reasons
            if 'content' not in data:
                logger.warning(
                    f"Anthropic response missing 'content' field. "
                    f"stop_reason: {data.get('stop_reason')}"
                )

            # Extract response with empty content warning (2A-11)
            content = data.get('content', [])
            if not content:
                logger.warning(
                    f"Anthropic returned empty content for model {model}. "
                    f"stop_reason: {data.get('stop_reason')}"
                )
                text = ''
            else:
                text = content[0].get('text', '')

            stop_reason = data.get('stop_reason', 'end_turn')

            # Token usage
            usage = data.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            total_tokens = input_tokens + output_tokens

            # Calculate cost using actual tokens (2A-05)
            pricing = ANTHROPIC_PRICING.get(model, {'input': 3.0, 'output': 15.0})
            cost = (
                (input_tokens / 1_000_000) * pricing['input'] +
                (output_tokens / 1_000_000) * pricing['output']
            )

            self._update_stats(total_tokens, cost)

            logger.debug(
                f"Anthropic generate: model={model}, tokens={total_tokens}, "
                f"cost=${cost:.4f}, latency={latency_ms}ms"
            )

            return LLMResponse(
                text=text,
                tokens_used=total_tokens,
                model=model,
                finish_reason=stop_reason,
                latency_ms=latency_ms,
                cost_usd=cost,
                input_tokens=input_tokens,  # 2A-05
                output_tokens=output_tokens,  # 2A-05
                raw_response=data,
            )

        except asyncio.TimeoutError:
            raise RuntimeError("Anthropic request timed out")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Anthropic connection error: {e}")

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        if not self.api_key:
            return False

        try:
            # Anthropic doesn't have a simple health endpoint,
            # so we make a minimal request
            session = await self._get_session(aiohttp.ClientTimeout(total=10))
            headers = self._get_headers(self.api_key, auth_type="x-api-key")
            headers['anthropic-version'] = self.api_version

            payload = {
                'model': 'claude-3-haiku-20240307',
                'max_tokens': 1,
                'messages': [{'role': 'user', 'content': 'hi'}],
            }
            async with session.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=headers
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            return False
