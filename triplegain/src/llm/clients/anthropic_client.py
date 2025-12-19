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

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (as of Dec 2024)
ANTHROPIC_PRICING = {
    'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
    'claude-3-opus-20240229': {'input': 15.00, 'output': 75.00},
    'claude-3-sonnet-20240229': {'input': 3.00, 'output': 15.00},
    'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
}


class AnthropicClient(BaseLLMClient):
    """
    Anthropic API client for Claude models.

    Supports Claude 3 Opus, Sonnet, and Haiku.
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

        payload = {
            'model': model,
            'system': system_prompt,
            'messages': [
                {'role': 'user', 'content': user_message},
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
        }

        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': self.api_version,
            'Content-Type': 'application/json',
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error = await response.json()
                        raise RuntimeError(
                            f"Anthropic API error: {response.status} - {error}"
                        )

                    data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract response
            content = data.get('content', [])
            text = content[0]['text'] if content else ''
            stop_reason = data.get('stop_reason', 'end_turn')

            # Token usage
            usage = data.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            total_tokens = input_tokens + output_tokens

            # Calculate cost
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
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                headers = {
                    'x-api-key': self.api_key,
                    'anthropic-version': self.api_version,
                    'Content-Type': 'application/json',
                }
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
