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

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (as of Dec 2024)
OPENAI_PRICING = {
    'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
    'gpt-4-turbo-preview': {'input': 10.00, 'output': 30.00},
    'gpt-4o': {'input': 2.50, 'output': 10.00},
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    'gpt-4': {'input': 30.00, 'output': 60.00},
}


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client for GPT models.

    Supports all GPT-4 and GPT-4o variants.
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

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error = await response.json()
                        raise RuntimeError(
                            f"OpenAI API error: {response.status} - {error}"
                        )

                    data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract response
            choice = data['choices'][0]
            text = choice['message']['content']
            finish_reason = choice.get('finish_reason', 'stop')

            # Token usage
            usage = data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost
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
                raw_response=data,
            )

        except asyncio.TimeoutError:
            raise RuntimeError("OpenAI request timed out")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"OpenAI connection error: {e}")

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self.api_key:
            return False

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                headers = {'Authorization': f'Bearer {self.api_key}'}
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
