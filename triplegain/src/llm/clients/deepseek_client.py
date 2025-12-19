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

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (as of Dec 2024)
DEEPSEEK_PRICING = {
    'deepseek-chat': {'input': 0.14, 'output': 0.28},
    'deepseek-coder': {'input': 0.14, 'output': 0.28},
    'deepseek-reasoner': {'input': 0.55, 'output': 2.19},
}


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek API client.

    Supports DeepSeek V3 (deepseek-chat) and DeepSeek Coder.
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
                            f"DeepSeek API error: {response.status} - {error}"
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
            logger.warning(f"DeepSeek health check failed: {e}")
            return False
