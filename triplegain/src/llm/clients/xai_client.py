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

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (as of Dec 2024)
XAI_PRICING = {
    'grok-2': {'input': 2.00, 'output': 10.00},
    'grok-2-1212': {'input': 2.00, 'output': 10.00},
    'grok-2-latest': {'input': 2.00, 'output': 10.00},
    'grok-beta': {'input': 5.00, 'output': 15.00},
}


class XAIClient(BaseLLMClient):
    """
    xAI API client for Grok models.

    Grok has access to real-time X data for sentiment analysis.
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
                            f"xAI API error: {response.status} - {error}"
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

        Args:
            model: Model name
            system_prompt: System prompt
            user_message: User message
            search_enabled: Enable real-time search for grounding
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLMResponse with generated text
        """
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

        # Note: xAI search features may have specific API params
        # This is a placeholder for when the API supports explicit search control

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
                            f"xAI API error: {response.status} - {error}"
                        )

                    data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            choice = data['choices'][0]
            text = choice['message']['content']

            usage = data.get('usage', {})
            total_tokens = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)

            pricing = XAI_PRICING.get(model, {'input': 2.0, 'output': 10.0})
            cost = (
                (usage.get('prompt_tokens', 0) / 1_000_000) * pricing['input'] +
                (usage.get('completion_tokens', 0) / 1_000_000) * pricing['output']
            )

            self._update_stats(total_tokens, cost)

            return LLMResponse(
                text=text,
                tokens_used=total_tokens,
                model=model,
                finish_reason=choice.get('finish_reason', 'stop'),
                latency_ms=latency_ms,
                cost_usd=cost,
                raw_response=data,
            )

        except asyncio.TimeoutError:
            raise RuntimeError("xAI request timed out")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"xAI connection error: {e}")
