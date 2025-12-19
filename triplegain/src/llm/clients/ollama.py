"""
Ollama Client - Local LLM integration via Ollama.

Provides low-latency access to local models like Qwen 2.5 7B.
Used for Tier 1 (execution) tasks: TA, Regime Detection, real-time decisions.
"""

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """
    Ollama client for local LLM inference.

    Default configuration:
    - Base URL: http://localhost:11434
    - Model: qwen2.5:7b
    - Model path: /media/rese/2tb_drive/ollama_config/
    """

    provider_name = "ollama"

    def __init__(self, config: dict):
        """
        Initialize Ollama client.

        Args:
            config: Configuration dictionary with:
                - base_url: Ollama API URL (default: http://localhost:11434)
                - timeout_seconds: Request timeout (default: 30)
                - default_model: Default model to use
        """
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.timeout = aiohttp.ClientTimeout(
            total=config.get('timeout_seconds', 30)
        )
        self.default_model = config.get('default_model', 'qwen2.5:7b')

        # Ollama-specific options
        self.default_options = config.get('default_options', {
            'temperature': 0.3,
            'top_p': 0.9,
            'top_k': 40,
            'num_predict': 1024,
            'num_ctx': 8192,
            'repeat_penalty': 1.1,
        })

    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate a response using Ollama.

        Args:
            model: Model name (e.g., 'qwen2.5:7b')
            system_prompt: System prompt
            user_message: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated text
        """
        model = model or self.default_model
        start_time = time.perf_counter()

        # Build request payload
        payload = {
            'model': model,
            'prompt': user_message,
            'system': system_prompt,
            'stream': False,
            'options': {
                **self.default_options,
                'temperature': temperature,
                'num_predict': max_tokens,
            }
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Ollama API error: {response.status} - {error_text}"
                        )

                    data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract metrics
            text = data.get('response', '')
            eval_count = data.get('eval_count', 0)
            prompt_eval_count = data.get('prompt_eval_count', 0)
            total_tokens = eval_count + prompt_eval_count

            # Update stats (local = no cost)
            self._update_stats(total_tokens, 0.0)

            logger.debug(
                f"Ollama generate: model={model}, tokens={total_tokens}, "
                f"latency={latency_ms}ms"
            )

            return LLMResponse(
                text=text,
                tokens_used=total_tokens,
                model=model,
                finish_reason=data.get('done_reason', 'stop'),
                latency_ms=latency_ms,
                cost_usd=0.0,
                raw_response=data,
            )

        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Ollama timeout after {latency_ms}ms")
            raise RuntimeError(f"Ollama request timed out after {latency_ms}ms")

        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            raise RuntimeError(f"Ollama connection failed: {e}")

    async def health_check(self) -> bool:
        """
        Check if Ollama is running and responsive.

        Returns:
            True if healthy, False otherwise
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        logger.debug(f"Ollama healthy, {len(models)} models available")
                        return True
                    return False
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> list[dict]:
        """
        List available models in Ollama.

        Returns:
            List of model info dictionaries
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('models', [])
                    return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def is_model_available(self, model: str) -> bool:
        """
        Check if a specific model is available.

        Args:
            model: Model name to check

        Returns:
            True if model is available
        """
        models = await self.list_models()
        model_names = [m.get('name', '') for m in models]
        return model in model_names or f"{model}:latest" in model_names

    async def generate_with_chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate using chat-style API (alternative format).

        Args:
            model: Model name
            messages: List of {role, content} messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLMResponse with generated text
        """
        model = model or self.default_model
        start_time = time.perf_counter()

        payload = {
            'model': model,
            'messages': messages,
            'stream': False,
            'options': {
                **self.default_options,
                'temperature': temperature,
                'num_predict': max_tokens,
            }
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Ollama chat API error: {response.status} - {error_text}"
                        )

                    data = await response.json()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract response
            message = data.get('message', {})
            text = message.get('content', '')
            eval_count = data.get('eval_count', 0)
            prompt_eval_count = data.get('prompt_eval_count', 0)
            total_tokens = eval_count + prompt_eval_count

            self._update_stats(total_tokens, 0.0)

            return LLMResponse(
                text=text,
                tokens_used=total_tokens,
                model=model,
                finish_reason=data.get('done_reason', 'stop'),
                latency_ms=latency_ms,
                cost_usd=0.0,
                raw_response=data,
            )

        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Ollama chat timeout after {latency_ms}ms")
            raise RuntimeError(f"Ollama chat request timed out after {latency_ms}ms")

        except aiohttp.ClientError as e:
            logger.error(f"Ollama chat connection error: {e}")
            raise RuntimeError(f"Ollama chat connection failed: {e}")
