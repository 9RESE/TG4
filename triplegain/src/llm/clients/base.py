"""
Base LLM Client - Abstract interface for all LLM providers.

Provides a unified interface for:
- Ollama (local)
- OpenAI
- Anthropic
- DeepSeek
- xAI (Grok)

Includes:
- Rate limiting per provider
- Exponential backoff retry logic
- Cost tracking
"""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

logger = logging.getLogger(__name__)


def parse_json_response(response_text: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Parse JSON from LLM response, handling common response formats.

    Handles:
    - Plain JSON
    - Markdown-wrapped JSON (```json\\n{...}\\n```)
    - JSON embedded in text
    - Triple backtick without language specifier

    Args:
        response_text: Raw response text from LLM

    Returns:
        Tuple of (parsed_dict or None, error_message or None)
    """
    if not response_text or not response_text.strip():
        return None, "Empty response"

    text = response_text.strip()

    # Strategy 1: Try to extract from markdown code block
    # Matches ```json\n{...}\n``` or ```\n{...}\n```
    markdown_patterns = [
        r'```json\s*\n?([\s\S]*?)\n?```',  # ```json ... ```
        r'```\s*\n?([\s\S]*?)\n?```',       # ``` ... ```
    ]

    for pattern in markdown_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                extracted = match.group(1).strip()
                return json.loads(extracted), None
            except json.JSONDecodeError as e:
                logger.debug(f"Markdown JSON extraction failed: {e}")
                continue

    # Strategy 2: Find JSON object in text (handles { ... } anywhere)
    # Use a simple bracket matching approach
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group()), None
        except json.JSONDecodeError as e:
            logger.debug(f"JSON object extraction failed: {e}")

    # Strategy 3: Try full text as JSON
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        logger.debug(f"Full text JSON parsing failed: {e}")

    # Strategy 4: More aggressive search for nested JSON
    start_idx = text.find('{')
    if start_idx >= 0:
        depth = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

        if end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx]), None
            except json.JSONDecodeError:
                pass

    return None, f"Failed to extract valid JSON from response: {text[:100]}..."


def validate_json_schema(data: dict, required_fields: list[str], field_types: Optional[dict] = None) -> tuple[bool, list[str]]:
    """
    Validate JSON data against expected schema.

    Args:
        data: Parsed JSON dictionary
        required_fields: List of required field names
        field_types: Optional dict of field_name -> expected type

    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if field_types:
        for field, expected_type in field_types.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    errors.append(
                        f"Field '{field}' expected {expected_type.__name__}, "
                        f"got {type(data[field]).__name__}"
                    )

    return len(errors) == 0, errors


# Default rate limits per provider (requests per minute)
DEFAULT_RATE_LIMITS = {
    'ollama': 120,      # Local, high limit
    'openai': 60,       # OpenAI Tier 1
    'anthropic': 60,    # Anthropic default
    'deepseek': 60,     # DeepSeek default
    'xai': 60,          # xAI default
}

# Model cost per 1K tokens (input, output)
# Note: These are converted from standard per-1M pricing
# e.g., OpenAI $10/1M = $0.01/1K, Anthropic $3/1M = $0.003/1K
# Calculation: (tokens / 1000) * cost_per_1k
MODEL_COSTS = {
    # OpenAI
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
    'gpt-4o': {'input': 0.005, 'output': 0.015},
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
    # Anthropic
    'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
    'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075},
    'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
    # DeepSeek
    'deepseek-chat': {'input': 0.00014, 'output': 0.00028},
    'deepseek-reasoner': {'input': 0.00055, 'output': 0.00219},
    # xAI
    'grok-2-1212': {'input': 0.002, 'output': 0.010},
    'grok-beta': {'input': 0.005, 'output': 0.015},
    # Ollama (local, free)
    'qwen2.5:7b': {'input': 0.0, 'output': 0.0},
}


class RateLimiter:
    """
    Async rate limiter using sliding window algorithm.

    Thread-safe for concurrent async operations.
    """

    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests = requests_per_minute
        self.window_seconds = 60.0
        self._request_times: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """
        Acquire a slot for a request, waiting if necessary.

        Returns:
            Time waited in seconds (0 if no wait needed)
        """
        async with self._lock:
            now = time.monotonic()

            # Remove old requests outside the window
            cutoff = now - self.window_seconds
            self._request_times = [t for t in self._request_times if t > cutoff]

            # Check if we need to wait
            if len(self._request_times) >= self.max_requests:
                # Calculate wait time until oldest request expires
                oldest = self._request_times[0]
                wait_time = (oldest + self.window_seconds) - now + 0.01  # Small buffer
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    now = time.monotonic()
                    # Clean up again after waiting
                    cutoff = now - self.window_seconds
                    self._request_times = [t for t in self._request_times if t > cutoff]

            # Record this request
            self._request_times.append(now)
            return max(0, wait_time) if 'wait_time' in dir() else 0

    @property
    def available_requests(self) -> int:
        """Get number of available requests in current window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        active = [t for t in self._request_times if t > cutoff]
        return max(0, self.max_requests - len(active))


@dataclass
class LLMResponse:
    """
    Standardized response from any LLM provider.

    All LLM clients return this consistent format.
    """
    text: str
    tokens_used: int
    model: str
    finish_reason: str = "stop"
    latency_ms: int = 0

    # Cost tracking (in USD)
    cost_usd: float = 0.0

    # Provider-specific metadata
    raw_response: Optional[dict] = None


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All providers implement:
    - generate(): Main generation method
    - health_check(): Verify connection/availability

    Includes:
    - Rate limiting per provider
    - Exponential backoff retry
    - Cost calculation
    """

    provider_name: str = "base"

    def __init__(self, config: dict):
        """
        Initialize LLM client.

        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0

        # Rate limiting
        rate_limit = config.get(
            'rate_limit_rpm',
            DEFAULT_RATE_LIMITS.get(self.provider_name, 60)
        )
        self._rate_limiter = RateLimiter(rate_limit)

        # Retry configuration
        self._max_retries = config.get('max_retries', 3)
        self._base_delay = config.get('retry_base_delay', 1.0)
        self._max_delay = config.get('retry_max_delay', 30.0)

    async def generate_with_retry(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate with rate limiting and exponential backoff retry.

        Args:
            model: Model identifier
            system_prompt: System/persona prompt
            user_message: User query/task
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response tokens

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                # Wait for rate limit
                wait_time = await self._rate_limiter.acquire()
                if wait_time > 0:
                    logger.debug(f"{self.provider_name}: Rate limited, waited {wait_time:.2f}s")

                # Make the actual request
                response = await self.generate(
                    model=model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Calculate and update cost
                cost = self._calculate_cost(model, response.tokens_used)
                response.cost_usd = cost
                self._update_stats(response.tokens_used, cost)

                return response

            except Exception as e:
                last_exception = e
                if attempt < self._max_retries:
                    # Exponential backoff
                    delay = min(
                        self._base_delay * (2 ** attempt),
                        self._max_delay
                    )
                    logger.warning(
                        f"{self.provider_name}: Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"{self.provider_name}: All {self._max_retries + 1} attempts failed"
                    )

        raise last_exception

    @abstractmethod
    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            model: Model identifier
            system_prompt: System/persona prompt
            user_message: User query/task
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response tokens

        Returns:
            LLMResponse with generated text and metadata
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM provider is available.

        Returns:
            True if healthy, False otherwise
        """
        pass

    def _calculate_cost(self, model: str, tokens_used: int) -> float:
        """
        Calculate cost for a request.

        Args:
            model: Model name
            tokens_used: Total tokens used (input + output)

        Returns:
            Cost in USD
        """
        costs = MODEL_COSTS.get(model)
        if not costs:
            return 0.0

        # Approximate split: 70% input, 30% output
        input_tokens = int(tokens_used * 0.7)
        output_tokens = tokens_used - input_tokens

        input_cost = (input_tokens / 1000) * costs['input']
        output_cost = (output_tokens / 1000) * costs['output']

        return input_cost + output_cost

    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "provider": self.provider_name,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "available_rate_limit": self._rate_limiter.available_requests,
        }

    def _update_stats(self, tokens: int, cost: float):
        """Update internal statistics."""
        self._total_requests += 1
        self._total_tokens += tokens
        self._total_cost += cost
