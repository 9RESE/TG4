"""
Base Agent Class - Abstract interface for all TripleGain agents.

All agents inherit from BaseAgent and implement:
- process(): Main analysis/decision method
- get_output_schema(): JSON schema for output validation
- store_output(): Persist output to database

Includes:
- Thread-safe output caching with asyncio locks
- TTL-based cache expiration
- Database-backed persistence
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, Any, TYPE_CHECKING
import json
import time
import uuid

if TYPE_CHECKING:
    from ..data.market_snapshot import MarketSnapshot
    from ..llm.prompt_builder import PromptBuilder, PortfolioContext

logger = logging.getLogger(__name__)


@dataclass
class AgentOutput:
    """
    Base class for all agent outputs.

    Every agent output includes:
    - Identification (agent, timestamp, symbol)
    - Confidence score
    - Reasoning (for explainability)
    - Performance metadata (latency, tokens)
    """
    agent_name: str
    timestamp: datetime
    symbol: str
    confidence: float
    reasoning: str

    # Performance metadata
    latency_ms: int = 0
    tokens_used: int = 0
    model_used: str = ""

    # Unique identifier
    output_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Decimal):
                result[key] = float(value)
            else:
                result[key] = value
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate output against constraints.

        Returns:
            tuple: (is_valid, list of error messages)
        """
        errors = []

        # Confidence must be in [0, 1]
        if not 0 <= self.confidence <= 1:
            errors.append(f"Confidence {self.confidence} not in [0, 1]")

        # Must have reasoning
        if not self.reasoning or len(self.reasoning) < 10:
            errors.append("Reasoning too short or missing")

        return len(errors) == 0, errors


class BaseAgent(ABC):
    """
    Abstract base class for all TripleGain agents.

    Agents process market data and produce structured outputs.
    All outputs are validated against JSON schemas and stored to database.

    Features:
    - Thread-safe output caching with asyncio locks
    - TTL-based cache expiration per symbol
    - Database-backed persistence
    """

    # Class attributes to be overridden by subclasses
    agent_name: str = "base"
    llm_tier: str = "tier1_local"
    model: str = ""

    def __init__(
        self,
        llm_client,
        prompt_builder: 'PromptBuilder',
        config: dict,
        db_pool=None
    ):
        """
        Initialize agent.

        Args:
            llm_client: LLM client for generation
            prompt_builder: Prompt builder for assembling prompts
            config: Agent configuration
            db_pool: Database pool for storing outputs
        """
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.config = config
        self.db = db_pool

        # Performance tracking
        self._total_invocations = 0
        self._total_latency_ms = 0
        self._total_tokens = 0

        # Last output for quick access (updated on each process() call)
        self._last_output: Optional[AgentOutput] = None

        # Thread-safe cache with TTL
        self._cache: dict[str, tuple[AgentOutput, datetime]] = {}
        self._cache_lock = asyncio.Lock()
        self._cache_ttl_seconds = config.get('cache_ttl_seconds', 300)

    @abstractmethod
    async def process(
        self,
        snapshot: 'MarketSnapshot',
        portfolio_context: Optional['PortfolioContext'] = None,
        **kwargs
    ) -> AgentOutput:
        """
        Process market data and produce output.

        Args:
            snapshot: Market data snapshot
            portfolio_context: Optional portfolio state
            **kwargs: Additional agent-specific inputs

        Returns:
            AgentOutput subclass with analysis results
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> dict:
        """
        Return JSON schema for output validation.

        Returns:
            dict: JSON schema
        """
        pass

    async def store_output(
        self,
        output: AgentOutput,
    ) -> None:
        """
        Store output to agent_outputs table and update cache.

        Args:
            output: Agent output to store
        """
        # Always update cache (thread-safe)
        await self.cache_output(output)

        if self.db is None:
            logger.debug(f"No database configured, skipping output storage for {self.agent_name}")
            return

        try:
            query = """
                INSERT INTO agent_outputs (
                    id, agent_name, timestamp, symbol, output_data,
                    confidence, latency_ms, tokens_used, model_used
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            await self.db.execute(
                query,
                output.output_id,
                self.agent_name,
                output.timestamp,
                output.symbol,
                output.to_json(),
                output.confidence,
                output.latency_ms,
                output.tokens_used,
                output.model_used,
            )
            logger.debug(f"Stored output {output.output_id} for {self.agent_name}")
        except Exception as e:
            logger.error(f"Failed to store output: {e}")

    async def get_latest_output(
        self,
        symbol: str,
        max_age_seconds: int = 300
    ) -> Optional[AgentOutput]:
        """
        Get most recent output for a symbol.

        Thread-safe with TTL-based cache expiration.

        Args:
            symbol: Trading symbol
            max_age_seconds: Maximum age of cached output

        Returns:
            Most recent AgentOutput or None
        """
        # Check in-memory cache first (thread-safe)
        async with self._cache_lock:
            if symbol in self._cache:
                output, cached_at = self._cache[symbol]
                age_seconds = (datetime.now(timezone.utc) - cached_at).total_seconds()
                if age_seconds <= max_age_seconds:
                    logger.debug(f"{self.agent_name}: Cache hit for {symbol} (age={age_seconds:.1f}s)")
                    return output
                else:
                    # Expired, remove from cache
                    del self._cache[symbol]

        # Check database if available
        if self.db is not None:
            try:
                query = """
                    SELECT output_data, timestamp
                    FROM agent_outputs
                    WHERE agent_name = $1 AND symbol = $2
                    AND timestamp > NOW() - INTERVAL '%s seconds'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """ % max_age_seconds

                row = await self.db.fetchrow(query, self.agent_name, symbol)
                if row:
                    output = self._parse_stored_output(row['output_data'])
                    if output:
                        # Update cache
                        async with self._cache_lock:
                            self._cache[symbol] = (output, row['timestamp'])
                        return output
            except Exception as e:
                logger.error(f"Failed to get latest output: {e}")

        return None

    async def cache_output(self, output: AgentOutput) -> None:
        """
        Cache output in thread-safe manner.

        Args:
            output: Agent output to cache
        """
        async with self._cache_lock:
            self._cache[output.symbol] = (output, datetime.now(timezone.utc))

    async def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached outputs.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        async with self._cache_lock:
            if symbol:
                self._cache.pop(symbol, None)
            else:
                self._cache.clear()

    def _parse_stored_output(self, output_json: str) -> Optional[AgentOutput]:
        """Parse stored JSON output back to AgentOutput."""
        # Subclasses should override this for proper deserialization
        try:
            data = json.loads(output_json)
            return AgentOutput(**data)
        except Exception as e:
            logger.error(f"Failed to parse stored output: {e}")
            return None

    async def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, int, int]:
        """
        Call the LLM and return response with metadata.

        Args:
            system_prompt: System prompt
            user_message: User message

        Returns:
            tuple: (response_text, latency_ms, tokens_used)
        """
        start_time = time.perf_counter()

        try:
            response = await self.llm.generate(
                model=self.model,
                system_prompt=system_prompt,
                user_message=user_message,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            tokens_used = response.tokens_used

            # Update stats
            self._total_invocations += 1
            self._total_latency_ms += latency_ms
            self._total_tokens += tokens_used

            return response.text, latency_ms, tokens_used

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"LLM call failed for {self.agent_name}: {e}")
            raise

    def get_stats(self) -> dict:
        """Get agent performance statistics."""
        return {
            "agent_name": self.agent_name,
            "total_invocations": self._total_invocations,
            "total_latency_ms": self._total_latency_ms,
            "average_latency_ms": (
                self._total_latency_ms / self._total_invocations
                if self._total_invocations > 0 else 0
            ),
            "total_tokens": self._total_tokens,
            "average_tokens": (
                self._total_tokens / self._total_invocations
                if self._total_invocations > 0 else 0
            ),
        }

    @property
    def last_output(self) -> Optional[AgentOutput]:
        """
        Get the most recent output from this agent.

        Returns:
            The last AgentOutput produced by process(), or None if not yet called
        """
        return self._last_output
