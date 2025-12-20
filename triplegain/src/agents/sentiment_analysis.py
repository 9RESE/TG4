"""
Sentiment Analysis Agent - Dual-model sentiment from Grok and GPT.

Uses Grok (xAI) and GPT (OpenAI) with web search capabilities to gather
real-time market sentiment from news and social media.

Invoked: Every 30 minutes
Latency target: <20s total
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any, TYPE_CHECKING

from .base_agent import BaseAgent, AgentOutput

if TYPE_CHECKING:
    from ..data.market_snapshot import MarketSnapshot
    from ..llm.prompt_builder import PromptBuilder, PortfolioContext
    from ..llm.clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker State
# =============================================================================

@dataclass
class CircuitBreakerState:
    """
    Circuit breaker state for provider failure handling.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests are blocked
    - HALF_OPEN: Testing if provider has recovered
    """
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    half_open_attempts: int = 0

    def record_success(self) -> None:
        """Reset on success."""
        self.consecutive_failures = 0
        self.is_open = False
        self.half_open_attempts = 0

    def record_failure(self, failure_threshold: int, cooldown_seconds: int) -> None:
        """Record failure and potentially open circuit."""
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.consecutive_failures >= failure_threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker OPENED after {self.consecutive_failures} failures. "
                f"Will retry after {cooldown_seconds}s cooldown."
            )

    def should_allow_request(self, cooldown_seconds: int) -> bool:
        """Check if request should be allowed."""
        if not self.is_open:
            return True

        # Check if cooldown has passed
        if self.last_failure_time:
            elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
            if elapsed >= cooldown_seconds:
                # Enter half-open state
                self.half_open_attempts += 1
                logger.info(f"Circuit breaker HALF-OPEN, attempt {self.half_open_attempts}")
                return True

        return False


class SentimentBias(Enum):
    """Sentiment bias levels."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"

    @classmethod
    def from_score(cls, score: float) -> 'SentimentBias':
        """
        Convert numeric score to sentiment bias.

        Args:
            score: Score from -1 (very bearish) to 1 (very bullish)

        Returns:
            SentimentBias enum value
        """
        if score >= 0.6:
            return cls.VERY_BULLISH
        elif score >= 0.2:
            return cls.BULLISH
        elif score >= -0.2:
            return cls.NEUTRAL
        elif score >= -0.6:
            return cls.BEARISH
        else:
            return cls.VERY_BEARISH


class FearGreedLevel(Enum):
    """Fear/Greed market assessment."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"

    @classmethod
    def from_score(cls, score: float) -> 'FearGreedLevel':
        """
        Convert score to fear/greed level.

        Uses same boundary logic as SentimentBias (>= for consistency):
        - EXTREME_GREED: score >= 0.6
        - GREED: score >= 0.2 and < 0.6
        - NEUTRAL: score >= -0.2 and < 0.2
        - FEAR: score >= -0.6 and < -0.2
        - EXTREME_FEAR: score < -0.6

        Args:
            score: Score from -1 to 1

        Returns:
            FearGreedLevel enum value
        """
        # Use consistent >= boundaries matching SentimentBias
        if score >= 0.6:
            return cls.EXTREME_GREED
        elif score >= 0.2:
            return cls.GREED
        elif score >= -0.2:
            return cls.NEUTRAL
        elif score >= -0.6:
            return cls.FEAR
        else:
            return cls.EXTREME_FEAR


class EventImpact(Enum):
    """Impact level for key events."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EventSignificance(Enum):
    """Significance level for key events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class KeyEvent:
    """Significant market event detected in sentiment analysis."""
    event: str
    impact: EventImpact
    significance: EventSignificance
    source: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "event": self.event,
            "impact": self.impact.value,
            "significance": self.significance.value,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'KeyEvent':
        """Create from dictionary."""
        return cls(
            event=data.get("event", "Unknown event"),
            impact=EventImpact(data.get("impact", "neutral")),
            significance=EventSignificance(data.get("significance", "medium")),
            source=data.get("source", "unknown"),
        )


@dataclass
class ProviderResult:
    """Result from a single sentiment provider (Grok or GPT)."""
    provider: str  # "grok" or "gpt"
    model: str
    bias: SentimentBias
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    events: list[KeyEvent] = field(default_factory=list)
    narratives: list[str] = field(default_factory=list)
    reasoning: str = ""
    latency_ms: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "bias": self.bias.value,
            "score": self.score,
            "confidence": self.confidence,
            "events": [e.to_dict() for e in self.events],
            "narratives": self.narratives,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "success": self.success,
            "error": self.error,
        }


# JSON Schema for sentiment output validation
SENTIMENT_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["timestamp", "symbol", "bias", "confidence", "sources"],
    "properties": {
        "timestamp": {"type": "string", "format": "date-time"},
        "symbol": {"type": "string"},
        "bias": {
            "type": "string",
            "enum": ["very_bullish", "bullish", "neutral", "bearish", "very_bearish"]
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "sentiment_scores": {
            "type": "object",
            "properties": {
                "social_media": {"type": "number", "minimum": -1, "maximum": 1},
                "news": {"type": "number", "minimum": -1, "maximum": 1},
                "overall": {"type": "number", "minimum": -1, "maximum": 1}
            }
        },
        "key_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {"type": "string"},
                    "impact": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "significance": {"type": "string", "enum": ["low", "medium", "high"]},
                    "source": {"type": "string"}
                }
            },
            "maxItems": 5
        },
        "market_narratives": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3
        },
        "fear_greed_assessment": {
            "type": "string",
            "enum": ["extreme_fear", "fear", "neutral", "greed", "extreme_greed"]
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string"},
                    "query_type": {"type": "string"},
                    "result_count": {"type": "integer"}
                }
            }
        },
        "reasoning": {"type": "string", "maxLength": 500}
    }
}


@dataclass
class SentimentOutput(AgentOutput):
    """Sentiment Analysis Agent output."""
    # Core sentiment
    bias: SentimentBias = SentimentBias.NEUTRAL

    # Sentiment scores
    social_score: float = 0.0  # -1 to 1 (from Grok/Twitter)
    news_score: float = 0.0  # -1 to 1 (from GPT/news)
    overall_score: float = 0.0  # -1 to 1

    # Provider analysis/reasoning (for trading decision context)
    social_analysis: str = ""  # Grok's Twitter/X sentiment analysis
    news_analysis: str = ""  # GPT's news sentiment analysis

    # Fear/Greed assessment
    fear_greed: FearGreedLevel = FearGreedLevel.NEUTRAL

    # Key events (max 5)
    key_events: list[KeyEvent] = field(default_factory=list)

    # Market narratives (max 3)
    market_narratives: list[str] = field(default_factory=list)

    # Provider availability
    grok_available: bool = False
    gpt_available: bool = False

    # Provider results for debugging
    provider_results: list[ProviderResult] = field(default_factory=list)

    # Total cost tracking
    total_cost_usd: float = 0.0

    def validate(self) -> tuple[bool, list[str]]:
        """Validate sentiment-specific constraints."""
        is_valid, errors = super().validate()

        # Scores must be in valid range
        for score_name, score_val in [
            ("social_score", self.social_score),
            ("news_score", self.news_score),
            ("overall_score", self.overall_score),
        ]:
            if not -1 <= score_val <= 1:
                errors.append(f"{score_name} {score_val} not in [-1, 1]")
                is_valid = False

        # Max 5 key events
        if len(self.key_events) > 5:
            errors.append(f"Too many key_events: {len(self.key_events)} > 5")
            is_valid = False

        # Max 3 narratives
        if len(self.market_narratives) > 3:
            errors.append(f"Too many narratives: {len(self.market_narratives)} > 3")
            is_valid = False

        return is_valid, errors

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "bias": self.bias.value,
            "social_score": self.social_score,
            "news_score": self.news_score,
            "overall_score": self.overall_score,
            "social_analysis": self.social_analysis,
            "news_analysis": self.news_analysis,
            "fear_greed": self.fear_greed.value,
            "key_events": [e.to_dict() for e in self.key_events],
            "market_narratives": self.market_narratives,
            "grok_available": self.grok_available,
            "gpt_available": self.gpt_available,
            "total_cost_usd": self.total_cost_usd,
        })
        return base


# Prompt templates
GROK_SENTIMENT_PROMPT = """Analyze current market sentiment for {asset} cryptocurrency.

TASK: Search the web and X/Twitter for recent sentiment about {asset}.

WEB SEARCH - Look for:
- Breaking news (last 24 hours) about {asset}
- Price analysis articles
- Regulatory developments
- Institutional adoption news
- Technical developments and upgrades
- Partnership announcements

SOCIAL/X ANALYSIS - Look for:
- Trending discussions about {asset}
- Sentiment from crypto influencers
- Official project announcements
- Community sentiment and engagement
- Any FUD or controversies

Based on your findings, provide a sentiment analysis.

RESPOND WITH JSON ONLY:
{{
  "bias": "very_bullish|bullish|neutral|bearish|very_bearish",
  "score": <number from -1.0 (very bearish) to 1.0 (very bullish)>,
  "confidence": <number from 0.0 to 1.0>,
  "key_events": [
    {{
      "event": "Brief description of significant event",
      "impact": "positive|negative|neutral",
      "significance": "low|medium|high",
      "source": "Source name"
    }}
  ],
  "narratives": ["Current market narrative 1", "Narrative 2"],
  "reasoning": "Brief explanation of your assessment (max 200 chars)"
}}

Important:
- Include up to 5 key events, prioritize by significance
- Include up to 3 current market narratives
- Be objective and evidence-based
- If no significant news, return neutral sentiment with low confidence"""

GPT_SENTIMENT_PROMPT = """Analyze current market sentiment for {asset} cryptocurrency.

TASK: Provide fundamental and news-based sentiment analysis for {asset}.

ANALYZE BASED ON YOUR KNOWLEDGE:
- Recent regulatory developments affecting {asset}
- Known institutional investment trends
- Fundamental analysis and market conditions
- Macro economic factors affecting crypto markets
- Known upcoming events (upgrades, halvings, token unlocks)
- General market structure and sentiment indicators

Focus on FUNDAMENTAL FACTORS rather than short-term price action.

NOTE: This analysis is based on model knowledge up to training cutoff.
For real-time news, Grok social analysis provides complementary data.

RESPOND WITH JSON ONLY:
{{
  "bias": "very_bullish|bullish|neutral|bearish|very_bearish",
  "score": <number from -1.0 (very bearish) to 1.0 (very bullish)>,
  "confidence": <number from 0.0 to 1.0>,
  "key_events": [
    {{
      "event": "Brief description of significant factor",
      "impact": "positive|negative|neutral",
      "significance": "low|medium|high",
      "source": "Source name"
    }}
  ],
  "narratives": ["Current market narrative 1", "Narrative 2"],
  "reasoning": "Brief explanation of your assessment (max 200 chars)"
}}

Important:
- Include up to 5 key factors, prioritize by significance
- Include up to 3 current market narratives
- Be objective and evidence-based
- Acknowledge knowledge limitations with lower confidence
- If limited knowledge, return neutral sentiment with low confidence"""


class SentimentAnalysisAgent(BaseAgent):
    """
    Sentiment Analysis Agent using dual-model (Grok + GPT) approach.

    Queries both Grok (for social/Twitter) and GPT (for news) in parallel,
    then aggregates results using weighted scoring.

    Features:
    - Parallel provider queries for speed
    - Graceful degradation on provider failure
    - Configurable weighting between providers
    - Event deduplication
    - Fear/Greed assessment
    """

    agent_name = "sentiment_analysis"
    llm_tier = "tier2_api"

    def __init__(
        self,
        llm_clients: dict,  # {"grok": XAIClient, "gpt": OpenAIClient}
        prompt_builder: 'PromptBuilder',
        config: dict,
        db_pool=None,
    ):
        """
        Initialize SentimentAnalysisAgent.

        Args:
            llm_clients: Dictionary of LLM clients {"grok": XAIClient, "gpt": OpenAIClient}
            prompt_builder: Prompt builder
            config: Agent configuration
            db_pool: Database pool for output storage
        """
        # Use first available client for base class
        first_client = next(iter(llm_clients.values()), None)
        super().__init__(first_client, prompt_builder, config, db_pool)

        self.llm_clients = llm_clients

        # Provider configuration
        providers_config = config.get('providers', {})

        # Grok config
        grok_config = providers_config.get('grok', {})
        self.grok_enabled = grok_config.get('enabled', True)
        self.grok_model = grok_config.get('model', 'grok-2')
        self.grok_timeout_ms = grok_config.get('timeout_ms', 30000)
        grok_weight = grok_config.get('weight', {})
        self.grok_social_weight = grok_weight.get('social', 0.6)
        self.grok_news_weight = grok_weight.get('news', 0.4)

        # GPT config
        gpt_config = providers_config.get('gpt', {})
        self.gpt_enabled = gpt_config.get('enabled', True)
        self.gpt_model = gpt_config.get('model', 'gpt-4-turbo')
        self.gpt_timeout_ms = gpt_config.get('timeout_ms', 30000)
        gpt_weight = gpt_config.get('weight', {})
        self.gpt_social_weight = gpt_weight.get('social', 0.4)
        self.gpt_news_weight = gpt_weight.get('news', 0.6)

        # Aggregation config
        agg_config = config.get('aggregation', {})
        self.min_providers = agg_config.get('min_providers', 1)
        self.min_confidence = agg_config.get('min_confidence', 0.3)

        # Output config
        output_config = config.get('output', {})
        self.max_events = output_config.get('max_events', 5)
        self.max_narratives = output_config.get('max_narratives', 3)

        # Retry config (loaded from grok config, applies to both)
        retry_config = grok_config.get('retry', {})
        self.max_retries = retry_config.get('max_attempts', 2)
        self.backoff_ms = retry_config.get('backoff_ms', 5000)

        # Circuit breaker config
        cb_config = config.get('circuit_breaker', {})
        self.cb_failure_threshold = cb_config.get('failure_threshold', 3)
        self.cb_cooldown_seconds = cb_config.get('cooldown_seconds', 300)  # 5 min default

        # Circuit breaker states (per provider)
        self._circuit_breakers: dict[str, CircuitBreakerState] = {
            "grok": CircuitBreakerState(),
            "gpt": CircuitBreakerState(),
        }

        # Rate limiting for refresh (requests per minute per user)
        self._rate_limit_rpm = config.get('rate_limit', {}).get('refresh_rpm', 5)

    async def process(
        self,
        snapshot: Optional['MarketSnapshot'] = None,
        portfolio_context: Optional['PortfolioContext'] = None,
        symbol: Optional[str] = None,
        include_twitter: bool = True,
        **kwargs
    ) -> SentimentOutput:
        """
        Analyze market sentiment using dual-model approach.

        Args:
            snapshot: Market data snapshot (optional, symbol can be passed directly)
            portfolio_context: Optional portfolio state
            symbol: Trading symbol (e.g., "BTC" or "BTC/USDT")
            include_twitter: Whether to request Twitter analysis from Grok

        Returns:
            SentimentOutput with aggregated sentiment analysis
        """
        # Determine symbol
        if symbol:
            asset = symbol.split('/')[0] if '/' in symbol else symbol
            full_symbol = symbol
        elif snapshot:
            asset = snapshot.symbol.split('/')[0] if '/' in snapshot.symbol else snapshot.symbol
            full_symbol = snapshot.symbol
        else:
            raise ValueError("Either snapshot or symbol must be provided")

        logger.debug(f"Sentiment Agent processing {asset}")

        # Query providers in parallel
        results = await self._query_providers(asset, include_twitter)

        # Aggregate results
        output = self._aggregate_results(full_symbol, results)

        # Validate output
        is_valid, validation_errors = output.validate()
        if not is_valid:
            logger.warning(f"Sentiment output validation issues: {validation_errors}")

        # Store output
        await self.store_output(output)
        await self._store_provider_responses(output)

        # Cache for quick retrieval
        self._last_output = output

        logger.info(
            f"Sentiment Agent: {full_symbol} bias={output.bias.value} "
            f"confidence={output.confidence:.2f} "
            f"grok={output.grok_available} gpt={output.gpt_available} "
            f"latency={output.latency_ms}ms"
        )

        return output

    async def _query_providers(
        self,
        asset: str,
        include_twitter: bool = True,
    ) -> list[ProviderResult]:
        """
        Query all enabled providers in parallel.

        Args:
            asset: Asset symbol (e.g., "BTC")
            include_twitter: Whether to include Twitter in Grok query

        Returns:
            List of ProviderResult from each provider
        """
        tasks = []

        if self.grok_enabled and "grok" in self.llm_clients:
            tasks.append(self._query_grok(asset, include_twitter))

        if self.gpt_enabled and "gpt" in self.llm_clients:
            tasks.append(self._query_gpt(asset))

        if not tasks:
            logger.warning("No sentiment providers available")
            return []

        # Run in parallel with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        provider_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Provider query failed with exception: {result}")
            elif isinstance(result, ProviderResult):
                provider_results.append(result)

        return provider_results

    async def _query_grok(
        self,
        asset: str,
        include_twitter: bool = True,
    ) -> ProviderResult:
        """
        Query Grok for sentiment with web/Twitter search.

        Implements:
        - Timeout enforcement (asyncio.wait_for)
        - Retry with exponential backoff
        - Circuit breaker pattern

        Args:
            asset: Asset symbol
            include_twitter: Whether to include Twitter search

        Returns:
            ProviderResult with Grok's analysis
        """
        provider = "grok"
        circuit_breaker = self._circuit_breakers[provider]

        # Check circuit breaker
        if not circuit_breaker.should_allow_request(self.cb_cooldown_seconds):
            logger.warning(f"Circuit breaker OPEN for {provider}, skipping query")
            return ProviderResult(
                provider=provider,
                model=self.grok_model,
                bias=SentimentBias.NEUTRAL,
                score=0.0,
                confidence=0.0,
                latency_ms=0,
                success=False,
                error="Circuit breaker open",
            )

        start_time = time.perf_counter()
        last_error = None

        # Retry with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff: backoff_ms * 2^(attempt-1)
                    wait_ms = self.backoff_ms * (2 ** (attempt - 1))
                    logger.info(f"Grok retry attempt {attempt + 1}, waiting {wait_ms}ms")
                    await asyncio.sleep(wait_ms / 1000)

                client = self.llm_clients.get("grok")
                if not client:
                    raise RuntimeError("Grok client not available")

                # Build prompt
                prompt = GROK_SENTIMENT_PROMPT.format(asset=asset)
                system_prompt = "You are a cryptocurrency market analyst with access to real-time web and social media data. Respond only with valid JSON."

                # Call Grok with search - wrapped in timeout
                timeout_seconds = self.grok_timeout_ms / 1000

                if hasattr(client, 'generate_with_search'):
                    response = await asyncio.wait_for(
                        client.generate_with_search(
                            model=self.grok_model,
                            system_prompt=system_prompt,
                            user_message=prompt,
                            search_enabled=include_twitter,
                            temperature=0.3,
                            max_tokens=2048,
                        ),
                        timeout=timeout_seconds,
                    )
                else:
                    response = await asyncio.wait_for(
                        client.generate(
                            model=self.grok_model,
                            system_prompt=system_prompt,
                            user_message=prompt,
                            temperature=0.3,
                            max_tokens=2048,
                        ),
                        timeout=timeout_seconds,
                    )

                latency_ms = int((time.perf_counter() - start_time) * 1000)

                # Parse response
                parsed = self._parse_provider_response(response.text, provider)

                # Success - reset circuit breaker
                circuit_breaker.record_success()

                return ProviderResult(
                    provider=provider,
                    model=self.grok_model,
                    bias=SentimentBias(parsed.get("bias", "neutral")),
                    score=parsed.get("score", 0.0),
                    confidence=parsed.get("confidence", 0.5),
                    events=[KeyEvent.from_dict(e) for e in parsed.get("key_events", [])[:self.max_events]],
                    narratives=parsed.get("narratives", [])[:self.max_narratives],
                    reasoning=parsed.get("reasoning", "")[:500],
                    latency_ms=latency_ms,
                    tokens_used=response.tokens_used,
                    cost_usd=response.cost_usd,
                    success=True,
                    raw_response=parsed,
                )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.grok_timeout_ms}ms"
                logger.warning(f"Grok query timeout (attempt {attempt + 1}/{self.max_retries + 1})")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Grok query failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")

        # All retries failed
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        circuit_breaker.record_failure(self.cb_failure_threshold, self.cb_cooldown_seconds)
        logger.error(f"Grok query failed after {self.max_retries + 1} attempts: {last_error}")

        return ProviderResult(
            provider=provider,
            model=self.grok_model,
            bias=SentimentBias.NEUTRAL,
            score=0.0,
            confidence=0.0,
            latency_ms=latency_ms,
            success=False,
            error=last_error,
        )

    async def _query_gpt(self, asset: str) -> ProviderResult:
        """
        Query GPT for sentiment with web search.

        Implements:
        - Timeout enforcement (asyncio.wait_for)
        - Retry with exponential backoff
        - Circuit breaker pattern

        NOTE: For actual web search, use gpt-4o-search-preview model or
        OpenAI's Responses API with web_search tool. Standard GPT-4 models
        do not have web search capability and will use training data only.

        Args:
            asset: Asset symbol

        Returns:
            ProviderResult with GPT's analysis
        """
        provider = "gpt"
        circuit_breaker = self._circuit_breakers[provider]

        # Check circuit breaker
        if not circuit_breaker.should_allow_request(self.cb_cooldown_seconds):
            logger.warning(f"Circuit breaker OPEN for {provider}, skipping query")
            return ProviderResult(
                provider=provider,
                model=self.gpt_model,
                bias=SentimentBias.NEUTRAL,
                score=0.0,
                confidence=0.0,
                latency_ms=0,
                success=False,
                error="Circuit breaker open",
            )

        start_time = time.perf_counter()
        last_error = None

        # Retry with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff: backoff_ms * 2^(attempt-1)
                    wait_ms = self.backoff_ms * (2 ** (attempt - 1))
                    logger.info(f"GPT retry attempt {attempt + 1}, waiting {wait_ms}ms")
                    await asyncio.sleep(wait_ms / 1000)

                client = self.llm_clients.get("gpt")
                if not client:
                    raise RuntimeError("GPT client not available")

                # Build prompt
                prompt = GPT_SENTIMENT_PROMPT.format(asset=asset)
                system_prompt = (
                    "You are a cryptocurrency market analyst. "
                    "Provide objective sentiment analysis based on your knowledge. "
                    "Respond only with valid JSON."
                )

                # Call GPT with timeout
                # NOTE: For web search, use gpt-4o-search-preview or Responses API
                timeout_seconds = self.gpt_timeout_ms / 1000

                response = await asyncio.wait_for(
                    client.generate(
                        model=self.gpt_model,
                        system_prompt=system_prompt,
                        user_message=prompt,
                        temperature=0.3,
                        max_tokens=2048,
                    ),
                    timeout=timeout_seconds,
                )

                latency_ms = int((time.perf_counter() - start_time) * 1000)

                # Parse response
                parsed = self._parse_provider_response(response.text, provider)

                # Success - reset circuit breaker
                circuit_breaker.record_success()

                return ProviderResult(
                    provider=provider,
                    model=self.gpt_model,
                    bias=SentimentBias(parsed.get("bias", "neutral")),
                    score=parsed.get("score", 0.0),
                    confidence=parsed.get("confidence", 0.5),
                    events=[KeyEvent.from_dict(e) for e in parsed.get("key_events", [])[:self.max_events]],
                    narratives=parsed.get("narratives", [])[:self.max_narratives],
                    reasoning=parsed.get("reasoning", "")[:500],
                    latency_ms=latency_ms,
                    tokens_used=response.tokens_used,
                    cost_usd=response.cost_usd,
                    success=True,
                    raw_response=parsed,
                )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.gpt_timeout_ms}ms"
                logger.warning(f"GPT query timeout (attempt {attempt + 1}/{self.max_retries + 1})")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"GPT query failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")

        # All retries failed
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        circuit_breaker.record_failure(self.cb_failure_threshold, self.cb_cooldown_seconds)
        logger.error(f"GPT query failed after {self.max_retries + 1} attempts: {last_error}")

        return ProviderResult(
            provider=provider,
            model=self.gpt_model,
            bias=SentimentBias.NEUTRAL,
            score=0.0,
            confidence=0.0,
            latency_ms=latency_ms,
            success=False,
            error=last_error,
        )

    def _parse_provider_response(self, response_text: str, provider: str) -> dict:
        """
        Parse JSON response from provider.

        Args:
            response_text: Raw LLM response
            provider: Provider name for logging

        Returns:
            Parsed dictionary with normalized values
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response_text.strip())

            return self._normalize_parsed_output(parsed)

        except json.JSONDecodeError as e:
            logger.warning(f"{provider} JSON decode error: {e}")
            return self._get_fallback_response()

    def _normalize_parsed_output(self, parsed: dict) -> dict:
        """Normalize and validate parsed output."""
        # Normalize bias
        bias = str(parsed.get("bias", "neutral")).lower()
        valid_biases = ["very_bullish", "bullish", "neutral", "bearish", "very_bearish"]
        if bias not in valid_biases:
            bias = "neutral"
        parsed["bias"] = bias

        # Clamp score
        score = parsed.get("score", 0.0)
        try:
            score = float(score)
            parsed["score"] = max(-1.0, min(1.0, score))
        except (ValueError, TypeError):
            parsed["score"] = 0.0

        # Clamp confidence
        confidence = parsed.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            parsed["confidence"] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            parsed["confidence"] = 0.5

        # Normalize key_events
        events = parsed.get("key_events", [])
        if not isinstance(events, list):
            events = []
        normalized_events = []
        for event in events[:5]:
            if isinstance(event, dict):
                normalized_events.append({
                    "event": str(event.get("event", "Unknown"))[:200],
                    "impact": event.get("impact", "neutral") if event.get("impact") in ["positive", "negative", "neutral"] else "neutral",
                    "significance": event.get("significance", "medium") if event.get("significance") in ["low", "medium", "high"] else "medium",
                    "source": str(event.get("source", "unknown"))[:100],
                })
        parsed["key_events"] = normalized_events

        # Normalize narratives
        narratives = parsed.get("narratives", [])
        if not isinstance(narratives, list):
            narratives = []
        parsed["narratives"] = [str(n)[:200] for n in narratives[:3]]

        # Truncate reasoning
        reasoning = parsed.get("reasoning", "")
        parsed["reasoning"] = str(reasoning)[:500]

        return parsed

    def _get_fallback_response(self) -> dict:
        """Get fallback response when parsing fails."""
        return {
            "bias": "neutral",
            "score": 0.0,
            "confidence": 0.3,
            "key_events": [],
            "narratives": [],
            "reasoning": "Failed to parse provider response",
        }

    def _aggregate_results(
        self,
        symbol: str,
        results: list[ProviderResult],
    ) -> SentimentOutput:
        """
        Aggregate results from multiple providers.

        Grok provides social/Twitter sentiment, GPT provides news sentiment.
        These are separate data sources passed independently to trading LLMs.

        Args:
            symbol: Trading symbol
            results: List of provider results

        Returns:
            Aggregated SentimentOutput
        """
        timestamp = datetime.now(timezone.utc)

        # Filter successful results
        successful = [r for r in results if r.success]

        if not successful:
            # Both failed - return neutral with low confidence
            logger.warning("All sentiment providers failed, returning neutral")
            return SentimentOutput(
                agent_name=self.agent_name,
                timestamp=timestamp,
                symbol=symbol,
                confidence=self.min_confidence,
                reasoning="All sentiment providers failed",
                latency_ms=sum(r.latency_ms for r in results),
                bias=SentimentBias.NEUTRAL,
                fear_greed=FearGreedLevel.NEUTRAL,
                grok_available=False,
                gpt_available=False,
                provider_results=results,
                total_cost_usd=sum(r.cost_usd for r in results),
            )

        # Get provider results - each measures different data
        grok_result = next((r for r in successful if r.provider == "grok"), None)
        gpt_result = next((r for r in successful if r.provider == "gpt"), None)

        # Social score comes from Grok (Twitter/X analysis)
        # News score comes from GPT (news/fundamental analysis)
        social_score = grok_result.score if grok_result else 0.0
        news_score = gpt_result.score if gpt_result else 0.0

        # Calculate overall score using configured weights
        # Weights determine relative importance of social vs news sentiment
        # Default: Grok social=0.6, GPT news=0.6 (each provider's primary strength)
        if grok_result and gpt_result:
            # Both providers available - use weighted average
            # Social weight from Grok config, news weight from GPT config
            total_weight = self.grok_social_weight + self.gpt_news_weight
            overall_score = (
                (social_score * self.grok_social_weight) +
                (news_score * self.gpt_news_weight)
            ) / total_weight
            logger.debug(
                f"Weighted aggregation: social={social_score:.2f}*{self.grok_social_weight} + "
                f"news={news_score:.2f}*{self.gpt_news_weight} = {overall_score:.2f}"
            )
        elif grok_result:
            # Only Grok available - use social score
            overall_score = social_score
        elif gpt_result:
            # Only GPT available - use news score
            overall_score = news_score
        else:
            overall_score = 0.0

        # Confidence is average of provider confidences
        base_confidence = sum(r.confidence for r in successful) / len(successful)
        final_confidence = max(self.min_confidence, base_confidence)

        # Determine final bias from overall score
        final_bias = SentimentBias.from_score(overall_score)

        # Determine fear/greed
        fear_greed = FearGreedLevel.from_score(overall_score)

        # Aggregate events with deduplication
        all_events = []
        for r in successful:
            all_events.extend(r.events)
        deduped_events = self._deduplicate_events(all_events)[:self.max_events]

        # Aggregate narratives with deduplication
        all_narratives = []
        for r in successful:
            all_narratives.extend(r.narratives)
        deduped_narratives = list(dict.fromkeys(all_narratives))[:self.max_narratives]

        # Extract provider analysis for trading decision context
        social_analysis = grok_result.reasoning if grok_result else ""
        news_analysis = gpt_result.reasoning if gpt_result else ""

        # Build combined reasoning summary
        reasoning_parts = []
        if social_analysis:
            reasoning_parts.append(f"Social: {social_analysis[:200]}")
        if news_analysis:
            reasoning_parts.append(f"News: {news_analysis[:200]}")
        combined_reasoning = " | ".join(reasoning_parts)[:500]

        # Calculate totals
        total_latency = max(r.latency_ms for r in results)  # Parallel, so take max
        total_tokens = sum(r.tokens_used for r in results)
        total_cost = sum(r.cost_usd for r in results)

        return SentimentOutput(
            agent_name=self.agent_name,
            timestamp=timestamp,
            symbol=symbol,
            confidence=final_confidence,
            reasoning=combined_reasoning,
            latency_ms=total_latency,
            tokens_used=total_tokens,
            model_used=f"grok:{self.grok_model}+gpt:{self.gpt_model}",
            bias=final_bias,
            social_score=social_score,
            news_score=news_score,
            overall_score=overall_score,
            social_analysis=social_analysis,
            news_analysis=news_analysis,
            fear_greed=fear_greed,
            key_events=deduped_events,
            market_narratives=deduped_narratives,
            grok_available=grok_result is not None,
            gpt_available=gpt_result is not None,
            provider_results=results,
            total_cost_usd=total_cost,
        )

    def _deduplicate_events(self, events: list[KeyEvent]) -> list[KeyEvent]:
        """
        Remove duplicate or similar events.

        Args:
            events: List of key events

        Returns:
            Deduplicated list, prioritized by significance
        """
        if not events:
            return []

        # Sort by significance (high > medium > low)
        significance_order = {"high": 0, "medium": 1, "low": 2}
        sorted_events = sorted(
            events,
            key=lambda e: significance_order.get(e.significance.value, 2)
        )

        # Deduplicate by event text similarity
        seen = set()
        unique = []
        for event in sorted_events:
            # Create a normalized key for comparison
            key = event.event.lower().strip()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(event)

        return unique

    async def _store_provider_responses(self, output: SentimentOutput) -> None:
        """
        Store individual provider responses to database.

        Args:
            output: SentimentOutput with provider_results
        """
        if not self.db:
            return

        try:
            for result in output.provider_results:
                query = """
                    INSERT INTO sentiment_provider_responses (
                        sentiment_output_id, provider, model,
                        raw_response, parsed_bias, parsed_score, parsed_confidence,
                        latency_ms, success, error_message, timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """
                await self.db.execute(
                    query,
                    output.output_id,
                    result.provider,
                    result.model,
                    json.dumps(result.raw_response) if result.raw_response else None,
                    result.bias.value,
                    result.score,
                    result.confidence,
                    result.latency_ms,
                    result.success,
                    result.error,
                    output.timestamp,
                )
        except Exception as e:
            logger.error(f"Failed to store provider responses: {e}")

    async def store_output(self, output: SentimentOutput) -> None:
        """
        Store sentiment output to sentiment_outputs table.

        Args:
            output: SentimentOutput to store
        """
        # Update cache
        await self.cache_output(output)

        if not self.db:
            logger.debug("No database configured, skipping output storage")
            return

        try:
            query = """
                INSERT INTO sentiment_outputs (
                    id, timestamp, symbol,
                    bias, confidence, social_score, news_score, overall_score,
                    social_analysis, news_analysis,
                    fear_greed, grok_available, gpt_available,
                    key_events, market_narratives, reasoning, total_latency_ms, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            """
            await self.db.execute(
                query,
                output.output_id,
                output.timestamp,
                output.symbol,
                output.bias.value,
                output.confidence,
                output.social_score,
                output.news_score,
                output.overall_score,
                output.social_analysis,  # Provider analysis text
                output.news_analysis,    # Provider analysis text
                output.fear_greed.value,
                output.grok_available,
                output.gpt_available,
                json.dumps([e.to_dict() for e in output.key_events]),
                json.dumps(output.market_narratives),
                output.reasoning,
                output.latency_ms,
                datetime.now(timezone.utc),
            )
            logger.debug(f"Stored sentiment output {output.output_id}")
        except Exception as e:
            logger.error(f"Failed to store sentiment output: {e}")

    def get_output_schema(self) -> dict:
        """Return JSON schema for validation."""
        return SENTIMENT_OUTPUT_SCHEMA

    def _parse_stored_output(self, output_json: str) -> Optional[SentimentOutput]:
        """Parse stored JSON output back to SentimentOutput."""
        try:
            data = json.loads(output_json)
            return SentimentOutput(
                agent_name=data.get("agent_name", self.agent_name),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                symbol=data["symbol"],
                confidence=data["confidence"],
                reasoning=data.get("reasoning", ""),
                latency_ms=data.get("latency_ms", 0),
                tokens_used=data.get("tokens_used", 0),
                model_used=data.get("model_used", ""),
                output_id=data.get("output_id", ""),
                bias=SentimentBias(data.get("bias", "neutral")),
                social_score=data.get("social_score", 0.0),
                news_score=data.get("news_score", 0.0),
                overall_score=data.get("overall_score", 0.0),
                social_analysis=data.get("social_analysis", ""),
                news_analysis=data.get("news_analysis", ""),
                fear_greed=FearGreedLevel(data.get("fear_greed", "neutral")),
                key_events=[KeyEvent.from_dict(e) for e in data.get("key_events", [])],
                market_narratives=data.get("market_narratives", []),
                grok_available=data.get("grok_available", False),
                gpt_available=data.get("gpt_available", False),
                total_cost_usd=data.get("total_cost_usd", 0.0),
            )
        except Exception as e:
            logger.error(f"Failed to parse stored sentiment output: {e}")
            return None
