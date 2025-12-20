"""
Sentiment API Routes - Endpoints for sentiment analysis.

Phase 7 Endpoints:
- GET /api/v1/sentiment/{symbol} - Latest sentiment for symbol
- GET /api/v1/sentiment/{symbol}/history - Historical sentiment
- POST /api/v1/sentiment/{symbol}/refresh - Force sentiment refresh (rate limited)
- GET /api/v1/sentiment/all - Latest for all symbols

Security:
- All endpoints require authentication
- Refresh endpoint has rate limiting (5 requests per minute per user)
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional, Any


# =============================================================================
# Simple In-Memory Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter for API endpoints.

    Uses a sliding window approach to limit requests per user.
    For production, consider using Redis-based rate limiting.
    """

    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[datetime]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        """
        Check if request is allowed for user.

        Args:
            user_id: User identifier

        Returns:
            True if allowed, False if rate limited
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Clean old requests
        self._requests[user_id] = [
            ts for ts in self._requests[user_id]
            if ts > cutoff
        ]

        # Check limit
        if len(self._requests[user_id]) >= self.max_requests:
            return False

        # Record new request
        self._requests[user_id].append(now)
        return True

    def get_retry_after(self, user_id: str) -> int:
        """
        Get seconds until rate limit resets.

        Args:
            user_id: User identifier

        Returns:
            Seconds until oldest request expires
        """
        if not self._requests[user_id]:
            return 0

        oldest = min(self._requests[user_id])
        reset_at = oldest + timedelta(seconds=self.window_seconds)
        remaining = (reset_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(remaining))


# Rate limiter for refresh endpoint (5 requests per minute per user)
_refresh_rate_limiter = RateLimiter(max_requests=5, window_seconds=60)

try:
    from fastapi import APIRouter, HTTPException, Query, Depends, Request
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None
    BaseModel = object

from .validation import validate_symbol_or_raise
from .security import (
    get_current_user,
    User,
    log_security_event,
    SecurityEventType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

if FASTAPI_AVAILABLE:
    class KeyEventResponse(BaseModel):
        """Key event in sentiment analysis."""
        event: str = Field(..., description="Event description")
        impact: str = Field(..., description="Impact: positive, negative, neutral")
        significance: str = Field(..., description="Significance: low, medium, high")
        source: str = Field(..., description="Source of the event")

    class SentimentResponse(BaseModel):
        """Sentiment analysis response."""
        timestamp: datetime = Field(..., description="Analysis timestamp")
        symbol: str = Field(..., description="Trading symbol")
        bias: str = Field(..., description="Sentiment bias: very_bullish to very_bearish")
        confidence: float = Field(..., ge=0, le=1, description="Confidence score")
        social_score: float = Field(..., ge=-1, le=1, description="Social media sentiment")
        news_score: float = Field(..., ge=-1, le=1, description="News sentiment")
        overall_score: float = Field(..., ge=-1, le=1, description="Overall sentiment")
        social_analysis: str = Field("", description="Grok's Twitter/X sentiment analysis")
        news_analysis: str = Field("", description="GPT's news sentiment analysis")
        fear_greed: str = Field(..., description="Fear/Greed assessment")
        key_events: list[KeyEventResponse] = Field(default_factory=list)
        market_narratives: list[str] = Field(default_factory=list)
        grok_available: bool = Field(False, description="Grok provider succeeded (social sentiment)")
        gpt_available: bool = Field(False, description="GPT provider succeeded (news sentiment)")
        reasoning: str = Field("", description="Combined reasoning for the sentiment")

    class SentimentHistoryParams(BaseModel):
        """Query parameters for sentiment history."""
        hours: int = Field(24, ge=1, le=168, description="Hours of history (1-168)")
        limit: int = Field(48, ge=1, le=500, description="Max results")


# =============================================================================
# Router Factory
# =============================================================================

def create_sentiment_router(
    sentiment_agent=None,
    db_pool=None,
) -> 'APIRouter':
    """
    Create sentiment router with injected dependencies.

    Args:
        sentiment_agent: SentimentAnalysisAgent instance
        db_pool: Database connection pool

    Returns:
        FastAPI router with sentiment endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")

    router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])

    # =========================================================================
    # STATIC ROUTES FIRST - Must be defined before parameterized routes
    # =========================================================================
    # FastAPI matches routes in order, so /all and /stats must come before
    # /{symbol} to prevent "all" or "stats" being interpreted as symbols.

    # -------------------------------------------------------------------------
    # GET /api/v1/sentiment/all
    # -------------------------------------------------------------------------
    @router.get(
        "/all",
        response_model=dict[str, SentimentResponse],
        summary="Get all latest sentiments",
        description="Returns latest sentiment for all configured symbols.",
    )
    async def get_all_sentiments(
        current_user: User = Depends(get_current_user),
    ) -> dict[str, SentimentResponse]:
        """Get latest sentiment for all symbols."""
        if not db_pool:
            raise HTTPException(
                status_code=503,
                detail="Database not available",
            )

        try:
            # Use the latest_sentiment view
            query = """
                SELECT * FROM latest_sentiment
                ORDER BY symbol
            """
            rows = await db_pool.fetch(query)
            return {row['symbol']: _row_to_response(row) for row in rows}
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve sentiments: {e}",
            )

    # -------------------------------------------------------------------------
    # GET /api/v1/sentiment/stats
    # -------------------------------------------------------------------------
    @router.get(
        "/stats",
        response_model=dict,
        summary="Get provider statistics",
        description="Returns performance statistics for sentiment providers.",
    )
    async def get_provider_stats(
        current_user: User = Depends(get_current_user),
    ) -> dict:
        """Get provider performance statistics."""
        if not db_pool:
            raise HTTPException(
                status_code=503,
                detail="Database not available",
            )

        try:
            query = """
                SELECT * FROM sentiment_provider_stats
            """
            rows = await db_pool.fetch(query)
            return {
                "providers": [dict(row) for row in rows],
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve provider stats: {e}",
            )

    # =========================================================================
    # PARAMETERIZED ROUTES - After static routes
    # =========================================================================

    # -------------------------------------------------------------------------
    # GET /api/v1/sentiment/{symbol}
    # -------------------------------------------------------------------------
    @router.get(
        "/{symbol}",
        response_model=SentimentResponse,
        summary="Get latest sentiment",
        description="Returns the most recent sentiment analysis for a symbol.",
    )
    async def get_sentiment(
        symbol: str,
        current_user: User = Depends(get_current_user),
    ) -> SentimentResponse:
        """Get latest sentiment for a symbol."""
        validate_symbol_or_raise(symbol)

        # Try to get from cache first
        if sentiment_agent:
            output = await sentiment_agent.get_latest_output(
                symbol,
                max_age_seconds=1800,  # 30 minutes
            )
            if output:
                return _output_to_response(output)

        # Try database
        if db_pool:
            try:
                query = """
                    SELECT * FROM sentiment_outputs
                    WHERE symbol = $1
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                row = await db_pool.fetchrow(query, symbol)
                if row:
                    return _row_to_response(row)
            except Exception as e:
                logger.error(f"Database query failed: {e}")

        raise HTTPException(
            status_code=404,
            detail=f"No sentiment data found for {symbol}",
        )

    # -------------------------------------------------------------------------
    # GET /api/v1/sentiment/{symbol}/history
    # -------------------------------------------------------------------------
    @router.get(
        "/{symbol}/history",
        response_model=list[SentimentResponse],
        summary="Get sentiment history",
        description="Returns historical sentiment for a symbol.",
    )
    async def get_sentiment_history(
        symbol: str,
        hours: int = Query(24, ge=1, le=168, description="Hours of history"),
        limit: int = Query(48, ge=1, le=500, description="Max results"),
        current_user: User = Depends(get_current_user),
    ) -> list[SentimentResponse]:
        """Get sentiment history for a symbol."""
        validate_symbol_or_raise(symbol)

        if not db_pool:
            raise HTTPException(
                status_code=503,
                detail="Database not available",
            )

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            query = """
                SELECT * FROM sentiment_outputs
                WHERE symbol = $1 AND timestamp > $2
                ORDER BY timestamp DESC
                LIMIT $3
            """
            rows = await db_pool.fetch(query, symbol, cutoff, limit)
            return [_row_to_response(row) for row in rows]
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve sentiment history: {e}",
            )

    # -------------------------------------------------------------------------
    # POST /api/v1/sentiment/{symbol}/refresh
    # -------------------------------------------------------------------------
    @router.post(
        "/{symbol}/refresh",
        response_model=SentimentResponse,
        summary="Force sentiment refresh",
        description="Triggers a fresh sentiment analysis for a symbol. Rate limited to 5 requests per minute.",
    )
    async def refresh_sentiment(
        request: Request,
        symbol: str,
        include_twitter: bool = Query(True, description="Include Twitter analysis"),
        current_user: User = Depends(get_current_user),
    ) -> SentimentResponse:
        """Force a fresh sentiment analysis (rate limited)."""
        validate_symbol_or_raise(symbol)

        # Check rate limit (5 requests per minute per user)
        if not _refresh_rate_limiter.is_allowed(current_user.user_id):
            retry_after = _refresh_rate_limiter.get_retry_after(current_user.user_id)
            log_security_event(
                request,
                SecurityEventType.RATE_LIMIT,
                user_id=current_user.user_id,
                details={"action": "sentiment_refresh", "symbol": symbol, "retry_after": retry_after},
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        if not sentiment_agent:
            raise HTTPException(
                status_code=503,
                detail="Sentiment agent not available",
            )

        # Log the refresh request
        log_security_event(
            request,
            SecurityEventType.API_ACCESS,
            user_id=current_user.user_id,
            details={"action": "sentiment_refresh", "symbol": symbol},
        )

        try:
            output = await sentiment_agent.process(
                symbol=symbol,
                include_twitter=include_twitter,
            )
            return _output_to_response(output)
        except Exception as e:
            logger.error(f"Sentiment refresh failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Sentiment analysis failed: {e}",
            )

    return router


# =============================================================================
# Helper Functions
# =============================================================================

def _output_to_response(output) -> 'SentimentResponse':
    """Convert SentimentOutput to API response."""
    return SentimentResponse(
        timestamp=output.timestamp,
        symbol=output.symbol,
        bias=output.bias.value,
        confidence=output.confidence,
        social_score=output.social_score,
        news_score=output.news_score,
        overall_score=output.overall_score,
        social_analysis=output.social_analysis,
        news_analysis=output.news_analysis,
        fear_greed=output.fear_greed.value,
        key_events=[
            KeyEventResponse(
                event=e.event,
                impact=e.impact.value,
                significance=e.significance.value,
                source=e.source,
            )
            for e in output.key_events
        ],
        market_narratives=output.market_narratives,
        grok_available=output.grok_available,
        gpt_available=output.gpt_available,
        reasoning=output.reasoning,
    )


def _row_to_response(row) -> 'SentimentResponse':
    """Convert database row to API response."""
    # Parse JSON fields
    key_events_data = row.get('key_events') or []
    if isinstance(key_events_data, str):
        key_events_data = json.loads(key_events_data)

    narratives_data = row.get('market_narratives') or []
    if isinstance(narratives_data, str):
        narratives_data = json.loads(narratives_data)

    return SentimentResponse(
        timestamp=row['timestamp'],
        symbol=row['symbol'],
        bias=row['bias'],
        confidence=float(row['confidence']),
        social_score=float(row.get('social_score') or 0),
        news_score=float(row.get('news_score') or 0),
        overall_score=float(row.get('overall_score') or 0),
        social_analysis=row.get('social_analysis') or '',
        news_analysis=row.get('news_analysis') or '',
        fear_greed=row.get('fear_greed') or 'neutral',
        key_events=[
            KeyEventResponse(
                event=e.get('event', ''),
                impact=e.get('impact', 'neutral'),
                significance=e.get('significance', 'medium'),
                source=e.get('source', 'unknown'),
            )
            for e in key_events_data
        ],
        market_narratives=narratives_data,
        grok_available=row.get('grok_available', False),
        gpt_available=row.get('gpt_available', False),
        reasoning=row.get('reasoning') or '',
    )
