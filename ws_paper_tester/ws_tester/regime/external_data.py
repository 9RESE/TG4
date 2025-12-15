"""
External Data Fetcher

Fetches external market sentiment data from free APIs:
- Fear & Greed Index from Alternative.me
- BTC Dominance from CoinGecko

Features:
- Async HTTP requests with aiohttp
- 5-minute caching to reduce API calls
- Graceful fallback to cached/default values on errors
- Rate limiting with exponential backoff
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

import aiohttp

from .types import ExternalSentiment

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple rate limiter with exponential backoff.

    Tracks API calls per endpoint and implements backoff on failures.
    """

    def __init__(
        self,
        min_interval_seconds: float = 1.0,
        max_backoff_seconds: float = 300.0,
        backoff_factor: float = 2.0
    ):
        """
        Initialize rate limiter.

        Args:
            min_interval_seconds: Minimum time between requests (default 1s)
            max_backoff_seconds: Maximum backoff duration (default 5 minutes)
            backoff_factor: Multiplier for exponential backoff (default 2x)
        """
        self.min_interval = min_interval_seconds
        self.max_backoff = max_backoff_seconds
        self.backoff_factor = backoff_factor

        self._last_request: Dict[str, datetime] = {}
        self._failure_count: Dict[str, int] = {}
        self._backoff_until: Dict[str, datetime] = {}

    async def acquire(self, endpoint: str) -> bool:
        """
        Acquire permission to make a request.

        Blocks if rate limit would be exceeded.
        Returns False if in backoff period.

        Args:
            endpoint: API endpoint identifier

        Returns:
            True if request can proceed, False if blocked by backoff
        """
        now = datetime.now(timezone.utc)

        # Check if in backoff period
        backoff_until = self._backoff_until.get(endpoint)
        if backoff_until and now < backoff_until:
            wait_seconds = (backoff_until - now).total_seconds()
            logger.debug(f"Rate limiter: {endpoint} in backoff for {wait_seconds:.1f}s")
            return False

        # Check minimum interval
        last_request = self._last_request.get(endpoint)
        if last_request:
            elapsed = (now - last_request).total_seconds()
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                await asyncio.sleep(wait_time)

        self._last_request[endpoint] = datetime.now(timezone.utc)
        return True

    def record_success(self, endpoint: str) -> None:
        """Record successful request, reset failure count."""
        self._failure_count[endpoint] = 0
        self._backoff_until.pop(endpoint, None)

    def record_failure(self, endpoint: str) -> None:
        """
        Record failed request, calculate backoff.

        Uses exponential backoff: backoff = min_interval * (factor ^ failures)
        """
        failures = self._failure_count.get(endpoint, 0) + 1
        self._failure_count[endpoint] = failures

        # Calculate backoff duration
        backoff_seconds = min(
            self.min_interval * (self.backoff_factor ** failures),
            self.max_backoff
        )

        self._backoff_until[endpoint] = (
            datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
        )

        logger.warning(
            f"Rate limiter: {endpoint} failed ({failures}x), "
            f"backoff for {backoff_seconds:.1f}s"
        )

    def get_call_count(self, endpoint: str) -> int:
        """Get number of failures for an endpoint."""
        return self._failure_count.get(endpoint, 0)


class ExternalDataFetcher:
    """
    Fetch external market sentiment data with caching.

    Data sources:
    - Fear & Greed Index: https://api.alternative.me/fng/
    - BTC Dominance: https://api.coingecko.com/api/v3/global

    Both APIs are free and don't require authentication.
    """

    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

    CACHE_TTL = timedelta(minutes=5)
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self, enabled: bool = True):
        """
        Initialize the external data fetcher.

        Args:
            enabled: If False, always returns None (for testing/offline mode)
        """
        self.enabled = enabled
        self._cache: Optional[ExternalSentiment] = None
        self._cache_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

        # Rate limiter for API calls (1 request/second min, 5 minute max backoff)
        self._rate_limiter = RateLimiter(
            min_interval_seconds=1.0,
            max_backoff_seconds=300.0,
            backoff_factor=2.0
        )

    async def fetch(self) -> Optional[ExternalSentiment]:
        """
        Fetch external sentiment data with caching.

        Returns cached data if still fresh (within TTL).
        On error, returns stale cache if available, else None.

        Returns:
            ExternalSentiment data or None if unavailable
        """
        if not self.enabled:
            return None

        async with self._lock:
            now = datetime.now(timezone.utc)

            # Return cached if fresh
            if self._is_cache_valid(now):
                return self._cache

            # Fetch fresh data
            try:
                sentiment = await self._fetch_all()
                self._cache = sentiment
                self._cache_time = now
                logger.debug(
                    f"External data fetched: F&G={sentiment.fear_greed_value}, "
                    f"BTC.D={sentiment.btc_dominance:.1f}%"
                )
                return sentiment

            except asyncio.TimeoutError:
                logger.warning("External data fetch timeout")
                return self._get_fallback()

            except aiohttp.ClientError as e:
                logger.warning(f"External data fetch error: {e}")
                return self._get_fallback()

            except Exception as e:
                logger.error(f"Unexpected error fetching external data: {e}")
                return self._get_fallback()

    async def _fetch_all(self) -> ExternalSentiment:
        """
        Fetch all external data sources.

        Returns:
            ExternalSentiment with fresh data
        """
        # Fetch both in parallel
        fg_task = asyncio.create_task(self._fetch_fear_greed())
        btc_task = asyncio.create_task(self._fetch_btc_dominance())

        fear_greed = await fg_task
        btc_dominance = await btc_task

        return ExternalSentiment(
            fear_greed_value=fear_greed['value'],
            fear_greed_classification=fear_greed['classification'],
            btc_dominance=btc_dominance,
            last_updated=datetime.now(timezone.utc),
        )

    async def _fetch_fear_greed(self) -> dict:
        """
        Fetch Fear & Greed Index from Alternative.me.

        Returns:
            Dict with 'value' (int 0-100) and 'classification' (str)

        Raises:
            Exception: If rate limited or API error
        """
        endpoint = "fear_greed"

        # Check rate limiter
        if not await self._rate_limiter.acquire(endpoint):
            raise Exception(f"Rate limited: {endpoint} in backoff period")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.FEAR_GREED_URL,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    self._rate_limiter.record_success(endpoint)
                    return {
                        'value': int(data['data'][0]['value']),
                        'classification': data['data'][0]['value_classification'],
                    }
        except Exception as e:
            self._rate_limiter.record_failure(endpoint)
            raise

    async def _fetch_btc_dominance(self) -> float:
        """
        Fetch BTC dominance from CoinGecko.

        Returns:
            BTC dominance as percentage (e.g., 56.5)

        Raises:
            Exception: If rate limited or API error
        """
        endpoint = "btc_dominance"

        # Check rate limiter
        if not await self._rate_limiter.acquire(endpoint):
            raise Exception(f"Rate limited: {endpoint} in backoff period")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.COINGECKO_GLOBAL_URL,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    self._rate_limiter.record_success(endpoint)
                    return data['data']['market_cap_percentage']['btc']
        except Exception as e:
            self._rate_limiter.record_failure(endpoint)
            raise

    def _is_cache_valid(self, now: datetime) -> bool:
        """
        Check if cached data is still valid.

        Args:
            now: Current time

        Returns:
            True if cache exists and is within TTL
        """
        return (
            self._cache is not None and
            self._cache_time is not None and
            now - self._cache_time < self.CACHE_TTL
        )

    def _get_fallback(self) -> Optional[ExternalSentiment]:
        """
        Get fallback data (stale cache or default).

        Returns:
            Stale cached data if available, else default neutral values
        """
        if self._cache is not None:
            logger.info("Using stale cached external data")
            return self._cache

        # Return neutral defaults
        logger.info("Using default external data (neutral)")
        return ExternalSentiment(
            fear_greed_value=50,
            fear_greed_classification="Neutral",
            btc_dominance=55.0,  # Approximate historical average
            last_updated=datetime.now(timezone.utc),
        )

    def get_cached(self) -> Optional[ExternalSentiment]:
        """
        Get cached data without fetching.

        Useful for synchronous access when async isn't needed.

        Returns:
            Cached ExternalSentiment or None
        """
        return self._cache

    def invalidate_cache(self) -> None:
        """Force cache invalidation."""
        self._cache = None
        self._cache_time = None

    @property
    def cache_age_seconds(self) -> Optional[float]:
        """
        Get the age of the cache in seconds.

        Returns:
            Seconds since last fetch, or None if no cache
        """
        if self._cache_time is None:
            return None
        return (datetime.now(timezone.utc) - self._cache_time).total_seconds()
