"""
External Data Fetcher

Fetches external market sentiment data from free APIs:
- Fear & Greed Index from Alternative.me
- BTC Dominance from CoinGecko

Features:
- Async HTTP requests with aiohttp
- 5-minute caching to reduce API calls
- Graceful fallback to cached/default values on errors
- Rate limit awareness
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

from .types import ExternalSentiment

logger = logging.getLogger(__name__)


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
            now = datetime.utcnow()

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
            last_updated=datetime.utcnow(),
        )

    async def _fetch_fear_greed(self) -> dict:
        """
        Fetch Fear & Greed Index from Alternative.me.

        Returns:
            Dict with 'value' (int 0-100) and 'classification' (str)
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.FEAR_GREED_URL,
                timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

                return {
                    'value': int(data['data'][0]['value']),
                    'classification': data['data'][0]['value_classification'],
                }

    async def _fetch_btc_dominance(self) -> float:
        """
        Fetch BTC dominance from CoinGecko.

        Returns:
            BTC dominance as percentage (e.g., 56.5)
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.COINGECKO_GLOBAL_URL,
                timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

                return data['data']['market_cap_percentage']['btc']

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
            last_updated=datetime.utcnow(),
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
        return (datetime.utcnow() - self._cache_time).total_seconds()
