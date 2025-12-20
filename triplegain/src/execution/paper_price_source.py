"""
Paper Price Source - Price provider for paper trading.

Phase 6: Paper Trading Integration

Provides realistic prices for paper trading from multiple sources:
1. Live WebSocket feed (real-time prices, simulated execution)
2. Database cache (recent historical prices)
3. Mock prices (for testing)
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class PaperPriceSource:
    """
    Price source for paper trading.

    Options:
    1. live_feed: Real WebSocket prices from Kraken (most realistic)
    2. historical: Database cached prices from candle data
    3. mock: Static prices for testing

    The price source maintains a cache for fast lookups and
    integrates with WebSocket feeds for real-time updates.
    """

    def __init__(
        self,
        source_type: str = "live_feed",
        db_connection: Optional[Any] = None,
        websocket_feed: Optional[Any] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize price source.

        Args:
            source_type: "live_feed", "historical", or "mock"
            db_connection: Database connection for historical prices
            websocket_feed: WebSocket feed for live prices
            config: Optional configuration
        """
        self.source_type = source_type.lower()
        self.db = db_connection
        self.ws_feed = websocket_feed
        self.config = config or {}

        # Price cache for fast lookups
        self._cache: Dict[str, Decimal] = {}
        self._cache_time: Dict[str, datetime] = {}

        # Mock prices (fallback and for testing)
        self._mock_prices: Dict[str, Decimal] = {
            "BTC/USDT": Decimal("45000"),
            "XRP/USDT": Decimal("0.60"),
            "XRP/BTC": Decimal("0.000013"),
            "ETH/USDT": Decimal("2500"),
            "BTC/USD": Decimal("45000"),
            "XRP/USD": Decimal("0.60"),
            "ETH/USD": Decimal("2500"),
        }

        # Cache expiry (seconds)
        self._cache_max_age = self.config.get("price_cache_max_age_seconds", 60)

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._ws_updates = 0
        self._db_fetches = 0

        logger.info(f"PaperPriceSource initialized: source_type={source_type}")

    def get_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current price for a symbol.

        Tries sources in order:
        1. WebSocket feed (if source_type=live_feed)
        2. Cache (if fresh enough)
        3. Database (if source_type=historical)
        4. Mock prices (fallback)

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Current price as Decimal, or None if unavailable
        """
        symbol = symbol.upper()

        # Try WebSocket feed first (live_feed mode)
        if self.source_type == "live_feed" and self.ws_feed:
            try:
                price = self._get_ws_price(symbol)
                if price:
                    self._cache[symbol] = price
                    self._cache_time[symbol] = datetime.now(timezone.utc)
                    return price
            except Exception as e:
                logger.debug(f"WebSocket price fetch failed for {symbol}: {e}")

        # Check cache
        if symbol in self._cache:
            cache_age = (datetime.now(timezone.utc) - self._cache_time.get(symbol, datetime.min.replace(tzinfo=timezone.utc))).total_seconds()
            if cache_age < self._cache_max_age:
                self._cache_hits += 1
                return self._cache[symbol]

        self._cache_misses += 1

        # Try database (historical mode)
        if self.source_type == "historical" and self.db:
            price = self._get_db_price(symbol)
            if price:
                self._cache[symbol] = price
                self._cache_time[symbol] = datetime.now(timezone.utc)
                return price

        # Fallback to mock prices
        if symbol in self._mock_prices:
            logger.debug(f"Using mock price for {symbol}: {self._mock_prices[symbol]}")
            return self._mock_prices[symbol]

        logger.warning(f"No price available for {symbol}")
        return None

    async def get_price_async(self, symbol: str) -> Optional[Decimal]:
        """
        Get current price for a symbol (async version).

        CRITICAL-03: Properly async method that awaits database calls.
        Use this method in async contexts for proper database integration.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Current price as Decimal, or None if unavailable
        """
        symbol = symbol.upper()

        # Try WebSocket feed first (live_feed mode)
        if self.source_type == "live_feed" and self.ws_feed:
            try:
                price = self._get_ws_price(symbol)
                if price:
                    self._cache[symbol] = price
                    self._cache_time[symbol] = datetime.now(timezone.utc)
                    return price
            except Exception as e:
                logger.debug(f"WebSocket price fetch failed for {symbol}: {e}")

        # Check cache
        if symbol in self._cache:
            cache_age = (datetime.now(timezone.utc) - self._cache_time.get(symbol, datetime.min.replace(tzinfo=timezone.utc))).total_seconds()
            if cache_age < self._cache_max_age:
                self._cache_hits += 1
                return self._cache[symbol]

        self._cache_misses += 1

        # Try database (historical mode) - properly async
        if self.source_type == "historical" and self.db:
            price = await self._get_db_price_async(symbol)
            if price:
                self._cache[symbol] = price
                self._cache_time[symbol] = datetime.now(timezone.utc)
                return price

        # Fallback to mock prices
        if symbol in self._mock_prices:
            logger.debug(f"Using mock price for {symbol}: {self._mock_prices[symbol]}")
            return self._mock_prices[symbol]

        logger.warning(f"No price available for {symbol}")
        return None

    def _get_ws_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get price from WebSocket feed.

        Args:
            symbol: Trading pair

        Returns:
            Price from WebSocket, or None
        """
        if not self.ws_feed:
            return None

        # Try different attribute names for WebSocket feeds
        if hasattr(self.ws_feed, "get_last_price"):
            price = self.ws_feed.get_last_price(symbol)
        elif hasattr(self.ws_feed, "get_price"):
            price = self.ws_feed.get_price(symbol)
        elif hasattr(self.ws_feed, "prices"):
            price = self.ws_feed.prices.get(symbol)
        else:
            return None

        if price:
            self._ws_updates += 1
            return Decimal(str(price))
        return None

    async def _get_db_price_async(self, symbol: str) -> Optional[Decimal]:
        """
        Get most recent price from database (async version).

        CRITICAL-03: Properly async method for use in async context.

        Args:
            symbol: Trading pair

        Returns:
            Price from database, or None
        """
        if not self.db:
            return None

        try:
            query = """
                SELECT close FROM candles_1m
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = await self.db.fetchrow(query, symbol)

            if result:
                self._db_fetches += 1
                return Decimal(str(result["close"]))

        except Exception as e:
            logger.debug(f"Database price fetch failed for {symbol}: {e}")

        return None

    def _get_db_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get most recent price from database (sync version).

        Note: This sync version cannot properly fetch from async DB in async context.
        Use get_price_async() in async contexts for proper database integration.

        Args:
            symbol: Trading pair

        Returns:
            Price from database, or None (returns cached/mock in async context)
        """
        if not self.db:
            return None

        import asyncio

        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in async context - can't block, return cached or mock
                logger.debug(f"Sync _get_db_price called in async context for {symbol}, using cache/mock")
                return self._mock_prices.get(symbol)
            except RuntimeError:
                # Not in async context - safe to use sync methods
                pass

            # Not in async context, try to run synchronously
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self._get_db_price_async(symbol))
                    return result
                finally:
                    loop.close()
            except Exception as e:
                logger.debug(f"Sync database fetch failed for {symbol}: {e}")

        except Exception as e:
            logger.debug(f"Database price fetch failed for {symbol}: {e}")

        return None

    def update_price(
        self,
        symbol: str,
        price: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Update price cache (called by WebSocket handler).

        MEDIUM-02: Only updates if new timestamp is newer than cached timestamp.

        Args:
            symbol: Trading pair
            price: Current price
            timestamp: Optional timestamp of the price (uses now if not provided)

        Returns:
            True if cache was updated, False if skipped (stale price)
        """
        symbol = symbol.upper()
        now = timestamp or datetime.now(timezone.utc)

        # MEDIUM-02: Check if existing price is newer
        if symbol in self._cache_time:
            existing_time = self._cache_time[symbol]
            if existing_time > now:
                logger.debug(
                    f"Skipping stale price update for {symbol}: "
                    f"existing={existing_time}, incoming={now}"
                )
                return False

        self._cache[symbol] = price
        self._cache_time[symbol] = now
        self._ws_updates += 1
        return True

    def update_prices(
        self,
        prices: Dict[str, Decimal],
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Batch update price cache.

        MEDIUM-02: Only updates prices with newer timestamps.

        Args:
            prices: Dictionary of symbol -> price
            timestamp: Optional timestamp for all prices

        Returns:
            Number of prices updated
        """
        now = timestamp or datetime.now(timezone.utc)
        updated_count = 0

        for symbol, price in prices.items():
            if self.update_price(symbol, price, now):
                updated_count += 1

        return updated_count

    def set_mock_price(self, symbol: str, price: Decimal) -> None:
        """
        Set a mock price (for testing).

        Args:
            symbol: Trading pair
            price: Mock price
        """
        symbol = symbol.upper()
        self._mock_prices[symbol] = price
        # Also update cache for immediate effect
        self._cache[symbol] = price
        self._cache_time[symbol] = datetime.now(timezone.utc)

    def get_all_prices(self) -> Dict[str, Decimal]:
        """
        Get all cached prices.

        Returns:
            Dictionary of symbol -> price
        """
        return self._cache.copy()

    def get_mock_prices(self) -> Dict[str, Decimal]:
        """
        Get all mock prices.

        Returns:
            Dictionary of symbol -> mock price
        """
        return self._mock_prices.copy()

    def clear_cache(self) -> None:
        """Clear the price cache."""
        self._cache.clear()
        self._cache_time.clear()

    def get_stats(self) -> dict:
        """Get price source statistics."""
        return {
            "source_type": self.source_type,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "ws_updates": self._ws_updates,
            "db_fetches": self._db_fetches,
            "has_ws_feed": self.ws_feed is not None,
            "has_db": self.db is not None,
        }

    def is_price_fresh(self, symbol: str, max_age_seconds: int = 60) -> bool:
        """
        Check if cached price is fresh enough.

        Args:
            symbol: Trading pair
            max_age_seconds: Maximum acceptable age

        Returns:
            True if price exists and is fresh
        """
        symbol = symbol.upper()
        if symbol not in self._cache_time:
            return False

        age = (datetime.now(timezone.utc) - self._cache_time[symbol]).total_seconds()
        return age < max_age_seconds


class MockPriceSource(PaperPriceSource):
    """
    Mock price source for testing.

    Provides static prices that can be manipulated for test scenarios.
    """

    def __init__(self, initial_prices: Optional[Dict[str, Decimal]] = None):
        """
        Initialize mock price source.

        Args:
            initial_prices: Optional initial price dictionary
        """
        super().__init__(source_type="mock")

        if initial_prices:
            self._mock_prices.update(initial_prices)
            self._cache.update(initial_prices)
            now = datetime.now(timezone.utc)
            for symbol in initial_prices:
                self._cache_time[symbol] = now

    def simulate_price_change(
        self,
        symbol: str,
        change_pct: float,
    ) -> Decimal:
        """
        Simulate a price change for testing.

        Args:
            symbol: Trading pair
            change_pct: Percentage change (positive or negative)

        Returns:
            New price
        """
        symbol = symbol.upper()
        current = self._mock_prices.get(symbol, Decimal("100"))
        new_price = current * (Decimal("1") + Decimal(str(change_pct)) / Decimal("100"))
        self.set_mock_price(symbol, new_price)
        return new_price

    def simulate_flash_crash(self, symbol: str, crash_pct: float = 10.0) -> Decimal:
        """
        Simulate a flash crash for testing stop-loss triggers.

        Args:
            symbol: Trading pair
            crash_pct: Crash percentage (positive number)

        Returns:
            New (crashed) price
        """
        return self.simulate_price_change(symbol, -abs(crash_pct))

    def simulate_pump(self, symbol: str, pump_pct: float = 10.0) -> Decimal:
        """
        Simulate a price pump for testing take-profit triggers.

        Args:
            symbol: Trading pair
            pump_pct: Pump percentage (positive number)

        Returns:
            New (pumped) price
        """
        return self.simulate_price_change(symbol, abs(pump_pct))
