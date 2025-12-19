"""
Order Execution Manager - Order lifecycle management with Kraken API.

NOT an LLM agent - purely rule-based execution.

Features:
- Order placement with market/limit orders
- Order monitoring and status updates
- Contingent orders (stop loss, take profit)
- Retry logic with exponential backoff
- Rate limiting compliance with token bucket
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..orchestration.message_bus import MessageBus, MessageTopic
    from ..risk.rules_engine import TradeProposal

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API call throttling.

    Implements a simple token bucket algorithm that:
    - Starts with a full bucket of tokens
    - Removes tokens on each request
    - Refills tokens at a steady rate
    - Blocks when bucket is empty
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Refill tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate

            # Wait for tokens to be available
            await asyncio.sleep(wait_time)

            # Update after waiting
            self.tokens = 0  # All tokens consumed
            return wait_time

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (approximate)."""
        now = time.monotonic()
        elapsed = now - self.last_update
        return min(self.capacity, self.tokens + elapsed * self.rate)


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = "pending"              # Created locally, not sent
    OPEN = "open"                    # Submitted to exchange
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    ERROR = "error"


class OrderType(Enum):
    """Order types supported."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"
    STOP_LOSS_LIMIT = "stop-loss-limit"
    TAKE_PROFIT_LIMIT = "take-profit-limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    external_id: Optional[str] = None  # Kraken order ID (txid)
    filled_size: Decimal = Decimal(0)
    filled_price: Optional[Decimal] = None
    leverage: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None

    # Related orders
    parent_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "size": str(self.size),
            "price": str(self.price) if self.price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "status": self.status.value,
            "external_id": self.external_id,
            "filled_size": str(self.filled_size),
            "filled_price": str(self.filled_price) if self.filled_price else None,
            "leverage": self.leverage,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
            "parent_order_id": self.parent_order_id,
            "stop_loss_order_id": self.stop_loss_order_id,
            "take_profit_order_id": self.take_profit_order_id,
        }


@dataclass
class ExecutionResult:
    """Result of trade execution attempt."""
    success: bool
    order: Optional[Order] = None
    position_id: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "order": self.order.to_dict() if self.order else None,
            "position_id": self.position_id,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
        }


# Symbol mapping: Internal -> Kraken
SYMBOL_MAP = {
    "BTC/USDT": "XBTUSDT",
    "XRP/USDT": "XRPUSDT",
    "XRP/BTC": "XRPXBT",
    "ETH/USDT": "ETHUSDT",
}

# Reverse mapping
KRAKEN_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}


class OrderExecutionManager:
    """
    Manages order execution on Kraken exchange.

    Handles:
    - Order placement (market, limit, stop-loss, take-profit)
    - Order monitoring and status updates
    - Contingent orders (stop-loss/take-profit after fill)
    - Retry logic with exponential backoff
    - Rate limiting compliance

    NOT an LLM agent - purely rule-based execution.
    """

    def __init__(
        self,
        kraken_client,
        message_bus: Optional['MessageBus'] = None,
        position_tracker=None,
        db_pool=None,
        config: Optional[dict] = None,
    ):
        """
        Initialize OrderExecutionManager.

        Args:
            kraken_client: Kraken API client
            message_bus: MessageBus for event publishing
            position_tracker: PositionTracker instance
            db_pool: Database pool for persistence
            config: Execution configuration
        """
        self.kraken = kraken_client
        self.bus = message_bus
        self.position_tracker = position_tracker
        self.db = db_pool
        self.config = config or {}

        # Configuration
        orders_config = self.config.get('orders', {})
        self._default_order_type = orders_config.get('default_type', 'limit')
        self._time_in_force = orders_config.get('time_in_force', 'GTC')
        self._max_retries = orders_config.get('max_retry_count', 3)
        self._retry_delay_seconds = orders_config.get('retry_delay_seconds', 5)

        # Rate limiting with token bucket
        rate_limit = self.config.get('kraken', {}).get('rate_limit', {})
        calls_per_minute = rate_limit.get('calls_per_minute', 60)
        order_calls_per_minute = rate_limit.get('order_calls_per_minute', 30)

        # Create rate limiters
        # General API calls: tokens refill at rate per second, capacity = burst limit
        self._api_rate_limiter = TokenBucketRateLimiter(
            rate=calls_per_minute / 60.0,  # tokens per second
            capacity=min(10, calls_per_minute),  # burst capacity
        )
        # Order-specific calls: more restrictive
        self._order_rate_limiter = TokenBucketRateLimiter(
            rate=order_calls_per_minute / 60.0,  # tokens per second
            capacity=min(5, order_calls_per_minute),  # burst capacity
        )

        # Order tracking
        self._open_orders: dict[str, Order] = {}
        self._order_history: list[Order] = []
        self._monitoring_tasks: dict[str, asyncio.Task] = {}

        # Statistics
        self._total_orders_placed = 0
        self._total_orders_filled = 0
        self._total_orders_cancelled = 0
        self._total_errors = 0

        # Locks for thread safety - separate locks to reduce contention
        self._lock = asyncio.Lock()  # For open orders
        self._history_lock = asyncio.Lock()  # For order history
        self._max_history_size = self.config.get('orders', {}).get('max_history_size', 1000)

    async def execute_trade(
        self,
        proposal: 'TradeProposal',
    ) -> ExecutionResult:
        """
        Execute a validated trade proposal.

        Args:
            proposal: Risk-validated trade proposal

        Returns:
            ExecutionResult with order/position details
        """
        start_time = time.perf_counter()

        # Validate proposal size
        if proposal.size_usd <= 0:
            logger.warning(f"Invalid trade size: {proposal.size_usd}")
            return ExecutionResult(
                success=False,
                error_message=f"Invalid trade size: {proposal.size_usd} (must be > 0)",
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        try:
            # Determine order type
            order_type = OrderType.LIMIT if proposal.entry_price else OrderType.MARKET

            # Calculate size in base currency
            size = await self._calculate_size(proposal)

            # Validate calculated size
            if size <= 0:
                return ExecutionResult(
                    success=False,
                    error_message=f"Invalid calculated order size: {size}",
                    execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                )

            # Create order
            order = Order(
                id=str(uuid.uuid4()),
                symbol=proposal.symbol,
                side=OrderSide.BUY if proposal.side == "buy" else OrderSide.SELL,
                order_type=order_type,
                size=size,
                price=Decimal(str(proposal.entry_price)) if proposal.entry_price else None,
                leverage=proposal.leverage,
            )

            # Place order on exchange
            success = await self._place_order(order)

            if not success:
                return ExecutionResult(
                    success=False,
                    order=order,
                    error_message=order.error_message or "Order placement failed",
                    execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                )

            # Track the order
            async with self._lock:
                self._open_orders[order.id] = order
                self._total_orders_placed += 1

            # Store to database
            await self._store_order(order)

            # Start monitoring
            self._monitoring_tasks[order.id] = asyncio.create_task(
                self._monitor_order(order, proposal)
            )

            # Publish execution event
            if self.bus:
                from ..orchestration.message_bus import MessageTopic, create_message, MessagePriority
                await self.bus.publish(create_message(
                    topic=MessageTopic.EXECUTION_EVENTS,
                    source="order_execution_manager",
                    payload={
                        "event_type": "order_placed",
                        "order_id": order.id,
                        "external_id": order.external_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "size": str(order.size),
                        "price": str(order.price) if order.price else None,
                    },
                ))

            return ExecutionResult(
                success=True,
                order=order,
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Trade execution failed: {e}", exc_info=True)
            self._total_errors += 1

            return ExecutionResult(
                success=False,
                error_message=str(e),
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

    async def _place_order(self, order: Order) -> bool:
        """
        Place order on Kraken exchange with retry logic and rate limiting.

        Args:
            order: Order to place

        Returns:
            True if order placed successfully
        """
        kraken_symbol = self._to_kraken_symbol(order.symbol)

        for attempt in range(self._max_retries):
            try:
                # Acquire rate limit tokens for order placement
                wait_time = await self._order_rate_limiter.acquire(1)
                if wait_time > 0:
                    logger.debug(f"Rate limited: waited {wait_time:.2f}s before order placement")

                # Build order params
                params = {
                    "pair": kraken_symbol,
                    "type": order.side.value,
                    "ordertype": order.order_type.value,
                    "volume": str(order.size),
                }

                if order.price:
                    params["price"] = str(order.price)

                if order.stop_price:
                    params["price2"] = str(order.stop_price)

                if order.leverage > 1:
                    params["leverage"] = str(order.leverage)

                # Place order
                if self.kraken:
                    result = await self.kraken.add_order(**params)

                    if result.get("error"):
                        error_msg = str(result["error"])
                        logger.warning(f"Kraken order error: {error_msg}")
                        order.error_message = error_msg

                        # Don't retry certain errors
                        if "Invalid" in error_msg or "insufficient" in error_msg.lower():
                            order.status = OrderStatus.ERROR
                            return False

                        # Retry with backoff
                        await asyncio.sleep(self._retry_delay_seconds * (2 ** attempt))
                        continue

                    # Extract order ID
                    txids = result.get("result", {}).get("txid", [])
                    if txids:
                        order.external_id = txids[0]
                        order.status = OrderStatus.OPEN
                        order.updated_at = datetime.now(timezone.utc)
                        return True
                else:
                    # Mock mode (no Kraken client)
                    order.external_id = f"mock_{uuid.uuid4().hex[:8]}"
                    order.status = OrderStatus.OPEN
                    order.updated_at = datetime.now(timezone.utc)
                    logger.info(f"Mock order placed: {order.id}")
                    return True

            except asyncio.TimeoutError:
                logger.warning(f"Order placement timeout (attempt {attempt + 1})")
                await asyncio.sleep(self._retry_delay_seconds * (2 ** attempt))

            except Exception as e:
                logger.error(f"Order placement error: {e}")
                order.error_message = str(e)
                await asyncio.sleep(self._retry_delay_seconds * (2 ** attempt))

        order.status = OrderStatus.ERROR
        return False

    async def _monitor_order(
        self,
        order: Order,
        proposal: 'TradeProposal',
    ) -> None:
        """
        Monitor order until filled, cancelled, or expired.

        Args:
            order: Order to monitor
            proposal: Original trade proposal (for contingent orders)
        """
        poll_interval = 5  # seconds
        max_wait_time = 3600  # 1 hour max wait
        start_time = time.time()

        while order.status in [OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
            if time.time() - start_time > max_wait_time:
                logger.warning(f"Order {order.id} monitoring timeout")
                break

            await asyncio.sleep(poll_interval)

            try:
                # Acquire rate limit tokens for API query
                await self._api_rate_limiter.acquire(1)

                # Query order status
                if self.kraken and order.external_id:
                    result = await self.kraken.query_orders(txid=order.external_id)

                    if result.get("error"):
                        logger.warning(f"Order query error: {result['error']}")
                        continue

                    order_info = result.get("result", {}).get(order.external_id, {})
                    kraken_status = order_info.get("status", "")

                    if kraken_status == "closed":
                        order.status = OrderStatus.FILLED
                        order.filled_size = Decimal(str(order_info.get("vol_exec", 0)))
                        order.filled_price = Decimal(str(order_info.get("price", 0)))
                        order.updated_at = datetime.now(timezone.utc)

                        # Handle fill
                        await self._handle_order_fill(order, proposal)

                    elif kraken_status == "canceled":
                        order.status = OrderStatus.CANCELLED
                        order.updated_at = datetime.now(timezone.utc)

                    elif kraken_status == "expired":
                        order.status = OrderStatus.EXPIRED
                        order.updated_at = datetime.now(timezone.utc)

                    elif kraken_status == "pending":
                        # Still open
                        pass

                else:
                    # Mock mode - simulate fill after short delay
                    await asyncio.sleep(2)
                    order.status = OrderStatus.FILLED
                    order.filled_size = order.size
                    order.filled_price = order.price or Decimal("45000")
                    order.updated_at = datetime.now(timezone.utc)
                    await self._handle_order_fill(order, proposal)

            except Exception as e:
                logger.error(f"Order monitoring error: {e}")

        # Cleanup - use separate locks to avoid race condition
        await self._update_order(order)

        # Remove from open orders
        async with self._lock:
            if order.id in self._open_orders:
                del self._open_orders[order.id]

        # Add to history with its own lock and size limit
        async with self._history_lock:
            self._order_history.append(order)
            # Cleanup old history to prevent memory growth
            if len(self._order_history) > self._max_history_size:
                # Keep only the most recent orders
                self._order_history = self._order_history[-self._max_history_size:]

    async def _handle_order_fill(
        self,
        order: Order,
        proposal: 'TradeProposal',
    ) -> None:
        """Handle order fill - create position and contingent orders."""
        logger.info(f"Order filled: {order.id} at {order.filled_price}")
        self._total_orders_filled += 1

        # Create position in tracker
        position_id = None
        if self.position_tracker:
            position = await self.position_tracker.open_position(
                symbol=order.symbol,
                side="long" if order.side == OrderSide.BUY else "short",
                size=order.filled_size,
                entry_price=order.filled_price or Decimal(0),
                leverage=order.leverage,
                order_id=order.id,
            )
            position_id = position.id if position else None

        # Place contingent orders (stop-loss, take-profit)
        if proposal.stop_loss:
            await self._place_stop_loss(order, proposal, position_id)

        if proposal.take_profit:
            await self._place_take_profit(order, proposal, position_id)

        # Publish fill event
        if self.bus:
            from ..orchestration.message_bus import MessageTopic, create_message, MessagePriority
            await self.bus.publish(create_message(
                topic=MessageTopic.EXECUTION_EVENTS,
                source="order_execution_manager",
                payload={
                    "event_type": "order_filled",
                    "order_id": order.id,
                    "external_id": order.external_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "size": str(order.filled_size),
                    "fill_price": str(order.filled_price),
                    "position_id": position_id,
                },
                priority=MessagePriority.HIGH,
            ))

    async def _place_stop_loss(
        self,
        parent_order: Order,
        proposal: 'TradeProposal',
        position_id: Optional[str],
    ) -> Optional[Order]:
        """Place stop-loss order after fill."""
        sl_order = Order(
            id=str(uuid.uuid4()),
            symbol=parent_order.symbol,
            side=OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY,
            order_type=OrderType.STOP_LOSS,
            size=parent_order.filled_size,
            stop_price=Decimal(str(proposal.stop_loss)),
            parent_order_id=parent_order.id,
        )

        success = await self._place_order(sl_order)

        if success:
            parent_order.stop_loss_order_id = sl_order.id
            async with self._lock:
                self._open_orders[sl_order.id] = sl_order
            await self._store_order(sl_order)
            logger.info(f"Stop-loss placed: {sl_order.id} at {proposal.stop_loss}")
            return sl_order

        return None

    async def _place_take_profit(
        self,
        parent_order: Order,
        proposal: 'TradeProposal',
        position_id: Optional[str],
    ) -> Optional[Order]:
        """Place take-profit order after fill."""
        tp_order = Order(
            id=str(uuid.uuid4()),
            symbol=parent_order.symbol,
            side=OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY,
            order_type=OrderType.TAKE_PROFIT,
            size=parent_order.filled_size,
            price=Decimal(str(proposal.take_profit)),
            parent_order_id=parent_order.id,
        )

        success = await self._place_order(tp_order)

        if success:
            parent_order.take_profit_order_id = tp_order.id
            async with self._lock:
                self._open_orders[tp_order.id] = tp_order
            await self._store_order(tp_order)
            logger.info(f"Take-profit placed: {tp_order.id} at {proposal.take_profit}")
            return tp_order

        return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Internal order ID

        Returns:
            True if cancelled successfully
        """
        async with self._lock:
            order = self._open_orders.get(order_id)

        if not order:
            logger.warning(f"Order not found: {order_id}")
            return False

        if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
            logger.warning(f"Cannot cancel order in status: {order.status.value}")
            return False

        try:
            # Acquire rate limit tokens for cancel operation
            await self._order_rate_limiter.acquire(1)

            if self.kraken and order.external_id:
                result = await self.kraken.cancel_order(txid=order.external_id)

                if result.get("error"):
                    logger.warning(f"Cancel order error: {result['error']}")
                    return False

            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now(timezone.utc)
            self._total_orders_cancelled += 1

            await self._update_order(order)

            # Publish cancel event
            if self.bus:
                from ..orchestration.message_bus import MessageTopic, create_message
                await self.bus.publish(create_message(
                    topic=MessageTopic.EXECUTION_EVENTS,
                    source="order_execution_manager",
                    payload={
                        "event_type": "order_cancelled",
                        "order_id": order.id,
                        "external_id": order.external_id,
                        "symbol": order.symbol,
                    },
                ))

            return True

        except Exception as e:
            logger.error(f"Cancel order failed: {e}")
            return False

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all open orders, optionally filtered by symbol."""
        async with self._lock:
            orders = list(self._open_orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        # Check open orders first
        async with self._lock:
            if order_id in self._open_orders:
                return self._open_orders[order_id]

        # Check history with separate lock
        async with self._history_lock:
            for order in self._order_history:
                if order.id == order_id:
                    return order

        return None

    async def sync_with_exchange(self) -> int:
        """
        Sync local order state with exchange.

        Returns:
            Number of orders synced
        """
        if not self.kraken:
            return 0

        try:
            # Acquire rate limit tokens for sync operation
            await self._api_rate_limiter.acquire(1)

            result = await self.kraken.open_orders()

            if result.get("error"):
                logger.warning(f"Sync error: {result['error']}")
                return 0

            exchange_orders = result.get("result", {}).get("open", {})
            synced = 0

            for txid, order_info in exchange_orders.items():
                # Check if we have this order tracked
                found = False
                async with self._lock:
                    for order in self._open_orders.values():
                        if order.external_id == txid:
                            found = True
                            break

                if not found:
                    # Unknown order - log it
                    logger.warning(f"Unknown exchange order: {txid}")

                synced += 1

            return synced

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return 0

    async def _calculate_size(self, proposal: 'TradeProposal') -> Decimal:
        """Calculate order size in base currency."""
        if proposal.entry_price and proposal.entry_price > 0:
            return Decimal(str(proposal.size_usd)) / Decimal(str(proposal.entry_price))
        return Decimal(str(proposal.size_usd))

    def _to_kraken_symbol(self, symbol: str) -> str:
        """Convert internal symbol to Kraken format."""
        return SYMBOL_MAP.get(symbol, symbol.replace("/", ""))

    def _from_kraken_symbol(self, kraken_symbol: str) -> str:
        """Convert Kraken symbol to internal format."""
        return KRAKEN_SYMBOL_MAP.get(kraken_symbol, kraken_symbol)

    async def _store_order(self, order: Order) -> None:
        """Store order to database."""
        if not self.db:
            return

        try:
            query = """
                INSERT INTO order_status_log (
                    order_id, external_id, status, timestamp, details
                ) VALUES ($1, $2, $3, $4, $5)
            """
            await self.db.execute(
                query,
                uuid.UUID(order.id),
                order.external_id,
                order.status.value,
                order.updated_at,
                order.to_dict(),
            )
        except Exception as e:
            logger.error(f"Failed to store order: {e}")

    async def _update_order(self, order: Order) -> None:
        """Update order status in database."""
        if not self.db:
            return

        try:
            query = """
                INSERT INTO order_status_log (
                    order_id, external_id, status, timestamp, details
                ) VALUES ($1, $2, $3, $4, $5)
            """
            await self.db.execute(
                query,
                uuid.UUID(order.id),
                order.external_id,
                order.status.value,
                order.updated_at,
                order.to_dict(),
            )
        except Exception as e:
            logger.error(f"Failed to update order: {e}")

    def get_stats(self) -> dict:
        """Get execution statistics."""
        return {
            "total_orders_placed": self._total_orders_placed,
            "total_orders_filled": self._total_orders_filled,
            "total_orders_cancelled": self._total_orders_cancelled,
            "total_errors": self._total_errors,
            "open_orders_count": len(self._open_orders),
            "history_count": len(self._order_history),
            "max_history_size": self._max_history_size,
            "api_rate_limit_tokens": self._api_rate_limiter.available_tokens,
            "order_rate_limit_tokens": self._order_rate_limiter.available_tokens,
        }
