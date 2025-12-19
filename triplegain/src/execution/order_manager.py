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

    # F10: Fee tracking
    fee_amount: Decimal = Decimal(0)
    fee_currency: str = ""

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
            "fee_amount": str(self.fee_amount),
            "fee_currency": self.fee_currency,
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

        # Position limits configuration
        limits_config = self.config.get('position_limits', {})
        self._max_positions_per_symbol = limits_config.get('max_per_symbol', 2)
        self._max_positions_total = limits_config.get('max_total', 5)

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

        # Check position limits (only for buy orders that open new positions)
        if proposal.side == "buy":
            limit_check = await self._check_position_limits(proposal.symbol)
            if not limit_check["allowed"]:
                logger.warning(f"Position limit exceeded: {limit_check['reason']}")
                return ExecutionResult(
                    success=False,
                    error_message=limit_check["reason"],
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

            # F11: Always store order for audit trail (success or failure)
            await self._store_order(order)

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

                # Handle price parameters based on order type
                # For stop-loss/take-profit: trigger price goes in 'price'
                # For limit variants: limit price in 'price', trigger in 'price2'
                if order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                    # Simple stop/TP orders: trigger price in 'price'
                    if order.stop_price:
                        params["price"] = str(order.stop_price)
                    elif order.price:
                        params["price"] = str(order.price)
                elif order.order_type in [OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
                    # Limit variants: limit price in 'price', trigger in 'price2'
                    if order.price:
                        params["price"] = str(order.price)
                    if order.stop_price:
                        params["price2"] = str(order.stop_price)
                else:
                    # Market/Limit orders
                    if order.price:
                        params["price"] = str(order.price)

                if order.leverage > 1:
                    params["leverage"] = str(order.leverage)

                # Place order
                if self.kraken:
                    result = await self.kraken.add_order(**params)

                    if result.get("error"):
                        error_msg = str(result["error"])
                        logger.warning(f"Kraken order error: {error_msg}")
                        order.error_message = error_msg

                        # F13: Consistent case-insensitive error checking
                        error_lower = error_msg.lower()
                        if "invalid" in error_lower or "insufficient" in error_lower:
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

                    vol_exec = Decimal(str(order_info.get("vol_exec", 0)))
                    vol_total = Decimal(str(order_info.get("vol", order.size)))

                    if kraken_status == "closed":
                        order.status = OrderStatus.FILLED
                        order.filled_size = vol_exec
                        order.filled_price = Decimal(str(order_info.get("price", 0)))
                        order.updated_at = datetime.now(timezone.utc)

                        # F10: Extract fee information from Kraken response
                        order.fee_amount = Decimal(str(order_info.get("fee", 0)))
                        order.fee_currency = order_info.get("fee_asset", "")

                        # Handle fill
                        await self._handle_order_fill(order, proposal)

                        # F16: Handle OCO (cancel sibling order)
                        if order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                            await self.handle_oco_fill(order)

                    elif kraken_status in ["open", "pending"]:
                        # Check for partial fill - new fills since last check
                        if vol_exec > order.filled_size:
                            old_filled = order.filled_size
                            order.filled_size = vol_exec
                            order.filled_price = Decimal(str(order_info.get("price", 0)))
                            order.updated_at = datetime.now(timezone.utc)

                            if vol_exec < vol_total:
                                # Partial fill detected
                                order.status = OrderStatus.PARTIALLY_FILLED
                                logger.info(
                                    f"Partial fill detected: {order.id} "
                                    f"{old_filled} -> {vol_exec} / {vol_total}"
                                )

                                # Handle partial fill (create partial position, optional SL/TP)
                                await self._handle_partial_fill(order, proposal, vol_exec, vol_total)
                            else:
                                # Fully filled now
                                order.status = OrderStatus.FILLED
                                await self._handle_order_fill(order, proposal)

                    elif kraken_status == "canceled":
                        order.status = OrderStatus.CANCELLED
                        order.updated_at = datetime.now(timezone.utc)

                        # Check if it was partially filled before cancellation
                        if vol_exec > 0:
                            logger.warning(
                                f"Order {order.id} cancelled with partial fill: {vol_exec} of {vol_total}"
                            )
                            # Still need to handle the partial fill that occurred
                            await self._handle_partial_fill(order, proposal, vol_exec, vol_total, is_final=True)

                    elif kraken_status == "expired":
                        order.status = OrderStatus.EXPIRED
                        order.updated_at = datetime.now(timezone.utc)

                        # Check if it was partially filled before expiry
                        if vol_exec > 0:
                            logger.warning(
                                f"Order {order.id} expired with partial fill: {vol_exec} of {vol_total}"
                            )
                            await self._handle_partial_fill(order, proposal, vol_exec, vol_total, is_final=True)

                else:
                    # Mock mode - simulate fill after short delay
                    await asyncio.sleep(2)
                    order.status = OrderStatus.FILLED
                    order.filled_size = order.size
                    # Use order price, position tracker price cache, or fallback
                    fill_price = order.price
                    if not fill_price and self.position_tracker:
                        fill_price = self.position_tracker._price_cache.get(order.symbol)
                    if not fill_price:
                        # Fallback to reasonable defaults for mock mode
                        mock_prices = {
                            "BTC/USDT": Decimal("45000"),
                            "XRP/USDT": Decimal("0.60"),
                            "XRP/BTC": Decimal("0.000013"),
                            "ETH/USDT": Decimal("2500"),
                        }
                        fill_price = mock_prices.get(order.symbol, Decimal("100"))
                    order.filled_price = fill_price
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
        """
        Handle order fill - create position and contingent orders.

        Implements best-effort atomicity:
        - Position is always created first
        - Contingent orders are placed after position creation
        - Failures are logged and alerted (not silently ignored)
        - Position remains even if contingent orders fail
        """
        logger.info(f"Order filled: {order.id} at {order.filled_price}")
        self._total_orders_filled += 1

        position = None
        position_id = None
        sl_order = None
        tp_order = None
        failed_contingent: list[str] = []

        try:
            # Step 1: Create position in tracker (required)
            if self.position_tracker:
                position = await self.position_tracker.open_position(
                    symbol=order.symbol,
                    side="long" if order.side == OrderSide.BUY else "short",
                    size=order.filled_size,
                    entry_price=order.filled_price or Decimal(0),
                    leverage=order.leverage,
                    order_id=order.id,
                )
                if position:
                    position_id = position.id
                else:
                    raise RuntimeError("Failed to create position for filled order")
            else:
                logger.warning("No position tracker - position will not be tracked")

            # Step 2: Place contingent orders and check for failures
            if proposal.stop_loss:
                sl_order = await self._place_stop_loss(order, proposal, position_id)
                if not sl_order:
                    failed_contingent.append("stop_loss")
                    logger.critical(
                        f"CRITICAL: Stop-loss placement failed for position {position_id}! "
                        f"Position is UNPROTECTED at SL={proposal.stop_loss}"
                    )

            if proposal.take_profit:
                tp_order = await self._place_take_profit(order, proposal, position_id)
                if not tp_order:
                    failed_contingent.append("take_profit")
                    logger.warning(f"Take-profit placement failed for position {position_id}")

            # Step 3: Update position with order links (for F14)
            if position and self.position_tracker and (sl_order or tp_order):
                await self.position_tracker.update_order_links(
                    position_id,
                    stop_loss_order_id=sl_order.id if sl_order else None,
                    take_profit_order_id=tp_order.id if tp_order else None,
                )

            # Step 4: Publish alert for failed contingent orders
            if failed_contingent and self.bus:
                from ..orchestration.message_bus import MessageTopic, create_message, MessagePriority
                await self.bus.publish(create_message(
                    topic=MessageTopic.RISK_ALERTS,
                    source="order_execution_manager",
                    payload={
                        "alert_type": "contingent_order_failure",
                        "severity": "critical" if "stop_loss" in failed_contingent else "high",
                        "position_id": position_id,
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "failed_orders": failed_contingent,
                        "message": (
                            "IMMEDIATE ATTENTION: Position lacks stop-loss protection"
                            if "stop_loss" in failed_contingent
                            else "Take-profit order failed to place"
                        ),
                        "requested_stop_loss": str(proposal.stop_loss) if proposal.stop_loss else None,
                        "requested_take_profit": str(proposal.take_profit) if proposal.take_profit else None,
                    },
                    priority=MessagePriority.URGENT,
                ))

        except Exception as e:
            logger.critical(f"Fill handling failed: {e}", exc_info=True)

            # Publish emergency alert
            if self.bus:
                from ..orchestration.message_bus import MessageTopic, create_message, MessagePriority
                await self.bus.publish(create_message(
                    topic=MessageTopic.RISK_ALERTS,
                    source="order_execution_manager",
                    payload={
                        "alert_type": "fill_handling_failure",
                        "severity": "critical",
                        "position_id": position_id,
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "error": str(e),
                        "message": (
                            f"Order fill handling failed for {order.symbol}. "
                            f"Position may be unprotected. Immediate review required."
                        ),
                    },
                    priority=MessagePriority.URGENT,
                ))

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

    async def _handle_partial_fill(
        self,
        order: Order,
        proposal: 'TradeProposal',
        filled_size: Decimal,
        total_size: Decimal,
        is_final: bool = False,
    ) -> None:
        """
        Handle partial order fill.

        Args:
            order: The order with partial fill
            proposal: Original trade proposal
            filled_size: Amount filled so far
            total_size: Total order size
            is_final: True if this is the final state (cancelled/expired with partial fill)
        """
        fill_pct = (filled_size / total_size * 100) if total_size > 0 else Decimal(0)
        logger.info(
            f"Partial fill: {order.id} filled {filled_size}/{total_size} "
            f"({fill_pct:.1f}%) at {order.filled_price}"
        )

        # Only create position on first partial fill or final state
        # Check if we already have a position for this order
        position = None
        if self.position_tracker:
            # Check if position already exists for this order
            existing_positions = await self.position_tracker.get_open_positions(symbol=order.symbol)
            for pos in existing_positions:
                if pos.order_id == order.id:
                    position = pos
                    break

            if not position:
                # Create position for the partial fill
                position = await self.position_tracker.open_position(
                    symbol=order.symbol,
                    side="long" if order.side == OrderSide.BUY else "short",
                    size=filled_size,
                    entry_price=order.filled_price or Decimal(0),
                    leverage=order.leverage,
                    order_id=order.id,
                )
                logger.info(f"Created position for partial fill: {position.id}")
            else:
                # Update existing position size
                # Note: This is a simplified approach - in reality you might want
                # to track multiple fill events separately
                async with self.position_tracker._lock:
                    position.size = filled_size
                await self.position_tracker._update_position(position)
                logger.info(f"Updated position {position.id} size to {filled_size}")

        # If final state (cancelled/expired), place contingent orders now
        if is_final and position:
            if proposal.stop_loss:
                await self._place_stop_loss(order, proposal, position.id)
            if proposal.take_profit:
                await self._place_take_profit(order, proposal, position.id)

        # Publish partial fill event
        if self.bus:
            from ..orchestration.message_bus import MessageTopic, create_message, MessagePriority
            await self.bus.publish(create_message(
                topic=MessageTopic.EXECUTION_EVENTS,
                source="order_execution_manager",
                payload={
                    "event_type": "order_partial_fill",
                    "order_id": order.id,
                    "external_id": order.external_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "filled_size": str(filled_size),
                    "total_size": str(total_size),
                    "fill_pct": str(fill_pct),
                    "fill_price": str(order.filled_price),
                    "position_id": position.id if position else None,
                    "is_final": is_final,
                },
                priority=MessagePriority.NORMAL if not is_final else MessagePriority.HIGH,
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
            # Use price field for stop-loss trigger (not stop_price)
            # Kraken expects trigger price in 'price' for stop-loss orders
            price=Decimal(str(proposal.stop_loss)),
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
        """
        Get order by ID.

        F15: Uses nested locks to prevent race condition where order
        moves from open_orders to history between lock releases.
        """
        # F15: Hold both locks to prevent race between open orders and history
        async with self._lock:
            if order_id in self._open_orders:
                return self._open_orders[order_id]

            # Check history while still holding the main lock
            # This prevents order from moving between the checks
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
        """
        Calculate order size in base currency.

        For limit orders: uses the entry_price from proposal
        For market orders: fetches current price to convert USD to base currency

        Raises:
            ValueError: If price cannot be determined for market orders
        """
        # Get price for conversion
        if proposal.entry_price and proposal.entry_price > 0:
            price = Decimal(str(proposal.entry_price))
        else:
            # For market orders, get current price from tracker or API
            price = await self._get_current_price(proposal.symbol)
            if not price or price <= 0:
                raise ValueError(
                    f"Cannot calculate size without price for {proposal.symbol}. "
                    f"For market orders, current price must be available."
                )

        return Decimal(str(proposal.size_usd)) / price

    async def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current market price for size calculation.

        Checks position tracker's price cache first, then falls back to Kraken API.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")

        Returns:
            Current price as Decimal, or None if unavailable
        """
        # Try position tracker's price cache first (fastest)
        if self.position_tracker:
            cached = self.position_tracker._price_cache.get(symbol)
            if cached and cached > 0:
                logger.debug(f"Using cached price for {symbol}: {cached}")
                return cached

        # Fall back to Kraken API
        if self.kraken:
            try:
                # Acquire rate limit token
                await self._api_rate_limiter.acquire(1)

                kraken_symbol = self._to_kraken_symbol(symbol)
                ticker = await self.kraken.get_ticker(kraken_symbol)

                if ticker and not ticker.get("error"):
                    result = ticker.get("result", {})
                    # Get the first pair's data
                    pair_data = list(result.values())[0] if result else {}
                    if "c" in pair_data:  # 'c' = last trade closed [price, lot-volume]
                        price = Decimal(pair_data["c"][0])
                        logger.debug(f"Fetched price for {symbol} from Kraken: {price}")
                        return price
                else:
                    logger.warning(f"Ticker error for {symbol}: {ticker.get('error')}")
            except Exception as e:
                logger.warning(f"Failed to get ticker for {symbol}: {e}")

        return None

    async def _check_position_limits(self, symbol: str) -> dict:
        """
        Check if opening a new position would exceed limits.

        Args:
            symbol: Trading symbol for the new position

        Returns:
            dict with 'allowed' bool and 'reason' string
        """
        # Get current open positions from tracker
        if not self.position_tracker:
            # If no tracker, allow the trade (limits can't be enforced)
            return {"allowed": True, "reason": None}

        try:
            open_positions = await self.position_tracker.get_open_positions()

            # Check total position limit
            total_count = len(open_positions)
            if total_count >= self._max_positions_total:
                return {
                    "allowed": False,
                    "reason": f"Max total positions ({self._max_positions_total}) reached. "
                              f"Current: {total_count}",
                }

            # Check per-symbol position limit
            symbol_count = sum(1 for p in open_positions if p.symbol == symbol)
            if symbol_count >= self._max_positions_per_symbol:
                return {
                    "allowed": False,
                    "reason": f"Max positions for {symbol} ({self._max_positions_per_symbol}) reached. "
                              f"Current: {symbol_count}",
                }

            return {"allowed": True, "reason": None}

        except Exception as e:
            logger.warning(f"Failed to check position limits: {e}")
            # On error, allow the trade but log the issue
            return {"allowed": True, "reason": None}

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

    async def place_stop_loss_update(
        self,
        position,  # Position object
        new_stop_price: Decimal,
    ) -> Optional[Order]:
        """
        Place a new stop-loss order for an existing position.

        F08: Used when modifying SL price - cancels old order, places new one.

        Args:
            position: Position object
            new_stop_price: New stop-loss price

        Returns:
            New Order if successful, None otherwise
        """
        # Determine order side (opposite of position)
        from .position_tracker import PositionSide
        order_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        sl_order = Order(
            id=str(uuid.uuid4()),
            symbol=position.symbol,
            side=order_side,
            order_type=OrderType.STOP_LOSS,
            size=position.size,
            price=new_stop_price,  # Use price for stop trigger
        )

        success = await self._place_order(sl_order)

        if success:
            async with self._lock:
                self._open_orders[sl_order.id] = sl_order
            await self._store_order(sl_order)
            logger.info(f"Stop-loss update placed: {sl_order.id} at {new_stop_price}")
            return sl_order

        return None

    async def place_take_profit_update(
        self,
        position,  # Position object
        new_take_profit_price: Decimal,
    ) -> Optional[Order]:
        """
        Place a new take-profit order for an existing position.

        F08: Used when modifying TP price - cancels old order, places new one.

        Args:
            position: Position object
            new_take_profit_price: New take-profit price

        Returns:
            New Order if successful, None otherwise
        """
        from .position_tracker import PositionSide
        order_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        tp_order = Order(
            id=str(uuid.uuid4()),
            symbol=position.symbol,
            side=order_side,
            order_type=OrderType.TAKE_PROFIT,
            size=position.size,
            price=new_take_profit_price,
        )

        success = await self._place_order(tp_order)

        if success:
            async with self._lock:
                self._open_orders[tp_order.id] = tp_order
            await self._store_order(tp_order)
            logger.info(f"Take-profit update placed: {tp_order.id} at {new_take_profit_price}")
            return tp_order

        return None

    async def get_orders_for_position(self, position_id: str) -> list[Order]:
        """
        Get all orders related to a position.

        F12: Used to find orphan orders when closing a position.

        Args:
            position_id: Position ID

        Returns:
            List of related orders
        """
        related_orders: list[Order] = []

        # Check open orders
        async with self._lock:
            for order in self._open_orders.values():
                # Check if order is related to this position
                # (parent order created the position, or it's a SL/TP for the position)
                if order.parent_order_id:
                    # Get parent order and check if it created this position
                    parent = self._open_orders.get(order.parent_order_id)
                    if parent and parent.id == position_id:
                        related_orders.append(order)

        # Also check by position tracker's order links if available
        if self.position_tracker:
            position = await self.position_tracker.get_position(position_id)
            if position:
                if position.stop_loss_order_id:
                    sl_order = await self.get_order(position.stop_loss_order_id)
                    if sl_order and sl_order not in related_orders:
                        related_orders.append(sl_order)
                if position.take_profit_order_id:
                    tp_order = await self.get_order(position.take_profit_order_id)
                    if tp_order and tp_order not in related_orders:
                        related_orders.append(tp_order)

        return related_orders

    async def cancel_orphan_orders(self, position_id: str) -> int:
        """
        Cancel all orphan orders for a closed position.

        F12: When a position closes (via SL, TP, or manual), cancel the other order.

        Args:
            position_id: Position ID that was closed

        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0
        related_orders = await self.get_orders_for_position(position_id)

        for order in related_orders:
            if order.status in [OrderStatus.OPEN, OrderStatus.PENDING]:
                success = await self.cancel_order(order.id)
                if success:
                    cancelled_count += 1
                    logger.info(f"Cancelled orphan order {order.id} for position {position_id}")
                else:
                    logger.warning(f"Failed to cancel orphan order {order.id}")

        return cancelled_count

    async def handle_oco_fill(self, filled_order: Order) -> None:
        """
        Handle one-cancels-other logic when SL or TP fills.

        F16: When SL fills, cancel TP (and vice versa).

        Args:
            filled_order: The order that was filled (SL or TP)
        """
        if not filled_order.parent_order_id:
            return

        # Get the parent order to find sibling orders
        parent = await self.get_order(filled_order.parent_order_id)
        if not parent:
            return

        # Determine which order to cancel
        if filled_order.order_type == OrderType.STOP_LOSS:
            # SL filled, cancel TP
            if parent.take_profit_order_id:
                await self.cancel_order(parent.take_profit_order_id)
                logger.info(f"OCO: Cancelled TP {parent.take_profit_order_id} after SL fill")
        elif filled_order.order_type == OrderType.TAKE_PROFIT:
            # TP filled, cancel SL
            if parent.stop_loss_order_id:
                await self.cancel_order(parent.stop_loss_order_id)
                logger.info(f"OCO: Cancelled SL {parent.stop_loss_order_id} after TP fill")

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
