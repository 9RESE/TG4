"""
Paper Order Executor - Simulated order execution for paper trading.

Phase 6: Paper Trading Integration

Features:
- Configurable fill delay
- Slippage simulation
- Partial fill simulation (optional)
- Fee calculation
- Real-time price source integration
- Position creation and tracking
"""

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .order_manager import Order, OrderSide, OrderStatus, OrderType, ExecutionResult
from .paper_portfolio import PaperPortfolio, InsufficientBalanceError

if TYPE_CHECKING:
    from ..risk.rules_engine import TradeProposal

logger = logging.getLogger(__name__)


@dataclass
class PaperFillResult:
    """Result of a paper order fill simulation."""
    order: Order
    filled: bool
    fill_price: Optional[Decimal] = None
    fill_size: Optional[Decimal] = None
    fee: Decimal = Decimal("0")
    fee_currency: str = ""
    error_message: Optional[str] = None
    latency_ms: int = 0


class PaperOrderExecutor:
    """
    Simulated order execution for paper trading.

    Features:
    - Configurable fill delay (network latency simulation)
    - Slippage simulation for market orders
    - Partial fill simulation (optional)
    - Fee calculation per trade
    - Real-time price source integration
    - Position creation/tracking integration
    """

    def __init__(
        self,
        config: dict,
        paper_portfolio: PaperPortfolio,
        price_source: Callable[[str], Optional[Decimal]],
        position_tracker=None,
    ):
        """
        Initialize paper executor.

        Args:
            config: Execution configuration (from execution.yaml)
            paper_portfolio: Portfolio for balance tracking
            price_source: Function to get current price for a symbol
            position_tracker: Optional PositionTracker for position management
        """
        self.config = config
        self.portfolio = paper_portfolio
        self.get_price = price_source
        self.position_tracker = position_tracker

        # Extract paper trading settings
        paper_config = config.get("paper_trading", {})
        self.fill_delay_ms = paper_config.get("fill_delay_ms", 100)
        self.slippage_pct = Decimal(str(paper_config.get("simulated_slippage_pct", 0.1)))
        self.simulate_partial_fills = paper_config.get("simulate_partial_fills", False)

        # Get fee rates from symbol config
        self.symbol_fees: Dict[str, Decimal] = {}
        for symbol, cfg in config.get("symbols", {}).items():
            self.symbol_fees[symbol] = Decimal(str(cfg.get("fee_pct", 0.26)))

        # Default fee if symbol not configured
        self.default_fee_pct = Decimal("0.26")

        # Order tracking
        self._open_orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}

        # Statistics (CRITICAL-02: Protected by lock for thread-safety)
        self._total_orders_placed = 0
        self._total_orders_filled = 0
        self._total_orders_rejected = 0
        self._stats_lock = asyncio.Lock()  # CRITICAL-02: Thread-safe statistics

        logger.info(
            f"PaperOrderExecutor initialized: delay={self.fill_delay_ms}ms, "
            f"slippage={self.slippage_pct}%"
        )

    async def execute_trade(
        self,
        proposal: 'TradeProposal',
    ) -> ExecutionResult:
        """
        Execute a validated trade proposal in paper mode.

        Args:
            proposal: Risk-validated trade proposal

        Returns:
            ExecutionResult with order/position details
        """
        import time
        start_time = time.perf_counter()

        # Create order from proposal
        order_type = OrderType.LIMIT if proposal.entry_price else OrderType.MARKET

        # Get current price for size calculation
        current_price = self.get_price(proposal.symbol)
        if not current_price and not proposal.entry_price:
            return ExecutionResult(
                success=False,
                error_message=f"No price available for {proposal.symbol}",
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        # Calculate size in base currency with proper precision (HIGH-02)
        price_for_calc = Decimal(str(proposal.entry_price)) if proposal.entry_price else current_price
        raw_size = Decimal(str(proposal.size_usd)) / price_for_calc

        # HIGH-02: Quantize size based on symbol config for proper precision
        symbol_config = self.config.get("symbols", {}).get(proposal.symbol, {})
        size_decimals = symbol_config.get("size_decimals", 8)  # Default to 8 decimals
        quantize_str = "0." + "0" * size_decimals if size_decimals > 0 else "1"
        size = raw_size.quantize(Decimal(quantize_str))

        order = Order(
            id=str(uuid.uuid4()),
            symbol=proposal.symbol,
            side=OrderSide.BUY if proposal.side == "buy" else OrderSide.SELL,
            order_type=order_type,
            size=size,
            price=Decimal(str(proposal.entry_price)) if proposal.entry_price else None,
            leverage=proposal.leverage,
        )

        # Execute order
        try:
            result = await self.execute_order(order)

            if result.filled:
                # CRITICAL-02: Thread-safe statistics update
                async with self._stats_lock:
                    self._total_orders_placed += 1
                    self._total_orders_filled += 1

                # Create position if position tracker available
                position_id = None
                if self.position_tracker:
                    position = await self.position_tracker.open_position(
                        symbol=order.symbol,
                        side="long" if order.side == OrderSide.BUY else "short",
                        size=result.fill_size or order.size,
                        entry_price=result.fill_price or order.price or current_price,
                        leverage=order.leverage,
                        order_id=order.id,
                        stop_loss=Decimal(str(proposal.stop_loss)) if proposal.stop_loss else None,
                        take_profit=Decimal(str(proposal.take_profit)) if proposal.take_profit else None,
                    )
                    position_id = position.id if position else None
                    logger.info(f"📝 Paper position opened: {position_id}")

                return ExecutionResult(
                    success=True,
                    order=result.order,
                    position_id=position_id,
                    execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                )
            else:
                # CRITICAL-02: Thread-safe statistics update
                async with self._stats_lock:
                    self._total_orders_rejected += 1
                return ExecutionResult(
                    success=False,
                    order=result.order,
                    error_message=result.error_message or "Order not filled",
                    execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                )

        except Exception as e:
            logger.error(f"Paper trade execution failed: {e}", exc_info=True)
            # CRITICAL-02: Thread-safe statistics update
            async with self._stats_lock:
                self._total_orders_rejected += 1
            return ExecutionResult(
                success=False,
                error_message=str(e),
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

    async def execute_order(self, order: Order) -> PaperFillResult:
        """
        Execute a paper trading order.

        Args:
            order: Order to execute

        Returns:
            PaperFillResult with fill information
        """
        import time
        start_time = time.perf_counter()

        # Simulate network/exchange latency
        delay_seconds = self.fill_delay_ms / 1000
        await asyncio.sleep(delay_seconds)

        # Get current market price
        current_price = self.get_price(order.symbol)
        if current_price is None:
            order.status = OrderStatus.ERROR
            order.error_message = f"No price available for {order.symbol}"
            return PaperFillResult(
                order=order,
                filled=False,
                error_message=order.error_message,
                latency_ms=int((time.perf_counter() - start_time) * 1000),
            )

        # Determine fill price based on order type
        fill_price = self._calculate_fill_price(order, current_price)

        # Check if limit order would fill
        if order.order_type == OrderType.LIMIT:
            if not self._would_limit_fill(order, current_price):
                order.status = OrderStatus.OPEN
                # Store for later monitoring
                self._open_orders[order.id] = order
                return PaperFillResult(
                    order=order,
                    filled=False,
                    latency_ms=int((time.perf_counter() - start_time) * 1000),
                )

        # Simulate partial fills if enabled
        if self.simulate_partial_fills and random.random() < 0.2:
            filled_pct = Decimal(str(random.uniform(0.3, 0.9)))
            fill_size = order.size * filled_pct
            order.filled_size = fill_size
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            fill_size = order.size
            order.filled_size = fill_size
            order.status = OrderStatus.FILLED

        order.filled_price = fill_price
        order.updated_at = datetime.now(timezone.utc)

        # Calculate and record fee
        fee_pct = self.symbol_fees.get(order.symbol, self.default_fee_pct)
        fee = fill_size * fill_price * fee_pct / Decimal("100")
        order.fee_amount = fee
        order.fee_currency = order.symbol.split("/")[1]  # Quote currency

        # Update portfolio balances
        side = "buy" if order.side == OrderSide.BUY else "sell"
        try:
            self.portfolio.execute_trade(
                symbol=order.symbol,
                side=side,
                size=fill_size,
                price=fill_price,
                fee_pct=fee_pct,
                order_id=order.id,
            )
        except InsufficientBalanceError as e:
            # CRITICAL-01: Use REJECTED for business logic rejections (not ERROR)
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            # CRITICAL-02: Thread-safe statistics update
            async with self._stats_lock:
                self._total_orders_rejected += 1
            logger.debug(f"Paper order rejected: {e}")
            return PaperFillResult(
                order=order,
                filled=False,
                error_message=str(e),
                latency_ms=int((time.perf_counter() - start_time) * 1000),
            )

        # Add to history with MEDIUM-03 persistence before trimming
        self._order_history.append(order)
        if len(self._order_history) > 1000:
            # MEDIUM-03: Persist old orders to database before trimming
            await self._persist_orders_before_trim(self._order_history[:-1000])
            self._order_history = self._order_history[-1000:]

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            f"📝 Paper order filled: {order.side.value.upper()} {fill_size} {order.symbol} "
            f"@ {fill_price} (fee: {fee}, latency: {latency_ms}ms)"
        )

        return PaperFillResult(
            order=order,
            filled=True,
            fill_price=fill_price,
            fill_size=fill_size,
            fee=fee,
            fee_currency=order.fee_currency,
            latency_ms=latency_ms,
        )

    def _calculate_fill_price(
        self,
        order: Order,
        current_price: Decimal,
    ) -> Decimal:
        """
        Calculate fill price including slippage.

        Market orders get slippage.
        Limit orders fill at limit price (if they fill).
        Stop orders trigger at stop price then fill with slippage.

        Args:
            order: The order being executed
            current_price: Current market price

        Returns:
            Calculated fill price
        """
        if order.order_type == OrderType.LIMIT:
            return order.price  # Limit orders fill at limit price

        # Apply slippage for market and stop orders
        slippage_multiplier = self.slippage_pct / Decimal("100")

        # Add some randomness to slippage (up to configured amount)
        random_factor = Decimal(str(random.uniform(0.5, 1.0)))
        actual_slippage = slippage_multiplier * random_factor

        if order.side == OrderSide.BUY:
            # Buying: pay slightly more (slippage up)
            return current_price * (Decimal("1") + actual_slippage)
        else:
            # Selling: receive slightly less (slippage down)
            return current_price * (Decimal("1") - actual_slippage)

    def _would_limit_fill(self, order: Order, current_price: Decimal) -> bool:
        """
        Check if a limit order would fill at current price.

        Args:
            order: Limit order to check
            current_price: Current market price

        Returns:
            True if order would fill
        """
        if order.price is None:
            return False

        if order.side == OrderSide.BUY:
            # Buy limit fills if market price <= limit price
            return current_price <= order.price
        else:
            # Sell limit fills if market price >= limit price
            return current_price >= order.price

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open paper order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if order_id in self._open_orders:
            order = self._open_orders[order_id]
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now(timezone.utc)
            del self._open_orders[order_id]
            self._order_history.append(order)
            logger.info(f"Paper order cancelled: {order_id}")
            return True

        logger.warning(f"Paper order not found for cancel: {order_id}")
        return False

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open paper orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        orders = list(self._open_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order if found, None otherwise
        """
        if order_id in self._open_orders:
            return self._open_orders[order_id]

        for order in self._order_history:
            if order.id == order_id:
                return order

        return None

    def get_stats(self) -> dict:
        """Get paper executor statistics."""
        return {
            "total_orders_placed": self._total_orders_placed,
            "total_orders_filled": self._total_orders_filled,
            "total_orders_rejected": self._total_orders_rejected,
            "open_orders_count": len(self._open_orders),
            "history_count": len(self._order_history),
            "fill_delay_ms": self.fill_delay_ms,
            "slippage_pct": float(self.slippage_pct),
            "portfolio_trade_count": self.portfolio.trade_count,
            "portfolio_fees_paid": float(self.portfolio.total_fees_paid),
        }

    async def close_all_positions(self) -> int:
        """
        Close all open positions (emergency).

        Returns:
            Number of positions closed
        """
        if not self.position_tracker:
            logger.warning("No position tracker - cannot close positions")
            return 0

        closed = 0
        positions = await self.position_tracker.get_open_positions()

        for position in positions:
            current_price = self.get_price(position.symbol)
            if current_price:
                await self.position_tracker.close_position(
                    position_id=position.id,
                    exit_price=current_price,
                    reason="emergency_close",
                )
                closed += 1
                logger.info(f"Emergency closed paper position: {position.id}")

        return closed

    async def reduce_positions(self, reduction_pct: int) -> int:
        """
        Reduce all positions by a percentage (emergency).

        Args:
            reduction_pct: Percentage to reduce (0-100)

        Returns:
            Number of positions modified
        """
        if not self.position_tracker:
            logger.warning("No position tracker - cannot reduce positions")
            return 0

        modified = 0
        # For paper trading, we could implement partial position closing
        # For now, just log the intent
        logger.warning(f"Position reduction by {reduction_pct}% requested (paper mode)")
        return modified

    async def _persist_orders_before_trim(self, orders: List[Order]) -> None:
        """
        Persist orders to database before trimming from memory.

        MEDIUM-03 / NEW-HIGH-01: Ensures order history is not lost when memory
        limit is reached. Orders are persisted to paper_orders table.

        Args:
            orders: List of orders to persist
        """
        if not orders:
            return

        # Check if we have a database connection
        if not hasattr(self, '_db') or self._db is None:
            logger.warning(
                f"NEW-HIGH-01: No database connection - {len(orders)} orders will be lost. "
                "Call set_database() to enable order persistence."
            )
            return

        try:
            # Get session_id from portfolio if available
            session_id = getattr(self.portfolio, 'session_id', None)

            persisted_count = 0
            for order in orders:
                try:
                    # Map OrderStatus enum to database-compatible string
                    status_map = {
                        OrderStatus.PENDING: 'pending',
                        OrderStatus.OPEN: 'open',
                        OrderStatus.PARTIALLY_FILLED: 'partially_filled',
                        OrderStatus.FILLED: 'filled',
                        OrderStatus.CANCELLED: 'cancelled',
                        OrderStatus.EXPIRED: 'expired',
                        OrderStatus.ERROR: 'error',
                        OrderStatus.REJECTED: 'error',  # Map REJECTED to 'error' for DB
                    }
                    db_status = status_map.get(order.status, 'error')

                    # Map OrderType to database-compatible string
                    order_type_str = order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type)

                    await self._db.execute(
                        """
                        INSERT INTO paper_orders (
                            external_id, session_id, symbol, side, order_type,
                            size, price, filled_size, filled_price,
                            fee_amount, fee_currency, leverage, status,
                            created_at, updated_at, error_message
                        ) VALUES (
                            $1, $2, $3, $4, $5,
                            $6, $7, $8, $9,
                            $10, $11, $12, $13,
                            $14, $15, $16
                        )
                        ON CONFLICT DO NOTHING
                        """,
                        order.id,  # external_id
                        session_id,
                        order.symbol,
                        order.side.value if hasattr(order.side, 'value') else str(order.side),
                        order_type_str,
                        float(order.size),
                        float(order.price) if order.price else None,
                        float(order.filled_size) if order.filled_size else 0,
                        float(order.filled_price) if order.filled_price else None,
                        float(order.fee_amount) if order.fee_amount else 0,
                        order.fee_currency if hasattr(order, 'fee_currency') else None,
                        order.leverage if hasattr(order, 'leverage') else 1,
                        db_status,
                        order.created_at if hasattr(order, 'created_at') else datetime.now(timezone.utc),
                        order.updated_at if hasattr(order, 'updated_at') else datetime.now(timezone.utc),
                        order.error_message if hasattr(order, 'error_message') else None,
                    )
                    persisted_count += 1
                except Exception as order_err:
                    logger.debug(f"Failed to persist order {order.id}: {order_err}")

            logger.info(
                f"NEW-HIGH-01: Persisted {persisted_count}/{len(orders)} orders "
                f"to database before trimming from memory"
            )

        except Exception as e:
            logger.error(f"Failed to persist orders before trim: {e}")

    def set_database(self, db) -> None:
        """
        Set database connection for order persistence.

        Args:
            db: Database connection pool
        """
        self._db = db
        logger.debug("Database connection set for paper executor")
