"""
Position Tracker - Open position tracking and P&L monitoring.

NOT an LLM agent - purely rule-based position management.

Features:
- Track open positions with entry details
- Real-time unrealized P&L calculation
- Position snapshots for time-series analysis
- Position modification and closure
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..orchestration.message_bus import MessageBus

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side (direction)."""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position status."""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


@dataclass
class Position:
    """Open position representation."""
    id: str
    symbol: str
    side: PositionSide
    size: Decimal
    entry_price: Decimal
    leverage: int = 1
    status: PositionStatus = PositionStatus.OPEN
    order_id: Optional[str] = None  # Opening order ID
    external_id: Optional[str] = None  # Kraken position ID (for sync)
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    stop_loss_order_id: Optional[str] = None  # F14: Link to SL order
    take_profit_order_id: Optional[str] = None  # F14: Link to TP order
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    exit_price: Optional[Decimal] = None

    # P&L tracking
    realized_pnl: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)
    unrealized_pnl_pct: Decimal = Decimal(0)

    # Fee tracking (F10)
    total_fees: Decimal = Decimal(0)
    fee_currency: str = ""

    # Trailing stop fields
    trailing_stop_enabled: bool = False
    trailing_stop_activated: bool = False
    trailing_stop_highest_price: Optional[Decimal] = None  # For LONG positions
    trailing_stop_lowest_price: Optional[Decimal] = None   # For SHORT positions
    trailing_stop_distance_pct: Decimal = Decimal("1.5")

    # Position metadata
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate position fields after initialization."""
        # Validate leverage (must be positive, max 5x per system constraints)
        if self.leverage < 1:
            raise ValueError(f"Leverage must be >= 1, got {self.leverage}")
        if self.leverage > 5:
            raise ValueError(f"Leverage must be <= 5 (system limit), got {self.leverage}")

        # Validate size (must be positive)
        if self.size <= 0:
            raise ValueError(f"Position size must be > 0, got {self.size}")

        # Validate entry price (must be non-negative)
        if self.entry_price < 0:
            raise ValueError(f"Entry price must be >= 0, got {self.entry_price}")

    def calculate_pnl(self, current_price: Decimal) -> tuple[Decimal, Decimal]:
        """
        Calculate unrealized P&L.

        Args:
            current_price: Current market price

        Returns:
            Tuple of (unrealized_pnl_usd, unrealized_pnl_pct)
        """
        if self.entry_price == 0:
            return Decimal(0), Decimal(0)

        if self.side == PositionSide.LONG:
            price_diff = current_price - self.entry_price
            pnl = price_diff * self.size * self.leverage
            pnl_pct = (price_diff / self.entry_price) * 100 * self.leverage
        else:
            price_diff = self.entry_price - current_price
            pnl = price_diff * self.size * self.leverage
            pnl_pct = (price_diff / self.entry_price) * 100 * self.leverage

        return pnl, pnl_pct

    def update_pnl(self, current_price: Decimal) -> None:
        """Update unrealized P&L with current price."""
        self.unrealized_pnl, self.unrealized_pnl_pct = self.calculate_pnl(current_price)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "size": str(self.size),
            "entry_price": str(self.entry_price),
            "leverage": self.leverage,
            "status": self.status.value,
            "order_id": self.order_id,
            "external_id": self.external_id,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "stop_loss_order_id": self.stop_loss_order_id,
            "take_profit_order_id": self.take_profit_order_id,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "unrealized_pnl_pct": str(self.unrealized_pnl_pct),
            "total_fees": str(self.total_fees),
            "fee_currency": self.fee_currency,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "trailing_stop_activated": self.trailing_stop_activated,
            "trailing_stop_highest_price": str(self.trailing_stop_highest_price) if self.trailing_stop_highest_price else None,
            "trailing_stop_lowest_price": str(self.trailing_stop_lowest_price) if self.trailing_stop_lowest_price else None,
            "trailing_stop_distance_pct": str(self.trailing_stop_distance_pct),
            "notes": self.notes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            side=PositionSide(data["side"]),
            size=Decimal(data["size"]),
            entry_price=Decimal(data["entry_price"]),
            leverage=data.get("leverage", 1),
            status=PositionStatus(data.get("status", "open")),
            order_id=data.get("order_id"),
            external_id=data.get("external_id"),
            stop_loss=Decimal(data["stop_loss"]) if data.get("stop_loss") else None,
            take_profit=Decimal(data["take_profit"]) if data.get("take_profit") else None,
            stop_loss_order_id=data.get("stop_loss_order_id"),
            take_profit_order_id=data.get("take_profit_order_id"),
            opened_at=datetime.fromisoformat(data["opened_at"]) if data.get("opened_at") else datetime.now(timezone.utc),
            closed_at=datetime.fromisoformat(data["closed_at"]) if data.get("closed_at") else None,
            exit_price=Decimal(data["exit_price"]) if data.get("exit_price") else None,
            realized_pnl=Decimal(data.get("realized_pnl", "0")),
            unrealized_pnl=Decimal(data.get("unrealized_pnl", "0")),
            unrealized_pnl_pct=Decimal(data.get("unrealized_pnl_pct", "0")),
            total_fees=Decimal(data.get("total_fees", "0")),
            fee_currency=data.get("fee_currency", ""),
            trailing_stop_enabled=data.get("trailing_stop_enabled", False),
            trailing_stop_activated=data.get("trailing_stop_activated", False),
            trailing_stop_highest_price=Decimal(data["trailing_stop_highest_price"]) if data.get("trailing_stop_highest_price") else None,
            trailing_stop_lowest_price=Decimal(data["trailing_stop_lowest_price"]) if data.get("trailing_stop_lowest_price") else None,
            trailing_stop_distance_pct=Decimal(data.get("trailing_stop_distance_pct", "1.5")),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
        )


@dataclass
class PositionSnapshot:
    """Point-in-time snapshot of a position."""
    position_id: str
    symbol: str
    timestamp: datetime
    current_price: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "current_price": str(self.current_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "unrealized_pnl_pct": str(self.unrealized_pnl_pct),
        }


class PositionTracker:
    """
    Tracks open positions and monitors P&L.

    Features:
    - Maintain accurate position state
    - Real-time P&L calculation
    - Position snapshots for time-series analysis
    - Integration with risk engine for exposure tracking
    """

    def __init__(
        self,
        message_bus: Optional['MessageBus'] = None,
        risk_engine=None,
        db_pool=None,
        config: Optional[dict] = None,
    ):
        """
        Initialize PositionTracker.

        Args:
            message_bus: MessageBus for event publishing
            risk_engine: RiskManagementEngine for exposure updates
            db_pool: Database pool for persistence
            config: Position tracking configuration
        """
        self.bus = message_bus
        self.risk_engine = risk_engine
        self.db = db_pool
        self.config = config or {}

        # Position tracking
        self._positions: dict[str, Position] = {}
        self._closed_positions: list[Position] = []
        self._snapshots: list[PositionSnapshot] = []

        # Configuration
        tracking_config = self.config.get('position_tracking', {})
        self._snapshot_interval_seconds = tracking_config.get('snapshot_interval_seconds', 60)
        self._max_snapshots = tracking_config.get('max_snapshots', 10000)

        # Trailing stop configuration
        trailing_config = self.config.get('trailing_stop', {})
        self._trailing_stop_enabled = trailing_config.get('enabled', False)
        self._trailing_stop_activation_pct = Decimal(str(trailing_config.get('activation_pct', 1.0)))
        self._trailing_stop_distance_pct = Decimal(str(trailing_config.get('trail_distance_pct', 1.5)))

        # Price cache for P&L calculations
        self._price_cache: dict[str, Decimal] = {}

        # F09: Separate interval for SL/TP trigger checks (faster than snapshots)
        self._trigger_check_interval_seconds = tracking_config.get('trigger_check_interval_seconds', 5)

        # Background tasks
        self._snapshot_task: Optional[asyncio.Task] = None
        self._trigger_check_task: Optional[asyncio.Task] = None
        self._running = False

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Statistics
        self._total_positions_opened = 0
        self._total_positions_closed = 0
        self._total_realized_pnl = Decimal(0)

    async def start(self) -> None:
        """Start position tracking and snapshot task."""
        self._running = True
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        # F09: Start separate faster trigger check loop
        self._trigger_check_task = asyncio.create_task(self._trigger_check_loop())
        await self._load_positions()
        logger.info("PositionTracker started")

    async def stop(self) -> None:
        """Stop position tracking."""
        self._running = False
        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
        # F09: Stop trigger check task
        if self._trigger_check_task:
            self._trigger_check_task.cancel()
            try:
                await self._trigger_check_task
            except asyncio.CancelledError:
                pass
        logger.info("PositionTracker stopped")

    async def open_position(
        self,
        symbol: str,
        side: str,
        size: Decimal,
        entry_price: Decimal,
        leverage: int = 1,
        order_id: Optional[str] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            size: Position size in base currency
            entry_price: Entry price
            leverage: Position leverage
            order_id: Related order ID
            stop_loss: Stop-loss price
            take_profit: Take-profit price

        Returns:
            New Position object
        """
        position = Position(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=PositionSide.LONG if side == "long" else PositionSide.SHORT,
            size=size,
            entry_price=entry_price,
            leverage=leverage,
            order_id=order_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        async with self._lock:
            self._positions[position.id] = position
            self._total_positions_opened += 1

        # Persist to database
        await self._store_position(position)

        # Update risk engine
        await self._update_risk_exposure()

        # Publish event
        if self.bus:
            from ..orchestration.message_bus import MessageTopic, create_message
            await self.bus.publish(create_message(
                topic=MessageTopic.PORTFOLIO_UPDATES,
                source="position_tracker",
                payload={
                    "event_type": "position_opened",
                    "position": position.to_dict(),
                },
            ))

        logger.info(f"Position opened: {position.id} {symbol} {side} {size} @ {entry_price}")
        return position

    async def close_position(
        self,
        position_id: str,
        exit_price: Decimal,
        reason: str = "manual",
        order_manager=None,
    ) -> Optional[Position]:
        """
        Close an open position.

        F12: If order_manager is provided, cancels orphan SL/TP orders.

        Args:
            position_id: Position ID to close
            exit_price: Exit/closing price
            reason: Reason for closing
            order_manager: Optional OrderExecutionManager to cancel orphan orders

        Returns:
            Closed position or None if not found
        """
        async with self._lock:
            position = self._positions.get(position_id)

            if not position:
                logger.warning(f"Position not found: {position_id}")
                return None

            if position.status != PositionStatus.OPEN:
                logger.warning(f"Position already closed: {position_id}")
                return position

            # Update position
            position.status = PositionStatus.CLOSED
            position.exit_price = exit_price
            position.closed_at = datetime.now(timezone.utc)

            # Calculate final P&L
            pnl, pnl_pct = position.calculate_pnl(exit_price)
            position.realized_pnl = pnl
            position.unrealized_pnl = Decimal(0)
            position.unrealized_pnl_pct = Decimal(0)

            # Move to closed list
            del self._positions[position_id]
            self._closed_positions.append(position)
            self._total_positions_closed += 1
            self._total_realized_pnl += pnl

        # Persist to database
        await self._update_position(position)

        # Update risk engine
        await self._update_risk_exposure()

        # Report to risk engine
        if self.risk_engine:
            is_win = position.realized_pnl > 0
            self.risk_engine.record_trade_result(is_win)
            if is_win:
                self.risk_engine.apply_post_trade_cooldown()

        # Publish event
        if self.bus:
            from ..orchestration.message_bus import MessageTopic, create_message, MessagePriority
            await self.bus.publish(create_message(
                topic=MessageTopic.PORTFOLIO_UPDATES,
                source="position_tracker",
                payload={
                    "event_type": "position_closed",
                    "position": position.to_dict(),
                    "reason": reason,
                },
                priority=MessagePriority.HIGH,
            ))

        logger.info(
            f"Position closed: {position_id} @ {exit_price}, "
            f"P&L: {pnl:.2f} ({pnl_pct:.2f}%)"
        )

        # F12: Cancel orphan orders (SL/TP that are no longer needed)
        if order_manager:
            cancelled = await order_manager.cancel_orphan_orders(position_id)
            if cancelled > 0:
                logger.info(f"Cancelled {cancelled} orphan orders for closed position {position_id}")

        return position

    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        order_manager=None,
    ) -> Optional[Position]:
        """
        Modify position stop-loss or take-profit.

        F08: If order_manager is provided, also updates exchange orders.

        Args:
            position_id: Position ID to modify
            stop_loss: New stop-loss price
            take_profit: New take-profit price
            order_manager: Optional OrderExecutionManager to update exchange orders

        Returns:
            Modified position or None if not found
        """
        async with self._lock:
            position = self._positions.get(position_id)

            if not position:
                logger.warning(f"Position not found: {position_id}")
                return None

            old_sl = position.stop_loss
            old_tp = position.take_profit

            if stop_loss is not None:
                position.stop_loss = stop_loss

            if take_profit is not None:
                position.take_profit = take_profit

        # Persist to database
        await self._update_position(position)

        # F08: Update exchange orders if manager provided
        if order_manager:
            if stop_loss is not None and old_sl != stop_loss:
                # Cancel old SL order and place new one
                if position.stop_loss_order_id:
                    cancelled = await order_manager.cancel_order(position.stop_loss_order_id)
                    if cancelled:
                        logger.info(f"Cancelled old SL order {position.stop_loss_order_id}")

                # Place new SL order
                new_sl_order = await order_manager.place_stop_loss_update(
                    position=position,
                    new_stop_price=stop_loss,
                )
                if new_sl_order:
                    position.stop_loss_order_id = new_sl_order.id
                    await self._update_position(position)
                else:
                    logger.warning(f"Failed to place updated SL order for position {position_id}")

            if take_profit is not None and old_tp != take_profit:
                # Cancel old TP order and place new one
                if position.take_profit_order_id:
                    cancelled = await order_manager.cancel_order(position.take_profit_order_id)
                    if cancelled:
                        logger.info(f"Cancelled old TP order {position.take_profit_order_id}")

                # Place new TP order
                new_tp_order = await order_manager.place_take_profit_update(
                    position=position,
                    new_take_profit_price=take_profit,
                )
                if new_tp_order:
                    position.take_profit_order_id = new_tp_order.id
                    await self._update_position(position)
                else:
                    logger.warning(f"Failed to place updated TP order for position {position_id}")

        logger.info(f"Position modified: {position_id} SL={stop_loss} TP={take_profit}")
        return position

    async def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        async with self._lock:
            return self._positions.get(position_id)

    async def get_open_positions(
        self,
        symbol: Optional[str] = None,
    ) -> list[Position]:
        """
        Get all open positions, optionally filtered by symbol.

        Args:
            symbol: Filter by symbol

        Returns:
            List of open positions
        """
        async with self._lock:
            positions = list(self._positions.values())

        if symbol:
            positions = [p for p in positions if p.symbol == symbol]

        return positions

    async def get_closed_positions(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Position]:
        """
        Get closed positions with filtering.

        Args:
            symbol: Filter by symbol
            since: Only positions closed after this time
            limit: Maximum positions to return

        Returns:
            List of closed positions (newest first)
        """
        positions = list(reversed(self._closed_positions))

        if symbol:
            positions = [p for p in positions if p.symbol == symbol]

        if since:
            positions = [p for p in positions if p.closed_at and p.closed_at >= since]

        return positions[:limit]

    async def update_prices(self, prices: dict[str, Decimal]) -> None:
        """
        Update price cache and recalculate P&L for all positions.

        Args:
            prices: Dict of symbol -> current price
        """
        self._price_cache.update(prices)

        async with self._lock:
            for position in self._positions.values():
                if position.symbol in prices:
                    position.update_pnl(prices[position.symbol])

    async def get_total_exposure(self) -> dict[str, Any]:
        """Calculate total exposure across all open positions."""
        async with self._lock:
            positions = list(self._positions.values())

        total_exposure_usd = Decimal(0)
        exposure_by_symbol: dict[str, Decimal] = {}

        for pos in positions:
            # Get current price
            current_price = self._price_cache.get(pos.symbol, pos.entry_price)
            position_value = pos.size * current_price * pos.leverage

            total_exposure_usd += position_value

            if pos.symbol not in exposure_by_symbol:
                exposure_by_symbol[pos.symbol] = Decimal(0)
            exposure_by_symbol[pos.symbol] += position_value

        return {
            "total_exposure_usd": float(total_exposure_usd),
            "exposure_by_symbol": {k: float(v) for k, v in exposure_by_symbol.items()},
            "open_positions_count": len(positions),
        }

    async def get_total_unrealized_pnl(self) -> dict[str, Any]:
        """Get total unrealized P&L across all positions."""
        async with self._lock:
            positions = list(self._positions.values())

        total_pnl = Decimal(0)
        pnl_by_position: dict[str, Decimal] = {}

        for pos in positions:
            total_pnl += pos.unrealized_pnl
            pnl_by_position[pos.id] = pos.unrealized_pnl

        return {
            "total_unrealized_pnl": float(total_pnl),
            "pnl_by_position": {k: float(v) for k, v in pnl_by_position.items()},
        }

    async def check_sl_tp_triggers(
        self,
        current_prices: dict[str, Decimal],
    ) -> list[tuple[Position, str]]:
        """
        Check if any positions have hit SL/TP levels.

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            List of (position, trigger_type) tuples where trigger_type is 'stop_loss' or 'take_profit'
        """
        triggered: list[tuple[Position, str]] = []

        async with self._lock:
            for position in list(self._positions.values()):
                if position.status != PositionStatus.OPEN:
                    continue

                price = current_prices.get(position.symbol)
                if not price:
                    continue

                # Check stop-loss
                if position.stop_loss:
                    if position.side == PositionSide.LONG and price <= position.stop_loss:
                        triggered.append((position, "stop_loss"))
                        logger.warning(
                            f"SL triggered for {position.id}: price {price} <= SL {position.stop_loss}"
                        )
                        continue
                    elif position.side == PositionSide.SHORT and price >= position.stop_loss:
                        triggered.append((position, "stop_loss"))
                        logger.warning(
                            f"SL triggered for {position.id}: price {price} >= SL {position.stop_loss}"
                        )
                        continue

                # Check take-profit
                if position.take_profit:
                    if position.side == PositionSide.LONG and price >= position.take_profit:
                        triggered.append((position, "take_profit"))
                        logger.info(
                            f"TP triggered for {position.id}: price {price} >= TP {position.take_profit}"
                        )
                    elif position.side == PositionSide.SHORT and price <= position.take_profit:
                        triggered.append((position, "take_profit"))
                        logger.info(
                            f"TP triggered for {position.id}: price {price} <= TP {position.take_profit}"
                        )

        return triggered

    async def _process_sl_tp_triggers(self) -> None:
        """Process SL/TP triggers and close positions if needed."""
        if not self._price_cache:
            return

        triggered = await self.check_sl_tp_triggers(self._price_cache)

        for position, trigger_type in triggered:
            price = self._price_cache.get(position.symbol)
            if price:
                await self.close_position(
                    position_id=position.id,
                    exit_price=price,
                    reason=trigger_type,
                )

    async def update_trailing_stops(self, current_prices: dict[str, Decimal]) -> None:
        """
        Update trailing stops for all positions based on current prices.

        This method:
        1. Activates trailing stops when profit exceeds activation threshold
        2. Updates the trailing stop level as price moves in favorable direction
        3. Triggers stop when price retraces by trail distance

        Args:
            current_prices: Dict of symbol -> current price
        """
        if not self._trailing_stop_enabled:
            return

        async with self._lock:
            for position in list(self._positions.values()):
                if position.status != PositionStatus.OPEN:
                    continue

                if not position.trailing_stop_enabled:
                    continue

                price = current_prices.get(position.symbol)
                if not price or position.entry_price == 0:
                    continue

                # Calculate current profit percentage
                if position.side == PositionSide.LONG:
                    profit_pct = ((price - position.entry_price) / position.entry_price) * 100
                else:
                    profit_pct = ((position.entry_price - price) / position.entry_price) * 100

                # Check if trailing stop should be activated
                if not position.trailing_stop_activated:
                    if profit_pct >= self._trailing_stop_activation_pct:
                        position.trailing_stop_activated = True
                        if position.side == PositionSide.LONG:
                            position.trailing_stop_highest_price = price
                        else:
                            position.trailing_stop_lowest_price = price
                        logger.info(
                            f"Trailing stop activated for {position.id}: "
                            f"profit {float(profit_pct):.2f}% >= {float(self._trailing_stop_activation_pct)}%"
                        )
                    continue

                # Update trailing stop for active positions
                trail_distance = position.trailing_stop_distance_pct

                if position.side == PositionSide.LONG:
                    # Update highest price and calculate new stop
                    if price > (position.trailing_stop_highest_price or Decimal(0)):
                        position.trailing_stop_highest_price = price
                        new_stop = price * (1 - trail_distance / 100)
                        if position.stop_loss is None or new_stop > position.stop_loss:
                            old_stop = position.stop_loss
                            position.stop_loss = new_stop
                            logger.debug(
                                f"Trailing stop updated for {position.id}: "
                                f"SL {old_stop} -> {new_stop}"
                            )
                else:
                    # SHORT: Update lowest price and calculate new stop
                    if price < (position.trailing_stop_lowest_price or Decimal("inf")):
                        position.trailing_stop_lowest_price = price
                        new_stop = price * (1 + trail_distance / 100)
                        if position.stop_loss is None or new_stop < position.stop_loss:
                            old_stop = position.stop_loss
                            position.stop_loss = new_stop
                            logger.debug(
                                f"Trailing stop updated for {position.id}: "
                                f"SL {old_stop} -> {new_stop}"
                            )

    async def enable_trailing_stop_for_position(
        self,
        position_id: str,
        distance_pct: Optional[Decimal] = None,
    ) -> bool:
        """
        Enable trailing stop for a specific position.

        F07: Now async and thread-safe with proper locking.

        Args:
            position_id: Position ID
            distance_pct: Optional custom trail distance percentage

        Returns:
            True if successfully enabled
        """
        async with self._lock:
            position = self._positions.get(position_id)
            if not position:
                return False

            position.trailing_stop_enabled = True
            if distance_pct is not None:
                position.trailing_stop_distance_pct = distance_pct
            else:
                position.trailing_stop_distance_pct = self._trailing_stop_distance_pct

        logger.info(f"Trailing stop enabled for position {position_id}")
        return True

    async def update_order_links(
        self,
        position_id: str,
        stop_loss_order_id: Optional[str] = None,
        take_profit_order_id: Optional[str] = None,
    ) -> bool:
        """
        Update position with linked order IDs.

        F14: Links contingent orders to positions for tracking.

        Args:
            position_id: Position ID to update
            stop_loss_order_id: ID of the stop-loss order
            take_profit_order_id: ID of the take-profit order

        Returns:
            True if successfully updated
        """
        async with self._lock:
            position = self._positions.get(position_id)
            if not position:
                logger.warning(f"Position not found for order link update: {position_id}")
                return False

            if stop_loss_order_id:
                position.stop_loss_order_id = stop_loss_order_id
            if take_profit_order_id:
                position.take_profit_order_id = take_profit_order_id

        # Persist the update
        await self._update_position(position)
        logger.debug(f"Updated order links for position {position_id}: SL={stop_loss_order_id}, TP={take_profit_order_id}")
        return True

    async def sync_with_exchange(self, kraken_client) -> dict:
        """
        Synchronize local position state with Kraken exchange.

        F05: Detects positions that exist on exchange but not locally,
        and positions that exist locally but not on exchange.

        Args:
            kraken_client: Kraken API client

        Returns:
            dict with sync results: added, removed, updated counts, and alerts
        """
        if not kraken_client:
            return {"error": "No Kraken client available"}

        try:
            # Get open positions from Kraken
            result = await kraken_client.open_positions()
            if result.get("error"):
                logger.warning(f"Position sync error: {result['error']}")
                return {"error": result["error"]}

            exchange_positions = result.get("result", {})
            sync_result = {
                "added": 0,
                "removed": 0,
                "updated": 0,
                "alerts": [],
                "exchange_position_count": len(exchange_positions),
                "local_position_count": len(self._positions),
            }

            exchange_ids = set()

            for pos_id, pos_info in exchange_positions.items():
                exchange_ids.add(pos_id)

                # Check if we're tracking this position
                local_pos = self._find_position_by_external_id(pos_id)

                if not local_pos:
                    # New position on exchange we don't know about
                    logger.warning(f"Unknown exchange position detected: {pos_id}")
                    sync_result["alerts"].append({
                        "type": "unknown_position",
                        "severity": "high",
                        "position_id": pos_id,
                        "details": {
                            "pair": pos_info.get("pair"),
                            "type": pos_info.get("type"),
                            "vol": pos_info.get("vol"),
                            "cost": pos_info.get("cost"),
                        },
                        "message": f"Position {pos_id} exists on exchange but not tracked locally",
                    })
                    sync_result["added"] += 1
                else:
                    # Verify size matches
                    exchange_size = Decimal(str(pos_info.get("vol", 0)))
                    if abs(local_pos.size - exchange_size) > Decimal("0.0001"):
                        logger.warning(
                            f"Position size mismatch: local={local_pos.size}, "
                            f"exchange={exchange_size}"
                        )
                        sync_result["alerts"].append({
                            "type": "size_mismatch",
                            "severity": "medium",
                            "position_id": local_pos.id,
                            "external_id": pos_id,
                            "local_size": str(local_pos.size),
                            "exchange_size": str(exchange_size),
                        })
                        sync_result["updated"] += 1

            # Check for positions we have that exchange doesn't
            async with self._lock:
                for pos_id, local_pos in list(self._positions.items()):
                    if local_pos.external_id and local_pos.external_id not in exchange_ids:
                        logger.warning(f"Local position not on exchange: {pos_id}")
                        sync_result["alerts"].append({
                            "type": "missing_on_exchange",
                            "severity": "high",
                            "position_id": pos_id,
                            "external_id": local_pos.external_id,
                            "symbol": local_pos.symbol,
                            "message": "Position tracked locally but not found on exchange (may have been closed/liquidated)",
                        })
                        sync_result["removed"] += 1

            # Publish sync alerts if any issues found
            if sync_result["alerts"] and self.bus:
                from ..orchestration.message_bus import MessageTopic, create_message, MessagePriority
                await self.bus.publish(create_message(
                    topic=MessageTopic.RISK_ALERTS,
                    source="position_tracker",
                    payload={
                        "alert_type": "position_sync_discrepancy",
                        "severity": "high" if sync_result["alerts"] else "info",
                        "sync_result": sync_result,
                        "message": f"Position sync found {len(sync_result['alerts'])} discrepancies",
                    },
                    priority=MessagePriority.HIGH,
                ))

            return sync_result

        except Exception as e:
            logger.error(f"Exchange position sync failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _find_position_by_external_id(self, external_id: str) -> Optional[Position]:
        """Find position by its exchange/external ID."""
        for position in self._positions.values():
            if position.external_id == external_id:
                return position
        return None

    async def _snapshot_loop(self) -> None:
        """Background task to capture position snapshots (every 60s by default)."""
        while self._running:
            try:
                await asyncio.sleep(self._snapshot_interval_seconds)
                await self._capture_snapshots()
                # Update trailing stops (also checked in trigger loop, but update here too)
                await self.update_trailing_stops(self._price_cache)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot error: {e}")

    async def _trigger_check_loop(self) -> None:
        """
        F09: Separate fast loop for SL/TP trigger checks.

        Runs more frequently than snapshot loop (every 5s by default)
        to catch stop-loss and take-profit triggers quickly.
        """
        while self._running:
            try:
                await asyncio.sleep(self._trigger_check_interval_seconds)
                # Only process if we have price data
                if self._price_cache:
                    await self._process_sl_tp_triggers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trigger check error: {e}")

    async def _capture_snapshots(self) -> None:
        """Capture current state snapshots for all positions."""
        now = datetime.now(timezone.utc)

        async with self._lock:
            for position in self._positions.values():
                current_price = self._price_cache.get(position.symbol, position.entry_price)

                snapshot = PositionSnapshot(
                    position_id=position.id,
                    symbol=position.symbol,
                    timestamp=now,
                    current_price=current_price,
                    unrealized_pnl=position.unrealized_pnl,
                    unrealized_pnl_pct=position.unrealized_pnl_pct,
                )

                self._snapshots.append(snapshot)

            # Trim snapshots if too many
            if len(self._snapshots) > self._max_snapshots:
                self._snapshots = self._snapshots[-self._max_snapshots:]

        # Store snapshots to database
        await self._store_snapshots()

    async def _update_risk_exposure(self) -> None:
        """Update risk engine with current exposure."""
        if not self.risk_engine:
            return

        exposure = await self.get_total_exposure()
        open_symbols = list(set(p.symbol for p in self._positions.values()))

        # Calculate exposure percentages
        exposures: dict[str, float] = {}
        total = exposure["total_exposure_usd"]
        for symbol, value in exposure["exposure_by_symbol"].items():
            exposures[symbol] = (value / total * 100) if total > 0 else 0

        self.risk_engine.update_positions(open_symbols, exposures)

    async def _load_positions(self) -> None:
        """Load open positions from database on startup."""
        if not self.db:
            return

        try:
            query = """
                SELECT id, symbol, side, size, entry_price, leverage, status,
                       order_id, stop_loss, take_profit, opened_at, notes
                FROM positions
                WHERE status = 'open'
            """
            rows = await self.db.fetch(query)

            for row in rows:
                position = Position(
                    id=str(row['id']),
                    symbol=row['symbol'],
                    side=PositionSide(row['side']),
                    size=Decimal(str(row['size'])),
                    entry_price=Decimal(str(row['entry_price'])),
                    leverage=row['leverage'],
                    status=PositionStatus(row['status']),
                    order_id=row.get('order_id'),
                    stop_loss=Decimal(str(row['stop_loss'])) if row.get('stop_loss') else None,
                    take_profit=Decimal(str(row['take_profit'])) if row.get('take_profit') else None,
                    opened_at=row['opened_at'],
                    notes=row.get('notes', ''),
                )
                self._positions[position.id] = position

            logger.info(f"Loaded {len(self._positions)} open positions from database")

        except Exception as e:
            logger.warning(f"Failed to load positions: {e}")

    async def _store_position(self, position: Position) -> None:
        """Store position to database using string representation for Decimal precision."""
        if not self.db:
            return

        try:
            query = """
                INSERT INTO positions (
                    id, symbol, side, size, entry_price, leverage, status,
                    order_id, stop_loss, take_profit, opened_at, notes
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """
            await self.db.execute(
                query,
                uuid.UUID(position.id),
                position.symbol,
                position.side.value,
                str(position.size),  # Use str for Decimal precision
                str(position.entry_price),  # Use str for Decimal precision
                position.leverage,
                position.status.value,
                position.order_id,
                str(position.stop_loss) if position.stop_loss else None,
                str(position.take_profit) if position.take_profit else None,
                position.opened_at,
                position.notes,
            )
        except Exception as e:
            logger.error(f"Failed to store position: {e}")

    async def _update_position(self, position: Position) -> None:
        """Update position in database using string representation for Decimal precision."""
        if not self.db:
            return

        try:
            query = """
                UPDATE positions SET
                    status = $2,
                    stop_loss = $3,
                    take_profit = $4,
                    closed_at = $5,
                    exit_price = $6,
                    realized_pnl = $7
                WHERE id = $1
            """
            await self.db.execute(
                query,
                uuid.UUID(position.id),
                position.status.value,
                str(position.stop_loss) if position.stop_loss else None,
                str(position.take_profit) if position.take_profit else None,
                position.closed_at,
                str(position.exit_price) if position.exit_price else None,
                str(position.realized_pnl),  # Use str for Decimal precision
            )
        except Exception as e:
            logger.error(f"Failed to update position: {e}")

    async def _store_snapshots(self) -> None:
        """Store position snapshots to database using string representation for Decimal precision."""
        if not self.db or not self._snapshots:
            return

        # Only store recent snapshots (last batch)
        recent_snapshots = self._snapshots[-len(self._positions):]

        try:
            for snapshot in recent_snapshots:
                query = """
                    INSERT INTO position_snapshots (
                        timestamp, position_id, symbol, current_price,
                        unrealized_pnl, unrealized_pnl_pct
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """
                await self.db.execute(
                    query,
                    snapshot.timestamp,
                    uuid.UUID(snapshot.position_id),
                    snapshot.symbol,
                    str(snapshot.current_price),  # Use str for Decimal precision
                    str(snapshot.unrealized_pnl),  # Use str for Decimal precision
                    str(snapshot.unrealized_pnl_pct),  # Use str for Decimal precision
                )
        except Exception as e:
            logger.error(f"Failed to store snapshots: {e}")

    def get_stats(self) -> dict:
        """Get position tracking statistics."""
        return {
            "total_positions_opened": self._total_positions_opened,
            "total_positions_closed": self._total_positions_closed,
            "open_positions_count": len(self._positions),
            "total_realized_pnl": float(self._total_realized_pnl),
            "snapshots_count": len(self._snapshots),
        }
