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
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    exit_price: Optional[Decimal] = None

    # P&L tracking
    realized_pnl: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)
    unrealized_pnl_pct: Decimal = Decimal(0)

    # Position metadata
    notes: str = ""
    tags: list[str] = field(default_factory=list)

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
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "unrealized_pnl_pct": str(self.unrealized_pnl_pct),
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
            stop_loss=Decimal(data["stop_loss"]) if data.get("stop_loss") else None,
            take_profit=Decimal(data["take_profit"]) if data.get("take_profit") else None,
            opened_at=datetime.fromisoformat(data["opened_at"]) if data.get("opened_at") else datetime.now(timezone.utc),
            closed_at=datetime.fromisoformat(data["closed_at"]) if data.get("closed_at") else None,
            exit_price=Decimal(data["exit_price"]) if data.get("exit_price") else None,
            realized_pnl=Decimal(data.get("realized_pnl", "0")),
            unrealized_pnl=Decimal(data.get("unrealized_pnl", "0")),
            unrealized_pnl_pct=Decimal(data.get("unrealized_pnl_pct", "0")),
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

        # Price cache for P&L calculations
        self._price_cache: dict[str, Decimal] = {}

        # Background task
        self._snapshot_task: Optional[asyncio.Task] = None
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
    ) -> Optional[Position]:
        """
        Close an open position.

        Args:
            position_id: Position ID to close
            exit_price: Exit/closing price
            reason: Reason for closing

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
        return position

    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Optional[Position]:
        """
        Modify position stop-loss or take-profit.

        Args:
            position_id: Position ID to modify
            stop_loss: New stop-loss price
            take_profit: New take-profit price

        Returns:
            Modified position or None if not found
        """
        async with self._lock:
            position = self._positions.get(position_id)

            if not position:
                logger.warning(f"Position not found: {position_id}")
                return None

            if stop_loss is not None:
                position.stop_loss = stop_loss

            if take_profit is not None:
                position.take_profit = take_profit

        # Persist to database
        await self._update_position(position)

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

    async def _snapshot_loop(self) -> None:
        """Background task to capture position snapshots."""
        while self._running:
            try:
                await asyncio.sleep(self._snapshot_interval_seconds)
                await self._capture_snapshots()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot error: {e}")

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
        """Store position to database."""
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
                float(position.size),
                float(position.entry_price),
                position.leverage,
                position.status.value,
                position.order_id,
                float(position.stop_loss) if position.stop_loss else None,
                float(position.take_profit) if position.take_profit else None,
                position.opened_at,
                position.notes,
            )
        except Exception as e:
            logger.error(f"Failed to store position: {e}")

    async def _update_position(self, position: Position) -> None:
        """Update position in database."""
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
                float(position.stop_loss) if position.stop_loss else None,
                float(position.take_profit) if position.take_profit else None,
                position.closed_at,
                float(position.exit_price) if position.exit_price else None,
                float(position.realized_pnl),
            )
        except Exception as e:
            logger.error(f"Failed to update position: {e}")

    async def _store_snapshots(self) -> None:
        """Store position snapshots to database."""
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
                    float(snapshot.current_price),
                    float(snapshot.unrealized_pnl),
                    float(snapshot.unrealized_pnl_pct),
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
