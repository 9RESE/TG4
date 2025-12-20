"""
Hodl Bag Manager - Automated profit allocation for long-term accumulation.

Phase 8: Hodl Bag System

Features:
- Automatic 10% profit allocation to hodl bags
- 1/3 split across USDT, XRP, BTC
- Per-asset purchase thresholds
- Paper trading mode support
- Excluded from rebalancing and trading
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..orchestration.message_bus import MessageBus

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Hodl bag transaction types."""
    ACCUMULATION = "accumulation"
    WITHDRAWAL = "withdrawal"
    ADJUSTMENT = "adjustment"


@dataclass
class HodlBagState:
    """Current state of a hodl bag for a single asset."""
    asset: str
    balance: Decimal
    cost_basis_usd: Decimal
    current_value_usd: Optional[Decimal] = None
    unrealized_pnl_usd: Optional[Decimal] = None
    unrealized_pnl_pct: Optional[Decimal] = None
    first_accumulation: Optional[datetime] = None
    last_accumulation: Optional[datetime] = None
    pending_usd: Decimal = Decimal(0)
    accumulation_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "asset": self.asset,
            "balance": str(self.balance),
            "cost_basis_usd": str(self.cost_basis_usd),
            "current_value_usd": str(self.current_value_usd) if self.current_value_usd else None,
            "unrealized_pnl_usd": str(self.unrealized_pnl_usd) if self.unrealized_pnl_usd else None,
            "unrealized_pnl_pct": str(self.unrealized_pnl_pct) if self.unrealized_pnl_pct else None,
            "first_accumulation": self.first_accumulation.isoformat() if self.first_accumulation else None,
            "last_accumulation": self.last_accumulation.isoformat() if self.last_accumulation else None,
            "pending_usd": str(self.pending_usd),
            "accumulation_count": self.accumulation_count,
        }


@dataclass
class HodlAllocation:
    """Allocation from a profitable trade."""
    trade_id: str
    profit_usd: Decimal
    total_allocation_usd: Decimal
    usdt_amount_usd: Decimal
    xrp_amount_usd: Decimal
    btc_amount_usd: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "trade_id": self.trade_id,
            "profit_usd": str(self.profit_usd),
            "total_allocation_usd": str(self.total_allocation_usd),
            "usdt_amount_usd": str(self.usdt_amount_usd),
            "xrp_amount_usd": str(self.xrp_amount_usd),
            "btc_amount_usd": str(self.btc_amount_usd),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HodlTransaction:
    """A single hodl bag transaction record."""
    id: str
    timestamp: datetime
    asset: str
    transaction_type: TransactionType
    amount: Decimal  # Asset amount
    price_usd: Decimal
    value_usd: Decimal
    source_trade_id: Optional[str] = None
    order_id: Optional[str] = None
    fee_usd: Decimal = Decimal(0)
    notes: Optional[str] = None
    is_paper: bool = False

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "transaction_type": self.transaction_type.value,
            "amount": str(self.amount),
            "price_usd": str(self.price_usd),
            "value_usd": str(self.value_usd),
            "source_trade_id": self.source_trade_id,
            "order_id": self.order_id,
            "fee_usd": str(self.fee_usd),
            "notes": self.notes,
            "is_paper": self.is_paper,
        }


@dataclass
class HodlPending:
    """Pending hodl accumulation."""
    id: str
    asset: str
    amount_usd: Decimal
    source_trade_id: str
    source_profit_usd: Decimal
    created_at: datetime
    executed_at: Optional[datetime] = None
    execution_transaction_id: Optional[str] = None
    is_paper: bool = False

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "asset": self.asset,
            "amount_usd": str(self.amount_usd),
            "source_trade_id": self.source_trade_id,
            "source_profit_usd": str(self.source_profit_usd),
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "execution_transaction_id": self.execution_transaction_id,
            "is_paper": self.is_paper,
        }


@dataclass
class HodlThresholds:
    """Per-asset accumulation thresholds."""
    usdt: Decimal = Decimal("1")
    xrp: Decimal = Decimal("25")
    btc: Decimal = Decimal("15")

    def get(self, asset: str) -> Decimal:
        """Get threshold for asset."""
        return getattr(self, asset.lower(), Decimal("25"))

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "usdt": str(self.usdt),
            "xrp": str(self.xrp),
            "btc": str(self.btc),
        }


# Symbol mapping for Kraken
HODL_SYMBOL_MAP = {
    "BTC": "BTC/USDT",
    "XRP": "XRP/USDT",
}


class HodlBagManager:
    """
    Manages hodl bag accumulation from trading profits.

    Features:
    - Process trade profits for hodl allocation (10% default)
    - Split across USDT/XRP/BTC (33.33% each)
    - Track pending accumulation per asset
    - Execute purchases when threshold reached
    - Paper trading mode support
    - Exclude hodl balances from trading capital
    """

    def __init__(
        self,
        config: dict,
        db_pool=None,
        kraken_client=None,
        price_source: Optional[Callable[[str], Optional[Decimal]]] = None,
        message_bus: Optional['MessageBus'] = None,
        is_paper_mode: bool = True,
    ):
        """
        Initialize HodlBagManager.

        Args:
            config: Hodl bag configuration from hodl.yaml
            db_pool: Database pool for persistence
            kraken_client: Kraken API client for live trading
            price_source: Function to get current price for a symbol
            message_bus: MessageBus for event publishing
            is_paper_mode: Whether running in paper trading mode
        """
        self.config = config
        self.db = db_pool
        self.kraken = kraken_client
        self.get_price = price_source
        self.bus = message_bus
        self.is_paper_mode = is_paper_mode

        # Load configuration
        hodl_config = config.get('hodl_bags', config)
        self.enabled = hodl_config.get('enabled', True)

        # Allocation settings
        self.allocation_pct = Decimal(str(hodl_config.get('allocation_pct', 10)))

        # Split percentages
        split = hodl_config.get('split', {})
        self.usdt_pct = Decimal(str(split.get('usdt_pct', 33.34)))
        self.xrp_pct = Decimal(str(split.get('xrp_pct', 33.33)))
        self.btc_pct = Decimal(str(split.get('btc_pct', 33.33)))

        # Thresholds
        min_acc = hodl_config.get('min_accumulation', {})
        self.thresholds = HodlThresholds(
            usdt=Decimal(str(min_acc.get('usdt', 1))),
            xrp=Decimal(str(min_acc.get('xrp', 25))),
            btc=Decimal(str(min_acc.get('btc', 15))),
        )

        # Execution settings
        execution = hodl_config.get('execution', {})
        self.order_type = execution.get('order_type', 'market')
        self.max_retries = execution.get('max_retries', 3)
        self.retry_delay_seconds = execution.get('retry_delay_seconds', 30)

        # Safety limits
        limits = hodl_config.get('limits', {})
        self.max_single_accumulation_usd = Decimal(str(limits.get('max_single_accumulation_usd', 1000)))
        self.daily_accumulation_limit_usd = Decimal(str(limits.get('daily_accumulation_limit_usd', 5000)))
        self.min_profit_to_allocate_usd = Decimal(str(limits.get('min_profit_to_allocate_usd', 1.0)))

        # In-memory state (loaded from DB on start)
        self._hodl_bags: Dict[str, HodlBagState] = {}
        self._pending: Dict[str, Decimal] = {"USDT": Decimal(0), "XRP": Decimal(0), "BTC": Decimal(0)}
        self._daily_accumulated_usd = Decimal(0)
        self._daily_reset_date: Optional[datetime] = None

        # Price cache
        self._price_cache: Dict[str, Decimal] = {}
        self._price_cache_time: Optional[datetime] = None

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Statistics
        self._total_allocations = 0
        self._total_allocated_usd = Decimal(0)
        self._total_executions = 0

        logger.info(
            f"HodlBagManager initialized: allocation={self.allocation_pct}%, "
            f"split=USDT:{self.usdt_pct}%/XRP:{self.xrp_pct}%/BTC:{self.btc_pct}%, "
            f"paper_mode={is_paper_mode}"
        )

    async def start(self) -> None:
        """Start the hodl bag manager and load state from database."""
        if not self.enabled:
            logger.info("HodlBagManager disabled")
            return

        await self._load_state()
        logger.info("HodlBagManager started")

    async def stop(self) -> None:
        """Stop the hodl bag manager."""
        logger.info("HodlBagManager stopped")

    async def process_trade_profit(
        self,
        trade_id: str,
        profit_usd: Decimal,
        source_symbol: str,
    ) -> Optional[HodlAllocation]:
        """
        Process a profitable trade for hodl allocation.

        Called by PositionTracker when a position closes with profit.

        Args:
            trade_id: Unique identifier for the trade
            profit_usd: Realized profit in USD
            source_symbol: Trading symbol (e.g., "BTC/USDT")

        Returns:
            HodlAllocation if allocation was made, None otherwise
        """
        if not self.enabled:
            logger.debug("Hodl bags disabled, skipping allocation")
            return None

        if profit_usd <= 0:
            logger.debug(f"No profit to allocate (profit: ${float(profit_usd):.2f})")
            return None

        if profit_usd < self.min_profit_to_allocate_usd:
            logger.debug(
                f"Profit ${float(profit_usd):.2f} below minimum "
                f"${float(self.min_profit_to_allocate_usd):.2f}"
            )
            return None

        async with self._lock:
            # Check daily limit
            self._check_daily_reset()
            if self._daily_accumulated_usd >= self.daily_accumulation_limit_usd:
                logger.warning(
                    f"Daily accumulation limit reached: ${float(self._daily_accumulated_usd):.2f} "
                    f">= ${float(self.daily_accumulation_limit_usd):.2f}"
                )
                return None

            # Calculate allocation
            allocation = self._calculate_allocation(trade_id, profit_usd)

            # Apply single accumulation cap
            if allocation.total_allocation_usd > self.max_single_accumulation_usd:
                # Scale down proportionally
                scale = self.max_single_accumulation_usd / allocation.total_allocation_usd
                allocation = HodlAllocation(
                    trade_id=trade_id,
                    profit_usd=profit_usd,
                    total_allocation_usd=self.max_single_accumulation_usd,
                    usdt_amount_usd=(allocation.usdt_amount_usd * scale).quantize(Decimal("0.01")),
                    xrp_amount_usd=(allocation.xrp_amount_usd * scale).quantize(Decimal("0.01")),
                    btc_amount_usd=(allocation.btc_amount_usd * scale).quantize(Decimal("0.01")),
                )
                logger.info(f"Allocation capped to ${float(self.max_single_accumulation_usd):.2f}")

            # Record pending allocations
            await self._record_pending(allocation)

            # Update daily total
            self._daily_accumulated_usd += allocation.total_allocation_usd

            # Update statistics
            self._total_allocations += 1
            self._total_allocated_usd += allocation.total_allocation_usd

        logger.info(
            f"Hodl allocation from trade {trade_id}: "
            f"${float(allocation.total_allocation_usd):.2f} "
            f"(USDT: ${float(allocation.usdt_amount_usd):.2f}, "
            f"XRP: ${float(allocation.xrp_amount_usd):.2f}, "
            f"BTC: ${float(allocation.btc_amount_usd):.2f})"
        )

        # Check thresholds and execute if reached
        for asset in ["USDT", "XRP", "BTC"]:
            await self._check_and_execute(asset)

        # Publish event
        if self.bus:
            from ..orchestration.message_bus import MessageTopic, create_message
            await self.bus.publish(create_message(
                topic=MessageTopic.PORTFOLIO_UPDATES,
                source="hodl_bag_manager",
                payload={
                    "event_type": "hodl_allocation",
                    "allocation": allocation.to_dict(),
                },
            ))

        return allocation

    def _calculate_allocation(self, trade_id: str, profit_usd: Decimal) -> HodlAllocation:
        """Calculate hodl allocation from profit."""
        # Total allocation (10% of profit)
        total = (profit_usd * self.allocation_pct / Decimal(100)).quantize(Decimal("0.01"))

        # Split across assets
        usdt = (total * self.usdt_pct / Decimal(100)).quantize(Decimal("0.01"))
        xrp = (total * self.xrp_pct / Decimal(100)).quantize(Decimal("0.01"))
        btc = (total * self.btc_pct / Decimal(100)).quantize(Decimal("0.01"))

        # Ensure total matches (put any rounding difference in USDT)
        actual_total = usdt + xrp + btc
        if actual_total != total:
            usdt += (total - actual_total)

        return HodlAllocation(
            trade_id=trade_id,
            profit_usd=profit_usd,
            total_allocation_usd=total,
            usdt_amount_usd=usdt,
            xrp_amount_usd=xrp,
            btc_amount_usd=btc,
        )

    async def _record_pending(self, allocation: HodlAllocation) -> None:
        """Record pending allocations to database and in-memory state."""
        now = datetime.now(timezone.utc)

        # Record USDT (held immediately, but track as pending for consistency)
        if allocation.usdt_amount_usd > 0:
            self._pending["USDT"] += allocation.usdt_amount_usd
            await self._store_pending("USDT", allocation.usdt_amount_usd, allocation)

        # Record XRP pending
        if allocation.xrp_amount_usd > 0:
            self._pending["XRP"] += allocation.xrp_amount_usd
            await self._store_pending("XRP", allocation.xrp_amount_usd, allocation)

        # Record BTC pending
        if allocation.btc_amount_usd > 0:
            self._pending["BTC"] += allocation.btc_amount_usd
            await self._store_pending("BTC", allocation.btc_amount_usd, allocation)

    async def _store_pending(
        self,
        asset: str,
        amount_usd: Decimal,
        allocation: HodlAllocation,
    ) -> None:
        """Store pending record to database."""
        if not self.db:
            return

        try:
            query = """
                INSERT INTO hodl_pending (
                    id, asset, amount_usd, source_trade_id, source_profit_usd,
                    created_at, is_paper
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            await self.db.execute(
                query,
                uuid.uuid4(),
                asset,
                float(amount_usd),
                uuid.UUID(allocation.trade_id) if allocation.trade_id else None,
                float(allocation.profit_usd),
                allocation.timestamp,
                self.is_paper_mode,
            )
        except Exception as e:
            logger.error(f"Failed to store pending: {e}")

    async def _check_and_execute(self, asset: str) -> Optional[Decimal]:
        """Check if threshold reached and execute accumulation."""
        threshold = self.thresholds.get(asset)
        pending = self._pending.get(asset, Decimal(0))

        if pending < threshold:
            return None

        return await self.execute_accumulation(asset)

    async def execute_accumulation(self, asset: str) -> Optional[Decimal]:
        """
        Execute pending accumulation for an asset.

        For USDT: Just records the transaction (no purchase needed)
        For XRP/BTC: Places market buy order or simulates in paper mode

        Args:
            asset: Asset to accumulate (BTC, XRP, USDT)

        Returns:
            Amount of asset accumulated, or None if failed
        """
        async with self._lock:
            pending_usd = self._pending.get(asset, Decimal(0))

            if pending_usd <= 0:
                logger.debug(f"No pending {asset} to execute")
                return None

            threshold = self.thresholds.get(asset)
            if pending_usd < threshold:
                logger.debug(
                    f"{asset} pending ${float(pending_usd):.2f} "
                    f"below threshold ${float(threshold):.2f}"
                )
                return None

            try:
                if asset == "USDT":
                    # USDT is just held, no purchase needed
                    amount = pending_usd
                    price = Decimal(1)
                    order_id = None
                else:
                    # Get current price
                    symbol = HODL_SYMBOL_MAP.get(asset)
                    if not symbol:
                        logger.error(f"Unknown asset: {asset}")
                        return None

                    price = await self._get_current_price(symbol)
                    if not price or price <= 0:
                        logger.error(f"Could not get price for {symbol}")
                        return None

                    # Calculate amount
                    amount = (pending_usd / price).quantize(
                        Decimal("0.00000001") if asset == "BTC" else Decimal("0.000001"),
                        rounding=ROUND_DOWN
                    )

                    # Execute purchase
                    order_id = await self._execute_purchase(asset, symbol, amount, price, pending_usd)
                    if not order_id and not self.is_paper_mode:
                        logger.error(f"Failed to execute {asset} purchase")
                        return None

                # Record transaction
                transaction = await self._record_transaction(
                    asset=asset,
                    amount=amount,
                    price_usd=price,
                    value_usd=pending_usd,
                    order_id=order_id,
                )

                # Update hodl bag state
                await self._update_hodl_bag(asset, amount, pending_usd)

                # Mark pending as executed
                await self._mark_pending_executed(asset, transaction.id if transaction else None)

                # Clear pending
                self._pending[asset] = Decimal(0)

                # Update statistics
                self._total_executions += 1

                logger.info(
                    f"Hodl {asset} executed: {float(amount)} @ ${float(price):.2f} "
                    f"= ${float(pending_usd):.2f}"
                )

                # Publish event
                if self.bus:
                    from ..orchestration.message_bus import MessageTopic, create_message, MessagePriority
                    await self.bus.publish(create_message(
                        topic=MessageTopic.PORTFOLIO_UPDATES,
                        source="hodl_bag_manager",
                        payload={
                            "event_type": "hodl_execution",
                            "asset": asset,
                            "amount": str(amount),
                            "price_usd": str(price),
                            "value_usd": str(pending_usd),
                            "order_id": order_id,
                            "is_paper": self.is_paper_mode,
                        },
                        priority=MessagePriority.NORMAL,
                    ))

                return amount

            except Exception as e:
                logger.error(f"Hodl accumulation failed for {asset}: {e}", exc_info=True)
                return None

    async def _execute_purchase(
        self,
        asset: str,
        symbol: str,
        amount: Decimal,
        price: Decimal,
        value_usd: Decimal,
    ) -> Optional[str]:
        """Execute purchase on exchange or simulate in paper mode."""
        if self.is_paper_mode:
            # Paper mode - simulate purchase
            order_id = f"paper_hodl_{uuid.uuid4().hex[:8]}"
            logger.info(f"Paper hodl purchase: {symbol} {float(amount)} @ ${float(price):.2f}")
            return order_id

        # Live mode - place actual order
        if not self.kraken:
            logger.error("No Kraken client for live hodl purchase")
            return None

        try:
            # Convert to Kraken symbol format
            kraken_symbol = self._to_kraken_symbol(symbol)

            # Place market buy order
            result = await self.kraken.add_order(
                pair=kraken_symbol,
                type="buy",
                ordertype=self.order_type,
                volume=str(amount),
            )

            if result.get("error"):
                logger.error(f"Kraken order error: {result['error']}")
                return None

            # Get order ID
            txids = result.get("result", {}).get("txid", [])
            if txids:
                order_id = txids[0]
                logger.info(f"Hodl purchase placed: {order_id}")

                # Wait for fill
                await self._wait_for_fill(order_id)
                return order_id

            return None

        except Exception as e:
            logger.error(f"Failed to execute hodl purchase: {e}")
            return None

    async def _wait_for_fill(self, order_id: str, timeout_seconds: int = 60) -> bool:
        """Wait for order to fill."""
        if not self.kraken:
            return False

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout_seconds:
            try:
                result = await self.kraken.query_orders(txid=order_id)
                if result.get("error"):
                    logger.warning(f"Order query error: {result['error']}")
                    await asyncio.sleep(2)
                    continue

                order_info = result.get("result", {}).get(order_id, {})
                status = order_info.get("status", "")

                if status == "closed":
                    return True
                elif status in ["canceled", "expired"]:
                    logger.warning(f"Hodl order {order_id} {status}")
                    return False

                await asyncio.sleep(2)

            except Exception as e:
                logger.warning(f"Error waiting for fill: {e}")
                await asyncio.sleep(2)

        logger.warning(f"Timeout waiting for hodl order {order_id}")
        return False

    async def _record_transaction(
        self,
        asset: str,
        amount: Decimal,
        price_usd: Decimal,
        value_usd: Decimal,
        order_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[HodlTransaction]:
        """Record transaction to database."""
        transaction = HodlTransaction(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            transaction_type=TransactionType.ACCUMULATION,
            amount=amount,
            price_usd=price_usd,
            value_usd=value_usd,
            order_id=order_id,
            notes=notes,
            is_paper=self.is_paper_mode,
        )

        if not self.db:
            return transaction

        try:
            query = """
                INSERT INTO hodl_transactions (
                    id, timestamp, asset, transaction_type, amount,
                    price_usd, value_usd, order_id, notes, is_paper
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            await self.db.execute(
                query,
                uuid.UUID(transaction.id),
                transaction.timestamp,
                transaction.asset,
                transaction.transaction_type.value,
                float(transaction.amount),
                float(transaction.price_usd),
                float(transaction.value_usd),
                transaction.order_id,
                transaction.notes,
                transaction.is_paper,
            )
            return transaction

        except Exception as e:
            logger.error(f"Failed to record transaction: {e}")
            return transaction

    async def _update_hodl_bag(
        self,
        asset: str,
        amount: Decimal,
        value_usd: Decimal,
    ) -> None:
        """Update hodl bag balance and cost basis."""
        now = datetime.now(timezone.utc)

        # Update in-memory state
        if asset not in self._hodl_bags:
            self._hodl_bags[asset] = HodlBagState(
                asset=asset,
                balance=Decimal(0),
                cost_basis_usd=Decimal(0),
            )

        bag = self._hodl_bags[asset]
        bag.balance += amount
        bag.cost_basis_usd += value_usd
        bag.last_accumulation = now
        bag.accumulation_count += 1

        if not bag.first_accumulation:
            bag.first_accumulation = now

        # Persist to database
        if not self.db:
            return

        try:
            query = """
                UPDATE hodl_bags SET
                    balance = balance + $2,
                    cost_basis_usd = cost_basis_usd + $3,
                    first_accumulation = COALESCE(first_accumulation, $4),
                    last_accumulation = $4,
                    updated_at = NOW()
                WHERE asset = $1
            """
            await self.db.execute(
                query,
                asset,
                float(amount),
                float(value_usd),
                now,
            )
        except Exception as e:
            logger.error(f"Failed to update hodl bag: {e}")

    async def _mark_pending_executed(
        self,
        asset: str,
        transaction_id: Optional[str],
    ) -> None:
        """Mark all pending for asset as executed."""
        if not self.db:
            return

        try:
            query = """
                UPDATE hodl_pending SET
                    executed_at = NOW(),
                    execution_transaction_id = $2
                WHERE asset = $1 AND executed_at IS NULL
            """
            await self.db.execute(
                query,
                asset,
                uuid.UUID(transaction_id) if transaction_id else None,
            )
        except Exception as e:
            logger.error(f"Failed to mark pending executed: {e}")

    async def force_accumulation(self, asset: str) -> bool:
        """
        Force immediate accumulation regardless of threshold.

        Args:
            asset: Asset to accumulate (BTC, XRP, USDT)

        Returns:
            True if accumulation was executed
        """
        pending = self._pending.get(asset, Decimal(0))
        if pending <= 0:
            logger.info(f"No pending {asset} to force accumulate")
            return False

        result = await self.execute_accumulation(asset)
        return result is not None

    async def get_hodl_state(self) -> Dict[str, HodlBagState]:
        """Get current state of all hodl bags."""
        # Update valuations
        await self._update_valuations()

        async with self._lock:
            return dict(self._hodl_bags)

    async def get_pending(self, asset: Optional[str] = None) -> Dict[str, Decimal]:
        """Get pending accumulation amounts."""
        async with self._lock:
            if asset:
                return {asset: self._pending.get(asset, Decimal(0))}
            return dict(self._pending)

    async def get_transaction_history(
        self,
        asset: Optional[str] = None,
        limit: int = 100,
    ) -> List[HodlTransaction]:
        """Get transaction history."""
        if not self.db:
            return []

        try:
            if asset:
                query = """
                    SELECT * FROM hodl_transactions
                    WHERE asset = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                """
                rows = await self.db.fetch(query, asset, limit)
            else:
                query = """
                    SELECT * FROM hodl_transactions
                    ORDER BY timestamp DESC
                    LIMIT $1
                """
                rows = await self.db.fetch(query, limit)

            return [
                HodlTransaction(
                    id=str(row['id']),
                    timestamp=row['timestamp'],
                    asset=row['asset'],
                    transaction_type=TransactionType(row['transaction_type']),
                    amount=Decimal(str(row['amount'])),
                    price_usd=Decimal(str(row['price_usd'])),
                    value_usd=Decimal(str(row['value_usd'])),
                    source_trade_id=str(row['source_trade_id']) if row.get('source_trade_id') else None,
                    order_id=row.get('order_id'),
                    fee_usd=Decimal(str(row.get('fee_usd', 0))),
                    notes=row.get('notes'),
                    is_paper=row.get('is_paper', False),
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get transaction history: {e}")
            return []

    async def calculate_metrics(self) -> dict:
        """Calculate hodl bag performance metrics."""
        await self._update_valuations()

        async with self._lock:
            total_cost_basis = Decimal(0)
            total_current_value = Decimal(0)
            total_balance_by_asset = {}

            for asset, bag in self._hodl_bags.items():
                total_cost_basis += bag.cost_basis_usd
                total_balance_by_asset[asset] = bag.balance

                if bag.current_value_usd:
                    total_current_value += bag.current_value_usd

            unrealized_pnl = total_current_value - total_cost_basis
            unrealized_pnl_pct = (
                (unrealized_pnl / total_cost_basis * 100)
                if total_cost_basis > 0
                else Decimal(0)
            )

            return {
                "total_cost_basis_usd": float(total_cost_basis),
                "total_current_value_usd": float(total_current_value),
                "unrealized_pnl_usd": float(unrealized_pnl),
                "unrealized_pnl_pct": float(unrealized_pnl_pct),
                "balances": {k: str(v) for k, v in total_balance_by_asset.items()},
                "pending_usd": {k: str(v) for k, v in self._pending.items()},
                "total_allocations": self._total_allocations,
                "total_allocated_usd": float(self._total_allocated_usd),
                "total_executions": self._total_executions,
                "thresholds": self.thresholds.to_dict(),
            }

    async def _update_valuations(self) -> None:
        """Update current valuations for all hodl bags."""
        for asset, bag in self._hodl_bags.items():
            if bag.balance <= 0:
                bag.current_value_usd = Decimal(0)
                bag.unrealized_pnl_usd = Decimal(0)
                bag.unrealized_pnl_pct = Decimal(0)
                continue

            if asset == "USDT":
                price = Decimal(1)
            else:
                symbol = HODL_SYMBOL_MAP.get(asset)
                if symbol:
                    price = await self._get_current_price(symbol)
                else:
                    price = None

            if price:
                bag.current_value_usd = bag.balance * price
                bag.unrealized_pnl_usd = bag.current_value_usd - bag.cost_basis_usd
                if bag.cost_basis_usd > 0:
                    bag.unrealized_pnl_pct = (bag.unrealized_pnl_usd / bag.cost_basis_usd * 100)
                else:
                    bag.unrealized_pnl_pct = Decimal(0)

    async def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol."""
        # Check price source function first
        if self.get_price:
            price = self.get_price(symbol)
            if price:
                return price

        # Check cache (5 second TTL)
        now = datetime.now(timezone.utc)
        if self._price_cache_time:
            age = (now - self._price_cache_time).total_seconds()
            if age < 5 and symbol in self._price_cache:
                return self._price_cache[symbol]

        # Try Kraken API
        if self.kraken:
            try:
                kraken_symbol = self._to_kraken_symbol(symbol)
                result = await self.kraken.get_ticker(kraken_symbol)

                if result and not result.get("error"):
                    pair_data = list(result.get("result", {}).values())[0] if result.get("result") else {}
                    if "c" in pair_data:
                        price = Decimal(pair_data["c"][0])
                        self._price_cache[symbol] = price
                        self._price_cache_time = now
                        return price

            except Exception as e:
                logger.debug(f"Failed to get price for {symbol}: {e}")

        # Fallback prices for paper trading
        fallback_prices = self.config.get('hodl_bags', {}).get('prices', {}).get('fallback', {})
        if symbol in fallback_prices:
            return Decimal(str(fallback_prices[symbol]))

        return None

    def _to_kraken_symbol(self, symbol: str) -> str:
        """Convert internal symbol to Kraken format."""
        mapping = {
            "BTC/USDT": "XBTUSDT",
            "XRP/USDT": "XRPUSDT",
        }
        return mapping.get(symbol, symbol.replace("/", ""))

    def _check_daily_reset(self) -> None:
        """Reset daily accumulation counter if new day."""
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if self._daily_reset_date != today:
            self._daily_accumulated_usd = Decimal(0)
            self._daily_reset_date = today

    async def _load_state(self) -> None:
        """Load hodl bag state from database."""
        if not self.db:
            # Initialize empty state
            for asset in ["USDT", "XRP", "BTC"]:
                self._hodl_bags[asset] = HodlBagState(
                    asset=asset,
                    balance=Decimal(0),
                    cost_basis_usd=Decimal(0),
                )
            return

        try:
            # Load hodl bags
            query = """
                SELECT asset, balance, cost_basis_usd, first_accumulation,
                       last_accumulation, last_valuation_usd, last_valuation_timestamp
                FROM hodl_bags
            """
            rows = await self.db.fetch(query)

            for row in rows:
                self._hodl_bags[row['asset']] = HodlBagState(
                    asset=row['asset'],
                    balance=Decimal(str(row['balance'])),
                    cost_basis_usd=Decimal(str(row['cost_basis_usd'])),
                    current_value_usd=Decimal(str(row['last_valuation_usd'])) if row.get('last_valuation_usd') else None,
                    first_accumulation=row.get('first_accumulation'),
                    last_accumulation=row.get('last_accumulation'),
                )

            # Load pending totals
            query = """
                SELECT asset, SUM(amount_usd) as total
                FROM hodl_pending
                WHERE executed_at IS NULL
                GROUP BY asset
            """
            rows = await self.db.fetch(query)

            for row in rows:
                self._pending[row['asset']] = Decimal(str(row['total']))

            # Count accumulations per asset
            query = """
                SELECT asset, COUNT(*) as count
                FROM hodl_transactions
                WHERE transaction_type = 'accumulation'
                GROUP BY asset
            """
            rows = await self.db.fetch(query)

            for row in rows:
                if row['asset'] in self._hodl_bags:
                    self._hodl_bags[row['asset']].accumulation_count = row['count']

            logger.info(
                f"Loaded hodl state: "
                f"bags={len(self._hodl_bags)}, "
                f"pending={sum(self._pending.values()):.2f} USD"
            )

        except Exception as e:
            logger.warning(f"Failed to load hodl state: {e}")
            # Initialize empty state
            for asset in ["USDT", "XRP", "BTC"]:
                self._hodl_bags[asset] = HodlBagState(
                    asset=asset,
                    balance=Decimal(0),
                    cost_basis_usd=Decimal(0),
                )

    def get_stats(self) -> dict:
        """Get hodl bag manager statistics."""
        return {
            "enabled": self.enabled,
            "is_paper_mode": self.is_paper_mode,
            "allocation_pct": float(self.allocation_pct),
            "total_allocations": self._total_allocations,
            "total_allocated_usd": float(self._total_allocated_usd),
            "total_executions": self._total_executions,
            "daily_accumulated_usd": float(self._daily_accumulated_usd),
            "daily_limit_usd": float(self.daily_accumulation_limit_usd),
            "pending": {k: float(v) for k, v in self._pending.items()},
            "thresholds": self.thresholds.to_dict(),
        }
