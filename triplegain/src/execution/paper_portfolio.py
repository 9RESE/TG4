"""
Paper Portfolio - Simulated balance tracking for paper trading.

Phase 6: Paper Trading Integration

Tracks:
- Asset balances (USDT, BTC, XRP)
- Unrealized P&L from open positions
- Realized P&L from closed trades
- Trade history for analysis
- Fee deductions
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class InsufficientBalanceError(Exception):
    """Raised when paper trading balance is insufficient for a trade."""
    pass


class InvalidTradeError(Exception):
    """Raised when trade parameters are invalid."""
    pass


@dataclass
class PaperTradeRecord:
    """Record of a single paper trade for history tracking."""
    id: str
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    size: Decimal
    price: Decimal
    value: Decimal
    fee: Decimal
    fee_currency: str
    balance_after: Dict[str, Decimal]
    notes: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "size": str(self.size),
            "price": str(self.price),
            "value": str(self.value),
            "fee": str(self.fee),
            "fee_currency": self.fee_currency,
            "balance_after": {k: str(v) for k, v in self.balance_after.items()},
            "notes": self.notes,
        }


@dataclass
class PaperPortfolio:
    """
    Simulated portfolio for paper trading.

    Tracks:
    - Asset balances (USDT, BTC, XRP)
    - Unrealized P&L from open positions
    - Realized P&L from closed trades
    - Trade history for analysis
    """

    # Current balances
    balances: Dict[str, Decimal] = field(default_factory=dict)

    # Starting balances (for P&L calculation)
    initial_balances: Dict[str, Decimal] = field(default_factory=dict)

    # Running totals
    realized_pnl: Decimal = Decimal("0")
    total_fees_paid: Decimal = Decimal("0")
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Trade history
    trade_history: list = field(default_factory=list)
    max_history_size: int = 1000

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Session tracking
    session_id: str = ""

    @classmethod
    def from_config(cls, config: dict, session_id: str = "") -> "PaperPortfolio":
        """
        Create portfolio from execution config.

        Args:
            config: Execution configuration dictionary
            session_id: Optional session identifier

        Returns:
            New PaperPortfolio instance
        """
        paper_config = config.get("paper_trading", {})
        initial = paper_config.get("initial_balance", {"USDT": 10000})

        # Convert to Decimal, handling various input types
        balances = {}
        for k, v in initial.items():
            try:
                balances[k.upper()] = Decimal(str(v))
            except (InvalidOperation, ValueError):
                logger.warning(f"Invalid balance value for {k}: {v}, using 0")
                balances[k.upper()] = Decimal("0")

        if not session_id:
            session_id = f"paper_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        return cls(
            balances=balances.copy(),
            initial_balances=balances.copy(),
            session_id=session_id,
        )

    def get_balance(self, asset: str) -> Decimal:
        """
        Get balance for an asset.

        Args:
            asset: Asset symbol (e.g., "USDT", "BTC")

        Returns:
            Current balance (Decimal), 0 if not found
        """
        return self.balances.get(asset.upper(), Decimal("0"))

    def has_sufficient_balance(self, asset: str, amount: Decimal) -> bool:
        """
        Check if sufficient balance exists for an operation.

        Args:
            asset: Asset symbol
            amount: Required amount (positive)

        Returns:
            True if balance >= amount
        """
        return self.get_balance(asset) >= amount

    def adjust_balance(
        self,
        asset: str,
        amount: Decimal,
        reason: str = "",
    ) -> Decimal:
        """
        Adjust balance for an asset.

        Args:
            asset: Asset symbol (e.g., "USDT", "BTC")
            amount: Amount to add (positive) or subtract (negative)
            reason: Reason for adjustment (for logging)

        Returns:
            New balance after adjustment

        Raises:
            InsufficientBalanceError: If resulting balance would be negative
        """
        asset = asset.upper()
        current = self.balances.get(asset, Decimal("0"))
        new_balance = current + amount

        if new_balance < 0:
            raise InsufficientBalanceError(
                f"Insufficient {asset} balance. "
                f"Have: {current}, Need: {abs(amount)}"
            )

        self.balances[asset] = new_balance
        self.last_updated = datetime.now(timezone.utc)

        logger.debug(
            f"Paper balance adjusted: {asset} {current} → {new_balance} "
            f"({'+' if amount > 0 else ''}{amount}) [{reason}]"
        )

        return new_balance

    def execute_trade(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        size: Decimal,
        price: Decimal,
        fee_pct: Decimal = Decimal("0.26"),
        order_id: str = "",
    ) -> Dict[str, Any]:
        """
        Execute a paper trade, adjusting balances.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: "buy" or "sell"
            size: Amount of base currency
            price: Execution price
            fee_pct: Fee percentage (default 0.26% taker)
            order_id: Optional order ID for tracking

        Returns:
            Trade result dictionary with fee, value, and new balances

        Raises:
            InvalidTradeError: If trade parameters are invalid
            InsufficientBalanceError: If balance is insufficient
        """
        # Validate inputs
        if "/" not in symbol:
            raise InvalidTradeError(f"Invalid symbol format: {symbol}. Expected 'BASE/QUOTE'")

        if size <= 0:
            raise InvalidTradeError(f"Trade size must be positive, got {size}")

        if price <= 0:
            raise InvalidTradeError(f"Trade price must be positive, got {price}")

        base, quote = symbol.split("/")
        base = base.upper()
        quote = quote.upper()

        value = size * price
        fee = value * (fee_pct / Decimal("100"))

        side_lower = side.lower()
        if side_lower not in ("buy", "sell"):
            raise InvalidTradeError(f"Side must be 'buy' or 'sell', got {side}")

        if side_lower == "buy":
            # Buying: spend quote currency, receive base currency
            required_quote = value + fee
            if not self.has_sufficient_balance(quote, required_quote):
                current = self.get_balance(quote)
                raise InsufficientBalanceError(
                    f"Insufficient {quote} balance to buy {size} {base}. "
                    f"Need: {required_quote}, Have: {current}"
                )

            self.adjust_balance(quote, -(value + fee), f"Buy {size} {base}")
            self.adjust_balance(base, size, f"Received from buy")
        else:
            # Selling: spend base currency, receive quote currency
            if not self.has_sufficient_balance(base, size):
                current = self.get_balance(base)
                raise InsufficientBalanceError(
                    f"Insufficient {base} balance to sell. "
                    f"Need: {size}, Have: {current}"
                )

            self.adjust_balance(base, -size, f"Sell {size} {base}")
            self.adjust_balance(quote, value - fee, f"Received from sell")

        self.total_fees_paid += fee
        self.trade_count += 1

        # Create trade record
        import uuid
        trade_id = order_id or str(uuid.uuid4())
        trade_record = PaperTradeRecord(
            id=trade_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=side_lower,
            size=size,
            price=price,
            value=value,
            fee=fee,
            fee_currency=quote,
            balance_after=self.balances.copy(),
        )

        # Add to history with size limit
        self.trade_history.append(trade_record)
        if len(self.trade_history) > self.max_history_size:
            self.trade_history = self.trade_history[-self.max_history_size:]

        logger.info(
            f"📝 Paper trade: {side_lower.upper()} {size} {symbol} @ {price} "
            f"(value: {value}, fee: {fee})"
        )

        return {
            "success": True,
            "trade_id": trade_id,
            "fee": fee,
            "fee_currency": quote,
            "value": value,
            "size": size,
            "price": price,
            "balances": {k: float(v) for k, v in self.balances.items()},
        }

    def record_realized_pnl(self, pnl: Decimal, is_win: bool) -> None:
        """
        Record realized P&L from a closed position.

        Args:
            pnl: Realized profit/loss amount
            is_win: True if profitable trade
        """
        self.realized_pnl += pnl
        if is_win:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        logger.debug(f"Paper P&L recorded: {'+' if pnl > 0 else ''}{pnl}")

    def get_equity_usd(self, prices: Dict[str, Decimal]) -> Decimal:
        """
        Calculate total portfolio equity in USD.

        Args:
            prices: Current prices for conversion (e.g., {"BTC/USDT": 45000})

        Returns:
            Total equity in USD (USDT equivalent)
        """
        equity = Decimal("0")

        for asset, balance in self.balances.items():
            if balance == 0:
                continue

            if asset in ("USDT", "USD", "USDC"):
                equity += balance
            else:
                # Find price to convert to USD
                pair = f"{asset}/USDT"
                if pair in prices:
                    equity += balance * prices[pair]
                else:
                    # Try USD pair
                    pair_usd = f"{asset}/USD"
                    if pair_usd in prices:
                        equity += balance * prices[pair_usd]
                    else:
                        logger.warning(f"No price found for {asset}, excluding from equity")

        return equity

    def get_pnl_summary(self, current_prices: Dict[str, Decimal]) -> dict:
        """
        Get comprehensive P&L summary.

        Args:
            current_prices: Current market prices for unrealized P&L

        Returns:
            Dictionary with P&L metrics
        """
        current_equity = self.get_equity_usd(current_prices)

        # Calculate initial equity (in USDT terms)
        initial_equity = Decimal("0")
        for asset, balance in self.initial_balances.items():
            if asset in ("USDT", "USD", "USDC"):
                initial_equity += balance
            else:
                pair = f"{asset}/USDT"
                if pair in current_prices:
                    # Use current price for initial holdings
                    # This isn't perfect but gives a reasonable baseline
                    initial_equity += balance * current_prices[pair]
                else:
                    initial_equity += balance  # Assume 1:1 if no price

        total_pnl = current_equity - initial_equity
        pnl_pct = (total_pnl / initial_equity * 100) if initial_equity else Decimal("0")

        win_rate = Decimal("0")
        if self.winning_trades + self.losing_trades > 0:
            win_rate = Decimal(str(self.winning_trades)) / Decimal(str(self.winning_trades + self.losing_trades)) * 100

        return {
            "session_id": self.session_id,
            "initial_equity_usd": float(initial_equity),
            "current_equity_usd": float(current_equity),
            "total_pnl_usd": float(total_pnl),
            "total_pnl_pct": float(pnl_pct),
            "realized_pnl_usd": float(self.realized_pnl),
            "total_fees_usd": float(self.total_fees_paid),
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": float(win_rate),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    def get_balances_dict(self) -> Dict[str, float]:
        """Get balances as float dictionary for JSON serialization."""
        return {k: float(v) for k, v in self.balances.items()}

    def reset(self, new_balances: Optional[Dict[str, Decimal]] = None) -> None:
        """
        Reset portfolio to initial state or new balances.

        Args:
            new_balances: Optional new starting balances. If None, uses initial_balances.
        """
        if new_balances:
            self.balances = {k.upper(): Decimal(str(v)) for k, v in new_balances.items()}
            self.initial_balances = self.balances.copy()
        else:
            self.balances = self.initial_balances.copy()

        self.realized_pnl = Decimal("0")
        self.total_fees_paid = Decimal("0")
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_history = []
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)

        logger.info(f"Paper portfolio reset: {self.balances}")

    def to_dict(self) -> dict:
        """Serialize for storage/API response."""
        return {
            "session_id": self.session_id,
            "balances": {k: str(v) for k, v in self.balances.items()},
            "initial_balances": {k: str(v) for k, v in self.initial_balances.items()},
            "realized_pnl": str(self.realized_pnl),
            "total_fees_paid": str(self.total_fees_paid),
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperPortfolio":
        """
        Deserialize from storage.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Restored PaperPortfolio instance
        """
        return cls(
            balances={k: Decimal(v) for k, v in data.get("balances", {}).items()},
            initial_balances={k: Decimal(v) for k, v in data.get("initial_balances", {}).items()},
            realized_pnl=Decimal(data.get("realized_pnl", "0")),
            total_fees_paid=Decimal(data.get("total_fees_paid", "0")),
            trade_count=data.get("trade_count", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            last_updated=datetime.fromisoformat(data["last_updated"]) if "last_updated" in data else datetime.now(timezone.utc),
            session_id=data.get("session_id", ""),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "PaperPortfolio":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # HIGH-01: Database Session Persistence
    # =========================================================================

    async def persist_to_db(self, db) -> bool:
        """
        Persist portfolio state to database.

        HIGH-01: Saves current session state for recovery on restart.

        Args:
            db: Database connection pool with execute/fetchrow methods

        Returns:
            True if persisted successfully
        """
        if not db:
            logger.warning("No database connection - cannot persist portfolio")
            return False

        try:
            # Upsert session state (matches existing 005_paper_trading.sql schema)
            # Schema uses: id, current_balances, initial_balances, ended_at
            query = """
                INSERT INTO paper_sessions (
                    id, initial_balances, current_balances, realized_pnl,
                    total_fees_paid, trade_count, winning_trades, losing_trades,
                    created_at, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'active')
                ON CONFLICT (id) DO UPDATE SET
                    current_balances = $3,
                    realized_pnl = $4,
                    total_fees_paid = $5,
                    trade_count = $6,
                    winning_trades = $7,
                    losing_trades = $8
            """
            await db.execute(
                query,
                self.session_id,
                json.dumps({k: str(v) for k, v in self.initial_balances.items()}),
                json.dumps({k: str(v) for k, v in self.balances.items()}),
                str(self.realized_pnl),
                str(self.total_fees_paid),
                self.trade_count,
                self.winning_trades,
                self.losing_trades,
                self.created_at,
            )
            logger.debug(f"Paper portfolio persisted: session={self.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to persist paper portfolio: {e}")
            return False

    @classmethod
    async def load_from_db(cls, db, session_id: Optional[str] = None) -> Optional["PaperPortfolio"]:
        """
        Load portfolio state from database.

        HIGH-01: Restores session state on startup.

        Args:
            db: Database connection pool
            session_id: Specific session to load, or None for most recent active

        Returns:
            Restored PaperPortfolio or None if not found
        """
        if not db:
            return None

        try:
            if session_id:
                query = """
                    SELECT * FROM paper_sessions
                    WHERE id = $1 AND status = 'active'
                """
                row = await db.fetchrow(query, session_id)
            else:
                # Get most recent active session
                query = """
                    SELECT * FROM paper_sessions
                    WHERE status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                row = await db.fetchrow(query)

            if not row:
                logger.info("No active paper session found in database")
                return None

            # Parse stored JSON data (matches 005_paper_trading.sql schema)
            balances = {k: Decimal(v) for k, v in json.loads(row['current_balances']).items()}
            initial_balances = {k: Decimal(v) for k, v in json.loads(row['initial_balances']).items()}

            portfolio = cls(
                balances=balances,
                initial_balances=initial_balances,
                realized_pnl=Decimal(str(row['realized_pnl'])) if row['realized_pnl'] else Decimal("0"),
                total_fees_paid=Decimal(str(row['total_fees_paid'])) if row['total_fees_paid'] else Decimal("0"),
                trade_count=row['trade_count'] or 0,
                winning_trades=row['winning_trades'] or 0,
                losing_trades=row['losing_trades'] or 0,
                created_at=row['created_at'],
                last_updated=datetime.now(timezone.utc),
                session_id=row['id'],
            )

            logger.info(f"Paper portfolio restored from database: session={row['id']}")
            return portfolio

        except Exception as e:
            logger.error(f"Failed to load paper portfolio from database: {e}")
            return None

    async def end_session(self, db) -> bool:
        """
        End current session (mark as ended in database).

        Args:
            db: Database connection pool

        Returns:
            True if session ended successfully
        """
        if not db:
            return False

        try:
            query = """
                UPDATE paper_sessions
                SET status = 'ended', ended_at = $2
                WHERE id = $1
            """
            await db.execute(query, self.session_id, datetime.now(timezone.utc))
            logger.info(f"Paper session ended: {self.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to end paper session: {e}")
            return False
