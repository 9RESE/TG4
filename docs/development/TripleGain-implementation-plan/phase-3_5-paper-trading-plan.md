# Phase 3.5: Paper Trading Integration Plan

**Version**: 1.0
**Date**: 2025-12-19
**Status**: PLANNING
**Priority**: CRITICAL - Must complete before any live trading

---

## Executive Summary

The TripleGain system has foundational infrastructure for paper trading but lacks complete implementation. Configuration exists in `execution.yaml` but is never read. Mock mode exists in `OrderManager` but uses hardcoded values instead of configuration. This plan outlines a comprehensive integration of paper trading as the **default execution mode**.

### Current State Assessment

| Component | Status | Gap |
|-----------|--------|-----|
| Paper trading config | EXISTS | Never read by code |
| Mock order placement | PARTIAL | Works, but no slippage |
| Fill simulation | PARTIAL | Hardcoded 2s delay, static prices |
| Portfolio balance | MISSING | Config ignored |
| Fee simulation | MISSING | Fees not deducted |
| Price source | MISSING | Market orders fail in paper mode |
| Data isolation | MISSING | Same DB for paper/live |

### Risk Statement

**CRITICAL**: Without proper paper trading, the system could accidentally execute real trades during testing, resulting in financial loss. Paper trading must be the enforced default with explicit opt-in for live trading.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING MODE SWITCH                                │
│                     ┌─────────────────────────────────────┐                 │
│                     │   TradingMode.PAPER (DEFAULT)       │                 │
│                     │   TradingMode.LIVE (Explicit Only)  │                 │
│                     └─────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION LAYER                                      │
│  ┌────────────────────────┐         ┌────────────────────────┐              │
│  │   PaperOrderExecutor   │         │   LiveOrderExecutor    │              │
│  │   (SimulatedExchange)  │         │   (KrakenClient)       │              │
│  └────────────────────────┘         └────────────────────────┘              │
│              │                                  │                            │
│              ▼                                  ▼                            │
│  ┌────────────────────────┐         ┌────────────────────────┐              │
│  │  PaperPortfolioTracker │         │  LivePortfolioTracker  │              │
│  │  (Simulated Balances)  │         │  (Kraken Balances)     │              │
│  └────────────────────────┘         └────────────────────────┘              │
│              │                                  │                            │
│              ▼                                  ▼                            │
│  ┌────────────────────────┐         ┌────────────────────────┐              │
│  │  paper_trading_* tables │        │  live_trading_* tables │              │
│  │  (Isolated DB Schema)  │         │  (Production Schema)   │              │
│  └────────────────────────┘         └────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Implementation Plan

### Phase 3.5.1: Trading Mode Infrastructure

**Goal**: Create a system-wide trading mode flag that defaults to PAPER.

#### 6.1.1 TradingMode Enum

**File**: `triplegain/src/execution/trading_mode.py` (NEW)

```python
from enum import Enum
import os

class TradingMode(Enum):
    """Trading execution mode."""
    PAPER = "paper"
    LIVE = "live"

def get_trading_mode() -> TradingMode:
    """
    Get the current trading mode.

    CRITICAL SAFETY: Defaults to PAPER unless explicitly set to LIVE
    via environment variable AND config file.

    Requires BOTH:
    - TRIPLEGAIN_TRADING_MODE=live (env var)
    - execution.yaml trading_mode: live (config)

    This dual-confirmation prevents accidental live trading.
    """
    env_mode = os.environ.get("TRIPLEGAIN_TRADING_MODE", "paper").lower()
    config_mode = load_config("execution").get("trading_mode", "paper").lower()

    # Require BOTH env and config to agree on "live" mode
    if env_mode == "live" and config_mode == "live":
        logger.warning("⚠️ LIVE TRADING MODE ENABLED - REAL MONEY AT RISK ⚠️")
        return TradingMode.LIVE

    if env_mode == "live" or config_mode == "live":
        logger.warning(
            "Live mode requested but not confirmed in both env and config. "
            f"Env: {env_mode}, Config: {config_mode}. Defaulting to PAPER."
        )

    return TradingMode.PAPER
```

#### 6.1.2 Config Update

**File**: `config/execution.yaml`

```yaml
# CRITICAL: Trading mode setting
# PAPER = Simulated trades with virtual balance (DEFAULT)
# LIVE = Real trades with real money (requires env confirmation)
trading_mode: paper  # NEVER change to "live" without understanding the risks

paper_trading:
  enabled: true  # Backwards compatibility flag

  # Starting portfolio for paper trading
  initial_balance:
    USDT: 10000
    BTC: 0.0
    XRP: 0.0

  # Fill simulation settings
  fill_delay_ms: 100          # Network latency simulation
  simulated_slippage_pct: 0.1 # 0.1% slippage on market orders
  simulate_partial_fills: false

  # Price source for paper trading
  # Options: "live_feed" (use real WebSocket prices), "historical" (use DB), "mock"
  price_source: live_feed

  # Database table prefix for isolation
  db_table_prefix: "paper_"
```

#### 6.1.3 Startup Validation

**File**: `triplegain/src/execution/__init__.py`

```python
def validate_trading_mode_on_startup():
    """
    Validate trading mode configuration at startup.

    CRITICAL: This function MUST be called during application startup
    to prevent accidental live trading.
    """
    mode = get_trading_mode()

    if mode == TradingMode.LIVE:
        # Require explicit confirmation for live trading
        if not os.environ.get("TRIPLEGAIN_CONFIRM_LIVE_TRADING") == "I_UNDERSTAND_THE_RISKS":
            raise RuntimeError(
                "SAFETY CHECK FAILED: Live trading requires explicit confirmation. "
                "Set TRIPLEGAIN_CONFIRM_LIVE_TRADING='I_UNDERSTAND_THE_RISKS' to proceed."
            )

        # Check Kraken credentials exist
        if not os.environ.get("KRAKEN_API_KEY"):
            raise RuntimeError("KRAKEN_API_KEY required for live trading")
        if not os.environ.get("KRAKEN_API_SECRET"):
            raise RuntimeError("KRAKEN_API_SECRET required for live trading")

        logger.critical("=" * 60)
        logger.critical("🔴 LIVE TRADING MODE - REAL MONEY TRANSACTIONS 🔴")
        logger.critical("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("🟢 PAPER TRADING MODE - SIMULATED EXECUTION 🟢")
        logger.info("=" * 60)

    return mode
```

---

### Phase 3.5.2: Paper Portfolio Tracker

**Goal**: Track simulated balances, P&L, and position limits in paper mode.

#### 6.2.1 PaperPortfolio Class

**File**: `triplegain/src/execution/paper_portfolio.py` (NEW)

```python
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Optional
import json

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

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_config(cls, config: dict) -> "PaperPortfolio":
        """Create portfolio from execution config."""
        paper_config = config.get("paper_trading", {})
        initial = paper_config.get("initial_balance", {"USDT": 10000})

        # Convert to Decimal
        balances = {k: Decimal(str(v)) for k, v in initial.items()}

        return cls(
            balances=balances.copy(),
            initial_balances=balances.copy(),
        )

    def get_balance(self, asset: str) -> Decimal:
        """Get balance for an asset."""
        return self.balances.get(asset.upper(), Decimal("0"))

    def adjust_balance(
        self,
        asset: str,
        amount: Decimal,
        reason: str = ""
    ) -> None:
        """
        Adjust balance for an asset.

        Args:
            asset: Asset symbol (e.g., "USDT", "BTC")
            amount: Amount to add (positive) or subtract (negative)
            reason: Reason for adjustment (for logging)
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

    def execute_trade(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        size: Decimal,
        price: Decimal,
        fee_pct: Decimal = Decimal("0.26"),
    ) -> Decimal:
        """
        Execute a paper trade, adjusting balances.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: "buy" or "sell"
            size: Amount of base currency
            price: Execution price
            fee_pct: Fee percentage (default 0.26% taker)

        Returns:
            Fee amount in quote currency
        """
        base, quote = symbol.split("/")
        value = size * price
        fee = value * (fee_pct / Decimal("100"))

        if side.lower() == "buy":
            # Spend quote currency, receive base currency
            self.adjust_balance(quote, -(value + fee), f"Buy {size} {base}")
            self.adjust_balance(base, size, f"Received from buy")
        else:
            # Spend base currency, receive quote currency
            self.adjust_balance(base, -size, f"Sell {size} {base}")
            self.adjust_balance(quote, value - fee, f"Received from sell")

        self.total_fees_paid += fee
        self.trade_count += 1

        return fee

    def get_equity_usd(self, prices: Dict[str, Decimal]) -> Decimal:
        """
        Calculate total portfolio equity in USD.

        Args:
            prices: Current prices for conversion (e.g., {"BTC/USDT": 45000})
        """
        equity = Decimal("0")

        for asset, balance in self.balances.items():
            if asset in ("USDT", "USD"):
                equity += balance
            else:
                # Find price to convert to USD
                pair = f"{asset}/USDT"
                if pair in prices:
                    equity += balance * prices[pair]
                # else: skip (can't price this asset)

        return equity

    def get_pnl_summary(self, current_prices: Dict[str, Decimal]) -> dict:
        """Get P&L summary."""
        current_equity = self.get_equity_usd(current_prices)
        initial_equity = sum(
            self.initial_balances.get(k, Decimal("0"))
            for k in ["USDT", "USD"]
        )

        total_pnl = current_equity - initial_equity
        pnl_pct = (total_pnl / initial_equity * 100) if initial_equity else Decimal("0")

        return {
            "initial_equity_usd": float(initial_equity),
            "current_equity_usd": float(current_equity),
            "total_pnl_usd": float(total_pnl),
            "total_pnl_pct": float(pnl_pct),
            "realized_pnl_usd": float(self.realized_pnl),
            "total_fees_usd": float(self.total_fees_paid),
            "trade_count": self.trade_count,
        }

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "balances": {k: str(v) for k, v in self.balances.items()},
            "initial_balances": {k: str(v) for k, v in self.initial_balances.items()},
            "realized_pnl": str(self.realized_pnl),
            "total_fees_paid": str(self.total_fees_paid),
            "trade_count": self.trade_count,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperPortfolio":
        """Deserialize from storage."""
        return cls(
            balances={k: Decimal(v) for k, v in data.get("balances", {}).items()},
            initial_balances={k: Decimal(v) for k, v in data.get("initial_balances", {}).items()},
            realized_pnl=Decimal(data.get("realized_pnl", "0")),
            total_fees_paid=Decimal(data.get("total_fees_paid", "0")),
            trade_count=data.get("trade_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            last_updated=datetime.fromisoformat(data["last_updated"]) if "last_updated" in data else datetime.now(timezone.utc),
        )


class InsufficientBalanceError(Exception):
    """Raised when paper trading balance is insufficient."""
    pass
```

---

### Phase 3.5.3: Paper Order Executor

**Goal**: Replace hardcoded mock logic with configurable, realistic simulation.

#### 6.3.1 PaperOrderExecutor Class

**File**: `triplegain/src/execution/paper_executor.py` (NEW)

```python
import asyncio
import random
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, Callable

class PaperOrderExecutor:
    """
    Simulated order execution for paper trading.

    Features:
    - Configurable fill delay
    - Slippage simulation
    - Partial fill simulation (optional)
    - Fee calculation
    - Real-time price source integration
    """

    def __init__(
        self,
        config: dict,
        paper_portfolio: "PaperPortfolio",
        price_source: Callable[[str], Optional[Decimal]],
    ):
        """
        Initialize paper executor.

        Args:
            config: Execution configuration
            paper_portfolio: Portfolio for balance tracking
            price_source: Function to get current price for a symbol
        """
        self.config = config
        self.portfolio = paper_portfolio
        self.get_price = price_source

        # Extract paper trading settings
        paper_config = config.get("paper_trading", {})
        self.fill_delay_ms = paper_config.get("fill_delay_ms", 100)
        self.slippage_pct = Decimal(str(paper_config.get("simulated_slippage_pct", 0.1)))
        self.simulate_partial_fills = paper_config.get("simulate_partial_fills", False)

        # Get fee rates from symbol config
        self.symbol_fees = {}
        for symbol, cfg in config.get("symbols", {}).items():
            self.symbol_fees[symbol] = Decimal(str(cfg.get("fee_pct", 0.26)))

    async def execute_order(self, order: "Order") -> "Order":
        """
        Execute a paper trading order.

        Args:
            order: Order to execute

        Returns:
            Updated order with fill information
        """
        # Simulate network/exchange latency
        delay_seconds = self.fill_delay_ms / 1000
        await asyncio.sleep(delay_seconds)

        # Get current market price
        current_price = self.get_price(order.symbol)
        if current_price is None:
            order.status = OrderStatus.ERROR
            order.error_message = f"No price available for {order.symbol}"
            return order

        # Determine fill price based on order type
        fill_price = self._calculate_fill_price(order, current_price)

        # Check if limit order would fill
        if order.type == OrderType.LIMIT:
            if not self._would_limit_fill(order, current_price):
                order.status = OrderStatus.OPEN
                return order  # Order remains open, not filled

        # Simulate partial fills if enabled
        if self.simulate_partial_fills and random.random() < 0.2:
            filled_pct = Decimal(str(random.uniform(0.3, 0.9)))
            order.filled_size = order.size * filled_pct
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.filled_size = order.size
            order.status = OrderStatus.FILLED

        order.filled_price = fill_price
        order.filled_at = datetime.now(timezone.utc)

        # Calculate and record fee
        fee_pct = self.symbol_fees.get(order.symbol, Decimal("0.26"))
        order.fee_amount = (order.filled_size * fill_price * fee_pct / Decimal("100"))
        order.fee_currency = order.symbol.split("/")[1]  # Quote currency

        # Update portfolio balances
        side = "buy" if order.side in (OrderSide.BUY, OrderSide.LONG) else "sell"
        try:
            self.portfolio.execute_trade(
                symbol=order.symbol,
                side=side,
                size=order.filled_size,
                price=fill_price,
                fee_pct=fee_pct,
            )
        except InsufficientBalanceError as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return order

        order.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"📝 Paper order filled: {order.side.value} {order.filled_size} {order.symbol} "
            f"@ {fill_price} (fee: {order.fee_amount})"
        )

        return order

    def _calculate_fill_price(
        self,
        order: "Order",
        current_price: Decimal
    ) -> Decimal:
        """
        Calculate fill price including slippage.

        Market orders get slippage.
        Limit orders fill at limit price (if they fill).
        Stop orders trigger at stop price then fill with slippage.
        """
        if order.type == OrderType.LIMIT:
            return order.price  # Limit orders fill at limit price

        # Apply slippage for market and stop orders
        slippage_multiplier = self.slippage_pct / Decimal("100")

        if order.side in (OrderSide.BUY, OrderSide.LONG):
            # Buying: pay slightly more (slippage up)
            return current_price * (Decimal("1") + slippage_multiplier)
        else:
            # Selling: receive slightly less (slippage down)
            return current_price * (Decimal("1") - slippage_multiplier)

    def _would_limit_fill(self, order: "Order", current_price: Decimal) -> bool:
        """Check if a limit order would fill at current price."""
        if order.side in (OrderSide.BUY, OrderSide.LONG):
            # Buy limit fills if market price <= limit price
            return current_price <= order.price
        else:
            # Sell limit fills if market price >= limit price
            return current_price >= order.price
```

---

### Phase 3.5.4: Price Source Integration

**Goal**: Provide realistic prices for paper trading.

#### 6.4.1 Price Source Options

```python
class PaperPriceSource:
    """
    Price source for paper trading.

    Options:
    1. Live WebSocket feed (real-time prices, simulated execution)
    2. Database cache (recent historical prices)
    3. Mock prices (for testing)
    """

    def __init__(
        self,
        source_type: str,
        db_connection: Optional[Any] = None,
        websocket_feed: Optional[Any] = None,
    ):
        self.source_type = source_type
        self.db = db_connection
        self.ws_feed = websocket_feed

        # Price cache for fast lookups
        self._cache: Dict[str, Decimal] = {}
        self._cache_time: Dict[str, datetime] = {}

        # Mock prices (fallback)
        self._mock_prices = {
            "BTC/USDT": Decimal("45000"),
            "XRP/USDT": Decimal("0.60"),
            "XRP/BTC": Decimal("0.000013"),
        }

    def get_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol."""

        if self.source_type == "live_feed" and self.ws_feed:
            # Get from WebSocket feed
            price = self.ws_feed.get_last_price(symbol)
            if price:
                self._cache[symbol] = Decimal(str(price))
                return self._cache[symbol]

        elif self.source_type == "historical" and self.db:
            # Get most recent price from database
            return self._get_db_price(symbol)

        # Fallback to cache or mock
        if symbol in self._cache:
            return self._cache[symbol]

        return self._mock_prices.get(symbol)

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update price cache (called by WebSocket handler)."""
        self._cache[symbol] = price
        self._cache_time[symbol] = datetime.now(timezone.utc)

    def _get_db_price(self, symbol: str) -> Optional[Decimal]:
        """Get most recent price from database."""
        # Query most recent candle close price
        result = self.db.execute(
            """
            SELECT close FROM candles_1m
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (symbol,)
        )
        if result:
            return Decimal(str(result[0][0]))
        return None
```

---

### Phase 3.5.5: Database Isolation

**Goal**: Separate paper trading data from live trading data.

#### 6.5.1 Migration for Paper Trading Tables

**File**: `migrations/005_paper_trading_tables.sql` (NEW)

```sql
-- ============================================================================
-- Migration: 005_paper_trading_tables.sql
-- Description: Create isolated tables for paper trading data
-- Created: 2025-12-19
-- Phase: 6 - Paper Trading
-- ============================================================================

-- ============================================================================
-- PAPER TRADING PORTFOLIO STATE
-- ============================================================================

CREATE TABLE IF NOT EXISTS paper_portfolio_state (
    id VARCHAR(50) PRIMARY KEY DEFAULT 'current',
    portfolio_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE paper_portfolio_state IS 'Persists paper trading portfolio state';

-- ============================================================================
-- PAPER TRADING ORDERS (mirrors orders table structure)
-- ============================================================================

CREATE TABLE IF NOT EXISTS paper_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    type VARCHAR(20) NOT NULL,
    size DECIMAL(20, 10) NOT NULL,
    price DECIMAL(20, 10),
    filled_size DECIMAL(20, 10) DEFAULT 0,
    filled_price DECIMAL(20, 10),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    fee_amount DECIMAL(20, 10) DEFAULT 0,
    fee_currency VARCHAR(10),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    filled_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_paper_orders_status ON paper_orders(status, created_at DESC);
CREATE INDEX idx_paper_orders_symbol ON paper_orders(symbol, created_at DESC);

-- ============================================================================
-- PAPER TRADING POSITIONS (mirrors positions table structure)
-- ============================================================================

CREATE TABLE IF NOT EXISTS paper_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size DECIMAL(20, 10) NOT NULL,
    entry_price DECIMAL(20, 10) NOT NULL,
    current_price DECIMAL(20, 10),
    leverage INT NOT NULL DEFAULT 1,
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    stop_loss DECIMAL(20, 10),
    take_profit DECIMAL(20, 10),
    unrealized_pnl DECIMAL(20, 10) DEFAULT 0,
    realized_pnl DECIMAL(20, 10) DEFAULT 0,
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    exit_price DECIMAL(20, 10),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_paper_positions_status ON paper_positions(status);
CREATE INDEX idx_paper_positions_symbol ON paper_positions(symbol, status);

-- ============================================================================
-- PAPER TRADING TRADE HISTORY
-- ============================================================================

CREATE TABLE IF NOT EXISTS paper_trade_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size DECIMAL(20, 10) NOT NULL,
    price DECIMAL(20, 10) NOT NULL,
    value_usd DECIMAL(20, 2),
    fee_amount DECIMAL(20, 10),
    fee_currency VARCHAR(10),
    order_id UUID,
    position_id UUID,
    balance_after JSONB,
    notes TEXT
);

-- Convert to hypertable for efficient time-series storage
SELECT create_hypertable('paper_trade_history', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- ============================================================================
-- PAPER TRADING PERFORMANCE SNAPSHOTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS paper_performance_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    equity_usd DECIMAL(20, 2) NOT NULL,
    pnl_usd DECIMAL(20, 2) NOT NULL,
    pnl_pct DECIMAL(10, 4) NOT NULL,
    open_positions INT DEFAULT 0,
    trade_count INT DEFAULT 0,
    win_rate DECIMAL(5, 2),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown_pct DECIMAL(10, 4),
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('paper_performance_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- ============================================================================
-- DATA RETENTION
-- ============================================================================

-- Keep paper trading data for 90 days (configurable)
SELECT add_retention_policy('paper_trade_history', INTERVAL '90 days',
    if_not_exists => TRUE);
SELECT add_retention_policy('paper_performance_snapshots', INTERVAL '90 days',
    if_not_exists => TRUE);

-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Migration 005_paper_trading_tables completed successfully.';
END $$;
```

---

### Phase 3.5.6: Integration with Coordinator

**Goal**: Modify coordinator to use paper or live execution based on mode.

#### 6.6.1 Coordinator Updates

**File**: `triplegain/src/orchestration/coordinator.py` (MODIFY)

```python
# Add to CoordinatorAgent.__init__
def __init__(
    self,
    # ... existing params ...
    trading_mode: TradingMode = TradingMode.PAPER,  # NEW
):
    self.trading_mode = trading_mode

    # Initialize appropriate executor
    if trading_mode == TradingMode.PAPER:
        self.paper_portfolio = PaperPortfolio.from_config(config)
        self.price_source = PaperPriceSource(
            source_type=config.get("paper_trading", {}).get("price_source", "live_feed"),
            websocket_feed=self.ws_feed,
        )
        self.paper_executor = PaperOrderExecutor(
            config=config,
            paper_portfolio=self.paper_portfolio,
            price_source=self.price_source.get_price,
        )
        logger.info("🟢 Coordinator initialized in PAPER trading mode")
    else:
        # Live mode - use existing execution manager with Kraken
        logger.warning("🔴 Coordinator initialized in LIVE trading mode")

# Modify _execute_validated_trade
async def _execute_validated_trade(self, trade: ValidatedTrade) -> bool:
    """Execute a validated trade using appropriate executor."""

    if self.trading_mode == TradingMode.PAPER:
        # Use paper executor
        order = self._create_order_from_trade(trade)
        result = await self.paper_executor.execute_order(order)

        if result.status == OrderStatus.FILLED:
            # Create paper position
            position = await self._create_paper_position(trade, result)
            logger.info(f"📝 Paper position opened: {position.id}")
            return True
        else:
            logger.warning(f"Paper order failed: {result.error_message}")
            return False
    else:
        # Use live execution (existing code)
        return await self.execution_manager.execute_trade(trade)
```

---

### Phase 3.5.7: API Endpoints for Paper Trading

**Goal**: Add API endpoints to manage paper trading.

#### 6.7.1 Paper Trading Routes

**File**: `triplegain/src/api/routes_paper_trading.py` (NEW)

```python
from fastapi import APIRouter, Depends
from triplegain.src.execution.paper_portfolio import PaperPortfolio
from triplegain.src.api.security import get_current_user, require_role, UserRole

router = APIRouter(prefix="/paper", tags=["Paper Trading"])

@router.get("/portfolio")
async def get_paper_portfolio(user = Depends(get_current_user)):
    """Get current paper trading portfolio state."""
    # Return portfolio balances, P&L, etc.
    pass

@router.post("/reset")
@require_role(UserRole.ADMIN)
async def reset_paper_portfolio(
    initial_balance: dict,
    user = Depends(get_current_user),
):
    """Reset paper trading portfolio to initial state."""
    # Reset balances, clear positions
    pass

@router.get("/history")
async def get_paper_trade_history(
    limit: int = 100,
    offset: int = 0,
    user = Depends(get_current_user),
):
    """Get paper trading history."""
    pass

@router.get("/performance")
async def get_paper_performance(user = Depends(get_current_user)):
    """Get paper trading performance metrics."""
    # Return win rate, Sharpe, drawdown, etc.
    pass

@router.get("/positions")
async def get_paper_positions(user = Depends(get_current_user)):
    """Get open paper trading positions."""
    pass

@router.post("/positions/{position_id}/close")
async def close_paper_position(
    position_id: str,
    exit_price: float = None,  # Optional manual exit price
    user = Depends(get_current_user),
):
    """Close a paper trading position."""
    pass
```

---

## Implementation Order

### Priority 1: Core Infrastructure (MUST HAVE)
| Task | Effort | Dependencies | Files |
|------|--------|--------------|-------|
| 6.1.1 TradingMode enum | 2h | None | trading_mode.py |
| 6.1.2 Config updates | 1h | None | execution.yaml |
| 6.1.3 Startup validation | 2h | 6.1.1 | execution/__init__.py |
| 6.2.1 PaperPortfolio class | 4h | None | paper_portfolio.py |
| 6.3.1 PaperOrderExecutor | 6h | 6.2.1 | paper_executor.py |

### Priority 2: Integration (MUST HAVE)
| Task | Effort | Dependencies | Files |
|------|--------|--------------|-------|
| 6.4.1 Price source | 4h | None | paper_price_source.py |
| 6.6.1 Coordinator updates | 4h | 6.3.1, 6.4.1 | coordinator.py |
| Update OrderManager | 2h | 6.3.1 | order_manager.py |

### Priority 3: Persistence (SHOULD HAVE)
| Task | Effort | Dependencies | Files |
|------|--------|--------------|-------|
| 6.5.1 DB migration | 2h | None | 005_paper_trading.sql |
| Portfolio persistence | 3h | 6.5.1, 6.2.1 | paper_portfolio.py |
| Trade history logging | 2h | 6.5.1 | paper_executor.py |

### Priority 4: API & Monitoring (NICE TO HAVE)
| Task | Effort | Dependencies | Files |
|------|--------|--------------|-------|
| 6.7.1 API routes | 4h | 6.2.1 | routes_paper_trading.py |
| Performance metrics | 3h | 6.5.1 | paper_performance.py |
| Dashboard integration | 4h | 6.7.1 | Phase 7 |

---

## Testing Strategy

### Unit Tests
- `test_paper_portfolio.py` - Balance tracking, trade execution, P&L calculation
- `test_paper_executor.py` - Order simulation, slippage, fills
- `test_trading_mode.py` - Mode switching, safety checks

### Integration Tests
- Test full trade flow in paper mode
- Test mode switching requires both env + config
- Test balance updates match expected P&L

### End-to-End Tests
- Run coordinator in paper mode for 24h
- Verify all signals result in paper trades
- Verify DB isolation (paper tables only)

---

## Safety Checklist

Before any live trading:

- [ ] Paper trading is the default mode
- [ ] Live mode requires dual confirmation (env + config)
- [ ] Live mode requires `TRIPLEGAIN_CONFIRM_LIVE_TRADING` env var
- [ ] Database tables are separate (paper_* vs production)
- [ ] API clearly indicates trading mode in responses
- [ ] Coordinator logs trading mode at startup
- [ ] Position tracker respects trading mode
- [ ] Risk engine works identically in both modes
- [ ] All tests pass in both modes

---

## Success Criteria

1. **Default Paper Mode**: System starts in paper trading without any configuration
2. **Realistic Simulation**: Slippage, fees, and delays match config
3. **Balance Tracking**: Portfolio balances update correctly on each trade
4. **Data Isolation**: Paper trading data is fully separate from live data
5. **Safe Live Transition**: Live trading requires explicit multi-step confirmation
6. **Performance Metrics**: Win rate, Sharpe, drawdown calculated for paper trades
7. **API Integration**: All paper trading operations available via API

---

## Timeline Estimate

| Phase | Effort | Calendar |
|-------|--------|----------|
| Priority 1 (Core) | 15h | 2 days |
| Priority 2 (Integration) | 10h | 1.5 days |
| Priority 3 (Persistence) | 7h | 1 day |
| Priority 4 (API) | 11h | 1.5 days |
| Testing & QA | 8h | 1 day |
| **Total** | **51h** | **~7 days** |

---

## Related Documents

- [Phase 5: Configuration & Integration](./phase-5-configuration.md)
- [Execution Layer Design](../../architecture/07-deployment/README.md)
- [Risk Management](../TripleGain-master-design/02-risk-management.md)
