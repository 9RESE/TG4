"""
Unit tests for HodlBagManager - Automated profit allocation for long-term accumulation.

Phase 8: Hodl Bag System

Tests cover:
- HodlBagManager initialization and configuration
- Profit allocation calculation
- Split percentage distribution
- Per-asset threshold tracking
- Pending accumulation management
- Accumulation execution
- Transaction recording
- Daily limits and caps
- Paper trading mode
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from triplegain.src.execution.hodl_bag import (
    HodlBagManager,
    HodlBagState,
    HodlAllocation,
    HodlTransaction,
    HodlPending,
    HodlThresholds,
    TransactionType,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_config():
    """Basic hodl bag configuration."""
    return {
        'hodl_bags': {
            'enabled': True,
            'allocation_pct': 10,
            'split': {
                'usdt_pct': 33.34,
                'xrp_pct': 33.33,
                'btc_pct': 33.33,
            },
            'min_accumulation': {
                'usdt': 1,
                'xrp': 25,
                'btc': 15,
            },
            'execution': {
                'order_type': 'market',
                'max_retries': 3,
                'retry_delay_seconds': 30,
            },
            'limits': {
                'max_single_accumulation_usd': 1000,
                'daily_accumulation_limit_usd': 5000,
                'min_profit_to_allocate_usd': 1.0,
            },
            'prices': {
                'fallback': {
                    'BTC/USDT': 45000,
                    'XRP/USDT': 0.60,
                }
            }
        }
    }


@pytest.fixture
def mock_db_pool():
    """Mock database pool."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchrow = AsyncMock(return_value=None)
    return mock_db


@pytest.fixture
def hodl_manager(basic_config):
    """Create HodlBagManager instance."""
    return HodlBagManager(
        config=basic_config,
        db_pool=None,
        kraken_client=None,
        price_source=None,
        message_bus=None,
        is_paper_mode=True,
    )


@pytest.fixture
def hodl_manager_with_db(basic_config, mock_db_pool):
    """Create HodlBagManager instance with mock database."""
    return HodlBagManager(
        config=basic_config,
        db_pool=mock_db_pool,
        kraken_client=None,
        price_source=None,
        message_bus=None,
        is_paper_mode=True,
    )


# =============================================================================
# HodlThresholds Tests
# =============================================================================

class TestHodlThresholds:
    """Tests for HodlThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = HodlThresholds()
        assert thresholds.usdt == Decimal("1")
        assert thresholds.xrp == Decimal("25")
        assert thresholds.btc == Decimal("15")

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = HodlThresholds(
            usdt=Decimal("5"),
            xrp=Decimal("50"),
            btc=Decimal("20"),
        )
        assert thresholds.usdt == Decimal("5")
        assert thresholds.xrp == Decimal("50")
        assert thresholds.btc == Decimal("20")

    def test_get_threshold_by_asset(self):
        """Test getting threshold by asset name."""
        thresholds = HodlThresholds()
        assert thresholds.get("USDT") == Decimal("1")
        assert thresholds.get("XRP") == Decimal("25")
        assert thresholds.get("BTC") == Decimal("15")

    def test_get_unknown_asset_returns_default(self):
        """Test that unknown asset returns default threshold."""
        thresholds = HodlThresholds()
        assert thresholds.get("ETH") == Decimal("25")

    def test_to_dict(self):
        """Test serialization to dictionary."""
        thresholds = HodlThresholds()
        result = thresholds.to_dict()
        assert result == {
            "usdt": "1",
            "xrp": "25",
            "btc": "15",
        }


# =============================================================================
# HodlBagState Tests
# =============================================================================

class TestHodlBagState:
    """Tests for HodlBagState dataclass."""

    def test_basic_state(self):
        """Test basic state creation."""
        state = HodlBagState(
            asset="BTC",
            balance=Decimal("0.001"),
            cost_basis_usd=Decimal("45"),
        )
        assert state.asset == "BTC"
        assert state.balance == Decimal("0.001")
        assert state.cost_basis_usd == Decimal("45")
        assert state.current_value_usd is None
        assert state.pending_usd == Decimal(0)

    def test_state_with_pnl(self):
        """Test state with P&L values."""
        state = HodlBagState(
            asset="BTC",
            balance=Decimal("0.001"),
            cost_basis_usd=Decimal("45"),
            current_value_usd=Decimal("50"),
            unrealized_pnl_usd=Decimal("5"),
            unrealized_pnl_pct=Decimal("11.11"),
        )
        assert state.unrealized_pnl_usd == Decimal("5")
        assert state.unrealized_pnl_pct == Decimal("11.11")

    def test_state_to_dict(self):
        """Test serialization to dictionary."""
        state = HodlBagState(
            asset="BTC",
            balance=Decimal("0.001"),
            cost_basis_usd=Decimal("45"),
        )
        result = state.to_dict()
        assert result["asset"] == "BTC"
        assert result["balance"] == "0.001"
        assert result["cost_basis_usd"] == "45"


# =============================================================================
# HodlAllocation Tests
# =============================================================================

class TestHodlAllocation:
    """Tests for HodlAllocation dataclass."""

    def test_allocation_creation(self):
        """Test allocation creation."""
        allocation = HodlAllocation(
            trade_id="trade-123",
            profit_usd=Decimal("100"),
            total_allocation_usd=Decimal("10"),
            usdt_amount_usd=Decimal("3.34"),
            xrp_amount_usd=Decimal("3.33"),
            btc_amount_usd=Decimal("3.33"),
        )
        assert allocation.trade_id == "trade-123"
        assert allocation.profit_usd == Decimal("100")
        assert allocation.total_allocation_usd == Decimal("10")

    def test_allocation_to_dict(self):
        """Test serialization to dictionary."""
        allocation = HodlAllocation(
            trade_id="trade-123",
            profit_usd=Decimal("100"),
            total_allocation_usd=Decimal("10"),
            usdt_amount_usd=Decimal("3.34"),
            xrp_amount_usd=Decimal("3.33"),
            btc_amount_usd=Decimal("3.33"),
        )
        result = allocation.to_dict()
        assert result["trade_id"] == "trade-123"
        assert result["profit_usd"] == "100"
        assert "timestamp" in result


# =============================================================================
# HodlTransaction Tests
# =============================================================================

class TestHodlTransaction:
    """Tests for HodlTransaction dataclass."""

    def test_transaction_creation(self):
        """Test transaction creation."""
        transaction = HodlTransaction(
            id="txn-123",
            timestamp=datetime.now(timezone.utc),
            asset="BTC",
            transaction_type=TransactionType.ACCUMULATION,
            amount=Decimal("0.001"),
            price_usd=Decimal("45000"),
            value_usd=Decimal("45"),
        )
        assert transaction.asset == "BTC"
        assert transaction.transaction_type == TransactionType.ACCUMULATION
        assert transaction.amount == Decimal("0.001")

    def test_transaction_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(timezone.utc)
        transaction = HodlTransaction(
            id="txn-123",
            timestamp=now,
            asset="BTC",
            transaction_type=TransactionType.ACCUMULATION,
            amount=Decimal("0.001"),
            price_usd=Decimal("45000"),
            value_usd=Decimal("45"),
        )
        result = transaction.to_dict()
        assert result["id"] == "txn-123"
        assert result["asset"] == "BTC"
        assert result["transaction_type"] == "accumulation"


# =============================================================================
# HodlBagManager Initialization Tests
# =============================================================================

class TestHodlBagManagerInit:
    """Tests for HodlBagManager initialization."""

    def test_basic_initialization(self, hodl_manager):
        """Test basic manager initialization."""
        assert hodl_manager.enabled is True
        assert hodl_manager.allocation_pct == Decimal("10")
        assert hodl_manager.is_paper_mode is True

    def test_split_percentages(self, hodl_manager):
        """Test split percentage configuration."""
        assert hodl_manager.usdt_pct == Decimal("33.34")
        assert hodl_manager.xrp_pct == Decimal("33.33")
        assert hodl_manager.btc_pct == Decimal("33.33")

    def test_threshold_configuration(self, hodl_manager):
        """Test threshold configuration."""
        assert hodl_manager.thresholds.usdt == Decimal("1")
        assert hodl_manager.thresholds.xrp == Decimal("25")
        assert hodl_manager.thresholds.btc == Decimal("15")

    def test_safety_limits(self, hodl_manager):
        """Test safety limit configuration."""
        assert hodl_manager.max_single_accumulation_usd == Decimal("1000")
        assert hodl_manager.daily_accumulation_limit_usd == Decimal("5000")
        assert hodl_manager.min_profit_to_allocate_usd == Decimal("1.0")

    def test_disabled_manager(self, basic_config):
        """Test manager when disabled."""
        basic_config['hodl_bags']['enabled'] = False
        manager = HodlBagManager(config=basic_config, is_paper_mode=True)
        assert manager.enabled is False


# =============================================================================
# Profit Allocation Tests
# =============================================================================

class TestProfitAllocation:
    """Tests for profit allocation calculation."""

    def test_calculate_allocation_basic(self, hodl_manager):
        """Test basic allocation calculation."""
        allocation = hodl_manager._calculate_allocation(
            trade_id="trade-123",
            profit_usd=Decimal("100"),
        )

        # 10% of 100 = 10
        assert allocation.total_allocation_usd == Decimal("10.00")

        # Split should be ~33.33% each
        assert allocation.usdt_amount_usd == Decimal("3.34")  # Gets rounding remainder
        assert allocation.xrp_amount_usd == Decimal("3.33")
        assert allocation.btc_amount_usd == Decimal("3.33")

        # Total should equal allocation
        total = allocation.usdt_amount_usd + allocation.xrp_amount_usd + allocation.btc_amount_usd
        assert total == allocation.total_allocation_usd

    def test_calculate_allocation_large_profit(self, hodl_manager):
        """Test allocation with large profit."""
        allocation = hodl_manager._calculate_allocation(
            trade_id="trade-456",
            profit_usd=Decimal("5000"),
        )

        # 10% of 5000 = 500
        assert allocation.total_allocation_usd == Decimal("500.00")

    def test_calculate_allocation_small_profit(self, hodl_manager):
        """Test allocation with small profit."""
        allocation = hodl_manager._calculate_allocation(
            trade_id="trade-789",
            profit_usd=Decimal("10"),
        )

        # 10% of 10 = 1
        assert allocation.total_allocation_usd == Decimal("1.00")

    def test_calculate_allocation_fractional(self, hodl_manager):
        """Test allocation with fractional amounts."""
        allocation = hodl_manager._calculate_allocation(
            trade_id="trade-abc",
            profit_usd=Decimal("33.33"),
        )

        # 10% of 33.33 = 3.33
        assert allocation.total_allocation_usd == Decimal("3.33")


# =============================================================================
# Process Trade Profit Tests
# =============================================================================

class TestProcessTradeProfit:
    """Tests for process_trade_profit method."""

    @pytest.mark.asyncio
    async def test_process_profit_basic(self, hodl_manager):
        """Test basic profit processing."""
        allocation = await hodl_manager.process_trade_profit(
            trade_id="trade-123",
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        assert allocation is not None
        assert allocation.total_allocation_usd == Decimal("10.00")

        # Check pending was updated
        # Note: USDT has $1 threshold so it gets executed immediately
        # XRP/BTC stay pending until they reach their thresholds (25/15)
        pending = await hodl_manager.get_pending()
        # USDT gets executed immediately (threshold $1)
        assert pending["USDT"] == Decimal("0")
        # XRP/BTC stay pending (below threshold)
        assert pending["XRP"] == Decimal("3.33")
        assert pending["BTC"] == Decimal("3.33")

    @pytest.mark.asyncio
    async def test_process_profit_disabled(self, basic_config):
        """Test that no allocation when disabled."""
        basic_config['hodl_bags']['enabled'] = False
        manager = HodlBagManager(config=basic_config, is_paper_mode=True)

        allocation = await manager.process_trade_profit(
            trade_id="trade-123",
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        assert allocation is None

    @pytest.mark.asyncio
    async def test_process_profit_zero(self, hodl_manager):
        """Test no allocation for zero profit."""
        allocation = await hodl_manager.process_trade_profit(
            trade_id="trade-123",
            profit_usd=Decimal("0"),
            source_symbol="BTC/USDT",
        )

        assert allocation is None

    @pytest.mark.asyncio
    async def test_process_profit_negative(self, hodl_manager):
        """Test no allocation for loss."""
        allocation = await hodl_manager.process_trade_profit(
            trade_id="trade-123",
            profit_usd=Decimal("-50"),
            source_symbol="BTC/USDT",
        )

        assert allocation is None

    @pytest.mark.asyncio
    async def test_process_profit_below_minimum(self, hodl_manager):
        """Test no allocation for profit below minimum."""
        allocation = await hodl_manager.process_trade_profit(
            trade_id="trade-123",
            profit_usd=Decimal("0.50"),  # Below 1.0 minimum
            source_symbol="BTC/USDT",
        )

        assert allocation is None

    @pytest.mark.asyncio
    async def test_process_profit_capped(self, hodl_manager):
        """Test allocation is capped at maximum."""
        allocation = await hodl_manager.process_trade_profit(
            trade_id="trade-123",
            profit_usd=Decimal("20000"),  # Would be 2000 allocation, exceeds 1000 cap
            source_symbol="BTC/USDT",
        )

        assert allocation is not None
        assert allocation.total_allocation_usd == hodl_manager.max_single_accumulation_usd

    @pytest.mark.asyncio
    async def test_process_profit_accumulates_pending(self, hodl_manager):
        """Test that multiple profits accumulate pending."""
        # First profit
        await hodl_manager.process_trade_profit(
            trade_id="trade-1",
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        # Second profit
        await hodl_manager.process_trade_profit(
            trade_id="trade-2",
            profit_usd=Decimal("100"),
            source_symbol="XRP/USDT",
        )

        pending = await hodl_manager.get_pending()
        # Each profit adds 3.33-3.34 per asset
        # Note: USDT has $1 threshold, so it gets executed after each profit
        # So USDT pending stays at 0
        assert pending["USDT"] == Decimal("0")  # Executed immediately each time
        assert pending["XRP"] == Decimal("6.66")   # 3.33 + 3.33
        assert pending["BTC"] == Decimal("6.66")   # 3.33 + 3.33


# =============================================================================
# Accumulation Execution Tests
# =============================================================================

class TestAccumulationExecution:
    """Tests for accumulation execution."""

    @pytest.mark.asyncio
    async def test_execute_usdt_accumulation(self, hodl_manager):
        """Test USDT accumulation (no purchase needed)."""
        # Add pending USDT
        hodl_manager._pending["USDT"] = Decimal("5.00")

        # USDT threshold is 1, so should execute
        amount = await hodl_manager.execute_accumulation("USDT")

        assert amount == Decimal("5.00")
        assert hodl_manager._pending["USDT"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_execute_xrp_accumulation_paper_mode(self, hodl_manager):
        """Test XRP accumulation in paper mode."""
        # Add pending XRP (above threshold)
        hodl_manager._pending["XRP"] = Decimal("30.00")

        # Execute accumulation
        amount = await hodl_manager.execute_accumulation("XRP")

        # Should have calculated XRP amount at fallback price (0.60)
        # 30 USD / 0.60 = 50 XRP
        assert amount is not None
        assert amount == Decimal("50.000000")  # Rounded to 6 decimals for XRP
        assert hodl_manager._pending["XRP"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_execute_btc_accumulation_paper_mode(self, hodl_manager):
        """Test BTC accumulation in paper mode."""
        # Add pending BTC (above threshold)
        hodl_manager._pending["BTC"] = Decimal("20.00")

        # Execute accumulation
        amount = await hodl_manager.execute_accumulation("BTC")

        # Should have calculated BTC amount at fallback price (45000)
        # 20 USD / 45000 = 0.00044444 BTC
        assert amount is not None
        assert hodl_manager._pending["BTC"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_execute_accumulation_below_threshold(self, hodl_manager):
        """Test no execution when below threshold."""
        # Add pending XRP (below 25 threshold)
        hodl_manager._pending["XRP"] = Decimal("20.00")

        amount = await hodl_manager.execute_accumulation("XRP")

        assert amount is None
        assert hodl_manager._pending["XRP"] == Decimal("20.00")

    @pytest.mark.asyncio
    async def test_execute_accumulation_no_pending(self, hodl_manager):
        """Test no execution when no pending."""
        amount = await hodl_manager.execute_accumulation("XRP")

        assert amount is None

    @pytest.mark.asyncio
    async def test_force_accumulation_below_threshold(self, hodl_manager):
        """Test force accumulation even below threshold.

        M1 Fix: force_accumulation now bypasses threshold check.
        """
        # Add pending XRP (below threshold of $25)
        hodl_manager._pending["XRP"] = Decimal("10.00")

        # M1 Fix: Force accumulation now properly bypasses threshold
        success = await hodl_manager.force_accumulation("XRP")

        # Force should succeed because M1 fix bypasses threshold check
        assert success is True  # Now works even below threshold

        # Verify XRP pending was cleared
        assert hodl_manager._pending["XRP"] == Decimal(0)


# =============================================================================
# Daily Limit Tests
# =============================================================================

class TestDailyLimits:
    """Tests for daily accumulation limits."""

    @pytest.mark.asyncio
    async def test_daily_limit_enforced(self, hodl_manager):
        """Test daily limit enforcement."""
        # Set daily accumulated to near limit
        hodl_manager._daily_accumulated_usd = Decimal("4999")
        hodl_manager._daily_reset_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Try to allocate more
        allocation = await hodl_manager.process_trade_profit(
            trade_id="trade-123",
            profit_usd=Decimal("100"),  # Would add 10
            source_symbol="BTC/USDT",
        )

        assert allocation is not None

        # Now at limit
        hodl_manager._daily_accumulated_usd = Decimal("5000")

        # Next allocation should fail
        allocation = await hodl_manager.process_trade_profit(
            trade_id="trade-456",
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        assert allocation is None

    def test_daily_reset_check(self, hodl_manager):
        """Test daily reset mechanism."""
        # Set accumulated from yesterday
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        hodl_manager._daily_accumulated_usd = Decimal("3000")
        hodl_manager._daily_reset_date = yesterday.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Check reset
        hodl_manager._check_daily_reset()

        assert hodl_manager._daily_accumulated_usd == Decimal("0")


# =============================================================================
# State and Metrics Tests
# =============================================================================

class TestStateAndMetrics:
    """Tests for state management and metrics."""

    @pytest.mark.asyncio
    async def test_get_hodl_state_empty(self, hodl_manager):
        """Test getting state when empty."""
        await hodl_manager.start()
        state = await hodl_manager.get_hodl_state()

        assert "USDT" in state
        assert "XRP" in state
        assert "BTC" in state

    @pytest.mark.asyncio
    async def test_get_pending(self, hodl_manager):
        """Test getting pending amounts."""
        hodl_manager._pending["USDT"] = Decimal("5.00")
        hodl_manager._pending["XRP"] = Decimal("10.00")

        pending = await hodl_manager.get_pending()

        assert pending["USDT"] == Decimal("5.00")
        assert pending["XRP"] == Decimal("10.00")
        assert pending["BTC"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_pending_single_asset(self, hodl_manager):
        """Test getting pending for single asset."""
        hodl_manager._pending["XRP"] = Decimal("15.00")

        pending = await hodl_manager.get_pending("XRP")

        assert "XRP" in pending
        assert pending["XRP"] == Decimal("15.00")

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, hodl_manager):
        """Test metrics calculation."""
        await hodl_manager.start()

        # Add some state
        hodl_manager._hodl_bags["BTC"] = HodlBagState(
            asset="BTC",
            balance=Decimal("0.001"),
            cost_basis_usd=Decimal("45"),
            current_value_usd=Decimal("50"),
        )
        hodl_manager._total_allocations = 5
        hodl_manager._total_allocated_usd = Decimal("100")

        metrics = await hodl_manager.calculate_metrics()

        assert "total_cost_basis_usd" in metrics
        assert "total_current_value_usd" in metrics
        assert metrics["total_allocations"] == 5
        assert metrics["total_allocated_usd"] == 100.0

    def test_get_stats(self, hodl_manager):
        """Test getting statistics."""
        hodl_manager._total_allocations = 10
        hodl_manager._total_executions = 3
        hodl_manager._daily_accumulated_usd = Decimal("250")

        stats = hodl_manager.get_stats()

        assert stats["enabled"] is True
        assert stats["is_paper_mode"] is True
        assert stats["total_allocations"] == 10
        assert stats["total_executions"] == 3
        assert stats["daily_accumulated_usd"] == 250.0


# =============================================================================
# Price Source Tests
# =============================================================================

class TestPriceSource:
    """Tests for price source functionality."""

    @pytest.mark.asyncio
    async def test_fallback_prices(self, hodl_manager):
        """Test fallback prices are used."""
        price = await hodl_manager._get_current_price("BTC/USDT")
        assert price == Decimal("45000")

        price = await hodl_manager._get_current_price("XRP/USDT")
        assert price == Decimal("0.60")

    @pytest.mark.asyncio
    async def test_custom_price_source(self, basic_config):
        """Test custom price source function."""
        def custom_price_source(symbol):
            prices = {
                "BTC/USDT": Decimal("50000"),
                "XRP/USDT": Decimal("1.00"),
            }
            return prices.get(symbol)

        manager = HodlBagManager(
            config=basic_config,
            price_source=custom_price_source,
            is_paper_mode=True,
        )

        price = await manager._get_current_price("BTC/USDT")
        assert price == Decimal("50000")


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for hodl bag flow."""

    @pytest.mark.asyncio
    async def test_full_accumulation_flow(self, hodl_manager):
        """Test full flow from profit to accumulation."""
        await hodl_manager.start()

        # Process multiple profitable trades
        for i in range(8):
            await hodl_manager.process_trade_profit(
                trade_id=f"trade-{i}",
                profit_usd=Decimal("100"),
                source_symbol="BTC/USDT",
            )

        # Check pending (should trigger XRP accumulation at 25 threshold)
        pending = await hodl_manager.get_pending()

        # 8 trades * 3.33 per asset = ~26.64 each
        # XRP should have triggered (>25)
        # BTC should have triggered (>15)
        # USDT should have triggered multiple times (>1)

        # Check state
        state = await hodl_manager.get_hodl_state()
        assert "BTC" in state
        assert "XRP" in state
        assert "USDT" in state

    @pytest.mark.asyncio
    async def test_threshold_triggers(self, hodl_manager):
        """Test that thresholds trigger correctly."""
        await hodl_manager.start()

        # Add just below threshold
        hodl_manager._pending["XRP"] = Decimal("24.99")
        hodl_manager._pending["BTC"] = Decimal("14.99")

        # Process small profit that pushes over threshold
        await hodl_manager.process_trade_profit(
            trade_id="trade-push",
            profit_usd=Decimal("10"),  # Adds ~0.33 to each
            source_symbol="BTC/USDT",
        )

        # XRP should have triggered (24.99 + 0.33 > 25)
        pending = await hodl_manager.get_pending()

        # Check that executions happened
        assert hodl_manager._total_executions >= 1


# =============================================================================
# L5 Fix: Slippage Protection Tests
# =============================================================================

class TestSlippageProtection:
    """L5 Fix: Tests for slippage tracking and protection."""

    @pytest.fixture
    def hodl_manager_with_slippage(self):
        """Create HodlBagManager with slippage config."""
        config = {
            "hodl_bags": {
                "enabled": True,
                "allocation_pct": 10,
                "split": {"usdt_pct": 33.34, "xrp_pct": 33.33, "btc_pct": 33.33},
                "min_accumulation": {"usdt": 1, "xrp": 25, "btc": 15},
                "execution": {
                    "order_type": "market",
                    "max_retries": 3,
                    "retry_delay_seconds": 1,
                    "max_slippage_pct": 0.5,  # L5: Slippage config
                },
                "limits": {
                    "max_single_accumulation_usd": 1000,
                    "daily_accumulation_limit_usd": 5000,
                    "min_profit_to_allocate_usd": 1.0,
                },
                "prices": {
                    "fallback": {"BTC/USDT": 100000, "XRP/USDT": 2.50},
                },
            }
        }
        return HodlBagManager(
            config=config,
            db_pool=None,
            kraken_client=None,
            price_source=lambda s: {
                "BTC/USDT": Decimal("100000"),
                "XRP/USDT": Decimal("2.50"),
            }.get(s),
            message_bus=None,
            is_paper_mode=True,
        )

    def test_slippage_config_loaded(self, hodl_manager_with_slippage):
        """Test slippage configuration is loaded."""
        manager = hodl_manager_with_slippage
        assert manager.max_slippage_pct == Decimal("0.5")

    def test_slippage_stats_initialized(self, hodl_manager_with_slippage):
        """Test slippage statistics are initialized."""
        manager = hodl_manager_with_slippage
        assert manager._total_slippage_events == 0
        assert manager._max_slippage_observed == Decimal(0)
        assert manager._slippage_warnings == 0

    def test_record_slippage_zero(self, hodl_manager_with_slippage):
        """Test recording zero slippage."""
        manager = hodl_manager_with_slippage
        slippage = manager._record_slippage("BTC/USDT", Decimal("100000"), Decimal("100000"))

        assert slippage == Decimal("0.00")
        assert manager._total_slippage_events == 1
        assert manager._slippage_warnings == 0

    def test_record_slippage_within_threshold(self, hodl_manager_with_slippage):
        """Test slippage within acceptable threshold."""
        manager = hodl_manager_with_slippage
        # 0.3% slippage (below 0.5% threshold)
        expected = Decimal("100000")
        actual = Decimal("100300")  # 0.3% higher
        slippage = manager._record_slippage("BTC/USDT", expected, actual)

        assert slippage == Decimal("0.30")
        assert manager._total_slippage_events == 1
        assert manager._slippage_warnings == 0  # Below threshold, no warning

    def test_record_slippage_exceeds_threshold(self, hodl_manager_with_slippage):
        """Test slippage exceeding threshold triggers warning."""
        manager = hodl_manager_with_slippage
        # 1% slippage (above 0.5% threshold)
        expected = Decimal("100000")
        actual = Decimal("101000")  # 1% higher
        slippage = manager._record_slippage("BTC/USDT", expected, actual)

        assert slippage == Decimal("1.00")
        assert manager._total_slippage_events == 1
        assert manager._slippage_warnings == 1  # Above threshold

    def test_max_slippage_tracked(self, hodl_manager_with_slippage):
        """Test maximum slippage is tracked."""
        manager = hodl_manager_with_slippage

        # First event: 0.2%
        manager._record_slippage("BTC/USDT", Decimal("100000"), Decimal("100200"))
        assert manager._max_slippage_observed == Decimal("0.20")

        # Second event: 0.5% (higher)
        manager._record_slippage("BTC/USDT", Decimal("100000"), Decimal("100500"))
        assert manager._max_slippage_observed == Decimal("0.50")

        # Third event: 0.1% (lower, max unchanged)
        manager._record_slippage("BTC/USDT", Decimal("100000"), Decimal("100100"))
        assert manager._max_slippage_observed == Decimal("0.50")

    def test_negative_slippage_favorable(self, hodl_manager_with_slippage):
        """Test negative slippage (got better price) is tracked."""
        manager = hodl_manager_with_slippage
        # Favorable slippage: paid less than expected (exceeds threshold)
        expected = Decimal("100000")
        actual = Decimal("99400")  # 0.6% lower (favorable, exceeds 0.5% threshold)
        slippage = manager._record_slippage("BTC/USDT", expected, actual)

        assert slippage == Decimal("-0.60")
        assert manager._total_slippage_events == 1
        # Absolute value exceeds threshold (0.6 > 0.5)
        assert manager._slippage_warnings == 1

    def test_slippage_in_stats(self, hodl_manager_with_slippage):
        """Test slippage included in get_stats."""
        manager = hodl_manager_with_slippage

        # Record some slippage
        manager._record_slippage("BTC/USDT", Decimal("100000"), Decimal("100200"))
        manager._record_slippage("BTC/USDT", Decimal("100000"), Decimal("100800"))  # Warning

        stats = manager.get_stats()

        assert "slippage" in stats
        assert stats["slippage"]["max_slippage_pct"] == 0.5
        assert stats["slippage"]["total_events"] == 2
        assert stats["slippage"]["max_observed_pct"] == 0.8  # 0.8%
        assert stats["slippage"]["warnings"] == 1

    def test_slippage_with_zero_expected_price(self, hodl_manager_with_slippage):
        """Test slippage calculation handles zero expected price."""
        manager = hodl_manager_with_slippage
        slippage = manager._record_slippage("BTC/USDT", Decimal("0"), Decimal("100"))

        assert slippage == Decimal(0)

    @pytest.mark.asyncio
    async def test_paper_mode_records_zero_slippage(self, hodl_manager_with_slippage):
        """Test paper mode records zero slippage on execution."""
        manager = hodl_manager_with_slippage
        await manager.start()

        # Process trades to trigger execution
        for i in range(8):
            await manager.process_trade_profit(
                trade_id=f"trade-{i}",
                profit_usd=Decimal("100"),
                source_symbol="BTC/USDT",
            )

        # Slippage events should be recorded (paper mode = 0% slippage each time)
        assert manager._total_slippage_events > 0
        assert manager._max_slippage_observed == Decimal("0.00")
        assert manager._slippage_warnings == 0
