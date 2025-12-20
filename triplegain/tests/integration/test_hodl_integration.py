"""
Integration tests for Hodl Bag System.

Tests validate:
- End-to-end profit flow: Trade profit -> Hodl allocation -> Threshold execution
- Coordinator lifecycle with hodl manager
- Position tracker integration
- Database persistence patterns
- API endpoints functionality
- Message bus event publishing

Phase 8: Hodl Bag Profit Allocation System Integration Tests
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def hodl_config():
    """Create hodl bag configuration."""
    return {
        "hodl_bags": {
            "enabled": True,
            "allocation_pct": 10,
            "split": {
                "usdt_pct": 33.34,
                "xrp_pct": 33.33,
                "btc_pct": 33.33,
            },
            "min_accumulation": {
                "usdt": 1,
                "xrp": 25,
                "btc": 15,
            },
            "execution": {
                "order_type": "market",
                "max_retries": 3,
                "retry_delay_seconds": 1,  # Fast for tests
                "max_slippage_pct": 0.5,
            },
            "limits": {
                "max_single_accumulation_usd": 1000,
                "daily_accumulation_limit_usd": 5000,
                "min_profit_to_allocate_usd": 1.0,
            },
            "prices": {
                "cache_duration_seconds": 5,
                "fallback": {
                    "BTC/USDT": 100000,
                    "XRP/USDT": 2.50,
                },
            },
        }
    }


@pytest.fixture
def mock_message_bus():
    """Create mock message bus for event verification."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def mock_price_source():
    """Create mock price source function."""
    prices = {
        "BTC/USDT": Decimal("100000"),
        "XRP/USDT": Decimal("2.50"),
    }
    return lambda symbol: prices.get(symbol)


@pytest.fixture
async def hodl_manager(hodl_config, mock_message_bus, mock_price_source):
    """Create HodlBagManager instance for testing."""
    from triplegain.src.execution.hodl_bag import HodlBagManager

    manager = HodlBagManager(
        config=hodl_config,
        db_pool=None,  # No DB for unit integration tests
        kraken_client=None,
        price_source=mock_price_source,
        message_bus=mock_message_bus,
        is_paper_mode=True,
    )
    await manager.start()
    yield manager
    await manager.stop()


# =============================================================================
# End-to-End Profit Flow Tests
# =============================================================================

class TestProfitFlowIntegration:
    """Test complete profit flow from trade to hodl allocation."""

    @pytest.mark.asyncio
    async def test_profit_allocation_flow(self, hodl_manager):
        """Test full flow: profit -> allocation -> pending."""
        # Process a profitable trade
        allocation = await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        # Verify allocation was made
        assert allocation is not None
        assert allocation.total_allocation_usd == Decimal("10.00")  # 10% of $100

        # Verify split (roughly 1/3 each)
        assert allocation.usdt_amount_usd == Decimal("3.34")  # 33.34%
        assert allocation.xrp_amount_usd == Decimal("3.33")  # 33.33%
        assert allocation.btc_amount_usd == Decimal("3.33")  # 33.33%

        # Verify pending updated
        # USDT has $1 threshold, so $3.34 immediately executes
        # XRP has $25 threshold, so $3.33 stays pending
        # BTC has $15 threshold, so $3.33 stays pending
        pending = await hodl_manager.get_pending()
        assert pending["USDT"] == Decimal("0")  # Immediately executed (> $1 threshold)
        assert pending["XRP"] == Decimal("3.33")
        assert pending["BTC"] == Decimal("3.33")

    @pytest.mark.asyncio
    async def test_threshold_execution_flow(self, hodl_manager):
        """Test accumulation executes when threshold reached."""
        # Process multiple trades to reach threshold ($25 for XRP)
        # Each trade: $100 profit -> $10 allocation -> ~$3.33 per asset
        #
        # Thresholds: USDT=$1, BTC=$15, XRP=$25
        # BTC: executes at trade 5 (~$16.66), then trades 6-8 accumulate ($9.99)
        # XRP: executes at trade 8 (~$26.66)
        # USDT: executes every trade (always > $1 threshold)

        for i in range(8):
            await hodl_manager.process_trade_profit(
                trade_id=str(uuid.uuid4()),
                profit_usd=Decimal("100"),
                source_symbol="BTC/USDT",
            )

        pending = await hodl_manager.get_pending()

        # XRP: 8 trades * $3.33 = $26.64 >= $25 threshold -> executed
        assert pending["XRP"] == Decimal(0)

        # BTC: threshold $15
        # - After 5 trades: $16.65 -> executes -> pending = 0
        # - Trades 6-8: 3 * $3.33 = $9.99 (below $15 threshold, not executed)
        assert pending["BTC"] == Decimal("9.99")

        # USDT: executes after each trade (> $1 threshold)
        assert pending["USDT"] == Decimal(0)

    @pytest.mark.asyncio
    async def test_force_accumulation_bypasses_threshold(self, hodl_manager):
        """Test force accumulation works below threshold."""
        # Add small pending amount (below threshold)
        await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("30"),  # Only $3 total allocation
            source_symbol="BTC/USDT",
        )

        # XRP pending should be ~$1 (below $25 threshold)
        pending = await hodl_manager.get_pending()
        xrp_pending = pending["XRP"]
        assert xrp_pending < Decimal("25")
        assert xrp_pending > Decimal(0)

        # Force accumulation
        success = await hodl_manager.force_accumulation("XRP")
        assert success is True

        # Pending should now be 0
        pending = await hodl_manager.get_pending()
        assert pending["XRP"] == Decimal(0)

    @pytest.mark.asyncio
    async def test_daily_limit_enforced(self, hodl_manager):
        """Test daily accumulation limit is enforced."""
        # Process many large trades to exceed daily limit ($5000)
        total_allocated = Decimal(0)

        for i in range(100):  # Try to allocate way more than limit
            allocation = await hodl_manager.process_trade_profit(
                trade_id=str(uuid.uuid4()),
                profit_usd=Decimal("1000"),  # $100 allocation per trade
                source_symbol="BTC/USDT",
            )
            if allocation:
                total_allocated += allocation.total_allocation_usd

        # Should be capped at daily limit
        assert total_allocated <= Decimal("5000")


# =============================================================================
# Coordinator Integration Tests
# =============================================================================

class TestCoordinatorIntegration:
    """Test coordinator lifecycle with hodl manager."""

    @pytest.mark.asyncio
    async def test_hodl_manager_lifecycle(self, hodl_config, mock_message_bus, mock_price_source):
        """Test hodl manager start/stop lifecycle."""
        from triplegain.src.execution.hodl_bag import HodlBagManager

        manager = HodlBagManager(
            config=hodl_config,
            db_pool=None,
            kraken_client=None,
            price_source=mock_price_source,
            message_bus=mock_message_bus,
            is_paper_mode=True,
        )

        # Start
        await manager.start()
        assert manager.enabled is True

        # Process trade while running
        allocation = await manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("50"),
            source_symbol="XRP/USDT",
        )
        assert allocation is not None

        # Stop
        await manager.stop()

    @pytest.mark.asyncio
    async def test_hodl_stats_available_for_display(self, hodl_manager):
        """Test stats method returns expected data for display."""
        # Process some trades
        await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        stats = hodl_manager.get_stats()

        # Verify all required fields for coordinator display
        assert "enabled" in stats
        assert "is_paper_mode" in stats
        assert "allocation_pct" in stats
        assert "total_allocations" in stats
        assert "total_allocated_usd" in stats
        assert "pending" in stats
        assert "thresholds" in stats
        assert stats["total_allocations"] == 1
        assert stats["total_allocated_usd"] == 10.0  # 10% of $100


# =============================================================================
# Position Tracker Integration Tests
# =============================================================================

class TestPositionTrackerIntegration:
    """Test position tracker integrates with hodl manager."""

    @pytest.mark.asyncio
    async def test_hodl_allocation_on_profit(self, hodl_manager, mock_message_bus):
        """Test that profitable close triggers hodl allocation."""
        # Simulate position tracker calling hodl manager
        allocation = await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("50"),
            source_symbol="BTC/USDT",
        )

        assert allocation is not None
        assert allocation.profit_usd == Decimal("50")
        assert allocation.total_allocation_usd == Decimal("5.00")  # 10%

    @pytest.mark.asyncio
    async def test_no_allocation_on_loss(self, hodl_manager):
        """Test no allocation made on losing trade."""
        allocation = await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("-100"),  # Loss
            source_symbol="BTC/USDT",
        )

        assert allocation is None

    @pytest.mark.asyncio
    async def test_no_allocation_below_minimum(self, hodl_manager):
        """Test no allocation for tiny profits."""
        allocation = await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("0.50"),  # Below $1 minimum
            source_symbol="BTC/USDT",
        )

        assert allocation is None


# =============================================================================
# Message Bus Event Integration Tests
# =============================================================================

class TestMessageBusIntegration:
    """Test hodl manager publishes events correctly."""

    @pytest.mark.asyncio
    async def test_allocation_event_published(self, hodl_manager, mock_message_bus):
        """Test allocation event published to message bus."""
        await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        # Verify event was published
        mock_message_bus.publish.assert_called()

        # Get the last call
        calls = mock_message_bus.publish.call_args_list
        assert len(calls) > 0

        # Verify event payload
        last_message = calls[-1][0][0]  # First arg of last call
        assert last_message.payload["event_type"] == "hodl_allocation"

    @pytest.mark.asyncio
    async def test_execution_event_published(self, hodl_manager, mock_message_bus):
        """Test execution event published when threshold reached."""
        # Process enough trades to trigger execution
        for i in range(10):
            await hodl_manager.process_trade_profit(
                trade_id=str(uuid.uuid4()),
                profit_usd=Decimal("100"),
                source_symbol="BTC/USDT",
            )

        # Verify execution events were published
        calls = mock_message_bus.publish.call_args_list
        execution_events = [
            c for c in calls
            if c[0][0].payload.get("event_type") == "hodl_execution"
        ]
        assert len(execution_events) > 0


# =============================================================================
# State Serialization Tests
# =============================================================================

class TestStateSerialization:
    """Test hodl state can be serialized for API and storage."""

    @pytest.mark.asyncio
    async def test_hodl_state_serializable(self, hodl_manager):
        """Test hodl state can be serialized to JSON."""
        await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        state = await hodl_manager.get_hodl_state()

        # Should serialize without error
        for asset, bag_state in state.items():
            state_dict = bag_state.to_dict()
            json_str = json.dumps(state_dict)
            parsed = json.loads(json_str)
            assert parsed["asset"] == asset

    @pytest.mark.asyncio
    async def test_metrics_serializable(self, hodl_manager):
        """Test metrics can be serialized for API."""
        await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        metrics = await hodl_manager.calculate_metrics()

        # Should serialize without error
        json_str = json.dumps(metrics)
        parsed = json.loads(json_str)
        assert "total_cost_basis_usd" in parsed
        assert "total_allocations" in parsed


# =============================================================================
# Concurrent Operation Tests
# =============================================================================

class TestConcurrentOperations:
    """Test thread-safe concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_profit_processing(self, hodl_manager):
        """Test multiple concurrent profit allocations."""
        async def process_profit():
            return await hodl_manager.process_trade_profit(
                trade_id=str(uuid.uuid4()),
                profit_usd=Decimal("100"),
                source_symbol="BTC/USDT",
            )

        # Process 10 trades concurrently
        results = await asyncio.gather(*[process_profit() for _ in range(10)])

        # All should succeed
        successful = [r for r in results if r is not None]
        assert len(successful) == 10

        # Total allocated should be correct
        stats = hodl_manager.get_stats()
        assert stats["total_allocations"] == 10
        assert stats["total_allocated_usd"] == 100.0  # 10 * $10

    @pytest.mark.asyncio
    async def test_concurrent_state_access(self, hodl_manager):
        """Test concurrent state access is thread-safe."""
        await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        async def get_state():
            return await hodl_manager.get_hodl_state()

        async def get_pending():
            return await hodl_manager.get_pending()

        async def get_metrics():
            return await hodl_manager.calculate_metrics()

        # Access state concurrently
        results = await asyncio.gather(
            get_state(),
            get_pending(),
            get_metrics(),
            get_state(),
            get_pending(),
        )

        # All should return valid data
        assert len(results) == 5
        for r in results:
            assert r is not None


# =============================================================================
# Retry Logic Integration Tests
# =============================================================================

class TestRetryLogicIntegration:
    """Test retry logic integration (M2 fix verification)."""

    @pytest.mark.asyncio
    async def test_retry_config_applied(self, hodl_config, mock_message_bus, mock_price_source):
        """Test retry configuration is loaded."""
        from triplegain.src.execution.hodl_bag import HodlBagManager

        manager = HodlBagManager(
            config=hodl_config,
            db_pool=None,
            kraken_client=None,
            price_source=mock_price_source,
            message_bus=mock_message_bus,
            is_paper_mode=True,
        )

        assert manager.max_retries == 3
        assert manager.retry_delay_seconds == 1

    @pytest.mark.asyncio
    async def test_paper_mode_succeeds_without_retry(self, hodl_manager):
        """Test paper mode doesn't need retries."""
        # Process enough to trigger execution
        for i in range(8):
            await hodl_manager.process_trade_profit(
                trade_id=str(uuid.uuid4()),
                profit_usd=Decimal("100"),
                source_symbol="BTC/USDT",
            )

        # All should succeed in paper mode
        stats = hodl_manager.get_stats()
        assert stats["total_executions"] > 0


# =============================================================================
# Snapshot Integration Tests (L2 fix verification)
# =============================================================================

class TestSnapshotIntegration:
    """Test snapshot functionality integration."""

    @pytest.mark.asyncio
    async def test_snapshot_creation_no_db(self, hodl_manager):
        """Test snapshot creation handles no DB gracefully."""
        await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        # Should return 0 without error when no DB
        count = await hodl_manager.create_daily_snapshot()
        assert count == 0

    @pytest.mark.asyncio
    async def test_snapshots_method_no_db(self, hodl_manager):
        """Test get_snapshots handles no DB gracefully."""
        snapshots = await hodl_manager.get_snapshots(asset="BTC", days=30)
        assert snapshots == []


# =============================================================================
# Price Cache Integration Tests (M3 fix verification)
# =============================================================================

class TestPriceCacheIntegration:
    """Test price cache thread-safety."""

    @pytest.mark.asyncio
    async def test_concurrent_price_access(self, hodl_manager):
        """Test concurrent price access is thread-safe."""
        async def get_price():
            return await hodl_manager._get_current_price("BTC/USDT")

        # Access price concurrently
        results = await asyncio.gather(*[get_price() for _ in range(10)])

        # All should return same value
        assert all(r == Decimal("100000") for r in results)


# =============================================================================
# Configuration Integration Tests
# =============================================================================

class TestConfigurationIntegration:
    """Test configuration is applied correctly."""

    @pytest.mark.asyncio
    async def test_allocation_percentage_applied(self, hodl_manager):
        """Test allocation percentage from config is used."""
        allocation = await hodl_manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        assert allocation.total_allocation_usd == Decimal("10.00")  # 10%

    @pytest.mark.asyncio
    async def test_thresholds_applied(self, hodl_manager):
        """Test thresholds from config are used."""
        thresholds = hodl_manager.thresholds

        assert thresholds.usdt == Decimal("1")
        assert thresholds.xrp == Decimal("25")
        assert thresholds.btc == Decimal("15")

    @pytest.mark.asyncio
    async def test_split_percentages_applied(self, hodl_manager):
        """Test split percentages from config are used."""
        assert hodl_manager.usdt_pct == Decimal("33.34")
        assert hodl_manager.xrp_pct == Decimal("33.33")
        assert hodl_manager.btc_pct == Decimal("33.33")


# =============================================================================
# Disabled Mode Tests
# =============================================================================

class TestDisabledMode:
    """Test behavior when hodl bags are disabled."""

    @pytest.mark.asyncio
    async def test_disabled_no_allocation(self, mock_message_bus, mock_price_source):
        """Test no allocation when disabled."""
        from triplegain.src.execution.hodl_bag import HodlBagManager

        config = {
            "hodl_bags": {
                "enabled": False,
            }
        }

        manager = HodlBagManager(
            config=config,
            db_pool=None,
            kraken_client=None,
            price_source=mock_price_source,
            message_bus=mock_message_bus,
            is_paper_mode=True,
        )

        allocation = await manager.process_trade_profit(
            trade_id=str(uuid.uuid4()),
            profit_usd=Decimal("100"),
            source_symbol="BTC/USDT",
        )

        assert allocation is None
