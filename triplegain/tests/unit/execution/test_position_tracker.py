"""
Unit tests for PositionTracker - Position tracking and P&L monitoring.

Tests cover:
- Position creation and serialization
- P&L calculation
- Position lifecycle (open, modify, close)
- Exposure calculation
- Position snapshots
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from triplegain.src.execution.position_tracker import (
    Position,
    PositionTracker,
    PositionSnapshot,
    PositionSide,
    PositionStatus,
)


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test basic position creation."""
        position = Position(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )
        assert position.symbol == "BTC/USDT"
        assert position.side == PositionSide.LONG
        assert position.size == Decimal("0.1")
        assert position.entry_price == Decimal("45000")
        assert position.status == PositionStatus.OPEN

    def test_position_with_leverage(self):
        """Test position with leverage."""
        position = Position(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            leverage=5,
        )
        assert position.leverage == 5

    def test_position_with_stops(self):
        """Test position with stop-loss and take-profit."""
        position = Position(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            stop_loss=Decimal("44000"),
            take_profit=Decimal("47000"),
        )
        assert position.stop_loss == Decimal("44000")
        assert position.take_profit == Decimal("47000")

    def test_calculate_pnl_long_profit(self):
        """Test P&L calculation for profitable long position."""
        position = Position(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            leverage=1,
        )
        # Price went up to 46000
        pnl, pnl_pct = position.calculate_pnl(Decimal("46000"))

        # (46000 - 45000) * 0.1 = 100
        assert pnl == Decimal("100")
        # (1000 / 45000) * 100 = 2.22%
        assert abs(float(pnl_pct) - 2.222) < 0.01

    def test_calculate_pnl_long_loss(self):
        """Test P&L calculation for losing long position."""
        position = Position(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            leverage=1,
        )
        # Price went down to 44000
        pnl, pnl_pct = position.calculate_pnl(Decimal("44000"))

        # (44000 - 45000) * 0.1 = -100
        assert pnl == Decimal("-100")

    def test_calculate_pnl_short_profit(self):
        """Test P&L calculation for profitable short position."""
        position = Position(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            leverage=1,
        )
        # Price went down to 44000
        pnl, pnl_pct = position.calculate_pnl(Decimal("44000"))

        # (45000 - 44000) * 0.1 = 100
        assert pnl == Decimal("100")

    def test_calculate_pnl_short_loss(self):
        """Test P&L calculation for losing short position."""
        position = Position(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            leverage=1,
        )
        # Price went up to 46000
        pnl, pnl_pct = position.calculate_pnl(Decimal("46000"))

        # (45000 - 46000) * 0.1 = -100
        assert pnl == Decimal("-100")

    def test_calculate_pnl_with_leverage(self):
        """Test P&L calculation with leverage."""
        position = Position(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            leverage=5,
        )
        # Price went up to 46000
        pnl, pnl_pct = position.calculate_pnl(Decimal("46000"))

        # (46000 - 45000) * 0.1 * 5 = 500
        assert pnl == Decimal("500")
        # PnL% is also leveraged
        assert float(pnl_pct) > 10  # Should be ~11.11%

    def test_calculate_pnl_zero_entry(self):
        """Test P&L calculation with zero entry price."""
        position = Position(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("0"),
        )
        pnl, pnl_pct = position.calculate_pnl(Decimal("45000"))
        assert pnl == Decimal("0")
        assert pnl_pct == Decimal("0")

    def test_update_pnl(self):
        """Test updating P&L with current price."""
        position = Position(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )
        position.update_pnl(Decimal("46000"))

        assert position.unrealized_pnl == Decimal("100")
        assert float(position.unrealized_pnl_pct) > 2

    def test_position_to_dict(self):
        """Test position serialization to dictionary."""
        position = Position(
            id="test-123",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            leverage=2,
            stop_loss=Decimal("44000"),
            take_profit=Decimal("47000"),
        )
        d = position.to_dict()
        assert d["id"] == "test-123"
        assert d["symbol"] == "BTC/USDT"
        assert d["side"] == "long"
        assert d["size"] == "0.1"
        assert d["entry_price"] == "45000"
        assert d["leverage"] == 2

    def test_position_from_dict(self):
        """Test position deserialization from dictionary."""
        data = {
            "id": "test-123",
            "symbol": "BTC/USDT",
            "side": "long",
            "size": "0.1",
            "entry_price": "45000",
            "leverage": 2,
            "status": "open",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        position = Position.from_dict(data)
        assert position.id == "test-123"
        assert position.symbol == "BTC/USDT"
        assert position.side == PositionSide.LONG
        assert position.size == Decimal("0.1")


class TestPositionSnapshot:
    """Tests for PositionSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test snapshot creation."""
        snapshot = PositionSnapshot(
            position_id="pos-123",
            symbol="BTC/USDT",
            timestamp=datetime.now(timezone.utc),
            current_price=Decimal("45500"),
            unrealized_pnl=Decimal("50"),
            unrealized_pnl_pct=Decimal("1.11"),
        )
        assert snapshot.position_id == "pos-123"
        assert snapshot.current_price == Decimal("45500")

    def test_snapshot_to_dict(self):
        """Test snapshot serialization."""
        snapshot = PositionSnapshot(
            position_id="pos-123",
            symbol="BTC/USDT",
            timestamp=datetime.now(timezone.utc),
            current_price=Decimal("45500"),
            unrealized_pnl=Decimal("50"),
            unrealized_pnl_pct=Decimal("1.11"),
        )
        d = snapshot.to_dict()
        assert d["position_id"] == "pos-123"
        assert d["current_price"] == "45500"


class TestPositionSide:
    """Tests for PositionSide enum."""

    def test_sides_exist(self):
        """Test expected sides exist."""
        assert PositionSide.LONG.value == "long"
        assert PositionSide.SHORT.value == "short"


class TestPositionStatus:
    """Tests for PositionStatus enum."""

    def test_statuses_exist(self):
        """Test expected statuses exist."""
        expected = ["open", "closing", "closed", "liquidated"]
        status_values = [s.value for s in PositionStatus]
        for expected_status in expected:
            assert expected_status in status_values


@pytest.fixture
def mock_message_bus():
    """Create a mock message bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock(return_value=1)
    return bus


@pytest.fixture
def mock_risk_engine():
    """Create a mock risk engine."""
    engine = MagicMock()
    engine.record_trade_result = MagicMock()
    engine.apply_post_trade_cooldown = MagicMock()
    engine.update_positions = MagicMock()
    return engine


@pytest.fixture
def position_tracker_config():
    """Create position tracker configuration."""
    return {
        "position_tracking": {
            "snapshot_interval_seconds": 60,
            "max_snapshots": 1000,
        },
    }


class TestPositionTracker:
    """Tests for PositionTracker."""

    @pytest.fixture
    def position_tracker(self, mock_message_bus, mock_risk_engine, position_tracker_config):
        """Create a position tracker instance for testing."""
        return PositionTracker(
            message_bus=mock_message_bus,
            risk_engine=mock_risk_engine,
            db_pool=None,
            config=position_tracker_config,
        )

    def test_tracker_creation(self, position_tracker):
        """Test position tracker creation."""
        assert position_tracker._snapshot_interval_seconds == 60
        assert len(position_tracker._positions) == 0

    @pytest.mark.asyncio
    async def test_open_position(self, position_tracker, mock_message_bus):
        """Test opening a position."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            leverage=2,
            order_id="order-123",
        )

        assert position is not None
        assert position.symbol == "BTC/USDT"
        assert position.side == PositionSide.LONG
        assert position.size == Decimal("0.1")
        assert position.entry_price == Decimal("45000")
        assert position.leverage == 2
        assert position.order_id == "order-123"
        assert position.status == PositionStatus.OPEN

        # Should be tracked
        assert position.id in position_tracker._positions

        # Should publish event
        mock_message_bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_open_position_with_stops(self, position_tracker):
        """Test opening a position with stop-loss and take-profit."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            stop_loss=Decimal("44000"),
            take_profit=Decimal("47000"),
        )

        assert position.stop_loss == Decimal("44000")
        assert position.take_profit == Decimal("47000")

    @pytest.mark.asyncio
    async def test_close_position(self, position_tracker, mock_message_bus, mock_risk_engine):
        """Test closing a position."""
        # Open position first
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )

        # Close position
        closed = await position_tracker.close_position(
            position_id=position.id,
            exit_price=Decimal("46000"),
            reason="take_profit",
        )

        assert closed is not None
        assert closed.status == PositionStatus.CLOSED
        assert closed.exit_price == Decimal("46000")
        assert closed.realized_pnl == Decimal("100")  # (46000-45000) * 0.1

        # Should be removed from open positions
        assert position.id not in position_tracker._positions

        # Should be in closed positions
        assert closed in position_tracker._closed_positions

        # Should report to risk engine
        mock_risk_engine.record_trade_result.assert_called_with(True)

    @pytest.mark.asyncio
    async def test_close_position_with_loss(self, position_tracker, mock_risk_engine):
        """Test closing a position with a loss."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )

        closed = await position_tracker.close_position(
            position_id=position.id,
            exit_price=Decimal("44000"),
        )

        assert closed.realized_pnl == Decimal("-100")
        mock_risk_engine.record_trade_result.assert_called_with(False)

    @pytest.mark.asyncio
    async def test_close_position_not_found(self, position_tracker):
        """Test closing a non-existent position."""
        result = await position_tracker.close_position(
            position_id="nonexistent",
            exit_price=Decimal("45000"),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_close_already_closed_position(self, position_tracker):
        """Test closing an already closed position."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )
        position.status = PositionStatus.CLOSED
        position_tracker._positions[position.id] = position

        result = await position_tracker.close_position(
            position_id=position.id,
            exit_price=Decimal("46000"),
        )
        assert result.status == PositionStatus.CLOSED

    @pytest.mark.asyncio
    async def test_modify_position_stop_loss(self, position_tracker):
        """Test modifying position stop-loss."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            stop_loss=Decimal("44000"),
        )

        modified = await position_tracker.modify_position(
            position_id=position.id,
            stop_loss=Decimal("44500"),
        )

        assert modified is not None
        assert modified.stop_loss == Decimal("44500")

    @pytest.mark.asyncio
    async def test_modify_position_take_profit(self, position_tracker):
        """Test modifying position take-profit."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            take_profit=Decimal("47000"),
        )

        modified = await position_tracker.modify_position(
            position_id=position.id,
            take_profit=Decimal("48000"),
        )

        assert modified is not None
        assert modified.take_profit == Decimal("48000")

    @pytest.mark.asyncio
    async def test_modify_position_not_found(self, position_tracker):
        """Test modifying a non-existent position."""
        result = await position_tracker.modify_position(
            position_id="nonexistent",
            stop_loss=Decimal("44000"),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_position(self, position_tracker):
        """Test getting a specific position."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )

        found = await position_tracker.get_position(position.id)
        assert found is not None
        assert found.id == position.id

        not_found = await position_tracker.get_position("nonexistent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_get_open_positions(self, position_tracker):
        """Test getting open positions."""
        await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )
        await position_tracker.open_position(
            symbol="XRP/USDT",
            side="short",
            size=Decimal("100"),
            entry_price=Decimal("0.60"),
        )

        # All positions
        positions = await position_tracker.get_open_positions()
        assert len(positions) == 2

        # Filter by symbol
        btc_positions = await position_tracker.get_open_positions("BTC/USDT")
        assert len(btc_positions) == 1
        assert btc_positions[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_get_closed_positions(self, position_tracker):
        """Test getting closed positions."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )
        await position_tracker.close_position(position.id, Decimal("46000"))

        closed = await position_tracker.get_closed_positions()
        assert len(closed) == 1
        assert closed[0].status == PositionStatus.CLOSED

    @pytest.mark.asyncio
    async def test_update_prices(self, position_tracker):
        """Test updating prices and recalculating P&L."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )

        await position_tracker.update_prices({"BTC/USDT": Decimal("46000")})

        updated = await position_tracker.get_position(position.id)
        assert updated.unrealized_pnl == Decimal("100")

    @pytest.mark.asyncio
    async def test_get_total_exposure(self, position_tracker):
        """Test calculating total exposure."""
        await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
            leverage=2,
        )

        position_tracker._price_cache["BTC/USDT"] = Decimal("45000")

        exposure = await position_tracker.get_total_exposure()
        # 0.1 * 45000 * 2 = 9000
        assert exposure["total_exposure_usd"] == 9000.0
        assert "BTC/USDT" in exposure["exposure_by_symbol"]

    @pytest.mark.asyncio
    async def test_get_total_unrealized_pnl(self, position_tracker):
        """Test getting total unrealized P&L."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )
        position.unrealized_pnl = Decimal("100")
        position_tracker._positions[position.id] = position

        pnl = await position_tracker.get_total_unrealized_pnl()
        assert pnl["total_unrealized_pnl"] == 100.0

    @pytest.mark.asyncio
    async def test_start_stop(self, position_tracker):
        """Test starting and stopping the tracker."""
        await position_tracker.start()
        assert position_tracker._running is True
        assert position_tracker._snapshot_task is not None

        await position_tracker.stop()
        assert position_tracker._running is False

    def test_get_stats(self, position_tracker):
        """Test getting tracker statistics."""
        stats = position_tracker.get_stats()
        assert "total_positions_opened" in stats
        assert "total_positions_closed" in stats
        assert "open_positions_count" in stats
        assert "total_realized_pnl" in stats


class TestPositionTrackerStatistics:
    """Tests for position tracker statistics."""

    @pytest.fixture
    def position_tracker(self, mock_message_bus, mock_risk_engine, position_tracker_config):
        """Create a position tracker instance for testing."""
        return PositionTracker(
            message_bus=mock_message_bus,
            risk_engine=mock_risk_engine,
            db_pool=None,
            config=position_tracker_config,
        )

    @pytest.mark.asyncio
    async def test_statistics_updated_on_open(self, position_tracker):
        """Test statistics are updated when opening a position."""
        await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )

        assert position_tracker._total_positions_opened == 1

    @pytest.mark.asyncio
    async def test_statistics_updated_on_close(self, position_tracker):
        """Test statistics are updated when closing a position."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("45000"),
        )
        await position_tracker.close_position(position.id, Decimal("46000"))

        assert position_tracker._total_positions_closed == 1
        assert position_tracker._total_realized_pnl == Decimal("100")


class TestPositionTrackerNewFeatures:
    """Tests for new PositionTracker features (F05, F07, F09, F14)."""

    @pytest.fixture
    def mock_message_bus(self):
        """Create mock message bus."""
        return AsyncMock()

    @pytest.fixture
    def position_tracker_config(self):
        """Create position tracker config."""
        return {
            "position_tracking": {
                "snapshot_interval_seconds": 60,
                "max_snapshots": 1000,
                "trigger_check_interval_seconds": 5,
            },
            "trailing_stop": {
                "enabled": True,
                "activation_pct": 1.0,
                "trail_distance_pct": 1.5,
            },
        }

    @pytest.fixture
    def position_tracker(self, mock_message_bus, position_tracker_config):
        """Create a position tracker instance for testing."""
        return PositionTracker(
            message_bus=mock_message_bus,
            risk_engine=None,
            db_pool=None,
            config=position_tracker_config,
        )

    def test_position_has_new_fields(self):
        """Test F14: Position has order ID reference fields."""
        position = Position(
            id="test-123",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )
        assert hasattr(position, "stop_loss_order_id")
        assert hasattr(position, "take_profit_order_id")
        assert hasattr(position, "external_id")
        assert hasattr(position, "total_fees")
        assert hasattr(position, "fee_currency")

    def test_position_new_fields_in_to_dict(self):
        """Test F14: New fields included in serialization."""
        position = Position(
            id="test-123",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            stop_loss_order_id="sl-order-456",
            take_profit_order_id="tp-order-789",
            external_id="kraken-pos-123",
            total_fees=Decimal("0.5"),
            fee_currency="USDT",
        )
        d = position.to_dict()
        assert d["stop_loss_order_id"] == "sl-order-456"
        assert d["take_profit_order_id"] == "tp-order-789"
        assert d["external_id"] == "kraken-pos-123"
        assert d["total_fees"] == "0.5"
        assert d["fee_currency"] == "USDT"

    def test_position_from_dict_with_new_fields(self):
        """Test F14: Deserialization includes new fields."""
        data = {
            "id": "test-123",
            "symbol": "BTC/USDT",
            "side": "long",
            "size": "0.1",
            "entry_price": "50000",
            "stop_loss_order_id": "sl-order-456",
            "take_profit_order_id": "tp-order-789",
            "external_id": "kraken-pos-123",
            "total_fees": "0.5",
            "fee_currency": "USDT",
        }
        position = Position.from_dict(data)
        assert position.stop_loss_order_id == "sl-order-456"
        assert position.take_profit_order_id == "tp-order-789"
        assert position.external_id == "kraken-pos-123"
        assert position.total_fees == Decimal("0.5")
        assert position.fee_currency == "USDT"

    @pytest.mark.asyncio
    async def test_enable_trailing_stop_async(self, position_tracker):
        """Test F07: enable_trailing_stop_for_position is async and uses lock."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        result = await position_tracker.enable_trailing_stop_for_position(
            position.id,
            distance_pct=Decimal("2.0"),
        )
        assert result is True

        updated = await position_tracker.get_position(position.id)
        assert updated.trailing_stop_enabled is True
        assert updated.trailing_stop_distance_pct == Decimal("2.0")

    @pytest.mark.asyncio
    async def test_enable_trailing_stop_not_found(self, position_tracker):
        """Test F07: enable_trailing_stop returns False for non-existent position."""
        result = await position_tracker.enable_trailing_stop_for_position(
            "nonexistent-id",
            distance_pct=Decimal("2.0"),
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_order_links(self, position_tracker):
        """Test F14: Update position with order links."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        result = await position_tracker.update_order_links(
            position.id,
            stop_loss_order_id="sl-order-123",
            take_profit_order_id="tp-order-456",
        )
        assert result is True

        updated = await position_tracker.get_position(position.id)
        assert updated.stop_loss_order_id == "sl-order-123"
        assert updated.take_profit_order_id == "tp-order-456"

    @pytest.mark.asyncio
    async def test_update_order_links_not_found(self, position_tracker):
        """Test F14: Update order links returns False for non-existent position."""
        result = await position_tracker.update_order_links(
            "nonexistent-id",
            stop_loss_order_id="sl-order-123",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_sync_with_exchange_no_client(self, position_tracker):
        """Test F05: sync_with_exchange handles no client gracefully."""
        result = await position_tracker.sync_with_exchange(kraken_client=None)
        assert "error" in result
        assert result["error"] == "No Kraken client available"

    @pytest.mark.asyncio
    async def test_sync_with_exchange_success(self, position_tracker):
        """Test F05: sync_with_exchange detects discrepancies."""
        mock_kraken = AsyncMock()
        mock_kraken.open_positions.return_value = {
            "result": {
                "pos-123": {
                    "pair": "XBTUSDT",
                    "type": "buy",
                    "vol": "0.1",
                    "cost": "5000",
                },
            }
        }

        result = await position_tracker.sync_with_exchange(mock_kraken)
        assert result["exchange_position_count"] == 1
        assert result["local_position_count"] == 0
        # Unknown position on exchange should be flagged
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["type"] == "unknown_position"

    @pytest.mark.asyncio
    async def test_sync_with_exchange_detects_size_mismatch(self, position_tracker):
        """Test F05: sync_with_exchange detects size mismatches."""
        # Open a position locally
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )
        position.external_id = "pos-123"
        position_tracker._positions[position.id] = position

        mock_kraken = AsyncMock()
        mock_kraken.open_positions.return_value = {
            "result": {
                "pos-123": {
                    "pair": "XBTUSDT",
                    "type": "buy",
                    "vol": "0.2",  # Different size than local
                    "cost": "10000",
                },
            }
        }

        result = await position_tracker.sync_with_exchange(mock_kraken)
        assert result["updated"] == 1
        assert any(a["type"] == "size_mismatch" for a in result["alerts"])

    @pytest.mark.asyncio
    async def test_find_position_by_external_id(self, position_tracker):
        """Test F05: Helper to find position by external ID."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )
        position.external_id = "kraken-123"
        position_tracker._positions[position.id] = position

        found = position_tracker._find_position_by_external_id("kraken-123")
        assert found is not None
        assert found.id == position.id

        not_found = position_tracker._find_position_by_external_id("nonexistent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_trigger_check_interval_configured(self, position_tracker):
        """Test F09: Trigger check interval is configurable."""
        assert position_tracker._trigger_check_interval_seconds == 5

    @pytest.mark.asyncio
    async def test_start_creates_trigger_check_task(self, position_tracker):
        """Test F09: Start creates separate trigger check task."""
        await position_tracker.start()
        assert position_tracker._running is True
        assert position_tracker._snapshot_task is not None
        assert position_tracker._trigger_check_task is not None

        await position_tracker.stop()
        assert position_tracker._running is False

    @pytest.mark.asyncio
    async def test_close_position_with_order_manager(self, position_tracker):
        """Test F12: close_position cancels orphan orders."""
        mock_order_manager = AsyncMock()
        mock_order_manager.cancel_orphan_orders = AsyncMock(return_value=2)

        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        closed = await position_tracker.close_position(
            position.id,
            exit_price=Decimal("51000"),
            reason="test",
            order_manager=mock_order_manager,
        )
        assert closed is not None
        mock_order_manager.cancel_orphan_orders.assert_called_once_with(position.id)

    @pytest.mark.asyncio
    async def test_modify_position_with_order_manager(self, position_tracker):
        """Test F08: modify_position updates exchange orders."""
        mock_order_manager = AsyncMock()
        mock_order_manager.cancel_order = AsyncMock(return_value=True)
        mock_sl_order = MagicMock(id="new-sl-123")
        mock_order_manager.place_stop_loss_update = AsyncMock(return_value=mock_sl_order)

        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            stop_loss=Decimal("49000"),
        )
        position.stop_loss_order_id = "old-sl-order"
        position_tracker._positions[position.id] = position

        modified = await position_tracker.modify_position(
            position.id,
            stop_loss=Decimal("49500"),
            order_manager=mock_order_manager,
        )
        assert modified is not None
        mock_order_manager.cancel_order.assert_called()
        mock_order_manager.place_stop_loss_update.assert_called()

    @pytest.mark.asyncio
    async def test_check_sl_tp_triggers_long_stop_loss(self, position_tracker):
        """Test SL trigger detection for LONG position."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            stop_loss=Decimal("49000"),
        )

        # Price drops below SL
        triggered = await position_tracker.check_sl_tp_triggers({
            "BTC/USDT": Decimal("48500"),
        })
        assert len(triggered) == 1
        assert triggered[0][0].id == position.id
        assert triggered[0][1] == "stop_loss"

    @pytest.mark.asyncio
    async def test_check_sl_tp_triggers_long_take_profit(self, position_tracker):
        """Test TP trigger detection for LONG position."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            take_profit=Decimal("52000"),
        )

        # Price rises above TP
        triggered = await position_tracker.check_sl_tp_triggers({
            "BTC/USDT": Decimal("52500"),
        })
        assert len(triggered) == 1
        assert triggered[0][0].id == position.id
        assert triggered[0][1] == "take_profit"

    @pytest.mark.asyncio
    async def test_check_sl_tp_triggers_short_stop_loss(self, position_tracker):
        """Test SL trigger detection for SHORT position."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="short",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            stop_loss=Decimal("51000"),
        )

        # Price rises above SL for SHORT
        triggered = await position_tracker.check_sl_tp_triggers({
            "BTC/USDT": Decimal("51500"),
        })
        assert len(triggered) == 1
        assert triggered[0][0].id == position.id
        assert triggered[0][1] == "stop_loss"

    @pytest.mark.asyncio
    async def test_check_sl_tp_triggers_short_take_profit(self, position_tracker):
        """Test TP trigger detection for SHORT position."""
        position = await position_tracker.open_position(
            symbol="BTC/USDT",
            side="short",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            take_profit=Decimal("48000"),
        )

        # Price drops below TP for SHORT
        triggered = await position_tracker.check_sl_tp_triggers({
            "BTC/USDT": Decimal("47500"),
        })
        assert len(triggered) == 1
        assert triggered[0][0].id == position.id
        assert triggered[0][1] == "take_profit"
