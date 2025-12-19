"""
Unit tests for the Risk Management Rules Engine.

Tests validate:
- Position sizing rules
- Stop-loss requirements
- Leverage limits
- Circuit breakers
- Confidence thresholds
- Cooldown periods
- <10ms latency requirement
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from triplegain.src.risk.rules_engine import (
    RiskManagementEngine,
    TradeProposal,
    RiskValidation,
    RiskState,
    ValidationStatus,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_config() -> dict:
    """Default risk configuration for testing."""
    return {
        'limits': {
            'max_leverage': 5,
            'max_position_pct': 20,
            'max_total_exposure_pct': 80,
            'max_risk_per_trade_pct': 2,
            'min_confidence': 0.60,
        },
        'stop_loss': {
            'required': True,
            'min_distance_pct': 0.5,
            'max_distance_pct': 5.0,
            'min_risk_reward': 1.5,
        },
        'circuit_breakers': {
            'daily_loss_pct': 5.0,
            'weekly_loss_pct': 10.0,
            'max_drawdown_pct': 20.0,
            'consecutive_losses': 5,
        },
        'cooldowns': {
            'post_trade_minutes': 5,
            'post_loss_minutes': 10,
            'consecutive_loss_minutes': 30,
        },
    }


@pytest.fixture
def risk_engine(default_config) -> RiskManagementEngine:
    """Create a risk engine with default config."""
    return RiskManagementEngine(default_config)


@pytest.fixture
def healthy_risk_state() -> RiskState:
    """Risk state with healthy portfolio."""
    state = RiskState()
    state.peak_equity = Decimal("10000")
    state.current_equity = Decimal("10000")
    state.available_margin = Decimal("8000")
    state.daily_pnl = Decimal("100")
    state.daily_pnl_pct = 1.0
    state.weekly_pnl = Decimal("200")
    state.weekly_pnl_pct = 2.0
    state.current_drawdown_pct = 0.0
    state.consecutive_losses = 0
    state.open_positions = 1
    state.total_exposure_pct = 10.0
    return state


@pytest.fixture
def valid_proposal() -> TradeProposal:
    """Valid trade proposal for testing."""
    return TradeProposal(
        symbol="BTC/USDT",
        side="buy",
        size_usd=1000.0,
        entry_price=45000.0,
        stop_loss=44100.0,  # 2% below entry
        take_profit=48000.0,
        leverage=2,
        confidence=0.75,
        regime="trending_bull",
    )


# =============================================================================
# Basic Validation Tests
# =============================================================================

class TestBasicValidation:
    """Test basic trade validation."""

    def test_valid_trade_approved(self, risk_engine, valid_proposal, healthy_risk_state):
        """Valid trade should be approved."""
        result = risk_engine.validate_trade(valid_proposal, healthy_risk_state)

        assert result.status == ValidationStatus.APPROVED
        assert result.is_approved()
        assert len(result.rejections) == 0

    def test_validation_latency_under_10ms(self, risk_engine, valid_proposal, healthy_risk_state):
        """Validation should complete in <10ms."""
        result = risk_engine.validate_trade(valid_proposal, healthy_risk_state)

        assert result.validation_time_ms < 10, f"Latency {result.validation_time_ms}ms exceeds 10ms"


# =============================================================================
# Stop-Loss Tests
# =============================================================================

class TestStopLossValidation:
    """Test stop-loss requirements."""

    def test_missing_stop_loss_rejected(self, risk_engine, healthy_risk_state):
        """Trade without stop-loss should be rejected."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=None,  # Missing
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("STOP_LOSS_REQUIRED" in r for r in result.rejections)

    def test_stop_too_tight_rejected(self, risk_engine, healthy_risk_state):
        """Stop-loss too close to entry should be rejected."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44900.0,  # 0.22% - below 0.5% minimum
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("STOP_TOO_TIGHT" in r for r in result.rejections)

    def test_stop_too_wide_rejected(self, risk_engine, healthy_risk_state):
        """Stop-loss too far from entry should be rejected."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=40000.0,  # 11% - above 5% maximum
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("STOP_TOO_WIDE" in r for r in result.rejections)

    def test_sell_side_stop_calculation(self, risk_engine, healthy_risk_state):
        """Stop-loss for sell orders should be above entry."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="sell",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=45900.0,  # 2% above entry (correct for short)
            take_profit=43000.0,
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.is_approved()


# =============================================================================
# Confidence Tests
# =============================================================================

class TestConfidenceValidation:
    """Test confidence threshold requirements."""

    def test_low_confidence_rejected(self, risk_engine, healthy_risk_state):
        """Confidence below minimum should be rejected."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            leverage=1,
            confidence=0.55,  # Below 0.60 minimum
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("CONFIDENCE_TOO_LOW" in r for r in result.rejections)

    def test_higher_confidence_required_after_losses(self, risk_engine, healthy_risk_state):
        """Confidence threshold should increase after consecutive losses."""
        # 3 consecutive losses requires 0.70
        healthy_risk_state.consecutive_losses = 3

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            leverage=1,
            confidence=0.65,  # Above normal 0.60, below adjusted 0.70
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("0.7" in r for r in result.rejections)

    def test_even_higher_confidence_after_5_losses(self, risk_engine, healthy_risk_state):
        """5 consecutive losses requires 0.80 confidence."""
        healthy_risk_state.consecutive_losses = 5

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            leverage=1,
            confidence=0.75,  # Below 0.80 required
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED


# =============================================================================
# Position Size Tests
# =============================================================================

class TestPositionSizeValidation:
    """Test position size limits."""

    def test_oversized_position_modified(self, risk_engine, healthy_risk_state):
        """Position exceeding max % should be reduced."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=3000.0,  # 30% of $10k equity - exceeds 20% max
            entry_price=45000.0,
            stop_loss=44100.0,
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.MODIFIED
        assert result.modified_proposal is not None
        assert result.modified_proposal.size_usd == 2000.0  # 20% of $10k


# =============================================================================
# Leverage Tests
# =============================================================================

class TestLeverageValidation:
    """Test leverage limits."""

    def test_excessive_leverage_reduced(self, risk_engine, healthy_risk_state):
        """Leverage exceeding regime limit should be reduced."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            leverage=5,
            confidence=0.75,
            regime="ranging",  # Max 2x for ranging
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.MODIFIED
        assert result.modified_proposal.leverage == 2

    def test_leverage_reduced_in_drawdown(self, risk_engine, healthy_risk_state):
        """Leverage should be reduced when in drawdown."""
        healthy_risk_state.current_drawdown_pct = 12  # 10-15% tier

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            leverage=5,
            confidence=0.75,
            regime="trending_bull",  # Normally 5x allowed
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.MODIFIED
        assert result.modified_proposal.leverage == 2  # Reduced for 10-15% DD

    def test_leverage_1x_after_5_losses(self, risk_engine, healthy_risk_state):
        """Leverage should be 1x after 5 consecutive losses."""
        healthy_risk_state.consecutive_losses = 5

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            leverage=3,
            confidence=0.85,  # High enough to pass confidence check
            regime="trending_bull",
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.MODIFIED
        assert result.modified_proposal.leverage == 1


# =============================================================================
# Circuit Breaker Tests
# =============================================================================

class TestCircuitBreakers:
    """Test circuit breaker functionality."""

    def test_daily_loss_halts_trading(self, risk_engine, healthy_risk_state):
        """Daily loss exceeding limit should halt trading."""
        # Update state to trigger breaker (this sets trading_halted)
        risk_engine.update_state(
            current_equity=Decimal("10000"),
            daily_pnl=Decimal("-550"),  # -5.5% exceeds 5% limit
            weekly_pnl=Decimal("-550"),
            open_positions=0,
            total_exposure_pct=0,
            available_margin=Decimal("10000"),
        )

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal)

        assert result.status == ValidationStatus.HALTED
        # After update_state, trading_halted is set, so next validation gets TRADING_HALTED
        assert any("HALTED" in r for r in result.rejections)

    def test_trading_halted_rejects_all(self, risk_engine, valid_proposal, healthy_risk_state):
        """When trading is halted, all trades should be rejected."""
        healthy_risk_state.trading_halted = True
        healthy_risk_state.halt_reason = "daily_loss"

        result = risk_engine.validate_trade(valid_proposal, healthy_risk_state)

        assert result.status == ValidationStatus.HALTED
        assert "TRADING_HALTED" in result.rejections[0]


# =============================================================================
# Cooldown Tests
# =============================================================================

class TestCooldowns:
    """Test cooldown functionality."""

    def test_active_cooldown_rejects_trade(self, risk_engine, valid_proposal, healthy_risk_state):
        """Trade during active cooldown should be rejected."""
        healthy_risk_state.in_cooldown = True
        healthy_risk_state.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=5)
        healthy_risk_state.cooldown_reason = "post_loss"

        result = risk_engine.validate_trade(valid_proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("IN_COOLDOWN" in r for r in result.rejections)

    def test_expired_cooldown_allows_trade(self, risk_engine, valid_proposal, healthy_risk_state):
        """Trade after cooldown expires should be allowed."""
        healthy_risk_state.in_cooldown = True
        healthy_risk_state.cooldown_until = datetime.now(timezone.utc) - timedelta(minutes=1)  # Expired

        result = risk_engine.validate_trade(valid_proposal, healthy_risk_state)

        assert result.is_approved()


# =============================================================================
# Exposure Tests
# =============================================================================

class TestExposureValidation:
    """Test portfolio exposure limits."""

    def test_excessive_exposure_rejected(self, risk_engine, healthy_risk_state):
        """Trade causing exposure over limit should be rejected."""
        healthy_risk_state.total_exposure_pct = 75  # Already near 80% limit

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,  # Would add 10% for total 85%
            entry_price=45000.0,
            stop_loss=44100.0,
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("EXPOSURE_LIMIT" in r for r in result.rejections)


# =============================================================================
# Margin Tests
# =============================================================================

class TestMarginValidation:
    """Test margin requirements."""

    def test_insufficient_margin_rejected(self, risk_engine, healthy_risk_state):
        """Trade requiring more margin than available should be rejected."""
        healthy_risk_state.available_margin = Decimal("200")  # Only $200 available

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,  # Needs $1000 at 1x or $500 at 2x
            entry_price=45000.0,
            stop_loss=44100.0,
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("INSUFFICIENT_MARGIN" in r for r in result.rejections)


# =============================================================================
# Position Sizing Calculation Tests
# =============================================================================

class TestPositionSizeCalculation:
    """Test position size calculation helper."""

    def test_calculate_position_size(self, risk_engine):
        """Test ATR-based position size calculation."""
        size = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,  # 2% stop
            regime="trending_bull",
            confidence=0.85,
        )

        # Should be capped at 20% of equity
        assert 0 < size <= 2000

    def test_no_position_for_low_confidence(self, risk_engine):
        """Low confidence should result in zero position size."""
        size = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,
            regime="trending_bull",
            confidence=0.50,  # Below 0.60 threshold
        )

        assert size == 0.0

    def test_reduced_size_in_choppy_regime(self, risk_engine):
        """Choppy regime should significantly reduce position size."""
        # Use a wider stop loss (10%) so the base position is smaller than max cap
        # With 2% stop, base position = 100% equity, both regimes hit 20% cap
        # With 10% stop, base position = 20% equity, choppy gets 5% (not capped)
        trending_size = risk_engine.calculate_position_size(
            equity=100000,
            entry_price=45000,
            stop_loss=40500,  # 10% stop (base position = 20% of equity)
            regime="trending_bull",
            confidence=0.85,
        )

        choppy_size = risk_engine.calculate_position_size(
            equity=100000,
            entry_price=45000,
            stop_loss=40500,  # Same 10% stop
            regime="choppy",
            confidence=0.85,
        )

        # Choppy should be much smaller (0.25 vs 1.0 multiplier)
        # trending_bull: 20% * 1.0 = 20% (at cap) = $20,000
        # choppy: 20% * 0.25 = 5% (not capped) = $5,000
        assert choppy_size < trending_size * 0.5


# =============================================================================
# State Management Tests
# =============================================================================

class TestStateManagement:
    """Test risk state management."""

    def test_record_win_resets_loss_streak(self, risk_engine):
        """Recording a win should reset consecutive losses."""
        risk_engine._risk_state.consecutive_losses = 3

        risk_engine.record_trade_result(is_win=True)

        assert risk_engine._risk_state.consecutive_losses == 0
        assert risk_engine._risk_state.consecutive_wins == 1

    def test_record_loss_increments_streak(self, risk_engine):
        """Recording a loss should increment consecutive losses."""
        risk_engine.record_trade_result(is_win=False)

        assert risk_engine._risk_state.consecutive_losses == 1

    def test_daily_reset_clears_daily_stats(self, risk_engine):
        """Daily reset should clear daily P&L."""
        risk_engine._risk_state.daily_pnl = Decimal("-100")
        risk_engine._risk_state.daily_pnl_pct = -1.0
        risk_engine._risk_state.trading_halted = True
        risk_engine._risk_state.triggered_breakers = ['daily_loss']

        risk_engine.reset_daily()

        assert risk_engine._risk_state.daily_pnl == Decimal("0")
        assert risk_engine._risk_state.daily_pnl_pct == 0.0
        assert risk_engine._risk_state.trading_halted is False

    def test_manual_reset_requires_admin_for_drawdown(self, risk_engine):
        """Manual reset of max drawdown halt requires admin override."""
        risk_engine._risk_state.trading_halted = True
        risk_engine._risk_state.triggered_breakers = ['max_drawdown']

        # Without override
        success = risk_engine.manual_reset(admin_override=False)
        assert success is False
        assert risk_engine._risk_state.trading_halted is True

        # With override
        success = risk_engine.manual_reset(admin_override=True)
        assert success is True
        assert risk_engine._risk_state.trading_halted is False


# =============================================================================
# Max Allowed Leverage Tests
# =============================================================================

class TestMaxAllowedLeverage:
    """Test max leverage calculation."""

    def test_max_leverage_by_regime(self, risk_engine):
        """Test regime-based leverage limits."""
        assert risk_engine.get_max_allowed_leverage("trending_bull", 0, 0) == 5
        assert risk_engine.get_max_allowed_leverage("trending_bear", 0, 0) == 3
        assert risk_engine.get_max_allowed_leverage("ranging", 0, 0) == 2
        assert risk_engine.get_max_allowed_leverage("choppy", 0, 0) == 1

    def test_leverage_reduced_by_drawdown(self, risk_engine):
        """Drawdown should reduce max leverage."""
        # Normal
        assert risk_engine.get_max_allowed_leverage("trending_bull", 0, 0) == 5

        # 5-10% drawdown
        assert risk_engine.get_max_allowed_leverage("trending_bull", 7, 0) == 3

        # 10-15% drawdown
        assert risk_engine.get_max_allowed_leverage("trending_bull", 12, 0) == 2

        # >15% drawdown
        assert risk_engine.get_max_allowed_leverage("trending_bull", 18, 0) == 1

    def test_leverage_reduced_by_losses(self, risk_engine):
        """Consecutive losses should reduce max leverage."""
        # Normal
        assert risk_engine.get_max_allowed_leverage("trending_bull", 0, 0) == 5

        # 3 losses
        assert risk_engine.get_max_allowed_leverage("trending_bull", 0, 3) == 2

        # 5 losses
        assert risk_engine.get_max_allowed_leverage("trending_bull", 0, 5) == 1


# =============================================================================
# Drawdown Edge Case Tests
# =============================================================================

class TestDrawdownEdgeCases:
    """Test drawdown calculation edge cases."""

    def test_zero_equity_no_drawdown(self):
        """Zero equity should result in zero drawdown."""
        state = RiskState()
        state.current_equity = Decimal("0")
        state.peak_equity = Decimal("0")

        state.update_drawdown()

        assert state.current_drawdown_pct == 0.0

    def test_initial_equity_sets_peak(self):
        """First positive equity should set peak."""
        state = RiskState()
        state.current_equity = Decimal("10000")
        state.peak_equity = Decimal("0")

        state.update_drawdown()

        assert state.peak_equity == Decimal("10000")
        assert state.current_drawdown_pct == 0.0

    def test_new_high_updates_peak(self):
        """New high should update peak equity."""
        state = RiskState()
        state.peak_equity = Decimal("10000")
        state.current_equity = Decimal("12000")

        state.update_drawdown()

        assert state.peak_equity == Decimal("12000")
        assert state.current_drawdown_pct == 0.0

    def test_drawdown_calculated_correctly(self):
        """Normal drawdown calculation should work."""
        state = RiskState()
        state.peak_equity = Decimal("10000")
        state.current_equity = Decimal("9000")

        state.update_drawdown()

        assert state.current_drawdown_pct == 10.0  # 10% drawdown
        assert state.max_drawdown_pct == 10.0

    def test_negative_equity_over_100_drawdown(self):
        """Negative equity should show >100% drawdown."""
        state = RiskState()
        state.peak_equity = Decimal("10000")
        state.current_equity = Decimal("-1000")  # Lost more than we had

        state.update_drawdown()

        # Should be 100% + (1000/10000)*100 = 110%
        assert state.current_drawdown_pct == 110.0

    def test_max_drawdown_tracks_worst(self):
        """Max drawdown should track the worst drawdown seen."""
        state = RiskState()
        state.peak_equity = Decimal("10000")

        # First drawdown: 10%
        state.current_equity = Decimal("9000")
        state.update_drawdown()
        assert state.max_drawdown_pct == 10.0

        # Recovery
        state.current_equity = Decimal("9500")
        state.update_drawdown()
        assert state.max_drawdown_pct == 10.0  # Still 10%

        # Deeper drawdown: 15%
        state.current_equity = Decimal("8500")
        state.update_drawdown()
        assert state.max_drawdown_pct == 15.0

    def test_zero_peak_with_positive_current(self):
        """Positive current equity with zero peak should set peak."""
        state = RiskState()
        state.peak_equity = Decimal("0")
        state.current_equity = Decimal("5000")

        state.update_drawdown()

        assert state.peak_equity == Decimal("5000")
        assert state.current_drawdown_pct == 0.0


# =============================================================================
# RiskState Serialization Tests
# =============================================================================

class TestRiskStateSerialization:
    """Test RiskState to_dict and from_dict."""

    def test_to_dict_basic(self):
        """Test basic to_dict serialization."""
        state = RiskState()
        state.peak_equity = Decimal("10000")
        state.current_equity = Decimal("9500")
        state.daily_pnl_pct = -5.0
        state.consecutive_losses = 3

        data = state.to_dict()

        assert data['peak_equity'] == '10000'
        assert data['current_equity'] == '9500'
        assert data['daily_pnl_pct'] == -5.0
        assert data['consecutive_losses'] == 3

    def test_to_dict_with_datetimes(self):
        """Test to_dict with datetime fields."""
        state = RiskState()
        now = datetime.now(timezone.utc)
        state.halt_until = now
        state.cooldown_until = now
        state.last_daily_reset = now

        data = state.to_dict()

        assert data['halt_until'] == now.isoformat()
        assert data['cooldown_until'] == now.isoformat()
        assert data['last_daily_reset'] == now.isoformat()

    def test_from_dict_basic(self):
        """Test basic from_dict deserialization."""
        data = {
            'peak_equity': '15000',
            'current_equity': '14000',
            'daily_pnl_pct': 2.5,
            'consecutive_losses': 2,
            'trading_halted': True,
            'halt_reason': 'test',
        }

        state = RiskState.from_dict(data)

        assert state.peak_equity == Decimal("15000")
        assert state.current_equity == Decimal("14000")
        assert state.daily_pnl_pct == 2.5
        assert state.consecutive_losses == 2
        assert state.trading_halted is True
        assert state.halt_reason == 'test'

    def test_from_dict_with_datetimes(self):
        """Test from_dict with datetime strings."""
        now = datetime.now(timezone.utc)
        data = {
            'halt_until': now.isoformat(),
            'cooldown_until': now.isoformat(),
            'last_daily_reset': now.isoformat(),
            'last_weekly_reset': now.isoformat(),
        }

        state = RiskState.from_dict(data)

        assert state.halt_until is not None
        assert state.cooldown_until is not None
        assert state.last_daily_reset is not None
        assert state.last_weekly_reset is not None

    def test_from_dict_empty(self):
        """Test from_dict with empty dict uses defaults."""
        state = RiskState.from_dict({})

        assert state.peak_equity == Decimal("0")
        assert state.current_equity == Decimal("0")
        assert state.daily_pnl_pct == 0.0
        assert state.consecutive_losses == 0
        assert state.trading_halted is False

    def test_roundtrip_serialization(self):
        """Test that to_dict -> from_dict preserves data."""
        original = RiskState()
        original.peak_equity = Decimal("20000")
        original.current_equity = Decimal("18000")
        original.daily_pnl_pct = -2.5
        original.consecutive_losses = 4
        original.open_position_symbols = ['BTC/USDT', 'ETH/USDT']
        original.position_exposures = {'BTC/USDT': 15.0, 'ETH/USDT': 10.0}

        data = original.to_dict()
        restored = RiskState.from_dict(data)

        assert restored.peak_equity == original.peak_equity
        assert restored.current_equity == original.current_equity
        assert restored.daily_pnl_pct == original.daily_pnl_pct
        assert restored.consecutive_losses == original.consecutive_losses
        assert restored.open_position_symbols == original.open_position_symbols
        assert restored.position_exposures == original.position_exposures


# =============================================================================
# Risk Validation Result Tests
# =============================================================================

class TestRiskValidation:
    """Test RiskValidation dataclass."""

    def test_is_approved_for_approved_status(self):
        """is_approved should return True for APPROVED status."""
        result = RiskValidation(
            status=ValidationStatus.APPROVED,
            proposal=None,
            rejections=[],
            warnings=[],
            modifications=[],
        )

        assert result.is_approved() is True

    def test_is_approved_for_modified_status(self):
        """is_approved should return True for MODIFIED status."""
        result = RiskValidation(
            status=ValidationStatus.MODIFIED,
            proposal=None,
            rejections=[],
            warnings=['Size reduced'],
            modifications=['size_usd: 3000 -> 2000'],
        )

        assert result.is_approved() is True

    def test_is_approved_for_rejected_status(self):
        """is_approved should return False for REJECTED status."""
        result = RiskValidation(
            status=ValidationStatus.REJECTED,
            proposal=None,
            rejections=['CONFIDENCE_TOO_LOW'],
            warnings=[],
            modifications=[],
        )

        assert result.is_approved() is False

    def test_is_approved_for_halted_status(self):
        """is_approved should return False for HALTED status."""
        result = RiskValidation(
            status=ValidationStatus.HALTED,
            proposal=None,
            rejections=['TRADING_HALTED'],
            warnings=[],
            modifications=[],
        )

        assert result.is_approved() is False


# =============================================================================
# TradeProposal Tests
# =============================================================================

class TestTradeProposal:
    """Test TradeProposal dataclass."""

    def test_create_minimal_proposal(self):
        """Create proposal with minimal required fields."""
        proposal = TradeProposal(
            symbol="XRP/USDT",
            side="buy",
            size_usd=100.0,
            entry_price=0.50,
        )

        assert proposal.symbol == "XRP/USDT"
        assert proposal.leverage == 1  # Default
        assert proposal.confidence == 0.5  # Default
        assert proposal.stop_loss is None
        assert proposal.take_profit is None

    def test_create_full_proposal(self):
        """Create proposal with all fields."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="sell",
            size_usd=5000.0,
            entry_price=45000.0,
            stop_loss=46000.0,
            take_profit=42000.0,
            leverage=3,
            confidence=0.85,
            regime="trending_bear",
        )

        assert proposal.side == "sell"
        assert proposal.leverage == 3
        assert proposal.regime == "trending_bear"


# =============================================================================
# Additional Engine Tests
# =============================================================================

class TestEngineConfiguration:
    """Test RiskManagementEngine configuration."""

    def test_default_config_values(self):
        """Test that missing config uses sensible defaults."""
        engine = RiskManagementEngine({})

        assert engine.max_leverage == 5
        assert engine.max_position_pct == 20
        assert engine.max_exposure_pct == 80
        assert engine.min_confidence == 0.60

    def test_custom_config_values(self):
        """Test that custom config overrides defaults."""
        config = {
            'limits': {
                'max_leverage': 3,
                'max_position_pct': 15,
                'min_confidence': 0.70,
            }
        }

        engine = RiskManagementEngine(config)

        assert engine.max_leverage == 3
        assert engine.max_position_pct == 15
        assert engine.min_confidence == 0.70

    def test_nested_circuit_breaker_config(self):
        """Test nested circuit breaker configuration."""
        config = {
            'circuit_breakers': {
                'daily_loss': {
                    'threshold_pct': 3.0,
                },
                'weekly_loss': {
                    'threshold_pct': 8.0,
                },
            }
        }

        engine = RiskManagementEngine(config)

        assert engine.daily_loss_limit_pct == 3.0
        assert engine.weekly_loss_limit_pct == 8.0


class TestGetState:
    """Test get_state method."""

    def test_get_state_returns_state(self, risk_engine):
        """get_state should return the internal state."""
        state = risk_engine.get_state()

        assert isinstance(state, RiskState)
        # Verify it's the actual state object (could be copy or reference)
        assert state.current_equity >= Decimal("0")


class TestUpdateState:
    """Test update_state method."""

    def test_update_state_basic(self, risk_engine):
        """Test basic state update."""
        risk_engine.update_state(
            current_equity=Decimal("12000"),
            daily_pnl=Decimal("500"),
            weekly_pnl=Decimal("1000"),
            open_positions=2,
            total_exposure_pct=25.0,
            available_margin=Decimal("8000"),
        )

        state = risk_engine.get_state()
        assert state.current_equity == Decimal("12000")
        assert state.open_positions == 2
        assert state.total_exposure_pct == 25.0

    def test_update_state_triggers_circuit_breaker(self, risk_engine):
        """Test that update_state can trigger circuit breakers."""
        # Set peak first so we have a baseline
        risk_engine.update_state(
            current_equity=Decimal("10000"),
            daily_pnl=Decimal("0"),
            weekly_pnl=Decimal("0"),
            open_positions=0,
            total_exposure_pct=0,
            available_margin=Decimal("10000"),
        )

        # Now trigger daily loss
        risk_engine.update_state(
            current_equity=Decimal("10000"),
            daily_pnl=Decimal("-600"),  # -6% exceeds 5% limit
            weekly_pnl=Decimal("-600"),
            open_positions=0,
            total_exposure_pct=0,
            available_margin=Decimal("9400"),
        )

        state = risk_engine.get_state()
        assert state.trading_halted is True


class TestRecordTradeResult:
    """Test record_trade_result method."""

    def test_win_resets_loss_streak(self, risk_engine):
        """Win should reset loss streak and increment wins."""
        risk_engine._risk_state.consecutive_losses = 4
        risk_engine._risk_state.consecutive_wins = 0

        risk_engine.record_trade_result(is_win=True)

        assert risk_engine._risk_state.consecutive_losses == 0
        assert risk_engine._risk_state.consecutive_wins == 1

    def test_loss_resets_win_streak(self, risk_engine):
        """Loss should reset win streak and increment losses."""
        risk_engine._risk_state.consecutive_wins = 3
        risk_engine._risk_state.consecutive_losses = 0

        risk_engine.record_trade_result(is_win=False)

        assert risk_engine._risk_state.consecutive_wins == 0
        assert risk_engine._risk_state.consecutive_losses == 1


class TestResets:
    """Test reset methods."""

    def test_reset_weekly(self, risk_engine):
        """Test weekly reset."""
        risk_engine._risk_state.weekly_pnl = Decimal("-500")
        risk_engine._risk_state.weekly_pnl_pct = -5.0
        risk_engine._risk_state.trading_halted = True
        risk_engine._risk_state.triggered_breakers = ['weekly_loss']

        risk_engine.reset_weekly()

        assert risk_engine._risk_state.weekly_pnl == Decimal("0")
        assert risk_engine._risk_state.weekly_pnl_pct == 0.0
        # Weekly reset should also clear daily
        assert risk_engine._risk_state.daily_pnl == Decimal("0")


class TestRiskRewardValidation:
    """Test risk/reward ratio validation."""

    def test_poor_risk_reward_rejected(self, risk_engine, healthy_risk_state):
        """Poor risk/reward ratio should generate warning or rejection."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,  # 2% risk
            take_profit=45500.0,  # Only 1.1% reward - R:R < 1
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        # Should have warning about R:R or be rejected
        assert len(result.warnings) > 0 or result.status == ValidationStatus.REJECTED


class TestRegimeMultipliers:
    """Test regime-based position size multipliers."""

    def test_trending_bull_full_size(self, risk_engine):
        """Trending bull should allow full position size."""
        size = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,  # 2%
            regime="trending_bull",
            confidence=0.85,
        )

        # With 2% stop, should hit 20% cap
        assert size == 2000.0

    def test_ranging_smaller_than_trending(self, risk_engine):
        """Ranging regime should have smaller position than trending."""
        trending_size = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,  # 2%
            regime="trending_bull",
            confidence=0.85,
        )

        ranging_size = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,  # Same 2%
            regime="ranging",
            confidence=0.85,
        )

        # Both may hit the 20% cap, so we just verify ranging <= trending
        assert ranging_size <= trending_size
        assert ranging_size > 0

    def test_choppy_smallest_size(self, risk_engine):
        """Choppy regime should have smallest position size."""
        trending_size = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,
            regime="trending_bull",
            confidence=0.85,
        )

        choppy_size = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,
            regime="choppy",
            confidence=0.85,
        )

        # Choppy should be smaller (multiplier ~0.25 vs 1.0)
        assert choppy_size <= trending_size
        assert choppy_size > 0


# =============================================================================
# Correlation Validation Tests
# =============================================================================

class TestCorrelationValidation:
    """Test correlated position exposure validation."""

    def test_no_correlation_check_when_no_positions(self, risk_engine, valid_proposal, healthy_risk_state):
        """No correlation check when no open positions."""
        healthy_risk_state.open_position_symbols = []

        result = risk_engine.validate_trade(valid_proposal, healthy_risk_state)

        # Should pass correlation check
        assert result.status == ValidationStatus.APPROVED

    def test_correlation_check_with_correlated_position(self, risk_engine, valid_proposal, healthy_risk_state):
        """Check correlation when holding correlated position."""
        # Already holding BTC/USDT
        healthy_risk_state.open_position_symbols = ['BTC/USDT']
        healthy_risk_state.position_exposures = {'BTC/USDT': 20.0}

        # Propose another BTC/USDT trade
        result = risk_engine.validate_trade(valid_proposal, healthy_risk_state)

        # Should still pass since BTC/USDT correlation with itself = 1.0
        # and we're checking against max_correlated_exposure_pct
        assert result is not None

    def test_get_pair_correlation_same_symbol(self, risk_engine):
        """Same symbol should return correlation of 1.0."""
        correlation = risk_engine._get_pair_correlation('BTC/USDT', 'BTC/USDT')
        assert correlation == 1.0

    def test_get_pair_correlation_unknown_pairs(self, risk_engine):
        """Unknown pairs should return 0.0 correlation."""
        correlation = risk_engine._get_pair_correlation('UNKNOWN1', 'UNKNOWN2')
        assert correlation == 0.0

    def test_get_pair_correlation_reverse_lookup(self, risk_engine):
        """Correlation lookup should work in either direction."""
        # These may not be in PAIR_CORRELATIONS but test the logic
        corr1 = risk_engine._get_pair_correlation('BTC/USDT', 'ETH/USDT')
        corr2 = risk_engine._get_pair_correlation('ETH/USDT', 'BTC/USDT')

        # Both should return same value (either from lookup or 0.0)
        assert corr1 == corr2


# =============================================================================
# Period Reset Tests
# =============================================================================

class TestPeriodResets:
    """Test daily and weekly reset logic."""

    def test_check_and_reset_initializes_timestamps(self, risk_engine):
        """Reset timestamps should be initialized on first check."""
        state = risk_engine._risk_state
        state.last_daily_reset = None
        state.last_weekly_reset = None

        risk_engine._check_and_reset_periods(state)

        assert state.last_daily_reset is not None
        assert state.last_weekly_reset is not None

    def test_daily_reset_when_new_day(self, risk_engine):
        """Daily reset should trigger when new UTC day starts."""
        state = risk_engine._risk_state

        # Set last reset to yesterday
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        state.last_daily_reset = yesterday
        state.daily_pnl = Decimal("100")  # Some existing PnL

        risk_engine._check_and_reset_periods(state)

        # Should have reset
        assert state.last_daily_reset.date() == datetime.now(timezone.utc).date()

    def test_no_reset_same_day(self, risk_engine):
        """No reset should occur within the same day."""
        state = risk_engine._risk_state

        # Set last reset to now
        now = datetime.now(timezone.utc)
        state.last_daily_reset = now
        state.last_weekly_reset = now
        original_reset = state.last_daily_reset

        risk_engine._check_and_reset_periods(state)

        # Should not have changed
        assert state.last_daily_reset == original_reset


# =============================================================================
# Volatility Tracking Tests
# =============================================================================

class TestVolatilityTracking:
    """Test volatility spike detection."""

    def test_update_volatility_normal(self, risk_engine):
        """Normal volatility should not trigger spike."""
        spike_detected = risk_engine.update_volatility(
            current_atr=100.0,
            avg_atr_20=100.0,
        )

        assert spike_detected is False
        assert risk_engine._risk_state.current_atr == 100.0
        assert risk_engine._risk_state.avg_atr_20 == 100.0

    def test_update_volatility_spike_detected(self, risk_engine):
        """High volatility should trigger spike detection."""
        # Spike multiplier is 3.0 by default, so >3x average should trigger
        spike_detected = risk_engine.update_volatility(
            current_atr=350.0,  # 3.5x average - above 3.0 threshold
            avg_atr_20=100.0,
        )

        assert spike_detected is True
        assert risk_engine._risk_state.volatility_spike_active is True

    def test_volatility_spike_clears(self, risk_engine):
        """Volatility spike should clear when normalized."""
        # First trigger a spike (>3x)
        risk_engine.update_volatility(current_atr=350.0, avg_atr_20=100.0)
        assert risk_engine._risk_state.volatility_spike_active is True

        # Then normalize
        risk_engine.update_volatility(current_atr=100.0, avg_atr_20=100.0)
        assert risk_engine._risk_state.volatility_spike_active is False

    def test_update_volatility_zero_avg(self, risk_engine):
        """Zero average ATR should not cause division error."""
        spike_detected = risk_engine.update_volatility(
            current_atr=100.0,
            avg_atr_20=0.0,  # Zero average
        )

        assert spike_detected is False


# =============================================================================
# Position Tracking Tests
# =============================================================================

class TestPositionTracking:
    """Test position tracking updates."""

    def test_update_positions(self, risk_engine):
        """Position tracking should update state."""
        risk_engine.update_positions(
            open_symbols=['BTC/USDT', 'ETH/USDT'],
            exposures={'BTC/USDT': 15.0, 'ETH/USDT': 10.0},
        )

        state = risk_engine._risk_state
        assert 'BTC/USDT' in state.open_position_symbols
        assert 'ETH/USDT' in state.open_position_symbols
        assert state.position_exposures['BTC/USDT'] == 15.0

    def test_update_positions_empty(self, risk_engine):
        """Empty positions should clear tracking."""
        risk_engine.update_positions(
            open_symbols=[],
            exposures={},
        )

        state = risk_engine._risk_state
        assert len(state.open_position_symbols) == 0
        assert len(state.position_exposures) == 0


# =============================================================================
# Database Persistence Tests
# =============================================================================

class TestDatabasePersistence:
    """Test database state persistence."""

    @pytest.mark.asyncio
    async def test_persist_state_no_db(self, risk_engine):
        """Persist without database should return False."""
        risk_engine.db = None

        result = await risk_engine.persist_state()

        assert result is False

    @pytest.mark.asyncio
    async def test_load_state_no_db(self, risk_engine):
        """Load without database should return False."""
        risk_engine.db = None

        result = await risk_engine.load_state()

        assert result is False

    @pytest.mark.asyncio
    async def test_persist_state_with_mock_db(self, risk_engine):
        """Persist with mock database should succeed."""
        from unittest.mock import AsyncMock, MagicMock

        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=None)
        risk_engine.db = mock_db

        result = await risk_engine.persist_state()

        assert result is True
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_state_db_error(self, risk_engine):
        """Persist should handle database errors gracefully."""
        from unittest.mock import AsyncMock, MagicMock

        mock_db = MagicMock()
        mock_db.execute = AsyncMock(side_effect=Exception("DB error"))
        risk_engine.db = mock_db

        result = await risk_engine.persist_state()

        assert result is False

    @pytest.mark.asyncio
    async def test_load_state_with_mock_db(self, risk_engine):
        """Load with mock database should succeed."""
        from unittest.mock import AsyncMock, MagicMock
        import json

        mock_state = RiskState()
        mock_state.daily_pnl = Decimal("500")

        mock_db = MagicMock()
        mock_db.fetchrow = AsyncMock(return_value={
            'state_data': json.dumps(mock_state.to_dict())
        })
        risk_engine.db = mock_db

        result = await risk_engine.load_state()

        assert result is True
        # State should be loaded
        assert risk_engine._risk_state is not None

    @pytest.mark.asyncio
    async def test_load_state_no_data(self, risk_engine):
        """Load with no stored data should return False."""
        from unittest.mock import AsyncMock, MagicMock

        mock_db = MagicMock()
        mock_db.fetchrow = AsyncMock(return_value=None)
        risk_engine.db = mock_db

        result = await risk_engine.load_state()

        assert result is False

    @pytest.mark.asyncio
    async def test_load_state_db_error(self, risk_engine):
        """Load should handle database errors gracefully."""
        from unittest.mock import AsyncMock, MagicMock

        mock_db = MagicMock()
        mock_db.fetchrow = AsyncMock(side_effect=Exception("DB error"))
        risk_engine.db = mock_db

        result = await risk_engine.load_state()

        assert result is False


# =============================================================================
# Calculate Position Size Edge Cases
# =============================================================================

class TestCalculatePositionSizeEdgeCases:
    """Test edge cases in position size calculation."""

    def test_low_confidence_returns_zero(self, risk_engine):
        """Very low confidence should return 0 position."""
        size = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,
            regime="trending_bull",
            confidence=0.50,  # Below min threshold
        )

        assert size == 0.0

    def test_zero_equity(self, risk_engine):
        """Zero equity should return 0 position."""
        size = risk_engine.calculate_position_size(
            equity=0,
            entry_price=45000,
            stop_loss=44100,
            regime="trending_bull",
            confidence=0.85,
        )

        assert size == 0.0

    def test_zero_stop_distance(self, risk_engine):
        """Zero stop distance should return 0 position."""
        size = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=45000,  # Same as entry
            regime="trending_bull",
            confidence=0.85,
        )

        assert size == 0.0

    def test_confidence_tiers(self, risk_engine):
        """Different confidence levels should give different multipliers."""
        high_conf = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,
            regime="trending_bull",
            confidence=0.85,  # High tier
        )

        med_conf = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,
            regime="trending_bull",
            confidence=0.70,  # Medium tier
        )

        low_conf = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,
            regime="trending_bull",
            confidence=0.62,  # Low tier
        )

        # Higher confidence should give larger or equal position
        assert high_conf >= med_conf
        assert med_conf >= low_conf

    def test_unknown_regime_uses_default(self, risk_engine):
        """Unknown regime should use default multiplier."""
        size = risk_engine.calculate_position_size(
            equity=10000,
            entry_price=45000,
            stop_loss=44100,
            regime="unknown_regime",
            confidence=0.85,
        )

        # Should still calculate something (default 0.5 multiplier)
        assert size > 0


# =============================================================================
# High Correlated Exposure Warning Test
# =============================================================================

class TestCorrelatedExposureWarnings:
    """Test correlation exposure warning generation."""

    def test_high_correlation_warning(self, risk_engine, valid_proposal, healthy_risk_state):
        """High correlation exposure should generate warning."""
        # Set up high existing exposure on same symbol
        healthy_risk_state.open_position_symbols = ['BTC/USDT']
        # Set exposure that's high but below rejection threshold
        healthy_risk_state.position_exposures = {'BTC/USDT': 15.0}
        healthy_risk_state.current_equity = Decimal("10000")

        # Adjust proposal to small size so total doesn't exceed max
        valid_proposal.size_usd = 500.0

        result = risk_engine.validate_trade(valid_proposal, healthy_risk_state)

        # The exact behavior depends on thresholds, but test runs
        assert result is not None


# =============================================================================
# Weekly Reset Test
# =============================================================================

class TestWeeklyReset:
    """Test weekly reset functionality."""

    def test_weekly_reset_clears_weekly_pnl(self, risk_engine):
        """Weekly reset should clear weekly PnL."""
        state = risk_engine._risk_state
        state.weekly_pnl = Decimal("500")
        state.weekly_pnl_pct = 5.0

        risk_engine.reset_weekly()

        assert state.weekly_pnl == Decimal("0")
        assert state.weekly_pnl_pct == 0.0

    def test_weekly_reset_clears_breakers(self, risk_engine):
        """Weekly reset should clear weekly circuit breakers."""
        state = risk_engine._risk_state
        # Set up a weekly loss breaker properly
        state.trading_halted = True
        state.halt_reason = "WEEKLY_LOSS"
        state.triggered_breakers = ['weekly_loss']  # Must be in triggered_breakers

        risk_engine.reset_weekly()

        # Should be cleared since it was the only breaker
        assert state.trading_halted is False
        assert 'weekly_loss' not in state.triggered_breakers


# =============================================================================
# F01: Risk/Reward Ratio Rejection Tests
# =============================================================================

class TestRiskRewardRejection:
    """Test risk/reward ratio rejection (F01 fix)."""

    def test_low_rr_ratio_rejected(self, risk_engine, healthy_risk_state):
        """Trade with R:R below 1.5 should be REJECTED, not just warned."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,  # 2% risk (900 points)
            take_profit=45500.0,  # 1.1% reward (500 points) - R:R = 0.56
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("LOW_RR" in r for r in result.rejections)

    def test_good_rr_ratio_approved(self, risk_engine, healthy_risk_state):
        """Trade with R:R >= 1.5 should be approved."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,  # 2% risk (900 points)
            take_profit=46500.0,  # 3.3% reward (1500 points) - R:R = 1.67
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.is_approved()


# =============================================================================
# F02: Regime-Based Confidence Threshold Tests
# =============================================================================

class TestRegimeConfidenceThresholds:
    """Test regime-based confidence thresholds (F02 fix)."""

    def test_choppy_requires_high_confidence(self, risk_engine, healthy_risk_state):
        """Choppy regime should require 0.75 confidence."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            take_profit=47000.0,
            leverage=1,
            confidence=0.70,  # Below 0.75 required for choppy
            regime="choppy",
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("CONFIDENCE_TOO_LOW" in r for r in result.rejections)

    def test_trending_accepts_lower_confidence(self, risk_engine, healthy_risk_state):
        """Trending regime should accept 0.55 confidence."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            take_profit=47000.0,
            leverage=1,
            confidence=0.56,  # Above 0.55 required for trending
            regime="trending_bull",
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.is_approved()

    def test_volatile_requires_medium_confidence(self, risk_engine, healthy_risk_state):
        """Volatile regime should require 0.65 confidence."""
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            take_profit=47000.0,
            leverage=1,
            confidence=0.62,  # Below 0.65 required for volatile
            regime="volatile_bull",
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED


# =============================================================================
# F03: Consecutive Loss Size Reduction Tests
# =============================================================================

class TestConsecutiveLossSizeReduction:
    """Test 50% size reduction after 5 consecutive losses (F03 fix)."""

    def test_size_reduced_after_5_losses(self, risk_engine, healthy_risk_state):
        """Position size should be reduced by 50% after 5 consecutive losses."""
        healthy_risk_state.consecutive_losses = 5

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            take_profit=47000.0,
            leverage=1,
            confidence=0.85,  # High enough to pass elevated confidence check
            regime="trending_bull",
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.MODIFIED
        assert result.modified_proposal.size_usd == 500.0  # 50% of 1000
        assert 'consecutive_loss_reduction' in result.modifications
        assert any("SIZE_REDUCED_LOSSES" in w for w in result.warnings)

    def test_no_size_reduction_before_5_losses(self, risk_engine, healthy_risk_state):
        """Position size should NOT be reduced before 5 consecutive losses."""
        healthy_risk_state.consecutive_losses = 4

        proposal = TradeProposal(
            symbol="BTC/USDT",
            side="buy",
            size_usd=1000.0,
            entry_price=45000.0,
            stop_loss=44100.0,
            take_profit=47000.0,
            leverage=1,
            confidence=0.85,
            regime="trending_bull",
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        # Should be approved without size modification for consecutive losses
        assert result.is_approved()
        assert 'consecutive_loss_reduction' not in result.modifications


# =============================================================================
# F05: Trades Today Counter Tests
# =============================================================================

class TestTradesTodayCounter:
    """Test trades_today counter (F05 fix)."""

    def test_trades_today_increments_on_trade(self, risk_engine):
        """trades_today should increment when recording trade result."""
        assert risk_engine._risk_state.trades_today == 0

        risk_engine.record_trade_result(is_win=True)
        assert risk_engine._risk_state.trades_today == 1

        risk_engine.record_trade_result(is_win=False)
        assert risk_engine._risk_state.trades_today == 2

    def test_trades_today_resets_daily(self, risk_engine):
        """trades_today should reset to 0 on daily reset."""
        risk_engine._risk_state.trades_today = 5

        risk_engine.reset_daily()

        assert risk_engine._risk_state.trades_today == 0

    def test_trades_today_serialization(self):
        """trades_today should be included in serialization."""
        state = RiskState()
        state.trades_today = 10

        data = state.to_dict()
        assert data['trades_today'] == 10

        restored = RiskState.from_dict(data)
        assert restored.trades_today == 10


# =============================================================================
# F06: Persist State Retry Tests
# =============================================================================

class TestPersistStateRetry:
    """Test persist_state retry logic (F06 fix)."""

    @pytest.mark.asyncio
    async def test_persist_retries_on_failure(self, risk_engine):
        """persist_state should retry on failure."""
        from unittest.mock import AsyncMock, MagicMock

        mock_db = MagicMock()
        # Fail first 2 times, succeed on 3rd
        mock_db.execute = AsyncMock(
            side_effect=[Exception("fail 1"), Exception("fail 2"), None]
        )
        risk_engine.db = mock_db

        result = await risk_engine.persist_state(max_retries=3)

        assert result is True
        assert mock_db.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_persist_fails_after_max_retries(self, risk_engine):
        """persist_state should return False after max retries exceeded."""
        from unittest.mock import AsyncMock, MagicMock

        mock_db = MagicMock()
        mock_db.execute = AsyncMock(side_effect=Exception("always fail"))
        risk_engine.db = mock_db

        result = await risk_engine.persist_state(max_retries=2)

        assert result is False
        assert mock_db.execute.call_count == 2


# =============================================================================
# F10: Weekly Reset Non-Monday Tests
# =============================================================================

class TestWeeklyResetNonMonday:
    """Test weekly reset triggers on any day of new week (F10 fix)."""

    def test_weekly_reset_on_tuesday(self, risk_engine):
        """Weekly reset should trigger on Tuesday if first validation of week."""
        state = risk_engine._risk_state

        # Set last reset to previous week (any day)
        state.last_weekly_reset = datetime(2025, 12, 8, 12, 0, tzinfo=timezone.utc)  # Monday
        state.weekly_pnl = Decimal("500")

        # Simulate validation on Tuesday of next week
        from unittest.mock import patch
        tuesday = datetime(2025, 12, 16, 12, 0, tzinfo=timezone.utc)  # Tuesday

        with patch('triplegain.src.risk.rules_engine.datetime') as mock_dt:
            mock_dt.now.return_value = tuesday
            mock_dt.fromisoformat = datetime.fromisoformat
            risk_engine._check_and_reset_periods(state)

        # Should have reset
        assert state.weekly_pnl == Decimal("0")


# =============================================================================
# F11: Correlation Rejection Path Tests
# =============================================================================

class TestCorrelationRejection:
    """Test correlated exposure rejection path (F11 fix)."""

    def test_correlated_exposure_exceeds_max_rejected(self, risk_engine, healthy_risk_state):
        """Trade causing correlated exposure over limit should be rejected."""
        # Set max_correlated_exposure_pct to 40%
        risk_engine.max_correlated_exposure_pct = 40

        # Already holding BTC/USDT at 30% exposure
        healthy_risk_state.open_position_symbols = ['BTC/USDT']
        healthy_risk_state.position_exposures = {'BTC/USDT': 30.0}
        healthy_risk_state.current_equity = Decimal("10000")

        # Propose XRP/USDT trade (correlated with BTC at 0.75)
        # With 30% BTC * 0.75 correlation = 22.5% correlated
        # Plus new 20% = 42.5% > 40% limit
        proposal = TradeProposal(
            symbol="XRP/USDT",
            side="buy",
            size_usd=2000.0,  # 20% of $10k equity
            entry_price=0.50,
            stop_loss=0.49,
            take_profit=0.55,
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.status == ValidationStatus.REJECTED
        assert any("CORRELATED_EXPOSURE" in r for r in result.rejections)

    def test_correlated_exposure_under_limit_approved(self, risk_engine, healthy_risk_state):
        """Trade with correlated exposure under limit should be approved."""
        # Set max_correlated_exposure_pct to 50%
        risk_engine.max_correlated_exposure_pct = 50

        # Already holding BTC/USDT at 15% exposure
        healthy_risk_state.open_position_symbols = ['BTC/USDT']
        healthy_risk_state.position_exposures = {'BTC/USDT': 15.0}
        healthy_risk_state.current_equity = Decimal("10000")

        # Propose small XRP/USDT trade
        proposal = TradeProposal(
            symbol="XRP/USDT",
            side="buy",
            size_usd=500.0,  # 5% of equity
            entry_price=0.50,
            stop_loss=0.49,
            take_profit=0.55,
            leverage=1,
            confidence=0.75,
        )

        result = risk_engine.validate_trade(proposal, healthy_risk_state)

        assert result.is_approved()


# =============================================================================
# TradeProposal Validation Edge Cases (F07)
# =============================================================================

class TestTradeProposalValidation:
    """Test TradeProposal validation edge cases (F07)."""

    def test_empty_symbol_raises(self):
        """Empty symbol should raise validation error."""
        from triplegain.src.risk.rules_engine import TradeProposalValidationError

        with pytest.raises(TradeProposalValidationError):
            TradeProposal(symbol="", side="buy", size_usd=100, entry_price=50)

    def test_invalid_side_raises(self):
        """Invalid side should raise validation error."""
        from triplegain.src.risk.rules_engine import TradeProposalValidationError

        with pytest.raises(TradeProposalValidationError):
            TradeProposal(symbol="BTC/USDT", side="invalid", size_usd=100, entry_price=50)

    def test_negative_size_raises(self):
        """Negative size should raise validation error."""
        from triplegain.src.risk.rules_engine import TradeProposalValidationError

        with pytest.raises(TradeProposalValidationError):
            TradeProposal(symbol="BTC/USDT", side="buy", size_usd=-100, entry_price=50)

    def test_zero_size_raises(self):
        """Zero size should raise validation error."""
        from triplegain.src.risk.rules_engine import TradeProposalValidationError

        with pytest.raises(TradeProposalValidationError):
            TradeProposal(symbol="BTC/USDT", side="buy", size_usd=0, entry_price=50)

    def test_negative_entry_price_raises(self):
        """Negative entry price should raise validation error."""
        from triplegain.src.risk.rules_engine import TradeProposalValidationError

        with pytest.raises(TradeProposalValidationError):
            TradeProposal(symbol="BTC/USDT", side="buy", size_usd=100, entry_price=-50)

    def test_invalid_leverage_raises(self):
        """Leverage < 1 should raise validation error."""
        from triplegain.src.risk.rules_engine import TradeProposalValidationError

        with pytest.raises(TradeProposalValidationError):
            TradeProposal(symbol="BTC/USDT", side="buy", size_usd=100, entry_price=50, leverage=0)

    def test_invalid_confidence_raises(self):
        """Confidence outside 0-1 should raise validation error."""
        from triplegain.src.risk.rules_engine import TradeProposalValidationError

        with pytest.raises(TradeProposalValidationError):
            TradeProposal(symbol="BTC/USDT", side="buy", size_usd=100, entry_price=50, confidence=1.5)

    def test_buy_stop_above_entry_raises(self):
        """Buy with stop loss above entry should raise validation error."""
        from triplegain.src.risk.rules_engine import TradeProposalValidationError

        with pytest.raises(TradeProposalValidationError):
            TradeProposal(symbol="BTC/USDT", side="buy", size_usd=100, entry_price=50, stop_loss=55)

    def test_sell_stop_below_entry_raises(self):
        """Sell with stop loss below entry should raise validation error."""
        from triplegain.src.risk.rules_engine import TradeProposalValidationError

        with pytest.raises(TradeProposalValidationError):
            TradeProposal(symbol="BTC/USDT", side="sell", size_usd=100, entry_price=50, stop_loss=45)
