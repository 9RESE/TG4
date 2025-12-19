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
