"""
Risk Management Rules Engine - Deterministic risk validation.

NO LLM DEPENDENCY - Purely rule-based for:
- Sub-10ms execution
- Predictable behavior
- No network latency

Implements all risk layers from the design:
1. Pre-trade validation
2. Portfolio risk checks
3. Drawdown circuit breakers
4. Position management
5. System safeguards
6. Volatility spike detection
7. Correlated position checking
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Any

logger = logging.getLogger(__name__)


# Correlation matrix for trading pairs (BTC/XRP correlation)
PAIR_CORRELATIONS = {
    ('BTC/USDT', 'XRP/USDT'): 0.75,
    ('BTC/USDT', 'XRP/BTC'): 0.60,
    ('XRP/USDT', 'XRP/BTC'): 0.85,
}


class ValidationStatus(Enum):
    """Trade validation result status."""
    APPROVED = "approved"
    MODIFIED = "modified"  # Approved with adjustments
    REJECTED = "rejected"
    HALTED = "halted"  # Trading is halted


class TradeProposalValidationError(ValueError):
    """Raised when TradeProposal validation fails."""
    pass


@dataclass
class TradeProposal:
    """
    Trade proposal to be validated by the risk engine.

    All trades must pass risk validation before execution.
    """
    symbol: str
    side: str  # "buy" or "sell"
    size_usd: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: int = 1
    confidence: float = 0.5

    # Context for validation
    regime: str = "ranging"
    agent_confidences: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate inputs after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate trade proposal inputs.

        Raises:
            TradeProposalValidationError: If any validation fails
        """
        errors = []

        # Symbol validation
        if not self.symbol or not isinstance(self.symbol, str):
            errors.append("Symbol must be a non-empty string")

        # Side validation
        if self.side not in ("buy", "sell"):
            errors.append(f"Side must be 'buy' or 'sell', got '{self.side}'")

        # Size validation - CRITICAL: Must be positive
        if not isinstance(self.size_usd, (int, float)) or self.size_usd <= 0:
            errors.append(f"Size must be positive, got {self.size_usd}")

        # Entry price validation - CRITICAL: Must be positive
        if not isinstance(self.entry_price, (int, float)) or self.entry_price <= 0:
            errors.append(f"Entry price must be positive, got {self.entry_price}")

        # Stop loss validation
        if self.stop_loss is not None:
            if not isinstance(self.stop_loss, (int, float)) or self.stop_loss <= 0:
                errors.append(f"Stop loss must be positive, got {self.stop_loss}")
            elif self.side == "buy" and self.stop_loss >= self.entry_price:
                errors.append(f"Stop loss ({self.stop_loss}) must be below entry ({self.entry_price}) for buy")
            elif self.side == "sell" and self.stop_loss <= self.entry_price:
                errors.append(f"Stop loss ({self.stop_loss}) must be above entry ({self.entry_price}) for sell")

        # Take profit validation
        if self.take_profit is not None:
            if not isinstance(self.take_profit, (int, float)) or self.take_profit <= 0:
                errors.append(f"Take profit must be positive, got {self.take_profit}")
            elif self.side == "buy" and self.take_profit <= self.entry_price:
                errors.append(f"Take profit ({self.take_profit}) must be above entry ({self.entry_price}) for buy")
            elif self.side == "sell" and self.take_profit >= self.entry_price:
                errors.append(f"Take profit ({self.take_profit}) must be below entry ({self.entry_price}) for sell")

        # Leverage validation - CRITICAL: Must be positive integer
        if not isinstance(self.leverage, int) or self.leverage < 1:
            errors.append(f"Leverage must be positive integer >= 1, got {self.leverage}")

        # Confidence validation
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= self.confidence <= 1.0):
            errors.append(f"Confidence must be between 0 and 1, got {self.confidence}")

        if errors:
            error_msg = "Trade proposal validation failed: " + "; ".join(errors)
            logger.warning(error_msg)
            raise TradeProposalValidationError(error_msg)

    def calculate_risk_reward(self) -> Optional[float]:
        """Calculate risk/reward ratio."""
        if not self.stop_loss or not self.take_profit:
            return None

        if self.side == "buy":
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit

        if risk <= 0:
            return None

        return reward / risk


@dataclass
class RiskValidation:
    """Result of risk validation."""
    status: ValidationStatus
    proposal: TradeProposal
    modified_proposal: Optional[TradeProposal] = None

    # Detailed results
    rejections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    modifications: dict = field(default_factory=dict)

    # Risk metrics
    risk_per_trade_pct: float = 0.0
    portfolio_exposure_pct: float = 0.0
    margin_utilization_pct: float = 0.0

    # Validation timing
    validation_time_ms: int = 0

    def is_approved(self) -> bool:
        """Check if trade is approved (possibly with modifications)."""
        return self.status in [ValidationStatus.APPROVED, ValidationStatus.MODIFIED]


@dataclass
class RiskState:
    """Current risk state for the portfolio."""
    # Equity tracking
    peak_equity: Decimal = Decimal("0")
    current_equity: Decimal = Decimal("0")
    available_margin: Decimal = Decimal("0")

    # P&L tracking
    daily_pnl: Decimal = Decimal("0")
    daily_pnl_pct: float = 0.0
    weekly_pnl: Decimal = Decimal("0")
    weekly_pnl_pct: float = 0.0

    # Drawdown tracking
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    # Trade tracking
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    open_positions: int = 0
    total_exposure_pct: float = 0.0

    # Circuit breaker state
    trading_halted: bool = False
    halt_reason: str = ""
    halt_until: Optional[datetime] = None
    triggered_breakers: list[str] = field(default_factory=list)

    # Cooldown state
    in_cooldown: bool = False
    cooldown_until: Optional[datetime] = None
    cooldown_reason: str = ""

    # Volatility tracking
    current_atr: float = 0.0
    avg_atr_20: float = 0.0
    volatility_spike_active: bool = False

    # Position tracking for correlation
    open_position_symbols: list[str] = field(default_factory=list)
    position_exposures: dict = field(default_factory=dict)  # symbol -> exposure_pct

    # Timestamp tracking for daily/weekly reset
    last_daily_reset: Optional[datetime] = None
    last_weekly_reset: Optional[datetime] = None

    def update_drawdown(self):
        """
        Update drawdown calculations.

        Handles edge cases:
        - Zero peak equity: No calculation (initial state)
        - Negative equity: Clamps to 100% drawdown
        - Current > peak: Updates peak (no drawdown)
        """
        # Guard against invalid equity states
        if self.current_equity <= 0 and self.peak_equity <= 0:
            # Initial state or invalid - reset drawdown
            self.current_drawdown_pct = 0.0
            return

        # Update peak if current is higher
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            self.current_drawdown_pct = 0.0
            return

        # Calculate drawdown only if we have valid peak equity
        if self.peak_equity > 0:
            if self.current_equity < 0:
                # Edge case: negative equity means 100%+ drawdown
                self.current_drawdown_pct = 100.0 + float(
                    abs(self.current_equity) / self.peak_equity * 100
                )
            else:
                self.current_drawdown_pct = float(
                    (self.peak_equity - self.current_equity) / self.peak_equity * 100
                )
            self.max_drawdown_pct = max(self.max_drawdown_pct, self.current_drawdown_pct)

    def to_dict(self) -> dict:
        """Serialize RiskState to dictionary for persistence."""
        return {
            'peak_equity': str(self.peak_equity),
            'current_equity': str(self.current_equity),
            'available_margin': str(self.available_margin),
            'daily_pnl': str(self.daily_pnl),
            'daily_pnl_pct': self.daily_pnl_pct,
            'weekly_pnl': str(self.weekly_pnl),
            'weekly_pnl_pct': self.weekly_pnl_pct,
            'current_drawdown_pct': self.current_drawdown_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'open_positions': self.open_positions,
            'total_exposure_pct': self.total_exposure_pct,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'halt_until': self.halt_until.isoformat() if self.halt_until else None,
            'triggered_breakers': self.triggered_breakers,
            'in_cooldown': self.in_cooldown,
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
            'cooldown_reason': self.cooldown_reason,
            'current_atr': self.current_atr,
            'avg_atr_20': self.avg_atr_20,
            'volatility_spike_active': self.volatility_spike_active,
            'open_position_symbols': self.open_position_symbols,
            'position_exposures': self.position_exposures,
            'last_daily_reset': self.last_daily_reset.isoformat() if self.last_daily_reset else None,
            'last_weekly_reset': self.last_weekly_reset.isoformat() if self.last_weekly_reset else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RiskState':
        """Deserialize RiskState from dictionary."""
        state = cls()
        state.peak_equity = Decimal(data.get('peak_equity', '0'))
        state.current_equity = Decimal(data.get('current_equity', '0'))
        state.available_margin = Decimal(data.get('available_margin', '0'))
        state.daily_pnl = Decimal(data.get('daily_pnl', '0'))
        state.daily_pnl_pct = data.get('daily_pnl_pct', 0.0)
        state.weekly_pnl = Decimal(data.get('weekly_pnl', '0'))
        state.weekly_pnl_pct = data.get('weekly_pnl_pct', 0.0)
        state.current_drawdown_pct = data.get('current_drawdown_pct', 0.0)
        state.max_drawdown_pct = data.get('max_drawdown_pct', 0.0)
        state.consecutive_losses = data.get('consecutive_losses', 0)
        state.consecutive_wins = data.get('consecutive_wins', 0)
        state.open_positions = data.get('open_positions', 0)
        state.total_exposure_pct = data.get('total_exposure_pct', 0.0)
        state.trading_halted = data.get('trading_halted', False)
        state.halt_reason = data.get('halt_reason', '')
        state.halt_until = (
            datetime.fromisoformat(data['halt_until'])
            if data.get('halt_until') else None
        )
        state.triggered_breakers = data.get('triggered_breakers', [])
        state.in_cooldown = data.get('in_cooldown', False)
        state.cooldown_until = (
            datetime.fromisoformat(data['cooldown_until'])
            if data.get('cooldown_until') else None
        )
        state.cooldown_reason = data.get('cooldown_reason', '')
        state.current_atr = data.get('current_atr', 0.0)
        state.avg_atr_20 = data.get('avg_atr_20', 0.0)
        state.volatility_spike_active = data.get('volatility_spike_active', False)
        state.open_position_symbols = data.get('open_position_symbols', [])
        state.position_exposures = data.get('position_exposures', {})
        state.last_daily_reset = (
            datetime.fromisoformat(data['last_daily_reset'])
            if data.get('last_daily_reset') else None
        )
        state.last_weekly_reset = (
            datetime.fromisoformat(data['last_weekly_reset'])
            if data.get('last_weekly_reset') else None
        )
        return state


class RiskManagementEngine:
    """
    Deterministic risk management engine.

    All validations are rule-based with <10ms target latency.
    NO LLM calls - predictable, auditable decisions.
    """

    def __init__(self, config: dict, db_pool=None):
        """
        Initialize RiskManagementEngine.

        Args:
            config: Risk configuration from risk.yaml
            db_pool: Optional database pool for state persistence
        """
        self.config = config
        self.db = db_pool

        # Load limits from config
        limits = config.get('limits', {})
        self.max_leverage = limits.get('max_leverage', 5)
        self.max_position_pct = limits.get('max_position_pct', 20)
        self.max_exposure_pct = limits.get('max_total_exposure_pct', 80)
        self.max_risk_per_trade_pct = limits.get('max_risk_per_trade_pct', 2)
        self.min_confidence = limits.get('min_confidence', 0.60)
        self.max_correlated_exposure_pct = limits.get('max_correlated_exposure_pct', 40)

        # Stop-loss requirements
        stops = config.get('stop_loss', {})
        self.require_stop_loss = stops.get('required', True)
        self.min_stop_pct = stops.get('min_distance_pct', 0.5)
        self.max_stop_pct = stops.get('max_distance_pct', 5.0)
        self.min_risk_reward = stops.get('min_risk_reward', 1.5)

        # Circuit breaker thresholds - handle both flat and nested config
        breakers = config.get('circuit_breakers', {})
        # Support nested structure from yaml
        daily_loss = breakers.get('daily_loss', {})
        weekly_loss = breakers.get('weekly_loss', {})
        max_drawdown = breakers.get('max_drawdown', {})
        cons_losses = breakers.get('consecutive_losses', {})

        self.daily_loss_limit_pct = (
            daily_loss.get('threshold_pct', 5.0)
            if isinstance(daily_loss, dict) else breakers.get('daily_loss_pct', 5.0)
        )
        self.weekly_loss_limit_pct = (
            weekly_loss.get('threshold_pct', 10.0)
            if isinstance(weekly_loss, dict) else breakers.get('weekly_loss_pct', 10.0)
        )
        self.max_drawdown_limit_pct = (
            max_drawdown.get('threshold_pct', 20.0)
            if isinstance(max_drawdown, dict) else breakers.get('max_drawdown_pct', 20.0)
        )
        self.consecutive_loss_threshold = (
            cons_losses.get('threshold_count', 5)
            if isinstance(cons_losses, dict) else breakers.get('consecutive_losses', 5)
        )

        # Volatility spike threshold (ATR > 3x average)
        self.volatility_spike_multiplier = config.get('volatility_spike_multiplier', 3.0)
        self.volatility_size_reduction_pct = config.get('volatility_size_reduction_pct', 50)

        # Cooldown settings
        cooldowns = config.get('cooldowns', {})
        self.post_trade_cooldown_min = cooldowns.get('post_trade_minutes', 5)
        self.post_loss_cooldown_min = cooldowns.get('post_loss_minutes', 10)
        self.consecutive_loss_cooldown_min = cooldowns.get('consecutive_loss_3_minutes', 30)
        self.volatility_spike_cooldown_min = cooldowns.get('volatility_spike_minutes', 15)

        # Leverage limits by regime - load from config or use defaults
        self.regime_leverage_limits = config.get('regime_leverage_limits', {
            'trending_bull': 5,
            'trending_bear': 3,
            'ranging': 2,
            'volatile_bull': 2,
            'volatile_bear': 2,
            'choppy': 1,
            'breakout_potential': 3,
            'unknown': 1,
        })

        # Confidence thresholds by consecutive losses - LOAD FROM CONFIG
        confidence_config = config.get('confidence', {})
        base_confidence = confidence_config.get('base_minimum', 0.60)
        after_3_losses = confidence_config.get('after_3_losses', 0.70)
        after_5_losses = confidence_config.get('after_5_losses', 0.80)

        self.confidence_thresholds = {
            0: base_confidence,
            3: after_3_losses,
            5: after_5_losses,
        }

        # Entry strictness adjustments (loaded from regime)
        self.entry_strictness_adjustments = {
            'relaxed': -0.05,    # Lower required confidence
            'normal': 0.0,       # No adjustment
            'strict': 0.05,      # Higher required confidence
            'very_strict': 0.10, # Much higher required confidence
        }

        # Track state with thread-safe lock
        self._risk_state = RiskState()
        self._state_lock = threading.Lock()  # Thread safety for concurrent validation

    def validate_trade(
        self,
        proposal: TradeProposal,
        risk_state: Optional[RiskState] = None,
        entry_strictness: str = "normal",
    ) -> RiskValidation:
        """
        Validate a trade proposal against all risk rules.

        Thread-safe: Uses internal lock to prevent race conditions.
        CRITICAL: Circuit breaker checks ALWAYS use internal state to prevent bypass.

        Args:
            proposal: Trade proposal to validate
            risk_state: Current risk state for non-circuit-breaker checks (uses internal if None)
            entry_strictness: Entry strictness from regime detection

        Returns:
            RiskValidation with approval/rejection decision
        """
        start_time = time.perf_counter()

        # Use lock for circuit breaker checks to ensure thread safety
        # CRITICAL: This prevents race condition where circuit breaker could be bypassed
        with self._state_lock:
            # Auto-reset daily/weekly if needed
            self._check_and_reset_periods(self._risk_state)

            # CRITICAL: Circuit breaker checks ALWAYS use internal state
            # This prevents bypass via stale external state
            internal_state = self._risk_state

            # Check circuit breakers first - use INTERNAL state (halt trading)
            if internal_state.trading_halted:
                result = RiskValidation(
                    status=ValidationStatus.HALTED,
                    proposal=proposal,
                    rejections=[f"TRADING_HALTED: {internal_state.halt_reason}"],
                    validation_time_ms=int((time.perf_counter() - start_time) * 1000),
                )
                return result

            # Check cooldown - use INTERNAL state
            if internal_state.in_cooldown and internal_state.cooldown_until:
                if datetime.now(timezone.utc) < internal_state.cooldown_until:
                    result = RiskValidation(
                        status=ValidationStatus.REJECTED,
                        proposal=proposal,
                        rejections=[f"IN_COOLDOWN: {internal_state.cooldown_reason}"],
                        validation_time_ms=int((time.perf_counter() - start_time) * 1000),
                    )
                    return result

            # Snapshot current state for remaining checks (outside lock is ok)
            # For other validations, allow external state override
            state = risk_state or self._risk_state

        result = RiskValidation(
            status=ValidationStatus.APPROVED,
            proposal=proposal,
        )

        # NOTE: Circuit breakers already checked above under lock
        # The following check is for secondary validation after other checks
        if state.trading_halted:
            result.status = ValidationStatus.HALTED
            result.rejections.append(f"TRADING_HALTED: {state.halt_reason}")
            result.validation_time_ms = int((time.perf_counter() - start_time) * 1000)
            return result

        # Check cooldown
        if state.in_cooldown and state.cooldown_until:
            if datetime.now(timezone.utc) < state.cooldown_until:
                result.status = ValidationStatus.REJECTED
                result.rejections.append(f"IN_COOLDOWN: {state.cooldown_reason}")
                result.validation_time_ms = int((time.perf_counter() - start_time) * 1000)
                return result

        # 1. Stop-loss validation
        if not self._validate_stop_loss(proposal, result):
            result.validation_time_ms = int((time.perf_counter() - start_time) * 1000)
            return result

        # 2. Confidence validation WITH entry strictness
        if not self._validate_confidence(proposal, state, result, entry_strictness):
            result.validation_time_ms = int((time.perf_counter() - start_time) * 1000)
            return result

        # 3. Position size validation (may modify)
        modified_size = self._validate_position_size(proposal, state, result)

        # 4. Volatility spike check - reduce size by 50% if active
        if state.volatility_spike_active:
            reduction = self.volatility_size_reduction_pct / 100
            original_size = modified_size or proposal.size_usd
            reduced_size = original_size * (1 - reduction)
            modified_size = reduced_size
            result.warnings.append(
                f"VOLATILITY_SPIKE: Size reduced by {self.volatility_size_reduction_pct}% "
                f"(${original_size:.2f} -> ${reduced_size:.2f})"
            )
            result.modifications['volatility_adjustment'] = {
                'original': original_size,
                'modified': reduced_size,
                'reason': 'Volatility spike detected (ATR > 3x average)',
            }

        # 5. Leverage validation (may modify)
        modified_leverage = self._validate_leverage(proposal, state, result)

        # 6. Portfolio exposure validation
        if not self._validate_exposure(proposal, state, result):
            result.validation_time_ms = int((time.perf_counter() - start_time) * 1000)
            return result

        # 7. Correlated position check
        if not self._validate_correlation(proposal, state, result):
            result.validation_time_ms = int((time.perf_counter() - start_time) * 1000)
            return result

        # 8. Margin validation
        if not self._validate_margin(proposal, state, result):
            result.validation_time_ms = int((time.perf_counter() - start_time) * 1000)
            return result

        # Check circuit breakers
        breaker_result = self._check_circuit_breakers(state)
        if breaker_result['halt_trading']:
            result.status = ValidationStatus.HALTED
            result.rejections.append(
                f"CIRCUIT_BREAKER: {breaker_result['triggered_breakers']}"
            )
            result.validation_time_ms = int((time.perf_counter() - start_time) * 1000)
            return result

        # Apply modifications if any
        if modified_size or modified_leverage:
            result.status = ValidationStatus.MODIFIED
            result.modified_proposal = TradeProposal(
                symbol=proposal.symbol,
                side=proposal.side,
                size_usd=modified_size or proposal.size_usd,
                entry_price=proposal.entry_price,
                stop_loss=proposal.stop_loss,
                take_profit=proposal.take_profit,
                leverage=modified_leverage or proposal.leverage,
                confidence=proposal.confidence,
                regime=proposal.regime,
            )

        # Calculate final risk metrics
        final_size = modified_size or proposal.size_usd
        if float(state.current_equity) > 0:
            result.risk_per_trade_pct = (final_size / float(state.current_equity)) * 100
        result.portfolio_exposure_pct = state.total_exposure_pct + result.risk_per_trade_pct

        result.validation_time_ms = int((time.perf_counter() - start_time) * 1000)

        logger.debug(
            f"Trade validation: {result.status.value} in {result.validation_time_ms}ms"
        )

        return result

    def _validate_stop_loss(
        self,
        proposal: TradeProposal,
        result: RiskValidation
    ) -> bool:
        """Validate stop-loss requirements."""
        if self.require_stop_loss and not proposal.stop_loss:
            result.status = ValidationStatus.REJECTED
            result.rejections.append("STOP_LOSS_REQUIRED: No stop-loss specified")
            return False

        if proposal.stop_loss:
            # Calculate stop distance
            if proposal.side == "buy":
                stop_distance_pct = (
                    (proposal.entry_price - proposal.stop_loss) / proposal.entry_price * 100
                )
            else:
                stop_distance_pct = (
                    (proposal.stop_loss - proposal.entry_price) / proposal.entry_price * 100
                )

            # Check minimum distance
            if stop_distance_pct < self.min_stop_pct:
                result.status = ValidationStatus.REJECTED
                result.rejections.append(
                    f"STOP_TOO_TIGHT: {stop_distance_pct:.2f}% < {self.min_stop_pct}% min"
                )
                return False

            # Check maximum distance
            if stop_distance_pct > self.max_stop_pct:
                result.status = ValidationStatus.REJECTED
                result.rejections.append(
                    f"STOP_TOO_WIDE: {stop_distance_pct:.2f}% > {self.max_stop_pct}% max"
                )
                return False

        # Validate risk/reward if take_profit set
        if proposal.stop_loss and proposal.take_profit:
            rr = proposal.calculate_risk_reward()
            if rr and rr < self.min_risk_reward:
                result.warnings.append(
                    f"LOW_RR: {rr:.2f} < {self.min_risk_reward} minimum"
                )

        return True

    def _validate_confidence(
        self,
        proposal: TradeProposal,
        state: RiskState,
        result: RiskValidation,
        entry_strictness: str = "normal",
    ) -> bool:
        """
        Validate confidence requirements.

        Args:
            proposal: Trade proposal
            state: Current risk state
            result: Validation result to update
            entry_strictness: Entry strictness from regime detection

        Returns:
            True if confidence is valid
        """
        # Get minimum confidence based on consecutive losses
        min_conf = self.min_confidence
        for losses, threshold in sorted(
            self.confidence_thresholds.items(), reverse=True
        ):
            if state.consecutive_losses >= losses:
                min_conf = threshold
                break

        # Apply entry strictness adjustment from regime detection
        strictness_adjustment = self.entry_strictness_adjustments.get(
            entry_strictness, 0.0
        )
        adjusted_min_conf = min(1.0, min_conf + strictness_adjustment)

        if strictness_adjustment != 0:
            logger.debug(
                f"Entry strictness '{entry_strictness}' adjusted min confidence: "
                f"{min_conf:.2f} -> {adjusted_min_conf:.2f}"
            )

        if proposal.confidence < adjusted_min_conf:
            result.status = ValidationStatus.REJECTED
            result.rejections.append(
                f"CONFIDENCE_TOO_LOW: {proposal.confidence:.2f} < {adjusted_min_conf:.2f} required "
                f"(base={min_conf:.2f}, strictness={entry_strictness})"
            )
            return False

        # Check for agent disagreement
        if proposal.agent_confidences:
            ta_conf = proposal.agent_confidences.get('technical_analysis', 0)
            sent_conf = proposal.agent_confidences.get('sentiment', 0)

            if ta_conf > 0 and sent_conf > 0:
                diff = abs(ta_conf - sent_conf)
                if diff > 0.2:
                    result.warnings.append(
                        f"AGENT_DISAGREEMENT: TA={ta_conf:.2f}, Sentiment={sent_conf:.2f}"
                    )

        return True

    def _validate_position_size(
        self,
        proposal: TradeProposal,
        state: RiskState,
        result: RiskValidation
    ) -> Optional[float]:
        """
        Validate position size, returning modified size if needed.

        Returns:
            Modified size if adjustment needed, None otherwise
        """
        if float(state.current_equity) == 0:
            return None

        equity = float(state.current_equity)
        max_size = equity * (self.max_position_pct / 100)
        max_risk_size = equity * (self.max_risk_per_trade_pct / 100)

        modified_size = None

        # Check position size limit
        if proposal.size_usd > max_size:
            modified_size = max_size
            result.modifications['size_usd'] = {
                'original': proposal.size_usd,
                'modified': max_size,
                'reason': f"Exceeds max position size ({self.max_position_pct}%)"
            }
            result.warnings.append(
                f"SIZE_REDUCED: ${proposal.size_usd:.2f} -> ${max_size:.2f}"
            )

        # Check risk per trade (if stop loss set)
        if proposal.stop_loss:
            if proposal.side == "buy":
                risk_pct = (proposal.entry_price - proposal.stop_loss) / proposal.entry_price
            else:
                risk_pct = (proposal.stop_loss - proposal.entry_price) / proposal.entry_price

            position_risk = proposal.size_usd * risk_pct
            if position_risk > max_risk_size:
                safe_size = max_risk_size / risk_pct
                if modified_size is None or safe_size < modified_size:
                    modified_size = safe_size
                    result.modifications['size_usd'] = {
                        'original': proposal.size_usd,
                        'modified': safe_size,
                        'reason': f"Risk exceeds {self.max_risk_per_trade_pct}% of equity"
                    }

        return modified_size

    def _validate_leverage(
        self,
        proposal: TradeProposal,
        state: RiskState,
        result: RiskValidation
    ) -> Optional[int]:
        """
        Validate leverage, returning modified leverage if needed.

        Returns:
            Modified leverage if adjustment needed, None otherwise
        """
        max_allowed = self.max_leverage

        # Adjust for regime
        regime_limit = self.regime_leverage_limits.get(proposal.regime, 2)
        max_allowed = min(max_allowed, regime_limit)

        # Adjust for drawdown
        if state.current_drawdown_pct >= 15:
            max_allowed = min(max_allowed, 1)
        elif state.current_drawdown_pct >= 10:
            max_allowed = min(max_allowed, 2)
        elif state.current_drawdown_pct >= 5:
            max_allowed = min(max_allowed, 3)

        # Adjust for consecutive losses
        if state.consecutive_losses >= 5:
            max_allowed = 1
        elif state.consecutive_losses >= 3:
            max_allowed = min(max_allowed, 2)

        if proposal.leverage > max_allowed:
            result.modifications['leverage'] = {
                'original': proposal.leverage,
                'modified': max_allowed,
                'reason': f"Reduced due to regime/drawdown/losses"
            }
            result.warnings.append(
                f"LEVERAGE_REDUCED: {proposal.leverage}x -> {max_allowed}x"
            )
            return max_allowed

        return None

    def _validate_exposure(
        self,
        proposal: TradeProposal,
        state: RiskState,
        result: RiskValidation
    ) -> bool:
        """
        Validate total portfolio exposure.

        CRITICAL: Exposure includes leverage multiplier.
        $1000 at 5x leverage = 50% exposure on $10k portfolio, not 10%!
        """
        if float(state.current_equity) == 0:
            return True

        # CRITICAL FIX: Include leverage in exposure calculation
        # Example: $1000 position at 5x = $5000 actual market exposure
        actual_exposure_usd = proposal.size_usd * proposal.leverage
        position_exposure = (actual_exposure_usd / float(state.current_equity)) * 100
        new_total = state.total_exposure_pct + position_exposure

        if new_total > self.max_exposure_pct:
            result.status = ValidationStatus.REJECTED
            result.rejections.append(
                f"EXPOSURE_LIMIT: Would be {new_total:.1f}% > {self.max_exposure_pct}% max "
                f"(${proposal.size_usd:.2f} x {proposal.leverage}x leverage = ${actual_exposure_usd:.2f} exposure)"
            )
            return False

        return True

    def _validate_margin(
        self,
        proposal: TradeProposal,
        state: RiskState,
        result: RiskValidation
    ) -> bool:
        """Validate sufficient margin available."""
        required_margin = proposal.size_usd / proposal.leverage

        if required_margin > float(state.available_margin):
            result.status = ValidationStatus.REJECTED
            result.rejections.append(
                f"INSUFFICIENT_MARGIN: Need ${required_margin:.2f}, "
                f"have ${float(state.available_margin):.2f}"
            )
            return False

        return True

    def _check_circuit_breakers(self, state: RiskState) -> dict:
        """Check all circuit breakers."""
        result = {
            'halt_trading': False,
            'close_positions': False,
            'reduce_positions_pct': 0,
            'triggered_breakers': [],
        }

        # Daily loss check
        if abs(state.daily_pnl_pct) >= self.daily_loss_limit_pct and state.daily_pnl_pct < 0:
            result['halt_trading'] = True
            result['triggered_breakers'].append('daily_loss')

        # Weekly loss check
        if abs(state.weekly_pnl_pct) >= self.weekly_loss_limit_pct and state.weekly_pnl_pct < 0:
            result['halt_trading'] = True
            result['reduce_positions_pct'] = 50
            result['triggered_breakers'].append('weekly_loss')

        # Max drawdown check
        if state.current_drawdown_pct >= self.max_drawdown_limit_pct:
            result['halt_trading'] = True
            result['close_positions'] = True
            result['triggered_breakers'].append('max_drawdown')

        return result

    def update_state(
        self,
        current_equity: Decimal,
        daily_pnl: Decimal,
        weekly_pnl: Decimal,
        open_positions: int,
        total_exposure_pct: float,
        available_margin: Decimal,
    ) -> RiskState:
        """
        Update risk state with current portfolio values.

        Args:
            current_equity: Current portfolio equity
            daily_pnl: Today's P&L
            weekly_pnl: This week's P&L
            open_positions: Number of open positions
            total_exposure_pct: Current exposure percentage
            available_margin: Available margin for new trades

        Returns:
            Updated RiskState
        """
        self._risk_state.current_equity = current_equity
        self._risk_state.daily_pnl = daily_pnl
        self._risk_state.weekly_pnl = weekly_pnl
        self._risk_state.open_positions = open_positions
        self._risk_state.total_exposure_pct = total_exposure_pct
        self._risk_state.available_margin = available_margin

        # Calculate percentages
        if float(current_equity) > 0:
            self._risk_state.daily_pnl_pct = float(daily_pnl / current_equity * 100)
            self._risk_state.weekly_pnl_pct = float(weekly_pnl / current_equity * 100)

        # Update drawdown
        self._risk_state.update_drawdown()

        # Check circuit breakers
        breaker_result = self._check_circuit_breakers(self._risk_state)
        if breaker_result['halt_trading']:
            self._risk_state.trading_halted = True
            self._risk_state.triggered_breakers = breaker_result['triggered_breakers']
            self._risk_state.halt_reason = ', '.join(breaker_result['triggered_breakers'])

        return self._risk_state

    def record_trade_result(self, is_win: bool) -> None:
        """
        Record trade result for consecutive tracking.

        Args:
            is_win: True if trade was profitable
        """
        if is_win:
            self._risk_state.consecutive_wins += 1
            self._risk_state.consecutive_losses = 0
        else:
            self._risk_state.consecutive_losses += 1
            self._risk_state.consecutive_wins = 0

            # Apply cooldown for consecutive losses
            if self._risk_state.consecutive_losses >= 5:
                self._apply_cooldown(
                    self.consecutive_loss_cooldown_min * 2,
                    f"5+ consecutive losses"
                )
            elif self._risk_state.consecutive_losses >= 3:
                self._apply_cooldown(
                    self.consecutive_loss_cooldown_min,
                    f"3+ consecutive losses"
                )
            else:
                self._apply_cooldown(
                    self.post_loss_cooldown_min,
                    "Post-loss cooldown"
                )

    def apply_post_trade_cooldown(self) -> None:
        """Apply standard post-trade cooldown."""
        self._apply_cooldown(
            self.post_trade_cooldown_min,
            "Post-trade cooldown"
        )

    def _apply_cooldown(self, minutes: int, reason: str) -> None:
        """Apply a cooldown period."""
        self._risk_state.in_cooldown = True
        self._risk_state.cooldown_until = (
            datetime.now(timezone.utc) + timedelta(minutes=minutes)
        )
        self._risk_state.cooldown_reason = reason
        logger.info(f"Cooldown applied: {reason} for {minutes} minutes")

    def reset_daily(self) -> None:
        """Reset daily tracking (call at UTC midnight)."""
        self._risk_state.daily_pnl = Decimal("0")
        self._risk_state.daily_pnl_pct = 0.0

        # Clear daily halt if it was triggered
        if 'daily_loss' in self._risk_state.triggered_breakers:
            self._risk_state.triggered_breakers.remove('daily_loss')
            if not self._risk_state.triggered_breakers:
                self._risk_state.trading_halted = False
                self._risk_state.halt_reason = ""

        logger.info("Daily risk state reset")

    def reset_weekly(self) -> None:
        """Reset weekly tracking (call at UTC Monday midnight)."""
        self._risk_state.weekly_pnl = Decimal("0")
        self._risk_state.weekly_pnl_pct = 0.0

        # Clear weekly halt if it was triggered
        if 'weekly_loss' in self._risk_state.triggered_breakers:
            self._risk_state.triggered_breakers.remove('weekly_loss')
            if not self._risk_state.triggered_breakers:
                self._risk_state.trading_halted = False
                self._risk_state.halt_reason = ""

        logger.info("Weekly risk state reset")

    def manual_reset(self, admin_override: bool = False) -> bool:
        """
        Manually reset trading halt (requires admin override).

        Args:
            admin_override: Must be True to reset max drawdown halt

        Returns:
            True if reset successful
        """
        if 'max_drawdown' in self._risk_state.triggered_breakers:
            if not admin_override:
                logger.warning("Max drawdown halt requires admin override")
                return False

        self._risk_state.trading_halted = False
        self._risk_state.halt_reason = ""
        self._risk_state.triggered_breakers = []
        self._risk_state.in_cooldown = False
        self._risk_state.cooldown_until = None
        self._risk_state.cooldown_reason = ""

        logger.info("Risk state manually reset")
        return True

    def get_state(self) -> RiskState:
        """Get current risk state."""
        return self._risk_state

    def get_max_allowed_leverage(
        self,
        regime: str,
        drawdown_pct: float,
        consecutive_losses: int
    ) -> int:
        """Calculate maximum allowed leverage for current conditions."""
        max_lev = self.regime_leverage_limits.get(regime, 2)

        if drawdown_pct >= 15:
            max_lev = min(max_lev, 1)
        elif drawdown_pct >= 10:
            max_lev = min(max_lev, 2)
        elif drawdown_pct >= 5:
            max_lev = min(max_lev, 3)

        if consecutive_losses >= 5:
            max_lev = 1
        elif consecutive_losses >= 3:
            max_lev = min(max_lev, 2)

        return min(max_lev, self.max_leverage)

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        regime: str,
        confidence: float,
    ) -> float:
        """
        Calculate appropriate position size based on risk parameters.

        Uses ATR-based position sizing adjusted for regime and confidence.

        Args:
            equity: Total portfolio equity
            entry_price: Planned entry price
            stop_loss: Stop-loss price
            regime: Current market regime
            confidence: Signal confidence

        Returns:
            Position size in USD
        """
        # Base risk per trade
        risk_per_trade = equity * (self.max_risk_per_trade_pct / 100)

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return 0.0

        stop_distance_pct = (stop_distance / entry_price) * 100

        # Base position size from risk
        base_size = (risk_per_trade / stop_distance_pct) * 100

        # Confidence adjustment
        if confidence >= 0.85:
            conf_mult = 1.0
        elif confidence >= 0.75:
            conf_mult = 0.75
        elif confidence >= 0.65:
            conf_mult = 0.5
        elif confidence >= 0.60:
            conf_mult = 0.25
        else:
            return 0.0  # No trade

        # Regime adjustment
        regime_mult = {
            'trending_bull': 1.0,
            'trending_bear': 0.8,
            'ranging': 0.6,
            'volatile_bull': 0.4,
            'volatile_bear': 0.4,
            'choppy': 0.25,
            'breakout_potential': 0.75,
        }.get(regime, 0.5)

        # Final size
        final_size = base_size * conf_mult * regime_mult

        # Cap at max position size
        max_size = equity * (self.max_position_pct / 100)
        final_size = min(final_size, max_size)

        return final_size

    def _validate_correlation(
        self,
        proposal: TradeProposal,
        state: RiskState,
        result: RiskValidation
    ) -> bool:
        """
        Validate correlated position exposure.

        Checks if adding this position would exceed max correlated exposure
        when combined with existing positions in correlated pairs.

        Args:
            proposal: Trade proposal
            state: Current risk state
            result: Validation result to update

        Returns:
            True if correlated exposure is acceptable
        """
        if not state.open_position_symbols:
            return True

        total_correlated_exposure = 0.0

        # Calculate correlated exposure with existing positions
        for existing_symbol in state.open_position_symbols:
            correlation = self._get_pair_correlation(proposal.symbol, existing_symbol)
            if correlation > 0.5:  # Consider significantly correlated
                existing_exposure = state.position_exposures.get(existing_symbol, 0.0)
                # Weighted by correlation
                correlated_contribution = existing_exposure * correlation
                total_correlated_exposure += correlated_contribution

        # Add proposed position exposure
        if float(state.current_equity) > 0:
            proposed_exposure = (proposal.size_usd / float(state.current_equity)) * 100
            total_correlated_exposure += proposed_exposure

        if total_correlated_exposure > self.max_correlated_exposure_pct:
            result.status = ValidationStatus.REJECTED
            result.rejections.append(
                f"CORRELATED_EXPOSURE: {total_correlated_exposure:.1f}% > "
                f"{self.max_correlated_exposure_pct}% max correlated exposure"
            )
            return False

        if total_correlated_exposure > self.max_correlated_exposure_pct * 0.8:
            result.warnings.append(
                f"HIGH_CORRELATION: Correlated exposure at {total_correlated_exposure:.1f}%"
            )

        return True

    def _get_pair_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation coefficient between two trading pairs."""
        if symbol1 == symbol2:
            return 1.0

        # Check direct correlation
        key = (symbol1, symbol2)
        if key in PAIR_CORRELATIONS:
            return PAIR_CORRELATIONS[key]

        # Check reverse
        key_rev = (symbol2, symbol1)
        if key_rev in PAIR_CORRELATIONS:
            return PAIR_CORRELATIONS[key_rev]

        return 0.0

    def _check_and_reset_periods(self, state: RiskState) -> None:
        """
        Check if daily/weekly reset is needed and perform it.

        This implements automatic reset without requiring external scheduler.
        """
        now = datetime.now(timezone.utc)

        # Check for daily reset (at UTC midnight)
        if state.last_daily_reset is None:
            state.last_daily_reset = now
        else:
            # Check if we're in a new UTC day
            if now.date() > state.last_daily_reset.date():
                self.reset_daily()
                state.last_daily_reset = now
                logger.info(f"Auto daily reset triggered at {now.isoformat()}")

        # Check for weekly reset (Monday UTC midnight)
        if state.last_weekly_reset is None:
            state.last_weekly_reset = now
        else:
            # Check if we're in a new week (Monday = 0)
            current_week = now.isocalendar()[1]
            last_week = state.last_weekly_reset.isocalendar()[1]
            if current_week != last_week and now.weekday() == 0:
                self.reset_weekly()
                state.last_weekly_reset = now
                logger.info(f"Auto weekly reset triggered at {now.isoformat()}")

    def update_volatility(
        self,
        current_atr: float,
        avg_atr_20: float,
    ) -> bool:
        """
        Update volatility tracking and detect spikes.

        Args:
            current_atr: Current ATR value
            avg_atr_20: 20-period average ATR

        Returns:
            True if volatility spike detected
        """
        self._risk_state.current_atr = current_atr
        self._risk_state.avg_atr_20 = avg_atr_20

        # Check for spike
        if avg_atr_20 > 0:
            spike_detected = current_atr > (avg_atr_20 * self.volatility_spike_multiplier)

            if spike_detected and not self._risk_state.volatility_spike_active:
                self._risk_state.volatility_spike_active = True
                self._apply_cooldown(
                    self.volatility_spike_cooldown_min,
                    f"Volatility spike: ATR {current_atr:.4f} > {avg_atr_20 * self.volatility_spike_multiplier:.4f}"
                )
                logger.warning(
                    f"Volatility spike detected: ATR={current_atr:.4f}, "
                    f"Avg={avg_atr_20:.4f}, Ratio={current_atr/avg_atr_20:.2f}x"
                )
                return True
            elif not spike_detected and self._risk_state.volatility_spike_active:
                self._risk_state.volatility_spike_active = False
                logger.info("Volatility normalized, spike condition cleared")

        return False

    def update_positions(
        self,
        open_symbols: list[str],
        exposures: dict[str, float],
    ) -> None:
        """
        Update open position tracking for correlation checks.

        Args:
            open_symbols: List of symbols with open positions
            exposures: Dict of symbol -> exposure percentage
        """
        self._risk_state.open_position_symbols = open_symbols
        self._risk_state.position_exposures = exposures

    async def persist_state(self) -> bool:
        """
        Persist current risk state to database.

        Returns:
            True if persisted successfully
        """
        if self.db is None:
            logger.debug("No database configured, skipping state persistence")
            return False

        try:
            state_json = json.dumps(self._risk_state.to_dict())

            query = """
                INSERT INTO risk_state (id, state_data, updated_at)
                VALUES ('current', $1, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    state_data = $1,
                    updated_at = NOW()
            """
            await self.db.execute(query, state_json)
            logger.debug("Risk state persisted to database")
            return True

        except Exception as e:
            logger.error(f"Failed to persist risk state: {e}")
            return False

    async def load_state(self) -> bool:
        """
        Load risk state from database on startup.

        Returns:
            True if loaded successfully
        """
        if self.db is None:
            logger.debug("No database configured, using fresh state")
            return False

        try:
            query = """
                SELECT state_data FROM risk_state
                WHERE id = 'current'
            """
            row = await self.db.fetchrow(query)

            if row and row['state_data']:
                state_data = json.loads(row['state_data'])
                self._risk_state = RiskState.from_dict(state_data)
                logger.info("Risk state loaded from database")
                return True
            else:
                logger.info("No persisted risk state found, using fresh state")
                return False

        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")
            return False
