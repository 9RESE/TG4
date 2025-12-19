# Risk Management Standards

**Version**: 1.0
**Last Updated**: 2025-12-19
**Status**: Active - Production Standard

---

## Overview

This document establishes mandatory standards for risk management implementations in the TripleGain trading system. These standards emerged from comprehensive code review of Phase 3 risk management implementation and represent battle-tested patterns for financial risk control.

---

## 1. Core Principles

### 1.1 Deterministic Behavior

**MANDATORY**: All risk management logic MUST be deterministic and rules-based.

```python
# ✅ CORRECT: Pure rules-based logic
def validate_trade(proposal, state):
    if proposal.confidence < 0.60:
        return REJECTED
    if state.daily_pnl_pct < -5.0:
        return HALTED
    return APPROVED

# ❌ WRONG: LLM-dependent risk decisions
def validate_trade(proposal, state):
    llm_advice = ask_llm("Should we take this trade?")
    if llm_advice == "yes":
        return APPROVED
```

**Rationale**: Financial risk cannot depend on non-deterministic AI responses. All risk decisions must be auditable and reproducible.

---

### 1.2 Performance Requirements

**MANDATORY**: All risk validation MUST complete in <10ms.

```python
# ✅ CORRECT: Fast, in-memory validation
def validate_trade(proposal, state):
    start = time.perf_counter()
    # ... validation logic ...
    latency_ms = (time.perf_counter() - start) * 1000
    assert latency_ms < 10, f"Validation too slow: {latency_ms}ms"

# ❌ WRONG: Network calls in validation hot path
def validate_trade(proposal, state):
    db_state = fetch_from_database()  # Network I/O = slow
    llm_check = call_risk_api()        # Network I/O = slow
```

**Enforcement**: All validation functions MUST include latency assertion in unit tests.

---

### 1.3 Defense in Depth

**MANDATORY**: Implement multiple independent risk layers.

```python
# ✅ CORRECT: Layered validation
def validate_trade(proposal, state):
    # Layer 1: Pre-trade validation
    if not validate_stop_loss(proposal):
        return REJECTED

    # Layer 2: Position limits
    if not validate_position_size(proposal, state):
        return REJECTED

    # Layer 3: Portfolio exposure
    if not validate_exposure(proposal, state):
        return REJECTED

    # Layer 4: Circuit breakers
    if check_circuit_breakers(state).halt_trading:
        return HALTED

    return APPROVED
```

**Rationale**: Single point of failure is unacceptable in financial systems. Each layer can independently halt dangerous trades.

---

## 2. Configuration Standards

### 2.1 All Risk Parameters MUST Be Configurable

**MANDATORY**: No hardcoded risk limits in code.

```yaml
# ✅ CORRECT: Config-driven limits
limits:
  max_leverage: 5
  max_position_pct: 20
  max_total_exposure_pct: 80
  max_risk_per_trade_pct: 2
  min_confidence: 0.60
  max_correlated_exposure_pct: 40  # All limits in config

circuit_breakers:
  daily_loss:
    threshold_pct: 5.0
    action: halt_new_trades
```

```python
# ✅ CORRECT: Load from config
def __init__(self, config: dict):
    limits = config.get('limits', {})
    self.max_leverage = limits.get('max_leverage', 5)
    # Fallback defaults acceptable, but MUST be documented

# ❌ WRONG: Hardcoded limits
def __init__(self):
    self.max_leverage = 5  # What if we need to change this?
    self.daily_loss_limit = 5.0  # Not configurable!
```

**Discovered Issue**: During review, `max_correlated_exposure_pct` was referenced in code but missing from config. This is a **VIOLATION** of this standard.

**Fix Template**:
```yaml
# If code references a parameter, config MUST define it
limits:
  parameter_name: default_value  # With comment explaining purpose
```

---

### 2.2 Magic Numbers MUST Be Named Constants or Config

```python
# ✅ CORRECT: Named constants
CORRELATION_WARNING_THRESHOLD_MULTIPLIER = 0.8

if correlated_exposure > max_limit * CORRELATION_WARNING_THRESHOLD_MULTIPLIER:
    result.warnings.append("HIGH_CORRELATION")

# ✅ BETTER: Config parameter
config.yaml:
  limits:
    correlated_exposure_warn_pct: 32  # 80% of 40% max

# ❌ WRONG: Magic number
if total_correlated_exposure > self.max_correlated_exposure_pct * 0.8:
    # What is 0.8? Why 80%?
```

---

## 3. Circuit Breaker Standards

### 3.1 Mandatory Circuit Breakers

**REQUIRED**: All trading systems MUST implement these circuit breakers:

1. **Daily Loss Limit** (Default: 5% of equity)
   - Halts new trades until daily reset (UTC midnight)
   - Does NOT close existing positions

2. **Weekly Loss Limit** (Default: 10% of equity)
   - Halts new trades
   - MAY reduce existing positions by 50%
   - Resets Monday UTC midnight

3. **Maximum Drawdown** (Default: 20% from peak)
   - Closes ALL positions immediately
   - Halts all trading
   - Requires manual intervention to resume

4. **Consecutive Losses** (Default: 5 losses)
   - Applies cooldown period (30-60 min)
   - Reduces next trade size by 50%
   - Auto-resumes after cooldown

### 3.2 Circuit Breaker Implementation Pattern

```python
# ✅ STANDARD PATTERN
@dataclass
class RiskState:
    trading_halted: bool = False
    halt_reason: str = ""
    halt_until: Optional[datetime] = None
    triggered_breakers: list[str] = field(default_factory=list)

def check_circuit_breakers(state: RiskState) -> dict:
    """Check all circuit breakers, return actions."""
    result = {
        'halt_trading': False,
        'close_positions': False,
        'triggered_breakers': [],
    }

    # Daily loss
    if state.daily_pnl_pct < -5.0:
        result['halt_trading'] = True
        result['triggered_breakers'].append('daily_loss')

    # Weekly loss
    if state.weekly_pnl_pct < -10.0:
        result['halt_trading'] = True
        result['reduce_positions_pct'] = 50
        result['triggered_breakers'].append('weekly_loss')

    # Max drawdown
    if state.current_drawdown_pct > 20.0:
        result['halt_trading'] = True
        result['close_positions'] = True
        result['triggered_breakers'].append('max_drawdown')

    return result
```

### 3.3 Automatic vs Manual Reset

**STANDARD**:
- Daily/Weekly circuit breakers: **Automatic reset**
- Max drawdown circuit breaker: **Manual reset with admin override**

```python
def reset_daily(self) -> None:
    """Auto-reset at UTC midnight."""
    self._risk_state.daily_pnl = Decimal("0")
    if 'daily_loss' in self._risk_state.triggered_breakers:
        self._risk_state.triggered_breakers.remove('daily_loss')
        if not self._risk_state.triggered_breakers:
            self._risk_state.trading_halted = False

def manual_reset(self, admin_override: bool = False) -> bool:
    """Manual reset requires admin override for max_drawdown."""
    if 'max_drawdown' in self._risk_state.triggered_breakers:
        if not admin_override:
            return False  # Require explicit override
    self._risk_state.trading_halted = False
    return True
```

**Rationale**: Daily/weekly losses are part of normal trading variance. Max drawdown indicates systemic failure requiring human review.

---

## 4. Position Sizing Standards

### 4.1 ATR-Based Position Sizing

**RECOMMENDED**: Use ATR-based position sizing for volatility adjustment.

```python
def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    regime: str,
    confidence: float,
) -> float:
    """Calculate position size based on risk parameters."""
    # Base risk per trade (e.g., 2% of equity)
    risk_per_trade = equity * (MAX_RISK_PER_TRADE_PCT / 100)

    # Stop distance
    stop_distance = abs(entry_price - stop_loss)
    stop_distance_pct = (stop_distance / entry_price) * 100

    # Base position size
    base_size = (risk_per_trade / stop_distance_pct) * 100

    # Adjust for confidence
    conf_mult = get_confidence_multiplier(confidence)

    # Adjust for regime
    regime_mult = get_regime_multiplier(regime)

    # Final size
    final_size = base_size * conf_mult * regime_mult

    # Cap at max position size
    max_size = equity * (MAX_POSITION_PCT / 100)
    return min(final_size, max_size)
```

### 4.2 Position Size Limits

**MANDATORY LIMITS**:

```python
MAX_POSITION_PCT = 20       # Single position max 20% of equity
MAX_RISK_PER_TRADE_PCT = 2  # Max 2% risk per trade
MAX_TOTAL_EXPOSURE_PCT = 80 # Total exposure max 80% of equity
```

**Edge Case Handling**:
```python
# ✅ CORRECT: Handle zero/invalid inputs
if equity <= 0:
    return 0.0
if stop_distance <= 0:
    return 0.0
if confidence < MIN_CONFIDENCE:
    return 0.0  # No trade
```

---

## 5. Leverage Standards

### 5.1 Regime-Adjusted Leverage

**MANDATORY**: Leverage MUST be adjusted based on market regime.

```python
REGIME_LEVERAGE_LIMITS = {
    'trending_bull': 5,
    'trending_bear': 3,
    'ranging': 2,
    'volatile_bull': 2,
    'volatile_bear': 2,
    'choppy': 1,
    'breakout_potential': 3,
    'unknown': 1,  # Conservative default
}
```

### 5.2 Drawdown-Adjusted Leverage

**MANDATORY**: Reduce leverage during drawdown.

```python
def apply_drawdown_leverage_limit(base_leverage: int, drawdown_pct: float) -> int:
    """Reduce leverage based on current drawdown."""
    if drawdown_pct >= 15:
        return min(base_leverage, 1)  # Only 1x leverage
    elif drawdown_pct >= 10:
        return min(base_leverage, 2)  # Max 2x
    elif drawdown_pct >= 5:
        return min(base_leverage, 3)  # Max 3x
    return base_leverage
```

### 5.3 Consecutive Loss Leverage Reduction

```python
# After 3 losses: max 2x leverage
if consecutive_losses >= 5:
    max_leverage = 1
elif consecutive_losses >= 3:
    max_leverage = min(max_leverage, 2)
```

---

## 6. Stop-Loss Standards

### 6.1 Mandatory Stop-Loss

**ABSOLUTE REQUIREMENT**: Every trade MUST have a stop-loss before execution.

```python
def validate_stop_loss(proposal: TradeProposal) -> bool:
    """Validate stop-loss requirements."""
    if not proposal.stop_loss:
        return False  # REJECT: No stop-loss

    # Calculate stop distance
    if proposal.side == "buy":
        stop_distance_pct = (proposal.entry_price - proposal.stop_loss) / proposal.entry_price * 100
    else:
        stop_distance_pct = (proposal.stop_loss - proposal.entry_price) / proposal.entry_price * 100

    # Validate distance
    if stop_distance_pct < MIN_STOP_PCT:  # Default 0.5%
        return False  # REJECT: Too tight

    if stop_distance_pct > MAX_STOP_PCT:  # Default 5%
        return False  # REJECT: Too wide

    return True
```

### 6.2 Risk/Reward Ratio Validation

**RECOMMENDED**: Minimum R:R ratio of 1.5:1

```python
def calculate_risk_reward(proposal: TradeProposal) -> Optional[float]:
    """Calculate risk/reward ratio."""
    if not proposal.stop_loss or not proposal.take_profit:
        return None

    if proposal.side == "buy":
        risk = proposal.entry_price - proposal.stop_loss
        reward = proposal.take_profit - proposal.entry_price
    else:
        risk = proposal.stop_loss - proposal.entry_price
        reward = proposal.entry_price - proposal.take_profit

    return reward / risk if risk > 0 else None

# Usage
rr = calculate_risk_reward(proposal)
if rr and rr < MIN_RISK_REWARD:  # 1.5
    if ENFORCE_MIN_RISK_REWARD:
        return REJECTED
    else:
        warnings.append(f"LOW_RR: {rr:.2f}")
```

**Discovered Pattern**: Current implementation only **warns** about low R:R. Consider making enforcement configurable:

```yaml
stop_loss:
  min_risk_reward: 1.5
  enforce_min_risk_reward: false  # Warning vs rejection
```

---

## 7. Confidence Threshold Standards

### 7.1 Tiered Confidence System

**STANDARD PATTERN**:

```python
CONFIDENCE_TIERS = {
    'very_high': {'min': 0.85, 'multiplier': 1.0},
    'high':      {'min': 0.75, 'multiplier': 0.75},
    'medium':    {'min': 0.65, 'multiplier': 0.50},
    'low':       {'min': 0.60, 'multiplier': 0.25},
    'none':      {'min': 0.00, 'multiplier': 0.0},  # No trade
}

MIN_CONFIDENCE = 0.60  # Absolute minimum
```

### 7.2 Consecutive Loss Adjustment

**MANDATORY**: Increase confidence requirement after losses.

```python
def get_adjusted_min_confidence(consecutive_losses: int) -> float:
    """Adjust minimum confidence based on loss streak."""
    if consecutive_losses >= 5:
        return 0.80  # Very high confidence required
    elif consecutive_losses >= 3:
        return 0.70  # High confidence required
    else:
        return 0.60  # Base minimum
```

### 7.3 Entry Strictness (Regime-Based)

**NEW PATTERN** (discovered in implementation):

```python
ENTRY_STRICTNESS_ADJUSTMENTS = {
    'relaxed': -0.05,      # Lower required confidence
    'normal': 0.0,         # No adjustment
    'strict': 0.05,        # Higher required confidence
    'very_strict': 0.10,   # Much higher required confidence
}

# Apply during validation
adjusted_min_conf = min(1.0, base_min_conf + strictness_adjustment)
```

**Testing Requirement**: MUST test all strictness levels (discovered gap in review).

---

## 8. Correlation Risk Standards

### 8.1 Pair Correlation Matrix

**REQUIRED**: Define correlation coefficients for traded pairs.

```python
# Correlation matrix (based on historical data)
PAIR_CORRELATIONS = {
    ('BTC/USDT', 'XRP/USDT'): 0.75,
    ('BTC/USDT', 'XRP/BTC'): 0.60,
    ('XRP/USDT', 'XRP/BTC'): 0.85,
}

def get_pair_correlation(symbol1: str, symbol2: str) -> float:
    """Get correlation coefficient between pairs."""
    if symbol1 == symbol2:
        return 1.0  # Perfect correlation

    key = (symbol1, symbol2)
    if key in PAIR_CORRELATIONS:
        return PAIR_CORRELATIONS[key]

    # Check reverse
    key_rev = (symbol2, symbol1)
    if key_rev in PAIR_CORRELATIONS:
        return PAIR_CORRELATIONS[key_rev]

    return 0.0  # Unknown pairs assumed uncorrelated
```

### 8.2 Correlated Exposure Limit

**MANDATORY**: Limit total correlated position exposure.

```python
MAX_CORRELATED_EXPOSURE_PCT = 40  # Max 40% in correlated positions

def validate_correlation(proposal, state):
    """Check if new position would exceed correlated exposure limit."""
    total_correlated = 0.0

    for existing_symbol in state.open_position_symbols:
        correlation = get_pair_correlation(proposal.symbol, existing_symbol)
        if correlation > 0.5:  # Significantly correlated
            existing_exposure = state.position_exposures[existing_symbol]
            # Weight by correlation strength
            total_correlated += existing_exposure * correlation

    # Add proposed position
    proposed_exposure = (proposal.size_usd / state.current_equity) * 100
    total_correlated += proposed_exposure

    if total_correlated > MAX_CORRELATED_EXPOSURE_PCT:
        return REJECTED
```

---

## 9. Volatility Standards

### 9.1 Volatility Spike Detection

**STANDARD PATTERN**: Use ATR ratio for spike detection.

```python
VOLATILITY_SPIKE_MULTIPLIER = 3.0  # ATR > 3x average = spike

def update_volatility(current_atr: float, avg_atr_20: float) -> bool:
    """Detect volatility spikes."""
    if avg_atr_20 <= 0:
        return False  # Cannot calculate

    spike_detected = current_atr > (avg_atr_20 * VOLATILITY_SPIKE_MULTIPLIER)

    if spike_detected:
        # Apply cooldown
        apply_cooldown(VOLATILITY_SPIKE_COOLDOWN_MIN)
        return True

    return False
```

### 9.2 Volatility Size Reduction

**MANDATORY**: Reduce position size during high volatility.

```python
VOLATILITY_SIZE_REDUCTION_PCT = 50  # Reduce by 50%

if volatility_spike_active:
    reduced_size = original_size * (1 - VOLATILITY_SIZE_REDUCTION_PCT / 100)
    modifications['volatility_adjustment'] = {
        'original': original_size,
        'modified': reduced_size,
        'reason': 'Volatility spike detected',
    }
```

---

## 10. Cooldown Standards

### 10.1 Mandatory Cooldown Periods

**REQUIRED COOLDOWNS**:

```python
COOLDOWN_PERIODS = {
    'post_trade': 5,           # After ANY trade
    'post_loss': 10,           # After losing trade
    'consecutive_loss_3': 30,  # After 3 losses
    'consecutive_loss_5': 60,  # After 5 losses
    'volatility_spike': 15,    # After spike detected
    'post_rebalance': 30,      # After portfolio rebalance
}
```

### 10.2 Cooldown Implementation

```python
@dataclass
class RiskState:
    in_cooldown: bool = False
    cooldown_until: Optional[datetime] = None
    cooldown_reason: str = ""

def apply_cooldown(minutes: int, reason: str) -> None:
    """Apply cooldown period."""
    self._risk_state.in_cooldown = True
    self._risk_state.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    self._risk_state.cooldown_reason = reason

def check_cooldown() -> tuple[bool, str]:
    """Check if in active cooldown."""
    if not self._risk_state.in_cooldown:
        return False, ""

    if datetime.now(timezone.utc) < self._risk_state.cooldown_until:
        return True, self._risk_state.cooldown_reason

    # Expired
    self._risk_state.in_cooldown = False
    return False, ""
```

---

## 11. State Management Standards

### 11.1 Drawdown Calculation

**CRITICAL EDGE CASES** (discovered in review):

```python
def update_drawdown(state: RiskState) -> None:
    """Update drawdown with proper edge case handling."""

    # Edge case 1: Initial state (zero equity)
    if state.current_equity <= 0 and state.peak_equity <= 0:
        state.current_drawdown_pct = 0.0
        return

    # Edge case 2: New high (update peak)
    if state.current_equity > state.peak_equity:
        state.peak_equity = state.current_equity
        state.current_drawdown_pct = 0.0
        return

    # Edge case 3: Negative equity (>100% drawdown)
    if state.peak_equity > 0:
        if state.current_equity < 0:
            state.current_drawdown_pct = 100.0 + float(
                abs(state.current_equity) / state.peak_equity * 100
            )
        else:
            # Normal drawdown calculation
            state.current_drawdown_pct = float(
                (state.peak_equity - state.current_equity) / state.peak_equity * 100
            )

        # Track max drawdown seen
        state.max_drawdown_pct = max(state.max_drawdown_pct, state.current_drawdown_pct)
```

**Rationale**: Drawdown calculation has multiple edge cases that can cause division by zero or incorrect results. MUST handle:
- Zero initial equity
- Negative equity (account blow-up)
- New equity highs

### 11.2 State Persistence

**REQUIRED**: Risk state MUST be persistable for crash recovery.

```python
@dataclass
class RiskState:
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            'peak_equity': str(self.peak_equity),  # Decimal → str
            'current_equity': str(self.current_equity),
            'halt_until': self.halt_until.isoformat() if self.halt_until else None,
            # ... all fields ...
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RiskState':
        """Deserialize from dict."""
        state = cls()
        state.peak_equity = Decimal(data.get('peak_equity', '0'))
        state.halt_until = (
            datetime.fromisoformat(data['halt_until'])
            if data.get('halt_until') else None
        )
        # ... all fields ...
        return state
```

**Database Schema**:
```sql
CREATE TABLE risk_state (
    id TEXT PRIMARY KEY,  -- 'current' for active state
    state_data JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 12. Testing Standards

### 12.1 Mandatory Test Coverage

**REQUIREMENTS**:

1. **Latency Test** (verify <10ms):
```python
def test_validation_latency_under_10ms():
    result = risk_engine.validate_trade(proposal, state)
    assert result.validation_time_ms < 10
```

2. **All Circuit Breakers**:
```python
def test_daily_loss_halts_trading()
def test_weekly_loss_halts_trading()
def test_max_drawdown_closes_all()
def test_consecutive_losses_applies_cooldown()
```

3. **Edge Cases**:
```python
def test_zero_equity_no_drawdown()
def test_negative_equity_over_100_drawdown()
def test_zero_stop_distance()
def test_insufficient_margin()
```

4. **State Serialization**:
```python
def test_roundtrip_serialization()
def test_datetime_serialization()
```

### 12.2 Test Coverage Minimum

**TARGET**: 90%+ code coverage for risk module

**MINIMUM**: 85% coverage required to pass CI

```bash
pytest triplegain/tests/unit/risk/ --cov=triplegain/src/risk --cov-fail-under=85
```

---

## 13. Error Handling Standards

### 13.1 Fail-Safe Defaults

**PRINCIPLE**: When uncertain, REJECT the trade.

```python
# ✅ CORRECT: Conservative defaults
try:
    risk_metrics = calculate_complex_risk()
except Exception as e:
    logger.error(f"Risk calculation failed: {e}")
    return ValidationStatus.REJECTED  # Fail safe

# ❌ WRONG: Optimistic defaults
try:
    risk_metrics = calculate_complex_risk()
except:
    return ValidationStatus.APPROVED  # DANGEROUS!
```

### 13.2 Validation Result Structure

**STANDARD PATTERN**:

```python
@dataclass
class RiskValidation:
    status: ValidationStatus  # APPROVED, MODIFIED, REJECTED, HALTED
    proposal: TradeProposal
    modified_proposal: Optional[TradeProposal] = None

    # Detailed feedback
    rejections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    modifications: dict = field(default_factory=dict)

    # Metrics
    validation_time_ms: int = 0

    def is_approved(self) -> bool:
        return self.status in [ValidationStatus.APPROVED, ValidationStatus.MODIFIED]
```

---

## 14. Documentation Standards

### 14.1 Configuration Documentation

**MANDATORY**: Every config parameter MUST have inline comment.

```yaml
# ✅ CORRECT: Documented parameters
limits:
  # Maximum leverage allowed regardless of regime
  max_leverage: 5

  # Maximum single position as % of equity
  max_position_pct: 20

  # Maximum total portfolio exposure as % of equity
  max_total_exposure_pct: 80

# ❌ WRONG: Undocumented
limits:
  max_leverage: 5
  max_position_pct: 20
```

### 14.2 Method Documentation

**REQUIRED**: All public methods MUST have docstrings.

```python
def validate_trade(
    self,
    proposal: TradeProposal,
    risk_state: Optional[RiskState] = None,
    entry_strictness: str = "normal",
) -> RiskValidation:
    """
    Validate a trade proposal against all risk rules.

    Performs multi-layer validation including stop-loss requirements,
    confidence thresholds, position sizing, leverage limits, portfolio
    exposure, and circuit breaker checks.

    Args:
        proposal: Trade proposal to validate
        risk_state: Current risk state (uses internal state if None)
        entry_strictness: Entry strictness from regime detection
            ('relaxed', 'normal', 'strict', 'very_strict')

    Returns:
        RiskValidation with approval/rejection decision, modifications,
        warnings, and validation latency.

    Validation Steps:
        1. Stop-loss validation
        2. Confidence validation
        3. Position size validation
        4. Leverage validation
        5. Portfolio exposure validation
        6. Correlation validation
        7. Margin validation
        8. Circuit breaker check

    Performance:
        Target: <10ms validation latency
        Actual: Typically 2-5ms
    """
```

---

## 15. Patterns to Avoid

### 15.1 Anti-Patterns

**DON'T**:

1. **Hardcode Risk Limits**:
```python
# ❌ BAD
if position_size > 2000:  # What if equity changes?
    return REJECTED
```

2. **Skip Edge Case Handling**:
```python
# ❌ BAD
drawdown = (peak - current) / peak  # Division by zero!
```

3. **Network I/O in Validation**:
```python
# ❌ BAD
def validate_trade(proposal):
    db_state = await db.fetch_state()  # Too slow!
```

4. **Non-Deterministic Logic**:
```python
# ❌ BAD
if random.random() > 0.5:
    return APPROVED
```

5. **Optimistic Error Handling**:
```python
# ❌ BAD
try:
    validate_complex_rule()
except:
    pass  # Ignore errors = approve by default
```

---

## 16. Review Checklist

Before committing risk management code:

- [ ] All risk parameters in config (no hardcoded limits)
- [ ] Validation completes in <10ms (test included)
- [ ] All circuit breakers implemented and tested
- [ ] Edge cases handled (zero equity, negative equity, division by zero)
- [ ] State serialization implemented and tested
- [ ] All public methods have docstrings
- [ ] Test coverage >85%
- [ ] No network I/O in validation hot path
- [ ] No LLM dependencies in risk logic
- [ ] Fail-safe defaults (reject when uncertain)

---

## 17. Maintenance

**Version History**:
- v1.0 (2025-12-19): Initial standards from Phase 3 review

**Review Schedule**: Quarterly or after major incidents

**Change Process**: Standards changes require:
1. Architecture decision record (ADR)
2. Risk assessment
3. Test coverage verification
4. Documentation update

---

## Appendix: Lessons Learned from Phase 3 Review

### What Went Well

1. **Comprehensive test coverage** (90 tests, 100% passing)
2. **Sub-10ms latency achieved** (typically 2-5ms)
3. **Proper edge case handling** (drawdown calculation)
4. **Clean separation of concerns** (rules_engine.py is focused)

### Issues Found

1. **Missing config parameters** (max_correlated_exposure_pct)
2. **Design vs implementation mismatch** (risk_per_trade 1% vs 2%)
3. **Incomplete features** (time-based exits configured but not enforced)
4. **Test gaps** (entry strictness not tested)

### Best Practices Established

1. **Config-driven everything**: All limits must be configurable
2. **Latency testing**: Include <10ms assertion in unit tests
3. **Edge case documentation**: Document WHY edge cases are handled
4. **State persistence**: Risk state must survive crashes

---

**This is a living document. Update as patterns emerge.**
