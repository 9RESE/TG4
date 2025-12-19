# Phase 3A Risk Management Engine - Review Findings

**Review Date**: 2025-12-19
**Reviewer**: Claude Opus 4.5
**Files Reviewed**: `triplegain/src/risk/rules_engine.py` (1350 lines)
**Config Reviewed**: `config/risk.yaml` (260 lines)
**Test Coverage**: 85% (90 tests passing)
**Status**: COMPLETE - 11 findings identified

---

## Executive Summary

The Risk Management Engine is **fundamentally sound** with correct implementation of critical safety features. However, several **design deviations** and **missing features** were identified that could impact trading safety. Most critically:

1. **R:R ratio not enforced** - only warns, doesn't reject
2. **Regime-based confidence thresholds missing** - could allow low-confidence trades in choppy markets
3. **Consecutive loss size reduction missing** - design specifies 50% reduction but not implemented

**Risk Assessment**: Medium - No critical security vulnerabilities, but capital protection could be stronger.

---

## Findings Summary

| ID | Priority | Category | Title | Line |
|----|----------|----------|-------|------|
| F01 | P1 | Logic | R:R ratio warning instead of rejection | 648-652 |
| F02 | P1 | Logic | Regime-based confidence thresholds not implemented | 656-716 |
| F03 | P1 | Logic | Consecutive loss size reduction missing | 941-970 |
| F04 | P2 | Logic | Modified proposal not re-validated | 580-593 |
| F05 | P2 | Logic | Trades today counter missing | N/A |
| F06 | P2 | Logic | State persistence failure not retried | 1291-1318 |
| F07 | P2 | Coverage | Test coverage gaps for critical paths | N/A |
| F08 | P3 | Performance | Redundant circuit breaker check | 569-577 |
| F09 | P3 | Design | Config divergence from design spec | N/A |
| F10 | P3 | Logic | Weekly reset requires Monday weekday check | 1232 |
| F11 | P3 | Edge Case | Correlation rejection path not fully tested | 1175-1181 |

---

## Detailed Findings

### Finding F01: Risk/Reward Ratio Warning Instead of Rejection

**File**: `triplegain/src/risk/rules_engine.py:648-652`
**Priority**: P1 - High (Financial Impact)
**Category**: Logic

#### Description

The design specification states "Minimum R:R enforced (1.5:1)" but the implementation only generates a warning when R:R is below the threshold, allowing trades with poor risk/reward ratios to execute.

#### Current Code

```python
# Validate risk/reward if take_profit set
if proposal.stop_loss and proposal.take_profit:
    rr = proposal.calculate_risk_reward()
    if rr and rr < self.min_risk_reward:
        result.warnings.append(
            f"LOW_RR: {rr:.2f} < {self.min_risk_reward} minimum"
        )
```

#### Recommended Fix

```python
# Validate risk/reward if take_profit set
if proposal.stop_loss and proposal.take_profit:
    rr = proposal.calculate_risk_reward()
    if rr and rr < self.min_risk_reward:
        result.status = ValidationStatus.REJECTED
        result.rejections.append(
            f"LOW_RR: {rr:.2f} < {self.min_risk_reward} minimum R:R required"
        )
        return False
```

#### Financial Impact

- **Scenario**: A trade with 1:1 R:R would be allowed despite 1.5:1 requirement
- **Worst case**: Over time, poor R:R trades erode edge, reducing long-term profitability
- **Probability**: Medium - depends on signal quality

---

### Finding F02: Regime-Based Confidence Thresholds Not Implemented

**File**: `triplegain/src/risk/rules_engine.py:656-716`
**Priority**: P1 - High (Design Deviation)
**Category**: Logic

#### Description

The master design specifies regime-based confidence thresholds:
- trending: 0.55
- ranging: 0.60
- volatile: 0.65
- choppy: 0.75

But the implementation uses a flat 0.60 minimum with adjustments only for consecutive losses, not regime. This allows low-confidence trades in choppy/volatile markets.

#### Current Code

```python
def _validate_confidence(
    self,
    proposal: TradeProposal,
    state: RiskState,
    result: RiskValidation,
    entry_strictness: str = "normal",
) -> bool:
    # Get minimum confidence based on consecutive losses
    min_conf = self.min_confidence  # Always 0.60 base
    for losses, threshold in sorted(
        self.confidence_thresholds.items(), reverse=True
    ):
        if state.consecutive_losses >= losses:
            min_conf = threshold
            break
    # ... entry_strictness adjustment only
```

#### Recommended Fix

```python
# Add regime-based confidence thresholds
REGIME_CONFIDENCE_THRESHOLDS = {
    'trending_bull': 0.55,
    'trending_bear': 0.55,
    'ranging': 0.60,
    'volatile_bull': 0.65,
    'volatile_bear': 0.65,
    'choppy': 0.75,
    'breakout_potential': 0.60,
    'unknown': 0.70,
}

def _validate_confidence(self, proposal, state, result, entry_strictness="normal"):
    # Start with regime-based minimum
    regime_min = REGIME_CONFIDENCE_THRESHOLDS.get(proposal.regime, 0.60)

    # Apply consecutive loss adjustments ON TOP of regime minimum
    loss_adjustment = 0.0
    if state.consecutive_losses >= 5:
        loss_adjustment = 0.20
    elif state.consecutive_losses >= 3:
        loss_adjustment = 0.10

    min_conf = min(1.0, regime_min + loss_adjustment)
    # ... rest of validation
```

#### Financial Impact

- **Scenario**: Choppy market with 0.60 confidence trade allowed when 0.75 required
- **Worst case loss**: Higher loss rate in choppy conditions, potentially -5% per bad trade
- **Probability**: High - choppy markets are common

---

### Finding F03: Consecutive Loss Size Reduction Missing

**File**: `triplegain/src/risk/rules_engine.py:941-970`
**Priority**: P1 - High (Design Deviation)
**Category**: Logic

#### Description

The design specifies "After 5 consecutive losses: 50% position size reduction" but the implementation only applies cooldown and leverage reduction, not explicit position size reduction during validation.

#### Current Code

```python
def record_trade_result(self, is_win: bool) -> None:
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
        # ... NO SIZE REDUCTION APPLIED
```

#### Recommended Fix

Add size reduction to `_validate_position_size`:

```python
def _validate_position_size(self, proposal, state, result) -> Optional[float]:
    # ... existing code ...

    # Apply consecutive loss size reduction
    if state.consecutive_losses >= 5:
        reduction = 0.50  # 50% reduction
        if modified_size:
            modified_size *= (1 - reduction)
        else:
            modified_size = proposal.size_usd * (1 - reduction)
        result.modifications['consecutive_loss_reduction'] = {
            'losses': state.consecutive_losses,
            'reduction_pct': reduction * 100,
            'original': proposal.size_usd,
            'modified': modified_size,
        }
        result.warnings.append(
            f"SIZE_REDUCED_LOSSES: {state.consecutive_losses} consecutive losses, "
            f"size reduced by {reduction*100:.0f}%"
        )

    return modified_size
```

#### Financial Impact

- **Scenario**: After 5 losses, full-size trades continue, amplifying losing streak
- **Worst case loss**: 5 more losses at full size = -10% equity
- **Probability**: Medium - losing streaks occur

---

### Finding F04: Modified Proposal Not Re-Validated

**File**: `triplegain/src/risk/rules_engine.py:580-593`
**Priority**: P2 - Medium
**Category**: Logic

#### Description

When a proposal is modified (size/leverage reduced), the modified proposal is created but not re-validated. While modifications are reductive and generally safe, edge cases could exist.

#### Current Code

```python
# Apply modifications if any
if modified_size or modified_leverage:
    result.status = ValidationStatus.MODIFIED
    result.modified_proposal = TradeProposal(
        symbol=proposal.symbol,
        # ... creates new proposal but doesn't validate it
    )
```

#### Recommended Fix

Either:
1. Add a final sanity check on the modified proposal
2. Document that modifications are always reductive and safe

#### Financial Impact

- **Scenario**: Modified proposal violates a constraint not checked during modification
- **Worst case**: Minimal - modifications reduce risk
- **Probability**: Low

---

### Finding F05: Trades Today Counter Missing

**File**: N/A (Not implemented)
**Priority**: P2 - Medium
**Category**: Logic

#### Description

The design mentions tracking `trades_today` but this counter is not implemented in `RiskState` or updated anywhere. This prevents implementing daily trade limits if desired.

#### Recommended Fix

Add to `RiskState`:

```python
@dataclass
class RiskState:
    # ... existing fields ...
    trades_today: int = 0

    # In serialization
    def to_dict(self) -> dict:
        return {
            # ... existing fields ...
            'trades_today': self.trades_today,
        }
```

Update in `record_trade_result`:
```python
def record_trade_result(self, is_win: bool) -> None:
    self._risk_state.trades_today += 1
    # ... rest of method
```

---

### Finding F06: State Persistence Failure Not Retried

**File**: `triplegain/src/risk/rules_engine.py:1291-1318`
**Priority**: P2 - Medium
**Category**: Logic

#### Description

If `persist_state` fails, the error is logged but no retry is attempted. This could lead to state inconsistency on restart if the DB write fails after a significant state change.

#### Current Code

```python
async def persist_state(self) -> bool:
    try:
        # ... persist logic ...
    except Exception as e:
        logger.error(f"Failed to persist risk state: {e}")
        return False  # No retry
```

#### Recommended Fix

```python
async def persist_state(self, max_retries: int = 3) -> bool:
    if self.db is None:
        return False

    for attempt in range(max_retries):
        try:
            state_json = json.dumps(self._risk_state.to_dict())
            await self.db.execute(query, state_json)
            return True
        except Exception as e:
            logger.warning(f"Persist attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff

    logger.error("Failed to persist risk state after all retries")
    return False
```

---

### Finding F07: Test Coverage Gaps for Critical Paths

**File**: N/A
**Priority**: P2 - Medium
**Category**: Coverage

#### Description

Current test coverage is 85% but several critical paths are not fully tested:

1. **TradeProposal validation edge cases** (lines 89-132)
   - Invalid symbol, side, size validation
   - Stop loss direction validation for both buy/sell

2. **Volatility spike size reduction** (lines 537-545)
   - Integration with other size modifications

3. **Correlation rejection path** (lines 1175-1181)
   - Rejection when max correlated exposure exceeded

4. **Weekly auto-reset on Monday** (lines 1232-1235)
   - Triggered on weekday check

#### Recommended Tests to Add

```python
# TradeProposal validation
def test_proposal_invalid_symbol_raises():
    with pytest.raises(TradeProposalValidationError):
        TradeProposal(symbol="", side="buy", size_usd=100, entry_price=50)

def test_proposal_negative_size_raises():
    with pytest.raises(TradeProposalValidationError):
        TradeProposal(symbol="BTC/USDT", side="buy", size_usd=-100, entry_price=50)

# Correlation rejection
def test_correlated_exposure_exceeds_limit_rejected():
    # Setup state with high correlated exposure
    # Validate trade that would exceed limit
    # Assert REJECTED status
```

---

### Finding F08: Redundant Circuit Breaker Check

**File**: `triplegain/src/risk/rules_engine.py:569-577`
**Priority**: P3 - Low
**Category**: Performance

#### Description

Circuit breakers are checked twice in `validate_trade`:
1. At the beginning (lines 477-495) using internal state
2. Near the end (lines 569-577) using external state

The second check is redundant when using internal state, but provides additional safety when external state is passed.

#### Current Code

```python
# First check (line 477)
if internal_state.trading_halted:
    # ... reject ...

# Second check (line 569)
breaker_result = self._check_circuit_breakers(state)
if breaker_result['halt_trading']:
    # ... reject ...
```

#### Recommendation

Document the purpose of the dual check or remove if truly redundant. The current design appears intentional for defense-in-depth.

---

### Finding F09: Config Divergence from Design Spec

**File**: `config/risk.yaml`
**Priority**: P3 - Low
**Category**: Design

#### Description

Several config values differ from the master design:

| Parameter | Design | Config | Impact |
|-----------|--------|--------|--------|
| Max exposure | 60% (checklist) | 80% | Higher risk |
| Ranging leverage | 3x (table) | 2x | More conservative |
| Correlation limit | N/A | 40% | Implemented but not in design |

These are generally more conservative than design, which is acceptable for safety.

#### Recommendation

Update design document or config to match. Conservative deviations are acceptable.

---

### Finding F10: Weekly Reset Requires Monday Weekday Check

**File**: `triplegain/src/risk/rules_engine.py:1232`
**Priority**: P3 - Low
**Category**: Logic

#### Description

The weekly reset logic checks for Monday but the condition could miss the reset if validation doesn't occur exactly on Monday.

#### Current Code

```python
if current_week != last_week and now.weekday() == 0:  # Monday = 0
    self.reset_weekly()
```

#### Issue

If no trade validation occurs on Monday, the reset won't trigger until the next Monday.

#### Recommended Fix

```python
if current_week != last_week:  # Any day of new week
    self.reset_weekly()
```

---

### Finding F11: Correlation Rejection Path Not Fully Tested

**File**: `triplegain/src/risk/rules_engine.py:1175-1181`
**Priority**: P3 - Low
**Category**: Edge Case

#### Description

The correlation validation can reject trades when `total_correlated_exposure > max_correlated_exposure_pct`, but this rejection path is not explicitly tested.

#### Missing Test

```python
def test_correlated_exposure_exceeds_max_rejected(risk_engine, healthy_risk_state):
    """Trade causing correlated exposure over limit should be rejected."""
    # Set max_correlated_exposure_pct to 40%
    risk_engine.max_correlated_exposure_pct = 40

    # Already holding BTC/USDT at 30% exposure
    healthy_risk_state.open_position_symbols = ['BTC/USDT']
    healthy_risk_state.position_exposures = {'BTC/USDT': 30.0}

    # Propose XRP/USDT trade (correlated with BTC)
    # With correlation 0.75, 30% * 0.75 = 22.5% correlated
    # Plus new 20% = 42.5% > 40% limit
    proposal = TradeProposal(
        symbol="XRP/USDT",
        side="buy",
        size_usd=2000.0,  # 20% of $10k equity
        entry_price=0.50,
        stop_loss=0.49,
        leverage=1,
        confidence=0.75,
    )

    result = risk_engine.validate_trade(proposal, healthy_risk_state)

    assert result.status == ValidationStatus.REJECTED
    assert any("CORRELATED_EXPOSURE" in r for r in result.rejections)
```

---

## Verification Checklist Status

### 1. Trade Validation Logic

| Check | Status | Notes |
|-------|--------|-------|
| Risk per trade calculation | PASS | Correct formula |
| Max position size enforced (20%) | PASS | Checked at validation |
| Position size adjusted for regime | PARTIAL | Only in calculate_position_size, not validation |
| Size reduction on consecutive losses | FAIL | Not implemented (F03) |
| Max leverage enforced (5x) | PASS | Hard limit enforced |
| Leverage reduced per regime | PASS | Correct regime limits |
| Leverage validated before execution | PASS | |
| Minimum confidence checked | PARTIAL | Not regime-based (F02) |
| Below-threshold trades rejected | PASS | |
| R:R calculation correct | PASS | Direction-aware |
| Minimum R:R enforced (1.5:1) | FAIL | Warning only (F01) |
| Total exposure calculated correctly | PASS | Includes leverage |
| Max total exposure enforced | PASS | 80% limit |
| Correlated positions considered | PASS | |

### 2. Circuit Breakers

| Check | Status | Notes |
|-------|--------|-------|
| Daily loss limit triggered at 5% | PASS | |
| Weekly loss limit triggered at 10% | PASS | |
| Max drawdown halt at 20% | PASS | |
| Consecutive losses cooldown | PASS | But no size reduction |
| Volatility spike 50% reduction | PASS | Correctly implemented |
| State transitions correct | PASS | |
| Multiple breakers handled | PASS | |
| Manual reset mechanism | PASS | Requires admin override for drawdown |

### 3. Mathematical Accuracy

| Check | Status | Notes |
|-------|--------|-------|
| Position size formula | PASS | Correct |
| Division by zero protected | PASS | |
| Drawdown calculation | PASS | Handles edge cases |
| R:R direction-aware | PASS | |

### 4. Security

| Check | Status | Notes |
|-------|--------|-------|
| No external input in risk calcs | PASS | Proposals from agents |
| Parameters from config only | PASS | |
| No SQL injection | PASS | Parameterized queries |
| Circuit breakers use internal state | PASS | Cannot be bypassed |

### 5. Performance

| Check | Status | Notes |
|-------|--------|-------|
| <10ms validation | PASS | Tested at ~1-2ms |
| No blocking I/O in validation | PASS | |
| State cached in memory | PASS | |

---

## Summary by Priority

### P1 - Must Fix Before Production (3 findings)

1. **F01**: R:R ratio enforcement - Change from warning to rejection
2. **F02**: Regime-based confidence - Add regime thresholds per design
3. **F03**: Consecutive loss size reduction - Implement 50% reduction

### P2 - Should Fix (4 findings)

1. **F04**: Re-validate modified proposals or document safety
2. **F05**: Add trades_today counter
3. **F06**: Add retry logic to persist_state
4. **F07**: Add missing test coverage

### P3 - Nice to Have (4 findings)

1. **F08**: Document or remove redundant circuit breaker check
2. **F09**: Align config with design document
3. **F10**: Fix weekly reset timing logic
4. **F11**: Add correlation rejection test

---

## Positive Findings

The review also identified several well-implemented aspects:

1. **Thread Safety**: Proper locking for circuit breaker state
2. **Defense in Depth**: Multiple validation layers
3. **Leverage Controls**: Regime + drawdown + consecutive loss adjustments
4. **Volatility Handling**: Correct spike detection and size reduction
5. **State Persistence**: Proper serialization with database support
6. **Error Handling**: Graceful degradation on failures
7. **Audit Trail**: Modifications and warnings tracked in result

---

## Recommendations

### Immediate (Before Paper Trading)

1. Fix F01 (R:R rejection) - Critical for profitability
2. Fix F02 (regime confidence) - Critical for choppy market safety
3. Fix F03 (loss size reduction) - Critical for drawdown protection

### Before Live Trading

1. Achieve 95%+ test coverage on risk module
2. Add integration tests with simulated market conditions
3. Add monitoring/alerting for circuit breaker triggers

### Technical Debt

1. Consider separating validation logic into smaller, testable functions
2. Add metrics collection for validation timing and rejection rates
3. Document all design decisions in ADRs

---

*Review completed: 2025-12-19*
*Next phase: 3B (Order Manager & Position Tracker)*
