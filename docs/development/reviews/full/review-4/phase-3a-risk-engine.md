# Review Phase 3A: Risk Management Engine

**Status**: Ready for Review
**Estimated Context**: ~1,500 tokens (code) + review
**Priority**: CRITICAL - Protects capital
**Output**: `findings/phase-3a-findings.md`
**DO NOT IMPLEMENT FIXES**

---

## Files to Review

| File | Lines | Purpose |
|------|-------|---------|
| `triplegain/src/risk/rules_engine.py` | ~800 | Rule-based risk validation |

**Total**: ~800 lines (dedicated phase due to criticality)

---

## Pre-Review: Load Files

```bash
# Read this file before starting review
cat triplegain/src/risk/rules_engine.py

# Also review the config
cat config/risk.yaml
```

---

## Why Separate Review Phase?

The Risk Management Engine is the **most critical** component because:

1. **Capital Protection**: Only barrier between bad decisions and real losses
2. **No LLM Dependency**: Must be 100% deterministic and auditable
3. **Low Latency**: Must execute in <10ms
4. **High Reliability**: Zero tolerance for bugs

A bug here could result in:
- Unauthorized leverage exposure
- Exceeding loss limits
- Bypassing circuit breakers
- Financial ruin

---

## Review Checklist

### 1. Trade Validation Logic

#### Position Sizing
- [ ] Risk per trade calculation correct:
  ```
  risk_amount = equity * risk_per_trade_pct
  max_size = risk_amount / stop_loss_distance
  ```
- [ ] Max position size enforced (20% of equity)
- [ ] Position size adjusted for regime
- [ ] Size reduction on consecutive losses

#### Leverage Control
- [ ] Max leverage enforced (5x absolute maximum)
- [ ] Leverage reduced per regime:
  - [ ] trending: 5x
  - [ ] ranging: 3x
  - [ ] volatile: 2x
  - [ ] choppy: 1x
- [ ] Leverage validated before execution

#### Confidence Thresholds
- [ ] Minimum confidence checked:
  - [ ] trending: 0.55
  - [ ] ranging: 0.60
  - [ ] volatile: 0.65
  - [ ] choppy: 0.75
- [ ] Below-threshold trades rejected

#### Risk-Reward Ratio
- [ ] R:R calculation correct:
  ```
  rr_ratio = (take_profit - entry) / (entry - stop_loss)
  ```
- [ ] Minimum R:R enforced (1.5:1)
- [ ] R:R validated for both long and short

#### Exposure Limits
- [ ] Total exposure calculated correctly
- [ ] Max total exposure enforced (60%)
- [ ] Correlated positions considered
- [ ] New position would exceed limit → rejected

---

### 2. Circuit Breakers

#### Daily Loss Limit
- [ ] Daily P&L tracked correctly
- [ ] Threshold: -5% daily loss
- [ ] Action: Halt all trading
- [ ] Cooldown: Until next trading day
- [ ] Time zone handling correct (UTC?)

#### Weekly Loss Limit
- [ ] Weekly P&L tracked correctly
- [ ] Threshold: -10% weekly loss
- [ ] Action: Halt all trading
- [ ] Cooldown: Until next week
- [ ] Week boundary definition correct

#### Maximum Drawdown
- [ ] Peak equity tracked
- [ ] Drawdown calculated: (peak - current) / peak
- [ ] Threshold: -20% drawdown
- [ ] Action: Halt trading
- [ ] Requires manual reset

#### Consecutive Losses
- [ ] Loss counter tracked correctly
- [ ] Threshold: 5 consecutive losses
- [ ] Action: 50% position size reduction
- [ ] Cooldown: 3 hours
- [ ] Counter reset on win

#### Volatility Spike
- [ ] ATR vs historical average comparison
- [ ] Threshold: ATR > 3x average
- [ ] Action: 50% position size reduction
- [ ] Reset: When volatility normalizes

---

### 3. Circuit Breaker State Machine

```
NORMAL → (trigger condition) → ACTIVE → (cooldown expires) → NORMAL
                                  ↓
                              (manual reset required)
                                  ↓
                               LOCKED
```

- [ ] State transitions correct
- [ ] Multiple active breakers handled
- [ ] Cooldown timing accurate
- [ ] Manual reset mechanism works
- [ ] State persisted to database

---

### 4. Validation Result Handling

#### Status Types
- [ ] APPROVED: Trade passes all checks unchanged
- [ ] MODIFIED: Trade passes with adjustments
- [ ] REJECTED: Trade fails critical checks

#### Modification Handling
- [ ] Size modifications applied correctly
- [ ] Leverage modifications applied correctly
- [ ] Modified trade re-validated
- [ ] Modifications logged

#### Rejection Handling
- [ ] Clear rejection reasons provided
- [ ] All failed checks listed
- [ ] Rejection logged
- [ ] No partial execution on rejection

---

### 5. Risk State Management

#### State Tracking
- [ ] daily_loss_pct updated after each trade
- [ ] weekly_loss_pct updated correctly
- [ ] max_drawdown_pct tracked
- [ ] peak_equity_usd updated on new highs
- [ ] consecutive_losses counter accurate
- [ ] trades_today counter accurate

#### State Persistence
- [ ] State saved to risk_state table
- [ ] State recovered on restart
- [ ] State consistency maintained
- [ ] No state corruption on crash

#### Time-Based Resets
- [ ] Daily reset timing correct
- [ ] Weekly reset timing correct
- [ ] Time zone handling consistent

---

## Critical Code Paths

### Path 1: Trade Validation
```python
validate_trade(proposal, portfolio, regime) →
    check_circuit_breakers() →
    check_confidence_threshold() →
    check_position_size() →
    check_leverage() →
    check_rr_ratio() →
    check_exposure_limit() →
    check_correlation() →
    return ValidationResult
```

- [ ] All checks execute in order
- [ ] Early return on rejection
- [ ] Modifications accumulate correctly

### Path 2: Post-Trade Update
```python
update_risk_state(trade_result) →
    update_pnl() →
    update_peak_equity() →
    update_drawdown() →
    update_consecutive_losses() →
    check_circuit_breakers() →
    persist_state()
```

- [ ] Called after every trade close
- [ ] Both wins and losses handled
- [ ] Circuit breakers triggered if needed

---

## Edge Case Analysis

| Edge Case | Expected Behavior | Test? |
|-----------|-------------------|-------|
| Proposal size = 0 | Reject | [ ] |
| Proposal size negative | Reject | [ ] |
| Stop loss beyond entry | Reject | [ ] |
| Take profit before entry | Reject | [ ] |
| Leverage = 0 | Reject or default to 1 | [ ] |
| Confidence > 1.0 | Clamp or reject | [ ] |
| Equity = 0 | Reject all trades | [ ] |
| First trade of day | daily_loss_pct = 0 | [ ] |
| Trade during halt | Reject | [ ] |
| Multiple circuit breakers | All respected | [ ] |

---

## Mathematical Accuracy

### Position Size Calculation
```python
# Correct formula
risk_amount = equity * risk_per_trade_pct  # e.g., $10,000 * 0.01 = $100
stop_loss_distance = abs(entry_price - stop_loss) / entry_price  # e.g., 2%
max_size_usd = risk_amount / stop_loss_distance  # $100 / 0.02 = $5,000
```

- [ ] Division by zero protected
- [ ] Decimal arithmetic used
- [ ] Result rounded appropriately

### Drawdown Calculation
```python
# Correct formula
drawdown_pct = (peak_equity - current_equity) / peak_equity * 100
```

- [ ] Peak updated correctly (max of current and peak)
- [ ] Drawdown never negative
- [ ] Decimal precision maintained

### R:R Calculation
```python
# For long
risk = entry - stop_loss
reward = take_profit - entry
rr_ratio = reward / risk

# For short
risk = stop_loss - entry
reward = entry - take_profit
rr_ratio = reward / risk
```

- [ ] Direction-aware calculation
- [ ] Division by zero protected

---

## Security Review

- [ ] No external input in risk calculations
- [ ] Parameters from validated config only
- [ ] No SQL injection in state queries
- [ ] No possibility of bypassing checks
- [ ] Rate limiting not applicable (rule-based)
- [ ] Audit log for all decisions

---

## Performance Review

- [ ] All checks complete in <10ms
- [ ] No blocking I/O in validation path
- [ ] Database queries minimized
- [ ] State cached in memory
- [ ] No unnecessary calculations

---

## Test Coverage Check

```bash
pytest --cov=triplegain/src/risk \
       --cov-report=term-missing \
       triplegain/tests/unit/risk/
```

**Required test scenarios:**

1. [ ] Trade below all limits → APPROVED
2. [ ] Trade exceeds position size → MODIFIED
3. [ ] Trade exceeds leverage → MODIFIED
4. [ ] Trade fails R:R ratio → REJECTED
5. [ ] Trade during circuit breaker → REJECTED
6. [ ] Daily loss limit triggered
7. [ ] Weekly loss limit triggered
8. [ ] Max drawdown triggered
9. [ ] Consecutive losses trigger
10. [ ] Volatility spike trigger
11. [ ] Multiple modifications in one trade
12. [ ] Circuit breaker cooldown expiry
13. [ ] State persistence/recovery
14. [ ] Time-based resets

---

## Design Conformance

### Implementation Plan 2.3 (Risk Management)
- [ ] All circuit breakers implemented
- [ ] All thresholds match spec
- [ ] Cooldown periods match spec

### Master Design (03-risk-management-rules-engine.md)
- [ ] Position sizing formula correct
- [ ] Regime-based adjustments correct
- [ ] Validation flow matches design

---

## Findings Template

```markdown
## Finding: [Title]

**File**: `triplegain/src/risk/rules_engine.py:123`
**Priority**: P0/P1/P2/P3 (Most findings here should be P0 or P1)
**Category**: Security/Logic/Performance

### Description
[What was found - be specific about the risk]

### Current Code
```python
# current implementation
```

### Recommended Fix
```python
# recommended fix
```

### Financial Impact
[Specific scenario where this could cause loss]
- Worst case loss: $X or X% of portfolio
- Probability: Low/Medium/High
```

---

## Review Completion

After completing this phase:

1. [ ] All validation logic reviewed line-by-line
2. [ ] Circuit breaker logic verified
3. [ ] Mathematical formulas verified
4. [ ] Edge cases analyzed
5. [ ] Performance verified (<10ms)
6. [ ] Test coverage verified (aim for 100%)
7. [ ] Findings documented (priority P0/P1)
8. [ ] Ready for Phase 3B

---

## Sign-Off Requirement

Due to the critical nature of this component, consider:

1. **Independent verification** of all findings
2. **Mathematical review** of position sizing formulas
3. **Stress testing** with edge case inputs
4. **Paper trading validation** before live deployment

---

*Phase 3A Review Plan v1.0 - CRITICAL COMPONENT*
