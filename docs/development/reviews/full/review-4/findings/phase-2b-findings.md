# Phase 2B Review Findings: Core Agents

**Review Date**: 2025-12-19
**Reviewer**: Claude Opus 4.5
**Status**: Complete
**Files Reviewed**: 5 (~2,500 lines)

---

## Executive Summary

The Core Agents implementation is **solid overall** with good adherence to the design specification. Test coverage is strong at 88% (188 tests passing). However, several issues were identified that should be addressed before production deployment:

| Priority | Count | Description |
|----------|-------|-------------|
| P0 (Critical) | 0 | None - No immediate blockers |
| P1 (High) | 2 | Trading safety issues |
| P2 (Medium) | 5 | Logic/design gaps |
| P3 (Low) | 5 | Quality improvements |

**Recommendation**: Address P1 issues before paper trading; P2 issues before live trading.

---

## Test Coverage Summary

```
Name                                           Stmts   Miss  Cover
------------------------------------------------------------------------------------------
triplegain/src/agents/__init__.py                  5      0   100%
triplegain/src/agents/base_agent.py              132      4    96%
triplegain/src/agents/portfolio_rebalance.py     246     51    76%
triplegain/src/agents/regime_detection.py        183     11    94%
triplegain/src/agents/technical_analysis.py      150      9    93%
triplegain/src/agents/trading_decision.py        308     30    89%
------------------------------------------------------------------------------------------
TOTAL                                           1024    105    88%
```

All 188 agent tests passing.

---

## Findings

### Finding P1-01: No Minimum Quorum for Consensus

**File**: `triplegain/src/agents/trading_decision.py:556-664`
**Priority**: P1 (High)
**Category**: Logic/Safety

#### Description

The consensus calculation proceeds even if only 1-2 models respond (due to timeouts or failures). With 6 models configured, a consensus based on 2 models is statistically unreliable and could lead to poor trading decisions.

#### Current Code

```python
def _calculate_consensus(self, decisions: list[ModelDecision]) -> ConsensusResult:
    valid_decisions = [d for d in decisions if d.is_valid()]

    if not valid_decisions:
        return ConsensusResult(...)  # Returns HOLD with 0 confidence

    # No check for minimum number of valid decisions
    # Proceeds with consensus even with 1-2 models
```

#### Recommended Fix

```python
def _calculate_consensus(self, decisions: list[ModelDecision]) -> ConsensusResult:
    valid_decisions = [d for d in decisions if d.is_valid()]

    # Require minimum quorum (e.g., 4 of 6 models)
    MIN_QUORUM = self.config.get('min_quorum', 4)

    if len(valid_decisions) < MIN_QUORUM:
        logger.warning(
            f"Insufficient quorum: {len(valid_decisions)}/{len(decisions)} valid, "
            f"need {MIN_QUORUM}. Forcing HOLD."
        )
        return ConsensusResult(
            final_action='HOLD',
            final_confidence=0.0,
            consensus_strength=0.0,
            votes={},
            total_models=len(decisions),
            agreeing_models=0,
            agreement_type='insufficient_quorum',
            confidence_boost=0.0,
            model_decisions=decisions,
        )

    # Continue with consensus calculation...
```

#### Impact

Without fix: Could execute trades based on 1-2 model votes when others timeout, leading to unreliable decisions.

---

### Finding P1-02: Missing Regime Flapping Prevention

**File**: `triplegain/src/agents/regime_detection.py:304-310`
**Priority**: P1 (High)
**Category**: Logic/Safety

#### Description

The regime detection agent tracks regime changes but has no hysteresis or smoothing to prevent rapid oscillation between regimes. In volatile markets, this could cause:
- Rapid switching between choppy/ranging/volatile
- Whiplash in position sizing and entry strictness
- Excessive parameter adjustments

#### Current Code

```python
# Track regime changes - immediate switch on any change
if current_regime != self._previous_regime:
    self._regime_start_time = datetime.now(timezone.utc)
    self._periods_in_current_regime = 0
    self._previous_regime = current_regime
else:
    self._periods_in_current_regime += 1
```

#### Recommended Fix

```python
def _should_change_regime(self, new_regime: str, confidence: float) -> bool:
    """Apply hysteresis to prevent regime flapping."""
    if self._previous_regime is None:
        return True  # First detection

    if new_regime == self._previous_regime:
        return False  # No change

    # Require higher confidence for regime changes
    min_change_confidence = self.config.get('min_regime_change_confidence', 0.7)
    if confidence < min_change_confidence:
        logger.debug(
            f"Regime change suppressed: {self._previous_regime} -> {new_regime} "
            f"(confidence {confidence:.2f} < {min_change_confidence})"
        )
        return False

    # Require minimum stability period before switching back
    if self._pending_regime == new_regime:
        self._pending_regime_count += 1
        if self._pending_regime_count >= self.config.get('regime_change_periods', 2):
            return True  # Confirmed change
    else:
        self._pending_regime = new_regime
        self._pending_regime_count = 1

    return False
```

#### Impact

Without fix: Rapid regime oscillation could cause inconsistent risk parameters and poor trading performance.

---

### Finding P2-01: SQL Injection Risk in Base Agent

**File**: `triplegain/src/agents/base_agent.py:253-255`
**Priority**: P2 (Medium)
**Category**: Security

#### Description

The `get_latest_output` method uses Python string formatting to inject `max_age_seconds` into the SQL query, which is a potential SQL injection vector.

#### Current Code

```python
query = """
    SELECT output_data, timestamp
    FROM agent_outputs
    WHERE agent_name = $1 AND symbol = $2
    AND timestamp > NOW() - INTERVAL '%s seconds'
    ORDER BY timestamp DESC
    LIMIT 1
""" % max_age_seconds  # String interpolation!
```

#### Recommended Fix

```python
# Use a fixed interval with parameterized comparison
query = """
    SELECT output_data, timestamp
    FROM agent_outputs
    WHERE agent_name = $1 AND symbol = $2
    AND timestamp > NOW() - make_interval(secs => $3)
    ORDER BY timestamp DESC
    LIMIT 1
"""
row = await self.db.fetchrow(query, self.agent_name, symbol, max_age_seconds)
```

#### Impact

While `max_age_seconds` is typically controlled internally, following parameterized query best practices prevents future vulnerabilities if the function signature changes.

---

### Finding P2-02: Hodl Bag Profit Allocation Not Implemented

**File**: `triplegain/src/agents/portfolio_rebalance.py`
**Priority**: P2 (Medium)
**Category**: Logic (Missing Feature)

#### Description

The design specification states that 10% of realized profits should be allocated to hodl bags. This functionality is not implemented in the portfolio rebalance agent.

The agent correctly:
- Excludes hodl bags from rebalancing calculations (lines 340-362)
- Fetches hodl bag amounts from DB or config (lines 644-670)

But missing:
- Logic to calculate 10% of realized profits
- Logic to add to hodl bag amounts
- Integration with trade execution to trigger profit allocation

#### Recommended Fix

Add a method to handle profit allocation:

```python
async def allocate_profits_to_hodl(
    self,
    trade_result: dict,
) -> dict[str, Decimal]:
    """
    Allocate 10% of realized profits to hodl bags.

    Args:
        trade_result: Completed trade with profit/loss

    Returns:
        dict of asset -> amount added to hodl
    """
    if not self.hodl_enabled:
        return {}

    pnl = Decimal(str(trade_result.get('realized_pnl_usd', 0)))
    if pnl <= 0:
        return {}  # No profit to allocate

    allocation_pct = Decimal(str(self.config.get('hodl_profit_allocation_pct', 10)))
    hodl_amount = (pnl * allocation_pct / 100)

    # Distribute proportionally or to specific asset
    # Implementation depends on design preference
    ...
```

#### Impact

Without fix: Hodl bag feature is incomplete - profits won't accumulate as designed.

---

### Finding P2-03: Target Allocation Validation Too Lenient

**File**: `triplegain/src/agents/portfolio_rebalance.py:176-182`
**Priority**: P2 (Medium)
**Category**: Logic/Configuration

#### Description

The agent only logs a warning if target allocations don't sum to 100%, but continues operation. This could lead to unexpected rebalancing behavior.

#### Current Code

```python
total_allocation = self.target_btc_pct + self.target_xrp_pct + self.target_usdt_pct
if abs(total_allocation - 100) > Decimal('0.1'):
    logger.warning(
        f"Target allocations sum to {float(total_allocation):.2f}%, not 100%. "
        # ... continues execution
    )
```

#### Recommended Fix

```python
total_allocation = self.target_btc_pct + self.target_xrp_pct + self.target_usdt_pct
if abs(total_allocation - 100) > Decimal('0.1'):
    raise ValueError(
        f"Target allocations must sum to 100% (+/- 0.1%), "
        f"got {float(total_allocation):.2f}%: "
        f"BTC={float(self.target_btc_pct):.2f}%, "
        f"XRP={float(self.target_xrp_pct):.2f}%, "
        f"USDT={float(self.target_usdt_pct):.2f}%"
    )
```

#### Impact

Without fix: Misconfigured allocations could cause unexpected portfolio drift.

---

### Finding P2-04: Regime State Not Persistent

**File**: `triplegain/src/agents/regime_detection.py:234-236`
**Priority**: P2 (Medium)
**Category**: Logic/State Management

#### Description

The regime tracking state (`_previous_regime`, `_regime_start_time`, `_periods_in_current_regime`) is stored in instance variables. If the agent restarts, this state is lost, potentially causing:
- Incorrect regime duration reporting
- Sudden regime transitions on restart
- Inaccurate transition probability tracking

#### Current Code

```python
def __init__(...):
    ...
    # Regime tracking - lost on restart
    self._previous_regime: Optional[str] = None
    self._regime_start_time: Optional[datetime] = None
    self._periods_in_current_regime = 0
```

#### Recommended Fix

```python
async def _load_regime_state(self) -> None:
    """Load regime state from database on startup."""
    if self.db is None:
        return

    try:
        query = """
            SELECT regime, timestamp
            FROM agent_outputs
            WHERE agent_name = 'regime_detection'
            ORDER BY timestamp DESC LIMIT 1
        """
        row = await self.db.fetchrow(query)
        if row:
            self._previous_regime = row['regime']
            self._regime_start_time = row['timestamp']
    except Exception as e:
        logger.warning(f"Failed to load regime state: {e}")
```

#### Impact

Without fix: Restart during operation could cause inconsistent regime reporting.

---

### Finding P2-05: Output Schema Action Mismatch

**File**: `triplegain/src/agents/trading_decision.py:39`
**Priority**: P2 (Medium)
**Category**: Schema/Design Conformance

#### Description

The implementation uses `CLOSE_LONG` and `CLOSE_SHORT` actions, but the design specification (implementation plan section 2.4) shows only `CLOSE` as the action type.

#### Current Code

```python
VALID_ACTIONS = ["BUY", "SELL", "HOLD", "CLOSE_LONG", "CLOSE_SHORT"]
```

#### Design Specification

```json
"action": {
  "type": "string",
  "enum": ["BUY", "SELL", "HOLD", "CLOSE"]
}
```

#### Recommended Fix

Either:
1. Update the design specification to include `CLOSE_LONG`/`CLOSE_SHORT` (preferred for clarity)
2. Or consolidate to single `CLOSE` action with side specified in parameters

#### Impact

Downstream consumers expecting the design schema may fail on `CLOSE_LONG`/`CLOSE_SHORT` actions.

---

### Finding P3-01: Missing Prompt Hash for TA Agent

**File**: `triplegain/src/agents/technical_analysis.py`
**Priority**: P3 (Low)
**Category**: Quality/Observability

#### Description

The review checklist specifies that outputs should include a `prompt_hash` for caching purposes. This is not implemented.

#### Recommended Fix

Add prompt hash to output metadata:

```python
import hashlib

def _compute_prompt_hash(self, prompt) -> str:
    """Compute hash of prompt for caching/deduplication."""
    content = f"{prompt.system_prompt}|{prompt.user_message}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

---

### Finding P3-02: Float vs Decimal Inconsistency

**File**: Multiple agent files
**Priority**: P3 (Low)
**Category**: Quality/Precision

#### Description

`portfolio_rebalance.py` correctly uses `Decimal` for financial calculations, but other agents (`trading_decision.py`, `technical_analysis.py`) use `float` for confidence scores and prices.

While confidence scores (0-1) and latency values are fine as floats, entry prices, stop losses, and take profits should use `Decimal` for precision.

#### Current Code (trading_decision.py)

```python
@dataclass
class ModelDecision:
    entry_price: Optional[float] = None  # Should be Decimal
    stop_loss: Optional[float] = None    # Should be Decimal
    take_profit: Optional[float] = None  # Should be Decimal
```

#### Recommended Fix

Use `Decimal` for all price values to prevent floating-point precision issues in financial calculations.

---

### Finding P3-03: Cache TTL Config Key Inconsistency

**File**: `triplegain/src/agents/base_agent.py:144`
**Priority**: P3 (Low)
**Category**: Configuration

#### Description

The cache TTL configuration uses `cache_ttl_seconds` but the implementation plan (config/agents.yaml) shows different key patterns for different agents.

#### Recommended Fix

Standardize configuration key naming across all agents.

---

### Finding P3-04: Missing Validation for Stop Loss Bounds

**File**: `triplegain/src/agents/trading_decision.py:822-845`
**Priority**: P3 (Low)
**Category**: Logic/Validation

#### Description

The decision query mentions `stop_loss_pct` should be 1.0-5.0 per the design, but the normalization code doesn't enforce this bound.

#### Current Code

```python
# No clamping of stop_loss in _normalize_decision()
```

#### Recommended Fix

```python
# In _normalize_decision:
if 'stop_loss_pct' in parsed:
    parsed['stop_loss_pct'] = max(1.0, min(5.0, float(parsed['stop_loss_pct'])))
```

---

### Finding P3-05: Hardcoded Model Names

**File**: `triplegain/src/agents/trading_decision.py:219-226`
**Priority**: P3 (Low)
**Category**: Quality/Maintainability

#### Description

The default model configuration is hardcoded in the `__init__` method. While configurable via the config dict, the defaults reference specific model versions that may become outdated.

#### Recommended Fix

Move default model configuration to a separate constants file or configuration file, making updates easier.

---

## Checklist Verification

### Base Agent (`base_agent.py`) - PASS

| Check | Status | Notes |
|-------|--------|-------|
| Abstract base class properly structured | PASS | Uses ABC correctly |
| Required methods defined | PASS | process, get_output_schema, store_output |
| Type hints complete | PASS | With TYPE_CHECKING imports |
| Docstrings comprehensive | PASS | All public methods documented |
| Output storage implemented | PASS | To agent_outputs table |
| Latency tracking | PASS | In _call_llm |
| Token/cost tracking | PASS | Accumulates in stats |
| Error handling hooks | PASS | Try/except in key methods |
| Easy to add new agents | PASS | Just extend BaseAgent |
| Configuration injection clean | PASS | Via __init__ config dict |
| LLM client injection clean | PASS | Passed in constructor |

### Technical Analysis Agent - PASS (with minor issues)

| Check | Status | Notes |
|-------|--------|-------|
| Output matches JSON schema | PASS | All fields present |
| Prompt correctly requests JSON | PASS | _get_analysis_query() |
| Confidence bounds enforced | PASS | 0.0-1.0 clamped |
| Trend strength bounds enforced | PASS | 0.0-1.0 clamped |
| Momentum score bounds enforced | PASS | -1.0 to 1.0 clamped |
| Uses pre-computed indicators | PASS | From snapshot.indicators |
| Multi-timeframe data | PASS | Via prompt_builder |
| LLM timeout handled | PASS | Retry mechanism |
| Invalid JSON handled | PASS | Fallback to indicators |
| Prompt hash recorded | FAIL | Not implemented (P3-01) |

### Regime Detection Agent - PARTIAL

| Check | Status | Notes |
|-------|--------|-------|
| All 7 regime types supported | PASS | VALID_REGIMES list |
| Output matches JSON schema | PASS | All fields present |
| Multipliers in valid ranges | PASS | Clamped in normalize |
| Uses TA output if provided | PASS | Lines 259-267 |
| ADX used for trend strength | PASS | In indicators fallback |
| Previous regime considered | PASS | _previous_regime tracking |
| Transition smoothing | FAIL | No hysteresis (P1-02) |
| Regime duration tracking | PASS | _periods_in_current_regime |

### Trading Decision Agent - PARTIAL

| Check | Status | Notes |
|-------|--------|-------|
| All 6 models invoked | PASS | Configurable via models dict |
| asyncio.gather used | PASS | Uses asyncio.wait |
| Timeout handling per model | PASS | timeout_seconds config |
| Failed model handling | PASS | Graceful with error tracking |
| Vote counting correct | PASS | Verified logic |
| Unanimous boost (+0.15) | PASS | Line 600 |
| Strong majority boost (+0.10) | PASS | Line 603 |
| Majority boost (+0.05) | PASS | Line 606 |
| Split forces HOLD | PASS | Line 609-612 |
| Minimum quorum check | FAIL | Not implemented (P1-01) |
| Model comparisons stored | PASS | _store_model_comparisons() |

### Portfolio Rebalance Agent - PARTIAL

| Check | Status | Notes |
|-------|--------|-------|
| Balances fetched correctly | PASS | _get_balances() |
| Prices converted to USD | PASS | _get_current_prices() |
| Percentages calculated | PASS | Lines 375-384 |
| Hodl bags excluded | PASS | Lines 340-362 |
| Threshold check correct | PASS | max_deviation_pct |
| Target allocation correct | PASS | 33.33/33.33/33.34 |
| Trade amounts correct | PASS | _calculate_rebalance_trades |
| Minimum trade size enforced | PASS | min_trade_usd |
| Sell first priority | PASS | Priority 1 for sell |
| Hodl bag profit allocation | FAIL | Not implemented (P2-02) |

---

## Critical Questions Answered

### 1. Consensus Safety: What if only 2 models respond?

**ISSUE FOUND**: No minimum quorum check. Consensus proceeds with any number of valid responses >= 1. This is unsafe for trading decisions.

**Recommendation**: Add minimum quorum requirement (suggest 4/6).

### 2. Confidence Inflation: Could confidence boost exceed 1.0?

**SAFE**: Line 311 correctly uses `min(1.0, consensus.final_confidence + consensus.confidence_boost)`.

### 3. Decimal Precision: Are all financial calculations using Decimal?

**PARTIAL**: `portfolio_rebalance.py` uses Decimal correctly. Other agents use float for prices which could cause precision issues in edge cases.

### 4. Race Conditions: Shared mutable state issues?

**SAFE**: Uses `asyncio.Lock` for cache operations. Parallel model execution is independent.

### 5. Output Validation: What if LLM returns invalid JSON?

**SAFE**: All agents have fallback parsing:
- Try regex extraction of JSON from response
- Try parsing entire response as JSON
- Fall back to indicator-based heuristics
- Return conservative fallback output with low confidence

### 6. Regime Flapping: How is rapid regime switching prevented?

**NOT PREVENTED**: No hysteresis or smoothing mechanism. Regime can change every 5-minute cycle.

---

## Design Conformance Summary

| Spec Section | Implementation | Conformance |
|--------------|----------------|-------------|
| 2.1 TA Agent | technical_analysis.py | CONFORMS |
| 2.2 Regime Agent | regime_detection.py | MOSTLY CONFORMS (missing hysteresis) |
| 2.3 Risk Engine | Not in this review scope | N/A |
| 2.4 Trading Decision | trading_decision.py | MOSTLY CONFORMS (action enum differs) |
| Portfolio Rebalance | portfolio_rebalance.py | MOSTLY CONFORMS (hodl profit missing) |

---

## Recommendations

### Before Paper Trading (P1)
1. **P1-01**: Add minimum quorum (4/6) for trading consensus
2. **P1-02**: Implement regime change hysteresis

### Before Live Trading (P2)
1. **P2-01**: Fix SQL injection in base agent
2. **P2-02**: Implement hodl bag profit allocation
3. **P2-03**: Make allocation validation stricter
4. **P2-04**: Persist regime state across restarts
5. **P2-05**: Reconcile action schema with design

### Code Quality (P3)
1. Add prompt hash tracking
2. Standardize Decimal usage for prices
3. Standardize config key naming
4. Add stop_loss bounds validation
5. Externalize model configuration defaults

---

## Appendix: Test Coverage Gaps

The following code paths have lower coverage and should be prioritized for additional testing:

1. `portfolio_rebalance.py` (76% coverage):
   - Lines 598-606: Kraken API integration
   - Lines 626-635: Kraken ticker integration
   - Lines 651-662: Database hodl bag fetching
   - DCA edge cases

2. `trading_decision.py` (89% coverage):
   - Lines 499-502: JSON parsing edge cases
   - Lines 603-607: Strong majority threshold edge cases

---

*Review completed 2025-12-19 | Phase 2B Core Agents | v1.0*
