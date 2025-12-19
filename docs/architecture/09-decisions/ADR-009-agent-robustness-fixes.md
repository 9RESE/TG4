# ADR-009: Agent Layer Robustness and Safety Fixes

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Development Team
**Related Review**: Phase 2B Findings (docs/development/reviews/full/review-4/findings/phase-2b-findings.md)

## Context

The Phase 2B code review identified 12 issues (2 P1, 5 P2, 5 P3) in the agent layer. These issues impacted trading safety, data integrity, and code maintainability.

Key problems identified:
1. **Trading Safety**: No minimum quorum for multi-model consensus
2. **Stability**: Regime detection susceptible to flapping on noise
3. **Security**: SQL injection vulnerability in base agent
4. **Data Integrity**: Float precision issues for trading prices
5. **Feature Gap**: Hodl bag profit allocation not implemented

## Decision

We implemented comprehensive fixes across the agent layer:

### 1. Minimum Quorum for Consensus (P1-01)

**Pattern**: Require minimum valid responses before trading

```python
# Default requires 4/6 models to respond successfully
self.min_quorum = config.get('min_quorum', 4)

def _calculate_consensus(self, decisions):
    valid_decisions = [d for d in decisions if d.is_valid()]
    if len(valid_decisions) < self.min_quorum:
        logger.warning(f"Insufficient quorum: {len(valid_decisions)}/{len(decisions)} valid")
        return ConsensusResult(
            final_action='HOLD',
            agreement_type='insufficient_quorum',
            ...
        )
```

**Rationale**: Trading on 2-3 model responses when others timed out leads to unreliable decisions. Forcing HOLD on insufficient data prevents low-quality trades.

**Location**: `triplegain/src/agents/trading_decision.py:232-234,620-645`

### 2. Regime Change Hysteresis (P1-02)

**Pattern**: Require consecutive confirmations and higher confidence

```python
# Regime hysteresis settings
self.min_regime_change_confidence = config.get('min_regime_change_confidence', 0.7)
self.regime_change_periods = config.get('regime_change_periods', 2)

def _should_change_regime(self, new_regime: str, confidence: float) -> bool:
    # Require higher confidence for regime changes
    if confidence < self.min_regime_change_confidence:
        return False

    # Track consecutive confirmations
    if self._pending_regime == new_regime:
        self._pending_regime_count += 1
        return self._pending_regime_count >= self.regime_change_periods
```

**Rationale**: Market noise can cause temporary regime detection shifts. Requiring 2+ consecutive confirmations at >70% confidence filters out false regime changes.

**Location**: `triplegain/src/agents/regime_detection.py:238-245,594-653`

### 3. SQL Injection Prevention (P2-01)

**Pattern**: Parameterized queries for all dynamic values

```python
# Before (vulnerable):
query = "... INTERVAL '%s seconds'" % max_age_seconds

# After (safe):
query = "... make_interval(secs => $3)"
await self.db.fetchrow(query, agent_name, symbol, max_age_seconds)
```

**Rationale**: String interpolation in SQL allows injection attacks. PostgreSQL's `make_interval()` function safely accepts numeric parameters.

**Location**: `triplegain/src/agents/base_agent.py:248-259`

### 4. Decimal for Price Fields (P3-02)

**Pattern**: Use Decimal instead of float for all trading prices

```python
@dataclass
class ModelDecision:
    entry_price: Optional[Decimal] = None  # Was: float
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

@dataclass
class ConsensusResult:
    avg_entry_price: Optional[Decimal] = None
    avg_stop_loss: Optional[Decimal] = None
    avg_take_profit: Optional[Decimal] = None
```

**Rationale**: Floating-point precision errors ($45000.00000001) can cause order matching issues on exchanges. Decimal provides exact arithmetic.

**Location**: `triplegain/src/agents/trading_decision.py:56-59,89-93,139-144`

### 5. Hodl Bag Profit Allocation (P2-02)

**Pattern**: Automatic profit allocation to long-term holdings

```python
async def allocate_profits_to_hodl(self, trade_result: dict) -> dict[str, Decimal]:
    """Allocate 10% of realized profits to hodl bags."""
    pnl = Decimal(str(trade_result.get('realized_pnl_usd', 0)))
    if pnl <= 0:
        return {}

    hodl_amount_usd = pnl * self.hodl_profit_allocation_pct / 100

    # Distribute proportionally between BTC and XRP
    if self.hodl_distribution_strategy == 'proportional':
        # Split based on target allocation
        ...
```

**Rationale**: Per design spec, 10% of trading profits should compound in long-term holdings. This feature was specified but not implemented.

**Location**: `triplegain/src/agents/portfolio_rebalance.py:676-788`

### 6. Regime State Persistence (P2-04)

**Pattern**: Restore regime state on restart

```python
async def load_regime_state(self) -> bool:
    """Load regime state from database on startup."""
    query = """
        SELECT output_data, timestamp
        FROM agent_outputs
        WHERE agent_name = 'regime_detection'
        ORDER BY timestamp DESC LIMIT 1
    """
    row = await self.db.fetchrow(query)
    if row:
        self._previous_regime = output_data.get('regime', 'ranging')
        self._regime_start_time = row['timestamp']
```

**Rationale**: Without persistence, restarts cause incorrect regime duration tracking and potentially sudden regime transitions.

**Location**: `triplegain/src/agents/regime_detection.py:250-305`

### 7. Configuration Standardization (P3-03, P3-05)

**Pattern**: Externalized defaults and documented naming conventions

```python
# Externalized model configuration
DEFAULT_MODEL_CONFIG = {
    'qwen': {'provider': 'ollama', 'model': 'qwen2.5:7b'},
    'gpt4': {'provider': 'openai', 'model': 'gpt-4-turbo'},
    # ...
}

# Naming conventions documented:
# - TTL configs: '_seconds' suffix (cache_ttl_seconds)
# - Percentages: '_pct' suffix (threshold_pct)
# - Booleans: 'enabled' suffix (hodl_enabled)
```

**Rationale**: Hardcoded defaults scattered through `__init__` are hard to update. Centralized constants and consistent naming improve maintainability.

**Location**: `triplegain/src/agents/trading_decision.py:48-81`, `base_agent.py:142-145`

## Consequences

### Positive
- **Safety**: Trading requires 4/6 models; no trades on thin consensus
- **Stability**: Regime flapping eliminated via 2-period hysteresis
- **Security**: SQL injection vulnerability closed
- **Precision**: Price calculations use exact Decimal arithmetic
- **Features**: Hodl bag allocation now works as designed
- **Reliability**: Regime state survives restarts

### Negative
- Slightly higher bar for trading (4 models vs previous 1+)
- Regime changes take longer to confirm (2 periods minimum)
- Test fixtures needed updating for Decimal prices

### Neutral
- New `agreement_type='insufficient_quorum'` value for debugging

## Alternatives Considered

1. **Configurable per-decision quorum**: Allow 2/6 for low-confidence signals
   - Rejected: Complexity outweighs benefit; uniform 4/6 is simpler

2. **Exponential backoff for regime hysteresis**: Longer waits after recent changes
   - Rejected: Simple consecutive count is sufficient for current needs

3. **Float with rounding for prices**: Round to 8 decimals
   - Rejected: Decimal eliminates rounding entirely; cleaner solution

## References

- [Phase 2B Findings](../../../development/reviews/full/review-4/findings/phase-2b-findings.md)
- [Core Agents Design](../../../development/TripleGain-master-design/03-agent-system.md)
- [Python Decimal Documentation](https://docs.python.org/3/library/decimal.html)
