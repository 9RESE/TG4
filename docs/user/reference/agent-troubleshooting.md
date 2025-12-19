# Agent Troubleshooting Guide

**Last Updated**: 2025-12-19
**Applies To**: TripleGain v1.4+

---

## Common Agent Issues

This guide covers common issues discovered during code review and testing of the TripleGain agent system.

---

## 1. Portfolio Rebalance Agent Not Found

### Symptom

```python
from triplegain.src.agents import PortfolioRebalanceAgent
# ImportError: cannot import name 'PortfolioRebalanceAgent'
```

### Cause

Missing export in `agents/__init__.py` (Issue P1-003)

### Solution

**Temporary Workaround**:
```python
from triplegain.src.agents.portfolio_rebalance import PortfolioRebalanceAgent
```

**Permanent Fix**: Update `/triplegain/src/agents/__init__.py`:
```python
from .portfolio_rebalance import (
    PortfolioRebalanceAgent,
    RebalanceOutput,
    PortfolioAllocation,
    RebalanceTrade
)

__all__ = [
    # ... existing exports ...
    'PortfolioRebalanceAgent',
    'RebalanceOutput',
]
```

---

## 2. SQL Interval Error in Agent Output Retrieval

### Symptom

```
asyncpg.exceptions.PostgresSyntaxError: syntax error at or near "300"
LINE 3: AND timestamp > NOW() - INTERVAL '300 seconds'
```

### Cause

SQL string interpolation vulnerability (Issue P2-001)

### Location

`base_agent.py` line 252

### Solution

Replace in `BaseAgent.get_latest_output()`:

**Before**:
```python
query = """
    SELECT output_data, timestamp
    FROM agent_outputs
    WHERE agent_name = $1 AND symbol = $2
    AND timestamp > NOW() - INTERVAL '%s seconds'
    ORDER BY timestamp DESC
    LIMIT 1
""" % max_age_seconds
```

**After**:
```python
query = """
    SELECT output_data, timestamp
    FROM agent_outputs
    WHERE agent_name = $1 AND symbol = $2
    AND timestamp > NOW() - $3::INTERVAL
    ORDER BY timestamp DESC
    LIMIT 1
"""
row = await self.db.fetchrow(
    query,
    self.agent_name,
    symbol,
    f'{max_age_seconds} seconds'
)
```

---

## 3. Regime Detection Periods Incorrect for Multi-Symbol

### Symptom

Regime `periods_in_regime` counter shows wrong values when processing multiple symbols (e.g., BTC and XRP tracked by same agent instance).

**Example**:
```
BTC: trending_bull for 5 periods  # Correct
XRP: trending_bull for 5 periods  # Wrong - should be 2
```

### Cause

Regime state stored at instance level, not per-symbol (Issue P2-002)

### Location

`regime_detection.py` lines 233-236

### Solution

**Short-term**: Use separate agent instances per symbol
```python
btc_regime_agent = RegimeDetectionAgent(...)
xrp_regime_agent = RegimeDetectionAgent(...)
```

**Long-term**: Make state symbol-keyed:
```python
# In __init__
self._regime_state: dict[str, dict] = {}

# In process()
if symbol not in self._regime_state:
    self._regime_state[symbol] = {
        'regime': None,
        'start_time': None,
        'periods': 0
    }

state = self._regime_state[symbol]
if current_regime != state['regime']:
    state['regime'] = current_regime
    state['start_time'] = datetime.now(timezone.utc)
    state['periods'] = 0
else:
    state['periods'] += 1
```

---

## 4. Consensus Always Picks Same Action on Ties

### Symptom

When 6 models split evenly (2 BUY, 2 SELL, 2 HOLD), the winning action is always the same one (e.g., always BUY).

### Cause

Tie-breaking uses dictionary insertion order, not safety priority (Issue P1-001)

### Location

`trading_decision.py` line 586

### Solution

Implement explicit safety-based tie-breaking:

```python
# Add to TradingDecisionAgent class
SAFETY_PRIORITY = {
    'HOLD': 0,         # Safest
    'CLOSE_LONG': 1,
    'CLOSE_SHORT': 1,
    'SELL': 2,
    'BUY': 3,          # Riskiest
}

# In _calculate_consensus()
winning_action = max(
    votes.keys(),
    key=lambda a: (
        votes[a],  # Primary: most votes
        -self.SAFETY_PRIORITY.get(a, 99)  # Tie-break: safest
    )
)
```

**Result**: On ties, HOLD wins over BUY/SELL (safer default)

---

## 5. Hodl Bags Exceed Balance Error

### Symptom

```
WARNING: BTC hodl bag (0.5) exceeds balance (0.3), clamping to 0
```

Then rebalancing proceeds with incorrect assumptions.

### Cause

Data integrity check warns but continues (Issue P2-004)

### Location

`portfolio_rebalance.py` lines 345-362

### Recommended Solution

**Option 1**: Fail loudly
```python
if available_btc < 0:
    raise ValueError(
        f"Hodl bag integrity error: BTC hodl ({hodl_bags['BTC']}) "
        f"exceeds balance ({balances['BTC']}). Fix hodl_bags table."
    )
```

**Option 2**: Return error output (preferred for production)
```python
if available_btc < 0 or available_xrp < 0 or available_usdt < 0:
    return RebalanceOutput(
        agent_name=self.agent_name,
        timestamp=datetime.now(timezone.utc),
        symbol="PORTFOLIO",
        confidence=0.0,
        reasoning="Hodl bag exceeds balance - data integrity error",
        action="no_action",
        trades=[],
    )
```

**Fix Root Cause**: Check `hodl_bags` table for corruption:
```sql
SELECT
    hb.asset,
    hb.amount AS hodl_amount,
    -- Compare to actual balance from Kraken
FROM hodl_bags hb
WHERE hb.account_id = 'default';
```

---

## 6. LLM Timeout Causes Incomplete Consensus

### Symptom

Trading decision shows fewer than 6 models voted:
```
Consensus: BUY (3/6 models agree, 50% strength)
```

But no clear indication if this is acceptable.

### Cause

Model timeouts are handled gracefully (good!), but minimum threshold not enforced (Issue P1-002)

### Location

`trading_decision.py` lines 399-414

### Current Behavior

- Timeout after 30 seconds
- Timed-out models logged but excluded from consensus
- Consensus calculated from available results

### Recommended Improvement

Add minimum model threshold:

```python
# In config
min_models_required: 3  # Need at least half

# In _query_all_models()
if len(decisions) < self.config.get('min_models_required', 3):
    logger.error(
        f"Only {len(decisions)}/6 models responded, "
        f"minimum {self.config['min_models_required']} required"
    )
    # Return HOLD with low confidence
    return self._create_insufficient_data_output()
```

---

## 7. ADX Normalization Produces Weak Trend Strength

### Symptom

Fallback regime detection shows weak trend even with strong ADX:
```
Regime: trending_bull
Trend Strength: 0.25  # Should be higher
ADX: 45  # Strong trend!
```

### Cause

ADX/100 normalization doesn't match indicator semantics (Issue P3-003)

### Location

`regime_detection.py` line 542

### Solution

Use non-linear normalization:

```python
# ADX interpretation:
# 0-25: No trend
# 25-50: Strong trend
# 50+: Very strong trend

def normalize_adx(adx: float) -> float:
    """Convert ADX (0-100) to trend strength (0-1)."""
    if adx < 25:
        return adx / 50  # 0-25 ADX → 0-0.5 strength
    else:
        return 0.5 + min(0.5, (adx - 25) / 50)  # 25-75 ADX → 0.5-1.0

# Use in fallback
'trend_strength': normalize_adx(adx) if adx else 0.25,
```

---

## 8. Target Allocation Doesn't Sum to 100%

### Symptom

```
WARNING: Target allocations sum to 99.99%, not 100%.
BTC=33.33%, XRP=33.33%, USDT=33.33%
```

Agent continues but rebalancing logic may be incorrect.

### Cause

Configuration validation only warns (Issue P3-005)

### Location

`portfolio_rebalance.py` lines 176-182

### Solution

Fail on initialization:

```python
# In __init__
total_allocation = self.target_btc_pct + self.target_xrp_pct + self.target_usdt_pct
if abs(total_allocation - 100) > Decimal('0.01'):  # Allow 0.01% rounding
    raise ValueError(
        f"Target allocations must sum to 100%, got {float(total_allocation):.2f}%. "
        f"Check config/portfolio.yaml"
    )
```

**Fix config**:
```yaml
# config/portfolio.yaml
target_allocation:
  btc_pct: 33.33
  xrp_pct: 33.33
  usdt_pct: 33.34  # ← Extra 0.01% to reach 100%
```

---

## 9. Cache Returns Stale Data

### Symptom

Agent returns outdated analysis even with fresh market data.

### Diagnosis

Check cache TTL:
```python
# In agent config
cache_ttl_seconds: 300  # 5 minutes
```

If TTL too long for your use case, reduce it:

```yaml
# config/agents.yaml
agents:
  technical_analysis:
    cache_ttl_seconds: 60  # 1 minute for faster updates
```

### Clear Cache Manually

```python
# In code
await agent.clear_cache(symbol='BTC/USDT')  # Clear one symbol
await agent.clear_cache()  # Clear all
```

### Verify Cache Miss

Enable debug logging:
```python
import logging
logging.getLogger('triplegain.src.agents').setLevel(logging.DEBUG)
```

Look for:
```
DEBUG: TA Agent: Cache hit for BTC/USDT (age=45.2s)
DEBUG: TA Agent: Cache miss for BTC/USDT (expired)
```

---

## 10. DCA Batches Below Minimum Trade Size

### Symptom

```
INFO: DCA batches would be below minimum ($10.00), executing immediately
```

### Cause

Total trade divided by 6 batches results in <$10 per batch

### Example

```
Total rebalance: $50
DCA batches: 6
Per batch: $8.33 < $10 minimum
```

### Solution

Automatic: Agent reduces batch count to meet minimum

**Manual Override** (if you want small batches):
```yaml
# config/portfolio.yaml
rebalancing:
  min_trade_usd: 5.0  # Lower minimum
```

**Or increase DCA threshold**:
```yaml
rebalancing:
  dca:
    threshold_usd: 100  # Only DCA for >$100 rebalances
```

---

## 11. Model Cost Tracking Missing Per-Model Breakdown

### Symptom

Can see total cost but not which models are expensive:
```python
stats = trading_agent.get_stats()
print(stats['total_cost_usd'])  # $2.50
# How much did GPT-4 vs Qwen cost?
```

### Workaround

Query `model_comparisons` table:
```sql
SELECT
    model_name,
    SUM(cost_usd) AS total_cost,
    AVG(cost_usd) AS avg_cost,
    COUNT(*) AS calls
FROM model_comparisons
WHERE timestamp > NOW() - INTERVAL '1 day'
GROUP BY model_name
ORDER BY total_cost DESC;
```

### Enhancement Needed

See Issue P3-004 for implementation in `get_stats()`

---

## 12. Floating Point in Financial Calculations

### Symptom

```
Total: $9999.999999999998  # Should be $10000.00
```

### Cause

Using `float` instead of `Decimal`

### Fix

**Check all financial math uses Decimal**:

```python
from decimal import Decimal

# BAD
total = price * quantity

# GOOD
total = Decimal(str(price)) * Decimal(str(quantity))
```

**Note**: Always convert to `str` first to avoid float precision loss:
```python
Decimal(1.1)  # BAD: Decimal('1.100000000000000088817841970012523233890533447265625')
Decimal('1.1')  # GOOD: Decimal('1.1')
```

---

## Performance Issues

### Agent Latency Exceeds Target

**Symptom**: TA agent takes >500ms consistently

**Diagnosis**:
1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Check model loaded: Should see `qwen2.5:7b`
3. Monitor Ollama: `docker logs -f ollama` (if in Docker)

**Common Causes**:
- Model not pre-loaded (first call loads model, slow)
- CPU/GPU throttling
- Network latency to Ollama

**Solutions**:
1. Pre-load model: `ollama run qwen2.5:7b` (keep terminal open)
2. Increase timeout: `timeout_ms: 10000` in config
3. Use faster model: `model: "qwen2.5:3b"` (less accurate but faster)

---

## Database Issues

### Agent Outputs Not Persisting

**Symptom**: Agent runs but no records in `agent_outputs` table

**Check**:
```python
# In agent initialization
if agent.db is None:
    print("No database configured!")
```

**Solution**:
```python
# Pass db_pool to agent
from triplegain.src.data.database import get_pool

db_pool = await get_pool()
agent = TechnicalAnalysisAgent(
    llm_client=llm,
    prompt_builder=builder,
    config=config,
    db_pool=db_pool  # ← Required for persistence
)
```

### AsyncPG Connection Pool Exhausted

**Symptom**:
```
asyncpg.exceptions.TooManyConnectionsError: pool exhausted
```

**Solution**: Increase pool size in config:
```python
# src/data/database.py
pool = await asyncpg.create_pool(
    dsn=db_url,
    min_size=5,   # Increase from 2
    max_size=20,  # Increase from 10
)
```

---

## Testing Issues

### Schema Validation Tests Failing

**Symptom**:
```
jsonschema.exceptions.ValidationError: 'trending_bull' is not of type 'string'
```

**Cause**: Schema expects string, output provides enum

**Fix**: Ensure output serialization matches schema:
```python
def to_dict(self):
    return {
        'regime': self.regime,  # Already string from enum
        # NOT: 'regime': self.regime.value
    }
```

---

## Configuration Issues

### Missing Environment Variables

**Required**:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="sk-..."
export XAI_BEARER_API_KEY="xai-..."
export DATABASE_URL="postgresql://user:pass@localhost/triplegain"
```

**Check**:
```python
import os
required = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
missing = [k for k in required if not os.getenv(k)]
if missing:
    print(f"Missing: {missing}")
```

---

## Logging and Debugging

### Enable Debug Logging

```python
import logging

# All agents
logging.getLogger('triplegain.src.agents').setLevel(logging.DEBUG)

# Specific agent
logging.getLogger('triplegain.src.agents.technical_analysis').setLevel(logging.DEBUG)

# LLM calls
logging.getLogger('triplegain.src.llm').setLevel(logging.DEBUG)
```

### Trace Agent Execution

```python
# Add to agent process()
logger.info(f"Input: {snapshot.to_dict()}")
logger.info(f"Prompt: {prompt.user_message[:200]}...")
logger.info(f"Response: {response_text[:200]}...")
logger.info(f"Output: {output.to_dict()}")
```

---

## Getting Help

### 1. Check Logs
```bash
tail -f logs/triplegain.log | grep ERROR
```

### 2. Enable Debug Mode
```yaml
# config/logging.yaml
level: DEBUG
```

### 3. Review Test Output
```bash
pytest triplegain/tests/unit/agents/ -v --tb=short
```

### 4. Check Agent Stats
```python
stats = agent.get_stats()
print(f"Invocations: {stats['total_invocations']}")
print(f"Avg Latency: {stats['average_latency_ms']:.0f}ms")
print(f"Errors: {stats.get('error_count', 0)}")
```

---

**Last Updated**: 2025-12-19
**Next Review**: After Phase 4 completion

