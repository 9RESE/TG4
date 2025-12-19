# Review Phase 2B: Core Agents

**Status**: Ready for Review
**Estimated Context**: ~5,000 tokens (code) + review
**Priority**: Critical - Core trading logic
**Output**: `findings/phase-2b-findings.md`
**DO NOT IMPLEMENT FIXES**

---

## Files to Review

| File | Lines | Purpose |
|------|-------|---------|
| `triplegain/src/agents/base_agent.py` | ~250 | Base agent interface |
| `triplegain/src/agents/technical_analysis.py` | ~550 | TA signal generation |
| `triplegain/src/agents/regime_detection.py` | ~450 | Market regime classification |
| `triplegain/src/agents/trading_decision.py` | ~700 | 6-model trading decisions |
| `triplegain/src/agents/portfolio_rebalance.py` | ~550 | Portfolio allocation |

**Total**: ~2,500 lines

---

## Pre-Review: Load Files

```bash
# Read these files before starting review
cat triplegain/src/agents/base_agent.py
cat triplegain/src/agents/technical_analysis.py
cat triplegain/src/agents/regime_detection.py
cat triplegain/src/agents/trading_decision.py
cat triplegain/src/agents/portfolio_rebalance.py
```

---

## Review Checklist

### 1. Base Agent (`base_agent.py`)

#### Interface Design
- [ ] Abstract base class properly structured
- [ ] Required methods defined (process, get_output_schema, etc.)
- [ ] Type hints complete
- [ ] Docstrings comprehensive

#### Common Functionality
- [ ] Output storage implemented
- [ ] Latency tracking
- [ ] Token/cost tracking
- [ ] Error handling hooks

#### Extensibility
- [ ] Easy to add new agents
- [ ] Configuration injection clean
- [ ] LLM client injection clean

---

### 2. Technical Analysis Agent (`technical_analysis.py`)

#### Output Schema Compliance
- [ ] Output matches JSON schema in implementation plan
- [ ] All required fields present:
  - [ ] timestamp
  - [ ] symbol
  - [ ] trend (direction, strength, timeframe_alignment)
  - [ ] momentum (score, rsi_signal, macd_signal)
  - [ ] key_levels (resistance, support, current_position)
  - [ ] signals (primary, secondary, warnings)
  - [ ] bias
  - [ ] confidence
  - [ ] reasoning

#### Logic Correctness
- [ ] Prompt correctly requests JSON output
- [ ] Response parsing handles all fields
- [ ] Confidence bounds enforced (0.0-1.0)
- [ ] Trend strength bounds enforced (0.0-1.0)
- [ ] Momentum score bounds enforced (-1.0 to 1.0)

#### Indicator Usage
- [ ] Uses pre-computed indicators (doesn't ask LLM to calculate)
- [ ] Indicator values passed correctly in prompt
- [ ] Multi-timeframe data included

#### Error Handling
- [ ] LLM timeout handled
- [ ] Invalid JSON response handled
- [ ] Missing fields handled with defaults
- [ ] Out-of-range values clamped or rejected

#### Output Storage
- [ ] All outputs stored to agent_outputs table
- [ ] Latency recorded
- [ ] Tokens recorded
- [ ] Prompt hash recorded (for caching)

---

### 3. Regime Detection Agent (`regime_detection.py`)

#### Regime Classification
- [ ] All 7 regime types supported:
  - [ ] trending_bull
  - [ ] trending_bear
  - [ ] ranging
  - [ ] volatile_bull
  - [ ] volatile_bear
  - [ ] choppy
  - [ ] breakout_potential

#### Output Compliance
- [ ] Output matches JSON schema
- [ ] Required fields present:
  - [ ] timestamp
  - [ ] symbol
  - [ ] regime
  - [ ] confidence
  - [ ] characteristics (volatility, trend_strength, volume_profile)
  - [ ] recommended_adjustments

#### Recommended Adjustments
- [ ] position_size_multiplier in valid range (0.25-1.5)
- [ ] stop_loss_multiplier in valid range (0.5-2.0)
- [ ] take_profit_multiplier in valid range (0.5-3.0)
- [ ] entry_strictness enum valid

#### Logic Correctness
- [ ] Uses TA output if provided
- [ ] ADX used for trend strength
- [ ] Choppiness index used for choppy detection
- [ ] ATR used for volatility assessment

#### Regime Transitions
- [ ] Previous regime considered (if available)
- [ ] Transition smoothing (avoid flapping)
- [ ] Regime duration tracking

---

### 4. Trading Decision Agent (`trading_decision.py`)

#### 6-Model Parallel Execution
- [ ] All 6 models invoked:
  - [ ] GPT (OpenAI)
  - [ ] Grok (xAI)
  - [ ] DeepSeek V3
  - [ ] Claude Sonnet
  - [ ] Claude Opus
  - [ ] Qwen 2.5 7B (Ollama)
- [ ] asyncio.gather used for parallelism
- [ ] Timeout handling per model
- [ ] Failed model handling (graceful degradation)

#### Consensus Calculation
- [ ] Vote counting correct
- [ ] Majority action determined correctly
- [ ] Agreement levels correct:
  - [ ] Unanimous: 6/6 = +0.15 boost
  - [ ] Strong majority: 5/6 = +0.10 boost
  - [ ] Majority: 4/6 = +0.05 boost
  - [ ] Split: <=3/6 = HOLD, no boost

#### Output Schema Compliance
- [ ] All required fields:
  - [ ] timestamp
  - [ ] symbol
  - [ ] action (BUY, SELL, HOLD, CLOSE)
  - [ ] confidence
  - [ ] parameters (entry_type, entry_price, size_pct, leverage, stop_loss_pct, take_profit_pct, time_horizon)
  - [ ] reasoning

#### Parameter Bounds
- [ ] size_pct: 0-100
- [ ] leverage: 1-5
- [ ] stop_loss_pct: 1.0-5.0
- [ ] take_profit_pct: reasonable range
- [ ] confidence: 0.0-1.0 (after boost)

#### Input Processing
- [ ] TA output integrated correctly
- [ ] Regime output integrated correctly
- [ ] Sentiment output integrated (when available)
- [ ] Portfolio context considered

#### Comparison Storage
- [ ] All model outputs stored to model_comparisons table
- [ ] Consensus result stored
- [ ] Individual latencies recorded
- [ ] Individual costs recorded

---

### 5. Portfolio Rebalance Agent (`portfolio_rebalance.py`)

#### Allocation Calculation
- [ ] Current balances fetched correctly
- [ ] Prices converted to USD correctly
- [ ] Percentages calculated correctly
- [ ] Hodl bags excluded from calculation

#### Rebalancing Logic
- [ ] Threshold check correct (default 5%)
- [ ] Target allocation correct (33.33/33.33/33.33)
- [ ] Trade amounts calculated correctly
- [ ] Minimum trade size enforced ($10)

#### Trade Generation
- [ ] Correct symbols used
- [ ] Buy/sell direction correct
- [ ] Priority ordering (sell first to free up capital)
- [ ] Execution type (limit vs market)

#### Hodl Bag Handling
- [ ] Hodl bags tracked separately
- [ ] Hodl bags excluded from rebalancing
- [ ] Profit allocation to hodl bags (10%)

---

## Critical Questions

1. **Consensus Safety**: What if only 2 models respond? Is there a minimum quorum?
2. **Confidence Inflation**: Could confidence boost exceed 1.0?
3. **Decimal Precision**: Are all financial calculations using Decimal?
4. **Race Conditions**: Are there any shared mutable state issues in parallel execution?
5. **Output Validation**: What if LLM returns invalid JSON?
6. **Regime Flapping**: How is rapid regime switching prevented?

---

## Trading Logic Review

### Decision Flow
```
TA Output → Regime → Trading Decision → Risk Validation → Execution
           ↓
    (modifies confidence thresholds)
```

- [ ] TA bias correctly influences trading decision
- [ ] Regime adjustments applied to position size
- [ ] Confidence thresholds regime-aware
- [ ] Choppy regime increases strictness

### Signal Quality
- [ ] Long bias requires bullish trend + positive momentum
- [ ] Short bias requires bearish trend + negative momentum
- [ ] Neutral/HOLD when conflicting signals
- [ ] High confidence requires multi-timeframe alignment

---

## Test Coverage Check

```bash
pytest --cov=triplegain/src/agents \
       --cov-report=term-missing \
       triplegain/tests/unit/agents/
```

Expected tests:
- [ ] Each agent's process() method
- [ ] Output schema validation
- [ ] Consensus calculation edge cases
- [ ] Error handling scenarios
- [ ] Parameter bounds enforcement

---

## Design Conformance

### Implementation Plan 2.1 (TA Agent)
- [ ] Output schema matches spec
- [ ] Invocation frequency matches (per minute)
- [ ] Model assignment correct (Qwen 2.5 7B)

### Implementation Plan 2.2 (Regime Agent)
- [ ] All 7 regimes supported
- [ ] Adjustments match spec
- [ ] Invocation frequency matches (5 min)

### Implementation Plan 2.4 (Trading Decision)
- [ ] 6-model parallel execution
- [ ] Consensus rules match spec
- [ ] Invocation frequency matches (hourly)

---

## Findings Template

```markdown
## Finding: [Title]

**File**: `triplegain/src/agents/filename.py:123`
**Priority**: P0/P1/P2/P3
**Category**: Security/Logic/Performance/Quality

### Description
[What was found]

### Current Code
```python
# current implementation
```

### Recommended Fix
```python
# recommended fix
```

### Impact
[Trading impact if not fixed]
```

---

## Review Completion

After completing this phase:

1. [ ] All 5 agent files reviewed
2. [ ] Output schemas verified
3. [ ] Consensus logic verified
4. [ ] Parameter bounds verified
5. [ ] Findings documented
6. [ ] Ready for Phase 3A

---

*Phase 2B Review Plan v1.0*
