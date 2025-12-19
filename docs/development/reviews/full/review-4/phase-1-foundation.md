# Review Phase 1: Foundation Layer

**Status**: Ready for Review
**Estimated Context**: ~4,000 tokens (code) + review
**Priority**: High - All other phases depend on this
**Output**: `findings/phase-1-findings.md`

---

## Files to Review

| File | Lines | Purpose |
|------|-------|---------|
| `triplegain/src/data/database.py` | ~350 | Database connection, queries |
| `triplegain/src/data/indicator_library.py` | ~650 | Technical indicator calculations |
| `triplegain/src/data/market_snapshot.py` | ~550 | Market data aggregation |
| `triplegain/src/llm/prompt_builder.py` | ~500 | Prompt template system |
| `triplegain/src/utils/config.py` | ~150 | Configuration loader |

**Total**: ~2,200 lines

---

## Pre-Review: Load Files

```bash
# Read these files before starting review
cat triplegain/src/data/database.py
cat triplegain/src/data/indicator_library.py
cat triplegain/src/data/market_snapshot.py
cat triplegain/src/llm/prompt_builder.py
cat triplegain/src/utils/config.py
```

---

## Review Checklist

### 1. Database Layer (`database.py`)

#### Connection Management
- [ ] Connection pooling implemented correctly
- [ ] Connections properly closed/returned to pool
- [ ] Timeout handling for queries
- [ ] Reconnection logic for dropped connections

#### Query Safety
- [ ] All queries use parameterized statements (no f-strings in SQL)
- [ ] Input validation before queries
- [ ] Proper error handling for query failures
- [ ] Transaction management (commit/rollback)

#### Data Types
- [ ] Decimal used for financial values (not float)
- [ ] Timezone-aware datetime handling
- [ ] UUID handling for IDs
- [ ] JSON/JSONB field handling

#### Performance
- [ ] Indexes utilized in queries
- [ ] Batch operations for bulk inserts
- [ ] Query result caching where appropriate
- [ ] Connection pool size appropriate

---

### 2. Indicator Library (`indicator_library.py`)

#### Calculation Accuracy
- [ ] EMA calculation matches standard formula
- [ ] RSI calculation produces 0-100 range values
- [ ] MACD signal/histogram calculation correct
- [ ] ATR uses true range (not simple range)
- [ ] Bollinger Bands use standard deviation correctly
- [ ] ADX calculation follows Wilder's method

#### Edge Cases
- [ ] Handle empty/null input arrays
- [ ] Handle single-element arrays
- [ ] Handle arrays shorter than period
- [ ] NaN/Inf handling
- [ ] Zero volume handling for VWAP/OBV

#### Data Type Handling
- [ ] Decimal used for financial calculations
- [ ] Float conversion only for numpy operations
- [ ] Precision preserved in results

#### Performance
- [ ] Numpy vectorized operations used
- [ ] No unnecessary loops
- [ ] Memory efficient (no large copies)
- [ ] Caching for repeated calculations

#### Specific Indicator Checks

| Indicator | Check | Expected Range |
|-----------|-------|----------------|
| RSI | `0 <= value <= 100` | 0-100 |
| MACD | Sign consistency with price direction | -Inf to +Inf |
| EMA | Smoothing factor = 2/(period+1) | Price range |
| ATR | Always positive | > 0 |
| BB Position | `0 <= value <= 1` for normal conditions | 0-1 (can exceed) |
| ADX | `0 <= value <= 100` | 0-100 |
| Choppiness | `0 <= value <= 100` | 0-100 |

---

### 3. Market Snapshot Builder (`market_snapshot.py`)

#### Data Completeness
- [ ] All required fields populated
- [ ] Missing data flagged appropriately
- [ ] Data age calculated correctly
- [ ] Multiple timeframes aggregated properly

#### Snapshot Accuracy
- [ ] Current price is truly current (not stale)
- [ ] 24h price change calculated correctly
- [ ] MTF alignment score calculation correct
- [ ] Order book features extracted properly

#### Token Budget
- [ ] `to_prompt_format()` respects token budget
- [ ] Truncation prioritizes important data
- [ ] `to_compact_format()` sufficiently compact for Tier 1

#### Error Handling
- [ ] Database unavailable handling
- [ ] Missing candle data handling
- [ ] Order book unavailable handling
- [ ] Timeout handling

#### Concurrency
- [ ] Thread-safe snapshot building
- [ ] Async operations used correctly
- [ ] No race conditions in data aggregation

---

### 4. Prompt Builder (`prompt_builder.py`)

#### Template System
- [ ] Templates load correctly from disk
- [ ] Missing template error handling
- [ ] Variable substitution works correctly
- [ ] No injection vulnerabilities in templates

#### Token Management
- [ ] Token estimation reasonably accurate
- [ ] Token budget enforced
- [ ] Truncation maintains valid JSON
- [ ] Different budgets for Tier 1 vs Tier 2

#### Context Injection
- [ ] Portfolio context formatted correctly
- [ ] Market data formatted correctly
- [ ] Additional context (agent outputs) handled
- [ ] Sensitive data not leaked into prompts

#### Output Quality
- [ ] System prompts clear and complete
- [ ] User message structured properly
- [ ] JSON output instructions explicit
- [ ] Confidence score instructions clear

---

### 5. Config Loader (`config.py`)

#### Environment Variables
- [ ] Sensitive values from env vars
- [ ] Default values appropriate
- [ ] Missing required vars handled with clear error
- [ ] Type conversion correct

#### File Loading
- [ ] YAML parsing error handling
- [ ] Schema validation
- [ ] Config file not found handling
- [ ] Config hot-reloading (if supported)

#### Security
- [ ] No secrets logged
- [ ] No secrets in default values
- [ ] Config file permissions appropriate

---

## Critical Questions

1. **Decimal Precision**: Are all price/amount calculations using Decimal, not float?
2. **SQL Injection**: Are there ANY string-interpolated SQL queries?
3. **Indicator Accuracy**: Do RSI/MACD/EMA calculations match industry standards?
4. **Data Freshness**: How is stale data detected and handled?
5. **Error Propagation**: Do errors bubble up appropriately or get swallowed?

---

## Test Coverage Check

Verify tests exist for:

```bash
# Run coverage for Phase 1 files
pytest --cov=triplegain/src/data --cov=triplegain/src/utils \
       --cov=triplegain/src/llm/prompt_builder \
       --cov-report=term-missing
```

Expected minimum coverage: 80%

---

## Design Conformance

### Implementation Plan 1.1 (Data Pipeline)
- [ ] Schema matches specification in 01-phase-1-foundation.md
- [ ] All tables created (agent_outputs, trading_decisions, etc.)
- [ ] Indexes created as specified

### Implementation Plan 1.2 (Indicator Library)
- [ ] All specified indicators implemented
- [ ] Parameters match specification
- [ ] Output format matches spec

### Implementation Plan 1.3 (Market Snapshot)
- [ ] MarketSnapshot dataclass matches spec
- [ ] CandleSummary format correct
- [ ] MTF state calculation implemented

### Implementation Plan 1.4 (Prompt Templates)
- [ ] Token budgets match spec
- [ ] Agent-specific settings correct
- [ ] Template structure matches design

---

## Findings Template

```markdown
## Finding: [Title]

**File**: `triplegain/src/data/filename.py:123`
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
[What could happen if not fixed]

### Test to Add
```python
# suggested test case
```
```

---

## Review Completion

After completing this phase:

1. [ ] All files reviewed
2. [ ] All checklist items addressed
3. [ ] Findings documented with priorities
4. [ ] Critical issues flagged
5. [ ] Ready for Phase 2A

---

*Phase 1 Review Plan v1.0*
