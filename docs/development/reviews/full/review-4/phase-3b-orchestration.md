# Review Phase 3B: Orchestration Layer

**Status**: Ready for Review
**Estimated Context**: ~3,000 tokens (code) + review
**Priority**: High - System coordination
**Output**: `findings/phase-3b-findings.md`

---

## Files to Review

| File | Lines | Purpose |
|------|-------|---------|
| `triplegain/src/orchestration/message_bus.py` | ~400 | Inter-agent communication |
| `triplegain/src/orchestration/coordinator.py` | ~700 | Agent orchestration |

**Total**: ~1,100 lines

---

## Pre-Review: Load Files

```bash
# Read these files before starting review
cat triplegain/src/orchestration/message_bus.py
cat triplegain/src/orchestration/coordinator.py

# Also review config
cat config/orchestration.yaml
```

---

## Review Checklist

### 1. Message Bus (`message_bus.py`)

#### Data Structures
- [ ] Message dataclass complete:
  - [ ] id (UUID)
  - [ ] timestamp
  - [ ] topic (enum)
  - [ ] source
  - [ ] priority
  - [ ] payload (dict)
  - [ ] correlation_id
  - [ ] ttl_seconds
- [ ] MessageTopic enum complete:
  - [ ] MARKET_DATA
  - [ ] TA_SIGNALS
  - [ ] REGIME_UPDATES
  - [ ] SENTIMENT_UPDATES
  - [ ] TRADING_SIGNALS
  - [ ] RISK_ALERTS
  - [ ] EXECUTION_EVENTS
  - [ ] PORTFOLIO_UPDATES
  - [ ] SYSTEM_EVENTS
- [ ] MessagePriority enum (LOW, NORMAL, HIGH, URGENT)
- [ ] Subscription dataclass complete

#### Publish/Subscribe
- [ ] publish() delivers to all subscribers
- [ ] subscribe() registers handler correctly
- [ ] unsubscribe() removes subscription
- [ ] Filter functions work correctly
- [ ] Handler exceptions don't crash other subscribers

#### Thread Safety
- [ ] Lock used for message history access
- [ ] Lock used for subscription modification
- [ ] async with lock used correctly
- [ ] No deadlock potential

#### Message History
- [ ] Messages stored in history
- [ ] TTL enforced (expired messages cleaned)
- [ ] get_latest() returns correct message
- [ ] Memory bounded (max history size?)

#### Error Handling
- [ ] Handler exception caught and logged
- [ ] Publish continues after handler error
- [ ] Invalid message handled
- [ ] Topic not found handled

---

### 2. Coordinator Agent (`coordinator.py`)

#### State Machine
- [ ] States defined:
  - [ ] RUNNING
  - [ ] PAUSED
  - [ ] HALTED
- [ ] State transitions correct:
  - [ ] RUNNING → PAUSED (pause())
  - [ ] PAUSED → RUNNING (resume())
  - [ ] Any → HALTED (stop(), circuit breaker)
- [ ] State checked before actions

#### Scheduled Tasks
- [ ] Tasks configured:
  - [ ] TA: every 60s
  - [ ] Regime: every 300s (5 min)
  - [ ] Sentiment: every 1800s (30 min) - Phase 4
  - [ ] Trading: every 3600s (1 hour)
  - [ ] Portfolio: every 3600s (1 hour)
- [ ] Last run timestamp tracked
- [ ] Interval calculation correct
- [ ] Task enable/disable works

#### Task Execution
- [ ] Due tasks identified correctly
- [ ] Tasks execute for all symbols
- [ ] Task errors handled (don't stop other tasks)
- [ ] Task execution logged

#### Trading Signal Handling
- [ ] HOLD signals ignored (no execution)
- [ ] Conflict detection called
- [ ] Risk validation called
- [ ] Execution triggered on approval
- [ ] Rejection logged

#### Conflict Detection
- [ ] TA vs Sentiment conflict detected
- [ ] Regime appropriateness checked
- [ ] Confidence difference threshold (0.2)
- [ ] Multiple conflict types handled

#### Conflict Resolution (LLM)
- [ ] DeepSeek V3 as primary
- [ ] Claude Sonnet as fallback
- [ ] Prompt built correctly
- [ ] Response parsed correctly
- [ ] Actions: proceed, wait, modify, abort

#### Risk Integration
- [ ] Portfolio context fetched
- [ ] Regime fetched for validation
- [ ] Risk engine called correctly
- [ ] Modified proposal used if applicable
- [ ] Rejection handled

#### Message Bus Integration
- [ ] Subscribed to TRADING_SIGNALS
- [ ] Subscribed to RISK_ALERTS
- [ ] Publishes after agent execution
- [ ] Correct message format

#### Emergency Handling
- [ ] Risk alerts trigger halt
- [ ] Circuit breaker respected
- [ ] Positions optionally closed
- [ ] Dashboard notified (if applicable)

#### State Persistence
- [ ] State saved to database
- [ ] State recovered on restart
- [ ] In-flight tasks handled on restart

---

## Critical Questions

1. **Task Starvation**: What if a task takes longer than its interval?
2. **Concurrent Execution**: Are multiple instances of same task prevented?
3. **Symbol Independence**: Does one symbol's error affect others?
4. **Conflict Resolution Timeout**: What if LLM takes too long?
5. **State Consistency**: What if crash during trade execution?
6. **Memory Leak**: Is message history properly bounded?

---

## Timing Analysis

### Schedule Alignment

```
Time 0:00   TA (all symbols)
Time 0:01   TA
...
Time 0:05   TA + Regime
...
Time 0:30   TA + Sentiment (Phase 4)
...
Time 1:00   TA + Trading + Portfolio + Regime
```

- [ ] No resource contention at aligned times
- [ ] Parallel agent execution supported
- [ ] Staggered execution if needed

### Latency Budget

| Step | Max Time | Cumulative |
|------|----------|------------|
| Fetch snapshot | 200ms | 200ms |
| TA analysis | 500ms | 700ms |
| Regime detection | 500ms | 1200ms |
| Trading decision (6 models) | 30s | 31.2s |
| Conflict resolution | 5s | 36.2s |
| Risk validation | 10ms | 36.21s |
| Execution | 2s | 38.21s |

- [ ] Timeout at each stage
- [ ] Overall timeout enforced
- [ ] Partial completion handled

---

## Message Flow Verification

### TA Signal Flow
```
TA Agent → publish(TA_SIGNALS) → Message Bus
                                      ↓
                              Trading Agent (subscribed)
```

### Trading Signal Flow
```
Trading Agent → publish(TRADING_SIGNALS) → Message Bus
                                                ↓
                                          Coordinator (subscribed)
                                                ↓
                                          Risk Validation
                                                ↓
                                          Execution
```

### Risk Alert Flow
```
Risk Engine → publish(RISK_ALERTS) → Message Bus
                                          ↓
                                    Coordinator (subscribed)
                                          ↓
                                    Halt Trading
```

- [ ] All flows implemented
- [ ] No message loss
- [ ] Priority respected

---

## Error Handling Matrix

| Error | Expected Behavior | Recovery |
|-------|-------------------|----------|
| Agent timeout | Log, skip, continue | Next schedule |
| Agent exception | Log, skip, continue | Next schedule |
| Message bus publish fail | Log, raise | Caller handles |
| Database unavailable | Pause trading | Manual intervention |
| LLM provider down | Use fallback | Automatic |
| All LLMs down | Skip trading decision | Next hour |
| Risk validation error | Reject trade | Log and continue |
| Execution error | Log, alert | Manual review |

---

## Concurrency Review

- [ ] async def used throughout
- [ ] await on all I/O operations
- [ ] No blocking calls in async context
- [ ] asyncio.gather for parallel operations
- [ ] asyncio.Lock for shared state
- [ ] Task cancellation handled

---

## Test Coverage Check

```bash
pytest --cov=triplegain/src/orchestration \
       --cov-report=term-missing \
       triplegain/tests/unit/orchestration/
```

Expected tests:
- [ ] Message publish/subscribe
- [ ] Message TTL expiration
- [ ] Filter function behavior
- [ ] Coordinator state transitions
- [ ] Schedule execution
- [ ] Conflict detection scenarios
- [ ] Risk integration
- [ ] Error handling

---

## Design Conformance

### Implementation Plan 3.1 (Communication Protocol)
- [ ] Message schema matches spec
- [ ] All topics implemented
- [ ] Pub/sub pattern implemented

### Implementation Plan 3.2 (Coordinator)
- [ ] Schedule configuration matches spec
- [ ] Conflict resolution logic matches spec
- [ ] State machine matches design

---

## Findings Template

```markdown
## Finding: [Title]

**File**: `triplegain/src/orchestration/filename.py:123`
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
[System impact if not fixed]
```

---

## Review Completion

After completing this phase:

1. [ ] Message bus logic verified
2. [ ] Coordinator state machine verified
3. [ ] Schedule timing verified
4. [ ] Conflict resolution verified
5. [ ] Error handling verified
6. [ ] Findings documented
7. [ ] Ready for Phase 3C

---

*Phase 3B Review Plan v1.0*
