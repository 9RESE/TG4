# ADR-004: Phase 3 Orchestration Architecture

**Status**: Accepted
**Date**: 2025-12-18
**Context**: Phase 3 Orchestration Implementation

## Context

Phase 3 requires implementing the orchestration layer for TripleGain:
- Inter-agent communication protocol
- Agent scheduling and coordination
- Portfolio rebalancing
- Order execution and position management

Key considerations:
1. Agents must communicate efficiently without tight coupling
2. Coordinator must handle conflicts between agent outputs
3. Portfolio must maintain 33/33/33 BTC/XRP/USDT allocation
4. Order execution must be reliable with proper error handling

## Decision Drivers

- **Decoupling**: Agents should communicate without direct dependencies
- **Reliability**: System must handle failures gracefully
- **Performance**: Message passing must be low-latency
- **Flexibility**: Easy to add new agents or modify schedules
- **Auditability**: All decisions and conflicts must be logged

## Decisions Made

### 1. In-Memory Pub/Sub Message Bus

**Decision**: Implement in-memory pub/sub message bus with topic-based routing.

**Rationale**:
- Simple, zero-latency communication
- No external dependencies (Redis, RabbitMQ)
- Sufficient for single-process deployment
- Easy to test and debug
- Built-in TTL for message expiration

**Implementation**:
- `MessageBus` class with async subscribe/publish
- Topic-based routing (`MessageTopic` enum)
- Priority levels for urgent messages
- Optional filter functions for subscribers
- Thread-safe with `asyncio.Lock`

**Alternatives Considered**:
- Redis Pub/Sub: External dependency, overkill for current scale
- Direct method calls: Tight coupling, harder to extend
- Event sourcing: Too complex for initial implementation

### 2. Coordinator Agent with LLM Conflict Resolution

**Decision**: Implement coordinator that uses DeepSeek V3 / Claude Sonnet for conflict resolution.

**Rationale**:
- LLM can reason about conflicting agent outputs
- Only invoked when actual conflicts exist (cost optimization)
- Fallback model ensures availability
- Conservative default (wait/abort) when uncertain

**Implementation**:
- Scheduled task execution for each agent type
- Conflict detection based on signal disagreement
- JSON-structured conflict resolution prompts
- Parse resolution with fallback to conservative action

**Conflict Types Handled**:
| Conflict | Detection | Resolution |
|----------|-----------|------------|
| TA vs Sentiment | Bias disagreement with close confidence | LLM arbitration |
| Regime incompatibility | Choppy regime + active trade signal | Usually abort |
| Multi-symbol conflict | Same direction on correlated pairs | LLM prioritization |

**Alternatives Considered**:
- Rules-based resolution: Too rigid for complex scenarios
- Always-on LLM: Too expensive, unnecessary latency
- Manual resolution: Not scalable, defeats automation

### 3. Portfolio Rebalancing Agent with Hodl Bag Exclusion

**Decision**: Implement portfolio agent with configurable target allocation and hodl bag protection.

**Rationale**:
- 33/33/33 allocation provides diversification
- Hodl bags accumulate long-term holdings tax-efficiently
- 5% threshold prevents excessive trading costs
- LLM determines execution strategy (limit vs market)

**Implementation**:
- Hourly allocation check
- Deviation calculation against target
- Trade generation for overweight/underweight assets
- Sells executed before buys (fund availability)
- Hodl bags excluded from rebalancing calculations

**Alternatives Considered**:
- Fixed rebalancing schedule: May miss opportunities or execute unnecessarily
- Percentage-based triggers: 5% chosen as balance between cost and drift
- No hodl bags: Loses long-term accumulation benefit

### 4. Order Execution Manager with Position Tracker (Non-LLM)

**Decision**: Implement purely rule-based order execution and position tracking.

**Rationale**:
- Execution requires deterministic, fast responses
- No LLM overhead for time-critical operations
- Retry logic must be predictable
- Position P&L must be calculated accurately

**Implementation**:
- Order lifecycle: PENDING → OPEN → FILLED/CANCELLED
- Retry with exponential backoff on transient errors
- Contingent order placement (stop-loss, take-profit)
- Real-time P&L calculation
- Position snapshots for time-series analysis

**Components**:
| Component | Purpose |
|-----------|---------|
| `OrderExecutionManager` | Order lifecycle, Kraken API integration |
| `PositionTracker` | P&L calculation, SL/TP monitoring |
| `order_status_log` table | Order state history |
| `position_snapshots` table | Time-series position data |

**Alternatives Considered**:
- LLM execution decisions: Too slow, unpredictable
- Third-party execution service: Added complexity, cost

### 5. API Routes with Dependency Injection

**Decision**: Use FastAPI router factory pattern with optional component injection.

**Rationale**:
- Components can be None during testing or development
- Proper 503 responses when components unavailable
- Clean separation of concerns
- Easy to mock for testing

**Implementation**:
- `create_orchestration_router()` factory function
- Optional parameters for all components
- Availability checks before operations
- Comprehensive error handling with proper HTTP codes

## Consequences

### Positive

- **Loose Coupling**: Agents communicate via messages, easy to add/remove
- **Testability**: Each component can be tested in isolation
- **Observability**: All messages and decisions logged
- **Flexibility**: Schedules and thresholds configurable
- **Reliability**: Graceful degradation when components unavailable

### Negative

- **Single Process**: Message bus limited to single process (acceptable for current scale)
- **State Loss**: In-memory message history lost on restart (mitigated by DB persistence)
- **Complexity**: More components to maintain than monolithic approach

### Risks Mitigated

| Risk | Mitigation |
|------|------------|
| Agent deadlock | Async message handling, timeouts |
| Execution failure | Retry logic, error state tracking |
| Conflict loops | Conservative resolution default |
| Memory growth | TTL-based message expiration |

## Implementation Summary

**Files Created**:
- `triplegain/src/orchestration/message_bus.py` (190 lines)
- `triplegain/src/orchestration/coordinator.py` (349 lines)
- `triplegain/src/agents/portfolio_rebalance.py` (184 lines)
- `triplegain/src/execution/order_manager.py` (321 lines)
- `triplegain/src/execution/position_tracker.py` (270 lines)
- `triplegain/src/api/routes_orchestration.py` (233 lines)
- `config/orchestration.yaml`
- `config/portfolio.yaml`
- `config/execution.yaml`
- `migrations/003_phase3_orchestration.sql`

**Test Coverage**:
- 227 new tests for Phase 3 components
- 916 total tests (was 689)
- 87% code coverage

## References

- [Phase 3 Implementation Plan](../../development/TripleGain-implementation-plan/03-phase-3-orchestration.md)
- [Multi-Agent Architecture](../../development/TripleGain-master-design/01-multi-agent-architecture.md)
- [Risk Management Design](../../development/TripleGain-master-design/03-risk-management-rules-engine.md)

---

*ADR-004 v1.0 - December 2025*
