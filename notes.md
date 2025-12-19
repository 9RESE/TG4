# Notes

Prompt:
# Task: Begin TripleGain Phase 3 Implementation - Orchestration
## Context
Read these files first:
1. CLAUDE.md - Project memory and constraints
2. docs/development/TripleGain-implementation-plan/03-phase-3-orchestration.md - Phase 3 spec
3. docs/development/TripleGain-master-design/01-multi-agent-architecture.md - Agent architecture
4. docs/development/TripleGain-master-design/03-risk-management-rules-engine.md - Risk rules
5. config/agents.yaml - LLM provider configuration (API keys in .env)
## Phases 1 & 2 Complete (DO NOT recreate)
Located in `triplegain/src/`:
**Phase 1 - Foundation:**
- Indicator Library: `data/indicator_library.py` (17+ indicators)
- Market Snapshot: `data/market_snapshot.py` (multi-timeframe aggregation)
- Prompt Builder: `llm/prompt_builder.py` (tier-aware templates)
- Database schema with 7 tables, compression, retention policies
- API endpoints: health, indicators, snapshots, debug
- Config system: YAML-based with env var substitution
**Phase 2 - Core Agents:**
- Base Agent Class: `agents/base_agent.py` (abstract interface, AgentOutput dataclass)
- Technical Analysis Agent: `agents/technical_analysis.py` (Qwen 2.5 7B via Ollama)
- Regime Detection Agent: `agents/regime_detection.py` (7 regime types)
- Risk Management Engine: `risk/rules_engine.py` (rules-based, <10ms, circuit breakers)
- Trading Decision Agent: `agents/trading_decision.py` (6-model A/B testing)
- LLM Clients: `llm/clients/` (Ollama, OpenAI, Anthropic, DeepSeek, xAI)
- API Endpoints: Agent invoke, risk state, model comparison routes
- Database Migration: model_comparisons table
- 689 tests passing, 90% coverage
## Phase 3 Deliverables
### 3.1 Agent Communication Protocol (`triplegain/src/orchestration/message_bus.py`)
- In-memory message bus with pub/sub pattern
- Thread-safe async implementation
- Message topics:
  - market_data, ta_signals, regime_updates, sentiment_updates
  - trading_signals, risk_alerts, execution_events, portfolio_updates
- Message schema: id, timestamp, topic, source, priority, payload, correlation_id, ttl
- Message priority: LOW, NORMAL, HIGH, URGENT
- Subscription filtering with optional filter functions
- Message history with TTL cleanup
### 3.2 Coordinator Agent (`triplegain/src/orchestration/coordinator.py`)
- Model: DeepSeek V3 (primary), Claude Sonnet (fallback) - for conflict resolution only
- Orchestrates all agent execution via scheduled tasks:
  - TA Agent: Every minute
  - Regime Agent: Every 5 minutes
  - Sentiment Agent: Every 30 minutes (Phase 4, disabled)
  - Trading Decision: Every hour
  - Portfolio Check: Every hour
- Conflict detection and LLM-based resolution:
  - TA vs Sentiment conflicts (confidence diff < 0.2)
  - Regime conflicts (trading in choppy market)
  - Risk engine modifications
- Emergency handling: circuit breaker responses, execution errors
- State management: RUNNING, PAUSED, HALTED
### 3.3 Portfolio Rebalancing Agent (`triplegain/src/agents/portfolio_rebalance.py`)
- Model: DeepSeek V3
- Target allocation: 33/33/33 BTC/XRP/USDT
- Rebalancing threshold: 5% deviation
- Hodl bag exclusion from calculations
- Output: PortfolioAllocation, RebalanceTrade list
- Determines execution strategy (market vs limit, sequencing)
### 3.4 Order Execution Manager (`triplegain/src/execution/`)
**NOT an LLM agent - purely rule-based**
- `order_manager.py` - Order lifecycle management
- `position_tracker.py` - Open position tracking
- Order types: MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT
- Order states: PENDING → OPEN → FILLED/CANCELLED/EXPIRED
- Contingent orders: Stop loss and take profit placed after fill
- Position management: P&L tracking, modification, closing
- Kraken API integration with rate limiting
### 3.5 Configuration Files
- `config/orchestration.yaml` - Coordinator schedules, conflict settings
- `config/portfolio.yaml` - Target allocation, rebalancing settings, hodl bags
- `config/execution.yaml` - Kraken API, order settings, position limits
### 3.6 Database Migration
```sql
-- Order status tracking
CREATE TABLE order_status_log (
    id SERIAL PRIMARY KEY,
    order_id UUID NOT NULL,
    external_id VARCHAR(50),
    status VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    details JSONB
);
-- Position real-time tracking (TimescaleDB hypertable)
CREATE TABLE position_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    position_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    current_price DECIMAL(20, 10),
    unrealized_pnl DECIMAL(20, 10),
    unrealized_pnl_pct DECIMAL(10, 4),
    PRIMARY KEY (timestamp, position_id)
);
```
### 3.7 API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/v1/coordinator/status | GET | Get coordinator state |
| /api/v1/coordinator/pause | POST | Pause trading |
| /api/v1/coordinator/resume | POST | Resume trading |
| /api/v1/portfolio/allocation | GET | Get current allocation |
| /api/v1/portfolio/rebalance | POST | Force rebalancing |
| /api/v1/positions | GET | Get open positions |
| /api/v1/positions/{id}/close | POST | Close position |
| /api/v1/orders | GET | Get open orders |
| /api/v1/orders/{id}/cancel | POST | Cancel order |
## Constraints
- Reuse ALL existing Phase 1 & 2 code - do not recreate
- API keys are in `.env`, referenced via `config/agents.yaml` (OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, XAI_BEARER_API_KEY)
- All agent outputs validated against JSON schemas
- Store all outputs to database
- Test-first approach (pytest) - maintain 90%+ coverage
- Coordinator LLM invoked only on conflicts (not for scheduling)
- Order Execution Manager is NOT an LLM agent
- Handle LLM timeouts gracefully with fallbacks
- Message bus must be thread-safe for async operations
## LLM Model Configuration
| Component | Model | Provider | When Invoked |
|-----------|-------|----------|--------------|
| Coordinator | DeepSeek V3 | DeepSeek | On conflict only |
| Coordinator (fallback) | Claude Sonnet | Anthropic | DeepSeek failure |
| Portfolio Rebalance | DeepSeek V3 | DeepSeek | On rebalance decision |
| TA Agent | Qwen 2.5 7B | Ollama | Every minute |
| Regime Agent | Qwen 2.5 7B | Ollama | Every 5 minutes |
| Trading Decision | 6-Model A/B | All | Hourly |
## Starting Instructions
1. Create `triplegain/src/orchestration/` directory
2. Implement `message_bus.py` with Message, MessageTopic, MessageBus classes
3. Add unit tests for message bus (pub/sub, filtering, TTL cleanup)
4. Implement `coordinator.py` with ScheduledTask, CoordinatorAgent classes
5. Create `config/orchestration.yaml` with schedules and conflict settings
6. Implement `triplegain/src/agents/portfolio_rebalance.py`
7. Create `config/portfolio.yaml` with allocation and rebalancing settings
8. Create `triplegain/src/execution/` directory
9. Implement `order_manager.py` with Order, OrderStatus, OrderExecutionManager
10. Implement `position_tracker.py` with Position tracking
11. Create `config/execution.yaml` with Kraken settings
12. Create database migration `migrations/003_phase3_orchestration.sql`
13. Add API routes in `triplegain/src/api/routes/` for coordinator, portfolio, positions, orders
14. Write integration tests for full agent pipeline
15. Ensure all tests pass and coverage remains at 90%+
## Acceptance Criteria
- [ ] Message bus delivers messages to all subscribers
- [ ] Coordinator executes scheduled tasks at correct intervals
- [ ] Coordinator detects and resolves conflicts using LLM
- [ ] Portfolio rebalancing calculates correct trades
- [ ] Order execution manager places orders on Kraken
- [ ] Position tracker maintains accurate position state
- [ ] All API endpoints functional
- [ ] 90%+ test coverage maintained
- [ ] Integration test: TA → Regime → Trading → Risk → Execution pipeline