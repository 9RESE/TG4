# Phase 3: Orchestration - Feature Documentation

**Version**: 1.4
**Status**: COMPLETE (with all enhancements and review fixes)
**Date**: 2025-12-19

## Overview

Phase 3 implements the orchestration layer for TripleGain, enabling agents to work together cohesively. This includes inter-agent communication, coordinated execution, portfolio management, and order execution.

## Components Implemented

### 1. Message Bus

**Location**: `triplegain/src/orchestration/message_bus.py`

The message bus provides pub/sub communication between agents:

```python
class MessageBus:
    async def publish(message: Message) -> int: ...
    async def subscribe(subscriber_id, topic, handler, filter_fn=None): ...
    async def unsubscribe(subscriber_id, topic=None): ...
    async def get_latest(topic, source=None) -> Optional[Message]: ...
```

**Message Topics**:
| Topic | Publisher | Subscribers |
|-------|-----------|-------------|
| `MARKET_DATA` | Data pipeline | All agents |
| `TA_SIGNALS` | TA Agent | Trading Agent, Coordinator |
| `REGIME_UPDATES` | Regime Agent | Trading Agent, Coordinator |
| `SENTIMENT_UPDATES` | Sentiment Agent | Trading Agent, Coordinator |
| `TRADING_SIGNALS` | Trading Agent | Coordinator, Risk Engine |
| `RISK_ALERTS` | Risk Engine | Coordinator |
| `EXECUTION_EVENTS` | Order Manager | Coordinator, Portfolio |
| `PORTFOLIO_UPDATES` | Portfolio Agent | Coordinator |
| `SYSTEM_EVENTS` | All | All |

**Message Structure**:
```python
@dataclass
class Message:
    id: str                          # UUID
    timestamp: datetime              # UTC timestamp
    topic: MessageTopic              # Routing topic
    source: str                      # Publisher agent name
    priority: MessagePriority        # LOW, NORMAL, HIGH, URGENT
    payload: dict                    # Message data
    correlation_id: Optional[str]    # For request-response
    ttl_seconds: int                 # Time-to-live (default 300)
```

**Features**:
- Thread-safe with `asyncio.Lock`
- TTL-based message expiration
- Optional filter functions per subscription
- Message history for `get_latest()` queries

### 2. Coordinator Agent

**Location**: `triplegain/src/orchestration/coordinator.py`

The coordinator orchestrates agent execution and resolves conflicts:

**LLM Configuration**:
| Role | Model | Invocation |
|------|-------|------------|
| Primary | DeepSeek V3 | On conflict |
| Fallback | Claude Sonnet | When primary fails |

**Coordinator States**:
- `RUNNING`: Normal operation, executing schedules
- `PAUSED`: Schedules run, trading signals ignored
- `HALTED`: Circuit breaker triggered, all trading stopped

**Degradation Levels** (v1.1):
- `NORMAL`: All systems operational
- `REDUCED`: Skip non-critical agents (sentiment)
- `LIMITED`: Skip optional agents, reduce LLM calls
- `EMERGENCY`: Only risk-based decisions, no LLM

**State Persistence** (v1.1):
- Coordinator state automatically saved to database on stop
- Statistics, task schedules, and enabled/disabled state restored on start
- Stored in `coordinator_state` table

**Consensus Building** (v1.1):
When trading signals are received, coordinator builds consensus from:
- TA Agent agreement (trend direction)
- Regime Agent agreement (favorable regime)
- Sentiment Agent agreement (bullish/bearish)

Confidence multiplier applied based on agreement ratio:
- 66%+ agreement: 1.0-1.3x confidence boost
- 33-66% agreement: 1.0x (neutral)
- <33% agreement: 0.85-1.0x confidence reduction

**Scheduled Tasks**:
| Task | Agent | Interval | Symbols |
|------|-------|----------|---------|
| `ta_analysis` | Technical Analysis | 60s | BTC/USDT, XRP/USDT |
| `regime_detection` | Regime Detection | 300s | BTC/USDT, XRP/USDT |
| `sentiment_analysis` | Sentiment | 1800s | BTC/USDT, XRP/USDT |
| `trading_decision` | Trading Decision | 3600s | BTC/USDT, XRP/USDT |
| `portfolio_check` | Portfolio Rebalance | 3600s | PORTFOLIO |

**Conflict Detection**:
```python
async def _detect_conflicts(signal: dict) -> list[ConflictInfo]:
    # 1. TA vs Sentiment disagreement
    # 2. Regime incompatibility (choppy + trade signal)
    # 3. Close confidence values (< 0.2 difference)
```

**Conflict Resolution**:
```python
@dataclass
class ConflictResolution:
    action: str        # "proceed", "wait", "modify", "abort"
    confidence: float  # Resolution confidence
    reasoning: str     # Explanation
    modifications: Optional[dict]  # Parameter adjustments
```

### 3. Portfolio Rebalancing Agent

**Location**: `triplegain/src/agents/portfolio_rebalance.py`

Maintains target portfolio allocation with hodl bag protection:

**Target Allocation**:
| Asset | Target % |
|-------|----------|
| BTC | 33.33% |
| XRP | 33.33% |
| USDT | 33.34% |

**Configuration**:
```yaml
portfolio:
  target_allocation:
    btc_pct: 33.33
    xrp_pct: 33.33
    usdt_pct: 33.34

  rebalancing:
    threshold_pct: 5.0      # Trigger when deviation exceeds
    min_trade_usd: 10.0     # Minimum trade size
    execution_type: limit   # Default order type
    dca:
      enabled: true
      threshold_usd: 500    # DCA for trades > $500
      batches: 6            # Split into 6 batches
      interval_hours: 4     # 4 hours between batches

  hodl_bags:
    enabled: true
    allocation_pct: 10      # % of profits to hodl
```

**DCA (Dollar Cost Averaging) Execution** (v1.2):
For large rebalances exceeding the threshold (default $500):
- Trades split into configurable number of batches (default 6)
- Each batch executed at scheduled intervals (default 4 hours)
- Reduces market impact and slippage for large trades
- Scheduled trades stored in database for persistence

**Output Schema**:
```python
@dataclass
class RebalanceOutput(AgentOutput):
    action: str                      # "no_action" or "rebalance"
    current_allocation: PortfolioAllocation
    trades: list[RebalanceTrade]     # Required trades
    execution_strategy: str          # LLM-determined strategy
    reasoning: str
```

**Rebalancing Logic**:
1. Get current balances (excluding hodl bags)
2. Calculate current allocation percentages
3. Compare to target allocation
4. If max deviation > threshold (5%), generate trades
5. LLM determines execution strategy (limit/market, sequencing)
6. Execute sells first (fund availability), then buys

**Trade Execution Routing** (v1.1):
- Coordinator automatically routes rebalance trades to execution manager
- Each trade validated with risk engine before execution
- `rebalance_trade_executed` events published to message bus

### 4. Order Execution Manager

**Location**: `triplegain/src/execution/order_manager.py`

Handles order lifecycle on Kraken exchange:

**Order Types**:
| Type | Description |
|------|-------------|
| `MARKET` | Execute immediately at market price |
| `LIMIT` | Execute at specified price or better |
| `STOP_LOSS` | Trigger sell when price drops to level |
| `TAKE_PROFIT` | Trigger sell when price rises to level |

**Order States**:
```
PENDING → OPEN → FILLED
                → PARTIALLY_FILLED → FILLED
                → CANCELLED
                → EXPIRED
                → ERROR
```

**Symbol Mapping** (Internal → Kraken):
| Internal | Kraken |
|----------|--------|
| BTC/USDT | XBTUSDT |
| XRP/USDT | XRPUSDT |

**Features**:
- Retry with exponential backoff
- Contingent order placement (SL/TP after fill)
- Order monitoring with state transitions
- Mock mode for testing (no Kraken client)

**Rate Limiting** (v1.1):
Token bucket algorithm for API call throttling:
```python
class TokenBucketRateLimiter:
    # General API: 60 calls/min, 10-token burst
    # Order API: 30 calls/min, 5-token burst
    async def acquire(tokens: int = 1) -> float  # Returns wait time
```

Applied to: `_place_order()`, `_monitor_order()`, `cancel_order()`, `sync_with_exchange()`

**Input Validation** (v1.1):
- Trade size validation (must be > 0)
- Calculated order size validation
- Returns `ExecutionResult` with error message for invalid inputs

**Order History Cleanup** (v1.1):
- Configurable `max_history_size` (default: 1000)
- Automatic cleanup when limit exceeded
- Separate lock for history to avoid race conditions

**Position Limits Enforcement** (v1.2):
Before placing any order, the system checks:
```python
position_limits:
  max_per_symbol: 2    # Max 2 open positions per symbol
  max_total: 5         # Max 5 total open positions
```
- Returns `ExecutionResult` with error message when limits exceeded
- Only enforced for buy orders (position opening)
- Graceful handling when position tracker unavailable

**Mock Mode Price Improvements** (v1.2):
- Uses position tracker's price cache for current market prices
- Symbol-specific fallback prices (BTC: $45000, XRP: $0.60, etc.)
- More realistic paper trading simulation

**Key Methods**:
```python
async def execute_trade(proposal: TradeProposal) -> ExecutionResult
async def cancel_order(order_id: str) -> bool
async def get_open_orders(symbol: Optional[str]) -> list[Order]
async def sync_with_exchange() -> int  # Returns synced count
```

### 5. Position Tracker

**Location**: `triplegain/src/execution/position_tracker.py`

Tracks open positions with P&L calculation:

**Position Structure**:
```python
@dataclass
class Position:
    id: str
    symbol: str
    side: str              # "long" or "short"
    entry_price: Decimal
    size: Decimal
    leverage: int
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal
    status: str            # "open", "closed"
```

**P&L Calculation**:
```python
# Long position
unrealized_pnl = (current_price - entry_price) * size * leverage

# Short position
unrealized_pnl = (entry_price - current_price) * size * leverage
```

**Features**:
- Real-time P&L updates
- Position snapshots for time-series
- Risk engine integration for exposure updates

**Automatic SL/TP Monitoring** (v1.1):
```python
async def check_sl_tp_triggers(current_prices: dict[str, Decimal]) -> list[tuple[Position, str]]:
    # Checks all open positions against current prices
    # Returns list of (position, trigger_type) where trigger_type is 'stop_loss' or 'take_profit'
    # Correctly handles both LONG and SHORT positions
```

Integrated into snapshot loop - automatically closes triggered positions.

**Position Validation** (v1.1):
```python
def __post_init__(self):
    # Validates on creation:
    # - leverage: 1-5 (system max)
    # - size: must be > 0
    # - entry_price: must be >= 0
```

**Decimal Precision** (v1.1):
- All Decimal values stored as strings in database
- Preserves full precision through database round-trip

**Trailing Stops** (v1.2):
Dynamic stop-loss that follows favorable price movements:
```yaml
trailing_stop:
  enabled: true
  activation_pct: 1.0      # Activate when profit >= 1%
  trail_distance_pct: 1.5  # Trail 1.5% behind highest/lowest
  update_interval_seconds: 60
```

**Trailing Stop Logic**:
- **Activation**: Stop activates when unrealized profit reaches threshold
- **LONG positions**: Tracks highest price, stop at (highest - trail_distance%)
- **SHORT positions**: Tracks lowest price, stop at (lowest + trail_distance%)
- **Stop-loss update**: Only moves in favorable direction (never against trade)
- Integrated into snapshot loop for automatic updates

```python
# Enable trailing stop for specific position
position_tracker.enable_trailing_stop_for_position(
    position_id="pos-123",
    distance_pct=Decimal("2.0")  # Optional custom distance
)
```

### 6. API Validation Module

**Location**: `triplegain/src/api/validation.py`

Provides centralized validation utilities for API endpoints:

```python
# Symbol format validation and normalization
def validate_symbol(symbol: str, strict: bool = True) -> tuple[bool, str]:
    # Accepts both BTC/USDT and BTC_USDT formats
    # Returns (is_valid, error_message)

def validate_symbol_or_raise(symbol: str, strict: bool = True) -> str:
    # Validates and normalizes symbol (e.g., BTC_USDT -> BTC/USDT)
    # Raises HTTPException(400) if invalid

def normalize_symbol(symbol: str) -> str:
    # Converts BTC_USDT -> BTC/USDT
```

**Supported Symbols**:
- BTC/USDT, XRP/USDT, XRP/BTC, ETH/USDT, ETH/BTC

**Features**:
- Accepts both slash and underscore separators for URL compatibility
- Normalizes all symbols to standard slash format internally
- Strict mode enforces supported symbols only

### 7. API Routes

**Location**: `triplegain/src/api/routes_orchestration.py`

**Coordinator Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/coordinator/status` | GET | Get coordinator state and statistics |
| `/coordinator/pause` | POST | Pause trading |
| `/coordinator/resume` | POST | Resume trading |
| `/coordinator/task/{name}/run` | POST | Force run specific task |
| `/coordinator/task/{name}/enable` | POST | Enable scheduled task |
| `/coordinator/task/{name}/disable` | POST | Disable scheduled task |

**Portfolio Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/portfolio/allocation` | GET | Get current allocation |
| `/portfolio/rebalance` | POST | Force portfolio rebalance |

**Position Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/positions` | GET | List positions (filter by status/symbol) |
| `/positions/{id}` | GET | Get specific position |
| `/positions/{id}/close` | POST | Close position (v1.2: exit_price in body) |
| `/positions/{id}` | PATCH | Modify SL/TP |
| `/positions/exposure` | GET | Get total exposure |

**Close Position Request Body** (v1.2):
```json
{
  "exit_price": 46000.0,
  "reason": "take_profit"
}
```

**Order Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/orders` | GET | List open orders |
| `/orders/{id}` | GET | Get specific order |
| `/orders/{id}/cancel` | POST | Cancel order |
| `/orders/sync` | POST | Sync with exchange |

**Statistics Endpoint**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stats/execution` | GET | Execution statistics |

## Configuration Files

### config/orchestration.yaml
```yaml
coordinator:
  llm:
    primary:
      provider: deepseek
      model: deepseek-chat
    fallback:
      provider: anthropic
      model: claude-3-5-sonnet-20241022

  symbols:
    - BTC/USDT
    - XRP/USDT

  schedules:
    technical_analysis:
      interval_seconds: 60
      enabled: true
    regime_detection:
      interval_seconds: 300
      enabled: true
    sentiment_analysis:
      interval_seconds: 1800
      enabled: false  # Phase 4
    trading_decision:
      interval_seconds: 3600
      enabled: true
    portfolio_rebalance:
      interval_seconds: 3600
      enabled: true

  conflicts:
    invoke_llm_threshold: 0.2
    max_resolution_time_ms: 10000
```

### config/portfolio.yaml
```yaml
portfolio:
  target_allocation:
    btc_pct: 33.33
    xrp_pct: 33.33
    usdt_pct: 33.34

  rebalancing:
    threshold_pct: 5.0
    min_trade_usd: 10.0
    execution_type: limit
    check_interval_seconds: 3600

  hodl_bags:
    enabled: true
    btc_amount: 0
    xrp_amount: 0
    usdt_amount: 0
```

### config/execution.yaml
```yaml
execution:
  kraken:
    api_key: ${KRAKEN_API_KEY}
    api_secret: ${KRAKEN_API_SECRET}
    rate_limit:
      calls_per_minute: 60
      order_calls_per_minute: 30

  orders:
    default_type: limit
    time_in_force: GTC
    max_retry_count: 3
    retry_delay_seconds: 5

  positions:
    sync_interval_seconds: 30
    max_open_positions: 6
```

## Database Migration

**File**: `migrations/003_phase3_orchestration.sql`

**Tables Created**:
| Table | Purpose |
|-------|---------|
| `order_status_log` | Order state history |
| `positions` | Open/closed positions |
| `position_snapshots` | Time-series P&L (hypertable) |
| `hodl_bags` | Hodl bag balances |
| `hodl_bag_history` | Hodl bag changes |
| `coordinator_state` | Coordinator persistence |
| `rebalancing_history` | Rebalancing trades |
| `conflict_resolution_log` | Conflict decisions |
| `execution_events` | Execution audit log |
| `scheduled_trades` | DCA scheduled trades (v1.2) |

## Testing

**Test Files**:
- `tests/unit/orchestration/test_message_bus.py` - 52 tests
- `tests/unit/orchestration/test_coordinator.py` - 62 tests
- `tests/unit/execution/test_order_manager.py` - 36 tests
- `tests/unit/execution/test_position_tracker.py` - 34 tests
- `tests/unit/agents/test_portfolio_rebalance.py` - 30 tests
- `tests/unit/api/test_routes_orchestration.py` - 43 tests

**Total Phase 3 Tests**: 257 tests
**Total Project Tests**: 916
**Coverage**: 87%

## Integration Points

### With Phase 1
- **MarketSnapshot**: Data source for agents
- **IndicatorLibrary**: Technical indicators
- **Database**: Persists state and history

### With Phase 2
- **BaseAgent**: Portfolio agent inherits from this
- **Risk Engine**: Validates trades before execution
- **LLM Clients**: DeepSeek/Anthropic for conflict resolution
- **Trading Agent**: Produces signals for coordinator

### With Phase 4 (Future)
- **Sentiment Agent**: Will integrate with coordinator
- **Dashboard**: Will consume API endpoints
- **A/B Testing Analytics**: Will use model comparison data

## Design Decisions

1. **In-Memory Message Bus**: Simple, fast, sufficient for single-process deployment.

2. **LLM Conflict Resolution**: Only invoked when conflicts detected, optimizes API costs.

3. **Portfolio Rebalancing**: 5% threshold balances trading costs vs allocation drift.

4. **Non-LLM Execution**: Order execution must be deterministic and fast.

5. **Mock Mode**: All components work without Kraken client for testing.

## References

- [Phase 3 Implementation Plan](../TripleGain-implementation-plan/03-phase-3-orchestration.md)
- [ADR-004: Phase 3 Architecture](../../architecture/09-decisions/ADR-004-phase3-orchestration-architecture.md)
- [Multi-Agent Architecture](../TripleGain-master-design/01-multi-agent-architecture.md)
- [Deep Code Review](../reviews/phase-3/phase-3-deep-code-review.md)
- [Fixes Implemented](../reviews/phase-3/phase-3-fixes-implemented.md)
- [Follow-Up Review](../reviews/phase-3/phase-3-follow-up-review.md)
- [Comprehensive Code Review](../reviews/full/triplegain-comprehensive-code-review.md)

## Changelog

### v1.4 (2025-12-19) - Security & Robustness Fixes
- **API Security**: All 20+ endpoints now return generic error messages instead of exposing stack traces
- **Validation Module**: New `triplegain/src/api/validation.py` with centralized symbol validation
- **Conflict Resolution Timeout**: Added `asyncio.wait_for()` to enforce max resolution time
- **Degradation Recovery Events**: System now publishes events when recovering from degraded state
- **Token Estimation Safety Margin**: Added 10% safety margin to account for BPE encoding variations
- **DCA Batch Rounding**: Batches now sum correctly to original total with remainder handling
- **DCA Sub-Minimum Trades**: Batch count auto-reduces when individual batches would be too small
- **Target Allocation Validation**: Warns when portfolio allocations don't sum to 100%
- **Hodl Bag Warning**: Logs warning when hodl bags exceed available balance
- **Zero Equity Handling**: Returns target allocation when equity is zero to prevent false rebalancing
- **LLM Fallback Transparency**: Added `used_fallback_strategy` field to indicate default usage

### v1.3 (2025-12-19) - Comprehensive Review Fixes
- **Supertrend Fix**: Initial direction now uses midpoint (hl2) instead of upper_band for accuracy
- **Async Error Handling**: Added failure detection threshold in market snapshot builder (max 50% failures)
- **Truncation Logging**: Added warning log when prompt content is truncated
- **Type Coercion**: Config loader now auto-converts string values to int/float/bool
- **Symbol Validation**: API endpoints now validate symbol format (BASE/QUOTE or BASE_QUOTE)
- **Timeframe Validation**: API endpoints now validate timeframe against allowed values

### v1.2 (2025-12-19) - Enhancements
- **DCA Execution**: Large rebalances (>$500) split into batches over 24h
- **Trailing Stops**: Dynamic stop-loss that follows favorable price movements
- **Position Limits**: Enforces max positions per symbol and total
- **Mock Mode Prices**: Uses price cache for realistic paper trading
- **API Improvement**: Exit price moved to request body for RESTful consistency

### v1.1 (2025-12-19) - Deep Review Fixes
- **Coordinator**: Added consensus building, state persistence, graceful degradation
- **Order Manager**: Added token bucket rate limiting, input validation, history cleanup
- **Position Tracker**: Added automatic SL/TP monitoring, position validation, Decimal precision
- **Portfolio Agent**: Added trade execution routing through coordinator

### v1.0 (2025-12-18) - Initial Release
- Message Bus with pub/sub pattern
- Coordinator Agent with conflict resolution
- Portfolio Rebalancing Agent
- Order Execution Manager
- Position Tracker
- API Routes

---

*Phase 3 Feature Documentation v1.4 - December 2025*
