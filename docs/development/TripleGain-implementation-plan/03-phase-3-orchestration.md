# Phase 3: Orchestration

**Phase Status**: ✅ COMPLETE
**Completion Date**: 2025-12-18
**Dependencies**: Phase 2 (All Core Agents) ✅
**Deliverable**: Agents working together, executing trades

---

## Overview

Phase 3 integrates all agents into a cohesive system that can execute actual trades. This phase implements the communication layer, coordinator logic, portfolio rebalancing, and order execution management.

### Components

| Component | Description | Depends On |
|-----------|-------------|------------|
| 3.1 Agent Communication Protocol | Inter-agent message passing | All Phase 2 agents |
| 3.2 Coordinator Agent | Orchestrates agent execution | 3.1 |
| 3.3 Portfolio Rebalancing Agent | Maintains 33/33/33 allocation | 3.1, 3.2 |
| 3.4 Order Execution Manager | Executes trades on Kraken | 3.2, Phase 2 Risk Engine |

---

## 3.1 Agent Communication Protocol

### Purpose

Standardized messaging protocol for agents to share outputs, coordinate decisions, and handle conflicts.

### Message Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT COMMUNICATION ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       MESSAGE BUS (In-Memory)                        │   │
│  │                                                                      │   │
│  │  Topics:                                                            │   │
│  │  • market_data       - Market snapshots                             │   │
│  │  • ta_signals        - Technical analysis outputs                   │   │
│  │  • regime_updates    - Regime classification changes                │   │
│  │  • sentiment_updates - Sentiment analysis outputs                   │   │
│  │  • trading_signals   - Trading decision outputs                     │   │
│  │  • risk_alerts       - Risk management notifications                │   │
│  │  • execution_events  - Order execution updates                      │   │
│  │  • portfolio_updates - Portfolio state changes                      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Publishers:                          Subscribers:                          │
│  • TA Agent → ta_signals              • Trading Agent ← ta_signals         │
│  • Regime Agent → regime_updates      • Trading Agent ← regime_updates     │
│  • Sentiment Agent → sentiment        • Trading Agent ← sentiment          │
│  • Trading Agent → trading_signals    • Risk Engine ← trading_signals      │
│  • Risk Engine → risk_alerts          • Execution ← trading_signals        │
│  • Execution → execution_events       • Portfolio ← execution_events       │
│  • Portfolio → portfolio_updates      • Coordinator ← all                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Message Schema

```python
# src/orchestration/message_bus.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum
import uuid

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageTopic(Enum):
    MARKET_DATA = "market_data"
    TA_SIGNALS = "ta_signals"
    REGIME_UPDATES = "regime_updates"
    SENTIMENT_UPDATES = "sentiment_updates"
    TRADING_SIGNALS = "trading_signals"
    RISK_ALERTS = "risk_alerts"
    EXECUTION_EVENTS = "execution_events"
    PORTFOLIO_UPDATES = "portfolio_updates"
    SYSTEM_EVENTS = "system_events"


@dataclass
class Message:
    """Standard message format for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    topic: MessageTopic = MessageTopic.SYSTEM_EVENTS
    source: str = ""  # Agent name
    priority: MessagePriority = MessagePriority.NORMAL
    payload: dict = field(default_factory=dict)
    correlation_id: Optional[str] = None  # For request-response patterns
    ttl_seconds: int = 300  # Time-to-live

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "topic": self.topic.value,
            "source": self.source,
            "priority": self.priority.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "ttl_seconds": self.ttl_seconds
        }


@dataclass
class Subscription:
    """Subscription to message topic."""
    subscriber_id: str
    topic: MessageTopic
    handler: callable
    filter_fn: Optional[callable] = None  # Optional message filter


class MessageBus:
    """
    In-memory message bus for agent communication.

    Thread-safe, supports pub/sub pattern.
    """

    def __init__(self, config: dict):
        self.config = config
        self._subscriptions: dict[MessageTopic, list[Subscription]] = {}
        self._message_history: list[Message] = []
        self._lock = asyncio.Lock()

    async def publish(self, message: Message) -> None:
        """
        Publish message to topic.

        Args:
            message: Message to publish
        """
        async with self._lock:
            # Store in history
            self._message_history.append(message)
            self._cleanup_expired()

            # Get subscriptions for topic
            subscriptions = self._subscriptions.get(message.topic, [])

            # Notify subscribers
            for sub in subscriptions:
                if sub.filter_fn is None or sub.filter_fn(message):
                    try:
                        await sub.handler(message)
                    except Exception as e:
                        # Log error but don't fail other subscribers
                        pass

    async def subscribe(
        self,
        subscriber_id: str,
        topic: MessageTopic,
        handler: callable,
        filter_fn: Optional[callable] = None
    ) -> None:
        """
        Subscribe to topic.

        Args:
            subscriber_id: Unique subscriber identifier
            topic: Topic to subscribe to
            handler: Async function to call on message
            filter_fn: Optional filter function
        """
        async with self._lock:
            if topic not in self._subscriptions:
                self._subscriptions[topic] = []

            sub = Subscription(
                subscriber_id=subscriber_id,
                topic=topic,
                handler=handler,
                filter_fn=filter_fn
            )
            self._subscriptions[topic].append(sub)

    async def unsubscribe(
        self,
        subscriber_id: str,
        topic: Optional[MessageTopic] = None
    ) -> None:
        """Unsubscribe from topic(s)."""
        pass

    async def get_latest(
        self,
        topic: MessageTopic,
        source: Optional[str] = None
    ) -> Optional[Message]:
        """Get latest message for topic."""
        async with self._lock:
            for msg in reversed(self._message_history):
                if msg.topic == topic:
                    if source is None or msg.source == source:
                        return msg
            return None

    def _cleanup_expired(self) -> None:
        """Remove expired messages from history."""
        now = datetime.utcnow()
        self._message_history = [
            m for m in self._message_history
            if (now - m.timestamp).total_seconds() < m.ttl_seconds
        ]
```

### Agent Message Types

```yaml
# Message payload schemas by topic

message_schemas:
  ta_signals:
    required_fields:
      - symbol
      - trend_direction
      - trend_strength
      - bias
      - confidence
    example:
      symbol: "BTC/USDT"
      trend_direction: "bullish"
      trend_strength: 0.72
      momentum_score: 0.45
      bias: "long"
      confidence: 0.68

  regime_updates:
    required_fields:
      - symbol
      - regime
      - confidence
      - position_size_multiplier
    example:
      symbol: "BTC/USDT"
      regime: "trending_bull"
      confidence: 0.78
      position_size_multiplier: 1.0
      entry_strictness: "normal"

  trading_signals:
    required_fields:
      - symbol
      - action
      - confidence
    example:
      symbol: "BTC/USDT"
      action: "BUY"
      confidence: 0.72
      entry_price: 45250.0
      stop_loss_pct: 2.0
      take_profit_pct: 4.0
      leverage: 2

  risk_alerts:
    required_fields:
      - alert_type
      - severity
      - message
    example:
      alert_type: "circuit_breaker"
      severity: "high"
      message: "Daily loss limit reached"
      action_required: "halt_trading"

  execution_events:
    required_fields:
      - event_type
      - order_id
    example:
      event_type: "order_filled"
      order_id: "ord_123"
      symbol: "BTC/USDT"
      side: "buy"
      size: 0.05
      fill_price: 45250.0
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Pub/sub basic | Publish reaches subscriber | Message delivered |
| Multi-subscriber | Multiple subscribers receive | All receive |
| Filter function | Filter blocks message | Not delivered |
| TTL expiration | Old messages cleaned | Memory stable |
| Thread safety | Concurrent pub/sub | No race conditions |

---

## 3.2 Coordinator Agent

### Purpose

Orchestrates agent execution, resolves conflicts between agent outputs, and manages the overall trading workflow.

### LLM Assignment

| Property | Value |
|----------|-------|
| Primary Model | DeepSeek V3 |
| Fallback Model | Claude Sonnet |
| Invocation | On conflict / On demand |
| Latency Target | < 5s |
| Tier | Tier 2 (API) |

### Coordination Logic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COORDINATOR DECISION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SCHEDULED EXECUTION                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Every minute:                                                        │   │
│  │   → Trigger TA Agent for all symbols                                │   │
│  │   → Publish ta_signals                                              │   │
│  │                                                                      │   │
│  │ Every 5 minutes:                                                    │   │
│  │   → Trigger Regime Agent (uses latest TA)                           │   │
│  │   → Publish regime_updates                                          │   │
│  │                                                                      │   │
│  │ Every 30 minutes:                                                   │   │
│  │   → Trigger Sentiment Agent (Grok + GPT)                            │   │
│  │   → Publish sentiment_updates                                       │   │
│  │                                                                      │   │
│  │ Every hour:                                                          │   │
│  │   → Trigger Trading Decision Agent (6-model)                        │   │
│  │   → Publish trading_signals                                         │   │
│  │   → If action != HOLD: → Risk Validation → Execution                │   │
│  │                                                                      │   │
│  │ Every hour:                                                          │   │
│  │   → Check portfolio allocation                                       │   │
│  │   → If deviation > threshold: → Portfolio Rebalancing               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. CONFLICT RESOLUTION (Coordinator LLM invoked)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ When invoked:                                                        │   │
│  │   • TA says LONG, Sentiment says BEARISH (confidence diff < 0.2)   │   │
│  │   • Regime change during open position                              │   │
│  │   • Risk engine modified trade significantly                        │   │
│  │   • Multiple symbols have conflicting signals                       │   │
│  │                                                                      │   │
│  │ Coordinator decides:                                                 │   │
│  │   • Which signal to prioritize                                      │   │
│  │   • Whether to proceed or wait                                      │   │
│  │   • How to adjust parameters                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. EMERGENCY HANDLING                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ On risk_alert (circuit_breaker):                                     │   │
│  │   → Halt scheduled trading                                          │   │
│  │   → Optionally close positions (per configuration)                  │   │
│  │   → Notify dashboard                                                 │   │
│  │                                                                      │   │
│  │ On execution_error:                                                  │   │
│  │   → Retry with backoff                                              │   │
│  │   → If persistent: alert and pause symbol                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Interface Definition

```python
# src/orchestration/coordinator.py

from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

class CoordinatorState(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"  # Circuit breaker triggered


@dataclass
class ScheduledTask:
    """Scheduled agent invocation."""
    name: str
    agent: str
    interval_seconds: int
    symbols: list[str]
    handler: Callable
    last_run: Optional[datetime] = None
    enabled: bool = True


@dataclass
class ConflictResolution:
    """Result of coordinator conflict resolution."""
    action: str  # "proceed", "wait", "modify", "abort"
    reasoning: str
    modifications: Optional[dict] = None
    confidence: float = 0.0


class CoordinatorAgent:
    """
    Orchestrates agent execution and resolves conflicts.

    Uses DeepSeek V3 / Claude Sonnet for conflict resolution.
    """

    agent_name = "coordinator"

    def __init__(
        self,
        message_bus: MessageBus,
        agents: dict,  # name -> agent instance
        llm_client,
        config: dict
    ):
        self.bus = message_bus
        self.agents = agents
        self.llm = llm_client
        self.config = config
        self._state = CoordinatorState.RUNNING
        self._scheduled_tasks: list[ScheduledTask] = []
        self._setup_schedules()
        self._setup_subscriptions()

    async def start(self) -> None:
        """Start coordinator main loop."""
        while True:
            if self._state == CoordinatorState.RUNNING:
                await self._execute_due_tasks()
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop coordinator."""
        self._state = CoordinatorState.HALTED

    async def pause(self) -> None:
        """Pause trading (scheduled tasks still run for analysis)."""
        self._state = CoordinatorState.PAUSED

    async def resume(self) -> None:
        """Resume from paused state."""
        self._state = CoordinatorState.RUNNING

    async def _execute_due_tasks(self) -> None:
        """Execute tasks that are due."""
        now = datetime.utcnow()

        for task in self._scheduled_tasks:
            if not task.enabled:
                continue

            if task.last_run is None or \
               (now - task.last_run).total_seconds() >= task.interval_seconds:

                for symbol in task.symbols:
                    try:
                        await task.handler(symbol)
                    except Exception as e:
                        await self._handle_task_error(task, symbol, e)

                task.last_run = now

    async def _handle_trading_signal(self, message: Message) -> None:
        """
        Handle trading signal from Trading Decision Agent.

        Validates with Risk Engine and executes if approved.
        """
        if self._state != CoordinatorState.RUNNING:
            return

        signal = message.payload
        symbol = signal["symbol"]

        if signal["action"] == "HOLD":
            return

        # Check for conflicts
        conflicts = await self._detect_conflicts(signal)

        if conflicts:
            resolution = await self._resolve_conflicts(signal, conflicts)
            if resolution.action == "abort":
                return
            if resolution.action == "modify" and resolution.modifications:
                signal.update(resolution.modifications)

        # Create trade proposal
        proposal = self._create_trade_proposal(signal)

        # Get current regime and portfolio
        regime = await self._get_current_regime(symbol)
        portfolio = await self._get_portfolio_context()

        # Validate with risk engine
        validation = await self.agents["risk_management"].validate_trade(
            proposal=proposal,
            portfolio_context=portfolio,
            regime=regime.regime if regime else "neutral"
        )

        if validation.status == ValidationStatus.REJECTED:
            await self._log_rejection(proposal, validation)
            return

        # Use modified proposal if applicable
        final_proposal = validation.modified_proposal or proposal

        # Execute trade
        await self.agents["execution_manager"].execute_trade(final_proposal)

    async def _detect_conflicts(self, signal: dict) -> list[dict]:
        """Detect conflicts between agent outputs."""
        conflicts = []

        # Get latest outputs
        ta = await self.bus.get_latest(MessageTopic.TA_SIGNALS, source="technical_analysis")
        regime = await self.bus.get_latest(MessageTopic.REGIME_UPDATES, source="regime_detection")
        sentiment = await self.bus.get_latest(MessageTopic.SENTIMENT_UPDATES)

        # Check TA vs Sentiment conflict
        if ta and sentiment:
            ta_bias = ta.payload.get("bias", "neutral")
            sent_bias = sentiment.payload.get("bias", "neutral")

            if (ta_bias == "long" and sent_bias == "bearish") or \
               (ta_bias == "short" and sent_bias == "bullish"):

                conf_diff = abs(
                    ta.payload.get("confidence", 0.5) -
                    sentiment.payload.get("confidence", 0.5)
                )

                if conf_diff < 0.2:  # Close confidence = conflict
                    conflicts.append({
                        "type": "ta_sentiment_conflict",
                        "ta_bias": ta_bias,
                        "sentiment_bias": sent_bias,
                        "confidence_diff": conf_diff
                    })

        # Check regime appropriateness
        if regime:
            regime_type = regime.payload.get("regime", "neutral")
            action = signal.get("action")

            if regime_type == "choppy" and action != "HOLD":
                conflicts.append({
                    "type": "regime_conflict",
                    "regime": regime_type,
                    "action": action,
                    "recommendation": "Avoid trading in choppy regime"
                })

        return conflicts

    async def _resolve_conflicts(
        self,
        signal: dict,
        conflicts: list[dict]
    ) -> ConflictResolution:
        """
        Use LLM to resolve conflicts between agents.

        Invoked only when conflicts detected.
        """
        # Build conflict resolution prompt
        prompt = self._build_conflict_prompt(signal, conflicts)

        # Call DeepSeek V3
        response = await self.llm.generate(
            model="deepseek-v3",
            system_prompt=COORDINATOR_SYSTEM_PROMPT,
            user_message=prompt
        )

        # Parse response
        return self._parse_resolution(response.text)

    def _setup_schedules(self) -> None:
        """Configure scheduled tasks."""
        self._scheduled_tasks = [
            ScheduledTask(
                name="ta_analysis",
                agent="technical_analysis",
                interval_seconds=60,
                symbols=self.config["symbols"],
                handler=self._run_ta_agent
            ),
            ScheduledTask(
                name="regime_detection",
                agent="regime_detection",
                interval_seconds=300,
                symbols=self.config["symbols"],
                handler=self._run_regime_agent
            ),
            ScheduledTask(
                name="trading_decision",
                agent="trading_decision",
                interval_seconds=3600,
                symbols=self.config["symbols"],
                handler=self._run_trading_agent
            ),
            ScheduledTask(
                name="portfolio_check",
                agent="portfolio_rebalance",
                interval_seconds=3600,
                symbols=["PORTFOLIO"],
                handler=self._check_portfolio_allocation
            ),
        ]

    def _setup_subscriptions(self) -> None:
        """Subscribe to relevant message topics."""
        asyncio.create_task(
            self.bus.subscribe(
                subscriber_id=self.agent_name,
                topic=MessageTopic.TRADING_SIGNALS,
                handler=self._handle_trading_signal
            )
        )
        asyncio.create_task(
            self.bus.subscribe(
                subscriber_id=self.agent_name,
                topic=MessageTopic.RISK_ALERTS,
                handler=self._handle_risk_alert
            )
        )
```

### Configuration

```yaml
# config/orchestration.yaml

coordinator:
  # LLM for conflict resolution
  llm:
    primary:
      provider: deepseek
      model: deepseek-v3
    fallback:
      provider: anthropic
      model: claude-sonnet-4-20250514

  # Symbols to trade
  symbols:
    - BTC/USDT
    - XRP/USDT

  # Schedule configuration
  schedules:
    technical_analysis:
      interval_seconds: 60
      enabled: true

    regime_detection:
      interval_seconds: 300
      enabled: true

    sentiment_analysis:
      interval_seconds: 1800  # 30 minutes
      enabled: true

    trading_decision:
      interval_seconds: 3600
      enabled: true

    portfolio_rebalance:
      interval_seconds: 3600
      enabled: true

  # Conflict resolution
  conflicts:
    invoke_llm_threshold: 0.2  # Confidence diff below this = conflict
    max_resolution_time_ms: 10000

  # State management
  state:
    persist_to_db: true
    recovery_on_restart: true
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Schedule execution | Tasks run on schedule | Correct timing |
| Conflict detection | Conflicts identified | Correct detection |
| Conflict resolution | LLM resolves conflict | Valid resolution |
| State persistence | State survives restart | Correct recovery |
| Emergency handling | Circuit breaker handled | Trading halted |

---

## 3.3 Portfolio Rebalancing Agent

### Purpose

Monitors portfolio allocation and executes rebalancing trades to maintain 33/33/33 BTC/XRP/USDT target.

### LLM Assignment

| Property | Value |
|----------|-------|
| Model | DeepSeek V3 |
| Provider | DeepSeek |
| Invocation | Hourly check, execute on deviation |
| Latency Target | < 3s |
| Tier | Tier 2 (API) |

### Rebalancing Logic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PORTFOLIO REBALANCING FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. HOURLY CHECK                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Get current balances:                                                │   │
│  │   BTC: 0.0345 BTC ($1,535)                                          │   │
│  │   XRP: 2,150 XRP ($1,290)                                           │   │
│  │   USDT: 1,275 USDT                                                  │   │
│  │   ─────────────────────                                              │   │
│  │   Total: $4,100                                                      │   │
│  │                                                                      │   │
│  │ Calculate allocation:                                                │   │
│  │   BTC: 37.4% (target: 33.33%, deviation: +4.1%)                     │   │
│  │   XRP: 31.5% (target: 33.33%, deviation: -1.8%)                     │   │
│  │   USDT: 31.1% (target: 33.33%, deviation: -2.2%)                    │   │
│  │                                                                      │   │
│  │ Max deviation: 4.1%                                                  │   │
│  │ Threshold: 5.0%                                                      │   │
│  │ Action: No rebalance needed (4.1% < 5.0%)                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. REBALANCING TRIGGERED (deviation > 5%)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Target allocation: $4,100 / 3 = $1,366.67 each                      │   │
│  │                                                                      │   │
│  │ Required changes:                                                    │   │
│  │   BTC: $1,535 → $1,367 = SELL $168 worth                           │   │
│  │   XRP: $1,290 → $1,367 = BUY $77 worth                             │   │
│  │   USDT: $1,275 → $1,367 = BUY $92 worth                            │   │
│  │                                                                      │   │
│  │ LLM decides execution strategy:                                      │   │
│  │   • Single large trade or multiple smaller trades                   │   │
│  │   • Market or limit orders                                          │   │
│  │   • Sequencing to minimize slippage                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. HODL BAG EXCLUSION                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Hodl bags are EXCLUDED from rebalancing calculation:                 │   │
│  │                                                                      │   │
│  │ Total BTC: 0.0500 BTC                                               │   │
│  │   - Hodl bag: 0.0155 BTC (locked)                                   │   │
│  │   - Available: 0.0345 BTC (used in rebalancing)                     │   │
│  │                                                                      │   │
│  │ Rebalancing only considers available balances                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["timestamp", "action", "current_allocation", "target_allocation"],
  "properties": {
    "timestamp": {"type": "string", "format": "date-time"},
    "action": {
      "type": "string",
      "enum": ["no_action", "rebalance"]
    },
    "current_allocation": {
      "type": "object",
      "properties": {
        "btc_pct": {"type": "number"},
        "xrp_pct": {"type": "number"},
        "usdt_pct": {"type": "number"},
        "max_deviation_pct": {"type": "number"}
      }
    },
    "target_allocation": {
      "type": "object",
      "properties": {
        "btc_pct": {"type": "number", "default": 33.33},
        "xrp_pct": {"type": "number", "default": 33.33},
        "usdt_pct": {"type": "number", "default": 33.33}
      }
    },
    "rebalance_trades": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "symbol": {"type": "string"},
          "action": {"type": "string", "enum": ["buy", "sell"]},
          "amount_usd": {"type": "number"},
          "execution_type": {"type": "string", "enum": ["market", "limit"]},
          "priority": {"type": "integer"}
        }
      }
    },
    "reasoning": {"type": "string"}
  }
}
```

### Interface Definition

```python
# src/agents/portfolio_rebalance.py

@dataclass
class PortfolioAllocation:
    """Current portfolio allocation."""
    total_equity_usd: Decimal
    btc_value_usd: Decimal
    xrp_value_usd: Decimal
    usdt_value_usd: Decimal
    btc_pct: Decimal
    xrp_pct: Decimal
    usdt_pct: Decimal
    max_deviation_pct: Decimal
    hodl_excluded: dict  # Hodl bag amounts excluded


@dataclass
class RebalanceTrade:
    """Single rebalancing trade."""
    symbol: str
    action: str  # "buy" or "sell"
    amount_usd: Decimal
    execution_type: str  # "market" or "limit"
    priority: int  # Execution order


@dataclass
class RebalanceOutput(AgentOutput):
    """Portfolio Rebalancing Agent output."""
    action: str  # "no_action" or "rebalance"
    current_allocation: PortfolioAllocation
    trades: list[RebalanceTrade]
    reasoning: str


class PortfolioRebalanceAgent(BaseAgent):
    """
    Portfolio Rebalancing Agent using DeepSeek V3.

    Monitors allocation and decides rebalancing strategy.
    """

    agent_name = "portfolio_rebalance"
    llm_tier = "tier2_api"
    model = "deepseek-v3"

    def __init__(
        self,
        llm_client,
        prompt_builder,
        kraken_client,
        config: dict
    ):
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.kraken = kraken_client
        self.config = config

    async def check_allocation(self) -> PortfolioAllocation:
        """
        Check current portfolio allocation.

        Returns:
            PortfolioAllocation with current state
        """
        # Get balances from Kraken
        balances = await self.kraken.get_balances()

        # Get current prices
        prices = await self._get_current_prices()

        # Calculate allocations excluding hodl bags
        hodl_bags = await self._get_hodl_bags()

        available_btc = Decimal(balances.get("XXBT", 0)) - hodl_bags.get("BTC", Decimal(0))
        available_xrp = Decimal(balances.get("XXRP", 0)) - hodl_bags.get("XRP", Decimal(0))
        available_usdt = Decimal(balances.get("USDT", 0)) - hodl_bags.get("USDT", Decimal(0))

        btc_value = available_btc * prices["BTC/USDT"]
        xrp_value = available_xrp * prices["XRP/USDT"]
        usdt_value = available_usdt

        total = btc_value + xrp_value + usdt_value

        btc_pct = (btc_value / total * 100) if total > 0 else Decimal(0)
        xrp_pct = (xrp_value / total * 100) if total > 0 else Decimal(0)
        usdt_pct = (usdt_value / total * 100) if total > 0 else Decimal(0)

        target_pct = Decimal("33.33")
        max_dev = max(
            abs(btc_pct - target_pct),
            abs(xrp_pct - target_pct),
            abs(usdt_pct - target_pct)
        )

        return PortfolioAllocation(
            total_equity_usd=total,
            btc_value_usd=btc_value,
            xrp_value_usd=xrp_value,
            usdt_value_usd=usdt_value,
            btc_pct=btc_pct,
            xrp_pct=xrp_pct,
            usdt_pct=usdt_pct,
            max_deviation_pct=max_dev,
            hodl_excluded=hodl_bags
        )

    async def process(
        self,
        force: bool = False
    ) -> RebalanceOutput:
        """
        Check and potentially rebalance portfolio.

        Args:
            force: Force rebalancing even if below threshold

        Returns:
            RebalanceOutput with decision and trades
        """
        allocation = await self.check_allocation()

        threshold = Decimal(str(self.config["rebalance_threshold_pct"]))

        if allocation.max_deviation_pct < threshold and not force:
            return RebalanceOutput(
                action="no_action",
                current_allocation=allocation,
                trades=[],
                reasoning=f"Deviation {allocation.max_deviation_pct:.1f}% below threshold {threshold}%"
            )

        # Use LLM to decide rebalancing strategy
        prompt = self.prompt_builder.build_prompt(
            agent_name=self.agent_name,
            snapshot=None,
            additional_context={
                "allocation": allocation.__dict__,
                "threshold": float(threshold)
            }
        )

        response = await self.llm.generate(
            model=self.model,
            system_prompt=prompt.system_prompt,
            user_message=prompt.user_message
        )

        # Parse trades
        trades = self._parse_rebalance_trades(response.text)

        return RebalanceOutput(
            action="rebalance",
            current_allocation=allocation,
            trades=trades,
            reasoning=self._extract_reasoning(response.text)
        )

    def _calculate_rebalance_trades(
        self,
        allocation: PortfolioAllocation
    ) -> list[RebalanceTrade]:
        """Calculate required trades for rebalancing."""
        target_value = allocation.total_equity_usd / 3
        trades = []

        # BTC
        btc_diff = target_value - allocation.btc_value_usd
        if abs(btc_diff) > 10:  # Minimum $10
            trades.append(RebalanceTrade(
                symbol="BTC/USDT",
                action="buy" if btc_diff > 0 else "sell",
                amount_usd=abs(btc_diff),
                execution_type="limit",
                priority=1 if btc_diff < 0 else 2  # Sell first
            ))

        # XRP
        xrp_diff = target_value - allocation.xrp_value_usd
        if abs(xrp_diff) > 10:
            trades.append(RebalanceTrade(
                symbol="XRP/USDT",
                action="buy" if xrp_diff > 0 else "sell",
                amount_usd=abs(xrp_diff),
                execution_type="limit",
                priority=1 if xrp_diff < 0 else 2
            ))

        return sorted(trades, key=lambda t: t.priority)
```

### Configuration

```yaml
# config/portfolio.yaml

portfolio:
  # Target allocation
  target_allocation:
    btc_pct: 33.33
    xrp_pct: 33.33
    usdt_pct: 33.33

  # Rebalancing settings
  rebalancing:
    threshold_pct: 5.0  # Rebalance when deviation exceeds this
    min_trade_usd: 10.0  # Minimum trade size
    execution_type: limit  # Default to limit orders
    check_interval_seconds: 3600  # Hourly check

  # Hodl bag configuration
  hodl_bags:
    enabled: true
    allocation_pct: 10  # 10% of profits to hodl bags
    assets:
      - BTC
      - XRP
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Allocation calculation | Correct percentages | Within 0.01% |
| Threshold check | No action below threshold | No trades generated |
| Trade calculation | Correct amounts | Within $1 |
| Hodl exclusion | Hodl bags excluded | Correct calculation |

---

## 3.4 Order Execution Manager

### Purpose

Executes trades on Kraken exchange, manages order lifecycle, and tracks position state.

### NOT an LLM Agent

The Execution Manager is **purely rule-based** - it executes decisions from other agents.

### Order Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORDER EXECUTION FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. RECEIVE TRADE PROPOSAL (Risk-validated)                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TradeProposal:                                                       │   │
│  │   symbol: BTC/USDT                                                   │   │
│  │   action: BUY                                                        │   │
│  │   side: long                                                         │   │
│  │   size: 0.05 BTC                                                     │   │
│  │   entry_price: 45,200 (limit)                                        │   │
│  │   stop_loss: 44,300                                                  │   │
│  │   take_profit: 47,000                                                │   │
│  │   leverage: 2x                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. CREATE ORDERS                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Primary Order:                                                       │   │
│  │   → Kraken AddOrder (buy limit 0.05 @ 45,200)                       │   │
│  │   → Order ID: ORD-123                                               │   │
│  │                                                                      │   │
│  │ Contingent Orders (after fill):                                      │   │
│  │   → Stop Loss: sell stop 0.05 @ 44,300                              │   │
│  │   → Take Profit: sell limit 0.05 @ 47,000                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. MONITOR ORDER STATE                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ States:                                                              │   │
│  │   pending → open → filled                                           │   │
│  │                  → partially_filled                                  │   │
│  │                  → cancelled                                         │   │
│  │                  → expired                                           │   │
│  │                                                                      │   │
│  │ On fill:                                                            │   │
│  │   → Record trade_execution                                          │   │
│  │   → Place contingent orders                                         │   │
│  │   → Update portfolio snapshot                                       │   │
│  │   → Publish execution_event                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  4. POSITION MANAGEMENT                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Track open positions:                                                │   │
│  │   • Entry price, size, side                                         │   │
│  │   • Current P&L                                                      │   │
│  │   • Stop loss / take profit orders                                  │   │
│  │                                                                      │   │
│  │ On position close:                                                   │   │
│  │   → Calculate realized P&L                                          │   │
│  │   → Update trade_executions                                         │   │
│  │   → Allocate profits to hodl bag (if applicable)                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Interface Definition

```python
# src/execution/order_manager.py

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    ERROR = "error"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"


@dataclass
class Order:
    """Order representation."""
    id: str
    external_id: Optional[str]  # Kraken order ID
    symbol: str
    side: str  # "buy" or "sell"
    order_type: OrderType
    size: Decimal
    price: Optional[Decimal]
    status: OrderStatus
    filled_size: Decimal = Decimal(0)
    filled_price: Optional[Decimal] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class Position:
    """Open position."""
    id: str
    symbol: str
    side: str  # "long" or "short"
    size: Decimal
    entry_price: Decimal
    entry_time: datetime
    leverage: int
    stop_loss_order_id: Optional[str]
    take_profit_order_id: Optional[str]
    unrealized_pnl: Decimal = Decimal(0)
    unrealized_pnl_pct: Decimal = Decimal(0)


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    success: bool
    order: Optional[Order]
    position: Optional[Position]
    error_message: Optional[str]


class OrderExecutionManager:
    """
    Manages order execution on Kraken.

    Handles order lifecycle, position tracking, and contingent orders.
    """

    def __init__(
        self,
        kraken_client,
        message_bus: MessageBus,
        db_pool,
        config: dict
    ):
        self.kraken = kraken_client
        self.bus = message_bus
        self.db = db_pool
        self.config = config
        self._open_orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}

    async def execute_trade(
        self,
        proposal: TradeProposal
    ) -> ExecutionResult:
        """
        Execute a validated trade proposal.

        Args:
            proposal: Risk-validated trade proposal

        Returns:
            ExecutionResult with order/position details
        """
        try:
            # Convert to Kraken order format
            kraken_symbol = self._to_kraken_symbol(proposal.symbol)
            order_type = "limit" if proposal.entry_price else "market"

            # Place primary order
            result = await self.kraken.add_order(
                pair=kraken_symbol,
                type="buy" if proposal.side == "long" else "sell",
                ordertype=order_type,
                volume=str(proposal.size),
                price=str(proposal.entry_price) if proposal.entry_price else None,
                leverage=str(proposal.leverage) if proposal.leverage > 1 else None
            )

            if "error" in result and result["error"]:
                return ExecutionResult(
                    success=False,
                    order=None,
                    position=None,
                    error_message=str(result["error"])
                )

            # Create order record
            order_id = str(uuid.uuid4())
            external_id = result["result"]["txid"][0]

            order = Order(
                id=order_id,
                external_id=external_id,
                symbol=proposal.symbol,
                side="buy" if proposal.side == "long" else "sell",
                order_type=OrderType.LIMIT if order_type == "limit" else OrderType.MARKET,
                size=proposal.size,
                price=proposal.entry_price,
                status=OrderStatus.OPEN,
                created_at=datetime.utcnow()
            )

            self._open_orders[order_id] = order
            await self._store_order(order)

            # Start monitoring order
            asyncio.create_task(self._monitor_order(order, proposal))

            return ExecutionResult(
                success=True,
                order=order,
                position=None,  # Position created on fill
                error_message=None
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                order=None,
                position=None,
                error_message=str(e)
            )

    async def _monitor_order(
        self,
        order: Order,
        proposal: TradeProposal
    ) -> None:
        """Monitor order until filled or cancelled."""
        while order.status in [OrderStatus.OPEN, OrderStatus.PENDING]:
            await asyncio.sleep(5)  # Poll every 5 seconds

            # Check order status
            status = await self.kraken.query_orders(
                txid=order.external_id
            )

            kraken_status = status["result"][order.external_id]["status"]

            if kraken_status == "closed":
                # Order filled
                fill_info = status["result"][order.external_id]
                order.status = OrderStatus.FILLED
                order.filled_size = Decimal(fill_info["vol_exec"])
                order.filled_price = Decimal(fill_info["price"])

                # Create position
                position = await self._create_position(order, proposal)

                # Place contingent orders
                await self._place_contingent_orders(position, proposal)

                # Publish event
                await self._publish_execution_event(order, position)

            elif kraken_status == "canceled":
                order.status = OrderStatus.CANCELLED

            elif kraken_status == "expired":
                order.status = OrderStatus.EXPIRED

        # Update stored order
        await self._update_order(order)

    async def _place_contingent_orders(
        self,
        position: Position,
        proposal: TradeProposal
    ) -> None:
        """Place stop loss and take profit orders."""
        # Stop loss
        if proposal.stop_loss:
            sl_result = await self.kraken.add_order(
                pair=self._to_kraken_symbol(position.symbol),
                type="sell" if position.side == "long" else "buy",
                ordertype="stop-loss",
                volume=str(position.size),
                price=str(proposal.stop_loss)
            )
            position.stop_loss_order_id = sl_result["result"]["txid"][0]

        # Take profit
        if proposal.take_profit:
            tp_result = await self.kraken.add_order(
                pair=self._to_kraken_symbol(position.symbol),
                type="sell" if position.side == "long" else "buy",
                ordertype="take-profit",
                volume=str(position.size),
                price=str(proposal.take_profit)
            )
            position.take_profit_order_id = tp_result["result"]["txid"][0]

    async def close_position(
        self,
        position_id: str,
        reason: str
    ) -> ExecutionResult:
        """Close an open position."""
        pass

    async def modify_position(
        self,
        position_id: str,
        new_stop_loss: Optional[Decimal] = None,
        new_take_profit: Optional[Decimal] = None
    ) -> bool:
        """Modify stop loss / take profit for position."""
        pass

    async def get_open_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    async def sync_with_exchange(self) -> None:
        """Sync local state with exchange state."""
        pass
```

### Configuration

```yaml
# config/execution.yaml

execution:
  # Kraken API settings
  kraken:
    api_key: ${KRAKEN_API_KEY}
    api_secret: ${KRAKEN_API_SECRET}
    rate_limit:
      calls_per_minute: 60
      order_calls_per_minute: 30

  # Order settings
  orders:
    default_type: limit
    time_in_force: GTC  # Good Till Cancelled
    max_retry_count: 3
    retry_delay_seconds: 5

  # Position management
  positions:
    sync_interval_seconds: 30
    max_open_positions: 6  # 2 per symbol

  # Slippage protection
  slippage:
    max_slippage_pct: 0.5
    use_limit_orders: true

  # Logging
  logging:
    log_all_orders: true
    log_to_db: true
```

### Database Updates for Execution

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

CREATE INDEX idx_order_status_log_order
    ON order_status_log (order_id, timestamp DESC);

-- Position real-time tracking
CREATE TABLE position_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    position_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    current_price DECIMAL(20, 10),
    unrealized_pnl DECIMAL(20, 10),
    unrealized_pnl_pct DECIMAL(10, 4),
    PRIMARY KEY (timestamp, position_id)
);

SELECT create_hypertable('position_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Order placement | Order sent to Kraken | Order ID returned |
| Order monitoring | Status updates tracked | Correct transitions |
| Position creation | Position created on fill | All fields populated |
| Contingent orders | SL/TP placed | Orders active |
| Error handling | API error handled | Graceful failure |

---

## Phase 3 Acceptance Criteria

### Functional Requirements

| Requirement | Test Method | Acceptance |
|-------------|-------------|------------|
| Message bus working | Pub/sub test | Messages delivered |
| Coordinator schedules | Timing test | Correct intervals |
| Portfolio rebalancing | Allocation test | Trades generated |
| Order execution | Paper trade | Orders placed |
| Position tracking | Position query | Correct state |

### Integration Requirements

| Requirement | Acceptance |
|-------------|------------|
| Full agent pipeline | TA → Regime → Trading → Risk → Execution |
| Message propagation | All agents receive relevant messages |
| Conflict resolution | Coordinator resolves conflicts |
| State persistence | Survives restart |

### Deliverables Checklist

- [x] `src/orchestration/message_bus.py` ✅
- [x] `src/orchestration/coordinator.py` ✅
- [x] `src/agents/portfolio_rebalance.py` ✅
- [x] `src/execution/order_manager.py` ✅
- [x] `src/execution/position_tracker.py` ✅
- [x] Configuration files (`orchestration.yaml`, `portfolio.yaml`, `execution.yaml`) ✅
- [x] Database migrations (`003_phase3_orchestration.sql`) ✅
- [x] Unit tests (227 new tests, 916 total) ✅
- [x] API routes (`routes_orchestration.py`) ✅
- [ ] Integration tests (paper trading mode) - Phase 5

---

## API Endpoints (Phase 3)

```yaml
# API Routes for Phase 3

endpoints:
  # Coordinator
  - path: /api/v1/coordinator/status
    method: GET
    description: Get coordinator state
    response: CoordinatorStatus

  - path: /api/v1/coordinator/pause
    method: POST
    description: Pause trading

  - path: /api/v1/coordinator/resume
    method: POST
    description: Resume trading

  # Portfolio
  - path: /api/v1/portfolio/allocation
    method: GET
    description: Get current allocation
    response: PortfolioAllocation

  - path: /api/v1/portfolio/rebalance
    method: POST
    description: Force rebalancing

  # Positions
  - path: /api/v1/positions
    method: GET
    description: Get open positions
    response: list[Position]

  - path: /api/v1/positions/{id}/close
    method: POST
    description: Close position

  # Orders
  - path: /api/v1/orders
    method: GET
    description: Get open orders
    response: list[Order]

  - path: /api/v1/orders/{id}/cancel
    method: POST
    description: Cancel order
```

---

## References

- Design: [01-multi-agent-architecture.md](../TripleGain-master-design/01-multi-agent-architecture.md)
- Design: [03-risk-management-rules-engine.md](../TripleGain-master-design/03-risk-management-rules-engine.md)

---

*Phase 3 Implementation Plan v1.1 - COMPLETE - December 2025*
