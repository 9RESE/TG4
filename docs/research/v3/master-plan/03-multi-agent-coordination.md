# Multi-Agent Coordination Design

**Version:** 1.0
**Date:** December 2025
**Status:** Design Phase

---

## Overview

This document defines the coordination patterns for the multi-agent system in TripleGain. The design draws from TradingAgents, Nof1.ai, and AutoHedge frameworks, adapted for the specific needs of a tri-asset cryptocurrency trading system.

---

## Table of Contents

1. [Agent Roles and Responsibilities](#1-agent-roles-and-responsibilities)
2. [Communication Patterns](#2-communication-patterns)
3. [Coordination Protocols](#3-coordination-protocols)
4. [Consensus Mechanisms](#4-consensus-mechanisms)
5. [State Management](#5-state-management)
6. [Failure Handling](#6-failure-handling)

---

## 1. Agent Roles and Responsibilities

### 1.1 Agent Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT HIERARCHY                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Coordinator   │
                              │    (Master)     │
                              └────────┬────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
  │   ANALYSIS      │        │   DECISION      │        │   EXECUTION     │
  │     TIER        │        │     TIER        │        │     TIER        │
  │                 │        │                 │        │                 │
  │ ┌─────────────┐ │        │ ┌─────────────┐ │        │ ┌─────────────┐ │
  │ │  Technical  │ │───────▶│ │   Trading   │ │───────▶│ │   Order     │ │
  │ │   Agent     │ │        │ │    Agent    │ │        │ │  Manager    │ │
  │ └─────────────┘ │        │ │    (LLM)    │ │        │ └─────────────┘ │
  │                 │        │ └─────────────┘ │        │                 │
  │ ┌─────────────┐ │        │                 │        │ ┌─────────────┐ │
  │ │   Regime    │ │───────▶│ ┌─────────────┐ │───────▶│ │  Position   │ │
  │ │  Detector   │ │        │ │    Risk     │ │        │ │  Tracker    │ │
  │ └─────────────┘ │        │ │   Agent     │ │        │ └─────────────┘ │
  │                 │        │ └─────────────┘ │        │                 │
  │ ┌─────────────┐ │        │                 │        │ ┌─────────────┐ │
  │ │ Rebalancing │ │───────▶│ ┌─────────────┐ │───────▶│ │  Executor   │ │
  │ │   Agent     │ │        │ │   Model     │ │        │ │(Paper/Live) │ │
  │ └─────────────┘ │        │ │  Selector   │ │        │ └─────────────┘ │
  └─────────────────┘        │ └─────────────┘ │        └─────────────────┘
                              └─────────────────┘
```

### 1.2 Agent Specifications

| Agent | Type | Latency | Priority | Dependencies |
|-------|------|---------|----------|--------------|
| **Coordinator** | Orchestrator | <10ms | Critical | All agents |
| **Technical Agent** | Analysis | <100ms | High | DataSnapshot, Indicators |
| **Regime Detector** | Analysis | <50ms | High | DataSnapshot |
| **Trading Agent** | Decision | <10s | High | Technical, Regime |
| **Risk Agent** | Decision | <50ms | Critical | Portfolio, Trading |
| **Rebalancing Agent** | Analysis | <1s | Medium | Portfolio |
| **Model Selector** | Decision | <10ms | Medium | Performance History |
| **Order Manager** | Execution | <100ms | Critical | Risk Assessment |
| **Position Tracker** | Execution | <10ms | High | Exchange Events |
| **Executor** | Execution | <500ms | Critical | Order Manager |

### 1.3 Agent Interface Contracts

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime

@dataclass
class AgentInput:
    """Base input for all agents."""
    timestamp: datetime
    request_id: str
    data: dict[str, Any]

@dataclass
class AgentOutput:
    """Base output for all agents."""
    agent_id: str
    timestamp: datetime
    request_id: str
    success: bool
    result: Optional[Any]
    error: Optional[str]
    latency_ms: float

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, agent_id: str, config: dict):
        self.agent_id = agent_id
        self.config = config
        self.is_healthy = True

    @abstractmethod
    async def process(self, input: AgentInput) -> AgentOutput:
        """Process input and return output."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check agent health status."""
        pass

    @abstractmethod
    async def reset(self) -> None:
        """Reset agent state."""
        pass

    async def pre_process(self, input: AgentInput) -> AgentInput:
        """Hook for pre-processing input."""
        return input

    async def post_process(self, output: AgentOutput) -> AgentOutput:
        """Hook for post-processing output."""
        return output
```

---

## 2. Communication Patterns

### 2.1 Message Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime

class MessageType(Enum):
    # Data messages
    MARKET_UPDATE = "market_update"
    PORTFOLIO_UPDATE = "portfolio_update"
    POSITION_UPDATE = "position_update"

    # Control messages
    START_CYCLE = "start_cycle"
    STOP_CYCLE = "stop_cycle"
    PAUSE_TRADING = "pause_trading"
    RESUME_TRADING = "resume_trading"

    # Agent messages
    SIGNAL_REQUEST = "signal_request"
    SIGNAL_RESPONSE = "signal_response"
    DECISION_REQUEST = "decision_request"
    DECISION_RESPONSE = "decision_response"
    RISK_REQUEST = "risk_request"
    RISK_RESPONSE = "risk_response"

    # Execution messages
    ORDER_REQUEST = "order_request"
    ORDER_RESPONSE = "order_response"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"

    # System messages
    HEALTH_CHECK = "health_check"
    HEALTH_RESPONSE = "health_response"
    ERROR = "error"
    ALERT = "alert"

@dataclass
class Message:
    """Inter-agent message structure."""
    id: str
    type: MessageType
    source: str
    target: str  # "*" for broadcast
    timestamp: datetime
    payload: dict[str, Any]
    correlation_id: Optional[str] = None
    priority: int = 5  # 1 (highest) to 10 (lowest)
    ttl_seconds: int = 30
```

### 2.2 Message Bus Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MESSAGE BUS ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Message Bus   │
                              │    (Redis)      │
                              └────────┬────────┘
                                       │
     ┌─────────────────┬───────────────┼───────────────┬─────────────────┐
     │                 │               │               │                 │
     ▼                 ▼               ▼               ▼                 ▼
┌─────────┐      ┌─────────┐    ┌─────────┐    ┌─────────┐      ┌─────────┐
│ Channel │      │ Channel │    │ Channel │    │ Channel │      │ Channel │
│ signals │      │decisions│    │  risk   │    │ orders  │      │ system  │
└────┬────┘      └────┬────┘    └────┬────┘    └────┬────┘      └────┬────┘
     │                 │               │               │                 │
     ├── Technical     ├── Trading     ├── Risk        ├── Order         ├── Coordinator
     │   Agent         │   Agent       │   Agent       │   Manager       │
     │                 │               │               │                 │
     └── Regime        └── Model       └──             └── Executor      └── Health
         Detector          Selector                                           Monitor
```

### 2.3 Message Bus Implementation

```python
import asyncio
import json
from datetime import datetime
from typing import Callable, Dict, List
import redis.asyncio as redis

class MessageBus:
    """Redis-based message bus for agent communication."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.subscribers: Dict[str, List[Callable]] = {}
        self._running = False

    async def publish(self, channel: str, message: Message) -> None:
        """Publish message to channel."""
        await self.redis.publish(
            channel,
            json.dumps({
                "id": message.id,
                "type": message.type.value,
                "source": message.source,
                "target": message.target,
                "timestamp": message.timestamp.isoformat(),
                "payload": message.payload,
                "correlation_id": message.correlation_id,
                "priority": message.priority
            })
        )

    async def subscribe(self, channel: str, handler: Callable) -> None:
        """Subscribe to channel with handler."""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        self.subscribers[channel].append(handler)

    async def start_listening(self) -> None:
        """Start listening for messages on subscribed channels."""
        self._running = True
        pubsub = self.redis.pubsub()

        for channel in self.subscribers.keys():
            await pubsub.subscribe(channel)

        while self._running:
            message = await pubsub.get_message(ignore_subscribe_messages=True)
            if message:
                channel = message["channel"].decode()
                data = json.loads(message["data"])
                msg = Message(
                    id=data["id"],
                    type=MessageType(data["type"]),
                    source=data["source"],
                    target=data["target"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    payload=data["payload"],
                    correlation_id=data.get("correlation_id"),
                    priority=data.get("priority", 5)
                )
                for handler in self.subscribers.get(channel, []):
                    asyncio.create_task(handler(msg))

    async def stop(self) -> None:
        """Stop listening."""
        self._running = False
        await self.redis.close()
```

---

## 3. Coordination Protocols

### 3.1 Trading Cycle Protocol

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRADING CYCLE PROTOCOL                               │
└─────────────────────────────────────────────────────────────────────────────┘

  Phase 1: DATA COLLECTION (Parallel)
  ─────────────────────────────────────

  Coordinator ──START_CYCLE──▶ All Agents

  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
  │  Technical Agent   │  │  Regime Detector   │  │ Rebalancing Agent  │
  │                    │  │                    │  │                    │
  │  Calculate         │  │  Detect current    │  │  Check allocation  │
  │  indicators for    │  │  market regime     │  │  drift             │
  │  all pairs         │  │                    │  │                    │
  └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼

  Phase 2: SIGNAL GENERATION
  ─────────────────────────────────────

                          ┌────────────────────┐
                          │    Coordinator     │
                          │                    │
                          │ Collect all agent  │
                          │ outputs, build     │
                          │ MarketContext      │
                          └─────────┬──────────┘
                                    │
                                    ▼
                          ┌────────────────────┐
                          │   Model Selector   │
                          │                    │
                          │ Choose best LLM    │
                          │ based on recent    │
                          │ performance        │
                          └─────────┬──────────┘
                                    │
                                    ▼
                          ┌────────────────────┐
                          │   Trading Agent    │
                          │      (LLM)         │
                          │                    │
                          │ Generate trading   │
                          │ decision with      │
                          │ reasoning          │
                          └─────────┬──────────┘
                                    │
                                    ▼

  Phase 3: RISK VALIDATION
  ─────────────────────────────────────

                          ┌────────────────────┐
                          │    Risk Agent      │
                          │                    │
                          │ Validate decision: │
                          │ - Position sizing  │
                          │ - Stop-loss check  │
                          │ - Drawdown limits  │
                          │ - Correlation      │
                          └─────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    APPROVED              REJECTED
                         │                     │
                         ▼                     ▼
  Phase 4: EXECUTION              Log rejection reason
  ─────────────────────           Return to Phase 1

                          ┌────────────────────┐
                          │   Order Manager    │
                          │                    │
                          │ Create order with  │
                          │ adjusted params    │
                          └─────────┬──────────┘
                                    │
                                    ▼
                          ┌────────────────────┐
                          │     Executor       │
                          │                    │
                          │ Submit to exchange │
                          │ (paper or live)    │
                          └─────────┬──────────┘
                                    │
                                    ▼
                          ┌────────────────────┐
                          │  Position Tracker  │
                          │                    │
                          │ Update portfolio   │
                          │ state              │
                          └─────────┬──────────┘
                                    │
                                    ▼

  Phase 5: LOGGING
  ─────────────────────────────────────

                          ┌────────────────────┐
                          │    Coordinator     │
                          │                    │
                          │ Log decision +     │
                          │ reasoning to DB    │
                          │ Update metrics     │
                          └────────────────────┘
```

### 3.2 Coordinator State Machine

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class CoordinatorState(Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    DECIDING = "deciding"
    VALIDATING = "validating"
    EXECUTING = "executing"
    LOGGING = "logging"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class CycleContext:
    """Context maintained across a trading cycle."""
    cycle_id: str
    start_time: datetime
    market_data: Optional[dict] = None
    technical_signals: Optional[dict] = None
    regime: Optional[str] = None
    rebalance_recommendation: Optional[dict] = None
    selected_model: Optional[str] = None
    trading_decision: Optional[dict] = None
    risk_assessment: Optional[dict] = None
    order: Optional[dict] = None
    execution_result: Optional[dict] = None

class Coordinator:
    """Main coordinator for multi-agent system."""

    def __init__(self, agents: dict[str, BaseAgent], message_bus: MessageBus):
        self.agents = agents
        self.bus = message_bus
        self.state = CoordinatorState.IDLE
        self.current_cycle: Optional[CycleContext] = None

    async def run_cycle(self) -> CycleContext:
        """Execute one complete trading cycle."""
        self.current_cycle = CycleContext(
            cycle_id=generate_uuid(),
            start_time=datetime.utcnow()
        )

        try:
            # Phase 1: Data Collection (Parallel)
            self.state = CoordinatorState.COLLECTING
            await self._collect_data()

            # Phase 2: Analysis
            self.state = CoordinatorState.ANALYZING
            await self._analyze()

            # Phase 3: Decision
            self.state = CoordinatorState.DECIDING
            await self._make_decision()

            # Phase 4: Validation
            self.state = CoordinatorState.VALIDATING
            if not await self._validate():
                self.state = CoordinatorState.LOGGING
                await self._log_cycle()
                return self.current_cycle

            # Phase 5: Execution
            self.state = CoordinatorState.EXECUTING
            await self._execute()

            # Phase 6: Logging
            self.state = CoordinatorState.LOGGING
            await self._log_cycle()

        except Exception as e:
            self.state = CoordinatorState.ERROR
            await self._handle_error(e)

        finally:
            self.state = CoordinatorState.IDLE

        return self.current_cycle

    async def _collect_data(self) -> None:
        """Collect data from all analysis agents in parallel."""
        tasks = [
            self.agents['technical'].process(AgentInput(
                timestamp=datetime.utcnow(),
                request_id=self.current_cycle.cycle_id,
                data={'symbols': ['BTC/USDT', 'XRP/USDT', 'XRP/BTC']}
            )),
            self.agents['regime'].process(AgentInput(
                timestamp=datetime.utcnow(),
                request_id=self.current_cycle.cycle_id,
                data={}
            )),
            self.agents['rebalancing'].process(AgentInput(
                timestamp=datetime.utcnow(),
                request_id=self.current_cycle.cycle_id,
                data={}
            ))
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        self.current_cycle.technical_signals = results[0].result if results[0].success else None
        self.current_cycle.regime = results[1].result if results[1].success else None
        self.current_cycle.rebalance_recommendation = results[2].result if results[2].success else None

    async def _analyze(self) -> None:
        """Select model and prepare context."""
        selector_result = await self.agents['model_selector'].process(AgentInput(
            timestamp=datetime.utcnow(),
            request_id=self.current_cycle.cycle_id,
            data={'recent_performance': await self._get_recent_performance()}
        ))
        self.current_cycle.selected_model = selector_result.result

    async def _make_decision(self) -> None:
        """Query trading agent for decision."""
        decision_result = await self.agents['trading'].process(AgentInput(
            timestamp=datetime.utcnow(),
            request_id=self.current_cycle.cycle_id,
            data={
                'technical_signals': self.current_cycle.technical_signals,
                'regime': self.current_cycle.regime,
                'rebalance': self.current_cycle.rebalance_recommendation,
                'model': self.current_cycle.selected_model
            }
        ))
        self.current_cycle.trading_decision = decision_result.result

    async def _validate(self) -> bool:
        """Validate decision with risk agent."""
        risk_result = await self.agents['risk'].process(AgentInput(
            timestamp=datetime.utcnow(),
            request_id=self.current_cycle.cycle_id,
            data={'decision': self.current_cycle.trading_decision}
        ))
        self.current_cycle.risk_assessment = risk_result.result
        return risk_result.result.get('approved', False)

    async def _execute(self) -> None:
        """Execute the trade."""
        if self.current_cycle.trading_decision['action'] == 'HOLD':
            return

        order_result = await self.agents['order_manager'].process(AgentInput(
            timestamp=datetime.utcnow(),
            request_id=self.current_cycle.cycle_id,
            data={
                'decision': self.current_cycle.trading_decision,
                'risk_assessment': self.current_cycle.risk_assessment
            }
        ))
        self.current_cycle.order = order_result.result

        exec_result = await self.agents['executor'].process(AgentInput(
            timestamp=datetime.utcnow(),
            request_id=self.current_cycle.cycle_id,
            data={'order': self.current_cycle.order}
        ))
        self.current_cycle.execution_result = exec_result.result

    async def _log_cycle(self) -> None:
        """Log cycle results to database."""
        # Implementation: Save to TimescaleDB
        pass

    async def _handle_error(self, error: Exception) -> None:
        """Handle cycle errors."""
        # Implementation: Log error, send alert
        pass
```

---

## 4. Consensus Mechanisms

### 4.1 Multi-Model Consensus

When running multiple LLMs in parallel for comparison, use weighted voting:

```python
@dataclass
class ModelVote:
    model: str
    action: str
    confidence: float
    weight: float  # Based on recent performance

def calculate_consensus(votes: list[ModelVote], threshold: float = 0.6) -> Optional[str]:
    """Calculate consensus action from multiple model votes."""

    # Group by action
    action_scores: dict[str, float] = {}
    for vote in votes:
        if vote.confidence >= threshold:
            weighted_score = vote.confidence * vote.weight
            action_scores[vote.action] = action_scores.get(vote.action, 0) + weighted_score

    if not action_scores:
        return None

    # Find winning action
    total_score = sum(action_scores.values())
    winning_action = max(action_scores, key=action_scores.get)
    winning_score = action_scores[winning_action]

    # Require majority weighted consensus
    if winning_score / total_score >= 0.5:
        return winning_action

    return None

def aggregate_decisions(decisions: list[TradingDecision]) -> TradingDecision:
    """Aggregate multiple model decisions into one."""

    # Use consensus for action
    votes = [
        ModelVote(
            model=d.model_used,
            action=d.action,
            confidence=d.confidence,
            weight=get_model_weight(d.model_used)  # From performance history
        )
        for d in decisions
    ]

    consensus_action = calculate_consensus(votes)
    if not consensus_action or consensus_action == 'HOLD':
        # Default to HOLD if no consensus
        return TradingDecision(
            action='HOLD',
            symbol=decisions[0].symbol,
            confidence=0.0,
            reasoning="No consensus among models",
            model_used="ensemble"
        )

    # Use highest-confidence agreeing decision for parameters
    agreeing = [d for d in decisions if d.action == consensus_action]
    best = max(agreeing, key=lambda d: d.confidence)

    return TradingDecision(
        action=consensus_action,
        symbol=best.symbol,
        confidence=sum(d.confidence for d in agreeing) / len(agreeing),
        position_size_pct=best.position_size_pct,
        leverage=best.leverage,
        entry_price=best.entry_price,
        stop_loss=best.stop_loss,
        take_profit=best.take_profit,
        invalidation=best.invalidation,
        reasoning=f"Ensemble ({len(agreeing)}/{len(decisions)} agree): {best.reasoning}",
        model_used="ensemble"
    )
```

### 4.2 Agent Conflict Resolution

```python
class ConflictResolver:
    """Resolve conflicts between agent recommendations."""

    PRIORITY_ORDER = ['risk', 'trading', 'technical', 'rebalancing']

    def resolve(self, agent_outputs: dict[str, AgentOutput]) -> dict:
        """Resolve conflicts based on priority and rules."""

        # Risk agent has absolute veto
        risk_output = agent_outputs.get('risk')
        if risk_output and not risk_output.result.get('approved'):
            return {
                'action': 'HOLD',
                'reason': f"Risk agent rejected: {risk_output.result.get('rejection_reason')}"
            }

        # Trading agent drives primary decision
        trading_output = agent_outputs.get('trading')
        if not trading_output or not trading_output.success:
            return {
                'action': 'HOLD',
                'reason': 'Trading agent failed or unavailable'
            }

        decision = trading_output.result

        # Technical agent can downgrade confidence
        technical_output = agent_outputs.get('technical')
        if technical_output and technical_output.success:
            tech_signal = technical_output.result
            if tech_signal['direction'] != decision['action']:
                # Technical disagrees - reduce confidence
                decision['confidence'] *= 0.8
                decision['reasoning'] += f" [Note: Technical analysis suggests {tech_signal['direction']}]"

        # Rebalancing can suggest alternative
        rebalance_output = agent_outputs.get('rebalancing')
        if rebalance_output and rebalance_output.success:
            rebalance = rebalance_output.result
            if rebalance.get('priority') == 'HIGH':
                # High priority rebalance overrides trading
                return {
                    'action': 'REBALANCE',
                    'recommendation': rebalance,
                    'reason': 'High priority portfolio rebalancing'
                }

        return decision
```

---

## 5. State Management

### 5.1 Shared State Schema

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class SharedState:
    """State shared across all agents via Redis."""

    # Market state
    last_prices: dict[str, float] = field(default_factory=dict)
    current_regime: str = "unknown"
    volatility_state: str = "normal"

    # Portfolio state
    balances: dict[str, float] = field(default_factory=dict)
    open_positions: list[dict] = field(default_factory=list)
    total_value_usdt: float = 0.0

    # Trading state
    is_trading_enabled: bool = True
    cooldown_until: Optional[datetime] = None
    consecutive_losses: int = 0
    daily_pnl_pct: float = 0.0
    current_drawdown_pct: float = 0.0

    # Model state
    active_model: str = "claude-sonnet-4-5"
    model_performance: dict[str, dict] = field(default_factory=dict)

    # Timing
    last_trade_time: dict[str, datetime] = field(default_factory=dict)
    last_cycle_time: Optional[datetime] = None
```

### 5.2 State Synchronization

```python
import json
import redis.asyncio as redis
from typing import Optional

class StateManager:
    """Manage shared state across agents."""

    def __init__(self, redis_url: str, key_prefix: str = "triplegain"):
        self.redis = redis.from_url(redis_url)
        self.prefix = key_prefix

    def _key(self, name: str) -> str:
        return f"{self.prefix}:{name}"

    async def get_state(self) -> SharedState:
        """Retrieve complete shared state."""
        data = await self.redis.hgetall(self._key("state"))
        if not data:
            return SharedState()

        return SharedState(
            last_prices=json.loads(data.get(b"last_prices", b"{}")),
            current_regime=data.get(b"current_regime", b"unknown").decode(),
            volatility_state=data.get(b"volatility_state", b"normal").decode(),
            balances=json.loads(data.get(b"balances", b"{}")),
            open_positions=json.loads(data.get(b"open_positions", b"[]")),
            total_value_usdt=float(data.get(b"total_value_usdt", 0)),
            is_trading_enabled=data.get(b"is_trading_enabled", b"1") == b"1",
            consecutive_losses=int(data.get(b"consecutive_losses", 0)),
            daily_pnl_pct=float(data.get(b"daily_pnl_pct", 0)),
            current_drawdown_pct=float(data.get(b"current_drawdown_pct", 0)),
            active_model=data.get(b"active_model", b"claude-sonnet-4-5").decode(),
        )

    async def update_state(self, **kwargs) -> None:
        """Update specific state fields."""
        updates = {}
        for key, value in kwargs.items():
            if isinstance(value, (dict, list)):
                updates[key] = json.dumps(value)
            elif isinstance(value, bool):
                updates[key] = "1" if value else "0"
            elif isinstance(value, datetime):
                updates[key] = value.isoformat()
            else:
                updates[key] = str(value)

        if updates:
            await self.redis.hset(self._key("state"), mapping=updates)

    async def acquire_lock(self, name: str, timeout: int = 10) -> bool:
        """Acquire distributed lock."""
        return await self.redis.set(
            self._key(f"lock:{name}"),
            "1",
            nx=True,
            ex=timeout
        )

    async def release_lock(self, name: str) -> None:
        """Release distributed lock."""
        await self.redis.delete(self._key(f"lock:{name}"))
```

---

## 6. Failure Handling

### 6.1 Agent Failure Matrix

| Agent | Failure Mode | Impact | Recovery Action |
|-------|--------------|--------|-----------------|
| **Technical** | Timeout | Medium | Use cached values, reduce confidence |
| **Technical** | Error | Medium | Skip cycle, retry next |
| **Regime** | Timeout | Low | Assume "unknown", reduce position size |
| **Regime** | Error | Low | Default to "normal" regime |
| **Trading** | Timeout | High | Default to HOLD |
| **Trading** | Parse Error | High | Retry once, then HOLD |
| **Risk** | Timeout | Critical | Block all trades |
| **Risk** | Error | Critical | Block all trades, alert |
| **Executor** | Timeout | Critical | Retry with backoff |
| **Executor** | Error | Critical | Cancel order, alert |

### 6.2 Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreaker:
    """Circuit breaker for agent calls."""

    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_requests: int = 3

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_successes: int = 0

    def can_execute(self) -> bool:
        """Check if request should proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.half_open_successes = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_success(self) -> None:
        """Record successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_requests:
                self.state = CircuitState.CLOSED
                self.failure_count = 0

        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
```

### 6.3 Graceful Degradation

```python
class DegradedModeManager:
    """Manage system degradation when components fail."""

    def __init__(self, state_manager: StateManager):
        self.state = state_manager
        self.degradation_level = 0  # 0=normal, 1=degraded, 2=minimal, 3=stopped

    async def assess_system_health(self, agent_health: dict[str, bool]) -> int:
        """Assess and update degradation level."""
        critical_agents = ['risk', 'executor', 'position_tracker']
        important_agents = ['trading', 'technical']
        optional_agents = ['regime', 'rebalancing', 'model_selector']

        critical_failures = sum(1 for a in critical_agents if not agent_health.get(a, True))
        important_failures = sum(1 for a in important_agents if not agent_health.get(a, True))
        optional_failures = sum(1 for a in optional_agents if not agent_health.get(a, True))

        if critical_failures > 0:
            self.degradation_level = 3  # Stop trading
        elif important_failures > 0:
            self.degradation_level = 2  # Minimal trading
        elif optional_failures > 0:
            self.degradation_level = 1  # Degraded but functional
        else:
            self.degradation_level = 0  # Normal

        return self.degradation_level

    def get_allowed_actions(self) -> list[str]:
        """Get allowed actions for current degradation level."""
        if self.degradation_level == 0:
            return ['LONG', 'SHORT', 'HOLD', 'CLOSE', 'REBALANCE']
        elif self.degradation_level == 1:
            return ['LONG', 'SHORT', 'HOLD', 'CLOSE']  # No rebalancing
        elif self.degradation_level == 2:
            return ['HOLD', 'CLOSE']  # Close-only mode
        else:
            return []  # No trading

    def get_position_size_modifier(self) -> float:
        """Get position size multiplier for degraded mode."""
        modifiers = {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}
        return modifiers.get(self.degradation_level, 0.0)
```

---

## Appendix: Agent Communication Sequence Examples

### A.1 Successful Trade Execution

```
Coordinator     Technical      Regime      Trading       Risk        Executor
     │              │            │            │            │             │
     │─START_CYCLE─▶│            │            │            │             │
     │─START_CYCLE──────────────▶│            │            │             │
     │              │            │            │            │             │
     │◀─SIGNALS─────│            │            │            │             │
     │◀─REGIME──────────────────│            │            │             │
     │              │            │            │            │             │
     │─────────────CONTEXT──────────────────▶│            │             │
     │              │            │            │            │             │
     │◀───────────DECISION──────────────────│            │             │
     │              │            │            │            │             │
     │───────────────────────VALIDATE────────────────────▶│             │
     │              │            │            │            │             │
     │◀──────────────────────APPROVED───────────────────│             │
     │              │            │            │            │             │
     │──────────────────────────────────EXECUTE──────────────────────▶│
     │              │            │            │            │             │
     │◀─────────────────────────────────FILLED──────────────────────│
     │              │            │            │            │             │
```

### A.2 Risk Rejection

```
Coordinator     Technical      Trading       Risk
     │              │            │            │
     │─START_CYCLE─▶│            │            │
     │◀─SIGNALS─────│            │            │
     │              │            │            │
     │──────────CONTEXT────────▶│            │
     │◀─────────DECISION────────│            │
     │              │            │            │
     │──────────────────VALIDATE─────────────▶│
     │              │            │            │
     │◀─────────────────REJECTED─────────────│
     │              │         (max drawdown)  │
     │              │            │            │
     │─LOG_REJECTION───────────────────────▶ DB
```

---

*Document Version: 1.0*
*Last Updated: December 2025*
