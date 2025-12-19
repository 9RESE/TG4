"""
Coordinator Agent - Orchestrates agent execution and resolves conflicts.

The Coordinator:
- Schedules agent invocations at configured intervals
- Detects conflicts between agent outputs
- Uses LLM (DeepSeek V3 / Claude Sonnet fallback) for conflict resolution
- Manages trading workflow from signal to execution
- Handles emergencies and circuit breaker responses
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable, Optional, TYPE_CHECKING

from .message_bus import (
    Message,
    MessageBus,
    MessagePriority,
    MessageTopic,
    create_message,
)

if TYPE_CHECKING:
    from ..risk.rules_engine import RiskManagementEngine, TradeProposal

logger = logging.getLogger(__name__)


# Coordinator system prompt for conflict resolution
COORDINATOR_SYSTEM_PROMPT = """You are a trading system coordinator responsible for resolving conflicts between trading agents.

Your role is to analyze conflicting signals and decide:
1. Whether to PROCEED with the trade (possibly with modifications)
2. Whether to WAIT for better conditions
3. Whether to ABORT the trade entirely

Consider:
- Signal confidence levels from each agent
- Current market regime
- Risk management implications
- The specific nature of the conflict

Respond in JSON format:
{
    "action": "proceed" | "wait" | "abort" | "modify",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "modifications": {  // only if action is "modify"
        "leverage": N,
        "size_reduction_pct": N,
        "entry_adjustment_pct": N
    }
}"""


class CoordinatorState(Enum):
    """Coordinator operational state."""
    RUNNING = "running"      # Normal operation
    PAUSED = "paused"        # Analysis continues, no new trades
    HALTED = "halted"        # Circuit breaker triggered


@dataclass
class ScheduledTask:
    """Scheduled agent invocation configuration."""
    name: str
    agent_name: str
    interval_seconds: int
    symbols: list[str]
    handler: Callable[[str], Awaitable[Any]]
    last_run: Optional[datetime] = None
    enabled: bool = True
    run_on_start: bool = False

    def is_due(self, now: datetime) -> bool:
        """Check if task is due for execution."""
        if not self.enabled:
            return False
        if self.last_run is None:
            return self.run_on_start
        elapsed = (now - self.last_run).total_seconds()
        return elapsed >= self.interval_seconds


@dataclass
class ConflictResolution:
    """Result of coordinator conflict resolution."""
    action: str  # "proceed", "wait", "modify", "abort"
    reasoning: str
    confidence: float = 0.0
    modifications: Optional[dict] = None
    resolved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def should_proceed(self) -> bool:
        """Check if resolution allows trade to proceed."""
        return self.action in ["proceed", "modify"]


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""
    conflict_type: str
    description: str
    agents_involved: list[str]
    details: dict = field(default_factory=dict)


class CoordinatorAgent:
    """
    Orchestrates agent execution and resolves conflicts.

    The Coordinator:
    - Manages scheduled execution of all agents
    - Detects conflicts between agent signals
    - Uses LLM for conflict resolution when needed
    - Routes validated trades to execution
    - Handles circuit breaker responses

    LLM Usage:
    - Primary: DeepSeek V3 (for conflict resolution only)
    - Fallback: Claude Sonnet (if DeepSeek fails)
    """

    agent_name = "coordinator"

    def __init__(
        self,
        message_bus: MessageBus,
        agents: dict,  # name -> agent instance
        llm_client,
        config: dict,
        risk_engine: Optional['RiskManagementEngine'] = None,
        execution_manager=None,
        db_pool=None,
    ):
        """
        Initialize CoordinatorAgent.

        Args:
            message_bus: MessageBus for inter-agent communication
            agents: Dictionary of agent name -> agent instance
            llm_client: LLM client for conflict resolution
            config: Coordinator configuration
            risk_engine: RiskManagementEngine for trade validation
            execution_manager: OrderExecutionManager for trade execution
            db_pool: Database pool for persistence
        """
        self.bus = message_bus
        self.agents = agents
        self.llm = llm_client
        self.config = config
        self.risk_engine = risk_engine
        self.execution_manager = execution_manager
        self.db = db_pool

        # State
        self._state = CoordinatorState.RUNNING
        self._scheduled_tasks: list[ScheduledTask] = []
        self._main_loop_task: Optional[asyncio.Task] = None

        # LLM configuration
        llm_config = config.get('llm', {})
        primary = llm_config.get('primary', {})
        fallback = llm_config.get('fallback', {})
        self._primary_model = primary.get('model', 'deepseek-chat')
        self._primary_provider = primary.get('provider', 'deepseek')
        self._fallback_model = fallback.get('model', 'claude-3-5-sonnet-20241022')
        self._fallback_provider = fallback.get('provider', 'anthropic')

        # Conflict detection settings
        conflict_config = config.get('conflicts', {})
        self._confidence_diff_threshold = conflict_config.get('invoke_llm_threshold', 0.2)
        self._max_resolution_time_ms = conflict_config.get('max_resolution_time_ms', 10000)

        # Statistics
        self._total_task_runs = 0
        self._total_conflicts_detected = 0
        self._total_conflicts_resolved = 0
        self._total_trades_routed = 0

    async def start(self) -> None:
        """Start the coordinator main loop."""
        self._setup_schedules()
        await self._setup_subscriptions()
        self._main_loop_task = asyncio.create_task(self._main_loop())
        logger.info("CoordinatorAgent started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        self._state = CoordinatorState.HALTED
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        logger.info("CoordinatorAgent stopped")

    async def pause(self) -> None:
        """Pause trading (scheduled tasks still run for analysis)."""
        self._state = CoordinatorState.PAUSED
        await self.bus.publish(create_message(
            topic=MessageTopic.SYSTEM_EVENTS,
            source=self.agent_name,
            payload={"event": "coordinator_paused"},
            priority=MessagePriority.HIGH,
        ))
        logger.info("CoordinatorAgent paused")

    async def resume(self) -> None:
        """Resume from paused state."""
        if self._state == CoordinatorState.PAUSED:
            self._state = CoordinatorState.RUNNING
            await self.bus.publish(create_message(
                topic=MessageTopic.SYSTEM_EVENTS,
                source=self.agent_name,
                payload={"event": "coordinator_resumed"},
                priority=MessagePriority.HIGH,
            ))
            logger.info("CoordinatorAgent resumed")

    @property
    def state(self) -> CoordinatorState:
        """Get current coordinator state."""
        return self._state

    async def _main_loop(self) -> None:
        """Main coordinator loop - executes scheduled tasks."""
        while self._state != CoordinatorState.HALTED:
            try:
                if self._state in [CoordinatorState.RUNNING, CoordinatorState.PAUSED]:
                    await self._execute_due_tasks()
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Coordinator main loop error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on errors

    async def _execute_due_tasks(self) -> None:
        """Execute tasks that are due for execution."""
        now = datetime.now(timezone.utc)

        for task in self._scheduled_tasks:
            if not task.is_due(now):
                continue

            for symbol in task.symbols:
                try:
                    logger.debug(f"Executing task {task.name} for {symbol}")
                    await task.handler(symbol)
                    self._total_task_runs += 1
                except Exception as e:
                    logger.error(f"Task {task.name} failed for {symbol}: {e}", exc_info=True)
                    await self._handle_task_error(task, symbol, e)

            task.last_run = now

    def _setup_schedules(self) -> None:
        """Configure scheduled tasks from config."""
        schedules = self.config.get('schedules', {})
        symbols = self.config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # Technical Analysis - every minute
        ta_config = schedules.get('technical_analysis', {})
        if ta_config.get('enabled', True) and 'technical_analysis' in self.agents:
            self._scheduled_tasks.append(ScheduledTask(
                name="ta_analysis",
                agent_name="technical_analysis",
                interval_seconds=ta_config.get('interval_seconds', 60),
                symbols=symbols,
                handler=self._run_ta_agent,
                run_on_start=True,
            ))

        # Regime Detection - every 5 minutes
        regime_config = schedules.get('regime_detection', {})
        if regime_config.get('enabled', True) and 'regime_detection' in self.agents:
            self._scheduled_tasks.append(ScheduledTask(
                name="regime_detection",
                agent_name="regime_detection",
                interval_seconds=regime_config.get('interval_seconds', 300),
                symbols=symbols,
                handler=self._run_regime_agent,
            ))

        # Sentiment Analysis - every 30 minutes (Phase 4, disabled by default)
        sentiment_config = schedules.get('sentiment_analysis', {})
        if sentiment_config.get('enabled', False) and 'sentiment_analysis' in self.agents:
            self._scheduled_tasks.append(ScheduledTask(
                name="sentiment_analysis",
                agent_name="sentiment_analysis",
                interval_seconds=sentiment_config.get('interval_seconds', 1800),
                symbols=symbols,
                handler=self._run_sentiment_agent,
            ))

        # Trading Decision - every hour
        trading_config = schedules.get('trading_decision', {})
        if trading_config.get('enabled', True) and 'trading_decision' in self.agents:
            self._scheduled_tasks.append(ScheduledTask(
                name="trading_decision",
                agent_name="trading_decision",
                interval_seconds=trading_config.get('interval_seconds', 3600),
                symbols=symbols,
                handler=self._run_trading_agent,
            ))

        # Portfolio Rebalance - every hour
        rebalance_config = schedules.get('portfolio_rebalance', {})
        if rebalance_config.get('enabled', True) and 'portfolio_rebalance' in self.agents:
            self._scheduled_tasks.append(ScheduledTask(
                name="portfolio_check",
                agent_name="portfolio_rebalance",
                interval_seconds=rebalance_config.get('interval_seconds', 3600),
                symbols=["PORTFOLIO"],
                handler=self._check_portfolio_allocation,
            ))

        logger.info(f"Configured {len(self._scheduled_tasks)} scheduled tasks")

    async def _setup_subscriptions(self) -> None:
        """Subscribe to relevant message topics."""
        # Subscribe to trading signals
        await self.bus.subscribe(
            subscriber_id=self.agent_name,
            topic=MessageTopic.TRADING_SIGNALS,
            handler=self._handle_trading_signal,
        )

        # Subscribe to risk alerts
        await self.bus.subscribe(
            subscriber_id=self.agent_name,
            topic=MessageTopic.RISK_ALERTS,
            handler=self._handle_risk_alert,
        )

        # Subscribe to execution events
        await self.bus.subscribe(
            subscriber_id=self.agent_name,
            topic=MessageTopic.EXECUTION_EVENTS,
            handler=self._handle_execution_event,
        )

    # -------------------------------------------------------------------------
    # Agent Execution Handlers
    # -------------------------------------------------------------------------

    async def _run_ta_agent(self, symbol: str) -> None:
        """Run Technical Analysis agent for a symbol."""
        if 'technical_analysis' not in self.agents:
            return

        agent = self.agents['technical_analysis']
        snapshot = await self._get_market_snapshot(symbol)

        if snapshot:
            output = await agent.process(snapshot)

            # Publish TA signals
            await self.bus.publish(create_message(
                topic=MessageTopic.TA_SIGNALS,
                source="technical_analysis",
                payload=output.to_dict(),
            ))

    async def _run_regime_agent(self, symbol: str) -> None:
        """Run Regime Detection agent for a symbol."""
        if 'regime_detection' not in self.agents:
            return

        agent = self.agents['regime_detection']
        snapshot = await self._get_market_snapshot(symbol)

        if snapshot:
            # Get latest TA output
            ta_output = None
            if 'technical_analysis' in self.agents:
                ta_output = self.agents['technical_analysis'].last_output

            output = await agent.process(snapshot, ta_output=ta_output)

            # Publish regime update
            await self.bus.publish(create_message(
                topic=MessageTopic.REGIME_UPDATES,
                source="regime_detection",
                payload=output.to_dict(),
            ))

    async def _run_sentiment_agent(self, symbol: str) -> None:
        """Run Sentiment Analysis agent (Phase 4)."""
        # Phase 4 - Sentiment agent not implemented yet
        pass

    async def _run_trading_agent(self, symbol: str) -> None:
        """Run Trading Decision agent for a symbol."""
        if 'trading_decision' not in self.agents:
            return

        agent = self.agents['trading_decision']
        snapshot = await self._get_market_snapshot(symbol)

        if snapshot:
            # Get supporting agent outputs
            ta_output = None
            regime_output = None

            if 'technical_analysis' in self.agents:
                ta_output = self.agents['technical_analysis'].last_output

            if 'regime_detection' in self.agents:
                regime_output = self.agents['regime_detection'].last_output

            output = await agent.process(
                snapshot,
                ta_output=ta_output,
                regime_output=regime_output,
            )

            # Publish trading signal
            await self.bus.publish(create_message(
                topic=MessageTopic.TRADING_SIGNALS,
                source="trading_decision",
                payload=output.to_dict(),
                priority=MessagePriority.HIGH,
            ))

    async def _check_portfolio_allocation(self, _: str) -> None:
        """Check portfolio allocation and trigger rebalancing if needed."""
        if 'portfolio_rebalance' not in self.agents:
            return

        agent = self.agents['portfolio_rebalance']
        output = await agent.process()

        if output.action == "rebalance":
            # Publish portfolio update with rebalance trades
            await self.bus.publish(create_message(
                topic=MessageTopic.PORTFOLIO_UPDATES,
                source="portfolio_rebalance",
                payload=output.to_dict(),
                priority=MessagePriority.HIGH,
            ))

    async def _get_market_snapshot(self, symbol: str):
        """Get market snapshot for a symbol."""
        if 'snapshot_builder' in self.agents:
            return await self.agents['snapshot_builder'].build_snapshot(symbol)
        return None

    # -------------------------------------------------------------------------
    # Message Handlers
    # -------------------------------------------------------------------------

    async def _handle_trading_signal(self, message: Message) -> None:
        """
        Handle trading signal from Trading Decision Agent.

        Validates with Risk Engine and executes if approved.
        """
        if self._state != CoordinatorState.RUNNING:
            logger.debug(f"Skipping trading signal - coordinator state: {self._state.value}")
            return

        signal = message.payload
        symbol = signal.get("symbol", "")

        # Skip HOLD signals
        if signal.get("action") == "HOLD":
            logger.debug(f"HOLD signal for {symbol} - no action needed")
            return

        # Check for conflicts
        conflicts = await self._detect_conflicts(signal)

        if conflicts:
            self._total_conflicts_detected += 1
            resolution = await self._resolve_conflicts(signal, conflicts)

            if not resolution.should_proceed():
                logger.info(f"Conflict resolution: {resolution.action} - {resolution.reasoning}")
                return

            if resolution.action == "modify" and resolution.modifications:
                signal = self._apply_modifications(signal, resolution.modifications)

        # Route to execution
        await self._route_to_execution(signal)

    async def _handle_risk_alert(self, message: Message) -> None:
        """Handle risk alerts (circuit breaker, etc.)."""
        alert = message.payload
        alert_type = alert.get("alert_type", "")

        if alert_type == "circuit_breaker":
            severity = alert.get("severity", "low")
            if severity in ["high", "critical"]:
                self._state = CoordinatorState.HALTED
                logger.warning(f"Circuit breaker triggered: {alert.get('message')}")

    async def _handle_execution_event(self, message: Message) -> None:
        """Handle execution events (fills, cancels, errors)."""
        event = message.payload
        event_type = event.get("event_type", "")

        if event_type == "order_filled":
            logger.info(f"Order filled: {event.get('order_id')}")
        elif event_type == "order_error":
            logger.error(f"Order error: {event.get('error_message')}")

    async def _handle_task_error(
        self,
        task: ScheduledTask,
        symbol: str,
        error: Exception,
    ) -> None:
        """Handle task execution errors."""
        await self.bus.publish(create_message(
            topic=MessageTopic.SYSTEM_EVENTS,
            source=self.agent_name,
            payload={
                "event": "task_error",
                "task": task.name,
                "symbol": symbol,
                "error": str(error),
            },
            priority=MessagePriority.HIGH,
        ))

    # -------------------------------------------------------------------------
    # Conflict Detection and Resolution
    # -------------------------------------------------------------------------

    async def _detect_conflicts(self, signal: dict) -> list[ConflictInfo]:
        """Detect conflicts between agent outputs."""
        conflicts = []
        symbol = signal.get("symbol", "")

        # Get latest outputs from message bus
        ta_msg = await self.bus.get_latest(MessageTopic.TA_SIGNALS, max_age_seconds=120)
        regime_msg = await self.bus.get_latest(MessageTopic.REGIME_UPDATES, max_age_seconds=600)
        sentiment_msg = await self.bus.get_latest(MessageTopic.SENTIMENT_UPDATES, max_age_seconds=3600)

        ta_payload = ta_msg.payload if ta_msg else {}
        regime_payload = regime_msg.payload if regime_msg else {}
        sentiment_payload = sentiment_msg.payload if sentiment_msg else {}

        # Check TA vs Sentiment conflict
        if ta_payload and sentiment_payload:
            ta_bias = ta_payload.get("bias", "neutral")
            sent_bias = sentiment_payload.get("sentiment_bias", "neutral")

            # Opposing bias?
            opposing = (
                (ta_bias == "long" and sent_bias == "bearish") or
                (ta_bias == "short" and sent_bias == "bullish")
            )

            if opposing:
                ta_conf = ta_payload.get("confidence", 0.5)
                sent_conf = sentiment_payload.get("confidence", 0.5)
                conf_diff = abs(ta_conf - sent_conf)

                if conf_diff < self._confidence_diff_threshold:
                    conflicts.append(ConflictInfo(
                        conflict_type="ta_sentiment_conflict",
                        description=f"TA ({ta_bias}) vs Sentiment ({sent_bias}) with close confidence",
                        agents_involved=["technical_analysis", "sentiment_analysis"],
                        details={
                            "ta_bias": ta_bias,
                            "ta_confidence": ta_conf,
                            "sentiment_bias": sent_bias,
                            "sentiment_confidence": sent_conf,
                            "confidence_diff": conf_diff,
                        }
                    ))

        # Check regime appropriateness
        if regime_payload:
            regime_type = regime_payload.get("regime", "neutral")
            action = signal.get("action", "HOLD")

            # Trading in choppy market?
            if regime_type == "choppy" and action != "HOLD":
                conflicts.append(ConflictInfo(
                    conflict_type="regime_conflict",
                    description="Trading signal in choppy regime",
                    agents_involved=["regime_detection", "trading_decision"],
                    details={
                        "regime": regime_type,
                        "proposed_action": action,
                        "recommendation": "Avoid trading in choppy regime",
                    }
                ))

        return conflicts

    async def _resolve_conflicts(
        self,
        signal: dict,
        conflicts: list[ConflictInfo],
    ) -> ConflictResolution:
        """
        Use LLM to resolve conflicts between agents.

        Invoked only when conflicts are detected.
        """
        start_time = time.perf_counter()

        # Build conflict resolution prompt
        prompt = self._build_conflict_prompt(signal, conflicts)

        try:
            # Try primary model (DeepSeek V3)
            response = await self._call_llm_for_resolution(prompt)
            resolution = self._parse_resolution(response)
            self._total_conflicts_resolved += 1

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(f"Conflict resolved in {latency_ms}ms: {resolution.action}")

            return resolution

        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            # Default to conservative action
            return ConflictResolution(
                action="wait",
                reasoning=f"Resolution failed: {str(e)}",
                confidence=0.0,
            )

    def _build_conflict_prompt(
        self,
        signal: dict,
        conflicts: list[ConflictInfo],
    ) -> str:
        """Build prompt for conflict resolution."""
        conflict_details = []
        for c in conflicts:
            conflict_details.append(f"- {c.conflict_type}: {c.description}")
            for k, v in c.details.items():
                conflict_details.append(f"    {k}: {v}")

        return f"""Trading Signal Analysis Required

PROPOSED TRADE:
- Symbol: {signal.get('symbol')}
- Action: {signal.get('action')}
- Confidence: {signal.get('confidence', 0):.2f}
- Entry Price: {signal.get('entry_price')}
- Stop Loss: {signal.get('stop_loss')}
- Take Profit: {signal.get('take_profit')}

CONFLICTS DETECTED:
{chr(10).join(conflict_details)}

Based on this information, what action should be taken?
Remember to respond in JSON format with action, confidence, reasoning, and optional modifications."""

    async def _call_llm_for_resolution(self, prompt: str) -> str:
        """Call LLM for conflict resolution with fallback."""
        try:
            response = await self.llm.generate(
                model=self._primary_model,
                system_prompt=COORDINATOR_SYSTEM_PROMPT,
                user_message=prompt,
                max_tokens=500,
            )
            return response.text
        except Exception as e:
            logger.warning(f"Primary LLM failed, trying fallback: {e}")
            # Fallback to Claude Sonnet
            response = await self.llm.generate(
                model=self._fallback_model,
                system_prompt=COORDINATOR_SYSTEM_PROMPT,
                user_message=prompt,
                max_tokens=500,
            )
            return response.text

    def _parse_resolution(self, response_text: str) -> ConflictResolution:
        """Parse LLM response into ConflictResolution."""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                return ConflictResolution(
                    action=data.get("action", "wait"),
                    reasoning=data.get("reasoning", ""),
                    confidence=data.get("confidence", 0.5),
                    modifications=data.get("modifications"),
                )
        except json.JSONDecodeError:
            logger.error(f"Failed to parse resolution JSON: {response_text}")

        # Default conservative resolution
        return ConflictResolution(
            action="wait",
            reasoning="Could not parse LLM response",
            confidence=0.0,
        )

    def _apply_modifications(self, signal: dict, modifications: dict) -> dict:
        """Apply resolution modifications to signal."""
        modified = signal.copy()

        if "leverage" in modifications:
            modified["leverage"] = modifications["leverage"]

        if "size_reduction_pct" in modifications:
            original_size = modified.get("size_usd", 0)
            reduction = modifications["size_reduction_pct"] / 100
            modified["size_usd"] = original_size * (1 - reduction)

        if "entry_adjustment_pct" in modifications:
            original_entry = modified.get("entry_price", 0)
            adjustment = modifications["entry_adjustment_pct"] / 100
            modified["entry_price"] = original_entry * (1 + adjustment)

        return modified

    # -------------------------------------------------------------------------
    # Trade Routing
    # -------------------------------------------------------------------------

    async def _route_to_execution(self, signal: dict) -> None:
        """Route validated signal to execution manager."""
        if not self.risk_engine or not self.execution_manager:
            logger.warning("Risk engine or execution manager not configured")
            return

        from ..risk.rules_engine import TradeProposal

        # Create trade proposal
        proposal = TradeProposal(
            symbol=signal.get("symbol", ""),
            side="buy" if signal.get("action") == "BUY" else "sell",
            size_usd=signal.get("size_usd", 100),
            entry_price=signal.get("entry_price", 0),
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit"),
            leverage=signal.get("leverage", 1),
            confidence=signal.get("confidence", 0.5),
            regime=signal.get("regime", "ranging"),
        )

        # Validate with risk engine
        validation = self.risk_engine.validate_trade(proposal)

        if not validation.is_approved():
            logger.info(f"Trade rejected by risk engine: {validation.rejections}")
            await self.bus.publish(create_message(
                topic=MessageTopic.RISK_ALERTS,
                source=self.agent_name,
                payload={
                    "alert_type": "trade_rejected",
                    "symbol": proposal.symbol,
                    "rejections": validation.rejections,
                },
            ))
            return

        # Use modified proposal if applicable
        final_proposal = validation.modified_proposal or proposal

        # Execute trade
        try:
            result = await self.execution_manager.execute_trade(final_proposal)
            self._total_trades_routed += 1

            if result.success:
                await self.bus.publish(create_message(
                    topic=MessageTopic.EXECUTION_EVENTS,
                    source=self.agent_name,
                    payload={
                        "event_type": "order_placed",
                        "order_id": result.order.id if result.order else None,
                        "symbol": final_proposal.symbol,
                    },
                ))
            else:
                await self.bus.publish(create_message(
                    topic=MessageTopic.EXECUTION_EVENTS,
                    source=self.agent_name,
                    payload={
                        "event_type": "order_error",
                        "symbol": final_proposal.symbol,
                        "error_message": result.error_message,
                    },
                    priority=MessagePriority.HIGH,
                ))

        except Exception as e:
            logger.error(f"Execution failed: {e}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get coordinator status."""
        return {
            "state": self._state.value,
            "scheduled_tasks": [
                {
                    "name": t.name,
                    "agent": t.agent_name,
                    "interval_seconds": t.interval_seconds,
                    "enabled": t.enabled,
                    "last_run": t.last_run.isoformat() if t.last_run else None,
                }
                for t in self._scheduled_tasks
            ],
            "statistics": {
                "total_task_runs": self._total_task_runs,
                "total_conflicts_detected": self._total_conflicts_detected,
                "total_conflicts_resolved": self._total_conflicts_resolved,
                "total_trades_routed": self._total_trades_routed,
            },
        }

    def enable_task(self, task_name: str) -> bool:
        """Enable a scheduled task."""
        for task in self._scheduled_tasks:
            if task.name == task_name:
                task.enabled = True
                return True
        return False

    def disable_task(self, task_name: str) -> bool:
        """Disable a scheduled task."""
        for task in self._scheduled_tasks:
            if task.name == task_name:
                task.enabled = False
                return True
        return False

    async def force_run_task(self, task_name: str, symbol: str) -> bool:
        """Force immediate execution of a task."""
        for task in self._scheduled_tasks:
            if task.name == task_name:
                await task.handler(symbol)
                return True
        return False
