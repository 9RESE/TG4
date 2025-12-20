"""
Coordinator Agent - Orchestrates agent execution and resolves conflicts.

The Coordinator:
- Schedules agent invocations at configured intervals
- Detects conflicts between agent outputs
- Uses LLM (DeepSeek V3 / Claude Sonnet fallback) for conflict resolution
- Manages trading workflow from signal to execution
- Handles emergencies and circuit breaker responses
- Supports paper trading mode (Phase 6) with simulated execution

Phase 6 Additions:
- TradingMode integration (paper vs live)
- PaperOrderExecutor for simulated trades
- PaperPortfolio for balance tracking
- PaperPriceSource for realistic pricing
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
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
    from ..execution.paper_portfolio import PaperPortfolio
    from ..execution.paper_executor import PaperOrderExecutor
    from ..execution.paper_price_source import PaperPriceSource

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


class DegradationLevel(Enum):
    """
    Graceful degradation levels for system resilience.

    Levels are triggered automatically based on system health:
    - NORMAL: All systems operational
    - REDUCED: Some non-critical services degraded
    - LIMITED: Only essential services running
    - EMERGENCY: Minimum viable operation
    """
    NORMAL = 0       # All systems operational
    REDUCED = 1      # Skip non-critical agents (sentiment)
    LIMITED = 2      # Skip optional agents, reduce LLM calls
    EMERGENCY = 3    # Only risk-based decisions, no LLM


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
    depends_on: list[str] = field(default_factory=list)
    timeout_seconds: int = 30  # F08: Individual task timeout
    max_concurrent_symbols: int = 3  # F05: Max parallel symbol executions
    # F01: Concurrent execution guard
    _running: bool = field(default=False, repr=False)
    _started_at: Optional[datetime] = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def is_due(self, now: datetime) -> bool:
        """Check if task is due for execution."""
        if not self.enabled:
            return False
        # F01: Don't run if already running
        if self._running:
            return False
        if self.last_run is None:
            return self.run_on_start
        elapsed = (now - self.last_run).total_seconds()
        return elapsed >= self.interval_seconds

    def dependencies_satisfied(self, task_map: dict[str, 'ScheduledTask']) -> bool:
        """
        Check if all dependencies have run more recently than this task.

        F03: Task dependency enforcement - ensures tasks run in correct order.
        """
        if not self.depends_on:
            return True

        for dep_name in self.depends_on:
            dep_task = task_map.get(dep_name)
            if not dep_task:
                logger.warning(f"Task {self.name} depends on unknown task: {dep_name}")
                continue
            if dep_task.last_run is None:
                # Dependency hasn't run yet
                return False
            if self.last_run and dep_task.last_run < self.last_run:
                # Dependency hasn't run since we last ran
                return False
        return True


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
        trading_mode=None,  # Phase 6: TradingMode enum
        execution_config: Optional[dict] = None,  # Phase 6: execution.yaml config
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
            trading_mode: TradingMode enum (PAPER or LIVE) - Phase 6
            execution_config: Execution configuration (from execution.yaml) - Phase 6
        """
        self.bus = message_bus
        self.agents = agents
        self.llm = llm_client
        self.config = config
        self.risk_engine = risk_engine
        self.execution_manager = execution_manager
        self.db = db_pool
        self.execution_config = execution_config or {}

        # Phase 6: Trading mode initialization
        self._init_trading_mode(trading_mode)

        # State
        self._state = CoordinatorState.RUNNING
        self._degradation_level = DegradationLevel.NORMAL
        self._scheduled_tasks: list[ScheduledTask] = []
        self._main_loop_task: Optional[asyncio.Task] = None

        # Health tracking for degradation
        self._consecutive_llm_failures = 0
        self._consecutive_api_failures = 0
        self._max_failures_for_degradation = 3

        # F04: Hysteresis for degradation recovery
        self._recovery_threshold = config.get('degradation', {}).get('recovery_threshold', 3)
        self._success_count_for_recovery = 0

        # F07: Thread safety for scheduled trades
        self._scheduled_trades: list = []
        self._scheduled_trades_lock = asyncio.Lock()

        # F06: Track in-flight tasks for restart recovery
        self._task_map: dict[str, ScheduledTask] = {}  # name -> task for dependency lookup

        # F15: Emergency config-driven response
        self._emergency_config = config.get('emergency', {})

        # F16: Consensus thresholds (configurable)
        consensus_config = config.get('consensus', {})
        self._consensus_high_threshold = consensus_config.get('high_agreement_threshold', 0.66)
        self._consensus_low_threshold = consensus_config.get('low_agreement_threshold', 0.33)
        self._consensus_max_boost = consensus_config.get('max_confidence_boost', 0.30)
        self._consensus_max_penalty = consensus_config.get('max_confidence_penalty', 0.15)

        # F02: Max concurrent tasks to prevent starvation
        self._max_concurrent_tasks = config.get('schedules', {}).get('max_concurrent_tasks', 5)

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

    def _init_trading_mode(self, trading_mode) -> None:
        """
        Initialize trading mode and paper trading infrastructure.

        Phase 6: Paper Trading Integration

        Sets up:
        - Trading mode (PAPER or LIVE)
        - PaperPortfolio for balance tracking (paper mode)
        - PaperPriceSource for pricing (paper mode)
        - PaperOrderExecutor for simulated execution (paper mode)
        """
        from ..execution.trading_mode import TradingMode, get_trading_mode

        # Determine trading mode
        if trading_mode is None:
            self.trading_mode = get_trading_mode(self.execution_config)
        else:
            self.trading_mode = trading_mode

        # Initialize paper trading components if in paper mode
        self.paper_portfolio = None
        self.paper_price_source = None
        self.paper_executor = None

        if self.trading_mode == TradingMode.PAPER:
            self._init_paper_trading()
            logger.info("🟢 Coordinator initialized in PAPER trading mode")
        else:
            logger.warning("🔴 Coordinator initialized in LIVE trading mode")

    def _init_paper_trading(self) -> None:
        """
        Initialize paper trading components.

        Creates:
        - PaperPortfolio with initial balances from config
        - PaperPriceSource for realistic pricing
        - PaperOrderExecutor for simulated execution

        Note: Portfolio restoration from DB happens in start() for async support.
        """
        from ..execution.paper_portfolio import PaperPortfolio
        from ..execution.paper_price_source import PaperPriceSource
        from ..execution.paper_executor import PaperOrderExecutor

        # Create paper portfolio from config (will be replaced if DB restore succeeds)
        self.paper_portfolio = PaperPortfolio.from_config(self.execution_config)
        logger.info(f"Paper portfolio initialized: {self.paper_portfolio.get_balances_dict()}")

        # Create paper price source
        paper_config = self.execution_config.get("paper_trading", {})
        price_source_type = paper_config.get("price_source", "live_feed")

        self.paper_price_source = PaperPriceSource(
            source_type=price_source_type,
            db_connection=self.db,
            websocket_feed=None,  # Will be set if WS feed is available
            config=self.execution_config,
        )

        # Create paper executor
        self.paper_executor = PaperOrderExecutor(
            config=self.execution_config,
            paper_portfolio=self.paper_portfolio,
            price_source=self.paper_price_source.get_price,
            position_tracker=None,  # Will be set when position tracker is available
        )

    def set_websocket_feed(self, ws_feed) -> None:
        """
        Set WebSocket feed for real-time prices in paper mode.

        Args:
            ws_feed: WebSocket feed instance with get_price/get_last_price method
        """
        if self.paper_price_source:
            self.paper_price_source.ws_feed = ws_feed
            logger.info("WebSocket feed connected to paper price source")

    def set_position_tracker(self, position_tracker) -> None:
        """
        Set position tracker for paper trading.

        Args:
            position_tracker: PositionTracker instance
        """
        if self.paper_executor:
            self.paper_executor.position_tracker = position_tracker
            logger.info("Position tracker connected to paper executor")

    async def start(self) -> None:
        """Start the coordinator main loop."""
        self._setup_schedules()
        # Load persisted state before starting
        await self.load_state()

        # HIGH-01: Restore paper portfolio from database if configured
        await self._restore_paper_portfolio()

        await self._setup_subscriptions()
        self._main_loop_task = asyncio.create_task(self._main_loop())
        logger.info("CoordinatorAgent started")

    async def _restore_paper_portfolio(self) -> None:
        """
        Restore paper portfolio from database if persist_state is enabled.

        HIGH-01: Paper trading session persistence.
        """
        from ..execution.trading_mode import TradingMode

        if self.trading_mode != TradingMode.PAPER:
            return

        paper_config = self.execution_config.get("paper_trading", {})
        if not paper_config.get("persist_state", True):
            logger.debug("Paper portfolio persistence disabled")
            return

        if not self.db:
            logger.debug("No database connection - skipping portfolio restore")
            return

        try:
            from ..execution.paper_portfolio import PaperPortfolio

            restored = await PaperPortfolio.load_from_db(self.db)
            if restored:
                self.paper_portfolio = restored
                # Reconnect executor to restored portfolio
                if self.paper_executor:
                    self.paper_executor.portfolio = self.paper_portfolio
                logger.info(
                    f"📝 Paper portfolio restored: session={restored.session_id}, "
                    f"balances={restored.get_balances_dict()}"
                )
            else:
                logger.info("No previous paper session found - using fresh portfolio")

        except Exception as e:
            logger.error(f"Failed to restore paper portfolio: {e}")

    async def stop(self) -> None:
        """Stop the coordinator and persist state."""
        self._state = CoordinatorState.HALTED

        # HIGH-01: Persist paper portfolio before stopping
        await self._persist_paper_portfolio()

        # Persist state before stopping
        await self.persist_state()
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        logger.info("CoordinatorAgent stopped")

    async def _persist_paper_portfolio(self) -> None:
        """
        Persist paper portfolio to database on shutdown.

        HIGH-01: Paper trading session persistence.
        """
        from ..execution.trading_mode import TradingMode

        if self.trading_mode != TradingMode.PAPER:
            return

        paper_config = self.execution_config.get("paper_trading", {})
        if not paper_config.get("persist_state", True):
            return

        if not self.paper_portfolio or not self.db:
            return

        try:
            await self.paper_portfolio.persist_to_db(self.db)
            logger.info(f"📝 Paper portfolio persisted on shutdown: {self.paper_portfolio.session_id}")
        except Exception as e:
            logger.error(f"Failed to persist paper portfolio on shutdown: {e}")

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

    @property
    def degradation_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self._degradation_level

    def _check_degradation_level(self) -> None:
        """
        Update degradation level based on system health.

        Called after failures to potentially increase degradation,
        or periodically to recover from degradation.
        """
        # Increase degradation on consecutive failures
        total_failures = self._consecutive_llm_failures + self._consecutive_api_failures

        if total_failures >= self._max_failures_for_degradation * 3:
            new_level = DegradationLevel.EMERGENCY
        elif total_failures >= self._max_failures_for_degradation * 2:
            new_level = DegradationLevel.LIMITED
        elif total_failures >= self._max_failures_for_degradation:
            new_level = DegradationLevel.REDUCED
        else:
            new_level = DegradationLevel.NORMAL

        if new_level != self._degradation_level:
            old_level = self._degradation_level
            self._degradation_level = new_level
            logger.warning(
                f"Degradation level changed: {old_level.name} -> {new_level.name} "
                f"(LLM failures: {self._consecutive_llm_failures}, "
                f"API failures: {self._consecutive_api_failures})"
            )
            # Publish degradation change event (including recovery to NORMAL)
            asyncio.create_task(self._publish_degradation_event(old_level, new_level))

    def _record_llm_success(self) -> None:
        """
        Record successful LLM call with gradual recovery (F04 fix).

        Uses hysteresis to prevent rapid oscillation between degradation levels.
        Multiple consecutive successes are required to recover, but a single
        success contributes to the recovery counter.
        """
        self._success_count_for_recovery += 1
        if self._success_count_for_recovery >= self._recovery_threshold:
            # Only decrement (not reset to 0) for gradual recovery
            if self._consecutive_llm_failures > 0:
                self._consecutive_llm_failures -= 1
                logger.debug(
                    f"LLM recovery progress: {self._success_count_for_recovery} successes, "
                    f"failures now: {self._consecutive_llm_failures}"
                )
            self._success_count_for_recovery = 0
        self._check_degradation_level()

    def _record_llm_failure(self) -> None:
        """
        Record LLM failure and potentially increase degradation (F04 fix).

        Resets recovery progress on failure to prevent oscillation.
        """
        self._consecutive_llm_failures += 1
        self._success_count_for_recovery = 0  # Reset recovery progress
        self._check_degradation_level()

    def _record_api_success(self) -> None:
        """Record successful API call with gradual recovery."""
        # API success also contributes to recovery
        self._success_count_for_recovery += 1
        if self._success_count_for_recovery >= self._recovery_threshold:
            if self._consecutive_api_failures > 0:
                self._consecutive_api_failures -= 1
            self._success_count_for_recovery = 0
        self._check_degradation_level()

    def _record_api_failure(self) -> None:
        """Record API failure and potentially increase degradation."""
        self._consecutive_api_failures += 1
        self._success_count_for_recovery = 0  # Reset recovery progress
        self._check_degradation_level()

    async def _publish_degradation_event(
        self,
        old_level: DegradationLevel,
        new_level: DegradationLevel,
    ) -> None:
        """
        Publish degradation level change event.

        This allows other components to react to degradation changes,
        including recovery to NORMAL level.
        """
        event_type = "degradation_recovery" if new_level == DegradationLevel.NORMAL else "degradation_increased"
        await self.bus.publish(create_message(
            topic=MessageTopic.SYSTEM_EVENTS,
            source=self.agent_name,
            payload={
                "event": event_type,
                "old_level": old_level.name,
                "new_level": new_level.name,
                "llm_failures": self._consecutive_llm_failures,
                "api_failures": self._consecutive_api_failures,
            },
            priority=MessagePriority.HIGH,
        ))

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
        """
        Execute tasks that are due for execution.

        F01: Concurrent task execution guard prevents same task running twice
        F02: Parallel task execution prevents starvation from slow tasks
        F03: Task dependency enforcement ensures correct execution order
        F05: Parallel symbol execution within tasks for performance
        F08: Individual task timeouts prevent indefinite blocking
        """
        now = datetime.now(timezone.utc)

        # Execute scheduled DCA trades first
        await self._execute_scheduled_trades()

        # Collect due tasks that satisfy dependencies
        due_tasks = []
        for task in self._scheduled_tasks:
            if not task.is_due(now):
                continue
            # F03: Check dependencies
            if not task.dependencies_satisfied(self._task_map):
                logger.debug(f"Task {task.name} waiting for dependencies: {task.depends_on}")
                continue
            due_tasks.append(task)

        if not due_tasks:
            return

        # F02: Execute tasks concurrently with semaphore to prevent starvation
        semaphore = asyncio.Semaphore(self._max_concurrent_tasks)

        async def execute_task(task: ScheduledTask) -> None:
            """Execute a single task with all its symbols."""
            # F01: Acquire lock and check if already running
            async with task._lock:
                if task._running:
                    logger.debug(f"Task {task.name} already running, skipping")
                    return
                task._running = True
                task._started_at = datetime.now(timezone.utc)

            try:
                async with semaphore:
                    await self._execute_task_for_symbols(task)
                    task.last_run = datetime.now(timezone.utc)
            finally:
                task._running = False
                task._started_at = None

        # Execute all due tasks concurrently
        await asyncio.gather(
            *[execute_task(t) for t in due_tasks],
            return_exceptions=True
        )

    async def _execute_task_for_symbols(self, task: ScheduledTask) -> None:
        """
        Execute task for all symbols in parallel (F05).

        Uses semaphore to limit concurrent symbol executions.
        """
        # F05: Execute symbols in parallel with semaphore
        symbol_semaphore = asyncio.Semaphore(task.max_concurrent_symbols)

        async def run_for_symbol(symbol: str) -> None:
            async with symbol_semaphore:
                try:
                    # F08: Apply timeout to individual agent execution
                    logger.debug(f"Executing task {task.name} for {symbol}")
                    await asyncio.wait_for(
                        task.handler(symbol),
                        timeout=task.timeout_seconds
                    )
                    self._total_task_runs += 1
                except asyncio.TimeoutError:
                    logger.error(f"Task {task.name} timed out for {symbol} after {task.timeout_seconds}s")
                    self._record_api_failure()
                    await self._handle_task_error(
                        task, symbol,
                        TimeoutError(f"Task timed out after {task.timeout_seconds}s")
                    )
                except Exception as e:
                    logger.error(f"Task {task.name} failed for {symbol}: {e}", exc_info=True)
                    await self._handle_task_error(task, symbol, e)

        await asyncio.gather(
            *[run_for_symbol(s) for s in task.symbols],
            return_exceptions=True
        )

    def _setup_schedules(self) -> None:
        """
        Configure scheduled tasks from config.

        F03: Now includes dependency loading from config.
        F08: Includes timeout_seconds from config.
        """
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
                run_on_start=ta_config.get('run_on_start', True),
                depends_on=[],  # No dependencies
                timeout_seconds=ta_config.get('timeout_seconds', 30),
                max_concurrent_symbols=ta_config.get('max_concurrent', 3),
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
                run_on_start=regime_config.get('run_on_start', False),
                depends_on=regime_config.get('depends_on', ['ta_analysis']),  # F03
                timeout_seconds=regime_config.get('timeout_seconds', 30),
                max_concurrent_symbols=regime_config.get('max_concurrent', 3),
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
                run_on_start=sentiment_config.get('run_on_start', False),
                depends_on=sentiment_config.get('depends_on', []),
                timeout_seconds=sentiment_config.get('timeout_seconds', 60),
                max_concurrent_symbols=sentiment_config.get('max_concurrent', 3),
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
                run_on_start=trading_config.get('run_on_start', False),
                depends_on=trading_config.get('depends_on', ['ta_analysis', 'regime_detection']),  # F03
                timeout_seconds=trading_config.get('timeout_seconds', 60),
                max_concurrent_symbols=trading_config.get('max_concurrent', 3),
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
                run_on_start=rebalance_config.get('run_on_start', False),
                depends_on=rebalance_config.get('depends_on', []),
                timeout_seconds=rebalance_config.get('timeout_seconds', 30),
                max_concurrent_symbols=1,  # Portfolio is single entity
            ))

        # F03/F06: Build task map for dependency lookup and in-flight tracking
        self._task_map = {task.name: task for task in self._scheduled_tasks}

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

        # F13: Subscribe to coordinator commands (pause, resume, etc.)
        await self.bus.subscribe(
            subscriber_id=self.agent_name,
            topic=MessageTopic.COORDINATOR_COMMANDS,
            handler=self._handle_coordinator_command,
        )

    async def _handle_coordinator_command(self, message: Message) -> None:
        """
        Handle coordinator commands (F13).

        Supported commands:
        - pause: Pause trading (analysis continues)
        - resume: Resume trading
        - halt: Emergency halt
        - enable_task: Enable a specific task
        - disable_task: Disable a specific task
        - force_run: Force immediate task execution
        """
        command = message.payload.get("command", "")
        params = message.payload.get("params", {})

        logger.info(f"Received coordinator command: {command}")

        if command == "pause":
            await self.pause()
        elif command == "resume":
            await self.resume()
        elif command == "halt":
            self._state = CoordinatorState.HALTED
            logger.warning("Coordinator halted via command")
        elif command == "enable_task":
            task_name = params.get("task_name", "")
            if self.enable_task(task_name):
                logger.info(f"Task {task_name} enabled via command")
            else:
                logger.warning(f"Failed to enable task: {task_name}")
        elif command == "disable_task":
            task_name = params.get("task_name", "")
            if self.disable_task(task_name):
                logger.info(f"Task {task_name} disabled via command")
            else:
                logger.warning(f"Failed to disable task: {task_name}")
        elif command == "force_run":
            task_name = params.get("task_name", "")
            symbol = params.get("symbol", "")
            if await self.force_run_task(task_name, symbol):
                logger.info(f"Force ran task {task_name} for {symbol}")
            else:
                logger.warning(f"Failed to force run task: {task_name}")
        else:
            logger.warning(f"Unknown coordinator command: {command}")

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

            # Route rebalance trades to execution
            await self._execute_rebalance_trades(output)

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
        Includes consensus building to amplify/reduce confidence.
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

        # Build consensus from multiple agents
        consensus_multiplier = await self._build_consensus(signal)

        # Apply consensus multiplier to confidence
        original_confidence = signal.get("confidence", 0.5)
        adjusted_confidence = min(1.0, original_confidence * consensus_multiplier)
        signal["confidence"] = adjusted_confidence
        signal["consensus_multiplier"] = consensus_multiplier

        logger.debug(
            f"Confidence adjusted: {original_confidence:.2f} -> {adjusted_confidence:.2f} "
            f"(consensus: {consensus_multiplier:.2f})"
        )

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
        """
        Handle risk alerts with config-driven responses (F15).

        Uses emergency config to determine actions:
        - daily_loss: pause_trading
        - weekly_loss: reduce_positions
        - max_drawdown: halt_all, close_positions

        Falls back to severity-based handling if breach_type not configured.
        """
        alert = message.payload
        alert_type = alert.get("alert_type", "")
        breach_type = alert.get("breach_type", "")  # e.g., "daily_loss", "weekly_loss", "max_drawdown"
        severity = alert.get("severity", "low")

        if alert_type == "circuit_breaker":
            # F15: Get config-driven action for this breach type
            cb_config = self._emergency_config.get('circuit_breaker', {})
            breach_config = cb_config.get(breach_type, {})
            action = breach_config.get('action', None)

            # If no config for this breach_type, fall back to severity-based handling
            if action is None:
                if severity in ["high", "critical"]:
                    action = 'halt_all'
                else:
                    action = 'pause_trading'

            logger.warning(f"Circuit breaker triggered: {breach_type or severity} - {alert.get('message')}")

            if action == 'halt_all':
                self._state = CoordinatorState.HALTED
                logger.warning("Emergency: Halting all operations")

                # Check if we should close positions
                if breach_config.get('close_positions', False):
                    await self._emergency_close_positions()

            elif action == 'pause_trading':
                await self.pause()
                logger.warning("Emergency: Paused trading")

            elif action == 'reduce_positions':
                reduction_pct = breach_config.get('reduction_pct', 50)
                await self._emergency_reduce_positions(reduction_pct)
                logger.warning(f"Emergency: Reducing positions by {reduction_pct}%")

            # Notify if configured
            if breach_config.get('notify', True):
                await self._publish_emergency_notification(breach_type, action, alert)

    async def _emergency_close_positions(self) -> None:
        """Close all positions in emergency (F15)."""
        if not self.execution_manager:
            logger.error("Cannot close positions: no execution manager")
            return

        try:
            # This would need to be implemented in execution manager
            if hasattr(self.execution_manager, 'close_all_positions'):
                await self.execution_manager.close_all_positions()
                logger.warning("All positions closed due to emergency")
        except Exception as e:
            logger.error(f"Failed to close positions in emergency: {e}")

    async def _emergency_reduce_positions(self, reduction_pct: int) -> None:
        """Reduce positions by percentage in emergency (F15)."""
        if not self.execution_manager:
            logger.error("Cannot reduce positions: no execution manager")
            return

        try:
            # This would need to be implemented in execution manager
            if hasattr(self.execution_manager, 'reduce_positions'):
                await self.execution_manager.reduce_positions(reduction_pct)
                logger.warning(f"Positions reduced by {reduction_pct}% due to emergency")
        except Exception as e:
            logger.error(f"Failed to reduce positions in emergency: {e}")

    async def _publish_emergency_notification(
        self,
        breach_type: str,
        action: str,
        alert: dict,
    ) -> None:
        """Publish emergency notification (F15)."""
        await self.bus.publish(create_message(
            topic=MessageTopic.SYSTEM_EVENTS,
            source=self.agent_name,
            payload={
                "event": "emergency_action",
                "breach_type": breach_type,
                "action_taken": action,
                "alert_message": alert.get("message", ""),
                "severity": alert.get("severity", "unknown"),
            },
            priority=MessagePriority.URGENT,
        ))

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

    async def _build_consensus(self, signal: dict) -> float:
        """
        Build consensus from multiple agent signals.

        When multiple agents agree on direction, amplify confidence.
        Returns a confidence multiplier (1.0 = no change, >1.0 = amplified).

        F11: Documented consensus formula
        F16: Uses configurable thresholds from config
        """
        agreement_count = 0
        total_agents = 0
        signal_action = signal.get("action", "HOLD")

        if signal_action == "HOLD":
            return 1.0  # No consensus needed for HOLD

        # Get latest outputs from message bus
        ta_msg = await self.bus.get_latest(MessageTopic.TA_SIGNALS, max_age_seconds=120)
        regime_msg = await self.bus.get_latest(MessageTopic.REGIME_UPDATES, max_age_seconds=600)
        sentiment_msg = await self.bus.get_latest(MessageTopic.SENTIMENT_UPDATES, max_age_seconds=3600)

        # Check TA agreement
        if ta_msg:
            total_agents += 1
            ta_bias = ta_msg.payload.get("bias", "neutral")
            if (signal_action == "BUY" and ta_bias == "long") or \
               (signal_action == "SELL" and ta_bias == "short"):
                agreement_count += 1
                logger.debug(f"Consensus: TA agent agrees ({ta_bias})")

        # Check regime appropriateness
        if regime_msg:
            total_agents += 1
            regime = regime_msg.payload.get("regime", "neutral")
            # Trending regimes support trading
            if regime in ["trending_up", "trending_down", "breakout"]:
                if (signal_action == "BUY" and regime in ["trending_up", "breakout"]) or \
                   (signal_action == "SELL" and regime == "trending_down"):
                    agreement_count += 1
                    logger.debug(f"Consensus: Regime supports signal ({regime})")

        # Check sentiment agreement
        if sentiment_msg:
            total_agents += 1
            sent_bias = sentiment_msg.payload.get("sentiment_bias", "neutral")
            if (signal_action == "BUY" and sent_bias == "bullish") or \
               (signal_action == "SELL" and sent_bias == "bearish"):
                agreement_count += 1
                logger.debug(f"Consensus: Sentiment agrees ({sent_bias})")

        # Calculate consensus multiplier
        if total_agents == 0:
            return 1.0

        agreement_ratio = agreement_count / total_agents

        # F11/F16: Calculate multiplier with documented formula and configurable thresholds
        multiplier = self._calculate_consensus_multiplier(agreement_ratio)

        logger.info(
            f"Consensus: {agreement_count}/{total_agents} agents agree, "
            f"confidence multiplier: {multiplier:.2f}"
        )
        return multiplier

    def _calculate_consensus_multiplier(self, agreement_ratio: float) -> float:
        """
        Calculate confidence multiplier based on agent agreement (F11, F16).

        Rationale:
        - High agreement (default 66%+): Boost confidence up to 30%
          When most independent signals agree, we have higher conviction
        - Moderate agreement (33-66%): No adjustment
          Mixed signals warrant neither boost nor penalty
        - Low agreement (<33%): Reduce confidence up to 15%
          Conflicting signals suggest caution

        This encourages trading when multiple independent signals agree
        while being cautious when signals conflict.

        Formula:
        - High: 1.0 + (agreement_ratio - 0.5) * (max_boost * 2)
          At 66%, multiplier = 1.0 + (0.66 - 0.5) * 0.6 = 1.096
          At 100%, multiplier = 1.0 + (1.0 - 0.5) * 0.6 = 1.3
        - Low: (1.0 - max_penalty) + agreement_ratio * (max_penalty / low_threshold)
          At 0%, multiplier = 0.85 + 0 = 0.85
          At 33%, multiplier = 0.85 + 0.33 * 0.45 = ~1.0

        Args:
            agreement_ratio: Fraction of agents that agree (0.0 to 1.0)

        Returns:
            Confidence multiplier (typically 0.85 to 1.3)
        """
        # F16: Use configurable thresholds
        high_threshold = self._consensus_high_threshold  # Default: 0.66
        low_threshold = self._consensus_low_threshold    # Default: 0.33
        max_boost = self._consensus_max_boost            # Default: 0.30
        max_penalty = self._consensus_max_penalty        # Default: 0.15

        if agreement_ratio >= high_threshold:
            # High agreement: Linear interpolation from 1.0 at 50% to (1.0 + max_boost) at 100%
            multiplier = 1.0 + (agreement_ratio - 0.5) * (max_boost * 2)
        elif agreement_ratio >= low_threshold:
            # Moderate agreement: No adjustment
            multiplier = 1.0
        else:
            # Low agreement: Linear interpolation from (1.0 - max_penalty) at 0% to 1.0 at low_threshold
            multiplier = (1.0 - max_penalty) + agreement_ratio * (max_penalty / low_threshold)

        return multiplier

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
        Respects degradation level - in EMERGENCY mode, uses conservative defaults.
        """
        start_time = time.perf_counter()

        # In EMERGENCY degradation, skip LLM and use conservative defaults
        if self._degradation_level == DegradationLevel.EMERGENCY:
            logger.warning("Emergency degradation: skipping LLM conflict resolution")
            return ConflictResolution(
                action="wait",
                reasoning="Emergency degradation mode - conservative action",
                confidence=0.3,
            )

        # In LIMITED degradation, only resolve high-priority conflicts
        if self._degradation_level == DegradationLevel.LIMITED:
            # Only resolve regime conflicts, skip sentiment conflicts
            critical_conflicts = [c for c in conflicts if c.conflict_type == "regime_conflict"]
            if not critical_conflicts:
                logger.info("Limited degradation: skipping non-critical conflict resolution")
                return ConflictResolution(
                    action="proceed",
                    reasoning="Limited degradation mode - proceeding with caution",
                    confidence=0.5,
                )
            conflicts = critical_conflicts

        # Build conflict resolution prompt
        prompt = self._build_conflict_prompt(signal, conflicts)

        # Convert timeout from ms to seconds
        timeout_seconds = self._max_resolution_time_ms / 1000.0

        try:
            # Try primary model (DeepSeek V3) with timeout enforcement
            response = await asyncio.wait_for(
                self._call_llm_for_resolution(prompt),
                timeout=timeout_seconds,
            )
            resolution = self._parse_resolution(response)
            self._total_conflicts_resolved += 1
            self._record_llm_success()

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(f"Conflict resolved in {latency_ms}ms: {resolution.action}")

            return resolution

        except asyncio.TimeoutError:
            logger.warning(f"Conflict resolution timed out after {self._max_resolution_time_ms}ms")
            self._record_llm_failure()
            # Default to conservative action on timeout
            return ConflictResolution(
                action="wait",
                reasoning=f"Resolution timed out after {self._max_resolution_time_ms}ms",
                confidence=0.0,
            )
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            self._record_llm_failure()
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
        """
        Call LLM for conflict resolution with fallback (F14 fix).

        Explicitly catches exceptions from both primary and fallback LLMs
        to provide clear error messages.
        """
        # Try primary model
        try:
            response = await self.llm.generate(
                model=self._primary_model,
                system_prompt=COORDINATOR_SYSTEM_PROMPT,
                user_message=prompt,
                max_tokens=500,
            )
            return response.text
        except Exception as e:
            logger.warning(f"Primary LLM ({self._primary_model}) failed: {e}")

        # F14: Try fallback with explicit exception handling
        try:
            response = await self.llm.generate(
                model=self._fallback_model,
                system_prompt=COORDINATOR_SYSTEM_PROMPT,
                user_message=prompt,
                max_tokens=500,
            )
            return response.text
        except Exception as e:
            logger.error(f"Fallback LLM ({self._fallback_model}) also failed: {e}")
            raise RuntimeError(f"All LLM providers failed for conflict resolution") from e

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
        """
        Apply resolution modifications to signal with bounds validation.

        Validates that modifications are within reasonable bounds to prevent
        invalid trade parameters.
        """
        modified = signal.copy()

        if "leverage" in modifications:
            leverage = modifications["leverage"]
            # Bound leverage between 1 and max allowed (5x per system constraints)
            leverage = max(1, min(5, int(leverage)))
            modified["leverage"] = leverage

        if "size_reduction_pct" in modifications:
            reduction_pct = modifications["size_reduction_pct"]
            # Bound reduction between 0% and 100%
            reduction_pct = max(0, min(100, float(reduction_pct)))
            original_size = modified.get("size_usd", 0)
            reduction = reduction_pct / 100
            new_size = original_size * (1 - reduction)
            # Ensure size doesn't go negative or below minimum
            modified["size_usd"] = max(0, new_size)

        if "entry_adjustment_pct" in modifications:
            adjustment_pct = modifications["entry_adjustment_pct"]
            # Bound adjustment between -50% and +50%
            adjustment_pct = max(-50, min(50, float(adjustment_pct)))
            original_entry = modified.get("entry_price", 0)
            if original_entry > 0:
                adjustment = adjustment_pct / 100
                new_entry = original_entry * (1 + adjustment)
                # Ensure entry price stays positive
                modified["entry_price"] = max(0.0001, new_entry)

        return modified

    # -------------------------------------------------------------------------
    # Trade Routing
    # -------------------------------------------------------------------------

    async def _execute_rebalance_trades(self, output) -> None:
        """
        Execute rebalance trades from Portfolio Rebalance Agent.

        Trades are executed in order (sells first, then buys) to maintain
        proper allocation and minimize exposure during rebalancing.

        Args:
            output: PortfolioRebalanceAgent output with trades list
        """
        if self._state != CoordinatorState.RUNNING:
            logger.debug("Skipping rebalance - coordinator not running")
            return

        if not self.risk_engine or not self.execution_manager:
            logger.warning("Risk engine or execution manager not configured for rebalance")
            return

        # Access trades directly from RebalanceOutput dataclass
        trades = output.trades if hasattr(output, 'trades') else []
        if not trades:
            logger.debug("No rebalance trades to execute")
            return

        # Get DCA info
        dca_batches = getattr(output, 'dca_batches', 1)
        dca_interval_hours = getattr(output, 'dca_interval_hours', 0)

        # Separate immediate trades (batch 0) from scheduled trades
        immediate_trades = [t for t in trades if getattr(t, 'batch_index', 0) == 0]
        scheduled_trades = [t for t in trades if getattr(t, 'batch_index', 0) > 0]

        if dca_batches > 1:
            logger.info(
                f"DCA rebalance: {len(immediate_trades)} immediate trades, "
                f"{len(scheduled_trades)} scheduled across {dca_batches - 1} future batches"
            )

            # Store scheduled trades for later execution
            if scheduled_trades:
                await self._store_scheduled_trades(scheduled_trades)
        else:
            logger.info(f"Executing {len(immediate_trades)} rebalance trades immediately")

        from ..risk.rules_engine import TradeProposal

        for trade in immediate_trades:
            try:
                # Create trade proposal from RebalanceTrade object
                proposal = TradeProposal(
                    symbol=trade.symbol,
                    side=trade.action,  # "buy" or "sell"
                    size_usd=abs(float(trade.amount_usd)),
                    entry_price=0,  # Market order for rebalancing
                    leverage=1,  # No leverage for rebalancing
                    confidence=0.8,  # Rebalance trades have high confidence
                    regime="rebalance",
                )

                # Validate with risk engine
                validation = self.risk_engine.validate_trade(proposal)

                if not validation.is_approved():
                    logger.warning(
                        f"Rebalance trade rejected by risk: {trade.symbol} - {validation.rejections}"
                    )
                    continue

                # Execute trade
                final_proposal = validation.modified_proposal or proposal
                result = await self.execution_manager.execute_trade(final_proposal)

                if result.success:
                    logger.info(f"Rebalance trade executed: {trade.symbol} {trade.action}")
                    await self.bus.publish(create_message(
                        topic=MessageTopic.EXECUTION_EVENTS,
                        source=self.agent_name,
                        payload={
                            "event_type": "rebalance_trade_executed",
                            "order_id": result.order.id if result.order else None,
                            "symbol": trade.symbol,
                            "side": trade.action,
                            "batch_index": getattr(trade, 'batch_index', 0),
                        },
                    ))
                else:
                    logger.error(f"Rebalance trade failed: {trade.symbol} - {result.error_message}")

            except Exception as e:
                logger.error(f"Rebalance trade error for {trade.symbol}: {e}")

    async def _store_scheduled_trades(self, trades: list) -> None:
        """
        Store scheduled DCA trades for later execution (F07 fix).

        Thread-safe: uses lock when modifying in-memory list.
        """
        if not self.db:
            logger.warning("No database configured - scheduled trades will not persist")
            # F07: Store in memory with lock for thread safety
            async with self._scheduled_trades_lock:
                self._scheduled_trades.extend(trades)
            return

        try:
            for trade in trades:
                query = """
                    INSERT INTO scheduled_trades (
                        symbol, side, amount_usd, batch_index, scheduled_time, status
                    ) VALUES ($1, $2, $3, $4, $5, 'pending')
                """
                await self.db.execute(
                    query,
                    trade.symbol,
                    trade.action,
                    float(trade.amount_usd),
                    trade.batch_index,
                    trade.scheduled_time,
                )
            logger.info(f"Stored {len(trades)} scheduled DCA trades")
        except Exception as e:
            logger.error(f"Failed to store scheduled trades: {e}")

    async def _execute_scheduled_trades(self) -> None:
        """
        Execute any scheduled trades that are due (F07 fix).

        Thread-safe: uses lock when accessing in-memory list.
        """
        now = datetime.now(timezone.utc)

        # F07: Check in-memory scheduled trades with lock
        due_trades = []
        async with self._scheduled_trades_lock:
            if self._scheduled_trades:
                due_trades = [t for t in self._scheduled_trades if t.scheduled_time <= now]
                # Remove due trades from list while holding lock
                for trade in due_trades:
                    self._scheduled_trades.remove(trade)

        # Execute outside lock to avoid blocking
        if due_trades:
            logger.info(f"Executing {len(due_trades)} scheduled DCA trades from memory")
            for trade in due_trades:
                await self._execute_single_rebalance_trade(trade)

        # Check database scheduled trades
        if self.db:
            try:
                query = """
                    SELECT id, symbol, side, amount_usd, batch_index, scheduled_time
                    FROM scheduled_trades
                    WHERE status = 'pending' AND scheduled_time <= $1
                    ORDER BY scheduled_time, batch_index
                """
                rows = await self.db.fetch(query, now)

                for row in rows:
                    from ..agents.portfolio_rebalance import RebalanceTrade
                    trade = RebalanceTrade(
                        symbol=row['symbol'],
                        action=row['side'],
                        amount_usd=Decimal(str(row['amount_usd'])),
                        batch_index=row['batch_index'],
                        scheduled_time=row['scheduled_time'],
                    )
                    success = await self._execute_single_rebalance_trade(trade)

                    # Update status
                    update_query = """
                        UPDATE scheduled_trades SET status = $1, executed_at = $2 WHERE id = $3
                    """
                    status = 'executed' if success else 'failed'
                    await self.db.execute(update_query, status, now, row['id'])

            except Exception as e:
                logger.error(f"Failed to execute scheduled trades: {e}")

    async def _execute_single_rebalance_trade(self, trade) -> bool:
        """Execute a single rebalance trade."""
        if not self.risk_engine or not self.execution_manager:
            return False

        try:
            from ..risk.rules_engine import TradeProposal

            proposal = TradeProposal(
                symbol=trade.symbol,
                side=trade.action,
                size_usd=abs(float(trade.amount_usd)),
                entry_price=0,
                leverage=1,
                confidence=0.8,
                regime="rebalance",
            )

            validation = self.risk_engine.validate_trade(proposal)
            if not validation.is_approved():
                logger.warning(f"Scheduled trade rejected: {trade.symbol}")
                return False

            final_proposal = validation.modified_proposal or proposal
            result = await self.execution_manager.execute_trade(final_proposal)

            if result.success:
                logger.info(f"Scheduled trade executed: {trade.symbol} batch {trade.batch_index}")
                return True
            else:
                logger.error(f"Scheduled trade failed: {trade.symbol}")
                return False

        except Exception as e:
            logger.error(f"Scheduled trade error: {e}")
            return False

    async def _route_to_execution(self, signal: dict) -> None:
        """
        Route validated signal to execution manager.

        Phase 6: Routes to paper executor or live executor based on trading mode.
        """
        from ..execution.trading_mode import TradingMode

        # Check for required components based on trading mode
        if self.trading_mode == TradingMode.PAPER:
            if not self.risk_engine or not self.paper_executor:
                logger.warning("Risk engine or paper executor not configured")
                return
        else:
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
                    "trading_mode": self.trading_mode.value,
                },
            ))
            return

        # Use modified proposal if applicable
        final_proposal = validation.modified_proposal or proposal

        # Execute trade using appropriate executor
        try:
            # Phase 6: Route based on trading mode
            if self.trading_mode == TradingMode.PAPER:
                result = await self.paper_executor.execute_trade(final_proposal)
                execution_mode = "paper"
            else:
                result = await self.execution_manager.execute_trade(final_proposal)
                execution_mode = "live"

            self._total_trades_routed += 1

            if result.success:
                await self.bus.publish(create_message(
                    topic=MessageTopic.EXECUTION_EVENTS,
                    source=self.agent_name,
                    payload={
                        "event_type": "order_placed",
                        "order_id": result.order.id if result.order else None,
                        "position_id": result.position_id,
                        "symbol": final_proposal.symbol,
                        "trading_mode": execution_mode,
                    },
                ))

                # Log paper trading portfolio update
                if self.trading_mode == TradingMode.PAPER and self.paper_portfolio:
                    logger.info(
                        f"📝 Paper portfolio after trade: {self.paper_portfolio.get_balances_dict()}"
                    )
            else:
                await self.bus.publish(create_message(
                    topic=MessageTopic.EXECUTION_EVENTS,
                    source=self.agent_name,
                    payload={
                        "event_type": "order_error",
                        "symbol": final_proposal.symbol,
                        "error_message": result.error_message,
                        "trading_mode": execution_mode,
                    },
                    priority=MessagePriority.HIGH,
                ))

        except Exception as e:
            logger.error(f"Execution failed: {e}")

    # -------------------------------------------------------------------------
    # State Persistence
    # -------------------------------------------------------------------------

    async def persist_state(self) -> bool:
        """
        Persist coordinator state to database (F06 fix).

        Saves task schedules, statistics, enabled/disabled state,
        AND in-flight tasks to survive restarts.

        Returns:
            True if persisted successfully
        """
        if not self.db:
            return False

        try:
            # F06: Track in-flight tasks
            in_flight_tasks = [
                {
                    "name": t.name,
                    "started_at": t._started_at.isoformat() if t._started_at else None,
                }
                for t in self._scheduled_tasks
                if t._running
            ]

            state_data = {
                "state": self._state.value,
                "statistics": {
                    "total_task_runs": self._total_task_runs,
                    "total_conflicts_detected": self._total_conflicts_detected,
                    "total_conflicts_resolved": self._total_conflicts_resolved,
                    "total_trades_routed": self._total_trades_routed,
                },
                "tasks": {
                    t.name: {
                        "enabled": t.enabled,
                        "last_run": t.last_run.isoformat() if t.last_run else None,
                    }
                    for t in self._scheduled_tasks
                },
                "in_flight_tasks": in_flight_tasks,  # F06
            }

            query = """
                INSERT INTO coordinator_state (id, state_data, updated_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (id) DO UPDATE SET
                    state_data = $2,
                    updated_at = $3
            """
            await self.db.execute(
                query,
                "coordinator",  # Single coordinator instance
                json.dumps(state_data),
                datetime.now(timezone.utc),
            )
            logger.debug("Coordinator state persisted")
            return True

        except Exception as e:
            logger.error(f"Failed to persist coordinator state: {e}")
            return False

    async def load_state(self) -> bool:
        """
        Load coordinator state from database (F06 fix).

        Restores task schedules, statistics, enabled/disabled state,
        AND handles in-flight tasks from last session.

        Returns:
            True if state loaded successfully
        """
        if not self.db:
            return False

        try:
            query = """
                SELECT state_data FROM coordinator_state WHERE id = $1
            """
            row = await self.db.fetchrow(query, "coordinator")

            if not row:
                logger.info("No persisted coordinator state found")
                return False

            state_data = json.loads(row['state_data'])

            # Restore statistics
            stats = state_data.get("statistics", {})
            self._total_task_runs = stats.get("total_task_runs", 0)
            self._total_conflicts_detected = stats.get("total_conflicts_detected", 0)
            self._total_conflicts_resolved = stats.get("total_conflicts_resolved", 0)
            self._total_trades_routed = stats.get("total_trades_routed", 0)

            # Restore task state
            tasks = state_data.get("tasks", {})
            for task in self._scheduled_tasks:
                if task.name in tasks:
                    task_state = tasks[task.name]
                    task.enabled = task_state.get("enabled", True)
                    last_run = task_state.get("last_run")
                    if last_run:
                        task.last_run = datetime.fromisoformat(last_run)

            # F06: Handle in-flight tasks from previous session
            in_flight = state_data.get("in_flight_tasks", [])
            if in_flight:
                task_names = [t['name'] for t in in_flight]
                logger.warning(
                    f"Detected {len(in_flight)} in-flight tasks from previous session: "
                    f"{task_names}"
                )
                # Mark these tasks for immediate re-run by setting last_run to None
                # This ensures they run on next schedule check
                for in_flight_task in in_flight:
                    task_name = in_flight_task.get('name')
                    if task_name in self._task_map:
                        task = self._task_map[task_name]
                        # Option: Re-run immediately by marking run_on_start
                        task.run_on_start = True
                        logger.info(f"Task {task_name} marked for immediate re-run")

                # Publish alert about in-flight tasks
                await self.bus.publish(create_message(
                    topic=MessageTopic.SYSTEM_EVENTS,
                    source=self.agent_name,
                    payload={
                        "event": "in_flight_tasks_recovered",
                        "tasks": task_names,
                        "action": "marked_for_rerun",
                    },
                    priority=MessagePriority.HIGH,
                ))

            logger.info("Coordinator state loaded from database")
            return True

        except Exception as e:
            logger.error(f"Failed to load coordinator state: {e}")
            return False

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get coordinator status including trading mode (Phase 6)."""
        status = {
            "state": self._state.value,
            "degradation_level": self._degradation_level.name,
            "trading_mode": self.trading_mode.value if self.trading_mode else "unknown",
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
            "health": {
                "consecutive_llm_failures": self._consecutive_llm_failures,
                "consecutive_api_failures": self._consecutive_api_failures,
            },
        }

        # Phase 6: Add paper trading status if in paper mode
        from ..execution.trading_mode import TradingMode
        if self.trading_mode == TradingMode.PAPER:
            paper_status = {}
            if self.paper_portfolio:
                paper_status["portfolio"] = {
                    "session_id": self.paper_portfolio.session_id,
                    "balances": self.paper_portfolio.get_balances_dict(),
                    "trade_count": self.paper_portfolio.trade_count,
                    "total_fees_paid": float(self.paper_portfolio.total_fees_paid),
                }
            if self.paper_executor:
                paper_status["executor"] = self.paper_executor.get_stats()
            if self.paper_price_source:
                paper_status["price_source"] = self.paper_price_source.get_stats()
            status["paper_trading"] = paper_status

        return status

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
