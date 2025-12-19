"""
Trading Decision Agent - 6-Model A/B Testing for trading decisions.

Runs all 6 models in parallel and calculates consensus:
- Qwen 2.5 7B (Local baseline)
- GPT-4-turbo (OpenAI)
- Grok-2 (xAI)
- DeepSeek V3 (DeepSeek)
- Claude Sonnet (Anthropic)
- Claude Opus (Anthropic)

Invoked: Hourly or on significant market events.
All model responses are stored for performance tracking.
"""

import asyncio
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any, TYPE_CHECKING

from .base_agent import BaseAgent, AgentOutput

if TYPE_CHECKING:
    from ..data.market_snapshot import MarketSnapshot
    from ..llm.prompt_builder import PromptBuilder, PortfolioContext
    from ..llm.clients.base import BaseLLMClient
    from .technical_analysis import TAOutput
    from .regime_detection import RegimeOutput

logger = logging.getLogger(__name__)


# Valid trading actions
VALID_ACTIONS = ["BUY", "SELL", "HOLD", "CLOSE_LONG", "CLOSE_SHORT"]


@dataclass
class ModelDecision:
    """Individual model's trading decision."""
    model_name: str
    provider: str
    action: str  # BUY, SELL, HOLD, CLOSE_LONG, CLOSE_SHORT
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: Optional[float] = None
    reasoning: str = ""
    latency_ms: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if decision is valid."""
        return self.error is None and self.action in VALID_ACTIONS


@dataclass
class ConsensusResult:
    """Aggregated consensus from all models."""
    # Consensus decision
    final_action: str
    final_confidence: float
    consensus_strength: float  # 0-1, how strongly models agree

    # Vote breakdown
    votes: dict  # action -> count
    total_models: int
    agreeing_models: int

    # Consensus agreement type and confidence boost
    agreement_type: str = "split"  # unanimous, strong_majority, majority, split
    confidence_boost: float = 0.0  # Boost to apply based on agreement

    # Averaged parameters
    avg_entry_price: Optional[float] = None
    avg_stop_loss: Optional[float] = None
    avg_take_profit: Optional[float] = None
    avg_position_size_pct: Optional[float] = None

    # Individual decisions
    model_decisions: list[ModelDecision] = field(default_factory=list)

    # Performance metrics
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_latency_ms: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'final_action': self.final_action,
            'final_confidence': self.final_confidence,
            'consensus_strength': self.consensus_strength,
            'votes': self.votes,
            'total_models': self.total_models,
            'agreeing_models': self.agreeing_models,
            'avg_entry_price': self.avg_entry_price,
            'avg_stop_loss': self.avg_stop_loss,
            'avg_take_profit': self.avg_take_profit,
            'avg_position_size_pct': self.avg_position_size_pct,
            'model_decisions': [
                {
                    'model': d.model_name,
                    'provider': d.provider,
                    'action': d.action,
                    'confidence': d.confidence,
                    'reasoning': d.reasoning[:100],
                    'error': d.error,
                }
                for d in self.model_decisions
            ],
            'total_cost_usd': self.total_cost_usd,
            'total_tokens': self.total_tokens,
        }


@dataclass
class TradingDecisionOutput(AgentOutput):
    """Trading Decision Agent output."""
    # Consensus decision
    action: str = "HOLD"
    consensus_strength: float = 0.0

    # Trade parameters
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: Optional[float] = None
    leverage: int = 1

    # Context
    regime: str = "ranging"
    ta_bias: str = "neutral"

    # Model agreement details
    votes: dict = field(default_factory=dict)
    agreeing_models: int = 0
    total_models: int = 0

    # All model decisions (for tracking)
    model_decisions: list[ModelDecision] = field(default_factory=list)

    # Cost tracking
    total_cost_usd: float = 0.0

    def validate(self) -> tuple[bool, list[str]]:
        """Validate trading decision output."""
        is_valid, errors = super().validate()

        if self.action not in VALID_ACTIONS:
            errors.append(f"Invalid action: {self.action}")
            is_valid = False

        if not 0 <= self.consensus_strength <= 1:
            errors.append(f"Consensus strength {self.consensus_strength} not in [0, 1]")
            is_valid = False

        # If action is BUY/SELL, need entry price and stop loss
        if self.action in ["BUY", "SELL"]:
            if self.entry_price is None:
                errors.append("Entry price required for BUY/SELL")
                is_valid = False
            if self.stop_loss is None:
                errors.append("Stop loss required for BUY/SELL")
                is_valid = False

        return is_valid, errors


class TradingDecisionAgent(BaseAgent):
    """
    Trading Decision Agent with 6-model A/B testing.

    Queries all 6 models in parallel and calculates consensus.
    All decisions are stored for performance tracking and model comparison.
    """

    agent_name = "trading_decision"
    llm_tier = "multi_model"
    model = "ensemble"

    def __init__(
        self,
        llm_clients: dict[str, 'BaseLLMClient'],
        prompt_builder: 'PromptBuilder',
        config: dict,
        db_pool=None
    ):
        """
        Initialize TradingDecisionAgent.

        Args:
            llm_clients: Dict of provider_name -> LLM client
                Expected keys: ollama, openai, anthropic, deepseek, xai
            prompt_builder: Prompt builder
            config: Agent configuration
            db_pool: Database pool for output storage
        """
        # Call base class init with first available client (for stats/cache)
        first_client = next(iter(llm_clients.values()), None)
        super().__init__(
            llm_client=first_client,
            prompt_builder=prompt_builder,
            config=config,
            db_pool=db_pool
        )

        # Store all clients for multi-model querying
        self.llm_clients = llm_clients

        # Model configuration
        self.models = config.get('models', {
            'qwen': {'provider': 'ollama', 'model': 'qwen2.5:7b'},
            'gpt4': {'provider': 'openai', 'model': 'gpt-4-turbo'},
            'grok': {'provider': 'xai', 'model': 'grok-2-1212'},
            'deepseek': {'provider': 'deepseek', 'model': 'deepseek-chat'},
            'sonnet': {'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20241022'},
            'opus': {'provider': 'anthropic', 'model': 'claude-3-opus-20240229'},
        })

        # Consensus thresholds
        self.min_consensus = config.get('min_consensus', 0.5)  # 50% agreement
        self.high_consensus = config.get('high_consensus', 0.67)  # 2/3 agreement

        # Confidence boost per design spec
        # Unanimous (6/6): +0.15, Strong (5/6): +0.10, Majority (4/6): +0.05
        self.confidence_boosts = config.get('confidence_boosts', {
            'unanimous': 0.15,      # 6/6 or 100%
            'strong_majority': 0.10, # 5/6 or 83%+
            'majority': 0.05,        # 4/6 or 67%+
            'split': 0.0,            # <4/6 or <67%
        })

        # Timeout for parallel calls (per model)
        self.timeout_seconds = config.get('timeout_seconds', 30)

        # Cost tracking
        self._total_cost = 0.0

    async def process(
        self,
        snapshot: 'MarketSnapshot',
        portfolio_context: Optional['PortfolioContext'] = None,
        ta_output: Optional['TAOutput'] = None,
        regime_output: Optional['RegimeOutput'] = None,
        **kwargs
    ) -> TradingDecisionOutput:
        """
        Get trading decision from all models and calculate consensus.

        Args:
            snapshot: Market data snapshot
            portfolio_context: Portfolio state
            ta_output: TA agent output
            regime_output: Regime detection output

        Returns:
            TradingDecisionOutput with consensus decision
        """
        logger.info(f"Trading Decision Agent processing {snapshot.symbol}")
        start_time = datetime.now(timezone.utc)

        # Build context from other agents
        additional_context = {}
        if ta_output:
            additional_context['technical_analysis'] = {
                'trend_direction': ta_output.trend_direction,
                'trend_strength': ta_output.trend_strength,
                'momentum_score': ta_output.momentum_score,
                'bias': ta_output.bias,
                'confidence': ta_output.confidence,
                'primary_signal': ta_output.primary_signal,
                'warnings': ta_output.warnings,
            }
        if regime_output:
            additional_context['regime'] = {
                'regime': regime_output.regime,
                'volatility': regime_output.volatility,
                'trend_strength': regime_output.trend_strength,
                'position_size_multiplier': regime_output.position_size_multiplier,
                'entry_strictness': regime_output.entry_strictness,
            }

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            agent_name=self.agent_name,
            snapshot=snapshot,
            portfolio_context=portfolio_context,
            additional_context=additional_context if additional_context else None,
            query=self._get_decision_query(),
        )

        # Query all models in parallel
        decisions = await self._query_all_models(
            prompt.system_prompt,
            prompt.user_message,
            snapshot,
        )

        # Calculate consensus
        consensus = self._calculate_consensus(decisions)

        # Apply confidence boost based on agreement level
        boosted_confidence = min(1.0, consensus.final_confidence + consensus.confidence_boost)

        logger.debug(
            f"Confidence: base={consensus.final_confidence:.2f}, "
            f"boost={consensus.confidence_boost:.2f} ({consensus.agreement_type}), "
            f"final={boosted_confidence:.2f}"
        )

        # Create output
        output = TradingDecisionOutput(
            agent_name=self.agent_name,
            timestamp=datetime.now(timezone.utc),
            symbol=snapshot.symbol,
            confidence=boosted_confidence,  # Apply the confidence boost
            reasoning=self._build_consensus_reasoning(consensus),
            latency_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
            tokens_used=consensus.total_tokens,
            model_used="ensemble",
            action=consensus.final_action,
            consensus_strength=consensus.consensus_strength,
            entry_price=consensus.avg_entry_price,
            stop_loss=consensus.avg_stop_loss,
            take_profit=consensus.avg_take_profit,
            position_size_pct=consensus.avg_position_size_pct,
            regime=regime_output.regime if regime_output else "ranging",
            ta_bias=ta_output.bias if ta_output else "neutral",
            votes=consensus.votes,
            agreeing_models=consensus.agreeing_models,
            total_models=consensus.total_models,
            model_decisions=consensus.model_decisions,
            total_cost_usd=consensus.total_cost_usd,
        )

        # Validate output
        is_valid, validation_errors = output.validate()
        if not is_valid:
            logger.warning(f"Trading decision validation issues: {validation_errors}")

        # Store output
        await self.store_output(output)

        # Store individual model comparisons
        await self._store_model_comparisons(output, snapshot)

        # Update stats
        self._total_invocations += 1
        self._total_cost += consensus.total_cost_usd
        self._last_output = output

        logger.info(
            f"Trading Decision: {snapshot.symbol} action={output.action} "
            f"consensus={output.consensus_strength:.2f} cost=${output.total_cost_usd:.4f}"
        )

        return output

    async def _query_all_models(
        self,
        system_prompt: str,
        user_message: str,
        snapshot: 'MarketSnapshot',
    ) -> list[ModelDecision]:
        """
        Query all models in parallel with improved timeout handling.

        Uses asyncio.wait instead of gather to preserve partial results
        when some models time out.
        """
        tasks = {}

        for name, model_config in self.models.items():
            provider = model_config['provider']
            model = model_config['model']

            if provider in self.llm_clients:
                client = self.llm_clients[provider]
                task = asyncio.create_task(
                    self._query_single_model(
                        name, provider, model, client,
                        system_prompt, user_message, snapshot
                    )
                )
                tasks[task] = name  # Track which model each task belongs to
            else:
                logger.warning(f"No client for provider: {provider}")

        if not tasks:
            return []

        # Use asyncio.wait to preserve partial results on timeout
        done, pending = await asyncio.wait(
            tasks.keys(),
            timeout=self.timeout_seconds,
            return_when=asyncio.ALL_COMPLETED
        )

        # Cancel pending tasks and log which models timed out
        for task in pending:
            model_name = tasks[task]
            logger.warning(f"Model {model_name} timed out after {self.timeout_seconds}s")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Process completed results
        decisions = []
        for task in done:
            model_name = tasks[task]
            try:
                result = task.result()
                if isinstance(result, ModelDecision):
                    decisions.append(result)
            except Exception as e:
                logger.error(f"Model {model_name} failed: {e}")
                # Create error decision so we track the failure
                decisions.append(ModelDecision(
                    model_name=model_name,
                    provider=self.models[model_name]['provider'],
                    action='HOLD',
                    confidence=0.0,
                    reasoning=f"Error: {str(e)}",
                    error=str(e),
                ))

        logger.info(
            f"Model query complete: {len(done)} done, {len(pending)} timed out"
        )

        return decisions

    async def _query_single_model(
        self,
        name: str,
        provider: str,
        model: str,
        client: 'BaseLLMClient',
        system_prompt: str,
        user_message: str,
        snapshot: 'MarketSnapshot',
    ) -> ModelDecision:
        """Query a single model."""
        try:
            response = await client.generate(
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
            )

            # Parse response
            parsed = self._parse_decision(response.text, name)

            return ModelDecision(
                model_name=name,
                provider=provider,
                action=parsed.get('action', 'HOLD'),
                confidence=parsed.get('confidence', 0.5),
                entry_price=parsed.get('entry_price'),
                stop_loss=parsed.get('stop_loss'),
                take_profit=parsed.get('take_profit'),
                position_size_pct=parsed.get('position_size_pct'),
                reasoning=parsed.get('reasoning', ''),
                latency_ms=response.latency_ms,
                tokens_used=response.tokens_used,
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error(f"Model {name} failed: {e}")
            return ModelDecision(
                model_name=name,
                provider=provider,
                action='HOLD',
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                error=str(e),
            )

    def _parse_decision(self, response_text: str, model_name: str) -> dict:
        """Parse a model's response into structured decision."""
        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return self._normalize_decision(parsed)
            except json.JSONDecodeError:
                pass

        # Fallback: try full response
        try:
            parsed = json.loads(response_text.strip())
            return self._normalize_decision(parsed)
        except json.JSONDecodeError:
            pass

        # Last resort: extract action from text
        return self._extract_decision_from_text(response_text)

    def _normalize_decision(self, parsed: dict) -> dict:
        """Normalize parsed decision."""
        # Normalize action
        action = str(parsed.get('action', 'HOLD')).upper()
        if action not in VALID_ACTIONS:
            action = 'HOLD'
        parsed['action'] = action

        # Clamp confidence
        parsed['confidence'] = max(0.0, min(1.0, float(parsed.get('confidence', 0.5))))

        # Validate prices
        for key in ['entry_price', 'stop_loss', 'take_profit']:
            if key in parsed:
                try:
                    parsed[key] = float(parsed[key])
                except (ValueError, TypeError):
                    parsed[key] = None

        # Position size
        if 'position_size_pct' in parsed:
            try:
                parsed['position_size_pct'] = max(0, min(100, float(parsed['position_size_pct'])))
            except (ValueError, TypeError):
                parsed['position_size_pct'] = None

        # Truncate reasoning
        if 'reasoning' in parsed and len(str(parsed['reasoning'])) > 300:
            parsed['reasoning'] = str(parsed['reasoning'])[:297] + "..."

        return parsed

    def _extract_decision_from_text(self, text: str) -> dict:
        """Extract decision from unstructured text."""
        text_upper = text.upper()

        # Look for action keywords
        if 'BUY' in text_upper and 'SELL' not in text_upper:
            action = 'BUY'
        elif 'SELL' in text_upper and 'BUY' not in text_upper:
            action = 'SELL'
        elif 'CLOSE' in text_upper:
            action = 'CLOSE_LONG'  # Default to close long
        else:
            action = 'HOLD'

        return {
            'action': action,
            'confidence': 0.4,  # Lower confidence for text extraction
            'reasoning': 'Extracted from unstructured response',
        }

    def _calculate_consensus(self, decisions: list[ModelDecision]) -> ConsensusResult:
        """Calculate consensus from all model decisions with confidence boost."""
        valid_decisions = [d for d in decisions if d.is_valid()]

        if not valid_decisions:
            return ConsensusResult(
                final_action='HOLD',
                final_confidence=0.0,
                consensus_strength=0.0,
                votes={},
                total_models=len(decisions),
                agreeing_models=0,
                agreement_type='split',
                confidence_boost=0.0,
                model_decisions=decisions,
            )

        # Count votes
        votes = {}
        for d in valid_decisions:
            votes[d.action] = votes.get(d.action, 0) + 1

        # Find winning action
        max_votes = max(votes.values())
        winning_action = max(votes.keys(), key=lambda a: (votes[a], -list(votes.keys()).index(a)))

        # Calculate consensus strength
        total_valid = len(valid_decisions)
        consensus_strength = max_votes / total_valid if total_valid > 0 else 0

        # Determine agreement type and confidence boost
        # Per design: Unanimous (6/6): +0.15, Strong (5/6): +0.10, Majority (4/6): +0.05
        if consensus_strength >= 1.0:  # 100% (6/6)
            agreement_type = 'unanimous'
            boost = self.confidence_boosts.get('unanimous', 0.15)
        elif consensus_strength >= 0.83:  # 83%+ (5/6)
            agreement_type = 'strong_majority'
            boost = self.confidence_boosts.get('strong_majority', 0.10)
        elif consensus_strength >= 0.67:  # 67%+ (4/6)
            agreement_type = 'majority'
            boost = self.confidence_boosts.get('majority', 0.05)
        else:  # <67% (<4/6) - split decision, no boost
            agreement_type = 'split'
            boost = self.confidence_boosts.get('split', 0.0)
            # Still use the winning action (most votes), just with no confidence boost
            # The Risk Engine will handle low-confidence signals appropriately

        # Calculate average confidence (weighted by model agreement)
        agreeing_decisions = [d for d in valid_decisions if d.action == winning_action]
        if agreeing_decisions:
            avg_confidence = statistics.mean(d.confidence for d in agreeing_decisions)
        else:
            avg_confidence = 0.0

        # Average trade parameters from agreeing models
        avg_entry = None
        avg_stop = None
        avg_tp = None
        avg_size = None

        if winning_action in ['BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT']:
            entries = [d.entry_price for d in agreeing_decisions if d.entry_price]
            stops = [d.stop_loss for d in agreeing_decisions if d.stop_loss]
            tps = [d.take_profit for d in agreeing_decisions if d.take_profit]
            sizes = [d.position_size_pct for d in agreeing_decisions if d.position_size_pct]

            if entries:
                avg_entry = statistics.mean(entries)
            if stops:
                avg_stop = statistics.mean(stops)
            if tps:
                avg_tp = statistics.mean(tps)
            if sizes:
                avg_size = statistics.mean(sizes)

        # Total costs and tokens
        total_cost = sum(d.cost_usd for d in decisions)
        total_tokens = sum(d.tokens_used for d in decisions)
        total_latency = max((d.latency_ms for d in decisions), default=0)  # Parallel, so max

        return ConsensusResult(
            final_action=winning_action,
            final_confidence=avg_confidence,
            consensus_strength=consensus_strength,
            votes=votes,
            total_models=len(decisions),
            agreeing_models=max_votes,
            agreement_type=agreement_type,
            confidence_boost=boost,
            avg_entry_price=avg_entry,
            avg_stop_loss=avg_stop,
            avg_take_profit=avg_tp,
            avg_position_size_pct=avg_size,
            model_decisions=decisions,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
        )

    def _build_consensus_reasoning(self, consensus: ConsensusResult) -> str:
        """Build human-readable reasoning from consensus."""
        parts = []

        parts.append(
            f"Consensus: {consensus.final_action} "
            f"({consensus.agreeing_models}/{consensus.total_models} models agree, "
            f"{consensus.consensus_strength:.0%} strength)"
        )

        # Vote breakdown
        vote_str = ", ".join(f"{a}: {c}" for a, c in sorted(consensus.votes.items()))
        parts.append(f"Votes: {vote_str}")

        # Key model reasonings
        for d in consensus.model_decisions[:3]:  # Top 3
            if d.reasoning and len(d.reasoning) > 10:
                parts.append(f"{d.model_name}: {d.reasoning[:50]}...")

        return " | ".join(parts)

    async def _store_model_comparisons(
        self,
        output: TradingDecisionOutput,
        snapshot: 'MarketSnapshot',
    ) -> None:
        """Store individual model decisions for A/B tracking."""
        if self.db is None:
            return

        # Get current price for outcome tracking
        current_price = float(snapshot.current_price)

        try:
            for decision in output.model_decisions:
                query = """
                    INSERT INTO model_comparisons (
                        timestamp, symbol, model_name, provider,
                        action, confidence, entry_price, stop_loss, take_profit,
                        reasoning, latency_ms, tokens_used, cost_usd,
                        consensus_action, was_consensus, price_at_decision
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                    )
                """
                await self.db.execute(
                    query,
                    output.timestamp,
                    output.symbol,
                    decision.model_name,
                    decision.provider,
                    decision.action,
                    decision.confidence,
                    decision.entry_price,
                    decision.stop_loss,
                    decision.take_profit,
                    decision.reasoning,
                    decision.latency_ms,
                    decision.tokens_used,
                    decision.cost_usd,
                    output.action,
                    decision.action == output.action,
                    current_price,
                )
        except Exception as e:
            logger.error(f"Failed to store model comparisons: {e}")

    async def update_comparison_outcomes(
        self,
        symbol: str,
        timestamp_from: datetime,
        timestamp_to: datetime,
        price_1h: float,
        price_4h: float,
        price_24h: float,
    ) -> int:
        """
        Update model comparison outcomes after price data is available.

        Called by the orchestrator/scheduler after 1h, 4h, and 24h to populate
        outcome tracking data for A/B analysis.

        Args:
            symbol: Trading symbol
            timestamp_from: Start of time window
            timestamp_to: End of time window
            price_1h: Price 1 hour after decisions
            price_4h: Price 4 hours after decisions
            price_24h: Price 24 hours after decisions

        Returns:
            Number of records updated
        """
        if self.db is None:
            logger.warning("No database configured, cannot update outcomes")
            return 0

        try:
            # Get comparisons that need outcome updates
            select_query = """
                SELECT id, action, price_at_decision
                FROM model_comparisons
                WHERE symbol = $1
                AND timestamp BETWEEN $2 AND $3
                AND was_correct IS NULL
                AND price_at_decision IS NOT NULL
            """
            rows = await self.db.fetch(select_query, symbol, timestamp_from, timestamp_to)

            if not rows:
                return 0

            updated = 0
            for row in rows:
                comparison_id = row['id']
                action = row['action']
                price_at_decision = float(row['price_at_decision'])

                # Calculate if decision was correct and P&L
                if action == 'BUY':
                    was_correct = price_4h > price_at_decision
                    pnl_pct = ((price_4h - price_at_decision) / price_at_decision) * 100
                elif action == 'SELL':
                    was_correct = price_4h < price_at_decision
                    pnl_pct = ((price_at_decision - price_4h) / price_at_decision) * 100
                else:
                    # HOLD/CLOSE - correct if price stayed within 2%
                    price_change_pct = abs((price_4h - price_at_decision) / price_at_decision)
                    was_correct = price_change_pct < 0.02
                    pnl_pct = 0.0

                # Update the record
                update_query = """
                    UPDATE model_comparisons
                    SET price_after_1h = $1,
                        price_after_4h = $2,
                        price_after_24h = $3,
                        was_correct = $4,
                        outcome_pnl_pct = $5,
                        updated_at = NOW()
                    WHERE id = $6
                """
                await self.db.execute(
                    update_query,
                    price_1h, price_4h, price_24h,
                    was_correct, pnl_pct, comparison_id
                )
                updated += 1

            logger.info(f"Updated {updated} model comparison outcomes for {symbol}")
            return updated

        except Exception as e:
            logger.error(f"Failed to update comparison outcomes: {e}")
            return 0

    def _get_decision_query(self) -> str:
        """Get the trading decision query for LLMs."""
        return """Based on all provided data (market snapshot, technical analysis, regime detection, portfolio state), make a trading decision.

Respond with a JSON object:

{
  "action": "BUY|SELL|HOLD|CLOSE_LONG|CLOSE_SHORT",
  "confidence": 0.0-1.0,
  "entry_price": price_for_entry_or_null,
  "stop_loss": stop_loss_price_or_null,
  "take_profit": take_profit_price_or_null,
  "position_size_pct": 1-100_percent_of_available,
  "reasoning": "Brief explanation (max 200 chars)"
}

IMPORTANT RULES:
1. Only BUY/SELL if confidence >= 0.65
2. Always specify stop_loss for BUY/SELL (required)
3. Risk/reward ratio should be >= 1.5
4. Consider current regime when sizing
5. HOLD is valid if signals are mixed

Return ONLY the JSON object."""

    def get_output_schema(self) -> dict:
        """Return JSON schema for trading decision."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["action", "confidence"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": VALID_ACTIONS
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "entry_price": {"type": ["number", "null"]},
                "stop_loss": {"type": ["number", "null"]},
                "take_profit": {"type": ["number", "null"]},
                "position_size_pct": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "maximum": 100
                },
                "reasoning": {"type": "string", "maxLength": 300}
            }
        }

    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "agent_name": self.agent_name,
            "total_invocations": self._total_invocations,
            "total_cost_usd": self._total_cost,
            "avg_cost_per_decision": (
                self._total_cost / self._total_invocations
                if self._total_invocations > 0 else 0
            ),
            "models_configured": list(self.models.keys()),
        }
