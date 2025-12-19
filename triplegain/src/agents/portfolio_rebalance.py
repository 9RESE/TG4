"""
Portfolio Rebalancing Agent - Maintains target portfolio allocation.

Monitors allocation and executes rebalancing trades to maintain
33/33/33 BTC/XRP/USDT target allocation.

Features:
- Automatic allocation monitoring
- Hodl bag exclusion from calculations
- LLM-assisted execution strategy decisions
- DCA support for large rebalances
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Optional, Any

from .base_agent import BaseAgent, AgentOutput

logger = logging.getLogger(__name__)


# Portfolio Rebalancing System Prompt
REBALANCE_SYSTEM_PROMPT = """You are a portfolio rebalancing assistant. Your role is to analyze current portfolio allocation and determine the optimal rebalancing strategy.

Consider:
1. Current market conditions and volatility
2. Trading fees and slippage impact
3. Tax implications of selling vs buying
4. Optimal execution timing (avoid high volatility periods)

Respond in JSON format:
{
    "should_rebalance": true/false,
    "execution_strategy": "immediate" | "dca_24h" | "limit_orders" | "defer",
    "trades": [
        {
            "symbol": "BTC/USDT",
            "action": "buy" | "sell",
            "amount_usd": 100.0,
            "execution_type": "market" | "limit",
            "priority": 1
        }
    ],
    "reasoning": "Brief explanation of the decision"
}"""


@dataclass
class PortfolioAllocation:
    """Current portfolio allocation state."""
    total_equity_usd: Decimal
    btc_value_usd: Decimal
    xrp_value_usd: Decimal
    usdt_value_usd: Decimal
    btc_pct: Decimal
    xrp_pct: Decimal
    usdt_pct: Decimal
    max_deviation_pct: Decimal
    hodl_excluded: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "total_equity_usd": float(self.total_equity_usd),
            "btc_value_usd": float(self.btc_value_usd),
            "xrp_value_usd": float(self.xrp_value_usd),
            "usdt_value_usd": float(self.usdt_value_usd),
            "btc_pct": float(self.btc_pct),
            "xrp_pct": float(self.xrp_pct),
            "usdt_pct": float(self.usdt_pct),
            "max_deviation_pct": float(self.max_deviation_pct),
            "hodl_excluded": {k: float(v) for k, v in self.hodl_excluded.items()},
        }


@dataclass
class RebalanceTrade:
    """Single rebalancing trade."""
    symbol: str
    action: str  # "buy" or "sell"
    amount_usd: Decimal
    execution_type: str = "limit"  # "market" or "limit"
    priority: int = 1  # Execution order
    batch_index: int = 0  # DCA batch index (0 = immediate)
    scheduled_time: Optional[datetime] = None  # When to execute (for DCA)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "amount_usd": float(self.amount_usd),
            "execution_type": self.execution_type,
            "priority": self.priority,
            "batch_index": self.batch_index,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
        }


@dataclass
class RebalanceOutput(AgentOutput):
    """Portfolio Rebalancing Agent output."""
    action: str = "no_action"  # "no_action" or "rebalance"
    current_allocation: Optional[PortfolioAllocation] = None
    trades: list[RebalanceTrade] = field(default_factory=list)
    execution_strategy: str = "immediate"  # "immediate", "dca_24h", "limit_orders", "defer"
    dca_batches: int = 1  # Number of DCA batches
    dca_interval_hours: int = 0  # Hours between batches
    total_trade_value_usd: Decimal = Decimal(0)  # Total trade value
    used_fallback_strategy: bool = False  # True if LLM failed and defaults were used

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        base = super().to_dict()
        base.update({
            "action": self.action,
            "execution_strategy": self.execution_strategy,
            "current_allocation": self.current_allocation.to_dict() if self.current_allocation else None,
            "trades": [t.to_dict() for t in self.trades],
            "dca_batches": self.dca_batches,
            "dca_interval_hours": self.dca_interval_hours,
            "total_trade_value_usd": float(self.total_trade_value_usd),
            "used_fallback_strategy": self.used_fallback_strategy,
        })
        return base


class PortfolioRebalanceAgent(BaseAgent):
    """
    Portfolio Rebalancing Agent using DeepSeek V3.

    Monitors portfolio allocation and decides rebalancing strategy.

    Target allocation: 33/33/33 BTC/XRP/USDT
    Rebalancing threshold: 5% deviation (configurable)
    Hodl bags are excluded from rebalancing calculations.
    """

    agent_name = "portfolio_rebalance"
    llm_tier = "tier2_api"
    model = "deepseek-chat"

    def __init__(
        self,
        llm_client,
        prompt_builder,
        config: dict,
        kraken_client=None,
        db_pool=None,
    ):
        """
        Initialize PortfolioRebalanceAgent.

        Args:
            llm_client: LLM client for strategy decisions
            prompt_builder: Prompt builder (not used for this agent)
            config: Portfolio configuration
            kraken_client: Kraken API client for balance queries
            db_pool: Database pool for persistence
        """
        super().__init__(llm_client, prompt_builder, config, db_pool)
        self.kraken = kraken_client

        # Target allocation
        target = config.get('target_allocation', {})
        self.target_btc_pct = Decimal(str(target.get('btc_pct', 33.33)))
        self.target_xrp_pct = Decimal(str(target.get('xrp_pct', 33.33)))
        self.target_usdt_pct = Decimal(str(target.get('usdt_pct', 33.34)))

        # Validate target allocations sum to ~100%
        total_allocation = self.target_btc_pct + self.target_xrp_pct + self.target_usdt_pct
        if abs(total_allocation - 100) > Decimal('0.1'):
            logger.warning(
                f"Target allocations sum to {float(total_allocation):.2f}%, not 100%. "
                f"BTC={float(self.target_btc_pct):.2f}%, XRP={float(self.target_xrp_pct):.2f}%, "
                f"USDT={float(self.target_usdt_pct):.2f}%"
            )

        # Rebalancing settings
        rebalancing = config.get('rebalancing', {})
        self.threshold_pct = Decimal(str(rebalancing.get('threshold_pct', 5.0)))
        self.min_trade_usd = Decimal(str(rebalancing.get('min_trade_usd', 10.0)))
        self.default_execution_type = rebalancing.get('execution_type', 'limit')

        # DCA (Dollar Cost Averaging) settings
        dca_config = rebalancing.get('dca', {})
        self.dca_enabled = dca_config.get('enabled', True)
        self.dca_threshold_usd = Decimal(str(dca_config.get('threshold_usd', 500)))
        self.dca_batches = dca_config.get('batches', 6)
        self.dca_interval_hours = dca_config.get('interval_hours', 4)

        # Hodl bag configuration
        hodl = config.get('hodl_bags', {})
        self.hodl_enabled = hodl.get('enabled', True)

        # Price cache
        self._price_cache: dict[str, Decimal] = {}
        self._price_cache_time: Optional[datetime] = None

    async def process(
        self,
        snapshot=None,
        portfolio_context=None,
        force: bool = False,
        **kwargs
    ) -> RebalanceOutput:
        """
        Check and potentially rebalance portfolio.

        Args:
            snapshot: Not used for this agent
            portfolio_context: Not used for this agent
            force: Force rebalancing even if below threshold

        Returns:
            RebalanceOutput with decision and trades
        """
        start_time = time.perf_counter()

        try:
            # Get current allocation
            allocation = await self.check_allocation()

            # Check if rebalancing is needed
            if allocation.max_deviation_pct < self.threshold_pct and not force:
                output = RebalanceOutput(
                    agent_name=self.agent_name,
                    timestamp=datetime.now(timezone.utc),
                    symbol="PORTFOLIO",
                    confidence=1.0,
                    reasoning=(
                        f"Deviation {float(allocation.max_deviation_pct):.1f}% "
                        f"below threshold {float(self.threshold_pct)}%"
                    ),
                    action="no_action",
                    current_allocation=allocation,
                    trades=[],
                    latency_ms=int((time.perf_counter() - start_time) * 1000),
                )
                self._last_output = output
                await self.store_output(output)
                return output

            # Calculate required trades
            calculated_trades = self._calculate_rebalance_trades(allocation)

            # Calculate total trade value before DCA split
            total_trade_value = sum(t.amount_usd for t in calculated_trades)

            # Use LLM for execution strategy if we have trades
            execution_strategy = self.default_execution_type
            used_fallback = False
            if calculated_trades and self.llm:
                try:
                    strategy = await self._get_execution_strategy(allocation, calculated_trades)
                    execution_strategy = strategy.get('execution_strategy', self.default_execution_type)

                    # Update trades with LLM recommendations
                    if 'trades' in strategy:
                        calculated_trades = self._parse_llm_trades(strategy['trades'])
                        # Recalculate total trade value after LLM modifications
                        total_trade_value = sum(t.amount_usd for t in calculated_trades)
                except Exception as e:
                    logger.warning(f"LLM strategy decision failed, using defaults: {e}")
                    used_fallback = True

            # Apply DCA batching for large rebalances
            dca_trades, num_batches, interval_hours = self._create_dca_batches(calculated_trades)

            # Update execution strategy if DCA applied
            if num_batches > 1:
                execution_strategy = "dca_24h"

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            output = RebalanceOutput(
                agent_name=self.agent_name,
                timestamp=datetime.now(timezone.utc),
                symbol="PORTFOLIO",
                confidence=0.8,
                reasoning=(
                    f"Deviation {float(allocation.max_deviation_pct):.1f}% "
                    f"exceeds threshold {float(self.threshold_pct)}%"
                    + (f" (DCA: {num_batches} batches)" if num_batches > 1 else "")
                    + (" [using fallback strategy]" if used_fallback else "")
                ),
                action="rebalance",
                current_allocation=allocation,
                trades=dca_trades,
                execution_strategy=execution_strategy,
                dca_batches=num_batches,
                dca_interval_hours=interval_hours,
                total_trade_value_usd=total_trade_value,
                used_fallback_strategy=used_fallback,
                latency_ms=latency_ms,
                model_used=self.model,
            )

            self._last_output = output
            await self.store_output(output)
            return output

        except Exception as e:
            logger.error(f"Portfolio rebalancing failed: {e}", exc_info=True)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            return RebalanceOutput(
                agent_name=self.agent_name,
                timestamp=datetime.now(timezone.utc),
                symbol="PORTFOLIO",
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                action="no_action",
                trades=[],
                latency_ms=latency_ms,
            )

    async def check_allocation(self) -> PortfolioAllocation:
        """
        Check current portfolio allocation.

        Returns:
            PortfolioAllocation with current state
        """
        # Get balances
        balances = await self._get_balances()

        # Get current prices
        prices = await self._get_current_prices()

        # Get hodl bags
        hodl_bags = await self._get_hodl_bags()

        # Calculate available amounts (excluding hodl bags)
        available_btc = Decimal(str(balances.get('BTC', 0))) - hodl_bags.get('BTC', Decimal(0))
        available_xrp = Decimal(str(balances.get('XRP', 0))) - hodl_bags.get('XRP', Decimal(0))
        available_usdt = Decimal(str(balances.get('USDT', 0))) - hodl_bags.get('USDT', Decimal(0))

        # Ensure non-negative (log warning if hodl bags exceed balance)
        if available_btc < 0:
            logger.warning(
                f"BTC hodl bag ({float(hodl_bags.get('BTC', 0))}) exceeds balance "
                f"({float(balances.get('BTC', 0))}), clamping to 0"
            )
            available_btc = Decimal(0)
        if available_xrp < 0:
            logger.warning(
                f"XRP hodl bag ({float(hodl_bags.get('XRP', 0))}) exceeds balance "
                f"({float(balances.get('XRP', 0))}), clamping to 0"
            )
            available_xrp = Decimal(0)
        if available_usdt < 0:
            logger.warning(
                f"USDT hodl bag ({float(hodl_bags.get('USDT', 0))}) exceeds balance "
                f"({float(balances.get('USDT', 0))}), clamping to 0"
            )
            available_usdt = Decimal(0)

        # Calculate USD values
        btc_price = prices.get('BTC/USDT', Decimal(0))
        xrp_price = prices.get('XRP/USDT', Decimal(0))

        btc_value = available_btc * btc_price
        xrp_value = available_xrp * xrp_price
        usdt_value = available_usdt

        total = btc_value + xrp_value + usdt_value

        # Calculate percentages
        if total > 0:
            btc_pct = (btc_value / total * 100)
            xrp_pct = (xrp_value / total * 100)
            usdt_pct = (usdt_value / total * 100)
        else:
            # Zero equity - set to target allocation to avoid false positive rebalancing
            logger.warning("Total equity is zero - no rebalancing possible")
            btc_pct = self.target_btc_pct
            xrp_pct = self.target_xrp_pct
            usdt_pct = self.target_usdt_pct

        # Calculate max deviation from target
        max_dev = max(
            abs(btc_pct - self.target_btc_pct),
            abs(xrp_pct - self.target_xrp_pct),
            abs(usdt_pct - self.target_usdt_pct),
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
            hodl_excluded={
                'BTC': hodl_bags.get('BTC', Decimal(0)),
                'XRP': hodl_bags.get('XRP', Decimal(0)),
                'USDT': hodl_bags.get('USDT', Decimal(0)),
            },
        )

    def _calculate_rebalance_trades(
        self,
        allocation: PortfolioAllocation
    ) -> list[RebalanceTrade]:
        """Calculate required trades for rebalancing."""
        if allocation.total_equity_usd == 0:
            return []

        trades = []
        target_value = allocation.total_equity_usd / Decimal(3)

        # BTC
        btc_diff = target_value - allocation.btc_value_usd
        if abs(btc_diff) >= self.min_trade_usd:
            trades.append(RebalanceTrade(
                symbol="BTC/USDT",
                action="buy" if btc_diff > 0 else "sell",
                amount_usd=abs(btc_diff),
                execution_type=self.default_execution_type,
                priority=1 if btc_diff < 0 else 2,  # Sell first
            ))

        # XRP
        xrp_diff = target_value - allocation.xrp_value_usd
        if abs(xrp_diff) >= self.min_trade_usd:
            trades.append(RebalanceTrade(
                symbol="XRP/USDT",
                action="buy" if xrp_diff > 0 else "sell",
                amount_usd=abs(xrp_diff),
                execution_type=self.default_execution_type,
                priority=1 if xrp_diff < 0 else 2,
            ))

        # Sort by priority (sells first, then buys)
        return sorted(trades, key=lambda t: t.priority)

    def _create_dca_batches(
        self,
        trades: list[RebalanceTrade],
    ) -> tuple[list[RebalanceTrade], int, int]:
        """
        Split trades into DCA batches if total value exceeds threshold.

        Ensures:
        - Total DCA amount equals original trade amount (rounding handled)
        - Each batch meets minimum trade size (reduces batch count if needed)
        - First batch gets any rounding remainder

        Args:
            trades: Original trades to potentially split

        Returns:
            Tuple of (dca_trades, num_batches, interval_hours)
        """
        if not trades or not self.dca_enabled:
            return trades, 1, 0

        total_value = sum(t.amount_usd for t in trades)

        if total_value < self.dca_threshold_usd:
            # Below threshold - execute immediately
            return trades, 1, 0

        # Determine optimal batch count - reduce if batches would be too small
        num_batches = self.dca_batches
        interval_hours = self.dca_interval_hours

        # Check if batches would be too small - adjust batch count
        for trade in trades:
            batch_amount = trade.amount_usd / Decimal(num_batches)
            while batch_amount < self.min_trade_usd and num_batches > 1:
                num_batches -= 1
                batch_amount = trade.amount_usd / Decimal(num_batches)

        if num_batches == 1:
            # All batches too small - execute immediately
            logger.info(
                f"DCA batches would be below minimum (${float(self.min_trade_usd):.2f}), "
                f"executing immediately"
            )
            return trades, 1, 0

        dca_trades = []
        now = datetime.now(timezone.utc)

        for trade in trades:
            # Calculate batch amounts with proper rounding
            base_batch_amount = trade.amount_usd / Decimal(num_batches)

            # Round DOWN to 2 decimal places to prevent overflow
            # Using ROUND_DOWN ensures total never exceeds original amount
            rounded_batch_amount = base_batch_amount.quantize(Decimal('0.01'), rounding=ROUND_DOWN)

            # Calculate remainder to add to first batch (always positive with ROUND_DOWN)
            remainder = trade.amount_usd - (rounded_batch_amount * num_batches)

            for batch_idx in range(num_batches):
                scheduled_time = now + timedelta(hours=batch_idx * interval_hours)

                # Add remainder to first batch to ensure total equals original
                batch_amount = rounded_batch_amount
                if batch_idx == 0:
                    batch_amount += remainder

                if batch_amount >= self.min_trade_usd:
                    dca_trades.append(RebalanceTrade(
                        symbol=trade.symbol,
                        action=trade.action,
                        amount_usd=batch_amount,
                        execution_type=trade.execution_type,
                        priority=trade.priority,
                        batch_index=batch_idx,
                        scheduled_time=scheduled_time,
                    ))

        # Log DCA summary
        actual_total = sum(t.amount_usd for t in dca_trades)
        logger.info(
            f"Created {len(dca_trades)} DCA trades across {num_batches} batches "
            f"(total: ${float(actual_total):.2f}, original: ${float(total_value):.2f}, "
            f"threshold: ${float(self.dca_threshold_usd):.2f})"
        )

        return dca_trades, num_batches, interval_hours

    async def _get_execution_strategy(
        self,
        allocation: PortfolioAllocation,
        trades: list[RebalanceTrade],
    ) -> dict:
        """Use LLM to determine optimal execution strategy."""
        total_trade_value = sum(t.amount_usd for t in trades)

        prompt = f"""Portfolio Rebalancing Analysis

CURRENT ALLOCATION:
- BTC: {float(allocation.btc_pct):.1f}% (${float(allocation.btc_value_usd):.2f})
- XRP: {float(allocation.xrp_pct):.1f}% (${float(allocation.xrp_value_usd):.2f})
- USDT: {float(allocation.usdt_pct):.1f}% (${float(allocation.usdt_value_usd):.2f})
- Total: ${float(allocation.total_equity_usd):.2f}

TARGET ALLOCATION: 33.33% each

MAX DEVIATION: {float(allocation.max_deviation_pct):.1f}%

PROPOSED TRADES:
{json.dumps([t.to_dict() for t in trades], indent=2)}

TOTAL TRADE VALUE: ${float(total_trade_value):.2f}

Determine the optimal execution strategy considering fees, slippage, and market conditions."""

        response_text, latency_ms, tokens_used = await self._call_llm(
            system_prompt=REBALANCE_SYSTEM_PROMPT,
            user_message=prompt,
        )

        self._total_tokens += tokens_used

        # Parse JSON response
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {response_text}")

        return {"execution_strategy": self.default_execution_type}

    def _parse_llm_trades(self, trades_data: list[dict]) -> list[RebalanceTrade]:
        """Parse trades from LLM response."""
        trades = []
        for td in trades_data:
            try:
                trades.append(RebalanceTrade(
                    symbol=td.get('symbol', ''),
                    action=td.get('action', 'buy'),
                    amount_usd=Decimal(str(td.get('amount_usd', 0))),
                    execution_type=td.get('execution_type', 'limit'),
                    priority=td.get('priority', 1),
                ))
            except (ValueError, KeyError, InvalidOperation) as e:
                logger.warning(f"Failed to parse trade: {td} - {e}")
        return trades

    async def _get_balances(self) -> dict[str, float]:
        """Get current balances from Kraken or config."""
        if self.kraken:
            try:
                result = await self.kraken.get_balances()
                return {
                    'BTC': float(result.get('XXBT', 0)),
                    'XRP': float(result.get('XXRP', 0)),
                    'USDT': float(result.get('USDT', 0)),
                }
            except Exception as e:
                logger.warning(f"Failed to get Kraken balances: {e}")

        # Fallback to config or mock data
        balances = self.config.get('mock_balances', {})
        return {
            'BTC': float(balances.get('BTC', 0)),
            'XRP': float(balances.get('XRP', 0)),
            'USDT': float(balances.get('USDT', 0)),
        }

    async def _get_current_prices(self) -> dict[str, Decimal]:
        """Get current prices for BTC and XRP."""
        # Check cache (5 second TTL)
        now = datetime.now(timezone.utc)
        if self._price_cache_time:
            age = (now - self._price_cache_time).total_seconds()
            if age < 5 and self._price_cache:
                return self._price_cache

        if self.kraken:
            try:
                ticker = await self.kraken.get_ticker(['XBTUSDT', 'XRPUSDT'])
                self._price_cache = {
                    'BTC/USDT': Decimal(str(ticker.get('XBTUSDT', {}).get('c', [0])[0])),
                    'XRP/USDT': Decimal(str(ticker.get('XRPUSDT', {}).get('c', [0])[0])),
                }
                self._price_cache_time = now
                return self._price_cache
            except Exception as e:
                logger.warning(f"Failed to get Kraken prices: {e}")

        # Fallback to config or mock data
        prices = self.config.get('mock_prices', {})
        return {
            'BTC/USDT': Decimal(str(prices.get('BTC/USDT', 45000))),
            'XRP/USDT': Decimal(str(prices.get('XRP/USDT', 0.60))),
        }

    async def _get_hodl_bags(self) -> dict[str, Decimal]:
        """Get hodl bag amounts to exclude from rebalancing."""
        if not self.hodl_enabled:
            return {}

        # Try to get from database
        if self.db:
            try:
                query = """
                    SELECT asset, amount FROM hodl_bags
                    WHERE account_id = 'default'
                """
                rows = await self.db.fetch(query)
                return {
                    row['asset']: Decimal(str(row['amount']))
                    for row in rows
                }
            except Exception as e:
                logger.debug(f"Failed to get hodl bags from DB: {e}")

        # Fallback to config
        hodl = self.config.get('hodl_bags', {})
        return {
            'BTC': Decimal(str(hodl.get('btc_amount', 0))),
            'XRP': Decimal(str(hodl.get('xrp_amount', 0))),
            'USDT': Decimal(str(hodl.get('usdt_amount', 0))),
        }

    def get_output_schema(self) -> dict:
        """Return JSON schema for output validation."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["timestamp", "action", "current_allocation"],
            "properties": {
                "timestamp": {"type": "string", "format": "date-time"},
                "action": {
                    "type": "string",
                    "enum": ["no_action", "rebalance"]
                },
                "execution_strategy": {
                    "type": "string",
                    "enum": ["immediate", "dca_24h", "limit_orders", "defer"]
                },
                "current_allocation": {
                    "type": "object",
                    "properties": {
                        "total_equity_usd": {"type": "number"},
                        "btc_pct": {"type": "number"},
                        "xrp_pct": {"type": "number"},
                        "usdt_pct": {"type": "number"},
                        "max_deviation_pct": {"type": "number"},
                    }
                },
                "trades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "action": {"type": "string", "enum": ["buy", "sell"]},
                            "amount_usd": {"type": "number"},
                            "execution_type": {"type": "string"},
                            "priority": {"type": "integer"},
                        }
                    }
                },
                "reasoning": {"type": "string"},
            }
        }
