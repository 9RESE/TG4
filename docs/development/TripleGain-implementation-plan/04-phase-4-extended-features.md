# Phase 4: Extended Features

**Phase Status**: Pending Phase 3 Completion
**Dependencies**: Phase 3 (Orchestration, Execution)
**Deliverable**: Full feature set with monitoring and A/B tracking

---

## Overview

Phase 4 adds extended features that enhance the core trading system: sentiment analysis with web search, the hodl bag accumulation system, the 6-model A/B testing framework, and the monitoring dashboard.

### Components

| Component | Description | Depends On |
|-----------|-------------|------------|
| 4.1 Sentiment Analysis Agent | Grok X.com + GPT with web search | Phase 3 Message Bus |
| 4.2 Hodl Bag System | Profit allocation to long-term holdings | Phase 3 Execution |
| 4.3 LLM 6-Model A/B Framework | Performance tracking across models | Phase 2 Trading Agent |
| 4.4 Dashboard | React frontend for monitoring | All Phase 3 components |

---

## 4.1 Sentiment Analysis Agent

### Purpose

Gathers real-time market sentiment using Grok and GPT's native web search capabilities. Provides sentiment signals to the Trading Decision Agent.

### LLM Assignment

| Property | Value |
|----------|-------|
| Primary Model | Grok (xAI) - web search, Twitter access |
| Secondary Model | GPT (OpenAI) - web search |
| Invocation | Every 60 minutes |
| Latency Target | < 10s (web search involved) |
| Tier | Tier 2 (API) |

### Dual-Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SENTIMENT ANALYSIS DUAL-MODEL FLOW                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      EVERY 30 MINUTES                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│              ┌─────────────────────┴─────────────────────┐                 │
│              ▼                                           ▼                  │
│  ┌───────────────────────┐               ┌───────────────────────┐        │
│  │   GROK (xAI)          │               │   GPT (OpenAI)        │        │
│  │                       │               │                       │        │
│  │  Capabilities:        │               │  Capabilities:        │        │
│  │  • Web search         │               │  • Web search         │        │
│  │  • Twitter/X access   │               │  • News aggregation   │        │
│  │  • Real-time news     │               │  • Broad web access   │        │
│  │                       │               │                       │        │
│  │  Focus:               │               │  Focus:               │        │
│  │  • Crypto Twitter     │               │  • News articles      │        │
│  │  • Social sentiment   │               │  • Market analysis    │        │
│  │  • Influencer posts   │               │  • Regulatory news    │        │
│  └───────────┬───────────┘               └───────────┬───────────┘        │
│              │                                       │                     │
│              └─────────────────┬─────────────────────┘                     │
│                                ▼                                            │
│                    ┌───────────────────────┐                               │
│                    │  AGGREGATION LOGIC    │                               │
│                    │                       │                               │
│                    │  • Combine scores     │                               │
│                    │  • Weight by source   │                               │
│                    │  • Detect conflicts   │                               │
│                    │  • Generate summary   │                               │
│                    └───────────┬───────────┘                               │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                               │
│                    │  SENTIMENT OUTPUT     │                               │
│                    │                       │                               │
│                    │  bias: bullish/bear   │                               │
│                    │  confidence: 0.65     │                               │
│                    │  key_events: [...]    │                               │
│                    │  sources: [...]       │                               │
│                    └───────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["timestamp", "symbol", "bias", "confidence", "sources"],
  "properties": {
    "timestamp": {"type": "string", "format": "date-time"},
    "symbol": {"type": "string"},
    "bias": {
      "type": "string",
      "enum": ["very_bullish", "bullish", "neutral", "bearish", "very_bearish"]
    },
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "sentiment_scores": {
      "type": "object",
      "properties": {
        "social_media": {"type": "number", "minimum": -1, "maximum": 1},
        "news": {"type": "number", "minimum": -1, "maximum": 1},
        "overall": {"type": "number", "minimum": -1, "maximum": 1}
      }
    },
    "key_events": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "event": {"type": "string"},
          "impact": {"type": "string", "enum": ["positive", "negative", "neutral"]},
          "significance": {"type": "string", "enum": ["low", "medium", "high"]},
          "source": {"type": "string"}
        }
      }
    },
    "market_narratives": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Current market narratives being discussed"
    },
    "fear_greed_assessment": {
      "type": "string",
      "enum": ["extreme_fear", "fear", "neutral", "greed", "extreme_greed"]
    },
    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "provider": {"type": "string"},
          "query_type": {"type": "string"},
          "result_count": {"type": "integer"}
        }
      }
    },
    "reasoning": {"type": "string", "maxLength": 500}
  }
}
```

### Interface Definition

```python
# src/agents/sentiment_analysis.py

from dataclasses import dataclass
from typing import Optional
from enum import Enum

class SentimentBias(Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class KeyEvent:
    """Significant market event."""
    event: str
    impact: str  # positive, negative, neutral
    significance: str  # low, medium, high
    source: str


@dataclass
class SentimentOutput(AgentOutput):
    """Sentiment Analysis Agent output."""
    bias: SentimentBias
    confidence: float
    social_score: float  # -1 to 1
    news_score: float  # -1 to 1
    overall_score: float  # -1 to 1
    key_events: list[KeyEvent]
    market_narratives: list[str]
    fear_greed: str
    reasoning: str


@dataclass
class ProviderResult:
    """Result from a single LLM provider."""
    provider: str
    bias: SentimentBias
    confidence: float
    score: float
    events: list[KeyEvent]
    latency_ms: int


class SentimentAnalysisAgent(BaseAgent):
    """
    Sentiment Analysis Agent using Grok + GPT.

    Both models have web search capabilities.
    Grok has Twitter/X access for social sentiment.
    """

    agent_name = "sentiment_analysis"
    llm_tier = "tier2_api"

    def __init__(
        self,
        grok_client,
        gpt_client,
        prompt_builder,
        config: dict
    ):
        self.grok = grok_client
        self.gpt = gpt_client
        self.prompt_builder = prompt_builder
        self.config = config

    async def process(
        self,
        symbol: str,
        include_twitter: bool = True
    ) -> SentimentOutput:
        """
        Analyze market sentiment using Grok and GPT.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            include_twitter: Whether to include Twitter analysis (Grok)

        Returns:
            SentimentOutput with aggregated sentiment
        """
        # Extract base asset for search
        base_asset = symbol.split("/")[0]  # "BTC" from "BTC/USDT"

        # Run both providers in parallel
        grok_task = self._query_grok(base_asset, include_twitter)
        gpt_task = self._query_gpt(base_asset)

        grok_result, gpt_result = await asyncio.gather(
            grok_task, gpt_task,
            return_exceptions=True
        )

        # Handle failures
        results = []
        if isinstance(grok_result, ProviderResult):
            results.append(grok_result)
        if isinstance(gpt_result, ProviderResult):
            results.append(gpt_result)

        if not results:
            # Both failed - return neutral
            return self._create_neutral_output(symbol)

        # Aggregate results
        return self._aggregate_results(symbol, results)

    async def _query_grok(
        self,
        asset: str,
        include_twitter: bool
    ) -> ProviderResult:
        """Query Grok with web search and Twitter access."""
        prompt = self._build_grok_prompt(asset, include_twitter)

        start_time = time.time()
        response = await self.grok.generate(
            model="grok-2",
            system_prompt=SENTIMENT_SYSTEM_PROMPT,
            user_message=prompt,
            tools=["web_search", "twitter_search"] if include_twitter else ["web_search"]
        )
        latency = int((time.time() - start_time) * 1000)

        return self._parse_provider_result("grok", response.text, latency)

    async def _query_gpt(self, asset: str) -> ProviderResult:
        """Query GPT with web search."""
        prompt = self._build_gpt_prompt(asset)

        start_time = time.time()
        response = await self.gpt.generate(
            model="gpt-4-turbo",
            system_prompt=SENTIMENT_SYSTEM_PROMPT,
            user_message=prompt,
            tools=["web_search"]
        )
        latency = int((time.time() - start_time) * 1000)

        return self._parse_provider_result("gpt", response.text, latency)

    def _aggregate_results(
        self,
        symbol: str,
        results: list[ProviderResult]
    ) -> SentimentOutput:
        """
        Aggregate results from multiple providers.

        Weighting:
        - Grok (with Twitter): 60% for social sentiment
        - GPT: 60% for news sentiment
        """
        # Calculate weighted scores
        social_scores = []
        news_scores = []

        for r in results:
            if r.provider == "grok":
                social_scores.append((r.score, 0.6))
                news_scores.append((r.score, 0.4))
            else:
                social_scores.append((r.score, 0.4))
                news_scores.append((r.score, 0.6))

        social_score = sum(s * w for s, w in social_scores) / sum(w for _, w in social_scores)
        news_score = sum(s * w for s, w in news_scores) / sum(w for _, w in news_scores)
        overall_score = (social_score + news_score) / 2

        # Determine bias
        bias = self._score_to_bias(overall_score)

        # Calculate confidence (higher when providers agree)
        score_variance = np.var([r.score for r in results]) if len(results) > 1 else 0
        confidence = max(0.3, 1.0 - score_variance * 2)

        # Combine events
        all_events = []
        for r in results:
            all_events.extend(r.events)

        # Deduplicate events by similarity
        unique_events = self._deduplicate_events(all_events)

        return SentimentOutput(
            bias=bias,
            confidence=confidence,
            social_score=social_score,
            news_score=news_score,
            overall_score=overall_score,
            key_events=unique_events[:5],  # Top 5 events
            market_narratives=self._extract_narratives(results),
            fear_greed=self._assess_fear_greed(overall_score),
            reasoning=self._combine_reasoning(results)
        )

    def _build_grok_prompt(self, asset: str, include_twitter: bool) -> str:
        """Build prompt for Grok."""
        twitter_section = """
TWITTER/X ANALYSIS:
Search for recent tweets about {asset} from:
- Major crypto influencers
- Project official accounts
- Institutional traders
- Sentiment of replies and engagement
""" if include_twitter else ""

        return f"""
Analyze current market sentiment for {asset} cryptocurrency.

WEB SEARCH:
Search for recent news and analysis about {asset}:
- Breaking news (last 24 hours)
- Price analysis articles
- Regulatory developments
- Institutional adoption news
- Technical developments

{twitter_section}

Provide a comprehensive sentiment analysis based on your findings.
"""

    def _build_gpt_prompt(self, asset: str) -> str:
        """Build prompt for GPT."""
        return f"""
Analyze current market sentiment for {asset} cryptocurrency.

WEB SEARCH:
Search for recent news and analysis about {asset}:
- Breaking news from major crypto outlets
- Regulatory news and developments
- Institutional investment news
- Market analysis and predictions
- Technical and fundamental analysis articles

Focus on factual news rather than social media sentiment.
Provide a comprehensive sentiment analysis based on your findings.
"""
```

### Configuration

```yaml
# config/agents.yaml (sentiment_analysis section)

agents:
  sentiment_analysis:
    enabled: true

    invocation:
      trigger: scheduled
      interval_seconds: 1800  # 30 minutes
      symbols:
        - BTC/USDT
        - XRP/USDT

    providers:
      grok:
        enabled: true
        model: grok-2
        timeout_ms: 30000
        capabilities:
          - web_search
          - twitter_search
        weight:
          social: 0.6
          news: 0.4

      gpt:
        enabled: true
        model: gpt-4-turbo
        timeout_ms: 30000
        capabilities:
          - web_search
        weight:
          social: 0.4
          news: 0.6

    aggregation:
      min_providers: 1  # At least one must succeed
      confidence_boost_on_agreement: 0.1

    output:
      store_all: true
      cache_ttl_seconds: 1750
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Grok query | Web search returns results | Valid response |
| GPT query | Web search returns results | Valid response |
| Aggregation | Combines provider results | Correct weighting |
| Fallback | Single provider failure handled | Other provider used |
| Event deduplication | Similar events merged | No duplicates |

---

## 4.2 Hodl Bag System

### Purpose

Automatically allocates a percentage of trading profits to long-term "hodl bag" positions in BTC and XRP that are excluded from rebalancing and trading.

### Accumulation Logic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HODL BAG ACCUMULATION FLOW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PROFIT REALIZED (Trade closes with gain)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Trade Result:                                                        │   │
│  │   Symbol: BTC/USDT                                                   │   │
│  │   Realized P&L: +$85.00                                             │   │
│  │                                                                      │   │
│  │ Hodl Allocation: 10% of profit                                      │   │
│  │   To Hodl: $8.50                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. ALLOCATION CALCULATION                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Profit from BTC trade → 50% to BTC hodl, 50% to XRP hodl            │   │
│  │ Profit from XRP trade → 50% to XRP hodl, 50% to BTC hodl            │   │
│  │                                                                      │   │
│  │ This example ($8.50 from BTC trade):                                │   │
│  │   BTC hodl: +$4.25 worth                                            │   │
│  │   XRP hodl: +$4.25 worth                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. EXECUTION (Immediate or batched)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ If accumulated hodl amount > minimum threshold ($10):                │   │
│  │   → Execute purchase                                                 │   │
│  │   → Transfer to hodl balance (excluded from trading)                │   │
│  │   → Update hodl_bags table                                          │   │
│  │                                                                      │   │
│  │ Else:                                                                │   │
│  │   → Add to pending hodl accumulation                                │   │
│  │   → Execute when threshold reached                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  4. HODL BAG STATE                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ BTC Hodl Bag:                                                        │   │
│  │   Balance: 0.0155 BTC                                               │   │
│  │   Value: $698.25                                                    │   │
│  │   Cost Basis: $612.40                                               │   │
│  │   Unrealized Gain: +$85.85 (+14.0%)                                │   │
│  │   First Accumulation: 2025-01-15                                    │   │
│  │   Last Accumulation: 2025-12-18                                     │   │
│  │                                                                      │   │
│  │ XRP Hodl Bag:                                                        │   │
│  │   Balance: 850 XRP                                                  │   │
│  │   Value: $510.00                                                    │   │
│  │   Cost Basis: $442.00                                               │   │
│  │   Unrealized Gain: +$68.00 (+15.4%)                                │   │
│  │   First Accumulation: 2025-01-15                                    │   │
│  │   Last Accumulation: 2025-12-18                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Database Schema

```sql
-- Hodl bag holdings
CREATE TABLE hodl_bags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset VARCHAR(10) NOT NULL,  -- BTC, XRP
    balance DECIMAL(20, 10) NOT NULL DEFAULT 0,
    cost_basis_usd DECIMAL(20, 2) NOT NULL DEFAULT 0,
    first_accumulation TIMESTAMPTZ,
    last_accumulation TIMESTAMPTZ,
    last_valuation_usd DECIMAL(20, 2),
    last_valuation_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(asset)
);

-- Hodl bag transactions
CREATE TABLE hodl_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset VARCHAR(10) NOT NULL,
    transaction_type VARCHAR(20) NOT NULL CHECK (
        transaction_type IN ('accumulation', 'withdrawal', 'adjustment')
    ),
    amount DECIMAL(20, 10) NOT NULL,
    price_usd DECIMAL(20, 10) NOT NULL,
    value_usd DECIMAL(20, 2) NOT NULL,
    source_trade_id UUID,  -- Reference to trade_executions
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hodl_transactions_asset
    ON hodl_transactions (asset, timestamp DESC);

-- Pending hodl accumulation (not yet executed)
CREATE TABLE hodl_pending (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset VARCHAR(10) NOT NULL,
    amount_usd DECIMAL(20, 2) NOT NULL,
    source_trade_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    executed_at TIMESTAMPTZ,
    execution_transaction_id UUID
);

CREATE INDEX idx_hodl_pending_asset
    ON hodl_pending (asset, executed_at NULLS FIRST);
```

### Interface Definition

```python
# src/execution/hodl_bag.py

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

@dataclass
class HodlBagState:
    """Current state of a hodl bag."""
    asset: str
    balance: Decimal
    cost_basis_usd: Decimal
    current_value_usd: Decimal
    unrealized_pnl_usd: Decimal
    unrealized_pnl_pct: Decimal
    first_accumulation: Optional[datetime]
    last_accumulation: Optional[datetime]


@dataclass
class HodlAllocation:
    """Allocation to hodl bags from a profit."""
    btc_amount_usd: Decimal
    xrp_amount_usd: Decimal
    total_amount_usd: Decimal


class HodlBagManager:
    """
    Manages hodl bag accumulation from trading profits.

    Hodl bags are long-term holdings excluded from trading and rebalancing.
    """

    def __init__(
        self,
        db_pool,
        kraken_client,
        config: dict
    ):
        self.db = db_pool
        self.kraken = kraken_client
        self.config = config
        self._pending_accumulation: dict[str, Decimal] = {"BTC": Decimal(0), "XRP": Decimal(0)}

    async def process_trade_profit(
        self,
        trade_id: str,
        profit_usd: Decimal,
        source_symbol: str
    ) -> Optional[HodlAllocation]:
        """
        Process profit from a closed trade for hodl accumulation.

        Args:
            trade_id: ID of the profitable trade
            profit_usd: Realized profit in USD
            source_symbol: Symbol of the trade (e.g., "BTC/USDT")

        Returns:
            HodlAllocation if accumulation occurred, None if deferred
        """
        if profit_usd <= 0:
            return None

        # Calculate allocation percentage
        allocation_pct = Decimal(str(self.config["allocation_pct"])) / 100
        to_hodl = profit_usd * allocation_pct

        # Determine split (50/50 between BTC and XRP)
        btc_allocation = to_hodl / 2
        xrp_allocation = to_hodl / 2

        # Add to pending
        self._pending_accumulation["BTC"] += btc_allocation
        self._pending_accumulation["XRP"] += xrp_allocation

        # Record pending
        await self._record_pending(trade_id, "BTC", btc_allocation)
        await self._record_pending(trade_id, "XRP", xrp_allocation)

        # Check if threshold reached
        min_threshold = Decimal(str(self.config["min_accumulation_usd"]))

        allocations_made = []
        for asset in ["BTC", "XRP"]:
            if self._pending_accumulation[asset] >= min_threshold:
                allocation = await self._execute_accumulation(asset)
                if allocation:
                    allocations_made.append((asset, allocation))

        if allocations_made:
            return HodlAllocation(
                btc_amount_usd=next((a for k, a in allocations_made if k == "BTC"), Decimal(0)),
                xrp_amount_usd=next((a for k, a in allocations_made if k == "XRP"), Decimal(0)),
                total_amount_usd=sum(a for _, a in allocations_made)
            )

        return None

    async def _execute_accumulation(self, asset: str) -> Optional[Decimal]:
        """
        Execute hodl accumulation purchase.

        Args:
            asset: Asset to accumulate ("BTC" or "XRP")

        Returns:
            Amount in USD accumulated, or None if failed
        """
        amount_usd = self._pending_accumulation[asset]
        if amount_usd <= 0:
            return None

        # Get current price
        symbol = f"{asset}/USDT"
        prices = await self._get_current_prices()
        price = prices.get(symbol)

        if not price:
            return None

        # Calculate amount to buy
        amount = amount_usd / price

        # Execute purchase
        try:
            result = await self.kraken.add_order(
                pair=self._to_kraken_symbol(symbol),
                type="buy",
                ordertype="market",
                volume=str(amount)
            )

            if "error" in result and result["error"]:
                return None

            # Wait for fill
            fill_price = await self._wait_for_fill(result["result"]["txid"][0])

            # Update hodl bag
            await self._update_hodl_bag(
                asset=asset,
                amount=amount,
                price_usd=fill_price,
                value_usd=amount_usd
            )

            # Clear pending
            self._pending_accumulation[asset] = Decimal(0)
            await self._mark_pending_executed(asset)

            return amount_usd

        except Exception as e:
            return None

    async def get_hodl_state(self) -> dict[str, HodlBagState]:
        """Get current state of all hodl bags."""
        states = {}

        for asset in ["BTC", "XRP"]:
            bag = await self.db.fetchrow(
                "SELECT * FROM hodl_bags WHERE asset = $1",
                asset
            )

            if bag:
                # Get current price
                prices = await self._get_current_prices()
                current_price = prices.get(f"{asset}/USDT", Decimal(0))
                current_value = bag["balance"] * current_price

                unrealized_pnl = current_value - bag["cost_basis_usd"]
                unrealized_pnl_pct = (
                    (unrealized_pnl / bag["cost_basis_usd"]) * 100
                    if bag["cost_basis_usd"] > 0 else Decimal(0)
                )

                states[asset] = HodlBagState(
                    asset=asset,
                    balance=bag["balance"],
                    cost_basis_usd=bag["cost_basis_usd"],
                    current_value_usd=current_value,
                    unrealized_pnl_usd=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    first_accumulation=bag["first_accumulation"],
                    last_accumulation=bag["last_accumulation"]
                )
            else:
                states[asset] = HodlBagState(
                    asset=asset,
                    balance=Decimal(0),
                    cost_basis_usd=Decimal(0),
                    current_value_usd=Decimal(0),
                    unrealized_pnl_usd=Decimal(0),
                    unrealized_pnl_pct=Decimal(0),
                    first_accumulation=None,
                    last_accumulation=None
                )

        return states

    async def _update_hodl_bag(
        self,
        asset: str,
        amount: Decimal,
        price_usd: Decimal,
        value_usd: Decimal
    ) -> None:
        """Update hodl bag with new accumulation."""
        now = datetime.utcnow()

        # Upsert hodl bag
        await self.db.execute("""
            INSERT INTO hodl_bags (asset, balance, cost_basis_usd, first_accumulation, last_accumulation, updated_at)
            VALUES ($1, $2, $3, $4, $4, $4)
            ON CONFLICT (asset)
            DO UPDATE SET
                balance = hodl_bags.balance + $2,
                cost_basis_usd = hodl_bags.cost_basis_usd + $3,
                last_accumulation = $4,
                updated_at = $4
        """, asset, amount, value_usd, now)

        # Record transaction
        await self.db.execute("""
            INSERT INTO hodl_transactions (asset, transaction_type, amount, price_usd, value_usd)
            VALUES ($1, 'accumulation', $2, $3, $4)
        """, asset, amount, price_usd, value_usd)
```

### Configuration

```yaml
# config/hodl.yaml

hodl_bags:
  enabled: true

  # Allocation from profits
  allocation_pct: 10  # 10% of profits to hodl bags

  # Minimum accumulation before purchase
  min_accumulation_usd: 10

  # Assets to accumulate
  assets:
    - BTC
    - XRP

  # Split strategy
  split:
    btc_pct: 50
    xrp_pct: 50

  # Execution
  execution:
    order_type: market  # Immediate execution
    retry_on_failure: true
    max_retries: 3
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Profit allocation | Correct percentage allocated | 10% of profit |
| Split calculation | 50/50 BTC/XRP split | Equal amounts |
| Threshold batching | Below threshold accumulated | Not executed |
| Execution | Purchase executed on threshold | Order placed |
| State tracking | Balances updated correctly | Accurate state |

---

## 4.3 LLM 6-Model A/B Testing Framework

### Purpose

Comprehensive framework for tracking and comparing performance across all 6 LLM models used for trading decisions.

### Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    6-MODEL A/B TESTING FRAMEWORK                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODELS UNDER TEST:                                                         │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐   │
│  │ GPT         │ Grok        │ DeepSeek V3 │Claude Sonnet│Claude Opus  │   │
│  │ (OpenAI)    │ (xAI)       │ (DeepSeek)  │ (Anthropic) │ (Anthropic) │   │
│  └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘   │
│  ┌─────────────┐                                                           │
│  │ Qwen 2.5 7B │                                                           │
│  │ (Local)     │                                                           │
│  └─────────────┘                                                           │
│                                                                             │
│  TRACKING DIMENSIONS:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Decision Accuracy                                                 │   │
│  │    - Correct BUY/SELL/HOLD predictions (1h, 4h, 24h horizons)       │   │
│  │    - By market regime                                                │   │
│  │    - By symbol                                                       │   │
│  │                                                                      │   │
│  │ 2. Profitability                                                     │   │
│  │    - Simulated P&L if model's decision was executed alone           │   │
│  │    - Contribution to consensus decisions                             │   │
│  │    - Risk-adjusted returns (Sharpe)                                  │   │
│  │                                                                      │   │
│  │ 3. Confidence Calibration                                            │   │
│  │    - Does confidence match actual accuracy?                          │   │
│  │    - Expected Calibration Error (ECE)                               │   │
│  │                                                                      │   │
│  │ 4. Operational Metrics                                               │   │
│  │    - Latency (p50, p95, p99)                                        │   │
│  │    - Cost per decision                                               │   │
│  │    - Error rate                                                      │   │
│  │    - Timeout rate                                                    │   │
│  │                                                                      │   │
│  │ 5. Consensus Contribution                                            │   │
│  │    - How often on winning side of consensus                          │   │
│  │    - How often selected as parameter source                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LEADERBOARD UPDATE:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Composite Score = (Accuracy × 0.35) + (Profit × 0.25) +             │   │
│  │                   (Calibration × 0.20) + (Latency × 0.10) +         │   │
│  │                   (Cost × 0.10)                                      │   │
│  │                                                                      │   │
│  │ Updated: Hourly                                                      │   │
│  │ Statistical Significance: Pairwise t-tests (p < 0.05)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Database Schema

```sql
-- Model performance snapshots (updated hourly)
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_id VARCHAR(20) NOT NULL,

    -- Decision metrics
    total_decisions INTEGER NOT NULL DEFAULT 0,
    correct_decisions INTEGER NOT NULL DEFAULT 0,
    accuracy_pct DECIMAL(5, 2),

    -- By decision type
    buy_accuracy_pct DECIMAL(5, 2),
    sell_accuracy_pct DECIMAL(5, 2),
    hold_accuracy_pct DECIMAL(5, 2),

    -- Profitability
    simulated_pnl_usd DECIMAL(20, 2),
    simulated_pnl_pct DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),

    -- Calibration
    expected_calibration_error DECIMAL(5, 4),
    is_well_calibrated BOOLEAN,

    -- Operational
    avg_latency_ms INTEGER,
    p95_latency_ms INTEGER,
    total_cost_usd DECIMAL(10, 4),
    cost_per_decision_usd DECIMAL(10, 6),
    error_rate_pct DECIMAL(5, 2),

    -- Consensus
    consensus_win_rate_pct DECIMAL(5, 2),
    times_selected_for_params INTEGER,

    -- Composite score
    composite_score DECIMAL(5, 4),

    -- Period
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_performance_ts
    ON model_performance (timestamp DESC, model_id);

-- Leaderboard (latest rankings)
CREATE TABLE model_leaderboard (
    model_id VARCHAR(20) PRIMARY KEY,
    rank INTEGER NOT NULL,
    composite_score DECIMAL(5, 4) NOT NULL,
    accuracy_pct DECIMAL(5, 2),
    profit_pct DECIMAL(10, 4),
    calibration_score DECIMAL(5, 4),
    cost_efficiency DECIMAL(10, 4),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Pairwise significance tests
CREATE TABLE model_pairwise_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_a VARCHAR(20) NOT NULL,
    model_b VARCHAR(20) NOT NULL,
    metric VARCHAR(50) NOT NULL,
    t_statistic DECIMAL(10, 4),
    p_value DECIMAL(10, 6),
    is_significant BOOLEAN,
    winner VARCHAR(20),
    sample_size INTEGER
);

CREATE INDEX idx_pairwise_tests_ts
    ON model_pairwise_tests (timestamp DESC);
```

### Interface Definition

```python
# src/llm/model_comparison.py

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from scipy import stats
import numpy as np

@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    model_id: str
    total_decisions: int
    correct_decisions: int
    accuracy_pct: Decimal
    buy_accuracy_pct: Decimal
    sell_accuracy_pct: Decimal
    hold_accuracy_pct: Decimal
    simulated_pnl_pct: Decimal
    sharpe_ratio: Decimal
    calibration_error: Decimal
    avg_latency_ms: int
    total_cost_usd: Decimal
    error_rate_pct: Decimal
    consensus_win_rate_pct: Decimal
    composite_score: Decimal


@dataclass
class LeaderboardEntry:
    """Single entry in model leaderboard."""
    rank: int
    model_id: str
    composite_score: Decimal
    accuracy_pct: Decimal
    profit_pct: Decimal
    best_at: Optional[str]  # e.g., "BUY decisions"


@dataclass
class PairwiseTest:
    """Result of pairwise significance test."""
    model_a: str
    model_b: str
    metric: str
    t_statistic: float
    p_value: float
    is_significant: bool
    winner: Optional[str]


class ModelComparisonFramework:
    """
    Framework for comparing 6 LLM models.

    Tracks performance, calculates rankings, and tests significance.
    """

    MODELS = ["gpt", "grok", "deepseek", "claude_sonnet", "claude_opus", "qwen"]

    COMPOSITE_WEIGHTS = {
        "accuracy": 0.35,
        "profitability": 0.25,
        "calibration": 0.20,
        "latency": 0.10,
        "cost": 0.10
    }

    def __init__(self, db_pool, config: dict):
        self.db = db_pool
        self.config = config

    async def record_decision(
        self,
        model_id: str,
        decision: dict,
        latency_ms: int,
        cost_usd: Decimal,
        error: Optional[str] = None
    ) -> None:
        """
        Record a model decision for tracking.

        Args:
            model_id: Model identifier
            decision: Decision output (action, confidence, etc.)
            latency_ms: Response latency
            cost_usd: API cost
            error: Error message if failed
        """
        await self.db.execute("""
            INSERT INTO model_comparisons
            (timestamp, model_id, decision_data, latency_ms, cost_usd, error)
            VALUES (NOW(), $1, $2, $3, $4, $5)
        """, model_id, decision, latency_ms, cost_usd, error)

    async def update_outcome(
        self,
        decision_id: str,
        price_after_1h: Decimal,
        price_after_4h: Decimal,
        price_after_24h: Decimal
    ) -> None:
        """Update decision with actual outcome for accuracy calculation."""
        pass

    async def calculate_metrics(
        self,
        model_id: str,
        period_hours: int = 24
    ) -> ModelMetrics:
        """
        Calculate comprehensive metrics for a model.

        Args:
            model_id: Model identifier
            period_hours: Lookback period

        Returns:
            ModelMetrics with all calculated metrics
        """
        # Query decisions from period
        decisions = await self.db.fetch("""
            SELECT * FROM model_comparisons
            WHERE model_id = $1
              AND timestamp > NOW() - INTERVAL '$2 hours'
              AND outcome_correct IS NOT NULL
        """, model_id, period_hours)

        if not decisions:
            return self._empty_metrics(model_id)

        # Calculate accuracy
        total = len(decisions)
        correct = sum(1 for d in decisions if d["outcome_correct"])
        accuracy = Decimal(correct) / Decimal(total) * 100

        # Calculate by decision type
        buy_decisions = [d for d in decisions if d["action"] == "BUY"]
        sell_decisions = [d for d in decisions if d["action"] == "SELL"]
        hold_decisions = [d for d in decisions if d["action"] == "HOLD"]

        buy_accuracy = self._calc_accuracy(buy_decisions)
        sell_accuracy = self._calc_accuracy(sell_decisions)
        hold_accuracy = self._calc_accuracy(hold_decisions)

        # Calculate simulated P&L
        pnl = self._simulate_pnl(decisions)
        sharpe = self._calculate_sharpe(decisions)

        # Calculate calibration
        calibration_error = self._calculate_ece(decisions)

        # Operational metrics
        latencies = [d["latency_ms"] for d in decisions]
        avg_latency = int(np.mean(latencies))
        costs = [d["cost_usd"] for d in decisions]
        total_cost = sum(costs)
        error_rate = sum(1 for d in decisions if d.get("error")) / total * 100

        # Consensus contribution
        consensus_wins = sum(
            1 for d in decisions
            if d.get("was_consensus_winner", False)
        )
        consensus_win_rate = Decimal(consensus_wins) / Decimal(total) * 100

        # Composite score
        composite = self._calculate_composite_score(
            accuracy=accuracy,
            profit=pnl,
            calibration=1 - calibration_error,  # Higher is better
            latency=1000 / avg_latency if avg_latency > 0 else 0,  # Inverse
            cost=1 / float(total_cost) if total_cost > 0 else 1  # Inverse
        )

        return ModelMetrics(
            model_id=model_id,
            total_decisions=total,
            correct_decisions=correct,
            accuracy_pct=accuracy,
            buy_accuracy_pct=buy_accuracy,
            sell_accuracy_pct=sell_accuracy,
            hold_accuracy_pct=hold_accuracy,
            simulated_pnl_pct=pnl,
            sharpe_ratio=sharpe,
            calibration_error=calibration_error,
            avg_latency_ms=avg_latency,
            total_cost_usd=total_cost,
            error_rate_pct=error_rate,
            consensus_win_rate_pct=consensus_win_rate,
            composite_score=composite
        )

    async def update_leaderboard(self) -> list[LeaderboardEntry]:
        """
        Update model leaderboard based on recent performance.

        Returns:
            Sorted leaderboard entries
        """
        metrics = {}
        for model_id in self.MODELS:
            metrics[model_id] = await self.calculate_metrics(model_id)

        # Sort by composite score
        ranked = sorted(
            metrics.values(),
            key=lambda m: m.composite_score,
            reverse=True
        )

        leaderboard = []
        for rank, m in enumerate(ranked, 1):
            # Determine what model is best at
            best_at = None
            if m.buy_accuracy_pct == max(r.buy_accuracy_pct for r in ranked):
                best_at = "BUY decisions"
            elif m.sell_accuracy_pct == max(r.sell_accuracy_pct for r in ranked):
                best_at = "SELL decisions"
            elif m.hold_accuracy_pct == max(r.hold_accuracy_pct for r in ranked):
                best_at = "HOLD decisions"

            entry = LeaderboardEntry(
                rank=rank,
                model_id=m.model_id,
                composite_score=m.composite_score,
                accuracy_pct=m.accuracy_pct,
                profit_pct=m.simulated_pnl_pct,
                best_at=best_at
            )
            leaderboard.append(entry)

            # Store in DB
            await self.db.execute("""
                INSERT INTO model_leaderboard (model_id, rank, composite_score, accuracy_pct, profit_pct, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (model_id)
                DO UPDATE SET rank = $2, composite_score = $3, accuracy_pct = $4, profit_pct = $5, updated_at = NOW()
            """, m.model_id, rank, m.composite_score, m.accuracy_pct, m.simulated_pnl_pct)

        return leaderboard

    async def run_pairwise_tests(
        self,
        metric: str = "pnl"
    ) -> list[PairwiseTest]:
        """
        Run pairwise significance tests between all models.

        Returns:
            List of pairwise test results (15 comparisons for 6 models)
        """
        from itertools import combinations

        results = []

        # Get all decisions with outcomes
        model_data = {}
        for model_id in self.MODELS:
            decisions = await self.db.fetch("""
                SELECT simulated_pnl FROM model_comparisons
                WHERE model_id = $1
                  AND timestamp > NOW() - INTERVAL '7 days'
                  AND outcome_correct IS NOT NULL
            """, model_id)
            model_data[model_id] = [d["simulated_pnl"] for d in decisions]

        # Pairwise t-tests
        for model_a, model_b in combinations(self.MODELS, 2):
            data_a = model_data.get(model_a, [])
            data_b = model_data.get(model_b, [])

            if len(data_a) < 30 or len(data_b) < 30:
                # Not enough data
                continue

            t_stat, p_value = stats.ttest_ind(data_a, data_b)

            is_significant = p_value < 0.05
            winner = None
            if is_significant:
                winner = model_a if np.mean(data_a) > np.mean(data_b) else model_b

            test_result = PairwiseTest(
                model_a=model_a,
                model_b=model_b,
                metric=metric,
                t_statistic=t_stat,
                p_value=p_value,
                is_significant=is_significant,
                winner=winner
            )
            results.append(test_result)

            # Store in DB
            await self.db.execute("""
                INSERT INTO model_pairwise_tests
                (model_a, model_b, metric, t_statistic, p_value, is_significant, winner, sample_size)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, model_a, model_b, metric, t_stat, p_value, is_significant, winner, len(data_a))

        return results

    def _calculate_ece(self, decisions: list) -> Decimal:
        """Calculate Expected Calibration Error."""
        bins = [[] for _ in range(10)]

        for d in decisions:
            conf = d.get("confidence", 0.5)
            bin_idx = min(int(conf * 10), 9)
            bins[bin_idx].append(d)

        ece = 0
        total = len(decisions)

        for i, bin_decisions in enumerate(bins):
            if not bin_decisions:
                continue

            expected_conf = (i + 0.5) / 10
            actual_accuracy = sum(
                1 for d in bin_decisions if d["outcome_correct"]
            ) / len(bin_decisions)

            ece += abs(expected_conf - actual_accuracy) * len(bin_decisions) / total

        return Decimal(str(ece))

    def _calculate_composite_score(
        self,
        accuracy: Decimal,
        profit: Decimal,
        calibration: Decimal,
        latency: float,
        cost: float
    ) -> Decimal:
        """Calculate weighted composite score."""
        # Normalize all to 0-1 range
        norm_accuracy = float(accuracy) / 100
        norm_profit = max(0, min(1, (float(profit) + 50) / 100))  # -50% to +50% → 0-1
        norm_calibration = float(calibration)
        norm_latency = min(1, latency)  # Already inverted
        norm_cost = min(1, cost)  # Already inverted

        composite = (
            norm_accuracy * self.COMPOSITE_WEIGHTS["accuracy"] +
            norm_profit * self.COMPOSITE_WEIGHTS["profitability"] +
            norm_calibration * self.COMPOSITE_WEIGHTS["calibration"] +
            norm_latency * self.COMPOSITE_WEIGHTS["latency"] +
            norm_cost * self.COMPOSITE_WEIGHTS["cost"]
        )

        return Decimal(str(composite))
```

### Configuration

```yaml
# config/model_comparison.yaml

model_comparison:
  enabled: true

  models:
    - id: gpt
      name: "GPT (latest)"
      provider: openai

    - id: grok
      name: "Grok (latest)"
      provider: xai

    - id: deepseek
      name: "DeepSeek V3"
      provider: deepseek

    - id: claude_sonnet
      name: "Claude Sonnet"
      provider: anthropic

    - id: claude_opus
      name: "Claude Opus"
      provider: anthropic

    - id: qwen
      name: "Qwen 2.5 7B"
      provider: ollama

  # Scoring weights
  composite_weights:
    accuracy: 0.35
    profitability: 0.25
    calibration: 0.20
    latency: 0.10
    cost: 0.10

  # Update frequencies
  updates:
    metrics_interval_seconds: 3600  # Hourly
    leaderboard_interval_seconds: 3600
    pairwise_tests_interval_seconds: 86400  # Daily

  # Significance testing
  significance:
    p_value_threshold: 0.05
    min_sample_size: 30

  # Outcome evaluation horizons
  horizons:
    - 1h
    - 4h
    - 24h
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| Metrics calculation | Correct accuracy/P&L | Matches manual calc |
| Composite score | Weighted correctly | Sum to 1.0 |
| Leaderboard ranking | Correct order | Highest score = rank 1 |
| Pairwise tests | Correct p-values | Verified with scipy |
| ECE calculation | Calibration error correct | Validated algorithm |

---

## 4.4 Dashboard

### Purpose

React-based web dashboard for monitoring system health, viewing positions, and managing trading operations.

### Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Frontend | React 18 + TypeScript | Modern, type-safe |
| Charting | Lightweight Charts (TradingView) | Professional trading charts |
| State | Zustand | Simple, performant |
| Real-time | WebSocket + React Query | Efficient data sync |
| Styling | Tailwind CSS | Rapid development |
| API Layer | FastAPI (existing) | Python consistency |

### Dashboard Views

Reference: [05-user-interface-requirements.md](../TripleGain-master-design/05-user-interface-requirements.md)

1. **Portfolio Summary** - Equity, P&L, allocation
2. **Price Charts** - Multi-symbol, multi-timeframe with indicators
3. **Open Positions** - Real-time position monitoring
4. **Agent Status** - Health and last outputs per agent
5. **Decision Log** - Audit trail of trading decisions
6. **Model Leaderboard** - 6-model comparison dashboard
7. **System Health** - Connections, data pipeline, risk state
8. **Manual Controls** - Pause, close positions, emergency stop

### API Endpoints for Dashboard

```yaml
# API Routes for Dashboard

# Real-time (WebSocket)
websocket:
  - path: /ws/portfolio
    description: Portfolio updates stream
    message_types:
      - portfolio_snapshot
      - position_update
      - trade_execution

  - path: /ws/prices
    description: Price updates stream
    message_types:
      - ticker_update
      - candle_update

  - path: /ws/agents
    description: Agent status stream
    message_types:
      - agent_output
      - agent_status_change

  - path: /ws/alerts
    description: System alerts stream
    message_types:
      - risk_alert
      - system_alert

# REST endpoints
rest:
  # Portfolio
  - path: GET /api/v1/portfolio/summary
    response: PortfolioSummary

  - path: GET /api/v1/portfolio/history
    params: [period]
    response: list[PortfolioSnapshot]

  # Positions
  - path: GET /api/v1/positions
    response: list[Position]

  - path: POST /api/v1/positions/{id}/close
    body: { reason: string }

  - path: PUT /api/v1/positions/{id}/modify
    body: { stop_loss?: number, take_profit?: number }

  # Agents
  - path: GET /api/v1/agents
    response: list[AgentStatus]

  - path: GET /api/v1/agents/{name}/outputs
    params: [limit]
    response: list[AgentOutput]

  # Decisions
  - path: GET /api/v1/decisions
    params: [limit, symbol, action]
    response: list[TradingDecision]

  - path: GET /api/v1/decisions/{id}
    response: TradingDecisionDetail

  # Models
  - path: GET /api/v1/models/leaderboard
    response: list[LeaderboardEntry]

  - path: GET /api/v1/models/{id}/metrics
    params: [period]
    response: ModelMetrics

  - path: GET /api/v1/models/comparisons
    params: [limit]
    response: list[ModelComparison]

  # System
  - path: GET /api/v1/system/health
    response: SystemHealth

  - path: GET /api/v1/system/risk-state
    response: RiskState

  # Controls
  - path: POST /api/v1/controls/pause
    response: { status: string }

  - path: POST /api/v1/controls/resume
    response: { status: string }

  - path: POST /api/v1/controls/close-all
    body: { confirm: string }
    response: { closed_count: number }

  - path: POST /api/v1/controls/emergency-stop
    body: { confirm: string }
    response: { status: string }

  # Configuration
  - path: GET /api/v1/config
    response: SystemConfig

  - path: PUT /api/v1/config
    body: ConfigUpdate
```

### Dashboard Project Structure

```
dashboard/
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── StatusBar.tsx
│   │   ├── portfolio/
│   │   │   ├── PortfolioSummary.tsx
│   │   │   ├── AllocationChart.tsx
│   │   │   └── HodlBagCard.tsx
│   │   ├── charts/
│   │   │   ├── PriceChart.tsx
│   │   │   ├── EquityCurve.tsx
│   │   │   └── IndicatorPanel.tsx
│   │   ├── positions/
│   │   │   ├── PositionTable.tsx
│   │   │   ├── PositionCard.tsx
│   │   │   └── ClosePositionDialog.tsx
│   │   ├── agents/
│   │   │   ├── AgentStatusGrid.tsx
│   │   │   ├── AgentCard.tsx
│   │   │   └── AgentOutputViewer.tsx
│   │   ├── decisions/
│   │   │   ├── DecisionLog.tsx
│   │   │   ├── DecisionCard.tsx
│   │   │   └── DecisionDetail.tsx
│   │   ├── models/
│   │   │   ├── Leaderboard.tsx
│   │   │   ├── ModelComparison.tsx
│   │   │   └── ConsensusView.tsx
│   │   ├── system/
│   │   │   ├── HealthDashboard.tsx
│   │   │   ├── RiskState.tsx
│   │   │   └── AlertPanel.tsx
│   │   └── controls/
│   │       ├── ControlPanel.tsx
│   │       ├── ConfirmDialog.tsx
│   │       └── EmergencyStop.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── usePortfolio.ts
│   │   ├── usePositions.ts
│   │   └── useAgents.ts
│   ├── stores/
│   │   ├── portfolioStore.ts
│   │   ├── positionStore.ts
│   │   └── agentStore.ts
│   ├── services/
│   │   ├── api.ts
│   │   └── websocket.ts
│   ├── types/
│   │   └── index.ts
│   └── App.tsx
├── public/
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── vite.config.ts
```

### Test Requirements

| Test | Description | Acceptance |
|------|-------------|------------|
| WebSocket connection | Connects and receives updates | Stable connection |
| REST API calls | All endpoints respond | Valid responses |
| Real-time updates | UI updates on WS message | < 100ms latency |
| Chart rendering | Price charts render | Smooth performance |
| Controls work | Pause/resume/close | Actions execute |

---

## Phase 4 Acceptance Criteria

### Functional Requirements

| Requirement | Test Method | Acceptance |
|-------------|-------------|------------|
| Sentiment aggregates correctly | Manual review | Reasonable sentiment |
| Hodl bags accumulate | Profit tracking | Correct allocation |
| Model comparison tracks all 6 | Leaderboard check | All models present |
| Dashboard displays data | Visual inspection | All views functional |

### Integration Requirements

| Requirement | Acceptance |
|-------------|------------|
| Sentiment → Trading | Trading agent receives sentiment |
| Trades → Hodl bags | Profits trigger accumulation |
| All agents → Dashboard | Status visible for all |
| Model decisions → Comparison | All decisions tracked |

### Deliverables Checklist

- [ ] `src/agents/sentiment_analysis.py`
- [ ] `src/execution/hodl_bag.py`
- [ ] `src/llm/model_comparison.py`
- [ ] `dashboard/` React application
- [ ] API routes for dashboard
- [ ] WebSocket handlers
- [ ] Database migrations
- [ ] Unit tests
- [ ] Integration tests

---

## References

- Design: [01-multi-agent-architecture.md](../TripleGain-master-design/01-multi-agent-architecture.md)
- Design: [02-llm-integration-system.md](../TripleGain-master-design/02-llm-integration-system.md)
- Design: [03-risk-management-rules-engine.md](../TripleGain-master-design/03-risk-management-rules-engine.md)
- Design: [05-user-interface-requirements.md](../TripleGain-master-design/05-user-interface-requirements.md)
- Design: [06-evaluation-framework.md](../TripleGain-master-design/06-evaluation-framework.md)

---

*Phase 4 Implementation Plan v1.0 - December 2025*
