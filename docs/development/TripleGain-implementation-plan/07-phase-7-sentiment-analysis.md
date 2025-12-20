# Phase 7: Sentiment Analysis Agent

**Phase Status**: Ready to Start
**Dependencies**: Phase 3 (Message Bus), Phase 6 (Paper Trading)
**Deliverable**: Dual-model sentiment agent providing market sentiment signals

---

## Overview

The Sentiment Analysis Agent gathers real-time market sentiment using Grok and GPT's native web search capabilities. It provides sentiment signals to the Trading Decision Agent to enhance trading decisions with social and news context.

### Why This Phase Matters

| Benefit | Description |
|---------|-------------|
| Market Context | Captures narratives and events not visible in price data |
| Risk Awareness | Identifies regulatory news, hacks, or FUD before price impact |
| Conviction Boost | Strong alignment between sentiment and technicals increases confidence |
| Contrarian Signals | Extreme sentiment can indicate reversal opportunities |

### Phase Dependencies

```
Phase 3 (Message Bus) ────→ Sentiment Agent publishes to sentiment_updates topic
Phase 6 (Paper Trading) ──→ Testing sentiment integration before live execution
Existing LLM Clients ─────→ xai_client.py (Grok), openai_client.py (GPT)
```

---

## 7.1 Architecture

### Dual-Model Design

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

### Model Assignments

| Property | Grok (xAI) | GPT (OpenAI) |
|----------|------------|--------------|
| Provider | xAI API | OpenAI API |
| Model | grok-2 | gpt-4-turbo |
| Invocation | Every 30 minutes | Every 30 minutes |
| Latency Target | < 15s | < 15s |
| Tier | Tier 2 (API) | Tier 2 (API) |
| Primary Strength | Social/Twitter | News/Analysis |
| Weighting (Social) | 60% | 40% |
| Weighting (News) | 40% | 60% |

### Integration Points

| Integration | Direction | Description |
|-------------|-----------|-------------|
| Message Bus | Publish | Publishes to `sentiment_updates` topic |
| Trading Decision Agent | Consumer | Receives sentiment for decision context |
| Coordinator | Consumer | Uses for conflict detection |
| Database | Write | Stores all sentiment outputs |
| Dashboard (Phase 10) | Read | Displays sentiment trends |

---

## 7.2 Data Structures

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
      },
      "maxItems": 5
    },
    "market_narratives": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Current market narratives being discussed",
      "maxItems": 3
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

### Internal Data Classes

| Class | Purpose | Fields |
|-------|---------|--------|
| `SentimentBias` | Enum for bias values | very_bullish, bullish, neutral, bearish, very_bearish |
| `KeyEvent` | Significant market event | event, impact, significance, source |
| `SentimentOutput` | Agent output | bias, confidence, scores, events, narratives, reasoning |
| `ProviderResult` | Single provider response | provider, bias, confidence, score, events, latency_ms |

---

## 7.3 Component Details

### 7.3.1 Sentiment Analysis Agent

**File**: `triplegain/src/agents/sentiment_analysis.py`

**Responsibilities**:
- Query Grok and GPT in parallel for sentiment
- Aggregate results with weighted scoring
- Handle provider failures gracefully
- Publish sentiment updates to message bus

**Key Methods**:

| Method | Purpose | Parameters | Returns |
|--------|---------|------------|---------|
| `process()` | Main analysis entry | symbol, include_twitter | SentimentOutput |
| `_query_grok()` | Query Grok with web/Twitter | asset, include_twitter | ProviderResult |
| `_query_gpt()` | Query GPT with web search | asset | ProviderResult |
| `_aggregate_results()` | Combine provider outputs | symbol, results | SentimentOutput |
| `_score_to_bias()` | Convert numeric to enum | score | SentimentBias |
| `_assess_fear_greed()` | Map to fear/greed scale | overall_score | str |
| `_deduplicate_events()` | Remove similar events | events | list[KeyEvent] |

**Error Handling**:

| Scenario | Behavior |
|----------|----------|
| Grok fails, GPT succeeds | Use GPT result only |
| GPT fails, Grok succeeds | Use Grok result only |
| Both fail | Return neutral sentiment with low confidence |
| Timeout (>30s) | Return cached result if available |
| Rate limit | Exponential backoff, skip cycle if needed |

### 7.3.2 Prompt Templates

**Grok Prompt Template** (Social + News focus):

```
Analyze current market sentiment for {asset} cryptocurrency.

WEB SEARCH:
Search for recent news and analysis about {asset}:
- Breaking news (last 24 hours)
- Price analysis articles
- Regulatory developments
- Institutional adoption news
- Technical developments

TWITTER/X ANALYSIS:
Search for recent tweets about {asset} from:
- Major crypto influencers
- Project official accounts
- Institutional traders
- Sentiment of replies and engagement

Provide a comprehensive sentiment analysis with:
1. Overall bias (very_bullish/bullish/neutral/bearish/very_bearish)
2. Confidence level (0-1)
3. Key events with impact assessment
4. Current market narratives

Return JSON format.
```

**GPT Prompt Template** (News focus):

```
Analyze current market sentiment for {asset} cryptocurrency.

WEB SEARCH:
Search for recent news and analysis about {asset}:
- Breaking news from major crypto outlets
- Regulatory news and developments
- Institutional investment news
- Market analysis and predictions
- Technical and fundamental analysis articles

Focus on factual news rather than social media sentiment.
Provide a comprehensive sentiment analysis with:
1. Overall bias (very_bullish/bullish/neutral/bearish/very_bearish)
2. Confidence level (0-1)
3. Key events with impact assessment
4. Current market narratives

Return JSON format.
```

### 7.3.3 Aggregation Logic

**Weighting Strategy**:

| Source | Social Score Weight | News Score Weight |
|--------|---------------------|-------------------|
| Grok | 60% | 40% |
| GPT | 40% | 60% |

**Confidence Calculation**:
- Base confidence = average of provider confidences
- Boost +10% when providers agree on bias
- Reduce by variance when providers disagree
- Minimum confidence = 30%

**Score Combination**:

```
social_score = (grok_score × 0.6) + (gpt_score × 0.4)
news_score = (gpt_score × 0.6) + (grok_score × 0.4)
overall_score = (social_score + news_score) / 2
```

---

## 7.4 Configuration

**File**: `config/agents.yaml` (sentiment_analysis section)

```yaml
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
        retry:
          max_attempts: 2
          backoff_ms: 5000

      gpt:
        enabled: true
        model: gpt-4-turbo
        timeout_ms: 30000
        capabilities:
          - web_search
        weight:
          social: 0.4
          news: 0.6
        retry:
          max_attempts: 2
          backoff_ms: 5000

    aggregation:
      min_providers: 1  # At least one must succeed
      confidence_boost_on_agreement: 0.1
      min_confidence: 0.3

    output:
      store_all: true
      cache_ttl_seconds: 1750  # Just under 30 min
      max_events: 5
      max_narratives: 3

    # Integration with other agents
    integration:
      publish_topic: sentiment_updates
      ttl_seconds: 1800
```

---

## 7.5 Database Schema

### New Tables

```sql
-- Sentiment analysis outputs
CREATE TABLE sentiment_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,

    -- Sentiment values
    bias VARCHAR(20) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    social_score DECIMAL(5, 4),
    news_score DECIMAL(5, 4),
    overall_score DECIMAL(5, 4),
    fear_greed VARCHAR(20),

    -- Provider details
    grok_available BOOLEAN DEFAULT FALSE,
    gpt_available BOOLEAN DEFAULT FALSE,
    providers_agreed BOOLEAN,

    -- Extended data (JSON for flexibility)
    key_events JSONB,
    market_narratives JSONB,
    reasoning TEXT,

    -- Metadata
    total_latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_sentiment_outputs_symbol_ts
    ON sentiment_outputs (symbol, timestamp DESC);

CREATE INDEX idx_sentiment_outputs_bias
    ON sentiment_outputs (bias, timestamp DESC);

-- Provider response tracking (for debugging/analysis)
CREATE TABLE sentiment_provider_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sentiment_output_id UUID REFERENCES sentiment_outputs(id),
    provider VARCHAR(20) NOT NULL,
    model VARCHAR(50) NOT NULL,

    -- Response data
    raw_response JSONB,
    parsed_bias VARCHAR(20),
    parsed_score DECIMAL(5, 4),
    parsed_confidence DECIMAL(5, 4),

    -- Performance
    latency_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,

    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_provider_responses_output
    ON sentiment_provider_responses (sentiment_output_id);
```

### Migration File

**File**: `migrations/007_sentiment_analysis.sql`

---

## 7.6 API Endpoints

### New Routes

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| GET | `/api/v1/sentiment/{symbol}` | Latest sentiment for symbol | SentimentOutput |
| GET | `/api/v1/sentiment/{symbol}/history` | Historical sentiment | list[SentimentOutput] |
| POST | `/api/v1/sentiment/{symbol}/refresh` | Force sentiment refresh | SentimentOutput |
| GET | `/api/v1/sentiment/all` | Latest for all symbols | dict[symbol, SentimentOutput] |

### Route Implementation

**File**: `triplegain/src/api/routes_sentiment.py`

---

## 7.7 Integration with Trading Decision

### Message Bus Integration

The Sentiment Agent publishes to `sentiment_updates` topic:

```python
# In coordinator.py - add to scheduled tasks
ScheduledTask(
    name="sentiment_analysis",
    agent="sentiment_analysis",
    interval_seconds=1800,  # 30 minutes
    symbols=self.config["symbols"],
    handler=self._run_sentiment_agent
)
```

### Trading Decision Enhancement

Modify Trading Decision Agent to consume sentiment:

```python
# In trading_decision.py - process method
async def process(self, snapshot, ta_output, regime_output, sentiment_output=None):
    """
    Generate trading decision with optional sentiment context.
    """
    context = {
        "technical_analysis": ta_output,
        "regime": regime_output,
    }

    if sentiment_output:
        context["sentiment"] = {
            "bias": sentiment_output.bias.value,
            "confidence": sentiment_output.confidence,
            "key_events": [e.__dict__ for e in sentiment_output.key_events],
            "fear_greed": sentiment_output.fear_greed
        }

    # Include in prompt building
    prompt = self.prompt_builder.build_prompt(
        agent_name=self.agent_name,
        snapshot=snapshot,
        additional_context=context
    )
```

---

## 7.8 Test Requirements

### Unit Tests

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_grok_query` | Grok web search returns results | Valid response structure |
| `test_gpt_query` | GPT web search returns results | Valid response structure |
| `test_aggregation_both_succeed` | Combine two provider results | Correct weighting applied |
| `test_aggregation_single_provider` | One provider fails | Other provider used alone |
| `test_aggregation_both_fail` | Both providers fail | Neutral sentiment returned |
| `test_score_to_bias` | Numeric to enum conversion | Correct bias for ranges |
| `test_fear_greed_assessment` | Score to fear/greed | Correct category |
| `test_event_deduplication` | Similar events merged | No duplicates |
| `test_confidence_boost` | Providers agree | Confidence increased |
| `test_confidence_variance` | Providers disagree | Confidence reduced |
| `test_output_schema_validation` | Output matches schema | Valid JSON schema |
| `test_timeout_handling` | Provider times out | Graceful fallback |

### Integration Tests

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_message_bus_publish` | Sentiment published | Topic receives message |
| `test_coordinator_scheduling` | Agent runs on schedule | Correct interval |
| `test_trading_agent_receives` | Trading agent gets sentiment | Context includes sentiment |
| `test_database_storage` | Outputs persisted | Queryable from DB |
| `test_api_endpoint` | REST endpoint works | Valid response |

---

## 7.9 Deliverables Checklist

- [ ] `triplegain/src/agents/sentiment_analysis.py` - Agent implementation
- [ ] `triplegain/src/api/routes_sentiment.py` - API endpoints
- [ ] `config/agents.yaml` - Configuration updates
- [ ] `migrations/007_sentiment_analysis.sql` - Database migration
- [ ] `triplegain/tests/unit/agents/test_sentiment_analysis.py` - Unit tests
- [ ] `triplegain/tests/integration/test_sentiment_integration.py` - Integration tests
- [ ] Update `coordinator.py` to schedule sentiment agent
- [ ] Update `trading_decision.py` to consume sentiment
- [ ] Update prompt templates for Trading Decision Agent

---

## 7.10 Cost Estimation

### API Costs (Daily)

| Provider | Calls/Day | Tokens/Call | Cost/Day |
|----------|-----------|-------------|----------|
| Grok | 48 (30 min × 2 symbols) | ~2,000 | ~$0.20 |
| GPT-4-turbo | 48 | ~2,000 | ~$0.50 |
| **Total** | 96 | - | **~$0.70** |

### Within Budget

Total API budget: $5/day
Sentiment usage: ~$0.70/day (14% of budget)
Remaining for Trading + Coordinator: $4.30/day

---

## 7.11 Acceptance Criteria

### Functional Requirements

| Requirement | Test Method | Acceptance |
|-------------|-------------|------------|
| Grok queries web successfully | Manual test | Valid results |
| GPT queries web successfully | Manual test | Valid results |
| Aggregation produces valid output | Unit test | Schema validates |
| Publishes to message bus | Integration test | Message received |
| Trading agent uses sentiment | Integration test | Context includes sentiment |

### Non-Functional Requirements

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| Latency (total) | < 20s | Log timestamps |
| Success rate | > 95% | Error monitoring |
| Cache hit rate | > 80% | Cache metrics |
| Cost per analysis | < $0.02 | API cost tracking |

---

## References

- Design: [02-llm-integration-system.md](../TripleGain-master-design/02-llm-integration-system.md)
- Design: [01-multi-agent-architecture.md](../TripleGain-master-design/01-multi-agent-architecture.md)
- Existing: [xai_client.py](../../../triplegain/src/llm/clients/xai_client.py)
- Existing: [openai_client.py](../../../triplegain/src/llm/clients/openai_client.py)
- Phase 3: [03-phase-3-orchestration.md](./03-phase-3-orchestration.md)

---

*Phase 7 Implementation Plan v1.0 - December 2025*
