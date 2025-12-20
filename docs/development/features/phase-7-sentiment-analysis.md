# Phase 7: Sentiment Analysis Agent

**Version**: 0.5.2
**Status**: Complete
**Last Updated**: 2025-12-20

## Overview

The Sentiment Analysis Agent provides real-time market sentiment by combining social media analysis (via Grok/xAI) with news analysis (via OpenAI GPT with web search). This dual-model architecture enables comprehensive sentiment coverage for trading decisions.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │     Sentiment Analysis Agent        │
                    │   (triplegain/src/agents/          │
                    │    sentiment_analysis.py)           │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   Grok (xAI)    │     │  GPT (OpenAI)   │     │  Message Bus    │
    │  Social/Twitter │     │  Web Search     │     │   Publisher     │
    │  Real-time      │     │  Real-time News │     └─────────────────┘
    └─────────────────┘     └─────────────────┘
              │                       │
              └───────────┬───────────┘
                          ▼
              ┌─────────────────────┐
              │ Weighted Aggregation │
              │ social*0.6 + news*0.6│
              └─────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Trading  │ │Coordinator│ │ Database │
        │ Decision │ │  Agent   │ │ Storage  │
        └──────────┘ └──────────┘ └──────────┘
```

## Key Components

### 1. SentimentAnalysisAgent

**File**: `triplegain/src/agents/sentiment_analysis.py`
**Lines**: ~1,300

The main agent that orchestrates sentiment analysis across multiple providers.

```python
class SentimentAnalysisAgent(BaseAgent):
    """Dual-model sentiment agent using Grok + GPT with web search."""

    async def process(self, symbol: str, include_twitter: bool = True) -> SentimentOutput:
        """Analyze sentiment for a cryptocurrency symbol."""
```

### 2. CircuitBreakerState

Protects against cascading failures from provider outages.

```python
@dataclass
class CircuitBreakerState:
    consecutive_failures: int = 0
    is_open: bool = False
    half_open_attempts: int = 0
    max_half_open_attempts: int = 3

    def get_state(self) -> str:  # "CLOSED", "OPEN", "HALF_OPEN"
```

**Behavior**:
- Opens after 3 consecutive failures
- Cooldown period: 5 minutes
- Half-open state: Allows 3 test requests before re-opening

### 3. RateLimiter

Prevents API abuse with per-user rate limiting.

```python
class RateLimiter:
    def __init__(
        self,
        max_requests: int = 5,
        window_seconds: int = 60,
        cleanup_interval: int = 300,  # Auto-cleanup every 5 min
    ):
```

**Memory Management**: Periodic `cleanup_old_users()` prevents memory leaks from inactive users.

### 4. OpenAI Web Search

GPT now uses `gpt-4o-search-preview` for real-time news search.

```python
# openai_client.py
async def generate_with_search(
    self,
    model: str,
    system_prompt: str,
    user_message: str,
    search_context_size: str = "medium",  # low, medium, high
) -> LLMResponse:
```

## Configuration

### agents.yaml

```yaml
sentiment_analysis:
  enabled: true
  tier: tier2_api

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
        social: 0.6  # USED: Grok's social sentiment weight
        news: 0.4    # NOT USED: Reserved for future

    gpt:
      enabled: true
      model: gpt-4o-search-preview
      timeout_ms: 30000
      capabilities:
        - web_search
      web_search_options:
        search_context_size: medium
      weight:
        social: 0.4  # NOT USED: Reserved for future
        news: 0.6    # USED: GPT's news sentiment weight

  circuit_breaker:
    failure_threshold: 3
    cooldown_seconds: 300

  rate_limit:
    refresh_rpm: 5
```

## Output Schema

```python
@dataclass
class SentimentOutput(AgentOutput):
    symbol: str
    bias: SentimentBias  # very_bullish, bullish, neutral, bearish, very_bearish
    social_score: float  # -1.0 to 1.0 (from Grok)
    news_score: float    # -1.0 to 1.0 (from GPT)
    overall_score: float # Weighted average
    social_analysis: str  # Grok's reasoning
    news_analysis: str    # GPT's reasoning
    fear_greed: FearGreedLevel
    key_events: list[KeyEvent]  # Up to 5 significant events
    market_narratives: list[str]  # Up to 3 narratives
    grok_available: bool
    gpt_available: bool
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/sentiment/{symbol}` | Latest sentiment for symbol |
| GET | `/api/v1/sentiment/{symbol}/history` | Historical sentiment |
| POST | `/api/v1/sentiment/{symbol}/refresh` | Force refresh (rate limited) |
| GET | `/api/v1/sentiment/all` | All symbols' latest sentiment |
| GET | `/api/v1/sentiment/stats` | Provider statistics |

## Weighted Aggregation Formula

```python
# When both providers succeed:
total_weight = grok_social_weight + gpt_news_weight  # 0.6 + 0.6 = 1.2
overall_score = (
    (social_score * grok_social_weight) +
    (news_score * gpt_news_weight)
) / total_weight

# Example: social=0.5, news=0.4
# Result: (0.5*0.6 + 0.4*0.6) / 1.2 = 0.45
```

## Integration with Other Agents

### Trading Decision Agent

Receives sentiment context including scores and full analysis:

```python
# Passed to Trading Decision LLM prompt:
sentiment_context = {
    "bias": "bullish",
    "overall_score": 0.45,
    "social_analysis": "Whale accumulation spotted, bullish momentum",
    "news_analysis": "ETF inflows continue, institutional adoption growing",
    "key_events": [...]
}
```

### Coordinator Agent

Detects conflicts between sentiment and technical analysis:

```python
# Example conflict detection:
if ta_signal == "SELL" and sentiment_bias in ["bullish", "very_bullish"]:
    # Conflict detected, Coordinator resolves
```

## Testing

### Unit Tests (56 tests)

```bash
pytest triplegain/tests/unit/agents/test_sentiment_analysis.py -v
```

Coverage includes:
- SentimentBias/FearGreedLevel enums (10 tests)
- KeyEvent/ProviderResult dataclasses (5 tests)
- SentimentOutput validation/serialization (7 tests)
- Response parsing and normalization (4 tests)
- Aggregation logic (3 tests)
- CircuitBreakerState (9 tests)
- RateLimiter (8 tests)
- Weighted aggregation (2 tests)

### Integration Tests (11 tests)

```bash
pytest triplegain/tests/integration/test_sentiment_integration.py -v
```

Coverage includes:
- Message bus publish/subscribe
- Coordinator receives sentiment
- Trading agent context
- API endpoint integration
- Database storage
- End-to-end flow

## Database Schema

**Migration**: `008_sentiment_analysis_fixes.sql`

```sql
CREATE TABLE sentiment_outputs (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    bias TEXT NOT NULL,
    social_score DECIMAL(4,3),
    news_score DECIMAL(4,3),
    overall_score DECIMAL(4,3),
    social_analysis TEXT,
    news_analysis TEXT,
    confidence DECIMAL(4,3),
    fear_greed TEXT,
    key_events JSONB,
    market_narratives JSONB,
    grok_available BOOLEAN,
    gpt_available BOOLEAN
);

CREATE TABLE sentiment_provider_responses (
    id UUID PRIMARY KEY,
    sentiment_output_id UUID REFERENCES sentiment_outputs(id),
    provider TEXT NOT NULL,
    model TEXT,
    bias TEXT,
    score DECIMAL(4,3),
    latency_ms INTEGER,
    cost_usd DECIMAL(8,6),
    success BOOLEAN
);
```

## Error Handling

1. **Provider Timeout**: Wrapped in `asyncio.wait_for()` (30s default)
2. **Provider Failure**: Circuit breaker opens after 3 consecutive failures
3. **Both Providers Fail**: Returns neutral sentiment with minimum confidence (0.3)
4. **Rate Limit Exceeded**: Returns HTTP 429 with `Retry-After` header
5. **JSON Parse Error**: Falls back to neutral with low confidence

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.5.0 | 2025-12-19 | Initial implementation (dual-model, API, database) |
| 0.5.1 | 2025-12-19 | Circuit breaker, rate limiting, weighted aggregation |
| 0.5.2 | 2025-12-20 | OpenAI web search, memory cleanup, per-attempt latency |

## References

- [Phase 7 Deep Review](../reviews/phase-7/deep-review.md)
- [Phase 7 Deep Review v2](../reviews/phase-7/deep-review-v2.md)
- [OpenAI Web Search Guide](https://platform.openai.com/docs/guides/tools-web-search)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
