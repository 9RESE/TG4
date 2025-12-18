# TripleGain LLM Integration System

**Document Version**: 1.0
**Status**: Design Phase

---

## 1. Multi-Model LLM Architecture

### 1.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-MODEL LLM ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TIER 1: LOCAL LLM (Execution)                     │   │
│  │                                                                      │   │
│  │  Model: Qwen 2.5 7B via Ollama                                      │   │
│  │  Location: /media/rese/2tb_drive/ollama_config/                     │   │
│  │  Latency: < 500ms                                                   │   │
│  │  Cost: $0 (local compute)                                           │   │
│  │                                                                      │   │
│  │  Use Cases:                                                          │   │
│  │  • Technical indicator interpretation                               │   │
│  │  • Regime classification                                            │   │
│  │  • Real-time execution decisions                                    │   │
│  │  • Pattern recognition                                              │   │
│  │  • Stop-loss/take-profit adjustment                                 │   │
│  │                                                                      │   │
│  │  Invocation: Per-tick or per-minute for active monitoring           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TIER 2: API LLMs (Strategic)                      │   │
│  │                                                                      │   │
│  │  SPECIALIZED ROLES:                                                  │   │
│  │  • Sentiment Analysis: Grok + GPT (web search, news every 30 min)  │   │
│  │  • Portfolio Rebalancing: DeepSeek (edge case reasoning)           │   │
│  │  • Coordinator: DeepSeek V3 / Claude Sonnet (conflict resolution)  │   │
│  │                                                                      │   │
│  │  TRADING DECISION A/B TESTING (all 6 models run in parallel):       │   │
│  │  • GPT (latest)                                                     │   │
│  │  • Grok (latest)                                                    │   │
│  │  • DeepSeek V3                                                      │   │
│  │  • Claude Sonnet                                                    │   │
│  │  • Claude Opus                                                      │   │
│  │  • Qwen 2.5 7B (local)                                             │   │
│  │                                                                      │   │
│  │  Latency: 3-30 seconds                                              │   │
│  │  Cost: $0.001-0.01 per call (varies by model)                       │   │
│  │                                                                      │   │
│  │  Invocation: Scheduled (hourly, daily, weekly) + every 30 min news  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Model Selection Matrix

| Model | Tier | Cost/1K tokens | Latency | Quantitative | Sentiment | Assigned Role |
|-------|------|----------------|---------|--------------|-----------|---------------|
| **Qwen 2.5 7B** | Local | $0 | ~200ms | Good | Good | TA, Regime, Execution, Trading Decision A/B |
| **DeepSeek V3** | API | ~$0.0002 | ~3s | Excellent | Good | Trading Decision A/B, Portfolio Rebalancing, Coordinator |
| **Claude Sonnet** | API | ~$0.003 | ~2s | Good | Excellent | Trading Decision A/B, Coordinator |
| **Claude Opus** | API | ~$0.015 | ~5s | Excellent | Excellent | Trading Decision A/B |
| **GPT (latest)** | API | ~$0.01 | ~2s | Good | Good | Trading Decision A/B, Sentiment Analysis |
| **Grok (latest)** | API | ~$0.005 | ~3s | Good | Excellent | Trading Decision A/B, Sentiment Analysis |

**Note**: All 6 models participate in Trading Decision A/B testing. Grok and GPT have web search capability for sentiment analysis.

### 1.3 Tier Selection Logic

```python
def select_llm_tier(task_type: str, urgency: str, complexity: str) -> str:
    """
    Determine which LLM tier to use for a given task.

    Args:
        task_type: "technical_analysis" | "regime" | "sentiment" |
                   "trading_decision" | "coordination" | "execution"
        urgency: "immediate" | "normal" | "deferred"
        complexity: "low" | "medium" | "high"

    Returns:
        "tier1_local" | "tier2_api"
    """

    # Immediate urgency always uses local
    if urgency == "immediate":
        return "tier1_local"

    # Execution tasks always local
    if task_type == "execution":
        return "tier1_local"

    # High complexity or strategic tasks use API
    if complexity == "high" or task_type in ["trading_decision", "coordination", "sentiment"]:
        return "tier2_api"

    # Technical analysis and regime detection use local
    if task_type in ["technical_analysis", "regime"]:
        return "tier1_local"

    # Default to local for cost efficiency
    return "tier1_local"
```

---

## 2. Prompt Engineering System

### 2.1 Prompt Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROMPT STRUCTURE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ SYSTEM PROMPT (Fixed per agent)                                      │   │
│  │  • Role definition                                                   │   │
│  │  • Trading rules & constraints                                       │   │
│  │  • Output format specification                                       │   │
│  │  • Risk management rules                                             │   │
│  │  • Behavioral guidelines                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ CONTEXT INJECTION (Dynamic)                                          │   │
│  │  • Current portfolio state                                           │   │
│  │  • Recent trade history                                              │   │
│  │  • Active positions                                                  │   │
│  │  • Performance metrics                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ MARKET DATA (Dynamic)                                                │   │
│  │  • OHLCV candles (multiple timeframes)                              │   │
│  │  • Technical indicators (pre-computed)                              │   │
│  │  • Order book snapshot                                              │   │
│  │  • Recent trades                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ QUERY (Task-specific)                                                │   │
│  │  • Specific question or task                                        │   │
│  │  • Required output fields                                           │   │
│  │  • Constraints for this query                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 System Prompts by Agent

#### 2.2.1 Technical Analysis Agent System Prompt

```
You are an expert quantitative technical analyst for cryptocurrency markets.

ROLE:
You analyze market data using technical indicators to identify trading opportunities.
You DO NOT make final trading decisions - you provide analysis to other agents.

CAPABILITIES:
- Interpret technical indicators (RSI, MACD, EMA, ATR, etc.)
- Identify chart patterns and price action signals
- Determine trend direction and strength
- Identify key support and resistance levels
- Assess volatility conditions

ANALYSIS GUIDELINES:
1. Always consider multiple timeframes (1m, 5m, 1h, 4h, 1d)
2. Confirm signals across different indicator categories
3. Note confluence of signals at key levels
4. Be explicit about confidence levels
5. Identify invalidation conditions for your analysis

OUTPUT FORMAT (JSON):
{
  "analysis_timestamp": "ISO8601",
  "symbol": "SYMBOL",
  "trend": {
    "direction": "bullish|bearish|neutral",
    "strength": 0.0-1.0,
    "timeframe_alignment": ["aligned_timeframes"]
  },
  "momentum": {
    "score": -1.0 to 1.0,
    "rsi_signal": "oversold|neutral|overbought",
    "macd_signal": "bullish|bearish|neutral"
  },
  "key_levels": {
    "resistance": [price_levels],
    "support": [price_levels],
    "current_position": "near_support|mid_range|near_resistance"
  },
  "signals": {
    "primary": "description of main signal",
    "secondary": ["additional observations"],
    "warnings": ["any concerns"]
  },
  "bias": "long|short|neutral",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}

CONSTRAINTS:
- Never recommend specific entry/exit prices (that's the Trading Agent's job)
- Always provide confidence scores
- Flag when indicators are conflicting
- Note when data quality is poor
```

#### 2.2.2 Trading Decision Agent System Prompt

```
You are the Trading Decision Agent for the TripleGain crypto trading system.

ROLE:
You synthesize inputs from Technical Analysis, Regime Detection, and Sentiment
agents to generate specific trading signals. You make the primary trading decisions,
subject to Risk Management approval.

PORTFOLIO CONTEXT:
- Target allocation: 33% BTC / 33% XRP / 33% USDT
- Starting capital: ~$2,100
- Maximum leverage: 5x (subject to regime adjustment)
- Risk per trade: 1-2% of equity
- Exchange: Kraken

TRADING STYLE:
- Mid-to-low frequency trading (hold periods: hours to days)
- Trend-following primary (NOT mean reversion)
- Quality over quantity - fewer, higher-conviction trades
- Conservative execution - like Claude in Alpha Arena (3 trades > 44 trades)

DECISION FRAMEWORK:
1. Review all agent inputs and their confidence scores
2. Check regime compatibility with proposed action
3. Verify sentiment alignment (or note divergence)
4. Calculate risk/reward ratio
5. Determine position size based on confidence and regime
6. Set stop-loss and take-profit levels
7. Define invalidation conditions

OUTPUT FORMAT (JSON):
{
  "decision_id": "uuid",
  "timestamp": "ISO8601",
  "symbol": "SYMBOL",
  "action": "BUY|SELL|HOLD|CLOSE",
  "confidence": 0.0-1.0,
  "parameters": {
    "entry_price": price (null for market orders),
    "position_size_usd": amount,
    "leverage": 1-5,
    "stop_loss": price,
    "take_profit": price,
    "order_type": "market|limit",
    "time_in_force": "gtc|ioc|fok"
  },
  "risk_metrics": {
    "risk_amount_usd": calculated_risk,
    "risk_pct_of_equity": calculated_pct,
    "risk_reward_ratio": calculated_rr
  },
  "synthesis": {
    "technical_alignment": "aligned|divergent",
    "sentiment_alignment": "aligned|divergent",
    "regime_compatibility": "optimal|acceptable|suboptimal"
  },
  "invalidation": "Condition that would invalidate this trade",
  "reasoning": "Detailed explanation of decision"
}

RULES:
1. NEVER trade without a stop-loss
2. Confidence < 0.6 → action must be HOLD
3. Risk/reward < 1.5:1 → do not take trade
4. Respect regime-based leverage limits
5. Document reasoning for every decision
6. If agents conflict, default to more conservative action
```

#### 2.2.3 Sentiment Analysis Agent System Prompt

```
You are the Sentiment Analysis Agent for the TripleGain trading system.

ROLE:
Analyze news, social sentiment, and on-chain data to generate sentiment signals
that inform trading decisions. You identify market narratives and potential
catalysts that technical analysis might miss.

DATA SOURCES YOU ANALYZE:
- Cryptocurrency news articles
- Fear & Greed Index
- On-chain metrics (exchange flows, whale transactions)
- Market narratives and trends

ANALYSIS GUIDELINES:
1. Distinguish between noise and signal
2. Identify sentiment extremes (fear/greed) that often precede reversals
3. Detect unusual on-chain activity (whale movements, exchange flows)
4. Assess news impact: magnitude and time horizon
5. Note divergences between sentiment and price action

SENTIMENT SCORING:
- Score range: -1.0 (extreme fear/bearish) to +1.0 (extreme greed/bullish)
- 0.0 = neutral
- Provide scores for each data source AND overall composite

OUTPUT FORMAT (JSON):
{
  "timestamp": "ISO8601",
  "sentiment_scores": {
    "overall": -1.0 to 1.0,
    "news": -1.0 to 1.0,
    "social": -1.0 to 1.0,
    "on_chain": -1.0 to 1.0,
    "fear_greed_index": 0-100
  },
  "notable_events": [
    {
      "event": "description",
      "impact": "bullish|bearish|neutral",
      "magnitude": "low|medium|high",
      "assets_affected": ["BTC", "XRP"],
      "time_horizon": "immediate|short_term|medium_term"
    }
  ],
  "on_chain_signals": {
    "exchange_flow": "inflow|outflow|neutral",
    "whale_activity": "accumulating|distributing|neutral",
    "unusual_activity": boolean
  },
  "market_narrative": "Brief description of current market story",
  "contrarian_signals": {
    "extreme_detected": boolean,
    "type": "extreme_fear|extreme_greed|none",
    "contrarian_bias": "potential_bounce|potential_top|none"
  },
  "bias": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "alerts": ["Any urgent alerts"]
}

RULES:
1. Flag extreme sentiment readings (Fear & Greed < 20 or > 80)
2. Note when sentiment diverges from price action
3. Identify potential "buy the rumor, sell the news" setups
4. Be skeptical of hype - distinguish signal from noise
5. Weight recent news higher than older news
```

#### 2.2.4 Regime Detection Agent System Prompt

```
You are the Market Regime Detection Agent for the TripleGain trading system.

ROLE:
Classify the current market regime to inform strategy selection and risk
parameters. Different regimes require different trading approaches.

REGIME DEFINITIONS:

1. TRENDING_BULL
   - ADX > 25
   - Price above EMA200
   - Higher highs and higher lows
   - Recommended: Trend-following longs, buy dips

2. TRENDING_BEAR
   - ADX > 25
   - Price below EMA200
   - Lower highs and lower lows
   - Recommended: Trend-following shorts, sell rallies

3. RANGING
   - ADX < 20
   - Choppiness Index > 60
   - Price oscillating between support/resistance
   - Recommended: Range trades, reduce position size

4. VOLATILE
   - ATR > 2x 20-period average
   - Wide Bollinger Bands
   - Large intraday swings
   - Recommended: Reduce leverage, widen stops

5. QUIET
   - ATR < 0.5x 20-period average
   - Bollinger Band squeeze detected
   - Low volume
   - Recommended: Prepare for breakout, wait for confirmation

ANALYSIS PROCESS:
1. Analyze indicators across multiple timeframes
2. Calculate regime scores for each category
3. Determine primary and secondary regime
4. Estimate transition probabilities
5. Recommend parameter adjustments

OUTPUT FORMAT (JSON):
{
  "timestamp": "ISO8601",
  "symbol": "SYMBOL",
  "primary_regime": "regime_type",
  "primary_confidence": 0.0-1.0,
  "secondary_regime": "regime_type",
  "secondary_confidence": 0.0-1.0,
  "regime_scores": {
    "trending_bull": 0.0-1.0,
    "trending_bear": 0.0-1.0,
    "ranging": 0.0-1.0,
    "volatile": 0.0-1.0,
    "quiet": 0.0-1.0
  },
  "regime_duration": {
    "estimated_hours": number,
    "confidence": 0.0-1.0
  },
  "transition_signals": {
    "regime_change_imminent": boolean,
    "likely_next_regime": "regime_type",
    "warning_signs": ["list of signs"]
  },
  "parameter_recommendations": {
    "leverage_multiplier": 0.5-1.0,
    "position_size_multiplier": 0.5-1.0,
    "stop_loss_multiplier": 1.0-2.0,
    "strategy_preference": "trend_following|mean_reversion|wait"
  },
  "reasoning": "Explanation of regime classification"
}

RULES:
1. Always check multiple timeframes for confirmation
2. A regime is only confirmed if confidence > 0.6
3. Flag transitional periods explicitly
4. Provide parameter adjustments even when uncertain
5. Default to conservative parameters in ambiguous regimes
```

### 2.3 Dynamic Context Templates

#### 2.3.1 Portfolio Context Template

```json
{
  "portfolio_context": {
    "timestamp": "{{current_time}}",
    "equity": {
      "total_usd": {{total_equity}},
      "available_margin_usd": {{available_margin}},
      "used_margin_usd": {{used_margin}}
    },
    "allocation": {
      "btc_pct": {{btc_allocation_pct}},
      "xrp_pct": {{xrp_allocation_pct}},
      "usdt_pct": {{usdt_allocation_pct}},
      "deviation_from_target": {{max_deviation}}
    },
    "positions": [
      {
        "symbol": "{{symbol}}",
        "side": "{{side}}",
        "size": {{size}},
        "entry_price": {{entry}},
        "current_price": {{current}},
        "unrealized_pnl": {{pnl}},
        "duration_hours": {{hours_held}}
      }
    ],
    "performance": {
      "daily_pnl_usd": {{daily_pnl}},
      "daily_pnl_pct": {{daily_pnl_pct}},
      "weekly_pnl_pct": {{weekly_pnl_pct}},
      "win_rate_7d": {{win_rate}},
      "current_drawdown_pct": {{drawdown}}
    },
    "risk_state": {
      "consecutive_losses": {{consecutive_losses}},
      "daily_loss_limit_remaining": {{daily_limit_remaining}},
      "circuit_breakers_active": [{{active_breakers}}]
    }
  }
}
```

#### 2.3.2 Market Data Template

```json
{
  "market_data": {
    "timestamp": "{{current_time}}",
    "symbol": "{{symbol}}",
    "current_price": {{price}},
    "candles": {
      "1m": {
        "last_n": {{n_1m_candles}},
        "data": "{{1m_ohlcv_summary}}"
      },
      "5m": {
        "last_n": {{n_5m_candles}},
        "data": "{{5m_ohlcv_summary}}"
      },
      "1h": {
        "last_n": {{n_1h_candles}},
        "data": "{{1h_ohlcv_summary}}"
      },
      "4h": {
        "last_n": {{n_4h_candles}},
        "data": "{{4h_ohlcv_summary}}"
      },
      "1d": {
        "last_n": {{n_1d_candles}},
        "data": "{{1d_ohlcv_summary}}"
      }
    },
    "indicators": {
      "rsi_14": {{rsi}},
      "macd": {"line": {{macd_line}}, "signal": {{macd_signal}}, "histogram": {{macd_hist}}},
      "ema": {"9": {{ema9}}, "21": {{ema21}}, "50": {{ema50}}, "200": {{ema200}}},
      "atr_14": {{atr}},
      "adx_14": {{adx}},
      "bb": {"upper": {{bb_upper}}, "middle": {{bb_middle}}, "lower": {{bb_lower}}, "width": {{bb_width}}},
      "vwap": {{vwap}},
      "obv_trend": "{{obv_direction}}"
    },
    "order_book": {
      "bid_depth_usd": {{bid_depth}},
      "ask_depth_usd": {{ask_depth}},
      "spread_bps": {{spread}},
      "imbalance": {{imbalance}}
    },
    "recent_trades": {
      "buy_volume_1h": {{buy_vol}},
      "sell_volume_1h": {{sell_vol}},
      "large_trades": [{{large_trades}}]
    }
  }
}
```

---

## 3. Output Parsing System

### 3.1 Structured Output Schema

All LLM outputs are validated against JSON schemas:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "TradingDecisionOutput",
  "type": "object",
  "required": ["decision_id", "timestamp", "symbol", "action", "confidence"],
  "properties": {
    "decision_id": {"type": "string", "format": "uuid"},
    "timestamp": {"type": "string", "format": "date-time"},
    "symbol": {"type": "string", "enum": ["BTC/USDT", "XRP/USDT", "XRP/BTC"]},
    "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD", "CLOSE"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "parameters": {
      "type": "object",
      "properties": {
        "entry_price": {"type": ["number", "null"]},
        "position_size_usd": {"type": "number", "minimum": 0},
        "leverage": {"type": "integer", "minimum": 1, "maximum": 5},
        "stop_loss": {"type": "number"},
        "take_profit": {"type": "number"},
        "order_type": {"type": "string", "enum": ["market", "limit"]},
        "time_in_force": {"type": "string", "enum": ["gtc", "ioc", "fok"]}
      }
    },
    "reasoning": {"type": "string", "minLength": 10, "maxLength": 1000}
  }
}
```

### 3.2 Output Validation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     OUTPUT VALIDATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LLM Response                                                               │
│       │                                                                     │
│       v                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. JSON EXTRACTION                                                   │   │
│  │    • Find JSON block in response                                     │   │
│  │    • Handle markdown code blocks                                     │   │
│  │    • Clean up common formatting issues                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       v                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. SCHEMA VALIDATION                                                 │   │
│  │    • Validate against agent-specific schema                          │   │
│  │    • Check required fields present                                   │   │
│  │    • Verify enum values                                              │   │
│  │    • Validate numeric ranges                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       v                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. BUSINESS LOGIC VALIDATION                                         │   │
│  │    • Stop-loss must be set if action != HOLD                         │   │
│  │    • Position size within limits                                     │   │
│  │    • Leverage within regime-adjusted limits                          │   │
│  │    • Risk/reward ratio acceptable                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       v                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 4. SANITIZATION                                                      │   │
│  │    • Round prices to exchange precision                              │   │
│  │    • Normalize confidence scores                                     │   │
│  │    • Truncate overly long reasoning                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       v                                                                     │
│  Validated Output (or Error + Fallback)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Error Recovery

```python
ERROR_RECOVERY_STRATEGIES = {
    "json_parse_error": {
        "action": "retry_with_simpler_prompt",
        "max_retries": 2,
        "fallback": "use_cached_output"
    },
    "schema_validation_error": {
        "action": "attempt_field_repair",
        "max_retries": 1,
        "fallback": "use_default_values"
    },
    "business_logic_error": {
        "action": "apply_corrections",
        "max_retries": 0,
        "fallback": "return_hold_decision"
    },
    "timeout": {
        "action": "use_local_model",
        "max_retries": 1,
        "fallback": "use_cached_output"
    },
    "rate_limit": {
        "action": "wait_and_retry",
        "max_retries": 3,
        "fallback": "use_local_model"
    }
}
```

---

## 4. LLM Client Configuration

### 4.1 Ollama Configuration (Tier 1)

```yaml
# ollama_config.yaml
base_url: "http://localhost:11434"
model: "qwen2.5:7b"
model_path: "/media/rese/2tb_drive/ollama_config/"

default_options:
  temperature: 0.3  # Lower for more consistent outputs
  top_p: 0.9
  top_k: 40
  num_predict: 1024
  num_ctx: 8192
  repeat_penalty: 1.1

agent_specific_options:
  technical_analysis:
    temperature: 0.2
    num_predict: 512

  regime_detection:
    temperature: 0.2
    num_predict: 512

  trading_decision_execution:
    temperature: 0.1  # Very deterministic for execution
    num_predict: 256

health_check:
  endpoint: "/api/tags"
  interval_seconds: 60
  timeout_seconds: 5

connection:
  timeout_seconds: 30
  max_retries: 3
  retry_delay_seconds: 1
```

### 4.2 API Client Configuration (Tier 2)

```yaml
# api_llm_config.yaml
providers:
  deepseek:
    base_url: "https://api.deepseek.com/v1"
    api_key_env: "DEEPSEEK_API_KEY"
    model: "deepseek-chat"
    default_options:
      temperature: 0.3
      max_tokens: 2048
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 100000
    roles: ["trading_decision", "portfolio_rebalancing", "coordinator"]

  anthropic_sonnet:
    base_url: "https://api.anthropic.com/v1"
    api_key_env: "ANTHROPIC_API_KEY"
    model: "claude-sonnet-4-20250514"
    default_options:
      temperature: 0.3
      max_tokens: 2048
    rate_limit:
      requests_per_minute: 50
      tokens_per_minute: 100000
    roles: ["trading_decision", "coordinator"]

  anthropic_opus:
    base_url: "https://api.anthropic.com/v1"
    api_key_env: "ANTHROPIC_API_KEY"
    model: "claude-opus-4-20250514"
    default_options:
      temperature: 0.3
      max_tokens: 2048
    rate_limit:
      requests_per_minute: 30
      tokens_per_minute: 50000
    roles: ["trading_decision"]

  openai:
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-4o"
    default_options:
      temperature: 0.3
      max_tokens: 2048
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 150000
    roles: ["trading_decision", "sentiment_analysis"]
    capabilities: ["web_search"]

  xai_grok:
    base_url: "https://api.x.ai/v1"
    api_key_env: "XAI_API_KEY"
    model: "grok-2"
    default_options:
      temperature: 0.3
      max_tokens: 2048
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 100000
    roles: ["trading_decision", "sentiment_analysis"]
    capabilities: ["web_search", "real_time_data"]

# A/B Testing Configuration
ab_testing:
  trading_decision:
    enabled: true
    models:
      - "deepseek-chat"
      - "claude-sonnet-4-20250514"
      - "claude-opus-4-20250514"
      - "gpt-4o"
      - "grok-2"
      - "qwen2.5:7b"  # Local via Ollama
    parallel_execution: true
    comparison_mode: "all_models"  # Run all models, compare results

  sentiment_analysis:
    enabled: true
    models:
      - "grok-2"
      - "gpt-4o"
    parallel_execution: true
    web_search_enabled: true
    invocation_interval: "30m"

failover:
  strategy: "round_robin"  # Distribute load across models
  retry_count: 2
  retry_delay_seconds: 1
  fallback_to_local: true

cost_tracking:
  enabled: true
  daily_budget_usd: 5.00  # Increased for 6-model A/B testing
  alert_threshold_pct: 80
  per_model_tracking: true
```

### 4.3 Request/Response Logging

```yaml
# logging_config.yaml
llm_logging:
  enabled: true
  log_level: "INFO"

  request_logging:
    log_prompts: true
    log_model: true
    log_parameters: true
    truncate_prompts_at: 2000  # Characters

  response_logging:
    log_responses: true
    log_token_counts: true
    log_latency: true
    truncate_responses_at: 2000

  storage:
    type: "file"
    path: "logs/llm/"
    rotation: "daily"
    retention_days: 30

  metrics:
    track_latency: true
    track_tokens: true
    track_costs: true
    track_errors: true
    export_to: "prometheus"
```

---

## 5. Caching Strategy

### 5.1 Cache Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM CACHING SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L1 CACHE: In-Memory (per agent)                                      │   │
│  │                                                                      │   │
│  │  • TTL: 60 seconds (configurable per agent)                         │   │
│  │  • Purpose: Deduplicate rapid successive calls                       │   │
│  │  • Key: hash(prompt + model + temperature)                          │   │
│  │  • Size: Last 100 responses per agent                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    │ Miss                                   │
│                                    v                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L2 CACHE: Redis (shared)                                             │   │
│  │                                                                      │   │
│  │  • TTL: Varies by analysis type                                     │   │
│  │    - Regime: 5 minutes                                              │   │
│  │    - Technical: 1 minute                                            │   │
│  │    - Sentiment: 15 minutes                                          │   │
│  │    - Strategic (1h+): 30 minutes                                    │   │
│  │  • Key: hash(agent + symbol + timeframe + prompt_hash)              │   │
│  │  • Invalidation: Market data change > threshold                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    │ Miss                                   │
│                                    v                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L3 CACHE: Historical Output Store (TimescaleDB)                      │   │
│  │                                                                      │   │
│  │  • Purpose: Learning and backtesting                                │   │
│  │  • Retention: Indefinite                                            │   │
│  │  • Use: Analyze past decisions, train improvements                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Cache Invalidation Rules

```python
CACHE_INVALIDATION_RULES = {
    "technical_analysis": {
        "invalidate_on": [
            "new_candle_close",
            "price_change_pct > 0.5",
            "volume_spike > 3x_average"
        ],
        "max_age_seconds": 60
    },
    "regime_detection": {
        "invalidate_on": [
            "atr_change_pct > 20",
            "adx_cross_threshold",
            "ema_cross"
        ],
        "max_age_seconds": 300
    },
    "sentiment_analysis": {
        "invalidate_on": [
            "new_high_impact_news",
            "fear_greed_change > 10",
            "whale_alert"
        ],
        "max_age_seconds": 900
    },
    "trading_decision": {
        "invalidate_on": [
            "position_closed",
            "stop_loss_hit",
            "regime_change"
        ],
        "max_age_seconds": 1800
    }
}
```

---

## 6. Cost Management

### 6.1 Cost Estimation Model

| Agent | Avg Tokens/Call | Calls/Day | Daily Cost (DeepSeek) |
|-------|-----------------|-----------|----------------------|
| Technical Analysis | 1,500 | 1,440 (per minute) | ~$0.00 (local) |
| Regime Detection | 1,200 | 288 (every 5 min) | ~$0.00 (local) |
| Sentiment Analysis | 2,000 | 96 (every 15 min) | ~$0.04 |
| Trading Decision (Strategic) | 2,500 | 24 (hourly) | ~$0.01 |
| Coordinator | 1,500 | 10 (on conflict) | ~$0.003 |
| **TOTAL** | - | - | **~$0.05/day** |

### 6.2 Cost Control Measures

```yaml
cost_controls:
  daily_budget:
    total_usd: 1.00
    warning_threshold_pct: 50
    hard_limit_pct: 100

  per_call_limits:
    max_input_tokens: 4000
    max_output_tokens: 2000

  throttling:
    enable_at_budget_pct: 80
    strategy: "reduce_frequency"
    reduced_intervals:
      sentiment: "30m"  # From 15m
      trading_decision: "2h"  # From 1h

  fallback:
    at_budget_pct: 100
    action: "local_only"
    duration: "until_daily_reset"
```

---

## 7. Model Comparison Framework

### 7.1 Six-Model A/B Testing Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     6-MODEL PARALLEL COMPARISON                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Market Data → Prompt Builder → ┬──→ GPT (API)         ──┐                 │
│                                 ├──→ Grok (API)        ──┤                 │
│                                 ├──→ DeepSeek V3 (API) ──┼──→ Comparison   │
│                                 ├──→ Claude Sonnet (API)─┤     Engine      │
│                                 ├──→ Claude Opus (API) ──┤       │         │
│                                 └──→ Qwen 2.5 7B (Local)─┘       v         │
│                                                              Leaderboard   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 A/B Testing Configuration

```yaml
model_comparison:
  enabled: true
  mode: "parallel_all"  # All 6 models run simultaneously

  models:
    - id: "gpt"
      provider: "openai"
      model: "gpt-4o"
      tier: "api"

    - id: "grok"
      provider: "xai"
      model: "grok-2"
      tier: "api"

    - id: "deepseek"
      provider: "deepseek"
      model: "deepseek-chat"
      tier: "api"

    - id: "claude_sonnet"
      provider: "anthropic"
      model: "claude-sonnet-4-20250514"
      tier: "api"

    - id: "claude_opus"
      provider: "anthropic"
      model: "claude-opus-4-20250514"
      tier: "api"

    - id: "qwen"
      provider: "ollama"
      model: "qwen2.5:7b"
      tier: "local"

  execution_strategy:
    parallel: true
    timeout_seconds: 30
    wait_for_all: false  # Proceed when first N models respond
    min_responses: 4

  evaluation_criteria:
    accuracy:
      weight: 0.35
      measure: "decision_correct_after_1h"
      description: "Was the predicted direction correct?"

    profitability:
      weight: 0.25
      measure: "pnl_if_followed"
      description: "P&L if this model's decisions were executed"

    confidence_calibration:
      weight: 0.20
      measure: "predicted_confidence_vs_actual_outcome"
      description: "How well does confidence predict success?"

    latency:
      weight: 0.10
      measure: "p95_response_time"
      description: "Response time at 95th percentile"

    cost:
      weight: 0.10
      measure: "cost_per_correct_decision"
      description: "Cost efficiency"

  comparison_windows:
    - period: "1h"
      description: "Short-term accuracy"
    - period: "4h"
      description: "Medium-term accuracy"
    - period: "1d"
      description: "Daily accuracy"
    - period: "1w"
      description: "Weekly performance"

  leaderboard:
    update_frequency: "hourly"
    min_decisions_for_ranking: 50
    display_metrics:
      - "accuracy"
      - "profitability"
      - "sharpe_ratio"
      - "cost_efficiency"
```

### 7.3 Performance Tracking Schema

```json
{
  "model_leaderboard": {
    "period": "2025-12-01 to 2025-12-18",
    "total_decisions_evaluated": 432,
    "models": [
      {
        "rank": 1,
        "model": "deepseek-chat",
        "metrics": {
          "accuracy": 0.62,
          "profitability_pct": 8.5,
          "sharpe_ratio": 1.8,
          "confidence_calibration": 0.88,
          "avg_latency_ms": 2850,
          "total_cost_usd": 0.89,
          "cost_per_correct_decision": 0.003
        },
        "composite_score": 0.78
      },
      {
        "rank": 2,
        "model": "claude-opus-4-20250514",
        "metrics": {
          "accuracy": 0.60,
          "profitability_pct": 7.2,
          "sharpe_ratio": 1.6,
          "confidence_calibration": 0.85,
          "avg_latency_ms": 4500,
          "total_cost_usd": 2.45,
          "cost_per_correct_decision": 0.010
        },
        "composite_score": 0.72
      },
      {
        "rank": 3,
        "model": "grok-2",
        "metrics": {
          "accuracy": 0.58,
          "profitability_pct": 6.8,
          "sharpe_ratio": 1.5,
          "confidence_calibration": 0.82,
          "avg_latency_ms": 2900,
          "total_cost_usd": 1.20,
          "cost_per_correct_decision": 0.005
        },
        "composite_score": 0.70
      },
      {
        "rank": 4,
        "model": "claude-sonnet-4-20250514",
        "metrics": {
          "accuracy": 0.57,
          "profitability_pct": 5.5,
          "sharpe_ratio": 1.4,
          "confidence_calibration": 0.80,
          "avg_latency_ms": 2100,
          "total_cost_usd": 1.50,
          "cost_per_correct_decision": 0.006
        },
        "composite_score": 0.68
      },
      {
        "rank": 5,
        "model": "gpt-4o",
        "metrics": {
          "accuracy": 0.55,
          "profitability_pct": 4.2,
          "sharpe_ratio": 1.2,
          "confidence_calibration": 0.75,
          "avg_latency_ms": 2000,
          "total_cost_usd": 1.80,
          "cost_per_correct_decision": 0.008
        },
        "composite_score": 0.62
      },
      {
        "rank": 6,
        "model": "qwen2.5:7b",
        "metrics": {
          "accuracy": 0.52,
          "profitability_pct": 3.1,
          "sharpe_ratio": 1.0,
          "confidence_calibration": 0.70,
          "avg_latency_ms": 200,
          "total_cost_usd": 0.00,
          "cost_per_correct_decision": 0.000
        },
        "composite_score": 0.58
      }
    ]
  },
  "by_decision_type": {
    "BUY": {
      "best_model": "deepseek-chat",
      "accuracy": 0.65
    },
    "SELL": {
      "best_model": "claude-opus-4-20250514",
      "accuracy": 0.62
    },
    "HOLD": {
      "best_model": "grok-2",
      "accuracy": 0.70
    }
  },
  "consensus_analysis": {
    "unanimous_decisions": 85,
    "unanimous_accuracy": 0.75,
    "majority_decisions": 180,
    "majority_accuracy": 0.68,
    "split_decisions": 167,
    "split_accuracy": 0.48
  }
}
```

### 7.4 Consensus Decision Logic

```yaml
consensus_rules:
  # When to use consensus vs single model
  modes:
    unanimous:
      description: "All 6 models agree"
      action: "execute_with_high_confidence"
      confidence_boost: 0.15

    strong_majority:
      description: "5 of 6 models agree"
      action: "execute_with_confidence"
      confidence_boost: 0.10

    majority:
      description: "4 of 6 models agree"
      action: "execute_if_confidence_threshold_met"
      confidence_boost: 0.05

    split:
      description: "3-3 split or less"
      action: "defer_to_top_performer_or_hold"
      confidence_penalty: 0.10

  tie_breaker:
    method: "highest_confidence_from_top_3_performers"
    fallback: "hold"
```

---

## 8. Security Considerations

### 8.1 API Key Management

```yaml
security:
  api_keys:
    storage: "environment_variables"
    never_log: true
    rotation_reminder_days: 90

  request_signing:
    enabled: false  # LLM APIs don't typically require

  response_validation:
    verify_source: true
    check_tampering: false  # Not applicable for LLM responses
```

### 8.2 Prompt Injection Protection

```python
PROMPT_SAFETY_RULES = {
    "input_sanitization": {
        "remove_patterns": [
            r"ignore previous instructions",
            r"disregard all prior",
            r"system prompt:",
            r"you are now"
        ],
        "max_user_input_length": 5000,
        "escape_special_chars": True
    },

    "output_validation": {
        "reject_if_contains": [
            "api_key",
            "password",
            "secret"
        ],
        "action_whitelist": ["BUY", "SELL", "HOLD", "CLOSE"]
    }
}
```

---

*Document Version 1.0 - December 2025*
