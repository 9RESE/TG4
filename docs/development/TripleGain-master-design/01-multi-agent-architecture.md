# TripleGain Multi-Agent Architecture

**Document Version**: 1.0
**Status**: Design Phase

---

## 1. Architecture Overview

### 1.1 System Context Diagram

```
                                    TripleGain System
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                           EXTERNAL DATA LAYER                                │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐                │  │
│  │  │  Kraken   │  │   News    │  │ Sentiment │  │ On-Chain  │                │  │
│  │  │ WebSocket │  │   APIs    │  │   APIs    │  │ Analytics │                │  │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                │  │
│  └────────┼──────────────┼──────────────┼──────────────┼────────────────────────┘  │
│           │              │              │              │                            │
│           v              v              v              v                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                           DATA INGESTION LAYER                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │  │
│  │  │                    TimescaleDB (Historical Store)                    │    │  │
│  │  │  • 5-9 years OHLCV data  • 8 timeframes  • 3 trading pairs          │    │  │
│  │  └─────────────────────────────────────────────────────────────────────┘    │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │  │
│  │  │ Order Book    │  │ Trade Flow    │  │ External      │                   │  │
│  │  │ Collector     │  │ Collector     │  │ Data Cache    │                   │  │
│  │  └───────────────┘  └───────────────┘  └───────────────┘                   │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                           │
│                                        v                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                           AGENT ORCHESTRATION LAYER                          │  │
│  │                                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │  │
│  │  │                        ANALYSIS AGENTS (Tier 1)                      │   │  │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │  │
│  │  │  │  Technical   │  │   Regime     │  │  Sentiment   │               │   │  │
│  │  │  │   Analysis   │  │  Detection   │  │   Analysis   │               │   │  │
│  │  │  │   Agent      │  │    Agent     │  │    Agent     │               │   │  │
│  │  │  │   (Local)    │  │   (Local)    │  │    (API)     │               │   │  │
│  │  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │   │  │
│  │  └─────────┼─────────────────┼─────────────────┼──────────────────────────┘   │  │
│  │            │                 │                 │                              │  │
│  │            └─────────────────┼─────────────────┘                              │  │
│  │                              v                                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │  │
│  │  │                      DECISION AGENTS (Tier 2)                        │   │  │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │  │
│  │  │  │   Trading    │  │     Risk     │  │  Portfolio   │               │   │  │
│  │  │  │   Decision   │  │  Management  │  │  Rebalancing │               │   │  │
│  │  │  │    Agent     │  │    Agent     │  │    Agent     │               │   │  │
│  │  │  │    (API)     │  │   (Rules)    │  │   (Rules)    │               │   │  │
│  │  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │   │  │
│  │  └─────────┼─────────────────┼─────────────────┼──────────────────────────┘   │  │
│  │            │                 │                 │                              │  │
│  │            └─────────────────┼─────────────────┘                              │  │
│  │                              v                                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │  │
│  │  │                       COORDINATOR AGENT                              │   │  │
│  │  │           (Consensus Building & Conflict Resolution)                 │   │  │
│  │  └─────────────────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                           │
│                                        v                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                           EXECUTION LAYER                                    │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │  │
│  │  │ Order Manager │  │ Position      │  │ Trade Logger  │                   │  │
│  │  │               │  │ Tracker       │  │               │                   │  │
│  │  └───────┬───────┘  └───────────────┘  └───────────────┘                   │  │
│  └──────────┼──────────────────────────────────────────────────────────────────┘  │
│             │                                                                      │
│             v                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                           EXCHANGE INTERFACE                                 │  │
│  │                      Kraken REST API / WebSocket                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Agent Hierarchy

```
                         ┌─────────────────────────────┐
                         │     COORDINATOR AGENT       │  ← Final authority
                         │    (Consensus & Override)   │
                         └─────────────┬───────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              v                        v                        v
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   TRADING DECISION  │  │   RISK MANAGEMENT   │  │ PORTFOLIO REBALANCE │
│       AGENT         │  │       AGENT         │  │       AGENT         │
│  (Signal synthesis) │  │   (Veto authority)  │  │  (Allocation mgmt)  │
└──────────┬──────────┘  └─────────────────────┘  └─────────────────────┘
           │
           │  Receives input from:
           │
    ┌──────┴───────────────────────────────┐
    │                                      │
    v                                      v
┌─────────────────────┐          ┌─────────────────────┐
│  TECHNICAL ANALYSIS │          │  SENTIMENT ANALYSIS │
│       AGENT         │          │       AGENT         │
│  (Indicators, TA)   │          │   (News, Social)    │
└──────────┬──────────┘          └─────────────────────┘
           │
           v
┌─────────────────────┐
│   REGIME DETECTION  │
│       AGENT         │
│  (Market state ID)  │
└─────────────────────┘
```

---

## 2. Agent Specifications

### 2.1 Technical Analysis Agent

**Purpose**: Compute technical indicators and generate technical signals

| Property | Value |
|----------|-------|
| **Execution Tier** | Tier 1 (Local) |
| **LLM Model** | Qwen 2.5 7B via Ollama |
| **Latency Target** | < 200ms |
| **Decision Authority** | Advisory (provides signals to Trading Decision Agent) |
| **Veto Power** | None |

**Input**:
```yaml
market_data:
  candles:
    - timeframe: 1m
      lookback: 100
    - timeframe: 5m
      lookback: 60
    - timeframe: 1h
      lookback: 48
    - timeframe: 4h
      lookback: 30
    - timeframe: 1d
      lookback: 30
  order_book:
    depth: 10
  recent_trades:
    count: 100
```

**Computed Indicators**:
| Category | Indicators |
|----------|------------|
| Trend | EMA(9,21,50,200), ADX, Trend Slope, Supertrend |
| Momentum | RSI(14), MACD, Stochastic RSI, ROC |
| Volatility | ATR(14), Bollinger Bands, Keltner Channels |
| Volume | OBV, VWAP, Volume MA Ratio |
| Pattern | Choppiness Index, Squeeze Detection |

**Output**:
```json
{
  "agent": "technical_analysis",
  "timestamp": "2025-12-18T10:30:00Z",
  "symbol": "BTC/USDT",
  "signals": {
    "trend_direction": "bullish",
    "trend_strength": 0.72,
    "momentum_score": 0.65,
    "volatility_state": "normal",
    "volume_confirmation": true,
    "squeeze_detected": false
  },
  "indicators": {
    "rsi_14": 62.5,
    "macd_histogram": 150.2,
    "adx": 28.5,
    "atr_14": 1250.0,
    "bb_position": 0.65
  },
  "recommendations": {
    "bias": "long",
    "confidence": 0.68,
    "entry_zones": [42500, 42200],
    "resistance_levels": [43500, 44000],
    "support_levels": [42000, 41500]
  }
}
```

---

### 2.2 Regime Detection Agent

**Purpose**: Identify current market regime to route strategy parameters

| Property | Value |
|----------|-------|
| **Execution Tier** | Tier 1 (Local) |
| **LLM Model** | Qwen 2.5 7B via Ollama |
| **Latency Target** | < 300ms |
| **Decision Authority** | Advisory (informs strategy selection) |
| **Veto Power** | None |

**Regime Classification**:
| Regime | Characteristics | Strategy Implication |
|--------|-----------------|---------------------|
| **Trending Bull** | ADX > 25, Price > EMA200, Higher highs | Trend-following long |
| **Trending Bear** | ADX > 25, Price < EMA200, Lower lows | Trend-following short |
| **Ranging** | ADX < 20, Choppiness > 60 | Mean reversion / grid |
| **Volatile** | ATR > 2x normal, Wide BB | Reduced position size |
| **Quiet** | ATR < 0.5x normal, BB squeeze | Breakout anticipation |

**Input**:
```yaml
features:
  - adx_14
  - choppiness_14
  - atr_ratio  # Current ATR / 20-period avg ATR
  - bb_width
  - price_vs_ema200
  - higher_highs_count_20d
  - lower_lows_count_20d
  - trend_slope_1h
  - trend_slope_4h
  - trend_slope_1d
```

**Output**:
```json
{
  "agent": "regime_detection",
  "timestamp": "2025-12-18T10:30:00Z",
  "symbol": "BTC/USDT",
  "regime": {
    "primary": "trending_bull",
    "confidence": 0.78,
    "secondary": "volatile",
    "duration_hours": 48
  },
  "regime_scores": {
    "trending_bull": 0.78,
    "trending_bear": 0.05,
    "ranging": 0.10,
    "volatile": 0.45,
    "quiet": 0.02
  },
  "transition_probability": {
    "to_trending_bear": 0.08,
    "to_ranging": 0.15,
    "regime_change_alert": false
  },
  "parameter_adjustments": {
    "leverage_multiplier": 0.8,
    "position_size_multiplier": 1.0,
    "stop_loss_multiplier": 1.2
  }
}
```

---

### 2.3 Sentiment Analysis Agent

**Purpose**: Analyze news and social sentiment for trading signals

| Property | Value |
|----------|-------|
| **Execution Tier** | Tier 2 (API) |
| **LLM Model** | Grok / GPT |
| **Latency Target** | < 5 seconds |
| **Decision Authority** | Advisory |
| **Veto Power** | Can flag high-risk news events |
| **Invocation Frequency** | Every 15 minutes (not per-tick) |

**Data Sources**:
| Source | Type | Update Frequency |
|--------|------|------------------|
| CryptoPanic API | Aggregated news | Every 5 minutes |
| Fear & Greed Index | Market sentiment | Every 8 hours |
| On-chain Metrics | Whale alerts | Every 15 minutes |
| Twitter/X (future) | Social sentiment | Every 30 minutes |
| Web Search | Aggregated news | Every 30 minutes |

**Input**:
```yaml
news_items:
  - title: "SEC Approves New Bitcoin ETF"
    source: "Reuters"
    published_at: "2025-12-18T10:00:00Z"
    relevance_score: 0.95
  - title: "Major Exchange Reports Record Volume"
    source: "CoinDesk"
    published_at: "2025-12-18T09:30:00Z"
    relevance_score: 0.7
fear_greed_index: 72  # 0-100 scale
on_chain:
  whale_transactions_24h: 15
  exchange_inflows_btc: 5000
  exchange_outflows_btc: 8000
```

**Output**:
```json
{
  "agent": "sentiment_analysis",
  "timestamp": "2025-12-18T10:30:00Z",
  "symbols": ["BTC/USDT", "XRP/USDT"],
  "sentiment": {
    "overall_score": 0.65,
    "news_sentiment": 0.72,
    "social_sentiment": 0.58,
    "fear_greed_index": 72,
    "on_chain_sentiment": 0.60
  },
  "news_events": [
    {
      "event": "SEC ETF Approval",
      "impact": "bullish",
      "magnitude": "high",
      "affected_assets": ["BTC"],
      "time_horizon": "short_term"
    }
  ],
  "alerts": {
    "high_impact_news": true,
    "sentiment_extreme": false,
    "whale_activity_unusual": false
  },
  "recommendations": {
    "sentiment_bias": "bullish",
    "confidence": 0.68,
    "caution_flags": []
  }
}
```

---

### 2.4 Trading Decision Agent

**Purpose**: Synthesize all inputs and generate trading signals

| Property | Value |
|----------|-------|
| **Execution Tier** | Tier 2 (API) for strategic decisions, Tier 1 for execution |
| **LLM Model** | Multi-model A/B testing: GPT, Grok, DeepSeek V3, Claude Sonnet, Claude Opus, Qwen 2.5 7B (all models run in parallel for comparison) |
| **Latency Target** | Strategic: < 30s, Execution: < 500ms |
| **Decision Authority** | Primary signal generator |
| **Veto Power** | None (subject to Risk Management veto) |

**Decision Timeframes**:
| Timeframe | LLM Tier | Frequency | Purpose |
|-----------|----------|-----------|---------|
| 1M (Monthly) | Tier 2 API | Once/month | Strategic allocation |
| 1W (Weekly) | Tier 2 API | Once/week | Trend confirmation |
| 1D (Daily) | Tier 2 API | Once/day | Bias determination |
| 4H | Tier 2 API | Every 4 hours | Position management |
| 1H | Tier 2 API | Every hour | Entry/exit refinement |
| Real-time | Tier 1 Local | Per tick | Execution decisions |

**Input**:
```yaml
technical_analysis_output: <from TA Agent>
regime_detection_output: <from Regime Agent>
sentiment_analysis_output: <from Sentiment Agent>
current_positions:
  - symbol: "BTC/USDT"
    side: "long"
    size: 0.5
    entry_price: 42000
    unrealized_pnl: 250
    duration_hours: 12
portfolio_state:
  total_equity: 2100
  available_margin: 1500
  allocation:
    btc_pct: 35
    xrp_pct: 28
    usdt_pct: 37
```

**Output**:
```json
{
  "agent": "trading_decision",
  "timestamp": "2025-12-18T10:30:00Z",
  "decision_timeframe": "1h",
  "decisions": [
    {
      "symbol": "BTC/USDT",
      "action": "HOLD",
      "confidence": 0.75,
      "reasoning": "Strong uptrend confirmed by TA and sentiment. Current long position aligned with bias. No adjustment needed.",
      "adjustments": null
    },
    {
      "symbol": "XRP/USDT",
      "action": "BUY",
      "confidence": 0.68,
      "reasoning": "Technical breakout detected with bullish sentiment confirmation. Regime supports trend-following entry.",
      "parameters": {
        "entry_price": 0.62,
        "position_size_usd": 200,
        "leverage": 3,
        "stop_loss": 0.595,
        "take_profit": 0.67,
        "invalidation": "Close below 0.59 on 4H"
      }
    }
  ],
  "market_outlook": {
    "bias": "bullish",
    "confidence": 0.70,
    "key_levels": {
      "btc_support": 42000,
      "btc_resistance": 45000,
      "xrp_support": 0.58,
      "xrp_resistance": 0.70
    }
  }
}
```

---

### 2.5 Risk Management Agent

**Purpose**: Enforce risk rules and provide veto authority on trades

| Property | Value |
|----------|-------|
| **Execution Tier** | Rules-based (deterministic, no LLM) |
| **Latency Target** | < 10ms |
| **Decision Authority** | **VETO AUTHORITY** over all trades |
| **Override Capability** | Can reduce position sizes, reject trades, force exits |

**Risk Rules Engine**:
```
RULE: Maximum Position Size
  IF trade_size_usd > (portfolio_equity * max_position_pct)
  THEN reject OR reduce to limit

RULE: Maximum Leverage
  IF leverage > max_leverage_for_regime
  THEN reject OR reduce leverage

RULE: Required Stop Loss
  IF stop_loss NOT SET OR stop_loss_distance > max_stop_pct
  THEN reject

RULE: Drawdown Circuit Breaker
  IF daily_loss_pct > daily_loss_limit
  THEN halt_all_trading

RULE: Consecutive Loss Protection
  IF consecutive_losses >= max_consecutive_losses
  THEN cooldown_period

RULE: Confidence Threshold
  IF signal_confidence < min_confidence_threshold
  THEN reject

RULE: Correlated Position Limit
  IF correlated_exposure > max_correlated_exposure
  THEN reject OR reduce
```

**Output**:
```json
{
  "agent": "risk_management",
  "timestamp": "2025-12-18T10:30:00Z",
  "trade_evaluation": {
    "original_trade": {
      "symbol": "XRP/USDT",
      "action": "BUY",
      "size_usd": 200,
      "leverage": 3
    },
    "approval_status": "APPROVED_WITH_MODIFICATIONS",
    "modifications": {
      "leverage": 2,
      "reason": "Volatile regime detected, leverage reduced from 3x to 2x"
    },
    "risk_metrics": {
      "position_risk_usd": 12,
      "portfolio_risk_pct": 0.57,
      "max_loss_if_stopped": 12,
      "risk_reward_ratio": 2.5
    }
  },
  "portfolio_risk_state": {
    "total_exposure_pct": 45,
    "daily_pnl_pct": 1.2,
    "drawdown_pct": 3.5,
    "consecutive_losses": 0,
    "circuit_breakers_triggered": []
  },
  "warnings": [],
  "blocks": []
}
```

---

### 2.6 Portfolio Rebalancing Agent

**Purpose**: Maintain target portfolio allocation (33/33/33)

| Property | Value |
|----------|-------|
| **Execution Tier** | Rules-based with LLM override for edge cases |
| **LLM Model** | Deepseek (for edge case reasoning) |
| **Latency Target** | < 1 second |
| **Decision Authority** | Advisory (recommends rebalancing trades) |
| **Invocation Frequency** | Hourly check, action when deviation > threshold |

**Rebalancing Logic**:
```
TARGET_ALLOCATION:
  BTC: 33.33%
  XRP: 33.33%
  USDT: 33.33%

REBALANCE_THRESHOLD: 5%  # Trigger when any asset deviates > 5%

REBALANCE_STRATEGY:
  1. Calculate current allocation percentages
  2. Identify deviations from target
  3. If max_deviation > threshold:
     a. Calculate rebalancing trades needed
     b. Prioritize selling overweight assets
     c. Use limit orders (not market) for better execution
     d. Consider tax implications (if applicable)
     e. Respect minimum trade sizes
```

**Output**:
```json
{
  "agent": "portfolio_rebalancing",
  "timestamp": "2025-12-18T10:30:00Z",
  "current_allocation": {
    "btc_pct": 38.5,
    "xrp_pct": 28.2,
    "usdt_pct": 33.3
  },
  "target_allocation": {
    "btc_pct": 33.33,
    "xrp_pct": 33.33,
    "usdt_pct": 33.33
  },
  "deviations": {
    "btc_pct": 5.17,
    "xrp_pct": -5.13,
    "usdt_pct": -0.03
  },
  "rebalance_needed": true,
  "recommended_trades": [
    {
      "symbol": "BTC/USDT",
      "action": "SELL",
      "amount_usd": 108.57,
      "reason": "Reduce BTC allocation from 38.5% to 33.33%"
    },
    {
      "symbol": "XRP/USDT",
      "action": "BUY",
      "amount_usd": 107.73,
      "reason": "Increase XRP allocation from 28.2% to 33.33%"
    }
  ],
  "execution_priority": "low",
  "defer_until": "next_low_volatility_period"
}
```

---

### 2.7 Coordinator Agent

**Purpose**: Build consensus, resolve conflicts, and make final decisions

| Property | Value |
|----------|-------|
| **Execution Tier** | Tier 2 (API) |
| **LLM Model** | DeepSeek V3 / Claude Sonnet |
| **Latency Target** | < 10 seconds |
| **Decision Authority** | **FINAL AUTHORITY** (after Risk Management veto) |
| **Role** | Conflict resolution, consensus building, human escalation |

**Conflict Resolution Protocol**:
```
PRIORITY ORDER (highest to lowest):
  1. Risk Management Agent (VETO POWER - cannot be overridden)
  2. Coordinator Agent (final decision among non-vetoed options)
  3. Trading Decision Agent
  4. Portfolio Rebalancing Agent
  5. Sentiment Analysis Agent
  6. Technical Analysis Agent
  7. Regime Detection Agent

CONFLICT SCENARIOS:
  Scenario 1: TA says BUY, Sentiment says SELL
    → Coordinator weighs confidence scores
    → Higher confidence wins, or HOLD if close

  Scenario 2: Trading Decision says BUY, Risk says REJECT
    → Risk VETO is final
    → Log rejection reason for learning

  Scenario 3: Rebalancing conflicts with Trading signal
    → Trading signals take priority during active trades
    → Rebalancing deferred to low-activity periods
```

**Output**:
```json
{
  "agent": "coordinator",
  "timestamp": "2025-12-18T10:30:00Z",
  "consensus_process": {
    "input_signals": {
      "technical_analysis": {"bias": "long", "confidence": 0.72},
      "sentiment_analysis": {"bias": "long", "confidence": 0.68},
      "trading_decision": {"action": "BUY", "confidence": 0.70},
      "risk_management": {"approval": "APPROVED_WITH_MODIFICATIONS"},
      "portfolio_rebalancing": {"recommendation": "defer"}
    },
    "conflicts_detected": false,
    "resolution_method": null
  },
  "final_decision": {
    "action": "EXECUTE",
    "trade": {
      "symbol": "XRP/USDT",
      "side": "BUY",
      "size_usd": 200,
      "leverage": 2,
      "stop_loss": 0.595,
      "take_profit": 0.67
    },
    "confidence": 0.70,
    "rationale": "Strong consensus across all agents. Risk-adjusted parameters applied. Proceed with execution."
  },
  "audit_trail": {
    "decision_id": "dec_20251218_103000_001",
    "all_agent_outputs_logged": true,
    "human_override": false
  }
}
```

---

## 3. Agent Communication Protocol

### 3.1 Message Format

All inter-agent communication uses a standardized JSON envelope:

```json
{
  "message_id": "msg_uuid_v4",
  "timestamp": "ISO8601",
  "source_agent": "agent_name",
  "target_agent": "agent_name | broadcast",
  "message_type": "signal | request | response | alert | veto",
  "priority": "critical | high | normal | low",
  "payload": { /* agent-specific content */ },
  "correlation_id": "original_message_id (for responses)",
  "ttl_seconds": 60
}
```

### 3.2 Communication Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGENT COMMUNICATION FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DATA TRIGGER                                                            │
│     New candle / tick → Broadcast to all Analysis Agents                    │
│                                                                             │
│  2. PARALLEL ANALYSIS (async)                                               │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Technical Analysis ─┐                                           │    │
│     │ Regime Detection ───┼──→ Message Queue → Trading Decision       │    │
│     │ Sentiment Analysis ─┘    (waits for all or timeout)             │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  3. TRADING DECISION GENERATION                                             │
│     Trading Decision Agent synthesizes inputs → Generates signal            │
│                                                                             │
│  4. RISK EVALUATION (synchronous, blocking)                                 │
│     Signal → Risk Management Agent → Approve / Modify / Reject              │
│                                                                             │
│  5. COORDINATOR REVIEW (if conflicts)                                       │
│     Only invoked if conflicting signals detected                            │
│                                                                             │
│  6. EXECUTION                                                               │
│     Approved signal → Order Manager → Exchange                              │
│                                                                             │
│  7. FEEDBACK LOOP                                                           │
│     Execution result → Broadcast to all agents for learning                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Timing Constraints

| Phase | Max Duration | Timeout Action |
|-------|--------------|----------------|
| Analysis Agents (parallel) | 2 seconds | Proceed without late agents |
| Trading Decision | 5 seconds | Use cached decision |
| Risk Evaluation | 100ms | Reject (fail-safe) |
| Coordinator Review | 10 seconds | Use highest-confidence signal |
| Order Submission | 1 second | Retry with exponential backoff |

---

## 4. Agent State Management

### 4.1 Shared State Store

```yaml
shared_state:
  market_data:
    last_updated: timestamp
    candles: {symbol: {timeframe: [candles]}}
    order_book: {symbol: {bids: [], asks: []}}
    recent_trades: {symbol: [trades]}

  portfolio:
    equity: decimal
    available_margin: decimal
    positions: [{symbol, side, size, entry_price, unrealized_pnl}]
    allocation_pct: {btc: pct, xrp: pct, usdt: pct}

  risk_state:
    daily_pnl: decimal
    max_drawdown: decimal
    consecutive_losses: int
    circuit_breakers: [active_breakers]

  agent_outputs:
    technical_analysis: {output, timestamp, ttl}
    regime_detection: {output, timestamp, ttl}
    sentiment_analysis: {output, timestamp, ttl}
    trading_decision: {output, timestamp, ttl}

  trade_history:
    recent_trades: [{trade_details, outcome}]
    win_rate_7d: decimal
    avg_return_7d: decimal
```

### 4.2 Agent-Specific State

Each agent maintains private state for learning and adaptation:

```yaml
# Example: Trading Decision Agent state
trading_decision_state:
  decision_history:
    - decision_id: uuid
      timestamp: datetime
      signal: {action, confidence, reasoning}
      outcome: {result, pnl, accuracy}

  performance_metrics:
    total_decisions: int
    win_rate: decimal
    avg_confidence_on_wins: decimal
    avg_confidence_on_losses: decimal

  adaptive_parameters:
    confidence_calibration_offset: decimal
    preferred_timeframes: [timeframes]
    learned_patterns: [pattern_ids]
```

---

## 5. Hodl Bag Integration

### 5.1 Hodl Bag Logic

The Hodl Bag system accumulates long-term holdings from trading profits:

```
HODL_BAG_RULES:
  Allocation: 10% of realized profits → Hodl Bag

  Trigger Conditions:
    - Trade closes with profit > 0
    - Weekly rebalancing generates profit
    - Monthly allocation from positive portfolio growth

  Hodl Bag Assets:
    - BTC Hodl Bag: 33.3% of hodl allocation
    - XRP Hodl Bag: 33.3% of hodl allocation
    - USDT Hodl Bag: 33.3% of hodl allocation

  Restrictions:
    - Hodl Bag positions are NEVER sold for trading
    - Hodl Bag is excluded from 33/33/33 rebalancing calculation
    - Hodl Bag only grows, never decreases (unless explicit user action)
```

### 5.2 Integration with Portfolio Agent

```json
{
  "portfolio_calculation": {
    "trading_portfolio": {
      "btc_value_usd": 700,
      "xrp_value_usd": 580,
      "usdt_value_usd": 700,
      "total_trading_equity": 1980
    },
    "hodl_bags": {
      "btc_hodl_value_usd": 60,
      "xrp_hodl_value_usd": 60,
      "usdt_hodl_value_usd": 60,
      "total_hodl_value": 180
    },
    "total_equity": 2160,
    "trading_allocation_target": {
      "btc_pct": 33.33,
      "xrp_pct": 33.33,
      "usdt_pct": 33.33
    },
    "note": "Rebalancing calculated on trading_portfolio only, hodl_bags excluded"
  }
}
```

---

## 6. Human Override Interface

### 6.1 Override Capabilities

| Override Type | Authority Level | Description |
|---------------|-----------------|-------------|
| **Pause Trading** | User | Immediately halt all trading activity |
| **Force Exit** | User | Close specific or all positions |
| **Adjust Risk Parameters** | User | Modify leverage, position size limits |
| **Override Agent Decision** | User | Manually execute or reject a specific trade |
| **Emergency Stop** | System | Auto-triggered by circuit breakers |

### 6.2 Override Protocol

```yaml
override_request:
  type: "force_exit | pause | adjust_params | manual_trade"
  requester: "user | system"
  timestamp: datetime
  target:
    symbol: optional
    position_id: optional
    agent: optional
  action:
    description: "Close all BTC positions"
    parameters: {}
  reason: "Market uncertainty, prefer to be in cash"

override_response:
  request_id: uuid
  status: "executed | rejected | pending_confirmation"
  execution_details:
    trades_executed: []
    positions_closed: []
    parameters_changed: {}
  confirmation_required: boolean
```

---

## 7. Monitoring & Observability

### 7.1 Agent Health Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `agent_latency_ms` | Response time per agent | > 2x target |
| `agent_error_rate` | Errors / total invocations | > 1% |
| `agent_timeout_rate` | Timeouts / total invocations | > 0.5% |
| `decision_confidence_avg` | Rolling avg confidence | < 0.5 |
| `win_rate_7d` | Trading agent accuracy | < 40% |

### 7.2 Agent Audit Log

Every agent decision is logged:

```json
{
  "log_id": "uuid",
  "timestamp": "ISO8601",
  "agent": "trading_decision",
  "decision_type": "trade_signal",
  "inputs": {
    "market_data_snapshot_id": "uuid",
    "other_agent_outputs": ["uuid1", "uuid2"]
  },
  "output": { /* full agent output */ },
  "latency_ms": 450,
  "llm_model_used": "qwen-2.5-7b",
  "prompt_tokens": 1500,
  "completion_tokens": 350
}
```

---

## 8. Failure Modes & Recovery

### 8.1 Failure Scenarios

| Scenario | Detection | Recovery Action |
|----------|-----------|-----------------|
| LLM API timeout | Response > timeout | Use cached decision or local model |
| LLM API error | HTTP 5xx or error response | Retry with backoff, then fallback |
| Database unavailable | Connection failure | Use in-memory cache, queue writes |
| Exchange API down | Connection failure | Halt trading, alert user |
| Agent produces invalid output | Schema validation failure | Discard output, log error |
| All analysis agents timeout | No valid inputs to coordinator | Hold current positions, no new trades |

### 8.2 Graceful Degradation

```
DEGRADATION LEVELS:

Level 1: Normal Operation
  All agents functioning, full capability

Level 2: Reduced Capability
  - Sentiment Agent unavailable → Continue with TA + Regime only
  - Tier 2 API unavailable → Use Tier 1 (local) for all decisions

Level 3: Safe Mode
  - Multiple agent failures → No new trades
  - Risk Management only → Monitor and protect existing positions

Level 4: Emergency Halt
  - Critical system failure → Close all positions
  - Exchange API down → Alert user, await manual intervention
```

---

*Document Version 1.0 - December 2025*
