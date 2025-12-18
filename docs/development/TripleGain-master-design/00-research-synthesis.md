# TripleGain Research Synthesis

**Date**: December 2025
**Status**: Design Phase
**Document Version**: 1.0

---

## Executive Summary

This document synthesizes key findings from all research documentation to inform the TripleGain system design. The research spans AI trading systems, multi-agent architectures, algorithmic trading strategies, and LLM integration patterns.

---

## 1. Key Research Findings

### 1.1 LLM Trading Performance (Alpha Arena Results)

| Model | Performance | Key Insight |
|-------|-------------|-------------|
| **DeepSeek V3** | +25% to +48% | Best risk control, disciplined execution |
| **Qwen3 Max** | +31% (Winner) | Strong quantitative reasoning |
| **Claude Sonnet** | +10% | Conservative, only 3 trades (vs Gemini's 44) |
| **GPT-5** | -25% | Significant drawdown |
| **Gemini 2.5 Pro** | -39% | Hyperactive trading, 44 trades |

**Critical Insight**: Trade frequency does NOT correlate with success. Conservative, disciplined trading outperforms hyperactive strategies.

### 1.2 Multi-Agent Architecture Patterns

| Framework | Architecture | Performance |
|-----------|--------------|-------------|
| **TradingAgents** | Analyst Team → Research Debate → Trader → Risk Team → Fund Manager | Sharpe 6-8, Max DD <2% |
| **FinMem** | Working → Short-term → Long-term Memory Layers | Memory-enhanced learning |
| **AutoHedge** | Director → Quant → Risk Manager → Execution (Swarm) | Collaborative intelligence |
| **nof1.ai** | Multi-model cooperative with consensus | +25% live trading |

**Design Implication**: Specialized agents with clear decision authority outperform monolithic systems.

### 1.3 Algorithmic Strategy Evidence

| Strategy Type | Evidence Level | Margin Suitability |
|---------------|----------------|-------------------|
| Trend-Following Momentum | **High** | HIGH |
| Volatility Breakout | **Medium-High** | HIGH |
| LSTM + XGBoost Hybrid | **Medium-High** | Medium |
| Deep RL (DQN/PPO) | Medium | Medium |
| Grid Trading | Medium | Medium |
| Mean Reversion RSI | **Low - Does NOT work on crypto** | Low |

**Critical Insight**: Momentum/trend-following significantly outperforms mean reversion in cryptocurrency markets. Traditional RSI "buy the dip" strategies fail on Bitcoin.

### 1.4 LLM Integration Patterns

| Pattern | Use Case | Latency |
|---------|----------|---------|
| **Single-Agent** | Simple decisions | 1-3 sec |
| **News-Driven** | Sentiment-based signals | 3-5 sec |
| **Reasoning (CoT)** | Complex analysis | 5-10 sec |
| **RL-Enhanced** | Optimized execution | Training: hours, Inference: 1-3 sec |

**Multi-Model Architecture**:
- **Tier 1 (Local)**: Qwen 2.5 7B via Ollama for <500ms execution decisions
- **Tier 2 (API)**: Multiple models with specialized roles:
  - **Sentiment Analysis**: Grok and GPT (web search, aggregated news every 30 min)
  - **Trading Decision**: A/B testing all 6 models (GPT, Grok, DeepSeek V3, Claude Sonnet, Claude Opus, Qwen)
  - **Portfolio Rebalancing**: DeepSeek (for edge case reasoning)
  - **Coordinator**: DeepSeek V3 / Claude Sonnet (conflict resolution)

---

## 2. Technology Stack Analysis

### 2.1 Framework Comparison

| Framework | Strengths | Weaknesses | Recommendation |
|-----------|-----------|------------|----------------|
| **TensorTrade** | Modular, Gym-compatible | Inactive maintenance | Learning only |
| **FinRL** | Comprehensive RL | Complex, research-focused | RL experimentation |
| **Freqtrade** | Production-ready, mature | Not LLM-native | Live trading base |
| **Jesse** | No lookahead bias, multi-TF | No built-in RL | Backtesting |

**Decision**: Build custom system leveraging existing TimescaleDB infrastructure rather than adopting external framework.

### 2.2 LLM Provider Comparison

| Provider | Cost | Latency | Quantitative Reasoning | Recommendation |
|----------|------|---------|------------------------|----------------|
| **DeepSeek V3** | Lowest | Medium | Excellent | Primary API |
| **Qwen 2.5 7B** | Free (local) | Lowest | Good | Tier 1 execution |
| **Claude Sonnet** | Medium | Medium | Good | Sentiment analysis |
| **GPT-4o** | High | Medium | Good | Backup |

### 2.3 RL Algorithm Selection

| Algorithm | Best For | Action Space |
|-----------|----------|--------------|
| **PPO** | Position sizing, portfolio management | Continuous |
| **DQN** | Discrete signals (buy/sell/hold) | Discrete |
| **A2C** | Fast training, parallelization | Continuous |

**Decision**: PPO for position sizing optimization, using Stable-Baselines3.

---

## 3. Risk Management Research

### 3.1 Position Sizing

| Method | Formula | Use Case |
|--------|---------|----------|
| **ATR-Based** | Risk / (ATR x Multiplier) | Volatility-adjusted sizing |
| **Kelly Criterion** | (bp - q) / b | Optimal growth (use 0.25x) |
| **Fixed Fractional** | Portfolio x Fixed % | Simple, consistent |

### 3.2 Recommended Parameters

| Parameter | Conservative | Moderate | Aggressive |
|-----------|--------------|----------|------------|
| Max Leverage | 2x | 3x | 5x |
| Risk Per Trade | 0.5% | 1% | 2% |
| Max Drawdown Trigger | 10% | 15% | 20% |
| Kelly Fraction | 0.1x | 0.25x | 0.5x |
| Confidence Threshold | 0.7 | 0.6 | 0.5 |

### 3.3 Circuit Breakers

| Trigger | Action | Cooldown |
|---------|--------|----------|
| Daily loss > 5% | Halt trading | 24 hours |
| Max drawdown > 15% | Reduce position sizes 50% | Until recovery |
| 5 consecutive losses | Pause new trades | 30 minutes |
| Abnormal volatility (ATR > 3x normal) | Reduce leverage to 1x | Until normalized |

---

## 4. Existing Infrastructure Assessment

### 4.1 Database Infrastructure

| Component | Status | Coverage |
|-----------|--------|----------|
| TimescaleDB | Operational | 5-9 years data |
| XRP/USDT | Available | Since 2020-04-30 |
| BTC/USDT | Available | Since 2019-12-19 |
| XRP/BTC | Available | Since 2016-07-19 |
| Continuous Aggregates | 8 timeframes | 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w |
| Retention Policies | Active | 90 days trades, 365 days candles |

### 4.2 Data Collectors

| Collector | Status | Purpose |
|-----------|--------|---------|
| WebSocket DB Writer | Ready | Real-time candles |
| Gap Filler | Ready | Historical data recovery |
| Order Book Collector | Ready | Depth data |
| Private Data Collector | Ready | Trade history |

### 4.3 Local LLM Infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| Ollama | `/media/rese/2tb_drive/ollama_config/` | Ready |
| Qwen 2.5 7B | Via Ollama | Available |

---

## 5. Success Metrics (from Vision)

### 5.1 Portfolio Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Annual Return | > 50% | (Final - Initial) / Initial |
| Max Drawdown | < 20% | Peak-to-trough decline |
| Sharpe Ratio | > 1.5 | Risk-adjusted returns |
| Portfolio Balance | 33/33/33 | BTC/XRP/USDT allocation |
| Hodl Bag Growth | Positive | Accumulated holdings |

### 5.2 System Reliability

| Metric | Target |
|--------|--------|
| Uptime | > 99% |
| Decision Latency (Tier 1) | < 500ms |
| Decision Latency (Tier 2) | < 30s |
| Data Freshness | < 100ms |

---

## 6. Key Design Implications

### 6.1 Architecture Decisions

1. **Multi-Agent System**: Implement specialized agents (Technical, Sentiment, Risk, Trading Decision, Portfolio, Coordinator) with clear authority hierarchy
2. **Multi-Model A/B Testing**: Run all 6 LLMs (GPT, Grok, DeepSeek V3, Claude Sonnet, Claude Opus, Qwen) in parallel for Trading Decision Agent to compare performance
3. **Specialized Model Assignments**: Grok/GPT for sentiment (web search capability), DeepSeek V3/Claude Sonnet for coordination
4. **Rules-Based Risk**: Deterministic risk management engine, not LLM-dependent
5. **Momentum-First**: Primary strategies should be trend-following, not mean reversion
6. **Conservative Execution**: Quality over quantity - fewer, higher-conviction trades

### 6.2 Critical Constraints

1. **Latency Budget**: Tier 1 decisions must complete in < 500ms
2. **Cost Management**: API calls for strategic decisions only (1h+ timeframes)
3. **Risk First**: No trade without valid stop-loss and position sizing
4. **Human Override**: Always maintain manual intervention capability
5. **Auditability**: Log all agent decisions and reasoning traces

### 6.3 Anti-Patterns to Avoid

| Anti-Pattern | Why |
|--------------|-----|
| Mean reversion RSI | Does not work on crypto |
| Hyperactive trading | Gemini's 44 trades vs Claude's 3 |
| Full Kelly position sizing | Too aggressive for crypto volatility |
| Black-box LLM math | Pre-compute indicators, don't ask LLM to calculate |
| Single model dependency | Always have fallback strategies |

---

## 7. Research Gaps & Recommendations

### 7.1 Areas Requiring Further Research

1. **Sentiment Data Sources**: Twitter/X API access, on-chain analytics integration
2. **News Feed Integration**: Real-time news API selection
3. **Bybit Integration**: Futures trading capability (mentioned in vision)
4. **Cross-Exchange Arbitrage**: Opportunity assessment

### 7.2 Recommended Next Steps

1. Design detailed multi-agent architecture with message passing
2. Define LLM prompt templates for each agent
3. Specify risk management rules engine with exact thresholds
4. Design data pipeline from WebSocket to LLM prompt
5. Create evaluation framework with baselines
6. Design UI/dashboard requirements

---

## Sources

- AI Integration Research (ai-integration-research.md)
- BTC/USDT Algo Trading Research (btc-usdt-algo-trading-research.md)
- TensorTrade Deep Dive (tensortrade-deep-dive.md)
- Alpha Arena Agent Trading Deep Dive (alpha-arena-agent-trading-deep-dive.md)
- TripleGain Vision Document
- Kraken API Reference

---

*Document generated: December 2025*
