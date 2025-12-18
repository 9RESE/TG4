# 04 - Solution Strategy

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.10+ | ML ecosystem, async support |
| Database | TimescaleDB | Time-series optimized |
| Message Queue | Redis | Fast, simple pub/sub |
| Containerization | Docker | Reproducible deployment |
| LLM Integration | Multi-provider | Compare performance |

## Architecture Approach

### LLM-as-Strategy Pattern (Nof1.ai)

The system follows the LLM-as-Strategy pattern where:
1. Market context is formatted as structured prompts
2. LLM generates: action, confidence, position size, stop-loss, take-profit, reasoning
3. All decisions are logged for analysis
4. Auto-switch models based on performance thresholds

### Multi-Agent Design

Following TradingAgents/Nof1.ai patterns:

| Agent | Type | Responsibility |
|-------|------|----------------|
| Technical Analysis | Local/Fast | Compute indicators, detect patterns |
| Risk Management | Rules-based | Enforce trading constraints |
| Trading Decision | LLM | Generate trade signals |
| Portfolio Rebalancing | LLM/Rules | Optimize tri-asset allocation |

## Strategy Focus

Based on BTC/USDT research findings:

| Strategy | Role | Notes |
|----------|------|-------|
| Trend-following momentum | Primary | Proven effective in crypto |
| Volatility breakout | Secondary | Bollinger squeeze entries |
| Mean reversion RSI | Avoid | Doesn't work on BTC |

## Quality Approach

| Quality Goal | Approach |
|--------------|----------|
| Reliability | Redundant connections, automatic reconnection |
| Performance | Async I/O, efficient data structures |
| Safety | Rule-based risk manager, kill switch |
| Auditability | Structured logging, decision replay |
