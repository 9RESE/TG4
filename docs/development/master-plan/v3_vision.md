PROJECT: Multi-Asset LLM-Assisted Trading System (Codename: TripleGain)
Objective: Build an autonomous trading system that uses LLMs as decision-making agents to grow holdings across BTC, USDT, and XRP through strategic trading of BTC/USDT, XRP/USDT, and XRP/BTC pairs.
Starting Capital:
- 1,000 USDT
- 500 XRP (~$1,100 equivalent at current prices)
- Total: ~$2,100
Target Exchange: Kraken (existing WebSocket infrastructure) with potential expansion to Bybit for futures
Architecture Requirements:
1. Extend existing ws_paper_tester infrastructure (TimescaleDB, indicators, regime detection)
2. Multi-LLM comparison framework - Test Claude, Grok, GPT-4, Deepseek V3 with identical prompts
3. Multi-agent design following TradingAgents/Nof1.ai patterns:
- Technical Analysis Agent (local, fast)
- Risk Management Agent (rules-based)
- Trading Decision Agent (LLM)
- Portfolio Rebalancing Agent (optimize tri-asset growth)
LLM Integration Approach: LLM-as-Strategy (Nof1.ai pattern)
- Format market context as structured prompt
- LLM generates: action, confidence, position size, stop-loss, take-profit, reasoning
- Log all decisions for analysis and self-reflection loop
- Auto-switch models based on performance thresholds
Trading Constraints:
- Max leverage: 3x (conservative per Alpha Arena insights)
- Risk per trade: 1% of portfolio
- Mandatory stop-loss: 2% max distance from entry
- Min risk:reward: 2:1
- Max drawdown trigger: 10% (pause trading)
- Confidence threshold: 0.6 minimum to execute
- Cooldown: 30 minutes between trades per pair
Strategy Focus (per BTC/USDT research):
- Primary: Trend-following momentum (proven effective in crypto)
- Secondary: Volatility breakout (Bollinger squeeze entries)
- Avoid: Mean reversion RSI (doesn't work on BTC)
Infrastructure (Docker-based):
- TimescaleDB for historical data and trade logs
- Local LLM option (Qwen 2.5 7B via Ollama) for cost efficiency
- Redis for caching and state management
- Prometheus/Grafana for monitoring
Development Phases:
1. Phase 1: Foundation - LLM strategy module, prompt engineering, backtesting
2. Phase 2: Paper Trading - Minimum 30 days paper trading with all LLMs
3. Phase 3: Micro-Live - $100 USDT live with best-performing LLM
4. Phase 4: Scale - Gradual increase based on performance

Success Metrics:
- Sharpe Ratio > 1.5
- Max Drawdown < 15%
- Win Rate > 50%
- All three assets increasing over 90-day rolling window
Research Documents to Reference:
- docs/research/v3/ai-integration-research.md
- docs/research/v3/alpha-arena-agent-trading-deep-dive.md
- docs/research/v3/btc-usdt-algo-trading-research.md
- docs/research/v3/freqtrade-deep-dive.md
- docs/research/v3/tensortrade-deep-dive.md
Deliverables:
1. System architecture document with C4 diagrams
2. LLM prompt templates (system + user prompts)
3. Multi-agent coordination design
4. Risk management rules engine spec
5. Implementation roadmap with dependencies
6. Evaluation framework for comparing LLM performance
Put your documentation in docs/research/v3/master-plan/