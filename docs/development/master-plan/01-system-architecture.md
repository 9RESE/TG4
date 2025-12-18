# TripleGain System Architecture

**Version:** 1.0
**Date:** December 2025
**Status:** Design Phase

---

## Executive Summary

TripleGain is a multi-asset LLM-assisted trading system designed to autonomously grow holdings across BTC, USDT, and XRP through strategic trading. The architecture extends the existing ws_paper_tester infrastructure with a multi-agent LLM framework following patterns from TradingAgents and Nof1.ai Alpha Arena.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [C4 Model Diagrams](#2-c4-model-diagrams)
3. [Core Components](#3-core-components)
4. [Data Flow Architecture](#4-data-flow-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Infrastructure Design](#6-infrastructure-design)

---

## 1. System Overview

### 1.1 Objectives

| Objective | Target |
|-----------|--------|
| Primary | Grow BTC, USDT, XRP holdings autonomously |
| Trading Pairs | BTC/USDT, XRP/USDT, XRP/BTC |
| Starting Capital | $2,100 (~1,000 USDT + 500 XRP) |
| Target Exchange | Kraken (primary), Bybit (futures expansion) |

### 1.2 Key Design Principles

1. **Modular Architecture** - Independent, swappable components
2. **Multi-LLM Comparison** - Run Claude, GPT-4, Grok, Deepseek in parallel
3. **Safety First** - Conservative risk limits, mandatory stop-losses
4. **Observability** - Full logging of decisions and reasoning
5. **Cost Efficiency** - Local LLM option for routine analysis

### 1.3 System Boundaries

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TripleGain System                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUTS                           OUTPUTS                           │
│  ├── Market Data (WebSocket)      ├── Trade Orders                  │
│  ├── Historical OHLCV             ├── Performance Metrics           │
│  ├── Order Book Depth             ├── Trade Journal + Reasoning     │
│  └── Account Balances             └── Alerts/Notifications          │
│                                                                      │
│  EXTERNAL SYSTEMS                                                   │
│  ├── Kraken Exchange (Primary)                                      │
│  ├── Bybit Exchange (Future)                                        │
│  ├── LLM APIs (Claude, GPT-4, Grok, Deepseek)                      │
│  └── Ollama (Local LLM - Qwen 2.5 7B)                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. C4 Model Diagrams

### 2.1 Level 1: System Context Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM CONTEXT                                   │
└──────────────────────────────────────────────────────────────────────────────┘

                              ┌───────────────┐
                              │    Trader     │
                              │    (User)     │
                              └───────┬───────┘
                                      │ Monitors, Configures
                                      │ Receives Alerts
                                      ▼
┌─────────────────┐          ┌───────────────────┐          ┌─────────────────┐
│                 │          │                   │          │                 │
│  Kraken         │◀────────▶│   TripleGain      │◀────────▶│  LLM Providers  │
│  Exchange       │  Orders  │   Trading System  │  Queries │  (Claude, GPT,  │
│                 │  Data    │                   │          │   Grok, etc.)   │
└─────────────────┘          └───────────────────┘          └─────────────────┘
                                      ▲
                                      │ Historical Data
                                      │ Monitoring
                                      ▼
                             ┌─────────────────┐
                             │  TimescaleDB    │
                             │  + Redis        │
                             └─────────────────┘
```

### 2.2 Level 2: Container Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CONTAINER DIAGRAM                                │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           TripleGain System                                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        AGENT LAYER                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  Technical   │  │    Risk      │  │   Trading    │              │   │
│  │  │  Analysis    │  │  Management  │  │   Decision   │              │   │
│  │  │   Agent      │  │    Agent     │  │    Agent     │              │   │
│  │  │   (Local)    │  │  (Rules)     │  │    (LLM)     │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │         │                 │                 │                        │   │
│  │         └─────────────────┼─────────────────┘                        │   │
│  │                           ▼                                          │   │
│  │                    ┌──────────────┐                                  │   │
│  │                    │  Portfolio   │                                  │   │
│  │                    │ Rebalancing  │                                  │   │
│  │                    │    Agent     │                                  │   │
│  │                    └──────────────┘                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     COORDINATION LAYER                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Agent      │  │   Signal     │  │   Model      │              │   │
│  │  │ Coordinator  │  │  Aggregator  │  │  Selector    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     EXECUTION LAYER                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Order      │  │   Position   │  │   Paper/Live │              │   │
│  │  │   Manager    │  │   Tracker    │  │   Executor   │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DATA LAYER                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Kraken     │  │   Data       │  │  Indicator   │              │   │
│  │  │  WebSocket   │  │   Manager    │  │   Library    │              │   │
│  │  │   Client     │  │              │  │              │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

                    │                              │
                    ▼                              ▼
┌────────────────────────────┐      ┌────────────────────────────┐
│       TimescaleDB          │      │      Redis Cache           │
│  ├── Historical OHLCV      │      │  ├── Session State         │
│  ├── Trade Logs            │      │  ├── Price Cache           │
│  ├── Agent Decisions       │      │  └── Rate Limits           │
│  └── Performance Metrics   │      │                            │
└────────────────────────────┘      └────────────────────────────┘
```

### 2.3 Level 3: Component Diagram (Agent Layer)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AGENT LAYER COMPONENTS                                │
└──────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                     TECHNICAL ANALYSIS AGENT (LOCAL)                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Inputs                                                              │   │
│  │  ├── DataSnapshot (prices, candles, orderbook)                       │   │
│  │  └── Market Regime (from RegimeDetector)                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Processing                                                          │   │
│  │  ├── IndicatorCalculator                                            │   │
│  │  │   ├── Trend: EMA9, EMA21, EMA50, ADX                            │   │
│  │  │   ├── Momentum: RSI14, MACD, ROC                                 │   │
│  │  │   ├── Volatility: ATR14, Bollinger Bands, Keltner               │   │
│  │  │   └── Volume: OBV, VWAP, Volume MA Ratio                        │   │
│  │  ├── PatternRecognizer                                              │   │
│  │  │   ├── Bollinger Squeeze Detection                                │   │
│  │  │   ├── Multi-Timeframe Trend Alignment                           │   │
│  │  │   └── Support/Resistance Levels                                  │   │
│  │  └── SignalGenerator                                                 │   │
│  │      ├── Primary: Trend-Following Momentum                          │   │
│  │      └── Secondary: Volatility Breakout                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Outputs                                                             │   │
│  │  └── TechnicalSignal                                                │   │
│  │      ├── direction: LONG | SHORT | NEUTRAL                          │   │
│  │      ├── strength: float [0, 1]                                     │   │
│  │      ├── entry_price: float                                         │   │
│  │      ├── stop_loss: float                                           │   │
│  │      ├── take_profit: float                                         │   │
│  │      └── indicators: dict (all calculated values)                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                    RISK MANAGEMENT AGENT (RULES-BASED)                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Rule Categories                                                     │   │
│  │                                                                       │   │
│  │  Position Sizing Rules                                               │   │
│  │  ├── Max 1% portfolio risk per trade                                │   │
│  │  ├── Position size = risk_amount / (entry - stop_loss)              │   │
│  │  └── Max 20% portfolio in single position                           │   │
│  │                                                                       │   │
│  │  Stop-Loss Rules                                                     │   │
│  │  ├── Mandatory stop-loss on every trade                             │   │
│  │  ├── Max 2% distance from entry                                     │   │
│  │  └── Trailing stop after 1.5x risk captured                        │   │
│  │                                                                       │   │
│  │  Drawdown Protection                                                 │   │
│  │  ├── 10% drawdown: Pause all trading                                │   │
│  │  ├── 5 consecutive losses: 30-minute cooldown                       │   │
│  │  └── Daily loss limit: 3% of portfolio                              │   │
│  │                                                                       │   │
│  │  Leverage Rules                                                      │   │
│  │  ├── Max leverage: 3x                                               │   │
│  │  ├── Reduce leverage in high volatility                             │   │
│  │  └── No leverage for new/untested strategies                        │   │
│  │                                                                       │   │
│  │  Correlation Rules                                                   │   │
│  │  ├── Max 2 correlated positions                                     │   │
│  │  └── Reduce size when BTC correlation > 0.8                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Outputs                                                             │   │
│  │  └── RiskAssessment                                                 │   │
│  │      ├── approved: bool                                             │   │
│  │      ├── adjusted_size: float                                       │   │
│  │      ├── adjusted_leverage: float                                   │   │
│  │      ├── stop_loss_price: float                                     │   │
│  │      ├── take_profit_price: float                                   │   │
│  │      └── rejection_reason: Optional[str]                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                      TRADING DECISION AGENT (LLM)                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Model Options (Parallel Execution)                                  │   │
│  │  ├── Claude Sonnet 4.5 (Primary - Conservative)                     │   │
│  │  ├── GPT-4o (Backup - Balanced)                                     │   │
│  │  ├── Grok 4 (Comparison - Competitive)                              │   │
│  │  ├── Deepseek V3 (Comparison - Best Alpha Arena)                   │   │
│  │  └── Qwen 2.5 7B (Local - Cost Efficient)                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Processing Pipeline                                                 │   │
│  │  1. Receive MarketContext from Coordinator                          │   │
│  │  2. Format structured prompt (system + user)                        │   │
│  │  3. Query LLM with timeout (10s max)                               │   │
│  │  4. Parse JSON response                                             │   │
│  │  5. Validate output schema                                          │   │
│  │  6. Log reasoning for analysis                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Outputs                                                             │   │
│  │  └── TradingDecision                                                │   │
│  │      ├── action: LONG | SHORT | HOLD | CLOSE                        │   │
│  │      ├── symbol: str                                                │   │
│  │      ├── confidence: float [0, 1]                                   │   │
│  │      ├── position_size_pct: float [0, 20]                          │   │
│  │      ├── leverage: int [1, 3]                                       │   │
│  │      ├── entry_price: float                                         │   │
│  │      ├── stop_loss: float                                           │   │
│  │      ├── take_profit: float                                         │   │
│  │      ├── invalidation: str (early exit condition)                   │   │
│  │      ├── reasoning: str (2-3 sentences)                             │   │
│  │      └── model_used: str                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                     PORTFOLIO REBALANCING AGENT                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Objectives                                                          │   │
│  │  ├── Optimize tri-asset growth (BTC, USDT, XRP)                     │   │
│  │  ├── Maintain minimum reserves in each asset                        │   │
│  │  └── Exploit cross-pair arbitrage opportunities                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Rebalancing Triggers                                                │   │
│  │  ├── Asset allocation drift > 10% from target                       │   │
│  │  ├── Single asset comprises > 60% of portfolio                      │   │
│  │  ├── Significant profit in one asset (>20% gain)                   │   │
│  │  └── Market regime change detected                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Outputs                                                             │   │
│  │  └── RebalanceRecommendation                                        │   │
│  │      ├── source_asset: str                                          │   │
│  │      ├── target_asset: str                                          │   │
│  │      ├── amount: float                                              │   │
│  │      ├── priority: HIGH | MEDIUM | LOW                              │   │
│  │      └── reasoning: str                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Level 4: Code Diagram (Trading Decision Agent)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TRADING DECISION AGENT - CODE STRUCTURE                    │
└──────────────────────────────────────────────────────────────────────────────┘

ws_paper_tester/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                 # Abstract base class
│   │   ├── class BaseAgent(ABC)
│   │   │   ├── @abstractmethod process(context) -> AgentOutput
│   │   │   ├── @abstractmethod reset()
│   │   │   └── log_decision(decision, context)
│   │
│   ├── technical_agent.py            # Local technical analysis
│   │   ├── class TechnicalAgent(BaseAgent)
│   │   │   ├── __init__(indicator_config, signal_rules)
│   │   │   ├── calculate_indicators(snapshot) -> dict
│   │   │   ├── detect_patterns(indicators) -> list[Pattern]
│   │   │   ├── generate_signal(patterns, regime) -> TechnicalSignal
│   │   │   └── process(context) -> TechnicalSignal
│   │
│   ├── risk_agent.py                 # Rules-based risk management
│   │   ├── class RiskAgent(BaseAgent)
│   │   │   ├── __init__(risk_config)
│   │   │   ├── calculate_position_size(signal, portfolio) -> float
│   │   │   ├── validate_stop_loss(entry, stop) -> bool
│   │   │   ├── check_drawdown_limits(portfolio) -> bool
│   │   │   ├── assess_correlation_risk(positions) -> float
│   │   │   └── process(context) -> RiskAssessment
│   │
│   ├── trading_agent.py              # LLM-based trading decisions
│   │   ├── class TradingAgent(BaseAgent)
│   │   │   ├── __init__(model_configs: list[LLMConfig])
│   │   │   ├── format_prompt(context) -> tuple[str, str]
│   │   │   ├── query_llm(model, prompt) -> str
│   │   │   ├── parse_response(response) -> TradingDecision
│   │   │   ├── validate_decision(decision) -> bool
│   │   │   └── process(context) -> TradingDecision
│   │
│   ├── rebalancing_agent.py          # Portfolio optimization
│   │   ├── class RebalancingAgent(BaseAgent)
│   │   │   ├── __init__(target_allocation, thresholds)
│   │   │   ├── calculate_current_allocation(portfolio) -> dict
│   │   │   ├── detect_drift(current, target) -> float
│   │   │   ├── find_arbitrage_opportunities() -> list
│   │   │   └── process(context) -> RebalanceRecommendation
│   │
│   └── coordinator.py                # Multi-agent orchestration
│       ├── class AgentCoordinator
│       │   ├── __init__(agents: list[BaseAgent], config)
│       │   ├── build_context(snapshot, portfolio) -> MarketContext
│       │   ├── run_agents(context) -> dict[str, AgentOutput]
│       │   ├── aggregate_signals(outputs) -> ConsensusSignal
│       │   ├── select_model(performance_history) -> str
│       │   └── execute_cycle() -> Optional[Order]
│
├── prompts/
│   ├── __init__.py
│   ├── system_prompts.py             # System prompt templates
│   │   ├── TRADING_SYSTEM_PROMPT
│   │   ├── CONSERVATIVE_SYSTEM_PROMPT
│   │   └── AGGRESSIVE_SYSTEM_PROMPT
│   │
│   ├── user_prompts.py               # Dynamic prompt builders
│   │   ├── build_market_context_prompt(context) -> str
│   │   ├── build_position_context_prompt(positions) -> str
│   │   └── build_recent_trades_prompt(trades) -> str
│   │
│   └── output_parsers.py             # JSON output parsing
│       ├── parse_trading_decision(response) -> TradingDecision
│       ├── validate_schema(data, schema) -> bool
│       └── extract_reasoning(response) -> str
│
├── llm/
│   ├── __init__.py
│   ├── client.py                     # Multi-provider LLM client
│   │   ├── class LLMClient
│   │   │   ├── __init__(providers: dict)
│   │   │   ├── query(model, system, user, timeout) -> str
│   │   │   ├── batch_query(models, prompt) -> dict[str, str]
│   │   │   └── get_cost_estimate(model, tokens) -> float
│   │
│   ├── providers/
│   │   ├── anthropic.py              # Claude integration
│   │   ├── openai.py                 # GPT integration
│   │   ├── xai.py                    # Grok integration
│   │   ├── deepseek.py               # Deepseek integration
│   │   └── ollama.py                 # Local LLM (Qwen)
│   │
│   └── model_selector.py             # Performance-based selection
│       ├── class ModelSelector
│       │   ├── __init__(models, performance_window)
│       │   ├── update_performance(model, result)
│       │   ├── get_best_model() -> str
│       │   └── should_switch(current, threshold) -> bool
│
└── types/
    ├── __init__.py
    ├── agent_types.py                # Agent-specific types
    │   ├── @dataclass MarketContext
    │   ├── @dataclass TechnicalSignal
    │   ├── @dataclass RiskAssessment
    │   ├── @dataclass TradingDecision
    │   └── @dataclass RebalanceRecommendation
    │
    └── llm_types.py                  # LLM-specific types
        ├── @dataclass LLMConfig
        ├── @dataclass LLMResponse
        └── @dataclass ModelPerformance
```

---

## 3. Core Components

### 3.1 Data Layer Components

| Component | Responsibility | Technology |
|-----------|----------------|------------|
| **KrakenWSClient** | Real-time market data via WebSocket | Existing implementation |
| **DataManager** | Historical data management, caching | TimescaleDB + Redis |
| **IndicatorLibrary** | 25+ technical indicators | Existing implementation |
| **RegimeDetector** | Market state classification | Existing CompositeScorer |

### 3.2 Agent Layer Components

| Agent | Type | Latency Target | Cost |
|-------|------|----------------|------|
| **Technical Agent** | Local Python | <100ms | Free |
| **Risk Agent** | Local Python | <50ms | Free |
| **Trading Agent** | LLM API | <10s | $0.01-0.10/query |
| **Rebalancing Agent** | Hybrid | <5s | ~$0.01/query |

### 3.3 Execution Layer Components

| Component | Responsibility |
|-----------|----------------|
| **OrderManager** | Order creation, modification, cancellation |
| **PositionTracker** | Open positions, P&L tracking |
| **PaperExecutor** | Simulated execution (existing) |
| **LiveExecutor** | Real exchange execution (Phase 3+) |

---

## 4. Data Flow Architecture

### 4.1 Main Trading Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING LOOP (100ms cycle)                         │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐
  │   START     │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1. DATA INGESTION                                                       │
  │     └── KrakenWSClient.get_snapshot() → DataSnapshot                    │
  └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  2. TECHNICAL ANALYSIS (Parallel for each pair)                         │
  │     ├── TechnicalAgent.process(BTC/USDT) → TechnicalSignal              │
  │     ├── TechnicalAgent.process(XRP/USDT) → TechnicalSignal              │
  │     └── TechnicalAgent.process(XRP/BTC) → TechnicalSignal               │
  └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  3. SIGNAL FILTER (Only proceed if signal.strength > threshold)         │
  │     └── Skip to step 8 if no actionable signals                        │
  └─────────────────────────────────────────────────────────────────────────┘
         │ (Has signal)
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  4. RISK PRE-CHECK                                                       │
  │     ├── RiskAgent.check_drawdown_limits()                               │
  │     ├── RiskAgent.check_cooldown_period()                               │
  │     └── Skip to step 8 if risk limits exceeded                         │
  └─────────────────────────────────────────────────────────────────────────┘
         │ (Approved)
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  5. LLM TRADING DECISION                                                 │
  │     ├── Coordinator.build_context(snapshot, signals, portfolio)         │
  │     ├── TradingAgent.process(context) → TradingDecision                 │
  │     └── Skip to step 8 if decision.confidence < 0.6                    │
  └─────────────────────────────────────────────────────────────────────────┘
         │ (High confidence)
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  6. RISK VALIDATION                                                      │
  │     ├── RiskAgent.process(decision) → RiskAssessment                    │
  │     ├── Adjust position size if needed                                  │
  │     └── Reject if assessment.approved == False                          │
  └─────────────────────────────────────────────────────────────────────────┘
         │ (Approved)
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  7. ORDER EXECUTION                                                      │
  │     ├── OrderManager.create_order(decision, assessment)                 │
  │     ├── Executor.submit_order(order)                                    │
  │     └── PositionTracker.update(order)                                   │
  └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  8. LOGGING & MONITORING                                                 │
  │     ├── Log decision + reasoning to TimescaleDB                        │
  │     ├── Update Prometheus metrics                                       │
  │     └── Send alerts if configured                                       │
  └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────┐
  │   SLEEP     │
  │  (100ms)    │
  └─────────────┘
```

### 4.2 LLM Query Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM QUERY FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  MarketContext
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  PROMPT CONSTRUCTION                                                     │
  │  ├── System Prompt (trading rules, constraints, output format)          │
  │  └── User Prompt (market data, indicators, positions, recent trades)    │
  └─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  MODEL SELECTION                                                         │
  │  ├── Check performance history                                          │
  │  ├── Select best-performing model OR                                    │
  │  └── Query multiple models in parallel (comparison mode)                │
  └─────────────────────────────────────────────────────────────────────────┘
       │
       ├──────────────────┬──────────────────┬──────────────────┐
       ▼                  ▼                  ▼                  ▼
  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │  Claude  │      │  GPT-4   │      │   Grok   │      │ Deepseek │
  └────┬─────┘      └────┬─────┘      └────┬─────┘      └────┬─────┘
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  RESPONSE PARSING                                                        │
  │  ├── Extract JSON from response                                         │
  │  ├── Validate against schema                                            │
  │  └── Handle parsing errors (retry or skip)                              │
  └─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  DECISION AGGREGATION (if multi-model)                                  │
  │  ├── Weighted voting based on confidence                                │
  │  ├── Consensus threshold: 3/4 models agree                              │
  │  └── Track individual model performance                                 │
  └─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                          TradingDecision
```

---

## 5. Technology Stack

### 5.1 Core Technologies

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Runtime** | Python | 3.11+ | Primary language |
| **Database** | TimescaleDB | 2.x | Time-series data storage |
| **Cache** | Redis | 7.x | State caching, rate limiting |
| **Containers** | Docker | 24.x | Deployment |
| **Orchestration** | Docker Compose | 2.x | Service orchestration |

### 5.2 LLM Providers

| Provider | Model | Cost (input/output) | Use Case |
|----------|-------|---------------------|----------|
| **Anthropic** | Claude Sonnet 4.5 | $3/$15 per 1M tokens | Primary trading decisions |
| **OpenAI** | GPT-4o | $5/$15 per 1M tokens | Backup, comparison |
| **xAI** | Grok 4 | ~$5/$15 per 1M tokens | Comparison |
| **Deepseek** | V3 | ~$0.14/$0.28 per 1M tokens | Cost-efficient comparison |
| **Ollama** | Qwen 2.5 7B | Free (local) | Routine analysis |

### 5.3 Python Dependencies

```
# Core
asyncio>=3.11
aiohttp>=3.8
websockets>=11.0
pydantic>=2.0

# Data
pandas>=2.0
numpy>=1.24
ta-lib>=0.4.28

# Database
asyncpg>=0.28
redis>=5.0

# LLM
anthropic>=0.20
openai>=1.10
httpx>=0.25  # For custom API clients

# Monitoring
prometheus-client>=0.18
structlog>=23.0

# Testing
pytest>=7.4
pytest-asyncio>=0.21
```

---

## 6. Infrastructure Design

### 6.1 Docker Compose Architecture

```yaml
# docker-compose.yml
version: '3.8'

services:
  triplegain:
    build: .
    environment:
      - KRAKEN_API_KEY=${KRAKEN_API_KEY}
      - KRAKEN_SECRET=${KRAKEN_SECRET}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@timescaledb:5432/triplegain
      - REDIS_URL=redis://redis:6379
    depends_on:
      - timescaledb
      - redis
      - ollama
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=triplegain
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

volumes:
  timescaledb_data:
  redis_data:
  ollama_models:
  prometheus_data:
  grafana_data:
```

### 6.2 Database Schema

```sql
-- TimescaleDB Schema

-- Trade decisions and reasoning
CREATE TABLE trade_decisions (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    model VARCHAR(50) NOT NULL,
    action VARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    position_size DECIMAL(20,8),
    entry_price DECIMAL(20,8),
    stop_loss DECIMAL(20,8),
    take_profit DECIMAL(20,8),
    reasoning TEXT,
    raw_response JSONB,
    executed BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (id, timestamp)
);
SELECT create_hypertable('trade_decisions', 'timestamp');

-- Model performance tracking
CREATE TABLE model_performance (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    model VARCHAR(50) NOT NULL,
    decision_id INTEGER REFERENCES trade_decisions(id),
    entry_price DECIMAL(20,8),
    exit_price DECIMAL(20,8),
    pnl DECIMAL(20,8),
    pnl_pct DECIMAL(10,6),
    holding_time_minutes INTEGER,
    PRIMARY KEY (id, timestamp)
);
SELECT create_hypertable('model_performance', 'timestamp');

-- Portfolio snapshots
CREATE TABLE portfolio_snapshots (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    btc_balance DECIMAL(20,8),
    usdt_balance DECIMAL(20,8),
    xrp_balance DECIMAL(20,8),
    total_value_usdt DECIMAL(20,8),
    PRIMARY KEY (id, timestamp)
);
SELECT create_hypertable('portfolio_snapshots', 'timestamp');

-- Indexes
CREATE INDEX idx_decisions_symbol ON trade_decisions (symbol, timestamp DESC);
CREATE INDEX idx_decisions_model ON trade_decisions (model, timestamp DESC);
CREATE INDEX idx_performance_model ON model_performance (model, timestamp DESC);
```

### 6.3 Monitoring Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GRAFANA DASHBOARD LAYOUT                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┬─────────────────────────────────┐
│       PORTFOLIO VALUE           │        ASSET ALLOCATION         │
│   [Line chart: USD value]       │   [Pie chart: BTC/USDT/XRP]    │
├─────────────────────────────────┴─────────────────────────────────┤
│                    MODEL PERFORMANCE COMPARISON                    │
│   [Bar chart: PnL by model - Claude, GPT, Grok, Deepseek]        │
├─────────────────────────────────┬─────────────────────────────────┤
│     RECENT TRADES               │     OPEN POSITIONS              │
│   [Table: last 10 trades]       │   [Table: current positions]   │
├─────────────────────────────────┼─────────────────────────────────┤
│     WIN RATE BY PAIR            │     CONFIDENCE DISTRIBUTION     │
│   [Gauge: BTC, XRP pairs]       │   [Histogram: 0.6-1.0]         │
├─────────────────────────────────┴─────────────────────────────────┤
│                        DECISION LATENCY                           │
│   [Time series: LLM response times by provider]                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Sequence Diagrams

### A.1 Trade Execution Sequence

```
┌─────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐
│WebSocket│ │Technical │ │  Risk   │ │ Trading  │ │  LLM    │ │ Executor │
│ Client  │ │  Agent   │ │  Agent  │ │  Agent   │ │ Client  │ │          │
└────┬────┘ └────┬─────┘ └────┬────┘ └────┬─────┘ └────┬────┘ └────┬─────┘
     │           │            │           │            │           │
     │ snapshot  │            │           │            │           │
     │──────────>│            │           │            │           │
     │           │            │           │            │           │
     │           │ indicators │           │            │           │
     │           │───────────>│           │            │           │
     │           │            │           │            │           │
     │           │  approved  │           │            │           │
     │           │<───────────│           │            │           │
     │           │            │           │            │           │
     │           │    context │           │            │           │
     │           │────────────────────────>│            │           │
     │           │            │           │            │           │
     │           │            │           │  query     │           │
     │           │            │           │───────────>│           │
     │           │            │           │            │           │
     │           │            │           │  response  │           │
     │           │            │           │<───────────│           │
     │           │            │           │            │           │
     │           │            │ validate  │            │           │
     │           │            │<──────────│            │           │
     │           │            │           │            │           │
     │           │            │ approved  │            │           │
     │           │            │──────────>│            │           │
     │           │            │           │            │           │
     │           │            │           │    order   │           │
     │           │            │           │───────────────────────>│
     │           │            │           │            │           │
     │           │            │           │ confirmation           │
     │           │            │           │<───────────────────────│
     │           │            │           │            │           │
```

---

*Document Version: 1.0*
*Last Updated: December 2025*
*Next Review: After Phase 1 completion*
