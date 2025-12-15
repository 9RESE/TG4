update the docs with the recent work and ALL the changes in git. Ensure documentation complies with the documentation standards and expectations outlined in the claude.md file. Then commit.





- wavetrend /home/rese/Documents/rese/trading-bots/grok-4_1/src/strategies/wavetrend
- whale_sentiment /home/rese/Documents/rese/trading-bots/grok-4_1/src/strategies/whale_sentiment

i want to add the whale_sentiment from /home/rese/Documents/rese/trading-bots/grok-4_1/src/strategies/whale_sentiment/ to the ws_paper_tester/. as we are rewriting it for the platform I want to do deep research on the strategy and its use with XRP/usdt, btc/usdt, and xrp/btc. I then want you to make a document on your findings, recommendations, and implementation plan. We need to ensure the new strategy complies with ws_paper_tester/docs/development/strategy-development-guide.md and integrates with existing ws_paper_tester/ infrastructure



---
Task: Research and Plan Whale Sentiment Strategy for ws_paper_tester

Context

- Source Strategy: Legacy implementation requiring platform adaptation
  - src/strategies/whale_sentiment/strategy.py
- Target Location: ws_paper_tester/strategies/whale_sentiment/
- Must comply with: ws_paper_tester/docs/development/strategy-development-guide.md
- Must integrate with: Existing ws_paper_tester infrastructure (modular structure: config, indicators, regimes, signals, lifecycle, risk, exits)

Strategy Specifications

- Name: Whale Sentiment
- Timeframes: 1h primary (with 1m/5m tick-level execution)
- Style: Contrarian sentiment + volume spike detection (whale proxy)
- Pairs: XRP/USDT, BTC/USDT, XRP/BTC
- Core Logic: Volume spikes as whale activity proxy, RSI/price deviation as fear-greed proxy, contrarian entry on extreme sentiment

Required Research Areas

1. Whale tracking and sentiment analysis fundamentals
  - Academic foundations (behavioral finance, contrarian investing, smart money theory)
  - Volume spike analysis as institutional activity proxy
  - Fear & Greed Index methodology and effectiveness
  - RSI as sentiment indicator vs momentum indicator
2. Pair-specific characteristics:
  - XRP/USDT: Volume patterns, whale threshold calibration, retail vs institutional behavior
  - BTC/USDT: Institutional activity signatures, volume spike accuracy for whale detection
  - XRP/BTC: Cross-pair sentiment divergence, correlation with BTC dominance cycles
3. Source code analysis:
  - Extract core logic from strategy.py (WhaleSentiment class)
  - Map to ws_paper_tester module structure (config, indicators, signals, regimes, lifecycle)
  - Identify components requiring WebSocket adaptation (hourly → tick-based execution)
  - External data integration patterns (Whale Alert, Glassnode, social sentiment)
4. Entry/exit signal criteria for sentiment + whale proxy
  - Composite signal aggregation methodology
  - Confidence thresholds (current: 0.55)
  - Contrarian vs momentum mode switching
5. Risk management:
  - Position sizing based on sentiment confidence
  - Stop-loss placement for contrarian entries (inherently counter-trend)
  - Regime-based exposure (extreme fear vs neutral markets)
  - False signal protection (volume spikes without follow-through)
6. Known pitfalls and failure modes:
  - Volume spike false positives (exchange manipulation, wash trading)
  - Sentiment divergence (external data vs proxy indicators)
  - Contrarian trap (knife-catching in trending markets)
  - Cross-pair correlation breakdown

Deliverable

Create a research document at:
ws_paper_tester/docs/development/review/whale_sentiment/master-plan-v1.0.md

Structure:
- Executive Summary
- Research Findings (per area above)
- Source Code Analysis (legacy → new mapping)
- Pair-Specific Analysis (XRP/USDT, BTC/USDT, XRP/BTC)
- Recommended Approach
- Development Plan (phases, no code)
- Compliance Checklist (vs strategy-development-guide.md v1.0)

Constraints

- Documentation only - no implementation code
- Focus on adaptation to WebSocket tick environment, not blind copy
- Address external data dependency: strategy must function with proxy indicators only
- Consider circuit breaker and volatility regime requirements from existing strategies



# REUSABLE STRATEGY PROMPTS

## Task: Research and Plan Grid RSI Reversion Strategy for ws_paper_tester
### Context
- **Source Strategy:** Legacy implementation being ported
- `src/strategies/grid_base.py`
- `src/strategies/grid_wrappers.py`
- `src/grid_ensemble_orchestrator.py`
- **Target Location:** `ws_paper_tester/strategies/grid_rsi_reversion/`
- **Must comply with:** `ws_paper_tester/docs/development/strategy-development-guide.md`
- **Must integrate with:** Existing ws_paper_tester infrastructure
### Strategy Specifications
- **Name:** Grid RSI Reversion
- **Style:** Grid-based mean reversion with RSI filtering
- **Pairs:** XRP/USDT, BTC/USDT, XRP/BTC
### Required Research Areas
1. Grid trading fundamentals and mean reversion theory
    - Academic foundations (Ornstein-Uhlenbeck, half-life concepts)
    - Grid spacing optimization approaches
    - RSI reversion confluence strategies
2. Pair-specific characteristics:
    - XRP/USDT: Volatility patterns, typical range behavior, liquidity
    - BTC/USDT: Grid suitability, trending vs ranging frequency
    - XRP/BTC: Cross-pair dynamics, correlation factors
3. Source code analysis:
    - Extract core logic from legacy files
    - Identify components that translate vs require rewrite
    - Document ensemble orchestration patterns
4. Entry/exit signal criteria for grid + RSI combination
5. Risk management:
    - Grid exposure accumulation limits
    - Stop-loss placement for grid strategies
    - Capital allocation across grid levels
6. Known pitfalls and failure modes for grid strategies
### Deliverable
Create a research document at:
`ws_paper_tester/docs/development/review/grid_rsi_reversion/master-plan-v1.0.md`
Structure:
- Executive Summary
- Research Findings (per area above)
- Source Code Analysis (legacy → new mapping)
- Pair-Specific Analysis
- Recommended Approach
- Development Plan (phases, no code)
- Compliance Checklist (vs strategy-development-guide.md v2.0)
### Constraints
- Documentation only - no implementation code
- Focus on adaptation, not blind copy

## Task: Research and plan a new strategy for the ws_paper_tester framework.
### Context
- Location: ws_paper_tester/strategies/
- Existing strategies have been refined and are working
- Must comply with: ws_paper_tester/docs/development/strategy-development-guide.md
- Must integrate with existing ws_paper_tester/ infrastructure
- Docs: ws_paper_tester/docs/development/features/momentum_scalping ws_paper_tester/docs/development/review/momentum_scalping
### Strategy Specifications
- Name: Momentum Scalping
- Timeframes: 1m and 5m
- Style: Quick momentum bursts
- Pairs: XRP/USDT, BTC/USDT, XRP/BTC
### Required Research Areas
1. Momentum scalping fundamentals and best practices
2. Optimal indicators for 1m-5m momentum detection
3. Pair-specific characteristics:
    - XRP/USDT: Volatility patterns, typical spread, liquidity
    - BTC/USDT: Market structure, momentum behavior
    - XRP/BTC: Cross-pair dynamics, correlation factors
4. Entry/exit signal criteria for scalping timeframes
5. Risk management specific to momentum scalping
6. Known pitfalls and failure modes
### Deliverable
Create a research document at:
ws_paper_tester/docs/development/review/momentum_scalping/master-plan-v1.0.md
Structure:
- Executive Summary
- Research Findings (per area above)
- Pair-Specific Analysis
- Recommended Approach
- Development Plan (phases, no code)
- Compliance Checklist (vs strategy-development-guide.md)
Constraint: Documentation only - no implementation code.
---





## Deep Review Prompt (Copy & Fill Placeholders)
```
## Task: Deep Review of grid_rsi_reversion Strategy
### Scope
- **Strategy:** `ws_paper_tester/strategies/grid_rsi_reversion/`
- **Docs:** `ws_paper_tester/docs/development/features/grid_rsi_reversion/` (if exists)
- **Pairs:** XRP/USDT, BTC/USDT, XRP/BTC
### Review Requirements
#### 1. Strategy Research
Research the core theory behind this strategy type:
- Academic foundations and mathematical models
- Effectiveness in crypto markets specifically
- Optimal parameter selection from literature
- Market conditions where this strategy fails
#### 2. Pair-Specific Analysis
For each supported pair, analyze:
- Current market characteristics (volatility, liquidity, spread)
- Historical performance tendencies
- Optimal parameter recommendations
- Suitability assessment
#### 3. Code Review
Review against `ws_paper_tester/docs/development/strategy-development-guide.md` v2.0:
- Section 15: Volatility Regime Classification
- Section 16: Circuit Breaker Protection
- Section 17: Signal Rejection Tracking
- Section 18: Trade Flow Confirmation
- Section 22: Per-Symbol Configuration (SYMBOL_CONFIGS)
- Section 24: Correlation Monitoring
- R:R ratio (must be >= 1:1)
- Position sizing (USD-based)
- Indicator logging on all code paths
#### 4. Strategy Logic Analysis
- Entry/exit signal conditions
- Stop-loss and take-profit implementation
- Edge cases and failure modes
- Regime change handling
### Deliverable
Create documentation at: `ws_paper_tester/docs/development/review/grid_rsi_reversion/`
Include:
1. **Executive Summary** - Overall assessment and risk level
2. **Research Findings** - Academic and industry research
3. **Pair Analysis** - Market characteristics per pair
4. **Compliance Matrix** - Checklist against guide v2.0
5. **Critical Findings** - Prioritized (CRITICAL/HIGH/MEDIUM/LOW)
6. **Recommendations** - REC-XXX with priority and effort
7. **Research References** - Academic papers and sources
### Constraints
- Do NOT modify any code
- Do NOT include code snippets
- Reference specific line numbers for issues
```
---

## Implementation Prompt (Copy & Fill Placeholders)
```
## Task: Implement Review Findings for grid_rsi_reversion Strategy
### Files
- **Strategy:** `ws_paper_tester/strategies/grid_rsi_reversion/`
- **Review:** `ws_paper_tester/docs/development/review/grid_rsi_reversion/deep-review-v2.0.md`
- **Guide:** `ws_paper_tester/docs/development/strategy-development-guide.md`
### Instructions
1. Read the review document - identify all recommendations (REC-XXX)
2. Read the strategy development guide for compliance requirements
3. Categorize by priority:
   - **CRITICAL/HIGH** → Must implement
   - **MEDIUM + LOW effort** → Should implement
   - **LOW or HIGH effort** → Document for future
4. For each implementation:
   - Update code with comments referencing REC-ID
   - Add new config parameters to CONFIG/SYMBOL_CONFIGS
   - Add rejection/exit reasons to enums if needed
   - Update version history in docstring
5. Update STRATEGY_VERSION (semver)
### Acceptance Criteria
- All CRITICAL/HIGH, MEDIUM,and LOW findings addressed or justified
- ALL Recommendations implemented or explained why not
- All existing tests pass
- Indicators populated on all code paths
- R:R ratio >= 1:1 maintained
- Version history updated
### Output
Provide:
1. Summary of changes by REC-ID
2. Deferred changes and why
3. New compliance score estimate
4. New risks introduced
6. update the docs with the recent work and ALL the changes in git. Ensure documentation complies with the documentation standards and expectations outlined in the claude.md file.
7. Commit changes with documentation updates
```
---

# STRATEGY-SPECIFIC NOTES (Reference During Review/Implementation)
## Ratio Trading
- Pairs: XRP/BTC only (USDT pairs NOT suitable)
- Key concepts: cointegration, Engle-Granger, Johansen test, Z-score
- Watch for: correlation breakdown, dual-asset accumulation
## Mean Reversion
- Pairs: XRP/USDT, BTC/USDT, XRP/BTC
- Key concepts: Ornstein-Uhlenbeck, half-life, Bollinger+RSI
- Watch for: band walk, trending markets
## Order Flow
- Pairs: XRP/USDT, BTC/USDT
- Key concepts: VPIN, trade tape, aggressor detection, session awareness
- Watch for: asymmetric thresholds, position decay stages
## Market Making
- Pairs: XRP/USDT, BTC/USDT, XRP/BTC
- Key concepts: Avellaneda-Stoikov, inventory skew, micro-price
- Watch for: minimum spread > 0.2% (fees), stale inventory decay
## Momentum Scalping (v2.1.1 - 2025-12-14)
- Pairs: XRP/USDT, BTC/USDT, XRP/BTC
- Key concepts: RSI 7-8, MACD (6,13,5), EMA 8/21/50 ribbon, volume spikes
- v2.0 additions:
  - REC-001: XRP/BTC correlation pause (threshold 0.50)
  - REC-002: 5m trend filter (50 EMA alignment)
  - REC-003: ADX filter for BTC (threshold 25)
  - REC-004: Regime-based RSI bands (75/25 in HIGH vol)
- v2.1.0 additions (Deep Review v2.0 implementation):
  - REC-001: Raised correlation pause threshold to 0.60 (was 0.50)
  - REC-002: Raised ADX threshold for BTC to 30 (was 25)
  - REC-003: Changed RSI period for XRP to 8 (was 7)
  - REC-005: ATR-based trailing stops (1.5x ATR after 0.4% profit)
  - REC-006: DST handling documented in regimes.py
  - REC-007: Trade flow confirmation (imbalance threshold 0.1)
  - REC-008: Correlation lookback increased to 100 (was 50)
  - REC-009: Breakeven momentum exit option (disabled by default)
  - REC-010: Structured logging using Python logging module
- v2.1.1 additions (REC-012/REC-013 Monitoring):
  - REC-012: XRP Independence Monitoring (monitoring.py)
    - CorrelationMonitor tracks XRP-BTC correlation for weekly review
    - Escalation triggers: <0.70 for 30 days, or >50% pause rate
    - State persisted to logs/monitoring/monitoring_state.json
    - Weekly reports: logs/monitoring/correlation_report_*.json
  - REC-013: Market Sentiment Monitoring
    - SentimentMonitor tracks Fear & Greed Index
    - Classifications: Extreme Fear/Fear/Neutral/Greed/Extreme Greed
    - Prolonged extreme alerts (7+ consecutive days)
    - Volatility expansion signals for regime awareness
- Watch for: correlation breakdown on XRP/BTC, BTC trending markets (ADX>30)

## WaveTrend Oscillator (v1.0.0 - 2025-12-14)
- Pairs: XRP/USDT, BTC/USDT, XRP/BTC
- Key concepts: WaveTrend (LazyBear) dual-line crossover, zone-based filtering (OB/OS), divergence confirmation
- Core logic:
  - WT1/WT2 crossover signals in overbought/oversold zones
  - Configurable zone thresholds per symbol
  - Bullish divergence bonus in oversold zones
  - Bearish divergence bonus in overbought zones
- Entry:
  - Long: Bullish crossover (WT1 > WT2) from/in oversold zone
  - Short: Bearish crossover (WT1 < WT2) from/in overbought zone
- Exit:
  - Crossover reversal (primary)
  - Extreme zone profit taking
  - Stop loss / Take profit
- Config highlights:
  - wt_channel_length: 10 (ESA calculation)
  - wt_average_length: 21 (WT1 smoothing)
  - wt_ma_length: 4 (WT2 signal line)
  - Zones: ±60 (OB/OS), ±80 (extreme)
  - R:R ratio: 2:1 (1.5% SL, 3.0% TP for XRP/USDT)
- Watch for: neutral zone entries (less reliable), zone exit timing
