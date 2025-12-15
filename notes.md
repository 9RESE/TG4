# Project Notes

## Current Status: Indicator Library v1.0.0 Complete

**Date:** 2025-12-15
**Version:** ws_paper_tester v1.13.0

### Recently Completed
- Centralized Indicator Library (Phases 0-7) - DONE
  - 27 indicator functions across 7 modules
  - 46 tests passing with golden fixture regression
  - Code review findings addressed
  - Documentation complete

### Next Steps
- Phase 8: Strategy migration to use centralized library
- Continue deep reviews for remaining strategies

---






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
## Task: Deep Review of whale_sentiment Strategy
### Scope
- **Sources:
  - **Strategy: `ws_paper_tester/strategies/whale_sentiment/`
  - **Docs:** `ws_paper_tester/docs/development/features/whale_sentiment/` (if exists)
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
Create documentation at: `ws_paper_tester/docs/development/review/whale_sentiment/`
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
## Task: Implement Review Findings for whale_sentiment Strategy
### Files
  - **Strategy:** `ws_paper_tester/strategies/whale_sentiment/`
  - **Docs:** `ws_paper_tester/docs/development/features/whale_sentiment/`
  - **Review: `ws_paper_tester/docs/development/review/whale_sentiment/deep-review-v4.0.md`
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

## WaveTrend Oscillator (v1.1.0 - 2025-12-14)
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
- v1.1.0 additions (Deep Review v1.0):
  - REC-001: Optimized channel/average lengths (10/21)
  - REC-002: Zone confirmation (2 candles in zone)
  - REC-003: Divergence detection with confidence boost
  - REC-005: Extreme zone profit taking
  - REC-006: Enhanced indicator logging
  - REC-007: Per-symbol WT parameters
- Watch for: neutral zone entries (less reliable), zone exit timing

## Whale Sentiment (v1.4.0 - 2025-12-15)
- Pairs: XRP/USDT, BTC/USDT (XRP/BTC disabled by default)
- Key concepts: Volume spike detection (whale proxy), price deviation sentiment, contrarian mode
- Core logic:
  - Volume spikes >= 2x average detected as whale activity
  - Price deviation sentiment classification (RSI REMOVED per REC-021/REC-032)
  - ATR-based volatility regime classification with EXTREME pause
  - Composite confidence scoring with weighted components
- Entry (Contrarian mode - default):
  - Long: Fear sentiment + volume spike (buy fear)
  - Short: Greed sentiment + volume spike (sell greed)
- Exit:
  - Stop loss / Take profit (2.5% SL, 5.0% TP default)
  - Sentiment reversal (primary)
- Config highlights:
  - weight_volume_spike: 0.55 (primary signal)
  - weight_price_deviation: 0.35 (primary sentiment - RSI removed)
  - weight_trade_flow: 0.10 (confirmation)
  - short_size_multiplier: 0.60 (increased for extreme fear per REC-026)
  - Stricter circuit breaker: 2 losses, 45 min cooldown
  - Session-aware sizing: Asia 0.8x, Off-Hours 0.5x
- v1.4.0 additions (Deep Review v4.0):
  - REC-030: CRITICAL - Fixed undefined function reference bug
  - REC-031: Added EXTREME volatility regime (ATR > 6%) with trading pause
  - REC-032: Removed deprecated RSI code (calculate_rsi removed)
  - REC-033: Added scope and limitations documentation
  - Guide v2.0 compliance: 100%
- Previous versions: v1.3.0 (REC-021-027), v1.2.0 (REC-011-020), v1.1.0 (REC-001-010)
- Watch for: 25+ hour warmup requirement, XRP/BTC liquidity concerns, EXTREME volatility pause




- i want to look at the different strategies for redundant logic and data processing. Like does more then one strategy calculate the rsi?
- is there a algorithm we can make to track the tokens and overall market direction. Like a bear, bull, or flat/sideways detector that the strategies can use in their evals. Maybe strategies can have different parameters based on the algorithms output to maximize it effectiveness.
- do we have any algos the use margin/leverage?






















     Implementation Plan: Indicator Library Refactoring v1.0

     Source Plan: ws_paper_tester/docs/development/plans/indicator-library-refactoring-v1.0.md
     Status: Ready for implementation
     Scope: Consolidate ~45 duplicated indicator functions across 8 strategies into centralized library

     ---
     User Decisions

     - Test First: Create golden fixtures BEFORE any code changes
     - RSI Method: Use Wilder's smoothing (industry standard)
     - Approach: Pause for review after Phases 1-7 (before migration)

     ---
     Overview

     Refactor redundant indicator calculations from 8 strategy modules into a centralized ws_tester/indicators/ library, eliminating 60-70% code duplication while maintaining exact 
     behavioral compatibility.

     ---
     Implementation Phases

     Phase 0: Golden Test Fixtures (FIRST)

     Purpose: Capture current indicator behavior before any changes for regression testing.

     Create: tests/fixtures/indicator_test_data.py

     Fixtures to capture:
     - RSI outputs from all 4 implementations with identical inputs
     - EMA/SMA outputs
     - Volatility outputs (should be identical across 6 implementations)
     - ADX outputs
     - Bollinger Bands outputs
     - Correlation outputs
     - ATR outputs (note: whale_sentiment uses Wilder's smoothing, others use simple average)

     Format: Dictionary with input candles and expected outputs, tolerance 1e-10

     ---
     Phase 1: Core Infrastructure

     Create foundation files:
     - ws_tester/indicators/__init__.py - Public API exports
     - ws_tester/indicators/_types.py - Type definitions

     Types to define:
     PriceInput = Union[List[Candle], Tuple[Candle, ...], List[float]]
     BollingerResult = NamedTuple('BollingerResult', sma, upper, lower, std_dev)
     ATRResult = NamedTuple('ATRResult', atr, atr_pct, tr_series)
     TradeFlowResult = NamedTuple('TradeFlowResult', buy_volume, sell_volume, imbalance)
     TrendResult = NamedTuple('TrendResult', slope_pct, is_trending)

     Helpers:
     - extract_closes(data: PriceInput) -> List[float]
     - extract_hlc(data: PriceInput) -> Tuple[List, List, List]

     ---
     Phase 2: Moving Averages

     File: ws_tester/indicators/moving_averages.py

     | Function               | Source                                   |
     |------------------------|------------------------------------------|
     | calculate_sma()        | mean_reversion:14-19, wavetrend:102-116  |
     | calculate_sma_series() | wavetrend:119-140                        |
     | calculate_ema()        | momentum_scalping:13-39, wavetrend:45-71 |
     | calculate_ema_series() | momentum_scalping:42-67, wavetrend:74-99 |

     ---
     Phase 3: Oscillators

     File: ws_tester/indicators/oscillators.py

     | Function                      | Source                                            | Notes                                |
     |-------------------------------|---------------------------------------------------|--------------------------------------|
     | calculate_rsi()               | mean_reversion:22-57, momentum_scalping:70-112    | Default period=14, scalping uses 7   |
     | calculate_rsi_series()        | momentum_scalping:115-162                         |                                      |
     | calculate_adx()               | mean_reversion:204-310, momentum_scalping:654-760 | Wilder's smoothing                   |
     | calculate_macd()              | momentum_scalping:165-225                         | Default (12,26,9), scalping (6,13,5) |
     | calculate_macd_with_history() | momentum_scalping:228-310                         | Crossover detection                  |

     ---
     Phase 4: Volatility

     File: ws_tester/indicators/volatility.py

     | Function                    | Source                                            | Notes                  |
     |-----------------------------|---------------------------------------------------|------------------------|
     | calculate_volatility()      | 6 identical implementations                       | Returns std_dev * 100  |
     | calculate_atr()             | momentum_scalping:407-438, whale_sentiment:90-146 | Add rich_output flag   |
     | calculate_bollinger_bands() | mean_reversion:60-78, ratio_trading:11-38         | Return BollingerResult |
     | calculate_z_score()         | ratio_trading                                     | Simple calculation     |
     | get_volatility_regime()     | market_making:183-197                             | Regime classification  |

     ---
     Phase 5: Correlation

     File: ws_tester/indicators/correlation.py

     | Function                        | Source                                  |
     |---------------------------------|-----------------------------------------|
     | calculate_rolling_correlation() | 5+ implementations (Pearson on returns) |
     | calculate_correlation_trend()   | ratio_trading:260-290                   |

     ---
     Phase 6: Volume & Flow

     Files: ws_tester/indicators/volume.py, ws_tester/indicators/flow.py

     volume.py:
     - calculate_volume_ratio() - grid_rsi:405-450, momentum_scalping:313-342
     - calculate_volume_spike() - momentum_scalping:345-373
     - calculate_micro_price() - order_flow:36-58, market_making:14-35
     - calculate_vpin() - order_flow:61-120

     flow.py:
     - calculate_trade_flow() - grid_rsi:358-404, wavetrend:465-522, whale_sentiment:465-519
     - check_trade_flow_confirmation() - grid_rsi:533-580

     ---
     Phase 7: Trend

     File: ws_tester/indicators/trend.py

     | Function                  | Source                                        |
     |---------------------------|-----------------------------------------------|
     | calculate_trend_slope()   | mean_reversion:105-136, market_making:335-381 |
     | detect_trend_strength()   | ratio_trading:112-153                         |
     | calculate_trailing_stop() | ratio_trading:156-180, market_making:121-152  |

     ---
     Phase 8: Strategy Migration

     Order by complexity (simplest first):

     1. order_flow - 4 functions (migrate 2)
     2. whale_sentiment - 12 functions (migrate 3)
     3. wavetrend - 11 functions (migrate 4)
     4. mean_reversion - 7 functions (migrate all)
     5. grid_rsi_reversion - 11 functions (migrate 8)
     6. ratio_trading - 11 functions (migrate 6)
     7. momentum_scalping - 15+ functions (migrate 10)
     8. market_making - 18 functions (migrate 5)

     Per-strategy migration:
     1. Update imports to use from ws_tester.indicators import ...
     2. Remove duplicated functions from local indicators.py
     3. Keep strategy-specific functions in local module
     4. Update call sites for new return types (NamedTuples)
     5. Run tests to verify unchanged behavior

     ---
     Phase 9: Testing

     New test files:
     - tests/test_indicators.py - Unit tests for indicator library
     - tests/fixtures/indicator_test_data.py - Golden test fixtures

     Test categories:
     1. Unit tests - known input/output pairs
     2. Edge cases - empty data, insufficient data, extreme values
     3. Regression tests - golden fixtures with 1e-10 tolerance
     4. Integration tests - strategy signal generation unchanged

     ---
     Files to Create (11)

     | File                                    | Purpose                    |
     |-----------------------------------------|----------------------------|
     | ws_tester/indicators/__init__.py        | Public API                 |
     | ws_tester/indicators/_types.py          | Type definitions           |
     | ws_tester/indicators/moving_averages.py | SMA, EMA                   |
     | ws_tester/indicators/oscillators.py     | RSI, ADX, MACD             |
     | ws_tester/indicators/volatility.py      | ATR, Bollinger, volatility |
     | ws_tester/indicators/correlation.py     | Rolling correlation        |
     | ws_tester/indicators/volume.py          | Volume metrics             |
     | ws_tester/indicators/flow.py            | Trade flow                 |
     | ws_tester/indicators/trend.py           | Trend, trailing stops      |
     | tests/test_indicators.py                | Test suite                 |
     | tests/fixtures/indicator_test_data.py   | Test fixtures              |

     Files to Modify (8)

     | File                                        | Action              |
     |---------------------------------------------|---------------------|
     | strategies/mean_reversion/indicators.py     | Remove 7 functions  |
     | strategies/grid_rsi_reversion/indicators.py | Remove 8 functions  |
     | strategies/ratio_trading/indicators.py      | Remove 6 functions  |
     | strategies/momentum_scalping/indicators.py  | Remove 10 functions |
     | strategies/wavetrend/indicators.py          | Remove 4 functions  |
     | strategies/order_flow/indicators.py         | Remove 2 functions  |
     | strategies/whale_sentiment/indicators.py    | Remove 3 functions  |
     | strategies/market_making/calculations.py    | Remove 5 functions  |

     ---
     Critical Path

     Phase 0: Create Golden Test Fixtures (FIRST)
         ↓
     Phase 1 (Infrastructure)
         ↓
     Phases 2-7 (Indicator Modules) - Can be done in parallel
         ↓
     ═══════════════════════════════════════
         ⏸️  PAUSE FOR REVIEW HERE
     ═══════════════════════════════════════
         ↓
     Phase 8 (Strategy Migration) - Sequential, test after each
         ↓
     Phase 9 (Final Testing)

     ---
     Success Criteria

     - All existing tests pass (zero regressions)
     - No behavioral changes (within 1e-10 float tolerance)
     - ~60% reduction in indicator code lines
     - Single source of truth per indicator
     90% test coverage for indicator library
     - Clear API documentation

     ---
     Risk Mitigation

     | Risk                   | Mitigation                               |
     |------------------------|------------------------------------------|
     | Behavioral differences | Golden fixtures, regression tests        |
     | Breaking strategies    | Test after each migration                |
     | Performance regression | Benchmark critical paths                 |
     | Import cycles          | indicators/ imports only ws_tester.types |




     Phases 0-7 Complete - Ready for Review

  All 45 tests pass. Here's what was created:

  Files Created (11 total)

  Test Fixtures:
  - tests/fixtures/indicator_test_data.py - Golden fixtures capturing existing behavior

  Indicator Library:
  - ws_tester/indicators/__init__.py - Public API exports
  - ws_tester/indicators/_types.py - Type definitions (PriceInput, NamedTuples, extractors)
  - ws_tester/indicators/moving_averages.py - SMA, EMA
  - ws_tester/indicators/oscillators.py - RSI (Wilder's), ADX, MACD
  - ws_tester/indicators/volatility.py - ATR, Bollinger, volatility, z-score, regime
  - ws_tester/indicators/correlation.py - Rolling correlation, trend detection
  - ws_tester/indicators/volume.py - Volume ratio, spike, micro-price, VPIN
  - ws_tester/indicators/flow.py - Trade flow, flow confirmation
  - ws_tester/indicators/trend.py - Trend slope, strength, trailing stops

  Test Suite:
  - tests/test_indicators.py - 45 tests covering all functions