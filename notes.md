update the docs with the recent work and ALL the changes in git. Ensure documentation complies with the documentation standards and expectations outlined in the claude.md file. Then commit.





- wavetrend /home/rese/Documents/rese/trading-bots/grok-4_1/src/strategies/wavetrend
- whale_sentiment /home/rese/Documents/rese/trading-bots/grok-4_1/src/strategies/whale_sentiment
- grid_rsi_reversion /home/rese/Documents/rese/trading-bots/grok-4_1/src/strategies/grid_base.py /home/rese/Documents/rese/trading-bots/grok-4_1/src/strategies/grid_wrappers.py /home/rese/Documents/rese/trading-bots/grok-4_1/src/grid_ensemble_orchestrator.py


# REUSABLE STRATEGY PROMPTS

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
## Task: Deep Review of momentum_scalping Strategy
### Scope
- **Strategy:** `ws_paper_tester/strategies/momentum_scalping/`
- **Docs:** `ws_paper_tester/docs/development/features/momentum_scalping/` (if exists)
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
Create documentation at: `ws_paper_tester/docs/development/review/momentum_scalping/`
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
## Task: Implement Review Findings for momentum_scalping Strategy
### Files
- **Strategy:** `ws_paper_tester/strategies/momentum_scalping/`
- **Review:** `ws_paper_tester/docs/development/review/momentum_scalping/deep-review-v1.0.md`
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
## Momentum Scalping (v2.0.0 - 2025-12-14)
- Pairs: XRP/USDT, BTC/USDT, XRP/BTC
- Key concepts: RSI 7, MACD (6,13,5), EMA 8/21/50 ribbon, volume spikes
- v2.0 additions:
  - REC-001: XRP/BTC correlation pause (threshold 0.50)
  - REC-002: 5m trend filter (50 EMA alignment)
  - REC-003: ADX filter for BTC (threshold 25)
  - REC-004: Regime-based RSI bands (75/25 in HIGH vol)
- Watch for: correlation breakdown on XRP/BTC, BTC trending markets (ADX>25)
