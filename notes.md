update the docs with the recent work and ALL the changes in git. Ensure documentation complies with the documentation standards and expectations outlined in the claude.md file. Then commit.

cd ws_paper_tester
# Install dependencies
pip install -r requirements.txt
# Run with live Kraken WebSocket data
python ws_tester.py
# Run with simulated data
python ws_tester.py --simulated
# Run for 30 minutes with dashboard disabled
python ws_tester.py --duration 30 --no-dashboard
# Run tests
pytest tests/ -v



Strategies:
- btc and xrp usdt pairs- 9 week moving average on the 5 min and 1 hour. One candle opposite the trend the trend close position. 2 candles closed above/below the 9 is a trend(the equation for a trend is not a set definition and is open to improvement)
- does xrp follow btc predictably enough to trade on?

I want to develop these strategies. 
| Scalping (momentum) | 1m-5m             | Quick momentum bursts                      |
| Arbitrage           | Tick-level        | Cross-exchange price differences           |





---

# REUSABLE STRATEGY PROMPTS

## Deep Review Prompt (Copy & Fill Placeholders)
```
## Task: Deep Review of market_making Strategy
### Scope
- **Strategy:** `ws_paper_tester/strategies/market_making/`
- **Docs:** `ws_paper_tester/docs/development/features/market_making/` (if exists)
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
Create documentation at: `ws_paper_tester/docs/development/review/market_making/`
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
## Task: Implement Review Findings for {STRATEGY_NAME} Strategy
### Files
- **Strategy:** `ws_paper_tester/strategies/{strategy_dir}/`
- **Review:** `ws_paper_tester/docs/development/review/{strategy_dir}/{review_filename}`
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
- All CRITICAL/HIGH findings addressed or justified
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
5. Commit changes with documentation updates
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
