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





## Task: Deep Review of Ratio Trading Strategy
### Scope
- Location: `ws_paper_tester/strategies/ratio_trading/`
- Pairs: XRP/BTC (primary), evaluate XRP/USDT and BTC/USDT applicability
### Review Requirements
#### 1. Strategy Research
- Research ratio/pairs trading theory: cointegration vs correlation, Engle-Granger method, Johansen test
- Investigate statistical arbitrage and spread trading in crypto markets
- Research optimal Z-score entry/exit thresholds from academic literature
- Analyze Generalized Hurst Exponent (GHE) for pair selection
- Identify when ratio relationships break down (regulatory events, market stress)
#### 2. Pair-Specific Analysis
For XRP/BTC:
- Current correlation and cointegration status
- Historical ratio stability and half-life of mean reversion
- Liquidity and spread characteristics
- Suitability for ratio trading approach
For XRP/USDT and BTC/USDT:
- Explicitly document why these are NOT suitable for ratio trading
- Clarify the difference between ratio trading and standard mean reversion
#### 3. Code Review
- Review implementation against `ws_paper_tester/docs/development/strategy-development-guide.md` v2.0
- Check compliance with new v2.0 sections:
  - Section 15: Volatility Regime Classification
  - Section 16: Circuit Breaker Protection
  - Section 17: Signal Rejection Tracking
  - Section 22: Per-Symbol Configuration (SYMBOL_CONFIGS)
  - Section 24: Correlation Monitoring (critical for ratio trading)
  - Section 26: Strategy Scope Documentation
- Evaluate R:R ratio (must be >= 1:1)
- Check position sizing (USD-based, not asset units)
- Verify strategy scope is properly documented
#### 4. Strategy Logic Analysis
- Evaluate cointegration validation (or lack thereof)
- Assess Z-score calculation and threshold selection
- Review dual-asset accumulation tracking
- Identify correlation breakdown detection
- Check for regime change handling
### Deliverable
Create documentation at: `ws_paper_tester/docs/development/review/ratio_trading/`
Include:
1. **Executive Summary** - Overall assessment and risk level
2. **Research Findings** - Academic and industry research summary
3. **Pair Analysis** - Market characteristics and suitability per pair
4. **Compliance Matrix** - Checklist against strategy-development-guide.md v2.0
5. **Critical Findings** - Prioritized issues (CRITICAL/HIGH/MEDIUM/LOW)
6. **Recommendations** - Specific improvements with priority and effort estimates
7. **Research References** - Academic papers and industry sources
### Constraints
- Do NOT modify any code during this review
- Do NOT include code snippets in documentation
- Focus on analysis and recommendations only
- Reference specific line numbers when identifying issues



## Task
Update the ratio_trading strategy based on the latest deep review findings.
## Files
- **Strategy:** `ws_paper_tester/strategies/ratio_trading/`
- **Review:** `ws_paper_tester/docs/development/review/ratio_trading/ratio-trading-deep-review-v10.0.md`
- **Guide:** `ws_paper_tester/docs/development/review/market_maker/strategy-development-guide.md`
## Instructions
1. Read the review document and identify all recommendations (REC-XXX)
2. Read the strategy development guide for compliance requirements
3. Categorize recommendations by priority:
    - **CRITICAL/HIGH severity** → Must implement
    - **MEDIUM severity + LOW effort** → Should implement
    - **LOW severity or HIGH effort** → Document for future consideration
4. For each recommendation to implement:
    - Update code with clear comments referencing the REC-ID
    - Add any new config parameters to CONFIG
    - Add new rejection reasons to RejectionReason enum if needed
    - Add new exit reasons to ExitReason enum if needed
    - Update version history in docstring
5. Update STRATEGY_VERSION following semver:
    - Breaking changes → major bump
    - New features → minor bump
    - Bug fixes/config tweaks → patch bump
## Acceptance Criteria
- All CRITICAL/HIGH findings addressed or justified
- Compliance score maintained at 100% (v2.0 guide)
- All existing tests pass
- Indicators populated on all code paths (Guide Section 7)
- R:R ratio >= 1:1 maintained (Guide Section 4)
- Correlation monitoring functional (Guide Section 24)
- Strategy scope clearly documented (Guide Section 26)
- Version history updated with changes
## Strategy-Specific Considerations
- XRP/BTC is the ONLY supported pair (USDT pairs are NOT suitable)
- Correlation pause is now enabled by default (v4.0.0)
- Dual-asset accumulation tracking must be maintained
- Dynamic BTC price lookup for USD conversion must work
## Output
After implementation, provide:
1. Summary of changes made (by REC-ID)
2. Changes deferred and why
3. New compliance score estimate
4. Any new risks introduced
5. Update the docs with the recent work and ALL the changes in git. Ensure documentation complies with the documentation standards and expectations outlined in the claude.md file. Then commit.






## Task: Deep Review of Mean Reversion Strategy
### Scope
- Location: `ws_paper_tester/strategies/mean_reversion/`
- Pairs: XRP/USDT, BTC/USDT, XRP/BTC
### Review Requirements
#### 1. Strategy Research
- Research mean reversion theory: Ornstein-Uhlenbeck process, half-life of mean reversion, stationarity requirements
- Investigate mean reversion effectiveness in crypto markets (vs traditional markets)
- Analyze Bollinger Bands + RSI combination effectiveness in academic literature
- Research optimal lookback periods and deviation thresholds for crypto volatility
- Identify market conditions where mean reversion fails (trending markets, regime changes)
#### 2. Pair-Specific Analysis
For each pair (XRP/USDT, BTC/USDT, XRP/BTC):
- Current market characteristics (volatility, liquidity, spread)
- Historical mean reversion tendencies
- Optimal parameter recommendations
- Suitability assessment for mean reversion approach
#### 3. Code Review
- Review implementation against `ws_paper_tester/docs/development/strategy-development-guide.md` v2.0
- Check compliance with new v2.0 sections:
- Section 15: Volatility Regime Classification
- Section 16: Circuit Breaker Protection
- Section 17: Signal Rejection Tracking
- Section 18: Trade Flow Confirmation
- Section 19: Trend Filtering (critical for mean reversion)
- Section 22: Per-Symbol Configuration (SYMBOL_CONFIGS)
- Evaluate R:R ratio (must be >= 1:1)
- Assess indicator logging on early returns
- Check position sizing (USD-based, not asset units)
#### 4. Strategy Logic Analysis
- Evaluate entry/exit signal conditions
- Assess stop-loss and take-profit implementation
- Review cooldown mechanisms
- Identify edge cases and failure modes
- Check for "band walk" protection during trends
### Deliverable
Create documentation at: `ws_paper_tester/docs/development/review/mean_reversion/`
Include:
1. **Executive Summary** - Overall assessment and risk level
2. **Research Findings** - Academic and industry research summary
3. **Pair Analysis** - Market characteristics and suitability per pair
4. **Compliance Matrix** - Checklist against strategy-development-guide.md v2.0
5. **Critical Findings** - Prioritized issues (CRITICAL/HIGH/MEDIUM/LOW)
6. **Recommendations** - Specific improvements with priority and effort estimates
7. **Research References** - Academic papers and industry sources
### Constraints
- Do NOT modify any code during this review
- Do NOT include code snippets in documentation
- Focus on analysis and recommendations only
- Reference specific line numbers when identifying issues





## Task
Update the mean_reversion strategy based on the latest deep review findings.
## Files
- **Strategy:** `ws_paper_tester/strategies/mean_reversion/`
- **Review:** `ws_paper_tester/docs/development/review/mean_reversion/mean-reversion-deep-review-v8.0.md`
- **Guide:** `ws_paper_tester/docs/development/review/market_maker/strategy-development-guide.md`
## Instructions
1. Read the review document and identify all recommendations (REC-XXX)
2. Read the strategy development guide for compliance requirements
3. Categorize recommendations by priority:
    - **CRITICAL/HIGH severity** → Must implement
    - **MEDIUM severity + LOW effort** → Should implement
    - **LOW severity or HIGH effort** → Document for future consideration
4. For each recommendation to implement:
    - Update code with clear comments referencing the REC-ID
    - Add any new config parameters to CONFIG or SYMBOL_CONFIGS
    - Add new rejection reasons to RejectionReason enum if needed
    - Update version history in docstring
5. Update STRATEGY_VERSION following semver:
    - Breaking changes → major bump
    - New features → minor bump
    - Bug fixes/config tweaks → patch bump
## Acceptance Criteria
- All CRITICAL/HIGH findings addressed or justified
- Compliance score improves (check review's Compliance Matrix)
- All existing tests pass
- Indicators populated on all code paths (Guide Section 7)
- R:R ratio >= 1:1 maintained (Guide Section 4)
- Version history updated with changes
## Output
After implementation, provide:
1. Summary of changes made (by REC-ID)
2. Changes deferred and why
3. New compliance score estimate
4. Any new risks introduced
5. Update the docs with the recent work and ALL the changes in git. Ensure documentation complies with the documentation standards and expectations outlined in the claude.md file. Then commit. 




