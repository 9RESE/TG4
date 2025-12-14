# Order Flow Strategy Deep Review v4.0.0

**Review Date:** 2025-12-14
**Version Reviewed:** 4.0.0
**Reviewer:** Extended Strategic Analysis
**Status:** Comprehensive Code and Strategy Review
**Previous Review:** v3.1.0 (2025-12-13)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Order Flow Trading Research](#2-order-flow-trading-research)
3. [Trading Pair Analysis](#3-trading-pair-analysis)
4. [Code Quality Assessment](#4-code-quality-assessment)
5. [Strategy Development Guide Compliance](#5-strategy-development-guide-compliance)
6. [Critical Findings](#6-critical-findings)
7. [Recommendations](#7-recommendations)
8. [Research References](#8-research-references)

---

## 1. Executive Summary

### Overview

The Order Flow strategy v4.0.0 represents a significant evolution from v3.1.0, implementing all five major recommendations from the previous review. The strategy now incorporates VPIN (Volume-Synchronized Probability of Informed Trading), volatility regime classification, trading session awareness, progressive position decay, and cross-pair correlation management.

### Version 4.0.0 Implementation Status

| Recommendation | v3.1 Review | v4.0.0 Status |
|----------------|-------------|---------------|
| REC-001: VPIN Calculation | Recommended | **Implemented** |
| REC-002: Volatility Regime Classification | Recommended | **Implemented** |
| REC-003: Time-of-Day Session Awareness | Recommended | **Implemented** |
| REC-004: Progressive Position Decay | Recommended | **Implemented** |
| REC-005: Cross-Pair Correlation Management | Recommended | **Implemented** |

### Key Strengths

| Aspect | Assessment |
|--------|------------|
| Research Alignment | Excellent - VPIN implementation aligns with academic research on order flow toxicity |
| Volatility Adaptation | Strong - Four-tier regime classification handles market conditions appropriately |
| Session Awareness | Strong - Adjusts parameters for Asian, European, US, and overlap sessions |
| Risk Management | Excellent - Multi-layered protection with circuit breaker, correlation limits, progressive decay |
| Code Organization | Improved - Clear section organization with documented functions |

### Risk Assessment Summary

| Risk Level | Category | Description |
|------------|----------|-------------|
| LOW | Code Quality | Well-structured with comprehensive validation |
| LOW | Guide Compliance | Fully compliant with all requirements |
| MODERATE | VPIN Bucket Logic | Overflow distribution logic may need refinement |
| MODERATE | Signal Density | Multiple filtering layers may reduce signal frequency |
| LOW | Parameter Calibration | Current parameters are research-backed |

### Overall Verdict

**PRODUCTION READY - EXTENDED PAPER TESTING RECOMMENDED**

The v4.0.0 implementation represents a mature, research-backed order flow strategy with comprehensive risk management. Recommended for extended paper trading to gather performance metrics across different market conditions before live deployment.

---

## 2. Order Flow Trading Research

### Academic Foundation

#### Trade Flow vs Order Book Imbalance

Research by Silantyev (2019) on cryptocurrency markets established that trade flow imbalance explains contemporaneous price changes better than aggregate order book imbalance. Key findings:

- Trade flow shows actual market aggression (executed trades)
- Order book imbalance shows resting intention (pending orders)
- Cryptocurrency markets exhibit lower depth and update rates than traditional markets

**Strategy Alignment:** The order_flow strategy correctly prioritizes trade tape analysis, using order book data only for micro-price calculation and exit pricing.

#### VPIN Research for Bitcoin

The 2025 study "Bitcoin wild moves: Evidence from order flow toxicity and price jumps" demonstrates that VPIN significantly predicts future Bitcoin price jumps. Key insights:

- Average VPIN in crypto markets (0.45-0.47) is significantly higher than traditional markets (0.22-0.23)
- High VPIN indicates greater information-based trading and toxicity
- Positive serial correlation in VPIN and jump size suggests momentum effects

**Strategy Implementation:** v4.0.0 implements VPIN with configurable bucket count (50) and high threshold (0.7). The strategy pauses trading when VPIN exceeds threshold, protecting against adverse selection during informed trading periods.

#### Market Microstructure Dynamics

Cornell research (2024) on crypto market microstructure reveals:

- BTC Roll measure and VPIN are key drivers of Bitcoin price movements
- Trade in BTC and ETH leads price changes in other cryptocurrencies
- Greater serial correlation in crypto prices supports momentum-based strategies

**Strategy Alignment:** The strategy's momentum-based approach aligns with these findings. The exclusion of XRP/BTC pair is validated by research showing BTC's market-leading role.

### Industry Best Practices Alignment

The strategy follows the standard order flow trading pattern:

1. **Identify Imbalance** - Calculates buy/sell volume differential
2. **Confirm with Flow** - Trade flow confirmation requirement
3. **Adjust for Conditions** - VPIN, volatility regime, and session adjustments
4. **Enter Position** - Direction aligned with imbalance
5. **Manage Risk** - Multi-layered stops and circuit breakers
6. **Exit Strategy** - Progressive decay, trailing stops

---

## 3. Trading Pair Analysis

### XRP/USDT Analysis

#### Current Market Characteristics (December 2025)

| Metric | Value | Trading Implication |
|--------|-------|---------------------|
| 24h Volume | ~$1.2-1.6 billion | Excellent liquidity |
| Current Price | ~$2.02 | Moderate price point |
| 2025 High | $3.67 (July 2025) | Significant upside captured |
| Quarterly Volatility | 100-130% | Requires wider thresholds |
| BTC Correlation (3-month) | 0.84 | Strong but weakening |

#### XRP-Specific Considerations

**Strengths:**
- High liquidity enables efficient execution
- Growing independence from BTC (correlation declining 24.86% over 90 days)
- Regulatory clarity improving depth and market maker participation
- On-Demand Liquidity (ODL) usage provides unique volume patterns

**Risks:**
- Regulatory developments cause significant volatility spikes
- High correlation with BTC during market stress events
- Volume patterns affected by geographic timezone (Asia/US overlap)

#### Current Configuration Assessment

| Parameter | Value | Assessment |
|-----------|-------|------------|
| buy_imbalance_threshold | 0.30 | APPROPRIATE - Research-backed |
| sell_imbalance_threshold | 0.25 | APPROPRIATE - Lower for sells per research |
| position_size_usd | $25 | CONSERVATIVE - Suitable for paper trading |
| volume_spike_mult | 2.0 | APPROPRIATE - Standard confirmation |
| take_profit_pct | 1.0% | APPROPRIATE - Good for momentum |
| stop_loss_pct | 0.5% | APPROPRIATE - 2:1 R:R maintained |

### BTC/USDT Analysis

#### Current Market Characteristics

| Metric | Value | Trading Implication |
|--------|-------|---------------------|
| 24h Volume (Kraken) | ~$193M | Market-leading pair |
| Total Spot Volume | $1.39B+ (Kraken total) | Deep liquidity |
| Typical Spread | ~0.02% | Minimal slippage |
| Market Maker Depth | 100 levels available | Excellent depth |

#### BTC-Specific Considerations

**Strengths:**
- Highest liquidity in crypto markets
- Most researched for VPIN effectiveness
- Institutional flows provide information-rich signals
- Taker-driven market structure benefits trade flow analysis

**Risks:**
- ETF inflows change traditional liquidity patterns
- Correlation with traditional markets (S&P 500, gold) during certain periods
- Macro events cause sudden volatility regime shifts

#### Current Configuration Assessment

| Parameter | Value | Assessment |
|-----------|-------|------------|
| buy_imbalance_threshold | 0.25 | APPROPRIATE - Lower for high liquidity |
| sell_imbalance_threshold | 0.20 | APPROPRIATE - Even lower for institutional selling |
| position_size_usd | $50 | APPROPRIATE - Larger for BTC liquidity |
| volume_spike_mult | 1.8 | APPROPRIATE - Lower threshold for more signals |
| take_profit_pct | 0.8% | APPROPRIATE - Tighter for BTC |
| stop_loss_pct | 0.4% | APPROPRIATE - 2:1 R:R maintained |

### XRP/BTC Pair Assessment

The decision to exclude XRP/BTC from the order_flow strategy (made in v2.2.0) remains valid:

**Rationale:**
1. XRP/BTC correlation patterns show ratio trading behavior
2. Lower liquidity (~3x less than XRP/USDT)
3. Golden cross formation (May 2025) suggests longer-term trends
4. Better suited for ratio_trading or market_making strategies
5. Research shows BTC leads price discovery, not XRP/BTC ratio

**2025 Context:**
- XRP/BTC ratio formed first golden cross in May 2025
- XRP gaining 20% yearly vs BTC (1.13x outperformance)
- Decreasing correlation supports pairs trading strategies, not momentum

**Recommendation:** No change needed. XRP/BTC remains correctly excluded from order_flow.

---

## 4. Code Quality Assessment

### Code Organization

The v4.0.0 implementation shows significant organizational improvement:

| Section | Lines | Purpose | Quality |
|---------|-------|---------|---------|
| Section 1: Config & Validation | 218-286 | Configuration management | Excellent |
| Section 2: Indicator Calculations | 288-407 | VPIN, volatility, micro-price | Good |
| Section 3: Regime & Session | 409-516 | Classification functions | Excellent |
| Section 4: Risk Management | 518-691 | Fee checks, correlation, stops | Excellent |
| Section 5: Logging Helpers | 693-713 | Indicator building | Good |
| Section 6: Exit Checks | 715-881 | Trailing stop, decay exits | Good |
| Main Signal Generation | 883-1361 | Core logic | Good |
| Lifecycle Callbacks | 1363-1525 | on_start, on_fill, on_stop | Excellent |

### Function Complexity Analysis

| Function | Lines | Cyclomatic Complexity | Assessment |
|----------|-------|----------------------|------------|
| generate_signal | 58 | Medium | Acceptable - Main orchestrator |
| _evaluate_symbol | 394 | High | Consider refactoring |
| _calculate_vpin | 74 | Medium | Acceptable |
| on_fill | 107 | Medium | Acceptable |
| _check_position_decay_exit | 95 | Medium | Acceptable |

### Type Safety

| Aspect | Status | Notes |
|--------|--------|-------|
| Type hints | Present | All public functions typed |
| Enums | Used | VolatilityRegime, TradingSession |
| Dict typing | Partial | Config uses Dict[str, Any] |
| Return types | Present | Optional[Signal] correctly used |

### Error Handling

| Scenario | Handling | Assessment |
|----------|----------|------------|
| Missing price data | Early return with indicators | Good |
| Missing orderbook | Fallback to current_price | Good |
| Empty trades | Early return with status | Good |
| Invalid config | Validation warnings on startup | Good |
| Division by zero | Protected in all calculations | Good |

### Memory Management

| Aspect | Implementation | Assessment |
|--------|----------------|------------|
| Fill history | Bounded to last 50 | Good |
| Trade buffers | Uses tuples (immutable) | Excellent |
| Position entries | Cleaned on position close | Good |
| Indicator state | Overwritten each tick | Good |

---

## 5. Strategy Development Guide Compliance

### Required Components

| Component | Requirement | Status | Implementation |
|-----------|-------------|--------|----------------|
| STRATEGY_NAME | Lowercase with underscores | PASS | `"order_flow"` |
| STRATEGY_VERSION | Semantic versioning | PASS | `"4.0.0"` |
| SYMBOLS | List of trading pairs | PASS | `["XRP/USDT", "BTC/USDT"]` |
| CONFIG | Default configuration dict | PASS | 58 parameters |
| generate_signal() | Main signal function | PASS | Correct signature |

### Optional Components

| Component | Status | Quality |
|-----------|--------|---------|
| on_start() | PASS | Config validation, state init, feature logging |
| on_fill() | PASS | Position tracking, PnL tracking, circuit breaker |
| on_stop() | PASS | Summary statistics, cleanup |

### Signal Structure Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Required fields | PASS | action, symbol, size, price, reason |
| Stop loss positioning | PASS | Below entry for longs, above for shorts |
| Take profit positioning | PASS | Above entry for longs, below for shorts |
| Informative reason | PASS | Includes imbalance, volume, regime, session |
| Metadata usage | PASS | trailing_stop, position_decay flags |

### Position Management Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Position tracking | PASS | position_side, position_size, position_by_symbol |
| Max position limits | PASS | Checks and limits in _evaluate_symbol |
| Partial closes | PASS | Calculates available size correctly |
| on_fill updates | PASS | Comprehensive state updates |

### Risk Management Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| R:R ratio >= 1:1 | PASS | 2:1 for all pairs |
| Stop loss calculation | PASS | Price-based percentage |
| Cooldown mechanisms | PASS | Trade-based and time-based |
| Position limits | PASS | Per-pair and total exposure |

### Indicator Logging Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Always populate indicators | PASS | _build_base_indicators helper |
| Include inputs | PASS | All calculation inputs logged |
| Include decisions | PASS | Status, aligned flags, profitable flags |

### Per-Pair PnL Tracking

| Feature | Status | Implementation |
|---------|--------|----------------|
| pnl_by_symbol | PASS | Tracked in on_fill |
| trades_by_symbol | PASS | Tracked in on_fill |
| wins_by_symbol | PASS | Tracked in on_fill |
| losses_by_symbol | PASS | Tracked in on_fill |
| Indicator inclusion | PASS | pnl_symbol, trades_symbol logged |

---

## 6. Critical Findings

### Finding #1: VPIN Bucket Overflow Logic Complexity

**Severity:** LOW-MEDIUM
**Category:** Algorithm Implementation

**Description:** The VPIN calculation's bucket overflow logic uses proportional distribution based on the last trade side, which may not accurately represent the actual distribution of buy/sell volume across bucket boundaries.

**Current Implementation:**
- When a bucket overflows, the overflow is attributed based on the last trade's side
- This creates potential edge cases where rapid alternating trades may misattribute volume

**Impact:**
- Minor inaccuracies in VPIN calculation during high-frequency periods
- Unlikely to significantly affect trading decisions at 0.7 threshold

**Recommendation:** Monitor VPIN values during paper trading. If anomalies detected, consider implementing time-weighted bucket boundaries as described in academic literature.

### Finding #2: Multi-Layer Filtering May Reduce Signal Density

**Severity:** MEDIUM
**Category:** Strategy Performance

**Description:** The v4.0.0 implementation adds multiple filtering layers that may significantly reduce signal generation frequency:

1. Circuit breaker check
2. Time cooldown (5 seconds)
3. Trade count cooldown (10 trades)
4. VPIN pause (if > 0.7)
5. Volatility regime pause (EXTREME)
6. Trade flow confirmation
7. Fee profitability check
8. Correlation exposure limits

**Impact:**
- Lower signal frequency may improve quality but reduce trading opportunities
- Extended periods without signals during volatile markets
- May miss momentum moves that occur during high-VPIN periods

**Recommendation:** During paper trading, track signal rejection reasons to quantify the impact of each filter. Consider making some filters configurable for fine-tuning.

### Finding #3: Session Time Boundaries Are Fixed

**Severity:** LOW
**Category:** Enhancement Opportunity

**Description:** Trading session boundaries are hardcoded in UTC:
- Asia: 00:00-08:00 UTC
- Europe: 08:00-14:00 UTC
- US: 17:00-21:00 UTC
- US/Europe Overlap: 14:00-17:00 UTC

**Impact:**
- Does not account for daylight saving time changes
- US session starts at 17:00 UTC (actual US market open varies)
- Weekend behavior not differentiated

**Recommendation:** Consider making session boundaries configurable and adding weekend detection logic.

### Finding #4: Correlation Check Uses USD Exposure Only

**Severity:** LOW
**Category:** Risk Management

**Description:** The cross-pair correlation check only considers USD exposure amounts, not the actual correlation between asset returns.

**Current Implementation:**
- Tracks total_long_exposure and total_short_exposure
- Reduces size when multiple pairs signal same direction
- Does not calculate rolling correlation between XRP and BTC

**Impact:**
- May over-constrain during periods of low correlation
- May under-constrain during correlation spikes

**Recommendation:** Consider adding optional rolling correlation calculation for more dynamic exposure management.

### Finding #5: Position Decay Exit May Miss Optimal Exit Points

**Severity:** LOW
**Category:** Strategy Logic

**Description:** Progressive position decay checks profit against reduced TP targets but only triggers exit when profit exceeds the reduced threshold. This means:
- At 3 min: Needs 0.9% profit (90% of 1.0%)
- At 4 min: Needs 0.75% profit (75% of 1.0%)
- At 5 min: Needs 0.5% profit (50% of 1.0%)
- At 6+ min: Needs any profit

**Impact:**
- A position at 0.4% profit at 4 minutes won't exit (needs 0.75%)
- Position may hit stop loss while waiting for decay threshold

**Recommendation:** Consider adding option for "close at current profit if > fees" for any decay stage.

---

## 7. Recommendations

### Immediate Actions (This Sprint)

#### REC-001: Add Signal Rejection Logging

**Priority:** HIGH
**Effort:** LOW

Add detailed logging of why signals were rejected to track filter effectiveness:
- Count of rejections by filter type per session
- Include in on_stop summary statistics

**Benefit:** Enables tuning of filter parameters based on actual rejection patterns.

#### REC-002: Add Configuration Override Validation

**Priority:** MEDIUM
**Effort:** LOW

When strategy loads with config overrides, validate the overrides match expected parameter types and bounds.

**Benefit:** Prevents runtime errors from typos in config.yaml overrides.

### Short-Term Improvements (Next Sprint)

#### REC-003: Configurable Session Boundaries

**Priority:** MEDIUM
**Effort:** LOW

Move session time boundaries to CONFIG with sensible defaults.

**Benefit:** Allows adjustment for daylight saving time and regional testing.

#### REC-004: Enhanced Position Decay Options

**Priority:** LOW
**Effort:** LOW

Add configuration option for "close at any profit after fees" at intermediate decay stages.

**Benefit:** Reduces time in market when momentum has faded.

### Medium-Term Enhancements (Future Sprints)

#### REC-005: Dynamic Correlation Calculation

**Priority:** LOW
**Effort:** MEDIUM

Add optional rolling correlation calculation between XRP and BTC returns over configurable window.

**Benefit:** More accurate cross-pair exposure management during correlation regime changes.

#### REC-006: VPIN Calibration Analysis

**Priority:** LOW
**Effort:** MEDIUM

Collect VPIN values during paper trading and analyze relationship with subsequent price moves.

**Benefit:** May allow optimization of VPIN threshold for crypto-specific conditions (currently using 0.7 from traditional market research).

### Long-Term Research (Future Releases)

#### REC-007: Machine Learning Signal Enhancement

**Priority:** LOW
**Effort:** HIGH

Train models on historical signal data to predict signal quality.

**Research Context:** 2024 SSRN research shows ensemble models combining LSTM and order book data improve prediction accuracy.

#### REC-008: Absorption Pattern Detection

**Priority:** LOW
**Effort:** HIGH

Detect when large resting orders absorb incoming aggression before breakouts.

**Research Context:** Academic research shows absorption precedes institutional accumulation/distribution.

---

## 8. Research References

### Academic Research

- [Order Flow Analysis of Cryptocurrency Markets](https://www.researchgate.net/publication/332089928_Order_flow_analysis_of_cryptocurrency_markets) - Silantyev (2019) - Foundational trade flow vs order book research
- [Bitcoin Wild Moves: Evidence from Order Flow Toxicity and Price Jumps](https://www.sciencedirect.com/science/article/pii/S0275531925004192) - ScienceDirect (2025) - VPIN effectiveness for Bitcoin
- [Microstructure and Market Dynamics in Crypto Markets](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf) - Cornell (2024) - Comprehensive crypto microstructure analysis
- [Order Flow Imbalance - A High Frequency Trading Signal](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html) - Dean Markwick - Technical implementation guide
- [Price Impact of Order Book Imbalance in Cryptocurrency Markets](https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6/) - Towards Data Science

### Market Data Sources

- [XRP Price and Market Data](https://coinmarketcap.com/currencies/xrp/) - CoinMarketCap
- [XRP Metrics and Analysis](https://www.coinglass.com/currencies/XRP) - CoinGlass
- [Kraken Exchange Statistics](https://www.coingecko.com/en/exchanges/kraken) - CoinGecko
- [Kraken Market Data](https://www.amberdata.io/kraken-market-data) - Amberdata

### XRP/BTC Correlation Analysis

- [Correlation Between XRP and Bitcoin](https://www.macroaxis.com/invest/pair-correlation/XRP.CC/BTC.CC/XRP-vs-Bitcoin) - MacroAxis
- [Assessing XRP's Correlation with Bitcoin in 2025](https://eng.ambcrypto.com/assessing-xrps-correlation-with-bitcoin-and-what-it-means-for-its-price-in-2025/) - AMBCrypto
- [XRP-Bitcoin Pair Golden Cross Analysis](https://www.coindesk.com/markets/2025/05/21/xrp-btc-pair-flashes-first-golden-cross-hinting-at-major-bull-run-for-xrp/) - CoinDesk

### Industry Resources

- [Crypto Market Depth Analysis](https://www.krayondigital.com/blog/crypto-market-depth-how-it-impacts-trading) - Krayon Digital
- [Order Flow Trading in Crypto](https://bookmap.com/blog/digital-currency-trading-with-bookmap) - Bookmap
- [VPIN in Crypto Markets](https://medium.com/@lucasastorian/empirical-market-microstructure-f67eff3517e0) - Lucas Astorian

### Internal Documentation

- Strategy Development Guide v1.1
- Order Flow Strategy Review v3.1.0
- Order Flow Feature Documentation v3.1.0
- Market Making Strategy Review (comparison patterns)

---

## Appendix A: v4.0.0 Feature Implementation Summary

### VPIN Implementation (REC-001 from v3.1)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| use_vpin | True | Enable VPIN calculation |
| vpin_bucket_count | 50 | Volume buckets for calculation |
| vpin_high_threshold | 0.7 | Threshold for pause |
| vpin_pause_on_high | True | Pause when threshold exceeded |
| vpin_lookback_trades | 200 | Trade window for calculation |

### Volatility Regime Implementation (REC-002 from v3.1)

| Regime | Volatility | Threshold Mult | Size Mult |
|--------|------------|----------------|-----------|
| LOW | < 0.3% | 0.9 | 1.0 |
| MEDIUM | 0.3% - 0.8% | 1.0 | 1.0 |
| HIGH | 0.8% - 1.5% | 1.3 | 0.8 |
| EXTREME | > 1.5% | 1.5 | 0.5 |

### Session Awareness Implementation (REC-003 from v3.1)

| Session | Hours (UTC) | Threshold Mult | Size Mult |
|---------|-------------|----------------|-----------|
| ASIA | 00:00-08:00 | 1.2 | 0.8 |
| EUROPE | 08:00-14:00 | 1.0 | 1.0 |
| US_EUROPE_OVERLAP | 14:00-17:00 | 0.85 | 1.1 |
| US | 17:00-21:00 | 1.0 | 1.0 |

### Progressive Position Decay Implementation (REC-004 from v3.1)

| Age | TP Multiplier | Behavior |
|-----|---------------|----------|
| < 3 min | 1.0 | Full TP target |
| 3 min | 0.90 | 90% of original TP |
| 4 min | 0.75 | 75% of original TP |
| 5 min | 0.50 | 50% of original TP |
| 6+ min | 0.0 | Close at any profit |

### Cross-Pair Correlation Management Implementation (REC-005 from v3.1)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| use_correlation_management | True | Enable cross-pair limits |
| max_total_long_exposure | $150 | Maximum total long USD |
| max_total_short_exposure | $150 | Maximum total short USD |
| same_direction_size_mult | 0.75 | Reduce size if both pairs same direction |

---

## Appendix B: Indicator Reference (v4.0.0)

### Core Indicators

| Indicator | Type | Description |
|-----------|------|-------------|
| symbol | string | Current trading pair |
| status | string | Current evaluation status |
| trade_count | int | Total trades available |
| imbalance | float | Buy/sell imbalance (-1 to +1) |
| buy_volume | float | Volume from buy trades |
| sell_volume | float | Volume from sell trades |
| volume_spike | float | Recent volume vs average |

### Price Indicators

| Indicator | Type | Description |
|-----------|------|-------------|
| vwap | float | Volume-weighted average price |
| price | float | Current market price |
| micro_price | float | Volume-weighted micro-price |
| price_vs_vwap | float | Deviation from VWAP |

### Regime and Session Indicators

| Indicator | Type | Description |
|-----------|------|-------------|
| volatility_pct | float | Current volatility percentage |
| volatility_regime | string | LOW/MEDIUM/HIGH/EXTREME |
| trading_session | string | ASIA/EUROPE/US/US_EUROPE_OVERLAP |
| regime_threshold_mult | float | Regime adjustment factor |
| session_threshold_mult | float | Session adjustment factor |
| combined_threshold_mult | float | Total threshold adjustment |
| combined_size_mult | float | Total size adjustment |

### Threshold Indicators

| Indicator | Type | Description |
|-----------|------|-------------|
| base_buy_threshold | float | Unadjusted buy threshold |
| base_sell_threshold | float | Unadjusted sell threshold |
| effective_buy_threshold | float | Adjusted buy threshold |
| effective_sell_threshold | float | Adjusted sell threshold |
| adjusted_lookback | int | Regime-adjusted trade lookback |

### Advanced Indicators

| Indicator | Type | Description |
|-----------|------|-------------|
| vpin | float | VPIN value (0-1) |
| vpin_threshold | float | Current VPIN threshold |
| trade_flow | float | Trade flow imbalance |
| trade_flow_aligned | bool | Flow confirms signal |
| is_fee_profitable | bool | Trade profitable after fees |
| trailing_stop_price | float | Current trailing stop |

### Position and Risk Indicators

| Indicator | Type | Description |
|-----------|------|-------------|
| position_side | string | long/short/None |
| position_size | float | Current position USD |
| max_position | float | Maximum allowed position |
| adjusted_position_size | float | Size after adjustments |
| consecutive_losses | int | Current loss streak |
| pnl_symbol | float | Cumulative PnL for symbol |
| trades_symbol | int | Trade count for symbol |

---

**Document Version:** 1.0
**Last Updated:** 2025-12-14
**Author:** Extended Strategic Analysis
**Next Review:** After 2 weeks of paper trading data collection
