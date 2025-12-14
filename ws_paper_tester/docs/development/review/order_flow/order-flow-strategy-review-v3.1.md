# Order Flow Strategy Deep Review v3.1.0

**Review Date:** 2025-12-13
**Version Reviewed:** 3.1.0
**Reviewer:** Strategy Architecture Review (Extended Analysis)
**Status:** Comprehensive Review with Research-Backed Findings
**Previous Review:** v3.0.0 (2025-12-13)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Order Flow Trading Theory and Research](#2-order-flow-trading-theory-and-research)
3. [Current Implementation Assessment](#3-current-implementation-assessment)
4. [Strategy Development Guide Compliance](#4-strategy-development-guide-compliance)
5. [Trading Pair Analysis](#5-trading-pair-analysis)
6. [Critical Findings](#6-critical-findings)
7. [Recommendations](#7-recommendations)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [References](#9-references)

---

## 1. Executive Summary

### Overview

The Order Flow strategy v3.1.0 is a momentum-based trading strategy that analyzes real-time trade tape data to detect buy/sell imbalances and volume spikes. The strategy targets XRP/USDT and BTC/USDT pairs on Kraken, generating trading signals when significant order flow pressure is detected alongside volatility-adjusted thresholds.

### Key Strengths

| Aspect | Assessment |
|--------|------------|
| Trade Flow Analysis | Excellent - Primary signal based on actual executed trades rather than order book alone |
| Risk Management | Strong - 2:1 R:R ratio, circuit breaker, position decay handling |
| Fee Awareness | Strong - Validates profitability after round-trip fees |
| Configuration | Comprehensive - Per-pair overrides, asymmetric thresholds |
| Logging | Excellent - Always populates indicators, even on early returns |

### Risk Assessment Summary

| Risk Level | Category | Description |
|------------|----------|-------------|
| LOW | System Bugs | types.py trade slicing bug fixed in v3.1.0 |
| MODERATE | Signal Quality | Relies on 50-trade lookback which may not capture intraday patterns |
| MODERATE | Market Conditions | Strategy optimized for trending momentum, may underperform in ranging markets |
| LOW | Development Guide Compliance | Fully compliant with all requirements |

### Overall Verdict

**READY FOR PAPER TRADING WITH MONITORING**

The strategy is well-architected and follows best practices. The v3.1.0 implementation addresses all critical bugs identified in v3.0.0 review. Recommended for extended paper trading with close attention to performance metrics across different market conditions.

---

## 2. Order Flow Trading Theory and Research

### Academic Foundation

Order flow trading is grounded in market microstructure theory, which studies the mechanics of how trades are executed and how information is incorporated into prices.

#### Trade Flow vs Order Book Imbalance

Research by Silantyev (2019) on cryptocurrency markets demonstrates that **trade flow imbalance is superior to order book imbalance** for explaining contemporaneous price changes. The key insight is:

- Order book imbalance shows *intention* (orders waiting to be filled)
- Trade flow imbalance shows *action* (actual executions)

**Strategy Alignment:** The order_flow strategy correctly prioritizes trade tape analysis over order book imbalance, aligning with academic findings.

#### VPIN for Toxicity Detection

The Volume-Synchronized Probability of Informed Trading (VPIN) metric, developed by Easley et al., has proven effective in cryptocurrency markets for detecting "toxic" order flow that precedes large price moves. Research published in 2025 shows that VPIN significantly predicts future Bitcoin price jumps.

**Strategy Gap Identified:** The current implementation does not include VPIN calculation. This represents an enhancement opportunity.

#### VWAP as Fair Value Benchmark

VWAP (Volume-Weighted Average Price) provides a benchmark for fair value assessment. Research shows that:

- Price above VWAP suggests bullish momentum
- Price below VWAP suggests potential mean reversion opportunity
- Crypto VWAP has higher prediction error than traditional markets due to 24/7 trading

**Strategy Alignment:** The strategy correctly uses VWAP as a secondary confirmation signal, not primary, which is appropriate given crypto market dynamics.

### Industry Best Practices

Modern order flow trading follows a standardized pattern:

1. **Identify Imbalance** - Detect significant buy/sell volume differential
2. **Confirm with Flow** - Verify direction with recent trade tape
3. **Trade Direction** - Enter in direction of imbalance
4. **Protect Capital** - Stop loss below/above imbalance zone
5. **Take Profits** - Exit when imbalance reverses or target reached

**Strategy Alignment:** Excellent - The order_flow strategy follows this exact pattern.

### Cryptocurrency-Specific Considerations

Cryptocurrency markets differ from traditional markets in several important ways:

| Factor | Traditional Markets | Crypto Markets | Strategy Implication |
|--------|--------------------|-----------------|--------------------|
| Trading Hours | Limited | 24/7 | Volume patterns vary by time of day |
| Depth | Generally deep | Variable | Need higher thresholds for volatile periods |
| Maker/Taker | Balanced | Often taker-driven | Trade flow more predictive than order book |
| Manipulation | Regulated | Less regulated | Need robust circuit breaker mechanisms |

---

## 3. Current Implementation Assessment

### Version 3.1.0 Changes from v3.0.0

| Item | v3.0.0 Status | v3.1.0 Status |
|------|---------------|---------------|
| Asymmetric buy/sell thresholds | Recommended | Implemented |
| Symbol config validation | Recommended | Implemented |
| Undefined variable bug (base_threshold) | Present | Fixed |
| VWAP reversion threshold bug | Present | Fixed |
| types.py trade array slicing | Fixed | Verified Fixed |

### Core Signal Generation Flow

The strategy evaluates signals through a multi-stage process:

1. **Circuit Breaker Check** - Blocks trading after 3 consecutive losses for 15 minutes
2. **Time Cooldown Check** - Minimum 5 seconds between signals
3. **Per-Symbol Evaluation** - Each trading pair evaluated independently
4. **Trade Count Cooldown** - Minimum 10 new trades between signals
5. **Imbalance Calculation** - Buy vs sell volume in last 50 trades
6. **Volume Spike Detection** - Last 5 trades vs average
7. **Trade Flow Confirmation** - Validates flow direction matches signal
8. **Fee Profitability Check** - Ensures expected profit exceeds fees
9. **Position Limit Check** - Respects maximum position exposure

### Feature Completeness

| Feature | Status | Quality Assessment |
|---------|--------|-------------------|
| Primary momentum signals | Implemented | Good - Based on imbalance + volume spike |
| VWAP mean reversion signals | Implemented | Good - Secondary confirmation |
| Volatility adjustment | Implemented | Good - Dynamic threshold scaling |
| Trade flow confirmation | Implemented | Good - Additional validation layer |
| Fee-aware profitability | Implemented | Good - Prevents unprofitable trades |
| Micro-price calculation | Implemented | Good - Better price discovery |
| Position decay handling | Implemented | Moderate - Binary decay, consider progressive |
| Trailing stops | Implemented (disabled) | Good when enabled |
| Circuit breaker | Implemented | Excellent - Prevents runaway losses |
| Per-pair PnL tracking | Implemented | Excellent - Enables pair-specific analysis |

### Indicator Logging Quality

The strategy populates comprehensive indicators for debugging and analysis:

**Always Populated (even on early returns):**
- Symbol, trade count, status
- Position side and size
- Consecutive losses count
- Per-pair PnL and trade count

**Populated on Active Evaluation:**
- Imbalance value and direction
- Volume spike magnitude
- VWAP and price deviation
- Micro-price calculation
- Volatility percentage
- Threshold values (buy and sell separately)
- Trade flow alignment status
- Fee profitability status

---

## 4. Strategy Development Guide Compliance

### Required Components Assessment

| Component | Requirement | Status | Notes |
|-----------|-------------|--------|-------|
| STRATEGY_NAME | Lowercase with underscores | PASS | "order_flow" |
| STRATEGY_VERSION | Semantic versioning | PASS | "3.1.0" |
| SYMBOLS | List of trading pairs | PASS | ["XRP/USDT", "BTC/USDT"] |
| CONFIG | Default configuration dict | PASS | Comprehensive with all parameters |
| generate_signal() | Main signal function | PASS | Correct signature and return type |

### Optional Components Assessment

| Component | Status | Quality Notes |
|-----------|--------|---------------|
| on_start() | PASS | Validates config, initializes state comprehensively |
| on_fill() | PASS | Updates position, tracks per-pair metrics, handles circuit breaker |
| on_stop() | PASS | Generates final summary with per-pair breakdown |

### Signal Structure Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| Required fields present | PASS | action, symbol, size, price, reason all populated |
| Stop loss correctly positioned | PASS | Below entry for longs, above for shorts |
| Take profit correctly positioned | PASS | Above entry for longs, below for shorts |
| Reason informativeness | PASS | Includes imbalance, volume, volatility values |
| Metadata usage | PASS | Used for trailing_stop and position_decay flags |

### Position Management Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| Track position in state | PASS | position_side, position_size, position_by_symbol |
| Respect max_position_usd | PASS | Checks and limits before signaling |
| Handle partial closes | PASS | Calculates available size correctly |
| Update on_fill correctly | PASS | Comprehensive state updates including circuit breaker |

### Indicator Logging Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| Always populate indicators | PASS | _build_base_indicators() helper ensures this |
| Include calculation inputs | PASS | All intermediate values logged |
| Include decision factors | PASS | trade_flow_aligned, is_fee_profitable, etc. |

### Risk Management Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| R:R ratio >= 1:1 | PASS | 2:1 ratio for both pairs |
| Stop loss properly calculated | PASS | Price-based percentage calculation |
| Take profit properly calculated | PASS | Price-based percentage calculation |
| Cooldown mechanisms | PASS | Both trade-based and time-based |

---

## 5. Trading Pair Analysis

### XRP/USDT Analysis

#### Market Characteristics (Q4 2024)

Based on Ripple's Q4 2024 XRP Markets Report:

| Metric | Value | Trading Implication |
|--------|-------|---------------------|
| Q4 Price Movement | +280% | High momentum opportunities |
| Daily Spot Volume (avg) | $5B (post-election) | Excellent liquidity |
| Realized Volatility | 160-200% (Dec) | Requires wider thresholds |
| Major Exchange Share | Binance 36%, Upbit 20% | Cross-exchange considerations |

#### Current Strategy Configuration

| Parameter | Value | Assessment |
|-----------|-------|------------|
| buy_imbalance_threshold | 0.30 | APPROPRIATE - 30% imbalance for entry |
| sell_imbalance_threshold | 0.25 | APPROPRIATE - Lower for sells (research-backed) |
| position_size_usd | $25 | CONSERVATIVE - Suitable for paper trading |
| volume_spike_mult | 2.0 | APPROPRIATE - 2x normal volume trigger |
| take_profit_pct | 1.0% | APPROPRIATE - Good for momentum |
| stop_loss_pct | 0.5% | APPROPRIATE - 2:1 R:R maintained |

#### XRP-Specific Findings

**Strength:** XRP exhibits strong momentum characteristics that align well with order flow trading. The high correlation between spot volumes and price movements supports the strategy's approach.

**Consideration:** During the Q4 2024 volatility spike (200%+), the strategy's volatility adjustment mechanism would have increased thresholds by 1.5x, potentially missing some signals. Consider implementing volatility regime classification for better adaptation.

**Consideration:** XRP shows high correlation with BTC movements. Consider adding correlation-aware position management to avoid over-concentration during simultaneous signals.

### BTC/USDT Analysis

#### Market Characteristics

| Metric | Value | Trading Implication |
|--------|-------|---------------------|
| Daily Volume | Highest in crypto | Excellent liquidity, tight spreads |
| Typical Spread | ~0.02% | Minimal slippage concern |
| Trade Frequency | ~1.3 per minute | High frequency ideal for order flow |
| Institutional Presence | Significant | More information-driven flows |

#### Current Strategy Configuration

| Parameter | Value | Assessment |
|-----------|-------|------------|
| buy_imbalance_threshold | 0.25 | APPROPRIATE - Lower for high liquidity |
| sell_imbalance_threshold | 0.20 | APPROPRIATE - Even lower for sells |
| position_size_usd | $50 | APPROPRIATE - Larger for BTC liquidity |
| volume_spike_mult | 1.8 | APPROPRIATE - Lower for more signals |
| take_profit_pct | 0.8% | APPROPRIATE - Slightly tighter for BTC |
| stop_loss_pct | 0.4% | APPROPRIATE - 2:1 R:R maintained |

#### BTC-Specific Findings

**Strength:** Research shows BTC order flow is increasingly driven by takers rather than makers, which benefits trade tape analysis over order book analysis. The strategy's focus on trade flow aligns well with this market structure.

**Consideration:** VPIN research specifically on Bitcoin shows strong predictive power for price jumps. Implementing VPIN would be particularly valuable for BTC trading.

**Consideration:** BTC correlates with traditional markets (S&P 500, gold) during certain periods. Macro-awareness could enhance signal filtering.

### XRP/BTC Analysis (Removed Pair)

#### Removal Decision Validation

The decision to remove XRP/BTC from order_flow strategy (v2.2.0) was **CORRECT** for the following reasons:

1. **Different Trading Objective:** XRP/BTC is better suited for ratio/accumulation trading rather than momentum trading
2. **Lower Liquidity:** Approximately 3x lower daily volume than XRP/USDT
3. **Different Signal Characteristics:** Ratio movements are slower and less suited to order flow momentum signals
4. **Strategy Specialization:** market_making and ratio_trading strategies are better suited

**Recommendation:** No action needed. Removal was appropriate design decision.

---

## 6. Critical Findings

### Finding #1: Strategy Complexity vs Maintainability

**Severity:** MEDIUM
**Category:** Architecture

**Description:** The order_flow.py file has grown to 1200+ lines with 15+ helper functions. While each feature is well-implemented, the overall complexity presents maintainability challenges.

**Impact:**
- Difficult to modify individual features without risk of side effects
- Testing requires comprehensive test coverage
- New developers need significant onboarding time

**Recommendation:** Consider refactoring into modular components:
- Core signal generation
- Risk management utilities
- Position tracking utilities
- Indicator calculation utilities

### Finding #2: Trade Lookback Window Optimization

**Severity:** LOW
**Category:** Strategy Parameters

**Description:** The fixed 50-trade lookback may not be optimal across all market conditions. During high volatility (like XRP's 200% realized vol in Q4 2024), 50 trades may represent only minutes of activity. During low volatility, 50 trades may span hours.

**Impact:**
- Signal timing may be suboptimal in varying conditions
- Momentum detection effectiveness varies by market phase

**Recommendation:** Consider implementing dynamic lookback based on time window or volatility regime.

### Finding #3: Time-of-Day Awareness Gap

**Severity:** LOW
**Category:** Enhancement

**Description:** The strategy does not account for time-of-day volume patterns. Crypto markets show significant volume variation across timezone overlaps.

**Impact:**
- Same thresholds applied during low-volume Asian hours and high-volume US/Europe overlap
- Potentially missed signals during low volume periods
- Potentially false signals during volume spikes at session opens

**Recommendation:** Implement session-aware parameter adjustment:
- Asia hours (00:00-08:00 UTC): Wider thresholds
- Europe hours (08:00-14:00 UTC): Standard thresholds
- US hours (14:00-21:00 UTC): Standard thresholds
- US/Europe overlap (14:00-17:00 UTC): Tighter thresholds (peak liquidity)

### Finding #4: Correlation Risk Management

**Severity:** LOW
**Category:** Risk Management

**Description:** XRP and BTC prices often correlate. Simultaneous signals on both pairs could result in concentrated directional exposure.

**Impact:**
- Maximum position effectively doubled if both pairs signal simultaneously
- Correlation-driven drawdowns not separately managed

**Recommendation:** Consider implementing cross-pair correlation awareness:
- Track aggregate directional exposure across pairs
- Reduce position sizes when multiple pairs signal same direction
- Add total portfolio exposure limit

---

## 7. Recommendations

### High Priority (Implement This Sprint)

#### REC-001: Implement VPIN Calculation

**Rationale:** Research specifically shows VPIN predicts Bitcoin price jumps effectively. Adding VPIN would enhance signal quality particularly for BTC/USDT.

**Benefits:**
- Better detection of informed trading flow
- Early warning for large price moves
- Research-backed improvement

**Suggested Parameters:**
- Volume bucket size: Based on average daily volume / 50
- Look-back: 50 volume buckets
- High VPIN threshold: 0.7+ (consider pausing)

#### REC-002: Implement Volatility Regime Classification

**Rationale:** The current linear volatility scaling (1.0x to 1.5x) may not adequately handle extreme volatility (like XRP's 200%+ in Q4 2024).

**Suggested Regimes:**
- LOW (volatility < 0.3%): Tighter thresholds, normal sizes
- MEDIUM (0.3% - 0.8%): Standard parameters
- HIGH (0.8% - 1.5%): Wider thresholds, consider reduced sizes
- EXTREME (> 1.5%): Consider pausing or significantly reduced activity

### Medium Priority (Next Sprint)

#### REC-003: Add Time-of-Day Awareness

**Rationale:** Volume patterns vary significantly across trading sessions.

**Implementation Approach:**
- Define session windows (UTC-based)
- Apply multipliers to thresholds based on session
- Log session in indicators for analysis

#### REC-004: Progressive Position Decay

**Rationale:** Current position decay is binary (triggered after 5 minutes). A progressive decay would be more nuanced.

**Suggested Approach:**
- 3 minutes: 90% of original TP target
- 4 minutes: 75% of original TP target
- 5 minutes: 50% of original TP target
- 6+ minutes: Close at any profit

#### REC-005: Cross-Pair Correlation Management

**Rationale:** Prevent concentrated directional exposure when pairs move together.

**Suggested Approach:**
- Track total long exposure across all pairs
- Track total short exposure across all pairs
- Apply position reduction when both pairs signal same direction

### Enhancement Priority (Future Sprints)

#### REC-006: Absorption Detection

**Rationale:** Detecting when large resting orders absorb incoming aggression would enhance entry timing.

**Use Case:** Identify when institutional participants are accumulating/distributing before breakouts.

#### REC-007: Order Book Depth Change Tracking

**Rationale:** Changes in order book depth (beyond just imbalance) can signal institutional activity.

**Use Case:** Early warning when large orders are placed or pulled.

#### REC-008: Machine Learning Signal Enhancement

**Rationale:** Historical signal data can be used to train models for better signal filtering.

**Use Case:** Learn which market conditions produce highest quality signals.

---

## 8. Implementation Roadmap

### Phase 1: Immediate (Days 1-2)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Paper trading deployment | Critical | Low | Validate v3.1.0 in live conditions |
| Monitoring dashboard setup | High | Low | Track real-time performance |

### Phase 2: Short Term (Week 1-2)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| VPIN calculation implementation | High | Medium | Enhanced signal quality |
| Volatility regime classification | High | Medium | Better risk management |
| Unit test coverage expansion | Medium | Medium | Prevent regressions |

### Phase 3: Medium Term (Week 3-4)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Time-of-day awareness | Medium | Low | Session-optimized signals |
| Progressive position decay | Medium | Low | Better stale position handling |
| Cross-pair correlation | Medium | Medium | Portfolio-level risk management |

### Phase 4: Long Term (Month 2+)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Absorption detection | Low | High | Advanced institutional flow detection |
| Order book depth tracking | Low | Medium | Complementary signal source |
| ML signal enhancement | Low | High | Adaptive signal quality improvement |

---

## 9. References

### Academic Research

- [Order Flow Analysis of Cryptocurrency Markets](https://www.researchgate.net/publication/332089928_Order_flow_analysis_of_cryptocurrency_markets) - Silantyev (2019) - Foundational research on trade flow vs order book imbalance
- [Bitcoin Wild Moves: Order Flow Toxicity and Price Jumps](https://www.sciencedirect.com/science/article/pii/S0275531925004192) - ScienceDirect (2025) - VPIN research for Bitcoin
- [Deep Learning for VWAP Execution in Crypto Markets](https://arxiv.org/html/2502.13722v1) - arXiv (2025) - Crypto-specific VWAP challenges
- [Microstructure and Market Dynamics in Crypto Markets](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf) - Cornell (2024) - Comprehensive crypto microstructure analysis

### Industry Resources

- [Order Flow Trading In Crypto](https://www.webopedia.com/crypto/learn/order-flow-trading-in-crypto/) - Webopedia - Fundamentals overview
- [Digital Currency Trading with Bookmap](https://bookmap.com/blog/digital-currency-trading-with-bookmap) - Bookmap - Practical order flow tools
- [Order Flow Imbalance - A High Frequency Trading Signal](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html) - Dean Markwick - Technical implementation
- [Mastering VWAP in Crypto Trading](https://www.hyrotrader.com/blog/vwap-trading-strategy/) - HyroTrader (2025) - VWAP strategy guide
- [TWAP vs VWAP in Crypto Trading](https://coinmarketcap.com/academy/article/twap-vs-vwap) - CoinMarketCap Academy

### Market Data

- [Q4 2024 XRP Markets Report](https://ripple.com/insights/q4-2024-xrp-markets-report/) - Ripple - XRP market statistics
- [Q1 2024 XRP Markets Report](https://ripple.com/insights/q1-2024-xrp-markets-report/) - Ripple - XRP market evolution

### Internal Documentation

- Strategy Development Guide v1.1
- Order Flow Strategy Review v3.0.0
- Order Flow Feature Documentation v3.1.0
- Market Making Strategy Review (for comparison patterns)

---

## Appendix A: Configuration Quick Reference

### Global CONFIG Defaults

| Parameter | Default | Purpose |
|-----------|---------|---------|
| imbalance_threshold | 0.30 | Default imbalance trigger (symmetric fallback) |
| buy_imbalance_threshold | 0.30 | Buy signal threshold |
| sell_imbalance_threshold | 0.25 | Sell signal threshold (lower per research) |
| use_asymmetric_thresholds | True | Enable different buy/sell thresholds |
| volume_spike_mult | 2.0 | Volume spike multiplier for confirmation |
| lookback_trades | 50 | Number of trades to analyze |
| position_size_usd | 25.0 | Default trade size in USD |
| max_position_usd | 100.0 | Maximum position exposure |
| min_trade_size_usd | 5.0 | Minimum viable trade size |
| take_profit_pct | 1.0 | Take profit percentage |
| stop_loss_pct | 0.5 | Stop loss percentage |
| cooldown_trades | 10 | Minimum trades between signals |
| cooldown_seconds | 5.0 | Minimum time between signals |
| base_volatility_pct | 0.5 | Baseline volatility for scaling |
| volatility_lookback | 20 | Candles for volatility calculation |
| volatility_threshold_mult | 1.5 | Maximum threshold increase |
| use_trade_flow_confirmation | True | Enable trade flow validation |
| trade_flow_threshold | 0.15 | Minimum trade flow alignment |
| fee_rate | 0.001 | Per-trade fee (0.1%) |
| min_profit_after_fees_pct | 0.05 | Minimum profit after fees |
| use_fee_check | True | Enable fee profitability check |
| use_micro_price | True | Use volume-weighted micro-price |
| use_position_decay | True | Enable stale position handling |
| max_position_age_seconds | 300 | Position age before decay |
| position_decay_tp_multiplier | 0.5 | Reduced TP for stale positions |
| use_trailing_stop | False | Enable trailing stops |
| trailing_stop_activation | 0.3 | Profit % to activate trailing |
| trailing_stop_distance | 0.2 | Distance from high to trail |
| use_circuit_breaker | True | Enable loss circuit breaker |
| max_consecutive_losses | 3 | Losses before cooldown |
| circuit_breaker_minutes | 15 | Cooldown period |

### Per-Symbol Overrides

| Parameter | XRP/USDT | BTC/USDT |
|-----------|----------|----------|
| buy_imbalance_threshold | 0.30 | 0.25 |
| sell_imbalance_threshold | 0.25 | 0.20 |
| position_size_usd | $25 | $50 |
| volume_spike_mult | 2.0 | 1.8 |
| take_profit_pct | 1.0% | 0.8% |
| stop_loss_pct | 0.5% | 0.4% |

---

## Appendix B: State Structure Reference

### Position Tracking

| State Key | Type | Purpose |
|-----------|------|---------|
| position_side | string or None | 'long', 'short', or None |
| position_size | float | Current position in USD |
| position_by_symbol | dict | Per-symbol position tracking |
| position_entries | dict | Entry details for trailing stops |

### Signal Control

| State Key | Type | Purpose |
|-----------|------|---------|
| last_signal_idx | int | Trade index of last signal |
| total_trades_seen | int | Running trade count |
| last_signal_time | datetime or None | Timestamp of last signal |

### Performance Metrics

| State Key | Type | Purpose |
|-----------|------|---------|
| pnl_by_symbol | dict | Cumulative PnL per trading pair |
| trades_by_symbol | dict | Trade count per pair |
| wins_by_symbol | dict | Winning trade count per pair |
| losses_by_symbol | dict | Losing trade count per pair |

### Risk Management

| State Key | Type | Purpose |
|-----------|------|---------|
| consecutive_losses | int | Current loss streak |
| circuit_breaker_time | datetime or None | When circuit breaker triggered |

---

**Document Version:** 1.0
**Last Updated:** 2025-12-13
**Author:** Strategy Architecture Review (Claude Opus 4.5)
**Next Review:** After Phase 2 implementation or significant performance data
