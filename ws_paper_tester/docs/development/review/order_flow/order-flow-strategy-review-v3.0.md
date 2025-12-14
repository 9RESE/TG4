# Order Flow Strategy Deep Review v3.0.0

**Review Date:** 2025-12-13
**Version Reviewed:** 3.0.0
**Reviewer:** Strategy Architecture Review (Extended Analysis)
**Status:** Active Review with Findings and Recommendations

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Strategy Fundamentals](#2-strategy-fundamentals)
3. [Version 3.0.0 Implementation Review](#3-version-300-implementation-review)
4. [Strategy Development Guide Compliance](#4-strategy-development-guide-compliance)
5. [Critical Findings](#5-critical-findings)
6. [Order Flow Trading Research Analysis](#6-order-flow-trading-research-analysis)
7. [Trading Pair Analysis](#7-trading-pair-analysis)
8. [Recommendations](#8-recommendations)
9. [Implementation Priority](#9-implementation-priority)
10. [References](#10-references)

---

## 1. Executive Summary

### Overview

The Order Flow strategy v3.0.0 represents a major refactor implementing recommendations from the v2.2.0 review. The strategy analyzes real-time trade tape data to detect buy/sell imbalances and volume spikes, generating momentum-based trading signals for XRP/USDT and BTC/USDT pairs.

### Key Findings Summary

| Category | Status | Notes |
|----------|--------|-------|
| v2.2 Review Items Implemented | 12/14 | Most critical and high priority items addressed |
| Strategy Guide Compliance | PASS | Meets all required components |
| Critical Issues Found | 1 | Trade array slicing bug in types.py |
| High Priority Issues | 2 | Remaining improvements needed |
| Enhancements | 5 | Research-backed improvements |

### Risk Assessment

- **Overall Risk Level:** MODERATE (improved from HIGH in v2.2)
- **Primary Concern:** types.py trade array slicing affects VWAP and trade imbalance calculations
- **Secondary Concern:** Strategy complexity - consider simplification for maintainability
- **Trading Readiness:** Ready for paper trading with monitoring

### Version 3.0.0 Implementation Status

| Review Item | Status | Implementation |
|-------------|--------|----------------|
| CRIT-OF-001: Trade array slicing | FIXED | Line 521: `trades[-lookback:]` |
| CRIT-OF-002: Indicator logging | FIXED | `_build_base_indicators()` helper |
| HIGH-OF-001: Per-pair PnL tracking | FIXED | `pnl_by_symbol`, `trades_by_symbol` in state |
| HIGH-OF-002: Config validation | FIXED | `_validate_config()` function |
| HIGH-OF-003: Trade flow confirmation | FIXED | `_is_trade_flow_aligned()` function |
| HIGH-OF-004: Fee-aware profitability | FIXED | `_check_fee_profitability()` function |
| HIGH-OF-005: R:R ratio improvement | FIXED | 2:1 ratio (1.0% TP, 0.5% SL) |
| MED-OF-001: Micro-price calculation | FIXED | `_calculate_micro_price()` function |
| MED-OF-002: Position decay handling | FIXED | `_check_position_decay()` function |
| MED-OF-003: Trailing stop support | FIXED | `_calculate_trailing_stop()` function |
| MED-OF-004: Configurable min trade | FIXED | `min_trade_size_usd` in CONFIG |
| ENH-OF-007: Position entry tracking | FIXED | `position_entries` state tracking |
| ENH-OF-008: Circuit breaker | FIXED | `_check_circuit_breaker()` function |

---

## 2. Strategy Fundamentals

### Order Flow Trading Theory

Order flow trading is based on the principle that price movements are caused by imbalances in supply and demand. The strategy monitors the trade tape (time and sales) to detect:

1. **Buy/Sell Volume Imbalance**: When buy volume significantly exceeds sell volume (or vice versa), it indicates directional pressure
2. **Volume Spikes**: Sudden increases in trading activity often precede price moves
3. **VWAP Deviation**: Price relationship to Volume-Weighted Average Price provides context for fair value

### Strategy Logic Flow

```
Trade Tape Data
      |
      v
Calculate Buy/Sell Volume Imbalance
      |
      v
Detect Volume Spikes (last 5 vs avg)
      |
      v
VWAP Confirmation Check
      |
      v
Volatility-Adjusted Thresholds
      |
      v
Trade Flow Confirmation (HIGH-OF-003)
      |
      v
Fee Profitability Check (HIGH-OF-004)
      |
      v
Generate Signal (if conditions met)
```

### Core Signal Conditions

**Buy Signal (Long Entry):**
- Imbalance > effective_threshold
- Volume spike > volume_spike_mult
- Trade flow confirms buy direction
- Fee-profitable after round-trip

**Sell Signal (Close Long):**
- Negative imbalance with existing long position
- Volume spike indicates selling pressure
- Trade flow confirms sell direction

**Short Signal:**
- Imbalance < -effective_threshold
- Volume spike > volume_spike_mult
- Trade flow confirms sell direction
- No existing long position

**VWAP Mean Reversion (Secondary):**
- Buy pressure + price below VWAP
- Sell pressure + price above VWAP
- Reduced position size (75% of normal)

---

## 3. Version 3.0.0 Implementation Review

### Fixed Issues Analysis

#### CRIT-OF-001: Trade Array Slicing

**Previous Bug:** `trades[:lookback]` returned oldest trades
**Fix Applied:** Line 521 now correctly uses `trades[-lookback:]`

**Verification:** The strategy's internal trade analysis is correct. However, the underlying `DataSnapshot` methods still have this bug (see Critical Findings).

#### CRIT-OF-002: Indicator Logging

**Implementation:** New `_build_base_indicators()` helper function ensures indicators are always populated, even on early returns.

**Quality:** Indicators now include:
- Symbol, trade count, status
- Position side and size
- Consecutive losses count
- Per-pair PnL and trade count

#### HIGH-OF-001: Per-Pair PnL Tracking

**Implementation:**
- `state['pnl_by_symbol']` - cumulative P&L per trading pair
- `state['trades_by_symbol']` - trade count per pair
- `state['wins_by_symbol']` - winning trade count
- `state['losses_by_symbol']` - losing trade count

**Quality:** Comprehensive tracking enables pair-specific performance analysis.

#### HIGH-OF-002: Config Validation

**Implementation:** `_validate_config()` function validates:
- Required positive values (position_size_usd, stop_loss_pct, etc.)
- Bounds checks (imbalance_threshold 0.1-0.8, fee_rate 0-0.01)
- R:R ratio warnings

**Quality:** Good coverage. Consider adding validation for symbol-specific configs.

#### HIGH-OF-003: Trade Flow Confirmation

**Implementation:** `_is_trade_flow_aligned()` function:
- Checks if trade flow direction matches signal direction
- Configurable threshold (default 0.15)
- Can be disabled via `use_trade_flow_confirmation`

**Issue:** Uses `data.get_trade_imbalance()` which has the slicing bug.

#### HIGH-OF-004: Fee-Aware Profitability

**Implementation:** `_check_fee_profitability()` function:
- Calculates round-trip fees
- Compares expected profit vs minimum threshold
- Default: 0.1% fee rate, 0.05% minimum profit

**Quality:** Good implementation. Consider dynamic fee rates based on market conditions.

#### HIGH-OF-005: R:R Ratio Improvement

**Implementation:**
- XRP/USDT: 1.0% TP, 0.5% SL (2:1 R:R)
- BTC/USDT: 0.8% TP, 0.4% SL (2:1 R:R)

**Quality:** Appropriate for momentum strategy. Research supports 2:1 or higher for order flow trading.

### New Features in v3.0.0

#### Micro-Price Calculation (MED-OF-001)

Volume-weighted micro-price provides better price discovery:
```
micro = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
```

**Quality:** Good implementation. Used for more accurate price reference.

#### Position Decay Handling (MED-OF-002)

Stale positions (>5 minutes) get reduced TP requirements:
- `max_position_age_seconds`: 300
- `position_decay_tp_multiplier`: 0.5 (50% of original TP)

**Quality:** Helps exit stuck positions. Consider making decay progressive rather than binary.

#### Trailing Stop Support (MED-OF-003)

Trailing stops protect profits:
- `trailing_stop_activation`: 0.3% profit to activate
- `trailing_stop_distance`: 0.2% from high

**Quality:** Disabled by default. Good for momentum trading when enabled.

#### Circuit Breaker (ENH-OF-008)

Stops trading after consecutive losses:
- `max_consecutive_losses`: 3
- `circuit_breaker_minutes`: 15

**Quality:** Important risk management feature. Resets on winning trade or after cooldown.

---

## 4. Strategy Development Guide Compliance

### Required Components

| Requirement | Status | Notes |
|-------------|--------|-------|
| `STRATEGY_NAME` | PASS | `"order_flow"` - lowercase with underscores |
| `STRATEGY_VERSION` | PASS | `"3.0.0"` - semantic versioning |
| `SYMBOLS` | PASS | `["XRP/USDT", "BTC/USDT"]` |
| `CONFIG` | PASS | Comprehensive default configuration |
| `generate_signal()` | PASS | Correct signature and return type |

### Optional Components

| Requirement | Status | Notes |
|-------------|--------|-------|
| `on_start()` | PASS | Validates config, initializes state |
| `on_fill()` | PASS | Updates position, tracks per-pair metrics |
| `on_stop()` | PASS | Generates final summary |

### Signal Structure Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| Required fields (action, symbol, size, price, reason) | PASS | All present |
| Optional stop_loss | PASS | Correctly calculated per direction |
| Optional take_profit | PASS | Correctly calculated per direction |
| Reason informativeness | PASS | Includes imbalance, volume, volatility values |
| Metadata usage | PASS | Used for trailing_stop, position_decay flags |

### Stop Loss/Take Profit Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| Long SL below entry | PASS | `price * (1 - sl_pct / 100)` |
| Long TP above entry | PASS | `price * (1 + tp_pct / 100)` |
| Short SL above entry | PASS | `price * (1 + sl_pct / 100)` |
| Short TP below entry | PASS | `price * (1 - tp_pct / 100)` |

### Position Management Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| Track position in state | PASS | `position_side`, `position_size` |
| Respect max_position_usd | PASS | Checks before signaling |
| Handle partial closes | PASS | Calculates available size |
| Update on_fill correctly | PASS | Comprehensive state updates |

### Indicator Logging Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| Always populate indicators | PASS | `_build_base_indicators()` on early returns |
| Include calculation inputs | PASS | Comprehensive indicator set |
| Include decision factors | PASS | trade_flow_aligned, is_fee_profitable, etc. |

---

## 5. Critical Findings

### CRIT-NEW-001: types.py Trade Array Slicing Bug

**Location:** `ws_tester/types.py:147, 159`

**Description:** The `DataSnapshot.get_vwap()` and `DataSnapshot.get_trade_imbalance()` methods use `trades[:n_trades]` which retrieves the FIRST n trades (oldest) instead of the LAST n trades (newest).

**Current Code (types.py:147):**
```python
def get_vwap(self, symbol: str, n_trades: int = 50) -> Optional[float]:
    trades = self.trades.get(symbol, ())
    if not trades:
        return None
    trades = trades[:n_trades]  # BUG: Gets OLDEST trades
```

**Impact on order_flow.py:**
- Line 385: `data.get_trade_imbalance(symbol, n_trades)` - trade flow confirmation uses stale data
- Line 566: `data.get_vwap(symbol, lookback)` - VWAP calculated from oldest trades
- Line 611: `data.get_trade_imbalance(symbol, lookback)` - signal confirmation uses stale data

**Severity:** CRITICAL
- VWAP calculations reference outdated price/volume
- Trade flow confirmation based on historical data
- Signal generation may be delayed or inverted

**Recommended Fix (types.py):**
```python
def get_vwap(self, symbol: str, n_trades: int = 50) -> Optional[float]:
    trades = self.trades.get(symbol, ())
    if not trades:
        return None
    trades = trades[-n_trades:]  # CORRECT: Get NEWEST trades
```

**Note:** While order_flow.py's internal trade analysis (line 521) was correctly fixed in v3.0.0, the reliance on types.py methods reintroduces the bug for VWAP and trade flow confirmation.

### HIGH-NEW-001: Symbol Config Validation Gap

**Location:** `strategies/order_flow.py:138-183`

**Description:** The `_validate_config()` function validates global config but not symbol-specific overrides in `SYMBOL_CONFIGS`.

**Impact:**
- Invalid per-symbol configurations go undetected
- Could lead to incorrect signal generation for specific pairs

**Recommendation:** Add validation loop for SYMBOL_CONFIGS:
```python
for symbol, sym_cfg in SYMBOL_CONFIGS.items():
    for key in ['stop_loss_pct', 'take_profit_pct', 'position_size_usd']:
        if key in sym_cfg and sym_cfg[key] <= 0:
            errors.append(f"{symbol}.{key} must be positive")
```

### HIGH-NEW-002: Trade Flow Confirmation Threshold Asymmetry

**Location:** `strategies/order_flow.py:365-392`

**Description:** The `_is_trade_flow_aligned()` function uses symmetric thresholds for buy and sell confirmation, but cryptocurrency markets often exhibit asymmetric behavior.

**Current Implementation:**
```python
if direction == 'buy':
    return trade_flow > threshold  # Same threshold for buy
elif direction == 'sell':
    return trade_flow < -threshold  # Same threshold for sell
```

**Research Insight:** Studies show that selling pressure in crypto markets is often more impactful than buying pressure of equal magnitude. Consider asymmetric thresholds.

---

## 6. Order Flow Trading Research Analysis

### Academic Foundation

#### Trade Flow vs Order Book Imbalance

Research by Silantyev (2019) on BitMex data demonstrates that **trade flow imbalance is superior to order book imbalance** for explaining contemporaneous price changes. The strategy correctly prioritizes trade tape analysis over order book imbalance.

**Strategy Alignment:** GOOD - Primary signal based on trade tape imbalance

#### VPIN for Toxicity Detection

[ScienceDirect (2025)](https://www.sciencedirect.com/science/article/pii/S0275531925004192) research on Bitcoin shows that Volume-Synchronized Probability of Informed Trading (VPIN) significantly predicts future price jumps.

**Strategy Gap:** The strategy does not implement VPIN. This could enhance signal quality by detecting "toxic" order flow before major price moves.

#### VWAP as Fair Value Benchmark

[Zarattini & Aziz (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351) show VWAP-based strategies outperform in detecting market imbalances:
- Long positions above VWAP: Momentum continuation
- Long positions below VWAP: Mean reversion opportunity

**Strategy Alignment:** GOOD - Strategy implements both momentum (primary) and VWAP mean reversion (secondary) signals

#### Deep Learning for VWAP Execution

[arXiv (2025)](https://arxiv.org/html/2502.13722v1) demonstrates that cryptocurrency markets require specialized VWAP execution due to higher prediction error margins.

**Strategy Implication:** Consider that VWAP in crypto is less reliable than traditional markets. The strategy's use of VWAP as a secondary confirmation (not primary signal) is appropriate.

### Industry Best Practices

#### Order Flow Imbalance Strategy Pattern

From industry sources, the standard order flow strategy pattern is:

1. **Identify Imbalance** - Detect significant buy/sell volume differential
2. **Confirm with Flow** - Verify direction with recent trade tape
3. **Trade Direction** - Enter in direction of imbalance
4. **Protect Capital** - Stop loss below/above imbalance zone
5. **Take Profits** - Exit when imbalance reverses or target reached

**Strategy Alignment:** EXCELLENT - The order_flow strategy follows this pattern

#### Absorption Detection

Advanced order flow traders look for "absorption" - when resting orders absorb incoming aggressive orders without price movement. This indicates institutional accumulation/distribution.

**Strategy Gap:** No absorption detection implemented. This could enhance entry timing.

### Research-Based Improvements

Based on academic and industry research, the following enhancements would improve the strategy:

1. **VPIN Implementation** - Detect toxic order flow for better signal timing
2. **Absorption Detection** - Identify institutional activity
3. **Asymmetric Thresholds** - Different thresholds for buy vs sell
4. **Volatility Regime Classification** - Adapt strategy to market conditions
5. **Time-of-Day Awareness** - Crypto volume patterns vary by hour

---

## 7. Trading Pair Analysis

### XRP/USDT Analysis

#### Market Characteristics

| Metric | Value | Implication |
|--------|-------|-------------|
| Typical Spread | ~0.03% | Very tight, good for momentum |
| Daily Volume | High | Sufficient liquidity |
| Volatility | Moderate-High | Good for order flow signals |
| Trade Frequency | ~1 per 2 min | Adequate for lookback analysis |

#### Strategy Configuration Assessment

| Parameter | Current | Assessment |
|-----------|---------|------------|
| `imbalance_threshold` | 0.30 | APPROPRIATE - 30% imbalance for entry |
| `position_size_usd` | $25 | APPROPRIATE - Small size for testing |
| `volume_spike_mult` | 2.0 | APPROPRIATE - 2x normal volume |
| `take_profit_pct` | 1.0% | APPROPRIATE - 2:1 R:R with 0.5% SL |
| `stop_loss_pct` | 0.5% | APPROPRIATE - Reasonable risk per trade |

#### Pair-Specific Recommendations

1. **Consider tighter imbalance threshold (0.25-0.28)** for more signals during high volatility
2. **Add BTC correlation filter** - XRP often moves with BTC
3. **Implement time-based sizing** - Reduce size during low volume hours

### BTC/USDT Analysis

#### Market Characteristics

| Metric | Value | Implication |
|--------|-------|-------------|
| Typical Spread | ~0.02% | Very tight, highly liquid |
| Daily Volume | Very High | Excellent liquidity |
| Volatility | Moderate | More stable than XRP |
| Trade Frequency | ~1.3 per min | High frequency ideal for order flow |

#### Strategy Configuration Assessment

| Parameter | Current | Assessment |
|-----------|---------|------------|
| `imbalance_threshold` | 0.25 | APPROPRIATE - Lower for more liquid market |
| `position_size_usd` | $50 | APPROPRIATE - Larger for BTC liquidity |
| `volume_spike_mult` | 1.8 | APPROPRIATE - Lower for more signals |
| `take_profit_pct` | 0.8% | APPROPRIATE - Slightly tighter for BTC |
| `stop_loss_pct` | 0.4% | APPROPRIATE - 2:1 R:R maintained |

#### Pair-Specific Recommendations

1. **Consider even lower imbalance threshold (0.20-0.22)** for BTC's high liquidity
2. **Implement funding rate awareness** - Impacts short positions
3. **Add macro correlation** - BTC correlates with S&P 500 and gold

### XRP/BTC Analysis (Removed in v2.2.0)

#### Removal Rationale - VALIDATED

The decision to remove XRP/BTC from order_flow was correct because:

1. **Lower liquidity** - 96,604 XRP daily vs 286,505 for XRP/USDT
2. **Different objective** - Ratio trading (asset accumulation) vs momentum
3. **Strategy specialization** - market_making and ratio_trading better suited

#### Recommendation

No action needed. XRP/BTC removal was appropriate design decision.

---

## 8. Recommendations

### Critical Priority (Immediate)

#### REC-001: Fix types.py Trade Array Slicing

**File:** `ws_tester/types.py`

**Change Required:**
- Line 147: Change `trades[:n_trades]` to `trades[-n_trades:]`
- Line 159: Change `trades[:n_trades]` to `trades[-n_trades:]`

**Impact:** Fixes VWAP and trade imbalance calculations for entire system

### High Priority (This Sprint)

#### REC-002: Add Symbol Config Validation

Extend `_validate_config()` to validate SYMBOL_CONFIGS entries.

**Benefits:**
- Catch invalid per-pair configurations at startup
- Prevent runtime errors from misconfiguration

#### REC-003: Implement Asymmetric Thresholds

Add separate thresholds for buy vs sell signals:

**CONFIG Addition:**
```
'buy_imbalance_threshold': 0.30,
'sell_imbalance_threshold': 0.25,  # Lower for sell (more impactful)
```

**Rationale:** Research shows crypto selling pressure has larger price impact

### Medium Priority (Next Sprint)

#### REC-004: Add VPIN Calculation

Implement Volume-Synchronized Probability of Informed Trading for toxicity detection.

**Benefits:**
- Predict large price moves
- Filter out noise from toxic flow
- Research-backed improvement

#### REC-005: Implement Volatility Regime Classification

Instead of linear volatility scaling, use discrete regimes:

**Regimes:**
- LOW (volatility < 0.3%): Tighter thresholds, smaller sizes
- MEDIUM (0.3% - 0.8%): Standard parameters
- HIGH (> 0.8%): Wider thresholds, consider pausing

**Benefits:**
- More nuanced response to market conditions
- Better risk management during volatility spikes

### Enhancement Priority (Future)

#### REC-006: Implement Absorption Detection

Detect when large resting orders absorb incoming aggression without price movement.

**Benefits:**
- Identify institutional accumulation/distribution
- Better entry timing on breakouts

#### REC-007: Add Order Book Depth Analysis

Track depth changes at multiple price levels for institutional footprint.

**Benefits:**
- Early warning of large order placement/removal
- Complement trade tape analysis

#### REC-008: Time-of-Day Awareness

Adjust parameters based on time of day and market session:

**Sessions:**
- Asia hours (00:00-08:00 UTC): Lower volume, wider thresholds
- Europe hours (08:00-14:00 UTC): Moderate volume
- US hours (14:00-21:00 UTC): High volume, standard thresholds
- Overlap (14:00-17:00 UTC): Peak volume, tighter thresholds

---

## 9. Implementation Priority

### Phase 1: Critical Fix (Immediate - Day 1)

| Item | File | Effort | Impact |
|------|------|--------|--------|
| REC-001: types.py trade slicing | `ws_tester/types.py` | Low | Critical |

### Phase 2: High Priority (Days 2-3)

| Item | File | Effort | Impact |
|------|------|--------|--------|
| REC-002: Symbol config validation | `order_flow.py` | Low | High |
| REC-003: Asymmetric thresholds | `order_flow.py` | Low | Medium |

### Phase 3: Research Features (Week 2)

| Item | Effort | Impact |
|------|--------|--------|
| REC-004: VPIN calculation | Medium | High |
| REC-005: Volatility regimes | Medium | Medium |

### Phase 4: Advanced Features (Week 3+)

| Item | Effort | Impact |
|------|--------|--------|
| REC-006: Absorption detection | High | High |
| REC-007: Order book depth | Medium | Medium |
| REC-008: Time-of-day awareness | Medium | Medium |

---

## 10. References

### Academic Papers

- [Order Flow Analysis of Cryptocurrency Markets](https://medium.com/@eliquinox/order-flow-analysis-of-cryptocurrency-markets-b479a0216ad8) - Silantyev (2019)
- [VWAP: The Holy Grail for Day Trading](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351) - Zarattini & Aziz (2024)
- [Bitcoin Wild Moves: Order Flow Toxicity](https://www.sciencedirect.com/science/article/pii/S0275531925004192) - ScienceDirect (2025)
- [Deep Learning for VWAP Execution in Crypto](https://arxiv.org/html/2502.13722v1) - arXiv (2025)
- [Order Flow Imbalance - A High Frequency Trading Signal](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html) - Dean Markwick

### Industry Resources

- [Order Flow Trading In Crypto](https://www.webopedia.com/crypto/learn/order-flow-trading-in-crypto/) - Webopedia
- [Digital Currency Trading with Bookmap](https://bookmap.com/blog/digital-currency-trading-with-bookmap) - Bookmap
- [Expert Analysis of Bitcoin's Toxic Order Flow](https://the-kingfisher.medium.com/bitcoins-toxic-order-flow-tof-acab6b4a983a) - The Kingfisher
- [Mastering VWAP in Crypto Trading](https://www.hyrotrader.com/blog/vwap-trading-strategy/) - HyroTrader

### Internal References

- `ws_paper_tester/docs/development/review/market_maker/strategy-development-guide.md` v1.1
- `ws_paper_tester/docs/development/review/order_flow/order-flow-strategy-review-v2.2.md`
- `ws_paper_tester/strategies/market_making.py` v1.5.0 (feature parity reference)

---

**Document Version:** 1.0
**Last Updated:** 2025-12-13
**Author:** Strategy Architecture Review (Claude Opus 4.5)
**Next Review:** After Phase 1 implementation

---

## Appendix A: Configuration Reference

### Default CONFIG

```
imbalance_threshold: 0.3       # Min imbalance to trigger (30%)
volume_spike_mult: 2.0         # Volume spike multiplier
lookback_trades: 50            # Trades to analyze
position_size_usd: 25.0        # Trade size in USD
max_position_usd: 100.0        # Maximum exposure
min_trade_size_usd: 5.0        # Minimum per trade
take_profit_pct: 1.0           # Take profit at 1.0%
stop_loss_pct: 0.5             # Stop loss at 0.5%
cooldown_trades: 10            # Min trades between signals
cooldown_seconds: 5.0          # Min time between signals
base_volatility_pct: 0.5       # Baseline volatility
volatility_lookback: 20        # Candles for volatility
volatility_threshold_mult: 1.5 # High volatility multiplier
use_trade_flow_confirmation: True
trade_flow_threshold: 0.15     # Min trade flow alignment
fee_rate: 0.001                # 0.1% per trade
min_profit_after_fees_pct: 0.05
use_fee_check: True
use_micro_price: True
use_position_decay: True
max_position_age_seconds: 300
position_decay_tp_multiplier: 0.5
use_trailing_stop: False
trailing_stop_activation: 0.3
trailing_stop_distance: 0.2
use_circuit_breaker: True
max_consecutive_losses: 3
circuit_breaker_minutes: 15
```

### XRP/USDT Overrides

```
imbalance_threshold: 0.30
position_size_usd: 25.0
volume_spike_mult: 2.0
take_profit_pct: 1.0  # 2:1 R:R
stop_loss_pct: 0.5
```

### BTC/USDT Overrides

```
imbalance_threshold: 0.25
position_size_usd: 50.0
volume_spike_mult: 1.8
take_profit_pct: 0.8
stop_loss_pct: 0.4  # 2:1 R:R
```

## Appendix B: Strategy State Reference

```
state = {
    # Initialization
    'initialized': True,
    'config_validated': True,
    'config_warnings': [],

    # Position tracking
    'position_side': 'long' | 'short' | None,
    'position_size': 0.0,
    'position_by_symbol': {},
    'position_entries': {},

    # Signal control
    'last_signal_idx': 0,
    'total_trades_seen': 0,
    'last_signal_time': datetime | None,

    # Per-pair metrics
    'pnl_by_symbol': {},
    'trades_by_symbol': {},
    'wins_by_symbol': {},
    'losses_by_symbol': {},

    # Circuit breaker
    'consecutive_losses': 0,
    'circuit_breaker_time': datetime | None,

    # Logging
    'fills': [],
    'indicators': {},

    # Final summary (set by on_stop)
    'final_summary': {},
}
```
