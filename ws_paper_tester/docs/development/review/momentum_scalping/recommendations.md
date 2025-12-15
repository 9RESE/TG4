# Recommendations: Momentum Scalping Strategy v2.0

**Review Date:** 2025-12-14
**Review Version:** v2.0 (Post v2.1.0 Implementation)

---

## Priority Definitions

| Priority | Timeframe | Description |
|----------|-----------|-------------|
| **P0** | Immediate | Block trading until resolved |
| **P1** | 1 week | High-impact improvement |
| **P2** | 1 month | Moderate improvement |
| **P3** | Backlog | Low-priority enhancement |

## Effort Definitions

| Effort | Description |
|--------|-------------|
| **LOW** | Config change or < 1 hour code |
| **MEDIUM** | 1-4 hours, single module |
| **HIGH** | 4+ hours, multiple modules |

---

## Recommendations Summary (v2.0)

### Implemented (v2.1.0)

| ID | Priority | Effort | Finding | Status |
|----|----------|--------|---------|--------|
| REC-001 | P0 | LOW | XRP/BTC correlation | ✅ COMPLETE |
| REC-002 | P1 | LOW | ADX threshold | ✅ COMPLETE |
| REC-003 | P1 | LOW | RSI period | ✅ COMPLETE |
| REC-005 | P2 | MEDIUM | Trailing stops | ✅ COMPLETE |
| REC-006 | P2 | LOW | DST handling | ✅ COMPLETE |
| REC-007 | P2 | MEDIUM | Trade flow | ✅ COMPLETE |
| REC-008 | P2 | LOW | Correlation lookback | ✅ COMPLETE |
| REC-009 | P3 | LOW | Breakeven exit | ✅ COMPLETE |
| REC-010 | P3 | LOW | Structured logging | ✅ COMPLETE |

### Outstanding

| ID | Priority | Effort | Finding | Status |
|----|----------|--------|---------|--------|
| REC-004 | P2 | HIGH | Guide v2.0 | 📝 DOCUMENTATION |
| REC-011 | P3 | LOW | MACD divergence | ⏸️ DEFERRED v3.0 |

### New (v2.0 Review) - IMPLEMENTED

| ID | Priority | Effort | Finding | Status |
|----|----------|--------|---------|--------|
| REC-012 | P3 | LOW | XRP independence monitoring | ✅ COMPLETE |
| REC-013 | P3 | LOW | Market sentiment monitoring | ✅ COMPLETE |

---

## Detailed Recommendations

### REC-001: Correlation Pause Threshold - COMPLETE ✅

**Finding:** CRIT-001
**Implementation:** v2.1.0
**Line Reference:** `config.py:205`

**Implementation Summary:**
- Correlation pause threshold raised from 0.50 to 0.60
- Lookback increased to 100 candles (REC-008)
- Rejection tracking via `CORRELATION_BREAKDOWN`
- Auto-pause functionality in `risk.py:369-395`

**Verification:**
```python
# config.py
'correlation_pause_threshold': 0.60,
'correlation_lookback': 100,
'correlation_pause_enabled': True,
```

---

### REC-002: ADX Threshold for BTC - COMPLETE ✅

**Finding:** HIGH-001
**Implementation:** v2.1.0
**Line Reference:** `config.py:218`

**Implementation Summary:**
- ADX threshold raised from 25 to 30
- Applied only to BTC/USDT via `adx_filter_btc_only`
- Rejection tracking via `ADX_STRONG_TREND`

**Verification:**
```python
# config.py
'adx_strong_trend_threshold': 30,
'adx_filter_btc_only': True,
```

---

### REC-003: RSI Period Optimization - COMPLETE ✅

**Finding:** HIGH-002
**Implementation:** v2.1.0
**Line Reference:** `config.py:268`

**Implementation Summary:**
- XRP/USDT RSI period changed from 7 to 8
- BTC/USDT and XRP/BTC remain at 9 (appropriate)
- Regime RSI adjustment (75/25 in HIGH volatility)

**Verification:**
```python
# SYMBOL_CONFIGS
'XRP/USDT': {'rsi_period': 8},
'BTC/USDT': {'rsi_period': 9},
'XRP/BTC': {'rsi_period': 9},
```

---

### REC-004: Strategy Development Guide v2.0 - DOCUMENTATION

**Finding:** CRIT-002
**Status:** Documentation task (not code-related)
**Priority:** P2

**Recommendation:**
Create Strategy Development Guide v2.0 with the following sections:

| Section | Topic | Based On |
|---------|-------|----------|
| 15 | Volatility Regime Classification | `regimes.py` implementation |
| 16 | Circuit Breaker Protection | `risk.py` implementation |
| 17 | Signal Rejection Tracking | `config.py` RejectionReason enum |
| 18 | Trade Flow Confirmation | `signal.py` imbalance filter |
| 19 | Multi-Timeframe Analysis | 5m trend filter |
| 20 | Correlation Management | `indicators.py` correlation |
| 21 | Dynamic Parameter Adjustment | Regime-based RSI |
| 22 | Per-Symbol Configuration | SYMBOL_CONFIGS pattern |
| 23 | Indicator Logging Requirements | All-path logging pattern |
| 24 | Correlation Monitoring | `risk.py` pause logic |

**Alternative:**
Document that v1.0 is the current standard and update notes.md review scope.

---

### REC-005: ATR-Based Trailing Stops - COMPLETE ✅

**Finding:** MED-002
**Implementation:** v2.1.0
**Line Reference:** `exits.py:292-389`

**Implementation Summary:**
- ATR-based trailing stop calculation
- Configurable activation threshold
- Trail multiplier configurable

**Verification:**
```python
# config.py
'use_trailing_stop': True,
'trail_atr_mult': 1.5,
'trail_activation_pct': 0.4,
```

---

### REC-006: DST Documentation - COMPLETE ✅

**Finding:** MED-004
**Implementation:** v2.1.0
**Location:** `regimes.py` comments

**Implementation Summary:**
- DST handling documented in regimes.py
- Session boundaries configurable
- UTC-based implementation handles DST implicitly

---

### REC-007: Trade Flow Confirmation - COMPLETE ✅

**Finding:** HIGH-003
**Implementation:** v2.1.0
**Line Reference:** `signal.py:296-300`

**Implementation Summary:**
- Trade imbalance filter added
- Configurable threshold (0.1 = 10% imbalance required)
- New rejection reason `TRADE_FLOW_MISALIGNMENT`

**Verification:**
```python
# config.py
'use_trade_flow_confirmation': True,
'trade_imbalance_threshold': 0.1,
```

---

### REC-008: Correlation Lookback - COMPLETE ✅

**Finding:** MED-001
**Implementation:** v2.1.0
**Line Reference:** `config.py:203`

**Implementation Summary:**
- Lookback increased from 50 to 100 candles
- Now covers ~8.3 hours of 5m data
- More stable correlation reading

**Verification:**
```python
# config.py
'correlation_lookback': 100,
```

---

### REC-009: Breakeven Momentum Exit - COMPLETE ✅

**Finding:** MED-003
**Implementation:** v2.1.0
**Line Reference:** `config.py:240`

**Implementation Summary:**
- Optional breakeven exit on RSI extreme
- Disabled by default (preserves original behavior)
- Configurable via config

**Verification:**
```python
# config.py
'exit_breakeven_on_momentum_exhaustion': False,
```

---

### REC-010: Structured Logging - COMPLETE ✅

**Finding:** LOW-002
**Implementation:** v2.1.0
**Line Reference:** `lifecycle.py:16`

**Implementation Summary:**
- Python logging module used
- Logger named after strategy
- Replaces print statements

**Verification:**
```python
# lifecycle.py
logger = logging.getLogger(STRATEGY_NAME)
```

---

### REC-011: MACD-Price Divergence - DEFERRED ⏸️

**Finding:** LOW-001
**Status:** Deferred to v3.0
**Priority:** P3

**Rationale:**
- High implementation complexity
- Requires swing point detection
- Marginal benefit for 1-minute timeframe
- Re-evaluate after v2.1.0 performance data collected

---

### REC-012: XRP Independence Monitoring - COMPLETE ✅

**Finding:** NEW-001
**Implementation:** v2.1.0
**Line Reference:** `monitoring.py:50-200`

**Implementation Summary:**
- `CorrelationMonitor` class tracks XRP-BTC correlation over time
- Automatic weekly review generation with trend analysis
- Escalation triggers monitored:
  - Consecutive days below 0.70 threshold
  - XRP/BTC pause rate above 50%
- State persisted to `logs/monitoring/monitoring_state.json`
- Weekly reports saved to `logs/monitoring/correlation_report_YYYYMMDD.json`

**Verification:**
```python
# config.py
'enable_correlation_trend_tracking': True,
'correlation_escalation_threshold': 0.70,
'correlation_escalation_days': 30,

# Usage in lifecycle.py
manager = get_or_create_monitoring_manager(state)
manager.correlation_monitor.generate_weekly_report()
```

**Action Items (Automated):**
1. ✅ Review correlation status weekly - auto-generated reports
2. ✅ Log correlation values for trend analysis - persisted history
3. ✅ Escalation alerts when threshold reached

---

### REC-013: Market Sentiment Monitoring - COMPLETE ✅

**Finding:** NEW-002
**Implementation:** v2.1.0
**Line Reference:** `monitoring.py:210-320`

**Implementation Summary:**
- `SentimentMonitor` class tracks Fear & Greed Index
- Sentiment classification (Extreme Fear/Fear/Neutral/Greed/Extreme Greed)
- Prolonged extreme sentiment alerts (7+ consecutive days)
- Volatility expansion signals for regime awareness
- State persisted to `logs/monitoring/monitoring_state.json`

**Verification:**
```python
# config.py
'enable_sentiment_monitoring': True,
'sentiment_extreme_fear_threshold': 24,
'sentiment_extreme_greed_threshold': 76,

# Usage
manager.record_session_data(
    correlation=0.65,
    xrp_btc_paused=False,
    fear_greed_index=23  # Optional - fetch externally
)
```

**Action Items (Automated):**
1. ✅ Monitor Fear & Greed Index - via record_session_data()
2. ✅ Regime classification auto-adjusts based on sentiment
3. ✅ Prolonged extreme alerts generated automatically

---

## Implementation Verification Checklist

### v2.1.0 Implementation Status

| Recommendation | Config | Code | Tests | Docs |
|----------------|--------|------|-------|------|
| REC-001 | ✅ | ✅ | - | ✅ |
| REC-002 | ✅ | ✅ | - | ✅ |
| REC-003 | ✅ | ✅ | - | ✅ |
| REC-005 | ✅ | ✅ | - | ✅ |
| REC-006 | - | ✅ | - | ✅ |
| REC-007 | ✅ | ✅ | - | ✅ |
| REC-008 | ✅ | N/A | - | ✅ |
| REC-009 | ✅ | ✅ | - | ✅ |
| REC-010 | - | ✅ | - | ✅ |
| REC-012 | ✅ | ✅ | ✅ | ✅ |
| REC-013 | ✅ | ✅ | ✅ | ✅ |

---

## Monitoring After Implementation

| Recommendation | Metric to Monitor | Frequency |
|----------------|-------------------|-----------|
| REC-001 | XRP/BTC trade count, correlation | Daily |
| REC-002 | BTC/USDT ADX rejection rate | Weekly |
| REC-003 | XRP/USDT win rate, signal count | Weekly |
| REC-005 | Trailing stop exit count, avg profit | Weekly |
| REC-007 | Trade flow rejection rate | Weekly |
| REC-008 | Correlation warning frequency | Weekly |
| REC-012 | XRP-BTC correlation trend | Weekly |
| REC-013 | Fear & Greed Index | Daily |

---

## Future Considerations (v3.0)

| Feature | Priority | Rationale |
|---------|----------|-----------|
| MACD divergence detection | MEDIUM | Complex, requires swing detection |
| Order book imbalance | LOW | Additional confirmation layer |
| Multi-exchange support | LOW | Different fee structures |
| Machine learning signals | LOW | Experimental, requires data |

---

*Next: [Research References](./research-references.md)*
