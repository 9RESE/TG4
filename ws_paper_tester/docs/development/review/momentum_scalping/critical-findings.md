# Critical Findings: Momentum Scalping Strategy

**Review Date:** 2025-12-14

---

## Severity Definitions

| Level | Definition | Action Required |
|-------|------------|-----------------|
| **CRITICAL** | Strategy may incur significant losses | Immediate action before trading |
| **HIGH** | Reduced effectiveness or elevated risk | Action within 1 week |
| **MEDIUM** | Suboptimal performance | Action within 1 month |
| **LOW** | Minor improvements | Backlog |

---

## 1. CRITICAL Findings

### CRIT-001: XRP/BTC Correlation Breakdown

**Severity:** CRITICAL
**Affected Component:** `risk.py`, `signal.py`
**Affected Pair:** XRP/BTC

**Finding:**
XRP-BTC correlation has declined from historical ~0.85 to current 0.40-0.67 range. The 90-day correlation shows a 24.86% decline. This fundamentally undermines momentum scalping on the XRP/BTC pair because:

1. Momentum in XRP no longer reliably predicts XRP/BTC ratio movement
2. Momentum in BTC no longer reliably predicts XRP/BTC ratio movement
3. The ratio may move contrary to individual asset momentum

**Current Mitigation:**
- Correlation monitoring implemented (`risk.py:312-366`)
- Pause threshold at 0.50 (`config.py:201`)
- Rejection reason `CORRELATION_BREAKDOWN` added

**Residual Risk:**
- Correlation lookback is 50 candles on 5m (~4 hours)
- Rapid correlation changes may not be detected in time
- Pause threshold 0.50 may be too permissive given current range

**Recommendation:** See REC-001

---

### CRIT-002: Strategy Development Guide v2.0 Missing

**Severity:** CRITICAL
**Affected Component:** Documentation

**Finding:**
Review scope requested compliance against Strategy Development Guide v2.0 Sections 15-18, 22, 24. Only v1.0 is available at `ws_paper_tester/docs/development/strategy-development-guide.md`.

**Impact:**
- Cannot verify compliance against stated requirements
- v2.0 features may not be properly documented
- Review completeness is uncertain

**Recommendation:** See REC-004

---

## 2. HIGH Findings

### HIGH-001: ADX Threshold May Be Too Permissive for BTC

**Severity:** HIGH
**Affected Component:** `risk.py:401-447`
**Affected Pair:** BTC/USDT
**Line Reference:** `config.py:211`

**Finding:**
ADX threshold is set to 25, but research indicates:
- 2024 market data shows BTC ADX > 30 during major rallies
- BTC "tends to trend when at its maximum and bounce back at minimum"
- Current BTC ADX is ~24.81, just below threshold

**Risk:**
- ADX of 25.1 would allow entries that may immediately face strong trend
- Momentum scalping typically fails in ADX > 30 conditions
- BTC at/near ATH ($126K in Oct 2025) exhibits strong trending

**Current Configuration:**
```python
'adx_strong_trend_threshold': 25,
'adx_filter_btc_only': True,
```

**Recommendation:** See REC-002

---

### HIGH-002: RSI Period Too Fast for Crypto Markets

**Severity:** HIGH
**Affected Component:** `indicators.py:70-112`
**Affected Pair:** XRP/USDT
**Line Reference:** `config.py:82`, `config.py:238`

**Finding:**
RSI period 7 for XRP/USDT is optimized for ultra-fast scalping but research indicates:
- Crypto sustains overbought/oversold conditions longer than traditional markets
- Higher timeframes (9-14 period) provide more reliable signals
- 1-minute RSI with period 7 generates significant noise

**Academic Finding:**
> "Cryptocurrencies can remain in overbought or oversold conditions longer than traditional markets"

**Current Configuration:**
- XRP/USDT: RSI period 7 (aggressive)
- BTC/USDT: RSI period 9 (appropriate)
- XRP/BTC: RSI period 9 (appropriate)

**Recommendation:** See REC-003

---

### HIGH-003: Trade Flow Confirmation Not Implemented

**Severity:** HIGH
**Affected Component:** `signal.py`
**Affected Pairs:** All

**Finding:**
Section 18 (inferred) requires trade flow confirmation using:
- Trade tape analysis
- Buy/sell imbalance
- VWAP calculation

The strategy only uses volume confirmation, not trade flow direction.

**Available but Unused:**
```python
# DataSnapshot provides:
data.get_vwap('XRP/USDT', n_trades=50)
data.get_trade_imbalance('XRP/USDT', n_trades=50)
```

**Impact:**
- Cannot confirm momentum direction from actual order flow
- Volume spike may be sell-side, contradicting long signal
- Missing confirmation layer reduces signal quality

**Recommendation:** See REC-007

---

## 3. MEDIUM Findings

### MED-001: Correlation Lookback May Be Insufficient

**Severity:** MEDIUM
**Affected Component:** `indicators.py:587-648`
**Line Reference:** `config.py:199`

**Finding:**
Correlation lookback is 50 candles on 5m timeframe (~4 hours). Given:
- XRP showing independence from BTC in 2025
- Correlation declining 24.86% over 90 days
- Rapid correlation changes possible during news events

**Risk:**
- 4-hour lookback may miss rapid decoupling
- May not capture structural correlation changes
- May generate false "correlation OK" signals

**Recommendation:** See REC-008

---

### MED-002: No Trailing Stop Implementation

**Severity:** MEDIUM
**Affected Component:** `exits.py`
**Affected Pairs:** All

**Finding:**
Strategy Development Guide v1.0 Section 4 discusses trailing stops as "Manual Implementation" but the momentum scalping strategy does not implement them.

**Impact:**
- Profitable positions may give back gains before TP hit
- Time-based exit (180s) may trigger before optimal exit
- No profit protection mechanism beyond fixed TP

**Note:** `position_entries` already tracks `highest_price`/`lowest_price` in `lifecycle.py:151, 185`, providing the foundation for trailing stops.

**Recommendation:** See REC-005

---

### MED-003: Momentum Exhaustion Exit Only When Profitable

**Severity:** MEDIUM
**Affected Component:** `exits.py:204-271`
**Line Reference:** `exits.py:248`

**Finding:**
Momentum exhaustion exit only triggers when `pnl_pct > 0`:
```python
# exits.py:248
if pnl_pct <= 0:
    return None
```

**Risk:**
- RSI extreme (e.g., RSI > 80 for long) indicates reversal imminent
- If position is at breakeven, momentum exhaustion won't trigger
- May lead to stop loss instead of breakeven exit

**Rationale:** The current logic prevents exiting at a loss due to RSI alone, which has merit. However, it may miss reversal signals.

**Recommendation:** Consider optional breakeven exit on RSI extreme (configurable).

---

### MED-004: Session DST Handling Not Documented

**Severity:** MEDIUM
**Affected Component:** `regimes.py:97-145`
**Status:** REC-006 deferred from v1.0 review

**Finding:**
Session boundaries are UTC-based and configurable, but DST handling is not documented. During DST transitions:
- US markets shift by 1 hour relative to UTC
- European markets shift by 1 hour relative to UTC
- Session boundaries may be misaligned

**Impact:**
- OFF_HOURS session may include peak volume periods
- US_EUROPE_OVERLAP may be incorrectly classified
- Session-based position sizing may be miscalibrated

**Recommendation:** See REC-006

---

## 4. LOW Findings

### LOW-001: MACD-Price Divergence Not Detected

**Severity:** LOW
**Affected Component:** `indicators.py`
**Status:** REC-005 deferred from v1.0 review

**Finding:**
MACD-price divergence (price making higher highs while MACD makes lower highs) is a powerful reversal signal not currently implemented.

**Rationale for LOW severity:**
- High implementation complexity
- Marginal benefit for 1-minute scalping
- Requires swing point detection

**Recommendation:** Consider for v3.0 after v2.0 performance evaluation.

---

### LOW-002: Validation Warnings Printed to Console

**Severity:** LOW
**Affected Component:** `lifecycle.py:47-50`
**Line Reference:** `lifecycle.py:49`

**Finding:**
Configuration warnings are printed to console:
```python
for error in errors:
    print(f"[momentum_scalping] Config warning: {error}")
```

**Impact:**
- Warnings may be missed in production logs
- No structured logging (e.g., JSON format)
- Inconsistent with JSONL logging elsewhere

**Recommendation:** Use structured logging instead of print statements.

---

### LOW-003: Fee Rate Hardcoded Assumption

**Severity:** LOW
**Affected Component:** `config.py:177`
**Line Reference:** `config.py:177`

**Finding:**
Fee rate is 0.1% (0.001) which assumes standard Kraken maker/taker fees. Different exchanges or VIP tiers may have different fees.

```python
'fee_rate': 0.001,  # 0.1% per trade
```

**Impact:**
- Fee profitability check may be incorrect for other exchanges
- VIP users may have lower fees (0.06% - 0.08%)

**Recommendation:** Document fee rate assumptions or make exchange-aware.

---

## 5. Finding Summary

| ID | Severity | Finding | Status |
|----|----------|---------|--------|
| CRIT-001 | CRITICAL | XRP/BTC correlation breakdown | Mitigated, residual risk |
| CRIT-002 | CRITICAL | Guide v2.0 missing | Documentation gap |
| HIGH-001 | HIGH | ADX threshold too permissive | Needs adjustment |
| HIGH-002 | HIGH | RSI period too fast | Needs evaluation |
| HIGH-003 | HIGH | No trade flow confirmation | Not implemented |
| MED-001 | MEDIUM | Correlation lookback short | Needs evaluation |
| MED-002 | MEDIUM | No trailing stops | Not implemented |
| MED-003 | MEDIUM | Momentum exit only when profitable | By design |
| MED-004 | MEDIUM | DST handling undocumented | Documentation gap |
| LOW-001 | LOW | No MACD divergence | Deferred |
| LOW-002 | LOW | Console print warnings | Minor |
| LOW-003 | LOW | Fee rate assumption | Minor |

---

## 6. Risk Heat Map

```
                    IMPACT
                Low   Med   High  Critical
           ┌────────────────────────────────┐
     High  │ LOW-  MED-  HIGH- CRIT-       │
           │ 001   002   001   001         │
           │ LOW-  MED-  HIGH-             │
LIKELIHOOD │ 002   001   002               │
           │ LOW-  MED-  HIGH-             │
     Med   │ 003   003   003               │
           │       MED-                    │
     Low   │       004        CRIT-002     │
           └────────────────────────────────┘
```

---

*Next: [Recommendations](./recommendations.md)*
