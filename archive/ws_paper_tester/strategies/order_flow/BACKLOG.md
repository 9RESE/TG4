# Order Flow Strategy Backlog

**Strategy Version:** 5.0.0
**Last Updated:** 2025-12-14
**Created From:** Deep Review v7.0

---

## Implemented Recommendations

### REC-003: Add XRP/BTC Configuration - IMPLEMENTED v4.4.0

**Priority:** MEDIUM | **Effort:** LOW | **Status:** IMPLEMENTED

**Description:**
Added XRP/BTC ratio pair support with research-backed configuration parameters.

**Implementation (config.py):**
```python
'XRP/BTC': {
    'buy_imbalance_threshold': 0.35,     # Higher: 7-10x lower liquidity than USDT pairs
    'sell_imbalance_threshold': 0.30,    # Higher: requires stronger signal confirmation
    'imbalance_threshold': 0.35,         # Fallback threshold
    'position_size_usd': 15.0,           # Smaller: higher slippage risk in thin market
    'volume_spike_mult': 2.2,            # Higher: need stronger volume confirmation
    'take_profit_pct': 1.5,              # Wider: XRP 1.55x more volatile than BTC
    'stop_loss_pct': 0.75,               # Wider: maintains 2:1 R:R (1.5/0.75)
    'cooldown_trades': 15,               # Higher: fewer quality signals in low liquidity
},
```

**Research Basis (December 2025):**
- Liquidity: ~1,608 BTC/24h (~$160M) - 7-10x less than XRP/USDT
- Volatility: 234% daily, XRP is 1.55x more volatile than BTC
- Correlation: 0.84 (declining 24.86% over 90 days)
- Spread: Wider than USDT pairs due to lower liquidity
- Dynamics: Ratio pair behavior with mean reversion potential

**Sources:**
- [CoinGecko XRP Statistics](https://coinlaw.io/xrp-statistics/)
- [MacroAxis XRP-BTC Correlation](https://www.macroaxis.com/invest/pair-correlation/XRP.CC/BTC.CC/XRP-vs-Bitcoin)
- [Gate.com XRP-BTC Correlation Analysis](https://www.gate.com/crypto-wiki/article/how-is-the-price-correlation-between-xrp-and-bitcoin)
- [CME Group XRP Analysis](https://www.cmegroup.com/insights/economic-research/2025/how-xrp-relates-to-the-crypto-universe-and-the-broader-economy.html)

---

## Implemented Recommendations (v5.0.0)

### REC-005: Volume Anomaly Detection - IMPLEMENTED v5.0.0

**Priority:** LOW | **Effort:** MEDIUM | **Status:** IMPLEMENTED

**Description:**
Added basic wash trading indicators to filter manipulated signals during low-liquidity periods.

**Implementation (indicators.py):**
```python
def check_volume_anomaly(trades: Tuple, config: Dict, current_price: float, previous_price: float) -> Dict:
    """
    Detect potential wash trading patterns.

    Indicators:
    1. Volume consistency vs rolling average - detects abnormal volume levels
    2. Repetitive exact-size trades - detects potential wash trading patterns
    3. Volume spike without corresponding price movement - detects fake volume

    Returns:
        Dict with anomaly_detected, anomaly_types, confidence_score, details
    """
```

**Configuration (config.py):**
```python
'use_volume_anomaly_detection': True,
'volume_anomaly_pause_on_detect': True,
'volume_anomaly_low_ratio': 0.2,        # Flag if volume < 20% of rolling avg
'volume_anomaly_high_ratio': 5.0,       # Flag if volume > 5x rolling avg
'volume_anomaly_repetitive_threshold': 0.4,  # Flag if >40% trades same size
'volume_anomaly_repetitive_tolerance': 0.001,  # Size match tolerance (0.1%)
'volume_anomaly_price_move_threshold': 0.001,  # Min price move (0.1%)
'volume_anomaly_volume_spike_threshold': 3.0,  # Volume spike multiplier
'volume_anomaly_lookback_trades': 100,
```

**Research Background:**
- $2.57B suspected wash trading activity (Chainalysis 2025)
- Wash trading intensifies during low legitimate volume periods
- Power law distribution analysis can detect anomalies

---

### REC-006: Session-Specific VPIN Thresholds - IMPLEMENTED v5.0.0

**Priority:** LOW | **Effort:** MEDIUM | **Status:** IMPLEMENTED

**Description:**
Implemented session-aware VPIN thresholds since VPIN effectiveness varies with liquidity conditions.

**Implementation (config.py):**
```python
'use_session_vpin_thresholds': True,
'session_vpin_thresholds': {
    'ASIA': 0.65,           # More conservative during thin liquidity
    'EUROPE': 0.70,         # Standard threshold
    'US': 0.70,             # Standard threshold
    'US_EUROPE_OVERLAP': 0.75,  # Allow more signals during deep liquidity
    'OFF_HOURS': 0.60,      # Most conservative for thinnest liquidity
},
```

**Thresholds:**
| Session | VPIN Threshold | Rationale |
|---------|---------------|-----------|
| ASIA | 0.65 | More conservative during thin liquidity |
| EUROPE | 0.70 | Standard threshold |
| US_EUROPE_OVERLAP | 0.75 | Allow more signals during deep liquidity |
| US | 0.70 | Standard threshold |
| OFF_HOURS | 0.60 | Most conservative for thinnest liquidity |

**Integration:**
- signal.py updated to get session-specific VPIN threshold before comparison
- Indicators include `vpin_session_aware` flag and active threshold

---

## Known Limitations

### 1. Market Manipulation Vulnerability

The strategy relies on trade tape data without specific wash trading detection. Current mitigations are partial:
- VPIN may not detect wash trading specifically
- Volume spike requirement accepts inflated volume
- Session awareness reduces but doesn't eliminate exposure

**Mitigation Status:** Acceptable for paper testing; monitor for false signals during Asia/OFF_HOURS sessions.

### 2. Pseudonymous Trading

Cannot verify trade authenticity on pseudonymous blockchain exchanges. No practical mitigation available - inherent to crypto market structure.

### 3. Single Exchange Data

Strategy currently assumes single exchange data feed. Cross-exchange arbitrage or manipulation may not be detected.

---

## Future Enhancement Ideas

1. **Cross-Exchange VPIN**: Aggregate VPIN from multiple exchanges for more robust signal
2. **Machine Learning Integration**: Train model on historical manipulation patterns
3. **Adaptive Thresholds**: Auto-adjust thresholds based on recent performance
4. **Orderbook Depth Integration**: Use orderbook depth to validate signal strength
5. **Whale Detection**: Flag unusually large trades for manual review

---

## Implementation Tracking

| REC-ID | Status | Target Version | Notes |
|--------|--------|---------------|-------|
| REC-001 | IMPLEMENTED | 4.3.0 | Trade flow check for VWAP short |
| REC-002 | IMPLEMENTED | 4.3.0 | OFF_HOURS session type |
| REC-003 | IMPLEMENTED | 4.4.0 | XRP/BTC ratio pair configuration |
| REC-004 | IMPLEMENTED | 4.3.0 | Extended decay timing |
| REC-005 | IMPLEMENTED | 5.0.0 | Volume anomaly detection |
| REC-006 | IMPLEMENTED | 5.0.0 | Session-specific VPIN thresholds |
| REC-007 | IMPLEMENTED | 4.3.0 | Trailing stop documentation |
| REC-008 | IMPLEMENTED | 4.3.0 | This backlog file |

---

*Document Version: 1.2*
*Created: 2025-12-14*
*Updated: 2025-12-14 - REC-005, REC-006 implemented in v5.0.0*
