# Order Flow Strategy v5.0.0 - Session VPIN & Volume Anomaly Detection

**Release Date:** 2025-12-14
**Previous Version:** 4.4.0 (XRP/BTC Ratio Pair Support)
**Status:** Production Ready - Paper Testing Recommended

---

## Overview

Version 5.0.0 implements two deferred recommendations from deep-review-v7.0:
- **REC-005:** Volume Anomaly Detection - Basic wash trading indicators
- **REC-006:** Session-Specific VPIN Thresholds - Liquidity-aware VPIN thresholds

These features enhance market manipulation detection and improve signal quality during varying liquidity conditions.

---

## Feature 1: Session-Specific VPIN Thresholds (REC-006)

### Background

VPIN (Volume-Synchronized Probability of Informed Trading) effectiveness varies with market liquidity. A fixed threshold of 0.7 may be too permissive during thin liquidity periods (allowing toxic signals) or too restrictive during peak liquidity (filtering valid signals).

### Implementation

**File:** config.py:108-121

```python
# REC-006 (v5.0.0): Session-specific VPIN thresholds
'use_session_vpin_thresholds': True,
'session_vpin_thresholds': {
    'ASIA': 0.65,           # More conservative during thin liquidity
    'EUROPE': 0.70,         # Standard threshold
    'US': 0.70,             # Standard threshold
    'US_EUROPE_OVERLAP': 0.75,  # Allow more signals during deep liquidity
    'OFF_HOURS': 0.60,      # Most conservative for thinnest liquidity
},
```

**File:** signal.py:233-263 - Updated VPIN check

```python
# Get session-specific VPIN threshold if enabled
if config.get('use_session_vpin_thresholds', True):
    session_vpin_thresholds = config.get('session_vpin_thresholds', {})
    vpin_threshold = session_vpin_thresholds.get(session.name, vpin_threshold)
```

### Threshold Rationale

| Session | VPIN Threshold | Liquidity Level | Rationale |
|---------|---------------|-----------------|-----------|
| OFF_HOURS (21:00-24:00 UTC) | 0.60 | Lowest | Post-US, pre-Asia gap. Highest manipulation risk. |
| ASIA (00:00-08:00 UTC) | 0.65 | Low | Asian session typically has lower volume. |
| EUROPE (08:00-14:00 UTC) | 0.70 | Medium | Standard European trading hours. |
| US (17:00-21:00 UTC) | 0.70 | Medium-High | US-only session after overlap. |
| US_EUROPE_OVERLAP (14:00-17:00 UTC) | 0.75 | Highest | Peak global liquidity, lowest manipulation risk. |

### Impact on Trading

- **More signals during peak hours**: US_EUROPE_OVERLAP allows higher VPIN before pausing
- **Better protection during thin markets**: OFF_HOURS and ASIA filter more aggressively
- **Session-aware indicators**: Logs include `vpin_session_aware` flag and active threshold

---

## Feature 2: Volume Anomaly Detection (REC-005)

### Background

Wash trading and market manipulation are significant concerns in crypto markets (~$2.57B suspected wash trading per Chainalysis 2025). Volume anomaly detection provides an additional layer of protection beyond VPIN.

### Implementation

**File:** indicators.py:146-308 - New `check_volume_anomaly()` function

```python
def check_volume_anomaly(
    trades: Tuple,
    config: Dict[str, Any],
    current_price: float = 0.0,
    previous_price: float = 0.0
) -> Dict[str, Any]:
    """
    Detect potential wash trading patterns.

    Implements three indicators:
    1. Volume consistency vs rolling average
    2. Repetitive exact-size trades
    3. Volume spike without corresponding price movement
    """
```

### Detection Methods

#### 1. Volume Consistency Check

Detects abnormal volume levels compared to rolling average.

```python
'volume_anomaly_low_ratio': 0.2,   # Flag if volume < 20% of rolling avg
'volume_anomaly_high_ratio': 5.0,  # Flag if volume > 5x rolling avg
```

**Anomaly Types:**
- `low_volume`: Suspiciously quiet market (potential setup for manipulation)
- `high_volume`: Suspiciously high volume (potential wash trading)

#### 2. Repetitive Trade Detection

Detects patterns where too many trades have identical or near-identical sizes.

```python
'volume_anomaly_repetitive_threshold': 0.4,    # Flag if >40% trades same size
'volume_anomaly_repetitive_tolerance': 0.001,  # Size match tolerance (0.1%)
```

**Anomaly Type:** `repetitive_trades`

**Why it matters:** Legitimate trading produces varied order sizes. High repetition suggests automated wash trading.

#### 3. Volume-Price Divergence

Detects volume spikes without corresponding price movement (fake volume).

```python
'volume_anomaly_price_move_threshold': 0.001,  # Min price move (0.1%)
'volume_anomaly_volume_spike_threshold': 3.0,  # Volume spike multiplier
```

**Anomaly Type:** `volume_price_divergence`

**Why it matters:** Real volume moves price. Volume without price movement suggests wash trading.

### Configuration

**File:** config.py:123-138

```python
# REC-005 (v5.0.0): Volume Anomaly Detection
'use_volume_anomaly_detection': True,
'volume_anomaly_pause_on_detect': True,
'volume_anomaly_low_ratio': 0.2,
'volume_anomaly_high_ratio': 5.0,
'volume_anomaly_repetitive_threshold': 0.4,
'volume_anomaly_repetitive_tolerance': 0.001,
'volume_anomaly_price_move_threshold': 0.001,
'volume_anomaly_volume_spike_threshold': 3.0,
'volume_anomaly_lookback_trades': 100,
```

### Confidence Scoring

The system calculates a confidence score based on the number of anomalies detected:

| Anomalies Detected | Confidence Score | Action |
|--------------------|------------------|--------|
| 0 | 0.0 | Continue trading |
| 1 | 0.5 | Pause trading (if enabled) |
| 2 | 0.75 | Pause trading |
| 3 | 0.95 | Pause trading |

### Return Value

```python
{
    'anomaly_detected': bool,
    'anomaly_types': ['low_volume', 'repetitive_trades', ...],
    'confidence_score': 0.0-1.0,
    'details': {
        'volume_consistency': {...},
        'repetitive_trades': {...},
        'volume_price_divergence': {...}
    }
}
```

---

## Integration

### Signal Flow

```
Trades → VPIN Check → Volume Anomaly Check → Imbalance Check → Signal Generation
                ↓                ↓
          [vpin_pause]    [volume_anomaly_pause]
```

### Rejection Reasons

**File:** config.py:56-57 - Added new rejection reason

```python
class RejectionReason(Enum):
    ...
    VOLUME_ANOMALY = "volume_anomaly"  # REC-005 (v5.0.0)
```

### Indicators Output

When active, indicators include:

```python
{
    # Session VPIN (REC-006)
    'vpin_threshold': 0.65,  # Session-specific threshold
    'vpin_session_aware': True,

    # Volume Anomaly (REC-005)
    'volume_anomaly_detected': False,
    'volume_anomaly_types': [],
    'volume_anomaly_confidence': 0.0,
}
```

---

## Configuration Comparison

| Parameter | v4.4.0 | v5.0.0 | Notes |
|-----------|--------|--------|-------|
| VPIN threshold | 0.7 (fixed) | 0.60-0.75 (session-based) | More adaptive |
| Volume anomaly detection | N/A | Enabled | New feature |
| Rejection reasons | 14 | 15 (+VOLUME_ANOMALY) | Better tracking |

---

## Risk Assessment

### Benefits

| Benefit | Impact |
|---------|--------|
| Better manipulation protection | Fewer false signals from wash trading |
| Session-appropriate sensitivity | More signals during peak hours, better protection during thin markets |
| Detailed anomaly logging | Easier to diagnose market quality issues |

### Potential Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Over-filtering during legitimate high volume | LOW | `volume_anomaly_high_ratio: 5.0` is conservative |
| Missing signals during OFF_HOURS | LOW | Expected - thin liquidity has lower signal quality anyway |
| Repetitive detection false positives | LOW | 40% threshold is conservative; legitimate markets rarely exceed 20% |

### Configuration Recommendations

**Conservative (default):**
```python
'use_session_vpin_thresholds': True,
'use_volume_anomaly_detection': True,
'volume_anomaly_pause_on_detect': True,
```

**Aggressive (more signals, higher risk):**
```python
'use_session_vpin_thresholds': True,
'session_vpin_thresholds': {
    'ASIA': 0.70,
    'EUROPE': 0.75,
    'US': 0.75,
    'US_EUROPE_OVERLAP': 0.80,
    'OFF_HOURS': 0.65,
},
'use_volume_anomaly_detection': True,
'volume_anomaly_pause_on_detect': False,  # Log only, don't pause
```

---

## Testing Recommendations

### Pre-Paper Testing Validation

1. **Session Threshold Verification:**
   - Confirm correct session detection at boundary hours
   - Verify threshold changes as sessions rotate
   - Check indicator logs include session-aware VPIN threshold

2. **Volume Anomaly Detection:**
   - Verify detection triggers during known manipulation events
   - Confirm false positive rate is acceptable
   - Check anomaly details in rejection logs

### Metrics to Monitor

| Metric | Target | Action if Not Met |
|--------|--------|-------------------|
| VPIN rejections during OFF_HOURS | Higher than peak hours | Verify threshold is more conservative |
| Volume anomaly false positive rate | < 5% | Adjust thresholds |
| Signal quality improvement | Win rate stable or improved | Verify filters are working |
| Anomaly detection correlation with losses | Detected anomalies correlate with losing trades | Validates detection effectiveness |

---

## Related Files

### Modified
- `ws_paper_tester/strategies/order_flow/config.py` - v5.0.0, session VPIN thresholds, volume anomaly config
- `ws_paper_tester/strategies/order_flow/indicators.py` - Added `check_volume_anomaly()` function
- `ws_paper_tester/strategies/order_flow/signal.py` - Integrated both features
- `ws_paper_tester/strategies/order_flow/BACKLOG.md` - Marked REC-005, REC-006 as implemented

### Created
- `docs/development/features/order_flow/order-flow-v5.0.md` - This document

---

## Version History

- **5.0.0** (2025-12-14): Session VPIN thresholds and volume anomaly detection
  - REC-005: Volume anomaly detection with 3 detection methods
  - REC-006: Session-specific VPIN thresholds
  - New rejection reason: VOLUME_ANOMALY
- **4.4.0** (2025-12-14): XRP/BTC ratio pair support
- **4.3.0** (2025-12-14): Deep review v7.0 implementation
- **4.2.0** (2025-12-14): Deep review v5.0 implementation
- **4.1.1** (2025-12-14): Modular refactoring
- **4.1.0** (2025-12-14): Review recommendations
- **4.0.0** (2025-12-14): Major refactor with VPIN, regimes, sessions

---

**Document Version:** 1.0
**Last Updated:** 2025-12-14
