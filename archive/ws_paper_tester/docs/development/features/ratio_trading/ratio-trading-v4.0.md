# Ratio Trading Strategy v4.0.0 - Deep Review Optimization

**Release Date:** 2025-12-14
**Previous Version:** 3.0.0
**Status:** Production Ready
**Review Reference:** `docs/development/review/ratio_trading/ratio-trading-strategy-review-v4.0.md`

---

## Overview

Version 4.0.0 of the Ratio Trading strategy implements the HIGH and MEDIUM priority recommendations from the deep review v4.0 analysis. This release focuses on enhanced correlation protection based on research showing XRP/BTC correlation has declined ~24.86% over 90 days (from ~0.84 to ~0.54), approaching the previous warning threshold.

The key changes enable correlation-based trading pause by default and raise correlation thresholds to provide earlier warnings and more conservative protection against correlation breakdown.

## Changes from v3.0.0

### REC-023: Enable Correlation Pause by Default (HIGH Priority)

**Problem:** The correlation monitoring system was implemented in v3.0 but the pause feature was disabled by default (`correlation_pause_enabled: False`). This means the strategy would continue trading even if correlation dropped below the pause threshold.

**Research Evidence:**
- XRP/BTC correlation declined from ~0.84 to ~0.54 over 90 days (24.86% decline)
- Current correlation (~0.54) is close to the previous warning threshold (0.5)
- Pairs trading requires stable cointegration/correlation relationships

**Solution:** Enable correlation pause by default for conservative operation.

**Configuration Change:**
```python
# Before (v3.0.0)
'correlation_pause_enabled': False,   # Disabled by default

# After (v4.0.0)
'correlation_pause_enabled': True,    # REC-023: Enabled by default
```

**Behavior:**
- Trading automatically pauses when correlation drops below `correlation_pause_threshold`
- Warnings logged when correlation drops below `correlation_warning_threshold`
- Protects against trading during correlation breakdown periods
- Can be disabled via config override if desired

### REC-024: Raise Correlation Thresholds (MEDIUM Priority)

**Problem:** The previous thresholds (warning: 0.5, pause: 0.3) were too permissive given the current XRP/BTC correlation (~0.54).

**Research Evidence:**
- Research suggests pairs trading requires correlation > 0.6 for reliability
- Current correlation (~0.54) barely above the old warning threshold
- Earlier warnings allow for manual intervention before significant losses

**Solution:** Raise both thresholds to provide earlier alerts and more conservative pause triggers.

**Configuration Changes:**
```python
# Before (v3.0.0)
'correlation_warning_threshold': 0.5, # Warn if correlation below this
'correlation_pause_threshold': 0.3,   # Pause trading if below this

# After (v4.0.0)
'correlation_warning_threshold': 0.6, # REC-024: Raised from 0.5
'correlation_pause_threshold': 0.4,   # REC-024: Raised from 0.3
```

**New Correlation Behavior:**

| Correlation Level | Action |
|------------------|--------|
| >= 0.6 | Normal trading |
| 0.4 - 0.6 | Warning logged, trading continues |
| < 0.4 | Trading paused (if enabled) |

**Rationale:**
- Provides earlier warning when relationship weakens
- More conservative pause trigger given declining XRP/BTC correlation
- Aligns with academic research on pairs trading requirements

## Configuration Summary v4.0.0

### Changed Parameters

| Parameter | v3.0.0 | v4.0.0 | Change |
|-----------|--------|--------|--------|
| `correlation_pause_enabled` | `False` | `True` | Enabled by default |
| `correlation_warning_threshold` | `0.5` | `0.6` | +0.1 (earlier warning) |
| `correlation_pause_threshold` | `0.3` | `0.4` | +0.1 (more conservative) |

### Unchanged Parameters

All other configuration parameters remain unchanged from v3.0.0:

```python
CONFIG = {
    # Core Ratio Trading Parameters
    'lookback_periods': 20,
    'bollinger_std': 2.0,
    'entry_threshold': 1.5,           # REC-013 from v2.1
    'exit_threshold': 0.5,

    # Position Sizing (USD-based)
    'position_size_usd': 15.0,
    'max_position_usd': 50.0,
    'min_trade_size_usd': 5.0,

    # Risk Management (1:1 R:R)
    'stop_loss_pct': 0.6,
    'take_profit_pct': 0.6,

    # All other v3.0 parameters unchanged...
}
```

## Startup Output Changes

The `on_start()` function now displays v4.0 feature information:

```
[ratio_trading] v4.0.0 started
[ratio_trading] Symbol: XRP/BTC (ratio pair)
[ratio_trading] Entry threshold: 1.5 std (REC-013)
[ratio_trading] Core Features: VolatilityRegimes=True, CircuitBreaker=True, SpreadFilter=True
[ratio_trading] v2.1 Features: RSI=True, TrendFilter=True, TrailingStop=True, PositionDecay=True
[ratio_trading] v3.0 Features: CorrelationMonitoring=True, DynamicBTCPrice=True, SeparateExitTracking=True
[ratio_trading] v4.0 Features: CorrelationPauseEnabled=True, RaisedThresholds=True (research-validated)
[ratio_trading] Correlation (REC-023/024): warn<0.6, pause<0.4 (pause_enabled=True)
[ratio_trading] Position sizing: 15.0 USD, Max: 50.0 USD
[ratio_trading] R:R ratio: 0.6/0.6 (1.00:1)
```

## Risk Considerations

### Correlation Breakdown Risk

The primary motivation for v4.0 is the declining XRP/BTC correlation:

| Metric | Value | Source |
|--------|-------|--------|
| 90-day Correlation | ~0.54 | PortfoliosLab |
| Correlation Decline | -24.86% | MacroAxis |
| XRP vs BTC Volatility | XRP 1.55x higher | CME Group |

**Implications:**
- XRP is showing "independent streak" behavior in 2025
- Regulatory clarity (SEC vs Ripple) has changed XRP's market dynamics
- Traditional pairs trading assumptions may be weakening

### Mitigation Strategies

1. **Automatic Pause**: Trading pauses when correlation < 0.4
2. **Early Warning**: Warnings when correlation < 0.6
3. **Manual Override**: Can disable pause via config if confident
4. **Correlation History**: Stored for post-session analysis

## Migration from v3.0.0

### Breaking Changes

None. All changes are additive. The strategy is more conservative by default.

### Behavior Changes

- Trading will automatically pause if correlation drops below 0.4
- Warnings will be logged at correlation < 0.6 (was 0.5)
- Sessions may have more frequent correlation warnings

### To Match v3.0.0 Behavior

```yaml
# In config.yaml
strategy_overrides:
  ratio_trading:
    correlation_pause_enabled: false
    correlation_warning_threshold: 0.5
    correlation_pause_threshold: 0.3
```

### Recommended Production Settings (v4.0.0)

Use defaults, which are now research-validated and conservative:

```yaml
# In config.yaml - no overrides needed
# Default v4.0.0 settings provide optimal protection
```

## Files

- Strategy: `strategies/ratio_trading.py`
- Symbol: `XRP/BTC`
- Version: 4.0.0
- Review: `docs/development/review/ratio_trading/ratio-trading-strategy-review-v4.0.md`

## Strategy Development Guide Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| STRATEGY_NAME | PASS | `"ratio_trading"` |
| STRATEGY_VERSION | PASS | `"4.0.0"` |
| SYMBOLS list | PASS | `["XRP/BTC"]` |
| CONFIG dict | PASS | 40+ parameters |
| generate_signal() | PASS | Correct signature |
| Size in USD | PASS | `position_size_usd` |
| Stop loss correct | PASS | Below/above entry as appropriate |
| R:R ratio >= 1:1 | PASS | 0.6%/0.6% = 1:1 |
| Signal metadata | PASS | Full metadata with exit_reason |
| on_start() | PASS | Config validation, feature logging |
| on_fill() | PASS | Position tracking, dynamic BTC price |
| on_stop() | PASS | Comprehensive summary with correlation stats |
| Per-pair PnL | PASS | REC-006 |
| Config validation | PASS | REC-007 |
| Volatility regimes | PASS | REC-004 |
| Circuit breaker | PASS | REC-005 |
| RSI confirmation | PASS | REC-014 |
| Trend filter | PASS | REC-015 |
| Enhanced metrics | PASS | REC-016 |
| Trailing stops | PASS | Implemented |
| Position decay | PASS | Implemented |
| Dynamic BTC price | PASS | REC-018 |
| Separate exit tracking | PASS | REC-020 |
| Correlation monitoring | PASS | REC-021 |
| Correlation pause enabled | PASS | REC-023 (v4.0) |
| Raised thresholds | PASS | REC-024 (v4.0) |

**Overall Compliance: ~98%** (per v4.0 review)

## Future Enhancements (from v4.0 Review)

The following recommendations are documented for future consideration:

| Recommendation | Priority | Effort | Description |
|----------------|----------|--------|-------------|
| REC-025 | LOW | HIGH | Formal cointegration testing (ADF, Hurst) |
| REC-026 | LOW | MEDIUM | Hedge ratio optimization (OLS regression) |
| REC-027 | LOW | MEDIUM | Position scaling based on z-score magnitude |

These are not implemented in v4.0 but provide a roadmap for future improvements.

## Research References

- [Crypto Pairs Trading: Why Cointegration Beats Correlation](https://blog.amberdata.io/crypto-pairs-trading-why-cointegration-beats-correlation)
- [Pairs Trading in Cryptocurrency Markets (IEEE)](https://ieeexplore.ieee.org/document/9200323/)
- [Correlation Between XRP and Bitcoin (MacroAxis)](https://www.macroaxis.com/invest/pair-correlation/XRP.CC/BTC.CC/XRP-vs-Bitcoin)
- [How XRP Relates to the Crypto Universe (CME Group)](https://www.cmegroup.com/insights/economic-research/2025/how-xrp-relates-to-the-crypto-universe-and-the-broader-economy.html)

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 1.0.0 | 2025-12-14 | Initial implementation |
| 2.0.0 | 2025-12-14 | Major refactor (REC-002 to REC-010) |
| 2.1.0 | 2025-12-14 | Enhancement refactor (REC-013 to REC-017) |
| 3.0.0 | 2025-12-14 | Review recommendations (REC-018 to REC-022) |
| 4.0.0 | 2025-12-14 | Deep review optimizations (REC-023, REC-024) |
