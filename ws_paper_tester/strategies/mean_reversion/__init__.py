"""
Mean Reversion Strategy v4.2.0

Trades price deviations from moving average and VWAP.
Enhanced with volatility regimes, circuit breaker, multi-symbol support,
trend filtering, trailing stops, position decay, and comprehensive risk management.

SCOPE AND LIMITATIONS (REC-005 v4.1.0):
- Asset Types: Crypto-to-stablecoin (XRP/USDT, BTC/USDT) and ratio (XRP/BTC)
- Best For: Range-bound markets, moderate volatility (0.3-1.0%)
- NOT Suitable For: Strong directional trends, extreme volatility, low correlation periods
- Market Conditions to Pause:
  * Fear & Greed Index < 25 (Extreme Fear)
  * ADX > 30 (strong trend)
  * XRP/BTC correlation < 0.4

KEY ASSUMPTIONS:
- Price deviations from mean are temporary and will revert
- Pairs maintain reasonable correlation (XRP/BTC > 0.5)
- Market structure supports reversion within decay period (15-30 min)
- Transaction costs are ~0.2% round-trip

THEORETICAL BASIS:
- Ornstein-Uhlenbeck process: prices fluctuate but revert to equilibrium
- Bollinger Bands + RSI combination for overbought/oversold detection
- VWAP deviation for institutional price anchoring
- Research note: Academic studies (SSRN Oct 2024) show mean reversion less
  effective in BTC since 2022; XRP may exhibit better mean-reverting behavior

Version History:
- 1.0.0: Initial implementation
- 1.0.1: Fixed RSI edge case (LOW-007)
- 2.0.0: Major refactor per mean-reversion-strategy-review-v1.0.md
         - REC-001: Fixed R:R ratio to 1:1 (0.5%/0.5%)
         - REC-002: Added multi-symbol support (XRP/USDT, BTC/USDT)
         - REC-003: Added cooldown mechanisms
         - REC-004: Added volatility regime classification
         - REC-005: Added circuit breaker protection
         - REC-006: Added per-pair PnL tracking
         - REC-007: Added configuration validation
         - REC-008: Added trade flow confirmation
         - Finding #6: Added on_stop() callback
         - Code cleanup and optimization
- 3.0.0: Major enhancement per mean-reversion-strategy-review-v3.1.md
         - REC-001: Added XRP/BTC ratio trading pair
         - REC-002: Fixed hardcoded max_losses in on_fill
         - REC-003: Added wider stop-loss option research support
         - REC-004: Added optional trend filter
         - REC-006: Added trailing stops
         - REC-007: Added position decay
         - Finding #4: Refactored _evaluate_symbol for lower complexity
- 4.0.0: Optimization per mean-reversion-deep-review-v4.0.md
         - REC-001: Trailing stops disabled by default (research: fixed TP better for MR)
         - REC-002: Extended position decay timing (15 min start, 5 min intervals)
         - REC-003: Added trend confirmation period (reduce false positives in choppy markets)
         - REC-005: Added XRP/BTC correlation monitoring (rolling Pearson coefficient)
         - Gentler decay multipliers: [1.0, 0.85, 0.7, 0.5]
         - Wider trailing distance: 0.3% (if enabled)
         - Higher trailing activation: 0.4% (if enabled)
- 4.1.0: Risk adjustments per mean-reversion-deep-review-v5.0.md
         - REC-001: Reduced BTC/USDT position size ($50 -> $25) due to unfavorable
           market conditions (bearish trend, Extreme Fear sentiment, academic research)
         - REC-002: Added fee profitability check (Guide v2.0 Section 23 compliance)
           Round-trip fees validated before signal generation
         - REC-005: Added SCOPE AND LIMITATIONS documentation
         - New rejection reason: FEE_UNPROFITABLE
         - Compliance score: 89% -> 92% (fee checks + scope docs added)
- 4.2.0: Correlation risk management per mean-reversion-deep-review-v6.0.md
         - REC-001: Tightened correlation_warn_threshold (0.5 -> 0.4) for XRP/BTC
           XRP-BTC correlation dropped from ~80% to ~40%, making traditional ratio
           trading assumptions less reliable
         - REC-001: Added correlation_pause_threshold (0.25) to automatically pause
           XRP/BTC trading when correlation is critically low
         - REC-001: Added correlation_pause_enabled config flag for user control
         - New rejection reason: LOW_CORRELATION (Guide v2.0 Section 24 compliant)
         - Compliance score: 96% maintained (dynamic correlation pause added)
         - Deferred: REC-003 ADX filter, REC-004 band walk detection (LOW priority)
- 4.2.1: Refactored into modular package structure for maintainability
"""

# =============================================================================
# Public Interface - Re-exports for backward compatibility
# =============================================================================

# Strategy metadata
from .config import (
    STRATEGY_NAME,
    STRATEGY_VERSION,
    SYMBOLS,
    CONFIG,
    SYMBOL_CONFIGS,
    VolatilityRegime,
    RejectionReason,
)

# Main signal generation function
from .signals import generate_signal

# Lifecycle callbacks
from .lifecycle import on_start, on_fill, on_stop

# Define public API
__all__ = [
    # Metadata
    'STRATEGY_NAME',
    'STRATEGY_VERSION',
    'SYMBOLS',
    'CONFIG',
    'SYMBOL_CONFIGS',
    # Enums
    'VolatilityRegime',
    'RejectionReason',
    # Main function
    'generate_signal',
    # Callbacks
    'on_start',
    'on_fill',
    'on_stop',
]
