"""
Ratio Trading Strategy v4.3.0

Mean reversion strategy for XRP/BTC pair accumulation.
Trades the XRP/BTC ratio to grow holdings of both assets.

Strategy Logic:
- Calculate moving average of XRP/BTC ratio
- Use Bollinger Bands for entry/exit zones
- Buy when ratio is below lower band (XRP cheap vs BTC)
- Sell when ratio is above upper band (XRP expensive vs BTC)
- Rebalance to maintain balanced holdings

IMPORTANT: This strategy is designed ONLY for crypto-to-crypto ratio pairs
(XRP/BTC). It is NOT suitable for USDT-denominated pairs. For USDT pairs,
use the mean_reversion.py strategy instead.

WARNING - Trend Continuation Risk:
Bollinger Band touches can signal trend CONTINUATION rather than reversal.
Price exceeding the bands may indicate strong momentum, not necessarily a
mean reversion opportunity. The volatility regime system helps mitigate this
by pausing in EXTREME conditions and widening thresholds in HIGH volatility.

CRITICAL WARNING - XRP/BTC Correlation Status (December 2025):
XRP/BTC 3-month correlation has recovered to ~0.84 (up from crisis lows of ~0.40).
However, structural factors (XRP independence, ETF ecosystem, regulatory clarity)
suggest ongoing monitoring is essential. The strategy enables correlation_pause_enabled
by default (v4.0.0) and now includes correlation trend detection (v4.2.0) for
proactive protection.

- CONSERVATIVE: Pause XRP/BTC trading until correlation stabilizes above 0.6
- MODERATE: Use v4.0.0+ correlation protection (enabled by default)
- AGGRESSIVE: Lower correlation_pause_threshold to 0.3 (more trading, higher risk)

ALTERNATIVE PAIRS (REC-033):
If XRP/BTC correlation deteriorates, consider evaluating alternative pairs:
- ETH/BTC: Stronger historical cointegration (~0.80 correlation), higher liquidity
- LTC/BTC: Classical pairs candidate (~0.80 correlation)
- BCH/BTC: Bitcoin fork relationship (~0.75 correlation)
These would require strategy scope expansion (future enhancement).

FUTURE ENHANCEMENTS:
- REC-034/REC-040: Generalized Hurst Exponent (GHE) for mean-reversion validation
  H < 0.5 = mean-reverting (good), H >= 0.5 = trending (pause)
  2025 research shows GHE outperforms correlation/cointegration for crypto pair selection
  Implementation: Add _calculate_ghe() function, GHE_NOT_MEAN_REVERTING rejection reason
- REC-035: ADF Cointegration Test for formal cointegration validation
  Currently uses correlation as proxy; formal testing would be more robust
- REC-038: Half-Life Calculation for position management optimization
  Calculate spread half-life via Ornstein-Uhlenbeck process to adjust position decay
- REC-039: Multi-Pair Support Framework for trading alternative pairs (ETH/BTC, LTC/BTC)
  Would enable rapid pair switching without code changes if XRP/BTC correlation degrades
  Requires: Per-pair SYMBOL_CONFIGS, per-pair correlation tracking, pair selection logic

Version History:
- 1.0.0: Initial implementation
         - Mean reversion with Bollinger Bands
         - Dual-asset accumulation tracking
         - Research-based config from Kraken data
- 2.0.0: Major refactor per ratio-trading-strategy-review-v1.0.md
         - REC-002: Converted to USD-based position sizing
         - REC-003: Fixed R:R ratio to 1:1 (0.6%/0.6%)
         - REC-004: Added volatility regime classification
         - REC-005: Added circuit breaker protection
         - REC-006: Added per-pair PnL tracking
         - REC-007: Added configuration validation
         - REC-008: Added spread monitoring
         - REC-010: Added trade flow confirmation
         - Refactored generate_signal into smaller functions
         - Fixed take profit to use price-based percentage
         - Added rejection tracking
         - Added comprehensive on_stop() summary
- 2.1.0: Enhancement refactor per ratio-trading-strategy-review-v2.0.md
         - REC-013: Higher entry threshold (1.0 -> 1.5 std)
         - REC-014: Optional RSI confirmation filter
         - REC-015: Trend detection warning system
         - REC-016: Enhanced accumulation metrics
         - REC-017: Documentation updates (trend risk warning)
         - Added trailing stops (from mean reversion patterns)
         - Added position decay for stale positions
         - Fixed hardcoded max_losses in on_fill
- 3.0.0: Review recommendations per ratio-trading-strategy-review-v3.1.md
         - REC-018: Dynamic BTC price for USD conversion
         - REC-019: Fixed on_start print statement (already correct)
         - REC-020: Separate exit tracking from rejection tracking
         - REC-021: Rolling correlation monitoring with pause option
         - REC-022: Hedge ratio calculation (optional, future enhancement)
         - Added ExitReason enum for intentional exit tracking
         - Added correlation warning system
         - Improved accumulation metrics with real-time USD values
- 4.0.0: Deep review recommendations per ratio-trading-strategy-review-v4.0.md
         - REC-023: Enable correlation_pause_enabled by default (HIGH priority)
         - REC-024: Raised correlation thresholds for earlier warning/pause
           - correlation_warning_threshold: 0.5 -> 0.6
           - correlation_pause_threshold: 0.3 -> 0.4
         - Research-validated: XRP/BTC correlation declining (~24.86% over 90 days)
         - Strategy parameters confirmed aligned with academic research
- 4.1.0: Deep review v6.0 recommendations
         - REC-033: Document alternative pairs consideration (ETH/BTC, LTC/BTC)
           - Strategic recommendation for when XRP/BTC correlation remains low
           - Added prominent warning about correlation crisis in docstring
         - REC-034: Document GHE validation as future enhancement
         - REC-035: Document ADF cointegration test as future enhancement
         - REC-036: Add optional wider Bollinger Bands for crypto volatility
           - New config: bollinger_std_crypto (2.5 std), use_crypto_bollinger_std
           - Research suggests 2.5-3.0 std more appropriate for crypto volatility
           - Disabled by default; current mitigations (trend filter, RSI, volatility
             regimes) may make this unnecessary per review assessment
         - Compliance: Maintained 100% with Guide v2.0
         - Status: Production ready with correlation monitoring critical
- 4.2.0: Deep review v7.0 recommendations
         - REC-037: Correlation trend detection for proactive protection
           - New config: use_correlation_trend_detection, correlation_trend_lookback,
             correlation_trend_threshold, correlation_trend_level, correlation_trend_pause_enabled
           - Calculates slope of correlation over time via linear regression
           - Detects declining correlation trends before hitting absolute thresholds
           - Emits warnings when slope is negative and correlation < 0.7
           - Optional pause mode (disabled by default) for conservative operation
           - New RejectionReason: CORRELATION_DECLINING
           - New indicators: correlation_slope, correlation_trend, correlation_trend_warnings
         - REC-038: Document half-life calculation as future enhancement
         - Updated docstring with current XRP/BTC correlation recovery (~0.84 3-month)
         - Compliance: Maintained 100% with Guide v2.0
         - Status: Production ready with enhanced correlation monitoring
- 4.2.1: Deep review v8.0 validation
         - Review validates v4.2.0 as PRODUCTION READY with enhanced correlation monitoring
         - REC-039: Document multi-pair support framework as future enhancement
           - Would enable trading alternative pairs (ETH/BTC, LTC/BTC) if XRP/BTC correlation degrades
         - REC-040: Enhanced GHE documentation (builds on REC-034)
           - 2025 research validates GHE outperforms correlation for crypto pair selection
         - XRP/BTC correlation confirmed favorable (~0.84 3-month)
         - Regulatory clarity: SEC case resolved, 5+ U.S. XRP ETFs approved
         - All findings INFORMATIONAL or LOW severity, all addressed/documented
         - Compliance: Maintained 100% with Guide v2.0
         - Status: Production ready - no code changes required
         - Refactored: Split into modular subfolder structure for maintainability
- 4.3.0: Deep review v9.0 recommendations (December 2025)
         - REC-050: Added explicit fee profitability check
           - New config: use_fee_profitability_check, estimated_fee_rate (0.26% Kraken XRP/BTC),
             min_net_profit_pct (0.10%)
           - Ensures trades remain profitable after round-trip fees
           - New RejectionReason: FEE_NOT_PROFITABLE
         - Raised correlation_warning_threshold from 0.6 to 0.7 for earlier warning
           - Proactive detection given ongoing XRP structural changes (ETF ecosystem, regulatory clarity)
         - Increased position_decay_minutes from 5 to 10
           - Research suggests allowing more time for mean reversion in crypto pairs
           - Previous 5-min decay was too aggressive relative to typical half-life
         - REC-041/042: Ongoing correlation monitoring confirmed critical
         - REC-043: Weekly performance review recommended for first month of production
         - REC-044/045/046: Document ADF/Johansen/GHE as future enhancements (HIGH priority)
           - Formal cointegration testing would strengthen theoretical foundation
           - Currently uses correlation as proxy (acceptable but not optimal)
         - REC-047/048/049: Document half-life calc, multi-pair support, session awareness
         - Compliance: Maintained 100% with Guide v2.0
         - Status: Production ready with enhanced fee protection
"""

# Strategy metadata
from .config import STRATEGY_NAME, STRATEGY_VERSION, SYMBOLS, CONFIG

# Main signal generation
from .signals import generate_signal

# Lifecycle callbacks
from .lifecycle import on_start, on_fill, on_stop

# Enums for external use
from .enums import VolatilityRegime, RejectionReason, ExitReason

# Re-export for backwards compatibility and explicit interface
__all__ = [
    # Required strategy interface
    'STRATEGY_NAME',
    'STRATEGY_VERSION',
    'SYMBOLS',
    'CONFIG',
    'generate_signal',
    # Optional lifecycle callbacks
    'on_start',
    'on_fill',
    'on_stop',
    # Enums
    'VolatilityRegime',
    'RejectionReason',
    'ExitReason',
]
