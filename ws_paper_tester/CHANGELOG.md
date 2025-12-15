# Changelog

All notable changes to the WebSocket Paper Tester will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.12.2] - 2025-12-15

### Added
- **Whale Sentiment strategy v1.4.0** - Deep Review v4.0 Implementation
  - REC-030: CRITICAL - Fixed undefined `_classify_volatility_regime` function reference in signal metadata
  - REC-031: Added EXTREME volatility regime (ATR > 6%) with trading pause
    - New `volatility_extreme_threshold` config parameter (default: 6.0%)
    - New `EXTREME_VOLATILITY` rejection reason for tracking
    - `should_pause` flag in volatility adjustments
  - REC-032: Removed deprecated RSI code per clean code principles
    - Removed `calculate_rsi` function from indicators.py
    - Removed RSI config parameters from CONFIG
    - `detect_rsi_divergence` retained as stub returning 'none'
  - REC-033: Added scope and limitations documentation section
    - Strategy scope and intended use documented
    - Known limitations with mitigations
    - Conditions where strategy should NOT trade

### Changed
- **Whale Sentiment strategy** (`strategies/whale_sentiment/`)
  - Version updated to 1.4.0
  - Guide v2.0 compliance increased from 89% to 100%
  - `VolatilityRegime` enum now includes EXTREME
  - `classify_volatility_regime()` handles EXTREME threshold
  - `get_volatility_adjustments()` returns `should_pause` flag
  - `generate_signal()` checks for extreme volatility pause
  - Updated `__init__.py` exports (removed `calculate_rsi`, added `calculate_atr`)

### Documentation
- `docs/development/features/whale_sentiment/whale-sentiment-v1.4.md` created
- `docs/development/review/whale_sentiment/deep-review-v4.0.md` added
- Version history updated in config.py and __init__.py

## [1.12.1] - 2025-12-15

### Fixed
- **Test Infrastructure** - Fixed 17 pre-existing test failures
  - Updated `create_rich_snapshot()` to include XRP/BTC data for all tests
  - Added backward compatibility aliases for renamed functions in market_making and mean_reversion
  - Fixed version assertion in test_strategy_wrapper (validates semver format)
  - Fixed test_version_is_4_0_0 to test_version_is_4_3_0 (matches current version)

- **Market Making exports** - Added missing function exports for testing
  - `validate_config`, `calculate_micro_price`, `calculate_reservation_price`
  - `calculate_optimal_spread`, `check_fee_profitability`, `check_position_decay`
  - `get_xrp_usdt_price`, `build_entry_signal`, `calculate_trailing_stop`
  - Backward compatibility aliases with underscore prefix

- **Mean Reversion exports** - Added missing function exports for testing
  - `calculate_trend_slope`, `is_trending`, `calculate_correlation`
  - `get_decayed_take_profit`, `calculate_trailing_stop`
  - Backward compatibility aliases with underscore prefix

### Changed
- All 153 tests now pass (previously 136 passed, 17 failed)

## [1.12.0] - 2025-12-15

### Added
- **Whale Sentiment strategy v1.1.0** - Deep Review v1.0 Implementation
  - REC-001: Recalibrated confidence weights (volume 40%, RSI 15%)
    - Academic research shows RSI ineffective in crypto markets
    - Volume spike detection now primary signal
  - REC-003: Clarified trade flow logic for contrarian mode
    - Intentionally lenient to allow contrarian entries
    - Accepts mild opposing flow (±10% threshold)
  - REC-005: Enhanced indicator logging on circuit breaker/cooldown paths
    - Includes elapsed/remaining cooldown times
  - REC-007: XRP/BTC disabled by default (7-10x lower liquidity)
  - REC-008: Short size multiplier reduced to 0.5x (crypto squeeze risk)
  - REC-009: Updated research references to deep-review-v1.0.md
  - REC-010: Documented UTC timezone requirement for session boundaries
  - Deferred: REC-002 (candle persistence), REC-004 (volatility regime), REC-006 (backtest weights)

### Documentation
- `docs/development/features/whale_sentiment/whale-sentiment-v1.0.md` updated to v1.1.0
- `docs/development/review/whale_sentiment/deep-review-v1.0.md` added to version control
- Version history in config.py and __init__.py updated

## [1.11.1] - 2025-12-15

### Added
- **Whale Sentiment strategy v1.0.0** - Initial Implementation
  - Volume spike detection as whale activity proxy (2x average = spike)
  - RSI-based sentiment zones (Extreme Fear/Fear/Neutral/Greed/Extreme Greed)
  - Price deviation from recent high/low as supplementary signal
  - Contrarian mode: buy fear, sell greed (default)
  - Trade flow confirmation for signal validation
  - Cross-pair correlation management with blocking at 85%
  - Session-aware position sizing (Asia 0.8x, Off-Hours 0.5x)
  - Stricter circuit breaker (2 consecutive losses, 45 min cooldown)
  - Composite confidence scoring with weighted components
  - Support for XRP/USDT, BTC/USDT, XRP/BTC (XRP/BTC disabled by default in v1.1.0)

### Documentation
- `docs/development/features/whale_sentiment/whale-sentiment-v1.0.md` created
- `docs/development/review/whale_sentiment/master-plan-v1.0.md` updated to IMPLEMENTED

## [1.11.0] - 2025-12-14

### Added
- **WaveTrend Oscillator strategy v1.1.0** - Deep Review v1.0 Implementation
  - REC-001: Optimized channel/average lengths (10/21 from 9/12)
  - REC-002: Zone confirmation requirement (2 candles in zone before signal)
  - REC-003: Bullish/bearish divergence detection with confidence boost
  - REC-005: Extreme zone profit taking (exit at ±80)
  - REC-006: Enhanced indicator logging on all code paths
  - REC-007: Per-symbol WT parameters in SYMBOL_CONFIGS
  - Deferred: REC-004 (volatility regime), REC-008 (volume confirmation)

- **WaveTrend Oscillator strategy v1.0.0** - Initial Implementation
  - WaveTrend (LazyBear) dual-line crossover signals
  - Overbought/oversold zone filtering (±60 threshold)
  - Extreme zone detection (±80 threshold)
  - Session-aware position sizing
  - Cross-pair correlation management
  - Volatility regime classification (LOW/MEDIUM/HIGH/EXTREME)
  - Circuit breaker protection (3 consecutive losses)
  - Signal rejection tracking
  - Support for XRP/USDT, BTC/USDT, XRP/BTC

### Documentation
- `docs/development/features/wavetrend/wavetrend-v1.0.md` created
- `docs/development/features/wavetrend/wavetrend-v1.1.md` created
- `docs/development/review/wavetrend/master-plan-v1.0.md` updated to IMPLEMENTED
- `docs/development/review/wavetrend/deep-review-v1.0.md` created

## [1.10.2] - 2025-12-14

### Changed
- **Grid RSI Reversion strategy v1.2.0** - Deep Review v2.1 Implementation
  - REC-009: BTC/USDT `grid_spacing_pct` increased from 1.0% to 1.5%
    - Improves R:R ratio from 0.10:1 to 0.15:1 (50% improvement)
    - Trade-off: Fewer grid fills per range traversal vs better risk-adjusted returns
  - REC-010: Aligned `adx_recenter_threshold` from 25 to 30
    - Now matches main trend filter threshold for consistent behavior
    - Grid recentering blocked when market is trending (ADX > 30)
  - REC-011: Documented VPIN as future enhancement for regime detection
    - Would provide order flow-based regime classification complementing price volatility

### Documentation
- Deep review v2.1 updated with implementation status (97% compliance)
- Version history updated in `__init__.py` with detailed changelog

## [1.10.1] - 2025-12-14

### Changed
- **Grid RSI Reversion strategy v1.1.0** - Deep Review v2.0 Implementation
  - REC-001: Signal rejection tracking verified on all paths via `track_rejection()` function
    - 14 distinct rejection reasons tracked (WARMING_UP, REGIME_PAUSE, TREND_FILTER, etc.)
  - REC-002: Complete indicator logging on all code paths via `build_base_indicators()`
  - REC-003: Trade flow confirmation with volume analysis
    - `calculate_trade_flow()`, `calculate_volume_ratio()`, `check_trade_flow_confirmation()`
    - New rejection reasons: `FLOW_AGAINST_TRADE`, `LOW_VOLUME`
  - REC-004: Widened stop-loss parameters based on research
    - Default: 8% (was 3%), XRP/USDT: 5%, BTC/USDT: 10%, XRP/BTC: 8%
  - REC-005: Real correlation monitoring between symbols
    - `calculate_rolling_correlation()` for Pearson correlation on returns
    - `correlation_block_threshold: 0.85` to prevent correlated exposure
  - REC-006: Liquidity validation for XRP/BTC
    - `check_liquidity_threshold()` with `min_volume_usd: $100M` requirement
    - New rejection reason: `LOW_LIQUIDITY`
  - REC-007: Explicit R:R ratio calculation and validation
    - `calculate_grid_rr_ratio()` with signal metadata
    - R:R warnings in `validate_config()`
  - REC-008: Trend check before grid recentering
    - `check_trend_before_recenter` config option with ADX threshold

### Documentation
- Deep review v2.0 updated to IMPLEMENTED status
- Compliance score increased to ~95%
- All 8 recommendations from previous review implemented

## [1.10.0] - 2025-12-14

### Added
- **Grid RSI Reversion strategy v1.0.0** - New hybrid strategy combining grid trading with RSI mean reversion
  - Grid level setup with geometric (default) and arithmetic spacing options
  - RSI as confidence modifier for entries (not hard filter)
  - Adaptive RSI zones based on ATR volatility
  - Multi-level position accumulation with cycle tracking
  - Grid cycle completion (buy-sell pair) for profit taking
  - Trend filter using ADX > 30 to pause in trending markets
  - Per-symbol configuration for XRP/USDT, BTC/USDT, XRP/BTC
  - Volatility regime classification (LOW/MEDIUM/HIGH/EXTREME)
  - Circuit breaker with consecutive loss tracking
  - Comprehensive state management for grid levels and metadata

- **Enhanced logging infrastructure** (`ws_tester.py`)
  - `log_aggregated()` - Complete audit trail with data hash, signal, execution, portfolio
  - `log_status()` - Periodic status logging (every 30s)
  - `log_portfolio_snapshot()` - Per-strategy portfolio state with symbol stats

- **Feature Documentation**
  - `docs/development/features/grid_rsi_reversion/grid-rsi-reversion-v1.0.md`
  - `docs/development/review/grid_rsi_reversion/master-plan-v1.0.md` (updated to IMPLEMENTED status)

### Changed
- **Dashboard WebSocket** (`ws_tester/dashboard/server.py`)
  - Thread-safe queue using `queue.Queue` instead of `asyncio.Queue` for cross-thread safety
  - Custom JSON serializer handles Infinity, NaN, and datetime values
  - Non-blocking event polling with 50ms sleep
  - Graceful handling when no clients connected

- **WebSocket Client** (`ws_tester/data_layer.py`)
  - Compatible with websockets v11+ (uses `close_code` attribute instead of `.closed`)

### Fixed
- **Market Making v2.2.1**: Circuit breaker consecutive loss tracking
  - Fixed `on_fill()` signature to match strategy_loader interface (2 args, not 3)
  - Config now stored in `state['_config']` during `on_start()` for `on_fill()` access
  - Circuit breaker now correctly triggers after consecutive losses

### Documentation
- Market Making v2.2 docs updated with v2.2.1 patch notes
- Grid RSI Reversion feature documentation created following Arc42/Diataxis standards
- Master plan document marked as IMPLEMENTED

## [1.9.0] - 2025-12-14

### Added
- **Mean Reversion strategy v4.1.0** - Risk adjustments per mean-reversion-deep-review-v5.0.md
  - REC-001: Reduced BTC/USDT position size ($50 → $25) due to unfavorable market conditions
    - BTC in bearish territory (below all EMAs), Fear & Greed at "Extreme Fear" (23)
    - Academic research (SSRN Oct 2024) indicates mean reversion less effective in BTC
    - Proportionally reduced max_position ($150 → $75)
  - REC-002: Added fee profitability check (Guide v2.0 Section 23 compliance)
    - `_check_fee_profitability()` validates net profit after round-trip fees
    - New config parameters: `check_fee_profitability`, `estimated_fee_rate`, `min_net_profit_pct`
    - New rejection reason: `FEE_UNPROFITABLE`
  - REC-005: Added SCOPE AND LIMITATIONS documentation
    - Explicit documentation of suitable/unsuitable market conditions
    - Key assumptions and theoretical basis documented
    - Market conditions to pause trading enumerated

### Changed
- **Mean Reversion strategy** (`mean_reversion.py`)
  - Version updated to 4.1.0
  - BTC/USDT default position size reduced from $50 to $25
  - `on_start()` now logs FeeCheck feature status and fee parameters
  - `_generate_entry_signal()` validates fee profitability before generating signals
  - New indicator: `net_profit_pct` for fee-aware monitoring
  - Docstring includes comprehensive SCOPE AND LIMITATIONS section

### Documentation
- Strategy Development Guide compliance for Mean Reversion: 89% → 92%
- Added fee profitability validation (Section 23) and scope documentation (Section 26)
- Research references: SSRN Oct 2024 paper on mean reversion effectiveness in Bitcoin

## [1.8.0] - 2025-12-14

### Added
- **Ratio Trading strategy v4.0.0** - Deep review optimizations per ratio-trading-strategy-review-v4.0.md
  - REC-023: Enable `correlation_pause_enabled` by default (HIGH priority)
    - Trading automatically pauses when XRP/BTC correlation drops below threshold
    - Protects against correlation breakdown periods
    - Research shows XRP/BTC correlation declining (~24.86% over 90 days)
  - REC-024: Raised correlation thresholds for earlier warning/pause
    - `correlation_warning_threshold`: 0.5 → 0.6 (earlier warning)
    - `correlation_pause_threshold`: 0.3 → 0.4 (more conservative pause)
    - Research suggests pairs trading requires correlation > 0.6 for reliability
  - New v4.0 feature logging in `on_start()` showing raised thresholds and research validation

- **Feature Documentation**
  - `docs/development/features/ratio_trading/ratio-trading-v4.0.md` - v4.0.0 feature docs

### Changed
- **Ratio Trading strategy** (`ratio_trading.py`)
  - Version updated to 4.0.0
  - Default configuration now more conservative for correlation protection
  - `on_start()` logs v4.0 features with REC-023/024 reference
  - Strategy marked "Production Ready - Monitor Correlation Closely" in review

### Documentation
- Strategy Development Guide compliance for Ratio Trading remains at ~98%
- All HIGH and MEDIUM priority recommendations from v4.0 review implemented
- Research references updated with XRP/BTC correlation analysis

## [1.7.0] - 2025-12-14

### Added
- **Ratio Trading strategy v3.0.0** - Enhancement refactor per ratio-trading-strategy-review-v3.1.md
  - REC-018: Dynamic BTC price for USD conversion
    - `_get_btc_price_usd()` function fetches real-time BTC/USD price from market data
    - Configurable fallback price and symbol list (`btc_price_fallback`, `btc_price_symbols`)
    - Replaces hardcoded $100,000 BTC price for accurate conversions
  - REC-019: Confirmed on_start print statement correctly uses `config.get()` with defaults
  - REC-020: Separate exit tracking from rejection tracking
    - New `ExitReason` enum: TRAILING_STOP, POSITION_DECAY, TAKE_PROFIT, STOP_LOSS, MEAN_REVERSION, CORRELATION_EXIT
    - `_track_exit()` function tracks intentional exits with P&L by reason
    - Exit statistics in on_stop summary separate from rejections
  - REC-021: Rolling correlation monitoring
    - `_calculate_rolling_correlation()` for XRP/BTC correlation calculation
    - Configurable warning threshold (0.5) and pause threshold (0.3)
    - Optional trading pause when correlation drops below threshold
    - Correlation warnings counted and displayed in summary
  - REC-022: Hedge ratio config placeholders for future enhancement
  - New indicators: `btc_price_usd`, `correlation`, `use_correlation_monitoring`, `correlation_warnings`
  - Enhanced on_stop summary with exit statistics and correlation monitoring

- **Feature Documentation**
  - `docs/development/features/ratio_trading/ratio-trading-v3.0.md` - v3.0.0 feature docs

### Changed
- **Ratio Trading strategy** (`ratio_trading.py`)
  - Version updated to 3.0.0
  - `on_start()` now logs v3.0 features including correlation monitoring config
  - `on_fill()` uses dynamic BTC price from state for accurate conversions
  - `generate_signal()` includes correlation monitoring with optional trading pause
  - Trailing stop and position decay exits now tracked separately from rejections
  - Enhanced indicators with BTC price and correlation metrics

### Documentation
- Strategy Development Guide compliance for Ratio Trading: 100% (40+ config parameters)
- All v3.1 review recommendations implemented and documented

## [1.6.0] - 2025-12-14

### Added
- **Mean Reversion strategy v3.0.0** - Major enhancement per mean-reversion-deep-review-v3.1.md
  - REC-001: XRP/BTC ratio trading pair with volatility-optimized parameters
    - Wider deviation threshold (1.0% vs 0.5%)
    - Wider TP/SL (0.8%/0.8%) accounting for ratio volatility
    - Conservative position sizing ($15 USD, max $40)
    - Longer cooldown (20s) for less liquid pair
  - REC-002: Fixed hardcoded `max_losses=3` in `on_fill()` - now uses config value
  - REC-003: Research support for wider stop-loss on volatile pairs
  - REC-004: Optional trend filter using linear regression slope detection
    - `_calculate_trend_slope()` and `_is_trending()` functions
    - New `TRENDING_MARKET` rejection reason
    - Prevents new entries in trending markets while managing existing positions
  - REC-006: Trailing stops for profit protection
    - Activates at configurable profit threshold (default 0.3%)
    - Trails at configurable distance (default 0.2%)
    - Tracks highest/lowest price since entry
  - REC-007: Position decay for time-based TP reduction
    - Reduces TP target for aging positions
    - Configurable decay schedule (default: 100% → 75% → 50% → 25%)
    - Encourages earlier exits when mean reversion thesis weakens
  - Finding #4: Refactored `_evaluate_symbol()` into modular helper functions
    - `_check_trailing_stop_exit()`, `_check_position_decay_exit()`, `_generate_entry_signal()`
    - Reduced cyclomatic complexity
  - New indicators: `trend_slope`, `is_trending`, `decay_multiplier`, `decayed_tp`
  - 11 new configuration parameters (total: 39)

- **Feature Documentation**
  - `docs/development/features/mean_reversion/mean-reversion-v3.0.md` - v3.0.0 feature docs
  - `docs/development/review/mean_reversion/mean-reversion-strategy-review-v3.1.md` - Implementation record

### Changed
- **Mean Reversion strategy** (`mean_reversion.py`)
  - Version updated to 3.0.0
  - SYMBOLS extended to include XRP/BTC
  - `on_start()` logs new feature flags and stores config values for `on_fill()`
  - `_evaluate_symbol()` now checks trailing stops, position decay, and trend filter before entry signals
  - Enhanced indicators with trend and decay metrics

### Fixed
- Mean Reversion `on_fill()` hardcoded max_losses value (was always 3, now uses config)

### Documentation
- Strategy Development Guide compliance for Mean Reversion improved from 94% to 100%
- All review recommendations implemented and documented

## [1.5.0] - 2025-12-14

### Added
- **Ratio Trading strategy v2.0.0** - Major refactor per ratio-trading-strategy-review-v1.0.md
  - REC-002: USD-based position sizing (`position_size_usd`, `max_position_usd`)
  - REC-003: Fixed R:R ratio to 1:1 (0.6%/0.6%, was 0.5%/0.6%)
  - REC-004: Volatility regime classification (LOW/MEDIUM/HIGH/EXTREME)
  - REC-005: Circuit breaker protection (3 consecutive losses = 15min pause)
  - REC-006: Per-pair PnL tracking (`pnl_by_symbol`, `trades_by_symbol`, etc.)
  - REC-007: Configuration validation on startup
  - REC-008: Spread monitoring with `max_spread_pct` filter
  - REC-010: Optional trade flow confirmation
  - Signal metadata with strategy, signal_type, z-score, regime
  - Rejection tracking for all signal rejection reasons
  - Comprehensive `on_stop()` summary with accumulation stats

- **Deep Strategy Reviews**
  - `docs/development/review/ratio_trading/ratio-trading-strategy-review-v1.0.md`
    - Academic research on pairs/ratio trading and cointegration
    - XRP/BTC pair analysis with correlation/volatility data
    - Strategy Development Guide compliance analysis (was ~55%)
    - 12 recommendations prioritized by severity
  - `docs/development/review/mean_reversion/mean-reversion-deep-review-v2.0.md`
    - Extended analysis for mean reversion strategy patterns

- **Feature Documentation**
  - `docs/development/features/ratio_trading/ratio-trading-v1.0.md` - Initial version docs
  - `docs/development/features/ratio_trading/ratio-trading-v2.0.md` - v2.0.0 feature docs

### Changed
- **Ratio Trading strategy** (`ratio_trading.py`)
  - Version updated to 2.0.0
  - Refactored monolithic `generate_signal()` into modular helper functions
  - Take profit now uses price-based percentage instead of SMA target
  - Dual tracking: USD position for compliance + XRP position for ratio logic
  - Enhanced indicators with 25+ metrics including regime and spread info
  - `on_start()` validates config and logs feature flags
  - `on_fill()` tracks per-pair metrics and circuit breaker state
  - `on_stop()` logs comprehensive summary with rejections

### Documentation
- Reorganized ratio trading docs from `features/ratio/` to `features/ratio_trading/`
- All strategy reviews now follow consistent format with:
  - Executive summary with compliance scores
  - Academic research references
  - Pair-specific analysis
  - Prioritized recommendations with effort/impact matrix

## [1.4.0] - 2025-12-13

### Added
- **Market Making strategy v1.4.0** - Enhancements per deep strategy review v1.3
  - Configuration validation on startup with warnings for invalid/risky settings
  - Optional Avellaneda-Stoikov reservation price model (`use_reservation_price`, `gamma`)
  - Trailing stop support (`use_trailing_stop`, `trailing_stop_activation`, `trailing_stop_distance`)
  - Per-pair PnL and trade metrics tracking in strategy state
  - Position entry tracking for trailing stop calculations

- **Portfolio per-pair tracking** (`ws_tester/portfolio.py`)
  - New fields: `pnl_by_symbol`, `trades_by_symbol`, `wins_by_symbol`, `losses_by_symbol`
  - New methods: `record_trade_result()`, `get_symbol_stats()`, `get_all_symbol_stats()`
  - Enhanced `to_dict()` with asset values, per-pair breakdown, and symbol stats

- **Enhanced logging** (`ws_tester/logger.py`)
  - `log_fill()` now accepts `symbol_stats` for per-pair cumulative P&L display
  - New `log_portfolio_snapshot()` method for detailed portfolio state logging
  - Console output shows per-pair cumulative P&L on fills

- **New tests** (`tests/test_strategies.py`)
  - `TestMarketMakingV14Features`: Config validation, reservation price, trailing stop
  - `TestPortfolioPerPairTracking`: Per-pair PnL and stats tracking
  - 17 total strategy tests (6 new for v1.4.0)

- **Documentation**
  - `docs/development/market-making-strategy-review-v1.3.md` - Deep strategy review
  - `docs/development/features/market-making-v1.4.md` - Feature documentation

### Changed
- **Market Making strategy** (`market_making.py`)
  - Version updated to 1.4.0
  - `on_start()` now validates config and initializes trailing stop tracking
  - `on_fill()` now tracks position entries and per-pair metrics
  - `on_stop()` includes per-pair metrics in final summary
  - `_evaluate_symbol()` checks trailing stops and calculates reservation price
  - Enhanced indicators include `reservation_price`, `trailing_stop_price`, `pnl_symbol`, `trades_symbol`

- **Executor** (`ws_tester/executor.py`)
  - Now calls `portfolio.record_trade_result()` for per-pair metrics

### Documentation
- Created comprehensive v1.3 strategy review with:
  - Strategy Development Guide compliance analysis (95% score)
  - Deep research on market making techniques (Avellaneda-Stoikov, OBI, trade flow)
  - Pair-specific analysis for XRP/USDT, BTC/USDT, XRP/BTC
  - Industry comparison with Hummingbot A-S strategy
  - 12 academic and industry references

## [1.3.0] - 2025-12-13

### Added
- **Market Making strategy v1.3.0** - Major improvements based on deep strategy review
  - Volatility-adjusted spreads using `_calculate_volatility()` function
  - Signal cooldown mechanism with per-symbol `cooldown_seconds` config
  - Trade flow confirmation via `use_trade_flow` and `trade_flow_threshold` parameters
  - Enhanced indicator logging with volatility metrics
  - XRP/BTC size unit conversion (XRP → USD for Signal compatibility)

### Changed
- **Market Making strategy** (`market_making.py`)
  - Improved risk-reward ratios: BTC/USDT now 1:1 (was 0.5:1)
  - XRP/USDT TP increased from 0.3% to 0.4%
  - XRP/BTC TP increased from 0.25% to 0.3%
  - Stop/take profit now based on entry price (was mid price)
  - `on_fill()` now uses `value` field directly for USD pairs

### Fixed
- **MM-001**: XRP/BTC size units mismatch - now converts XRP to USD for Signal
- **MM-002**: Added volatility-adjusted spreads to prevent over-trading
- **MM-003**: Added signal cooldown to prevent rapid-fire trades
- **MM-004**: Fixed BTC/USDT risk-reward ratio (0.35%/0.35% = 1:1)
- **MM-005**: Fixed on_fill unit confusion for USD pairs
- **MM-006**: Stop/TP now based on entry price, not mid price
- **MM-007**: Added trade flow confirmation for signal validation
- **MM-008**: Added volatility metrics to indicator logging

### Documentation
- Created `docs/development/market-making-strategy-review-v1.2.md` with deep analysis
- All 8 identified issues documented with severity, impact, and fixes
- Research references from hftbacktest, DWF Labs, academic papers

## [1.2.0] - 2025-12-13

### Added
- **New Ratio Trading strategy** (`ratio_trading.py` v1.0.0)
  - Mean reversion on XRP/BTC using Bollinger Bands
  - Dedicated strategy for growing both XRP and BTC holdings
  - Tracks `xrp_accumulated` and `btc_accumulated` metrics
  - Research-based config: 60s cooldown, 20-period lookback
- **XRP/BTC Market Making** in `market_making.py` v1.2.0
  - Spread capture for dual-asset accumulation
  - XRP-denominated inventory tracking
  - No shorting on cross-pair (buy/sell only)
- **BTC/USDT Market Making** in `market_making.py` v1.2.0
  - High liquidity pair with tighter spreads (0.03% min)
  - Larger position sizes ($50 vs $20 for XRP)
  - Lower imbalance threshold (0.08) for more liquid market
- Feature documentation: `docs/development/features/ratio-trading-v1.0.md`

### Changed
- Order Flow strategy updated to v2.2.0
  - Removed XRP/BTC (now handled by dedicated strategies)
  - Focused on USDT pairs (XRP/USDT, BTC/USDT)
  - Cleaner separation of concerns
- Market Making strategy updated to v1.2.0
  - Added BTC/USDT and XRP/BTC to symbols list
  - Symbol-specific configuration support
  - Tracks XRP and BTC accumulation
- Test version assertion relaxed to `startswith('1.')` for flexibility

### Architecture
- XRP/BTC trading now split across two strategies:
  - `market_making.py` - Spread capture, inventory management
  - `ratio_trading.py` - Mean reversion, Bollinger Bands
- Market Making covers all three pairs: XRP/USDT, BTC/USDT, XRP/BTC
- Clear separation: Order Flow for momentum, Market Making for spread capture

## [1.1.0] - 2025-12-13

### Added
- **XRP/BTC ratio trading support** in Order Flow strategy v2.1.0 (moved to dedicated strategies in v1.2.0)
- **Starting assets support** in portfolio system
  - Portfolios can start with assets (e.g., 500 XRP) in addition to USDT
  - `starting_assets` config section in config.yaml
  - Assets tracked separately and included in equity calculation
- **Per-symbol configuration** in Order Flow strategy
  - `cooldown_trades`, `cooldown_seconds` per symbol
  - `volume_spike_mult` per symbol
  - `position_size_xrp` for XRP/BTC trades
- Feature documentation: `docs/development/features/order-flow-v2.1.md`

### Changed
- Order Flow strategy updated to v2.1.0
- Config.yaml symbols updated to include XRP/BTC
- Test fixtures updated to use USDT pairs consistently (XRP/USDT, BTC/USDT)
- SimulatedDataManager defaults now include USDT pairs
- Startup banner shows starting assets when configured

### Fixed
- Symbol mismatch between strategies (USDT) and tests (USD)
- Test `test_get_all_symbols` now correctly checks for XRP/USDT

## [1.0.2] - 2025-12-13

### Fixed
- **HIGH-002**: Config key mismatch where `execution` section in config.yaml was not loaded (code looked for `executor` key)
- **MEDIUM-002/003**: Strategies now properly use `short` action for new short positions instead of `sell`
  - `market_making.py`: Distinguishes between selling to close long vs opening short position
  - `mean_reversion.py`: Same fix applied with proper stop-loss/take-profit directions
- P&L calculation now includes entry fees for accurate profit tracking
- Position tracking now carries `entry_fee` through partial close operations

### Changed
- Dashboard default binding changed from `0.0.0.0:8080` to `127.0.0.1:8787` for security
- Dashboard now reads host/port from config.yaml `dashboard` section
- WebSocket client timeout for slow clients reduced from 5s to 2s
- `mean_reversion.py` `on_fill()` now properly handles all four action types: `buy`, `sell`, `short`, `cover`
- Position dataclass now includes `entry_fee` field for complete P&L tracking
- `Position.unrealized_pnl()` now accepts optional `exit_fee` parameter

### Added
- New test files for comprehensive coverage:
  - `tests/test_dashboard.py`: Dashboard API and WebSocket tests
  - `tests/test_integration.py`: End-to-end trading loop tests
  - `tests/test_websocket_client.py`: WebSocket client and message handling tests
- `ws_tester/credentials.py`: Secure credential loading from environment/.env files
- Test coverage expanded from 81 to 126 tests

### Security
- Dashboard binds to localhost by default (not all interfaces)
- Added credentials module with secure handling patterns

## [1.0.1] - 2025-12-13

### Fixed
- Critical initialization bug where component setup code was misplaced inside `_apply_strategy_overrides()` causing scope errors
- Memory leak from unbounded fills list - now uses bounded deque via `portfolio.add_fill()` (max 1000 fills)
- Async/sync mismatch in main loop - now uses `get_snapshot_async()` for proper async context
- Bare exception clauses replaced with specific exception types in logger, dashboard, and data layer

### Changed
- Executor parameters (fee_rate, slippage_rate, max_short_leverage) now configurable via constructor and config file
- Strategies refactored to iterate over their SYMBOLS list instead of hardcoding symbol names
- Removed sys.path manipulation from all strategy files
- Strategy callbacks (on_fill, on_start, on_stop) now log errors explicitly
- Test version assertions relaxed to `startswith('1.0')` for flexibility

### Added
- `_initialize_components()` method in WSTester for proper component initialization order
- `_evaluate_symbol()` helper functions in market_making and mean_reversion strategies
- Explicit error logging for strategy callback exceptions
- Configuration documentation in `docs/user/how-to/configure-paper-tester.md`

## [1.0.0] - 2025-12-13

### Added
- Initial release of WebSocket Paper Tester
- Real-time paper trading with Kraken WebSocket v2 API
- Simulated data mode for offline testing
- Strategy auto-discovery with security features (whitelist, SHA256 verification)
- Per-strategy isolated portfolios with thread-safe operations
- Paper execution engine with configurable fees and slippage
- Web dashboard with real-time updates via WebSocket
- Structured JSON Lines logging with rotation and compression
- Three example strategies: market_making, order_flow, mean_reversion
- API key authentication for dashboard
- Comprehensive test suite (81 tests passing)

### Security
- Strategy file hash verification
- Dashboard API key authentication
- Thread-safe portfolio operations with RLock
- Specific exception handling to prevent silent failures
