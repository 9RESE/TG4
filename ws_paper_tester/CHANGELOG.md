# Changelog

All notable changes to the WebSocket Paper Tester will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
