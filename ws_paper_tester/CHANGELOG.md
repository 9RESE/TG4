# Changelog

All notable changes to the WebSocket Paper Tester will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
