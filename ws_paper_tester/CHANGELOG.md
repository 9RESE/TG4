# Changelog

All notable changes to the WebSocket Paper Tester will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
