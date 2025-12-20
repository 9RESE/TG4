# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-12-19

### Added
- **Phase 6: Paper Trading Integration** - Complete paper trading infrastructure as default execution mode
- **TradingMode Enum**: System-wide trading mode with dual-confirmation for live trading (CRITICAL-01)
- **PaperPortfolio**: Full balance tracking, P&L calculation, trade history, and session persistence
- **PaperOrderExecutor**: Configurable fill simulation with slippage, delays, and fees
- **PaperPriceSource**: Multi-source price provider (live feed, historical DB, mock fallback)
- **Paper Trading API**: 6 new endpoints for portfolio, positions, trades, and reset operations
- **Database Migration 005**: Isolated paper trading tables (sessions, orders, positions, trades, snapshots)
- **OrderStatus.REJECTED**: New enum value for business logic rejections vs system errors (CRITICAL-01)
- **Session Persistence**: Paper portfolio saves/restores across restarts (HIGH-01)
- **53 New Unit Tests**: Full coverage of paper trading components and review fixes

### Fixed
- **Thread-Safe Statistics**: Added `asyncio.Lock` for concurrent counter updates (CRITICAL-02)
- **Async Price Queries**: Created `get_price_async()` for proper async database access (CRITICAL-03)
- **Size Quantization**: Trade sizes now respect symbol's `size_decimals` config (HIGH-02)
- **Entry Price Handling**: API uses `None` instead of `0` for market orders (MEDIUM-01)
- **Price Cache Timestamps**: Stale price updates rejected based on timestamp comparison (MEDIUM-02)
- **Order History Persistence**: Orders persisted before memory trimming (MEDIUM-03)

### Changed
- Coordinator now auto-initializes paper trading in PAPER mode
- Rate limiting added for `/paper/trade` and `/paper/reset` endpoints
- Audit logging enhanced with `RISK_RESET` event for portfolio resets
- Test count increased from 1045 to 1098

### Security
- Triple confirmation required for live trading (ENV + CONFIG + explicit confirmation)
- Paper trading endpoints protected by authentication and rate limiting
- Complete data isolation between paper and live trading

### Documentation
- [Phase 6 Feature Documentation](docs/development/features/phase-6-paper-trading.md)
- [ADR-013: Paper Trading Design](docs/architecture/09-decisions/ADR-013-paper-trading-design.md)
- [Phase 6 Code Review](docs/development/reviews/phase-6/phase-6-code-review.md)

## [0.3.7] - 2025-12-19

### Added
- **Configuration Validators**: Added validators for all 9 config files (agents, risk, orchestration, portfolio, execution) with `validate_all_configs_on_startup()` function (F04, F14)
- **Shared Test Fixtures**: Created `triplegain/tests/conftest.py` with common fixtures for configs, LLM mocks, market data, and domain objects (F09)
- **Module Exports**: Added convenience re-exports in `llm/__init__.py` and `src/__init__.py` for simpler imports (F08, F13)
- **Environment Template**: Created `.env.example` documenting all required environment variables (F03)
- **Portfolio Rebalance Agent Config**: Added explicit configuration for portfolio_rebalance agent in `agents.yaml` (F15)

### Fixed
- **Migration Ordering**: Renumbered duplicate `003_*.sql` migrations to `003` and `004` for deterministic execution (F01)
- **Template Extensions**: Fixed `.md` → `.txt` extension references in `agents.yaml` to match actual template files (F02)
- **Symbol Consistency**: Uncommented `XRP/BTC` in `orchestration.yaml` to match other config files (F05)
- **risk_state Duplication**: Added `DROP TABLE IF EXISTS CASCADE` in migration 003 to handle schema evolution (F10)
- **Token Budget Alignment**: Aligned `snapshot.yaml` token budgets with `prompts.yaml` market_data values (F11)
- **XAI Comment Cleanup**: Removed confusing OPENAI reference from XAI provider comment (F12)

### Changed
- Coordinator agent now enabled by default (Phase 3 complete) (F06)
- `CLAUDE.md` updated to document 80% max exposure setting (F07)
- Prompts config validator now handles both wrapped and unwrapped formats

### Documentation
- ADR-012: Configuration & Integration Fixes documenting all 15 Phase 5 findings

## [0.3.6] - 2025-12-19

### Added
- **Authentication Enforcement**: All API endpoints now require authentication via `Depends(get_current_user)` (F01)
- **Role-Based Access Control**: Applied `require_role()` decorator - ADMIN for coordinator/risk, TRADER for trading operations (F02, F03)
- **Security Headers Middleware**: Added X-Content-Type-Options, X-Frame-Options, CSP, HSTS headers (F04)
- **UUID Validation**: All position_id and order_id parameters validated via `_validate_uuid()` (F05)
- **Confirmation Token Mechanism**: Destructive operations require one-time token from `/confirm` endpoint (F09)
  - `GET /positions/{id}/confirm` -> token valid for 5 minutes
  - `POST /positions/{id}/close` requires confirmation_token
  - Same pattern for order cancellation and portfolio rebalancing
- **Audit Logging**: `log_security_event()` with `SecurityEventType` enum for security event tracking (F12)
- **Task Name Validation**: `VALID_COORDINATOR_TASKS` whitelist for coordinator task endpoints (F20)
- **Symbol Config Loading**: `get_supported_symbols()` loads from config with caching (F23)
- **Pagination**: Added `offset` and `limit` parameters to `/positions` and `/orders` endpoints (F27)

### Fixed
- **Rate Limiting Hash**: Replaced MD5 with SHA256 for IP hashing in rate limiter (F13)
- **Exception Handling Order**: HTTPException now caught before generic Exception to prevent masking (F22)
- **Symbol Normalization**: All responses use consistent `BTC/USDT` format instead of `BTC_USDT` (F21)
- **Debug Endpoint Security**: Debug endpoints require authentication in production mode (F16)

### Changed
- `/risk/reset` now requires ADMIN role and is rate-limited to 1 request per minute (F10, F11)
- All validation functions centralized in `validation.py` module (F26)
- JWT secret required in production mode, warns in debug mode (F25)
- Request/response logging middleware added for debugging (F24)

### Security
- **OWASP A01**: Broken Access Control - Fixed by enforcing authentication/authorization on all endpoints
- **OWASP A03**: Injection - Fixed by UUID and symbol validation
- **OWASP A07**: Security Misconfiguration - Fixed by security headers and production mode requirements
- 26 of 27 Phase 4 security findings addressed (Finding 18 deferred as optional enhancement)

## [0.3.5] - 2025-12-19

### Added
- **Exchange Position Sync**: `sync_with_exchange()` method in PositionTracker detects discrepancies between local and Kraken positions (F05)
- **Partial Fill Detection**: `_handle_partial_fill()` creates positions for partial fills, handles cancelled/expired orders with fills (F03)
- **OCO Orders**: `handle_oco_fill()` automatically cancels take-profit when stop-loss fills and vice versa (F16)
- **Fee Tracking**: Order and Position dataclasses now track `fee_amount`, `fee_currency`, `total_fees` (F10)
- **Order Links**: Position dataclass includes `stop_loss_order_id`, `take_profit_order_id`, `external_id` for order tracking (F14)
- **Trigger Check Loop**: Separate `_trigger_check_loop()` runs every 5s (configurable) for faster SL/TP detection (F09)
- **Price Lookup for Market Orders**: `_get_current_price()` fetches from cache or Kraken API for size calculation (F02)
- **32 new execution tests**: Comprehensive coverage for new features, 1045 total tests

### Fixed
- **Stop-Loss Kraken Parameter**: Fixed critical bug where stop-loss used wrong `price2` parameter instead of `price` for trigger (F01)
- **Market Order Size**: Fixed calculation that returned USD amount instead of base currency amount (F02)
- **Contingent Order Failures**: Now publishes `RISK_ALERTS` when SL/TP placement fails, position marked as unprotected (F04)
- **Non-Atomic Fill Handling**: Added transaction-like error handling with proper alerting on failures (F06)
- **Thread Safety**: `enable_trailing_stop_for_position()` now async with proper lock usage (F07)
- **Case-Insensitive Errors**: Fixed inconsistent error checking (`Invalid` vs `invalid`) in order placement (F13)
- **Race in get_order()**: Fixed race condition with nested lock acquisition (F15)
- **Orphan Orders**: `cancel_orphan_orders()` cleans up SL/TP when position closes (F12)
- **Failed Order Persistence**: Failed orders now stored for audit before returning error (F11)

### Changed
- `modify_position()` now accepts optional `order_manager` to sync SL/TP changes to exchange (F08)
- `close_position()` now accepts optional `order_manager` to cancel orphan orders (F12)
- `trigger_check_interval_seconds` config option added (default: 5 seconds)
- Execution test coverage improved from 47% to 63%

## [0.3.4] - 2025-12-19

### Added
- **Minimum Quorum**: Trading decisions now require 4/6 models (configurable `min_quorum`) to prevent trading on insufficient data (P1-01)
- **Regime Hysteresis**: Regime changes require consecutive confirmations and higher confidence to prevent flapping (P1-02)
- **Hodl Bag Allocation**: Added `allocate_profits_to_hodl()` method with configurable 10% profit allocation (P2-02)
- **Regime State Persistence**: `load_regime_state()` restores previous regime on restart from database (P2-04)
- **Prompt Hash Tracking**: Technical Analysis agent now includes SHA-256 prompt hash in output for caching (P3-01)
- **Default Model Config**: Externalized `DEFAULT_MODEL_CONFIG` constant for easier model updates (P3-05)

### Fixed
- **SQL Injection**: Replaced string interpolation with parameterized query using `make_interval()` (P2-01)
- **Allocation Validation**: Target allocation misconfiguration now raises `ValueError` instead of warning (P2-03)
- **Price Type Safety**: Changed `entry_price`, `stop_loss`, `take_profit` from `float` to `Decimal` in ModelDecision, ConsensusResult, TradingDecisionOutput (P3-02)
- **Stop Loss Bounds**: Added clamping of `stop_loss_pct` to 1.0-5.0% range per design spec (P3-04)

### Changed
- `ConsensusResult.agreement_type` now includes 'insufficient_quorum' value
- Config key naming convention documented (P3-03)
- Action schema documented with rationale for CLOSE_LONG/CLOSE_SHORT vs single CLOSE (P2-05)

## [0.3.3] - 2025-12-19

### Added
- **Connection Pooling**: All LLM clients now use shared `aiohttp.ClientSession` with `TCPConnector` for connection reuse (2A-02)
- **Error Type Detection**: New `_is_retryable()` method classifies errors - auth/forbidden/bad request errors fail immediately (2A-01)
- **API Key Sanitization**: `sanitize_error_message()` redacts API keys and tokens from error logs (2A-03)
- **Rate Limit Headers**: `_parse_rate_limit_headers()` and `RateLimiter.update_from_provider()` respect provider limits (2A-07)
- **JSON Response Mode**: All clients now request structured JSON output from providers (2A-06)
- **SSL Certificate Validation**: `create_ssl_context()` uses certifi for explicit HTTPS verification (2A-09)
- **User-Agent Header**: All API requests include `TripleGain/{version}` User-Agent (2A-10)
- **Response Schema Validation**: `_validate_response_schema()` warns on missing expected fields (2A-13)
- **52 new LLM tests**: Comprehensive coverage for JSON utilities, error handling, cost calculation

### Fixed
- **RateLimiter wait_time**: Fixed incorrect `'wait_time' in dir()` check to proper variable initialization (2A-04)
- **Cost Calculation**: Added `_calculate_cost_actual()` using real input/output token counts (2A-05)
- **Empty Anthropic Content**: Now logs warning when Claude returns empty content array (2A-11)
- **Search Method Warning**: `generate_with_search()` now warns that search is not yet implemented (2A-12)

### Changed
- `LLMResponse` dataclass now includes `input_tokens`, `output_tokens`, `parsed_json`, `parse_error` fields
- `generate_with_retry()` accepts optional `parse_json` parameter for automatic JSON parsing (2A-08)
- Error messages use new sanitized format: `{provider}: {type} - {message}`
- Test count increased from 917 to 969 (52 new tests)

## [0.3.2] - 2025-12-19

### Added
- **Database Reconnection**: Added `reconnect()` with exponential backoff and `execute_with_retry()` for automatic retry on connection errors (P3.1)
- **Timestamp Normalization**: Added `normalize_timestamp()` utility to handle datetime, ISO strings, Unix timestamps uniformly (P3.3)
- **Thread-Safe Config**: Added thread lock and `reset_config_loader()` for test isolation (P3.4)

### Fixed
- **VWAP NaN Handling**: Updated calculation to treat NaN volumes as zero and carry forward previous values (P3.2)
- **Token Estimation**: Made `chars_per_token` and `safety_margin` configurable, added comprehensive documentation (P3.5)

### Changed
- Token estimation now uses instance-level configuration instead of class constants

## [0.3.1] - 2025-12-19

### Fixed
- **Stochastic RSI Calculation**: Fixed smoothing bug where raw K values were overwritten, causing incorrect %D calculation (P1.2)
- **Order Book Processing**: Added support for both raw and database formats in market snapshot builder (P1.3)
- **Supertrend Direction**: Changed warmup period values from 0 to NaN for clarity (P2.2)
- **Async Timeout**: Added configurable timeout (default 30s) to prevent hanging snapshot builds (P2.3)
- **Template Validation**: Increased strictness to require 2/3 of keywords instead of 1/2 (P2.4)
- **Config Float Parsing**: Added support for scientific notation (e.g., 1e-5) in configuration (P2.5)
- **Template Fallback**: Added minimal fallback prompt when template file is missing (P2.6)

### Added
- Transaction context manager `db.transaction()` for atomic multi-statement operations (P2.1)
- Phase 1 Foundation review findings document
- ADR-007: Phase 1 Foundation Review Fixes

### Changed
- Stochastic RSI values will differ from previous versions (now mathematically correct)

## [0.3.0] - 2025-12-19

### Added
- Comprehensive API security layer (`triplegain/src/api/security.py`)
  - API key authentication with role-based access control
  - Tiered rate limiting (5/30/60 requests per minute by endpoint)
  - CORS configuration via environment variables
  - 1MB request size limit
  - 45s request timeout
- Input validation for TradeProposal with descriptive error messages
- Thread-safe circuit breaker with internal state management
- Robust JSON parsing for LLM responses (handles markdown-wrapped JSON)
- Leverage factored into exposure calculations

### Fixed
- Trading decision consensus: Force HOLD when consensus <= 50% (was: pick alphabetically on tie)
- DCA rounding: Use ROUND_DOWN to prevent exceeding original amount
- MessageBus deadlock: Two-phase publish pattern prevents re-entrant locking
- TA fallback confidence: Reduced from 0.4 to 0.25 for heuristic analysis

### Security
- Fixed: Unauthorized API access (added authentication)
- Fixed: DDoS vulnerability (added rate limiting)
- Fixed: CSRF risk (added CORS configuration)
- Fixed: Memory exhaustion (added request size limits)

## [0.2.0] - 2025-12-18

### Added
- Phase 3 Orchestration layer complete
  - Message bus with pub/sub pattern
  - Coordinator agent for conflict resolution
  - Order manager with position tracking
  - Execution engine with paper trading
- DCA (Dollar Cost Averaging) support in portfolio rebalance
- Trailing stop functionality
- Position limit enforcement

### Changed
- Test coverage increased from 67% to 87%
- Total tests: 917 passing

## [0.1.0] - 2025-12-17

### Added
- Phase 1 Foundation: Data pipeline, indicators, snapshots, prompts
- Phase 2 Core Agents: TA, Regime, Risk, Trading Decision
- 5 LLM client integrations (OpenAI, Anthropic, DeepSeek, Grok, Ollama)
- TimescaleDB continuous aggregates for market data
- Kraken WebSocket data collector
