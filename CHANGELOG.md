# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
