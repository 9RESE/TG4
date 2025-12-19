# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
