# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2025-12-15

### Added - Whale Sentiment Strategy v1.2.0 (Deep Review v2.0 Implementation)

#### REC-011: Candle Data Persistence
- New `persistence.py` module for saving/loading candle data
- Eliminates 25+ hour warmup requirement after restarts
- Configurable via `use_candle_persistence`, `candle_persistence_dir`, `max_candle_age_hours`
- Saves to `data/candles/{symbol}_5m.json`

#### REC-012: Warmup Progress Indicator
- Added `warmup_pct`, `warmup_eta_minutes`, `warmup_eta_hours` to indicators
- Shows completion percentage and estimated time remaining during warmup

#### REC-016: XRP/BTC Re-enablement Guard
- Added `enable_xrpbtc` config flag (default: false)
- Blocks XRP/BTC trading unless explicitly enabled
- Logs warning about 7-10x lower liquidity when enabled

#### REC-017: UTC Timezone Validation
- Added `require_utc_timezone` and `timezone_warning_only` config options
- Validates server timezone on startup
- Warns or blocks if not running in UTC

#### REC-018: Trade Flow Expected Indicator
- Added `trade_flow_expected` and `trade_flow_mode` indicators
- Clarifies that contrarian mode expects opposite flow direction
- Shows 'positive'/'negative' expected flow and 'contrarian'/'momentum' mode

#### REC-019: Per-Symbol Volume Window
- `volume_window` now configurable per-symbol in SYMBOL_CONFIGS
- Allows different baseline periods for different market characteristics

#### REC-020: Extracted Magic Numbers
- Added `volume_confidence_base` and `volume_confidence_bonus_per_ratio` to CONFIG
- Makes confidence calculation parameters configurable

### Changed

#### REC-013: RSI Removed from Confidence Calculation
- `weight_rsi_sentiment` changed from 0.15 to 0.00
- `weight_divergence` changed from 0.10 to 0.00
- `weight_volume_spike` increased from 0.40 to 0.55 (PRIMARY signal)
- `weight_price_deviation` increased from 0.20 to 0.35
- `weight_trade_flow` reduced from 0.15 to 0.10
- `min_confidence` reduced from 0.55 to 0.50 (fewer components)
- Based on academic evidence: PMC/NIH (2023), QuantifiedStrategies (2024)

### Documented

#### Deferred Recommendations
- REC-014: Volatility regime classification (HIGH effort)
- REC-015: Backtest confidence weights (HIGH effort)

### Documentation

- Created `deep-review-v2.0.md` with comprehensive strategy review
- Created `whale-sentiment-v1.2.md` feature documentation
- Updated version history in all module docstrings

## [1.1.0] - 2025-12-14

### Added - Whale Sentiment Strategy v1.1.0 (Deep Review v1.0 Implementation)

- REC-001: Recalibrated confidence weights (volume 40%, RSI 15%)
- REC-003: Clarified trade flow logic for contrarian mode
- REC-005: Enhanced indicator logging on all code paths
- REC-007: Disabled XRP/BTC by default (liquidity concerns)
- REC-008: Reduced short size multiplier to 0.5x (squeeze risk)
- REC-009: Updated research documentation references
- REC-010: Documented UTC timezone requirement for sessions

## [1.0.0] - 2025-12-13

### Added - Whale Sentiment Strategy Initial Implementation

- Volume spike detection as whale activity proxy
- RSI sentiment classification
- Fear/greed price deviation proxy
- Contrarian mode signal generation
- Trade flow confirmation
- Cross-pair correlation management
- Session-aware position sizing
- Circuit breaker protection
