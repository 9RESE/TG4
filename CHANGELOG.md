# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.15.1] - 2025-12-15

### Fixed
- Code review fixes for Historical Data System
- Improved error handling in gap filler

### Documentation
- Updated README with Historical Data System instructions
- Restored `.env.example` with improved documentation
- Created project documentation index (`docs/index.md`)
- Added comprehensive getting started guide (`docs/user/how-to/getting-started.md`)

## [1.15.0] - 2025-12-15

### Added - Historical Data System v1.0.0

Complete TimescaleDB-based historical data storage and retrieval system.

#### Data Module (`ws_paper_tester/data/`)
- `types.py`: Data types (HistoricalTrade, HistoricalCandle, DataGap, etc.)
- `websocket_db_writer.py`: Real-time WebSocket data persistence with buffering
- `historical_provider.py`: Query API for backtesting and strategy warmup
- `gap_filler.py`: Automatic gap detection and filling on startup
- `bulk_csv_importer.py`: Import Kraken historical CSV files
- `historical_backfill.py`: Fetch complete trade history from Kraken API

#### Database Schema
- `trades`: Individual trade ticks (daily partitioning, 7-day compression)
- `candles`: OHLCV candles (weekly partitioning, 30-day compression)
- `data_sync_status`: Sync state for gap detection and resumption
- `external_indicators`: External data (Fear & Greed, BTC Dominance)
- `backtest_runs`: Backtest results storage

#### Continuous Aggregates
Auto-computed timeframes: 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w from 1m base data.

#### Infrastructure
- Docker Compose configuration for TimescaleDB deployment
- 90%+ compression via TimescaleDB columnar compression
- Extended entry point (`main_with_historical.py`) with gap filler integration

#### HistoricalDataProvider API
- `get_candles()`: Query candles in time range with automatic view routing
- `get_latest_candles()`: Get N most recent candles for indicator warmup
- `replay_candles()`: AsyncIterator for backtesting with speed control
- `get_warmup_data()`: Convenience method for strategy indicator warmup
- `get_multi_timeframe_candles()`: Aligned MTF data for multi-timeframe analysis

#### Test Suite
- `tests/test_historical_data.py`: 17 unit tests for data types and provider logic

## [1.14.0] - 2025-12-15

### Added - Market Regime Detection System v1.0.0

Comprehensive Bull/Bear/Sideways market regime detector with multi-timeframe analysis.

#### Regime Module (`ws_tester/regime/`)
- `RegimeDetector`: Main orchestrator with hysteresis
- `CompositeScorer`: Weighted indicator scoring (ADX 25%, Chop 20%, MA 20%, RSI 15%, Volume 10%, Sentiment 10%)
- `MTFAnalyzer`: Multi-timeframe confluence analysis (1m, 5m, 15m, 1h)
- `ExternalDataFetcher`: Fear & Greed Index, BTC Dominance (cached, async)
- `ParameterRouter`: Strategy-specific parameter adjustments

#### Classifications
- 5 market regimes: STRONG_BULL, BULL, SIDEWAYS, BEAR, STRONG_BEAR
- 4 volatility states: LOW, MEDIUM, HIGH, EXTREME
- 5 trend strength levels: ABSENT, WEAK, EMERGING, STRONG, VERY_STRONG

#### New Indicators
- `calculate_choppiness()`: Choppiness Index (0-100)
- `calculate_adx_with_di()`: ADX with +DI/-DI
- `calculate_ma_alignment()`: MA alignment scoring

#### Test Suite
- `tests/test_regime.py`: 39 tests covering all components

## [1.13.1] - 2025-12-15

### Fixed - Ratio Trading v4.3.1
- Fixed TypeError in correlation monitoring when `None` values encountered
- Added proper `None` checks before comparisons
- Only append valid (non-None) correlation values to history

### Changed - Strategy Indicator Shims
Updated 8 strategy indicator modules to use centralized library:
- `grid_rsi_reversion/indicators.py`
- `market_making/calculations.py`
- `mean_reversion/indicators.py`
- `momentum_scalping/indicators.py`
- `order_flow/indicators.py`
- `ratio_trading/indicators.py`
- `wavetrend/indicators.py`
- `whale_sentiment/indicators.py`

## [1.13.0] - 2025-12-15

### Added - Centralized Indicator Library v1.0.0

Major refactoring consolidating ~45 duplicated indicator functions.

#### Indicator Modules (`ws_tester/indicators/`)
- `moving_averages.py`: SMA, EMA (single value and series)
- `oscillators.py`: RSI (Wilder's smoothing), ADX, MACD with crossover detection
- `volatility.py`: ATR, Bollinger Bands, volatility percentage, z-score
- `correlation.py`: Rolling Pearson correlation on returns
- `volume.py`: Volume ratio, volume spike, micro-price, VPIN
- `flow.py`: Trade flow analysis, flow confirmation
- `trend.py`: Trend slope (linear regression), trend strength, trailing stops

#### Structured Return Types
- `BollingerResult`, `ATRResult`, `TradeFlowResult`, `TrendResult`, `CorrelationTrendResult`

#### Test Suite
- `tests/test_indicators.py`: 46 tests covering all indicator functions
- `tests/fixtures/indicator_test_data.py`: Golden fixtures with pre-calculated values

## [1.12.0] - 2025-12-15

### Added - Whale Sentiment Strategy v1.4.0

Deep Review v4.0 Implementation:
- EXTREME volatility regime (ATR > 6%) with trading pause
- Fixed undefined `_classify_volatility_regime` function reference
- Removed deprecated RSI code
- Added scope and limitations documentation

## [1.11.0] - 2025-12-14

### Added - WaveTrend Oscillator Strategy v1.1.0

Deep Review v1.0 Implementation:
- Optimized channel/average lengths (10/21 from 9/12)
- Zone confirmation requirement (2 candles in zone before signal)
- Bullish/bearish divergence detection with confidence boost
- Extreme zone profit taking (exit at ±80)
- Per-symbol WT parameters in SYMBOL_CONFIGS

## [1.10.0] - 2025-12-14

### Added - Grid RSI Reversion Strategy v1.2.0

Hybrid strategy combining grid trading with RSI mean reversion:
- Grid level setup with geometric and arithmetic spacing
- RSI as confidence modifier for entries
- Adaptive RSI zones based on ATR volatility
- Multi-level position accumulation with cycle tracking
- Trend filter using ADX > 30 to pause in trending markets
- Signal rejection tracking
- Trade flow confirmation with volume analysis
- Real correlation monitoring between symbols

## [1.9.0] - 2025-12-14

### Changed - Mean Reversion Strategy v4.1.0
- Reduced BTC/USDT position size ($50 → $25) due to unfavorable conditions
- Added fee profitability check (validates net profit after round-trip fees)
- Added scope and limitations documentation

## [1.8.0] - 2025-12-14

### Changed - Ratio Trading Strategy v4.0.0
- Enabled `correlation_pause_enabled` by default
- Raised correlation thresholds (warning: 0.5→0.6, pause: 0.3→0.4)
- Research-validated settings for pairs trading

## [1.7.0] - 2025-12-14

### Added - Ratio Trading Strategy v3.0.0
- Dynamic BTC price for USD conversion
- Separate exit tracking from rejection tracking
- Rolling correlation monitoring with configurable thresholds
- Optional trading pause when correlation drops

## [1.6.0] - 2025-12-14

### Added - Mean Reversion Strategy v3.0.0
- XRP/BTC ratio trading pair support
- Fixed hardcoded max_losses in on_fill()
- Optional trend filter using linear regression slope
- Trailing stops for profit protection
- Position decay for time-based TP reduction

## [1.5.0] - 2025-12-14

### Added - Ratio Trading Strategy v2.0.0
- USD-based position sizing
- Fixed R:R ratio to 1:1
- Volatility regime classification
- Circuit breaker protection
- Per-pair PnL tracking
- Configuration validation on startup
- Spread monitoring

## [1.4.0] - 2025-12-13

### Added - Market Making Strategy v1.4.0
- Configuration validation on startup
- Avellaneda-Stoikov reservation price model
- Trailing stop support
- Per-pair PnL and trade metrics tracking
- Portfolio per-pair tracking in `ws_tester/portfolio.py`

## [1.3.0] - 2025-12-13

### Added - Market Making Strategy v1.3.0
- Volatility-adjusted spreads
- Signal cooldown mechanism
- Trade flow confirmation
- Enhanced indicator logging

## [1.2.0] - 2025-12-13

### Added
- New Ratio Trading strategy (v1.0.0) for XRP/BTC pair trading
- XRP/BTC and BTC/USDT Market Making support
- Feature documentation for all new strategies

## [1.1.0] - 2025-12-13

### Added
- XRP/BTC ratio trading support
- Starting assets support in portfolio system
- Per-symbol configuration in strategies

## [1.0.2] - 2025-12-13

### Fixed
- Config key mismatch for executor settings
- Strategy short position handling
- P&L calculation including entry fees
- Dashboard security (localhost binding)

### Added
- Comprehensive test suite (126 tests)
- Credential management module

## [1.0.1] - 2025-12-13

### Fixed
- Critical initialization bug in WSTester
- Memory leak from unbounded fills list
- Async/sync mismatch in main loop
- Bare exception clauses

## [1.0.0] - 2025-12-13

### Added - WebSocket Paper Tester Initial Release
- Real-time paper trading with Kraken WebSocket v2 API
- Simulated data mode for offline testing
- Strategy auto-discovery with security features
- Per-strategy isolated portfolios
- Paper execution engine with configurable fees
- Web dashboard with real-time updates
- Structured JSON Lines logging
- Three example strategies: market_making, order_flow, mean_reversion
- 81 tests passing
