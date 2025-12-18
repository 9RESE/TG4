"""
Market Regime Detection Module

Provides multi-layer market regime detection for adaptive trading strategies.

Components:
    - RegimeDetector: Main orchestrator class
    - CompositeScorer: Weighted indicator scoring
    - MTFAnalyzer: Multi-timeframe confluence
    - ExternalDataFetcher: Fear & Greed, BTC Dominance
    - ParameterRouter: Strategy parameter adjustments

Types:
    - MarketRegime: STRONG_BULL, BULL, SIDEWAYS, BEAR, STRONG_BEAR
    - VolatilityState: LOW, MEDIUM, HIGH, EXTREME
    - TrendStrength: ABSENT, WEAK, EMERGING, STRONG, VERY_STRONG
    - RegimeSnapshot: Complete regime state at a point in time
    - RegimeAdjustments: Strategy parameter multipliers

Usage:
    from ws_tester.regime import RegimeDetector, RegimeSnapshot

    # Initialize detector
    detector = RegimeDetector(symbols=['XRP/USDT', 'BTC/USDT'])

    # On each tick
    regime = await detector.detect(data_snapshot)

    # Check regime conditions
    if regime.is_favorable_for_mean_reversion():
        adjustments = detector.get_strategy_adjustments('mean_reversion', regime)
        # Apply adjustments to strategy config

    # Or check if strategy should trade
    if detector.should_trade('mean_reversion', regime):
        # Generate signals
        pass

Example with strategy:
    def generate_signal(data, config, state):
        regime = data.regime
        if regime is None:
            return None

        # Check if favorable
        if not regime.is_favorable_for_mean_reversion():
            return None

        # Adjust size based on confidence
        size = config['position_size_usd'] * regime.overall_confidence

        # Continue with signal generation...
"""

# Types
from .types import (
    MarketRegime,
    VolatilityState,
    TrendStrength,
    IndicatorScores,
    SymbolRegime,
    MTFConfluence,
    ExternalSentiment,
    RegimeSnapshot,
    RegimeAdjustments,
    DEFAULT_REGIME_ADJUSTMENTS,
)

# Components
from .detector import RegimeDetector
from .composite_scorer import CompositeScorer
from .mtf_analyzer import MTFAnalyzer
from .external_data import ExternalDataFetcher
from .parameter_router import ParameterRouter

__all__ = [
    # Types
    'MarketRegime',
    'VolatilityState',
    'TrendStrength',
    'IndicatorScores',
    'SymbolRegime',
    'MTFConfluence',
    'ExternalSentiment',
    'RegimeSnapshot',
    'RegimeAdjustments',
    'DEFAULT_REGIME_ADJUSTMENTS',
    # Components
    'RegimeDetector',
    'CompositeScorer',
    'MTFAnalyzer',
    'ExternalDataFetcher',
    'ParameterRouter',
]
