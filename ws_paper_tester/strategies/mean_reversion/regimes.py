"""
Mean Reversion Strategy - Volatility Regime Classification

Contains functions for classifying market volatility into regimes
and determining appropriate trading adjustments.
"""
from typing import Dict, Any

from .config import VolatilityRegime


def classify_volatility_regime(
    volatility_pct: float,
    config: Dict[str, Any]
) -> VolatilityRegime:
    """Classify current volatility into a regime."""
    low = config.get('regime_low_threshold', 0.3)
    medium = config.get('regime_medium_threshold', 0.8)
    high = config.get('regime_high_threshold', 1.5)

    if volatility_pct < low:
        return VolatilityRegime.LOW
    elif volatility_pct < medium:
        return VolatilityRegime.MEDIUM
    elif volatility_pct < high:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.EXTREME


def get_regime_adjustments(
    regime: VolatilityRegime,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Get threshold and size adjustments based on volatility regime."""
    adjustments = {
        'threshold_mult': 1.0,
        'size_mult': 1.0,
        'pause_trading': False,
    }

    if regime == VolatilityRegime.LOW:
        # Tighter thresholds in low volatility
        adjustments['threshold_mult'] = 0.8
        adjustments['size_mult'] = 1.0
    elif regime == VolatilityRegime.MEDIUM:
        adjustments['threshold_mult'] = 1.0
        adjustments['size_mult'] = 1.0
    elif regime == VolatilityRegime.HIGH:
        # Wider thresholds, smaller sizes in high volatility
        adjustments['threshold_mult'] = 1.3
        adjustments['size_mult'] = 0.8
    elif regime == VolatilityRegime.EXTREME:
        adjustments['threshold_mult'] = 1.5
        adjustments['size_mult'] = 0.5
        adjustments['pause_trading'] = config.get('regime_extreme_pause', True)

    return adjustments
