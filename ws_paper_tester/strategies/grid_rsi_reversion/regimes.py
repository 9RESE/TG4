"""
Grid RSI Reversion Strategy - Regime Classification

Contains volatility and session regime classification.
"""
from datetime import datetime
from typing import Dict, Any

from .config import VolatilityRegime, TradingSession


def classify_volatility_regime(
    volatility: float,
    config: Dict[str, Any]
) -> VolatilityRegime:
    """
    Classify current volatility regime.

    Args:
        volatility: Current volatility percentage
        config: Strategy configuration

    Returns:
        VolatilityRegime enum value
    """
    low_threshold = config.get('regime_low_threshold', 0.3)
    medium_threshold = config.get('regime_medium_threshold', 0.8)
    high_threshold = config.get('regime_high_threshold', 1.5)

    if volatility < low_threshold:
        return VolatilityRegime.LOW
    elif volatility < medium_threshold:
        return VolatilityRegime.MEDIUM
    elif volatility < high_threshold:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.EXTREME


def get_regime_adjustments(
    regime: VolatilityRegime,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get trading adjustments for volatility regime.

    Args:
        regime: Current volatility regime
        config: Strategy configuration

    Returns:
        Dict with adjustment parameters
    """
    adjustments = {
        'grid_spacing_mult': 1.0,
        'size_mult': 1.0,
        'pause_trading': False,
    }

    if regime == VolatilityRegime.LOW:
        # Low volatility: tighter grids, standard size
        adjustments['grid_spacing_mult'] = 0.8
        adjustments['size_mult'] = 1.0

    elif regime == VolatilityRegime.MEDIUM:
        # Medium volatility: standard settings
        adjustments['grid_spacing_mult'] = 1.0
        adjustments['size_mult'] = 1.0

    elif regime == VolatilityRegime.HIGH:
        # High volatility: wider grids, reduced size
        adjustments['grid_spacing_mult'] = 1.3
        adjustments['size_mult'] = 0.8

    elif regime == VolatilityRegime.EXTREME:
        # Extreme volatility: may pause trading
        adjustments['grid_spacing_mult'] = 1.5
        adjustments['size_mult'] = 0.5
        if config.get('regime_extreme_pause', True):
            adjustments['pause_trading'] = True

    return adjustments


def classify_trading_session(
    current_time: datetime,
    config: Dict[str, Any]
) -> TradingSession:
    """
    Classify current trading session based on UTC hour.

    Args:
        current_time: Current timestamp
        config: Strategy configuration

    Returns:
        TradingSession enum value
    """
    hour = current_time.hour
    boundaries = config.get('session_boundaries', {})

    asia_start = boundaries.get('asia_start', 0)
    asia_end = boundaries.get('asia_end', 8)
    europe_start = boundaries.get('europe_start', 8)
    europe_end = boundaries.get('europe_end', 14)
    overlap_start = boundaries.get('overlap_start', 14)
    overlap_end = boundaries.get('overlap_end', 17)
    us_start = boundaries.get('us_start', 17)
    us_end = boundaries.get('us_end', 21)

    if asia_start <= hour < asia_end:
        return TradingSession.ASIA
    elif europe_start <= hour < europe_end:
        return TradingSession.EUROPE
    elif overlap_start <= hour < overlap_end:
        return TradingSession.US_EUROPE_OVERLAP
    elif us_start <= hour < us_end:
        return TradingSession.US
    else:
        return TradingSession.OFF_HOURS


def get_session_adjustments(
    session: TradingSession,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get trading adjustments for session.

    Args:
        session: Current trading session
        config: Strategy configuration

    Returns:
        Dict with adjustment parameters
    """
    size_multipliers = config.get('session_size_multipliers', {
        'ASIA': 0.8,
        'EUROPE': 1.0,
        'US': 1.0,
        'US_EUROPE_OVERLAP': 1.1,
        'OFF_HOURS': 0.5,
    })

    spacing_multipliers = config.get('session_spacing_multipliers', {
        'ASIA': 1.2,
        'EUROPE': 1.0,
        'US': 1.0,
        'US_EUROPE_OVERLAP': 0.9,
        'OFF_HOURS': 1.4,
    })

    session_name = session.name

    return {
        'size_mult': size_multipliers.get(session_name, 1.0),
        'spacing_mult': spacing_multipliers.get(session_name, 1.0),
    }
