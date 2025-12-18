"""
Order Flow Strategy - Regime and Session Classification

Contains volatility regime classification and trading session awareness logic.
"""
from datetime import datetime
from typing import Dict, Any

from .config import VolatilityRegime, TradingSession


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
) -> Dict[str, float]:
    """Get threshold and size adjustments based on volatility regime."""
    adjustments = {
        'threshold_mult': 1.0,
        'size_mult': 1.0,
        'pause_trading': False,
    }

    if regime == VolatilityRegime.LOW:
        adjustments['threshold_mult'] = 0.9
        adjustments['size_mult'] = 1.0
    elif regime == VolatilityRegime.MEDIUM:
        adjustments['threshold_mult'] = 1.0
        adjustments['size_mult'] = 1.0
    elif regime == VolatilityRegime.HIGH:
        adjustments['threshold_mult'] = 1.3
        adjustments['size_mult'] = 0.8
    elif regime == VolatilityRegime.EXTREME:
        adjustments['threshold_mult'] = config.get('volatility_threshold_mult', 1.5)
        adjustments['size_mult'] = config.get('regime_extreme_reduce_size', 0.5)
        adjustments['pause_trading'] = config.get('regime_extreme_pause', False)

    return adjustments


def classify_trading_session(
    timestamp: datetime,
    config: Dict[str, Any]
) -> TradingSession:
    """
    REC-003: Classify current time into a trading session using configurable boundaries.
    REC-002 (v4.3.0): Added OFF_HOURS session for 21:00-24:00 UTC.

    Session boundaries are configurable in CONFIG['session_boundaries'] to allow
    adjustment for daylight saving time changes.

    Args:
        timestamp: Current timestamp (assumed UTC)
        config: Strategy configuration with session_boundaries

    Returns:
        TradingSession enum value
    """
    hour = timestamp.hour

    # Get configurable boundaries with defaults
    bounds = config.get('session_boundaries', {})
    overlap_start = bounds.get('overlap_start', 14)
    overlap_end = bounds.get('overlap_end', 17)
    europe_start = bounds.get('europe_start', 8)
    europe_end = bounds.get('europe_end', 14)
    us_start = bounds.get('us_start', 17)
    us_end = bounds.get('us_end', 21)
    # REC-002 (v4.3.0): OFF_HOURS boundaries
    off_hours_start = bounds.get('off_hours_start', 21)
    off_hours_end = bounds.get('off_hours_end', 24)

    if overlap_start <= hour < overlap_end:
        return TradingSession.US_EUROPE_OVERLAP
    elif europe_start <= hour < europe_end:
        return TradingSession.EUROPE
    elif us_start <= hour < us_end:
        return TradingSession.US
    # REC-002 (v4.3.0): Explicitly handle OFF_HOURS (21:00-24:00 UTC)
    elif off_hours_start <= hour < off_hours_end:
        return TradingSession.OFF_HOURS
    else:
        return TradingSession.ASIA


def get_session_adjustments(
    session: TradingSession,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Get threshold and size adjustments based on trading session."""
    threshold_mults = config.get('session_threshold_multipliers', {})
    size_mults = config.get('session_size_multipliers', {})

    session_name = session.name

    return {
        'threshold_mult': threshold_mults.get(session_name, 1.0),
        'size_mult': size_mults.get(session_name, 1.0),
    }
