"""
Momentum Scalping Strategy - Regime and Session Classification

Contains volatility regime classification and trading session awareness logic.
Based on research from master-plan-v1.0.md.
"""
from datetime import datetime
from typing import Dict, Any

from .config import VolatilityRegime, TradingSession


def classify_volatility_regime(
    volatility_pct: float,
    config: Dict[str, Any]
) -> VolatilityRegime:
    """
    Classify current volatility into a regime.

    Regime boundaries from config:
    - LOW: volatility < regime_low_threshold
    - MEDIUM: regime_low_threshold <= volatility < regime_medium_threshold
    - HIGH: regime_medium_threshold <= volatility < regime_high_threshold
    - EXTREME: volatility >= regime_high_threshold

    Args:
        volatility_pct: Current volatility percentage
        config: Strategy configuration

    Returns:
        VolatilityRegime enum value
    """
    low = config.get('regime_low_threshold', 0.2)
    medium = config.get('regime_medium_threshold', 0.6)
    high = config.get('regime_high_threshold', 1.2)

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
    """
    Get threshold and size adjustments based on volatility regime.

    Momentum scalping adjustments:
    - LOW: Slightly tighter thresholds (more signals in quiet markets)
    - MEDIUM: Standard thresholds (baseline)
    - HIGH: Wider thresholds (filter noise, reduce size)
    - EXTREME: Very wide thresholds or pause trading entirely

    Args:
        regime: Current volatility regime
        config: Strategy configuration

    Returns:
        Dict with threshold_mult, size_mult, and pause_trading
    """
    adjustments = {
        'threshold_mult': 1.0,
        'size_mult': 1.0,
        'pause_trading': False,
    }

    if regime == VolatilityRegime.LOW:
        # Slightly tighter - can be more aggressive in quiet markets
        adjustments['threshold_mult'] = 0.9
        adjustments['size_mult'] = 1.0

    elif regime == VolatilityRegime.MEDIUM:
        # Standard - baseline settings
        adjustments['threshold_mult'] = 1.0
        adjustments['size_mult'] = 1.0

    elif regime == VolatilityRegime.HIGH:
        # Wider thresholds, reduced size - more conservative
        adjustments['threshold_mult'] = 1.3
        adjustments['size_mult'] = 0.75

    elif regime == VolatilityRegime.EXTREME:
        # Most conservative - may pause entirely
        adjustments['threshold_mult'] = config.get('volatility_threshold_mult', 1.5)
        adjustments['size_mult'] = config.get('regime_extreme_reduce_size', 0.5)
        adjustments['pause_trading'] = config.get('regime_extreme_pause', True)

    return adjustments


def classify_trading_session(
    timestamp: datetime,
    config: Dict[str, Any]
) -> TradingSession:
    """
    Classify current time into a trading session using configurable boundaries.

    Session boundaries are configurable to allow adjustment for daylight saving
    time changes.

    Session characteristics from research:
    - ASIA (00:00-08:00 UTC): Lower volume, retail-heavy
    - EUROPE (08:00-14:00 UTC): Medium volume, FX overlap
    - US_EUROPE_OVERLAP (14:00-17:00 UTC): Peak volume, best liquidity
    - US (17:00-21:00 UTC): High volume, institutional
    - OFF_HOURS (21:00-24:00 UTC): Thinnest liquidity, highest risk

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
    off_hours_start = bounds.get('off_hours_start', 21)
    off_hours_end = bounds.get('off_hours_end', 24)

    # Check sessions in priority order
    if overlap_start <= hour < overlap_end:
        return TradingSession.US_EUROPE_OVERLAP
    elif europe_start <= hour < europe_end:
        return TradingSession.EUROPE
    elif us_start <= hour < us_end:
        return TradingSession.US
    elif off_hours_start <= hour < off_hours_end:
        return TradingSession.OFF_HOURS
    else:
        # Default to ASIA (00:00-08:00 UTC)
        return TradingSession.ASIA


def get_session_adjustments(
    session: TradingSession,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Get threshold and size adjustments based on trading session.

    Session adjustments from research:
    - Higher thresholds during low-liquidity periods (wider)
    - Smaller position sizes when liquidity is thin
    - Best conditions during US/Europe overlap

    Args:
        session: Current trading session
        config: Strategy configuration

    Returns:
        Dict with threshold_mult and size_mult
    """
    threshold_mults = config.get('session_threshold_multipliers', {
        'ASIA': 1.2,
        'EUROPE': 1.0,
        'US': 1.0,
        'US_EUROPE_OVERLAP': 0.9,
        'OFF_HOURS': 1.4,
    })

    size_mults = config.get('session_size_multipliers', {
        'ASIA': 0.8,
        'EUROPE': 1.0,
        'US': 1.0,
        'US_EUROPE_OVERLAP': 1.1,
        'OFF_HOURS': 0.5,
    })

    session_name = session.name

    return {
        'threshold_mult': threshold_mults.get(session_name, 1.0),
        'size_mult': size_mults.get(session_name, 1.0),
    }


def should_skip_session(
    session: TradingSession,
    config: Dict[str, Any]
) -> bool:
    """
    Check if trading should be skipped during current session.

    OFF_HOURS has the thinnest liquidity and highest risk.
    Some strategies may want to skip this session entirely.

    Args:
        session: Current trading session
        config: Strategy configuration

    Returns:
        True if trading should be skipped
    """
    # Currently we don't skip any sessions, just adjust parameters
    # This could be extended to skip OFF_HOURS if configured
    return False
