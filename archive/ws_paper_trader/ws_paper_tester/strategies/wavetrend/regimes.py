"""
WaveTrend Oscillator Strategy - Regime Classification

Contains functions for session awareness and zone regime classification.
The WaveTrend strategy uses zone-based regimes rather than volatility regimes.
"""
from datetime import datetime
from typing import Dict, Any

from .config import TradingSession, WaveTrendZone


def classify_trading_session(
    current_time: datetime,
    config: Dict[str, Any]
) -> TradingSession:
    """
    Classify current trading session based on UTC hour.

    Sessions:
    - Asia: 00:00 - 08:00 UTC
    - Europe: 08:00 - 14:00 UTC
    - US/Europe Overlap: 14:00 - 17:00 UTC (peak liquidity)
    - US: 17:00 - 21:00 UTC
    - Off Hours: 21:00 - 24:00 UTC

    Args:
        current_time: Current timestamp
        config: Strategy configuration with session boundaries

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
) -> Dict[str, float]:
    """
    Get position size adjustment for trading session.

    Args:
        session: Current trading session
        config: Strategy configuration

    Returns:
        Dict with size_mult adjustment
    """
    multipliers = config.get('session_size_multipliers', {})

    size_mult = multipliers.get(session.name, 1.0)

    return {
        'size_mult': size_mult,
    }


def get_zone_regime_adjustments(
    zone: WaveTrendZone,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get adjustments based on current WaveTrend zone.

    Different zones may warrant different trading approaches:
    - Extreme zones: Higher confidence signals
    - Neutral zones: Avoid trading (if require_zone_exit is True)

    Args:
        zone: Current WaveTrend zone
        config: Strategy configuration

    Returns:
        Dict with zone-based adjustments
    """
    adjustments = {
        'allow_entry': True,
        'confidence_bonus': 0.0,
        'zone_description': zone.name.lower(),
    }

    if zone == WaveTrendZone.EXTREME_OVERBOUGHT:
        adjustments['confidence_bonus'] = 0.05
        adjustments['zone_description'] = 'extreme_overbought'
    elif zone == WaveTrendZone.EXTREME_OVERSOLD:
        adjustments['confidence_bonus'] = 0.05
        adjustments['zone_description'] = 'extreme_oversold'
    elif zone == WaveTrendZone.NEUTRAL:
        # In neutral zone, signals may be less reliable
        adjustments['allow_entry'] = not config.get('require_zone_exit', True)
        adjustments['zone_description'] = 'neutral'

    return adjustments


def should_wait_for_zone_exit(
    crossover_occurred: bool,
    zone_when_crossover: WaveTrendZone,
    current_zone: WaveTrendZone,
    config: Dict[str, Any]
) -> bool:
    """
    Determine if we should wait for price to exit the zone before entry.

    The require_zone_exit configuration option:
    - When True: Entry signals wait for price to exit the overbought/oversold zone
    - When False: Entry signals execute immediately on crossover in zone

    Args:
        crossover_occurred: Whether a crossover was detected
        zone_when_crossover: Zone when crossover occurred
        current_zone: Current zone
        config: Strategy configuration

    Returns:
        True if we should wait, False if we can enter now
    """
    if not crossover_occurred:
        return False

    if not config.get('require_zone_exit', True):
        return False

    # For bullish crossover from oversold, wait until we exit oversold
    from .indicators import is_in_oversold_zone, is_in_overbought_zone
    if is_in_oversold_zone(zone_when_crossover) and is_in_oversold_zone(current_zone):
        return True

    # For bearish crossover from overbought, wait until we exit overbought
    if is_in_overbought_zone(zone_when_crossover) and is_in_overbought_zone(current_zone):
        return True

    return False
