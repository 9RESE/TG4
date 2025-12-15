"""
Whale Sentiment Strategy - Regime Classification

Contains functions for session awareness, sentiment regime classification,
and volatility regime adjustments.

v1.3.0 Additions:
- REC-023: Volatility regime classification and adjustments
- REC-025: Extended fear period detection
- REC-027: Dynamic confidence threshold support
"""
from datetime import datetime
from typing import Dict, Any, Optional

from .config import TradingSession, SentimentZone


# =============================================================================
# REC-023: Volatility Regime Classification
# REC-031: Added EXTREME regime with trading pause (v1.4.0)
# =============================================================================
class VolatilityRegime:
    """Volatility regime classifications."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    EXTREME = 'extreme'  # REC-031: Added for trading pause in extreme conditions
    UNKNOWN = 'unknown'


def classify_volatility_regime(
    atr_pct: Optional[float],
    config: Dict[str, Any]
) -> str:
    """
    REC-023: Classify volatility regime based on ATR percentage.
    REC-031: Added EXTREME regime classification (v1.4.0).

    Args:
        atr_pct: ATR as percentage of price
        config: Strategy configuration

    Returns:
        Volatility regime string ('low', 'medium', 'high', 'extreme', 'unknown')
    """
    if atr_pct is None:
        return VolatilityRegime.UNKNOWN

    low_threshold = config.get('volatility_low_threshold', 1.5)
    high_threshold = config.get('volatility_high_threshold', 3.5)
    extreme_threshold = config.get('volatility_extreme_threshold', 6.0)  # REC-031

    if atr_pct < low_threshold:
        return VolatilityRegime.LOW
    elif atr_pct >= extreme_threshold:
        return VolatilityRegime.EXTREME  # REC-031: Pause trading in extreme conditions
    elif atr_pct >= high_threshold:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.MEDIUM


def get_volatility_adjustments(
    volatility_regime: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    REC-023: Get parameter adjustments based on volatility regime.
    REC-031: Added should_pause flag for EXTREME regime (v1.4.0).

    Args:
        volatility_regime: Current volatility regime
        config: Strategy configuration

    Returns:
        Dict with size_mult, stop_mult, cooldown_mult, should_pause adjustments
    """
    adjustments = {
        'size_mult': 1.0,
        'stop_mult': 1.0,
        'cooldown_mult': 1.0,
        'should_pause': False,  # REC-031: Pause trading flag
    }

    if volatility_regime == VolatilityRegime.EXTREME:
        # REC-031: Extreme volatility - pause trading entirely
        adjustments['size_mult'] = 0.0  # No new entries
        adjustments['stop_mult'] = 2.0  # Very wide stops if position exists
        adjustments['cooldown_mult'] = 3.0  # Long cooldown
        adjustments['should_pause'] = True
    elif volatility_regime == VolatilityRegime.HIGH:
        adjustments['size_mult'] = config.get('volatility_high_size_mult', 0.75)
        adjustments['stop_mult'] = config.get('volatility_high_stop_mult', 1.5)
        adjustments['cooldown_mult'] = config.get('volatility_high_cooldown_mult', 1.5)
    elif volatility_regime == VolatilityRegime.LOW:
        # Tighter stops, can be slightly larger positions in low volatility
        adjustments['size_mult'] = 1.1
        adjustments['stop_mult'] = 0.8
        adjustments['cooldown_mult'] = 0.8

    return adjustments


# =============================================================================
# REC-025: Extended Fear Period Detection
# =============================================================================
def check_extended_fear_period(
    state: Dict[str, Any],
    sentiment_zone: SentimentZone,
    current_time: datetime,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    REC-025: Check for extended periods of extreme sentiment.

    Tracks consecutive hours in extreme sentiment zones and recommends
    size reduction or entry pause to prevent capital exhaustion.

    Args:
        state: Strategy state dict
        sentiment_zone: Current sentiment zone
        current_time: Current timestamp
        config: Strategy configuration

    Returns:
        Dict with is_extended, hours_in_extreme, size_mult, should_pause
    """
    result = {
        'is_extended': False,
        'hours_in_extreme': 0,
        'size_mult': 1.0,
        'should_pause': False,
        'zone_entered': None,
    }

    if not config.get('use_extended_fear_detection', True):
        return result

    # Initialize tracking in state if needed
    if 'extreme_zone_start' not in state:
        state['extreme_zone_start'] = None
        state['extreme_zone_type'] = None

    is_extreme = sentiment_zone in (SentimentZone.EXTREME_FEAR, SentimentZone.EXTREME_GREED)

    if is_extreme:
        # If not already tracking, start tracking
        if state['extreme_zone_start'] is None:
            state['extreme_zone_start'] = current_time
            state['extreme_zone_type'] = sentiment_zone.name

        # Calculate hours in extreme zone
        hours_in_extreme = (current_time - state['extreme_zone_start']).total_seconds() / 3600
        result['hours_in_extreme'] = hours_in_extreme
        result['zone_entered'] = state['extreme_zone_start'].isoformat()

        # Check thresholds
        threshold_hours = config.get('extended_fear_threshold_hours', 168)  # 7 days
        pause_hours = config.get('extended_fear_pause_hours', 336)  # 14 days

        if hours_in_extreme >= pause_hours:
            result['is_extended'] = True
            result['should_pause'] = True
            result['size_mult'] = 0.0  # No new entries
        elif hours_in_extreme >= threshold_hours:
            result['is_extended'] = True
            result['size_mult'] = config.get('extended_fear_size_reduction', 0.70)
    else:
        # Reset tracking if no longer in extreme zone
        state['extreme_zone_start'] = None
        state['extreme_zone_type'] = None

    return result


# =============================================================================
# REC-027: Dynamic Confidence Threshold
# =============================================================================
def calculate_dynamic_confidence_threshold(
    base_threshold: float,
    sentiment_zone: SentimentZone,
    volatility_regime: str,
    config: Dict[str, Any]
) -> float:
    """
    REC-027: Calculate dynamic confidence threshold based on conditions.

    Args:
        base_threshold: Base minimum confidence threshold
        sentiment_zone: Current sentiment zone
        volatility_regime: Current volatility regime
        config: Strategy configuration

    Returns:
        Adjusted confidence threshold
    """
    if not config.get('use_dynamic_confidence', True):
        return base_threshold

    threshold = base_threshold

    # Extreme sentiment bonus (easier entry in extreme zones)
    if sentiment_zone in (SentimentZone.EXTREME_FEAR, SentimentZone.EXTREME_GREED):
        bonus = config.get('confidence_extreme_bonus', -0.05)
        threshold += bonus

    # High volatility penalty (harder entry in volatile markets)
    if volatility_regime == VolatilityRegime.HIGH:
        penalty = config.get('confidence_high_volatility_penalty', 0.05)
        threshold += penalty

    # Clamp to valid range (0.40 - 0.60)
    threshold = max(0.40, min(0.60, threshold))

    return threshold


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


def get_sentiment_regime_adjustments(
    zone: SentimentZone,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get adjustments based on current sentiment zone.

    Different sentiment zones warrant different trading approaches:
    - Extreme zones: Higher confidence signals, more aggressive sizing
    - Neutral zones: No trading (contrarian needs extreme sentiment)

    Args:
        zone: Current sentiment zone
        config: Strategy configuration

    Returns:
        Dict with sentiment-based adjustments
    """
    adjustments = {
        'allow_entry': True,
        'confidence_bonus': 0.0,
        'size_multiplier': 1.0,
        'zone_description': zone.name.lower(),
    }

    if zone == SentimentZone.EXTREME_FEAR:
        adjustments['confidence_bonus'] = 0.10
        adjustments['size_multiplier'] = 1.1  # Slightly larger in extreme fear
        adjustments['zone_description'] = 'extreme_fear'
    elif zone == SentimentZone.EXTREME_GREED:
        adjustments['confidence_bonus'] = 0.10
        adjustments['size_multiplier'] = 0.9  # Slightly smaller (greed more dangerous)
        adjustments['zone_description'] = 'extreme_greed'
    elif zone == SentimentZone.FEAR:
        adjustments['confidence_bonus'] = 0.05
        adjustments['zone_description'] = 'fear'
    elif zone == SentimentZone.GREED:
        adjustments['confidence_bonus'] = 0.05
        adjustments['zone_description'] = 'greed'
    elif zone == SentimentZone.NEUTRAL:
        # In neutral zone, contrarian strategy should not trade
        adjustments['allow_entry'] = False
        adjustments['zone_description'] = 'neutral'

    return adjustments


def is_contrarian_opportunity(
    sentiment_zone: SentimentZone,
    whale_signal_name: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Determine if current conditions present a contrarian opportunity.

    Contrarian opportunities:
    - Fear sentiment + whale accumulation = bullish opportunity
    - Greed sentiment + whale distribution = bearish opportunity

    Also validates alignment:
    - Fear + accumulation is expected (smart money buying fear)
    - Fear + distribution is conflicting (may indicate more downside)

    Args:
        sentiment_zone: Current sentiment zone
        whale_signal_name: Whale signal classification name
        config: Strategy configuration

    Returns:
        Dict with opportunity details
    """
    result = {
        'has_opportunity': False,
        'direction': None,
        'alignment': 'none',
        'reason': '',
    }

    is_contrarian = config.get('contrarian_mode', True)

    # Fear sentiment
    if sentiment_zone in (SentimentZone.EXTREME_FEAR, SentimentZone.FEAR):
        if is_contrarian:
            result['direction'] = 'buy'  # Buy the fear

            # Check whale alignment
            if whale_signal_name == 'accumulation':
                result['has_opportunity'] = True
                result['alignment'] = 'strong'
                result['reason'] = 'Fear + whale accumulation (smart money buying)'
            elif whale_signal_name == 'neutral':
                result['has_opportunity'] = True
                result['alignment'] = 'moderate'
                result['reason'] = 'Fear without clear whale direction'
            elif whale_signal_name == 'distribution':
                result['has_opportunity'] = False
                result['alignment'] = 'conflicting'
                result['reason'] = 'Fear + whale distribution (wait for confirmation)'
        else:
            # Momentum mode: follow the fear (sell)
            result['direction'] = 'short'
            result['has_opportunity'] = True
            result['alignment'] = 'momentum'
            result['reason'] = 'Fear momentum following'

    # Greed sentiment
    elif sentiment_zone in (SentimentZone.EXTREME_GREED, SentimentZone.GREED):
        if is_contrarian:
            result['direction'] = 'short'  # Sell the greed

            # Check whale alignment
            if whale_signal_name == 'distribution':
                result['has_opportunity'] = True
                result['alignment'] = 'strong'
                result['reason'] = 'Greed + whale distribution (smart money selling)'
            elif whale_signal_name == 'neutral':
                result['has_opportunity'] = True
                result['alignment'] = 'moderate'
                result['reason'] = 'Greed without clear whale direction'
            elif whale_signal_name == 'accumulation':
                result['has_opportunity'] = False
                result['alignment'] = 'conflicting'
                result['reason'] = 'Greed + whale accumulation (wait for confirmation)'
        else:
            # Momentum mode: follow the greed (buy)
            result['direction'] = 'buy'
            result['has_opportunity'] = True
            result['alignment'] = 'momentum'
            result['reason'] = 'Greed momentum following'

    # Neutral sentiment - no contrarian opportunity
    else:
        result['reason'] = 'Neutral sentiment - no contrarian signal'

    return result


def should_reduce_size_for_sentiment(
    sentiment_zone: SentimentZone,
    direction: str,
    config: Dict[str, Any]
) -> float:
    """
    Determine if position size should be reduced based on sentiment direction.

    Reduce size when:
    - Shorting in greed (squeeze risk)
    - Buying in extreme fear (catching falling knife risk)

    Args:
        sentiment_zone: Current sentiment zone
        direction: Trade direction ('buy' or 'short')
        config: Strategy configuration

    Returns:
        Size multiplier (1.0 = no reduction)
    """
    multiplier = 1.0

    # Shorting in greed has squeeze risk
    if direction == 'short':
        if sentiment_zone == SentimentZone.EXTREME_GREED:
            multiplier *= 0.8  # 20% reduction
        elif sentiment_zone == SentimentZone.GREED:
            multiplier *= 0.9  # 10% reduction

    # Buying in extreme fear has falling knife risk
    if direction == 'buy':
        if sentiment_zone == SentimentZone.EXTREME_FEAR:
            # Only slight reduction - extreme fear is actually good for contrarian
            multiplier *= 0.95

    return multiplier
