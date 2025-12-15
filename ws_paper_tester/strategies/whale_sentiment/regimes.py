"""
Whale Sentiment Strategy - Regime Classification

Contains functions for session awareness and sentiment regime classification.
The Whale Sentiment strategy uses sentiment-based regimes rather than volatility regimes.
"""
from datetime import datetime
from typing import Dict, Any

from .config import TradingSession, SentimentZone


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
