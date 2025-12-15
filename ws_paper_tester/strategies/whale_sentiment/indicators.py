"""
Whale Sentiment Strategy - Indicator Calculations

Contains functions for calculating:
- ATR (Average True Range) for volatility regime - REC-023
- Volume spike detection (whale proxy)
- Fear/Greed price deviation (PRIMARY sentiment signal per REC-021)
- Trade flow analysis
- Composite confidence calculation
- Rolling correlation

REC-032: RSI code REMOVED (v1.4.0)
- calculate_rsi function removed per REC-032 clean code principles
- detect_rsi_divergence retained as stub for backwards compatibility
Academic research (PMC/NIH 2023, QuantifiedStrategies 2024) shows RSI
ineffectiveness in crypto markets.

===============================================================================
REC-012: Candle Aggregation Edge Cases Documentation
===============================================================================
This module expects candle data from ws_paper_tester DataSnapshot:
- data.candles_5m: Primary source (5-minute candles, aggregated from ticks)
- data.candles_1m: Fallback source (1-minute candles)

Candle Data Handling:
1. Candle Format: Each candle has (timestamp, open, high, low, close, volume)
2. Ordering: Candles are ordered oldest-first (newest at end of tuple)
3. Completeness: Only complete candles are used; partial candles are excluded
4. Timestamp Alignment: Candles are aligned to fixed time boundaries

Edge Cases Handled:
- Empty candle buffer: Returns None/empty results, signal generation skipped
- Insufficient candles: Minimum buffer check before calculations
- Gap handling: Gaps in data don't break calculations but may affect accuracy
- Overflow: Uses standard Python float, sufficient for typical price ranges
===============================================================================
"""
import warnings
from typing import List, Dict, Any, Optional, Tuple

from .config import SentimentZone, WhaleSignal


def calculate_ema(values: List[float], period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average.

    Formula:
    Multiplier = 2 / (Period + 1)
    EMA_today = (Value_today * Multiplier) + (EMA_yesterday * (1 - Multiplier))

    Args:
        values: List of values (oldest first)
        period: EMA period

    Returns:
        EMA value or None if insufficient data
    """
    if len(values) < period:
        return None

    # Initialize EMA with SMA for the first 'period' values
    sma = sum(values[:period]) / period
    multiplier = 2.0 / (period + 1)

    ema = sma
    for value in values[period:]:
        ema = (value * multiplier) + (ema * (1 - multiplier))

    return ema


def calculate_sma(values: List[float], period: int) -> Optional[float]:
    """
    Calculate Simple Moving Average.

    Args:
        values: List of values (oldest first)
        period: SMA period

    Returns:
        SMA value or None if insufficient data
    """
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def calculate_atr(candles, period: int = 14) -> Dict[str, Any]:
    """
    Calculate Average True Range (ATR) for volatility regime classification.

    REC-023: Added for volatility regime detection.

    ATR = Average of True Range over period
    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))

    Args:
        candles: Tuple/list of candle objects
        period: ATR period (default 14)

    Returns:
        Dict with atr, atr_pct (as % of price), tr_series
    """
    result = {
        'atr': None,
        'atr_pct': None,
        'tr_series': [],
    }

    if len(candles) < period + 1:
        return result

    # Calculate True Range series
    tr_series = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_series.append(tr)

    if len(tr_series) < period:
        return result

    # Calculate ATR using Wilder's smoothing
    atr = sum(tr_series[:period]) / period

    for i in range(period, len(tr_series)):
        atr = (atr * (period - 1) + tr_series[i]) / period

    result['atr'] = atr
    result['tr_series'] = tr_series

    # Calculate ATR as percentage of current price
    current_price = candles[-1].close
    if current_price > 0:
        result['atr_pct'] = (atr / current_price) * 100

    return result


def detect_volume_spike(
    candles,
    window: int = 288,
    spike_mult: float = 2.0
) -> Dict[str, Any]:
    """
    Detect volume spike as whale activity proxy.

    Volume Ratio = Current Volume / MA(Volume, window)
    Whale Activity = Volume Ratio >= spike_mult

    Args:
        candles: Tuple/list of candle objects with volume
        window: Volume average window (default 288 = 24h in 5m candles)
        spike_mult: Multiplier threshold for spike detection

    Returns:
        Dict with volume_ratio, has_spike, current_volume, avg_volume
    """
    result = {
        'volume_ratio': 0.0,
        'has_spike': False,
        'current_volume': 0.0,
        'avg_volume': 0.0,
        'spike_strength': 0.0,  # How much above threshold
    }

    if len(candles) < window + 1:
        return result

    # Get volume series
    volumes = [c.volume for c in candles]

    # Current volume (last complete candle)
    current_volume = volumes[-1]

    # Average volume over window (excluding current)
    avg_volume = sum(volumes[-window - 1:-1]) / window

    if avg_volume > 0:
        volume_ratio = current_volume / avg_volume
        result['volume_ratio'] = volume_ratio
        result['has_spike'] = volume_ratio >= spike_mult
        result['spike_strength'] = max(0, volume_ratio - spike_mult)

    result['current_volume'] = current_volume
    result['avg_volume'] = avg_volume

    return result


def classify_whale_signal(
    volume_spike: Dict[str, Any],
    candles,
    price_move_threshold: float = 0.1
) -> WhaleSignal:
    """
    Classify whale signal based on volume spike and price movement.

    Accumulation: Volume spike + price increase
    Distribution: Volume spike + price decrease
    Neutral: No volume spike

    Args:
        volume_spike: Volume spike detection result
        candles: Candle data for price change calculation
        price_move_threshold: Minimum % price move for classification

    Returns:
        WhaleSignal enum value
    """
    if not volume_spike.get('has_spike', False):
        return WhaleSignal.NEUTRAL

    if len(candles) < 2:
        return WhaleSignal.NEUTRAL

    # Calculate price change
    current_close = candles[-1].close
    prev_close = candles[-2].close

    if prev_close <= 0:
        return WhaleSignal.NEUTRAL

    price_change_pct = (current_close - prev_close) / prev_close * 100

    if abs(price_change_pct) < price_move_threshold:
        # Volume spike without price movement - potentially suspicious
        return WhaleSignal.NEUTRAL

    if price_change_pct > 0:
        return WhaleSignal.ACCUMULATION
    else:
        return WhaleSignal.DISTRIBUTION


def validate_volume_spike(
    volume_ratio: float,
    price_change_pct: float,
    spread_pct: float,
    trade_count: int,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Filter potential false positive volume spikes.

    Red flags:
    - Volume spike without price movement (wash trading)
    - Extremely wide spread (manipulation)
    - Very few trades (single large order vs distributed)

    Args:
        volume_ratio: Current volume ratio
        price_change_pct: Recent price change percentage
        spread_pct: Current bid-ask spread percentage
        trade_count: Number of trades during spike
        config: Configuration dict

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    # Volume spike without price movement = suspicious
    min_price_move = config.get('volume_spike_price_move_pct', 0.1)
    if volume_ratio > 3.0 and abs(price_change_pct) < min_price_move:
        return False, "volume_without_price_move"

    # Wide spread during spike = thin book / manipulation
    max_spread = config.get('max_spread_pct', 0.5)
    if spread_pct > max_spread:
        return False, "wide_spread"

    # Too few trades = single large order (less predictive)
    min_trades = config.get('min_spike_trades', 20)
    if trade_count < min_trades:
        return False, "insufficient_trade_count"

    return True, "valid"


def calculate_fear_greed_proxy(
    candles,
    lookback: int = 48,
    fear_deviation: float = -5.0,
    greed_deviation: float = 5.0
) -> Dict[str, Any]:
    """
    Calculate Fear/Greed sentiment from price deviation.

    Fear: Price significantly below recent high
    Greed: Price significantly above recent low

    Args:
        candles: Candle data for price history
        lookback: Number of candles for high/low reference
        fear_deviation: Threshold % from high for fear (-5%)
        greed_deviation: Threshold % from low for greed (+5%)

    Returns:
        Dict with from_high_pct, from_low_pct, sentiment indicators
    """
    result = {
        'from_high_pct': 0.0,
        'from_low_pct': 0.0,
        'recent_high': 0.0,
        'recent_low': 0.0,
        'current_price': 0.0,
        'is_fear': False,
        'is_greed': False,
    }

    if len(candles) < lookback:
        return result

    # Get recent candles for high/low
    recent_candles = candles[-lookback:]
    highs = [c.high for c in recent_candles]
    lows = [c.low for c in recent_candles]

    recent_high = max(highs)
    recent_low = min(lows)
    current_price = candles[-1].close

    result['recent_high'] = recent_high
    result['recent_low'] = recent_low
    result['current_price'] = current_price

    if recent_high > 0:
        result['from_high_pct'] = (current_price - recent_high) / recent_high * 100
        result['is_fear'] = result['from_high_pct'] <= fear_deviation

    if recent_low > 0:
        result['from_low_pct'] = (current_price - recent_low) / recent_low * 100
        result['is_greed'] = result['from_low_pct'] >= greed_deviation

    return result


def classify_sentiment_zone(
    fear_greed: Dict[str, Any],
    config: Dict[str, Any],
    rsi: Optional[float] = None  # DEPRECATED: Retained for backwards compatibility
) -> SentimentZone:
    """
    Classify market sentiment zone based on price deviation.

    REC-021: RSI COMPLETELY REMOVED (v1.3.0).
    Now uses ONLY price deviation for sentiment classification.
    Academic research shows RSI ineffectiveness in crypto markets.

    Classification based on deviation from recent high/low:
    - EXTREME_FEAR: from_high_pct <= extreme_fear_deviation (default -8%)
    - FEAR: from_high_pct <= fear_deviation (default -5%)
    - EXTREME_GREED: from_low_pct >= extreme_greed_deviation (default +8%)
    - GREED: from_low_pct >= greed_deviation (default +5%)
    - NEUTRAL: Neither fear nor greed conditions met

    Args:
        fear_greed: Fear/greed proxy result with from_high_pct, from_low_pct
        config: Configuration with thresholds
        rsi: DEPRECATED - ignored, retained for backwards compatibility

    Returns:
        SentimentZone enum value
    """
    # Get thresholds from config
    fear_deviation = config.get('fear_deviation_pct', -5.0)
    greed_deviation = config.get('greed_deviation_pct', 5.0)
    # REC-021: New extreme deviation thresholds for finer classification
    extreme_fear_deviation = config.get('extreme_fear_deviation_pct', fear_deviation * 1.6)  # -8%
    extreme_greed_deviation = config.get('extreme_greed_deviation_pct', greed_deviation * 1.6)  # +8%

    from_high_pct = fear_greed.get('from_high_pct', 0)
    from_low_pct = fear_greed.get('from_low_pct', 0)

    # Fear classification (price below recent high)
    if from_high_pct <= extreme_fear_deviation:
        return SentimentZone.EXTREME_FEAR
    elif from_high_pct <= fear_deviation:
        return SentimentZone.FEAR

    # Greed classification (price above recent low)
    if from_low_pct >= extreme_greed_deviation:
        return SentimentZone.EXTREME_GREED
    elif from_low_pct >= greed_deviation:
        return SentimentZone.GREED

    return SentimentZone.NEUTRAL


def get_sentiment_string(zone: SentimentZone) -> str:
    """Convert SentimentZone to string representation."""
    zone_strings = {
        SentimentZone.EXTREME_FEAR: 'extreme_fear',
        SentimentZone.FEAR: 'fear',
        SentimentZone.NEUTRAL: 'neutral',
        SentimentZone.GREED: 'greed',
        SentimentZone.EXTREME_GREED: 'extreme_greed',
    }
    return zone_strings.get(zone, 'unknown')


def is_fear_zone(zone: SentimentZone) -> bool:
    """Check if zone is fear or extreme fear."""
    return zone in (SentimentZone.FEAR, SentimentZone.EXTREME_FEAR)


def is_greed_zone(zone: SentimentZone) -> bool:
    """Check if zone is greed or extreme greed."""
    return zone in (SentimentZone.GREED, SentimentZone.EXTREME_GREED)


def is_extreme_zone(zone: SentimentZone) -> bool:
    """Check if zone is extreme (either direction)."""
    return zone in (SentimentZone.EXTREME_FEAR, SentimentZone.EXTREME_GREED)


def detect_rsi_divergence(
    closes: List[float],
    rsi_series: List[float],
    lookback: int = 14
) -> Dict[str, Any]:
    """
    REC-032: RSI divergence REMOVED (v1.4.0).
    REC-036: Scheduled for removal in v2.0.0 (v1.5.0).

    .. deprecated:: 1.4.0
        RSI-based indicators removed from strategy per REC-021.
        This function is a stub retained for backwards compatibility.
        **Will be removed in v2.0.0** - Update any code that calls this function.

    RSI-based indicators removed from strategy per REC-021. Function retained
    as stub for backwards compatibility with signal.py call at line 286.
    Always returns no divergence.

    Args:
        closes: List of closing prices (ignored)
        rsi_series: List of RSI values (ignored)
        lookback: Number of candles for comparison (ignored)

    Returns:
        Dict with divergence_type: 'none' (always)
    """
    # REC-036: Issue deprecation warning
    warnings.warn(
        "detect_rsi_divergence is deprecated and will be removed in v2.0.0. "
        "RSI-based indicators are no longer used in this strategy (REC-021).",
        DeprecationWarning,
        stacklevel=2
    )
    return {
        'bullish_divergence': False,
        'bearish_divergence': False,
        'divergence_type': 'none',
    }


def calculate_trade_flow(trades, lookback: int = 50) -> Dict[str, Any]:
    """
    Calculate buy/sell volume and imbalance from recent trades.

    Args:
        trades: Tuple/list of trade objects with side and value attributes
        lookback: Number of recent trades to analyze

    Returns:
        Dict with buy_volume, sell_volume, total_volume, imbalance, and trade_count
    """
    result = {
        'buy_volume': 0.0,
        'sell_volume': 0.0,
        'total_volume': 0.0,
        'imbalance': 0.0,  # Range: -1 (all sells) to +1 (all buys)
        'trade_count': 0,
        'valid': False,
    }

    if not trades or len(trades) == 0:
        return result

    # Get recent trades
    recent_trades = trades[-lookback:] if len(trades) > lookback else trades
    result['trade_count'] = len(recent_trades)

    if result['trade_count'] < 5:  # Need minimum trades for meaningful analysis
        return result

    # Aggregate buy/sell volumes
    for trade in recent_trades:
        # Handle different trade object formats
        if hasattr(trade, 'side'):
            side = trade.side
            value = getattr(trade, 'value', getattr(trade, 'size', 0) * getattr(trade, 'price', 1))
        elif isinstance(trade, dict):
            side = trade.get('side', '')
            value = trade.get('value', trade.get('size', 0) * trade.get('price', 1))
        else:
            continue

        if side == 'buy':
            result['buy_volume'] += value
        elif side == 'sell':
            result['sell_volume'] += value

    result['total_volume'] = result['buy_volume'] + result['sell_volume']

    # Calculate imbalance: (buy - sell) / total
    if result['total_volume'] > 0:
        result['imbalance'] = (result['buy_volume'] - result['sell_volume']) / result['total_volume']
        result['valid'] = True

    return result


def check_trade_flow_confirmation(
    trades,
    direction: str,
    threshold: float = 0.10,
    lookback: int = 50
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if trade flow supports the signal direction.

    REC-003: Contrarian Mode Logic Clarification
    ============================================
    For contrarian strategy, the trade flow check is intentionally lenient:

    - BUY signals (in fear): We ACCEPT mild selling pressure (imbalance >= -threshold)
      Rationale: Contrarian buys occur during fear/panic selling. Requiring positive
      flow would reject valid contrarian entries. We only reject if selling is EXTREME.

    - SHORT signals (in greed): We ACCEPT mild buying pressure (imbalance <= +threshold)
      Rationale: Contrarian shorts occur during FOMO buying. Requiring negative
      flow would reject valid contrarian entries. We only reject if buying is EXTREME.

    This differs from momentum strategies which would require flow alignment.
    The threshold (-0.10/+0.10) allows mild opposing flow while blocking extreme cases.

    Args:
        trades: Tuple/list of trade objects
        direction: Signal direction ('buy' or 'short')
        threshold: Maximum opposing imbalance to accept (0.10 = 10%)
        lookback: Number of recent trades to analyze

    Returns:
        Tuple of (is_confirmed, flow_data)
    """
    flow_data = calculate_trade_flow(trades, lookback)

    if not flow_data['valid']:
        # If we can't calculate trade flow, don't block the signal
        return True, flow_data

    imbalance = flow_data['imbalance']

    # For buy signals, we want positive imbalance (buying pressure)
    # For contrarian buys in fear, we accept neutral or mild selling
    if direction == 'buy':
        # Confirmed if imbalance is not strongly negative
        is_confirmed = imbalance >= -threshold
        flow_data['confirms_signal'] = is_confirmed
        flow_data['direction_wanted'] = 'positive (buy pressure)'
    # For short signals, we want negative imbalance (selling pressure)
    # For contrarian shorts in greed, we accept neutral or mild buying
    elif direction == 'short':
        # Confirmed if imbalance is not strongly positive
        is_confirmed = imbalance <= threshold
        flow_data['confirms_signal'] = is_confirmed
        flow_data['direction_wanted'] = 'negative (sell pressure)'
    else:
        is_confirmed = True  # Unknown direction, don't block
        flow_data['confirms_signal'] = True

    return is_confirmed, flow_data


def calculate_rolling_correlation(
    prices_a: List[float],
    prices_b: List[float],
    window: int = 20
) -> Optional[float]:
    """
    Calculate Pearson correlation on price returns.

    Args:
        prices_a: List of prices for asset A (oldest first)
        prices_b: List of prices for asset B (oldest first)
        window: Number of periods for correlation calculation

    Returns:
        Pearson correlation coefficient (-1 to +1) or None if insufficient data
    """
    if len(prices_a) < window + 1 or len(prices_b) < window + 1:
        return None

    # Calculate returns (percentage change)
    returns_a = []
    returns_b = []

    for i in range(1, window + 1):
        idx = -(window + 1 - i)
        if prices_a[idx - 1] > 0:
            returns_a.append((prices_a[idx] - prices_a[idx - 1]) / prices_a[idx - 1])
        if prices_b[idx - 1] > 0:
            returns_b.append((prices_b[idx] - prices_b[idx - 1]) / prices_b[idx - 1])

    if len(returns_a) != len(returns_b) or len(returns_a) < window // 2:
        return None

    n = len(returns_a)

    # Calculate means
    mean_a = sum(returns_a) / n
    mean_b = sum(returns_b) / n

    # Calculate covariance and standard deviations
    cov = 0.0
    var_a = 0.0
    var_b = 0.0

    for i in range(n):
        diff_a = returns_a[i] - mean_a
        diff_b = returns_b[i] - mean_b
        cov += diff_a * diff_b
        var_a += diff_a ** 2
        var_b += diff_b ** 2

    # Avoid division by zero
    if var_a < 1e-10 or var_b < 1e-10:
        return None

    std_a = (var_a / n) ** 0.5
    std_b = (var_b / n) ** 0.5

    if std_a < 1e-10 or std_b < 1e-10:
        return None

    correlation = (cov / n) / (std_a * std_b)

    # Clamp to valid range
    return max(-1.0, min(1.0, correlation))


def calculate_composite_confidence(
    whale_signal: WhaleSignal,
    sentiment_zone: SentimentZone,
    volume_ratio: float,
    trade_flow_imbalance: float,
    divergence: Dict[str, Any],
    is_contrarian: bool,
    direction: str,
    config: Dict[str, Any]
) -> Tuple[float, List[str]]:
    """
    Calculate weighted confidence from multiple signals.

    REC-013: RSI REMOVED from confidence calculation based on academic evidence
    showing RSI ineffectiveness in crypto markets (PMC/NIH 2023, QuantifiedStrategies 2024).

    Weights (configurable, v1.2.0):
    - Volume spike (whale proxy): 0.55 (PRIMARY signal)
    - Price deviation: 0.35 (sentiment classification)
    - Trade flow confirmation: 0.10
    - RSI/Divergence: 0.00 (REMOVED per REC-013)

    Maximum confidence capped at 0.90.

    Args:
        whale_signal: Volume spike classification
        sentiment_zone: Sentiment zone classification
        volume_ratio: Current volume ratio
        trade_flow_imbalance: Trade flow imbalance (-1 to +1)
        divergence: Divergence detection result (IGNORED per REC-013)
        is_contrarian: Whether in contrarian mode
        direction: Signal direction ('buy' or 'short')
        config: Configuration dict

    Returns:
        Tuple of (confidence value, list of reasons)
    """
    confidence = 0.0
    reasons = []

    # Get weights from config
    weight_volume = config.get('weight_volume_spike', 0.55)
    # REC-013: RSI weight should be 0 - removed from calculation
    weight_rsi = config.get('weight_rsi_sentiment', 0.00)
    weight_price = config.get('weight_price_deviation', 0.35)
    weight_flow = config.get('weight_trade_flow', 0.10)
    # REC-013: Divergence weight should be 0 - removed with RSI
    weight_div = config.get('weight_divergence', 0.00)

    # REC-020: Extracted magic numbers from config
    vol_conf_base = config.get('volume_confidence_base', 0.50)
    vol_conf_bonus = config.get('volume_confidence_bonus_per_ratio', 0.05)

    # 1. Volume spike contribution (PRIMARY signal per REC-013)
    if whale_signal != WhaleSignal.NEUTRAL:
        vol_conf = min(weight_volume, weight_volume * vol_conf_base + (volume_ratio - 2.0) * vol_conf_bonus)
        confidence += vol_conf
        reasons.append(f"Volume {volume_ratio:.1f}x ({whale_signal.name.lower()})")

    # 2. REC-013: RSI contribution REMOVED - skip if weight is 0
    if weight_rsi > 0:
        # Legacy RSI contribution (only if explicitly enabled)
        if sentiment_zone == SentimentZone.EXTREME_FEAR:
            confidence += weight_rsi
            reasons.append("Extreme fear (RSI)")
        elif sentiment_zone == SentimentZone.EXTREME_GREED:
            confidence += weight_rsi
            reasons.append("Extreme greed (RSI)")
        elif sentiment_zone in (SentimentZone.FEAR, SentimentZone.GREED):
            confidence += weight_rsi * 0.6
            reasons.append(f"{sentiment_zone.name.lower()} (RSI)")

    # 3. Price deviation contribution (now PRIMARY sentiment signal per REC-013)
    if is_extreme_zone(sentiment_zone):
        confidence += weight_price
        reasons.append(f"Price dev: extreme {sentiment_zone.name.lower().replace('extreme_', '')}")
    elif sentiment_zone != SentimentZone.NEUTRAL:
        confidence += weight_price * 0.6
        reasons.append(f"Price dev: {sentiment_zone.name.lower()}")

    # 4. Trade flow confirmation
    if abs(trade_flow_imbalance) > 0.10:
        flow_conf = min(weight_flow, abs(trade_flow_imbalance) * weight_flow * 2)
        # For contrarian: opposite flow is actually confirming
        if is_contrarian:
            if direction == 'buy' and trade_flow_imbalance < -0.10:
                confidence += flow_conf
                reasons.append(f"Sell pressure ({trade_flow_imbalance:.2f})")
            elif direction == 'short' and trade_flow_imbalance > 0.10:
                confidence += flow_conf
                reasons.append(f"Buy pressure ({trade_flow_imbalance:.2f})")
        else:
            if direction == 'buy' and trade_flow_imbalance > 0.10:
                confidence += flow_conf
                reasons.append(f"Buy flow ({trade_flow_imbalance:.2f})")
            elif direction == 'short' and trade_flow_imbalance < -0.10:
                confidence += flow_conf
                reasons.append(f"Sell flow ({trade_flow_imbalance:.2f})")

    # 5. REC-013: Divergence bonus REMOVED - skip if weight is 0
    if weight_div > 0:
        # Legacy divergence contribution (only if explicitly enabled)
        if direction == 'buy' and divergence.get('bullish_divergence', False):
            confidence += weight_div
            reasons.append("Bullish divergence")
        elif direction == 'short' and divergence.get('bearish_divergence', False):
            confidence += weight_div
            reasons.append("Bearish divergence")

    # Cap confidence
    max_conf = config.get('max_confidence', 0.90)
    if direction == 'buy':
        max_conf = min(max_conf, config.get('max_long_confidence', 0.90))
    else:
        max_conf = min(max_conf, config.get('max_short_confidence', 0.85))

    confidence = min(confidence, max_conf)

    return confidence, reasons
