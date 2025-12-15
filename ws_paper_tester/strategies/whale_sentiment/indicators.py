"""
Whale Sentiment Strategy - Indicator Calculations

Contains functions for calculating:
- RSI (Relative Strength Index)
- Volume spike detection (whale proxy)
- Fear/Greed price deviation
- Trade flow analysis
- Composite confidence calculation
- Rolling correlation

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


def calculate_rsi(candles, period: int = 14) -> Dict[str, Any]:
    """
    Calculate Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss

    Uses Wilder's smoothing (EMA with period N).

    Args:
        candles: Tuple/list of candle objects with close prices
        period: RSI period (default 14)

    Returns:
        Dict with rsi, avg_gain, avg_loss, and rsi_series
    """
    result = {
        'rsi': None,
        'prev_rsi': None,
        'avg_gain': None,
        'avg_loss': None,
        'rsi_series': [],
    }

    if len(candles) < period + 1:
        return result

    # Calculate price changes
    closes = [c.close for c in candles]
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    if len(changes) < period:
        return result

    # Calculate initial average gain and loss
    gains = [max(0, c) for c in changes[:period]]
    losses = [abs(min(0, c)) for c in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    rsi_series = []

    # Calculate RSI using Wilder's smoothing
    for i in range(period, len(changes)):
        change = changes[i]
        current_gain = max(0, change)
        current_loss = abs(min(0, change))

        # Wilder's smoothing
        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1 + rs))

        rsi_series.append(rsi)

    if not rsi_series:
        return result

    result['rsi'] = rsi_series[-1]
    result['prev_rsi'] = rsi_series[-2] if len(rsi_series) >= 2 else None
    result['avg_gain'] = avg_gain
    result['avg_loss'] = avg_loss
    result['rsi_series'] = rsi_series

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
    rsi: Optional[float],
    fear_greed: Dict[str, Any],
    config: Dict[str, Any]
) -> SentimentZone:
    """
    Classify market sentiment zone based on RSI and price deviation.

    Priority:
    1. Extreme zones from RSI (most reliable)
    2. Regular zones from RSI
    3. Fear/greed from price deviation (supplementary)

    Args:
        rsi: Current RSI value
        fear_greed: Fear/greed proxy result
        config: Configuration with thresholds

    Returns:
        SentimentZone enum value
    """
    if rsi is None:
        # Fall back to price deviation only
        if fear_greed.get('is_fear', False):
            return SentimentZone.FEAR
        elif fear_greed.get('is_greed', False):
            return SentimentZone.GREED
        return SentimentZone.NEUTRAL

    # RSI-based classification
    rsi_extreme_fear = config.get('rsi_extreme_fear', 25)
    rsi_fear = config.get('rsi_fear', 40)
    rsi_greed = config.get('rsi_greed', 60)
    rsi_extreme_greed = config.get('rsi_extreme_greed', 75)

    if rsi <= rsi_extreme_fear:
        return SentimentZone.EXTREME_FEAR
    elif rsi <= rsi_fear:
        return SentimentZone.FEAR
    elif rsi >= rsi_extreme_greed:
        return SentimentZone.EXTREME_GREED
    elif rsi >= rsi_greed:
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
    Detect price/RSI divergence for additional confirmation.

    Bullish divergence: Price lower low + RSI higher low
    Bearish divergence: Price higher high + RSI lower high

    Args:
        closes: List of closing prices
        rsi_series: List of RSI values
        lookback: Number of candles for comparison

    Returns:
        Dict with divergence type and details
    """
    result = {
        'bullish_divergence': False,
        'bearish_divergence': False,
        'divergence_type': 'none',
    }

    min_data = lookback * 2 + 5

    if len(closes) < min_data or len(rsi_series) < min_data:
        return result

    # Recent vs Prior period comparison
    recent_close = closes[-lookback:]
    prior_close = closes[-2 * lookback:-lookback]
    recent_rsi = rsi_series[-lookback:]
    prior_rsi = rsi_series[-2 * lookback:-lookback]

    # Find swing points
    recent_close_min = min(recent_close)
    recent_close_max = max(recent_close)
    prior_close_min = min(prior_close)
    prior_close_max = max(prior_close)

    recent_rsi_min = min(recent_rsi)
    recent_rsi_max = max(recent_rsi)
    prior_rsi_min = min(prior_rsi)
    prior_rsi_max = max(prior_rsi)

    # Bullish divergence: Price lower low, RSI higher low
    if recent_close_min < prior_close_min and recent_rsi_min > prior_rsi_min:
        result['bullish_divergence'] = True
        result['divergence_type'] = 'bullish'

    # Bearish divergence: Price higher high, RSI lower high
    if recent_close_max > prior_close_max and recent_rsi_max < prior_rsi_max:
        result['bearish_divergence'] = True
        result['divergence_type'] = 'bearish'

    return result


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

    Weights (configurable):
    - Volume spike (whale proxy): 0.30
    - Sentiment zone (RSI): 0.25
    - Sentiment zone (price dev): 0.20
    - Trade flow confirmation: 0.15
    - Divergence bonus: 0.10

    Maximum confidence capped at 0.90.

    Args:
        whale_signal: Volume spike classification
        sentiment_zone: Sentiment zone classification
        volume_ratio: Current volume ratio
        trade_flow_imbalance: Trade flow imbalance (-1 to +1)
        divergence: Divergence detection result
        is_contrarian: Whether in contrarian mode
        direction: Signal direction ('buy' or 'short')
        config: Configuration dict

    Returns:
        Tuple of (confidence value, list of reasons)
    """
    confidence = 0.0
    reasons = []

    # Get weights from config
    weight_volume = config.get('weight_volume_spike', 0.30)
    weight_rsi = config.get('weight_rsi_sentiment', 0.25)
    weight_price = config.get('weight_price_deviation', 0.20)
    weight_flow = config.get('weight_trade_flow', 0.15)
    weight_div = config.get('weight_divergence', 0.10)

    # 1. Volume spike contribution
    if whale_signal != WhaleSignal.NEUTRAL:
        vol_conf = min(weight_volume, weight_volume * 0.5 + (volume_ratio - 2.0) * 0.05)
        confidence += vol_conf
        reasons.append(f"Volume {volume_ratio:.1f}x ({whale_signal.name.lower()})")

    # 2. RSI sentiment contribution
    if sentiment_zone == SentimentZone.EXTREME_FEAR:
        confidence += weight_rsi
        reasons.append(f"Extreme fear")
    elif sentiment_zone == SentimentZone.EXTREME_GREED:
        confidence += weight_rsi
        reasons.append(f"Extreme greed")
    elif sentiment_zone in (SentimentZone.FEAR, SentimentZone.GREED):
        confidence += weight_rsi * 0.6
        reasons.append(f"{sentiment_zone.name.lower()}")

    # 3. Price deviation contribution (implicit in sentiment zone)
    if is_extreme_zone(sentiment_zone):
        confidence += weight_price
    elif sentiment_zone != SentimentZone.NEUTRAL:
        confidence += weight_price * 0.5

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
                reasons.append(f"Buy pressure ({trade_flow_imbalance:.2f})")
            elif direction == 'short' and trade_flow_imbalance < -0.10:
                confidence += flow_conf
                reasons.append(f"Sell pressure ({trade_flow_imbalance:.2f})")

    # 5. Divergence bonus
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
