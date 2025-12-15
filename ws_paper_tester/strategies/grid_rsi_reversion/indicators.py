"""
Grid RSI Reversion Strategy - Indicator Calculations

Contains functions for calculating RSI, ATR, and adaptive RSI zones.
Implements the RSI confidence calculation from legacy code.
"""
from typing import Dict, Any, List, Optional, Tuple

from .config import RSIZone, get_symbol_config


def calculate_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index using Wilder's smoothing method.

    Formula:
    RS = Average Gain / Average Loss (over N periods)
    RSI = 100 - (100 / (1 + RS))

    Args:
        closes: List of closing prices (oldest first)
        period: RSI period

    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(closes) < period + 1:
        return None

    # Calculate price changes
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Initial average gain/loss (first 'period' changes)
    gains = [max(0, change) for change in changes[:period]]
    losses = [max(0, -change) for change in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder's smoothing for remaining periods
    for change in changes[period:]:
        gain = max(0, change)
        loss = max(0, -change)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_atr(candles, period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range.

    Args:
        candles: Tuple/list of candle objects with high, low, close
        period: ATR period

    Returns:
        ATR value or None if insufficient data
    """
    if len(candles) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    # Simple average of recent true ranges
    return sum(true_ranges[-period:]) / period


def calculate_adx(candles, period: int = 14) -> Optional[float]:
    """
    Calculate Average Directional Index for trend strength.

    ADX measures trend strength (not direction):
    - ADX < 20: Weak trend / ranging
    - ADX 20-25: Trend developing
    - ADX 25-50: Strong trend
    - ADX > 50: Very strong trend

    Args:
        candles: Tuple/list of candle objects
        period: ADX period

    Returns:
        ADX value (0-100) or None if insufficient data
    """
    if len(candles) < period * 2:
        return None

    # Calculate True Range, +DM, -DM
    true_ranges = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_high = candles[i - 1].high
        prev_low = candles[i - 1].low
        prev_close = candles[i - 1].close

        # True Range
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0)

        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0)

    if len(true_ranges) < period:
        return None

    # Wilder's smoothing
    def wilder_smooth(values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return []
        smoothed = [sum(values[:period]) / period]
        for i in range(period, len(values)):
            smoothed.append((smoothed[-1] * (period - 1) + values[i]) / period)
        return smoothed

    atr = wilder_smooth(true_ranges, period)
    smooth_plus_dm = wilder_smooth(plus_dm, period)
    smooth_minus_dm = wilder_smooth(minus_dm, period)

    if not atr or not smooth_plus_dm or not smooth_minus_dm:
        return None

    # Calculate +DI and -DI
    dx_values = []
    for i in range(len(atr)):
        if atr[i] == 0:
            plus_di = 0
            minus_di = 0
        else:
            plus_di = 100 * smooth_plus_dm[i] / atr[i]
            minus_di = 100 * smooth_minus_dm[i] / atr[i]

        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_values.append(0)
        else:
            dx_values.append(100 * abs(plus_di - minus_di) / di_sum)

    if len(dx_values) < period:
        return None

    # Calculate ADX (smoothed DX)
    adx_values = wilder_smooth(dx_values, period)

    if not adx_values:
        return None

    return adx_values[-1]


def get_adaptive_rsi_zones(
    current_atr: Optional[float],
    current_price: float,
    config: Dict[str, Any],
    symbol: str
) -> Tuple[float, float]:
    """
    Get RSI zones adjusted by volatility.

    During volatile markets, RSI thresholds are expanded to reduce
    false signals from sustained overbought/oversold conditions.

    Args:
        current_atr: Current ATR value
        current_price: Current market price
        config: Strategy configuration
        symbol: Trading symbol

    Returns:
        Tuple of (oversold_threshold, overbought_threshold)
    """
    base_oversold = get_symbol_config(symbol, config, 'rsi_oversold')
    base_overbought = get_symbol_config(symbol, config, 'rsi_overbought')

    if not config.get('use_adaptive_rsi', True):
        return base_oversold, base_overbought

    if current_atr is None or current_atr <= 0 or current_price <= 0:
        return base_oversold, base_overbought

    # Calculate ATR as percentage of price
    atr_pct = (current_atr / current_price) * 100
    zone_expansion_limit = config.get('rsi_zone_expansion', 5)

    # Expand zones based on volatility
    expansion = min(zone_expansion_limit, atr_pct * 2)

    adaptive_oversold = max(15, base_oversold - expansion)
    adaptive_overbought = min(85, base_overbought + expansion)

    return adaptive_oversold, adaptive_overbought


def classify_rsi_zone(
    rsi: float,
    oversold: float,
    overbought: float
) -> RSIZone:
    """
    Classify RSI into zone.

    Args:
        rsi: Current RSI value
        oversold: Oversold threshold
        overbought: Overbought threshold

    Returns:
        RSIZone enum value
    """
    if rsi < oversold:
        return RSIZone.OVERSOLD
    elif rsi > overbought:
        return RSIZone.OVERBOUGHT
    else:
        return RSIZone.NEUTRAL


def calculate_rsi_confidence(
    side: str,
    rsi: float,
    oversold: float,
    overbought: float,
    config: Dict[str, Any]
) -> Tuple[float, str]:
    """
    Calculate confidence based on RSI position.

    From legacy RSIMeanReversionGrid implementation:
    - RSI in extreme zone: 0.7-1.0 confidence
    - RSI approaching zone: 0.5-0.8 confidence
    - RSI neutral: 0.2-0.4 confidence

    Args:
        side: 'buy' or 'sell'
        rsi: Current RSI value
        oversold: Oversold threshold
        overbought: Overbought threshold
        config: Strategy configuration

    Returns:
        Tuple of (confidence value 0-1, confidence reason)
    """
    if side == 'buy':
        if rsi < oversold:
            # Deep oversold - highest confidence
            boost = min(0.4, (oversold - rsi) / 50)
            confidence = min(1.0, 0.7 + boost)
            reason = f"deep_oversold (RSI={rsi:.1f}<{oversold})"
        elif rsi < oversold + 15:
            # Approaching oversold - moderate confidence
            boost = (oversold + 15 - rsi) / 30 * 0.3
            confidence = min(1.0, 0.5 + boost)
            reason = f"approaching_oversold (RSI={rsi:.1f})"
        else:
            # Neutral - low confidence
            confidence = max(0.2, 0.5 - 0.1)
            reason = f"neutral_rsi (RSI={rsi:.1f})"
    else:  # sell
        if rsi > overbought:
            # Deep overbought - highest confidence
            boost = min(0.4, (rsi - overbought) / 50)
            confidence = min(1.0, 0.7 + boost)
            reason = f"deep_overbought (RSI={rsi:.1f}>{overbought})"
        elif rsi > overbought - 15:
            # Approaching overbought - moderate confidence
            boost = (rsi - (overbought - 15)) / 30 * 0.3
            confidence = min(1.0, 0.5 + boost)
            reason = f"approaching_overbought (RSI={rsi:.1f})"
        else:
            # Neutral - low confidence
            confidence = max(0.2, 0.5 - 0.1)
            reason = f"neutral_rsi (RSI={rsi:.1f})"

    return confidence, reason


def calculate_position_size_multiplier(
    rsi: float,
    oversold: float,
    overbought: float,
    config: Dict[str, Any]
) -> float:
    """
    Calculate position size multiplier based on RSI.

    From legacy code: RSI extreme positions warrant larger sizes
    as mean reversion probability is higher.

    Args:
        rsi: Current RSI value
        oversold: Oversold threshold
        overbought: Overbought threshold
        config: Strategy configuration

    Returns:
        Position size multiplier (1.0 = base size)
    """
    extreme_multiplier = config.get('rsi_extreme_multiplier', 1.3)

    if rsi < oversold:
        # Oversold - larger buy sizes
        return extreme_multiplier
    elif rsi < oversold + 10:
        # Approaching oversold
        return 1.0 + (extreme_multiplier - 1.0) * 0.5
    elif rsi > overbought:
        # Overbought - larger sell sizes
        return extreme_multiplier
    elif rsi > overbought - 10:
        # Approaching overbought
        return 1.0 + (extreme_multiplier - 1.0) * 0.5
    else:
        return 1.0


def calculate_trade_flow(
    trades: tuple,
    lookback: int = 50
) -> Tuple[float, float, float]:
    """
    Calculate trade flow metrics from recent trades.

    REC-003: Trade flow confirmation to avoid entering against market momentum.

    Args:
        trades: Tuple of Trade objects with side, value attributes
        lookback: Number of trades to analyze

    Returns:
        Tuple of (buy_volume, sell_volume, flow_imbalance)
        flow_imbalance: -1 to +1 where positive = more buy volume
    """
    if not trades or len(trades) == 0:
        return 0.0, 0.0, 0.0

    recent_trades = trades[-lookback:] if len(trades) > lookback else trades

    buy_volume = 0.0
    sell_volume = 0.0

    for trade in recent_trades:
        # Handle both object and dict access
        if hasattr(trade, 'side'):
            side = trade.side
            value = getattr(trade, 'value', getattr(trade, 'size', 0) * getattr(trade, 'price', 0))
        else:
            side = trade.get('side', '')
            value = trade.get('value', trade.get('size', 0) * trade.get('price', 0))

        if side == 'buy':
            buy_volume += value
        elif side == 'sell':
            sell_volume += value

    total_volume = buy_volume + sell_volume
    if total_volume == 0:
        return 0.0, 0.0, 0.0

    # Flow imbalance: (buy - sell) / total, ranges from -1 to +1
    flow_imbalance = (buy_volume - sell_volume) / total_volume

    return buy_volume, sell_volume, flow_imbalance


def calculate_volume_ratio(
    candles,
    lookback: int = 20
) -> float:
    """
    Calculate current volume vs average volume ratio.

    REC-003: Volume confirmation for trade entries.

    Args:
        candles: Tuple of candle objects with volume attribute
        lookback: Number of candles for average calculation

    Returns:
        Volume ratio (current / average). >1 = above average, <1 = below average
    """
    if not candles or len(candles) < 2:
        return 1.0

    recent = candles[-lookback:] if len(candles) > lookback else candles

    volumes = []
    for candle in recent:
        vol = getattr(candle, 'volume', 0)
        if vol > 0:
            volumes.append(vol)

    if not volumes:
        return 1.0

    avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
    current_volume = volumes[-1] if volumes else 0

    if avg_volume == 0:
        return 1.0

    return current_volume / avg_volume


def check_liquidity_threshold(
    volume_24h: float,
    min_volume_usd: float
) -> Tuple[bool, str]:
    """
    Check if market liquidity meets minimum threshold.

    REC-006: Liquidity validation especially for XRP/BTC.

    Args:
        volume_24h: 24-hour trading volume in USD
        min_volume_usd: Minimum required volume

    Returns:
        Tuple of (is_sufficient, reason)
    """
    if min_volume_usd <= 0:
        return True, "liquidity_check_disabled"

    if volume_24h >= min_volume_usd:
        ratio = volume_24h / min_volume_usd
        return True, f"liquidity_ok (ratio={ratio:.2f}x)"
    else:
        ratio = volume_24h / min_volume_usd if min_volume_usd > 0 else 0
        return False, f"low_liquidity (ratio={ratio:.2f}x, need={min_volume_usd/1e6:.0f}M)"


def calculate_rolling_correlation(
    prices_a: List[float],
    prices_b: List[float],
    lookback: int = 20
) -> Optional[float]:
    """
    Calculate rolling correlation between two price series.

    REC-005: Real correlation monitoring for cross-pair exposure management.

    Uses Pearson correlation coefficient on returns.

    Args:
        prices_a: List of prices for first symbol
        prices_b: List of prices for second symbol
        lookback: Number of periods for correlation calculation

    Returns:
        Correlation coefficient (-1 to +1) or None if insufficient data
    """
    if len(prices_a) < lookback + 1 or len(prices_b) < lookback + 1:
        return None

    # Use last N prices
    a = prices_a[-(lookback + 1):]
    b = prices_b[-(lookback + 1):]

    # Calculate returns
    returns_a = [(a[i] - a[i-1]) / a[i-1] for i in range(1, len(a)) if a[i-1] != 0]
    returns_b = [(b[i] - b[i-1]) / b[i-1] for i in range(1, len(b)) if b[i-1] != 0]

    if len(returns_a) < 2 or len(returns_b) < 2:
        return None

    # Ensure same length
    min_len = min(len(returns_a), len(returns_b))
    returns_a = returns_a[-min_len:]
    returns_b = returns_b[-min_len:]

    # Calculate means
    mean_a = sum(returns_a) / len(returns_a)
    mean_b = sum(returns_b) / len(returns_b)

    # Calculate covariance and standard deviations
    covariance = sum((ra - mean_a) * (rb - mean_b) for ra, rb in zip(returns_a, returns_b)) / len(returns_a)
    variance_a = sum((r - mean_a) ** 2 for r in returns_a) / len(returns_a)
    variance_b = sum((r - mean_b) ** 2 for r in returns_b) / len(returns_b)

    std_a = variance_a ** 0.5
    std_b = variance_b ** 0.5

    if std_a == 0 or std_b == 0:
        return None

    correlation = covariance / (std_a * std_b)

    # Clamp to valid range
    return max(-1.0, min(1.0, correlation))


def check_trade_flow_confirmation(
    flow_imbalance: float,
    side: str,
    threshold: float = 0.1
) -> Tuple[bool, str]:
    """
    Check if trade flow confirms the intended trade direction.

    REC-003: Only enter when flow confirms direction to reduce adverse selection.

    Args:
        flow_imbalance: Flow imbalance from calculate_trade_flow (-1 to +1)
        side: Intended trade side ('buy' or 'sell')
        threshold: Minimum imbalance to require confirmation (default 0.1)

    Returns:
        Tuple of (is_confirmed, reason)
    """
    # For buys, we want positive or neutral flow (not heavily selling)
    # For sells, we want negative or neutral flow (not heavily buying)

    if side == 'buy':
        if flow_imbalance >= -threshold:
            # Flow is neutral or buying - confirmed
            return True, f"flow_confirmed (imbalance={flow_imbalance:.2f})"
        else:
            # Heavy selling - not confirmed
            return False, f"flow_against_buy (imbalance={flow_imbalance:.2f})"
    else:  # sell
        if flow_imbalance <= threshold:
            # Flow is neutral or selling - confirmed
            return True, f"flow_confirmed (imbalance={flow_imbalance:.2f})"
        else:
            # Heavy buying - not confirmed
            return False, f"flow_against_sell (imbalance={flow_imbalance:.2f})"


def calculate_grid_rr_ratio(
    grid_spacing_pct: float,
    stop_loss_pct: float,
    num_accumulation_levels: int = 1
) -> Tuple[float, str]:
    """
    Calculate Risk:Reward ratio for grid strategy.

    REC-007: Explicit R:R calculation and documentation.

    For grid strategies:
    - Reward = grid_spacing_pct (profit per cycle)
    - Risk = stop_loss_pct (max loss if stopped out)

    For accumulated positions, R:R degrades as more levels are filled.

    Args:
        grid_spacing_pct: Grid spacing as percentage
        stop_loss_pct: Stop loss percentage below lowest grid
        num_accumulation_levels: Number of filled levels (affects effective R:R)

    Returns:
        Tuple of (r:r ratio, description string)
    """
    if stop_loss_pct <= 0:
        return 0.0, "invalid_stop_loss"

    # Base R:R for single grid level
    base_rr = grid_spacing_pct / stop_loss_pct

    # R:R degrades with accumulation (average entry moves closer to stop)
    # Simplified: assume each level is equally spaced, average entry at midpoint
    if num_accumulation_levels > 1:
        # Average entry moves down by (levels-1)/2 * spacing from first entry
        avg_entry_offset = (num_accumulation_levels - 1) / 2 * grid_spacing_pct
        # Effective reward reduced, effective risk increased
        effective_reward = grid_spacing_pct
        effective_risk = stop_loss_pct + avg_entry_offset
        adjusted_rr = effective_reward / effective_risk
    else:
        adjusted_rr = base_rr

    # Generate description
    if adjusted_rr >= 2.0:
        desc = f"excellent ({adjusted_rr:.2f}:1)"
    elif adjusted_rr >= 1.5:
        desc = f"good ({adjusted_rr:.2f}:1)"
    elif adjusted_rr >= 1.0:
        desc = f"acceptable ({adjusted_rr:.2f}:1)"
    else:
        desc = f"poor ({adjusted_rr:.2f}:1) - consider wider spacing or tighter stop"

    return adjusted_rr, desc


def calculate_volatility(candles, lookback: int = 20) -> float:
    """
    Calculate price volatility from candle closes.

    Returns volatility as a percentage (std dev of returns * 100).

    Args:
        candles: Tuple of candle objects with .close attribute
        lookback: Number of candles for calculation

    Returns:
        Volatility percentage
    """
    if len(candles) < lookback + 1:
        return 0.0

    closes = [c.close for c in candles[-(lookback + 1):]]
    if len(closes) < 2:
        return 0.0

    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] != 0]

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    return (variance ** 0.5) * 100
