"""
WaveTrend Oscillator Strategy - Indicator Calculations

Contains functions for calculating WaveTrend oscillator, EMA, SMA,
zone classification, crossover detection, and divergence detection.

WaveTrend Formula (LazyBear):
1. HLC3 = (High + Low + Close) / 3        # Typical Price
2. ESA = EMA(HLC3, channel_length)        # Exponential Smoothed Average
3. D = EMA(|HLC3 - ESA|, channel_length)  # Average Deviation
4. CI = (HLC3 - ESA) / (0.015 * D)        # Channel Index
5. WT1 = EMA(CI, average_length)          # WaveTrend Line 1 (tci)
6. WT2 = SMA(WT1, ma_length)              # WaveTrend Line 2 (Signal)
"""
from typing import List, Dict, Any, Optional, Tuple

from .config import WaveTrendZone, CrossoverType, DivergenceType


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


def calculate_ema_series(values: List[float], period: int) -> List[float]:
    """
    Calculate EMA series for the entire value history.

    Args:
        values: List of values (oldest first)
        period: EMA period

    Returns:
        List of EMA values
    """
    if len(values) < period:
        return []

    # Initialize EMA with SMA for the first 'period' values
    sma = sum(values[:period]) / period
    multiplier = 2.0 / (period + 1)

    ema_series = [sma]
    ema = sma

    for value in values[period:]:
        ema = (value * multiplier) + (ema * (1 - multiplier))
        ema_series.append(ema)

    return ema_series


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


def calculate_sma_series(values: List[float], period: int) -> List[float]:
    """
    Calculate SMA series for the entire value history.

    Args:
        values: List of values (oldest first)
        period: SMA period

    Returns:
        List of SMA values
    """
    if len(values) < period:
        return []

    sma_series = []
    for i in range(period - 1, len(values)):
        sma = sum(values[i - period + 1:i + 1]) / period
        sma_series.append(sma)

    return sma_series


def calculate_wavetrend(
    candles,
    channel_length: int = 10,
    average_length: int = 21,
    ma_length: int = 4
) -> Dict[str, Any]:
    """
    Calculate WaveTrend Oscillator values.

    Formula:
    1. HLC3 = (High + Low + Close) / 3
    2. ESA = EMA(HLC3, channel_length)
    3. D = EMA(|HLC3 - ESA|, channel_length)
    4. CI = (HLC3 - ESA) / (0.015 * D)
    5. WT1 = EMA(CI, average_length)
    6. WT2 = SMA(WT1, ma_length)

    Args:
        candles: Tuple/list of candle objects with high, low, close
        channel_length: ESA and D calculation period (default 10)
        average_length: WT1 smoothing period (default 21)
        ma_length: WT2 signal line smoothing (default 4)

    Returns:
        Dict with wt1, wt2, prev_wt1, prev_wt2, diff, and series
    """
    result = {
        'wt1': None,
        'wt2': None,
        'prev_wt1': None,
        'prev_wt2': None,
        'diff': None,
        'wt1_series': [],
        'wt2_series': [],
    }

    min_candles = max(channel_length, average_length) + ma_length + 5
    if len(candles) < min_candles:
        return result

    # Step 1: Calculate HLC3 (Typical Price)
    hlc3 = [(c.high + c.low + c.close) / 3 for c in candles]

    # Step 2: Calculate ESA - EMA of HLC3
    esa_series = calculate_ema_series(hlc3, channel_length)
    if not esa_series:
        return result

    # Step 3: Calculate D - EMA of absolute deviation
    # Align hlc3 with esa_series (esa_series starts at index channel_length-1)
    offset = channel_length - 1
    deviation = [abs(hlc3[offset + i] - esa_series[i]) for i in range(len(esa_series))]
    d_series = calculate_ema_series(deviation, channel_length)
    if not d_series:
        return result

    # Step 4: Calculate CI - Channel Index
    # Align with d_series
    d_offset = channel_length - 1
    ci = []
    for i in range(len(d_series)):
        # Index in esa_series
        esa_idx = d_offset + i
        if esa_idx >= len(esa_series):
            break
        d_val = d_series[i]
        esa_val = esa_series[esa_idx]
        hlc3_idx = offset + esa_idx
        if hlc3_idx >= len(hlc3):
            break
        hlc3_val = hlc3[hlc3_idx]
        # Avoid division by zero
        ci_val = (hlc3_val - esa_val) / (0.015 * d_val + 1e-10)
        ci.append(ci_val)

    if len(ci) < average_length:
        return result

    # Step 5: Calculate WT1 - EMA of CI
    wt1_series = calculate_ema_series(ci, average_length)
    if not wt1_series or len(wt1_series) < ma_length + 1:
        return result

    # Step 6: Calculate WT2 - SMA of WT1
    wt2_series = calculate_sma_series(wt1_series, ma_length)
    if not wt2_series or len(wt2_series) < 2:
        return result

    # Align wt1_series with wt2_series for comparison
    # wt2_series[i] corresponds to wt1_series[i + ma_length - 1]
    wt1_aligned = wt1_series[ma_length - 1:]

    if len(wt1_aligned) < 2 or len(wt2_series) < 2:
        return result

    result['wt1'] = wt1_aligned[-1]
    result['wt2'] = wt2_series[-1]
    result['prev_wt1'] = wt1_aligned[-2] if len(wt1_aligned) >= 2 else None
    result['prev_wt2'] = wt2_series[-2] if len(wt2_series) >= 2 else None
    result['diff'] = result['wt1'] - result['wt2']
    result['wt1_series'] = wt1_aligned
    result['wt2_series'] = wt2_series

    return result


def classify_zone(wt1: float, config: Dict[str, Any]) -> WaveTrendZone:
    """
    Determine current WaveTrend zone.

    Args:
        wt1: Current WT1 value
        config: Configuration with zone thresholds

    Returns:
        WaveTrendZone enum value
    """
    extreme_ob = config.get('wt_extreme_overbought', 80)
    overbought = config.get('wt_overbought', 60)
    oversold = config.get('wt_oversold', -60)
    extreme_os = config.get('wt_extreme_oversold', -80)

    if wt1 >= extreme_ob:
        return WaveTrendZone.EXTREME_OVERBOUGHT
    elif wt1 >= overbought:
        return WaveTrendZone.OVERBOUGHT
    elif wt1 <= extreme_os:
        return WaveTrendZone.EXTREME_OVERSOLD
    elif wt1 <= oversold:
        return WaveTrendZone.OVERSOLD
    return WaveTrendZone.NEUTRAL


def get_zone_string(zone: WaveTrendZone) -> str:
    """Convert WaveTrendZone to string representation."""
    zone_strings = {
        WaveTrendZone.EXTREME_OVERBOUGHT: 'extreme_overbought',
        WaveTrendZone.OVERBOUGHT: 'overbought',
        WaveTrendZone.NEUTRAL: 'neutral',
        WaveTrendZone.OVERSOLD: 'oversold',
        WaveTrendZone.EXTREME_OVERSOLD: 'extreme_oversold',
    }
    return zone_strings.get(zone, 'unknown')


def detect_crossover(
    wt1: float,
    wt2: float,
    prev_wt1: Optional[float],
    prev_wt2: Optional[float]
) -> CrossoverType:
    """
    Detect WT1/WT2 crossover.

    Args:
        wt1: Current WT1 value
        wt2: Current WT2 value
        prev_wt1: Previous WT1 value
        prev_wt2: Previous WT2 value

    Returns:
        CrossoverType enum value
    """
    if prev_wt1 is None or prev_wt2 is None:
        return CrossoverType.NONE

    # Bullish: WT1 crosses above WT2
    if wt1 > wt2 and prev_wt1 <= prev_wt2:
        return CrossoverType.BULLISH

    # Bearish: WT1 crosses below WT2
    if wt1 < wt2 and prev_wt1 >= prev_wt2:
        return CrossoverType.BEARISH

    return CrossoverType.NONE


def detect_divergence(
    closes: List[float],
    wt1_series: List[float],
    lookback: int,
    oversold: float,
    overbought: float,
    current_wt: float
) -> DivergenceType:
    """
    Detect price/WaveTrend divergence.

    Bullish divergence: Price makes lower low + WaveTrend makes higher low
    Bearish divergence: Price makes higher high + WaveTrend makes lower high

    Args:
        closes: List of closing prices
        wt1_series: List of WT1 values
        lookback: Number of candles for comparison
        oversold: Oversold threshold
        overbought: Overbought threshold
        current_wt: Current WT1 value for zone check

    Returns:
        DivergenceType enum value
    """
    n = lookback
    min_data = n * 2 + 5

    if len(closes) < min_data or len(wt1_series) < min_data:
        return DivergenceType.NONE

    # Recent vs Prior period comparison
    recent_close = closes[-n:]
    prior_close = closes[-2*n:-n]
    recent_wt = wt1_series[-n:]
    prior_wt = wt1_series[-2*n:-n]

    # Find swing points
    recent_close_min = min(recent_close)
    recent_close_max = max(recent_close)
    prior_close_min = min(prior_close)
    prior_close_max = max(prior_close)

    recent_wt_min = min(recent_wt)
    recent_wt_max = max(recent_wt)
    prior_wt_min = min(prior_wt)
    prior_wt_max = max(prior_wt)

    # Bullish divergence: Price lower low, WT higher low
    if recent_close_min < prior_close_min and recent_wt_min > prior_wt_min:
        # More significant in oversold zone
        if current_wt < oversold:
            return DivergenceType.BULLISH

    # Bearish divergence: Price higher high, WT lower high
    if recent_close_max > prior_close_max and recent_wt_max < prior_wt_max:
        # More significant in overbought zone
        if current_wt > overbought:
            return DivergenceType.BEARISH

    return DivergenceType.NONE


def is_in_oversold_zone(zone: WaveTrendZone) -> bool:
    """Check if zone is oversold or extreme oversold."""
    return zone in (WaveTrendZone.OVERSOLD, WaveTrendZone.EXTREME_OVERSOLD)


def is_in_overbought_zone(zone: WaveTrendZone) -> bool:
    """Check if zone is overbought or extreme overbought."""
    return zone in (WaveTrendZone.OVERBOUGHT, WaveTrendZone.EXTREME_OVERBOUGHT)


def is_extreme_zone(zone: WaveTrendZone) -> bool:
    """Check if zone is extreme (either direction)."""
    return zone in (WaveTrendZone.EXTREME_OVERBOUGHT, WaveTrendZone.EXTREME_OVERSOLD)


def calculate_confidence(
    crossover: CrossoverType,
    current_zone: WaveTrendZone,
    prev_zone: WaveTrendZone,
    divergence: DivergenceType,
    config: Dict[str, Any]
) -> Tuple[float, List[str]]:
    """
    Calculate entry confidence based on WaveTrend conditions.

    Args:
        crossover: Type of crossover detected
        current_zone: Current WaveTrend zone
        prev_zone: Previous WaveTrend zone
        divergence: Type of divergence detected
        config: Strategy configuration

    Returns:
        Tuple of (confidence value, list of reasons)
    """
    reasons = []

    if crossover == CrossoverType.BULLISH:
        confidence = 0.55  # Base confidence for bullish crossover
        reasons.append("WT bullish crossover")

        # Zone-based confidence boost
        if is_in_oversold_zone(prev_zone):
            confidence = 0.75
            reasons.append(f"from {get_zone_string(prev_zone)} zone")
        elif is_in_oversold_zone(current_zone):
            confidence = 0.70
            reasons.append(f"in {get_zone_string(current_zone)} zone")

        # Divergence bonus
        if divergence == DivergenceType.BULLISH:
            confidence += 0.10
            reasons.append("bullish divergence")

        # Extreme zone bonus
        if is_extreme_zone(prev_zone):
            confidence += 0.05
            reasons.append("from extreme zone")

        # Cap at max long confidence
        max_conf = config.get('max_long_confidence', 0.92)
        confidence = min(confidence, max_conf)

    elif crossover == CrossoverType.BEARISH:
        confidence = 0.55  # Base confidence for bearish crossover
        reasons.append("WT bearish crossover")

        # Zone-based confidence boost
        if is_in_overbought_zone(prev_zone):
            confidence = 0.70
            reasons.append(f"from {get_zone_string(prev_zone)} zone")
        elif is_in_overbought_zone(current_zone):
            confidence = 0.65
            reasons.append(f"in {get_zone_string(current_zone)} zone")

        # Divergence bonus
        if divergence == DivergenceType.BEARISH:
            confidence += 0.10
            reasons.append("bearish divergence")

        # Extreme zone bonus
        if is_extreme_zone(prev_zone):
            confidence += 0.05
            reasons.append("from extreme zone")

        # Cap at max short confidence (lower than long)
        max_conf = config.get('max_short_confidence', 0.88)
        confidence = min(confidence, max_conf)

    else:
        confidence = 0.0
        reasons.append("no crossover")

    return confidence, reasons
