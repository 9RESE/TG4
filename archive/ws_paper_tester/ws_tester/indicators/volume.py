"""
Volume Calculations

Contains volume ratio, volume spike detection, micro-price, and VPIN.

Source implementations:
- Volume ratio: grid_rsi_reversion/indicators.py:407-443, momentum_scalping/indicators.py:313-342
- Volume spike: momentum_scalping/indicators.py:345-373
- Micro-price: order_flow/indicators.py:36-58, market_making/calculations.py:26-46
- VPIN: order_flow/indicators.py:55-143
"""
from typing import List, Tuple, Any

from ._types import PriceInput, extract_volumes
from ws_tester.types import OrderbookSnapshot, Trade


def calculate_volume_ratio(data: PriceInput, lookback: int = 20) -> float:
    """
    Calculate current volume ratio vs rolling average.

    Volume Ratio = Current Volume / SMA(Volume, lookback)

    Args:
        data: Candle data with volume attribute
        lookback: Lookback period for average calculation

    Returns:
        Volume ratio (1.0 = average, >1.0 = above average, <1.0 = below average)
        Returns 1.0 if insufficient data or no volume data
    """
    volumes = extract_volumes(data)

    if len(volumes) < lookback or not volumes:
        return 1.0

    # Calculate rolling average (excluding current)
    avg_volume = sum(volumes[-lookback:]) / lookback

    if avg_volume <= 0:
        return 1.0

    # Current volume is the last value
    current_volume = volumes[-1]

    return current_volume / avg_volume


def calculate_volume_spike(
    volumes: List[float],
    lookback: int = 20,
    recent_count: int = 3
) -> float:
    """
    Calculate volume spike ratio for recent candles.

    Compares average of recent candles to historical average.

    Args:
        volumes: List of volume values
        lookback: Lookback period for historical average
        recent_count: Number of recent candles to average

    Returns:
        Volume spike ratio
    """
    if len(volumes) < lookback + recent_count:
        return 1.0

    # Rolling average excluding recent candles
    avg_volume = sum(volumes[-(lookback + recent_count):-recent_count]) / lookback

    if avg_volume <= 0:
        return 1.0

    # Average of recent volumes
    recent_avg = sum(volumes[-recent_count:]) / recent_count

    return recent_avg / avg_volume


def calculate_micro_price(ob: OrderbookSnapshot) -> float:
    """
    Calculate volume-weighted micro-price.

    Micro-price provides better price discovery than simple mid-price
    by weighting by order sizes at best bid/ask.

    Formula: micro = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)

    Args:
        ob: OrderbookSnapshot with bids and asks

    Returns:
        Micro-price or mid-price if calculation fails
    """
    if not ob or not ob.bids or not ob.asks:
        return 0.0

    best_bid, bid_size = ob.bids[0]
    best_ask, ask_size = ob.asks[0]

    total_size = bid_size + ask_size
    if total_size <= 0:
        # Fallback to mid price
        return ob.mid if hasattr(ob, 'mid') else (best_bid + best_ask) / 2

    return (best_bid * ask_size + best_ask * bid_size) / total_size


def calculate_vpin(trades: Tuple[Trade, ...], bucket_count: int = 50) -> float:
    """
    Calculate VPIN (Volume-Synchronized Probability of Informed Trading).

    VPIN measures order flow toxicity by dividing trades into equal-volume
    buckets and measuring the imbalance in each.

    Formula:
    - Divide trades into equal-volume buckets
    - For each bucket, calculate |buy_volume - sell_volume| / total_volume
    - VPIN = average of bucket imbalances

    Args:
        trades: Tuple of Trade objects with size and side attributes
        bucket_count: Number of volume buckets to use

    Returns:
        VPIN value between 0 and 1 (higher = more informed trading)
    """
    if len(trades) < bucket_count:
        return 0.0

    # Calculate total volume and bucket size
    total_volume = sum(t.size for t in trades)
    if total_volume <= 0:
        return 0.0

    bucket_volume = total_volume / bucket_count

    # Build volume buckets with improved overflow handling
    buckets = []
    current_bucket_buy = 0.0
    current_bucket_sell = 0.0
    cumulative_volume = 0.0
    bucket_boundary = bucket_volume

    for trade in trades:
        trade_volume = trade.size
        trade_buy = trade_volume if trade.side == 'buy' else 0.0
        trade_sell = trade_volume if trade.side == 'sell' else 0.0

        # Handle trade that may span multiple buckets
        remaining_buy = trade_buy
        remaining_sell = trade_sell
        remaining_volume = trade_volume

        while remaining_volume > 0 and len(buckets) < bucket_count:
            # How much volume fits in current bucket
            space_in_bucket = bucket_boundary - cumulative_volume
            volume_for_bucket = min(remaining_volume, space_in_bucket)

            if remaining_volume > 0:
                # Proportionally split buy/sell based on trade composition
                proportion = volume_for_bucket / remaining_volume
                buy_portion = remaining_buy * proportion
                sell_portion = remaining_sell * proportion

                current_bucket_buy += buy_portion
                current_bucket_sell += sell_portion
                cumulative_volume += volume_for_bucket

                remaining_buy -= buy_portion
                remaining_sell -= sell_portion
                remaining_volume -= volume_for_bucket

            # Check if bucket is complete
            if cumulative_volume >= bucket_boundary - 1e-10:
                bucket_total = current_bucket_buy + current_bucket_sell
                if bucket_total > 0:
                    bucket_imbalance = abs(current_bucket_buy - current_bucket_sell) / bucket_total
                    buckets.append(bucket_imbalance)

                # Reset for next bucket
                current_bucket_buy = 0.0
                current_bucket_sell = 0.0
                bucket_boundary += bucket_volume

    # Handle final partial bucket if it has meaningful volume
    if current_bucket_buy + current_bucket_sell > bucket_volume * 0.5:
        bucket_total = current_bucket_buy + current_bucket_sell
        if bucket_total > 0:
            bucket_imbalance = abs(current_bucket_buy - current_bucket_sell) / bucket_total
            buckets.append(bucket_imbalance)

    if not buckets:
        return 0.0

    # VPIN is the average bucket imbalance
    return sum(buckets) / len(buckets)
