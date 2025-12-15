"""
Grid RSI Reversion Strategy - Indicator Calculations

Contains strategy-specific indicator functions for grid RSI reversion analysis.
Common indicators are imported from the centralized ws_tester.indicators library.

Technical indicators:
- RSI, ATR, ADX
- Adaptive RSI zones
- RSI confidence calculation from legacy code
"""
from typing import Dict, Any, List, Optional, Tuple

from .config import RSIZone, get_symbol_config

# Import common indicators from centralized library
from ws_tester.indicators import (
    calculate_rsi,
    calculate_atr,
    calculate_adx,
    calculate_volatility,
    calculate_volume_ratio,
    calculate_rolling_correlation,
    check_trade_flow_confirmation as _check_trade_flow_lib,
    calculate_trade_flow as _calculate_trade_flow_lib,
    TradeFlowResult,
)


def calculate_trade_flow(
    trades: tuple,
    lookback: int = 50
) -> Tuple[float, float, float]:
    """
    Calculate trade flow metrics from recent trades.

    REC-003: Trade flow confirmation to avoid entering against market momentum.

    This is a thin wrapper around ws_tester.indicators.calculate_trade_flow
    that returns (buy_volume, sell_volume, flow_imbalance) tuple for backward
    compatibility.

    Args:
        trades: Tuple of Trade objects with side, value attributes
        lookback: Number of trades to analyze

    Returns:
        Tuple of (buy_volume, sell_volume, flow_imbalance)
        flow_imbalance: -1 to +1 where positive = more buy volume
    """
    result = _calculate_trade_flow_lib(trades, lookback)
    return result.buy_volume, result.sell_volume, result.imbalance


def check_trade_flow_confirmation(
    flow_imbalance: float,
    side: str,
    threshold: float = 0.1
) -> Tuple[bool, str]:
    """
    Check if trade flow confirms the intended trade direction.

    REC-003: Only enter when flow confirms direction to reduce adverse selection.

    Uses ws_tester.indicators.check_trade_flow_confirmation with pre-calculated
    imbalance for backward compatibility.

    Args:
        flow_imbalance: Flow imbalance from calculate_trade_flow (-1 to +1)
        side: Intended trade side ('buy' or 'sell')
        threshold: Minimum imbalance to require confirmation (default 0.1)

    Returns:
        Tuple of (is_confirmed, reason)
    """
    # Use library version with pre-calculated imbalance
    direction = 'buy' if side == 'buy' else 'short'
    is_confirmed, flow_data = _check_trade_flow_lib(flow_imbalance, direction, threshold)

    # Generate reason string for backward compatibility
    if is_confirmed:
        reason = f"flow_confirmed (imbalance={flow_imbalance:.2f})"
    else:
        if side == 'buy':
            reason = f"flow_against_buy (imbalance={flow_imbalance:.2f})"
        else:
            reason = f"flow_against_sell (imbalance={flow_imbalance:.2f})"

    return is_confirmed, reason


# Re-export for backward compatibility
__all__ = [
    # From centralized library (re-exported)
    'calculate_rsi',
    'calculate_atr',
    'calculate_adx',
    'calculate_volatility',
    'calculate_volume_ratio',
    'calculate_rolling_correlation',
    # Wrapper functions
    'calculate_trade_flow',
    'check_trade_flow_confirmation',
    # Strategy-specific functions
    'get_adaptive_rsi_zones',
    'classify_rsi_zone',
    'calculate_rsi_confidence',
    'calculate_position_size_multiplier',
    'check_liquidity_threshold',
    'calculate_grid_rr_ratio',
]


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
