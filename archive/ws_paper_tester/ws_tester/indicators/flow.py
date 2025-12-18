"""
Trade Flow Calculations

Contains trade flow analysis and flow confirmation logic.

Source implementations:
- Trade flow: grid_rsi_reversion/indicators.py:358-404, wavetrend/indicators.py:465-522
- Flow confirmation: grid_rsi_reversion/indicators.py:533-580, whale_sentiment/indicators.py:522-581
"""
from typing import Dict, Any, Tuple, Union, Sequence

from ._types import TradeFlowResult
from ws_tester.types import Trade


def calculate_trade_flow(
    trades: Union[Tuple[Trade, ...], Sequence[Trade], Sequence[Dict[str, Any]]],
    lookback: int = 50
) -> TradeFlowResult:
    """
    Calculate buy/sell volume and imbalance from recent trades.

    Args:
        trades: Tuple/list of trade objects with side and value/size/price attributes
        lookback: Number of recent trades to analyze

    Returns:
        TradeFlowResult with buy_volume, sell_volume, imbalance, total_volume, trade_count, valid
        - imbalance: Range -1 (all sells) to +1 (all buys)
    """
    if not trades or len(trades) == 0:
        return TradeFlowResult(
            buy_volume=0.0,
            sell_volume=0.0,
            imbalance=0.0,
            total_volume=0.0,
            trade_count=0,
            valid=False
        )

    # Get recent trades
    recent_trades = trades[-lookback:] if len(trades) > lookback else trades
    trade_count = len(recent_trades)

    if trade_count < 5:  # Need minimum trades for meaningful analysis
        return TradeFlowResult(
            buy_volume=0.0,
            sell_volume=0.0,
            imbalance=0.0,
            total_volume=0.0,
            trade_count=trade_count,
            valid=False
        )

    buy_volume = 0.0
    sell_volume = 0.0

    # Aggregate buy/sell volumes
    for trade in recent_trades:
        # Handle different trade object formats
        if hasattr(trade, 'side'):
            side = trade.side
            # Try value property first, then calculate from size * price
            if hasattr(trade, 'value'):
                value = trade.value
            else:
                size = getattr(trade, 'size', 0)
                price = getattr(trade, 'price', 1)
                value = size * price
        elif isinstance(trade, dict):
            side = trade.get('side', '')
            value = trade.get('value', trade.get('size', 0) * trade.get('price', 1))
        else:
            continue

        if side == 'buy':
            buy_volume += value
        elif side == 'sell':
            sell_volume += value

    total_volume = buy_volume + sell_volume

    # Calculate imbalance: (buy - sell) / total
    if total_volume > 0:
        imbalance = (buy_volume - sell_volume) / total_volume
        valid = True
    else:
        imbalance = 0.0
        valid = False

    return TradeFlowResult(
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        imbalance=imbalance,
        total_volume=total_volume,
        trade_count=trade_count,
        valid=valid
    )


def check_trade_flow_confirmation(
    trades_or_imbalance: Union[Tuple[Trade, ...], Sequence[Trade], Sequence[Dict[str, Any]], float],
    direction: str,
    threshold: float = 0.10,
    lookback: int = 50
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if trade flow supports the signal direction.

    For momentum strategies: requires flow alignment with signal
    For contrarian strategies: accepts mild opposing flow

    The threshold determines how much opposing flow is acceptable:
    - BUY signals: Accept if imbalance >= -threshold (allows mild selling)
    - SHORT signals: Accept if imbalance <= +threshold (allows mild buying)

    Args:
        trades_or_imbalance: Either trades tuple or pre-calculated imbalance float
        direction: Signal direction ('buy' or 'short')
        threshold: Maximum opposing imbalance to accept (default 0.10 = 10%)
        lookback: Number of recent trades to analyze (if trades tuple provided)

    Returns:
        Tuple of (is_confirmed, flow_data_dict)
    """
    # Handle pre-calculated imbalance
    if isinstance(trades_or_imbalance, (int, float)):
        imbalance = float(trades_or_imbalance)
        flow_data = {
            'imbalance': imbalance,
            'valid': True,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'total_volume': 0.0,
            'trade_count': 0,
        }
    else:
        # Calculate from trades
        flow_result = calculate_trade_flow(trades_or_imbalance, lookback)
        flow_data = {
            'imbalance': flow_result.imbalance,
            'valid': flow_result.valid,
            'buy_volume': flow_result.buy_volume,
            'sell_volume': flow_result.sell_volume,
            'total_volume': flow_result.total_volume,
            'trade_count': flow_result.trade_count,
        }

        if not flow_result.valid:
            # If we can't calculate trade flow, don't block the signal
            flow_data['confirms_signal'] = True
            return True, flow_data

        imbalance = flow_result.imbalance

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
        flow_data['direction_wanted'] = 'unknown'

    return is_confirmed, flow_data
