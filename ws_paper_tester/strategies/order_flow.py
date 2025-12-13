"""
Order Flow Strategy
Trades based on trade tape analysis and buy/sell imbalance.
"""

from typing import Optional

# Note: Types are imported by the strategy loader which handles the import path
try:
    from ws_tester.types import DataSnapshot, Signal
except ImportError:
    DataSnapshot = None
    Signal = None


# Strategy metadata (required)
STRATEGY_NAME = "order_flow"
STRATEGY_VERSION = "1.0.1"
SYMBOLS = ["XRP/USD", "BTC/USD"]

# Configuration with defaults
CONFIG = {
    'imbalance_threshold': 0.3,    # Min imbalance to trigger (0.3 = 30%)
    'volume_spike_mult': 2.0,      # Volume spike multiplier
    'position_size_usd': 25,       # Size per trade in USD
    'lookback_trades': 50,         # Number of trades to analyze
    'cooldown_trades': 10,         # Min trades between signals
    'take_profit_pct': 0.5,        # Take profit at 0.5%
    'stop_loss_pct': 0.3,          # Stop loss at 0.3%
}


def generate_signal(data, config: dict, state: dict):
    """
    Generate order flow signal.

    Strategy:
    - Analyze recent trades for buy/sell imbalance
    - Look for volume spikes
    - Trade in direction of order flow
    """
    from ws_tester.types import Signal

    # Initialize state
    if 'last_signal_idx' not in state:
        state['last_signal_idx'] = 0
        state['total_trades_seen'] = 0

    # Try each symbol
    for symbol in SYMBOLS:
        trades = data.trades.get(symbol, ())
        if len(trades) < config['lookback_trades']:
            continue

        recent_trades = trades[:config['lookback_trades']]

        # Count trades since last signal
        state['total_trades_seen'] = len(trades)
        trades_since_signal = state['total_trades_seen'] - state['last_signal_idx']

        # Cooldown check
        if trades_since_signal < config['cooldown_trades']:
            continue

        # Calculate buy/sell imbalance
        buy_volume = sum(t.size for t in recent_trades if t.side == 'buy')
        sell_volume = sum(t.size for t in recent_trades if t.side == 'sell')
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            continue

        imbalance = (buy_volume - sell_volume) / total_volume  # -1 to 1

        # Calculate average volume
        avg_trade_size = total_volume / len(recent_trades)

        # Check for volume spike in last few trades
        last_5_volume = sum(t.size for t in recent_trades[:5])
        expected_5_volume = avg_trade_size * 5
        volume_spike = last_5_volume / expected_5_volume if expected_5_volume > 0 else 1.0

        # Calculate VWAP
        vwap = data.get_vwap(symbol, config['lookback_trades'])
        current_price = data.prices.get(symbol, 0)

        if not current_price or not vwap:
            continue

        price_vs_vwap = (current_price - vwap) / vwap  # Positive = above VWAP

        # Store indicators
        state['indicators'] = {
            'symbol': symbol,
            'imbalance': imbalance,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'volume_spike': volume_spike,
            'vwap': vwap,
            'price_vs_vwap': price_vs_vwap,
        }

        # Generate signal
        signal = None

        # Strong buy pressure with volume spike
        if imbalance > config['imbalance_threshold'] and volume_spike > config['volume_spike_mult']:
            signal = Signal(
                action='buy',
                symbol=symbol,
                size=config['position_size_usd'],
                price=current_price,
                reason=f"OF: Buy pressure (imbal={imbalance:.2f}, vol_spike={volume_spike:.1f}x)",
                stop_loss=current_price * (1 - config['stop_loss_pct'] / 100),
                take_profit=current_price * (1 + config['take_profit_pct'] / 100),
            )

        # Strong sell pressure with volume spike
        # HIGH-008 Fix: Use 'short' action instead of 'sell' when opening a new short position
        # 'sell' should only be used when we have a long position to close
        elif imbalance < -config['imbalance_threshold'] and volume_spike > config['volume_spike_mult']:
            # Check if we have a long position to sell, otherwise go short
            has_long_position = state.get('position_side') == 'long' and state.get('position_size', 0) > 0

            if has_long_position:
                # Sell existing long position
                signal = Signal(
                    action='sell',
                    symbol=symbol,
                    size=min(config['position_size_usd'], state.get('position_size', config['position_size_usd'])),
                    price=current_price,
                    reason=f"OF: Close long on sell pressure (imbal={imbalance:.2f}, vol_spike={volume_spike:.1f}x)",
                    # For closing a long, stop_loss protects against price going down further
                    stop_loss=current_price * (1 - config['stop_loss_pct'] / 100),
                    take_profit=current_price * (1 + config['take_profit_pct'] / 100),
                )
            else:
                # Open a short position
                signal = Signal(
                    action='short',
                    symbol=symbol,
                    size=config['position_size_usd'],
                    price=current_price,
                    reason=f"OF: Short on sell pressure (imbal={imbalance:.2f}, vol_spike={volume_spike:.1f}x)",
                    # For shorts, stop_loss is above entry (price going up is bad)
                    stop_loss=current_price * (1 + config['stop_loss_pct'] / 100),
                    # take_profit is below entry (price going down is good)
                    take_profit=current_price * (1 - config['take_profit_pct'] / 100),
                )

        # Buy pressure + price below VWAP (potential mean reversion)
        elif imbalance > config['imbalance_threshold'] * 0.7 and price_vs_vwap < -0.001:
            signal = Signal(
                action='buy',
                symbol=symbol,
                size=config['position_size_usd'] * 0.75,  # Smaller size
                price=current_price,
                reason=f"OF: Buy flow below VWAP (imbal={imbalance:.2f}, vs_vwap={price_vs_vwap:.4f})",
                stop_loss=current_price * (1 - config['stop_loss_pct'] / 100),
                take_profit=vwap,  # Target VWAP
            )

        if signal:
            state['last_signal_idx'] = state['total_trades_seen']
            return signal

    return None


def on_fill(fill: dict, state: dict) -> None:
    """Track fills and position state. HIGH-008 fix: Add position awareness."""
    if 'fills' not in state:
        state['fills'] = []
    state['fills'].append(fill)
    state['fills'] = state['fills'][-20:]  # Keep last 20

    # Track position state for proper sell/short decision
    side = fill.get('side', '')
    size_usd = fill.get('size', 0) * fill.get('price', 0)

    if side == 'buy':
        # Opening or adding to long position
        state['position_side'] = 'long'
        state['position_size'] = state.get('position_size', 0) + size_usd
    elif side == 'sell':
        # Closing long position
        state['position_size'] = max(0, state.get('position_size', 0) - size_usd)
        if state['position_size'] == 0:
            state['position_side'] = None
    elif side == 'short':
        # Opening or adding to short position
        state['position_side'] = 'short'
        state['position_size'] = state.get('position_size', 0) + size_usd
    elif side == 'cover':
        # Closing short position
        state['position_size'] = max(0, state.get('position_size', 0) - size_usd)
        if state['position_size'] == 0:
            state['position_side'] = None


def on_start(config: dict, state: dict) -> None:
    """Initialize state."""
    state['last_signal_idx'] = 0
    state['total_trades_seen'] = 0
    state['fills'] = []
    state['indicators'] = {}
    state['position_side'] = None  # 'long', 'short', or None
    state['position_size'] = 0
