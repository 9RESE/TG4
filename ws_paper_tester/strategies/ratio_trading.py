"""
Ratio Trading Strategy

Mean reversion strategy for XRP/BTC pair accumulation.
Trades the XRP/BTC ratio to grow holdings of both assets.

Strategy Logic:
- Calculate moving average of XRP/BTC ratio
- Use Bollinger Bands for entry/exit zones
- Buy when ratio is below lower band (XRP cheap vs BTC)
- Sell when ratio is above upper band (XRP expensive vs BTC)
- Rebalance to maintain balanced holdings

Version History:
- 1.0.0: Initial implementation
         - Mean reversion with Bollinger Bands
         - Dual-asset accumulation tracking
         - Research-based config from Kraken data
"""
from datetime import datetime
from typing import Dict, Any, Optional, List

from ws_tester.types import DataSnapshot, Signal


# =============================================================================
# REQUIRED: Strategy Metadata
# =============================================================================
STRATEGY_NAME = "ratio_trading"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["XRP/BTC"]


# =============================================================================
# REQUIRED: Default Configuration
# =============================================================================
CONFIG = {
    # Mean reversion parameters
    'lookback_periods': 20,        # Periods for moving average
    'bollinger_std': 2.0,          # Standard deviations for bands
    'entry_threshold': 1.0,        # Entry at N std devs from mean
    'exit_threshold': 0.5,         # Exit at N std devs (closer to mean)

    # Position sizing (in XRP)
    'position_size_xrp': 30.0,     # Base size per trade in XRP
    'max_position_xrp': 200.0,     # Maximum XRP exposure

    # Risk management
    'stop_loss_pct': 0.6,          # Stop loss percentage
    'take_profit_pct': 0.5,        # Take profit percentage

    # Cooldown
    'cooldown_seconds': 60.0,      # Minimum time between trades
    'min_candles': 10,             # Minimum candles before trading

    # Rebalancing
    'rebalance_threshold': 0.3,    # Rebalance when holdings differ by 30%
}


# =============================================================================
# Helper Functions
# =============================================================================
def _calculate_bollinger_bands(
    prices: List[float],
    lookback: int = 20,
    num_std: float = 2.0
) -> tuple:
    """
    Calculate Bollinger Bands.

    Returns:
        (sma, upper_band, lower_band, std_dev)
    """
    if len(prices) < lookback:
        return None, None, None, None

    recent = prices[-lookback:]

    # Simple Moving Average
    sma = sum(recent) / len(recent)

    # Standard Deviation
    variance = sum((p - sma) ** 2 for p in recent) / len(recent)
    std_dev = variance ** 0.5

    # Bands
    upper = sma + (num_std * std_dev)
    lower = sma - (num_std * std_dev)

    return sma, upper, lower, std_dev


def _calculate_z_score(price: float, sma: float, std_dev: float) -> float:
    """Calculate z-score (number of std devs from mean)."""
    if std_dev == 0:
        return 0.0
    return (price - sma) / std_dev


# =============================================================================
# REQUIRED: Main Signal Generation Function
# =============================================================================
def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """
    Generate ratio trading signal based on mean reversion.

    Strategy:
    - Track XRP/BTC price history
    - Calculate Bollinger Bands
    - Buy XRP when ratio below lower band (XRP cheap)
    - Sell XRP when ratio above upper band (XRP expensive)
    - Goal: Accumulate both XRP and BTC over time

    Args:
        data: Immutable market data snapshot
        config: Strategy configuration (CONFIG + overrides)
        state: Mutable strategy state dict

    Returns:
        Signal if trading opportunity found, None otherwise
    """
    # Lazy initialization
    if 'initialized' not in state:
        state['initialized'] = True
        state['price_history'] = []
        state['position_xrp'] = 0.0
        state['xrp_accumulated'] = 0.0
        state['btc_accumulated'] = 0.0
        state['last_signal_time'] = None
        state['trade_count'] = 0
        state['indicators'] = {}

    symbol = SYMBOLS[0]  # XRP/BTC
    current_time = data.timestamp

    # Time-based cooldown
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        if elapsed < config.get('cooldown_seconds', 60.0):
            return None

    # Get current price
    price = data.prices.get(symbol)
    if not price:
        return None

    # Update price history from candles or trades
    candles = data.candles_1m.get(symbol, ())
    if candles:
        # Use candle closes for history
        closes = [c.close for c in candles]
        state['price_history'] = closes[-50:]  # Keep last 50
    else:
        # Fall back to current price
        state['price_history'].append(price)
        state['price_history'] = state['price_history'][-50:]

    # Need minimum candles before trading
    if len(state['price_history']) < config.get('min_candles', 10):
        state['indicators'] = {
            'symbol': symbol,
            'price': price,
            'status': 'warming_up',
            'candles': len(state['price_history']),
        }
        return None

    # Calculate Bollinger Bands
    lookback = config.get('lookback_periods', 20)
    num_std = config.get('bollinger_std', 2.0)

    sma, upper, lower, std_dev = _calculate_bollinger_bands(
        state['price_history'],
        lookback,
        num_std
    )

    if sma is None:
        return None

    # Calculate z-score (how many std devs from mean)
    z_score = _calculate_z_score(price, sma, std_dev)

    # Entry/exit thresholds
    entry_threshold = config.get('entry_threshold', 1.0)
    exit_threshold = config.get('exit_threshold', 0.5)

    # Position sizing
    base_size = config.get('position_size_xrp', 30.0)
    max_position = config.get('max_position_xrp', 200.0)
    current_position = state.get('position_xrp', 0)

    # Calculate band widths for indicators
    band_width = (upper - lower) / sma * 100 if sma else 0  # As percentage

    # Store indicators for logging
    state['indicators'] = {
        'symbol': symbol,
        'price': round(price, 8),
        'sma': round(sma, 8),
        'upper_band': round(upper, 8),
        'lower_band': round(lower, 8),
        'std_dev': round(std_dev, 10),
        'z_score': round(z_score, 3),
        'band_width_pct': round(band_width, 4),
        'position_xrp': round(current_position, 2),
        'xrp_accumulated': round(state.get('xrp_accumulated', 0), 4),
        'btc_accumulated': round(state.get('btc_accumulated', 0), 8),
    }

    # Risk management percentages
    tp_pct = config.get('take_profit_pct', 0.5)
    sl_pct = config.get('stop_loss_pct', 0.6)

    signal = None

    # BUY Signal: Price below lower band (XRP cheap vs BTC)
    # Action: Spend BTC to buy XRP
    if z_score < -entry_threshold:
        available = max_position - current_position
        if available >= 5.0:
            trade_size = min(base_size, available)
            signal = Signal(
                action='buy',
                symbol=symbol,
                size=trade_size,
                price=price,
                reason=f"RT: Buy XRP (z={z_score:.2f}, below {-entry_threshold}σ)",
                stop_loss=price * (1 - sl_pct / 100),
                take_profit=sma,  # Target mean reversion
            )

    # SELL Signal: Price above upper band (XRP expensive vs BTC)
    # Action: Sell XRP to get BTC
    elif z_score > entry_threshold:
        # Can sell if we have position OR if we want to short (but we don't short)
        if current_position > 5.0:
            trade_size = min(base_size, current_position)
            signal = Signal(
                action='sell',
                symbol=symbol,
                size=trade_size,
                price=price,
                reason=f"RT: Sell XRP (z={z_score:.2f}, above {entry_threshold}σ)",
                stop_loss=price * (1 + sl_pct / 100),
                take_profit=sma,  # Target mean reversion
            )
        else:
            # No position to sell, but still want to accumulate BTC
            # Sell from starting XRP holdings if available
            trade_size = base_size
            signal = Signal(
                action='sell',
                symbol=symbol,
                size=trade_size,
                price=price,
                reason=f"RT: Sell XRP for BTC (z={z_score:.2f}, ratio expensive)",
                stop_loss=price * (1 + sl_pct / 100),
                take_profit=sma,
            )

    # EXIT/TAKE PROFIT: Position exists and price reverted toward mean
    elif current_position > 5.0 and abs(z_score) < exit_threshold:
        # We're long and price returned to mean - take profit
        trade_size = min(base_size, current_position * 0.5)  # Partial exit
        if trade_size >= 5.0:
            signal = Signal(
                action='sell',
                symbol=symbol,
                size=trade_size,
                price=price,
                reason=f"RT: Take profit (z={z_score:.2f}, near mean)",
            )

    if signal:
        state['last_signal_time'] = current_time
        state['trade_count'] += 1

    return signal


# =============================================================================
# OPTIONAL: Lifecycle Callbacks
# =============================================================================
def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Initialize strategy state on startup."""
    state['initialized'] = True
    state['price_history'] = []
    state['position_xrp'] = 0.0
    state['xrp_accumulated'] = 0.0
    state['btc_accumulated'] = 0.0
    state['last_signal_time'] = None
    state['trade_count'] = 0
    state['indicators'] = {}


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Track fills and update position/accumulation state.

    For XRP/BTC:
    - Buy: Spent BTC to get XRP → +XRP position
    - Sell: Sold XRP to get BTC → -XRP position, +BTC accumulated
    """
    side = fill.get('side', '')
    size = fill.get('size', 0)  # Size in XRP
    price = fill.get('price', 0)  # Price in BTC per XRP

    btc_value = size * price

    if side == 'buy':
        # Bought XRP with BTC
        state['position_xrp'] = state.get('position_xrp', 0) + size
        state['xrp_accumulated'] = state.get('xrp_accumulated', 0) + size
    elif side == 'sell':
        # Sold XRP for BTC
        state['position_xrp'] = state.get('position_xrp', 0) - size
        state['btc_accumulated'] = state.get('btc_accumulated', 0) + btc_value

    # Track fill history
    if 'fills' not in state:
        state['fills'] = []
    state['fills'].append(fill)
    state['fills'] = state['fills'][-20:]  # Keep last 20


def on_stop(state: Dict[str, Any]) -> None:
    """Called when strategy stops. Log final state."""
    state['final_summary'] = {
        'position_xrp': state.get('position_xrp', 0),
        'xrp_accumulated': state.get('xrp_accumulated', 0),
        'btc_accumulated': state.get('btc_accumulated', 0),
        'trade_count': state.get('trade_count', 0),
        'total_fills': len(state.get('fills', [])),
    }
