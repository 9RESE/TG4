"""
Mean Reversion Strategy
Trades price deviations from moving average and VWAP.
"""

from typing import Optional, List
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.types import DataSnapshot, Signal, Candle


# Strategy metadata (required)
STRATEGY_NAME = "mean_reversion"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["XRP/USD"]

# Configuration with defaults
CONFIG = {
    'lookback_candles': 20,       # Candles for MA calculation
    'deviation_threshold': 0.5,   # % deviation to trigger (0.5%)
    'position_size_usd': 20,      # Size per trade in USD
    'rsi_oversold': 35,           # RSI oversold level
    'rsi_overbought': 65,         # RSI overbought level
    'take_profit_pct': 0.4,       # Take profit at 0.4%
    'stop_loss_pct': 0.6,         # Stop loss at 0.6%
    'max_position': 50,           # Max position size in USD
}


def calculate_sma(candles: List[Candle], period: int) -> float:
    """Calculate simple moving average."""
    if len(candles) < period:
        return 0.0
    closes = [c.close for c in candles[-period:]]
    return sum(closes) / len(closes)


def calculate_rsi(candles: List[Candle], period: int = 14) -> float:
    """Calculate RSI indicator."""
    if len(candles) < period + 1:
        return 50.0  # Neutral

    gains = []
    losses = []

    for i in range(len(candles) - period, len(candles)):
        change = candles[i].close - candles[i-1].close
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_bollinger_bands(candles: List[Candle], period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands."""
    if len(candles) < period:
        return None, None, None

    closes = [c.close for c in candles[-period:]]
    sma = sum(closes) / len(closes)

    variance = sum((c - sma) ** 2 for c in closes) / len(closes)
    std = variance ** 0.5

    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)

    return lower, sma, upper


def generate_signal(data: DataSnapshot, config: dict, state: dict) -> Optional[Signal]:
    """
    Generate mean reversion signal.

    Strategy:
    - Calculate SMA and deviation
    - Use RSI for confirmation
    - Trade when price deviates significantly from mean
    """
    symbol = "XRP/USD"

    # Get candles
    candles_1m = data.candles_1m.get(symbol, ())
    candles_5m = data.candles_5m.get(symbol, ())

    if len(candles_5m) < config['lookback_candles']:
        return None

    current_price = data.prices.get(symbol, 0)
    if not current_price:
        return None

    # Calculate indicators
    sma = calculate_sma(list(candles_5m), config['lookback_candles'])
    rsi = calculate_rsi(list(candles_5m))
    bb_lower, bb_mid, bb_upper = calculate_bollinger_bands(list(candles_5m))

    if not sma or not bb_lower:
        return None

    # Calculate VWAP
    vwap = data.get_vwap(symbol, 50)

    # Calculate deviation from SMA
    deviation_pct = ((current_price - sma) / sma) * 100

    # Store indicators
    state['indicators'] = {
        'sma': sma,
        'rsi': rsi,
        'deviation_pct': deviation_pct,
        'bb_lower': bb_lower,
        'bb_mid': bb_mid,
        'bb_upper': bb_upper,
        'vwap': vwap,
        'position': state.get('position', 0),
    }

    current_position = state.get('position', 0)
    max_position = config['max_position']

    # Generate signals
    signal = None

    # Oversold: Price below SMA and RSI oversold
    if (deviation_pct < -config['deviation_threshold'] and
        rsi < config['rsi_oversold'] and
        current_position < max_position):

        # Extra confirmation: price near or below lower BB
        if current_price <= bb_lower * 1.005:
            signal = Signal(
                action='buy',
                symbol=symbol,
                size=config['position_size_usd'],
                price=current_price,
                reason=f"MR: Oversold (dev={deviation_pct:.2f}%, RSI={rsi:.1f})",
                stop_loss=current_price * (1 - config['stop_loss_pct'] / 100),
                take_profit=sma,  # Target the mean
            )

    # Overbought: Price above SMA and RSI overbought
    elif (deviation_pct > config['deviation_threshold'] and
          rsi > config['rsi_overbought'] and
          current_position > -max_position):

        # Extra confirmation: price near or above upper BB
        if current_price >= bb_upper * 0.995:
            signal = Signal(
                action='sell',
                symbol=symbol,
                size=config['position_size_usd'],
                price=current_price,
                reason=f"MR: Overbought (dev={deviation_pct:.2f}%, RSI={rsi:.1f})",
                stop_loss=current_price * (1 + config['stop_loss_pct'] / 100),
                take_profit=sma,  # Target the mean
            )

    # VWAP reversion opportunity
    elif vwap:
        vwap_deviation = ((current_price - vwap) / vwap) * 100

        # Price significantly below VWAP with neutral RSI
        if vwap_deviation < -0.3 and 40 < rsi < 60 and current_position < max_position:
            signal = Signal(
                action='buy',
                symbol=symbol,
                size=config['position_size_usd'] * 0.5,  # Half size
                price=current_price,
                reason=f"MR: VWAP reversion (vwap_dev={vwap_deviation:.2f}%)",
                stop_loss=current_price * (1 - config['stop_loss_pct'] / 100),
                take_profit=vwap,
            )

    return signal


def on_fill(fill: dict, state: dict) -> None:
    """Update position tracking."""
    size_usd = fill.get('size', 0) * fill.get('price', 0)
    if fill.get('side') == 'buy':
        state['position'] = state.get('position', 0) + size_usd
    else:
        state['position'] = state.get('position', 0) - size_usd

    state['last_fill'] = fill


def on_start(config: dict, state: dict) -> None:
    """Initialize state."""
    state['position'] = 0
    state['indicators'] = {}
