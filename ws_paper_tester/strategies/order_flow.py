"""
Order Flow Strategy

Trades based on trade tape analysis and buy/sell imbalance.
Enhanced with volatility adjustment and dynamic thresholds.

Version History:
- 1.0.0: Initial implementation
- 1.0.1: Added position awareness for sell vs short (HIGH-008 fix)
- 2.0.0: Major refactor per strategy-development-guide.md and market-making-strategy-review.md
         - Added volatility measurement and dynamic thresholds
         - Added time-based cooldown alongside trade-based
         - Enhanced indicator logging with volatility metrics
         - Fixed type hints and import patterns
         - Added on_stop() callback for compliance
         - Improved risk-reward ratio
- 2.1.0: Added XRP/BTC pair support for ratio trading
         - Research-backed config: 664 trades/day, 0.0446% spread
         - Symbol-specific cooldowns, thresholds, TP/SL
         - Separate logic for XRP/BTC (no shorting, direct sell)
         - Goal: Grow both XRP and BTC holdings through ratio trading
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import DataSnapshot, Signal


# =============================================================================
# REQUIRED: Strategy Metadata
# =============================================================================
STRATEGY_NAME = "order_flow"
STRATEGY_VERSION = "2.1.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT", "XRP/BTC"]


# =============================================================================
# REQUIRED: Default Configuration
# =============================================================================
CONFIG = {
    # Order flow parameters
    'imbalance_threshold': 0.3,        # Min imbalance to trigger (0.3 = 30%)
    'volume_spike_mult': 2.0,          # Volume spike multiplier
    'lookback_trades': 50,             # Number of trades to analyze

    # Position sizing
    'position_size_usd': 25.0,         # Size per trade in USD
    'max_position_usd': 100.0,         # Maximum position exposure

    # Risk management - Improved R:R ratio (was 0.5:0.3 = 1.67:1)
    'take_profit_pct': 0.5,            # Take profit at 0.5%
    'stop_loss_pct': 0.5,              # Stop loss at 0.5% (1:1 R:R)

    # Cooldown mechanisms
    'cooldown_trades': 10,             # Min trades between signals
    'cooldown_seconds': 5.0,           # Min time between signals (NEW)

    # Volatility adjustment parameters (NEW)
    'base_volatility_pct': 0.5,        # Baseline volatility for scaling
    'volatility_lookback': 20,         # Candles for volatility calculation
    'volatility_threshold_mult': 1.5,  # Increase threshold in high volatility

    # VWAP parameters
    'vwap_deviation_threshold': 0.001, # Min deviation from VWAP for reversion signal
}

# Per-symbol configurations
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'imbalance_threshold': 0.3,
        'position_size_usd': 25.0,
    },
    'BTC/USDT': {
        'imbalance_threshold': 0.25,   # Slightly lower for BTC
        'position_size_usd': 50.0,     # Larger size for BTC
    },
    'XRP/BTC': {
        # XRP/BTC ratio trading - optimized based on Kraken 24h data:
        # - 664 trades/day (~1 per 2 min), 96K XRP volume, 0.0446% spread
        'imbalance_threshold': 0.35,   # Higher threshold (wider spread, less noise)
        'volume_spike_mult': 1.5,      # Lower mult (fewer trades to detect spikes)
        'cooldown_seconds': 30.0,      # Longer cooldown (lower trade frequency)
        'cooldown_trades': 5,          # Fewer trades needed for cooldown
        'position_size_xrp': 30.0,     # Trade 30 XRP per signal (~6% of 500 XRP)
        'take_profit_pct': 0.4,        # Wider than spread (0.0446%)
        'stop_loss_pct': 0.4,          # 1:1 R:R
        'base_asset': 'XRP',           # Selling XRP to get BTC
        'quote_asset': 'BTC',
    },
}


# =============================================================================
# Helper Functions
# =============================================================================
def _calculate_volatility(candles, lookback: int = 20) -> float:
    """
    Calculate price volatility from candle closes.

    Returns volatility as a percentage (std dev of returns * 100).
    """
    if len(candles) < lookback + 1:
        return 0.0

    closes = [c.close for c in candles[-(lookback + 1):]]
    if len(closes) < 2:
        return 0.0

    # Calculate returns
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes))]

    if not returns:
        return 0.0

    # Calculate standard deviation of returns
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5

    return std_dev * 100  # Return as percentage


def _get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """Get symbol-specific config or fall back to global config."""
    symbol_cfg = SYMBOL_CONFIGS.get(symbol, {})
    return symbol_cfg.get(key, config.get(key))


# =============================================================================
# REQUIRED: Main Signal Generation Function
# =============================================================================
def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """
    Generate order flow signal based on trade tape analysis.

    Strategy:
    - Analyze recent trades for buy/sell imbalance
    - Look for volume spikes indicating momentum
    - Adjust thresholds based on market volatility
    - Trade in direction of order flow with VWAP confirmation

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
        state['last_signal_idx'] = 0
        state['total_trades_seen'] = 0
        state['last_signal_time'] = None
        state['position_side'] = None
        state['position_size'] = 0.0
        state['fills'] = []
        state['indicators'] = {}

    current_time = data.timestamp

    # Time-based cooldown check
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        if elapsed < config.get('cooldown_seconds', 5.0):
            return None

    # Evaluate each symbol
    for symbol in SYMBOLS:
        signal = _evaluate_symbol(data, config, state, symbol, current_time)
        if signal:
            return signal

    return None


def _evaluate_symbol(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_time: datetime
) -> Optional[Signal]:
    """Evaluate order flow for a specific symbol."""
    trades = data.trades.get(symbol, ())
    lookback = config.get('lookback_trades', 50)

    if len(trades) < lookback:
        return None

    recent_trades = trades[:lookback]

    # Get symbol-specific cooldown settings
    cooldown_trades = _get_symbol_config(symbol, config, 'cooldown_trades') or config.get('cooldown_trades', 10)

    # Trade-based cooldown check
    state['total_trades_seen'] = len(trades)
    trades_since_signal = state['total_trades_seen'] - state['last_signal_idx']

    if trades_since_signal < cooldown_trades:
        return None

    # Calculate buy/sell imbalance
    buy_volume = sum(t.size for t in recent_trades if t.side == 'buy')
    sell_volume = sum(t.size for t in recent_trades if t.side == 'sell')
    total_volume = buy_volume + sell_volume

    if total_volume == 0:
        return None

    imbalance = (buy_volume - sell_volume) / total_volume  # -1 to 1

    # Calculate average volume and volume spike
    avg_trade_size = total_volume / len(recent_trades)
    last_5_volume = sum(t.size for t in recent_trades[:5])
    expected_5_volume = avg_trade_size * 5
    volume_spike = last_5_volume / expected_5_volume if expected_5_volume > 0 else 1.0

    # Calculate VWAP
    vwap = data.get_vwap(symbol, lookback)
    current_price = data.prices.get(symbol, 0)

    if not current_price or not vwap:
        return None

    price_vs_vwap = (current_price - vwap) / vwap  # Positive = above VWAP

    # Calculate volatility from candles
    candles = data.candles_1m.get(symbol, ())
    volatility = _calculate_volatility(candles, config.get('volatility_lookback', 20))
    base_vol = config.get('base_volatility_pct', 0.5)

    # Dynamic threshold adjustment based on volatility
    vol_multiplier = 1.0
    if base_vol > 0 and volatility > 0:
        vol_ratio = volatility / base_vol
        if vol_ratio > 1.5:
            # High volatility: increase threshold to avoid noise
            vol_multiplier = min(vol_ratio, config.get('volatility_threshold_mult', 1.5))

    # Get symbol-specific config
    base_threshold = _get_symbol_config(symbol, config, 'imbalance_threshold')
    volume_spike_mult = _get_symbol_config(symbol, config, 'volume_spike_mult') or config.get('volume_spike_mult', 2.0)

    # Handle XRP/BTC differently - size in XRP, not USD
    is_xrp_btc = symbol == 'XRP/BTC'
    if is_xrp_btc:
        position_size = _get_symbol_config(symbol, config, 'position_size_xrp') or 30.0
        # For XRP/BTC, get symbol-specific TP/SL
        tp_pct = _get_symbol_config(symbol, config, 'take_profit_pct') or config.get('take_profit_pct', 0.4)
        sl_pct = _get_symbol_config(symbol, config, 'stop_loss_pct') or config.get('stop_loss_pct', 0.4)
    else:
        position_size = _get_symbol_config(symbol, config, 'position_size_usd')
        tp_pct = config.get('take_profit_pct', 0.5)
        sl_pct = config.get('stop_loss_pct', 0.5)

    # Adjusted threshold
    effective_threshold = base_threshold * vol_multiplier

    # Store indicators for logging
    state['indicators'] = {
        'symbol': symbol,
        'imbalance': round(imbalance, 4),
        'buy_volume': round(buy_volume, 2),
        'sell_volume': round(sell_volume, 2),
        'volume_spike': round(volume_spike, 2),
        'vwap': round(vwap, 6),
        'price_vs_vwap': round(price_vs_vwap, 6),
        'volatility_pct': round(volatility, 4),
        'vol_multiplier': round(vol_multiplier, 2),
        'effective_threshold': round(effective_threshold, 4),
        'trades_since_signal': trades_since_signal,
    }

    # Check position limits (different for XRP/BTC vs USD pairs)
    current_position = state.get('position_size', 0)
    if is_xrp_btc:
        # For XRP/BTC, we trade XRP directly - check XRP holdings
        # Skip position limit for now (managed by XRP balance in portfolio)
        max_position = 500.0  # Max XRP to trade
        min_trade = 5.0  # Minimum XRP per trade
    else:
        max_position = config.get('max_position_usd', 100.0)
        min_trade = 5.0  # Minimum USD per trade

    if current_position >= max_position:
        return None

    # Adjust size if needed to stay within limits
    available = max_position - current_position
    actual_size = min(position_size, available)
    if actual_size < min_trade:
        return None

    # Generate signal using symbol-specific volume spike multiplier
    signal = None

    # Strong buy pressure with volume spike
    # For XRP/BTC: buy = trade BTC for XRP (accumulate XRP)
    if imbalance > effective_threshold and volume_spike > volume_spike_mult:
        signal = Signal(
            action='buy',
            symbol=symbol,
            size=actual_size,
            price=current_price,
            reason=f"OF: Buy pressure (imbal={imbalance:.2f}, vol={volume_spike:.1f}x, volatility={volatility:.2f}%)",
            stop_loss=current_price * (1 - sl_pct / 100),
            take_profit=current_price * (1 + tp_pct / 100),
        )

    # Strong sell pressure with volume spike
    # For XRP/BTC: sell = trade XRP for BTC (accumulate BTC)
    elif imbalance < -effective_threshold and volume_spike > volume_spike_mult:
        # Check if we have a long position to sell, otherwise go short
        has_long = state.get('position_side') == 'long' and state.get('position_size', 0) > 0

        if has_long:
            # Sell existing long position
            close_size = min(actual_size, state.get('position_size', actual_size))
            signal = Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=current_price,
                reason=f"OF: Close long (imbal={imbalance:.2f}, vol={volume_spike:.1f}x)",
                stop_loss=current_price * (1 - sl_pct / 100),
                take_profit=current_price * (1 + tp_pct / 100),
            )
        else:
            # Open a short position (for USD pairs) or sell XRP (for XRP/BTC)
            if is_xrp_btc:
                # For XRP/BTC, sell means trading XRP for BTC
                signal = Signal(
                    action='sell',
                    symbol=symbol,
                    size=actual_size,
                    price=current_price,
                    reason=f"OF: Sell XRP for BTC (imbal={imbalance:.2f}, vol={volume_spike:.1f}x)",
                    stop_loss=current_price * (1 + sl_pct / 100),  # Stop above for sell
                    take_profit=current_price * (1 - tp_pct / 100),  # TP below for sell
                )
            else:
                signal = Signal(
                    action='short',
                    symbol=symbol,
                    size=actual_size,
                    price=current_price,
                    reason=f"OF: Short (imbal={imbalance:.2f}, vol={volume_spike:.1f}x, volatility={volatility:.2f}%)",
                    stop_loss=current_price * (1 + sl_pct / 100),
                    take_profit=current_price * (1 - tp_pct / 100),
                )

    # Buy pressure + price below VWAP (mean reversion opportunity)
    elif (imbalance > effective_threshold * 0.7 and
          price_vs_vwap < -config.get('vwap_deviation_threshold', 0.001)):
        reduced_size = actual_size * 0.75  # Smaller size for reversion plays
        if reduced_size >= min_trade:
            signal = Signal(
                action='buy',
                symbol=symbol,
                size=reduced_size,
                price=current_price,
                reason=f"OF: Buy below VWAP (imbal={imbalance:.2f}, dev={price_vs_vwap:.4f})",
                stop_loss=current_price * (1 - sl_pct / 100),
                take_profit=vwap,  # Target VWAP for reversion
            )

    # Sell pressure + price above VWAP (mean reversion short)
    elif (imbalance < -effective_threshold * 0.7 and
          price_vs_vwap > config.get('vwap_deviation_threshold', 0.001)):
        has_long = state.get('position_side') == 'long' and state.get('position_size', 0) > 0
        reduced_size = actual_size * 0.75

        if has_long and reduced_size >= min_trade:
            # Close long position if above VWAP with sell pressure
            close_size = min(reduced_size, state.get('position_size', reduced_size))
            signal = Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=current_price,
                reason=f"OF: Close long above VWAP (imbal={imbalance:.2f}, dev={price_vs_vwap:.4f})",
            )

    if signal:
        state['last_signal_idx'] = state['total_trades_seen']
        state['last_signal_time'] = current_time
        return signal

    return None


# =============================================================================
# OPTIONAL: Lifecycle Callbacks
# =============================================================================
def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Initialize strategy state on startup."""
    state['initialized'] = True
    state['last_signal_idx'] = 0
    state['total_trades_seen'] = 0
    state['last_signal_time'] = None
    state['position_side'] = None  # 'long', 'short', or None
    state['position_size'] = 0.0
    state['fills'] = []
    state['indicators'] = {}


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Track fills and update position state.

    Args:
        fill: Fill information dict with keys: side, size, price, value, symbol
        state: Mutable strategy state dict
    """
    # Track fill history (bounded)
    if 'fills' not in state:
        state['fills'] = []
    state['fills'].append(fill)
    state['fills'] = state['fills'][-20:]  # Keep last 20

    side = fill.get('side', '')
    value = fill.get('value', fill.get('size', 0) * fill.get('price', 0))

    if side == 'buy':
        # Opening or adding to long position
        state['position_side'] = 'long'
        state['position_size'] = state.get('position_size', 0) + value
    elif side == 'sell':
        # Closing long position
        state['position_size'] = max(0, state.get('position_size', 0) - value)
        if state['position_size'] < 0.01:
            state['position_side'] = None
            state['position_size'] = 0.0
    elif side == 'short':
        # Opening or adding to short position
        state['position_side'] = 'short'
        state['position_size'] = state.get('position_size', 0) + value
    elif side == 'cover':
        # Closing short position
        state['position_size'] = max(0, state.get('position_size', 0) - value)
        if state['position_size'] < 0.01:
            state['position_side'] = None
            state['position_size'] = 0.0


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops. Cleanup if needed.

    Args:
        state: Mutable strategy state dict
    """
    # Log final state for debugging
    final_position = state.get('position_size', 0)
    final_side = state.get('position_side')
    total_fills = len(state.get('fills', []))

    # Clear any large cached data
    state['indicators'] = {}

    # Keep summary for post-run analysis
    state['final_summary'] = {
        'position_side': final_side,
        'position_size': final_position,
        'total_fills': total_fills,
    }
