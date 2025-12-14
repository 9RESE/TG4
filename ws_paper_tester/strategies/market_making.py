"""
Market Making Strategy

Provides liquidity by placing orders on both sides of the spread.
Captures spread while managing inventory to stay balanced.

Version History:
- 1.0.0: Initial implementation
- 1.0.1: Added position awareness for sell vs short
- 1.1.0: Added XRP/BTC support for dual-asset accumulation
         - Symbol-specific configuration
         - XRP-denominated inventory tracking for XRP/BTC
         - Wider spreads and adjusted sizing for cross-pair
- 1.2.0: Added BTC/USDT support
         - Higher liquidity pair with tighter spreads
         - Larger position sizes appropriate for BTC
- 1.3.0: Major improvements per market-making-strategy-review-v1.2.md
         - MM-001: Fixed XRP/BTC size units (convert to USD)
         - MM-002: Added volatility-adjusted spreads
         - MM-003: Added signal cooldown mechanism
         - MM-004: Improved R:R ratios
         - MM-005: Fixed on_fill unit handling
         - MM-006: Stop/TP now based on entry price
         - MM-007: Added trade flow confirmation
         - MM-008: Enhanced indicator logging with volatility
"""
from datetime import datetime
from typing import Dict, Any, Optional, List

from ws_tester.types import DataSnapshot, Signal


# =============================================================================
# REQUIRED: Strategy Metadata
# =============================================================================
STRATEGY_NAME = "market_making"
STRATEGY_VERSION = "1.3.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT", "XRP/BTC"]


# =============================================================================
# REQUIRED: Default Configuration
# =============================================================================
CONFIG = {
    # General settings
    'min_spread_pct': 0.1,        # Minimum spread to trade (0.1%)
    'position_size_usd': 20,      # Size per trade in USD
    'max_inventory': 100,         # Max position size in USD
    'inventory_skew': 0.5,        # Reduce size when inventory builds
    'imbalance_threshold': 0.1,   # Orderbook imbalance to trigger

    # Risk management (MM-004: improved R:R)
    'take_profit_pct': 0.4,       # Take profit at 0.4% (was 0.3%)
    'stop_loss_pct': 0.5,         # Stop loss at 0.5%

    # Signal control (MM-003: cooldown)
    'cooldown_seconds': 5.0,      # Minimum time between signals

    # Volatility adjustment (MM-002)
    'base_volatility_pct': 0.5,   # Baseline volatility for scaling
    'volatility_lookback': 20,    # Candles for volatility calculation
    'volatility_threshold_mult': 1.5,  # Max multiplier in high volatility

    # Trade flow confirmation (MM-007)
    'use_trade_flow': True,       # Confirm with trade tape
    'trade_flow_threshold': 0.15, # Minimum trade imbalance alignment
}

# Per-symbol configurations
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'min_spread_pct': 0.05,       # Tighter spreads on USDT pair
        'position_size_usd': 20,
        'max_inventory': 100,
        'imbalance_threshold': 0.1,
        'take_profit_pct': 0.4,       # MM-004: was 0.3
        'stop_loss_pct': 0.5,
        'cooldown_seconds': 5.0,      # MM-003
    },
    'BTC/USDT': {
        # BTC/USDT market making - high liquidity pair
        # - Very tight spreads, high volume
        # - Larger position sizes (BTC trades bigger)
        'min_spread_pct': 0.03,       # Tighter min spread (more liquid)
        'position_size_usd': 50,      # Larger size for BTC
        'max_inventory': 200,         # Higher max inventory
        'imbalance_threshold': 0.08,  # Lower threshold (more liquid)
        'take_profit_pct': 0.35,      # MM-004: was 0.2, now 1:1 R:R
        'stop_loss_pct': 0.35,        # MM-004: was 0.4, now 1:1 R:R
        'cooldown_seconds': 3.0,      # MM-003: faster for BTC
    },
    'XRP/BTC': {
        # XRP/BTC market making - optimized from Kraken data:
        # - 664 trades/day, 0.0446% spread
        # - Goal: Accumulate both XRP and BTC through spread capture
        'min_spread_pct': 0.03,       # Lower threshold (spread is 0.0446%)
        'position_size_xrp': 25,      # Size in XRP (converted to USD in signal)
        'max_inventory_xrp': 150,     # Max XRP exposure
        'imbalance_threshold': 0.15,  # Slightly higher (less liquid)
        'take_profit_pct': 0.3,       # MM-004: was 0.25
        'stop_loss_pct': 0.4,
        'cooldown_seconds': 10.0,     # MM-003: slower for cross-pair
    },
}


# =============================================================================
# Helper Functions
# =============================================================================
def _get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """Get symbol-specific config or fall back to global config."""
    symbol_cfg = SYMBOL_CONFIGS.get(symbol, {})
    return symbol_cfg.get(key, config.get(key))


def _is_xrp_btc(symbol: str) -> bool:
    """Check if this is the XRP/BTC pair."""
    return symbol == 'XRP/BTC'


def _calculate_volatility(candles, lookback: int = 20) -> float:
    """
    Calculate price volatility from candle closes.

    Returns volatility as a percentage (std dev of returns * 100).
    """
    if not candles or len(candles) < lookback + 1:
        return 0.0

    closes = [c.close for c in candles[-(lookback + 1):]]
    if len(closes) < 2:
        return 0.0

    # Calculate returns
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] != 0]

    if not returns:
        return 0.0

    # Calculate standard deviation of returns
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5

    return std_dev * 100  # Return as percentage


def _get_trade_flow_imbalance(data: DataSnapshot, symbol: str, n_trades: int = 50) -> float:
    """Get trade flow imbalance from recent trades."""
    return data.get_trade_imbalance(symbol, n_trades)


# =============================================================================
# REQUIRED: Main Signal Generation Function
# =============================================================================
def generate_signal(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any]
) -> Optional[Signal]:
    """
    Generate market making signal.

    Strategy:
    - Trade when spread is wide enough to capture
    - Skew quotes based on inventory
    - Use stop-loss and take-profit
    - For XRP/BTC: accumulate both assets through spread capture
    - MM-002: Adjust thresholds based on volatility
    - MM-003: Respect cooldown between signals
    - MM-007: Confirm with trade flow

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
        state['inventory'] = 0
        state['inventory_by_symbol'] = {}
        state['xrp_accumulated'] = 0.0
        state['btc_accumulated'] = 0.0
        state['last_signal_time'] = None
        state['indicators'] = {}

    current_time = data.timestamp

    # MM-003: Global cooldown check
    if state.get('last_signal_time') is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        global_cooldown = config.get('cooldown_seconds', 5.0)
        if elapsed < global_cooldown:
            return None

    # Iterate over configured symbols
    for symbol in SYMBOLS:
        signal = _evaluate_symbol(data, config, state, symbol, current_time)
        if signal is not None:
            state['last_signal_time'] = current_time
            return signal

    return None


def _evaluate_symbol(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_time: datetime
) -> Optional[Signal]:
    """Evaluate a single symbol for market making opportunity."""
    # Get orderbook
    ob = data.orderbooks.get(symbol)
    if not ob or not ob.best_bid or not ob.best_ask:
        return None

    price = data.prices.get(symbol, 0)
    if not price:
        return None

    # Calculate spread
    spread_pct = ob.spread_pct

    # Get symbol-specific config
    min_spread = _get_symbol_config(symbol, config, 'min_spread_pct')
    imbalance_threshold = _get_symbol_config(symbol, config, 'imbalance_threshold')
    tp_pct = _get_symbol_config(symbol, config, 'take_profit_pct')
    sl_pct = _get_symbol_config(symbol, config, 'stop_loss_pct')
    symbol_cooldown = _get_symbol_config(symbol, config, 'cooldown_seconds')

    is_cross_pair = _is_xrp_btc(symbol)

    # MM-002: Calculate volatility for dynamic threshold adjustment
    candles = data.candles_1m.get(symbol, ())
    volatility = _calculate_volatility(candles, config.get('volatility_lookback', 20))
    base_vol = config.get('base_volatility_pct', 0.5)

    # Dynamic threshold adjustment based on volatility
    vol_multiplier = 1.0
    if base_vol > 0 and volatility > 0:
        vol_ratio = volatility / base_vol
        if vol_ratio > 1.0:
            # High volatility: increase threshold to avoid noise
            vol_multiplier = min(vol_ratio, config.get('volatility_threshold_mult', 1.5))

    effective_threshold = imbalance_threshold * vol_multiplier
    effective_min_spread = min_spread * vol_multiplier

    # MM-007: Get trade flow imbalance for confirmation
    trade_flow = _get_trade_flow_imbalance(data, symbol, 50)
    trade_flow_threshold = config.get('trade_flow_threshold', 0.15)
    use_trade_flow = config.get('use_trade_flow', True)

    # Get inventory (different units for XRP/BTC)
    if 'inventory_by_symbol' not in state:
        state['inventory_by_symbol'] = {}

    if is_cross_pair:
        # For XRP/BTC, track inventory in XRP
        inventory = state['inventory_by_symbol'].get(symbol, 0)  # in XRP
        max_inventory = _get_symbol_config(symbol, config, 'max_inventory_xrp') or 150
        base_size_xrp = _get_symbol_config(symbol, config, 'position_size_xrp') or 25
        # MM-001: Convert XRP size to USD for Signal
        xrp_usdt_price = data.prices.get('XRP/USDT', 2.35)
        base_size = base_size_xrp * xrp_usdt_price
    else:
        # For USD pairs, track in USD
        inventory = state['inventory_by_symbol'].get(symbol, 0)  # in USD
        max_inventory = _get_symbol_config(symbol, config, 'max_inventory') or 100
        base_size = _get_symbol_config(symbol, config, 'position_size_usd') or 20
        base_size_xrp = 0  # Not applicable

    # MM-008: Enhanced indicator logging with volatility
    state['indicators'] = {
        'symbol': symbol,
        'spread_pct': round(spread_pct, 4),
        'min_spread_pct': round(min_spread, 4),
        'effective_min_spread': round(effective_min_spread, 4),
        'best_bid': round(ob.best_bid, 8),
        'best_ask': round(ob.best_ask, 8),
        'mid': round(ob.mid, 8),
        'inventory': round(inventory, 4),
        'max_inventory': max_inventory,
        'imbalance': round(ob.imbalance, 4),
        'effective_threshold': round(effective_threshold, 4),
        'is_cross_pair': is_cross_pair,
        # MM-008: Volatility metrics
        'volatility_pct': round(volatility, 4),
        'vol_multiplier': round(vol_multiplier, 2),
        # MM-007: Trade flow
        'trade_flow': round(trade_flow, 4),
        'trade_flow_aligned': False,  # Will be set below
    }

    # Check minimum spread (with volatility adjustment)
    if spread_pct < effective_min_spread:
        return None

    # Calculate position size with inventory skew
    skew_factor = 1.0 - abs(inventory / max_inventory) * config.get('inventory_skew', 0.5)
    position_size = base_size * max(skew_factor, 0.1)

    # Minimum trade size check (in USD)
    min_size = 5.0
    if position_size < min_size:
        return None

    # Decide action based on orderbook imbalance and inventory
    imbalance = ob.imbalance  # positive = more bids (buy pressure)

    # MM-007: Check trade flow alignment
    def is_trade_flow_aligned(direction: str) -> bool:
        """Check if trade flow confirms the signal direction."""
        if not use_trade_flow:
            return True
        if direction == 'buy':
            return trade_flow > trade_flow_threshold
        elif direction == 'sell':
            return trade_flow < -trade_flow_threshold
        return True

    # If we're long and see selling pressure, reduce position
    if inventory > 0 and imbalance < -0.2:
        close_size = min(position_size, abs(inventory))
        if is_cross_pair:
            # MM-001: Convert back to USD for XRP/BTC
            xrp_usdt_price = data.prices.get('XRP/USDT', 2.35)
            close_size = close_size * xrp_usdt_price
        if close_size >= min_size:
            # MM-006: Use entry price (best_bid for sells)
            entry_price = ob.best_bid
            state['indicators']['trade_flow_aligned'] = is_trade_flow_aligned('sell')
            return Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=entry_price,
                reason=f"MM: Reduce long on sell pressure (imbal={imbalance:.2f})",
                stop_loss=entry_price * (1 - sl_pct / 100),
                take_profit=entry_price * (1 + tp_pct / 100),
                metadata={'xrp_size': close_size / xrp_usdt_price if is_cross_pair else None},
            )

    # If we're short and see buying pressure, cover
    if inventory < 0 and imbalance > 0.2:
        cover_size = min(position_size, abs(inventory))
        if is_cross_pair:
            xrp_usdt_price = data.prices.get('XRP/USDT', 2.35)
            cover_size = cover_size * xrp_usdt_price
        if cover_size >= min_size:
            entry_price = ob.best_ask
            state['indicators']['trade_flow_aligned'] = is_trade_flow_aligned('buy')
            return Signal(
                action='buy',
                symbol=symbol,
                size=cover_size,
                price=entry_price,
                reason=f"MM: Reduce short on buy pressure (imbal={imbalance:.2f})",
                stop_loss=entry_price * (1 + sl_pct / 100),
                take_profit=entry_price * (1 - tp_pct / 100),
                metadata={'xrp_size': cover_size / xrp_usdt_price if is_cross_pair else None},
            )

    # If inventory allows, trade based on spread capture
    if inventory < max_inventory and imbalance > effective_threshold:
        # MM-007: Check trade flow confirmation
        if not is_trade_flow_aligned('buy'):
            state['indicators']['trade_flow_aligned'] = False
            return None

        state['indicators']['trade_flow_aligned'] = True

        # More bids than asks - buy side has support
        # For XRP/BTC: buying means spending BTC to get XRP
        entry_price = ob.best_ask
        signal_size = position_size
        if is_cross_pair:
            xrp_usdt_price = data.prices.get('XRP/USDT', 2.35)
            # position_size is already in USD from MM-001 fix
            xrp_size = position_size / xrp_usdt_price
            metadata = {'xrp_size': xrp_size}
        else:
            metadata = None

        return Signal(
            action='buy',
            symbol=symbol,
            size=signal_size,
            price=entry_price,
            reason=f"MM: Spread capture buy (spread={spread_pct:.3f}%, imbal={imbalance:.2f}, vol={volatility:.2f}%)",
            stop_loss=entry_price * (1 - sl_pct / 100),
            take_profit=entry_price * (1 + tp_pct / 100),
            metadata=metadata,
        )

    if inventory > -max_inventory and imbalance < -effective_threshold:
        # MM-007: Check trade flow confirmation
        if not is_trade_flow_aligned('sell'):
            state['indicators']['trade_flow_aligned'] = False
            return None

        state['indicators']['trade_flow_aligned'] = True

        # More asks than bids - sell pressure
        # For XRP/BTC: selling means spending XRP to get BTC
        if inventory > 0:
            # We have a long position - sell to reduce
            close_size = min(position_size, inventory)
            if is_cross_pair:
                xrp_usdt_price = data.prices.get('XRP/USDT', 2.35)
                close_size_usd = close_size * xrp_usdt_price
                metadata = {'xrp_size': close_size}
            else:
                close_size_usd = close_size
                metadata = None

            if close_size_usd >= min_size:
                entry_price = ob.best_bid
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=close_size_usd,
                    price=entry_price,
                    reason=f"MM: Reduce long on imbalance (spread={spread_pct:.3f}%, vol={volatility:.2f}%)",
                    stop_loss=entry_price * (1 - sl_pct / 100),
                    take_profit=entry_price * (1 + tp_pct / 100),
                    metadata=metadata,
                )
        else:
            # We're flat or short
            if is_cross_pair:
                # For XRP/BTC, we don't short - just sell to accumulate BTC
                entry_price = ob.best_bid
                xrp_usdt_price = data.prices.get('XRP/USDT', 2.35)
                xrp_size = position_size / xrp_usdt_price
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=position_size,
                    price=entry_price,
                    reason=f"MM: Sell XRP for BTC (spread={spread_pct:.3f}%, imbal={imbalance:.2f})",
                    stop_loss=entry_price * (1 + sl_pct / 100),
                    take_profit=entry_price * (1 - tp_pct / 100),
                    metadata={'xrp_size': xrp_size},
                )
            else:
                # For USD pairs, open short position
                entry_price = ob.best_bid
                return Signal(
                    action='short',
                    symbol=symbol,
                    size=position_size,
                    price=entry_price,
                    reason=f"MM: Spread capture short (spread={spread_pct:.3f}%, vol={volatility:.2f}%)",
                    stop_loss=entry_price * (1 + sl_pct / 100),
                    take_profit=entry_price * (1 - tp_pct / 100),
                )

    return None


# =============================================================================
# OPTIONAL: Lifecycle Callbacks
# =============================================================================
def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Initialize state."""
    state['initialized'] = True
    state['inventory'] = 0
    state['inventory_by_symbol'] = {}
    state['xrp_accumulated'] = 0.0
    state['btc_accumulated'] = 0.0
    state['last_signal_time'] = None
    state['indicators'] = {}


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Update inventory on fill.

    MM-005: Fixed unit handling - use value field from executor for USD pairs.

    For XRP/BTC:
    - Buy: +XRP inventory, track BTC spent
    - Sell: -XRP inventory, track BTC received
    """
    symbol = fill.get('symbol', 'XRP/USDT')
    side = fill.get('side', '')
    size = fill.get('size', 0)  # Base asset size from executor
    price = fill.get('price', 0)
    # MM-005: Use value from executor if available (always in quote currency)
    value = fill.get('value', size * price)

    # Initialize inventory tracking by symbol
    if 'inventory_by_symbol' not in state:
        state['inventory_by_symbol'] = {}

    current_inventory = state['inventory_by_symbol'].get(symbol, 0)
    is_cross_pair = _is_xrp_btc(symbol)

    if is_cross_pair:
        # For XRP/BTC, size is in XRP, price is in BTC
        xrp_amount = size
        btc_amount = size * price

        if side == 'buy':
            # Bought XRP with BTC
            state['inventory_by_symbol'][symbol] = current_inventory + xrp_amount
            state['xrp_accumulated'] = state.get('xrp_accumulated', 0) + xrp_amount
        elif side == 'sell':
            # Sold XRP for BTC
            state['inventory_by_symbol'][symbol] = current_inventory - xrp_amount
            state['btc_accumulated'] = state.get('btc_accumulated', 0) + btc_amount
    else:
        # MM-005: For USD pairs, use value (USD) directly
        # The executor provides size in base asset, so value = size * price is USD
        size_usd = value

        if side == 'buy':
            state['inventory_by_symbol'][symbol] = current_inventory + size_usd
        elif side == 'sell':
            state['inventory_by_symbol'][symbol] = current_inventory - size_usd
        elif side == 'short':
            state['inventory_by_symbol'][symbol] = current_inventory - size_usd
        elif side == 'cover':
            state['inventory_by_symbol'][symbol] = current_inventory + size_usd

    # Keep legacy inventory for backward compatibility
    state['inventory'] = sum(state['inventory_by_symbol'].values())
    state['last_fill'] = fill


def on_stop(state: Dict[str, Any]) -> None:
    """Called when strategy stops."""
    state['final_summary'] = {
        'inventory_by_symbol': state.get('inventory_by_symbol', {}),
        'xrp_accumulated': state.get('xrp_accumulated', 0),
        'btc_accumulated': state.get('btc_accumulated', 0),
    }
