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
"""
from datetime import datetime
from typing import Dict, Any, Optional

from ws_tester.types import DataSnapshot, Signal


# =============================================================================
# REQUIRED: Strategy Metadata
# =============================================================================
STRATEGY_NAME = "market_making"
STRATEGY_VERSION = "1.1.0"
SYMBOLS = ["XRP/USDT", "XRP/BTC"]


# =============================================================================
# REQUIRED: Default Configuration
# =============================================================================
CONFIG = {
    # General settings
    'min_spread_pct': 0.1,        # Minimum spread to trade (0.1%)
    'position_size_usd': 20,      # Size per trade in USD
    'max_inventory': 100,         # Max position size in USD
    'inventory_skew': 0.5,        # Reduce size when inventory builds
    'take_profit_pct': 0.3,       # Take profit at 0.3%
    'stop_loss_pct': 0.5,         # Stop loss at 0.5%
    'imbalance_threshold': 0.1,   # Orderbook imbalance to trigger
}

# Per-symbol configurations
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'min_spread_pct': 0.05,       # Tighter spreads on USDT pair
        'position_size_usd': 20,
        'max_inventory': 100,
        'imbalance_threshold': 0.1,
    },
    'XRP/BTC': {
        # XRP/BTC market making - optimized from Kraken data:
        # - 664 trades/day, 0.0446% spread
        # - Goal: Accumulate both XRP and BTC through spread capture
        'min_spread_pct': 0.03,       # Lower threshold (spread is 0.0446%)
        'position_size_xrp': 25,      # Size in XRP (not USD)
        'max_inventory_xrp': 150,     # Max XRP exposure
        'imbalance_threshold': 0.15,  # Slightly higher (less liquid)
        'take_profit_pct': 0.25,      # Tighter TP for spread capture
        'stop_loss_pct': 0.4,         # Tighter SL
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
        state['indicators'] = {}

    # Iterate over configured symbols
    for symbol in SYMBOLS:
        signal = _evaluate_symbol(data, config, state, symbol)
        if signal is not None:
            return signal

    return None


def _evaluate_symbol(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str
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

    is_cross_pair = _is_xrp_btc(symbol)

    # Get inventory (different units for XRP/BTC)
    if 'inventory_by_symbol' not in state:
        state['inventory_by_symbol'] = {}

    if is_cross_pair:
        # For XRP/BTC, track inventory in XRP
        inventory = state['inventory_by_symbol'].get(symbol, 0)  # in XRP
        max_inventory = _get_symbol_config(symbol, config, 'max_inventory_xrp') or 150
        base_size = _get_symbol_config(symbol, config, 'position_size_xrp') or 25
    else:
        # For USD pairs, track in USD
        inventory = state['inventory_by_symbol'].get(symbol, 0)  # in USD
        max_inventory = _get_symbol_config(symbol, config, 'max_inventory') or 100
        base_size = _get_symbol_config(symbol, config, 'position_size_usd') or 20

    # Store indicators for logging
    state['indicators'] = {
        'symbol': symbol,
        'spread_pct': round(spread_pct, 4),
        'min_spread_pct': round(min_spread, 4),
        'best_bid': round(ob.best_bid, 8),
        'best_ask': round(ob.best_ask, 8),
        'mid': round(ob.mid, 8),
        'inventory': round(inventory, 4),
        'max_inventory': max_inventory,
        'imbalance': round(ob.imbalance, 4),
        'is_cross_pair': is_cross_pair,
    }

    # Check minimum spread
    if spread_pct < min_spread:
        return None

    # Calculate position size with inventory skew
    skew_factor = 1.0 - abs(inventory / max_inventory) * config.get('inventory_skew', 0.5)
    position_size = base_size * max(skew_factor, 0.1)

    # Minimum trade size check
    min_size = 5.0 if is_cross_pair else 5.0
    if position_size < min_size:
        return None

    # Decide action based on orderbook imbalance and inventory
    imbalance = ob.imbalance  # positive = more bids (buy pressure)

    # If we're long and see selling pressure, reduce position
    if inventory > 0 and imbalance < -0.2:
        close_size = min(position_size, abs(inventory))
        if close_size >= min_size:
            return Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=ob.best_bid,
                reason=f"MM: Reduce long on sell pressure (imbal={imbalance:.2f})",
                stop_loss=ob.mid * (1 - sl_pct / 100),
                take_profit=ob.mid * (1 + tp_pct / 100),
            )

    # If we're short and see buying pressure, cover
    if inventory < 0 and imbalance > 0.2:
        cover_size = min(position_size, abs(inventory))
        if cover_size >= min_size:
            return Signal(
                action='buy',
                symbol=symbol,
                size=cover_size,
                price=ob.best_ask,
                reason=f"MM: Reduce short on buy pressure (imbal={imbalance:.2f})",
                stop_loss=ob.mid * (1 + sl_pct / 100),
                take_profit=ob.mid * (1 - tp_pct / 100),
            )

    # If inventory allows, trade based on spread capture
    if inventory < max_inventory and imbalance > imbalance_threshold:
        # More bids than asks - buy side has support
        # For XRP/BTC: buying means spending BTC to get XRP
        return Signal(
            action='buy',
            symbol=symbol,
            size=position_size,
            price=ob.best_ask,
            reason=f"MM: Spread capture buy (spread={spread_pct:.3f}%, imbal={imbalance:.2f})",
            stop_loss=ob.mid * (1 - sl_pct / 100),
            take_profit=ob.mid * (1 + tp_pct / 100),
        )

    if inventory > -max_inventory and imbalance < -imbalance_threshold:
        # More asks than bids - sell pressure
        # For XRP/BTC: selling means spending XRP to get BTC
        if inventory > 0:
            # We have a long position - sell to reduce
            close_size = min(position_size, inventory)
            if close_size >= min_size:
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=close_size,
                    price=ob.best_bid,
                    reason=f"MM: Reduce long on imbalance (spread={spread_pct:.3f}%)",
                    stop_loss=ob.mid * (1 - sl_pct / 100),
                    take_profit=ob.mid * (1 + tp_pct / 100),
                )
        else:
            # We're flat or short
            if is_cross_pair:
                # For XRP/BTC, we don't short - just sell to accumulate BTC
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=position_size,
                    price=ob.best_bid,
                    reason=f"MM: Sell XRP for BTC (spread={spread_pct:.3f}%, imbal={imbalance:.2f})",
                    stop_loss=ob.mid * (1 + sl_pct / 100),
                    take_profit=ob.mid * (1 - tp_pct / 100),
                )
            else:
                # For USD pairs, open short position
                return Signal(
                    action='short',
                    symbol=symbol,
                    size=position_size,
                    price=ob.best_bid,
                    reason=f"MM: Spread capture short (spread={spread_pct:.3f}%)",
                    stop_loss=ob.mid * (1 + sl_pct / 100),
                    take_profit=ob.mid * (1 - tp_pct / 100),
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
    state['indicators'] = {}


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Update inventory on fill.

    For XRP/BTC:
    - Buy: +XRP inventory, track BTC spent
    - Sell: -XRP inventory, track BTC received
    """
    symbol = fill.get('symbol', 'XRP/USDT')
    side = fill.get('side', '')
    size = fill.get('size', 0)
    price = fill.get('price', 0)

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
        # For USD pairs, track in USD value
        size_usd = size * price if size < 1000 else size  # Handle both XRP and USD sizes

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
