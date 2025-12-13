"""
Market Making Strategy
Provides liquidity by placing orders on both sides of the spread.
"""

from typing import Optional
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ws_tester.types import DataSnapshot, Signal


# Strategy metadata (required)
STRATEGY_NAME = "market_making"
STRATEGY_VERSION = "1.0.0"
SYMBOLS = ["XRP/USD"]

# Configuration with defaults
CONFIG = {
    'min_spread_pct': 0.1,        # Minimum spread to trade (0.1%)
    'position_size_usd': 20,      # Size per trade in USD
    'max_inventory': 100,         # Max position size in USD
    'inventory_skew': 0.5,        # Reduce size when inventory builds
    'take_profit_pct': 0.3,       # Take profit at 0.3%
    'stop_loss_pct': 0.5,         # Stop loss at 0.5%
}


def generate_signal(data: DataSnapshot, config: dict, state: dict) -> Optional[Signal]:
    """
    Generate market making signal.

    Strategy:
    - Trade when spread is wide enough to capture
    - Skew quotes based on inventory
    - Use stop-loss and take-profit
    """
    symbol = "XRP/USD"

    # Get orderbook
    ob = data.orderbooks.get(symbol)
    if not ob or not ob.best_bid or not ob.best_ask:
        return None

    price = data.prices.get(symbol, 0)
    if not price:
        return None

    # Calculate spread
    spread_pct = ob.spread_pct

    # Store indicators for logging
    state['indicators'] = {
        'spread_pct': spread_pct,
        'best_bid': ob.best_bid,
        'best_ask': ob.best_ask,
        'mid': ob.mid,
        'inventory': state.get('inventory', 0),
        'imbalance': ob.imbalance,
    }

    # Check minimum spread
    if spread_pct < config['min_spread_pct']:
        return None

    # Get current inventory (positive = long, negative = short)
    inventory = state.get('inventory', 0)
    max_inventory = config['max_inventory']

    # Calculate position size with inventory skew
    base_size = config['position_size_usd']
    skew_factor = 1.0 - abs(inventory / max_inventory) * config['inventory_skew']
    position_size = base_size * max(skew_factor, 0.1)

    # Decide action based on orderbook imbalance and inventory
    imbalance = ob.imbalance  # positive = more bids

    # If we're long and see selling pressure, consider selling
    if inventory > 0 and imbalance < -0.2:
        return Signal(
            action='sell',
            symbol=symbol,
            size=min(position_size, abs(inventory)),
            price=ob.best_bid,
            reason=f"MM: Reduce long on sell pressure (imbal={imbalance:.2f})",
            stop_loss=ob.mid * (1 - config['stop_loss_pct'] / 100),
            take_profit=ob.mid * (1 + config['take_profit_pct'] / 100),
        )

    # If we're short and see buying pressure, consider covering
    if inventory < 0 and imbalance > 0.2:
        return Signal(
            action='buy',
            symbol=symbol,
            size=min(position_size, abs(inventory)),
            price=ob.best_ask,
            reason=f"MM: Reduce short on buy pressure (imbal={imbalance:.2f})",
            stop_loss=ob.mid * (1 + config['stop_loss_pct'] / 100),
            take_profit=ob.mid * (1 - config['take_profit_pct'] / 100),
        )

    # If inventory allows, trade based on spread capture
    if inventory < max_inventory and imbalance > 0.1:
        # More bids than asks - buy side has support
        return Signal(
            action='buy',
            symbol=symbol,
            size=position_size,
            price=ob.best_ask,
            reason=f"MM: Spread capture buy (spread={spread_pct:.2f}%)",
            stop_loss=ob.mid * (1 - config['stop_loss_pct'] / 100),
            take_profit=ob.mid * (1 + config['take_profit_pct'] / 100),
        )

    if inventory > -max_inventory and imbalance < -0.1:
        # More asks than bids - potential short opportunity
        return Signal(
            action='sell',
            symbol=symbol,
            size=position_size,
            price=ob.best_bid,
            reason=f"MM: Spread capture sell (spread={spread_pct:.2f}%)",
            stop_loss=ob.mid * (1 + config['stop_loss_pct'] / 100),
            take_profit=ob.mid * (1 - config['take_profit_pct'] / 100),
        )

    return None


def on_fill(fill: dict, state: dict) -> None:
    """Update inventory on fill."""
    size_usd = fill.get('size', 0) * fill.get('price', 0)
    if fill.get('side') == 'buy':
        state['inventory'] = state.get('inventory', 0) + size_usd
    else:
        state['inventory'] = state.get('inventory', 0) - size_usd

    state['last_fill'] = fill


def on_start(config: dict, state: dict) -> None:
    """Initialize state."""
    state['inventory'] = 0
    state['indicators'] = {}
