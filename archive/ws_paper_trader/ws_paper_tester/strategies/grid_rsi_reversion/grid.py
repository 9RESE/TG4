"""
Grid RSI Reversion Strategy - Grid Management

Contains grid-specific logic for:
- Grid level setup and calculation
- Grid spacing (arithmetic/geometric)
- Grid state management
- Cycle tracking and recentering
"""
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import uuid

from .config import get_symbol_config, GridType, get_grid_type


def calculate_grid_prices(
    center_price: float,
    num_grids: int,
    spacing_pct: float,
    range_pct: float,
    grid_type: GridType
) -> Tuple[List[float], List[float]]:
    """
    Calculate grid level prices.

    Args:
        center_price: Center price for the grid
        num_grids: Number of grid levels per side
        spacing_pct: Spacing between levels (percentage)
        range_pct: Range from center (percentage)
        grid_type: ARITHMETIC or GEOMETRIC

    Returns:
        Tuple of (buy_prices, sell_prices) sorted appropriately
    """
    # Ensure float type for calculations (handles Decimal from database)
    center_price = float(center_price)
    spacing_pct = float(spacing_pct)
    range_pct = float(range_pct)

    buy_prices = []
    sell_prices = []

    if grid_type == GridType.ARITHMETIC:
        # Fixed dollar spacing
        spacing = center_price * (spacing_pct / 100)
        for i in range(1, num_grids + 1):
            buy_prices.append(center_price - (i * spacing))
            sell_prices.append(center_price + (i * spacing))
    else:  # GEOMETRIC
        # Fixed percentage spacing
        multiplier = 1 + (spacing_pct / 100)
        for i in range(1, num_grids + 1):
            buy_prices.append(center_price / (multiplier ** i))
            sell_prices.append(center_price * (multiplier ** i))

    # Sort: buy prices descending (highest buy first), sell prices ascending
    buy_prices.sort(reverse=True)
    sell_prices.sort()

    # Limit to range
    lower_limit = center_price * (1 - range_pct / 100)
    upper_limit = center_price * (1 + range_pct / 100)

    buy_prices = [p for p in buy_prices if p >= lower_limit]
    sell_prices = [p for p in sell_prices if p <= upper_limit]

    return buy_prices, sell_prices


def setup_grid_levels(
    symbol: str,
    center_price: float,
    config: Dict[str, Any],
    atr: Optional[float] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Setup grid levels for a symbol.

    Args:
        symbol: Trading symbol
        center_price: Center price for the grid
        config: Strategy configuration
        atr: Optional ATR for dynamic spacing

    Returns:
        Tuple of (grid_levels list, grid_metadata dict)
    """
    # Get symbol-specific settings
    grid_type = get_grid_type(symbol, config)
    num_grids = get_symbol_config(symbol, config, 'num_grids')
    spacing_pct = get_symbol_config(symbol, config, 'grid_spacing_pct')
    range_pct = get_symbol_config(symbol, config, 'range_pct')
    position_size = get_symbol_config(symbol, config, 'position_size_usd')

    # ATR-based spacing adjustment (optional)
    # Convert to float to handle Decimal types from database
    if config.get('use_atr_spacing', True) and atr is not None and atr > 0:
        atr_float = float(atr)
        center_float = float(center_price)
        atr_spacing_pct = (atr_float / center_float) * 100 * config.get('atr_multiplier', 0.3)
        # Use larger of configured spacing or ATR-based
        spacing_pct = max(spacing_pct, atr_spacing_pct)

    # Calculate grid prices
    buy_prices, sell_prices = calculate_grid_prices(
        center_price, num_grids, spacing_pct, range_pct, grid_type
    )

    grid_levels = []
    grid_id = str(uuid.uuid4())[:8]

    # Create buy levels
    for i, price in enumerate(buy_prices):
        level = {
            'level_index': i,
            'price': price,
            'side': 'buy',
            'size': position_size,
            'filled': False,
            'fill_price': None,
            'fill_time': None,
            'order_id': f"{symbol.replace('/', '_')}_{grid_id}_buy_{i}",
            'matched_order_id': f"{symbol.replace('/', '_')}_{grid_id}_sell_{i}",
            'original_side': 'buy',
        }
        grid_levels.append(level)

    # Create sell levels (for closing buy positions)
    for i, price in enumerate(sell_prices):
        level = {
            'level_index': i + len(buy_prices),
            'price': price,
            'side': 'sell',
            'size': position_size,
            'filled': False,
            'fill_price': None,
            'fill_time': None,
            'order_id': f"{symbol.replace('/', '_')}_{grid_id}_sell_{i}",
            'matched_order_id': f"{symbol.replace('/', '_')}_{grid_id}_buy_{i}",
            'original_side': 'sell',
        }
        grid_levels.append(level)

    # Calculate grid boundaries
    lower_price = min(buy_prices) if buy_prices else center_price * 0.9
    upper_price = max(sell_prices) if sell_prices else center_price * 1.1

    metadata = {
        'center_price': center_price,
        'upper_price': upper_price,
        'lower_price': lower_price,
        'grid_spacing_pct': spacing_pct,
        'num_buy_levels': len(buy_prices),
        'num_sell_levels': len(sell_prices),
        'last_recenter_time': datetime.now(),
        'cycles_completed': 0,
        'cycles_at_last_recenter': 0,
        'grid_id': grid_id,
    }

    return grid_levels, metadata


def get_unfilled_levels(
    grid_levels: List[Dict[str, Any]],
    side: str
) -> List[Dict[str, Any]]:
    """
    Get unfilled grid levels for a specific side.

    Args:
        grid_levels: List of grid level dicts
        side: 'buy' or 'sell'

    Returns:
        List of unfilled levels for the specified side
    """
    return [
        level for level in grid_levels
        if level['side'] == side and not level['filled']
    ]


def get_nearest_unfilled_level(
    grid_levels: List[Dict[str, Any]],
    current_price: float,
    side: str,
    tolerance_pct: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Get the nearest unfilled grid level to current price.

    Args:
        grid_levels: List of grid level dicts
        current_price: Current market price
        side: 'buy' or 'sell'
        tolerance_pct: Price tolerance percentage

    Returns:
        Nearest unfilled level within tolerance, or None
    """
    unfilled = get_unfilled_levels(grid_levels, side)
    if not unfilled:
        return None

    tolerance = current_price * (tolerance_pct / 100)

    # Find levels within tolerance
    candidates = []
    for level in unfilled:
        price_diff = abs(current_price - level['price'])
        if price_diff <= tolerance:
            candidates.append((level, price_diff))

    if not candidates:
        return None

    # Return the closest one
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def check_price_at_grid_level(
    grid_levels: List[Dict[str, Any]],
    current_price: float,
    side: str,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Check if current price is at a grid level.

    For buy signals: price should be at or below the grid level price
    For sell signals: price should be at or above the grid level price

    Args:
        grid_levels: List of grid level dicts
        current_price: Current market price
        side: 'buy' or 'sell'
        config: Strategy configuration

    Returns:
        Grid level if price matches, None otherwise
    """
    tolerance_pct = config.get('slippage_tolerance_pct', 0.5)
    unfilled = get_unfilled_levels(grid_levels, side)

    for level in unfilled:
        tolerance = level['price'] * (tolerance_pct / 100)

        if side == 'buy':
            # Buy: price at or below level (within tolerance)
            if current_price <= level['price'] + tolerance:
                return level
        else:  # sell
            # Sell: price at or above level (within tolerance)
            if current_price >= level['price'] - tolerance:
                return level

    return None


def mark_level_filled(
    grid_levels: List[Dict[str, Any]],
    order_id: str,
    fill_price: float,
    fill_time: datetime
) -> bool:
    """
    Mark a grid level as filled.

    Args:
        grid_levels: List of grid level dicts
        order_id: Order ID of the level to mark
        fill_price: Actual fill price
        fill_time: Fill timestamp

    Returns:
        True if level was found and marked, False otherwise
    """
    for level in grid_levels:
        if level['order_id'] == order_id:
            level['filled'] = True
            level['fill_price'] = fill_price
            level['fill_time'] = fill_time
            return True
    return False


def count_filled_levels(
    grid_levels: List[Dict[str, Any]],
    side: str
) -> int:
    """
    Count filled grid levels for a side.

    Args:
        grid_levels: List of grid level dicts
        side: 'buy' or 'sell'

    Returns:
        Count of filled levels
    """
    return len([
        level for level in grid_levels
        if level['side'] == side and level['filled']
    ])


def check_cycle_completion(
    grid_levels: List[Dict[str, Any]],
    order_id: str
) -> Optional[Dict[str, Any]]:
    """
    Check if a fill completes a buy-sell cycle.

    Args:
        grid_levels: List of grid level dicts
        order_id: Order ID of the filled order

    Returns:
        The matching order if cycle completed, None otherwise
    """
    # Find the filled order
    filled_order = None
    for level in grid_levels:
        if level['order_id'] == order_id:
            filled_order = level
            break

    if not filled_order:
        return None

    # Find the matching order
    matched_id = filled_order.get('matched_order_id')
    if not matched_id:
        return None

    for level in grid_levels:
        if level['order_id'] == matched_id and level['filled']:
            return level

    return None


def should_recenter_grid(
    metadata: Dict[str, Any],
    current_time: datetime,
    config: Dict[str, Any],
    adx: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Check if grid should be recentered.

    REC-008: Enhanced with trend check before recentering.

    Recentering conditions:
    1. Completed cycles exceeds threshold since last recenter
    2. Minimum time has passed since last recenter
    3. REC-008: Not in a strong trend (ADX < threshold)

    Args:
        metadata: Grid metadata dict
        current_time: Current timestamp
        config: Strategy configuration
        adx: Current ADX value (REC-008)

    Returns:
        Tuple of (should_recenter, reason)
    """
    cycles_since_recenter = metadata.get('cycles_completed', 0) - metadata.get('cycles_at_last_recenter', 0)
    recenter_threshold = config.get('recenter_after_cycles', 5)

    if cycles_since_recenter < recenter_threshold:
        return False, f'insufficient_cycles ({cycles_since_recenter}/{recenter_threshold})'

    # Check minimum time interval
    last_recenter = metadata.get('last_recenter_time')
    if last_recenter:
        elapsed = (current_time - last_recenter).total_seconds()
        min_interval = config.get('min_recenter_interval', 3600)
        if elapsed < min_interval:
            return False, f'cooldown_active ({elapsed/60:.0f}/{min_interval/60:.0f}min)'

    # REC-008: Check trend before recentering
    # Don't recenter in strong trends - grid will likely break out again
    if config.get('check_trend_before_recenter', True) and adx is not None:
        adx_recenter_threshold = config.get('adx_recenter_threshold', 25)
        if adx > adx_recenter_threshold:
            return False, f'trending_market (ADX={adx:.1f}>{adx_recenter_threshold})'

    return True, f'recenter_ready (cycles={cycles_since_recenter})'


def recenter_grid(
    symbol: str,
    new_center_price: float,
    config: Dict[str, Any],
    old_metadata: Dict[str, Any],
    atr: Optional[float] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Recenter grid around a new price.

    Args:
        symbol: Trading symbol
        new_center_price: New center price
        config: Strategy configuration
        old_metadata: Previous grid metadata
        atr: Optional ATR for dynamic spacing

    Returns:
        Tuple of (new_grid_levels, new_metadata)
    """
    new_levels, new_metadata = setup_grid_levels(
        symbol, new_center_price, config, atr
    )

    # Preserve statistics
    new_metadata['cycles_completed'] = old_metadata.get('cycles_completed', 0)
    new_metadata['cycles_at_last_recenter'] = new_metadata['cycles_completed']
    new_metadata['last_recenter_time'] = datetime.now()

    return new_levels, new_metadata


def calculate_grid_stats(
    grid_levels: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate grid statistics.

    Args:
        grid_levels: List of grid level dicts
        metadata: Grid metadata

    Returns:
        Dict of grid statistics
    """
    buy_levels = [l for l in grid_levels if l['original_side'] == 'buy']
    sell_levels = [l for l in grid_levels if l['original_side'] == 'sell']

    filled_buys = count_filled_levels(grid_levels, 'buy')
    filled_sells = count_filled_levels(grid_levels, 'sell')

    # Calculate average fill prices
    buy_fill_prices = [l['fill_price'] for l in buy_levels if l['filled'] and l['fill_price']]
    sell_fill_prices = [l['fill_price'] for l in sell_levels if l['filled'] and l['fill_price']]

    avg_buy_price = sum(buy_fill_prices) / len(buy_fill_prices) if buy_fill_prices else None
    avg_sell_price = sum(sell_fill_prices) / len(sell_fill_prices) if sell_fill_prices else None

    return {
        'total_levels': len(grid_levels),
        'buy_levels': len(buy_levels),
        'sell_levels': len(sell_levels),
        'filled_buys': filled_buys,
        'filled_sells': filled_sells,
        'unfilled_buys': len(buy_levels) - filled_buys,
        'unfilled_sells': len(sell_levels) - filled_sells,
        'cycles_completed': metadata.get('cycles_completed', 0),
        'grid_spacing_pct': metadata.get('grid_spacing_pct'),
        'center_price': metadata.get('center_price'),
        'upper_price': metadata.get('upper_price'),
        'lower_price': metadata.get('lower_price'),
        'avg_buy_price': avg_buy_price,
        'avg_sell_price': avg_sell_price,
    }
