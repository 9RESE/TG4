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
- 1.4.0: Enhancements per market-making-strategy-review-v1.3.md
         - Added config validation on startup
         - Optional Avellaneda-Stoikov reservation price model
         - Trailing stop support
         - Enhanced per-pair metrics tracking
- 1.5.0: All recommendations from market-making-strategy-review-v1.4.md
         - MM-E03: Fee-aware profitability check
         - MM-009: Adjusted R:R ratios for 1:1 on XRP pairs
         - MM-E01: Micro-price calculation for better price discovery
         - MM-E02: Optimal spread calculation (A-S style)
         - MM-010: Refactored _evaluate_symbol into smaller functions
         - MM-011: Configurable fallback prices (no hardcoding)
         - MM-E04: Time-based position decay for stale positions
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import math

from ws_tester.types import DataSnapshot, Signal, OrderbookSnapshot


# =============================================================================
# REQUIRED: Strategy Metadata
# =============================================================================
STRATEGY_NAME = "market_making"
STRATEGY_VERSION = "1.5.0"
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

    # Risk management (MM-009: improved R:R to 1:1)
    'take_profit_pct': 0.5,       # Take profit at 0.5% (was 0.4%)
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

    # v1.4.0: Avellaneda-Stoikov reservation price model (optional)
    'use_reservation_price': False,  # Enable A-S style quote adjustment
    'gamma': 0.1,                    # Risk aversion parameter (0.01-1.0)
    # Higher gamma = more aggressive inventory reduction

    # v1.4.0: Trailing stop support
    'use_trailing_stop': False,      # Enable trailing stops
    'trailing_stop_activation': 0.2, # Activate trailing after 0.2% profit
    'trailing_stop_distance': 0.15,  # Trail at 0.15% from high

    # v1.5.0: Fee-aware profitability (MM-E03)
    'fee_rate': 0.001,               # 0.1% per trade (0.2% round-trip)
    'min_profit_after_fees_pct': 0.05,  # Minimum profit after fees (0.05%)
    'use_fee_check': True,           # Enable fee-aware profitability check

    # v1.5.0: Micro-price (MM-E01)
    'use_micro_price': True,         # Use volume-weighted micro-price

    # v1.5.0: Optimal spread calculation (MM-E02)
    'use_optimal_spread': False,     # Enable A-S optimal spread calculation
    'kappa': 1.5,                    # Market liquidity parameter

    # v1.5.0: Fallback prices (MM-011)
    'fallback_xrp_usdt': 2.50,       # Fallback XRP/USDT price if unavailable

    # v1.5.0: Position decay (MM-E04)
    'use_position_decay': True,      # Enable time-based position decay
    'max_position_age_seconds': 300, # Max age before widening TP (5 minutes)
    'position_decay_tp_multiplier': 0.5,  # Reduce TP by 50% for stale positions
}

# Per-symbol configurations (MM-009: adjusted R:R ratios)
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'min_spread_pct': 0.05,       # Tighter spreads on USDT pair
        'position_size_usd': 20,
        'max_inventory': 100,
        'imbalance_threshold': 0.1,
        'take_profit_pct': 0.5,       # MM-009: changed from 0.4 to 0.5 for 1:1
        'stop_loss_pct': 0.5,
        'cooldown_seconds': 5.0,
    },
    'BTC/USDT': {
        # BTC/USDT market making - high liquidity pair
        # - Very tight spreads, high volume
        # - Larger position sizes (BTC trades bigger)
        'min_spread_pct': 0.03,       # Tighter min spread (more liquid)
        'position_size_usd': 50,      # Larger size for BTC
        'max_inventory': 200,         # Higher max inventory
        'imbalance_threshold': 0.08,  # Lower threshold (more liquid)
        'take_profit_pct': 0.35,      # 1:1 R:R
        'stop_loss_pct': 0.35,
        'cooldown_seconds': 3.0,      # Faster for BTC
    },
    'XRP/BTC': {
        # XRP/BTC market making - optimized from Kraken data:
        # - 664 trades/day, 0.0446% spread
        # - Goal: Accumulate both XRP and BTC through spread capture
        'min_spread_pct': 0.03,       # Lower threshold (spread is 0.0446%)
        'position_size_xrp': 25,      # Size in XRP (converted to USD in signal)
        'max_inventory_xrp': 150,     # Max XRP exposure
        'imbalance_threshold': 0.15,  # Slightly higher (less liquid)
        'take_profit_pct': 0.4,       # MM-009: changed from 0.3 to 0.4 for 1:1
        'stop_loss_pct': 0.4,
        'cooldown_seconds': 10.0,     # Slower for cross-pair
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


def _validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration on startup.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Required positive values
    required_positive = [
        'position_size_usd',
        'max_inventory',
        'stop_loss_pct',
        'take_profit_pct',
        'cooldown_seconds',
    ]

    for key in required_positive:
        val = config.get(key)
        if val is None:
            errors.append(f"Missing required config: {key}")
        elif val <= 0:
            errors.append(f"{key} must be positive, got {val}")

    # Optional values with bounds
    gamma = config.get('gamma', 0.1)
    if gamma < 0.01 or gamma > 1.0:
        errors.append(f"gamma must be between 0.01 and 1.0, got {gamma}")

    inventory_skew = config.get('inventory_skew', 0.5)
    if inventory_skew < 0 or inventory_skew > 1.0:
        errors.append(f"inventory_skew must be between 0 and 1.0, got {inventory_skew}")

    # Warn about risky R:R ratios
    sl = config.get('stop_loss_pct', 0.5)
    tp = config.get('take_profit_pct', 0.5)
    if sl > 0 and tp > 0:
        rr_ratio = tp / sl
        if rr_ratio < 0.5:
            errors.append(f"Warning: Poor R:R ratio ({rr_ratio:.2f}:1), requires {100/(1+rr_ratio):.0f}% win rate")

    # v1.5.0: Validate fee settings
    fee_rate = config.get('fee_rate', 0.001)
    if fee_rate < 0 or fee_rate > 0.01:
        errors.append(f"fee_rate should be between 0 and 0.01, got {fee_rate}")

    return errors


def _calculate_micro_price(ob: OrderbookSnapshot) -> float:
    """
    Calculate volume-weighted micro-price (MM-E01).

    Micro-price provides better price discovery than simple mid-price
    by weighting by order sizes at best bid/ask.

    Formula: micro = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
    """
    if not ob.bids or not ob.asks:
        return ob.mid

    best_bid, bid_size = ob.bids[0]
    best_ask, ask_size = ob.asks[0]

    total_size = bid_size + ask_size
    if total_size <= 0:
        return ob.mid

    micro_price = (best_bid * ask_size + best_ask * bid_size) / total_size
    return micro_price


def _calculate_optimal_spread(
    volatility_pct: float,
    gamma: float,
    kappa: float,
    time_horizon: float = 1.0
) -> float:
    """
    Calculate Avellaneda-Stoikov optimal spread (MM-E02).

    Formula: optimal_spread = γ * σ² * T + (2/γ) * ln(1 + γ/κ)

    Args:
        volatility_pct: Price volatility as percentage
        gamma: Risk aversion parameter
        kappa: Market liquidity parameter
        time_horizon: Time horizon (normalized, default 1.0)

    Returns:
        Optimal spread as percentage
    """
    if gamma <= 0 or kappa <= 0:
        return 0.0

    sigma = volatility_pct / 100  # Convert to decimal
    sigma_sq = sigma ** 2

    # A-S optimal spread formula
    inventory_term = gamma * sigma_sq * time_horizon * 100  # Convert back to %
    liquidity_term = (2 / gamma) * math.log(1 + gamma / kappa) * 100

    return inventory_term + liquidity_term


def _calculate_reservation_price(
    mid_price: float,
    inventory: float,
    max_inventory: float,
    gamma: float,
    volatility_pct: float
) -> float:
    """
    Calculate Avellaneda-Stoikov reservation price.

    The reservation price adjusts the mid price based on inventory risk:
    r = s - q * γ * σ²

    Where:
    - s: mid price
    - q: normalized inventory (-1 to 1)
    - γ: risk aversion parameter
    - σ²: variance of price (volatility squared)

    Returns:
        Adjusted reservation price
    """
    if max_inventory <= 0:
        return mid_price

    # Normalize inventory to -1 to 1 range
    q = inventory / max_inventory

    # Convert volatility percentage to decimal variance
    sigma_sq = (volatility_pct / 100) ** 2

    # Calculate reservation price
    # Positive inventory (long) -> lower reservation price (favor selling)
    # Negative inventory (short) -> higher reservation price (favor buying)
    reservation = mid_price * (1 - q * gamma * sigma_sq * 100)

    return reservation


def _calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    side: str,
    activation_pct: float,
    trail_distance_pct: float
) -> Optional[float]:
    """
    Calculate trailing stop level.

    Args:
        entry_price: Original entry price
        highest_price: Highest price since entry (for longs) or lowest (for shorts)
        side: 'long' or 'short'
        activation_pct: Minimum profit % to activate trailing
        trail_distance_pct: Distance from high/low to trail

    Returns:
        Trailing stop price or None if not activated
    """
    if side == 'long':
        # Long: profit when price increases
        profit_pct = (highest_price - entry_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)
    elif side == 'short':
        # Short: profit when price decreases (highest_price is actually lowest)
        profit_pct = (entry_price - highest_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 + trail_distance_pct / 100)

    return None


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


def _check_fee_profitability(
    spread_pct: float,
    fee_rate: float,
    min_profit_pct: float
) -> Tuple[bool, float]:
    """
    Check if trade is profitable after fees (MM-E03).

    Args:
        spread_pct: Current spread as percentage
        fee_rate: Fee per trade (e.g., 0.001 for 0.1%)
        min_profit_pct: Minimum required profit after fees

    Returns:
        Tuple of (is_profitable, expected_profit_pct)
    """
    # Round-trip fees (entry + exit)
    round_trip_fee_pct = fee_rate * 2 * 100  # Convert to percentage

    # Expected profit from spread capture (we capture half the spread)
    expected_capture = spread_pct / 2

    # Net profit after fees
    net_profit_pct = expected_capture - round_trip_fee_pct

    is_profitable = net_profit_pct >= min_profit_pct

    return is_profitable, net_profit_pct


def _check_position_decay(
    position_entry: Dict[str, Any],
    current_time: datetime,
    max_age_seconds: float,
    tp_multiplier: float
) -> Tuple[bool, float]:
    """
    Check if position is stale and should have adjusted TP (MM-E04).

    Args:
        position_entry: Position entry data with 'entry_time'
        current_time: Current timestamp
        max_age_seconds: Maximum age before decay kicks in
        tp_multiplier: Multiplier to reduce TP (e.g., 0.5 = 50% of original)

    Returns:
        Tuple of (is_stale, adjusted_tp_multiplier)
    """
    entry_time = position_entry.get('entry_time')
    if not entry_time:
        return False, 1.0

    age_seconds = (current_time - entry_time).total_seconds()

    if age_seconds > max_age_seconds:
        # Position is stale - reduce TP requirement
        return True, tp_multiplier

    return False, 1.0


def _get_xrp_usdt_price(data: DataSnapshot, config: Dict[str, Any]) -> float:
    """
    Get XRP/USDT price with configurable fallback (MM-011).

    Args:
        data: Market data snapshot
        config: Strategy configuration

    Returns:
        XRP/USDT price
    """
    price = data.prices.get('XRP/USDT')
    if price and price > 0:
        return price

    # Use configurable fallback instead of hardcoded value
    return config.get('fallback_xrp_usdt', 2.50)


# =============================================================================
# Refactored Signal Generation Functions (MM-010)
# =============================================================================
def _calculate_effective_thresholds(
    config: Dict[str, Any],
    symbol: str,
    volatility: float
) -> Tuple[float, float, float]:
    """
    Calculate volatility-adjusted thresholds (MM-010 refactor).

    Returns:
        Tuple of (effective_min_spread, effective_imbalance_threshold, vol_multiplier)
    """
    min_spread = _get_symbol_config(symbol, config, 'min_spread_pct')
    imbalance_threshold = _get_symbol_config(symbol, config, 'imbalance_threshold')
    base_vol = config.get('base_volatility_pct', 0.5)

    vol_multiplier = 1.0
    if base_vol > 0 and volatility > 0:
        vol_ratio = volatility / base_vol
        if vol_ratio > 1.0:
            vol_multiplier = min(vol_ratio, config.get('volatility_threshold_mult', 1.5))

    effective_threshold = imbalance_threshold * vol_multiplier
    effective_min_spread = min_spread * vol_multiplier

    return effective_min_spread, effective_threshold, vol_multiplier


def _check_trailing_stop_exit(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    price: float,
    ob: OrderbookSnapshot,
    inventory: float,
    is_cross_pair: bool
) -> Optional[Signal]:
    """
    Check if trailing stop should trigger exit (MM-010 refactor).

    Returns:
        Signal if trailing stop triggered, None otherwise
    """
    use_trailing = config.get('use_trailing_stop', False)
    if not use_trailing:
        return None

    if 'position_entries' not in state:
        return None

    pos_entry = state['position_entries'].get(symbol)
    if not pos_entry:
        return None

    trailing_activation = config.get('trailing_stop_activation', 0.2)
    trailing_distance = config.get('trailing_stop_distance', 0.15)

    # Update highest/lowest price for tracking
    if pos_entry['side'] == 'long':
        pos_entry['highest_price'] = max(pos_entry['highest_price'], price)
        tracking_price = pos_entry['highest_price']
    else:
        pos_entry['lowest_price'] = min(pos_entry['lowest_price'], price)
        tracking_price = pos_entry['lowest_price']

    trailing_stop_price = _calculate_trailing_stop(
        entry_price=pos_entry['entry_price'],
        highest_price=tracking_price,
        side=pos_entry['side'],
        activation_pct=trailing_activation,
        trail_distance_pct=trailing_distance
    )

    if trailing_stop_price is None:
        return None

    # Check if trailing stop is triggered
    if pos_entry['side'] == 'long' and price <= trailing_stop_price:
        close_size = abs(inventory)
        xrp_usdt_price = _get_xrp_usdt_price(data, config)
        if is_cross_pair:
            close_size = close_size * xrp_usdt_price

        return Signal(
            action='sell',
            symbol=symbol,
            size=close_size,
            price=ob.best_bid,
            reason=f"MM: Trailing stop hit (entry={pos_entry['entry_price']:.6f}, high={pos_entry['highest_price']:.6f}, trail={trailing_stop_price:.6f})",
            metadata={'trailing_stop': True, 'xrp_size': close_size / xrp_usdt_price if is_cross_pair else None},
        )

    elif pos_entry['side'] == 'short' and price >= trailing_stop_price:
        close_size = abs(inventory)
        xrp_usdt_price = _get_xrp_usdt_price(data, config)
        if is_cross_pair:
            close_size = close_size * xrp_usdt_price

        return Signal(
            action='cover',
            symbol=symbol,
            size=close_size,
            price=ob.best_ask,
            reason=f"MM: Trailing stop hit (entry={pos_entry['entry_price']:.6f}, low={pos_entry['lowest_price']:.6f}, trail={trailing_stop_price:.6f})",
            metadata={'trailing_stop': True, 'xrp_size': close_size / xrp_usdt_price if is_cross_pair else None},
        )

    return None


def _check_position_decay_exit(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    price: float,
    ob: OrderbookSnapshot,
    inventory: float,
    is_cross_pair: bool,
    current_time: datetime
) -> Optional[Signal]:
    """
    Check if stale position should be closed with reduced TP (MM-E04).

    Returns:
        Signal if stale position should exit, None otherwise
    """
    use_decay = config.get('use_position_decay', True)
    if not use_decay:
        return None

    if 'position_entries' not in state:
        return None

    pos_entry = state['position_entries'].get(symbol)
    if not pos_entry:
        return None

    max_age = config.get('max_position_age_seconds', 300)
    tp_mult = config.get('position_decay_tp_multiplier', 0.5)

    is_stale, adjusted_mult = _check_position_decay(pos_entry, current_time, max_age, tp_mult)

    if not is_stale:
        return None

    # Check if we're in profit and should exit with reduced TP
    entry_price = pos_entry['entry_price']
    tp_pct = _get_symbol_config(symbol, config, 'take_profit_pct')
    adjusted_tp_pct = tp_pct * adjusted_mult

    if pos_entry['side'] == 'long':
        profit_pct = (price - entry_price) / entry_price * 100
        if profit_pct >= adjusted_tp_pct:
            close_size = abs(inventory)
            xrp_usdt_price = _get_xrp_usdt_price(data, config)
            if is_cross_pair:
                close_size = close_size * xrp_usdt_price

            return Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=ob.best_bid,
                reason=f"MM: Stale position exit (age>{max_age}s, profit={profit_pct:.2f}%, adj_tp={adjusted_tp_pct:.2f}%)",
                metadata={'position_decay': True, 'xrp_size': close_size / xrp_usdt_price if is_cross_pair else None},
            )

    elif pos_entry['side'] == 'short':
        profit_pct = (entry_price - price) / entry_price * 100
        if profit_pct >= adjusted_tp_pct:
            close_size = abs(inventory)
            xrp_usdt_price = _get_xrp_usdt_price(data, config)
            if is_cross_pair:
                close_size = close_size * xrp_usdt_price

            return Signal(
                action='cover',
                symbol=symbol,
                size=close_size,
                price=ob.best_ask,
                reason=f"MM: Stale position exit (age>{max_age}s, profit={profit_pct:.2f}%, adj_tp={adjusted_tp_pct:.2f}%)",
                metadata={'position_decay': True, 'xrp_size': close_size / xrp_usdt_price if is_cross_pair else None},
            )

    return None


def _build_entry_signal(
    symbol: str,
    action: str,
    size: float,
    entry_price: float,
    reason: str,
    sl_pct: float,
    tp_pct: float,
    is_cross_pair: bool,
    xrp_usdt_price: float
) -> Signal:
    """
    Build a trading signal with proper stop/TP levels (MM-010 refactor).

    Returns:
        Constructed Signal object
    """
    # Calculate stop and take profit based on action direction
    if action in ('buy', 'cover'):
        stop_loss = entry_price * (1 - sl_pct / 100)
        take_profit = entry_price * (1 + tp_pct / 100)
    else:  # sell, short
        stop_loss = entry_price * (1 + sl_pct / 100)
        take_profit = entry_price * (1 - tp_pct / 100)

    # Calculate XRP size for cross-pair
    metadata = None
    if is_cross_pair:
        xrp_size = size / xrp_usdt_price
        metadata = {'xrp_size': xrp_size}

    return Signal(
        action=action,
        symbol=symbol,
        size=size,
        price=entry_price,
        reason=reason,
        stop_loss=stop_loss,
        take_profit=take_profit,
        metadata=metadata,
    )


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
    - MM-E03: Check fee profitability before entry
    - MM-E01: Use micro-price for better price discovery
    - MM-E04: Handle stale positions with decay

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

    is_cross_pair = _is_xrp_btc(symbol)
    xrp_usdt_price = _get_xrp_usdt_price(data, config)

    # MM-E01: Calculate micro-price for better price discovery
    use_micro = config.get('use_micro_price', True)
    if use_micro:
        micro_price = _calculate_micro_price(ob)
    else:
        micro_price = ob.mid

    # Calculate spread
    spread_pct = ob.spread_pct

    # Get symbol-specific config
    tp_pct = _get_symbol_config(symbol, config, 'take_profit_pct')
    sl_pct = _get_symbol_config(symbol, config, 'stop_loss_pct')

    # MM-002: Calculate volatility
    candles = data.candles_1m.get(symbol, ())
    volatility = _calculate_volatility(candles, config.get('volatility_lookback', 20))

    # Calculate effective thresholds (MM-010 refactor)
    effective_min_spread, effective_threshold, vol_multiplier = _calculate_effective_thresholds(
        config, symbol, volatility
    )

    # MM-E02: Calculate optimal spread if enabled
    use_optimal = config.get('use_optimal_spread', False)
    optimal_spread = 0.0
    if use_optimal and volatility > 0:
        gamma = config.get('gamma', 0.1)
        kappa = config.get('kappa', 1.5)
        optimal_spread = _calculate_optimal_spread(volatility, gamma, kappa)
        # Use maximum of configured min spread and A-S optimal spread
        effective_min_spread = max(effective_min_spread, optimal_spread)

    # MM-007: Get trade flow imbalance for confirmation
    trade_flow = _get_trade_flow_imbalance(data, symbol, 50)
    trade_flow_threshold = config.get('trade_flow_threshold', 0.15)
    use_trade_flow = config.get('use_trade_flow', True)

    # Get inventory (different units for XRP/BTC)
    if 'inventory_by_symbol' not in state:
        state['inventory_by_symbol'] = {}

    if is_cross_pair:
        inventory = state['inventory_by_symbol'].get(symbol, 0)  # in XRP
        max_inventory = _get_symbol_config(symbol, config, 'max_inventory_xrp') or 150
        base_size_xrp = _get_symbol_config(symbol, config, 'position_size_xrp') or 25
        base_size = base_size_xrp * xrp_usdt_price
    else:
        inventory = state['inventory_by_symbol'].get(symbol, 0)  # in USD
        max_inventory = _get_symbol_config(symbol, config, 'max_inventory') or 100
        base_size = _get_symbol_config(symbol, config, 'position_size_usd') or 20

    # v1.4.0: Calculate reservation price if enabled
    use_reservation = config.get('use_reservation_price', False)
    gamma = config.get('gamma', 0.1)
    reservation_price = micro_price  # Use micro-price as base

    if use_reservation and volatility > 0:
        reservation_price = _calculate_reservation_price(
            mid_price=micro_price,
            inventory=inventory,
            max_inventory=max_inventory,
            gamma=gamma,
            volatility_pct=volatility
        )

    # MM-E03: Check fee profitability
    use_fee_check = config.get('use_fee_check', True)
    fee_rate = config.get('fee_rate', 0.001)
    min_profit_pct = config.get('min_profit_after_fees_pct', 0.05)

    is_fee_profitable = True
    expected_profit = spread_pct / 2
    if use_fee_check:
        is_fee_profitable, expected_profit = _check_fee_profitability(
            spread_pct, fee_rate, min_profit_pct
        )

    # Check trailing stop exit (refactored)
    trailing_signal = _check_trailing_stop_exit(
        data, config, state, symbol, price, ob, inventory, is_cross_pair
    )
    if trailing_signal:
        return trailing_signal

    # Check position decay exit (MM-E04)
    decay_signal = _check_position_decay_exit(
        data, config, state, symbol, price, ob, inventory, is_cross_pair, current_time
    )
    if decay_signal:
        return decay_signal

    # Get trailing stop price for logging
    trailing_stop_price = None
    if config.get('use_trailing_stop', False) and 'position_entries' in state:
        pos_entry = state['position_entries'].get(symbol)
        if pos_entry:
            tracking_price = pos_entry.get('highest_price' if pos_entry['side'] == 'long' else 'lowest_price', price)
            trailing_stop_price = _calculate_trailing_stop(
                entry_price=pos_entry['entry_price'],
                highest_price=tracking_price,
                side=pos_entry['side'],
                activation_pct=config.get('trailing_stop_activation', 0.2),
                trail_distance_pct=config.get('trailing_stop_distance', 0.15)
            )

    # MM-008: Enhanced indicator logging
    state['indicators'] = {
        'symbol': symbol,
        'spread_pct': round(spread_pct, 4),
        'min_spread_pct': round(_get_symbol_config(symbol, config, 'min_spread_pct'), 4),
        'effective_min_spread': round(effective_min_spread, 4),
        'best_bid': round(ob.best_bid, 8),
        'best_ask': round(ob.best_ask, 8),
        'mid': round(ob.mid, 8),
        'micro_price': round(micro_price, 8),  # v1.5.0
        'inventory': round(inventory, 4),
        'max_inventory': max_inventory,
        'imbalance': round(ob.imbalance, 4),
        'effective_threshold': round(effective_threshold, 4),
        'is_cross_pair': is_cross_pair,
        # Volatility metrics
        'volatility_pct': round(volatility, 4),
        'vol_multiplier': round(vol_multiplier, 2),
        'optimal_spread': round(optimal_spread, 4) if use_optimal else None,  # v1.5.0
        # Trade flow
        'trade_flow': round(trade_flow, 4),
        'trade_flow_aligned': False,
        # Fee profitability (v1.5.0)
        'is_fee_profitable': is_fee_profitable,
        'expected_profit_pct': round(expected_profit, 4),
        # Reservation price and trailing stop
        'reservation_price': round(reservation_price, 8) if use_reservation else None,
        'trailing_stop_price': round(trailing_stop_price, 8) if trailing_stop_price else None,
        # Per-pair metrics
        'pnl_symbol': state.get('pnl_by_symbol', {}).get(symbol, 0),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
    }

    # Check minimum spread (with volatility adjustment)
    if spread_pct < effective_min_spread:
        return None

    # MM-E03: Skip if not profitable after fees
    if use_fee_check and not is_fee_profitable:
        return None

    # Calculate position size with inventory skew
    skew_factor = 1.0 - abs(inventory / max_inventory) * config.get('inventory_skew', 0.5)
    position_size = base_size * max(skew_factor, 0.1)

    # Minimum trade size check (in USD)
    min_size = 5.0
    if position_size < min_size:
        return None

    # Decide action based on orderbook imbalance and inventory
    imbalance = ob.imbalance

    # MM-007: Trade flow alignment check
    def is_trade_flow_aligned(direction: str) -> bool:
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
            close_size = close_size * xrp_usdt_price
        if close_size >= min_size:
            entry_price = ob.best_bid
            state['indicators']['trade_flow_aligned'] = is_trade_flow_aligned('sell')
            return _build_entry_signal(
                symbol=symbol,
                action='sell',
                size=close_size,
                entry_price=entry_price,
                reason=f"MM: Reduce long on sell pressure (imbal={imbalance:.2f})",
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                is_cross_pair=is_cross_pair,
                xrp_usdt_price=xrp_usdt_price,
            )

    # If we're short and see buying pressure, cover
    if inventory < 0 and imbalance > 0.2:
        cover_size = min(position_size, abs(inventory))
        if is_cross_pair:
            cover_size = cover_size * xrp_usdt_price
        if cover_size >= min_size:
            entry_price = ob.best_ask
            state['indicators']['trade_flow_aligned'] = is_trade_flow_aligned('buy')
            return _build_entry_signal(
                symbol=symbol,
                action='buy',
                size=cover_size,
                entry_price=entry_price,
                reason=f"MM: Reduce short on buy pressure (imbal={imbalance:.2f})",
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                is_cross_pair=is_cross_pair,
                xrp_usdt_price=xrp_usdt_price,
            )

    # If inventory allows, trade based on spread capture
    if inventory < max_inventory and imbalance > effective_threshold:
        if not is_trade_flow_aligned('buy'):
            state['indicators']['trade_flow_aligned'] = False
            return None

        state['indicators']['trade_flow_aligned'] = True
        entry_price = ob.best_ask

        return _build_entry_signal(
            symbol=symbol,
            action='buy',
            size=position_size,
            entry_price=entry_price,
            reason=f"MM: Spread capture buy (spread={spread_pct:.3f}%, imbal={imbalance:.2f}, vol={volatility:.2f}%, profit={expected_profit:.3f}%)",
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            is_cross_pair=is_cross_pair,
            xrp_usdt_price=xrp_usdt_price,
        )

    if inventory > -max_inventory and imbalance < -effective_threshold:
        if not is_trade_flow_aligned('sell'):
            state['indicators']['trade_flow_aligned'] = False
            return None

        state['indicators']['trade_flow_aligned'] = True

        if inventory > 0:
            # We have a long position - sell to reduce
            close_size = min(position_size, inventory)
            if is_cross_pair:
                close_size_usd = close_size * xrp_usdt_price
            else:
                close_size_usd = close_size

            if close_size_usd >= min_size:
                entry_price = ob.best_bid
                return _build_entry_signal(
                    symbol=symbol,
                    action='sell',
                    size=close_size_usd,
                    entry_price=entry_price,
                    reason=f"MM: Reduce long on imbalance (spread={spread_pct:.3f}%, vol={volatility:.2f}%)",
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    is_cross_pair=is_cross_pair,
                    xrp_usdt_price=xrp_usdt_price,
                )
        else:
            # We're flat or short
            if is_cross_pair:
                # For XRP/BTC, we don't short - just sell to accumulate BTC
                entry_price = ob.best_bid
                return _build_entry_signal(
                    symbol=symbol,
                    action='sell',
                    size=position_size,
                    entry_price=entry_price,
                    reason=f"MM: Sell XRP for BTC (spread={spread_pct:.3f}%, imbal={imbalance:.2f})",
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    is_cross_pair=is_cross_pair,
                    xrp_usdt_price=xrp_usdt_price,
                )
            else:
                # For USD pairs, open short position
                entry_price = ob.best_bid
                return _build_entry_signal(
                    symbol=symbol,
                    action='short',
                    size=position_size,
                    entry_price=entry_price,
                    reason=f"MM: Spread capture short (spread={spread_pct:.3f}%, vol={volatility:.2f}%, profit={expected_profit:.3f}%)",
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    is_cross_pair=is_cross_pair,
                    xrp_usdt_price=xrp_usdt_price,
                )

    return None


# =============================================================================
# OPTIONAL: Lifecycle Callbacks
# =============================================================================
def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Initialize state and validate configuration.

    v1.4.0: Added config validation and trailing stop tracking.
    v1.5.0: Enhanced with all v1.4 review recommendations.
    """
    # Validate configuration
    errors = _validate_config(config)
    if errors:
        for error in errors:
            print(f"[market_making] Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    # Core state
    state['initialized'] = True
    state['inventory'] = 0
    state['inventory_by_symbol'] = {}
    state['xrp_accumulated'] = 0.0
    state['btc_accumulated'] = 0.0
    state['last_signal_time'] = None
    state['indicators'] = {}

    # Position tracking for trailing stops and decay
    state['position_entries'] = {}

    # Per-pair metrics
    state['pnl_by_symbol'] = {}
    state['trades_by_symbol'] = {}


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Update inventory on fill.

    MM-005: Fixed unit handling - use value field from executor for USD pairs.
    v1.4.0: Added position tracking for trailing stops and per-pair metrics.
    v1.5.0: Added entry_time tracking for position decay (MM-E04).

    For XRP/BTC:
    - Buy: +XRP inventory, track BTC spent
    - Sell: -XRP inventory, track BTC received
    """
    symbol = fill.get('symbol', 'XRP/USDT')
    side = fill.get('side', '')
    size = fill.get('size', 0)
    price = fill.get('price', 0)
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())
    value = fill.get('value', size * price)

    # Initialize inventory tracking by symbol
    if 'inventory_by_symbol' not in state:
        state['inventory_by_symbol'] = {}

    current_inventory = state['inventory_by_symbol'].get(symbol, 0)
    is_cross_pair = _is_xrp_btc(symbol)

    if is_cross_pair:
        xrp_amount = size
        btc_amount = size * price

        if side == 'buy':
            state['inventory_by_symbol'][symbol] = current_inventory + xrp_amount
            state['xrp_accumulated'] = state.get('xrp_accumulated', 0) + xrp_amount
        elif side == 'sell':
            state['inventory_by_symbol'][symbol] = current_inventory - xrp_amount
            state['btc_accumulated'] = state.get('btc_accumulated', 0) + btc_amount
    else:
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

    # Track position entries for trailing stops and decay (MM-E04)
    if 'position_entries' not in state:
        state['position_entries'] = {}

    if side == 'buy':
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'highest_price': price,
                'lowest_price': price,
                'side': 'long',
                'entry_time': timestamp,  # v1.5.0: Track entry time for decay
            }
        else:
            pos = state['position_entries'][symbol]
            pos['highest_price'] = max(pos['highest_price'], price)
    elif side == 'sell':
        if symbol in state['position_entries']:
            del state['position_entries'][symbol]
    elif side == 'short':
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'highest_price': price,
                'lowest_price': price,
                'side': 'short',
                'entry_time': timestamp,
            }
        else:
            pos = state['position_entries'][symbol]
            pos['lowest_price'] = min(pos['lowest_price'], price)
    elif side == 'cover':
        if symbol in state['position_entries']:
            del state['position_entries'][symbol]

    # Track per-pair metrics
    if 'pnl_by_symbol' not in state:
        state['pnl_by_symbol'] = {}
    if 'trades_by_symbol' not in state:
        state['trades_by_symbol'] = {}

    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl
    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops.

    v1.4.0: Enhanced with per-pair metrics.
    v1.5.0: Added position decay stats.
    """
    state['final_summary'] = {
        'inventory_by_symbol': state.get('inventory_by_symbol', {}),
        'xrp_accumulated': state.get('xrp_accumulated', 0),
        'btc_accumulated': state.get('btc_accumulated', 0),
        'pnl_by_symbol': state.get('pnl_by_symbol', {}),
        'trades_by_symbol': state.get('trades_by_symbol', {}),
        'config_warnings': state.get('config_warnings', []),
        'position_entries': state.get('position_entries', {}),  # v1.5.0
    }
