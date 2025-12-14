"""
Order Flow Strategy

Trades based on trade tape analysis and buy/sell imbalance.
Enhanced with volatility adjustment, dynamic thresholds, and advanced features.

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
- 2.1.0: Added XRP/BTC pair support (moved to dedicated strategies in v2.2.0)
- 2.2.0: Removed XRP/BTC (now handled by market_making and ratio_trading)
         - Focused on USDT pairs for order flow momentum trading
         - Cleaner separation of concerns
- 3.0.0: Major refactor per order-flow-strategy-review-v2.2.md
         - CRIT-OF-001: Fixed trade array slicing (newest trades)
         - CRIT-OF-002: Always populate indicators, even on early returns
         - HIGH-OF-001: Added per-pair PnL tracking
         - HIGH-OF-002: Added config validation on startup
         - HIGH-OF-003: Added trade flow confirmation
         - HIGH-OF-004: Added fee-aware profitability check
         - HIGH-OF-005: Improved R:R ratio to 2:1 for momentum
         - MED-OF-001: Added micro-price calculation
         - MED-OF-002: Added position decay handling
         - MED-OF-003: Added trailing stop support
         - MED-OF-004: Configurable minimum trade size
         - ENH-OF-007: Position entry tracking for analysis
         - ENH-OF-008: Consecutive loss circuit breaker
- 3.1.0: Fixes per order-flow-strategy-review-v3.0.md
         - HIGH-NEW-001: Added SYMBOL_CONFIGS validation in _validate_config()
         - HIGH-NEW-002: Implemented asymmetric buy/sell thresholds
         - FIX: Fixed undefined base_threshold variable in indicator logging
         - FIX: Fixed VWAP reversion using wrong threshold variable
         - Updated indicator logging with separate buy/sell thresholds
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from ws_tester.types import DataSnapshot, Signal, OrderbookSnapshot


# =============================================================================
# REQUIRED: Strategy Metadata
# =============================================================================
STRATEGY_NAME = "order_flow"
STRATEGY_VERSION = "3.1.0"
SYMBOLS = ["XRP/USDT", "BTC/USDT"]


# =============================================================================
# REQUIRED: Default Configuration
# =============================================================================
CONFIG = {
    # Order flow parameters - Asymmetric thresholds (HIGH-NEW-002)
    'imbalance_threshold': 0.3,        # Default threshold (used if asymmetric disabled)
    'buy_imbalance_threshold': 0.30,   # Threshold for buy signals
    'sell_imbalance_threshold': 0.25,  # Lower for sell (selling pressure more impactful)
    'use_asymmetric_thresholds': True, # Enable asymmetric buy/sell thresholds
    'volume_spike_mult': 2.0,          # Volume spike multiplier
    'lookback_trades': 50,             # Number of trades to analyze

    # Position sizing
    'position_size_usd': 25.0,         # Size per trade in USD
    'max_position_usd': 100.0,         # Maximum position exposure
    'min_trade_size_usd': 5.0,         # Minimum USD per trade (MED-OF-004)

    # Risk management - Improved R:R ratio (HIGH-OF-005: 2:1 for momentum)
    'take_profit_pct': 1.0,            # Take profit at 1.0%
    'stop_loss_pct': 0.5,              # Stop loss at 0.5% (2:1 R:R)

    # Cooldown mechanisms
    'cooldown_trades': 10,             # Min trades between signals
    'cooldown_seconds': 5.0,           # Min time between signals

    # Volatility adjustment parameters
    'base_volatility_pct': 0.5,        # Baseline volatility for scaling
    'volatility_lookback': 20,         # Candles for volatility calculation
    'volatility_threshold_mult': 1.5,  # Increase threshold in high volatility

    # VWAP parameters
    'vwap_deviation_threshold': 0.001, # Min deviation from VWAP for reversion signal
    'vwap_reversion_size_mult': 0.75,  # Position size multiplier for VWAP reversion
    'vwap_reversion_threshold_mult': 0.7,  # Threshold multiplier for VWAP reversion

    # Trade flow confirmation (HIGH-OF-003)
    'use_trade_flow_confirmation': True,
    'trade_flow_threshold': 0.15,      # Minimum trade flow alignment

    # Fee-aware profitability (HIGH-OF-004)
    'fee_rate': 0.001,                 # 0.1% per trade (0.2% round-trip)
    'min_profit_after_fees_pct': 0.05, # Minimum profit after fees
    'use_fee_check': True,             # Enable fee-aware profitability check

    # Micro-price (MED-OF-001)
    'use_micro_price': True,           # Use volume-weighted micro-price

    # Position decay (MED-OF-002)
    'use_position_decay': True,        # Enable time-based position decay
    'max_position_age_seconds': 300,   # Max age before widening TP (5 minutes)
    'position_decay_tp_multiplier': 0.5,  # Reduce TP by 50% for stale positions

    # Trailing stops (MED-OF-003)
    'use_trailing_stop': False,        # Enable trailing stops
    'trailing_stop_activation': 0.3,   # Activate trailing after 0.3% profit
    'trailing_stop_distance': 0.2,     # Trail at 0.2% from high

    # Circuit breaker (ENH-OF-008)
    'use_circuit_breaker': True,       # Enable consecutive loss circuit breaker
    'max_consecutive_losses': 3,       # Max losses before cooldown
    'circuit_breaker_minutes': 15,     # Cooldown after max losses
}

# Per-symbol configurations
# v3.1.0: Added asymmetric thresholds (HIGH-NEW-002)
SYMBOL_CONFIGS = {
    'XRP/USDT': {
        'buy_imbalance_threshold': 0.30,   # XRP buy threshold
        'sell_imbalance_threshold': 0.25,  # Lower for sell (more impactful)
        'imbalance_threshold': 0.30,       # Legacy fallback
        'position_size_usd': 25.0,
        'volume_spike_mult': 2.0,
        'take_profit_pct': 1.0,            # 2:1 R:R
        'stop_loss_pct': 0.5,
    },
    'BTC/USDT': {
        'buy_imbalance_threshold': 0.25,   # BTC buy threshold (lower for high liquidity)
        'sell_imbalance_threshold': 0.20,  # Even lower for sell
        'imbalance_threshold': 0.25,       # Legacy fallback
        'position_size_usd': 50.0,         # Larger size for BTC
        'volume_spike_mult': 1.8,          # Lower for more signals
        'take_profit_pct': 0.8,            # Slightly tighter for BTC
        'stop_loss_pct': 0.4,              # 2:1 R:R
    },
}


# =============================================================================
# Helper Functions
# =============================================================================
def _get_symbol_config(symbol: str, config: Dict[str, Any], key: str) -> Any:
    """Get symbol-specific config or fall back to global config."""
    symbol_cfg = SYMBOL_CONFIGS.get(symbol, {})
    return symbol_cfg.get(key, config.get(key))


def _validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration on startup (HIGH-OF-002).

    v3.1.0: Added SYMBOL_CONFIGS validation (HIGH-NEW-001)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Required positive values
    required_positive = [
        'position_size_usd',
        'max_position_usd',
        'stop_loss_pct',
        'take_profit_pct',
        'lookback_trades',
        'cooldown_seconds',
    ]

    for key in required_positive:
        val = config.get(key)
        if val is None:
            errors.append(f"Missing required config: {key}")
        elif val <= 0:
            errors.append(f"{key} must be positive, got {val}")

    # Bounds checks
    imbalance = config.get('imbalance_threshold', 0.3)
    if imbalance < 0.1 or imbalance > 0.8:
        errors.append(f"imbalance_threshold should be 0.1-0.8, got {imbalance}")

    fee_rate = config.get('fee_rate', 0.001)
    if fee_rate < 0 or fee_rate > 0.01:
        errors.append(f"fee_rate should be 0-0.01, got {fee_rate}")

    # R:R ratio warning
    sl = config.get('stop_loss_pct', 0.5)
    tp = config.get('take_profit_pct', 1.0)
    if sl > 0 and tp > 0:
        rr_ratio = tp / sl
        if rr_ratio < 1.0:
            errors.append(f"Warning: R:R ratio ({rr_ratio:.2f}:1) < 1:1, requires >{100/(1+rr_ratio):.0f}% win rate")
        elif rr_ratio < 1.5:
            errors.append(f"Info: R:R ratio {rr_ratio:.2f}:1 is acceptable but consider 2:1+ for momentum")

    # HIGH-NEW-001: Validate SYMBOL_CONFIGS
    symbol_positive_keys = ['stop_loss_pct', 'take_profit_pct', 'position_size_usd', 'volume_spike_mult']
    for symbol, sym_cfg in SYMBOL_CONFIGS.items():
        for key in symbol_positive_keys:
            if key in sym_cfg and sym_cfg[key] <= 0:
                errors.append(f"{symbol}.{key} must be positive, got {sym_cfg[key]}")

        # Validate imbalance_threshold per symbol
        if 'imbalance_threshold' in sym_cfg:
            imb = sym_cfg['imbalance_threshold']
            if imb < 0.1 or imb > 0.8:
                errors.append(f"{symbol}.imbalance_threshold should be 0.1-0.8, got {imb}")

        # Per-symbol R:R check
        sym_sl = sym_cfg.get('stop_loss_pct')
        sym_tp = sym_cfg.get('take_profit_pct')
        if sym_sl and sym_tp and sym_sl > 0 and sym_tp > 0:
            rr = sym_tp / sym_sl
            if rr < 1.0:
                errors.append(f"Warning: {symbol} R:R ratio ({rr:.2f}:1) < 1:1")

    return errors


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
               for i in range(1, len(closes)) if closes[i - 1] != 0]

    if not returns:
        return 0.0

    # Calculate standard deviation of returns
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5

    return std_dev * 100  # Return as percentage


def _calculate_micro_price(ob: OrderbookSnapshot) -> float:
    """
    Calculate volume-weighted micro-price (MED-OF-001).

    Micro-price provides better price discovery than simple mid-price
    by weighting by order sizes at best bid/ask.

    Formula: micro = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
    """
    if not ob or not ob.bids or not ob.asks:
        return 0.0

    best_bid, bid_size = ob.bids[0]
    best_ask, ask_size = ob.asks[0]

    total_size = bid_size + ask_size
    if total_size <= 0:
        return ob.mid if ob else 0.0

    micro_price = (best_bid * ask_size + best_ask * bid_size) / total_size
    return micro_price


def _check_fee_profitability(
    expected_move_pct: float,
    fee_rate: float,
    min_profit_pct: float
) -> Tuple[bool, float]:
    """
    Check if trade is profitable after fees (HIGH-OF-004).

    Args:
        expected_move_pct: Expected price move percentage
        fee_rate: Fee per trade (e.g., 0.001 for 0.1%)
        min_profit_pct: Minimum required profit after fees

    Returns:
        Tuple of (is_profitable, expected_profit_pct)
    """
    # Round-trip fees (entry + exit)
    round_trip_fee_pct = fee_rate * 2 * 100  # Convert to percentage

    # Net profit after fees
    net_profit_pct = expected_move_pct - round_trip_fee_pct

    is_profitable = net_profit_pct >= min_profit_pct

    return is_profitable, net_profit_pct


def _calculate_trailing_stop(
    entry_price: float,
    highest_price: float,
    side: str,
    activation_pct: float,
    trail_distance_pct: float
) -> Optional[float]:
    """
    Calculate trailing stop level (MED-OF-003).

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
        profit_pct = (highest_price - entry_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 - trail_distance_pct / 100)
    elif side == 'short':
        profit_pct = (entry_price - highest_price) / entry_price * 100
        if profit_pct >= activation_pct:
            return highest_price * (1 + trail_distance_pct / 100)

    return None


def _check_position_decay(
    position_entry: Dict[str, Any],
    current_time: datetime,
    max_age_seconds: float,
    tp_multiplier: float
) -> Tuple[bool, float]:
    """
    Check if position is stale and should have adjusted TP (MED-OF-002).

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
        return True, tp_multiplier

    return False, 1.0


def _check_circuit_breaker(
    state: Dict[str, Any],
    current_time: datetime,
    max_losses: int,
    cooldown_minutes: float
) -> bool:
    """
    Check if circuit breaker is active (ENH-OF-008).

    Args:
        state: Strategy state
        current_time: Current timestamp
        max_losses: Maximum consecutive losses before circuit breaker
        cooldown_minutes: Cooldown period after circuit breaker triggers

    Returns:
        True if trading is blocked by circuit breaker
    """
    consecutive_losses = state.get('consecutive_losses', 0)

    if consecutive_losses < max_losses:
        return False

    # Check if cooldown has elapsed
    breaker_time = state.get('circuit_breaker_time')
    if breaker_time is None:
        return False

    elapsed_minutes = (current_time - breaker_time).total_seconds() / 60

    if elapsed_minutes >= cooldown_minutes:
        # Reset circuit breaker
        state['consecutive_losses'] = 0
        state['circuit_breaker_time'] = None
        return False

    return True


def _is_trade_flow_aligned(
    data: DataSnapshot,
    symbol: str,
    direction: str,
    threshold: float,
    n_trades: int = 50
) -> bool:
    """
    Check if trade flow confirms the signal direction (HIGH-OF-003).

    Args:
        data: Market data snapshot
        symbol: Trading pair
        direction: 'buy' or 'sell'
        threshold: Minimum trade flow imbalance for confirmation
        n_trades: Number of trades to analyze

    Returns:
        True if trade flow confirms direction
    """
    trade_flow = data.get_trade_imbalance(symbol, n_trades)

    if direction == 'buy':
        return trade_flow > threshold
    elif direction == 'sell':
        return trade_flow < -threshold

    return True


def _build_base_indicators(
    symbol: str,
    trade_count: int,
    status: str,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Build base indicators dict for early returns (CRIT-OF-002)."""
    return {
        'symbol': symbol,
        'trade_count': trade_count,
        'status': status,
        'position_side': state.get('position_side'),
        'position_size': state.get('position_size', 0),
        'consecutive_losses': state.get('consecutive_losses', 0),
        'pnl_symbol': state.get('pnl_by_symbol', {}).get(symbol, 0),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
    }


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
    - Confirm signals with trade flow direction (HIGH-OF-003)
    - Check fee profitability before entry (HIGH-OF-004)

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
        state['position_by_symbol'] = {}
        state['fills'] = []
        state['indicators'] = {}
        state['pnl_by_symbol'] = {}
        state['trades_by_symbol'] = {}
        state['position_entries'] = {}
        state['consecutive_losses'] = 0
        state['circuit_breaker_time'] = None

    current_time = data.timestamp

    # Circuit breaker check (ENH-OF-008)
    if config.get('use_circuit_breaker', True):
        max_losses = config.get('max_consecutive_losses', 3)
        cooldown_min = config.get('circuit_breaker_minutes', 15)

        if _check_circuit_breaker(state, current_time, max_losses, cooldown_min):
            # Set indicators for circuit breaker state
            state['indicators'] = _build_base_indicators(
                symbol='N/A',
                trade_count=0,
                status='circuit_breaker',
                state=state
            )
            state['indicators']['circuit_breaker_active'] = True
            return None

    # Time-based cooldown check
    if state['last_signal_time'] is not None:
        elapsed = (current_time - state['last_signal_time']).total_seconds()
        if elapsed < config.get('cooldown_seconds', 5.0):
            state['indicators'] = _build_base_indicators(
                symbol='N/A',
                trade_count=0,
                status='cooldown',
                state=state
            )
            state['indicators']['cooldown_remaining'] = config.get('cooldown_seconds', 5.0) - elapsed
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

    # CRIT-OF-002: Always set base indicators even on early return
    if len(trades) < lookback:
        state['indicators'] = _build_base_indicators(
            symbol=symbol,
            trade_count=len(trades),
            status='warming_up',
            state=state
        )
        state['indicators']['required_trades'] = lookback
        return None

    # CRIT-OF-001: FIX - Use negative index to get MOST RECENT trades
    recent_trades = trades[-lookback:]

    # Get symbol-specific cooldown settings
    cooldown_trades = _get_symbol_config(symbol, config, 'cooldown_trades') or config.get('cooldown_trades', 10)

    # Trade-based cooldown check
    state['total_trades_seen'] = len(trades)
    trades_since_signal = state['total_trades_seen'] - state['last_signal_idx']

    if trades_since_signal < cooldown_trades:
        state['indicators'] = _build_base_indicators(
            symbol=symbol,
            trade_count=len(trades),
            status='trade_cooldown',
            state=state
        )
        state['indicators']['trades_since_signal'] = trades_since_signal
        state['indicators']['cooldown_trades'] = cooldown_trades
        return None

    # Calculate buy/sell imbalance
    buy_volume = sum(t.size for t in recent_trades if t.side == 'buy')
    sell_volume = sum(t.size for t in recent_trades if t.side == 'sell')
    total_volume = buy_volume + sell_volume

    if total_volume == 0:
        state['indicators'] = _build_base_indicators(
            symbol=symbol,
            trade_count=len(trades),
            status='no_volume',
            state=state
        )
        return None

    imbalance = (buy_volume - sell_volume) / total_volume  # -1 to 1

    # Calculate average volume and volume spike
    # CRIT-OF-001: FIX - Use most recent 5 trades (end of array)
    avg_trade_size = total_volume / len(recent_trades)
    last_5_trades = recent_trades[-5:]
    last_5_volume = sum(t.size for t in last_5_trades)
    expected_5_volume = avg_trade_size * 5
    volume_spike = last_5_volume / expected_5_volume if expected_5_volume > 0 else 1.0

    # Calculate VWAP
    vwap = data.get_vwap(symbol, lookback)
    current_price = data.prices.get(symbol, 0)

    if not current_price or not vwap:
        state['indicators'] = _build_base_indicators(
            symbol=symbol,
            trade_count=len(trades),
            status='no_price_or_vwap',
            state=state
        )
        return None

    price_vs_vwap = (current_price - vwap) / vwap  # Positive = above VWAP

    # MED-OF-001: Calculate micro-price
    ob = data.orderbooks.get(symbol)
    micro_price = current_price
    if config.get('use_micro_price', True) and ob:
        micro_price = _calculate_micro_price(ob)

    # Calculate volatility from candles
    candles = data.candles_1m.get(symbol, ())
    volatility = _calculate_volatility(candles, config.get('volatility_lookback', 20))
    base_vol = config.get('base_volatility_pct', 0.5)

    # Dynamic threshold adjustment based on volatility
    vol_multiplier = 1.0
    if base_vol > 0 and volatility > 0:
        vol_ratio = volatility / base_vol
        if vol_ratio > 1.5:
            vol_multiplier = min(vol_ratio, config.get('volatility_threshold_mult', 1.5))

    # Get symbol-specific config
    volume_spike_mult = _get_symbol_config(symbol, config, 'volume_spike_mult') or config.get('volume_spike_mult', 2.0)
    position_size = _get_symbol_config(symbol, config, 'position_size_usd')
    tp_pct = _get_symbol_config(symbol, config, 'take_profit_pct')
    sl_pct = _get_symbol_config(symbol, config, 'stop_loss_pct')

    # HIGH-NEW-002: Asymmetric thresholds for buy vs sell
    use_asymmetric = config.get('use_asymmetric_thresholds', True)
    if use_asymmetric:
        base_buy_threshold = _get_symbol_config(symbol, config, 'buy_imbalance_threshold') or 0.30
        base_sell_threshold = _get_symbol_config(symbol, config, 'sell_imbalance_threshold') or 0.25
    else:
        base_buy_threshold = _get_symbol_config(symbol, config, 'imbalance_threshold') or 0.30
        base_sell_threshold = base_buy_threshold

    # Adjusted thresholds with volatility
    effective_buy_threshold = base_buy_threshold * vol_multiplier
    effective_sell_threshold = base_sell_threshold * vol_multiplier

    # HIGH-OF-003: Trade flow confirmation
    use_trade_flow = config.get('use_trade_flow_confirmation', True)
    trade_flow_threshold = config.get('trade_flow_threshold', 0.15)
    trade_flow = data.get_trade_imbalance(symbol, lookback)

    # HIGH-OF-004: Fee profitability check
    use_fee_check = config.get('use_fee_check', True)
    fee_rate = config.get('fee_rate', 0.001)
    min_profit_pct = config.get('min_profit_after_fees_pct', 0.05)

    is_fee_profitable = True
    expected_profit = tp_pct  # Expected profit is TP percentage
    if use_fee_check:
        is_fee_profitable, expected_profit = _check_fee_profitability(
            tp_pct, fee_rate, min_profit_pct
        )

    # Check position limits
    current_position = state.get('position_size', 0)
    max_position = config.get('max_position_usd', 100.0)
    min_trade = config.get('min_trade_size_usd', 5.0)  # MED-OF-004: Configurable

    # MED-OF-003: Check trailing stop exit
    trailing_signal = _check_trailing_stop_exit(
        data, config, state, symbol, current_price, ob
    )
    if trailing_signal:
        return trailing_signal

    # MED-OF-002: Check position decay exit
    decay_signal = _check_position_decay_exit(
        data, config, state, symbol, current_price, ob, current_time
    )
    if decay_signal:
        return decay_signal

    # Get trailing stop price for logging
    trailing_stop_price = None
    if config.get('use_trailing_stop', False):
        pos_entry = state.get('position_entries', {}).get(symbol)
        if pos_entry:
            tracking_price = pos_entry.get('highest_price' if pos_entry['side'] == 'long' else 'lowest_price', current_price)
            trailing_stop_price = _calculate_trailing_stop(
                entry_price=pos_entry['entry_price'],
                highest_price=tracking_price,
                side=pos_entry['side'],
                activation_pct=config.get('trailing_stop_activation', 0.3),
                trail_distance_pct=config.get('trailing_stop_distance', 0.2)
            )

    # Store indicators for logging (CRIT-OF-002: Always populate)
    state['indicators'] = {
        'symbol': symbol,
        'status': 'active',
        'trade_count': len(trades),
        'imbalance': round(imbalance, 4),
        'buy_volume': round(buy_volume, 2),
        'sell_volume': round(sell_volume, 2),
        'volume_spike': round(volume_spike, 2),
        'volume_spike_threshold': round(volume_spike_mult, 2),
        'vwap': round(vwap, 6),
        'price': round(current_price, 6),
        'micro_price': round(micro_price, 6),
        'price_vs_vwap': round(price_vs_vwap, 6),
        'volatility_pct': round(volatility, 4),
        'vol_multiplier': round(vol_multiplier, 2),
        'base_buy_threshold': round(base_buy_threshold, 4),
        'base_sell_threshold': round(base_sell_threshold, 4),
        'effective_buy_threshold': round(effective_buy_threshold, 4),
        'effective_sell_threshold': round(effective_sell_threshold, 4),
        'trades_since_signal': trades_since_signal,
        # Trade flow (HIGH-OF-003)
        'trade_flow': round(trade_flow, 4),
        'trade_flow_threshold': round(trade_flow_threshold, 4),
        'use_trade_flow': use_trade_flow,
        # Fee profitability (HIGH-OF-004)
        'is_fee_profitable': is_fee_profitable,
        'expected_profit_pct': round(expected_profit, 4),
        # Position info
        'position_side': state.get('position_side'),
        'position_size': round(current_position, 2),
        'max_position': max_position,
        # Trailing stop (MED-OF-003)
        'trailing_stop_price': round(trailing_stop_price, 6) if trailing_stop_price else None,
        # Per-pair metrics (HIGH-OF-001)
        'pnl_symbol': round(state.get('pnl_by_symbol', {}).get(symbol, 0), 4),
        'trades_symbol': state.get('trades_by_symbol', {}).get(symbol, 0),
        # Circuit breaker (ENH-OF-008)
        'consecutive_losses': state.get('consecutive_losses', 0),
    }

    if current_position >= max_position:
        state['indicators']['status'] = 'max_position_reached'
        return None

    # Adjust size if needed to stay within limits
    available = max_position - current_position
    actual_size = min(position_size, available)
    if actual_size < min_trade:
        state['indicators']['status'] = 'insufficient_size'
        return None

    # HIGH-OF-004: Skip if not profitable after fees
    if use_fee_check and not is_fee_profitable:
        state['indicators']['status'] = 'not_fee_profitable'
        return None

    # Generate signal
    signal = None

    # Strong buy pressure with volume spike (HIGH-NEW-002: use asymmetric buy threshold)
    if imbalance > effective_buy_threshold and volume_spike > volume_spike_mult:
        # HIGH-OF-003: Check trade flow confirmation
        if use_trade_flow and not _is_trade_flow_aligned(data, symbol, 'buy', trade_flow_threshold, lookback):
            state['indicators']['status'] = 'trade_flow_not_aligned'
            state['indicators']['trade_flow_aligned'] = False
            return None

        state['indicators']['trade_flow_aligned'] = True

        signal = Signal(
            action='buy',
            symbol=symbol,
            size=actual_size,
            price=current_price,
            reason=f"OF: Buy pressure (imbal={imbalance:.2f}, vol={volume_spike:.1f}x, volatility={volatility:.2f}%)",
            stop_loss=current_price * (1 - sl_pct / 100),
            take_profit=current_price * (1 + tp_pct / 100),
        )

    # Strong sell pressure with volume spike (HIGH-NEW-002: use asymmetric sell threshold)
    elif imbalance < -effective_sell_threshold and volume_spike > volume_spike_mult:
        # HIGH-OF-003: Check trade flow confirmation
        if use_trade_flow and not _is_trade_flow_aligned(data, symbol, 'sell', trade_flow_threshold, lookback):
            state['indicators']['status'] = 'trade_flow_not_aligned'
            state['indicators']['trade_flow_aligned'] = False
            return None

        state['indicators']['trade_flow_aligned'] = True

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
            # Open a short position
            signal = Signal(
                action='short',
                symbol=symbol,
                size=actual_size,
                price=current_price,
                reason=f"OF: Short (imbal={imbalance:.2f}, vol={volume_spike:.1f}x, volatility={volatility:.2f}%)",
                stop_loss=current_price * (1 + sl_pct / 100),
                take_profit=current_price * (1 - tp_pct / 100),
            )

    # VWAP mean reversion opportunities
    vwap_threshold_mult = config.get('vwap_reversion_threshold_mult', 0.7)
    vwap_size_mult = config.get('vwap_reversion_size_mult', 0.75)
    vwap_deviation = config.get('vwap_deviation_threshold', 0.001)

    # Buy pressure + price below VWAP (mean reversion opportunity)
    if signal is None and (imbalance > effective_buy_threshold * vwap_threshold_mult and
          price_vs_vwap < -vwap_deviation):
        reduced_size = actual_size * vwap_size_mult
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
    if signal is None and (imbalance < -effective_sell_threshold * vwap_threshold_mult and
          price_vs_vwap > vwap_deviation):
        has_long = state.get('position_side') == 'long' and state.get('position_size', 0) > 0
        reduced_size = actual_size * vwap_size_mult

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
        state['indicators']['status'] = 'signal_generated'
        return signal

    state['indicators']['status'] = 'no_signal'
    return None


def _check_trailing_stop_exit(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    ob: Optional[OrderbookSnapshot]
) -> Optional[Signal]:
    """
    Check if trailing stop should trigger exit (MED-OF-003).

    Returns:
        Signal if trailing stop triggered, None otherwise
    """
    if not config.get('use_trailing_stop', False):
        return None

    pos_entry = state.get('position_entries', {}).get(symbol)
    if not pos_entry:
        return None

    trailing_activation = config.get('trailing_stop_activation', 0.3)
    trailing_distance = config.get('trailing_stop_distance', 0.2)

    # Update highest/lowest price for tracking
    if pos_entry['side'] == 'long':
        pos_entry['highest_price'] = max(pos_entry.get('highest_price', current_price), current_price)
        tracking_price = pos_entry['highest_price']
    else:
        pos_entry['lowest_price'] = min(pos_entry.get('lowest_price', current_price), current_price)
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

    exit_price = current_price
    if ob:
        exit_price = ob.best_bid if pos_entry['side'] == 'long' else ob.best_ask

    # Check if trailing stop is triggered
    if pos_entry['side'] == 'long' and current_price <= trailing_stop_price:
        close_size = state.get('position_size', 0)
        if close_size > 0:
            return Signal(
                action='sell',
                symbol=symbol,
                size=close_size,
                price=exit_price,
                reason=f"OF: Trailing stop (entry={pos_entry['entry_price']:.6f}, high={pos_entry['highest_price']:.6f})",
                metadata={'trailing_stop': True},
            )

    elif pos_entry['side'] == 'short' and current_price >= trailing_stop_price:
        close_size = state.get('position_size', 0)
        if close_size > 0:
            return Signal(
                action='cover',
                symbol=symbol,
                size=close_size,
                price=exit_price,
                reason=f"OF: Trailing stop (entry={pos_entry['entry_price']:.6f}, low={pos_entry['lowest_price']:.6f})",
                metadata={'trailing_stop': True},
            )

    return None


def _check_position_decay_exit(
    data: DataSnapshot,
    config: Dict[str, Any],
    state: Dict[str, Any],
    symbol: str,
    current_price: float,
    ob: Optional[OrderbookSnapshot],
    current_time: datetime
) -> Optional[Signal]:
    """
    Check if stale position should be closed with reduced TP (MED-OF-002).

    Returns:
        Signal if stale position should exit, None otherwise
    """
    if not config.get('use_position_decay', True):
        return None

    pos_entry = state.get('position_entries', {}).get(symbol)
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

    exit_price = current_price
    if ob:
        exit_price = ob.best_bid if pos_entry['side'] == 'long' else ob.best_ask

    if pos_entry['side'] == 'long':
        profit_pct = (current_price - entry_price) / entry_price * 100
        if profit_pct >= adjusted_tp_pct:
            close_size = state.get('position_size', 0)
            if close_size > 0:
                age_seconds = (current_time - pos_entry.get('entry_time', current_time)).total_seconds()
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Stale exit (age={age_seconds:.0f}s, profit={profit_pct:.2f}%)",
                    metadata={'position_decay': True},
                )

    elif pos_entry['side'] == 'short':
        profit_pct = (entry_price - current_price) / entry_price * 100
        if profit_pct >= adjusted_tp_pct:
            close_size = state.get('position_size', 0)
            if close_size > 0:
                age_seconds = (current_time - pos_entry.get('entry_time', current_time)).total_seconds()
                return Signal(
                    action='cover',
                    symbol=symbol,
                    size=close_size,
                    price=exit_price,
                    reason=f"OF: Stale exit (age={age_seconds:.0f}s, profit={profit_pct:.2f}%)",
                    metadata={'position_decay': True},
                )

    return None


# =============================================================================
# OPTIONAL: Lifecycle Callbacks
# =============================================================================
def on_start(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Initialize strategy state on startup.

    v3.0.0: Added config validation and enhanced state tracking.
    """
    # HIGH-OF-002: Validate configuration
    errors = _validate_config(config)
    if errors:
        for error in errors:
            print(f"[order_flow] Config warning: {error}")
        state['config_warnings'] = errors
    state['config_validated'] = True

    # Core state
    state['initialized'] = True
    state['last_signal_idx'] = 0
    state['total_trades_seen'] = 0
    state['last_signal_time'] = None
    state['position_side'] = None  # 'long', 'short', or None
    state['position_size'] = 0.0
    state['position_by_symbol'] = {}
    state['fills'] = []
    state['indicators'] = {}

    # HIGH-OF-001: Per-pair metrics
    state['pnl_by_symbol'] = {}
    state['trades_by_symbol'] = {}
    state['wins_by_symbol'] = {}
    state['losses_by_symbol'] = {}

    # ENH-OF-007: Position entry tracking
    state['position_entries'] = {}

    # ENH-OF-008: Circuit breaker
    state['consecutive_losses'] = 0
    state['circuit_breaker_time'] = None


def on_fill(fill: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Track fills and update position state.

    v3.0.0: Added per-pair PnL tracking (HIGH-OF-001) and position entry tracking (ENH-OF-007).

    Args:
        fill: Fill information dict with keys: side, size, price, value, symbol, pnl
        state: Mutable strategy state dict
    """
    # Track fill history (bounded)
    if 'fills' not in state:
        state['fills'] = []
    state['fills'].append(fill)
    state['fills'] = state['fills'][-50:]  # Keep last 50

    side = fill.get('side', '')
    value = fill.get('value', fill.get('size', 0) * fill.get('price', 0))
    symbol = fill.get('symbol', 'XRP/USDT')
    price = fill.get('price', 0)
    pnl = fill.get('pnl', 0)
    timestamp = fill.get('timestamp', datetime.now())

    # HIGH-OF-001: Track per-pair metrics
    if 'pnl_by_symbol' not in state:
        state['pnl_by_symbol'] = {}
    if 'trades_by_symbol' not in state:
        state['trades_by_symbol'] = {}
    if 'wins_by_symbol' not in state:
        state['wins_by_symbol'] = {}
    if 'losses_by_symbol' not in state:
        state['losses_by_symbol'] = {}

    # Update per-pair PnL
    if pnl != 0:
        state['pnl_by_symbol'][symbol] = state['pnl_by_symbol'].get(symbol, 0) + pnl

        # Track wins/losses
        if pnl > 0:
            state['wins_by_symbol'][symbol] = state['wins_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = 0  # Reset on win
        else:
            state['losses_by_symbol'][symbol] = state['losses_by_symbol'].get(symbol, 0) + 1
            state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

            # ENH-OF-008: Check circuit breaker trigger
            max_losses = 3  # Use default, actual check in generate_signal
            if state['consecutive_losses'] >= max_losses:
                state['circuit_breaker_time'] = timestamp

    state['trades_by_symbol'][symbol] = state['trades_by_symbol'].get(symbol, 0) + 1

    # Update position state
    if side == 'buy':
        # Opening or adding to long position
        state['position_side'] = 'long'
        state['position_size'] = state.get('position_size', 0) + value

        # ENH-OF-007: Track position entry
        if 'position_entries' not in state:
            state['position_entries'] = {}
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'highest_price': price,
                'lowest_price': price,
                'side': 'long',
            }
        else:
            pos = state['position_entries'][symbol]
            pos['highest_price'] = max(pos.get('highest_price', price), price)

    elif side == 'sell':
        # Closing long position
        state['position_size'] = max(0, state.get('position_size', 0) - value)
        if state['position_size'] < 0.01:
            state['position_side'] = None
            state['position_size'] = 0.0
            # Clear position entry
            if symbol in state.get('position_entries', {}):
                del state['position_entries'][symbol]

    elif side == 'short':
        # Opening or adding to short position
        state['position_side'] = 'short'
        state['position_size'] = state.get('position_size', 0) + value

        # ENH-OF-007: Track position entry
        if 'position_entries' not in state:
            state['position_entries'] = {}
        if symbol not in state['position_entries']:
            state['position_entries'][symbol] = {
                'entry_price': price,
                'entry_time': timestamp,
                'highest_price': price,
                'lowest_price': price,
                'side': 'short',
            }
        else:
            pos = state['position_entries'][symbol]
            pos['lowest_price'] = min(pos.get('lowest_price', price), price)

    elif side == 'cover':
        # Closing short position
        state['position_size'] = max(0, state.get('position_size', 0) - value)
        if state['position_size'] < 0.01:
            state['position_side'] = None
            state['position_size'] = 0.0
            # Clear position entry
            if symbol in state.get('position_entries', {}):
                del state['position_entries'][symbol]


def on_stop(state: Dict[str, Any]) -> None:
    """
    Called when strategy stops. Cleanup if needed.

    v3.0.0: Enhanced with per-pair metrics summary.

    Args:
        state: Mutable strategy state dict
    """
    final_position = state.get('position_size', 0)
    final_side = state.get('position_side')
    total_fills = len(state.get('fills', []))

    # Calculate overall metrics
    total_pnl = sum(state.get('pnl_by_symbol', {}).values())
    total_trades = sum(state.get('trades_by_symbol', {}).values())
    total_wins = sum(state.get('wins_by_symbol', {}).values())
    total_losses = sum(state.get('losses_by_symbol', {}).values())

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # Clear large cached data
    state['indicators'] = {}

    # Keep summary for post-run analysis
    state['final_summary'] = {
        'position_side': final_side,
        'position_size': final_position,
        'total_fills': total_fills,
        # Per-pair metrics (HIGH-OF-001)
        'pnl_by_symbol': state.get('pnl_by_symbol', {}),
        'trades_by_symbol': state.get('trades_by_symbol', {}),
        'wins_by_symbol': state.get('wins_by_symbol', {}),
        'losses_by_symbol': state.get('losses_by_symbol', {}),
        # Aggregate metrics
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_rate': win_rate,
        # Config warnings
        'config_warnings': state.get('config_warnings', []),
    }
