"""
EMA-9 Trend Flip Strategy v1.0.0

A trend-following strategy based on the 9-period Exponential Moving Average (EMA).
Entry signals are generated when price "flips" from consistently opening on one side
of the EMA to opening on the opposite side, indicating a potential trend change.

Strategy Logic (Option 3 - 1H Timeframe):
- Build 1H candles from 1m candles
- Calculate EMA-9 on hourly candle open prices
- Track N consecutive candles opening above/below EMA
- Entry: When candle opens on opposite side after N consecutive candles
- Exit: Opposing EMA flip, stop loss, or take profit

Based on research: ws_paper_tester/docs/development/plans/ema9/ema-9-strategy-analysis.md

Version History:
- 1.0.0: Initial implementation based on EMA-9 Strategy Analysis
         - Option 3 (1H timeframe) implementation
         - Build hourly candles from 1-minute data
         - EMA-9 calculation on hourly opens
         - Flip detection with consecutive candle confirmation
         - Buffer percentage to reduce whipsaws
         - Exit on opposing EMA flip
         - ATR-based or fixed percentage stops
         - 2:1 risk-reward ratio targeting
         - Time-based cooldown mechanisms
         - Maximum hold time exit condition
"""

# =============================================================================
# Public API - Required exports for strategy interface
# =============================================================================
from .config import (
    STRATEGY_NAME,
    STRATEGY_VERSION,
    SYMBOLS,
    CONFIG,
    SYMBOL_CONFIGS,
    # Enums
    TrendDirection,
    RejectionReason,
    # Helper
    get_symbol_config,
)

from .signal import generate_signal

from .lifecycle import on_start, on_fill, on_stop, initialize_state, validate_config

from .warmup import (
    fetch_warmup_candles,
    warmup_from_db_sync,
    initialize_warmup_state,
    merge_warmup_with_realtime,
    check_warmup_status,
)

# =============================================================================
# Secondary exports (for advanced use / testing)
# =============================================================================
from .indicators import (
    calculate_ema,
    calculate_ema_series,
    build_hourly_candles,
    calculate_atr,
    get_candle_position,
    check_consecutive_positions,
)

from .exits import (
    check_exit_conditions,
    check_ema_flip_exit,
    check_max_hold_time_exit,
)

from .risk import (
    check_position_limits,
    calculate_entry_stops,
    create_entry_signal,
    track_rejection,
)

from .signal import build_indicators, check_circuit_breaker


__all__ = [
    # Required strategy interface
    'STRATEGY_NAME',
    'STRATEGY_VERSION',
    'SYMBOLS',
    'CONFIG',
    'SYMBOL_CONFIGS',
    'generate_signal',
    'on_start',
    'on_fill',
    'on_stop',
    # Enums
    'TrendDirection',
    'RejectionReason',
    # Helpers
    'get_symbol_config',
    'initialize_state',
    'validate_config',
    # Warmup
    'fetch_warmup_candles',
    'warmup_from_db_sync',
    'initialize_warmup_state',
    'merge_warmup_with_realtime',
    'check_warmup_status',
    # Indicators
    'calculate_ema',
    'calculate_ema_series',
    'build_hourly_candles',
    'calculate_atr',
    'get_candle_position',
    'check_consecutive_positions',
    # Exits
    'check_exit_conditions',
    'check_ema_flip_exit',
    'check_max_hold_time_exit',
    # Risk
    'check_position_limits',
    'calculate_entry_stops',
    'create_entry_signal',
    'track_rejection',
    # Signal helpers
    'build_indicators',
    'check_circuit_breaker',
]
