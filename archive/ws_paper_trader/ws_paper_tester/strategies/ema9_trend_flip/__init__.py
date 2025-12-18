"""
EMA-9 Trend Flip Strategy v1.0.1

A trend-following strategy based on the 9-period Exponential Moving Average (EMA).
Entry signals are generated when price "flips" from consistently closing on one side
of the EMA to closing on the opposite side, indicating a potential trend change.

Strategy Philosophy:
- The EMA flip IS the profit exit (no fixed take_profit_pct)
- Stop loss is for PROTECTION ONLY (wide stop for catastrophic moves)
- Exit on flip is ALWAYS enabled - it's the core strategy

Strategy Logic (1H Timeframe):
- Use 1H candles (pre-aggregated or built from 1m)
- Calculate EMA-9 on hourly candle CLOSE prices (full candle confirmation)
- Track N consecutive candles closing above/below EMA
- Entry: When candle closes on opposite side after N consecutive candles
- Exit: Opposing EMA flip (primary), stop loss (protection), max hold time (timeout)

Based on research: ws_paper_tester/docs/development/plans/ema9/ema-9-strategy-analysis.md

Version History:
- 1.0.1: Conceptual alignment - flip IS the exit
         - Removed take_profit_pct (flip is the profit exit)
         - Removed exit_on_flip toggle (always exit on flip)
         - Changed to CLOSE price (full candle confirmation)
         - Widened stop loss to 2.5% or 2x ATR (protection only)
         - Restructured exit priority: flip first, then stop loss
- 1.0.0: Initial implementation based on EMA-9 Strategy Analysis
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
)

from .risk import (
    check_position_limits,
    calculate_stop_loss,
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
    # Risk
    'check_position_limits',
    'calculate_stop_loss',
    'create_entry_signal',
    'track_rejection',
    # Signal helpers
    'build_indicators',
    'check_circuit_breaker',
]
