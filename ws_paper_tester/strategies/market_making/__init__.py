"""
Market Making Strategy v1.5.0

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
         - Refactored: Split into modular subfolder structure

Module Structure:
- config.py: Strategy metadata, CONFIG, SYMBOL_CONFIGS, validation
- calculations.py: Pure calculation functions (volatility, micro-price, etc.)
- signals.py: Signal generation logic
- lifecycle.py: on_start, on_fill, on_stop callbacks
"""

# Strategy metadata
from .config import STRATEGY_NAME, STRATEGY_VERSION, SYMBOLS, CONFIG, SYMBOL_CONFIGS

# Main signal generation
from .signals import generate_signal

# Lifecycle callbacks
from .lifecycle import on_start, on_fill, on_stop

# Re-export for backwards compatibility and explicit interface
__all__ = [
    # Required strategy interface
    'STRATEGY_NAME',
    'STRATEGY_VERSION',
    'SYMBOLS',
    'CONFIG',
    'SYMBOL_CONFIGS',
    'generate_signal',
    # Optional lifecycle callbacks
    'on_start',
    'on_fill',
    'on_stop',
]
