"""
Execution module - Order management and position tracking.

Phase 3 Components:
- OrderExecutionManager: Order lifecycle and Kraken API integration
- PositionTracker: Open position tracking and P&L monitoring

Phase 6 Components:
- TradingMode: Paper vs Live mode control
- PaperPortfolio: Simulated balance tracking
- PaperOrderExecutor: Paper trade execution with slippage/fees
- PaperPriceSource: Real-time price source for paper trading

Phase 8 Components:
- HodlBagManager: Automated profit allocation for long-term accumulation
"""

from .order_manager import (
    Order,
    OrderExecutionManager,
    OrderSide,
    OrderStatus,
    OrderType,
    ExecutionResult,
)
from .position_tracker import (
    Position,
    PositionSide,
    PositionStatus,
    PositionTracker,
    PositionSnapshot,
)
from .trading_mode import (
    TradingMode,
    TradingModeError,
    get_trading_mode,
    validate_trading_mode_on_startup,
    is_paper_mode,
    is_live_mode,
    get_db_table_prefix,
)
from .paper_portfolio import (
    PaperPortfolio,
    PaperTradeRecord,
    InsufficientBalanceError,
    InvalidTradeError,
)
from .paper_executor import (
    PaperOrderExecutor,
    PaperFillResult,
)
from .paper_price_source import (
    PaperPriceSource,
    MockPriceSource,
)
from .hodl_bag import (
    HodlBagManager,
    HodlBagState,
    HodlAllocation,
    HodlTransaction,
    HodlPending,
    HodlThresholds,
    TransactionType,
)

__all__ = [
    # Order Management
    'Order',
    'OrderExecutionManager',
    'OrderSide',
    'OrderStatus',
    'OrderType',
    'ExecutionResult',
    # Position Tracking
    'Position',
    'PositionSide',
    'PositionStatus',
    'PositionTracker',
    'PositionSnapshot',
    # Trading Mode (Phase 6)
    'TradingMode',
    'TradingModeError',
    'get_trading_mode',
    'validate_trading_mode_on_startup',
    'is_paper_mode',
    'is_live_mode',
    'get_db_table_prefix',
    # Paper Trading (Phase 6)
    'PaperPortfolio',
    'PaperTradeRecord',
    'InsufficientBalanceError',
    'InvalidTradeError',
    'PaperOrderExecutor',
    'PaperFillResult',
    'PaperPriceSource',
    'MockPriceSource',
    # Hodl Bag (Phase 8)
    'HodlBagManager',
    'HodlBagState',
    'HodlAllocation',
    'HodlTransaction',
    'HodlPending',
    'HodlThresholds',
    'TransactionType',
]
