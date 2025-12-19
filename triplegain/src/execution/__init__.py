"""
Execution module - Order management and position tracking.

Phase 3 Components:
- OrderExecutionManager: Order lifecycle and Kraken API integration
- PositionTracker: Open position tracking and P&L monitoring
"""

from .order_manager import (
    Order,
    OrderExecutionManager,
    OrderStatus,
    OrderType,
    ExecutionResult,
)
from .position_tracker import (
    Position,
    PositionTracker,
    PositionSnapshot,
)

__all__ = [
    'Order',
    'OrderExecutionManager',
    'OrderStatus',
    'OrderType',
    'ExecutionResult',
    'Position',
    'PositionTracker',
    'PositionSnapshot',
]
