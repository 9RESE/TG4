"""
Paper Trading API Routes - Endpoints for paper trading management.

Phase 6 Endpoints:
- GET /api/v1/paper/status - Paper trading status and portfolio
- GET /api/v1/paper/portfolio - Current paper portfolio details
- GET /api/v1/paper/trades - Paper trade history
- POST /api/v1/paper/trade - Execute a paper trade
- POST /api/v1/paper/reset - Reset paper portfolio
- GET /api/v1/paper/performance - Performance metrics
- GET /api/v1/paper/positions - Open paper positions

Security:
- All endpoints require authentication
- Reset requires ADMIN role
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, HTTPException, Query, Body, Depends, Request
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None
    BaseModel = object

from .validation import validate_symbol_or_raise
from .security import (
    get_current_user,
    require_role,
    User,
    UserRole,
    log_security_event,
    SecurityEventType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

if FASTAPI_AVAILABLE:
    class PaperTradeRequest(BaseModel):
        """Request body for paper trade execution."""
        symbol: str = Field(..., description="Trading symbol (e.g., BTC/USDT)")
        side: str = Field(..., pattern="^(buy|sell)$", description="Trade side")
        size_usd: float = Field(..., gt=0, le=10000, description="Trade size in USD")
        entry_price: Optional[float] = Field(None, gt=0, description="Entry price (optional for market)")
        stop_loss: Optional[float] = Field(None, gt=0, description="Stop-loss price")
        take_profit: Optional[float] = Field(None, gt=0, description="Take-profit price")
        leverage: int = Field(1, ge=1, le=5, description="Leverage (1-5)")

    class PaperResetRequest(BaseModel):
        """Request body for paper portfolio reset."""
        new_balances: Optional[Dict[str, float]] = Field(
            None,
            description="Optional new starting balances. If None, uses config defaults."
        )
        confirm: bool = Field(
            ...,
            description="Must be true to confirm reset"
        )

    class PaperPositionCloseRequest(BaseModel):
        """Request body for closing a paper position."""
        position_id: str = Field(..., description="Position ID to close")
        price: Optional[float] = Field(None, gt=0, description="Close price (current if None)")


def create_paper_trading_routes(app_state: Dict[str, Any]) -> 'APIRouter':
    """
    Create paper trading API routes.

    Args:
        app_state: Application state with coordinator, config, etc.

    Returns:
        FastAPI router with paper trading endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")

    router = APIRouter(prefix="/api/v1/paper", tags=["Paper Trading"])

    def get_paper_components():
        """Get paper trading components from coordinator."""
        coordinator = app_state.get("coordinator")
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        from ..execution.trading_mode import TradingMode
        if coordinator.trading_mode != TradingMode.PAPER:
            raise HTTPException(
                status_code=400,
                detail="Paper trading endpoints only available in PAPER mode"
            )

        return {
            "portfolio": coordinator.paper_portfolio,
            "executor": coordinator.paper_executor,
            "price_source": coordinator.paper_price_source,
            "coordinator": coordinator,
        }

    # =========================================================================
    # GET /api/v1/paper/status - Paper trading status
    # =========================================================================

    @router.get("/status")
    async def get_paper_status(
        current_user: User = Depends(get_current_user),
    ) -> Dict[str, Any]:
        """
        Get paper trading status and overview.

        Returns trading mode, portfolio summary, and system health.
        """
        components = get_paper_components()
        coordinator = components["coordinator"]

        status = coordinator.get_status()
        return {
            "trading_mode": "paper",
            "session_id": components["portfolio"].session_id if components["portfolio"] else None,
            "paper_trading": status.get("paper_trading", {}),
            "coordinator_state": status.get("state"),
        }

    # =========================================================================
    # GET /api/v1/paper/portfolio - Paper portfolio details
    # =========================================================================

    @router.get("/portfolio")
    async def get_paper_portfolio(
        current_user: User = Depends(get_current_user),
    ) -> Dict[str, Any]:
        """
        Get current paper portfolio details.

        Returns balances, P&L, trade statistics.
        """
        components = get_paper_components()
        portfolio = components["portfolio"]
        price_source = components["price_source"]

        if not portfolio:
            raise HTTPException(status_code=503, detail="Paper portfolio not initialized")

        # Get current prices for equity calculation
        current_prices = {}
        if price_source:
            current_prices = {
                symbol: price
                for symbol, price in price_source.get_all_prices().items()
            }

        return {
            "session_id": portfolio.session_id,
            "balances": portfolio.get_balances_dict(),
            "initial_balances": {k: float(v) for k, v in portfolio.initial_balances.items()},
            "pnl_summary": portfolio.get_pnl_summary(current_prices),
            "trade_count": portfolio.trade_count,
            "total_fees_paid": float(portfolio.total_fees_paid),
            "created_at": portfolio.created_at.isoformat(),
            "last_updated": portfolio.last_updated.isoformat(),
        }

    # =========================================================================
    # GET /api/v1/paper/trades - Paper trade history
    # =========================================================================

    @router.get("/trades")
    async def get_paper_trades(
        current_user: User = Depends(get_current_user),
        limit: int = Query(50, ge=1, le=500, description="Max trades to return"),
        symbol: Optional[str] = Query(None, description="Filter by symbol"),
    ) -> Dict[str, Any]:
        """
        Get paper trade history.

        Returns list of executed paper trades with details.
        """
        components = get_paper_components()
        portfolio = components["portfolio"]

        if not portfolio:
            raise HTTPException(status_code=503, detail="Paper portfolio not initialized")

        trades = list(reversed(portfolio.trade_history))[:limit]

        if symbol:
            symbol = symbol.upper()
            trades = [t for t in trades if t.symbol == symbol]

        return {
            "trades": [t.to_dict() for t in trades],
            "total_count": len(portfolio.trade_history),
            "returned_count": len(trades),
        }

    # =========================================================================
    # POST /api/v1/paper/trade - Execute a paper trade
    # =========================================================================

    @router.post("/trade")
    async def execute_paper_trade(
        request: PaperTradeRequest,
        current_user: User = Depends(get_current_user),
    ) -> Dict[str, Any]:
        """
        Execute a paper trade.

        Validates the trade through risk engine and executes through paper executor.
        """
        components = get_paper_components()
        executor = components["executor"]
        coordinator = components["coordinator"]

        if not executor:
            raise HTTPException(status_code=503, detail="Paper executor not initialized")

        # Validate symbol
        validate_symbol_or_raise(request.symbol)

        # Create trade proposal
        from ..risk.rules_engine import TradeProposal

        # MEDIUM-01: Use None for entry_price when not specified (not 0)
        # 0 could be misinterpreted as a valid price, None clearly means "market order"
        proposal = TradeProposal(
            symbol=request.symbol.upper(),
            side=request.side,
            size_usd=request.size_usd,
            entry_price=request.entry_price if request.entry_price and request.entry_price > 0 else None,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            leverage=request.leverage,
            confidence=0.75,  # Default confidence for manual paper trades
            regime="ranging",
        )

        # Validate with risk engine if available
        if coordinator.risk_engine:
            validation = coordinator.risk_engine.validate_trade(proposal)
            if not validation.is_approved():
                return {
                    "success": False,
                    "error": "Trade rejected by risk engine",
                    "rejections": validation.rejections,
                }
            proposal = validation.modified_proposal or proposal

        # Execute through paper executor
        try:
            result = await executor.execute_trade(proposal)

            log_security_event(
                SecurityEventType.DATA_ACCESS,
                current_user.id,
                f"Paper trade executed: {request.side} {request.size_usd} USD of {request.symbol}",
            )

            return {
                "success": result.success,
                "order_id": result.order.id if result.order else None,
                "position_id": result.position_id,
                "error_message": result.error_message,
                "execution_time_ms": result.execution_time_ms,
            }

        except Exception as e:
            logger.error(f"Paper trade execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # POST /api/v1/paper/reset - Reset paper portfolio
    # =========================================================================

    @router.post("/reset")
    async def reset_paper_portfolio(
        request: PaperResetRequest,
        current_user: User = Depends(require_role(UserRole.ADMIN)),
    ) -> Dict[str, Any]:
        """
        Reset paper portfolio to initial state.

        Requires ADMIN role. Clears all trades and positions.
        """
        if not request.confirm:
            raise HTTPException(
                status_code=400,
                detail="Reset requires confirm=true"
            )

        components = get_paper_components()
        portfolio = components["portfolio"]

        if not portfolio:
            raise HTTPException(status_code=503, detail="Paper portfolio not initialized")

        old_balances = portfolio.get_balances_dict()
        old_trade_count = portfolio.trade_count

        # Reset portfolio
        if request.new_balances:
            new_balances = {k: Decimal(str(v)) for k, v in request.new_balances.items()}
            portfolio.reset(new_balances)
        else:
            portfolio.reset()

        # Audit logging: Use RISK_RESET for portfolio resets (critical operation)
        log_security_event(
            SecurityEventType.RISK_RESET,
            current_user.id,
            f"Paper portfolio reset by ADMIN. Session: {portfolio.session_id}, "
            f"Old trades: {old_trade_count}, Old balances: {old_balances}, "
            f"New balances: {portfolio.get_balances_dict()}",
        )

        return {
            "success": True,
            "message": "Paper portfolio reset successfully",
            "old_balances": old_balances,
            "old_trade_count": old_trade_count,
            "new_balances": portfolio.get_balances_dict(),
            "session_id": portfolio.session_id,
        }

    # =========================================================================
    # GET /api/v1/paper/performance - Performance metrics
    # =========================================================================

    @router.get("/performance")
    async def get_paper_performance(
        current_user: User = Depends(get_current_user),
    ) -> Dict[str, Any]:
        """
        Get paper trading performance metrics.

        Returns P&L, win rate, Sharpe-like metrics.
        """
        components = get_paper_components()
        portfolio = components["portfolio"]
        executor = components["executor"]
        price_source = components["price_source"]

        if not portfolio:
            raise HTTPException(status_code=503, detail="Paper portfolio not initialized")

        # Get current prices
        current_prices = {}
        if price_source:
            current_prices = {
                symbol: price
                for symbol, price in price_source.get_all_prices().items()
            }

        pnl_summary = portfolio.get_pnl_summary(current_prices)

        return {
            "session_id": portfolio.session_id,
            "duration_hours": (
                datetime.now(timezone.utc) - portfolio.created_at
            ).total_seconds() / 3600,
            "performance": pnl_summary,
            "executor_stats": executor.get_stats() if executor else {},
            "price_source_stats": price_source.get_stats() if price_source else {},
        }

    # =========================================================================
    # GET /api/v1/paper/positions - Open paper positions
    # =========================================================================

    @router.get("/positions")
    async def get_paper_positions(
        current_user: User = Depends(get_current_user),
        symbol: Optional[str] = Query(None, description="Filter by symbol"),
    ) -> Dict[str, Any]:
        """
        Get open paper positions.

        Returns list of current open positions with unrealized P&L.
        """
        components = get_paper_components()
        executor = components["executor"]
        price_source = components["price_source"]

        if not executor or not executor.position_tracker:
            raise HTTPException(
                status_code=503,
                detail="Position tracker not initialized"
            )

        positions = await executor.position_tracker.get_open_positions(symbol=symbol)

        # Get current prices for P&L
        if price_source:
            current_prices = {
                sym: price
                for sym, price in price_source.get_all_prices().items()
            }
            await executor.position_tracker.update_prices(current_prices)

        return {
            "positions": [pos.to_dict() for pos in positions],
            "count": len(positions),
        }

    # =========================================================================
    # POST /api/v1/paper/positions/close - Close a paper position
    # =========================================================================

    @router.post("/positions/close")
    async def close_paper_position(
        request: PaperPositionCloseRequest,
        current_user: User = Depends(get_current_user),
    ) -> Dict[str, Any]:
        """
        Close an open paper position.

        Returns closed position details with realized P&L.
        """
        components = get_paper_components()
        executor = components["executor"]
        price_source = components["price_source"]

        if not executor or not executor.position_tracker:
            raise HTTPException(
                status_code=503,
                detail="Position tracker not initialized"
            )

        # Get position
        position = await executor.position_tracker.get_position(request.position_id)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")

        # Determine close price
        if request.price:
            close_price = Decimal(str(request.price))
        elif price_source:
            close_price = price_source.get_price(position.symbol)
            if not close_price:
                raise HTTPException(
                    status_code=400,
                    detail=f"No price available for {position.symbol}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Price required (no price source available)"
            )

        # Close position
        closed = await executor.position_tracker.close_position(
            position_id=request.position_id,
            exit_price=close_price,
            reason="manual_close",
        )

        if not closed:
            raise HTTPException(status_code=500, detail="Failed to close position")

        # Update portfolio P&L
        if components["portfolio"]:
            is_win = closed.realized_pnl > 0
            components["portfolio"].record_realized_pnl(closed.realized_pnl, is_win)

        log_security_event(
            SecurityEventType.DATA_ACCESS,
            current_user.id,
            f"Paper position closed: {request.position_id}, P&L: {closed.realized_pnl}",
        )

        return {
            "success": True,
            "position": closed.to_dict(),
            "realized_pnl": float(closed.realized_pnl),
        }

    return router


# =============================================================================
# Router factory for app initialization
# =============================================================================

def get_paper_trading_router(app_state: Dict[str, Any]) -> Optional['APIRouter']:
    """
    Get paper trading router if FastAPI is available.

    Args:
        app_state: Application state dictionary

    Returns:
        Router or None if FastAPI unavailable
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available - paper trading routes disabled")
        return None

    return create_paper_trading_routes(app_state)
