"""
Orchestration API Routes - Endpoints for coordinator, portfolio, positions, and orders.

Phase 3 Endpoints:
- GET/POST /api/v1/coordinator/* - Coordinator control
- GET/POST /api/v1/portfolio/* - Portfolio allocation and rebalancing
- GET/POST /api/v1/positions/* - Position management
- GET/POST /api/v1/orders/* - Order management
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Any
import uuid

try:
    from fastapi import APIRouter, HTTPException, Query, Body
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None
    BaseModel = object

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

if FASTAPI_AVAILABLE:
    class ForceRebalanceRequest(BaseModel):
        """Request for forced portfolio rebalancing."""
        execution_strategy: str = Field(
            "limit",
            description="Execution strategy: immediate, dca_24h, limit_orders"
        )

    class ClosePositionRequest(BaseModel):
        """Request to close a position."""
        reason: str = Field("manual", description="Reason for closing")

    class ModifyPositionRequest(BaseModel):
        """Request to modify a position."""
        stop_loss: Optional[float] = Field(None, description="New stop-loss price")
        take_profit: Optional[float] = Field(None, description="New take-profit price")


# =============================================================================
# Router Factory
# =============================================================================

def create_orchestration_router(
    coordinator=None,
    portfolio_agent=None,
    position_tracker=None,
    order_manager=None,
    message_bus=None,
) -> 'APIRouter':
    """
    Create orchestration router with dependencies.

    Args:
        coordinator: CoordinatorAgent instance
        portfolio_agent: PortfolioRebalanceAgent instance
        position_tracker: PositionTracker instance
        order_manager: OrderExecutionManager instance
        message_bus: MessageBus instance

    Returns:
        Configured APIRouter
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")

    router = APIRouter(prefix="/api/v1", tags=["orchestration"])

    # -------------------------------------------------------------------------
    # Coordinator Endpoints
    # -------------------------------------------------------------------------

    @router.get("/coordinator/status")
    async def get_coordinator_status():
        """
        Get current coordinator status.

        Returns:
            Coordinator state, scheduled tasks, and statistics
        """
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        try:
            return coordinator.get_status()
        except Exception as e:
            logger.error(f"Failed to get coordinator status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/coordinator/pause")
    async def pause_coordinator():
        """
        Pause trading (scheduled tasks still run for analysis).

        Returns:
            Updated coordinator status
        """
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        try:
            await coordinator.pause()
            return {
                "status": "paused",
                "message": "Trading paused. Analysis tasks continue running.",
                "coordinator": coordinator.get_status(),
            }
        except Exception as e:
            logger.error(f"Failed to pause coordinator: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/coordinator/resume")
    async def resume_coordinator():
        """
        Resume trading from paused state.

        Returns:
            Updated coordinator status
        """
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        try:
            await coordinator.resume()
            return {
                "status": "running",
                "message": "Trading resumed.",
                "coordinator": coordinator.get_status(),
            }
        except Exception as e:
            logger.error(f"Failed to resume coordinator: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/coordinator/task/{task_name}/run")
    async def force_run_task(
        task_name: str,
        symbol: str = Query("BTC/USDT", description="Symbol to run task for")
    ):
        """
        Force immediate execution of a scheduled task.

        Args:
            task_name: Name of the task to run
            symbol: Symbol to run the task for

        Returns:
            Task execution result
        """
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        try:
            success = await coordinator.force_run_task(task_name, symbol)
            if success:
                return {
                    "status": "success",
                    "message": f"Task {task_name} executed for {symbol}",
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Task {task_name} not found"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to run task: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/coordinator/task/{task_name}/enable")
    async def enable_task(task_name: str):
        """Enable a scheduled task."""
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        success = coordinator.enable_task(task_name)
        if success:
            return {"status": "enabled", "task": task_name}
        raise HTTPException(status_code=404, detail=f"Task {task_name} not found")

    @router.post("/coordinator/task/{task_name}/disable")
    async def disable_task(task_name: str):
        """Disable a scheduled task."""
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        success = coordinator.disable_task(task_name)
        if success:
            return {"status": "disabled", "task": task_name}
        raise HTTPException(status_code=404, detail=f"Task {task_name} not found")

    # -------------------------------------------------------------------------
    # Portfolio Endpoints
    # -------------------------------------------------------------------------

    @router.get("/portfolio/allocation")
    async def get_portfolio_allocation():
        """
        Get current portfolio allocation.

        Returns:
            Current allocation percentages and deviations
        """
        if not portfolio_agent:
            raise HTTPException(status_code=503, detail="Portfolio Agent not initialized")

        try:
            allocation = await portfolio_agent.check_allocation()
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "allocation": allocation.to_dict(),
                "target": {
                    "btc_pct": float(portfolio_agent.target_btc_pct),
                    "xrp_pct": float(portfolio_agent.target_xrp_pct),
                    "usdt_pct": float(portfolio_agent.target_usdt_pct),
                },
                "threshold_pct": float(portfolio_agent.threshold_pct),
                "needs_rebalancing": allocation.max_deviation_pct >= portfolio_agent.threshold_pct,
            }
        except Exception as e:
            logger.error(f"Failed to get portfolio allocation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/portfolio/rebalance")
    async def force_rebalance(
        request: ForceRebalanceRequest = Body(default_factory=ForceRebalanceRequest)
    ):
        """
        Force portfolio rebalancing.

        Args:
            request: Rebalancing options

        Returns:
            Rebalancing decision and trades
        """
        if not portfolio_agent:
            raise HTTPException(status_code=503, detail="Portfolio Agent not initialized")

        try:
            output = await portfolio_agent.process(force=True)
            return {
                "status": output.action,
                "execution_strategy": output.execution_strategy,
                "trades": [t.to_dict() for t in output.trades],
                "allocation": output.current_allocation.to_dict() if output.current_allocation else None,
                "reasoning": output.reasoning,
            }
        except Exception as e:
            logger.error(f"Failed to rebalance portfolio: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------------------------------
    # Position Endpoints
    # -------------------------------------------------------------------------

    @router.get("/positions")
    async def get_positions(
        symbol: Optional[str] = Query(None, description="Filter by symbol"),
        status: str = Query("open", description="Filter by status: open, closed, all")
    ):
        """
        Get positions.

        Args:
            symbol: Optional symbol filter
            status: Status filter (open, closed, all)

        Returns:
            List of positions
        """
        if not position_tracker:
            raise HTTPException(status_code=503, detail="Position Tracker not initialized")

        try:
            if status == "open":
                positions = await position_tracker.get_open_positions(symbol)
            elif status == "closed":
                positions = await position_tracker.get_closed_positions(symbol)
            else:
                open_positions = await position_tracker.get_open_positions(symbol)
                closed_positions = await position_tracker.get_closed_positions(symbol, limit=50)
                positions = open_positions + closed_positions

            return {
                "count": len(positions),
                "positions": [p.to_dict() for p in positions],
            }
        except Exception as e:
            logger.error(f"Failed to get positions: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/positions/exposure")
    async def get_exposure():
        """Get total portfolio exposure across all positions."""
        if not position_tracker:
            raise HTTPException(status_code=503, detail="Position Tracker not initialized")

        try:
            exposure = await position_tracker.get_total_exposure()
            pnl = await position_tracker.get_total_unrealized_pnl()
            return {
                **exposure,
                **pnl,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to get exposure: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/positions/{position_id}")
    async def get_position(position_id: str):
        """Get a specific position by ID."""
        if not position_tracker:
            raise HTTPException(status_code=503, detail="Position Tracker not initialized")

        try:
            position = await position_tracker.get_position(position_id)
            if not position:
                raise HTTPException(status_code=404, detail="Position not found")
            return position.to_dict()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get position: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/positions/{position_id}/close")
    async def close_position(
        position_id: str,
        request: ClosePositionRequest = Body(default_factory=ClosePositionRequest),
        exit_price: float = Query(..., description="Exit price for the position")
    ):
        """
        Close an open position.

        Args:
            position_id: Position ID to close
            request: Close request details
            exit_price: Exit price for P&L calculation

        Returns:
            Closed position details
        """
        if not position_tracker:
            raise HTTPException(status_code=503, detail="Position Tracker not initialized")

        try:
            position = await position_tracker.close_position(
                position_id,
                Decimal(str(exit_price)),
                reason=request.reason,
            )
            if not position:
                raise HTTPException(status_code=404, detail="Position not found")
            return {
                "status": "closed",
                "position": position.to_dict(),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to close position: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.patch("/positions/{position_id}")
    async def modify_position(
        position_id: str,
        request: ModifyPositionRequest = Body(...)
    ):
        """
        Modify position stop-loss or take-profit.

        Args:
            position_id: Position ID to modify
            request: Modification details

        Returns:
            Modified position details
        """
        if not position_tracker:
            raise HTTPException(status_code=503, detail="Position Tracker not initialized")

        try:
            position = await position_tracker.modify_position(
                position_id,
                stop_loss=Decimal(str(request.stop_loss)) if request.stop_loss else None,
                take_profit=Decimal(str(request.take_profit)) if request.take_profit else None,
            )
            if not position:
                raise HTTPException(status_code=404, detail="Position not found")
            return position.to_dict()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to modify position: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------------------------------
    # Order Endpoints
    # -------------------------------------------------------------------------

    @router.get("/orders")
    async def get_orders(
        symbol: Optional[str] = Query(None, description="Filter by symbol"),
    ):
        """
        Get open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        if not order_manager:
            raise HTTPException(status_code=503, detail="Order Manager not initialized")

        try:
            orders = await order_manager.get_open_orders(symbol)
            return {
                "count": len(orders),
                "orders": [o.to_dict() for o in orders],
            }
        except Exception as e:
            logger.error(f"Failed to get orders: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/orders/{order_id}")
    async def get_order(order_id: str):
        """Get a specific order by ID."""
        if not order_manager:
            raise HTTPException(status_code=503, detail="Order Manager not initialized")

        try:
            order = await order_manager.get_order(order_id)
            if not order:
                raise HTTPException(status_code=404, detail="Order not found")
            return order.to_dict()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get order: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/orders/{order_id}/cancel")
    async def cancel_order(order_id: str):
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation result
        """
        if not order_manager:
            raise HTTPException(status_code=503, detail="Order Manager not initialized")

        try:
            success = await order_manager.cancel_order(order_id)
            if success:
                return {"status": "cancelled", "order_id": order_id}
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to cancel order (may already be filled/cancelled)"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/orders/sync")
    async def sync_orders():
        """Sync local order state with exchange."""
        if not order_manager:
            raise HTTPException(status_code=503, detail="Order Manager not initialized")

        try:
            synced = await order_manager.sync_with_exchange()
            return {
                "status": "synced",
                "orders_synced": synced,
            }
        except Exception as e:
            logger.error(f"Failed to sync orders: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------------------------------
    # Statistics Endpoints
    # -------------------------------------------------------------------------

    @router.get("/stats/execution")
    async def get_execution_stats():
        """Get execution statistics."""
        stats = {}

        if order_manager:
            stats["orders"] = order_manager.get_stats()

        if position_tracker:
            stats["positions"] = position_tracker.get_stats()

        if message_bus:
            stats["message_bus"] = message_bus.get_stats()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **stats,
        }

    return router
