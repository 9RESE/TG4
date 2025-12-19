"""
Orchestration API Routes - Endpoints for coordinator, portfolio, positions, and orders.

Phase 3 Endpoints:
- GET/POST /api/v1/coordinator/* - Coordinator control
- GET/POST /api/v1/portfolio/* - Portfolio allocation and rebalancing
- GET/POST /api/v1/positions/* - Position management
- GET/POST /api/v1/orders/* - Order management

Security Fixes Applied (Phase 4 Review):
- Findings 1-3: All endpoints require authentication
- Finding 5: UUID validation for position/order IDs
- Finding 9: Confirmation mechanism for destructive operations
- Finding 19: Symbol validation
- Finding 20: Task name validation
- Finding 27: Pagination for list endpoints
"""

import logging
import secrets
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Any
from uuid import UUID

try:
    from fastapi import APIRouter, HTTPException, Query, Body, Depends, Request
    from pydantic import BaseModel, Field, UUID4
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

# Valid coordinator tasks (Finding 20)
VALID_COORDINATOR_TASKS = frozenset({
    "technical_analysis",
    "regime_detection",
    "trading_decision",
    "portfolio_rebalance",
    "risk_check",
})

# In-memory confirmation token store (use Redis in production)
_confirmation_tokens: dict[str, tuple[str, datetime]] = {}


# =============================================================================
# Pydantic Models
# =============================================================================

if FASTAPI_AVAILABLE:
    class ForceRebalanceRequest(BaseModel):
        """Request for forced portfolio rebalancing."""
        execution_strategy: str = Field(
            "limit",
            pattern="^(immediate|dca_24h|limit_orders|limit)$",
            description="Execution strategy: immediate, dca_24h, limit_orders, limit"
        )
        confirmation_token: Optional[str] = Field(
            None,
            description="Confirmation token from GET /portfolio/rebalance/confirm"
        )

    class ClosePositionRequest(BaseModel):
        """Request to close a position (Finding 9: requires confirmation)."""
        exit_price: float = Field(..., gt=0, description="Exit price for P&L calculation")
        reason: str = Field("manual", description="Reason for closing")
        confirmation_token: str = Field(
            ...,
            description="One-time confirmation token from GET /positions/{id}/confirm"
        )

    class ModifyPositionRequest(BaseModel):
        """Request to modify a position."""
        stop_loss: Optional[float] = Field(None, gt=0, description="New stop-loss price")
        take_profit: Optional[float] = Field(None, gt=0, description="New take-profit price")

    class CancelOrderRequest(BaseModel):
        """Request to cancel an order (Finding 9: requires confirmation)."""
        confirmation_token: str = Field(
            ...,
            description="One-time confirmation token from GET /orders/{id}/confirm"
        )
        reason: str = Field("manual", description="Reason for cancellation")


def _generate_confirmation_token(resource_id: str, action: str) -> str:
    """Generate a one-time confirmation token for destructive operations."""
    token = secrets.token_urlsafe(32)
    key = f"{action}:{resource_id}"
    # Token expires in 5 minutes
    _confirmation_tokens[key] = (token, datetime.now(timezone.utc))
    return token


def _verify_confirmation_token(resource_id: str, action: str, token: str) -> bool:
    """Verify and consume a confirmation token."""
    key = f"{action}:{resource_id}"
    stored = _confirmation_tokens.get(key)
    if not stored:
        return False

    stored_token, created_at = stored
    # Check expiry (5 minutes)
    if (datetime.now(timezone.utc) - created_at).total_seconds() > 300:
        del _confirmation_tokens[key]
        return False

    if stored_token != token:
        return False

    # Consume token (one-time use)
    del _confirmation_tokens[key]
    return True


def _validate_uuid(value: str, field_name: str) -> str:
    """Validate that a string is a valid UUID (Finding 5)."""
    try:
        UUID(value, version=4)
        return value
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} format: must be a valid UUID"
        )


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
    async def get_coordinator_status(
        user: User = Depends(get_current_user),
    ):
        """
        Get current coordinator status.

        Requires authentication.

        Returns:
            Coordinator state, scheduled tasks, and statistics
        """
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        try:
            return coordinator.get_status()
        except Exception as e:
            logger.error(f"Failed to get coordinator status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error getting coordinator status")

    @router.post("/coordinator/pause")
    async def pause_coordinator(
        request: Request,
        user: User = Depends(require_role(UserRole.ADMIN)),
    ):
        """
        Pause trading (scheduled tasks still run for analysis).

        Requires ADMIN role.

        Returns:
            Updated coordinator status
        """
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        client_ip = request.client.host if request.client else "unknown"

        try:
            await log_security_event(
                SecurityEventType.TRADING_PAUSED,
                user.user_id,
                client_ip,
                {"action": "pause"},
                request,
            )

            await coordinator.pause()
            return {
                "status": "paused",
                "message": "Trading paused. Analysis tasks continue running.",
                "paused_by": user.user_id,
                "coordinator": coordinator.get_status(),
            }
        except Exception as e:
            logger.error(f"Failed to pause coordinator: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error pausing coordinator")

    @router.post("/coordinator/resume")
    async def resume_coordinator(
        request: Request,
        user: User = Depends(require_role(UserRole.ADMIN)),
    ):
        """
        Resume trading from paused state.

        Requires ADMIN role.

        Returns:
            Updated coordinator status
        """
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        client_ip = request.client.host if request.client else "unknown"

        try:
            await log_security_event(
                SecurityEventType.TRADING_RESUMED,
                user.user_id,
                client_ip,
                {"action": "resume"},
                request,
            )

            await coordinator.resume()
            return {
                "status": "running",
                "message": "Trading resumed.",
                "resumed_by": user.user_id,
                "coordinator": coordinator.get_status(),
            }
        except Exception as e:
            logger.error(f"Failed to resume coordinator: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error resuming coordinator")

    @router.post("/coordinator/task/{task_name}/run")
    async def force_run_task(
        task_name: str,
        symbol: str = Query("BTC/USDT", description="Symbol to run task for"),
        user: User = Depends(require_role(UserRole.TRADER)),
    ):
        """
        Force immediate execution of a scheduled task.

        Requires TRADER role or higher.

        Args:
            task_name: Name of the task to run
            symbol: Symbol to run the task for

        Returns:
            Task execution result
        """
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        # Finding 20: Validate task name
        if task_name not in VALID_COORDINATOR_TASKS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task name: '{task_name}'. Valid: {', '.join(sorted(VALID_COORDINATOR_TASKS))}"
            )

        # Validate symbol (Finding 19)
        symbol = validate_symbol_or_raise(symbol, strict=False)

        try:
            success = await coordinator.force_run_task(task_name, symbol)
            if success:
                return {
                    "status": "success",
                    "message": f"Task {task_name} executed for {symbol}",
                    "triggered_by": user.user_id,
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
            raise HTTPException(status_code=500, detail="Internal server error running task")

    @router.post("/coordinator/task/{task_name}/enable")
    async def enable_task(
        task_name: str,
        user: User = Depends(require_role(UserRole.ADMIN)),
    ):
        """
        Enable a scheduled task.

        Requires ADMIN role.
        """
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        # Finding 20: Validate task name
        if task_name not in VALID_COORDINATOR_TASKS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task name: '{task_name}'. Valid: {', '.join(sorted(VALID_COORDINATOR_TASKS))}"
            )

        success = coordinator.enable_task(task_name)
        if success:
            return {"status": "enabled", "task": task_name, "enabled_by": user.user_id}
        raise HTTPException(status_code=404, detail=f"Task {task_name} not found")

    @router.post("/coordinator/task/{task_name}/disable")
    async def disable_task(
        task_name: str,
        user: User = Depends(require_role(UserRole.ADMIN)),
    ):
        """
        Disable a scheduled task.

        Requires ADMIN role.
        """
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        # Finding 20: Validate task name
        if task_name not in VALID_COORDINATOR_TASKS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task name: '{task_name}'. Valid: {', '.join(sorted(VALID_COORDINATOR_TASKS))}"
            )

        success = coordinator.disable_task(task_name)
        if success:
            return {"status": "disabled", "task": task_name, "disabled_by": user.user_id}
        raise HTTPException(status_code=404, detail=f"Task {task_name} not found")

    # -------------------------------------------------------------------------
    # Portfolio Endpoints
    # -------------------------------------------------------------------------

    @router.get("/portfolio/allocation")
    async def get_portfolio_allocation(
        user: User = Depends(get_current_user),
    ):
        """
        Get current portfolio allocation.

        Requires authentication.

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
            raise HTTPException(status_code=500, detail="Internal server error getting portfolio allocation")

    @router.get("/portfolio/rebalance/confirm")
    async def get_rebalance_confirmation(
        user: User = Depends(require_role(UserRole.TRADER)),
    ):
        """
        Get a confirmation token for portfolio rebalancing.

        Requires TRADER role or higher.
        The token must be included in the POST /portfolio/rebalance request.

        Returns:
            Confirmation token valid for 5 minutes
        """
        token = _generate_confirmation_token("portfolio", "rebalance")
        return {
            "confirmation_token": token,
            "expires_in_seconds": 300,
            "message": "Include this token in your rebalance request",
        }

    @router.post("/portfolio/rebalance")
    async def force_rebalance(
        http_request: Request,
        request: ForceRebalanceRequest = Body(default_factory=ForceRebalanceRequest),
        user: User = Depends(require_role(UserRole.TRADER)),
    ):
        """
        Force portfolio rebalancing.

        Requires TRADER role or higher.
        Requires confirmation token from GET /portfolio/rebalance/confirm.

        Args:
            request: Rebalancing options with confirmation token

        Returns:
            Rebalancing decision and trades
        """
        if not portfolio_agent:
            raise HTTPException(status_code=503, detail="Portfolio Agent not initialized")

        # Finding 9: Verify confirmation token
        if request.confirmation_token:
            if not _verify_confirmation_token("portfolio", "rebalance", request.confirmation_token):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid or expired confirmation token. Get a new one from GET /portfolio/rebalance/confirm"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Confirmation token required. Get one from GET /portfolio/rebalance/confirm"
            )

        client_ip = http_request.client.host if http_request.client else "unknown"

        try:
            await log_security_event(
                SecurityEventType.CRITICAL_OPERATION,
                user.user_id,
                client_ip,
                {"action": "portfolio_rebalance", "strategy": request.execution_strategy},
                http_request,
            )

            output = await portfolio_agent.process(force=True)
            return {
                "status": output.action,
                "execution_strategy": output.execution_strategy,
                "trades": [t.to_dict() for t in output.trades],
                "allocation": output.current_allocation.to_dict() if output.current_allocation else None,
                "reasoning": output.reasoning,
                "triggered_by": user.user_id,
            }
        except Exception as e:
            logger.error(f"Failed to rebalance portfolio: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error rebalancing portfolio")

    # -------------------------------------------------------------------------
    # Position Endpoints
    # -------------------------------------------------------------------------

    @router.get("/positions")
    async def get_positions(
        symbol: Optional[str] = Query(None, description="Filter by symbol"),
        status: str = Query("open", pattern="^(open|closed|all)$", description="Filter by status"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
        limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
        user: User = Depends(get_current_user),
    ):
        """
        Get positions with pagination (Finding 27).

        Requires authentication.

        Args:
            symbol: Optional symbol filter
            status: Status filter (open, closed, all)
            offset: Pagination offset
            limit: Maximum results to return

        Returns:
            Paginated list of positions
        """
        if not position_tracker:
            raise HTTPException(status_code=503, detail="Position Tracker not initialized")

        # Finding 19: Validate symbol if provided
        if symbol:
            symbol = validate_symbol_or_raise(symbol, strict=False)

        try:
            if status == "open":
                positions = await position_tracker.get_open_positions(symbol)
            elif status == "closed":
                positions = await position_tracker.get_closed_positions(symbol, limit=limit + offset)
            else:
                open_positions = await position_tracker.get_open_positions(symbol)
                closed_positions = await position_tracker.get_closed_positions(symbol, limit=limit + offset)
                positions = open_positions + closed_positions

            # Apply pagination
            total = len(positions)
            positions = positions[offset:offset + limit]

            return {
                "total": total,
                "offset": offset,
                "limit": limit,
                "count": len(positions),
                "positions": [p.to_dict() for p in positions],
            }
        except Exception as e:
            logger.error(f"Failed to get positions: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error getting positions")

    @router.get("/positions/exposure")
    async def get_exposure(
        user: User = Depends(get_current_user),
    ):
        """
        Get total portfolio exposure across all positions.

        Requires authentication.
        """
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
            raise HTTPException(status_code=500, detail="Internal server error getting exposure")

    @router.get("/positions/{position_id}")
    async def get_position(
        position_id: str,
        user: User = Depends(get_current_user),
    ):
        """
        Get a specific position by ID.

        Requires authentication.
        """
        if not position_tracker:
            raise HTTPException(status_code=503, detail="Position Tracker not initialized")

        # Finding 5: Validate UUID format
        position_id = _validate_uuid(position_id, "position_id")

        try:
            position = await position_tracker.get_position(position_id)
            if not position:
                raise HTTPException(status_code=404, detail="Position not found")
            return position.to_dict()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get position: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error getting position")

    @router.get("/positions/{position_id}/confirm")
    async def get_position_close_confirmation(
        position_id: str,
        user: User = Depends(require_role(UserRole.TRADER)),
    ):
        """
        Get a confirmation token for closing a position (Finding 9).

        Requires TRADER role or higher.

        Returns:
            Confirmation token valid for 5 minutes
        """
        # Validate UUID format
        position_id = _validate_uuid(position_id, "position_id")

        if not position_tracker:
            raise HTTPException(status_code=503, detail="Position Tracker not initialized")

        # Verify position exists
        position = await position_tracker.get_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")

        token = _generate_confirmation_token(position_id, "close_position")
        return {
            "position_id": position_id,
            "confirmation_token": token,
            "expires_in_seconds": 300,
            "message": "Include this token in your close request",
            "position_summary": {
                "symbol": position.symbol,
                "side": position.side,
                "size": str(position.size),
                "entry_price": str(position.entry_price),
            },
        }

    @router.post("/positions/{position_id}/close")
    async def close_position(
        position_id: str,
        http_request: Request,
        request: ClosePositionRequest = Body(...),
        user: User = Depends(require_role(UserRole.TRADER)),
    ):
        """
        Close an open position.

        Requires TRADER role or higher.
        Requires confirmation token from GET /positions/{id}/confirm.

        Args:
            position_id: Position ID to close
            request: Close request details (exit_price, reason, confirmation_token)

        Returns:
            Closed position details
        """
        if not position_tracker:
            raise HTTPException(status_code=503, detail="Position Tracker not initialized")

        # Finding 5: Validate UUID format
        position_id = _validate_uuid(position_id, "position_id")

        # Finding 9: Verify confirmation token
        if not _verify_confirmation_token(position_id, "close_position", request.confirmation_token):
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired confirmation token. Get a new one from GET /positions/{id}/confirm"
            )

        client_ip = http_request.client.host if http_request.client else "unknown"

        try:
            # Log critical operation
            await log_security_event(
                SecurityEventType.POSITION_CLOSE,
                user.user_id,
                client_ip,
                {
                    "position_id": position_id,
                    "exit_price": request.exit_price,
                    "reason": request.reason,
                },
                http_request,
            )

            position = await position_tracker.close_position(
                position_id,
                Decimal(str(request.exit_price)),
                reason=request.reason,
            )
            if not position:
                raise HTTPException(status_code=404, detail="Position not found")
            return {
                "status": "closed",
                "closed_by": user.user_id,
                "position": position.to_dict(),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to close position: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error closing position")

    @router.patch("/positions/{position_id}")
    async def modify_position(
        position_id: str,
        request: ModifyPositionRequest = Body(...),
        user: User = Depends(require_role(UserRole.TRADER)),
    ):
        """
        Modify position stop-loss or take-profit.

        Requires TRADER role or higher.

        Args:
            position_id: Position ID to modify
            request: Modification details

        Returns:
            Modified position details
        """
        if not position_tracker:
            raise HTTPException(status_code=503, detail="Position Tracker not initialized")

        # Finding 5: Validate UUID format
        position_id = _validate_uuid(position_id, "position_id")

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
            raise HTTPException(status_code=500, detail="Internal server error modifying position")

    # -------------------------------------------------------------------------
    # Order Endpoints
    # -------------------------------------------------------------------------

    @router.get("/orders")
    async def get_orders(
        symbol: Optional[str] = Query(None, description="Filter by symbol"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
        limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
        user: User = Depends(get_current_user),
    ):
        """
        Get open orders with pagination (Finding 27).

        Requires authentication.

        Args:
            symbol: Optional symbol filter
            offset: Pagination offset
            limit: Maximum results to return

        Returns:
            Paginated list of open orders
        """
        if not order_manager:
            raise HTTPException(status_code=503, detail="Order Manager not initialized")

        # Finding 19: Validate symbol if provided
        if symbol:
            symbol = validate_symbol_or_raise(symbol, strict=False)

        try:
            orders = await order_manager.get_open_orders(symbol)

            # Apply pagination
            total = len(orders)
            orders = orders[offset:offset + limit]

            return {
                "total": total,
                "offset": offset,
                "limit": limit,
                "count": len(orders),
                "orders": [o.to_dict() for o in orders],
            }
        except Exception as e:
            logger.error(f"Failed to get orders: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error getting orders")

    @router.get("/orders/{order_id}")
    async def get_order(
        order_id: str,
        user: User = Depends(get_current_user),
    ):
        """
        Get a specific order by ID.

        Requires authentication.
        """
        if not order_manager:
            raise HTTPException(status_code=503, detail="Order Manager not initialized")

        # Finding 5: Validate UUID format
        order_id = _validate_uuid(order_id, "order_id")

        try:
            order = await order_manager.get_order(order_id)
            if not order:
                raise HTTPException(status_code=404, detail="Order not found")
            return order.to_dict()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get order: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error getting order")

    @router.get("/orders/{order_id}/confirm")
    async def get_order_cancel_confirmation(
        order_id: str,
        user: User = Depends(require_role(UserRole.TRADER)),
    ):
        """
        Get a confirmation token for cancelling an order (Finding 9).

        Requires TRADER role or higher.

        Returns:
            Confirmation token valid for 5 minutes
        """
        # Validate UUID format
        order_id = _validate_uuid(order_id, "order_id")

        if not order_manager:
            raise HTTPException(status_code=503, detail="Order Manager not initialized")

        # Verify order exists
        order = await order_manager.get_order(order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")

        token = _generate_confirmation_token(order_id, "cancel_order")
        return {
            "order_id": order_id,
            "confirmation_token": token,
            "expires_in_seconds": 300,
            "message": "Include this token in your cancel request",
            "order_summary": {
                "symbol": order.symbol,
                "side": order.side,
                "type": order.order_type,
                "size": str(order.size),
                "price": str(order.price) if order.price else None,
            },
        }

    @router.post("/orders/{order_id}/cancel")
    async def cancel_order(
        order_id: str,
        http_request: Request,
        request: CancelOrderRequest = Body(...),
        user: User = Depends(require_role(UserRole.TRADER)),
    ):
        """
        Cancel an open order.

        Requires TRADER role or higher.
        Requires confirmation token from GET /orders/{id}/confirm.

        Args:
            order_id: Order ID to cancel
            request: Cancel request with confirmation token

        Returns:
            Cancellation result
        """
        if not order_manager:
            raise HTTPException(status_code=503, detail="Order Manager not initialized")

        # Finding 5: Validate UUID format
        order_id = _validate_uuid(order_id, "order_id")

        # Finding 9: Verify confirmation token
        if not _verify_confirmation_token(order_id, "cancel_order", request.confirmation_token):
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired confirmation token. Get a new one from GET /orders/{id}/confirm"
            )

        client_ip = http_request.client.host if http_request.client else "unknown"

        try:
            # Log critical operation
            await log_security_event(
                SecurityEventType.ORDER_CANCEL,
                user.user_id,
                client_ip,
                {
                    "order_id": order_id,
                    "reason": request.reason,
                },
                http_request,
            )

            success = await order_manager.cancel_order(order_id)
            if success:
                return {
                    "status": "cancelled",
                    "order_id": order_id,
                    "cancelled_by": user.user_id,
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to cancel order (may already be filled/cancelled)"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error cancelling order")

    @router.post("/orders/sync")
    async def sync_orders(
        user: User = Depends(require_role(UserRole.TRADER)),
    ):
        """
        Sync local order state with exchange.

        Requires TRADER role or higher.
        """
        if not order_manager:
            raise HTTPException(status_code=503, detail="Order Manager not initialized")

        try:
            synced = await order_manager.sync_with_exchange()
            return {
                "status": "synced",
                "orders_synced": synced,
                "synced_by": user.user_id,
            }
        except Exception as e:
            logger.error(f"Failed to sync orders: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error syncing orders")

    # -------------------------------------------------------------------------
    # Statistics Endpoints
    # -------------------------------------------------------------------------

    @router.get("/stats/execution")
    async def get_execution_stats(
        user: User = Depends(get_current_user),
    ):
        """
        Get execution statistics.

        Requires authentication.
        """
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
