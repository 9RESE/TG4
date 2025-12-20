"""API module - FastAPI endpoints for TripleGain.

Phase 3: Core API endpoints for agents and orchestration
Phase 6: Paper trading API endpoints
"""

from .app import create_app, get_app, register_paper_trading_routes, get_app_state
from .routes_paper_trading import get_paper_trading_router

__all__ = [
    'create_app',
    'get_app',
    'get_app_state',
    'get_paper_trading_router',
    'register_paper_trading_routes',
]
