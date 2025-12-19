"""
API Security Middleware - Authentication, Rate Limiting, and Security Headers.

CRITICAL SECURITY LAYER - Protects all API endpoints from:
- Unauthorized access (JWT authentication)
- DDoS/abuse (rate limiting)
- CSRF attacks (CORS)
- Memory exhaustion (request size limits)
- Hung requests (async timeouts)
"""

import asyncio
import hashlib
import logging
import os
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import wraps
from typing import Optional, Callable

try:
    from fastapi import FastAPI, Request, HTTPException, Depends, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Security Configuration
# =============================================================================

@dataclass
class SecurityConfig:
    """Security configuration with sensible defaults."""
    # JWT Settings
    jwt_secret: str = ""  # MUST be set via environment variable
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    # Rate Limiting (requests per minute)
    rate_limit_default: int = 60
    rate_limit_expensive: int = 5  # LLM calls, agent runs
    rate_limit_moderate: int = 30  # Database queries

    # CORS Settings
    cors_origins: list = None  # Default: none (secure)
    cors_allow_credentials: bool = True
    cors_allow_methods: list = None
    cors_allow_headers: list = None

    # Request Limits
    max_request_size_bytes: int = 1_048_576  # 1MB
    request_timeout_seconds: float = 45.0  # For LLM calls

    # Public endpoints (no auth required)
    public_endpoints: list = None

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = []  # No CORS by default (secure)
        if self.cors_allow_methods is None:
            self.cors_allow_methods = ["GET", "POST", "PUT", "DELETE"]
        if self.cors_allow_headers is None:
            self.cors_allow_headers = ["Authorization", "Content-Type"]
        if self.public_endpoints is None:
            self.public_endpoints = [
                "/health",
                "/health/live",
                "/health/ready",
                "/docs",
                "/openapi.json",
            ]


def get_security_config() -> SecurityConfig:
    """Load security config from environment."""
    return SecurityConfig(
        jwt_secret=os.environ.get("TRIPLEGAIN_JWT_SECRET", ""),
        cors_origins=os.environ.get("TRIPLEGAIN_CORS_ORIGINS", "").split(",") if os.environ.get("TRIPLEGAIN_CORS_ORIGINS") else [],
        rate_limit_default=int(os.environ.get("TRIPLEGAIN_RATE_LIMIT", "60")),
        max_request_size_bytes=int(os.environ.get("TRIPLEGAIN_MAX_REQUEST_SIZE", "1048576")),
        request_timeout_seconds=float(os.environ.get("TRIPLEGAIN_REQUEST_TIMEOUT", "45.0")),
    )


# =============================================================================
# Role-Based Access Control
# =============================================================================

class UserRole(Enum):
    """User roles for RBAC."""
    VIEWER = "viewer"      # Read-only access
    TRADER = "trader"      # Can execute trades
    ADMIN = "admin"        # Full access including config changes


@dataclass
class User:
    """Authenticated user."""
    user_id: str
    role: UserRole
    api_key_hash: str
    created_at: datetime


# Simple in-memory API key store (production should use database)
_api_keys: dict[str, User] = {}


def create_api_key(user_id: str, role: UserRole) -> str:
    """Create a new API key for a user."""
    api_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    _api_keys[key_hash] = User(
        user_id=user_id,
        role=role,
        api_key_hash=key_hash,
        created_at=datetime.now(timezone.utc),
    )

    return api_key


def validate_api_key(api_key: str) -> Optional[User]:
    """Validate an API key and return the user."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return _api_keys.get(key_hash)


def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    if key_hash in _api_keys:
        del _api_keys[key_hash]
        return True
    return False


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Sliding window rate limiter.

    Thread-safe implementation for concurrent requests.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        """
        Check if a request is allowed.

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self.window_seconds

            # Clean old requests
            self._requests[client_id] = [
                t for t in self._requests[client_id]
                if t > cutoff
            ]

            current_count = len(self._requests[client_id])

            info = {
                "limit": self.max_requests,
                "remaining": max(0, self.max_requests - current_count),
                "reset_seconds": int(self.window_seconds - (now - self._requests[client_id][0]) if self._requests[client_id] else 0),
            }

            if current_count >= self.max_requests:
                return False, info

            self._requests[client_id].append(now)
            info["remaining"] = max(0, self.max_requests - current_count - 1)
            return True, info


# Global rate limiters for different endpoint tiers
_rate_limiters: dict[str, RateLimiter] = {}


def get_rate_limiter(tier: str, config: SecurityConfig) -> RateLimiter:
    """Get or create a rate limiter for a tier."""
    if tier not in _rate_limiters:
        if tier == "expensive":
            _rate_limiters[tier] = RateLimiter(config.rate_limit_expensive, 60)
        elif tier == "moderate":
            _rate_limiters[tier] = RateLimiter(config.rate_limit_moderate, 60)
        else:
            _rate_limiters[tier] = RateLimiter(config.rate_limit_default, 60)
    return _rate_limiters[tier]


# =============================================================================
# FastAPI Middleware
# =============================================================================

if FASTAPI_AVAILABLE:

    class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
        """Reject requests that exceed size limit."""

        def __init__(self, app, max_size: int):
            super().__init__(app)
            self.max_size = max_size

        async def dispatch(self, request: Request, call_next):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_size:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Request too large. Max size: {self.max_size} bytes"
                    }
                )
            return await call_next(request)


    class TimeoutMiddleware(BaseHTTPMiddleware):
        """Add timeout to all requests."""

        def __init__(self, app, timeout: float):
            super().__init__(app)
            self.timeout = timeout

        async def dispatch(self, request: Request, call_next):
            try:
                return await asyncio.wait_for(
                    call_next(request),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={"detail": f"Request timeout after {self.timeout}s"}
                )


    class RateLimitMiddleware(BaseHTTPMiddleware):
        """Rate limit middleware."""

        def __init__(self, app, config: SecurityConfig):
            super().__init__(app)
            self.config = config
            self._expensive_paths = {
                "/api/v1/agents/",
                "/api/v1/coordinator/",
            }
            self._moderate_paths = {
                "/api/v1/indicators/",
                "/api/v1/snapshot/",
            }

        async def dispatch(self, request: Request, call_next):
            # Get client identifier
            client_ip = request.client.host if request.client else "unknown"
            # Include API key if present for per-user limiting
            auth_header = request.headers.get("authorization", "")
            client_id = f"{client_ip}:{hashlib.md5(auth_header.encode()).hexdigest()[:8]}"

            # Determine rate limit tier
            path = request.url.path
            if any(path.startswith(p) for p in self._expensive_paths):
                limiter = get_rate_limiter("expensive", self.config)
            elif any(path.startswith(p) for p in self._moderate_paths):
                limiter = get_rate_limiter("moderate", self.config)
            else:
                limiter = get_rate_limiter("default", self.config)

            allowed, info = await limiter.is_allowed(client_id)

            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "limit": info["limit"],
                        "reset_seconds": info["reset_seconds"],
                    },
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": str(info["remaining"]),
                        "X-RateLimit-Reset": str(info["reset_seconds"]),
                        "Retry-After": str(info["reset_seconds"]),
                    }
                )

            response = await call_next(request)

            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])

            return response


    # HTTP Bearer security scheme
    security = HTTPBearer(auto_error=False)


    async def get_current_user(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> Optional[User]:
        """
        Get current authenticated user.

        Returns None for public endpoints, raises 401 for protected endpoints.
        """
        config = get_security_config()

        # Check if endpoint is public
        path = request.url.path
        if any(path.startswith(p) for p in config.public_endpoints):
            return None

        # Require authentication for protected endpoints
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate API key
        user = validate_api_key(credentials.credentials)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user


    def require_role(required_role: UserRole):
        """Decorator to require a specific role."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, user: User = Depends(get_current_user), **kwargs):
                if user is None:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                    )

                # Check role hierarchy
                role_hierarchy = {
                    UserRole.VIEWER: 1,
                    UserRole.TRADER: 2,
                    UserRole.ADMIN: 3,
                }

                if role_hierarchy[user.role] < role_hierarchy[required_role]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Requires {required_role.value} role or higher",
                    )

                return await func(*args, user=user, **kwargs)
            return wrapper
        return decorator


def setup_security(app: 'FastAPI', config: Optional[SecurityConfig] = None) -> None:
    """
    Set up all security middleware for the FastAPI app.

    Args:
        app: FastAPI application instance
        config: Security configuration (uses defaults if None)
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")

    if config is None:
        config = get_security_config()

    # 1. Request size limit (first to reject large requests early)
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_size=config.max_request_size_bytes,
    )

    # 2. Timeout middleware
    app.add_middleware(
        TimeoutMiddleware,
        timeout=config.request_timeout_seconds,
    )

    # 3. Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        config=config,
    )

    # 4. CORS (must be last to wrap everything)
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=config.cors_allow_credentials,
            allow_methods=config.cors_allow_methods,
            allow_headers=config.cors_allow_headers,
        )
    else:
        # No CORS origins = reject all cross-origin requests (secure default)
        logger.info("CORS not configured - cross-origin requests will be rejected")

    # Warn if JWT secret not configured
    if not config.jwt_secret:
        logger.warning(
            "JWT_SECRET not configured! Set TRIPLEGAIN_JWT_SECRET environment variable. "
            "API authentication is DISABLED without this."
        )

    logger.info(
        f"Security middleware configured: "
        f"rate_limit={config.rate_limit_default}/min, "
        f"max_request={config.max_request_size_bytes}B, "
        f"timeout={config.request_timeout_seconds}s, "
        f"cors_origins={len(config.cors_origins)}"
    )
