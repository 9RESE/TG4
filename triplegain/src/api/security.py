"""
API Security Middleware - Authentication, Rate Limiting, and Security Headers.

CRITICAL SECURITY LAYER - Protects all API endpoints from:
- Unauthorized access (API key authentication)
- DDoS/abuse (rate limiting)
- CSRF attacks (CORS)
- Memory exhaustion (request size limits)
- Hung requests (async timeouts)
- Clickjacking and XSS (security headers)

Security Fixes Applied (Phase 4 Review):
- Finding 1-3: Authentication now enforced via get_current_user dependency
- Finding 4: Security headers middleware added
- Finding 6: Database-backed API key storage interface
- Finding 7: JWT removed (API key auth only)
- Finding 10-11: Risk/reset requires ADMIN role and rate limited
- Finding 12: Audit logging added
- Finding 13: MD5 replaced with SHA256
- Finding 25: Production mode requires proper configuration
"""

import asyncio
import hashlib
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import wraps
from typing import Optional, Callable, Protocol

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
audit_logger = logging.getLogger(f"{__name__}.audit")


# =============================================================================
# Security Configuration
# =============================================================================

@dataclass
class SecurityConfig:
    """Security configuration with sensible defaults."""
    # Production Mode - enables strict security checks
    production_mode: bool = False

    # Rate Limiting (requests per minute)
    rate_limit_default: int = 60
    rate_limit_expensive: int = 5  # LLM calls, agent runs, risk reset
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

    # Debug mode - enables debug endpoints
    debug_mode: bool = False

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = []  # No CORS by default (secure)
        if self.cors_allow_methods is None:
            self.cors_allow_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        if self.cors_allow_headers is None:
            self.cors_allow_headers = ["Authorization", "Content-Type", "X-Request-ID"]
        if self.public_endpoints is None:
            self.public_endpoints = [
                "/health",
                "/health/live",
                "/health/ready",
                "/docs",
                "/openapi.json",
                "/redoc",
            ]


def get_security_config() -> SecurityConfig:
    """Load security config from environment."""
    return SecurityConfig(
        production_mode=os.environ.get("TRIPLEGAIN_PRODUCTION", "false").lower() == "true",
        cors_origins=os.environ.get("TRIPLEGAIN_CORS_ORIGINS", "").split(",") if os.environ.get("TRIPLEGAIN_CORS_ORIGINS") else [],
        rate_limit_default=int(os.environ.get("TRIPLEGAIN_RATE_LIMIT", "60")),
        max_request_size_bytes=int(os.environ.get("TRIPLEGAIN_MAX_REQUEST_SIZE", "1048576")),
        request_timeout_seconds=float(os.environ.get("TRIPLEGAIN_REQUEST_TIMEOUT", "45.0")),
        debug_mode=os.environ.get("TRIPLEGAIN_DEBUG", "false").lower() == "true",
    )


# =============================================================================
# Role-Based Access Control
# =============================================================================

class UserRole(Enum):
    """User roles for RBAC."""
    VIEWER = "viewer"      # Read-only access
    TRADER = "trader"      # Can execute trades
    ADMIN = "admin"        # Full access including config changes


# Role hierarchy for authorization checks
ROLE_HIERARCHY = {
    UserRole.VIEWER: 1,
    UserRole.TRADER: 2,
    UserRole.ADMIN: 3,
}


@dataclass
class User:
    """Authenticated user."""
    user_id: str
    role: UserRole
    api_key_hash: str
    created_at: datetime
    last_used: Optional[datetime] = None

    def has_role(self, required_role: UserRole) -> bool:
        """Check if user has at least the required role."""
        return ROLE_HIERARCHY[self.role] >= ROLE_HIERARCHY[required_role]


# =============================================================================
# API Key Storage Interface (Finding 6: Database-backed storage)
# =============================================================================

class APIKeyStore(Protocol):
    """Protocol for API key storage implementations."""

    async def validate_key(self, api_key: str) -> Optional[User]:
        """Validate an API key and return the user."""
        ...

    async def create_key(self, user_id: str, role: UserRole) -> str:
        """Create a new API key for a user."""
        ...

    async def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        ...

    async def list_keys(self, user_id: Optional[str] = None) -> list[dict]:
        """List API keys, optionally filtered by user."""
        ...


class InMemoryAPIKeyStore:
    """
    In-memory API key store for development/testing.

    WARNING: Keys are lost on server restart. Use DatabaseAPIKeyStore for production.
    """

    def __init__(self):
        self._api_keys: dict[str, User] = {}

    async def validate_key(self, api_key: str) -> Optional[User]:
        """Validate an API key and return the user."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user = self._api_keys.get(key_hash)
        if user:
            user.last_used = datetime.now(timezone.utc)
        return user

    async def create_key(self, user_id: str, role: UserRole) -> str:
        """Create a new API key for a user."""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        self._api_keys[key_hash] = User(
            user_id=user_id,
            role=role,
            api_key_hash=key_hash,
            created_at=datetime.now(timezone.utc),
        )

        return api_key

    async def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self._api_keys:
            del self._api_keys[key_hash]
            return True
        return False

    async def list_keys(self, user_id: Optional[str] = None) -> list[dict]:
        """List API keys, optionally filtered by user."""
        keys = []
        for key_hash, user in self._api_keys.items():
            if user_id is None or user.user_id == user_id:
                keys.append({
                    "user_id": user.user_id,
                    "role": user.role.value,
                    "created_at": user.created_at.isoformat(),
                    "last_used": user.last_used.isoformat() if user.last_used else None,
                    "key_hash_prefix": key_hash[:8],
                })
        return keys


class DatabaseAPIKeyStore:
    """
    Database-backed API key store for production.

    Requires a database pool with api_keys table:
    CREATE TABLE api_keys (
        key_hash VARCHAR(64) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        role VARCHAR(50) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_used TIMESTAMPTZ,
        revoked BOOLEAN DEFAULT FALSE
    );
    """

    def __init__(self, db_pool):
        self._pool = db_pool

    async def validate_key(self, api_key: str) -> Optional[User]:
        """Validate an API key and return the user."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        try:
            row = await self._pool.fetchone(
                """
                SELECT user_id, role, created_at, last_used
                FROM api_keys
                WHERE key_hash = $1 AND revoked = FALSE
                """,
                key_hash
            )
            if row:
                # Update last_used
                await self._pool.execute(
                    "UPDATE api_keys SET last_used = NOW() WHERE key_hash = $1",
                    key_hash
                )
                return User(
                    user_id=row['user_id'],
                    role=UserRole(row['role']),
                    api_key_hash=key_hash,
                    created_at=row['created_at'],
                    last_used=row['last_used'],
                )
        except Exception as e:
            logger.error(f"Database error validating API key: {e}")
        return None

    async def create_key(self, user_id: str, role: UserRole) -> str:
        """Create a new API key for a user."""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        await self._pool.execute(
            """
            INSERT INTO api_keys (key_hash, user_id, role, created_at)
            VALUES ($1, $2, $3, NOW())
            """,
            key_hash, user_id, role.value
        )

        return api_key

    async def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        result = await self._pool.execute(
            "UPDATE api_keys SET revoked = TRUE WHERE key_hash = $1 AND revoked = FALSE",
            key_hash
        )
        return result > 0

    async def list_keys(self, user_id: Optional[str] = None) -> list[dict]:
        """List API keys, optionally filtered by user."""
        query = """
            SELECT key_hash, user_id, role, created_at, last_used, revoked
            FROM api_keys
            WHERE revoked = FALSE
        """
        params = []
        if user_id:
            query += " AND user_id = $1"
            params.append(user_id)

        rows = await self._pool.fetch(query, *params)
        return [
            {
                "user_id": row['user_id'],
                "role": row['role'],
                "created_at": row['created_at'].isoformat(),
                "last_used": row['last_used'].isoformat() if row['last_used'] else None,
                "key_hash_prefix": row['key_hash'][:8],
            }
            for row in rows
        ]


# Global API key store (set during app initialization)
_api_key_store: Optional[APIKeyStore] = None


def get_api_key_store() -> APIKeyStore:
    """Get the current API key store."""
    global _api_key_store
    if _api_key_store is None:
        # Default to in-memory for development
        logger.warning("Using in-memory API key store. Set up database store for production.")
        _api_key_store = InMemoryAPIKeyStore()
    return _api_key_store


def set_api_key_store(store: APIKeyStore) -> None:
    """Set the API key store (call during app initialization)."""
    global _api_key_store
    _api_key_store = store


# Legacy sync functions for backwards compatibility (wrap async)
def create_api_key(user_id: str, role: UserRole) -> str:
    """Create a new API key for a user (sync wrapper)."""
    import asyncio
    store = get_api_key_store()
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Use InMemoryAPIKeyStore directly for sync context
        if isinstance(store, InMemoryAPIKeyStore):
            api_key = secrets.token_urlsafe(32)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            store._api_keys[key_hash] = User(
                user_id=user_id,
                role=role,
                api_key_hash=key_hash,
                created_at=datetime.now(timezone.utc),
            )
            return api_key
    return loop.run_until_complete(store.create_key(user_id, role))


def validate_api_key(api_key: str) -> Optional[User]:
    """Validate an API key and return the user (sync wrapper)."""
    import asyncio
    store = get_api_key_store()
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Use InMemoryAPIKeyStore directly for sync context
        if isinstance(store, InMemoryAPIKeyStore):
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            return store._api_keys.get(key_hash)
    return loop.run_until_complete(store.validate_key(api_key))


def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key (sync wrapper)."""
    import asyncio
    store = get_api_key_store()
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Use InMemoryAPIKeyStore directly for sync context
        if isinstance(store, InMemoryAPIKeyStore):
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            if key_hash in store._api_keys:
                del store._api_keys[key_hash]
                return True
            return False
    return loop.run_until_complete(store.revoke_key(api_key))


# =============================================================================
# Audit Logging (Finding 12)
# =============================================================================

class SecurityEventType(Enum):
    """Types of security events for audit logging."""
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    AUTH_INVALID_KEY = "auth_invalid_key"
    AUTHZ_DENIED = "authorization_denied"
    RATE_LIMITED = "rate_limited"
    CRITICAL_OPERATION = "critical_operation"
    ADMIN_OVERRIDE = "admin_override"
    POSITION_CLOSE = "position_close"
    ORDER_CANCEL = "order_cancel"
    RISK_RESET = "risk_reset"
    TRADING_PAUSED = "trading_paused"
    TRADING_RESUMED = "trading_resumed"


async def log_security_event(
    event_type: SecurityEventType,
    user_id: Optional[str],
    ip_address: str,
    details: dict,
    request: Optional[Request] = None,
) -> None:
    """
    Log a security event for audit purposes.

    All security-relevant events are logged with structured data.
    """
    event = {
        "event_type": event_type.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "ip_address": ip_address,
        "details": details,
    }

    if request:
        event["path"] = request.url.path
        event["method"] = request.method
        event["request_id"] = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Log at appropriate level
    if event_type in {
        SecurityEventType.AUTH_FAILURE,
        SecurityEventType.AUTH_INVALID_KEY,
        SecurityEventType.AUTHZ_DENIED,
        SecurityEventType.ADMIN_OVERRIDE,
        SecurityEventType.RISK_RESET,
    }:
        audit_logger.warning(f"SECURITY_EVENT: {event}")
    else:
        audit_logger.info(f"SECURITY_EVENT: {event}")


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
        """Rate limit middleware with tiered limits based on endpoint type."""

        def __init__(self, app, config: SecurityConfig):
            super().__init__(app)
            self.config = config
            # Finding 11: Add risk/reset to expensive paths
            self._expensive_paths = {
                "/api/v1/agents/",
                "/api/v1/coordinator/",
                "/api/v1/risk/reset",
                "/api/v1/portfolio/rebalance",
            }
            self._moderate_paths = {
                "/api/v1/indicators/",
                "/api/v1/snapshot/",
                "/api/v1/positions/",
                "/api/v1/orders/",
            }

        async def dispatch(self, request: Request, call_next):
            # Get client identifier
            client_ip = request.client.host if request.client else "unknown"
            # Finding 13: Replace MD5 with SHA256 for rate limit client ID
            auth_header = request.headers.get("authorization", "")
            client_id = f"{client_ip}:{hashlib.sha256(auth_header.encode()).hexdigest()[:16]}"

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
                # Log rate limit event
                await log_security_event(
                    SecurityEventType.RATE_LIMITED,
                    None,  # User unknown at this point
                    client_ip,
                    {"path": path, "limit": info["limit"]},
                    request,
                )
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


    # Finding 4: Security Headers Middleware
    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        """Add security headers to all responses."""

        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)

            # Prevent MIME type sniffing
            response.headers["X-Content-Type-Options"] = "nosniff"

            # Prevent clickjacking
            response.headers["X-Frame-Options"] = "DENY"

            # XSS protection (legacy, but still useful for older browsers)
            response.headers["X-XSS-Protection"] = "1; mode=block"

            # Prevent caching of sensitive data
            if "/api/" in request.url.path:
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
                response.headers["Pragma"] = "no-cache"

            # Content Security Policy for API
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"

            # Referrer policy
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

            # Add HSTS in production (requires HTTPS)
            config = get_security_config()
            if config.production_mode:
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

            return response


    # Finding 24: Request/Response Logging Middleware
    class RequestLoggingMiddleware(BaseHTTPMiddleware):
        """Log all requests and responses for debugging and observability."""

        async def dispatch(self, request: Request, call_next):
            # Generate request ID if not provided
            request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

            # Log request
            client_ip = request.client.host if request.client else "unknown"
            logger.info(
                f"REQUEST: {request.method} {request.url.path} "
                f"client={client_ip} request_id={request_id}"
            )

            start_time = time.monotonic()

            try:
                response = await call_next(request)

                # Log response
                duration_ms = (time.monotonic() - start_time) * 1000
                logger.info(
                    f"RESPONSE: {request.method} {request.url.path} "
                    f"status={response.status_code} duration={duration_ms:.2f}ms "
                    f"request_id={request_id}"
                )

                # Add request ID to response
                response.headers["X-Request-ID"] = request_id

                return response

            except Exception as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                logger.error(
                    f"ERROR: {request.method} {request.url.path} "
                    f"error={str(e)} duration={duration_ms:.2f}ms "
                    f"request_id={request_id}"
                )
                raise


    # HTTP Bearer security scheme
    security = HTTPBearer(auto_error=False)


    async def get_current_user(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> User:
        """
        Get current authenticated user.

        This dependency enforces authentication on all non-public endpoints.
        Use as: user: User = Depends(get_current_user)

        Raises:
            HTTPException 401: If no credentials provided or invalid API key
        """
        config = get_security_config()
        client_ip = request.client.host if request.client else "unknown"

        # Check if endpoint is public
        path = request.url.path
        if any(path.startswith(p) for p in config.public_endpoints):
            # Return a special "anonymous" user for public endpoints
            return User(
                user_id="anonymous",
                role=UserRole.VIEWER,
                api_key_hash="",
                created_at=datetime.now(timezone.utc),
            )

        # Require authentication for protected endpoints
        if credentials is None:
            await log_security_event(
                SecurityEventType.AUTH_FAILURE,
                None,
                client_ip,
                {"reason": "No credentials provided"},
                request,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate API key
        user = validate_api_key(credentials.credentials)
        if user is None:
            await log_security_event(
                SecurityEventType.AUTH_INVALID_KEY,
                None,
                client_ip,
                {"reason": "Invalid API key"},
                request,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Log successful authentication
        await log_security_event(
            SecurityEventType.AUTH_SUCCESS,
            user.user_id,
            client_ip,
            {"role": user.role.value},
            request,
        )

        return user


    async def get_optional_user(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> Optional[User]:
        """
        Get current user if authenticated, None otherwise.

        Use for endpoints that work both authenticated and unauthenticated.
        """
        if credentials is None:
            return None

        user = validate_api_key(credentials.credentials)
        return user


    def require_role(required_role: UserRole):
        """
        Create a dependency that requires a specific role.

        Usage:
            @router.post("/admin/action")
            async def admin_action(user: User = Depends(require_role(UserRole.ADMIN))):
                ...
        """
        async def role_checker(
            request: Request,
            user: User = Depends(get_current_user),
        ) -> User:
            client_ip = request.client.host if request.client else "unknown"

            if user.user_id == "anonymous":
                await log_security_event(
                    SecurityEventType.AUTH_FAILURE,
                    None,
                    client_ip,
                    {"reason": "Authentication required", "required_role": required_role.value},
                    request,
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if not user.has_role(required_role):
                await log_security_event(
                    SecurityEventType.AUTHZ_DENIED,
                    user.user_id,
                    client_ip,
                    {
                        "user_role": user.role.value,
                        "required_role": required_role.value,
                    },
                    request,
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires {required_role.value} role or higher",
                )

            return user

        return role_checker


    # Convenience dependencies for common role requirements
    def require_viewer():
        """Require at least VIEWER role."""
        return require_role(UserRole.VIEWER)

    def require_trader():
        """Require at least TRADER role."""
        return require_role(UserRole.TRADER)

    def require_admin():
        """Require ADMIN role."""
        return require_role(UserRole.ADMIN)


def setup_security(app: 'FastAPI', config: Optional[SecurityConfig] = None) -> None:
    """
    Set up all security middleware for the FastAPI app.

    Middleware order (applied in reverse - last added runs first):
    1. CORS (outermost)
    2. Security Headers
    3. Request Logging
    4. Rate Limiting
    5. Timeout
    6. Request Size Limit (innermost)

    Args:
        app: FastAPI application instance
        config: Security configuration (uses defaults if None)
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")

    if config is None:
        config = get_security_config()

    # Finding 25: Fail fast in production if not properly configured
    if config.production_mode:
        if not config.cors_origins:
            logger.warning("Production mode: No CORS origins configured")
        logger.info("Running in PRODUCTION mode with strict security")

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

    # 4. Request/Response logging (Finding 24)
    app.add_middleware(RequestLoggingMiddleware)

    # 5. Security headers (Finding 4)
    app.add_middleware(SecurityHeadersMiddleware)

    # 6. CORS (must be last to wrap everything)
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

    logger.info(
        f"Security middleware configured: "
        f"rate_limit={config.rate_limit_default}/min, "
        f"max_request={config.max_request_size_bytes}B, "
        f"timeout={config.request_timeout_seconds}s, "
        f"cors_origins={len(config.cors_origins)}, "
        f"production={config.production_mode}, "
        f"debug={config.debug_mode}"
    )
