# ADR-011: API Security Fixes

## Status
Accepted

## Date
2025-12-19

## Context

During the Phase 4 code review of the API layer (app.py, routes_agents.py, routes_orchestration.py, security.py, validation.py), 27 security issues were identified. The most critical finding was that **authentication and authorization infrastructure was fully implemented but not enforced on any endpoints**. This meant:

1. **P0 Critical**: Any attacker with network access could pause/resume trading, close positions, cancel orders
2. **P0 Critical**: Role-based access control (RBAC) decorators existed but were never applied
3. **P1 High**: No UUID validation on position/order IDs allowed potential injection attacks
4. **P1 High**: Destructive operations (close position, cancel order, rebalance) lacked confirmation mechanisms

These issues needed immediate resolution before any production deployment.

## Decision

We implemented 26 of 27 fixes across the API layer (Finding 18 - Pydantic response models - deferred as optional enhancement):

### Critical Fixes (P0 - 3 issues)
- **F01**: Added `Depends(get_current_user)` to all endpoints in routes_agents.py and routes_orchestration.py
- **F02**: Applied `require_role(UserRole.ADMIN)` to sensitive operations (pause/resume coordinator, risk/reset)
- **F03**: Applied `require_role(UserRole.TRADER)` to trading operations (close position, cancel order, rebalance)

### High Priority Fixes (P1 - 8 issues)
- **F04**: Added `SecurityHeadersMiddleware` with X-Content-Type-Options, X-Frame-Options, CSP, HSTS headers
- **F05**: Added UUID validation via `_validate_uuid()` for all position_id and order_id parameters
- **F06**: Defined `APIKeyStorageInterface` abstract class with `DatabaseAPIKeyStorage` implementation stub
- **F07**: Documented JWT support as future enhancement, API key auth as current method
- **F08**: Refactored routes to use FastAPI dependency injection instead of global module state
- **F09**: Added confirmation token mechanism for destructive operations:
  - `GET /positions/{id}/confirm` returns one-time token
  - `POST /positions/{id}/close` requires confirmation_token
  - Same pattern for order cancellation and portfolio rebalancing
- **F10**: `/risk/reset` now requires `UserRole.ADMIN`
- **F11**: `/risk/reset` now rate-limited to 1 request per minute

### Medium Priority Fixes (P2 - 9 issues, 1 deferred)
- **F12**: Added `log_security_event()` with `SecurityEventType` enum for audit logging
- **F13**: Replaced MD5 with SHA256 for rate limiting IP hash
- **F14**: Added cross-field validation (stop_loss < entry_price, take_profit > entry_price) in trade proposals
- **F15**: Added global exception handler with `_register_exception_handlers()`
- **F16**: Debug endpoints require authentication in production mode
- **F17**: Implemented `GET /agents/outputs/{agent_name}` endpoint for agent output retrieval
- **F18**: (Deferred) Pydantic response models for OpenAPI documentation
- **F19**: Added symbol validation to orchestration routes via `validate_symbol_or_raise()`
- **F20**: Added task_name validation with `VALID_COORDINATOR_TASKS` whitelist
- **F21**: Fixed inconsistent symbol handling - all responses use normalized `BTC/USDT` format
- **F22**: Fixed exception handling order (HTTPException caught before generic Exception)

### Low Priority Fixes (P3 - 6 issues)
- **F23**: `SUPPORTED_SYMBOLS` now loaded from config via `get_supported_symbols()` with caching
- **F24**: Added request/response logging middleware
- **F25**: JWT secret required in production mode (warns in debug mode)
- **F26**: Centralized validation functions in `validation.py` module
- **F27**: Added pagination (`offset`, `limit`) to `/positions` and `/orders` list endpoints

## Technical Details

### Confirmation Token Flow
```python
# 1. Client requests confirmation token
GET /api/v1/positions/{id}/confirm
-> Returns: {"confirmation_token": "...", "expires_in_seconds": 300}

# 2. Client confirms operation with token
POST /api/v1/positions/{id}/close
Body: {"exit_price": 46000.0, "confirmation_token": "..."}
-> Position closed
```

### Security Event Types
```python
class SecurityEventType(Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    TRADING_PAUSED = "trading_paused"
    TRADING_RESUMED = "trading_resumed"
    POSITION_CLOSE = "position_close"
    ORDER_CANCEL = "order_cancel"
    CRITICAL_OPERATION = "critical_operation"
```

### Valid Coordinator Tasks
```python
VALID_COORDINATOR_TASKS = frozenset({
    "technical_analysis",
    "regime_detection",
    "trading_decision",
    "portfolio_rebalance",
    "risk_check",
})
```

## Consequences

### Positive
- All endpoints now require authentication (API key required)
- Role-based access control enforced (VIEWER, TRADER, ADMIN hierarchy)
- Destructive operations require explicit confirmation
- Security events logged for audit trail
- Input validation prevents injection attacks
- Standardized error responses with timestamps
- Pagination prevents unbounded responses

### Negative
- Breaking change: All API clients must now provide authentication headers
- Additional latency from confirmation token round-trip for destructive operations
- Slightly more complex client implementation

### Risks Mitigated
- OWASP A01: Broken Access Control - Fixed by enforcing authentication/authorization
- OWASP A03: Injection - Fixed by UUID and symbol validation
- OWASP A07: Security Misconfiguration - Fixed by security headers and production mode requirements

## Test Impact
- 1045 tests passing
- Updated tests to include authentication override for testing
- Updated tests to use valid UUID format for position/order IDs
- Updated tests to use confirmation tokens for destructive operations

## Related Documents
- [Phase 4 Findings](../../development/reviews/full/review-4/findings/phase-4-findings.md)
- [Security Implementation](../../development/features/phase-4-api-security.md) (to be created)
