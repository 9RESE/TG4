# Review Phase 4: API Layer

**Status**: Ready for Review
**Estimated Context**: ~3,500 tokens (code) + review
**Priority**: High - External interface security
**Output**: `findings/phase-4-findings.md`
**DO NOT IMPLEMENT FIXES**

---

## Files to Review

| File | Lines | Purpose |
|------|-------|---------|
| `triplegain/src/api/app.py` | ~300 | FastAPI application |
| `triplegain/src/api/routes_agents.py` | ~400 | Agent API routes |
| `triplegain/src/api/routes_orchestration.py` | ~350 | Orchestration API routes |
| `triplegain/src/api/validation.py` | ~300 | Input validation |
| `triplegain/src/api/security.py` | ~250 | Security middleware |

**Total**: ~1,600 lines

---

## Pre-Review: Load Files

```bash
# Read these files before starting review
cat triplegain/src/api/app.py
cat triplegain/src/api/routes_agents.py
cat triplegain/src/api/routes_orchestration.py
cat triplegain/src/api/validation.py
cat triplegain/src/api/security.py
```

---

## Review Checklist

### 1. FastAPI Application (`app.py`)

#### Application Setup
- [ ] FastAPI instance configured correctly
- [ ] CORS configured (if needed)
- [ ] Middleware applied
- [ ] Routers included
- [ ] Startup/shutdown events defined

#### Dependency Injection
- [ ] Database pool injected
- [ ] Agent instances injected
- [ ] Config injected
- [ ] Clean dependency lifecycle

#### Error Handling
- [ ] Global exception handler
- [ ] Custom exception classes
- [ ] Error response format consistent
- [ ] Stack traces not exposed in production

#### Health Check
- [ ] /health endpoint exists
- [ ] Database connectivity check
- [ ] Agent status check (optional)
- [ ] Returns appropriate status codes

#### Logging
- [ ] Request logging configured
- [ ] Response logging configured
- [ ] Sensitive data redacted
- [ ] Log levels appropriate

---

### 2. Agent Routes (`routes_agents.py`)

#### Endpoint Coverage
- [ ] GET /api/v1/agents/ta/{symbol}
- [ ] GET /api/v1/agents/regime/{symbol}
- [ ] POST /api/v1/agents/ta/{symbol}/run
- [ ] POST /api/v1/agents/trading/{symbol}/run
- [ ] GET /api/v1/agents/outputs/{agent_name}

#### Input Validation
- [ ] Symbol parameter validated
- [ ] Symbol enum or pattern check
- [ ] Invalid symbol returns 400
- [ ] Path parameters sanitized

#### Response Format
- [ ] Consistent JSON structure
- [ ] Timestamps in ISO format
- [ ] Decimal serialization correct
- [ ] Null handling consistent

#### Error Responses
- [ ] 400 for bad input
- [ ] 404 for not found
- [ ] 500 for server errors
- [ ] Error messages helpful but not revealing

#### Rate Limiting (if applicable)
- [ ] Rate limits on POST endpoints
- [ ] Per-IP or per-key limits
- [ ] 429 Too Many Requests returned

---

### 3. Orchestration Routes (`routes_orchestration.py`)

#### Endpoint Coverage
- [ ] GET /api/v1/coordinator/status
- [ ] POST /api/v1/coordinator/pause
- [ ] POST /api/v1/coordinator/resume
- [ ] GET /api/v1/portfolio/allocation
- [ ] POST /api/v1/portfolio/rebalance
- [ ] GET /api/v1/positions
- [ ] POST /api/v1/positions/{id}/close
- [ ] GET /api/v1/orders
- [ ] POST /api/v1/orders/{id}/cancel
- [ ] GET /api/v1/risk/state
- [ ] POST /api/v1/risk/validate

#### Critical Operations
- [ ] pause/resume require authentication
- [ ] rebalance requires confirmation
- [ ] position close requires confirmation
- [ ] order cancel requires confirmation

#### Input Validation
- [ ] UUID format validation for IDs
- [ ] Trade proposal schema validation
- [ ] Enum values validated

#### Response Format
- [ ] Coordinator status structure
- [ ] Portfolio allocation structure
- [ ] Position list structure
- [ ] Order list structure
- [ ] Risk state structure

---

### 4. Validation (`validation.py`)

#### Pydantic Models
- [ ] Request models defined
- [ ] Response models defined
- [ ] Field validators present
- [ ] Examples provided

#### Symbol Validation
- [ ] Allowed symbols list
- [ ] Format validation (BASE/QUOTE)
- [ ] Case normalization

#### Trade Proposal Validation
- [ ] action enum validation
- [ ] confidence bounds (0-1)
- [ ] size_pct bounds (0-100)
- [ ] leverage bounds (1-5)
- [ ] stop_loss_pct bounds
- [ ] take_profit_pct bounds
- [ ] Cross-field validation (SL < TP for long)

#### Numeric Validation
- [ ] Decimal handling
- [ ] Precision limits
- [ ] Positive value checks
- [ ] Range validation

---

### 5. Security (`security.py`)

#### Authentication
- [ ] API key authentication (if used)
- [ ] JWT validation (if used)
- [ ] No authentication for read-only? (config-dependent)
- [ ] Authentication for write operations

#### Authorization
- [ ] Role-based access (if applicable)
- [ ] Operation-level permissions
- [ ] Admin-only endpoints protected

#### Input Sanitization
- [ ] SQL injection prevention (use ORM/parameterized)
- [ ] XSS prevention (output encoding)
- [ ] Path traversal prevention
- [ ] Command injection prevention

#### Security Headers
- [ ] Content-Type enforcement
- [ ] X-Content-Type-Options: nosniff
- [ ] X-Frame-Options: DENY
- [ ] Strict-Transport-Security (production)

#### Rate Limiting
- [ ] Implemented for sensitive endpoints
- [ ] Per-IP or per-key tracking
- [ ] Configurable limits

#### Audit Logging
- [ ] Authentication attempts logged
- [ ] Authorization failures logged
- [ ] Critical operations logged
- [ ] IP addresses captured

---

## Critical Security Questions

1. **Authentication Bypass**: Can any endpoint be accessed without auth?
2. **Parameter Injection**: Can user input reach database queries directly?
3. **Information Disclosure**: Are error messages revealing internals?
4. **Privilege Escalation**: Can read-only user trigger trades?
5. **Denial of Service**: Can large requests crash the server?
6. **CORS Misconfiguration**: Are all origins allowed?

---

## OWASP Top 10 Checklist

| Risk | Status | Notes |
|------|--------|-------|
| A01 Broken Access Control | [ ] Check | Auth on all write endpoints |
| A02 Cryptographic Failures | [ ] Check | Secrets management |
| A03 Injection | [ ] Check | SQL, Command injection |
| A04 Insecure Design | [ ] Check | Security by design |
| A05 Security Misconfiguration | [ ] Check | Headers, CORS |
| A06 Vulnerable Components | [ ] Check | Dependencies |
| A07 Auth Failures | [ ] Check | Auth implementation |
| A08 Data Integrity | [ ] Check | Input validation |
| A09 Logging Failures | [ ] Check | Audit logging |
| A10 SSRF | [ ] Check | External URL handling |

---

## API Contract Verification

### Response Schema Consistency

```python
# Expected response envelope
{
    "success": bool,
    "data": object | null,
    "error": {
        "code": str,
        "message": str
    } | null,
    "timestamp": str  # ISO format
}
```

- [ ] All endpoints use consistent envelope
- [ ] Or raw data (document which)

### Error Response Format

```python
# 4xx/5xx response
{
    "detail": str,
    "error_code": str,  # optional
    "field_errors": dict  # for validation
}
```

- [ ] Consistent error format
- [ ] Appropriate status codes

---

## Performance Review

- [ ] Async handlers throughout
- [ ] No blocking calls
- [ ] Database connection pooling
- [ ] Response caching where appropriate
- [ ] Pagination for list endpoints
- [ ] Query parameter limits

---

## Test Coverage Check

```bash
pytest --cov=triplegain/src/api \
       --cov-report=term-missing \
       triplegain/tests/unit/api/
```

Expected tests:
- [ ] Each endpoint (happy path)
- [ ] Input validation (bad input)
- [ ] Authentication (missing/invalid)
- [ ] Authorization (insufficient)
- [ ] Error handling
- [ ] Rate limiting

---

## Design Conformance

### Implementation Plan API Endpoints
- [ ] Phase 1 endpoints implemented
- [ ] Phase 2 endpoints implemented
- [ ] Phase 3 endpoints implemented

### OpenAPI/Swagger
- [ ] Auto-generated docs accurate
- [ ] Examples provided
- [ ] Schema descriptions complete

---

## Findings Template

```markdown
## Finding: [Title]

**File**: `triplegain/src/api/filename.py:123`
**Priority**: P0/P1/P2/P3
**Category**: Security/Logic/Performance/Quality

### Description
[What was found]

### Current Code
```python
# current implementation
```

### Recommended Fix
```python
# recommended fix
```

### Security Impact
[Attack scenario if not fixed]
```

---

## Review Completion

After completing this phase:

1. [ ] All route files reviewed
2. [ ] Validation logic verified
3. [ ] Security measures verified
4. [ ] OWASP checklist addressed
5. [ ] Error handling verified
6. [ ] Findings documented
7. [ ] Ready for Phase 5

---

*Phase 4 Review Plan v1.0*
