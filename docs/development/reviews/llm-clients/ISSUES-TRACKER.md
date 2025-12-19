# LLM Client Issues Tracker

**Last Updated**: 2025-12-19
**Review Version**: 1.0

This document tracks all issues found in the LLM client code review.
Update status as issues are resolved.

---

## Critical Issues (P0)

### CRIT-1: Resource Leak - Session Management
- **Status**: 🔴 Open
- **Files**: `ollama.py`, `openai_client.py`, `anthropic_client.py`, `deepseek_client.py`, `xai_client.py`
- **Lines**: Multiple (all generate methods)
- **Assigned**: -
- **Est. Effort**: 2-3 hours
- **Blocked By**: -
- **PR**: -

**Description**: All clients create new aiohttp.ClientSession for each request.

**Solution**:
```python
# In __init__
self._session = None

async def initialize(self):
    self._session = aiohttp.ClientSession(timeout=self.timeout)

async def close(self):
    if self._session:
        await self._session.close()

async def __aenter__(self):
    await self.initialize()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()
```

**Testing**: Add test for session reuse, verify no leaks

---

### CRIT-2: RateLimiter Variable Reference Bug
- **Status**: 🔴 Open
- **File**: `base.py`
- **Line**: 105
- **Assigned**: -
- **Est. Effort**: 15 minutes
- **Blocked By**: -
- **PR**: -

**Description**: Incorrect variable existence check using `dir()`

**Solution**:
```python
async def acquire(self) -> float:
    wait_time = 0.0  # Initialize here
    async with self._lock:
        # ... rest of logic
        return wait_time  # Remove dir() check
```

**Testing**: Existing tests should catch this once fixed

---

### CRIT-3: No Total Timeout on Retries
- **Status**: 🔴 Open
- **File**: `base.py`
- **Lines**: 176-243
- **Assigned**: -
- **Est. Effort**: 30 minutes
- **Blocked By**: -
- **PR**: -

**Description**: Exponential backoff can cause 60+ second delays

**Solution**:
```python
async def generate_with_retry(
    self,
    # ... existing params
    timeout_seconds: Optional[float] = None,
) -> LLMResponse:
    start_time = time.monotonic()
    timeout = timeout_seconds or self.config.get('total_timeout_seconds', 30.0)

    for attempt in range(self._max_retries + 1):
        elapsed = time.monotonic() - start_time
        if elapsed >= timeout:
            raise TimeoutError(f"Request timed out after {elapsed:.1f}s")
        # ... rest
```

**Testing**: Add test with mock that always fails, verify timeout

---

### CRIT-4: Error Response Parsing Can Crash
- **Status**: 🔴 Open
- **Files**: All provider clients
- **Lines**: Multiple (all error handling)
- **Assigned**: -
- **Est. Effort**: 1 hour (5 clients)
- **Blocked By**: -
- **PR**: -

**Description**: Assumes error responses are JSON, crashes on HTML

**Solution**:
```python
if response.status != 200:
    try:
        error = await response.json()
        error_msg = error.get('error', {}).get('message', str(error))
    except (aiohttp.ContentTypeError, json.JSONDecodeError):
        error_msg = await response.text()

    raise RuntimeError(f"{self.provider_name} API error: {response.status} - {error_msg}")
```

**Testing**: Mock HTML error response, verify graceful handling

---

### CRIT-5: API Keys Not Validated at Init
- **Status**: 🔴 Open
- **Files**: All provider clients except Ollama
- **Lines**: Multiple (__init__ methods)
- **Assigned**: -
- **Est. Effort**: 1 hour (4 clients)
- **Blocked By**: -
- **PR**: -

**Description**: Missing/invalid API keys only fail at first request

**Solution**:
```python
def __init__(self, config: dict):
    super().__init__(config)
    self.api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')

    if not self.api_key:
        raise ValueError(
            "OpenAI API key not configured. "
            "Set api_key in config or OPENAI_API_KEY env var"
        )

    if len(self.api_key) < 20:
        raise ValueError("Invalid OpenAI API key: too short")
```

**Testing**: Test with missing key, verify ValueError at init

---

## High Priority Issues (P1)

### HIGH-1: Cost Calculation Approximation
- **Status**: 🔴 Open
- **File**: `base.py`
- **Lines**: 279-301
- **Est. Effort**: 2 hours
- **Blocked By**: Need to update LLMResponse dataclass

### HIGH-2: Health Check Costs Money (Anthropic)
- **Status**: 🔴 Open
- **File**: `anthropic_client.py`
- **Lines**: 148-177
- **Est. Effort**: 30 minutes

### HIGH-3: RateLimiter Thread Safety
- **Status**: 🔴 Open
- **File**: `base.py`
- **Lines**: 108-113
- **Est. Effort**: 15 minutes

### HIGH-4: No Response Validation
- **Status**: 🔴 Open
- **Files**: All provider clients
- **Est. Effort**: 2 hours

### HIGH-5: API Key Exposure in Logs
- **Status**: 🔴 Open
- **Files**: All clients
- **Est. Effort**: 1 hour

### HIGH-6: Ollama Model Validation Missing
- **Status**: 🔴 Open
- **File**: `ollama.py`
- **Lines**: 59-95
- **Est. Effort**: 30 minutes

### HIGH-7: XAI generate_with_search Unused Parameter
- **Status**: 🔴 Open
- **File**: `xai_client.py`
- **Lines**: 166-257
- **Est. Effort**: 15 minutes

### HIGH-8: No Session Cleanup in BaseLLMClient
- **Status**: 🔴 Open
- **File**: `base.py`
- **Est. Effort**: 1 hour
- **Blocked By**: Related to CRIT-1

---

## Medium Priority Issues (P2)

### MED-1: Template Validation Too Lenient
- **Status**: 🔴 Open
- **File**: `prompt_builder.py`
- **Lines**: 227-289
- **Est. Effort**: 30 minutes

### MED-2: Token Estimation Approximate
- **Status**: 🔴 Open
- **File**: `prompt_builder.py`
- **Lines**: 165-173
- **Est. Effort**: 2 hours

### MED-3: No Token Budget Enforcement
- **Status**: 🔴 Open
- **File**: `base.py`
- **Est. Effort**: 1 hour

### MED-4: No Retry Differentiation
- **Status**: 🔴 Open
- **Files**: All clients
- **Est. Effort**: 3 hours

### MED-5: Pricing Data Hardcoded
- **Status**: 🔴 Open
- **Files**: `base.py`, all clients
- **Est. Effort**: 1 hour

### MED-6: No Metrics/Observability
- **Status**: 🔴 Open
- **Files**: All clients
- **Est. Effort**: 4 hours

---

## Low Priority Issues (P3)

### LOW-1: Ollama generate_with_chat Unused
- **Status**: 🔴 Open
- **File**: `ollama.py`
- **Lines**: 201-276
- **Est. Effort**: 10 minutes (remove or document)

### LOW-2: Inconsistent Default Models
- **Status**: 🔴 Open
- **Files**: All clients
- **Est. Effort**: 30 minutes

### LOW-3: Magic Numbers in Prompt Builder
- **Status**: 🔴 Open
- **File**: `prompt_builder.py`
- **Est. Effort**: 15 minutes

### LOW-4: No Streaming Support
- **Status**: 🔴 Open (documented as future)
- **Files**: All clients
- **Est. Effort**: N/A (future feature)

---

## Issue Statistics

```
Total Issues: 23
├─ Critical (P0): 5  (22%)  [Must fix before production]
├─ High (P1):     8  (35%)  [Fix within 1 week]
├─ Medium (P2):   6  (26%)  [Fix within 2 weeks]
└─ Low (P3):      4  (17%)  [Fix when convenient]

Status:
├─ Open:         23 (100%)
├─ In Progress:   0 (0%)
├─ Fixed:         0 (0%)
└─ Wontfix:       0 (0%)
```

---

## Sprint Planning

### Sprint 1 (Week 1): Critical Fixes
**Goal**: Fix all P0 issues
**Estimated Effort**: 8-10 hours

- [ ] CRIT-2: RateLimiter bug (0.25h)
- [ ] CRIT-5: API key validation (1h)
- [ ] CRIT-4: Error parsing (1h)
- [ ] CRIT-3: Total timeout (0.5h)
- [ ] CRIT-1: Session management (3h)
- [ ] Testing & validation (3h)

### Sprint 2 (Week 2): High Priority
**Goal**: Fix all P1 issues
**Estimated Effort**: 10-12 hours

- [ ] HIGH-8: Session cleanup in base (1h)
- [ ] HIGH-3: RateLimiter thread safety (0.25h)
- [ ] HIGH-7: XAI parameter fix (0.25h)
- [ ] HIGH-6: Ollama model validation (0.5h)
- [ ] HIGH-2: Health check caching (0.5h)
- [ ] HIGH-1: Cost calculation (2h)
- [ ] HIGH-4: Response validation (2h)
- [ ] HIGH-5: API key sanitization (1h)
- [ ] Testing & validation (3h)

### Sprint 3 (Week 3): Medium Priority
**Goal**: Fix high-value P2 issues
**Estimated Effort**: 8-10 hours

- [ ] MED-1: Template validation (0.5h)
- [ ] MED-3: Token budget enforcement (1h)
- [ ] MED-2: Token estimation improvement (2h)
- [ ] MED-5: Pricing freshness (1h)
- [ ] MED-4: Retry differentiation (3h)
- [ ] Testing & validation (2h)

---

## Dependencies Graph

```
CRIT-1 (Session Mgmt)
  └─> HIGH-8 (Session Cleanup)
  └─> All other session-related fixes

CRIT-5 (API Key Validation)
  └─> HIGH-5 (Key Sanitization)

HIGH-1 (Cost Calculation)
  └─> Requires LLMResponse refactor

MED-4 (Retry Differentiation)
  └─> Requires exception hierarchy
  └─> Affects all clients
```

---

## Testing Checklist

After each fix:
- [ ] Unit tests pass
- [ ] Integration tests pass (if applicable)
- [ ] Manual testing with real API
- [ ] Performance regression test
- [ ] Security review (for key handling)
- [ ] Documentation updated
- [ ] Changelog entry added

---

## Risk Register

| Issue | Risk if Unfixed | Probability | Impact |
|-------|-----------------|-------------|--------|
| CRIT-1 | Production outage | High | Critical |
| CRIT-2 | Rate limiting malfunction | Medium | High |
| CRIT-3 | Trading decision delays | Medium | Critical |
| CRIT-4 | Unhelpful error messages | Medium | Medium |
| CRIT-5 | Late API failures | High | Medium |
| HIGH-1 | Inaccurate cost tracking | Medium | Medium |
| HIGH-2 | Wasted API costs | High | Low |

---

## Notes

- All issues tracked in this file
- Update status as work progresses
- Link PRs when created
- Mark blocked issues clearly
- Prioritize based on trading system impact

**Last Review**: 2025-12-19
**Next Review**: After Sprint 1 completion
