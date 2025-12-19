# Phase 2A Findings: LLM Integration Layer

**Review Date**: 2025-12-19
**Reviewer**: Claude Code (Opus 4.5)
**Status**: ✅ ALL ISSUES RESOLVED (2025-12-19)
**Total Findings**: 15 (3 P0, 5 P1, 6 P2, 1 P3) - All Fixed

---

## Resolution Summary

All 15 findings have been addressed with the following implementations:

1. **Error type detection** - Added `_is_retryable()` method with NON_RETRYABLE_PATTERNS
2. **Connection pooling** - Implemented `_get_session()` with TCPConnector in all clients
3. **API key sanitization** - Added `sanitize_error_message()` function
4. **JSON response mode** - Enabled for all providers (OpenAI format, Anthropic prompt, Ollama format)
5. **Rate limit headers** - Added `_parse_rate_limit_headers()` and `update_from_provider()`
6. **93 new tests** - Comprehensive test coverage for JSON utilities and new features

Tests passing: **969/969**

---

## Summary Table

| ID | Priority | Category | Status | Title |
|----|----------|----------|--------|-------|
| 2A-01 | P0 | Logic | ✅ FIXED | Retry logic retries non-retryable errors |
| 2A-02 | P0 | Performance | ✅ FIXED | New ClientSession per request (no pooling) |
| 2A-03 | P0 | Security | ✅ FIXED | API keys potentially in error logs |
| 2A-04 | P1 | Logic | ✅ FIXED | Incorrect wait_time check in RateLimiter |
| 2A-05 | P1 | Logic | ✅ FIXED | Cost calculation uses approximation, not actual tokens |
| 2A-06 | P1 | Quality | ✅ FIXED | JSON response mode not enabled |
| 2A-07 | P1 | Logic | ✅ FIXED | Rate limit headers not parsed |
| 2A-08 | P1 | Quality | ✅ FIXED | parse_json_response not integrated into client flow |
| 2A-09 | P2 | Security | ✅ FIXED | No explicit certificate validation |
| 2A-10 | P2 | Quality | ✅ FIXED | No User-Agent header set |
| 2A-11 | P2 | Logic | ✅ FIXED | Empty content returns '' without warning |
| 2A-12 | P2 | Logic | ✅ FIXED | generate_with_search doesn't enable search |
| 2A-13 | P2 | Quality | ✅ FIXED | No response schema validation |
| 2A-14 | P2 | Quality | ✅ FIXED | Low test coverage for JSON utilities (56%) |
| 2A-15 | P3 | Quality | ✅ FIXED | Duplicate pricing definitions |

---

## Detailed Findings

### Finding 2A-01: Retry Logic Retries Non-Retryable Errors

**File**: `triplegain/src/llm/clients/base.py:313-354`
**Priority**: P0
**Category**: Logic

#### Description
The `generate_with_retry` method catches all exceptions and retries regardless of error type. Authentication errors (401), forbidden (403), and bad request (400) errors should not be retried as they will always fail.

#### Current Code
```python
for attempt in range(self._max_retries + 1):
    try:
        # ...
        response = await self.generate(...)
        # ...
    except Exception as e:  # Catches ALL exceptions
        last_exception = e
        if attempt < self._max_retries:
            # Exponential backoff
            delay = min(
                self._base_delay * (2 ** attempt),
                self._max_delay
            )
            logger.warning(
                f"{self.provider_name}: Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)
```

#### Recommended Fix
```python
# Define non-retryable error patterns
NON_RETRYABLE_PATTERNS = [
    "401", "unauthorized", "api key",
    "403", "forbidden",
    "400", "bad request", "invalid",
    "404", "not found",
]

def _is_retryable(self, error: Exception) -> bool:
    """Check if error is retryable."""
    error_str = str(error).lower()
    return not any(pattern in error_str for pattern in NON_RETRYABLE_PATTERNS)

# In generate_with_retry:
except Exception as e:
    last_exception = e
    if attempt < self._max_retries and self._is_retryable(e):
        # ... retry logic
    else:
        logger.error(f"{self.provider_name}: Non-retryable error or max retries: {e}")
        raise
```

#### Impact
- Wasting API quota on failed retries
- Delayed error feedback (3+ attempts for auth errors)
- Potential rate limiting from repeated invalid requests

---

### Finding 2A-02: New ClientSession Per Request (No Connection Pooling)

**File**: All client files
**Priority**: P0
**Category**: Performance

#### Description
Each request creates a new `aiohttp.ClientSession`, which bypasses connection pooling and keep-alive benefits. This adds significant overhead per request.

#### Current Code (ollama.py:97-101)
```python
async with aiohttp.ClientSession(timeout=self.timeout) as session:
    async with session.post(
        f"{self.base_url}/api/generate",
        json=payload
    ) as response:
```

#### Recommended Fix
```python
class OllamaClient(BaseLLMClient):
    def __init__(self, config: dict):
        super().__init__(config)
        # ... existing init ...
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create shared session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,  # Connection pool size
                keepalive_timeout=30,
            )
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector,
            )
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate(self, ...):
        session = await self._get_session()
        async with session.post(...) as response:
            # ...
```

#### Impact
- Each request incurs TCP connection overhead (~50-200ms)
- SSL handshake repeated for each HTTPS request (~100-300ms)
- Higher latency and resource usage
- For Tier 1 (per-minute calls), this is ~1,440 unnecessary connections/day

---

### Finding 2A-03: API Keys Potentially Exposed in Error Logs

**File**: All API client files
**Priority**: P0
**Category**: Security

#### Description
When API errors occur, the full error response is logged or included in exception messages. This could expose request details including authorization headers in logs or error outputs.

#### Current Code (openai_client.py:98-101)
```python
if response.status != 200:
    error = await response.json()
    raise RuntimeError(
        f"OpenAI API error: {response.status} - {error}"  # Error dict may contain sensitive info
    )
```

#### Recommended Fix
```python
if response.status != 200:
    error = await response.json()
    # Sanitize error before logging
    error_message = error.get('error', {}).get('message', 'Unknown error')
    error_type = error.get('error', {}).get('type', 'unknown')
    raise RuntimeError(
        f"OpenAI API error: {response.status} - {error_type}: {error_message}"
    )
```

Additionally, add explicit warning in logger configuration:
```python
# Ensure API keys are never logged
logger = logging.getLogger(__name__)
# Consider adding a filter to redact API key patterns
```

#### Impact
- API keys could appear in log files
- Security audit failures
- Potential credential exposure if logs are shipped to external systems

---

### Finding 2A-04: Incorrect wait_time Check in RateLimiter

**File**: `triplegain/src/llm/clients/base.py:216`
**Priority**: P1
**Category**: Logic

#### Description
The return statement uses `'wait_time' in dir()` to check if a local variable was defined. However, `dir()` returns module-level attributes, not local variables. This always evaluates incorrectly.

#### Current Code
```python
return max(0, wait_time) if 'wait_time' in dir() else 0
```

#### Recommended Fix
```python
async def acquire(self) -> float:
    """Acquire a slot for a request, waiting if necessary."""
    wait_time = 0.0  # Initialize at start
    async with self._lock:
        now = time.monotonic()
        # ... existing logic ...

        if len(self._request_times) >= self.max_requests:
            oldest = self._request_times[0]
            wait_time = (oldest + self.window_seconds) - now + 0.01
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # ... cleanup ...

        self._request_times.append(now)
        return wait_time  # Now always defined
```

#### Impact
- Return value may be incorrect (always 0 instead of actual wait time)
- Logging/metrics about rate limit waits are inaccurate
- Debugging rate limit issues becomes difficult

---

### Finding 2A-05: Cost Calculation Uses Approximation Instead of Actual Tokens

**File**: `triplegain/src/llm/clients/base.py:390-412`
**Priority**: P1
**Category**: Logic

#### Description
The base class `_calculate_cost` method uses a 70/30 split approximation for input/output tokens, even though all API clients have actual token counts available from the response.

#### Current Code
```python
def _calculate_cost(self, model: str, tokens_used: int) -> float:
    costs = MODEL_COSTS.get(model)
    if not costs:
        return 0.0

    # Approximate split: 70% input, 30% output
    input_tokens = int(tokens_used * 0.7)
    output_tokens = tokens_used - input_tokens
    # ...
```

#### Recommended Fix
```python
def _calculate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """Calculate cost using actual token counts."""
    costs = MODEL_COSTS.get(model)
    if not costs:
        return 0.0

    input_cost = (input_tokens / 1000) * costs['input']
    output_cost = (output_tokens / 1000) * costs['output']
    return input_cost + output_cost

# Update clients to pass actual values:
cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
```

Note: Clients already calculate cost correctly with actual tokens. Consider removing the base class method or making it consistent.

#### Impact
- Cost tracking is ~10-30% inaccurate depending on prompt/response ratio
- Budget alerts may trigger at wrong times
- Historical cost analysis is unreliable

---

### Finding 2A-06: JSON Response Mode Not Enabled

**File**: All API client files
**Priority**: P1
**Category**: Quality

#### Description
None of the API clients request JSON response mode from the providers. This means LLMs may return non-JSON responses that require parsing/extraction.

#### Current Code (openai_client.py)
```python
payload = {
    'model': model,
    'messages': [...],
    'temperature': temperature,
    'max_tokens': max_tokens,
    # No response_format specified
}
```

#### Recommended Fix

**OpenAI:**
```python
payload = {
    'model': model,
    'messages': [...],
    'temperature': temperature,
    'max_tokens': max_tokens,
    'response_format': {'type': 'json_object'},  # Add this
}
```

**Anthropic:** (system prompt approach)
```python
# Add to system prompt or use tool_choice for structured output
system_prompt = system_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON."
```

**DeepSeek/xAI:** (OpenAI-compatible)
```python
payload = {
    # ...
    'response_format': {'type': 'json_object'},
}
```

**Ollama:**
```python
payload = {
    # ...
    'format': 'json',  # Ollama-specific
}
```

#### Impact
- LLMs may return markdown-wrapped JSON or explanatory text
- Relies on `parse_json_response` to extract JSON from arbitrary text
- Increased latency from larger responses with explanations
- Higher token usage/cost

---

### Finding 2A-07: Rate Limit Headers Not Parsed

**File**: All API client files
**Priority**: P1
**Category**: Logic

#### Description
API providers return rate limit information in response headers (e.g., `X-RateLimit-Remaining`, `X-RateLimit-Reset`), but clients don't read these headers. This means the rate limiter operates on estimates rather than actual limits.

#### Current Code
Rate limit headers are completely ignored in all clients.

#### Recommended Fix
```python
async def generate(self, ...):
    # ... make request ...
    async with session.post(...) as response:
        # Parse rate limit headers
        self._update_rate_limit_from_headers(response.headers)
        # ... rest of processing ...

def _update_rate_limit_from_headers(self, headers: dict) -> None:
    """Update rate limiter from provider headers."""
    remaining = headers.get('X-RateLimit-Remaining')
    reset_time = headers.get('X-RateLimit-Reset')
    limit = headers.get('X-RateLimit-Limit')

    if remaining is not None:
        try:
            self._rate_limiter.update_from_provider(
                remaining=int(remaining),
                reset_time=int(reset_time) if reset_time else None,
                limit=int(limit) if limit else None
            )
        except ValueError:
            pass  # Ignore malformed headers
```

#### Impact
- Rate limiting is based on estimated RPM, not actual quota
- Could hit actual rate limits despite local rate limiter
- Could be too conservative when limits are higher than estimated

---

### Finding 2A-08: parse_json_response Not Integrated Into Client Flow

**File**: `triplegain/src/llm/clients/base.py:29-102`
**Priority**: P1
**Category**: Quality

#### Description
The `parse_json_response` and `validate_json_schema` utility functions are defined in the base module but not used by any client. Clients return raw text, leaving JSON parsing to consumers (agents). This creates inconsistent error handling.

#### Current Usage
```python
# In trading_decision.py (agent), not client:
from ..llm.clients.base import parse_json_response

parsed, error = parse_json_response(response_text)
```

#### Recommended Fix
Add an optional `parse_json` parameter to `generate`:
```python
async def generate(
    self,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    parse_json: bool = False,  # New parameter
) -> LLMResponse:
    # ... existing logic ...

    response = LLMResponse(
        text=text,
        # ...
    )

    if parse_json:
        parsed, error = parse_json_response(text)
        if error:
            logger.warning(f"JSON parsing failed: {error}")
        response.parsed_json = parsed  # Add field to LLMResponse

    return response
```

#### Impact
- Duplicate parsing logic across agents
- Inconsistent error handling for JSON failures
- The well-tested parsing utilities go unused

---

### Finding 2A-09: No Explicit Certificate Validation

**File**: All API client files
**Priority**: P2
**Category**: Security

#### Description
HTTPS connections rely on aiohttp's default SSL context. While this is secure by default, there's no explicit verification that certificate validation is enabled, and no handling for certificate errors.

#### Current Code
```python
async with aiohttp.ClientSession(timeout=self.timeout) as session:
    async with session.post(url, ...) as response:
        # No explicit ssl parameter
```

#### Recommended Fix
```python
import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())

async with aiohttp.ClientSession(
    timeout=self.timeout,
    connector=aiohttp.TCPConnector(ssl=ssl_context)
) as session:
    # ...
```

Or at minimum, explicitly set `ssl=True`:
```python
connector = aiohttp.TCPConnector(ssl=True)  # Explicit
```

#### Impact
- Security audits may flag implicit SSL
- No explicit protection against SSL-related attacks
- Future code changes could accidentally disable verification

---

### Finding 2A-10: No User-Agent Header Set

**File**: All API client files
**Priority**: P2
**Category**: Quality

#### Description
None of the clients set a User-Agent header. While not required, it's best practice for API integrations and helps providers identify legitimate traffic.

#### Recommended Fix
```python
headers = {
    'Authorization': f'Bearer {self.api_key}',
    'Content-Type': 'application/json',
    'User-Agent': 'TripleGain/0.3.0 (https://github.com/user/triplegain)',  # Add this
}
```

#### Impact
- Harder for providers to track usage patterns
- May be flagged as suspicious traffic
- Missing opportunity for proper identification

---

### Finding 2A-11: Empty Content Returns '' Without Warning

**File**: `triplegain/src/llm/clients/anthropic_client.py:110`
**Priority**: P2
**Category**: Logic

#### Description
When Anthropic returns an empty content array, the client silently returns an empty string instead of logging a warning or raising an error.

#### Current Code
```python
content = data.get('content', [])
text = content[0]['text'] if content else ''  # Silent empty return
```

#### Recommended Fix
```python
content = data.get('content', [])
if not content:
    logger.warning(
        f"Anthropic returned empty content for model {model}. "
        f"stop_reason: {data.get('stop_reason')}"
    )
    text = ''
else:
    text = content[0].get('text', '')
```

#### Impact
- Empty responses go unnoticed
- Debugging issues becomes harder
- Downstream components receive empty strings without context

---

### Finding 2A-12: generate_with_search Doesn't Enable Search

**File**: `triplegain/src/llm/clients/xai_client.py:166-257`
**Priority**: P2
**Category**: Logic

#### Description
The `generate_with_search` method accepts a `search_enabled` parameter but doesn't actually enable search functionality. The code explicitly notes this is a placeholder.

#### Current Code
```python
async def generate_with_search(
    self,
    ...
    search_enabled: bool = True,  # Parameter accepted but not used
    ...
) -> LLMResponse:
    # ...
    # Note: xAI search features may have specific API params
    # This is a placeholder for when the API supports explicit search control
```

#### Recommended Fix
Either:
1. Research xAI's actual search API and implement it
2. Remove the method until search is supported
3. Document the limitation clearly and rename to `generate_with_potential_search`

```python
async def generate_with_search(
    self,
    ...
) -> LLMResponse:
    """
    Generate response with potential web search grounding.

    WARNING: Search functionality is not yet implemented.
    This method currently behaves identically to generate().

    See: https://docs.x.ai/api for current search capabilities.
    """
    logger.warning("generate_with_search: search_enabled parameter is not yet functional")
    return await self.generate(...)
```

#### Impact
- Misleading API - consumers expect search but don't get it
- Sentiment analysis agent may rely on non-existent search
- Silent failure of expected functionality

---

### Finding 2A-13: No Response Schema Validation

**File**: All client files
**Priority**: P2
**Category**: Quality

#### Description
Clients don't validate that responses match the expected schema from each provider. Invalid or malformed responses could cause unexpected downstream errors.

#### Current Code
```python
data = await response.json()
# No validation that 'choices', 'usage', etc. exist
choice = data['choices'][0]  # Could raise KeyError
```

#### Recommended Fix
```python
OPENAI_RESPONSE_SCHEMA = {
    'required': ['id', 'choices'],
    'choices_required': ['message', 'finish_reason'],
}

def _validate_response(self, data: dict) -> None:
    """Validate response matches expected schema."""
    for field in OPENAI_RESPONSE_SCHEMA['required']:
        if field not in data:
            raise RuntimeError(f"Missing required field in response: {field}")

    if data.get('choices'):
        choice = data['choices'][0]
        for field in OPENAI_RESPONSE_SCHEMA['choices_required']:
            if field not in choice:
                raise RuntimeError(f"Missing required field in choice: {field}")
```

#### Impact
- KeyError exceptions instead of meaningful errors
- Difficult to diagnose API changes or issues
- No guarantee response is valid before processing

---

### Finding 2A-14: Low Test Coverage for JSON Utilities

**File**: `triplegain/src/llm/clients/base.py:45-102, 117-132`
**Priority**: P2
**Category**: Quality

#### Description
The `parse_json_response` and `validate_json_schema` functions have lower test coverage (noted as missing lines 45-102, 117-132 in coverage report). These are critical utilities for response handling.

#### Current Coverage
```
base.py: 56% coverage
Missing: 45-102, 117-132
```

#### Recommended Fix
Add comprehensive tests for edge cases:
```python
# Tests to add:
def test_parse_json_response_plain_json():
    """Test parsing plain JSON."""

def test_parse_json_response_markdown_wrapped():
    """Test parsing ```json ... ``` blocks."""

def test_parse_json_response_no_language_spec():
    """Test parsing ``` ... ``` without 'json'."""

def test_parse_json_response_json_in_text():
    """Test extracting JSON from surrounding text."""

def test_parse_json_response_nested_json():
    """Test deeply nested JSON extraction."""

def test_parse_json_response_empty():
    """Test empty response handling."""

def test_parse_json_response_invalid():
    """Test invalid JSON returns error."""

def test_validate_json_schema_missing_required():
    """Test missing required field detection."""

def test_validate_json_schema_wrong_type():
    """Test wrong field type detection."""
```

#### Impact
- Critical parsing code may have untested edge cases
- Production failures from malformed LLM responses
- Reduced confidence in JSON handling

---

### Finding 2A-15: Duplicate Pricing Definitions

**File**: All client files + base.py
**Priority**: P3
**Category**: Quality

#### Description
Pricing data is defined in both the base module (`MODEL_COSTS`) and in each client file (`OPENAI_PRICING`, `ANTHROPIC_PRICING`, etc.). This duplication could lead to inconsistencies.

#### Current Code
```python
# In base.py:
MODEL_COSTS = {
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
    # ...
}

# In openai_client.py:
OPENAI_PRICING = {
    'gpt-4-turbo': {'input': 10.00, 'output': 30.00},  # Per 1M tokens
    # ...
}
```

Note: These are different units (per 1K vs per 1M tokens), which is documented but could confuse maintainers.

#### Recommended Fix
Consolidate pricing in one location with clear unit specification:
```python
# In base.py or separate pricing.py:
MODEL_PRICING_PER_1M = {
    'openai': {
        'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
        'gpt-4o': {'input': 2.50, 'output': 10.00},
    },
    'anthropic': {
        'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
    },
    # ...
}

def get_cost_per_1k(provider: str, model: str) -> dict:
    """Get cost per 1K tokens (converted from 1M pricing)."""
    pricing = MODEL_PRICING_PER_1M.get(provider, {}).get(model)
    if not pricing:
        return {'input': 0.0, 'output': 0.0}
    return {
        'input': pricing['input'] / 1000,
        'output': pricing['output'] / 1000,
    }
```

#### Impact
- Maintenance burden updating pricing in multiple places
- Risk of inconsistent pricing between base and clients
- Confusion about units (per 1K vs per 1M)

---

## Critical Questions Assessment

| Question | Status | Notes |
|----------|--------|-------|
| Are API keys ever logged? | **RISK** | Full error responses logged, may contain request details |
| What happens if LLM call >30s? | **OK** | Properly raises RuntimeError after timeout |
| Are retries idempotent? | **RISK** | Generation calls retried - could cause duplicate processing |
| Is cost per call calculated? | **OK** | Yes, in all API clients with actual token counts |
| Fallback if provider fails? | **MISSING** | Not implemented in clients, needs orchestration layer |
| Invalid JSON handling? | **OK** | `parse_json_response` handles gracefully (but not integrated) |

---

## Security Checklist

| Check | Status | Notes |
|-------|--------|-------|
| API keys from env vars | PASS | All clients use os.environ fallback |
| No hardcoded credentials | PASS | Only env var names, not values |
| API keys masked in logs | **FAIL** | Error responses may expose details (2A-03) |
| HTTPS for all API calls | PASS | All URLs use https:// |
| Certificate validation | **WARN** | Implicit, not explicit (2A-09) |
| No sensitive data in prompts logged | PASS | Prompts logged at DEBUG only |

---

## Performance Checklist

| Check | Status | Notes |
|-------|--------|-------|
| Connection pooling | **FAIL** | New session per request (2A-02) |
| Keep-alive connections | FAIL | Would be enabled with pooling |
| Appropriate timeouts | PASS | Ollama: 30s, API: 60s |
| Async doesn't block | PASS | Proper async/await usage |
| Memory efficient streaming | N/A | Streaming not implemented |

---

## Test Coverage Summary

| File | Coverage | Notes |
|------|----------|-------|
| `__init__.py` | 100% | Simple exports |
| `base.py` | 56% | JSON utilities untested (2A-14) |
| `ollama.py` | 94% | Missing some error paths |
| `openai_client.py` | 96% | Well tested |
| `anthropic_client.py` | 96% | Well tested |
| `deepseek_client.py` | 96% | Well tested |
| `xai_client.py` | 94% | Missing search method tests |
| **TOTAL** | **82%** | Above 80% threshold |

---

## Design Conformance

### Implementation Plan 2.4 (Trading Decision Agent)
| Requirement | Status | Notes |
|-------------|--------|-------|
| All 6 providers implemented | PASS | Ollama, OpenAI, Anthropic, DeepSeek, xAI |
| Parallel execution supported | PARTIAL | Clients support async, orchestration needed |
| Response format matches spec | PASS | LLMResponse dataclass |

### Master Design (02-llm-integration-system.md)
| Requirement | Status | Notes |
|-------------|--------|-------|
| Tier 1/Tier 2 distinction | PASS | Different timeouts configured |
| Token budgets enforced | FAIL | Not in clients, only prompt_builder |
| Cost tracking implemented | PASS | Per-call cost in all API clients |

---

## Recommendations

### Immediate (Before Production)
1. **Fix P0 issues** (2A-01, 2A-02, 2A-03)
   - Add error type detection for retries
   - Implement connection pooling
   - Sanitize error logs

### Short-term (Next Sprint)
2. **Fix P1 issues** (2A-04 through 2A-08)
   - Fix RateLimiter return value
   - Enable JSON response mode
   - Integrate parse_json_response

### Medium-term (Backlog)
3. **Address P2/P3 issues**
   - Add explicit SSL context
   - Improve test coverage for JSON utilities
   - Consolidate pricing definitions

---

## Appendix: Files Reviewed

| File | Lines | Last Modified |
|------|-------|---------------|
| `triplegain/src/llm/clients/__init__.py` | 28 | Phase 2 |
| `triplegain/src/llm/clients/base.py` | 429 | Phase 2 |
| `triplegain/src/llm/clients/ollama.py` | 277 | Phase 2 |
| `triplegain/src/llm/clients/openai_client.py` | 165 | Phase 2 |
| `triplegain/src/llm/clients/anthropic_client.py` | 178 | Phase 2 |
| `triplegain/src/llm/clients/deepseek_client.py` | 163 | Phase 2 |
| `triplegain/src/llm/clients/xai_client.py` | 258 | Phase 2 |
| `triplegain/src/llm/prompt_builder.py` | 383 | Phase 2 |

---

*Phase 2A Review Complete - 2025-12-19*
