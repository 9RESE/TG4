# Review Phase 2A: LLM Integration Layer

**Status**: Ready for Review
**Estimated Context**: ~3,500 tokens (code) + review
**Priority**: High - All agents depend on this
**Output**: `findings/phase-2a-findings.md`
**DO NOT IMPLEMENT FIXES**

---

## Files to Review

| File | Lines | Purpose |
|------|-------|---------|
| `triplegain/src/llm/clients/base.py` | ~200 | Base LLM client interface |
| `triplegain/src/llm/clients/ollama.py` | ~300 | Ollama (local) client |
| `triplegain/src/llm/clients/openai_client.py` | ~350 | OpenAI API client |
| `triplegain/src/llm/clients/anthropic_client.py` | ~300 | Anthropic API client |
| `triplegain/src/llm/clients/deepseek_client.py` | ~300 | DeepSeek API client |
| `triplegain/src/llm/clients/xai_client.py` | ~300 | xAI (Grok) API client |

**Total**: ~1,750 lines

---

## Pre-Review: Load Files

```bash
# Read these files before starting review
cat triplegain/src/llm/clients/base.py
cat triplegain/src/llm/clients/ollama.py
cat triplegain/src/llm/clients/openai_client.py
cat triplegain/src/llm/clients/anthropic_client.py
cat triplegain/src/llm/clients/deepseek_client.py
cat triplegain/src/llm/clients/xai_client.py
```

---

## Review Checklist

### 1. Base Client (`base.py`)

#### Interface Design
- [ ] Abstract base class properly defined
- [ ] Required methods have type hints
- [ ] Common functionality in base class
- [ ] Extension points clear for subclasses

#### Common Functionality
- [ ] Response parsing standardized
- [ ] Error types defined
- [ ] Retry logic abstracted
- [ ] Metrics/logging hooks

#### Type Safety
- [ ] Response types clearly defined
- [ ] Input validation in base class
- [ ] Generic types used appropriately

---

### 2. All Clients - Common Checks

For EACH client (Ollama, OpenAI, Anthropic, DeepSeek, xAI):

#### API Key Handling
- [ ] API key loaded from environment variable
- [ ] API key never logged
- [ ] API key not in default values
- [ ] Clear error if API key missing

#### Request Handling
- [ ] Timeout configured and enforced
- [ ] Request headers correct
- [ ] Content-Type set properly
- [ ] User-Agent set (if required)

#### Response Handling
- [ ] JSON parsing with error handling
- [ ] Response schema validation
- [ ] Token count extraction (if provided)
- [ ] Cost calculation (for API clients)

#### Error Handling
- [ ] HTTP error codes handled (400, 401, 403, 429, 500, etc.)
- [ ] Network timeout handling
- [ ] Connection error handling
- [ ] Rate limit detection and response
- [ ] Malformed response handling

#### Retry Logic
- [ ] Exponential backoff implemented
- [ ] Maximum retry count enforced
- [ ] Retry on transient errors only
- [ ] No retry on auth errors

---

### 3. Ollama Client (`ollama.py`)

#### Local Connection
- [ ] Ollama URL configurable
- [ ] Health check before requests
- [ ] Connection refused handling
- [ ] Model availability check

#### Model Management
- [ ] Model name validation
- [ ] Model pull if not available (optional)
- [ ] GPU/CPU selection (if applicable)

#### Performance
- [ ] Streaming response handling (if used)
- [ ] Context window management
- [ ] Memory usage considerations

#### Specific Checks
- [ ] Uses correct endpoint (`/api/generate` or `/api/chat`)
- [ ] Temperature/top_p parameters passed correctly
- [ ] JSON mode enforced for structured output

---

### 4. OpenAI Client (`openai_client.py`)

#### API Compliance
- [ ] Uses correct API version
- [ ] Uses chat completions endpoint
- [ ] Model name correct (`gpt-4-turbo`, etc.)
- [ ] Supports function calling (if used)

#### Token Management
- [ ] Input token count tracking
- [ ] Output token count tracking
- [ ] Token limit enforcement
- [ ] Cost calculation per call

#### Rate Limiting
- [ ] Rate limit headers parsed
- [ ] Backoff on 429 errors
- [ ] Request queuing (if needed)

#### Specific Checks
- [ ] `response_format: {"type": "json_object"}` used for JSON output
- [ ] System message passed correctly
- [ ] Max tokens configured

---

### 5. Anthropic Client (`anthropic_client.py`)

#### API Compliance
- [ ] Uses Messages API (not legacy Completion)
- [ ] Model name correct (`claude-sonnet-4-20250514`, etc.)
- [ ] API version header set
- [ ] Beta features properly enabled (if used)

#### Message Format
- [ ] System message in correct location
- [ ] User/assistant message alternation
- [ ] Content blocks formatted correctly

#### Specific Checks
- [ ] `max_tokens` parameter set (required)
- [ ] Stop sequences handled
- [ ] Tool use properly structured (if used)

---

### 6. DeepSeek Client (`deepseek_client.py`)

#### API Compliance
- [ ] Uses correct base URL
- [ ] Model name correct (`deepseek-v3`, etc.)
- [ ] OpenAI-compatible format used correctly

#### Specific Checks
- [ ] API differences from OpenAI handled
- [ ] Context length limits respected
- [ ] Response format handling

---

### 7. xAI Client (`xai_client.py`)

#### API Compliance
- [ ] Uses correct base URL for Grok API
- [ ] Model name correct (`grok-2`, etc.)
- [ ] Authentication method correct

#### Specific Checks
- [ ] API differences handled
- [ ] Web search capability (if available)
- [ ] Rate limits understood

---

## Critical Questions

1. **Secrets Exposure**: Are API keys ever logged or included in error messages?
2. **Timeout Handling**: What happens if an LLM call takes >30 seconds?
3. **Retry Safety**: Are retries idempotent? Could retries cause duplicate actions?
4. **Cost Tracking**: Is cost per call calculated and logged?
5. **Fallback Logic**: What happens if primary provider fails?
6. **JSON Validity**: How is invalid JSON response handled?

---

## Security Checklist

- [ ] API keys stored in environment variables only
- [ ] No hardcoded credentials
- [ ] API keys masked in logs (show only last 4 chars)
- [ ] HTTPS used for all API calls
- [ ] Certificate validation enabled
- [ ] No sensitive data in prompts logged

---

## Performance Checklist

- [ ] Connection pooling used (httpx/aiohttp)
- [ ] Keep-alive connections
- [ ] Appropriate timeout values:
  - Local (Ollama): 5-10s
  - API (Tier 2): 30-60s
- [ ] Async operations don't block
- [ ] Memory efficient streaming (if applicable)

---

## Error Handling Matrix

| Error Type | Expected Behavior | Retry? |
|------------|-------------------|--------|
| 400 Bad Request | Log and raise | No |
| 401 Unauthorized | Log, raise, alert | No |
| 403 Forbidden | Log, raise, alert | No |
| 404 Not Found | Log and raise | No |
| 429 Rate Limit | Backoff and retry | Yes |
| 500 Server Error | Backoff and retry | Yes |
| 502 Bad Gateway | Backoff and retry | Yes |
| 503 Unavailable | Backoff and retry | Yes |
| Network Timeout | Backoff and retry | Yes (limited) |
| Connection Error | Backoff and retry | Yes (limited) |
| Invalid JSON | Log and raise | No |
| Empty Response | Log and raise | Maybe |

---

## Test Coverage Check

```bash
pytest --cov=triplegain/src/llm/clients \
       --cov-report=term-missing \
       triplegain/tests/unit/llm/
```

Expected tests per client:
- [ ] Successful request
- [ ] API key missing
- [ ] Timeout handling
- [ ] Rate limit handling
- [ ] Invalid response handling
- [ ] Network error handling

---

## Design Conformance

### Implementation Plan 2.4 (Trading Decision Agent)
- [ ] All 6 providers implemented
- [ ] Parallel execution supported
- [ ] Response format matches spec

### Master Design (02-llm-integration-system.md)
- [ ] Tier 1/Tier 2 distinction respected
- [ ] Token budgets enforced
- [ ] Cost tracking implemented

---

## Findings Template

```markdown
## Finding: [Title]

**File**: `triplegain/src/llm/clients/filename.py:123`
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

### Impact
[What could happen if not fixed]
```

---

## Review Completion

After completing this phase:

1. [ ] All 6 clients reviewed
2. [ ] Security checklist complete
3. [ ] Error handling verified
4. [ ] Findings documented
5. [ ] Ready for Phase 2B

---

*Phase 2A Review Plan v1.0*
