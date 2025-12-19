# ADR-008: LLM Client Robustness and Performance Fixes

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Development Team
**Related Review**: Phase 2A Findings (docs/development/reviews/full/review-4/findings/phase-2a-findings.md)

## Context

The Phase 2A code review identified 15 issues (3 P0, 5 P1, 6 P2, 1 P3) in the LLM integration layer. These issues impacted security, performance, and reliability of the multi-provider LLM client architecture.

Key problems identified:
1. **Security**: API keys potentially exposed in error logs
2. **Performance**: No connection pooling (new session per request)
3. **Reliability**: Retry logic didn't distinguish error types
4. **Quality**: JSON response mode not enabled, low test coverage

## Decision

We implemented comprehensive fixes to the LLM client layer:

### 1. Error Type Detection (2A-01)

**Pattern**: Non-retryable error classification

```python
NON_RETRYABLE_PATTERNS = [
    "401", "unauthorized", "api key", "authentication",
    "403", "forbidden", "access denied",
    "400", "bad request", "invalid",
    "404", "not found",
    "422", "unprocessable",
]

def _is_retryable(self, error: Exception) -> bool:
    error_str = str(error).lower()
    return not any(pattern in error_str for pattern in NON_RETRYABLE_PATTERNS)
```

**Rationale**: Authentication errors, invalid requests, and missing resources will never succeed on retry. Failing immediately saves API quota and provides faster feedback.

### 2. Connection Pooling (2A-02)

**Pattern**: Shared session with TCP connector

```python
async def _get_session(self, timeout: aiohttp.ClientTimeout) -> aiohttp.ClientSession:
    if self._session is None or self._session.closed:
        self._connector = aiohttp.TCPConnector(
            limit=10,
            keepalive_timeout=30,
            ssl=create_ssl_context(),
        )
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=self._connector,
        )
    return self._session
```

**Rationale**: Connection pooling eliminates TCP/SSL handshake overhead per request (~50-300ms savings). Critical for Tier 1 per-minute calls.

### 3. API Key Sanitization (2A-03)

**Pattern**: Regex-based key redaction

```python
def sanitize_error_message(error: Any, provider: str) -> str:
    key_pattern = r'(sk-[a-zA-Z0-9]{20,}|[a-zA-Z0-9]{32,}|Bearer\s+[^\s]+)'
    error_message = re.sub(key_pattern, '[REDACTED]', error_message)
    return f"{provider}: {error_type} - {error_message}"
```

**Rationale**: Error messages may be logged or displayed. Sanitization prevents accidental credential exposure.

### 4. JSON Response Mode (2A-06)

**Decision**: Enable provider-specific JSON mode

| Provider | Implementation |
|----------|---------------|
| OpenAI/DeepSeek/xAI | `response_format: {type: 'json_object'}` |
| Anthropic | System prompt suffix instruction |
| Ollama | `format: 'json'` |

**Rationale**: Structured output reduces post-processing complexity and token waste from explanatory text.

### 5. LLMResponse Extensions (2A-05, 2A-08)

**Pattern**: Extended dataclass with actual token counts

```python
@dataclass
class LLMResponse:
    # Existing fields...
    parsed_json: Optional[dict] = None
    parse_error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
```

**Rationale**: Accurate cost tracking requires actual token counts, not 70/30 approximation. Auto-parsed JSON simplifies agent code.

## Consequences

### Positive
- **Security**: API keys cannot appear in logs or error messages
- **Performance**: ~50-300ms latency reduction per request from connection reuse
- **Reliability**: Non-retryable errors fail immediately instead of 3x retries
- **Accuracy**: Cost tracking uses actual token counts
- **Quality**: 157 LLM tests with 87% coverage

### Negative
- Session lifecycle management required (clients must call `close()`)
- Slightly more complex base class

### Neutral
- Error message format changed (may affect log parsing)

## Alternatives Considered

1. **Lazy error classification**: Check error type after first retry
   - Rejected: Wastes one retry attempt on known-bad errors

2. **Global session pool**: Single pool shared by all clients
   - Rejected: Per-provider pools allow different SSL/timeout settings

3. **Strict schema validation**: Raise error on missing fields
   - Rejected: Log warning is less disruptive; providers may evolve

## References

- [Phase 2A Findings](../../../development/reviews/full/review-4/findings/phase-2a-findings.md)
- [LLM Integration Design](../../../development/TripleGain-master-design/02-llm-integration-system.md)
- [aiohttp Connection Pooling](https://docs.aiohttp.org/en/stable/client_advanced.html#limiting-connection-pool-size)
