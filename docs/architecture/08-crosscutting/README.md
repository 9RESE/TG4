# 08 - Crosscutting Concepts

## Logging

### Structured Logging Format

All logs use JSON format for easy parsing:

```json
{
  "timestamp": "2024-12-17T10:30:00Z",
  "level": "INFO",
  "component": "strategy_engine",
  "event": "trade_signal",
  "data": {
    "pair": "BTC/USDT",
    "action": "BUY",
    "confidence": 0.75,
    "llm": "claude"
  }
}
```

### Log Levels

| Level | Usage |
|-------|-------|
| DEBUG | Detailed debugging information |
| INFO | Normal operation events |
| WARNING | Unexpected but handled situations |
| ERROR | Errors requiring attention |
| CRITICAL | System failures |

## Error Handling

### Strategy

1. **Retry with backoff**: Network errors, API rate limits
2. **Fail-safe defaults**: Use conservative values on error
3. **Circuit breaker**: Disable component after repeated failures
4. **Alert and log**: Always record errors for analysis

### Error Categories

| Category | Handling |
|----------|----------|
| Network | Retry 3x with exponential backoff |
| API Rate Limit | Wait and retry |
| Data Validation | Log and skip |
| Exchange Error | Alert, pause trading |
| LLM Error | Fallback to rules-based |

## Security

### API Key Management

- Store keys in environment variables
- Never log API keys
- Use separate keys for paper/live trading

### Data Protection

- Local-only execution
- No cloud storage of trading data
- Encrypted database connections

## Configuration Management

### Configuration Hierarchy

1. Environment variables (highest priority)
2. config.yaml file
3. Default values (lowest priority)

### Configuration Categories

| Category | File | Purpose |
|----------|------|---------|
| Trading | config/trading.yaml | Trading parameters |
| Risk | config/risk.yaml | Risk limits |
| Exchange | config/exchange.yaml | API configuration |
| LLM | config/llm.yaml | LLM provider settings |

## Monitoring

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| portfolio_value | Gauge | Total portfolio value in USDT |
| drawdown_current | Gauge | Current drawdown percentage |
| trades_total | Counter | Total trades executed |
| trade_latency | Histogram | Order execution latency |
| llm_latency | Histogram | LLM response latency |
