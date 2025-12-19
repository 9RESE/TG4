# 10 - Quality Requirements

## Quality Tree

```
                        Quality
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   Reliability        Performance          Safety
        │                  │                  │
   ┌────┴────┐        ┌────┴────┐        ┌────┴────┐
   │         │        │         │        │         │
Uptime   Recovery  Latency  Throughput  Risk    Audit
```

## Quality Scenarios

### Reliability

| ID | Scenario | Measure | Target |
|----|----------|---------|--------|
| R1 | System runs continuously | Uptime | 99.9% |
| R2 | WebSocket disconnects | Reconnect time | < 5 seconds |
| R3 | LLM provider unavailable | Fallback time | < 1 second |
| R4 | Database failure | Recovery time | < 1 minute |

### Performance

| ID | Scenario | Measure | Target |
|----|----------|---------|--------|
| P1 | Trade signal generated | Signal latency | < 100ms |
| P2 | Order submitted | Execution latency | < 200ms |
| P3 | LLM decision requested | Response time | < 5 seconds |
| P4 | Historical data query | Query time | < 1 second |

### Safety

| ID | Scenario | Measure | Target |
|----|----------|---------|--------|
| S1 | Max position size | Portfolio percentage | < 10% |
| S2 | Stop-loss execution | Trigger accuracy | 100% |
| S3 | Max drawdown reached | Trading halt | Immediate |
| S4 | Invalid signal | Rejection rate | 100% |

### Auditability

| ID | Scenario | Measure | Target |
|----|----------|---------|--------|
| A1 | Trade executed | Log completeness | 100% |
| A2 | LLM decision made | Reasoning logged | 100% |
| A3 | Error occurred | Error details logged | 100% |
| A4 | Historical analysis | Data retention | 1 year |

## Success Metrics

| Metric | Target | Measurement Period |
|--------|--------|-------------------|
| Sharpe Ratio | > 1.5 | 90 days rolling |
| Max Drawdown | < 15% | All time |
| Win Rate | > 50% | 90 days rolling |
| Asset Growth | Positive | 90 days rolling |

## Code Quality

### Test Coverage

| Component | Coverage | Tests |
|-----------|----------|-------|
| Foundation Layer | 87% | 224 |
| Core Agents | 87% | 188 |
| Risk Engine | 87% | 90 |
| Orchestration | 87% | 114 |
| Execution | 87% | 70 |
| LLM Clients | 87% | 157 |
| API Layer | 87% | 110 |
| **Total** | **87%** | **969** |

*Last updated: 2025-12-19*

### Code Review Status

| Review | Status | Issues Found | Fixed |
|--------|--------|--------------|-------|
| Review 1 | Complete | Initial comprehensive | N/A |
| Review 2 | Complete | Consolidated findings | 25/25 |
| Review 3 | Complete | LLM integration deep dive | 12/12 |
| Review 4 Phase 1 | Complete | Foundation layer (P1/P2) | 8/8 |
| Review 4 Phase 1 P3 | Complete | Foundation layer (P3) | 5/5 |
| Review 4 Phase 2A | Complete | LLM client robustness | 15/15 |
| Review 4 Phase 2B | Complete | Agent layer robustness | 12/12 |
| Review 4 Phase 3-6 | Pending | - | - |

See [ADRs](../09-decisions/) for detailed fix documentation.

### Agent Safety Controls

| Control | Description | Status |
|---------|-------------|--------|
| Minimum Quorum | Require 4/6 models for trading | ✅ v0.3.4 |
| Regime Hysteresis | 2+ consecutive confirmations for regime change | ✅ v0.3.4 |
| Decimal Pricing | Exact arithmetic for trade prices | ✅ v0.3.4 |
| SQL Parameterization | All dynamic SQL values parameterized | ✅ v0.3.4 |
| Stop-Loss Bounds | 1-5% range validation | ✅ v0.3.4 |

### Static Analysis

- **Type Hints**: 100% coverage on public APIs
- **Docstrings**: All modules, classes, and public methods documented
- **Linting**: Black, isort, flake8 compliant
