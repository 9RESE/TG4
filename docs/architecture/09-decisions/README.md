# 09 - Architecture Decision Records

## ADR Index

This directory contains Architecture Decision Records (ADRs) documenting significant technical decisions.

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](./ADR-001-phase1-foundation-architecture.md) | Phase 1 Foundation Architecture | Accepted | 2025-12-18 |
| [ADR-002](./ADR-002-phase2-core-agents-architecture.md) | Phase 2 Core Agents Architecture | Accepted | 2025-12-18 |
| [ADR-003](./ADR-003-phase2-code-review-fixes.md) | Phase 2 Code Review Fixes | Accepted | 2025-12-18 |
| [ADR-004](./ADR-004-phase3-orchestration-architecture.md) | Phase 3 Orchestration Architecture | Accepted | 2025-12-18 |
| [ADR-005](./ADR-005-security-robustness-fixes.md) | Security & Robustness Fixes | Accepted | 2025-12-19 |
| [ADR-006](./ADR-006-consolidated-review-fixes.md) | Consolidated Code Review Fixes | Accepted | 2025-12-19 |
| [ADR-007](./ADR-007-phase1-foundation-review-fixes.md) | Phase 1 Foundation Review Fixes | Accepted | 2025-12-19 |
| [ADR-008](./ADR-008-llm-client-robustness-fixes.md) | LLM Client Robustness & Performance Fixes | Accepted | 2025-12-19 |
| [ADR-009](./ADR-009-agent-robustness-fixes.md) | Agent Layer Robustness & Safety Fixes | Accepted | 2025-12-19 |
| [ADR-010](./ADR-010-execution-layer-robustness.md) | Execution Layer Robustness Fixes | Accepted | 2025-12-19 |

## ADR Template

When creating new ADRs, use this template:

```markdown
# ADR-XXX: Title

## Status

[Proposed | Accepted | Deprecated | Superseded]

## Context

What is the issue that we're seeing that is motivating this decision?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult to do because of this change?

## Alternatives Considered

What other options were considered and why were they rejected?
```

## How to Create an ADR

1. Copy the template above
2. Create a new file: `ADR-XXX-title.md`
3. Fill in all sections
4. Update this index
5. Submit for review

## Categories

ADRs are typically created for:

- Technology choices (language, framework, database)
- Architecture patterns (microservices, event-driven)
- Integration approaches (API design, protocols)
- Security decisions (authentication, encryption)
- Performance optimizations (caching, scaling)
