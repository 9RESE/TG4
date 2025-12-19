# Feature Documentation

Implementation documentation for TripleGain features.

## Overview

This directory contains detailed documentation for each major feature, including:

- Design decisions
- Implementation details
- Testing approach
- Integration points

## TripleGain Features

| Feature | Phase | Status | Documentation |
|---------|-------|--------|---------------|
| Foundation Layer | 1 | **COMPLETE** | [phase-1-foundation.md](./phase-1-foundation.md) |
| Indicator Library | 1 | **COMPLETE** | [phase-1-foundation.md](./phase-1-foundation.md#1-indicator-library) |
| Market Snapshot Builder | 1 | **COMPLETE** | [phase-1-foundation.md](./phase-1-foundation.md#2-market-snapshot-builder) |
| Prompt Builder | 1 | **COMPLETE** | [phase-1-foundation.md](./phase-1-foundation.md#3-prompt-builder) |
| Core Agents | 2 | **COMPLETE** | [phase-2-core-agents.md](./phase-2-core-agents.md) |
| Base Agent Framework | 2 | **COMPLETE** | [phase-2-core-agents.md](./phase-2-core-agents.md#1-base-agent-framework) |
| Technical Analysis Agent | 2 | **COMPLETE** | [phase-2-core-agents.md](./phase-2-core-agents.md#2-technical-analysis-agent) |
| Regime Detection Agent | 2 | **COMPLETE** | [phase-2-core-agents.md](./phase-2-core-agents.md#3-regime-detection-agent) |
| Risk Management Engine | 2 | **COMPLETE** | [phase-2-core-agents.md](./phase-2-core-agents.md#4-risk-management-engine) |
| Trading Decision Agent | 2 | **COMPLETE** | [phase-2-core-agents.md](./phase-2-core-agents.md#5-trading-decision-agent) |
| LLM Client Infrastructure | 2 | **COMPLETE** | [phase-2-core-agents.md](./phase-2-core-agents.md#6-llm-client-infrastructure) |
| Agent Orchestration | 3 | **COMPLETE** | [phase-3-orchestration.md](./phase-3-orchestration.md) |
| Message Bus | 3 | **COMPLETE** | [phase-3-orchestration.md](./phase-3-orchestration.md#1-message-bus) |
| Coordinator Agent | 3 | **COMPLETE** | [phase-3-orchestration.md](./phase-3-orchestration.md#2-coordinator-agent) |
| Execution Engine | 3 | **COMPLETE** | [phase-3-orchestration.md](./phase-3-orchestration.md#3-execution-engine) |
| API Security | 4 | **COMPLETE** | [phase-4-api-security.md](./phase-4-api-security.md) |
| Configuration & Integration | 5 | **COMPLETE** | [phase-5-configuration.md](./phase-5-configuration.md) |
| Sentiment Analysis | 6 | Planned | Pending |
| Multi-LLM A/B Testing | 6 | Planned | Pending |
| Dashboard | 6 | Planned | Pending |

## Archived Features (v2)

| Feature | Status | Documentation |
|---------|--------|---------------|
| WebSocket Paper Tester | Archived | See `archive/ws_paper_trader/` |
| Historical Data System | Archived | See `archive/ws_paper_trader/` |

## Feature Documentation Template

When documenting a feature, include:

1. **Overview**: What the feature does
2. **Design**: Architecture and design decisions
3. **Implementation**: Key code and algorithms
4. **Configuration**: How to configure the feature
5. **Testing**: Test coverage and approach
6. **Integration**: How it connects to other components

## Adding Feature Docs

1. Create a new file: `feature-name.md`
2. Use the template above
3. Update this index
4. Link from relevant architecture docs

---

*Feature docs will be added as implementation progresses.*
