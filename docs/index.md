# TG4 Documentation

**Multi-Asset LLM-Assisted Trading System (Codename: TripleGain)**

## Overview

TG4 is a local AI/ML crypto trading platform designed to accumulate BTC (45%), XRP (35%), and USDT (20%) through strategic trading. The system uses LLMs as decision-making agents with a multi-agent architecture.

**Current Phase**: Phase 21 - WebSocket Paper Tester + Historical Data System

---

## Documentation Structure

### Architecture (Arc42)

Technical architecture documentation following the Arc42 template.

| Section | Description |
|---------|-------------|
| [01 - Introduction](architecture/01-introduction/README.md) | System goals, stakeholders, requirements |
| [02 - Constraints](architecture/02-constraints/README.md) | Technical and organizational constraints |
| [03 - Context](architecture/03-context/README.md) | System context and external interfaces |
| [04 - Solution Strategy](architecture/04-solution-strategy/README.md) | Fundamental solution approach |
| [05 - Building Blocks](architecture/05-building-blocks/README.md) | System decomposition |
| [06 - Runtime](architecture/06-runtime/README.md) | Runtime scenarios |
| [07 - Deployment](architecture/07-deployment/README.md) | Infrastructure and deployment |
| [08 - Crosscutting](architecture/08-crosscutting/README.md) | Cross-cutting concerns |
| [09 - Decisions](architecture/09-decisions/README.md) | Architecture Decision Records (ADRs) |
| [10 - Quality](architecture/10-quality/README.md) | Quality requirements |
| [11 - Risks](architecture/11-risks/README.md) | Technical risks and debt |
| [12 - Glossary](architecture/12-glossary/README.md) | Terms and definitions |

#### Key Subsystem Documentation

| Subsystem | Description |
|-----------|-------------|
| [Kraken Historical Data System](architecture/05-building-blocks/kraken-db.md) | TimescaleDB-backed historical data storage and retrieval |

### C4 Diagrams

Visual architecture using the C4 model.

| Level | Description |
|-------|-------------|
| [Context](c4-diagrams/context/README.md) | System context diagram |
| [Containers](c4-diagrams/containers/README.md) | Container diagram |
| [Components](c4-diagrams/components/README.md) | Component diagrams |
| [Code](c4-diagrams/code/README.md) | Code-level diagrams |

#### Component Diagrams

| Component | Description |
|-----------|-------------|
| [Kraken DB](c4-diagrams/components/kraken-db.md) | Historical data system component diagrams |

### User Documentation (Diataxis)

End-user documentation following the Diataxis framework.

| Type | Description |
|------|-------------|
| [Tutorials](user/tutorials/README.md) | Learning-oriented guides |
| [How-To Guides](user/how-to/README.md) | Problem-solving recipes |
| [Reference](user/reference/README.md) | Technical reference information |
| [Explanation](user/explanation/README.md) | Understanding-oriented discussion |

#### Kraken Historical Data System

| Type | Document | Description |
|------|----------|-------------|
| Tutorial | [Quickstart](user/tutorials/kraken-db-quickstart.md) | Getting started with the historical data system |
| How-To | [Operations](user/how-to/kraken-db-operations.md) | Common database operations |
| Reference | [API](user/reference/kraken-db-api.md) | Complete API reference |
| Reference | [Data Holdings](user/reference/kraken-db-data-holdings.md) | Current database contents and statistics |
| Explanation | [Architecture](user/explanation/kraken-db-architecture.md) | Design decisions and trade-offs |

### Development

| Section | Description |
|---------|-------------|
| [Features](development/features/README.md) | Feature implementation docs |
| [Master Plan](development/master-plan/) | V3 implementation planning |
| [API](api/README.md) | API documentation |

### Research

| Section | Description |
|---------|-------------|
| [Master Plan Research](research/master_plan_research/) | Research and analysis |

---

## Quick Links

- [README](../README.md) - Project overview
- [CHANGELOG](../CHANGELOG.md) - Version history
- [V3 Vision](development/master-plan/v3_vision.md) - Project vision document

---

## Project Information

| Attribute | Value |
|-----------|-------|
| Target Exchange | Kraken (WebSocket) |
| Starting Capital | 1,000 USDT + 500 XRP (~$2,100) |
| Hardware | AMD Ryzen 9 7950X, 128GB DDR5, RX 6700 XT |
| Mode | Paper trading (no real funds) |
