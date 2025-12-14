# grok_4_1 Documentation

## Quick Links
- [Architecture Overview](./architecture/01-introduction-goals/)
- [Getting Started](./user/tutorials/)
- [API Reference](./api/)
- [Development Guide](./development/)

## Recent Updates (2025-12-14)

### Strategy Modular Refactoring
Large strategies have been refactored into modular package structures for better maintainability:

- **Mean Reversion v4.2.1**: Split from 1,772-line file into 7 focused modules
- **Ratio Trading v4.2.1**: Split into 10 focused modules with dual-asset tracking
- **Strategies Reference**: [Complete Strategy Documentation](./user/reference/strategies.md)

### Strategy Deep Reviews (15+ cycles)
Comprehensive review process with research-backed parameter tuning:

- Mean Reversion: 6 deep review cycles (v1.0 → v6.0)
- Ratio Trading: 8 deep review cycles (v1.0 → v8.0)
- All strategies now comply with Strategy Development Guide v2.0

Key improvements:
- XRP/BTC correlation monitoring (correlation dropped from ~80% to ~40%)
- Dynamic correlation pause thresholds
- Fee profitability validation before signals
- Extended position decay timing

### WebSocket Paper Tester
A lightweight, WebSocket-native paper trading system for rapid strategy development.

- **Feature Doc**: [WebSocket Paper Tester Design](./development/features/WEBSOCKET_PAPER_TESTER_DESIGN.md)
- **How-To Guide**: [Using the Paper Tester](./user/how-to/websocket-paper-tester.md)
- **Create Strategies**: [Creating Custom Strategies](./user/how-to/create-strategy.md)
- **Strategies Reference**: [Strategy Documentation](./user/reference/strategies.md)
- **ADR**: [ADR-001: WebSocket Paper Tester Architecture](./architecture/09-decisions/ADR-001-websocket-paper-tester.md)

## Documentation Structure

### Technical Architecture (Arc42)
| Section | Description |
|---------|-------------|
| [01-Introduction](./architecture/01-introduction-goals/) | Goals and stakeholders |
| [02-Constraints](./architecture/02-constraints/) | Technical and organizational constraints |
| [03-Context](./architecture/03-context-scope/) | System context and scope |
| [04-Strategy](./architecture/04-solution-strategy/) | Solution approach |
| [05-Building Blocks](./architecture/05-building-blocks/) | System decomposition |
| [06-Runtime](./architecture/06-runtime-view/) | Runtime behavior |
| [07-Deployment](./architecture/07-deployment-view/) | Infrastructure |
| [08-Crosscutting](./architecture/08-crosscutting/) | Cross-cutting concerns |
| [09-Decisions](./architecture/09-decisions/) | Architecture Decision Records |
| [10-Quality](./architecture/10-quality/) | Quality requirements |
| [11-Risks](./architecture/11-risks/) | Risks and technical debt |
| [12-Glossary](./architecture/12-glossary/) | Terms and definitions |

### User Documentation (Diataxis)
- [Tutorials](./user/tutorials/) - Learning-oriented guides
- [How-To Guides](./user/how-to/) - Task-oriented instructions
- [Reference](./user/reference/) - Technical reference
- [Explanation](./user/explanation/) - Conceptual guides

### Visual Architecture (C4 Model)
- [Context](./c4-diagrams/1-context/) - System context
- [Containers](./c4-diagrams/2-containers/) - Container diagram
- [Components](./c4-diagrams/3-components/) - Component diagram
- [Code](./c4-diagrams/4-code/) - Code-level diagrams

### Development
- [Features](./development/features/) - Implementation docs
- [API](./api/) - API documentation

---
*Updated: 2025-12-14*
