# TG4 - Local AI/ML Crypto Trading Platform

Documentation for the TG4 crypto trading platform with WebSocket paper trading and historical data system.

---

## Documentation Structure

This documentation follows the [Diataxis](https://diataxis.fr/) framework combined with [Arc42](https://arc42.org/) for architecture documentation.

### User Documentation (`/docs/user/`)

| Type | Description | Location |
|------|-------------|----------|
| **Tutorials** | Step-by-step learning guides | `/docs/user/tutorials/` |
| **How-To Guides** | Task-oriented instructions | `/docs/user/how-to/` |
| **Reference** | Technical specifications | `/docs/user/reference/` |
| **Explanation** | Conceptual discussions | `/docs/user/explanation/` |

### Architecture Documentation (`/docs/architecture/`)

Following Arc42 template:

| Section | Description |
|---------|-------------|
| 01-introduction | Goals and requirements |
| 02-constraints | Technical and organizational constraints |
| 03-context | System context and external interfaces |
| 04-solution-strategy | High-level approach |
| 05-building-blocks | Component decomposition |
| 06-runtime | Runtime scenarios |
| 07-deployment | Infrastructure and deployment |
| 08-concepts | Cross-cutting concepts |
| 09-decisions | Architecture Decision Records (ADRs) |
| 10-quality | Quality requirements |
| 11-risks | Technical risks |
| 12-glossary | Terms and definitions |

### Development Documentation (`/docs/development/`)

| Type | Description |
|------|-------------|
| **Features** | Feature specifications and implementation details |
| **Reviews** | Deep strategy reviews and analysis |
| **Plans** | Design documents and implementation plans |

---

## Quick Links

### Getting Started

- [Getting Started Guide](user/how-to/getting-started.md) - Complete setup and first run
- [WebSocket Paper Tester Guide](../ws_paper_tester/docs/user/how-to/operate-historical-data.md) - Historical data system operation

### System Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **WebSocket Paper Tester** | Strategy testing with live data | [ws_paper_tester/](../ws_paper_tester/) |
| **Historical Data System** | TimescaleDB storage and queries | [Historical Data Guide](../ws_paper_tester/docs/development/features/historical-data-system/) |
| **Market Regime Detection** | Bull/Bear/Sideways detection | [Regime Detection Guide](../ws_paper_tester/docs/development/features/regime_detection/) |
| **Centralized Indicators** | Shared indicator library | [Indicator Library](../ws_paper_tester/docs/development/features/indicators/) |

### Strategies

| Strategy | Version | Description |
|----------|---------|-------------|
| Market Making | v2.2.1 | Spread capture with inventory management |
| Mean Reversion | v4.2.0 | Bollinger Band reversion with trailing stops |
| Order Flow | v5.0.0 | Volume imbalance and VPIN-based signals |
| Ratio Trading | v4.3.1 | XRP/BTC pair trading with correlation monitoring |
| Grid RSI Reversion | v1.3.0 | Hybrid grid + RSI mean reversion |
| WaveTrend Oscillator | v1.1.0 | LazyBear WaveTrend crossover signals |
| Whale Sentiment | v1.6.0 | Volume spike detection with contrarian mode |
| Momentum Scalping | v2.1.1 | Short-term momentum capture |

---

## Current Phase

**Phase 21** - WebSocket Paper Tester + Historical Data System

### Recent Releases

| Version | Date | Highlights |
|---------|------|------------|
| v1.17.1 | 2025-12-17 | Grid RSI v1.3.0 configurable timeframes, research v3 |
| v1.17.0 | 2025-12-16 | ML v2.1 class weights, precision metrics, regime detection |
| v1.16.0 | 2025-12-16 | EMA-9 v2.0 strict candle mode, simplified exits |
| v1.15.1 | 2025-12-15 | Historical Data System code review fixes |
| v1.15.0 | 2025-12-15 | TimescaleDB integration, gap filler, multi-timeframe aggregates |
| v1.14.1 | 2025-12-15 | Regime Detection System code review fixes |

See [CHANGELOG](../ws_paper_tester/CHANGELOG.md) for complete history.

---

## Research & Planning

### AI Trading Research (v3)

Comprehensive research on AI integration and trading strategies:

| Document | Description |
|----------|-------------|
| [Research Synthesis Digest](research/v3/research-synthesis-digest.md) | Key findings and strategic insights |
| [AI Integration Research](research/v3/ai-integration-research.md) | AI/ML integration opportunities |
| [Alpha Arena Deep Dive](research/v3/alpha-arena-agent-trading-deep-dive.md) | LLM agent trading analysis |
| [BTC/USDT Algo Trading](research/v3/btc-usdt-algo-trading-research.md) | Algorithm research and recommendations |
| [Freqtrade Deep Dive](research/v3/freqtrade-deep-dive.md) | Platform comparison |
| [TensorTrade Deep Dive](research/v3/tensortrade-deep-dive.md) | RL framework analysis |

### V3 Master Plan

Architecture and implementation plans for next-generation trading:

| Document | Description |
|----------|-------------|
| [System Architecture](research/v3/master-plan/01-system-architecture.md) | Multi-agent system design |
| [LLM Prompt Templates](research/v3/master-plan/02-llm-prompt-templates.md) | Trading prompt engineering |
| [Multi-Agent Coordination](research/v3/master-plan/03-multi-agent-coordination.md) | Agent orchestration |
| [Risk Management Rules](research/v3/master-plan/04-risk-management-rules-engine.md) | Risk engine design |
| [Implementation Roadmap](research/v3/master-plan/05-implementation-roadmap.md) | Phased delivery plan |
| [LLM Evaluation Framework](research/v3/master-plan/06-llm-evaluation-framework.md) | Model comparison |

---

## Project Goals

- **Accumulation Target**: BTC (45%), XRP (35%), USDT (20%)
- **Starting Capital**: $1000 USDT + 500 XRP
- **Mode**: Paper trading only (no real funds)
- **Exchanges**: Kraken (WebSocket v2), Bitrue (planned)

---

## System Requirements

- **Python**: 3.10+
- **Docker**: For TimescaleDB (optional but recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 50GB+ for historical data storage
- **OS**: Linux (Ubuntu 22.04+ recommended), macOS, Windows with WSL2

---

## Support

- **Issues**: [GitHub Issues](https://github.com/9RESE/TG4/issues)
- **Repository**: [GitHub](https://github.com/9RESE/TG4)

---

*Last updated: 2025-12-17*
