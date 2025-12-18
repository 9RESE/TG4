# 01 - Introduction and Goals

## Requirements Overview

TG4 (TripleGain) is a multi-asset LLM-assisted trading system designed to autonomously grow holdings across BTC, USDT, and XRP through strategic trading.

### Primary Goals

1. **Autonomous Trading**: Use LLMs as decision-making agents for trade execution
2. **Multi-Asset Growth**: Accumulate BTC (45%), XRP (35%), USDT (20%)
3. **Risk Management**: Conservative leverage (max 3x), strict stop-losses
4. **Performance Tracking**: Maintain Sharpe Ratio > 1.5, Max Drawdown < 15%

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Execute trades on Kraken via WebSocket | High |
| FR-02 | Multi-LLM comparison (Claude, Grok, GPT-4, Deepseek) | High |
| FR-03 | Technical analysis with indicators | High |
| FR-04 | Risk management with stop-loss/take-profit | Critical |
| FR-05 | Portfolio rebalancing across three assets | Medium |
| FR-06 | Historical backtesting with TimescaleDB | High |

## Quality Goals

| Priority | Goal | Description |
|----------|------|-------------|
| 1 | Reliability | System must operate 24/7 without manual intervention |
| 2 | Performance | Trade execution within 100ms of signal |
| 3 | Safety | Never exceed risk parameters, mandatory stop-losses |
| 4 | Auditability | Full logging of all decisions and trades |

## Stakeholders

| Role | Description | Expectations |
|------|-------------|--------------|
| Developer/Operator | Primary user and maintainer | System stability, clear logs |
| Trading Agent (LLM) | Decision-making component | Accurate market context |
| Risk Manager (Rules) | Automated risk enforcement | Clear rule definitions |
