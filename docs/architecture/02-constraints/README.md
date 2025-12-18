# 02 - Architecture Constraints

## Technical Constraints

| Constraint | Description | Rationale |
|------------|-------------|-----------|
| TC-01 | Python 3.10+ | Async support, type hints, ML libraries |
| TC-02 | TimescaleDB | Time-series optimized storage |
| TC-03 | WebSocket-native | Real-time data from Kraken |
| TC-04 | Local execution | Privacy, no cloud dependencies |
| TC-05 | Docker deployment | Reproducible environment |

## Hardware Constraints

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9 7950X |
| RAM | 128GB DDR5 |
| GPU | RX 6700 XT (ROCm-enabled) |
| Storage | SSD for TimescaleDB |

## Organizational Constraints

| Constraint | Description |
|------------|-------------|
| OC-01 | Single developer project |
| OC-02 | Paper trading only (no real funds at risk) |
| OC-03 | Exchange: Kraken (10x margin), Bitrue (3x ETFs) |

## Trading Constraints

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max leverage | 3x | Conservative per Alpha Arena research |
| Risk per trade | 1% of portfolio | Standard risk management |
| Max stop-loss distance | 2% from entry | Limit downside |
| Min risk:reward | 2:1 | Ensure positive expectancy |
| Max drawdown trigger | 10% | Pause trading, review |
| Confidence threshold | 0.6 | Minimum to execute |
| Trade cooldown | 30 minutes per pair | Prevent overtrading |

## Conventions

- All times in UTC
- All prices in quote currency
- Logging format: JSON structured logs
- Configuration: YAML files
