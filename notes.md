# Development Notes

## Quick Start

```bash
cd ws_paper_tester
# Install dependencies
pip install -r requirements.txt
# Run with live Kraken WebSocket data
python ws_tester.py
# Run with simulated data
python ws_tester.py --simulated
# Run for 30 minutes with dashboard disabled
python ws_tester.py --duration 30 --no-dashboard
# Run tests
pytest tests/ -v
```

## Strategy Status

| Strategy | Version | Status | Last Updated |
|----------|---------|--------|--------------|
| Market Making | 1.5.0 | Production Ready | 2025-12-14 |
| Order Flow | 4.1.0 | Production Ready | 2025-12-14 |
| Mean Reversion | 2.0.0 | Production Ready | 2025-12-14 |
| Ratio Trading | 2.1.0 | Production Ready | 2025-12-14 |

## Recent Work

### Ratio Trading v2.1.0 (2025-12-14)
Implemented recommendations from ratio-trading-strategy-review-v2.0.md:
- REC-013: Higher entry threshold (1.0 -> 1.5 std)
- REC-014: RSI confirmation filter
- REC-015: Trend detection warning
- REC-016: Enhanced accumulation metrics
- Added trailing stops (from mean reversion patterns)
- Added position decay (from mean reversion patterns)
- Fixed hardcoded max_losses bug in on_fill

### Mean Reversion v2.0.0 (2025-12-14)
Implemented recommendations from mean-reversion-strategy-review-v1.0.md

### Order Flow v4.1.0 (2025-12-14)
Implemented recommendations from order-flow-strategy-review-v4.0.md

## Future Strategy Ideas

- **Scalping (momentum)**: 1m-5m timeframes, quick momentum bursts
- **Arbitrage**: Tick-level, cross-exchange price differences
- **Trend Following**: 9 week MA on 5m/1h, trend confirmation rules
- **XRP/BTC Correlation**: Does XRP follow BTC predictably enough to trade on?

## Documentation Structure

```
ws_paper_tester/docs/
├── development/
│   ├── features/        # Implementation docs per strategy
│   └── review/          # Code and strategy reviews
├── user/
│   └── how-to/         # Configuration guides
└── CODE_REVIEW_ISSUES.md
```
