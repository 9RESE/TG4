# How to Use the WebSocket Paper Tester

This guide explains how to run paper trading with live WebSocket data using the WebSocket Paper Tester.

## Prerequisites

- Python 3.12+
- Internet connection (for live Kraken WebSocket data)

## Quick Start

### 1. Install Dependencies

```bash
cd ws_paper_tester
pip install -r requirements.txt
```

### 2. Run with Simulated Data

For testing without connecting to Kraken:

```bash
python ws_tester.py --simulated
```

### 3. Run with Live Data

Connect to Kraken's WebSocket for real-time prices:

```bash
python ws_tester.py
```

### 4. Access the Dashboard

Open your browser to `http://localhost:8787` to see the real-time dashboard showing:
- Strategy leaderboard ranked by P&L
- Live trade feed
- Current prices
- Aggregate portfolio stats

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--duration` | 60 | Run duration in minutes |
| `--interval` | 100 | Main loop interval in milliseconds |
| `--symbols` | From strategies | Comma-separated symbols (e.g., `XRP/USD,BTC/USD`) |
| `--capital` | 100 | Starting capital per strategy in USD |
| `--no-dashboard` | False | Disable the web dashboard |
| `--simulated` | False | Use simulated data instead of live WebSocket |

## Examples

### Run for 30 minutes with higher capital

```bash
python ws_tester.py --duration 30 --capital 500
```

### Run without dashboard (headless mode)

```bash
python ws_tester.py --no-dashboard --simulated
```

### Specify custom symbols

```bash
python ws_tester.py --symbols XRP/USD,ETH/USD,BTC/USD
```

## Understanding the Output

### Console Output

The system prints periodic status updates every 30 seconds:

```
--- Status @ 14:30:22 (5.5 min) ---
Ticks: 3300 | Signals: 12 | Fills: 8
Total Equity: $302.45 | P&L: $+2.45

Leaderboard:
  1. order_flow: $108.23 (P&L: $+8.23, 15 trades)
  2. mean_reversion: $102.12 (P&L: $+2.12, 5 trades)
  3. market_making: $92.10 (P&L: $-7.90, 10 trades)
```

### Final Report

When the session ends, a complete summary is printed:

```
============================================================
SESSION COMPLETE
============================================================
Duration: 60.0 minutes
Mode: Live
Ticks: 36000
Signals: 142
Fills: 98

Total Capital: $300.00
Final Equity:  $315.67
Total P&L:     $+15.67 (+5.22%)
Win Rate:      58.2%

Strategy Performance:
------------------------------------------------------------
  1. order_flow           $ 115.23  P&L: $+15.23 (+15.23%)  Trades:  35  Win: 62.9%  [WIN]
  2. mean_reversion       $ 105.44  P&L:  $+5.44 ( +5.44%)  Trades:  28  Win: 57.1%  [WIN]
  3. market_making        $  95.00  P&L:  $-5.00 ( -5.00%)  Trades:  35  Win: 48.6%  [LOSS]
============================================================
```

## Log Files

Logs are stored in `ws_paper_tester/logs/`:

| Directory | Contents |
|-----------|----------|
| `system/` | System events (connections, errors) |
| `strategies/` | Per-strategy signal logs |
| `trades/` | All executed fills |
| `aggregated/` | Unified view of all events |

All logs are in JSON Lines format for easy parsing.

## Troubleshooting

### Dashboard not loading

1. Ensure FastAPI and uvicorn are installed: `pip install fastapi uvicorn`
2. Check if port 8787 is available (or configure different port in `config.yaml`)
3. Try running with `--no-dashboard` to verify the core system works

### No signals being generated

1. Check strategy configuration in `strategies/*.py`
2. Verify WebSocket connection is established (check console output)
3. Ensure sufficient candles have been collected (strategies need historical data)

### WebSocket connection fails

1. Check internet connection
2. Kraken may be experiencing issues - try `--simulated` mode
3. Check firewall settings for outbound WebSocket connections

## Understanding Position Types

The system supports four trading actions:

| Action | Description | When to Use |
|--------|-------------|-------------|
| `buy` | Open or add to a long position | Bullish signal, expect price to rise |
| `sell` | Close an existing long position | Taking profit or stop-loss on long |
| `short` | Open or add to a short position | Bearish signal, expect price to fall |
| `cover` | Close an existing short position | Taking profit or stop-loss on short |

Strategies automatically use the correct action based on current position:
- When flat and bearish signal → `short`
- When long and bearish signal → `sell`
- When short and bullish signal → `cover`

## Next Steps

- [Creating Custom Strategies](./create-strategy.md)
- [Understanding the Dashboard](./dashboard-guide.md)
- [Analyzing Log Files](./log-analysis.md)

---
*Last updated: 2025-12-13 (v1.0.2)*
