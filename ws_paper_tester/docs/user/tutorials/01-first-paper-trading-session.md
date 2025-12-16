# Tutorial: Your First Paper Trading Session

In this tutorial, you'll run your first paper trading session and understand what's happening.

**Time:** 10 minutes
**Prerequisites:** Python 3.10+, pip

---

## Step 1: Install Dependencies

```bash
cd ws_paper_tester
pip install -r requirements.txt
```

You should see packages like `websockets`, `fastapi`, `pyyaml` being installed.

## Step 2: Run in Simulated Mode

Let's start with simulated data (no internet required):

```bash
python ws_tester.py --duration 2 --simulated
```

You'll see output like:

```
================================================================================
                    WebSocket Paper Tester v1.15.1
================================================================================
Mode: SIMULATED
Duration: 2 minutes
Symbols: ['XRP/USDT', 'BTC/USDT', 'XRP/BTC']
Starting Capital: $100.00 per strategy
Dashboard: http://127.0.0.1:8787
--------------------------------------------------------------------------------

[10:30:00] Loaded 8 strategies
[10:30:00] Starting main trading loop...
```

## Step 3: Understand the Output

During the session, you'll see:

### Signal Generation
```
[10:30:15] [mean_reversion] BUY XRP/USDT $20.00 @ 2.3500
           Reason: RSI oversold (28.5), deviation -1.2%
```

This shows:
- `[mean_reversion]` - Which strategy generated the signal
- `BUY XRP/USDT` - Action and symbol
- `$20.00 @ 2.3500` - Size in USD and price
- `Reason:` - Why the strategy decided to trade

### Fill Execution
```
[10:30:15] [mean_reversion] FILLED BUY 8.51 XRP @ 2.3505 (fee: $0.02)
           P&L: $0.00 | Equity: $99.98
```

This shows:
- Actual fill with slippage (2.3505 vs 2.3500)
- Fee deducted
- Running P&L and equity

### Stop/Take-Profit Triggers
```
[10:31:42] [mean_reversion] STOP_LOSS XRP/USDT @ 2.3200
           P&L: -$0.65 | Equity: $99.33
```

## Step 4: View the Dashboard

Open your browser to `http://127.0.0.1:8787`

You'll see:
- **Portfolio Overview**: Equity, P&L, positions per strategy
- **Recent Trades**: Last 50 fills with details
- **Live Prices**: Current market data
- **Strategy Status**: Active/paused state

## Step 5: Check the Logs

After the session ends, examine the logs:

```bash
# View strategy signals
cat logs/strategies/mean_reversion_*.jsonl | head -5

# View trade fills
cat logs/trades/fills_*.jsonl | head -5

# View aggregated audit trail
cat logs/aggregated/unified_*.jsonl | head -5
```

Each log entry is JSON with timestamps, correlation IDs, and full context.

## Step 6: Run with Live Data

Now try with real Kraken data:

```bash
python ws_tester.py --duration 5
```

You'll see real prices from Kraken's WebSocket API. The strategies will generate signals based on actual market movements.

---

## What You Learned

- How to start a paper trading session
- What the console output means
- How to use the dashboard
- Where logs are stored
- Difference between simulated and live modes

## Next Steps

- [Writing Your First Strategy](02-writing-your-first-strategy.md) - Create your own trading algorithm
- [Configure Paper Tester](../how-to/configure-paper-tester.md) - Customize settings
