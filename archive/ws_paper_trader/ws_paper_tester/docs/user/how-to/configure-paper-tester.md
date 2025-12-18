# How to Configure the WebSocket Paper Tester

This guide explains how to configure the WebSocket Paper Tester for different trading scenarios.

## Configuration File

The tester uses `config.yaml` in the project root. All settings have sensible defaults.

### General Settings

```yaml
general:
  duration_minutes: 60        # Session length
  interval_ms: 100           # Main loop interval (ms)
  starting_capital: 100.0    # USD per strategy portfolio
```

### Symbol Configuration

```yaml
symbols:
  - XRP/USD
  - BTC/USD
```

Strategies will only trade symbols they are configured for AND that appear in this list.

### Data Source

```yaml
data:
  source: kraken              # 'kraken' or 'simulated'
  ws_url: wss://ws.kraken.com/v2
  reconnect_delay: 1.0
  max_reconnect_delay: 60.0
```

Use `simulated` for testing without network connectivity.

### Execution Settings

```yaml
execution:
  fee_rate: 0.001            # 0.1% per trade
  slippage_rate: 0.0005      # 0.05% slippage simulation
  max_short_leverage: 2.0    # Maximum short exposure relative to equity (2x default)
  max_long_leverage: 1.5     # Maximum long exposure relative to equity (1.5x default)
```

These parameters directly affect P&L calculations.

#### Leverage and Margin

The executor supports leveraged positions for both longs and shorts:

| Position Type | Default Leverage | Behavior |
|---------------|------------------|----------|
| Long | 1.5x | USDT can go negative (borrowed funds) |
| Short | 2.0x | Asset holdings go negative (borrowed to sell) |

**Margin Calls**: When equity drops below 25% of total position value (maintenance margin), all positions are automatically liquidated. This applies to both leveraged longs and shorts.

```yaml
# Conservative settings (no leverage on longs)
execution:
  max_long_leverage: 1.0     # Cash-only longs
  max_short_leverage: 2.0    # 2x shorts allowed

# Aggressive settings
execution:
  max_long_leverage: 2.0     # 2x longs
  max_short_leverage: 2.0    # 2x shorts
```

### Dashboard

```yaml
dashboard:
  enabled: true
  host: 127.0.0.1            # Localhost only (default, secure)
  port: 8787                 # Dashboard port
```

Set `enabled: false` to run without the web dashboard. For network access, change `host: 0.0.0.0`.

### Logging

```yaml
logging:
  base_dir: logs
  buffer_size: 100
  enable_aggregated: true
  console_output: true
```

Logs are written as JSON Lines (`.jsonl`) with automatic rotation and compression.

### Strategy Overrides

Override default strategy parameters without modifying strategy files:

```yaml
strategy_overrides:
  market_making:
    min_spread_pct: 0.1
    position_size_usd: 20
  order_flow:
    imbalance_threshold: 0.3
    position_size_usd: 25
  mean_reversion:
    lookback_candles: 20
    deviation_threshold: 0.5
```

## CLI Override

Command-line arguments override config file values:

```bash
# Override duration
python ws_tester.py --duration 120

# Override starting capital
python ws_tester.py --capital 500

# Use simulated data
python ws_tester.py --simulated

# Disable dashboard
python ws_tester.py --no-dashboard

# Specify config file
python ws_tester.py --config my_config.yaml
```

## Dashboard Authentication

Set the `WS_TESTER_API_KEY` environment variable to enable API key authentication:

```bash
export WS_TESTER_API_KEY="your-secret-key"
python ws_tester.py
```

Clients must include `X-API-Key` header or `api_key` query parameter.

## Strategy Security

For production, disable unsigned strategies:

1. Generate hashes for approved strategies:
   ```python
   from ws_tester.strategy_loader import generate_strategy_hashes, _save_strategy_hashes
   from pathlib import Path

   hashes = generate_strategy_hashes("strategies")
   _save_strategy_hashes(Path("strategies"), hashes)
   ```

2. Set in `strategy_loader.py`:
   ```python
   ALLOW_UNSIGNED_STRATEGIES = False
   ```

Modified strategy files will be blocked from loading.

## Common Configurations

### Development (Fast Iteration)

```yaml
general:
  duration_minutes: 5
  starting_capital: 100.0
data:
  source: simulated
dashboard:
  enabled: true
```

### Backtesting Equivalent

```yaml
general:
  duration_minutes: 480  # 8 hours
  starting_capital: 1000.0
data:
  source: simulated
logging:
  enable_aggregated: true
  console_output: false
```

### Live Paper Trading

```yaml
general:
  duration_minutes: 1440  # 24 hours
  starting_capital: 100.0
data:
  source: kraken
dashboard:
  enabled: true
  host: 127.0.0.1  # Localhost only
execution:
  fee_rate: 0.0026         # Match your actual exchange fees
  max_long_leverage: 1.5   # Moderate leverage
  max_short_leverage: 2.0
```

---

*Last Updated: 2025-12-15*
