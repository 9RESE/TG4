# Command Line Interface Reference

## Usage

```bash
python ws_tester.py [OPTIONS]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--duration`, `-d` | int | 60 | Session duration in minutes |
| `--capital`, `-c` | float | 100.0 | Starting capital per strategy (USD) |
| `--symbols`, `-s` | list | config | Trading symbols (comma-separated) |
| `--config` | path | config.yaml | Path to configuration file |
| `--simulated` | flag | false | Use simulated data instead of Kraken |
| `--no-dashboard` | flag | false | Disable web dashboard |
| `--log-level` | str | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Examples

### Quick Test (5 minutes, simulated)
```bash
python ws_tester.py --duration 5 --simulated
```

### Extended Session with Custom Capital
```bash
python ws_tester.py --duration 480 --capital 1000
```

### Specific Symbols Only
```bash
python ws_tester.py --symbols XRP/USDT,BTC/USDT
```

### Use Custom Config
```bash
python ws_tester.py --config my_config.yaml
```

### Headless Mode (No Dashboard)
```bash
python ws_tester.py --no-dashboard --duration 1440
```

### Debug Mode
```bash
python ws_tester.py --log-level DEBUG
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `WS_TESTER_API_KEY` | API key for dashboard authentication |
| `DATABASE_URL` | TimescaleDB connection string for historical data |
| `KRAKEN_API_KEY` | Kraken API key (optional, for REST operations) |
| `KRAKEN_API_SECRET` | Kraken API secret (optional) |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Normal exit (duration completed or Ctrl+C) |
| 1 | Configuration error |
| 2 | WebSocket connection failure |
| 3 | Strategy loading error |

## Signal Handling

- `Ctrl+C` (SIGINT): Graceful shutdown - closes positions, flushes logs
- `SIGTERM`: Same as Ctrl+C
- Second `Ctrl+C`: Force exit (may lose buffered data)
