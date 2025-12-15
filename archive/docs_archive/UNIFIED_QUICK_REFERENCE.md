# Unified Trading Platform - Quick Reference

## CLI Commands

```bash
# Paper Trading
python unified_trader.py paper                          # Default 60 min
python unified_trader.py paper --duration 120           # 2 hours
python unified_trader.py paper --interval 60            # 1 min intervals
python unified_trader.py paper --enable grid_arithmetic # Enable at runtime

# Experiments
python unified_trader.py experiment --preset aggressive
python unified_trader.py experiment --override strategy:param:value
python unified_trader.py experiment --preset conservative --duration 120

# Strategy Management
python unified_trader.py list                           # List all strategies
python unified_trader.py config --enable grid_arithmetic
python unified_trader.py config --disable ema9_scalper
python unified_trader.py config --set strategies.mean_reversion_vwap.rsi_oversold=32

# Analysis
python unified_trader.py analyze                        # List experiments
python unified_trader.py analyze --experiment exp_id    # Detailed analysis
python unified_trader.py compare exp_1 exp_2 exp_3      # Compare experiments

# Setup
python unified_trader.py init-config                    # Create config template
python unified_trader.py init-config --force            # Overwrite existing
```

## Available Strategies

| Strategy | Category | Default | Description |
|----------|----------|---------|-------------|
| `defensive_yield` | general | ON | RL-driven with yield |
| `mean_reversion_vwap` | general | ON | VWAP + RSI reversion |
| `xrp_btc_pair_trading` | general | ON | Cointegration stat arb |
| `ma_trend_follow` | general | ON | SMA-9 trend following |
| `xrp_btc_leadlag` | general | ON | Correlation lead-lag |
| `intraday_scalper` | scalper | ON | BB squeeze scalper |
| `ema9_scalper` | scalper | OFF | EMA-9 override |
| `grid_arithmetic` | grid | OFF | Fixed spacing |
| `grid_geometric` | grid | OFF | Percentage spacing |
| `grid_rsi_reversion` | grid | OFF | RSI-filtered |
| `grid_bb_squeeze` | grid | OFF | BB squeeze |
| `grid_trend_margin` | margin | OFF | 5x trend |
| `grid_dual_hedge` | margin | OFF | Hedge protection |
| `grid_time_weighted` | grid | OFF | Time-weighted |
| `grid_liq_hunter` | margin | OFF | Liquidation hunter |

## Key Parameters

### mean_reversion_vwap
```yaml
dev_threshold: 0.003    # VWAP deviation (tune: 0.002-0.005)
rsi_oversold: 35        # Buy threshold (tune: 30-40)
rsi_overbought: 65      # Sell threshold (tune: 60-70)
volume_mult: 1.3        # Volume filter (tune: 1.0-2.0)
max_leverage: 5         # Leverage (tune: 1-10)
```

### xrp_btc_pair_trading
```yaml
entry_z: 1.8            # Z-score entry (tune: 1.5-2.5)
exit_z: 0.5             # Z-score exit (tune: 0.3-0.7)
lookback: 336           # Hours for hedge ratio
max_leverage: 10
```

### intraday_scalper
```yaml
atr_threshold: 0.007    # 0.7% ATR to activate
rsi_oversold: 30        # Scalp buy
rsi_overbought: 70      # Scalp sell
stop_loss_pct: 0.005    # 0.5% stop
take_profit_pct: 0.01   # 1% target
```

## Experiment Presets

| Preset | Description |
|--------|-------------|
| `aggressive` | Higher leverage, tighter RSI thresholds |
| `conservative` | Lower leverage, wider thresholds |
| `grid_focus` | Only grid strategies enabled |

## Log Locations

```
logs/
├── strategies/{strategy}_{timestamp}.jsonl  # Per-strategy
├── experiments/{experiment_id}.jsonl        # Experiment
├── performance/{strategy}_metrics.json      # Rolling metrics
└── orchestrator/unified_{timestamp}.jsonl   # Master log
```

## Market Regimes

| Regime | Condition | Weight Adjustment |
|--------|-----------|-------------------|
| `high_volatility` | ATR > 4% | Scalpers ↑ |
| `low_volatility` | ATR < 1.5% | Defensive ↑ |
| `trend_up` | Price > SMA20 > SMA50 | Trend-follow ↑ |
| `trend_down` | Price < SMA20 < SMA50 | Defensive ↑ |
| `chop` | Sideways | Mean-reversion ↑ |

## Python API

```python
from unified_orchestrator import UnifiedOrchestrator
from portfolio import Portfolio

# Initialize
portfolio = Portfolio({'USDT': 2000.0, 'XRP': 0.0, 'BTC': 0.0})
orchestrator = UnifiedOrchestrator(portfolio, "strategies_config/unified.yaml")

# Runtime management
orchestrator.add_strategy('grid_arithmetic')
orchestrator.remove_strategy('ema9_scalper')
orchestrator.set_weight('mean_reversion_vwap', 0.25)
orchestrator.set_experiment_param('mean_reversion_vwap', 'rsi_oversold', 30)

# Trading loop
orchestrator.run_loop(duration_minutes=60, interval_seconds=300)

# Or manual cycle
decision = orchestrator.decide()
result = orchestrator.execute_paper(decision)

# Cleanup
orchestrator.close()
```

## Signal Format

```python
{
    'action': 'buy|sell|short|cover|hold|close',
    'symbol': 'XRP/USDT',
    'confidence': 0.75,     # 0.0-1.0
    'leverage': 5,          # 1-10
    'size': 0.10,           # Fraction of capital
    'reason': 'RSI oversold',
}
```

## Tuning Tips

1. **More signals**: Lower `rsi_oversold`, raise `rsi_overbought`
2. **Fewer signals**: Raise `rsi_oversold`, lower `rsi_overbought`
3. **Wider entries**: Increase `dev_threshold`, `entry_z`
4. **Tighter entries**: Decrease `dev_threshold`, `entry_z`
5. **Less risk**: Lower `max_leverage`, increase `stop_loss_pct`

## Troubleshooting

```bash
# Check enabled strategies
python unified_trader.py list

# View recent signals
cat logs/strategies/mean_reversion_vwap_*.jsonl | jq 'select(.type=="signal")'

# Check rejections
cat logs/orchestrator/unified_*.jsonl | jq 'select(.type=="orchestrator_decision")'

# NumPy conflict fix
pip install "numpy<2"
```
