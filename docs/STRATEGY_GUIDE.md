# TG4 Trading Strategy Guide

## Overview

TG4 is a local AI/ML crypto trading platform designed for **capital preservation in bear markets** while capturing upside in favorable conditions. The strategy uses reinforcement learning (PPO) combined with LSTM price prediction to make autonomous trading decisions.

**Core Philosophy**: Don't try to beat the market. Survive bears, compound yield, and capture rare high-probability opportunities.

---

## Table of Contents

1. [Strategy Summary](#strategy-summary)
2. [Trading Modes](#trading-modes)
3. [How It Works](#how-it-works)
4. [Key Indicators](#key-indicators)
5. [Position Sizing & Risk Management](#position-sizing--risk-management)
6. [Yield Accrual](#yield-accrual)
7. [Dashboard & Monitoring](#dashboard--monitoring)
8. [Quick Start Guide](#quick-start-guide)
9. [Configuration Reference](#configuration-reference)
10. [Performance Expectations](#performance-expectations)

---

## Strategy Summary

| Aspect | Description |
|--------|-------------|
| **Goal** | Accumulate BTC (45%), XRP (35%), USDT (20%) |
| **Starting Capital** | $1000 USDT + 500 XRP (~$2200 total) |
| **Mode** | Paper trading only (simulated) |
| **Exchanges** | Kraken (10x margin), Bitrue (3x ETFs) |
| **Update Frequency** | Every 10 minutes |
| **Yield** | 6.5% APY on parked USDT |

### The Three Pillars

1. **Defensive Parking**: In uncertain markets, park in USDT and earn yield
2. **Opportunistic Shorts**: Capture rare overbought rips with precision shorts
3. **Aggressive Offense**: Deploy leverage only on high-confidence dips

---

## Trading Modes

The system operates in three distinct modes based on market conditions:

### 1. Defensive Mode (Most Common)

**Trigger**: Neutral volatility OR no high-confidence signals

**Behavior**:
- Parks capital in USDT
- Earns 6.5% APY simulated yield
- No active positions
- Waits for better opportunities

**Why**: Most market conditions are noise. Sitting out preserves capital.

### 2. Offensive Mode (Rare)

**Trigger**:
- Volatility < 2% (ATR%)
- RSI < 30 (oversold)
- LSTM dip signal with confidence > 85%

**Behavior**:
- Deploys up to 10x leverage on Kraken (XRP)
- Uses 3x leveraged ETFs on Bitrue (BTC)
- Maximum 15% of portfolio at risk

**Why**: Low volatility + oversold conditions = high probability rebound.

### 3. Bear Mode (Opportunistic)

**Trigger**:
- Volatility > 4% (ATR%)
- RSI > 65-72 (overbought)
- Price above VWAP

**Sub-modes**:

| Type | RSI Threshold | Risk Level | Portfolio % |
|------|---------------|------------|-------------|
| Rip Short | > 72 | Aggressive | 8-12% |
| Selective Short | > 65 | Conservative | 5-8% |
| Opportunistic Short | > 68 | Moderate | 4% |
| Grind-Down | No signal | Park only | 0% (yield) |

**Why**: Overbought conditions in high volatility often revert to mean.

---

## How It Works

### Decision Flow

```
Every 10 minutes:
│
├─> Fetch latest OHLCV data (500 hourly candles)
│
├─> Calculate indicators:
│   ├─ ATR% (14-period volatility)
│   ├─ RSI (14-period momentum)
│   └─ VWAP (volume-weighted average price)
│
├─> LSTM generates confidence scores for:
│   ├─ Dip probability
│   └─ Direction prediction
│
├─> RL agent (PPO) selects action from 12-action space
│
├─> Orchestrator applies risk filters:
│   ├─ Mode determination (defensive/offensive/bear)
│   ├─ Confidence thresholds
│   └─ Position limits
│
├─> Execute trade (if conditions met)
│
├─> Apply yield to parked USDT
│
└─> Update dashboard & log results
```

### RL Action Space

| Action | Asset | Type |
|--------|-------|------|
| 0 | BTC | Buy |
| 1 | BTC | Hold |
| 2 | BTC | Sell |
| 3 | XRP | Buy |
| 4 | XRP | Hold |
| 5 | XRP | Sell |
| 6 | USDT | Park (sell to USDT) |
| 7 | USDT | Hold |
| 8 | USDT | Deploy (leverage) |
| 9 | BTC | Short (via BTC3S ETF) |
| 10 | XRP | Short (via Kraken margin) |
| 11 | ALL | Close all shorts |

---

## Key Indicators

### ATR% (Average True Range Percentage)

Measures volatility as a percentage of price.

| ATR% | Market State | Strategy Response |
|------|--------------|-------------------|
| < 2% | Low volatility (greed) | Look for offensive entries |
| 2-4% | Normal volatility | Defensive parking |
| > 4% | High volatility (fear) | Consider shorts or park |

### RSI (Relative Strength Index)

Measures momentum on a 0-100 scale.

| RSI | Condition | Strategy Response |
|-----|-----------|-------------------|
| < 30 | Oversold | Potential long entry |
| 30-65 | Neutral | No action |
| 65-68 | Mildly overbought | Watch for short setup |
| 68-72 | Opportunistic zone | Conservative short |
| > 72 | Extreme overbought | Aggressive short |

### VWAP (Volume-Weighted Average Price)

Institutional fair value indicator.

- **Price > VWAP**: Short candidates (overextended)
- **Price < VWAP**: Long candidates (undervalued)

### Fear/Greed Index

Composite score (0-100) calculated from volatility and RSI:

| Index | Label | Meaning |
|-------|-------|---------|
| 0-25 | Extreme Fear | High vol, low RSI |
| 25-45 | Fear | Elevated vol |
| 45-55 | Neutral | Balanced |
| 55-75 | Greed | Low vol, high RSI |
| 75-100 | Extreme Greed | Very low vol |

---

## Position Sizing & Risk Management

### Maximum Exposure Limits

| Position Type | Max Portfolio % | Max Leverage |
|---------------|-----------------|--------------|
| Margin Longs | 15% | 10x (Kraken) |
| ETF Longs | 15% | 3x (Bitrue) |
| Shorts (total) | 15% | 5x |
| Single Short | 8-12% | 3-5x |

### Short Position Management

| Exit Condition | Threshold | Action |
|----------------|-----------|--------|
| Take Profit | +15% gain | Close position |
| Stop Loss | -8% loss | Close position |
| RSI Mean Reversion | RSI < 40 | Auto-close |
| Time Decay | 14 days | Force close |

### Dynamic Position Sizing

Position size adjusts based on:
1. **Volatility**: Higher vol = smaller positions
2. **Confidence**: Higher confidence = larger positions
3. **Available capital**: Never exceed limits

```python
# Simplified logic
base_size = available_usdt * risk_percentage
confidence_adj = base_size * (confidence / 0.85)
volatility_adj = confidence_adj * (0.04 / current_volatility)
final_size = min(volatility_adj, max_allowed)
```

---

## Yield Accrual

### How Yield Works

When capital is parked in USDT (defensive mode), the system simulates lending yield:

| Platform | Simulated APY |
|----------|---------------|
| Kraken | 6% |
| Bitrue | 7% |
| **Average** | **6.5%** |

### Yield Calculation

```python
# Applied every cycle (10 minutes = 1/6 hour)
hourly_rate = 0.065 / 365 / 24  # ~0.000742% per hour
yield_per_cycle = usdt_balance * hourly_rate * (10/60)
```

### Projected Earnings

| USDT Balance | Daily Yield | Monthly Yield | Annual Yield |
|--------------|-------------|---------------|--------------|
| $1,000 | $0.18 | $5.42 | $65.00 |
| $2,000 | $0.36 | $10.83 | $130.00 |
| $3,000 | $0.53 | $16.25 | $195.00 |

---

## Dashboard & Monitoring

### Terminal Dashboard

The live dashboard displays:

```
────────────────────────────────────────────────────────
│                  TG4 LIVE DASHBOARD                  │
────────────────────────────────────────────────────────
│ Time: 2024-12-07 15:30:00                            │
│ Portfolio: $2,245.67                                 │
│ Mode: DEFENSIVE                                      │
│ Fear/Greed: 42.5 (Fear)                              │
────────────────────────────────────────────────────────
│ Volatility (ATR%):   3.25%                           │
│ RSI XRP:  52.3   RSI BTC:  48.7                      │
│ XRP: $2.3540   BTC: $99,500.00                       │
│ Yield This Cycle: $0.0031  Total: $5.8921            │
────────────────────────────────────────────────────────
```

### Log Files

| File | Contents |
|------|----------|
| `logs/trades.csv` | All executed trades with metadata |
| `logs/equity_curve.csv` | Portfolio value over time |

### Matplotlib Plots (Optional)

Enable with `--plots` flag:
- Portfolio value chart
- Fear/Greed gauge
- RSI indicators
- Volatility tracking

---

## Quick Start Guide

### 1. Installation

```bash
git clone https://github.com/9RESE/TG4.git
cd TG4
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the RL Agent

```bash
# Recommended: 2M timesteps (takes ~30-60 minutes on GPU)
python src/main.py --mode train-rl --timesteps 2000000 --device cuda

# Quick test: 500K timesteps
python src/main.py --mode train-rl --timesteps 500000 --device cuda
```

### 3. Run Live Paper Trading

```bash
# Standard mode (10-minute intervals, dashboard enabled)
python src/live_paper.py --interval 10

# With matplotlib plots (requires display)
python src/live_paper.py --interval 10 --plots

# Custom starting balance
python src/live_paper.py --usdt 2000 --xrp 0 --btc 0
```

### 4. Monitor Performance

```bash
# Watch trades in real-time
tail -f logs/trades.csv

# Watch equity curve
tail -f logs/equity_curve.csv
```

### 5. Stop Trading

Press `Ctrl+C` to gracefully stop. The system will print a session summary.

---

## Configuration Reference

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--interval` | 10 | Minutes between cycles |
| `--cycles` | infinite | Maximum cycles to run |
| `--usdt` | 1000 | Starting USDT balance |
| `--xrp` | 500 | Starting XRP balance |
| `--btc` | 0 | Starting BTC balance |
| `--log-dir` | logs | Directory for log files |
| `--no-dashboard` | false | Disable terminal dashboard |
| `--plots` | false | Enable matplotlib plots |

### Threshold Configuration

Edit `src/orchestrator.py` to adjust thresholds:

```python
# Confidence thresholds
LEVERAGE_CONFIDENCE_THRESHOLD = 0.80
OFFENSIVE_CONFIDENCE_THRESHOLD = 0.85
BEAR_CONFIDENCE_THRESHOLD = 0.78

# Volatility thresholds
GREED_VOL_THRESHOLD = 0.02  # Below = greed regime
BEAR_VOL_THRESHOLD = 0.04   # Above = bear regime
OPPORTUNISTIC_VOL_THRESHOLD = 0.042

# RSI thresholds
RSI_OVERBOUGHT = 65
RSI_OPPORTUNISTIC = 68
RSI_RIP_THRESHOLD = 72
RSI_SHORT_EXIT = 40
```

---

## Performance Expectations

### Historical Backtest Results

| Phase | Return | vs XRP | vs BTC | Mode Distribution |
|-------|--------|--------|--------|-------------------|
| Phase 12 | -7.5% | +5.9% | +4.0% | 100% defensive |
| Phase 13 | -7.1% | +6.3% | +4.4% | 100% defensive |
| Phase 14 | TBD | TBD | TBD | TBD |

### What to Expect

**In Bear Markets**:
- Returns of -5% to -10% (outperforming crypto by 5-10%)
- High defensive mode usage (80-100%)
- Consistent yield accrual
- Rare short captures on rips

**In Bull Markets**:
- Underperformance vs buy-and-hold
- More offensive mode usage
- Leveraged gains on confirmed dips

**In Sideways Markets**:
- Steady yield accumulation
- Minimal trading activity
- Capital preservation

### Key Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| Drawdown | < 15% | Maximum loss from peak |
| Win Rate (shorts) | > 60% | Profitable short trades |
| Yield/Month | $5-15 | From USDT parking |
| vs Benchmark | Positive | Outperform XRP/BTC blend |

---

## Troubleshooting

### Common Issues

**"RL model not loaded"**
- Train the model first: `python src/main.py --mode train-rl --timesteps 2000000`
- Check `models/` directory for saved model

**"Data fetch failed"**
- Check internet connection
- Kraken API may be rate-limited (wait 1 minute)

**"No trades executing"**
- Normal in defensive mode
- Check thresholds if market conditions seem favorable

**Dashboard not showing**
- Ensure terminal supports ANSI colors
- Try `--no-dashboard` for minimal output

---

## Disclaimer

This is a **paper trading simulation** for educational purposes only. Do not use real funds. Past performance does not guarantee future results. Cryptocurrency trading involves significant risk of loss.

---

*Last Updated: Phase 14 (December 2024)*
