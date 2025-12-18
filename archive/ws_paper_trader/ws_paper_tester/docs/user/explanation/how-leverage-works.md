# How Leverage Works

This document explains the leverage system in the WebSocket Paper Tester - what it is, why it matters, and how it affects your trading.

---

## What is Leverage?

Leverage allows you to control a larger position than your available capital would normally permit. It's essentially borrowing funds to amplify your trading capacity.

**Example without leverage:**
- You have $100
- Price of XRP is $2.50
- You can buy 40 XRP ($100 / $2.50)

**Example with 2x leverage:**
- You have $100
- Price of XRP is $2.50
- You can buy 80 XRP ($200 / $2.50)
- You've "borrowed" $100

## Why Different Leverage for Longs vs Shorts?

| Position | Default Leverage | Reasoning |
|----------|------------------|-----------|
| Long | 1.5x | Conservative - unlimited downside risk |
| Short | 2.0x | More aggressive - capped downside (asset to $0) |

### Long Position Risk

When you buy with leverage:
- If price goes **up**, you profit on the amplified position
- If price goes **down**, losses are amplified
- **No floor**: In theory, losses can exceed your initial capital

This is why longs default to 1.5x - more conservative to limit potential losses.

### Short Position Risk

When you short with leverage:
- If price goes **down**, you profit
- If price goes **up**, you lose money
- **Has a floor**: Maximum loss is if asset goes to infinity (practically, liquidation prevents this)

Shorts can handle 2x because the worst case (asset goes to $0) gives you maximum profit, not maximum loss.

## How the Paper Tester Implements Leverage

### Equity Calculation

```
Equity = USDT Balance + Value of Holdings

For leveraged longs:
  USDT can be negative (borrowed funds)
  Holdings are positive (you own the asset)

For leveraged shorts:
  USDT is positive (proceeds from short sale)
  Holdings are negative (you owe the asset)
```

### Maximum Position Calculation

```
Long:  Max Position Value = Equity × max_long_leverage
Short: Max Position Value = Equity × max_short_leverage
```

If you request a position larger than the max, it's automatically capped.

## Margin Calls and Liquidation

### What is a Margin Call?

A margin call occurs when your equity drops too low relative to your position size. It's the exchange saying "your collateral is insufficient - add more or we'll close your position."

### Maintenance Margin

The paper tester uses a **25% maintenance margin** ratio:

```
Liquidation triggers when:
  Equity < Total Position Value × 25%
```

**Example:**
- You have $100 and buy $150 worth of XRP (1.5x leverage)
- USDT balance: -$50 (borrowed)
- XRP holdings: $150 worth

If XRP drops 50%:
- XRP holdings now worth: $75
- Equity = -$50 + $75 = $25
- Position value: $75
- Maintenance requirement: $75 × 25% = $18.75
- $25 > $18.75, so no liquidation

If XRP drops 70%:
- XRP holdings now worth: $45
- Equity = -$50 + $45 = -$5 (negative!)
- This triggers immediate liquidation

### What Happens on Liquidation?

1. System detects equity below maintenance margin
2. All positions are force-closed at market price
3. Signal reason shows "MARGIN CALL"
4. Remaining equity (if any) is preserved

## Practical Considerations

### When to Use Leverage

**Good candidates for leverage:**
- High-conviction signals with strong confluence
- Tight stop-losses already in place
- Favorable risk-reward setups
- Markets you understand well

**Avoid leverage when:**
- Market is choppy or unpredictable
- Your strategy has low win rate
- You're still testing the strategy
- During high-volatility events

### Calculating Your True Risk

With leverage, your actual risk is amplified:

```
True Risk = Position Size × Stop Loss % × Leverage Factor

Example:
  Position: $100 (at 1.5x leverage, so $150 actual exposure)
  Stop Loss: 2% below entry
  True Risk: $150 × 2% = $3 (3% of equity)
```

Without leverage, $100 position with 2% stop = $2 risk (2% of equity).

### Monitoring Leveraged Positions

Watch these indicators:
- **Equity**: Should stay positive and above maintenance margin
- **USDT Balance**: Negative means you're using leverage on longs
- **Asset Holdings**: Negative means you have short positions

## Configuration

```yaml
# In config.yaml
execution:
  max_long_leverage: 1.5   # 1.0 = no leverage, 2.0 = 2x
  max_short_leverage: 2.0  # Standard for shorts
```

To disable leverage entirely:
```yaml
execution:
  max_long_leverage: 1.0
  max_short_leverage: 1.0
```

---

## Summary

- Leverage amplifies both gains and losses
- Longs use 1.5x default (conservative) due to unlimited downside
- Shorts use 2.0x default (more aggressive) with capped downside
- 25% maintenance margin triggers automatic liquidation
- Always calculate your true risk when using leverage
