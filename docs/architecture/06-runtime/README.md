# 06 - Runtime View

## Trading Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                     Trading Cycle (every candle)                 │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  Receive │───>│ Calculate│───>│   LLM    │───>│  Risk    │
  │   Data   │    │Indicators│    │ Decision │    │  Check   │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                        │
                                                        v
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │   Log    │<───│  Update  │<───│  Execute │<───│  Order   │
  │ Decision │    │ Position │    │   Order  │    │  Build   │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## Scenario: New Trade Signal

1. **Data Reception**: WebSocket receives new candle data
2. **Indicator Calculation**: Update EMA, RSI, Bollinger Bands
3. **Context Building**: Format market data for LLM prompt
4. **LLM Query**: Send context to trading LLM
5. **Signal Parsing**: Extract action, confidence, parameters
6. **Risk Check**: Validate against constraints
7. **Order Building**: Create order with stop-loss/take-profit
8. **Execution**: Submit order to exchange
9. **Position Update**: Record new position
10. **Logging**: Store decision and execution details

## Scenario: Stop-Loss Trigger

1. **Price Update**: WebSocket receives price tick
2. **Position Check**: Compare price to stop-loss levels
3. **Trigger Detection**: Stop-loss condition met
4. **Market Order**: Submit immediate market sell
5. **Position Close**: Update position to closed
6. **P&L Recording**: Calculate and log trade result
7. **Cooldown Start**: Begin trade cooldown period

## Scenario: Daily Rebalancing

1. **Schedule Trigger**: Daily at configured time
2. **Balance Query**: Get current asset balances
3. **Allocation Check**: Compare to target (45/35/20)
4. **Rebalance Decision**: Determine trades needed
5. **Trade Execution**: Execute rebalancing trades
6. **Balance Update**: Record new balances
