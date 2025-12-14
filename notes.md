update the docs with the recent work and ALL the changes in git. Ensure documentation complies with the documentation standards and expectations outlined in the claude.md file. Then commit.

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



Strategies:
- btc and xrp usdt pairs- 9 week moving average on the 5 min and 1 hour. One candle opposite the trend the trend close position. 2 candles closed above/below the 9 is a trend(the equation for a trend is not a set definition and is open to improvement)
- does xrp follow btc predictably enough to trade on?

I want to develop these strategies. 
| Strategy Type       | Suitable Interval | Notes                                      |
| Market Making       | Tick-level        | Bid/ask spread capture, requires WebSocket |
| Scalping (momentum) | 1m-5m             | Quick momentum bursts                      |
| Arbitrage           | Tick-level        | Cross-exchange price differences           |
| Order Flow          | Tick-level        | Trade tape analysis                        |



1. Market Making (market_making.py)

  Purpose: Provides liquidity by trading both sides of the spread

  Logic:
  - Trades when spread > 0.1% (configurable)
  - Uses orderbook imbalance for direction decisions
  - Skews position size based on current inventory
  - Tracks inventory by symbol in USD terms

  Signals Generated:
  | Condition                      | Action         |
  |--------------------------------|----------------|
  | Long inventory + sell pressure | Sell to reduce |
  | Short inventory + buy pressure | Buy to cover   |
  | Wide spread + bid imbalance    | Buy            |
  | Wide spread + ask imbalance    | Sell/Short     |

  Config Defaults:
  min_spread_pct: 0.1      # 0.1% minimum spread
  position_size_usd: 20    # $20 per trade
  max_inventory: 100       # Max $100 position
  inventory_skew: 0.5      # Reduce size as inventory builds
  take_profit_pct: 0.3     # 0.3% TP
  stop_loss_pct: 0.5       # 0.5% SL

  Symbols: XRP/USD

  ---
  2. Order Flow (order_flow.py)

  Purpose: Trades based on trade tape analysis and buy/sell imbalance

  Logic:
  - Analyzes last 50 trades for volume imbalance
  - Detects volume spikes (2x normal)
  - Uses VWAP for mean reversion confirmation
  - Position-aware (distinguishes sell vs short)

  Signals Generated:
  | Condition                                      | Action             |
  |------------------------------------------------|--------------------|
  | Buy imbalance > 30% + volume spike             | Buy                |
  | Sell imbalance > 30% + volume spike + has long | Sell               |
  | Sell imbalance > 30% + volume spike + no long  | Short              |
  | Buy imbalance + price below VWAP               | Buy (smaller size) |

  Config Defaults:
  imbalance_threshold: 0.3   # 30% imbalance to trigger
  volume_spike_mult: 2.0     # 2x volume spike
  position_size_usd: 25      # $25 per trade
  lookback_trades: 50        # Analyze last 50 trades
  cooldown_trades: 10        # Min trades between signals
  take_profit_pct: 0.5       # 0.5% TP
  stop_loss_pct: 0.3         # 0.3% SL

  Symbols: XRP/USD, BTC/USD

  ---
  3. Mean Reversion (mean_reversion.py)

  Purpose: Trades price deviations from moving average and VWAP

  Logic:
  - Calculates 20-period SMA from 5-minute candles
  - Uses RSI (14-period) for confirmation
  - Bollinger Bands for entry confirmation
  - VWAP deviation for additional opportunities

  Signals Generated:
  | Condition                                                | Action          |
  |----------------------------------------------------------|-----------------|
  | Price < SMA - 0.5% + RSI < 35 + near lower BB            | Buy             |
  | Price > SMA + 0.5% + RSI > 65 + near upper BB + has long | Sell            |
  | Price > SMA + 0.5% + RSI > 65 + near upper BB + no long  | Short           |
  | Price below VWAP + neutral RSI                           | Buy (half size) |

  Config Defaults:
  lookback_candles: 20       # 20 candles for SMA
  deviation_threshold: 0.5   # 0.5% deviation to trigger
  position_size_usd: 20      # $20 per trade
  rsi_oversold: 35           # RSI oversold level
  rsi_overbought: 65         # RSI overbought level
  take_profit_pct: 0.4       # 0.4% TP
  stop_loss_pct: 0.6         # 0.6% SL
  max_position: 50           # Max $50 position

  Symbols: XRP/USD

  ---
  Strategy Comparison

  | Strategy       | Style            | Data Used    | Symbols          | Risk Profile      |
  |----------------|------------------|--------------|------------------|-------------------|
  | Market Making  | Neutral/Scalping | Orderbook    | XRP/USD          | Low (tight stops) |
  | Order Flow     | Momentum         | Trade tape   | XRP/USD, BTC/USD | Medium            |
  | Mean Reversion | Counter-trend    | Candles/VWAP | XRP/USD          | Medium            |

  All strategies support:
  - Stop-loss and take-profit orders
  - Long and short positions
  - Position state tracking via on_fill() callback
  - Indicator logging via state['indicators']

Ratio Strategy

in the ws_paper_tester/ we have the strategy Order Flow. We need to do a deep code and strategy review. Deep research the strategy technique and its use with the XRP/USDT, BTC/USDT, XRP/BTC pairs. XRP/BTC trading goals is to accumulate both assets. USDT traded pairs goal is to accumulate USDT. I also want to ensure the strategy meets the requirements in ws_paper_tester/docs/development/strategy-development-guide.md. provide a document with your findings and recommendations for improvements and fixes. ultrathink

in the ws_paper_tester/ we have the strategy Order Flow. We need to refactor or rewrite it to implement the fixes and recommendations in ws_paper_tester/docs/development/market-making-strategy-review.md and ensure it meets the requirements in ws_paper_tester/docs/development/strategy-development-guide.md