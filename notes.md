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
| Scalping (momentum) | 1m-5m             | Quick momentum bursts                      |
| Arbitrage           | Tick-level        | Cross-exchange price differences           |


mean_reversion
ratio

in the ws_paper_tester/ we have the strategy mean_reversion. We need to do a deep code and strategy review. Deep research the strategy technique and its use with the XRP/USDT, BTC/USDT, XRP/BTC pairs. I also want to ensure the strategy meets the requirements in ws_paper_tester/docs/development/strategy-development-guide.md. provide a document in ws_paper_tester/docs/development/review/mean_reversion/ with your findings and recommendations for improvements and fixes. Do not include code in your documentation. ultrathink

in the ws_paper_tester/ we have the strategy mean_reversion. We need to refactor or rewrite it to implement the fixes and recommendations in ws_paper_tester/docs/development/review/mean_reversion/mean-reversion-deep-review-v4.0.md and ensure it meets the requirements in ws_paper_tester/docs/development/strategy-development-guide.md

in the ws_paper_tester/ we have the strategy ratio_trading. We need to do a deep code and strategy review. Deep research the strategy technique and its use with the XRP/USDT, BTC/USDT, XRP/BTC pairs. I also want to ensure the strategy meets the requirements in ws_paper_tester/docs/development/strategy-development-guide.md. provide a document in ws_paper_tester/docs/development/review/ratio_trading/ with your findings and recommendations for improvements and fixes. Do not include code in your documentation. ultrathink

in the ws_paper_tester/ we have the strategy ratio_trading. We need to refactor or rewrite it to implement the fixes and recommendations in ws_paper_tester/docs/development/review/ratio-trading-strategy-review-v2.1.md and ratio-trading-strategy-review-v3.1.md and ensure it meets the requirements in ws_paper_tester/docs/development/strategy-development-guide.md