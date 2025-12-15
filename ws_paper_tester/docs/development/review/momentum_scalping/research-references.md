# Research References: Momentum Scalping Strategy v2.0

**Review Date:** 2025-12-14
**Review Version:** v2.0 (Updated with December 2025 market data)

---

## Academic Sources

### Cryptocurrency RSI Effectiveness

1. **PMC/NIH Study - Effectiveness of RSI Signals in Cryptocurrency**
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC9920669/
   - Key Finding: "The long-only strategy results using RSI were relatively good, with 4 out of 10 indices making an overhold profit using the oversold RSI value as an entry signal."
   - Key Finding: "The authors advise against treating RSI oversold level as a long signal, as results show the upward, asymmetric nature of the cryptocurrency market may make primary RSI applications ineffective."
   - Relevance: Supports regime-based RSI band adjustment (REC-004)

### Momentum Trading Research

2. **CEPR - Momentum Trading and Predictable Crashes**
   - URL: https://cepr.org/voxeu/columns/momentum-trading-return-chasing-and-predictable-crashes
   - Key Finding: "The momentum strategy is most likely to crash when past returns are high. Capital available to momentum traders predicts sharp downturns in momentum profits."
   - Relevance: Supports circuit breaker mechanism

3. **Academic Reference - Tong Chio (2022)**
   - Finding: "MACD strategies on short timeframes underperform unless combined with additional momentum filters (like RSI or MFI)."
   - Relevance: Validates RSI + MACD combination approach

---

## Industry Sources

### MACD Settings for Scalping

4. **MC2 Finance - Best MACD Settings for 1 Minute Chart**
   - URL: https://www.mc2.fi/blog/best-macd-settings-for-1-minute-chart
   - Key Finding: "For 1-minute scalping, the best MACD settings are typically 6 (fast length), 13 (slow length), and 5 (signal line). This combination reacts fast (usually within 1–2 candles, or 60–120 seconds)."
   - Relevance: Validates current MACD configuration (6, 13, 5)

5. **FXOpen - 1-Minute Scalping Trading Strategies**
   - URL: https://fxopen.com/blog/en/1-minute-scalping-trading-strategies-with-examples/
   - Key Finding: Multiple scalping approaches with indicator combinations
   - Relevance: General scalping best practices

6. **OpoFinance - Best MACD Settings for 1 Minute Chart**
   - URL: https://blog.opofinance.com/en/macd-settings-for-1-minute-chart/
   - Key Finding: "The 5, 13, 6 configuration is a favored choice among scalpers due to its significantly heightened sensitivity."
   - Relevance: Confirms MACD settings are industry-standard

### RSI and MACD Combination

7. **Quantified Strategies - MACD and RSI Strategy**
   - URL: https://www.quantifiedstrategies.com/macd-and-rsi-strategy/
   - Key Finding: "73% Win Rate" in backtested scenarios combining MACD and RSI
   - Key Finding: "235 trades, average gain of 0.88% per trade, including commissions and slippage"
   - Relevance: Validates combined indicator approach

8. **WunderTrading - How to Use MACD with RSI**
   - URL: https://wundertrading.com/journal/en/learn/article/combine-macd-and-rsi
   - Key Finding: "MACD, being trend-focused and lagging, confirms overall market direction. RSI, being momentum-focused and leading, identifies potential exhaustion points."
   - Relevance: Explains indicator complementarity

### ADX for Cryptocurrency

9. **AltFINS - ADX Crypto Indicator**
   - URL: https://altfins.com/knowledge-base/new-up-down-trend-adx/
   - Key Finding: "ADX above 30 often signaled strong market moves in 2024. Bitcoin's ADX hit 35 during a major rally in April 2024."
   - Relevance: Supports ADX threshold of 30 for BTC

10. **Bybit Learn - ADX Indicator for Crypto**
    - URL: https://learn.bybit.com/en/indicators/what-is-adx-indicator
    - Key Finding: ADX interpretation levels for crypto trading
    - Relevance: Confirms ADX threshold guidance

11. **CryptoTailor - Mastering ADX for Crypto**
    - URL: https://cryptotailor.io/academy/indicators/mastering-adx-indicator-crypto-trend-strength
    - Key Finding: "If ADX falls below 20, it signals a sideways market. When ADX moves above 25, it suggests a trending market."
    - Relevance: Supports current ADX threshold logic

12. **PI42 - ADX Indicator Strategies**
    - URL: https://pi42.com/blog/adx-indicator/
    - Key Finding: "More than 60% of professional traders used ADX daily in 2024"
    - Key Finding: "When Bitcoin's RSI showed oversold in April, ADX confirmed trend strength with value of 40"
    - Relevance: Validates ADX + RSI combination

### XRP-BTC Correlation Analysis

13. **AMBCrypto - XRP Correlation with Bitcoin 2025**
    - URL: https://eng.ambcrypto.com/assessing-xrps-correlation-with-bitcoin-and-what-it-means-for-its-price-in-2025/
    - Key Finding: "XRP's correlation with Bitcoin is continuing to weaken... highlighting its growing independence in 2025, fueled by Ripple's expanding real-world footprint."
    - Relevance: Supports XRP/BTC pair pause recommendation

14. **Gate.com - XRP Bitcoin Price Correlation 2025**
    - URL: https://www.gate.com/crypto-wiki/article/how-is-the-price-correlation-between-xrp-and-bitcoin
    - Key Finding: "3-month correlation at 0.84 but declining"
    - Key Finding: "XRP's correlation with Bitcoin has seen a 90-day decline of 24.86%"
    - Relevance: Quantifies correlation decline

15. **CME Group - How XRP Relates to Crypto Universe**
    - URL: https://www.cmegroup.com/insights/economic-research/2025/how-xrp-relates-to-the-crypto-universe-and-the-broader-economy.html
    - Key Finding: "XRP shows more independent streak, correlating at +0.4 to +0.6 compared to BTC-ETH correlation of +0.8"
    - Relevance: Explains structural correlation difference

16. **MacroAxis - XRP BTC Correlation**
    - URL: https://www.macroaxis.com/invest/pair-correlation/XRP.CC/BTC.CC/XRP-vs-Bitcoin
    - Key Finding: Quantitative correlation analysis tools
    - Relevance: Correlation calculation methodology

### Scalping Strategy Risks

17. **CenterPoint Securities - Momentum Trading Guide**
    - URL: https://centerpointsecurities.com/momentum-trading/
    - Key Finding: "Momentum trading performs best in trending markets but struggles in range-bound conditions"
    - Relevance: Explains failure conditions

18. **HighStrike - Scalping Trading Strategy Guide 2025**
    - URL: https://highstrike.com/scalping-trading-strategy/
    - Key Finding: "Scalping requires intense focus. High trading frequency can generate significant slippage and commissions."
    - Relevance: Risk management considerations

19. **FXCC - Momentum Scalping Strategy**
    - URL: https://www.fxcc.com/momentum-scalping-strategy
    - Key Finding: "In scalping momentum strategies, the individual is trying to enter ahead of momentum and exit before momentum dissipates. Mistiming exit is a common failure point."
    - Relevance: Explains time-based exit rationale

### RSI in Crypto Markets

20. **CoinMarketCap - Crypto RSI Charts**
    - URL: https://coinmarketcap.com/charts/rsi/
    - Key Finding: Market-wide RSI data
    - Relevance: Benchmark RSI levels

21. **Quantified Strategies - RSI Trading Strategy**
    - URL: https://www.quantifiedstrategies.com/rsi-trading-strategy/
    - Key Finding: "91% Win Rate" with specific RSI strategy rules
    - Relevance: RSI strategy optimization

22. **Mind Math Money - RSI Indicator Explained**
    - URL: https://www.mindmathmoney.com/articles/the-rsi-indicator-how-to-use-the-rsi-indicator-relative-strength-index-for-trading-crypto-forex-and-stocks
    - Key Finding: "RSI works beautifully in sideways/ranging markets. In strong trends, RSI can stay overbought/oversold for weeks."
    - Relevance: Supports regime-based RSI adjustment

---

## Technical Documentation

### Platform Documentation

23. **Strategy Development Guide v1.0**
    - Location: `ws_paper_tester/docs/development/strategy-development-guide.md`
    - Content: Sections 1-12 covering strategy development requirements
    - Note: v2.0 not available

### Strategy Implementation

24. **Momentum Scalping v2.0.0 Release Notes**
    - Location: `ws_paper_tester/docs/development/features/momentum_scalping/momentum-scalping-v2.0.md`
    - Content: Implementation details for REC-001 through REC-004

---

## Research Summary by Topic

### RSI Effectiveness

| Source | Finding | Confidence |
|--------|---------|------------|
| PMC Study | RSI alone underperforms in crypto | HIGH |
| Quantified Strategies | RSI + MACD = 73% win rate | MEDIUM |
| Mind Math Money | RSI fails in trends | HIGH |

### MACD Settings

| Source | Recommended | Strategy Uses | Match |
|--------|-------------|---------------|-------|
| MC2 Finance | 6, 13, 5 | 6, 13, 5 | YES |
| OpoFinance | 5, 13, 6 | 6, 13, 5 | CLOSE |

### ADX Thresholds

| Source | Recommended | Strategy Uses | Assessment |
|--------|-------------|---------------|------------|
| PI42 | 30 for strong | 25 | LOW |
| AltFINS | 25-30 range | 25 | BORDERLINE |
| Bybit | 25+ = trending | 25 | MINIMUM |

### XRP-BTC Correlation

| Source | Finding | Implication |
|--------|---------|-------------|
| AMBCrypto | Weakening in 2025 | PAUSE XRP/BTC |
| Gate.com | -24.86% over 90 days | PAUSE XRP/BTC |
| CME Group | 0.4-0.6 independent | PAUSE XRP/BTC |

---

## December 2025 Market Data Sources (v2.0 Update)

### Bitcoin Market Status

25. **CoinDesk** - "Bitcoin Volatility Is Still Compressing"
    - URL: https://www.coindesk.com/markets/2025/12/10/bitcoin-volatility-is-still-compressing-dimming-year-end-rally-outlook
    - Key Finding: BTC annualized 30-day implied volatility dropped to 49%
    - Key Finding: Volatility compression suggests low odds of year-end rally
    - Relevance: Current BTC regime assessment

26. **CoinCodex** - "Bitcoin (BTC) Price Prediction 2025"
    - URL: https://coincodex.com/crypto/bitcoin/price-prediction/
    - Key Finding: RSI 44.94 (neutral), 3.27% 30-day volatility
    - Relevance: BTC technical indicators status

27. **CoinDCX** - "Bitcoin Price Prediction 2025"
    - URL: https://coindcx.com/blog/price-predictions/bitcoin-price-weekly/
    - Key Finding: BTC trading $90,000-$100,000 range
    - Relevance: Price context for position sizing

### XRP Market Status (December 2025)

28. **CoinGape** - "Top 3 Price Predictions for Bitcoin, Ethereum, and XRP"
    - URL: https://coingape.com/markets/top-3-price-predictions-for-bitcoin-ethereum-and-xrp-in-dec-2025/
    - Key Finding: XRP at $2.02, market cap $125.1 billion (4.63%)
    - Relevance: XRP current market position

29. **The Crypto Basic** - "XRP Bull Case Projection"
    - URL: https://thecryptobasic.com/2025/12/13/here-is-xrp-bull-case-projection-if-saylors-2045-bitcoin-prediction-materializes/
    - Key Finding: XRP showing growing institutional momentum
    - Relevance: XRP independence analysis

### 2025 Scalping Strategy Research

30. **Gate.io Web3** - "How Do MACD and RSI Indicators Signal Crypto Market Trends in 2025?"
    - URL: https://web3.gate.com/en/crypto-wiki/article/how-do-macd-and-rsi-indicators-signal-crypto-market-trends-in-2025-20251207
    - Key Finding: Crypto can sustain overbought conditions longer than traditional markets
    - Key Finding: NEAR Protocol 42% surge Nov 7 with RSI briefly exceeding 80
    - Relevance: RSI reliability in volatile conditions

31. **MC2 Finance** - "Best RSI for Scalping (2025 Guide)"
    - URL: https://www.mc2.fi/blog/best-rsi-for-scalping
    - Key Finding: RSI 7 ideal for scalping on 1-minute to 15-minute charts
    - Relevance: RSI period selection validation

32. **Memebell** - "Crypto Scalping Strategies for 2025"
    - URL: https://memebell.com/index.php/2025/02/08/crypto-scalping-strategies-for-2025/
    - Key Finding: 2025 features clearer regulation, more institutional players
    - Relevance: Market structure context

---

## Citation Notes

- All URLs accessed: December 14, 2025
- Market data reflects December 2025 conditions
- Academic citations should be verified in original sources
- Industry sources are trading blogs and educational content, not peer-reviewed

---

## Recommended Further Reading

1. **Academic Databases**
   - SSRN: Search "cryptocurrency momentum"
   - Google Scholar: Search "RSI effectiveness crypto"
   - JSTOR: Search "technical analysis profitability"

2. **Exchange Documentation**
   - Kraken API documentation for fee structures
   - Exchange market data for spread analysis

3. **Backtesting Studies**
   - TradingView community strategies
   - Quantified Strategies backtests

---

*End of Research References*
