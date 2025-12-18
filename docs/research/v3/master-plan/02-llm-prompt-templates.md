# LLM Prompt Templates for TripleGain

**Version:** 1.0
**Date:** December 2025
**Status:** Design Phase

---

## Overview

This document defines the prompt templates for the LLM-based Trading Decision Agent. These templates follow patterns proven effective in the Nof1.ai Alpha Arena competition, where structured prompts with clear output formats significantly improved model performance.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [System Prompts](#2-system-prompts)
3. [User Prompt Templates](#3-user-prompt-templates)
4. [Output Schema](#4-output-schema)
5. [Few-Shot Examples](#5-few-shot-examples)
6. [Error Handling](#6-error-handling)

---

## 1. Design Principles

### 1.1 Key Learnings from Alpha Arena

| Principle | Rationale | Implementation |
|-----------|-----------|----------------|
| **Structured Output** | Prevents parsing errors | JSON schema required |
| **Explicit Constraints** | Reduces hallucination | Hard-coded rules in prompt |
| **Confidence Scoring** | Enables position sizing | 0-1 scale mandatory |
| **Exit Plan Required** | Improves risk management | TP/SL/invalidation fields |
| **Reasoning Trace** | Enables learning/debugging | Short justification field |

### 1.2 Prompt Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       PROMPT ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SYSTEM PROMPT (Static)                                             │
│  ├── Role Definition                                                │
│  ├── Trading Rules & Constraints                                    │
│  ├── Risk Management Rules                                          │
│  ├── Output Format Specification                                    │
│  └── Behavioral Guidelines                                          │
│                                                                      │
│  USER PROMPT (Dynamic)                                              │
│  ├── Market Data Section                                            │
│  │   ├── Current Prices                                             │
│  │   ├── Technical Indicators                                       │
│  │   └── Market Regime                                              │
│  ├── Portfolio Section                                              │
│  │   ├── Current Balances                                           │
│  │   ├── Open Positions                                             │
│  │   └── Recent P&L                                                 │
│  ├── Trade History Section                                          │
│  │   └── Last 5 Decisions                                           │
│  └── Query Section                                                  │
│      └── Specific Question                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. System Prompts

### 2.1 Primary Trading System Prompt

```python
TRADING_SYSTEM_PROMPT = """You are an expert quantitative trader managing a cryptocurrency portfolio across BTC, USDT, and XRP. You make trading decisions based on technical analysis, market conditions, and risk management principles.

## TRADING RULES

### Position Management
- Maximum leverage: 3x
- Risk per trade: 1% of total portfolio value
- Maximum single position: 20% of portfolio
- Mandatory stop-loss on every trade
- Stop-loss maximum distance: 2% from entry price
- Minimum risk:reward ratio: 2:1

### Trade Execution
- Minimum confidence threshold: 0.6 to execute any trade
- Cooldown period: 30 minutes between trades on same pair
- Only trade during normal market conditions (no extreme volatility)

### Position Types
- LONG: Buy asset expecting price increase
- SHORT: Sell asset expecting price decrease (futures only)
- HOLD: Maintain current position, no new trades
- CLOSE: Exit existing position

## STRATEGY GUIDELINES

### Primary Strategy: Trend-Following Momentum
- Enter positions aligned with the prevailing trend
- Use multi-timeframe confirmation (1h and 4h alignment)
- RSI momentum (>50 for longs, <50 for shorts) - NOT mean reversion
- Wait for pullbacks to moving averages for entries

### Secondary Strategy: Volatility Breakout
- Trade Bollinger Band squeezes followed by expansion
- Confirm breakouts with volume spike
- Set stops at the opposite band

### Avoid
- Mean reversion RSI strategies (oversold/overbought) - proven ineffective on crypto
- Trading against strong trends
- Trading during low-volume periods

## RISK MANAGEMENT

### Drawdown Protection
- If daily loss exceeds 3%: Reduce position sizes by 50%
- If total drawdown exceeds 10%: STOP all trading, return HOLD only
- After 5 consecutive losses: Return HOLD for cooldown period

### Correlation Awareness
- BTC/USDT and XRP/USDT are often correlated
- Avoid having maximum positions in both simultaneously
- XRP/BTC pair offers decorrelation opportunities

## OUTPUT FORMAT

You MUST respond with valid JSON matching this exact schema:

```json
{
  "action": "LONG | SHORT | HOLD | CLOSE",
  "symbol": "BTC/USDT | XRP/USDT | XRP/BTC",
  "confidence": <float between 0.0 and 1.0>,
  "position_size_pct": <float between 0.0 and 20.0>,
  "leverage": <integer between 1 and 3>,
  "entry_price": <float - current market price for entries>,
  "stop_loss": <float - mandatory for LONG/SHORT>,
  "take_profit": <float - must provide 2:1 minimum R:R>,
  "invalidation": "<condition that would invalidate this trade>",
  "reasoning": "<2-3 sentence justification for decision>"
}
```

### Output Rules
1. If confidence < 0.6, action MUST be "HOLD"
2. For LONG/SHORT, all price fields (entry, stop_loss, take_profit) are required
3. For HOLD/CLOSE, price fields can be null
4. reasoning must be concise (max 100 words)
5. stop_loss distance from entry must not exceed 2%
6. take_profit must be at least 2x the stop_loss distance

## BEHAVIORAL GUIDELINES

1. Be disciplined - follow the rules strictly
2. Be conservative - when uncertain, choose HOLD
3. Be consistent - don't change strategy based on recent losses
4. Be patient - wait for high-probability setups
5. Never chase prices after missing an entry
6. Accept small losses to avoid large ones"""
```

### 2.2 Conservative Profile System Prompt

```python
CONSERVATIVE_SYSTEM_PROMPT = """You are an expert quantitative trader with a CONSERVATIVE risk profile managing a cryptocurrency portfolio.

## CONSERVATIVE MODIFICATIONS

All rules from the standard trading system apply, with these stricter limits:

### Enhanced Risk Controls
- Maximum leverage: 2x (reduced from 3x)
- Risk per trade: 0.5% of portfolio (reduced from 1%)
- Maximum single position: 10% of portfolio (reduced from 20%)
- Minimum confidence threshold: 0.7 (increased from 0.6)
- Minimum risk:reward ratio: 3:1 (increased from 2:1)

### Trading Restrictions
- Only trade in clear trending markets (ADX > 25)
- Avoid trading first 2 hours after major news events
- Maximum 2 open positions at any time
- Prefer XRP/BTC pair for lower correlation

### Behavioral Emphasis
- "When in doubt, stay out"
- Quality over quantity - fewer trades, higher conviction
- Preserve capital above all else
- Missing opportunities is better than taking losses

{Include standard output format from TRADING_SYSTEM_PROMPT}"""
```

### 2.3 Aggressive Profile System Prompt

```python
AGGRESSIVE_SYSTEM_PROMPT = """You are an expert quantitative trader with an AGGRESSIVE risk profile seeking higher returns.

## AGGRESSIVE MODIFICATIONS

All rules from the standard trading system apply, with these adjusted limits:

### Adjusted Risk Parameters
- Maximum leverage: 3x (full limit)
- Risk per trade: 1.5% of portfolio (increased from 1%)
- Maximum single position: 25% of portfolio (increased from 20%)
- Minimum confidence threshold: 0.55 (reduced from 0.6)
- Minimum risk:reward ratio: 1.5:1 (reduced from 2:1)

### Trading Opportunities
- Trade both trending and ranging markets
- Consider counter-trend entries at extreme levels
- Allow up to 3 open positions simultaneously
- Active trading on all three pairs

### Important Warnings
- NEVER exceed 3x leverage regardless of confidence
- NEVER skip stop-loss placement
- ALWAYS respect the 10% maximum drawdown rule
- Higher aggression requires even stricter discipline

{Include standard output format from TRADING_SYSTEM_PROMPT}"""
```

---

## 3. User Prompt Templates

### 3.1 Market Context Builder

```python
def build_market_context_prompt(context: MarketContext) -> str:
    """Build the market data section of the user prompt."""

    return f"""## CURRENT MARKET DATA

### Prices (as of {context.timestamp})
| Pair | Price | 24h Change | 24h Volume |
|------|-------|------------|------------|
| BTC/USDT | ${context.prices['BTC/USDT']:,.2f} | {context.changes['BTC/USDT']:+.2f}% | ${context.volumes['BTC/USDT']:,.0f} |
| XRP/USDT | ${context.prices['XRP/USDT']:.4f} | {context.changes['XRP/USDT']:+.2f}% | ${context.volumes['XRP/USDT']:,.0f} |
| XRP/BTC | {context.prices['XRP/BTC']:.8f} | {context.changes['XRP/BTC']:+.2f}% | {context.volumes['XRP/BTC']:.2f} BTC |

### Technical Indicators (1H Timeframe)

**BTC/USDT:**
- Trend: EMA9={context.indicators['BTC/USDT']['ema9']:,.2f}, EMA21={context.indicators['BTC/USDT']['ema21']:,.2f}, EMA50={context.indicators['BTC/USDT']['ema50']:,.2f}
- Momentum: RSI14={context.indicators['BTC/USDT']['rsi14']:.1f}, MACD={context.indicators['BTC/USDT']['macd']:.2f}, Signal={context.indicators['BTC/USDT']['macd_signal']:.2f}
- Volatility: ATR14={context.indicators['BTC/USDT']['atr14']:.2f}, BB_Width={context.indicators['BTC/USDT']['bb_width']:.4f}
- Volume: OBV_Trend={context.indicators['BTC/USDT']['obv_trend']}, Vol_MA_Ratio={context.indicators['BTC/USDT']['vol_ma_ratio']:.2f}

**XRP/USDT:**
- Trend: EMA9={context.indicators['XRP/USDT']['ema9']:.4f}, EMA21={context.indicators['XRP/USDT']['ema21']:.4f}, EMA50={context.indicators['XRP/USDT']['ema50']:.4f}
- Momentum: RSI14={context.indicators['XRP/USDT']['rsi14']:.1f}, MACD={context.indicators['XRP/USDT']['macd']:.6f}, Signal={context.indicators['XRP/USDT']['macd_signal']:.6f}
- Volatility: ATR14={context.indicators['XRP/USDT']['atr14']:.4f}, BB_Width={context.indicators['XRP/USDT']['bb_width']:.4f}
- Volume: OBV_Trend={context.indicators['XRP/USDT']['obv_trend']}, Vol_MA_Ratio={context.indicators['XRP/USDT']['vol_ma_ratio']:.2f}

**XRP/BTC:**
- Trend: EMA9={context.indicators['XRP/BTC']['ema9']:.8f}, EMA21={context.indicators['XRP/BTC']['ema21']:.8f}
- Momentum: RSI14={context.indicators['XRP/BTC']['rsi14']:.1f}
- Note: Use for decorrelation from USD pairs

### Market Regime
- Current Regime: {context.regime.name}
- Regime Confidence: {context.regime.confidence:.2f}
- ADX: {context.regime.adx:.1f} ({'Trending' if context.regime.adx > 25 else 'Ranging'})
- Volatility State: {context.regime.volatility_state}"""
```

### 3.2 Portfolio Context Builder

```python
def build_portfolio_context_prompt(portfolio: Portfolio, positions: list[Position]) -> str:
    """Build the portfolio section of the user prompt."""

    # Calculate totals
    total_usdt = portfolio.total_value_usdt

    # Open positions section
    if positions:
        positions_text = "\n### Open Positions\n"
        positions_text += "| Symbol | Side | Size | Entry | Current | Unrealized P&L |\n"
        positions_text += "|--------|------|------|-------|---------|---------------|\n"
        for pos in positions:
            pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
            if pos.side == 'SHORT':
                pnl_pct = -pnl_pct
            positions_text += f"| {pos.symbol} | {pos.side} | {pos.size:.4f} | {pos.entry_price:.4f} | {pos.current_price:.4f} | {pnl_pct:+.2f}% |\n"
    else:
        positions_text = "\n### Open Positions\nNo open positions.\n"

    return f"""## PORTFOLIO STATUS

### Balances
| Asset | Amount | USD Value | Allocation |
|-------|--------|-----------|------------|
| USDT | {portfolio.usdt_balance:,.2f} | ${portfolio.usdt_balance:,.2f} | {(portfolio.usdt_balance/total_usdt)*100:.1f}% |
| BTC | {portfolio.btc_balance:.6f} | ${portfolio.btc_value_usdt:,.2f} | {(portfolio.btc_value_usdt/total_usdt)*100:.1f}% |
| XRP | {portfolio.xrp_balance:.2f} | ${portfolio.xrp_value_usdt:,.2f} | {(portfolio.xrp_value_usdt/total_usdt)*100:.1f}% |
| **Total** | - | **${total_usdt:,.2f}** | 100% |

### Performance
- Starting Value: ${portfolio.starting_value:,.2f}
- Current Value: ${total_usdt:,.2f}
- Total P&L: {((total_usdt - portfolio.starting_value) / portfolio.starting_value) * 100:+.2f}%
- Today's P&L: {portfolio.daily_pnl_pct:+.2f}%
- Current Drawdown: {portfolio.current_drawdown_pct:.2f}%
{positions_text}"""
```

### 3.3 Trade History Builder

```python
def build_trade_history_prompt(recent_trades: list[Trade]) -> str:
    """Build the recent trades section of the user prompt."""

    if not recent_trades:
        return """## RECENT TRADES
No recent trades in the last 24 hours."""

    trades_text = """## RECENT TRADES (Last 5)

| Time | Symbol | Action | Entry | Exit | P&L | Model |
|------|--------|--------|-------|------|-----|-------|
"""

    wins = 0
    losses = 0

    for trade in recent_trades[-5:]:
        pnl_str = f"{trade.pnl_pct:+.2f}%" if trade.exit_price else "Open"
        if trade.pnl_pct and trade.pnl_pct > 0:
            wins += 1
        elif trade.pnl_pct and trade.pnl_pct < 0:
            losses += 1
        trades_text += f"| {trade.timestamp.strftime('%H:%M')} | {trade.symbol} | {trade.action} | {trade.entry_price:.4f} | {trade.exit_price or '-'} | {pnl_str} | {trade.model} |\n"

    trades_text += f"\nRecent Win/Loss: {wins}W / {losses}L"

    return trades_text
```

### 3.4 Complete User Prompt Assembly

```python
def build_user_prompt(
    context: MarketContext,
    portfolio: Portfolio,
    positions: list[Position],
    recent_trades: list[Trade],
    specific_query: str = None
) -> str:
    """Assemble the complete user prompt."""

    sections = [
        build_market_context_prompt(context),
        build_portfolio_context_prompt(portfolio, positions),
        build_trade_history_prompt(recent_trades),
    ]

    # Add specific query if provided
    query_section = specific_query or """## DECISION REQUIRED

Based on the current market conditions, portfolio status, and recent performance, provide your trading decision.

Analyze:
1. Current market trend and momentum
2. Best opportunity among the three pairs
3. Appropriate position size given risk parameters
4. Clear entry, stop-loss, and take-profit levels

Respond with your decision in the required JSON format."""

    sections.append(query_section)

    return "\n\n".join(sections)
```

---

## 4. Output Schema

### 4.1 JSON Schema Definition

```python
TRADING_DECISION_SCHEMA = {
    "type": "object",
    "required": ["action", "symbol", "confidence", "reasoning"],
    "properties": {
        "action": {
            "type": "string",
            "enum": ["LONG", "SHORT", "HOLD", "CLOSE"]
        },
        "symbol": {
            "type": "string",
            "enum": ["BTC/USDT", "XRP/USDT", "XRP/BTC"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "position_size_pct": {
            "type": ["number", "null"],
            "minimum": 0.0,
            "maximum": 20.0
        },
        "leverage": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": 3
        },
        "entry_price": {
            "type": ["number", "null"]
        },
        "stop_loss": {
            "type": ["number", "null"]
        },
        "take_profit": {
            "type": ["number", "null"]
        },
        "invalidation": {
            "type": ["string", "null"]
        },
        "reasoning": {
            "type": "string",
            "maxLength": 500
        }
    },
    "allOf": [
        {
            "if": {
                "properties": {"action": {"enum": ["LONG", "SHORT"]}}
            },
            "then": {
                "required": ["position_size_pct", "leverage", "entry_price", "stop_loss", "take_profit"]
            }
        }
    ]
}
```

### 4.2 Output Parser

```python
import json
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradingDecision:
    action: str
    symbol: str
    confidence: float
    position_size_pct: Optional[float]
    leverage: Optional[int]
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    invalidation: Optional[str]
    reasoning: str
    model_used: str
    raw_response: str

def parse_trading_decision(response: str, model: str) -> TradingDecision:
    """Parse LLM response into TradingDecision object."""

    # Try to extract JSON from response
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        raise ValueError("No JSON object found in response")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # Validate required fields
    required = ['action', 'symbol', 'confidence', 'reasoning']
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate action
    if data['action'] not in ['LONG', 'SHORT', 'HOLD', 'CLOSE']:
        raise ValueError(f"Invalid action: {data['action']}")

    # Validate symbol
    if data['symbol'] not in ['BTC/USDT', 'XRP/USDT', 'XRP/BTC']:
        raise ValueError(f"Invalid symbol: {data['symbol']}")

    # Validate confidence
    if not 0 <= data['confidence'] <= 1:
        raise ValueError(f"Confidence out of range: {data['confidence']}")

    # Validate LONG/SHORT has required fields
    if data['action'] in ['LONG', 'SHORT']:
        trade_required = ['position_size_pct', 'leverage', 'entry_price', 'stop_loss', 'take_profit']
        for field in trade_required:
            if not data.get(field):
                raise ValueError(f"Missing field for {data['action']}: {field}")

        # Validate stop-loss distance (max 2%)
        entry = data['entry_price']
        stop = data['stop_loss']
        stop_distance = abs(entry - stop) / entry
        if stop_distance > 0.02:
            raise ValueError(f"Stop-loss too far: {stop_distance*100:.2f}% > 2%")

        # Validate risk:reward (min 2:1)
        tp = data['take_profit']
        reward_distance = abs(tp - entry)
        risk_distance = abs(entry - stop)
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        if rr_ratio < 2.0:
            raise ValueError(f"Risk:reward too low: {rr_ratio:.2f} < 2.0")

    return TradingDecision(
        action=data['action'],
        symbol=data['symbol'],
        confidence=data['confidence'],
        position_size_pct=data.get('position_size_pct'),
        leverage=data.get('leverage'),
        entry_price=data.get('entry_price'),
        stop_loss=data.get('stop_loss'),
        take_profit=data.get('take_profit'),
        invalidation=data.get('invalidation'),
        reasoning=data['reasoning'],
        model_used=model,
        raw_response=response
    )
```

---

## 5. Few-Shot Examples

### 5.1 Example: LONG Entry

```json
{
  "action": "LONG",
  "symbol": "BTC/USDT",
  "confidence": 0.75,
  "position_size_pct": 10.0,
  "leverage": 2,
  "entry_price": 95000.00,
  "stop_loss": 93150.00,
  "take_profit": 98700.00,
  "invalidation": "Close if price breaks below EMA21 on 4H timeframe",
  "reasoning": "BTC showing bullish momentum with RSI at 58 and price above all major EMAs. Bollinger squeeze resolved upward with volume confirmation. Entry at current price with stop below recent swing low."
}
```

### 5.2 Example: SHORT Entry

```json
{
  "action": "SHORT",
  "symbol": "XRP/USDT",
  "confidence": 0.68,
  "position_size_pct": 8.0,
  "leverage": 2,
  "entry_price": 2.3500,
  "stop_loss": 2.3970,
  "take_profit": 2.2560,
  "invalidation": "Close if XRP/BTC shows strength (breaks above 0.000024)",
  "reasoning": "XRP showing bearish divergence on RSI with price failing at resistance. BTC weakness likely to drag XRP lower. Risk:reward of 2:1 with stop above recent high."
}
```

### 5.3 Example: HOLD Decision

```json
{
  "action": "HOLD",
  "symbol": "BTC/USDT",
  "confidence": 0.45,
  "position_size_pct": null,
  "leverage": null,
  "entry_price": null,
  "stop_loss": null,
  "take_profit": null,
  "invalidation": null,
  "reasoning": "Market in consolidation range with ADX at 18 indicating no clear trend. RSI neutral at 52. Waiting for clearer directional signal before committing capital."
}
```

### 5.4 Example: CLOSE Position

```json
{
  "action": "CLOSE",
  "symbol": "XRP/BTC",
  "confidence": 0.82,
  "position_size_pct": null,
  "leverage": null,
  "entry_price": null,
  "stop_loss": null,
  "take_profit": null,
  "invalidation": null,
  "reasoning": "Take profit target reached on existing XRP/BTC long. RSI showing overbought at 72 with bearish divergence forming. Locking in 3.2% gain before potential reversal."
}
```

---

## 6. Error Handling

### 6.1 Response Validation Flow

```python
def validate_and_parse_response(
    response: str,
    model: str,
    max_retries: int = 2
) -> Optional[TradingDecision]:
    """Validate LLM response with retry logic."""

    for attempt in range(max_retries + 1):
        try:
            decision = parse_trading_decision(response, model)

            # Additional business rule validation
            if decision.action in ['LONG', 'SHORT']:
                if decision.confidence < 0.6:
                    # Force to HOLD if below threshold
                    decision = TradingDecision(
                        action='HOLD',
                        symbol=decision.symbol,
                        confidence=decision.confidence,
                        position_size_pct=None,
                        leverage=None,
                        entry_price=None,
                        stop_loss=None,
                        take_profit=None,
                        invalidation=None,
                        reasoning=f"Confidence {decision.confidence:.2f} below threshold. Original: {decision.reasoning}",
                        model_used=model,
                        raw_response=response
                    )

            return decision

        except ValueError as e:
            if attempt < max_retries:
                # Log and retry with clarification prompt
                logging.warning(f"Parse attempt {attempt + 1} failed: {e}")
                # In practice, you might re-query the LLM with error feedback
                continue
            else:
                logging.error(f"All parse attempts failed: {e}")
                return None

    return None
```

### 6.2 Fallback Behavior

```python
def get_safe_fallback_decision(symbol: str, model: str) -> TradingDecision:
    """Return safe HOLD decision when parsing fails."""
    return TradingDecision(
        action='HOLD',
        symbol=symbol,
        confidence=0.0,
        position_size_pct=None,
        leverage=None,
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        invalidation=None,
        reasoning="Fallback HOLD due to response parsing failure. Manual review recommended.",
        model_used=model,
        raw_response=""
    )
```

### 6.3 Timeout Handling

```python
import asyncio
from typing import Optional

async def query_llm_with_timeout(
    client: LLMClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int = 10
) -> Optional[str]:
    """Query LLM with timeout protection."""
    try:
        response = await asyncio.wait_for(
            client.query(model, system_prompt, user_prompt),
            timeout=timeout_seconds
        )
        return response
    except asyncio.TimeoutError:
        logging.error(f"LLM query timed out after {timeout_seconds}s for model {model}")
        return None
    except Exception as e:
        logging.error(f"LLM query failed for model {model}: {e}")
        return None
```

---

## Appendix: Prompt Testing Checklist

### Before Production

- [ ] Test with all supported models (Claude, GPT-4, Grok, Deepseek, Qwen)
- [ ] Verify JSON output is consistently parseable
- [ ] Confirm confidence scores are well-calibrated
- [ ] Validate stop-loss/take-profit calculations
- [ ] Test edge cases (empty portfolio, max drawdown, all positions open)
- [ ] Verify behavior during high volatility scenarios
- [ ] Test timeout and error handling paths
- [ ] Compare outputs across models for consistency

### Production Monitoring

- [ ] Log all prompts and responses to TimescaleDB
- [ ] Track parsing success rate per model
- [ ] Monitor confidence distribution
- [ ] Alert on repeated parsing failures
- [ ] Track decision latency per model

---

*Document Version: 1.0*
*Last Updated: December 2025*
