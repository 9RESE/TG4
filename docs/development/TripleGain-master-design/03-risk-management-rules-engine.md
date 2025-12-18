# TripleGain Risk Management Rules Engine

**Document Version**: 1.0
**Status**: Design Phase

---

## 1. Risk Management Overview

### 1.1 Design Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RISK MANAGEMENT PHILOSOPHY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRINCIPLE 1: RULES-BASED, NOT LLM-DEPENDENT                               │
│  • Risk management is deterministic                                         │
│  • No LLM can override risk rules                                          │
│  • Sub-10ms execution for all risk checks                                  │
│                                                                             │
│  PRINCIPLE 2: DEFENSE IN DEPTH                                              │
│  • Multiple independent safety layers                                       │
│  • Each layer can independently halt trading                               │
│  • Graceful degradation, not catastrophic failure                          │
│                                                                             │
│  PRINCIPLE 3: CONSERVATIVE BY DEFAULT                                       │
│  • When uncertain, reduce exposure                                          │
│  • Errors default to rejection, not approval                               │
│  • Protection of capital is paramount                                       │
│                                                                             │
│  PRINCIPLE 4: FULL AUDITABILITY                                             │
│  • Every risk decision logged with reasoning                               │
│  • Complete audit trail for all modifications                              │
│  • Real-time monitoring and alerting                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Risk Layers Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RISK MANAGEMENT LAYERS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LAYER 1: PRE-TRADE VALIDATION                                              │
│  ├── Position size limits                                                   │
│  ├── Leverage limits                                                        │
│  ├── Confidence thresholds                                                  │
│  ├── Required stop-loss verification                                        │
│  └── Available margin check                                                 │
│                                                                             │
│  LAYER 2: PORTFOLIO RISK                                                    │
│  ├── Total exposure limits                                                  │
│  ├── Correlated position limits                                             │
│  ├── Concentration risk                                                     │
│  └── Margin utilization limits                                              │
│                                                                             │
│  LAYER 3: DRAWDOWN PROTECTION                                               │
│  ├── Daily loss limits                                                      │
│  ├── Weekly loss limits                                                     │
│  ├── Maximum drawdown circuit breaker                                       │
│  └── Consecutive loss protection                                            │
│                                                                             │
│  LAYER 4: POSITION MANAGEMENT                                               │
│  ├── Stop-loss monitoring                                                   │
│  ├── Take-profit monitoring                                                 │
│  ├── Time-based exit rules                                                  │
│  └── Trailing stop management                                               │
│                                                                             │
│  LAYER 5: SYSTEM SAFEGUARDS                                                 │
│  ├── Emergency stop (kill switch)                                           │
│  ├── Exchange connectivity checks                                           │
│  ├── Data quality validation                                                │
│  └── Anomaly detection                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Position Sizing Rules

### 2.1 Base Position Sizing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POSITION SIZING CALCULATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRIMARY METHOD: ATR-Based Position Sizing                                  │
│                                                                             │
│  position_size = (equity × risk_per_trade) / (entry - stop_loss)           │
│                                                                             │
│  Where:                                                                     │
│  • equity = current portfolio equity (USDT value)                          │
│  • risk_per_trade = max % of equity to risk (default: 1%)                  │
│  • entry = planned entry price                                             │
│  • stop_loss = stop-loss price                                             │
│                                                                             │
│  EXAMPLE:                                                                   │
│  Equity: $2,100                                                            │
│  Risk per trade: 1% = $21                                                  │
│  Entry: $45,000 (BTC)                                                      │
│  Stop-loss: $44,100 (2% below entry)                                       │
│  Stop distance: $900                                                        │
│                                                                             │
│  Position size = $21 / $900 = 0.0233 BTC                                   │
│  Position value = 0.0233 × $45,000 = $1,050                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Confidence-Adjusted Sizing

```python
CONFIDENCE_MULTIPLIERS = {
    "very_high": {    # confidence >= 0.85
        "multiplier": 1.0,
        "max_position_pct": 20
    },
    "high": {          # 0.75 <= confidence < 0.85
        "multiplier": 0.75,
        "max_position_pct": 15
    },
    "medium": {        # 0.65 <= confidence < 0.75
        "multiplier": 0.50,
        "max_position_pct": 10
    },
    "low": {           # 0.60 <= confidence < 0.65
        "multiplier": 0.25,
        "max_position_pct": 5
    },
    "insufficient": {  # confidence < 0.60
        "multiplier": 0.0,  # NO TRADE
        "max_position_pct": 0
    }
}

def calculate_confidence_adjusted_size(base_size: float, confidence: float) -> float:
    """
    Adjust position size based on signal confidence.
    """
    if confidence < 0.60:
        return 0.0  # No trade

    for level, params in CONFIDENCE_MULTIPLIERS.items():
        if confidence >= params.get("min_confidence", 0):
            return min(
                base_size * params["multiplier"],
                portfolio_equity * (params["max_position_pct"] / 100)
            )

    return 0.0
```

### 2.3 Regime-Adjusted Sizing

| Regime | Position Size Multiplier | Max Leverage | Stop Distance Multiplier |
|--------|-------------------------|--------------|--------------------------|
| Trending Bull | 1.0 | 5x | 1.0 |
| Trending Bear | 0.8 | 3x | 1.2 |
| Ranging | 0.6 | 2x | 1.5 |
| Volatile | 0.4 | 2x | 2.0 |
| Quiet | 0.5 | 3x | 0.8 |

```python
def apply_regime_adjustment(
    base_size: float,
    base_leverage: int,
    regime: str
) -> tuple[float, int]:
    """
    Adjust position size and leverage based on market regime.
    """
    adjustments = {
        "trending_bull": {"size_mult": 1.0, "max_leverage": 5},
        "trending_bear": {"size_mult": 0.8, "max_leverage": 3},
        "ranging": {"size_mult": 0.6, "max_leverage": 2},
        "volatile": {"size_mult": 0.4, "max_leverage": 2},
        "quiet": {"size_mult": 0.5, "max_leverage": 3},
    }

    adj = adjustments.get(regime, {"size_mult": 0.5, "max_leverage": 2})

    adjusted_size = base_size * adj["size_mult"]
    adjusted_leverage = min(base_leverage, adj["max_leverage"])

    return adjusted_size, adjusted_leverage
```

---

## 3. Stop-Loss Rules

### 3.1 Stop-Loss Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STOP-LOSS RULES                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RULE 1: MANDATORY STOP-LOSS                                                │
│  • Every trade MUST have a stop-loss                                        │
│  • Trades without stop-loss are REJECTED                                    │
│  • Stop-loss must be set BEFORE order submission                            │
│                                                                             │
│  RULE 2: MAXIMUM STOP DISTANCE                                              │
│  • Stop-loss cannot exceed max_stop_distance_pct of entry                   │
│  • Default: 3% for BTC, 5% for XRP                                         │
│  • Regime-adjusted (volatile regime allows wider stops)                     │
│                                                                             │
│  RULE 3: MINIMUM STOP DISTANCE                                              │
│  • Stop-loss must be at least min_stop_distance_pct from entry             │
│  • Default: 0.5% (to avoid noise triggers)                                  │
│  • Must exceed typical spread + slippage                                    │
│                                                                             │
│  RULE 4: ATR-BASED STOPS                                                    │
│  • Recommended: stop = entry - (ATR_14 × multiplier)                       │
│  • Default multiplier: 2.0                                                 │
│  • Regime-adjusted multiplier (see table below)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 ATR-Based Stop Calculation

```python
def calculate_atr_stop(
    entry_price: float,
    side: str,  # "long" or "short"
    atr_14: float,
    regime: str
) -> float:
    """
    Calculate stop-loss price using ATR.
    """
    ATR_MULTIPLIERS = {
        "trending_bull": 2.0,
        "trending_bear": 2.0,
        "ranging": 2.5,
        "volatile": 3.0,
        "quiet": 1.5,
    }

    multiplier = ATR_MULTIPLIERS.get(regime, 2.0)
    stop_distance = atr_14 * multiplier

    if side == "long":
        stop_price = entry_price - stop_distance
    else:  # short
        stop_price = entry_price + stop_distance

    return stop_price
```

### 3.3 Stop-Loss Types

| Stop Type | Description | Use Case |
|-----------|-------------|----------|
| **Fixed** | Static price level | Default for all trades |
| **Trailing** | Follows price at fixed distance | Trend-following trades |
| **ATR Trailing** | Trails by ATR multiple | Volatile markets |
| **Break-Even** | Move to entry after X% profit | Risk elimination |
| **Time-Based** | Close after N hours regardless | Prevent stale positions |

```python
TRAILING_STOP_CONFIG = {
    "activation_profit_pct": 1.0,  # Activate trailing after 1% profit
    "trail_distance_pct": 1.5,      # Trail by 1.5% from peak
    "trail_distance_atr": 1.5,      # OR trail by 1.5x ATR (regime-adjusted)
    "use_atr": True,                # Prefer ATR-based trailing
    "update_frequency_seconds": 60  # Check for trail update every minute
}
```

---

## 4. Leverage Limits

### 4.1 Maximum Leverage Rules

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LEVERAGE LIMITS                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ABSOLUTE MAXIMUM: 5x                                                        │
│  • System hard limit, cannot be overridden                                  │
│  • Applies regardless of Kraken's available leverage                        │
│                                                                             │
│  REGIME-ADJUSTED LIMITS:                                                    │
│  ┌───────────────────┬─────────────────┐                                    │
│  │ Regime            │ Max Leverage    │                                    │
│  ├───────────────────┼─────────────────┤                                    │
│  │ Trending Bull     │ 5x              │                                    │
│  │ Trending Bear     │ 3x              │                                    │
│  │ Ranging           │ 2x              │                                    │
│  │ Volatile          │ 2x              │                                    │
│  │ Quiet             │ 3x              │                                    │
│  │ Unknown/Error     │ 1x              │                                    │
│  └───────────────────┴─────────────────┘                                    │
│                                                                             │
│  DRAWDOWN-ADJUSTED LIMITS:                                                  │
│  ┌───────────────────┬─────────────────┐                                    │
│  │ Current Drawdown  │ Max Leverage    │                                    │
│  ├───────────────────┼─────────────────┤                                    │
│  │ < 5%              │ Regime limit    │                                    │
│  │ 5% - 10%          │ min(Regime, 3x) │                                    │
│  │ 10% - 15%         │ 2x              │                                    │
│  │ > 15%             │ 1x              │                                    │
│  └───────────────────┴─────────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Leverage Calculation

```python
def get_max_allowed_leverage(
    regime: str,
    current_drawdown_pct: float,
    consecutive_losses: int
) -> int:
    """
    Calculate maximum allowed leverage based on current conditions.
    """
    # Base limits by regime
    REGIME_LIMITS = {
        "trending_bull": 5,
        "trending_bear": 3,
        "ranging": 2,
        "volatile": 2,
        "quiet": 3,
        "unknown": 1
    }

    # Start with regime limit
    max_leverage = REGIME_LIMITS.get(regime, 1)

    # Reduce for drawdown
    if current_drawdown_pct >= 15:
        max_leverage = min(max_leverage, 1)
    elif current_drawdown_pct >= 10:
        max_leverage = min(max_leverage, 2)
    elif current_drawdown_pct >= 5:
        max_leverage = min(max_leverage, 3)

    # Reduce for consecutive losses
    if consecutive_losses >= 5:
        max_leverage = 1
    elif consecutive_losses >= 3:
        max_leverage = min(max_leverage, 2)

    return max_leverage
```

---

## 5. Drawdown Circuit Breakers

### 5.1 Circuit Breaker Definitions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CIRCUIT BREAKER RULES                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LEVEL 1: DAILY LOSS LIMIT                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Trigger: Daily realized + unrealized loss > 5% of equity            │   │
│  │ Action: Halt all new trades                                          │   │
│  │ Existing positions: Tighten stops to 1% from current price          │   │
│  │ Duration: Until next trading day (UTC midnight)                      │   │
│  │ Resume condition: Automatic at daily reset                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LEVEL 2: WEEKLY LOSS LIMIT                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Trigger: Weekly loss > 10% of equity                                 │   │
│  │ Action: Halt all new trades, reduce position sizes 50%               │   │
│  │ Existing positions: Close 50% of all positions                       │   │
│  │ Duration: Until Monday UTC midnight                                  │   │
│  │ Resume condition: Manual review required                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LEVEL 3: MAXIMUM DRAWDOWN                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Trigger: Total drawdown from peak > 20% of peak equity              │   │
│  │ Action: Close ALL positions, halt all trading                        │   │
│  │ Leverage: Reduce to 1x when trading resumes                          │   │
│  │ Duration: Until manual intervention                                   │   │
│  │ Resume condition: User must explicitly re-enable trading             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LEVEL 4: CONSECUTIVE LOSSES                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Trigger: 5 consecutive losing trades                                 │   │
│  │ Action: 30-minute cooldown, reduce next trade size by 50%           │   │
│  │ Duration: 30 minutes minimum                                         │   │
│  │ Resume condition: Automatic after cooldown                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Circuit Breaker Implementation

```python
@dataclass
class CircuitBreakerState:
    daily_loss_pct: float = 0.0
    weekly_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    current_equity: float = 0.0

    breakers_triggered: list = field(default_factory=list)
    trading_halted: bool = False
    halt_reason: str = ""
    resume_at: datetime = None


CIRCUIT_BREAKER_CONFIG = {
    "daily_loss": {
        "threshold_pct": 5.0,
        "action": "halt_new_trades",
        "duration": "until_daily_reset",
        "severity": 1
    },
    "weekly_loss": {
        "threshold_pct": 10.0,
        "action": "halt_and_reduce",
        "reduction_pct": 50,
        "duration": "until_weekly_reset",
        "severity": 2
    },
    "max_drawdown": {
        "threshold_pct": 20.0,
        "action": "close_all_halt",
        "duration": "until_manual_reset",
        "severity": 3
    },
    "consecutive_losses": {
        "threshold_count": 5,
        "action": "cooldown",
        "cooldown_minutes": 30,
        "size_reduction_pct": 50,
        "severity": 1
    }
}


def check_circuit_breakers(state: CircuitBreakerState) -> dict:
    """
    Check all circuit breakers and return actions needed.
    """
    actions = {
        "halt_trading": False,
        "close_positions": False,
        "reduce_positions_pct": 0,
        "reduce_new_trade_size_pct": 0,
        "cooldown_minutes": 0,
        "triggered_breakers": [],
        "severity": 0
    }

    # Check daily loss
    if state.daily_loss_pct >= CIRCUIT_BREAKER_CONFIG["daily_loss"]["threshold_pct"]:
        actions["halt_trading"] = True
        actions["triggered_breakers"].append("daily_loss")
        actions["severity"] = max(actions["severity"], 1)

    # Check weekly loss
    if state.weekly_loss_pct >= CIRCUIT_BREAKER_CONFIG["weekly_loss"]["threshold_pct"]:
        actions["halt_trading"] = True
        actions["reduce_positions_pct"] = 50
        actions["triggered_breakers"].append("weekly_loss")
        actions["severity"] = max(actions["severity"], 2)

    # Check max drawdown
    if state.max_drawdown_pct >= CIRCUIT_BREAKER_CONFIG["max_drawdown"]["threshold_pct"]:
        actions["halt_trading"] = True
        actions["close_positions"] = True
        actions["triggered_breakers"].append("max_drawdown")
        actions["severity"] = max(actions["severity"], 3)

    # Check consecutive losses
    if state.consecutive_losses >= CIRCUIT_BREAKER_CONFIG["consecutive_losses"]["threshold_count"]:
        actions["cooldown_minutes"] = 30
        actions["reduce_new_trade_size_pct"] = 50
        actions["triggered_breakers"].append("consecutive_losses")
        actions["severity"] = max(actions["severity"], 1)

    return actions
```

---

## 6. Confidence Thresholds

### 6.1 Minimum Confidence Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONFIDENCE THRESHOLD RULES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ABSOLUTE MINIMUM: 0.60                                                      │
│  • No trade executed if confidence < 0.60                                   │
│  • Applies to ALL trading decisions                                         │
│                                                                             │
│  TIERED CONFIDENCE THRESHOLDS:                                              │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ Confidence │ Action                │ Position Size  │ Leverage        │ │
│  ├────────────┼───────────────────────┼────────────────┼─────────────────┤ │
│  │ < 0.60     │ NO TRADE              │ 0%             │ N/A             │ │
│  │ 0.60-0.65  │ Small position        │ 25% of base    │ Max 2x          │ │
│  │ 0.65-0.75  │ Medium position       │ 50% of base    │ Max 3x          │ │
│  │ 0.75-0.85  │ Standard position     │ 75% of base    │ Regime limit    │ │
│  │ > 0.85     │ Full position         │ 100% of base   │ Regime limit    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  SPECIAL CASES:                                                             │
│                                                                             │
│  • Disagreement between agents (TA vs Sentiment):                           │
│    - If confidence difference > 0.2, reduce to lower confidence            │
│    - If opposite signals, confidence capped at 0.5 (no trade)              │
│                                                                             │
│  • After consecutive losses:                                                │
│    - Minimum confidence increased to 0.70 after 3 losses                   │
│    - Minimum confidence increased to 0.80 after 5 losses                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Confidence Validation

```python
def validate_confidence(
    confidence: float,
    agent_confidences: dict,
    consecutive_losses: int
) -> tuple[bool, str]:
    """
    Validate if confidence meets requirements for trading.

    Returns:
        tuple: (is_valid, rejection_reason)
    """
    # Base minimum
    min_confidence = 0.60

    # Adjust for consecutive losses
    if consecutive_losses >= 5:
        min_confidence = 0.80
    elif consecutive_losses >= 3:
        min_confidence = 0.70

    # Check absolute minimum
    if confidence < min_confidence:
        return False, f"Confidence {confidence} below minimum {min_confidence}"

    # Check agent agreement
    ta_confidence = agent_confidences.get("technical_analysis", 0)
    sentiment_confidence = agent_confidences.get("sentiment", 0)

    if ta_confidence > 0 and sentiment_confidence > 0:
        # Check for significant disagreement
        confidence_diff = abs(ta_confidence - sentiment_confidence)
        if confidence_diff > 0.2:
            effective_confidence = min(ta_confidence, sentiment_confidence)
            if effective_confidence < min_confidence:
                return False, f"Agent disagreement: TA={ta_confidence}, Sent={sentiment_confidence}"

    return True, "OK"
```

---

## 7. Cooldown Periods

### 7.1 Cooldown Rules

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COOLDOWN PERIOD RULES                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  POST-TRADE COOLDOWN:                                                       │
│  • After opening a position: 5-minute cooldown before next trade            │
│  • Purpose: Prevent overtrading, allow price action to develop              │
│                                                                             │
│  POST-LOSS COOLDOWN:                                                        │
│  • After a losing trade: 10-minute cooldown                                 │
│  • After 3 consecutive losses: 30-minute cooldown                           │
│  • After 5 consecutive losses: 60-minute cooldown                           │
│                                                                             │
│  VOLATILITY COOLDOWN:                                                       │
│  • After stop-loss triggered by spike: 15-minute cooldown                   │
│  • After unusual price movement (>3% in 5 min): 10-minute cooldown          │
│                                                                             │
│  REBALANCING COOLDOWN:                                                      │
│  • After portfolio rebalance: 30-minute cooldown before trading signals     │
│  • Purpose: Allow rebalance orders to fill                                  │
│                                                                             │
│  DAILY TRADING WINDOW:                                                      │
│  • Cooldown during low-liquidity hours (optional)                           │
│  • Default: No trading 22:00-02:00 UTC (weekend rollover period)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Cooldown Implementation

```python
@dataclass
class CooldownManager:
    cooldowns: dict = field(default_factory=dict)

    def add_cooldown(self, cooldown_type: str, duration_minutes: int):
        self.cooldowns[cooldown_type] = {
            "expires_at": datetime.utcnow() + timedelta(minutes=duration_minutes),
            "reason": cooldown_type
        }

    def is_in_cooldown(self) -> tuple[bool, str]:
        """Check if any cooldown is active."""
        now = datetime.utcnow()
        for cooldown_type, data in self.cooldowns.items():
            if data["expires_at"] > now:
                remaining = (data["expires_at"] - now).total_seconds() / 60
                return True, f"{cooldown_type}: {remaining:.1f} min remaining"

        return False, ""

    def clear_expired(self):
        """Remove expired cooldowns."""
        now = datetime.utcnow()
        self.cooldowns = {
            k: v for k, v in self.cooldowns.items()
            if v["expires_at"] > now
        }


COOLDOWN_TRIGGERS = {
    "post_trade": {"duration_minutes": 5},
    "post_loss": {"duration_minutes": 10},
    "consecutive_loss_3": {"duration_minutes": 30},
    "consecutive_loss_5": {"duration_minutes": 60},
    "volatility_spike": {"duration_minutes": 15},
    "unusual_movement": {"duration_minutes": 10},
    "post_rebalance": {"duration_minutes": 30}
}
```

---

## 8. Hodl Bag Allocation

### 8.1 Hodl Bag Rules

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HODL BAG ALLOCATION RULES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ALLOCATION TRIGGER:                                                        │
│  • 10% of realized profit allocated to Hodl Bags                           │
│  • Triggered on: Trade profit, rebalancing profit, positive weekly P&L      │
│                                                                             │
│  ALLOCATION DISTRIBUTION:                                                   │
│  • BTC Hodl Bag: 33.33% of allocation                                      │
│  • XRP Hodl Bag: 33.33% of allocation                                      │
│  • USDT Hodl Bag: 33.33% of allocation                                     │
│                                                                             │
│  MINIMUM ALLOCATION:                                                        │
│  • Minimum allocation: $10 per trigger                                     │
│  • If profit < $100, accumulate until threshold                            │
│                                                                             │
│  HODL BAG PROTECTION:                                                       │
│  • Hodl Bag positions are NEVER sold for trading                           │
│  • Hodl Bag is excluded from 33/33/33 rebalancing                          │
│  • Hodl Bag can only decrease via explicit user withdrawal                 │
│                                                                             │
│  EXECUTION:                                                                 │
│  • Hodl allocation uses limit orders (best execution)                      │
│  • Executed during low-volatility periods                                  │
│  • DCA over 24 hours if allocation > $100                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Hodl Bag Calculation

```python
@dataclass
class HodlBagState:
    btc_amount: float = 0.0
    xrp_amount: float = 0.0
    usdt_amount: float = 0.0
    pending_allocation_usd: float = 0.0  # Accumulated but not yet allocated

    def calculate_allocation(self, realized_profit_usd: float) -> dict:
        """
        Calculate Hodl Bag allocation from realized profit.
        """
        HODL_PCT = 0.10  # 10% of profits
        MIN_ALLOCATION_USD = 10.0
        BTC_SPLIT = 0.3333
        XRP_SPLIT = 0.3333
        USDT_SPLIT = 0.3334  # Slightly higher to account for rounding

        allocation_usd = realized_profit_usd * HODL_PCT

        # Add to pending
        self.pending_allocation_usd += allocation_usd

        # Check if threshold reached
        if self.pending_allocation_usd < MIN_ALLOCATION_USD:
            return {
                "execute": False,
                "pending_usd": self.pending_allocation_usd,
                "reason": f"Below minimum (${MIN_ALLOCATION_USD})"
            }

        # Calculate split
        total_allocation = self.pending_allocation_usd
        btc_allocation_usd = total_allocation * BTC_SPLIT
        xrp_allocation_usd = total_allocation * XRP_SPLIT
        usdt_allocation_usd = total_allocation * USDT_SPLIT

        # Reset pending
        self.pending_allocation_usd = 0.0

        return {
            "execute": True,
            "btc_allocation_usd": btc_allocation_usd,
            "xrp_allocation_usd": xrp_allocation_usd,
            "usdt_allocation_usd": usdt_allocation_usd,
            "total_usd": total_allocation,
            "execution_strategy": "dca_24h" if total_allocation > 100 else "immediate"
        }
```

---

## 9. Pre-Trade Validation Checklist

### 9.1 Complete Validation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRE-TRADE VALIDATION CHECKLIST                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  □ 1. STOP-LOSS SET                                                         │
│       ├── Stop-loss price specified                                         │
│       ├── Within allowed distance (0.5% - 5%)                              │
│       └── Risk/reward ratio >= 1.5:1                                       │
│                                                                             │
│  □ 2. CONFIDENCE CHECK                                                      │
│       ├── Confidence >= minimum threshold (0.60)                            │
│       ├── No significant agent disagreement                                 │
│       └── Adjusted for consecutive losses                                   │
│                                                                             │
│  □ 3. POSITION SIZE CHECK                                                   │
│       ├── Within max position % of equity                                   │
│       ├── Risk per trade <= 2% of equity                                   │
│       └── Regime-adjusted size applied                                      │
│                                                                             │
│  □ 4. LEVERAGE CHECK                                                        │
│       ├── Within regime-adjusted limit                                      │
│       ├── Within drawdown-adjusted limit                                    │
│       └── Sufficient margin available                                       │
│                                                                             │
│  □ 5. PORTFOLIO EXPOSURE CHECK                                              │
│       ├── Total exposure < max exposure limit                               │
│       ├── Correlated position limit not exceeded                            │
│       └── Margin utilization < limit                                        │
│                                                                             │
│  □ 6. COOLDOWN CHECK                                                        │
│       ├── No active cooldown periods                                        │
│       └── Outside restricted trading hours                                  │
│                                                                             │
│  □ 7. CIRCUIT BREAKER CHECK                                                 │
│       ├── No circuit breakers triggered                                     │
│       ├── Daily loss limit not reached                                      │
│       └── Not in halt state                                                 │
│                                                                             │
│  □ 8. SYSTEM HEALTH CHECK                                                   │
│       ├── Exchange connection active                                        │
│       ├── Data feeds current (< 1 min old)                                 │
│       └── No system errors                                                  │
│                                                                             │
│  ══════════════════════════════════════════════════════════════════════════│
│  ALL CHECKS PASS → APPROVE TRADE                                            │
│  ANY CHECK FAILS → REJECT TRADE (with specific reason)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Validation Implementation

```python
@dataclass
class TradeValidationResult:
    approved: bool
    modifications: dict = field(default_factory=dict)
    rejections: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    risk_metrics: dict = field(default_factory=dict)


def validate_trade(
    proposed_trade: dict,
    portfolio_state: dict,
    risk_state: dict,
    market_state: dict
) -> TradeValidationResult:
    """
    Comprehensive pre-trade validation.
    """
    result = TradeValidationResult(approved=True)

    # 1. Stop-loss check
    if not proposed_trade.get("stop_loss"):
        result.approved = False
        result.rejections.append("STOP_LOSS_REQUIRED: No stop-loss specified")
        return result

    # 2. Confidence check
    confidence = proposed_trade.get("confidence", 0)
    min_confidence = get_adjusted_min_confidence(risk_state["consecutive_losses"])
    if confidence < min_confidence:
        result.approved = False
        result.rejections.append(
            f"CONFIDENCE_TOO_LOW: {confidence} < {min_confidence}"
        )
        return result

    # 3. Position size check
    max_size = calculate_max_position_size(portfolio_state, market_state["regime"])
    if proposed_trade["size_usd"] > max_size:
        result.modifications["size_usd"] = max_size
        result.warnings.append(
            f"SIZE_REDUCED: {proposed_trade['size_usd']} -> {max_size}"
        )

    # 4. Leverage check
    max_leverage = get_max_allowed_leverage(
        market_state["regime"],
        risk_state["drawdown_pct"],
        risk_state["consecutive_losses"]
    )
    if proposed_trade["leverage"] > max_leverage:
        result.modifications["leverage"] = max_leverage
        result.warnings.append(
            f"LEVERAGE_REDUCED: {proposed_trade['leverage']} -> {max_leverage}"
        )

    # 5. Portfolio exposure check
    if portfolio_state["total_exposure_pct"] + proposed_trade["exposure_pct"] > 80:
        result.approved = False
        result.rejections.append("EXPOSURE_LIMIT: Would exceed 80% portfolio exposure")
        return result

    # 6. Cooldown check
    in_cooldown, cooldown_reason = check_cooldown()
    if in_cooldown:
        result.approved = False
        result.rejections.append(f"IN_COOLDOWN: {cooldown_reason}")
        return result

    # 7. Circuit breaker check
    breaker_actions = check_circuit_breakers(risk_state)
    if breaker_actions["halt_trading"]:
        result.approved = False
        result.rejections.append(
            f"CIRCUIT_BREAKER: {breaker_actions['triggered_breakers']}"
        )
        return result

    # 8. System health check
    if not market_state["exchange_connected"]:
        result.approved = False
        result.rejections.append("EXCHANGE_DISCONNECTED")
        return result

    # Calculate final risk metrics
    result.risk_metrics = calculate_trade_risk_metrics(
        proposed_trade, portfolio_state, result.modifications
    )

    return result
```

---

## 10. Risk Monitoring Dashboard Metrics

### 10.1 Real-Time Metrics

| Metric | Update Frequency | Alert Threshold |
|--------|------------------|-----------------|
| Portfolio P&L (daily) | Every tick | Loss > 3% |
| Total Exposure % | Every tick | > 70% |
| Margin Utilization % | Every tick | > 50% |
| Current Drawdown % | Every tick | > 10% |
| Open Positions Count | Every tick | > 5 |
| Avg Position Duration | Every minute | > 48 hours |
| Unrealized P&L | Every tick | Loss > 5% |
| Stop-Loss Proximity | Every tick | < 0.5% |

### 10.2 Historical Metrics

| Metric | Calculation Period | Target |
|--------|-------------------|--------|
| Win Rate | 7 days rolling | > 50% |
| Profit Factor | 7 days rolling | > 1.5 |
| Sharpe Ratio | 30 days rolling | > 1.5 |
| Max Drawdown | All time | < 20% |
| Avg Trade Duration | 7 days | 4-24 hours |
| Largest Winner | All time | Tracking |
| Largest Loser | All time | < 3% of equity |

---

*Document Version 1.0 - December 2025*
