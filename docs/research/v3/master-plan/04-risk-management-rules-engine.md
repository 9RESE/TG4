# Risk Management Rules Engine Specification

**Version:** 1.0
**Date:** December 2025
**Status:** Design Phase

---

## Overview

This document specifies the rules engine for the Risk Management Agent in the TripleGain system. The rules are designed based on insights from Alpha Arena research, BTC/USDT trading analysis, and established quantitative risk management principles.

---

## Table of Contents

1. [Risk Philosophy](#1-risk-philosophy)
2. [Position Sizing Rules](#2-position-sizing-rules)
3. [Stop-Loss Rules](#3-stop-loss-rules)
4. [Drawdown Protection](#4-drawdown-protection)
5. [Leverage Rules](#5-leverage-rules)
6. [Correlation Management](#6-correlation-management)
7. [Cooldown Rules](#7-cooldown-rules)
8. [Rule Engine Implementation](#8-rule-engine-implementation)

---

## 1. Risk Philosophy

### 1.1 Core Principles

| Principle | Description |
|-----------|-------------|
| **Capital Preservation** | Protecting capital is more important than maximizing gains |
| **Risk First** | Every trade must have defined risk before entry |
| **Consistency** | Rules apply uniformly, no exceptions based on conviction |
| **Adaptability** | Rules adjust based on market conditions |
| **Transparency** | All decisions logged with reasoning |

### 1.2 Risk Budget

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RISK BUDGET ALLOCATION                            │
└─────────────────────────────────────────────────────────────────────────────┘

Total Portfolio: $2,100 (Starting)

Risk Allocation:
├── Per Trade Risk: 1% = $21 max loss per trade
├── Daily Risk Budget: 3% = $63 max daily loss
├── Max Drawdown Threshold: 10% = $210 drawdown triggers pause
└── Reserve: 20% = $420 always in cash/stables

Position Limits:
├── Single Position Max: 20% of portfolio = $420
├── Total Exposure Max: 60% of portfolio = $1,260
└── Leverage-Adjusted Exposure: 100% max (3x on 33%)

Per-Pair Allocation:
├── BTC/USDT: Max 25% per trade
├── XRP/USDT: Max 20% per trade
└── XRP/BTC: Max 15% per trade (lower liquidity)
```

---

## 2. Position Sizing Rules

### 2.1 ATR-Based Position Sizing

```python
@dataclass
class PositionSizeRule:
    """Calculate position size based on ATR and risk parameters."""

    risk_per_trade_pct: float = 0.01  # 1% of portfolio
    atr_multiplier: float = 2.0       # Stop at 2x ATR
    max_position_pct: float = 0.20    # Max 20% of portfolio

    def calculate(
        self,
        portfolio_value: float,
        entry_price: float,
        atr: float,
        confidence: float
    ) -> PositionSize:
        """
        Calculate position size using ATR-based risk.

        Formula:
        risk_amount = portfolio_value * risk_per_trade_pct
        stop_distance = atr * atr_multiplier
        position_size = risk_amount / stop_distance
        """

        # Base risk amount
        risk_amount = portfolio_value * self.risk_per_trade_pct

        # Stop distance based on ATR
        stop_distance = atr * self.atr_multiplier
        stop_distance_pct = stop_distance / entry_price

        # Ensure stop doesn't exceed 2%
        if stop_distance_pct > 0.02:
            stop_distance_pct = 0.02
            stop_distance = entry_price * stop_distance_pct

        # Calculate base position size (in quote currency)
        position_size_quote = risk_amount / stop_distance_pct

        # Apply confidence scaling
        # Confidence 0.6-0.7: 50% size
        # Confidence 0.7-0.85: 75% size
        # Confidence 0.85+: 100% size
        confidence_multiplier = self._get_confidence_multiplier(confidence)
        position_size_quote *= confidence_multiplier

        # Apply maximum position limit
        max_size = portfolio_value * self.max_position_pct
        position_size_quote = min(position_size_quote, max_size)

        # Calculate position in base currency
        position_size_base = position_size_quote / entry_price

        return PositionSize(
            size_base=position_size_base,
            size_quote=position_size_quote,
            stop_distance=stop_distance,
            stop_distance_pct=stop_distance_pct,
            risk_amount=position_size_quote * stop_distance_pct,
            confidence_multiplier=confidence_multiplier
        )

    def _get_confidence_multiplier(self, confidence: float) -> float:
        if confidence >= 0.85:
            return 1.0
        elif confidence >= 0.70:
            return 0.75
        elif confidence >= 0.60:
            return 0.50
        else:
            return 0.0  # No trade below 0.6

@dataclass
class PositionSize:
    size_base: float
    size_quote: float
    stop_distance: float
    stop_distance_pct: float
    risk_amount: float
    confidence_multiplier: float
```

### 2.2 Kelly Criterion (Fractional)

```python
@dataclass
class KellyRule:
    """Fractional Kelly criterion for position sizing."""

    kelly_fraction: float = 0.25  # Use 1/4 Kelly
    max_kelly_size_pct: float = 0.20  # Never exceed 20%

    def calculate(
        self,
        portfolio_value: float,
        win_probability: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate position size using fractional Kelly.

        Full Kelly: f* = (bp - q) / b
        Where:
          f* = fraction of capital to bet
          b = win/loss ratio (odds)
          p = probability of winning
          q = probability of losing (1-p)
        """

        if win_probability <= 0 or win_probability >= 1:
            return 0.0

        q = 1 - win_probability
        b = win_loss_ratio

        # Full Kelly
        full_kelly = (b * win_probability - q) / b

        # Fractional Kelly
        fractional_kelly = full_kelly * self.kelly_fraction

        # Clamp to valid range
        fractional_kelly = max(0, min(fractional_kelly, self.max_kelly_size_pct))

        return portfolio_value * fractional_kelly
```

### 2.3 Position Sizing Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      POSITION SIZING DECISION TREE                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │ Calculate Base  │
                              │ ATR Position    │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Confidence      │
                              │ >= 0.6?         │
                              └────────┬────────┘
                                  │    │
                              NO ─┘    └─ YES
                              │              │
                              ▼              ▼
                        ┌─────────┐   ┌─────────────────┐
                        │ SIZE=0  │   │ Apply Confidence│
                        │ NO TRADE│   │ Multiplier      │
                        └─────────┘   └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ Size > Max 20%? │
                                      └────────┬────────┘
                                          │    │
                                      YES─┘    └─NO
                                      │              │
                                      ▼              │
                                ┌───────────┐       │
                                │ Cap at 20%│       │
                                └─────┬─────┘       │
                                      │             │
                                      └──────┬──────┘
                                             │
                                             ▼
                                      ┌─────────────────┐
                                      │ Drawdown > 5%?  │
                                      └────────┬────────┘
                                          │    │
                                      YES─┘    └─NO
                                      │              │
                                      ▼              │
                                ┌───────────┐       │
                                │ Reduce by │       │
                                │    50%    │       │
                                └─────┬─────┘       │
                                      │             │
                                      └──────┬──────┘
                                             │
                                             ▼
                                      ┌─────────────────┐
                                      │ High Volatility?│
                                      └────────┬────────┘
                                          │    │
                                      YES─┘    └─NO
                                      │              │
                                      ▼              │
                                ┌───────────┐       │
                                │ Reduce by │       │
                                │    30%    │       │
                                └─────┬─────┘       │
                                      │             │
                                      └──────┬──────┘
                                             │
                                             ▼
                                    ┌───────────────┐
                                    │ FINAL SIZE    │
                                    └───────────────┘
```

---

## 3. Stop-Loss Rules

### 3.1 Stop-Loss Requirements

```python
@dataclass
class StopLossRules:
    """Stop-loss validation and adjustment rules."""

    max_stop_distance_pct: float = 0.02     # 2% max from entry
    min_stop_distance_pct: float = 0.005    # 0.5% min (avoid noise)
    trailing_activation_rr: float = 1.5     # Activate at 1.5R profit

    def validate_stop_loss(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str
    ) -> ValidationResult:
        """Validate stop-loss placement."""

        if direction == 'LONG':
            stop_distance = (entry_price - stop_loss) / entry_price
            if stop_loss >= entry_price:
                return ValidationResult(
                    valid=False,
                    error="Stop-loss must be below entry for LONG"
                )
        else:  # SHORT
            stop_distance = (stop_loss - entry_price) / entry_price
            if stop_loss <= entry_price:
                return ValidationResult(
                    valid=False,
                    error="Stop-loss must be above entry for SHORT"
                )

        # Check max distance
        if stop_distance > self.max_stop_distance_pct:
            return ValidationResult(
                valid=False,
                error=f"Stop distance {stop_distance*100:.2f}% exceeds max {self.max_stop_distance_pct*100:.1f}%",
                suggested_stop=self._calculate_max_stop(entry_price, direction)
            )

        # Check min distance
        if stop_distance < self.min_stop_distance_pct:
            return ValidationResult(
                valid=False,
                error=f"Stop distance {stop_distance*100:.2f}% below min {self.min_stop_distance_pct*100:.1f}%",
                suggested_stop=self._calculate_min_stop(entry_price, direction)
            )

        return ValidationResult(valid=True)

    def _calculate_max_stop(self, entry: float, direction: str) -> float:
        if direction == 'LONG':
            return entry * (1 - self.max_stop_distance_pct)
        return entry * (1 + self.max_stop_distance_pct)

    def _calculate_min_stop(self, entry: float, direction: str) -> float:
        if direction == 'LONG':
            return entry * (1 - self.min_stop_distance_pct)
        return entry * (1 + self.min_stop_distance_pct)

@dataclass
class ValidationResult:
    valid: bool
    error: Optional[str] = None
    suggested_stop: Optional[float] = None
```

### 3.2 Trailing Stop Rules

```python
@dataclass
class TrailingStopRules:
    """Rules for trailing stop activation and adjustment."""

    activation_profit_multiple: float = 1.5  # Activate at 1.5x risk
    trail_distance_pct: float = 0.01         # Trail by 1%
    step_mode: bool = True                   # Use step-based trailing
    step_profit_levels: list = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        original_stop: float,
        direction: str,
        current_trailing_stop: Optional[float] = None
    ) -> Optional[float]:
        """Calculate new trailing stop position."""

        risk_distance = abs(entry_price - original_stop)

        if direction == 'LONG':
            profit = current_price - entry_price
            profit_multiple = profit / risk_distance if risk_distance > 0 else 0

            if profit_multiple < self.activation_profit_multiple:
                return None  # Not activated yet

            if self.step_mode:
                return self._step_trailing_stop_long(
                    entry_price, current_price, risk_distance, current_trailing_stop
                )
            else:
                new_stop = current_price * (1 - self.trail_distance_pct)
                if current_trailing_stop:
                    return max(new_stop, current_trailing_stop)
                return new_stop

        else:  # SHORT
            profit = entry_price - current_price
            profit_multiple = profit / risk_distance if risk_distance > 0 else 0

            if profit_multiple < self.activation_profit_multiple:
                return None

            if self.step_mode:
                return self._step_trailing_stop_short(
                    entry_price, current_price, risk_distance, current_trailing_stop
                )
            else:
                new_stop = current_price * (1 + self.trail_distance_pct)
                if current_trailing_stop:
                    return min(new_stop, current_trailing_stop)
                return new_stop

    def _step_trailing_stop_long(
        self,
        entry: float,
        current: float,
        risk: float,
        current_stop: Optional[float]
    ) -> float:
        """Calculate step-based trailing stop for long position."""
        profit_multiple = (current - entry) / risk
        new_stop = entry  # Start at breakeven

        for level in self.step_profit_levels:
            if profit_multiple >= level:
                # Move stop to lock in (level - 0.5)R profit
                new_stop = entry + (level - 0.5) * risk
            else:
                break

        if current_stop:
            return max(new_stop, current_stop)
        return new_stop

    def _step_trailing_stop_short(
        self,
        entry: float,
        current: float,
        risk: float,
        current_stop: Optional[float]
    ) -> float:
        """Calculate step-based trailing stop for short position."""
        profit_multiple = (entry - current) / risk
        new_stop = entry  # Start at breakeven

        for level in self.step_profit_levels:
            if profit_multiple >= level:
                new_stop = entry - (level - 0.5) * risk
            else:
                break

        if current_stop:
            return min(new_stop, current_stop)
        return new_stop
```

---

## 4. Drawdown Protection

### 4.1 Drawdown Levels and Actions

```python
@dataclass
class DrawdownRules:
    """Drawdown-based risk management rules."""

    # Drawdown thresholds
    warning_threshold: float = 0.05    # 5% - reduce size
    critical_threshold: float = 0.08   # 8% - minimal trading
    shutdown_threshold: float = 0.10   # 10% - stop all trading

    # Daily limits
    daily_loss_limit: float = 0.03     # 3% daily max loss

    # Recovery
    recovery_trades_required: int = 3  # Winning trades to resume
    recovery_profit_required: float = 0.02  # 2% profit to fully resume

    def assess(self, portfolio: Portfolio) -> DrawdownAssessment:
        """Assess current drawdown status."""

        current_dd = portfolio.current_drawdown_pct
        daily_pnl = portfolio.daily_pnl_pct

        # Check shutdown
        if current_dd >= self.shutdown_threshold:
            return DrawdownAssessment(
                level='SHUTDOWN',
                current_drawdown=current_dd,
                action='STOP_ALL_TRADING',
                size_multiplier=0.0,
                message=f"Drawdown {current_dd*100:.1f}% exceeds shutdown threshold"
            )

        # Check daily limit
        if daily_pnl <= -self.daily_loss_limit:
            return DrawdownAssessment(
                level='DAILY_LIMIT',
                current_drawdown=current_dd,
                action='STOP_FOR_DAY',
                size_multiplier=0.0,
                message=f"Daily loss {abs(daily_pnl)*100:.1f}% exceeds limit"
            )

        # Check critical
        if current_dd >= self.critical_threshold:
            return DrawdownAssessment(
                level='CRITICAL',
                current_drawdown=current_dd,
                action='CLOSE_ONLY',
                size_multiplier=0.0,
                message=f"Drawdown {current_dd*100:.1f}% - close-only mode"
            )

        # Check warning
        if current_dd >= self.warning_threshold:
            return DrawdownAssessment(
                level='WARNING',
                current_drawdown=current_dd,
                action='REDUCE_SIZE',
                size_multiplier=0.5,
                message=f"Drawdown {current_dd*100:.1f}% - reduce position sizes"
            )

        return DrawdownAssessment(
            level='NORMAL',
            current_drawdown=current_dd,
            action='NORMAL_TRADING',
            size_multiplier=1.0,
            message="Operating normally"
        )

@dataclass
class DrawdownAssessment:
    level: str
    current_drawdown: float
    action: str
    size_multiplier: float
    message: str
```

### 4.2 Drawdown State Machine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DRAWDOWN STATE MACHINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │     NORMAL      │
                              │   DD < 5%       │
                              │ Size: 100%      │
                              └────────┬────────┘
                                       │
                                       │ DD >= 5%
                                       ▼
                              ┌─────────────────┐
                              │    WARNING      │
                              │   5% <= DD < 8% │
                              │  Size: 50%      │
                              └────────┬────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │ DD < 5%                   │ DD >= 8%
                         │ (need 2% profit)          │
                         ▼                           ▼
                ┌─────────────────┐         ┌─────────────────┐
                │     NORMAL      │         │    CRITICAL     │
                │   DD < 5%       │         │  8% <= DD < 10% │
                │  Size: 100%     │         │   Size: 0%      │
                └─────────────────┘         │  CLOSE ONLY     │
                                            └────────┬────────┘
                                                     │
                                   ┌─────────────────┴─────────────┐
                                   │ DD < 5%                       │ DD >= 10%
                                   │ (need 3 winning trades)       │
                                   ▼                               ▼
                          ┌─────────────────┐             ┌─────────────────┐
                          │    WARNING      │             │    SHUTDOWN     │
                          │   5% <= DD < 8% │             │    DD >= 10%    │
                          │   Size: 50%     │             │    NO TRADING   │
                          └─────────────────┘             └────────┬────────┘
                                                                   │
                                                                   │ Manual reset
                                                                   │ required
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │    RECOVERY     │
                                                          │ Paper trade only│
                                                          │ Prove strategy  │
                                                          └─────────────────┘
```

---

## 5. Leverage Rules

### 5.1 Leverage Limits

```python
@dataclass
class LeverageRules:
    """Leverage management rules."""

    max_leverage: int = 3
    default_leverage: int = 2
    conservative_leverage: int = 1

    # Volatility-based adjustment
    high_volatility_threshold: float = 1.5  # ATR > 1.5x normal
    low_volatility_threshold: float = 0.7   # ATR < 0.7x normal

    def calculate_leverage(
        self,
        requested_leverage: int,
        current_atr: float,
        normal_atr: float,
        regime: str,
        drawdown_pct: float,
        confidence: float
    ) -> LeverageResult:
        """Calculate appropriate leverage based on conditions."""

        # Start with requested or default
        leverage = min(requested_leverage, self.max_leverage)

        # Reduce for high volatility
        atr_ratio = current_atr / normal_atr if normal_atr > 0 else 1.0
        if atr_ratio > self.high_volatility_threshold:
            leverage = min(leverage, self.conservative_leverage)
            reason = f"High volatility (ATR ratio: {atr_ratio:.2f})"
            return LeverageResult(leverage=leverage, reason=reason)

        # Reduce in choppy/ranging regime
        if regime in ['choppy', 'ranging']:
            leverage = min(leverage, self.default_leverage)

        # Reduce for drawdown
        if drawdown_pct >= 0.05:
            leverage = min(leverage, self.conservative_leverage)
            reason = f"Drawdown protection ({drawdown_pct*100:.1f}%)"
            return LeverageResult(leverage=leverage, reason=reason)

        # Reduce for low confidence
        if confidence < 0.7:
            leverage = min(leverage, self.default_leverage)

        return LeverageResult(
            leverage=leverage,
            reason="Standard conditions"
        )

@dataclass
class LeverageResult:
    leverage: int
    reason: str
```

### 5.2 Leverage Decision Matrix

| Condition | Max Leverage | Rationale |
|-----------|--------------|-----------|
| Normal conditions | 3x | Full capability |
| High volatility (ATR > 1.5x) | 1x | Protect against gaps |
| Choppy/Ranging regime | 2x | Reduced edge |
| Drawdown > 5% | 1x | Capital preservation |
| Confidence < 0.7 | 2x | Uncertainty adjustment |
| New strategy (< 30 trades) | 1x | Validation period |
| Multiple conditions | Minimum of all | Conservative approach |

---

## 6. Correlation Management

### 6.1 Correlation Rules

```python
@dataclass
class CorrelationRules:
    """Manage correlation risk across positions."""

    btc_xrp_correlation_threshold: float = 0.8  # Consider highly correlated
    max_correlated_positions: int = 2
    correlation_size_reduction: float = 0.5     # 50% size when correlated

    def assess_correlation_risk(
        self,
        symbol: str,
        open_positions: list[Position],
        correlation_matrix: dict[str, dict[str, float]]
    ) -> CorrelationAssessment:
        """Assess correlation risk for new position."""

        correlated_positions = []

        for position in open_positions:
            correlation = self._get_correlation(
                symbol, position.symbol, correlation_matrix
            )

            if abs(correlation) >= self.btc_xrp_correlation_threshold:
                correlated_positions.append({
                    'symbol': position.symbol,
                    'correlation': correlation,
                    'size': position.size_usdt
                })

        # Check position count
        if len(correlated_positions) >= self.max_correlated_positions:
            return CorrelationAssessment(
                allowed=False,
                reason=f"Already {len(correlated_positions)} correlated positions open",
                size_multiplier=0.0,
                correlated_positions=correlated_positions
            )

        # Allow with reduced size if some correlation exists
        if correlated_positions:
            return CorrelationAssessment(
                allowed=True,
                reason=f"Reducing size due to correlation with {correlated_positions[0]['symbol']}",
                size_multiplier=self.correlation_size_reduction,
                correlated_positions=correlated_positions
            )

        return CorrelationAssessment(
            allowed=True,
            reason="No significant correlation with open positions",
            size_multiplier=1.0,
            correlated_positions=[]
        )

    def _get_correlation(
        self,
        symbol1: str,
        symbol2: str,
        matrix: dict[str, dict[str, float]]
    ) -> float:
        """Get correlation between two symbols."""
        try:
            return matrix[symbol1][symbol2]
        except KeyError:
            # Default correlation assumptions
            if 'BTC' in symbol1 and 'BTC' in symbol2:
                return 1.0
            if 'XRP' in symbol1 and 'XRP' in symbol2:
                return 1.0
            if ('BTC' in symbol1 and 'XRP' in symbol2) or \
               ('XRP' in symbol1 and 'BTC' in symbol2):
                return 0.85  # Assume high correlation
            return 0.5

@dataclass
class CorrelationAssessment:
    allowed: bool
    reason: str
    size_multiplier: float
    correlated_positions: list[dict]
```

### 6.2 XRP/BTC as Decorrelation Tool

```python
class DecorrelationStrategy:
    """Use XRP/BTC pair for decorrelation."""

    def should_use_xrp_btc(
        self,
        btc_usdt_position: Optional[Position],
        xrp_usdt_position: Optional[Position],
        signal: TechnicalSignal
    ) -> bool:
        """Determine if XRP/BTC should be used for decorrelation."""

        # If we have USDT positions in both BTC and XRP
        if btc_usdt_position and xrp_usdt_position:
            # XRP/BTC can help hedge the pair
            return True

        # If signal suggests XRP strength relative to BTC
        if signal.symbol == 'XRP/BTC' and signal.strength > 0.7:
            return True

        # If BTC/USDT position is large and we want XRP exposure
        if btc_usdt_position and btc_usdt_position.size_pct > 0.15:
            # Use XRP/BTC instead of XRP/USDT for exposure
            return True

        return False
```

---

## 7. Cooldown Rules

### 7.1 Cooldown Triggers

```python
@dataclass
class CooldownRules:
    """Manage trading cooldowns."""

    # Time-based cooldowns
    min_time_between_trades_same_pair: int = 1800  # 30 minutes
    min_time_between_any_trade: int = 60           # 1 minute

    # Loss-based cooldowns
    consecutive_loss_cooldown_trigger: int = 5
    consecutive_loss_cooldown_duration: int = 3600  # 1 hour

    # Volatility-based cooldowns
    extreme_move_threshold: float = 0.05  # 5% move
    extreme_move_cooldown: int = 900      # 15 minutes

    def check_cooldown(
        self,
        symbol: str,
        last_trade_times: dict[str, datetime],
        consecutive_losses: int,
        recent_price_move: float
    ) -> CooldownResult:
        """Check if trading is in cooldown."""

        now = datetime.utcnow()

        # Check same-pair cooldown
        if symbol in last_trade_times:
            elapsed = (now - last_trade_times[symbol]).total_seconds()
            if elapsed < self.min_time_between_trades_same_pair:
                remaining = self.min_time_between_trades_same_pair - elapsed
                return CooldownResult(
                    in_cooldown=True,
                    reason=f"Same-pair cooldown ({remaining:.0f}s remaining)",
                    remaining_seconds=remaining
                )

        # Check any-trade cooldown
        if last_trade_times:
            last_any = max(last_trade_times.values())
            elapsed = (now - last_any).total_seconds()
            if elapsed < self.min_time_between_any_trade:
                remaining = self.min_time_between_any_trade - elapsed
                return CooldownResult(
                    in_cooldown=True,
                    reason=f"Global trade cooldown ({remaining:.0f}s remaining)",
                    remaining_seconds=remaining
                )

        # Check consecutive loss cooldown
        if consecutive_losses >= self.consecutive_loss_cooldown_trigger:
            return CooldownResult(
                in_cooldown=True,
                reason=f"{consecutive_losses} consecutive losses - in cooldown",
                remaining_seconds=self.consecutive_loss_cooldown_duration
            )

        # Check extreme move cooldown
        if abs(recent_price_move) > self.extreme_move_threshold:
            return CooldownResult(
                in_cooldown=True,
                reason=f"Extreme price move ({recent_price_move*100:.1f}%) - waiting for stabilization",
                remaining_seconds=self.extreme_move_cooldown
            )

        return CooldownResult(
            in_cooldown=False,
            reason="No cooldown active",
            remaining_seconds=0
        )

@dataclass
class CooldownResult:
    in_cooldown: bool
    reason: str
    remaining_seconds: float
```

---

## 8. Rule Engine Implementation

### 8.1 Complete Risk Agent

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class RiskAssessment:
    approved: bool
    adjusted_size: Optional[float]
    adjusted_leverage: Optional[int]
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    rejection_reasons: list[str]
    warnings: list[str]
    risk_score: float  # 0-1, higher = more risk

class RiskAgent:
    """Complete risk management agent."""

    def __init__(self, config: dict):
        self.position_rules = PositionSizeRule(**config.get('position', {}))
        self.stop_rules = StopLossRules(**config.get('stop_loss', {}))
        self.drawdown_rules = DrawdownRules(**config.get('drawdown', {}))
        self.leverage_rules = LeverageRules(**config.get('leverage', {}))
        self.correlation_rules = CorrelationRules(**config.get('correlation', {}))
        self.cooldown_rules = CooldownRules(**config.get('cooldown', {}))

    def assess(
        self,
        decision: TradingDecision,
        portfolio: Portfolio,
        market_context: MarketContext,
        open_positions: list[Position],
        trade_history: list[Trade]
    ) -> RiskAssessment:
        """Perform complete risk assessment."""

        rejection_reasons = []
        warnings = []
        risk_score = 0.0

        # 1. Check drawdown limits
        dd_assessment = self.drawdown_rules.assess(portfolio)
        if dd_assessment.level in ['SHUTDOWN', 'DAILY_LIMIT']:
            return RiskAssessment(
                approved=False,
                rejection_reasons=[dd_assessment.message],
                warnings=[],
                risk_score=1.0
            )
        if dd_assessment.level == 'CRITICAL':
            if decision.action not in ['HOLD', 'CLOSE']:
                rejection_reasons.append(dd_assessment.message)
        if dd_assessment.level == 'WARNING':
            warnings.append(dd_assessment.message)
            risk_score += 0.2

        # 2. Check cooldown
        cooldown = self.cooldown_rules.check_cooldown(
            symbol=decision.symbol,
            last_trade_times=self._get_last_trade_times(trade_history),
            consecutive_losses=self._count_consecutive_losses(trade_history),
            recent_price_move=market_context.recent_price_move.get(decision.symbol, 0)
        )
        if cooldown.in_cooldown and decision.action in ['LONG', 'SHORT']:
            rejection_reasons.append(cooldown.reason)

        # 3. Check correlation
        if decision.action in ['LONG', 'SHORT']:
            correlation = self.correlation_rules.assess_correlation_risk(
                symbol=decision.symbol,
                open_positions=open_positions,
                correlation_matrix=market_context.correlation_matrix
            )
            if not correlation.allowed:
                rejection_reasons.append(correlation.reason)
            elif correlation.size_multiplier < 1.0:
                warnings.append(correlation.reason)

        # 4. Validate stop-loss
        if decision.action in ['LONG', 'SHORT']:
            stop_validation = self.stop_rules.validate_stop_loss(
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                direction=decision.action
            )
            if not stop_validation.valid:
                rejection_reasons.append(stop_validation.error)

        # 5. Calculate adjusted position size
        adjusted_size = None
        if decision.action in ['LONG', 'SHORT'] and not rejection_reasons:
            position_calc = self.position_rules.calculate(
                portfolio_value=portfolio.total_value_usdt,
                entry_price=decision.entry_price,
                atr=market_context.indicators[decision.symbol]['atr14'],
                confidence=decision.confidence
            )

            # Apply drawdown modifier
            adjusted_size = position_calc.size_quote * dd_assessment.size_multiplier

            # Apply correlation modifier if needed
            if correlation and correlation.size_multiplier < 1.0:
                adjusted_size *= correlation.size_multiplier

        # 6. Calculate adjusted leverage
        adjusted_leverage = None
        if decision.action in ['LONG', 'SHORT'] and not rejection_reasons:
            leverage_result = self.leverage_rules.calculate_leverage(
                requested_leverage=decision.leverage,
                current_atr=market_context.indicators[decision.symbol]['atr14'],
                normal_atr=market_context.normal_atr[decision.symbol],
                regime=market_context.regime.name,
                drawdown_pct=portfolio.current_drawdown_pct,
                confidence=decision.confidence
            )
            adjusted_leverage = leverage_result.leverage
            if leverage_result.leverage < decision.leverage:
                warnings.append(f"Leverage reduced: {leverage_result.reason}")

        # 7. Calculate risk score
        risk_factors = [
            dd_assessment.current_drawdown * 5,  # Weight drawdown heavily
            0.1 if len(open_positions) > 0 else 0,
            0.1 if decision.leverage > 2 else 0,
            (1 - decision.confidence) * 0.3,
        ]
        risk_score = min(1.0, sum(risk_factors))

        return RiskAssessment(
            approved=len(rejection_reasons) == 0,
            adjusted_size=adjusted_size,
            adjusted_leverage=adjusted_leverage,
            stop_loss_price=decision.stop_loss,
            take_profit_price=decision.take_profit,
            rejection_reasons=rejection_reasons,
            warnings=warnings,
            risk_score=risk_score
        )

    def _get_last_trade_times(self, trades: list[Trade]) -> dict[str, datetime]:
        result = {}
        for trade in trades:
            if trade.symbol not in result or trade.timestamp > result[trade.symbol]:
                result[trade.symbol] = trade.timestamp
        return result

    def _count_consecutive_losses(self, trades: list[Trade]) -> int:
        count = 0
        for trade in reversed(trades):
            if trade.pnl_pct and trade.pnl_pct < 0:
                count += 1
            else:
                break
        return count
```

### 8.2 Rule Configuration

```yaml
# config/risk_rules.yaml

position:
  risk_per_trade_pct: 0.01
  atr_multiplier: 2.0
  max_position_pct: 0.20

stop_loss:
  max_stop_distance_pct: 0.02
  min_stop_distance_pct: 0.005
  trailing_activation_rr: 1.5

drawdown:
  warning_threshold: 0.05
  critical_threshold: 0.08
  shutdown_threshold: 0.10
  daily_loss_limit: 0.03
  recovery_trades_required: 3
  recovery_profit_required: 0.02

leverage:
  max_leverage: 3
  default_leverage: 2
  conservative_leverage: 1
  high_volatility_threshold: 1.5
  low_volatility_threshold: 0.7

correlation:
  btc_xrp_correlation_threshold: 0.8
  max_correlated_positions: 2
  correlation_size_reduction: 0.5

cooldown:
  min_time_between_trades_same_pair: 1800
  min_time_between_any_trade: 60
  consecutive_loss_cooldown_trigger: 5
  consecutive_loss_cooldown_duration: 3600
  extreme_move_threshold: 0.05
  extreme_move_cooldown: 900
```

---

## Appendix: Risk Rule Summary Table

| Rule Category | Parameter | Value | Trigger Action |
|---------------|-----------|-------|----------------|
| **Position Size** | Max per trade | 20% | Cap position |
| **Position Size** | Risk per trade | 1% | Size calculation |
| **Position Size** | Confidence < 0.6 | No trade | Block entry |
| **Stop-Loss** | Max distance | 2% | Reject/adjust stop |
| **Stop-Loss** | Min distance | 0.5% | Reject stop |
| **Drawdown** | Warning (5%) | Reduce 50% | Reduce size |
| **Drawdown** | Critical (8%) | Close only | Block new trades |
| **Drawdown** | Shutdown (10%) | Stop all | Block everything |
| **Drawdown** | Daily limit (3%) | Stop for day | Block until tomorrow |
| **Leverage** | Max normal | 3x | Cap leverage |
| **Leverage** | High volatility | 1x | Reduce leverage |
| **Leverage** | Drawdown > 5% | 1x | Reduce leverage |
| **Correlation** | High correlation | 2 positions max | Block additional |
| **Correlation** | Some correlation | 50% size | Reduce size |
| **Cooldown** | Same pair | 30 min | Block trade |
| **Cooldown** | 5 consecutive losses | 1 hour | Pause trading |
| **Cooldown** | Extreme move (5%) | 15 min | Wait for stability |

---

*Document Version: 1.0*
*Last Updated: December 2025*
