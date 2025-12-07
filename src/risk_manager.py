"""
Phase 11: Volatility-Aware Risk Manager
Dynamic leverage scaling + fear/greed detection + short position guards
Phase 11: Lowered short thresholds, 15% max exposure, RSI<40 auto-exit
"""
import numpy as np
from typing import Dict, Optional


class RiskManager:
    """
    Risk management with volatility-aware position sizing and leverage scaling.
    Defensive in high-vol periods, aggressive on calm dips.
    Phase 11: Tuned short thresholds, 15% max exposure, RSI<40 auto-exit.
    """

    def __init__(self, max_drawdown: float = 0.20, max_leverage: float = 10.0):
        self.max_dd = max_drawdown
        self.max_lev = max_leverage

        # Phase 11: Volatility thresholds (lowered for earlier short entries)
        self.vol_high = 0.04      # Phase 11: Lowered from 0.05 - ATR% for bear signal
        self.vol_extreme = 0.08   # ATR% above this = extreme (park USDT)
        self.vol_low = 0.02       # ATR% below this = calm market

        # Fear/Greed state (simulated from volatility)
        self.market_state = 'neutral'  # 'fear', 'extreme_fear', 'neutral', 'greed', 'bear'

        # Position sizing limits
        self.min_collateral_pct = 0.05  # Min 5% of USDT per trade
        self.max_collateral_pct = 0.10  # Max 10% of USDT per trade

        # Phase 11: Short position limits (tuned for more shorts)
        self.max_short_leverage = 5     # Cap short leverage at 5x
        self.max_short_exposure = 0.15  # Phase 11: Lowered from 0.20 to 15% max exposure
        self.short_rsi_threshold = 65   # Phase 11: Lowered from 70 - earlier overbought
        self.short_rsi_exit = 40        # Phase 11: Auto-exit shorts when RSI < 40
        self.short_stop_loss = 0.08     # 8% loss triggers stop loss for shorts
        self.short_take_profit = 0.15   # 15% gain triggers take profit for shorts

    def dynamic_leverage(self, volatility: float, base_max: int = 10) -> int:
        """
        Scale down leverage in high volatility (current Dec chop).
        Returns optimal leverage based on market conditions.

        Args:
            volatility: Current ATR as percentage (e.g., 0.05 = 5%)
            base_max: Maximum leverage in calm conditions

        Returns:
            int: Recommended leverage (1-10)
        """
        if volatility > self.vol_extreme:
            # Extreme volatility - no leverage, park USDT
            self.market_state = 'extreme_fear'
            return 1

        elif volatility > self.vol_high:
            # High volatility - cap at 3x
            self.market_state = 'fear'
            return min(base_max // 3, 3)

        elif volatility > self.vol_low:
            # Normal volatility - moderate leverage 5x
            self.market_state = 'neutral'
            return min(base_max // 2, 5)

        else:
            # Low volatility - full leverage on confirmed dips
            self.market_state = 'greed'
            return base_max

    def calculate_atr_pct(self, high: np.ndarray, low: np.ndarray,
                          close: np.ndarray, period: int = 14) -> float:
        """
        Calculate ATR as percentage of price.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            period: ATR period

        Returns:
            float: ATR as percentage (e.g., 0.05 = 5%)
        """
        if len(close) < period + 1:
            return 0.05  # Default to moderate volatility

        tr_list = []
        for i in range(1, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)

        atr = np.mean(tr_list[-period:])
        current_price = close[-1]

        return atr / current_price if current_price > 0 else 0.05

    def position_size(self, portfolio_value: float, volatility: float,
                      confidence: float = 0.8) -> float:
        """
        Kelly-inspired position sizing with volatility adjustment.

        Args:
            portfolio_value: Total portfolio value in USD
            volatility: Current volatility (ATR%)
            confidence: Signal confidence (0-1)

        Returns:
            float: Recommended position size in USD
        """
        # Base Kelly fraction
        kelly = confidence - (1 - confidence) / 2.0

        # Volatility-adjusted size
        vol_scalar = max(0.5, 1.0 - volatility * 5)  # Reduce size in high vol
        size_pct = kelly * vol_scalar

        # Clamp to limits
        size_pct = max(self.min_collateral_pct, min(size_pct, self.max_collateral_pct))

        return size_pct * portfolio_value

    def dynamic_position_size(self, usdt_balance: float, volatility: float,
                              confidence: float = 0.8) -> float:
        """
        Phase 8: Dynamic position sizing for margin trades.
        Risk 5-10% of USDT collateral based on conditions.

        Args:
            usdt_balance: Available USDT
            volatility: Current ATR%
            confidence: Signal confidence (0-1)

        Returns:
            float: USDT collateral to use
        """
        # Base risk: 5-10% of USDT
        if volatility > self.vol_high:
            risk_pct = self.min_collateral_pct  # 5% in high vol
        elif confidence > 0.9:
            risk_pct = self.max_collateral_pct  # 10% on high conviction
        else:
            risk_pct = 0.07  # 7% default

        # Scale by confidence
        risk_pct *= confidence

        return usdt_balance * risk_pct

    def should_park_usdt(self, volatility: float, recent_drawdown: float = 0.0) -> bool:
        """
        Check if conditions warrant parking in USDT (defensive mode).

        Args:
            volatility: Current ATR%
            recent_drawdown: Recent portfolio drawdown

        Returns:
            bool: True if should park in USDT
        """
        # Park during extreme fear
        if volatility > self.vol_extreme:
            return True

        # Park if recent drawdown exceeds threshold
        if recent_drawdown > self.max_dd * 0.5:  # 50% of max DD
            return True

        return False

    def check_liquidation(self, entry_price: float, current_price: float,
                          leverage: float, direction: str = 'long') -> bool:
        """Check if position would be liquidated."""
        if direction == 'long':
            liq_price = entry_price * (1 - 0.9 / leverage)
            return current_price <= liq_price
        else:
            liq_price = entry_price * (1 + 0.9 / leverage)
            return current_price >= liq_price

    def get_market_state(self) -> str:
        """Get current market state based on volatility analysis."""
        return self.market_state

    def get_risk_params(self, volatility: float, confidence: float) -> Dict:
        """
        Get all risk parameters for current conditions.

        Returns:
            dict: Complete risk parameter set
        """
        leverage = self.dynamic_leverage(volatility)
        should_park = self.should_park_usdt(volatility)

        return {
            'leverage': leverage,
            'market_state': self.market_state,
            'should_park': should_park,
            'volatility': volatility,
            'vol_category': 'extreme' if volatility > self.vol_extreme else
                           'high' if volatility > self.vol_high else
                           'normal' if volatility > self.vol_low else 'low'
        }

    # ========== Phase 10: Short Position Guards ==========

    def can_open_short(self, portfolio_value: float, current_short_exposure: float,
                       volatility: float, rsi: float) -> bool:
        """
        Phase 10: Check if conditions allow opening a short position.

        Args:
            portfolio_value: Total portfolio value in USD
            current_short_exposure: Current short exposure in USD
            volatility: Current ATR%
            rsi: Current RSI value

        Returns:
            bool: True if short can be opened
        """
        # Check max exposure limit (20% of portfolio)
        if current_short_exposure >= portfolio_value * self.max_short_exposure:
            return False

        # Only short in high volatility + overbought conditions
        if volatility < self.vol_high:
            return False  # Need high vol for shorts

        if rsi < self.short_rsi_threshold:
            return False  # Need overbought condition

        return True

    def short_position_size(self, usdt_balance: float, volatility: float,
                            confidence: float = 0.8) -> float:
        """
        Phase 10: Calculate optimal short position size.

        Args:
            usdt_balance: Available USDT for collateral
            volatility: Current ATR%
            confidence: Bear signal confidence (0-1)

        Returns:
            float: USDT collateral for short position
        """
        # Base risk: 5-8% of USDT for shorts (conservative)
        if volatility > self.vol_extreme:
            risk_pct = 0.05  # 5% in extreme vol
        elif confidence > 0.9:
            risk_pct = 0.08  # 8% on high conviction bear
        else:
            risk_pct = 0.06  # 6% default for shorts

        # Scale by confidence
        risk_pct *= confidence

        return usdt_balance * risk_pct

    def short_leverage(self, volatility: float) -> int:
        """
        Phase 10: Get optimal leverage for short position.
        More conservative than longs - max 5x.

        Args:
            volatility: Current ATR%

        Returns:
            int: Recommended short leverage (1-5)
        """
        if volatility > self.vol_extreme:
            return 2  # Minimal leverage in extreme vol

        elif volatility > self.vol_high:
            return 3  # Moderate leverage

        else:
            return min(self.max_short_leverage, 5)  # Max 5x for shorts

    def check_short_liquidation(self, entry_price: float, current_price: float,
                                 leverage: float) -> bool:
        """
        Phase 10: Check if short position would be liquidated.
        Shorts get liquidated when price goes UP.

        Args:
            entry_price: Entry price of short position
            current_price: Current market price
            leverage: Position leverage

        Returns:
            bool: True if position is liquidated
        """
        # Short liquidation: price rises to (entry * (1 + 0.9/leverage))
        liq_price = entry_price * (1 + 0.9 / leverage)
        return current_price >= liq_price

    def should_close_short(self, entry_price: float, current_price: float,
                           leverage: float, duration_hours: int = 0,
                           rsi: float = 50.0) -> tuple:
        """
        Phase 11: Check if short should be closed (take profit, stop loss, decay, or RSI exit).

        Args:
            entry_price: Entry price of short
            current_price: Current price
            leverage: Position leverage
            duration_hours: How long position has been open
            rsi: Current RSI value (Phase 11: auto-exit when RSI < 40)

        Returns:
            tuple: (should_close: bool, reason: str)
        """
        # Calculate P&L for short (profit when price drops)
        pnl_pct = ((entry_price - current_price) / entry_price) * leverage

        # Take profit at configured threshold
        if pnl_pct >= self.short_take_profit:
            return True, 'take_profit'

        # Stop loss at configured threshold
        if pnl_pct <= -self.short_stop_loss:
            return True, 'stop_loss'

        # Phase 11: Mean reversion exit - close when RSI drops below 40
        if rsi < self.short_rsi_exit:
            return True, 'rsi_mean_reversion'

        # Decay timeout: close after 336 hours (~14 days) regardless
        if duration_hours >= 336:
            return True, 'decay_timeout'

        # Check liquidation
        if self.check_short_liquidation(entry_price, current_price, leverage):
            return True, 'liquidation'

        return False, 'hold'

    def get_short_risk_params(self, volatility: float, rsi: float,
                               confidence: float) -> Dict:
        """
        Phase 11: Get all short-specific risk parameters.

        Returns:
            dict: Complete short risk parameter set
        """
        can_short = rsi > self.short_rsi_threshold and volatility > self.vol_high
        leverage = self.short_leverage(volatility)

        return {
            'can_short': can_short,
            'short_leverage': leverage,
            'rsi': rsi,
            'rsi_overbought': rsi > self.short_rsi_threshold,
            'rsi_exit_threshold': self.short_rsi_exit,  # Phase 11: RSI<40 auto-exit
            'stop_loss': self.short_stop_loss,
            'take_profit': self.short_take_profit,
            'max_exposure': self.max_short_exposure,
            'market_state': 'bear' if can_short else self.market_state
        }
