"""
Enhanced DCA with Dynamic Multiplier Strategy
Research: DCA with volatility-adjusted sizing outperforms fixed DCA

Beyond basic Dollar-Cost Averaging:
- Buy more when price drops below moving average (buy the dip)
- Buy less when price rises above moving average (avoid FOMO)
- Multiplier scales with deviation from average

Features:
- SMA-based fair value estimation
- Volatility-adjusted multipliers
- RSI confirmation for extreme dips
- Accumulation tracking and reporting
- Support for BTC, XRP, and any other asset
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class EnhancedDCA(BaseStrategy):
    """
    Enhanced Dollar-Cost Averaging with dynamic multipliers.

    Buys more when cheap, less when expensive, maximizing accumulation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # DCA base amount
        self.base_amount_usd = config.get('base_amount_usd', 50)  # $50 per interval
        self.dca_interval_hours = config.get('dca_interval_hours', 24)  # Daily DCA

        # Moving average for fair value
        self.ma_period = config.get('ma_period', 200)  # 200 SMA as fair value
        self.short_ma_period = config.get('short_ma_period', 50)  # For trend

        # Deviation multipliers
        self.multipliers = config.get('multipliers', {
            -0.30: 3.0,   # 30% below MA = 3x buy
            -0.20: 2.0,   # 20% below MA = 2x buy
            -0.10: 1.5,   # 10% below MA = 1.5x buy
            0.00: 1.0,    # At MA = 1x buy
            0.10: 0.75,   # 10% above MA = 0.75x buy
            0.20: 0.50,   # 20% above MA = 0.5x buy
            0.30: 0.25,   # 30% above MA = 0.25x buy
        })

        # RSI for extreme dip confirmation
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_extreme_oversold = config.get('rsi_extreme_oversold', 25)
        self.rsi_extreme_bonus = config.get('rsi_extreme_bonus', 0.5)  # +50% on extreme oversold

        # Symbols to accumulate
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])
        self.symbol_allocations = config.get('allocations', {
            'BTC/USDT': 0.60,  # 60% to BTC
            'XRP/USDT': 0.40   # 40% to XRP
        })

        # Accumulation tracking
        self.total_invested: Dict[str, float] = {}
        self.total_accumulated: Dict[str, float] = {}
        self.average_cost: Dict[str, float] = {}
        self.last_dca_time: Dict[str, datetime] = {}
        self.dca_history: List[Dict] = []

    def _calculate_sma(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return close.rolling(window=period).mean()

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _get_multiplier(self, deviation: float) -> float:
        """
        Get buy multiplier based on price deviation from MA.

        Interpolates between defined multiplier points.
        """
        sorted_devs = sorted(self.multipliers.keys())

        # Handle edge cases
        if deviation <= sorted_devs[0]:
            return self.multipliers[sorted_devs[0]]
        if deviation >= sorted_devs[-1]:
            return self.multipliers[sorted_devs[-1]]

        # Find surrounding points and interpolate
        for i, dev in enumerate(sorted_devs[:-1]):
            next_dev = sorted_devs[i + 1]
            if dev <= deviation < next_dev:
                # Linear interpolation
                ratio = (deviation - dev) / (next_dev - dev)
                return self.multipliers[dev] + ratio * (self.multipliers[next_dev] - self.multipliers[dev])

        return 1.0  # Default

    def _should_dca(self, symbol: str) -> bool:
        """Check if it's time for DCA based on interval."""
        last_time = self.last_dca_time.get(symbol)
        if last_time is None:
            return True

        elapsed = datetime.now() - last_time
        return elapsed >= timedelta(hours=self.dca_interval_hours)

    def _update_accumulation_stats(self, symbol: str, usd_spent: float,
                                   amount_bought: float, price: float):
        """Update accumulation tracking."""
        if symbol not in self.total_invested:
            self.total_invested[symbol] = 0
            self.total_accumulated[symbol] = 0
            self.average_cost[symbol] = 0

        self.total_invested[symbol] += usd_spent
        self.total_accumulated[symbol] += amount_bought

        # Update average cost
        if self.total_accumulated[symbol] > 0:
            self.average_cost[symbol] = self.total_invested[symbol] / self.total_accumulated[symbol]

        self.last_dca_time[symbol] = datetime.now()

        # Track history
        self.dca_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'price': price,
            'usd_spent': usd_spent,
            'amount': amount_bought,
            'total_accumulated': self.total_accumulated[symbol],
            'avg_cost': self.average_cost[symbol]
        })

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate DCA buy signals with dynamic multipliers.

        Always generates buy signals for accumulation, but amount varies.
        """
        signals = []

        for symbol in self.symbols:
            # Check if time for DCA
            if not self._should_dca(symbol):
                continue

            df = data.get(f'{symbol}_1d')  # Daily data for DCA
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(f'{symbol}_4h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            if df is None or len(df) < self.ma_period + 5:
                continue

            close = df['close']
            current_price = close.iloc[-1]

            # Calculate moving averages
            sma_long = self._calculate_sma(close, self.ma_period)
            sma_short = self._calculate_sma(close, self.short_ma_period)
            rsi = self._calculate_rsi(close)

            sma_value = sma_long.iloc[-1]
            rsi_value = rsi.iloc[-1]

            if pd.isna(sma_value):
                continue

            # Calculate deviation from MA
            deviation = (current_price - sma_value) / sma_value

            # Get base multiplier
            multiplier = self._get_multiplier(deviation)

            # RSI bonus for extreme oversold
            if rsi_value <= self.rsi_extreme_oversold:
                multiplier += self.rsi_extreme_bonus

            # Calculate buy amount
            allocation = self.symbol_allocations.get(symbol, 0.5)
            base_for_symbol = self.base_amount_usd * allocation
            buy_amount_usd = base_for_symbol * multiplier

            # Calculate crypto amount
            amount_to_buy = buy_amount_usd / current_price

            # Trend context
            trend = 'bullish' if sma_short.iloc[-1] > sma_long.iloc[-1] else 'bearish'

            # Confidence based on value (buy more when cheaper)
            confidence = 0.50 + (multiplier / 6)  # Scale 0.5-1.0

            signal = {
                'action': 'buy',
                'symbol': symbol,
                'size': buy_amount_usd,  # In USD
                'leverage': 1,  # Spot only for accumulation
                'confidence': min(confidence, 0.95),
                'reason': f"DCA: {deviation*100:.1f}% from MA, mult={multiplier:.2f}x, RSI={rsi_value:.0f}",
                'strategy': 'enhanced_dca',
                'multiplier': multiplier,
                'deviation_pct': deviation * 100,
                'usd_amount': buy_amount_usd,
                'crypto_amount': amount_to_buy,
                'current_price': current_price,
                'sma_value': sma_value,
                'trend': trend,
                'is_dca': True  # Flag for orchestrator to handle differently
            }

            # Update stats (assuming execution)
            self._update_accumulation_stats(symbol, buy_amount_usd, amount_to_buy, current_price)

            signals.append(signal)

        if signals:
            # For DCA, we might want to execute all buys, not just the best
            # Return the highest multiplier (best value) for now
            return max(signals, key=lambda x: x.get('multiplier', 0))

        # Check why no signals
        next_dca_times = {}
        for symbol in self.symbols:
            last = self.last_dca_time.get(symbol)
            if last:
                next_time = last + timedelta(hours=self.dca_interval_hours)
                next_dca_times[symbol] = str(next_time)

        # Calculate time until next DCA
        time_to_next = 'unknown'
        for symbol in self.symbols:
            last = self.last_dca_time.get(symbol)
            if last:
                next_time = last + timedelta(hours=self.dca_interval_hours)
                remaining = (next_time - datetime.now()).total_seconds() / 3600
                if remaining > 0:
                    time_to_next = f'{remaining:.1f}h'
                    break

        return {
            'action': 'hold',
            'symbol': self.symbols[0] if self.symbols else 'BTC/USDT',
            'confidence': 0.0,
            'reason': f'EnhancedDCA: Waiting for interval ({time_to_next} to next)',
            'strategy': 'enhanced_dca',
            'next_dca_times': next_dca_times,
            'total_accumulated': self.total_accumulated,
            'average_costs': self.average_cost,
            'indicators': {
                'interval_passed': False,
                'time_since_last': 0,  # Would calculate from last_dca_time
                'dca_interval_hours': self.dca_interval_hours,
                'accumulating': True,
                'total_invested_usd': sum(self.total_invested.values())
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Rule-based strategy, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status with accumulation stats."""
        base = super().get_status()

        # Calculate current value vs cost
        current_value = {}
        unrealized_pnl = {}
        for symbol, amount in self.total_accumulated.items():
            # Would need current price to calculate
            current_value[symbol] = amount  # Placeholder
            if symbol in self.average_cost and self.average_cost[symbol] > 0:
                unrealized_pnl[symbol] = 0  # Would calculate with current price

        base.update({
            'base_amount_usd': self.base_amount_usd,
            'dca_interval_hours': self.dca_interval_hours,
            'total_invested': self.total_invested,
            'total_accumulated': self.total_accumulated,
            'average_cost': self.average_cost,
            'last_dca_times': {s: str(t) for s, t in self.last_dca_time.items()},
            'dca_count': len(self.dca_history),
            'recent_dcas': self.dca_history[-5:] if self.dca_history else []
        })
        return base
