"""
WaveTrend Oscillator Strategy
Research: Top 10 TradingView indicator for 2025, less noise than RSI

WaveTrend by LazyBear is a momentum oscillator that identifies
overbought/oversold conditions with cleaner signals than RSI.

Features:
- Dual line crossover system (like MACD)
- Overbought/oversold zones
- Divergence detection
- Works well in volatile crypto markets

Signals:
- WT cross up from oversold = bullish
- WT cross down from overbought = bearish
- Divergence between price and WT = reversal warning
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


class WaveTrend(BaseStrategy):
    """
    WaveTrend Oscillator strategy.

    Cleaner momentum signals than RSI, especially for crypto.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # WaveTrend parameters
        self.channel_length = config.get('channel_length', 10)
        self.average_length = config.get('average_length', 21)
        self.ma_length = config.get('ma_length', 4)  # Smoothing

        # Threshold levels
        self.overbought = config.get('overbought', 60)
        self.oversold = config.get('oversold', -60)
        self.extreme_overbought = config.get('extreme_overbought', 80)
        self.extreme_oversold = config.get('extreme_oversold', -80)

        # Signal settings
        self.require_zone_exit = config.get('require_zone_exit', True)
        self.use_divergence = config.get('use_divergence', True)
        self.divergence_lookback = config.get('divergence_lookback', 14)

        # Position sizing
        self.base_size_pct = config.get('base_size_pct', 0.10)

        # Symbols
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # State
        self.last_wt1: Dict[str, float] = {}
        self.last_wt2: Dict[str, float] = {}
        self.last_zone: Dict[str, str] = {}

        # Position tracking for exit signals
        self.positions: Dict[str, str] = {}  # symbol -> 'long' or 'short'

    def _calculate_wavetrend(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate WaveTrend Oscillator.

        Formula:
        1. HLC3 = (High + Low + Close) / 3
        2. ESA = EMA(HLC3, channel_length)
        3. D = EMA(|HLC3 - ESA|, channel_length)
        4. CI = (HLC3 - ESA) / (0.015 * D)
        5. WT1 = EMA(CI, average_length)
        6. WT2 = SMA(WT1, ma_length)
        """
        hlc3 = (df['high'] + df['low'] + df['close']) / 3

        # ESA - EMA of HLC3
        esa = hlc3.ewm(span=self.channel_length, adjust=False).mean()

        # D - EMA of absolute deviation
        d = (hlc3 - esa).abs().ewm(span=self.channel_length, adjust=False).mean()

        # CI - Channel Index
        ci = (hlc3 - esa) / (0.015 * d + 1e-10)

        # WT1 - EMA of CI
        wt1 = ci.ewm(span=self.average_length, adjust=False).mean()

        # WT2 - SMA of WT1
        wt2 = wt1.rolling(window=self.ma_length).mean()

        return {
            'wt1': wt1,
            'wt2': wt2,
            'diff': wt1 - wt2
        }

    def _get_zone(self, wt1: float) -> str:
        """Determine current WT zone."""
        if wt1 >= self.extreme_overbought:
            return 'extreme_overbought'
        elif wt1 >= self.overbought:
            return 'overbought'
        elif wt1 <= self.extreme_oversold:
            return 'extreme_oversold'
        elif wt1 <= self.oversold:
            return 'oversold'
        return 'neutral'

    def _check_crossover(self, wt1: pd.Series, wt2: pd.Series) -> Optional[str]:
        """Check for WT1/WT2 crossover."""
        if len(wt1) < 2 or len(wt2) < 2:
            return None

        current_wt1 = wt1.iloc[-1]
        prev_wt1 = wt1.iloc[-2]
        current_wt2 = wt2.iloc[-1]
        prev_wt2 = wt2.iloc[-2]

        if pd.isna(current_wt1) or pd.isna(current_wt2):
            return None

        # Bullish cross: WT1 crosses above WT2
        if current_wt1 > current_wt2 and prev_wt1 <= prev_wt2:
            return 'bullish'

        # Bearish cross: WT1 crosses below WT2
        if current_wt1 < current_wt2 and prev_wt1 >= prev_wt2:
            return 'bearish'

        return None

    def _check_divergence(self, close: pd.Series, wt1: pd.Series) -> Optional[str]:
        """
        Check for divergence between price and WaveTrend.

        Bullish divergence: Price makes lower low, WT makes higher low
        Bearish divergence: Price makes higher high, WT makes lower high
        """
        if not self.use_divergence:
            return None

        n = self.divergence_lookback
        if len(close) < n + 5 or len(wt1) < n + 5:
            return None

        # Recent data
        recent_close = close.iloc[-n:]
        recent_wt = wt1.iloc[-n:]
        prior_close = close.iloc[-2*n:-n]
        prior_wt = wt1.iloc[-2*n:-n]

        current_close = close.iloc[-1]
        current_wt = wt1.iloc[-1]

        # Find swing points
        recent_close_min = recent_close.min()
        recent_close_max = recent_close.max()
        prior_close_min = prior_close.min()
        prior_close_max = prior_close.max()

        recent_wt_min = recent_wt.min()
        recent_wt_max = recent_wt.max()
        prior_wt_min = prior_wt.min()
        prior_wt_max = prior_wt.max()

        # Bullish divergence
        if recent_close_min < prior_close_min and recent_wt_min > prior_wt_min:
            if current_wt < self.oversold:  # More significant in oversold
                return 'bullish'

        # Bearish divergence
        if recent_close_max > prior_close_max and recent_wt_max < prior_wt_max:
            if current_wt > self.overbought:  # More significant in overbought
                return 'bearish'

        return None

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals based on WaveTrend oscillator.

        Priority:
        1. Exit signals (close existing positions on reversal)
        2. Crossover from extreme zone + divergence = highest confidence
        3. Crossover from overbought/oversold zone
        4. Simple crossover in neutral zone
        """
        signals = []

        for symbol in self.symbols:
            df = data.get(f'{symbol}_1h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(f'{symbol}_15m')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            min_bars = max(self.channel_length, self.average_length, self.divergence_lookback * 2) + 10
            if df is None or len(df) < min_bars:
                continue

            close = df['close']

            # Calculate WaveTrend
            wt = self._calculate_wavetrend(df)
            wt1 = wt['wt1']
            wt2 = wt['wt2']

            current_wt1 = wt1.iloc[-1]
            current_wt2 = wt2.iloc[-1]
            prev_zone = self.last_zone.get(symbol, 'neutral')
            current_zone = self._get_zone(current_wt1)

            # Store state
            self.last_wt1[symbol] = current_wt1
            self.last_wt2[symbol] = current_wt2
            self.last_zone[symbol] = current_zone

            # Check for signals
            crossover = self._check_crossover(wt1, wt2)
            divergence = self._check_divergence(close, wt1)

            # Current position for this symbol
            current_pos = self.positions.get(symbol)

            signal = None
            confidence = 0.55  # Base confidence

            # EXIT SIGNALS - Check first (priority over entries)
            # Phase 32: Use market orders for exits to ensure immediate execution
            if current_pos == 'long' and crossover == 'bearish':
                # Close long on bearish crossover
                confidence = 0.70
                if 'overbought' in current_zone:
                    confidence = 0.80  # Higher confidence in overbought
                signal = {
                    'action': 'sell',
                    'symbol': symbol,
                    'size': 1.0,  # Close full position
                    'leverage': 1,
                    'confidence': confidence,
                    'order_type': 'market',  # Phase 32: Immediate exit
                    'reason': f"WaveTrend bearish cross - closing long, WT1={current_wt1:.0f}",
                    'strategy': 'wavetrend',
                    'wt1': current_wt1,
                    'wt2': current_wt2,
                    'zone': current_zone
                }
                # Don't clear position tracking here - wait for on_order_filled
                print(f"[WaveTrend] EXIT SIGNAL: Sell {symbol} (was long)")

            elif current_pos == 'short' and crossover == 'bullish':
                # Cover short on bullish crossover
                confidence = 0.70
                if 'oversold' in current_zone:
                    confidence = 0.80  # Higher confidence in oversold
                signal = {
                    'action': 'cover',
                    'symbol': symbol,
                    'size': 1.0,  # Close full position
                    'leverage': 1,
                    'confidence': confidence,
                    'order_type': 'market',  # Phase 32: Immediate exit
                    'reason': f"WaveTrend bullish cross - covering short, WT1={current_wt1:.0f}",
                    'strategy': 'wavetrend',
                    'wt1': current_wt1,
                    'wt2': current_wt2,
                    'zone': current_zone
                }
                # Don't clear position tracking here - wait for on_order_filled
                print(f"[WaveTrend] EXIT SIGNAL: Cover {symbol} (was short)")

            # Also exit on extreme zone reversals (profit taking)
            elif current_pos == 'long' and current_zone == 'extreme_overbought':
                signal = {
                    'action': 'sell',
                    'symbol': symbol,
                    'size': 1.0,
                    'leverage': 1,
                    'confidence': 0.75,
                    'order_type': 'market',  # Phase 32: Immediate profit taking
                    'reason': f"WaveTrend extreme overbought - taking profit, WT1={current_wt1:.0f}",
                    'strategy': 'wavetrend',
                    'wt1': current_wt1,
                    'zone': current_zone
                }
                print(f"[WaveTrend] PROFIT TAKE: Sell {symbol} at extreme overbought")

            elif current_pos == 'short' and current_zone == 'extreme_oversold':
                signal = {
                    'action': 'cover',
                    'symbol': symbol,
                    'size': 1.0,
                    'leverage': 1,
                    'confidence': 0.75,
                    'order_type': 'market',  # Phase 32: Immediate profit taking
                    'reason': f"WaveTrend extreme oversold - taking profit, WT1={current_wt1:.0f}",
                    'strategy': 'wavetrend',
                    'wt1': current_wt1,
                    'zone': current_zone
                }
                print(f"[WaveTrend] PROFIT TAKE: Cover {symbol} at extreme oversold")

            # ENTRY SIGNALS - Only if no position and crossover detected
            # Phase 32: Use limit orders for entries to get better fills
            elif crossover == 'bullish' and current_pos is None:
                # Zone-based confidence
                if 'oversold' in prev_zone:
                    confidence = 0.75
                    if self.require_zone_exit and current_zone in ['oversold', 'extreme_oversold']:
                        continue  # Wait for zone exit
                elif current_zone in ['oversold', 'extreme_oversold']:
                    confidence = 0.70

                # Divergence bonus
                if divergence == 'bullish':
                    confidence += 0.10

                # Extreme zone bonus
                if 'extreme' in prev_zone:
                    confidence += 0.05

                current_price = close.iloc[-1]
                signal = {
                    'action': 'buy',
                    'symbol': symbol,
                    'size': self.base_size_pct,
                    'leverage': self.max_leverage,
                    'confidence': min(confidence, 0.92),
                    'order_type': 'limit',  # Phase 32: Limit order for better entry
                    'limit_price': current_price,  # Use current price as limit
                    'reason': f"WaveTrend bullish cross, WT1={current_wt1:.0f}, zone={current_zone}",
                    'strategy': 'wavetrend',
                    'wt1': current_wt1,
                    'wt2': current_wt2,
                    'zone': current_zone,
                    'divergence': divergence
                }
                # Don't track position here - wait for on_order_filled
                print(f"[WaveTrend] ENTRY SIGNAL: Buy {symbol} @ ${current_price:.4f}")

            elif crossover == 'bearish' and current_pos is None:
                if 'overbought' in prev_zone:
                    confidence = 0.70
                    if self.require_zone_exit and current_zone in ['overbought', 'extreme_overbought']:
                        continue
                elif current_zone in ['overbought', 'extreme_overbought']:
                    confidence = 0.65

                if divergence == 'bearish':
                    confidence += 0.10

                if 'extreme' in prev_zone:
                    confidence += 0.05

                current_price = close.iloc[-1]
                signal = {
                    'action': 'short',
                    'symbol': symbol,
                    'size': self.base_size_pct * 0.8,
                    'leverage': self.max_leverage,
                    'confidence': min(confidence, 0.88),
                    'order_type': 'limit',  # Phase 32: Limit order for better entry
                    'limit_price': current_price,  # Use current price as limit
                    'reason': f"WaveTrend bearish cross, WT1={current_wt1:.0f}, zone={current_zone}",
                    'strategy': 'wavetrend',
                    'wt1': current_wt1,
                    'wt2': current_wt2,
                    'zone': current_zone,
                    'divergence': divergence
                }
                # Don't track position here - wait for on_order_filled
                print(f"[WaveTrend] ENTRY SIGNAL: Short {symbol} @ ${current_price:.4f}")

            if signal:
                signals.append(signal)

        if signals:
            return max(signals, key=lambda x: x.get('confidence', 0))

        # Build detailed hold reason for diagnostics
        hold_reasons = []
        primary_symbol = self.symbols[0] if self.symbols else 'BTC/USDT'
        primary_wt1 = self.last_wt1.get(primary_symbol, 0)
        primary_zone = self.last_zone.get(primary_symbol, 'unknown')

        if primary_zone == 'neutral':
            hold_reasons.append('WT in neutral zone (no extremes)')
        elif 'oversold' not in primary_zone and 'overbought' not in primary_zone:
            hold_reasons.append(f'WT zone={primary_zone}')

        if not hold_reasons:
            hold_reasons.append('No WT crossover detected')

        return {
            'action': 'hold',
            'symbol': primary_symbol,
            'confidence': 0.0,
            'reason': f"WaveTrend: {', '.join(hold_reasons)}, WT1={primary_wt1:.1f}",
            'strategy': 'wavetrend',
            'indicators': {
                'wavetrend': primary_wt1,
                'wt1': primary_wt1,
                'wt2': self.last_wt2.get(primary_symbol, 0),
                'wt_zone': primary_zone,
                'in_oversold': 'oversold' in primary_zone,
                'in_overbought': 'overbought' in primary_zone,
                'has_position': bool(self.positions),
                'positions': list(self.positions.keys()) if self.positions else []
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Rule-based strategy, no model to update."""
        return True

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """
        Sync internal position state when orders are filled by orchestrator.

        This ensures position tracking stays in sync with actual executions,
        preventing duplicate entries or orphaned positions if orders fail.

        Args:
            order: Filled order details including action, symbol, price
        """
        action = order.get('action', '')
        symbol = order.get('symbol', '')
        price = order.get('price', 0)

        # Normalize symbol format
        if '/' not in symbol:
            # Convert 'XRPUSDT' to 'XRP/USDT' format
            for s in self.symbols:
                if symbol.replace('/', '') == s.replace('/', ''):
                    symbol = s
                    break

        # Track position changes
        prev_position = self.positions.get(symbol)

        if action == 'buy':
            self.positions[symbol] = 'long'
            print(f"[WaveTrend] Order filled: BUY {symbol} @ ${price:.4f} | Position: {prev_position} -> long")

        elif action == 'short':
            self.positions[symbol] = 'short'
            print(f"[WaveTrend] Order filled: SHORT {symbol} @ ${price:.4f} | Position: {prev_position} -> short")

        elif action in ['sell', 'cover', 'close']:
            removed = self.positions.pop(symbol, None)
            print(f"[WaveTrend] Order filled: {action.upper()} {symbol} @ ${price:.4f} | Position: {removed} -> None")

        # Log any state inconsistencies for debugging
        if action == 'sell' and prev_position != 'long':
            print(f"[WaveTrend] WARNING: Sold {symbol} but was not tracking long position (was: {prev_position})")
        elif action == 'cover' and prev_position != 'short':
            print(f"[WaveTrend] WARNING: Covered {symbol} but was not tracking short position (was: {prev_position})")
        elif action == 'buy' and prev_position is not None:
            print(f"[WaveTrend] WARNING: Bought {symbol} but already had position: {prev_position}")

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base = super().get_status()
        base.update({
            'channel_length': self.channel_length,
            'average_length': self.average_length,
            'overbought': self.overbought,
            'oversold': self.oversold,
            'wt_values': self.last_wt1,
            'zones': self.last_zone
        })
        return base
