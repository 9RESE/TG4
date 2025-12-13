"""
Volume Profile Strategy
Research: POC (Point of Control) identifies key S/R levels with high accuracy

Volume Profile shows where most trading occurred at each price level.

Key Concepts:
- Point of Control (POC): Price with highest volume = key S/R
- Value Area (VA): Range containing 70% of volume
- High Volume Nodes (HVN): Strong S/R levels
- Low Volume Nodes (LVN): Price tends to move quickly through these

Trading Logic:
- Price below Value Area Low = bullish opportunity
- Price above Value Area High = bearish opportunity
- POC acts as magnet (price tends to return to it)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class VolumeProfile(BaseStrategy):
    """
    Volume Profile trading strategy.

    Uses volume distribution across price levels to identify
    high-probability trading zones.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Volume Profile parameters
        self.lookback_bars = config.get('lookback_bars', 100)  # Bars for profile
        self.num_bins = config.get('num_bins', 50)  # Price level granularity
        self.value_area_pct = config.get('value_area_pct', 0.70)  # 70% for VA

        # Trading parameters
        self.poc_tolerance = config.get('poc_tolerance', 0.005)  # 0.5% from POC
        self.va_tolerance = config.get('va_tolerance', 0.01)  # 1% from VA edges

        # Momentum filter
        self.use_momentum_filter = config.get('use_momentum_filter', True)
        self.momentum_period = config.get('momentum_period', 14)

        # Position sizing
        self.base_size_pct = config.get('base_size_pct', 0.10)

        # Symbols
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])

        # State
        self.profiles: Dict[str, Dict] = {}
        self.last_poc: Dict[str, float] = {}
        self.last_va: Dict[str, Tuple[float, float]] = {}
        self.positions: Dict[str, Dict] = {}  # symbol -> {'side': 'long'/'short', 'entry': price, 'target': price}

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Volume Profile from OHLCV data.

        Returns POC, Value Area, and volume distribution.
        """
        if len(df) < self.lookback_bars:
            return None

        # Use recent data
        recent = df.tail(self.lookback_bars)

        high = recent['high'].max()
        low = recent['low'].min()

        if high == low:
            return None

        # Create price bins
        bin_edges = np.linspace(low, high, self.num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate volume at each price level
        # Simple approach: distribute candle volume across its range
        volume_at_price = np.zeros(self.num_bins)

        for _, row in recent.iterrows():
            candle_high = row['high']
            candle_low = row['low']
            candle_volume = row.get('volume', 1)

            # Find bins this candle spans
            for i, (edge_low, edge_high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                # Check overlap
                overlap_low = max(candle_low, edge_low)
                overlap_high = min(candle_high, edge_high)

                if overlap_high > overlap_low:
                    # Proportion of candle in this bin
                    candle_range = candle_high - candle_low if candle_high > candle_low else 1
                    overlap_ratio = (overlap_high - overlap_low) / candle_range
                    volume_at_price[i] += candle_volume * overlap_ratio

        # Find Point of Control (highest volume price)
        poc_idx = np.argmax(volume_at_price)
        poc = bin_centers[poc_idx]

        # Calculate Value Area (70% of volume)
        total_volume = volume_at_price.sum()
        target_volume = total_volume * self.value_area_pct

        # Start from POC and expand outward
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        current_volume = volume_at_price[poc_idx]

        while current_volume < target_volume:
            # Expand to side with more volume
            can_expand_low = va_low_idx > 0
            can_expand_high = va_high_idx < len(volume_at_price) - 1

            if not can_expand_low and not can_expand_high:
                break

            low_vol = volume_at_price[va_low_idx - 1] if can_expand_low else 0
            high_vol = volume_at_price[va_high_idx + 1] if can_expand_high else 0

            if low_vol >= high_vol and can_expand_low:
                va_low_idx -= 1
                current_volume += volume_at_price[va_low_idx]
            elif can_expand_high:
                va_high_idx += 1
                current_volume += volume_at_price[va_high_idx]
            elif can_expand_low:
                va_low_idx -= 1
                current_volume += volume_at_price[va_low_idx]

        va_low = bin_centers[va_low_idx]
        va_high = bin_centers[va_high_idx]

        # Identify High Volume Nodes (above average)
        avg_volume = volume_at_price.mean()
        hvn_indices = np.where(volume_at_price > avg_volume * 1.5)[0]
        hvn_levels = bin_centers[hvn_indices].tolist()

        # Identify Low Volume Nodes (below average)
        lvn_indices = np.where(volume_at_price < avg_volume * 0.5)[0]
        lvn_levels = bin_centers[lvn_indices].tolist()

        return {
            'poc': poc,
            'va_low': va_low,
            'va_high': va_high,
            'hvn': hvn_levels,
            'lvn': lvn_levels,
            'volume_dist': volume_at_price.tolist(),
            'price_levels': bin_centers.tolist(),
            'profile_high': high,
            'profile_low': low
        }

    def _calculate_momentum(self, close: pd.Series) -> float:
        """Calculate simple momentum."""
        if len(close) < self.momentum_period + 1:
            return 0

        return (close.iloc[-1] - close.iloc[-self.momentum_period]) / close.iloc[-self.momentum_period]

    def _get_price_position(self, price: float, profile: Dict) -> str:
        """Determine price position relative to Value Area."""
        if price > profile['va_high']:
            return 'above_va'
        elif price < profile['va_low']:
            return 'below_va'
        elif abs(price - profile['poc']) / profile['poc'] < self.poc_tolerance:
            return 'at_poc'
        else:
            return 'inside_va'

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate signals based on Volume Profile analysis.

        Trading Logic:
        - Price below VA with positive momentum = bullish (expect return to POC)
        - Price above VA with negative momentum = bearish (expect return to POC)
        - Price at POC = look for breakout direction
        """
        signals = []

        # CHECK EXITS FIRST - Priority over new entries
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            df = data.get(f'{symbol}_1h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            if df is None or len(df) < 2:
                continue

            current_price = df['close'].iloc[-1]
            side = pos['side']
            entry = pos['entry']
            target = pos['target']

            # Check if target (POC) reached
            if side == 'long':
                if current_price >= target:
                    self.positions.pop(symbol, None)
                    return {
                        'action': 'sell',
                        'symbol': symbol,
                        'size': 1.0,
                        'leverage': 1,
                        'confidence': 0.80,
                        'reason': f'Target POC ${target:.2f} reached from ${entry:.2f}',
                        'strategy': 'volume_profile'
                    }
                # Stop loss: price drops further below VA (2% below entry)
                elif current_price < entry * 0.98:
                    self.positions.pop(symbol, None)
                    return {
                        'action': 'sell',
                        'symbol': symbol,
                        'size': 1.0,
                        'leverage': 1,
                        'confidence': 0.75,
                        'reason': f'Stop loss hit, price ${current_price:.2f} < entry ${entry:.2f}',
                        'strategy': 'volume_profile'
                    }

            elif side == 'short':
                if current_price <= target:
                    self.positions.pop(symbol, None)
                    return {
                        'action': 'cover',
                        'symbol': symbol,
                        'size': 1.0,
                        'leverage': 1,
                        'confidence': 0.80,
                        'reason': f'Target POC ${target:.2f} reached from ${entry:.2f}',
                        'strategy': 'volume_profile'
                    }
                # Stop loss: price rises further above VA (2% above entry)
                elif current_price > entry * 1.02:
                    self.positions.pop(symbol, None)
                    return {
                        'action': 'cover',
                        'symbol': symbol,
                        'size': 1.0,
                        'leverage': 1,
                        'confidence': 0.75,
                        'reason': f'Stop loss hit, price ${current_price:.2f} > entry ${entry:.2f}',
                        'strategy': 'volume_profile'
                    }

        for symbol in self.symbols:
            df = data.get(f'{symbol}_1h')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            if df is None or len(df) < self.lookback_bars + 5:
                continue

            close = df['close']
            current_price = close.iloc[-1]

            # Calculate Volume Profile
            profile = self._calculate_volume_profile(df)
            if profile is None:
                continue

            self.profiles[symbol] = profile
            self.last_poc[symbol] = profile['poc']
            self.last_va[symbol] = (profile['va_low'], profile['va_high'])

            # Get position
            position = self._get_price_position(current_price, profile)

            # Calculate momentum
            momentum = self._calculate_momentum(close)
            momentum_bullish = momentum > 0.01
            momentum_bearish = momentum < -0.01

            signal = None

            # Skip if already in a position for this symbol
            if symbol in self.positions:
                continue

            # Strategy logic
            if position == 'below_va':
                # Price below value area
                if not self.use_momentum_filter or momentum_bullish:
                    # Expect reversion to POC
                    target = profile['poc']
                    distance_pct = (target - current_price) / current_price

                    signal = {
                        'action': 'buy',
                        'symbol': symbol,
                        'size': self.base_size_pct,
                        'leverage': self.max_leverage,
                        'confidence': 0.70 + min(distance_pct * 2, 0.15),
                        'reason': f"Below VA, target POC ${profile['poc']:.2f}",
                        'strategy': 'volume_profile',
                        'target': target,
                        'poc': profile['poc'],
                        'va_low': profile['va_low'],
                        'va_high': profile['va_high'],
                        'position': position
                    }
                    # Track position for exit logic
                    self.positions[symbol] = {
                        'side': 'long',
                        'entry': current_price,
                        'target': target
                    }

            elif position == 'above_va':
                # Price above value area
                if not self.use_momentum_filter or momentum_bearish:
                    target = profile['poc']
                    distance_pct = (current_price - target) / current_price

                    signal = {
                        'action': 'short',
                        'symbol': symbol,
                        'size': self.base_size_pct * 0.8,
                        'leverage': self.max_leverage,
                        'confidence': 0.65 + min(distance_pct * 2, 0.15),
                        'reason': f"Above VA, target POC ${profile['poc']:.2f}",
                        'strategy': 'volume_profile',
                        'target': target,
                        'poc': profile['poc'],
                        'va_low': profile['va_low'],
                        'va_high': profile['va_high'],
                        'position': position
                    }
                    # Track position for exit logic
                    self.positions[symbol] = {
                        'side': 'short',
                        'entry': current_price,
                        'target': target
                    }

            elif position == 'at_poc':
                # At POC - look for breakout
                if momentum_bullish:
                    signal = {
                        'action': 'buy',
                        'symbol': symbol,
                        'size': self.base_size_pct * 0.7,
                        'leverage': self.max_leverage,
                        'confidence': 0.60,
                        'reason': f"POC breakout attempt, bullish momentum",
                        'strategy': 'volume_profile',
                        'target': profile['va_high'],
                        'poc': profile['poc'],
                        'position': position
                    }
                    # Track position for exit logic
                    self.positions[symbol] = {
                        'side': 'long',
                        'entry': current_price,
                        'target': profile['va_high']
                    }
                elif momentum_bearish:
                    signal = {
                        'action': 'short',
                        'symbol': symbol,
                        'size': self.base_size_pct * 0.6,
                        'leverage': self.max_leverage,
                        'confidence': 0.55,
                        'reason': f"POC breakout attempt, bearish momentum",
                        'strategy': 'volume_profile',
                        'target': profile['va_low'],
                        'poc': profile['poc'],
                        'position': position
                    }
                    # Track position for exit logic
                    self.positions[symbol] = {
                        'side': 'short',
                        'entry': current_price,
                        'target': profile['va_low']
                    }

            if signal:
                signals.append(signal)

        if signals:
            return max(signals, key=lambda x: x.get('confidence', 0))

        # Build detailed hold reason for diagnostics
        primary_symbol = self.symbols[0] if self.symbols else 'BTC/USDT'
        profile = self.profiles.get(primary_symbol, {})
        poc = self.last_poc.get(primary_symbol, 0)
        va = self.last_va.get(primary_symbol, (0, 0))
        has_position = bool(self.positions)

        hold_reasons = []
        if profile:
            hold_reasons.append('Price inside Value Area (equilibrium)')
        else:
            hold_reasons.append('No volume profile calculated')

        if has_position:
            hold_reasons.append(f'Managing position: {list(self.positions.keys())}')

        return {
            'action': 'hold',
            'symbol': primary_symbol,
            'confidence': 0.0,
            'reason': f"VolumeProfile: {', '.join(hold_reasons)}",
            'strategy': 'volume_profile',
            'indicators': {
                'poc': poc,
                'va_low': va[0],
                'va_high': va[1],
                'near_volume_node': False,  # Inside VA = no edge
                'has_position': has_position,
                'positions': list(self.positions.keys()) if self.positions else []
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Rule-based strategy, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base = super().get_status()
        base.update({
            'lookback_bars': self.lookback_bars,
            'profiles': {s: {
                'poc': p.get('poc'),
                'va_low': p.get('va_low'),
                'va_high': p.get('va_high'),
                'hvn_count': len(p.get('hvn', [])),
                'lvn_count': len(p.get('lvn', []))
            } for s, p in self.profiles.items()}
        })
        return base
