"""
TWAP (Time-Weighted Average Price) Accumulator Strategy
Research: Minimize market impact for large orders

TWAP spreads large orders over time to achieve average execution price.
Ideal for:
- Building large positions without moving the market
- Accumulating crypto over a defined period
- Reducing slippage on large orders

Features:
- Configurable duration and interval
- Random jitter to avoid predictability
- Volume-weighted adjustments
- Progress tracking and reporting
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import random


class TWAPAccumulator(BaseStrategy):
    """
    TWAP execution strategy for large accumulation orders.

    Breaks large orders into smaller chunks executed at regular intervals.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # TWAP parameters
        self.total_amount_usd = config.get('total_amount_usd', 1000)  # Total to invest
        self.duration_hours = config.get('duration_hours', 24)  # Spread over 24h
        self.interval_minutes = config.get('interval_minutes', 30)  # Every 30 min

        # Randomization
        self.use_jitter = config.get('use_jitter', True)
        self.jitter_pct = config.get('jitter_pct', 0.20)  # +/- 20% timing jitter

        # Volume weighting
        self.use_volume_weight = config.get('use_volume_weight', True)
        self.volume_window = config.get('volume_window', 24)

        # Symbols
        self.symbols = config.get('symbols', ['BTC/USDT', 'XRP/USDT'])
        self.symbol_allocations = config.get('allocations', {
            'BTC/USDT': 0.60,
            'XRP/USDT': 0.40
        })

        # Calculate per-chunk amount
        total_intervals = int(self.duration_hours * 60 / self.interval_minutes)
        self.chunk_amount_usd = self.total_amount_usd / max(total_intervals, 1)

        # State tracking
        self.active_twaps: Dict[str, Dict] = {}
        self.completed_chunks: Dict[str, int] = {}
        self.total_executed_usd: Dict[str, float] = {}
        self.total_accumulated: Dict[str, float] = {}
        self.vwap_achieved: Dict[str, float] = {}
        self.last_execution: Dict[str, datetime] = {}
        self.twap_start_time: Optional[datetime] = None

    def start_twap(self, symbol: str, total_usd: float, duration_hours: float):
        """Initialize a TWAP execution for a symbol."""
        total_intervals = int(duration_hours * 60 / self.interval_minutes)

        self.active_twaps[symbol] = {
            'total_usd': total_usd,
            'remaining_usd': total_usd,
            'chunk_usd': total_usd / max(total_intervals, 1),
            'total_intervals': total_intervals,
            'completed_intervals': 0,
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(hours=duration_hours),
            'prices': [],
            'amounts': []
        }

        self.completed_chunks[symbol] = 0
        self.total_executed_usd[symbol] = 0
        self.total_accumulated[symbol] = 0

    def _should_execute_chunk(self, symbol: str) -> bool:
        """Check if it's time for next chunk execution."""
        if symbol not in self.active_twaps:
            return False

        twap = self.active_twaps[symbol]

        # Check if TWAP complete
        if twap['remaining_usd'] <= 0:
            return False
        if datetime.now() > twap['end_time']:
            return False

        # Check interval timing
        last_exec = self.last_execution.get(symbol)
        if last_exec is None:
            return True

        interval = timedelta(minutes=self.interval_minutes)

        # Add jitter
        if self.use_jitter:
            jitter = interval * random.uniform(-self.jitter_pct, self.jitter_pct)
            interval += jitter

        return datetime.now() >= last_exec + interval

    def _calculate_volume_weight(self, df: pd.DataFrame) -> float:
        """
        Calculate volume weight for current period.

        Execute more during high-volume periods for better fills.
        """
        if not self.use_volume_weight:
            return 1.0

        if 'volume' not in df.columns or len(df) < self.volume_window + 1:
            return 1.0

        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].iloc[-self.volume_window-1:-1].mean()

        if avg_vol <= 0:
            return 1.0

        ratio = current_vol / avg_vol

        # Scale: low volume = smaller chunk, high volume = larger chunk
        # But cap at reasonable bounds
        return max(0.5, min(1.5, ratio))

    def _update_vwap(self, symbol: str, price: float, amount: float):
        """Update VWAP calculation."""
        if symbol not in self.active_twaps:
            return

        twap = self.active_twaps[symbol]
        twap['prices'].append(price)
        twap['amounts'].append(amount)

        # Calculate VWAP
        total_value = sum(p * a for p, a in zip(twap['prices'], twap['amounts']))
        total_amount = sum(twap['amounts'])

        if total_amount > 0:
            self.vwap_achieved[symbol] = total_value / total_amount

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate TWAP chunk execution signals.
        """
        signals = []

        for symbol in self.symbols:
            # Initialize TWAP if not active
            if symbol not in self.active_twaps:
                allocation = self.symbol_allocations.get(symbol, 0.5)
                self.start_twap(symbol, self.total_amount_usd * allocation, self.duration_hours)

            # Check if should execute chunk
            if not self._should_execute_chunk(symbol):
                continue

            df = data.get(f'{symbol}_5m')  # Short timeframe for TWAP
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = data.get(symbol)

            if df is None or len(df) < 5:
                continue

            current_price = df['close'].iloc[-1]
            twap = self.active_twaps[symbol]

            # Calculate chunk size with volume weight
            volume_weight = self._calculate_volume_weight(df)
            chunk_usd = twap['chunk_usd'] * volume_weight

            # Don't exceed remaining amount
            chunk_usd = min(chunk_usd, twap['remaining_usd'])

            if chunk_usd < 1:  # Minimum $1
                continue

            crypto_amount = chunk_usd / current_price

            # Progress calculation
            progress = twap['completed_intervals'] / max(twap['total_intervals'], 1)

            signal = {
                'action': 'buy',
                'symbol': symbol,
                'size': chunk_usd,
                'leverage': 1,
                'confidence': 0.90,  # High confidence for scheduled execution
                'reason': f"TWAP chunk {twap['completed_intervals']+1}/{twap['total_intervals']}, {progress*100:.0f}% complete",
                'strategy': 'twap',
                'chunk_usd': chunk_usd,
                'crypto_amount': crypto_amount,
                'volume_weight': volume_weight,
                'remaining_usd': twap['remaining_usd'] - chunk_usd,
                'progress': progress,
                'is_twap': True
            }

            signals.append(signal)

        if signals:
            # Execute the first ready signal
            signal = signals[0]
            symbol = signal['symbol']

            # Update tracking
            twap = self.active_twaps[symbol]
            twap['completed_intervals'] += 1
            twap['remaining_usd'] -= signal['chunk_usd']

            self.completed_chunks[symbol] = twap['completed_intervals']
            self.total_executed_usd[symbol] = self.total_executed_usd.get(symbol, 0) + signal['chunk_usd']
            self.total_accumulated[symbol] = self.total_accumulated.get(symbol, 0) + signal['crypto_amount']
            self.last_execution[symbol] = datetime.now()

            # Update VWAP
            self._update_vwap(symbol, data[symbol]['close'].iloc[-1] if symbol in data else signal['chunk_usd'] / signal['crypto_amount'], signal['crypto_amount'])

            return signal

        # Report status
        status = {}
        for symbol in self.symbols:
            if symbol in self.active_twaps:
                twap = self.active_twaps[symbol]
                status[symbol] = {
                    'progress': f"{twap['completed_intervals']}/{twap['total_intervals']}",
                    'remaining_usd': twap['remaining_usd'],
                    'vwap': self.vwap_achieved.get(symbol, 0)
                }

        # Calculate time until next interval
        time_to_next = 'unknown'
        for symbol in self.symbols:
            last_exec = self.last_execution.get(symbol)
            if last_exec:
                next_exec = last_exec + timedelta(minutes=self.interval_minutes)
                remaining = (next_exec - datetime.now()).total_seconds() / 60
                if remaining > 0:
                    time_to_next = f'{remaining:.1f}m'
                    break

        return {
            'action': 'hold',
            'symbol': self.symbols[0] if self.symbols else 'BTC/USDT',
            'confidence': 0.0,
            'reason': f'TWAP: Waiting for next interval ({time_to_next} remaining)',
            'strategy': 'twap',
            'twap_status': status,
            'indicators': {
                'interval_minutes': self.interval_minutes,
                'duration_hours': self.duration_hours,
                'total_amount_usd': self.total_amount_usd,
                'active_twaps': len(self.active_twaps),
                'time_to_next_interval': time_to_next,
                'is_twap': True,
                'waiting_for_interval': True
            }
        }

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Rule-based strategy, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get TWAP execution status."""
        base = super().get_status()

        twap_details = {}
        for symbol, twap in self.active_twaps.items():
            twap_details[symbol] = {
                'total_usd': twap['total_usd'],
                'executed_usd': self.total_executed_usd.get(symbol, 0),
                'remaining_usd': twap['remaining_usd'],
                'chunks_completed': twap['completed_intervals'],
                'chunks_total': twap['total_intervals'],
                'accumulated': self.total_accumulated.get(symbol, 0),
                'vwap_achieved': self.vwap_achieved.get(symbol, 0),
                'start_time': str(twap['start_time']),
                'end_time': str(twap['end_time'])
            }

        base.update({
            'total_amount_usd': self.total_amount_usd,
            'duration_hours': self.duration_hours,
            'interval_minutes': self.interval_minutes,
            'active_twaps': twap_details
        })
        return base
