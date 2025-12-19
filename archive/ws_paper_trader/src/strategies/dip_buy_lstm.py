"""
Phase 8: Hybrid Dip-Buy LSTM Strategy
LSTM flags oversold dips (RSI <30 + volume surge) for RL confirmation.
Complements the RL agent by providing high-confidence entry signals.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class DipDetectorLSTM(nn.Module):
    """
    LSTM model for detecting high-probability dip entries.
    Outputs probability of price recovery within next N periods.
    """

    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output: probability of successful dip recovery
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


def calculate_rsi(close: np.ndarray, period: int = 14) -> float:
    """Calculate RSI indicator."""
    if len(close) < period + 1:
        return 50.0  # Neutral

    deltas = np.diff(close[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_volume_surge(volume: np.ndarray, lookback: int = 20) -> float:
    """Calculate volume surge ratio vs recent average."""
    if len(volume) < lookback:
        return 1.0

    avg_volume = np.mean(volume[-lookback:-1])
    current_volume = volume[-1]

    return current_volume / avg_volume if avg_volume > 0 else 1.0


def calculate_drawdown(close: np.ndarray, lookback: int = 20) -> float:
    """Calculate current drawdown from recent high."""
    if len(close) < lookback:
        return 0.0

    recent_high = np.max(close[-lookback:])
    current = close[-1]

    return (current - recent_high) / recent_high if recent_high > 0 else 0.0


def detect_dip_signal(
    close: np.ndarray,
    volume: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    rsi_threshold: float = 30.0,
    volume_surge_threshold: float = 1.5,
    drawdown_threshold: float = -0.05
) -> Dict:
    """
    Detect oversold dip conditions for potential entry.

    Args:
        close: Close prices array
        volume: Volume array
        high: High prices array
        low: Low prices array
        rsi_threshold: RSI below this = oversold
        volume_surge_threshold: Volume surge above this = capitulation
        drawdown_threshold: Drawdown below this = significant dip

    Returns:
        dict: Signal details with confidence score
    """
    rsi = calculate_rsi(close)
    vol_surge = calculate_volume_surge(volume)
    drawdown = calculate_drawdown(close)

    # Calculate ATR for volatility context
    atr = 0.0
    if len(close) >= 15:
        tr_list = []
        for i in range(-14, 0):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        atr = np.mean(tr_list)

    atr_pct = atr / close[-1] if close[-1] > 0 else 0.05

    # Dip conditions
    is_oversold = rsi < rsi_threshold
    has_volume_surge = vol_surge > volume_surge_threshold
    has_significant_drawdown = drawdown < drawdown_threshold

    # Calculate confidence score (0-1)
    confidence = 0.0

    # RSI contribution (0-0.4)
    if rsi < 20:
        confidence += 0.4
    elif rsi < 25:
        confidence += 0.3
    elif rsi < 30:
        confidence += 0.2
    elif rsi < 35:
        confidence += 0.1

    # Volume surge contribution (0-0.3)
    if vol_surge > 3.0:
        confidence += 0.3
    elif vol_surge > 2.0:
        confidence += 0.2
    elif vol_surge > 1.5:
        confidence += 0.1

    # Drawdown contribution (0-0.3)
    if drawdown < -0.10:
        confidence += 0.3
    elif drawdown < -0.07:
        confidence += 0.2
    elif drawdown < -0.05:
        confidence += 0.1

    # Is this a high-probability dip entry?
    is_dip = is_oversold and (has_volume_surge or has_significant_drawdown)
    leverage_ok = is_dip and confidence > 0.6 and atr_pct < 0.08

    return {
        'is_dip': is_dip,
        'confidence': confidence,
        'leverage_ok': leverage_ok,
        'rsi': rsi,
        'volume_surge': vol_surge,
        'drawdown': drawdown,
        'atr_pct': atr_pct,
        'details': {
            'is_oversold': is_oversold,
            'has_volume_surge': has_volume_surge,
            'has_significant_drawdown': has_significant_drawdown
        }
    }


def generate_dip_signals(data: Dict, symbol: str = 'XRP/USDT') -> Dict:
    """
    Generate dip-buy signals for a symbol.
    Hybrid approach: LSTM features + rule-based confirmation.

    Args:
        data: Dictionary of DataFrames keyed by symbol
        symbol: Symbol to analyze

    Returns:
        dict: Signal with confidence and entry recommendation
    """
    if symbol not in data:
        return {
            'signal': 'none',
            'confidence': 0.0,
            'is_dip': False,
            'leverage_ok': False
        }

    df = data[symbol]
    if len(df) < 50:
        return {
            'signal': 'none',
            'confidence': 0.0,
            'is_dip': False,
            'leverage_ok': False
        }

    close = df['close'].values
    volume = df['volume'].values
    high = df['high'].values
    low = df['low'].values

    # Get dip signal
    dip_signal = detect_dip_signal(close, volume, high, low)

    # Determine signal type
    if dip_signal['is_dip'] and dip_signal['confidence'] > 0.8:
        signal = 'strong_buy'
    elif dip_signal['is_dip'] and dip_signal['confidence'] > 0.6:
        signal = 'buy'
    elif dip_signal['is_dip']:
        signal = 'weak_buy'
    else:
        signal = 'hold'

    return {
        'signal': signal,
        'confidence': dip_signal['confidence'],
        'is_dip': dip_signal['is_dip'],
        'leverage_ok': dip_signal['leverage_ok'],
        'rsi': dip_signal['rsi'],
        'volume_surge': dip_signal['volume_surge'],
        'drawdown': dip_signal['drawdown'],
        'atr_pct': dip_signal['atr_pct']
    }


def combine_with_rl_decision(
    dip_signal: Dict,
    rl_action: str,
    rl_confidence: float = 0.5
) -> Dict:
    """
    Combine LSTM dip signal with RL agent decision.
    Both must agree for leverage execution.

    Args:
        dip_signal: Signal from generate_dip_signals
        rl_action: Action string from RL agent ('buy', 'hold', 'sell')
        rl_confidence: Confidence from RL model (optional)

    Returns:
        dict: Combined recommendation
    """
    lstm_confidence = dip_signal.get('confidence', 0)
    is_dip = dip_signal.get('is_dip', False)
    leverage_ok = dip_signal.get('leverage_ok', False)

    # Both must agree on buy for leverage
    both_agree_buy = is_dip and rl_action == 'buy'

    # Combined confidence (weighted average)
    combined_conf = lstm_confidence * 0.6 + rl_confidence * 0.4

    # Final recommendation
    if both_agree_buy and combined_conf > 0.8 and leverage_ok:
        recommendation = 'leverage_long'
        final_confidence = combined_conf
    elif both_agree_buy and combined_conf > 0.6:
        recommendation = 'spot_buy'
        final_confidence = combined_conf
    elif is_dip and combined_conf > 0.5:
        recommendation = 'small_buy'
        final_confidence = lstm_confidence
    else:
        recommendation = 'hold'
        final_confidence = 0.0

    return {
        'recommendation': recommendation,
        'confidence': final_confidence,
        'lstm_signal': dip_signal.get('signal', 'none'),
        'rl_action': rl_action,
        'agreement': both_agree_buy,
        'leverage_approved': both_agree_buy and leverage_ok and combined_conf > 0.8
    }
