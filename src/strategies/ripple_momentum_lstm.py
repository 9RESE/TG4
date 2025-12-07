from models.lstm_predictor import LSTMPredictor
import pandas as pd
import numpy as np


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range for volatility filtering"""
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=period).mean().values
    return atr


def generate_ripple_signals(data: pd.DataFrame, symbol: str = 'XRP/RLUSD'):
    """
    Hybrid strategy: LSTM predicts direction + volume filter + ATR volatility
    Strongly biased to accumulate XRP and RLUSD
    Tuned for RLUSD ecosystem focus
    """
    df = data[symbol]
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # Calculate ATR for volatility filter
    atr = calculate_atr(high, low, close, period=14)
    atr_threshold = np.nanmean(atr) * 0.5  # Only trade when volatility is meaningful

    predictor = LSTMPredictor()

    # Train on historical (first 80%)
    train_end = int(len(close) * 0.8)
    predictor.train(close[:train_end])

    signals = []
    for i in range(train_end, len(close)):
        window = close[max(0, i-100):i]
        lstm_bullish = predictor.predict_signal(window)

        # Stronger volume confirmation (2.0x instead of 1.5x)
        volume_spike = volume[i] > np.mean(volume[max(0, i-50):i]) * 2.0

        # Volatility filter: only trade when ATR > threshold
        volatility_ok = atr[i] > atr_threshold if not np.isnan(atr[i]) else True

        # Buy/hold signal if LSTM bullish + volume support + volatility
        signal = lstm_bullish and volume_spike and volatility_ok
        signals.append(signal)

    signal_series = pd.Series([False] * train_end + signals, index=df.index)
    return signal_series
