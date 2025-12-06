from models.lstm_predictor import LSTMPredictor
import pandas as pd
import numpy as np

def generate_ripple_signals(data: pd.DataFrame, symbol: str = 'XRP/RLUSD'):
    """
    Hybrid strategy: LSTM predicts direction + volume filter
    Strongly biased to accumulate XRP and RLUSD
    """
    close = data[symbol]['close'].values
    volume = data[symbol]['volume'].values

    predictor = LSTMPredictor()

    # Train on historical (first 80%)
    train_end = int(len(close) * 0.8)
    predictor.train(close[:train_end])

    signals = []
    for i in range(train_end, len(close)):
        window = close[max(0, i-100):i]
        lstm_bullish = predictor.predict_signal(window)
        volume_spike = volume[i] > np.mean(volume[max(0, i-50):i]) * 1.5

        # Buy/hold signal if LSTM bullish + volume support
        signal = lstm_bullish and volume_spike
        signals.append(signal)

    signal_series = pd.Series([False] * train_end + signals, index=data[symbol].index)
    return signal_series
