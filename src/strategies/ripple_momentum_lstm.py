from models.lstm_predictor import LSTMPredictor
import pandas as pd
import numpy as np


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range for volatility filtering"""
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=period).mean().values
    return atr


def check_rlusd_premium(data: dict, threshold=1.001):
    """
    Check if RLUSD is trading at premium (>$1.001).
    When RLUSD is at premium, it signals Ripple ecosystem demand.
    """
    if 'RLUSD/USDT' in data:
        rlusd_price = data['RLUSD/USDT']['close'].iloc[-1]
        return rlusd_price > threshold
    return False


def generate_ripple_signals(data: dict, symbol: str = 'XRP/RLUSD', rlusd_premium_boost: bool = True):
    """
    Hybrid strategy: LSTM predicts direction + volume filter + ATR volatility
    Strongly biased to accumulate XRP and RLUSD
    Tuned for RLUSD ecosystem focus

    Args:
        data: Dict of symbol -> DataFrame with OHLCV data
        symbol: Primary trading pair
        rlusd_premium_boost: If True, boost buy signals when RLUSD > $1.001
    """
    df = data[symbol]
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # Calculate ATR for volatility filter
    atr = calculate_atr(high, low, close, period=14)
    atr_threshold = np.nanmean(atr) * 0.5  # Only trade when volatility is meaningful

    # Check RLUSD premium for ecosystem demand signal
    rlusd_at_premium = check_rlusd_premium(data) if rlusd_premium_boost else False

    predictor = LSTMPredictor()

    # Train on historical (first 80%)
    train_end = int(len(close) * 0.8)
    predictor.train(close[:train_end])

    signals = []
    for i in range(train_end, len(close)):
        window = close[max(0, i-100):i]
        lstm_bullish = predictor.predict_signal(window)

        # Stronger volume confirmation (2.5x for higher conviction)
        volume_spike = volume[i] > np.mean(volume[max(0, i-50):i]) * 2.5

        # Volatility filter: only trade when ATR > threshold
        volatility_ok = atr[i] > atr_threshold if not np.isnan(atr[i]) else True

        # RLUSD premium filter: if RLUSD is at premium, lower volume requirement
        if rlusd_at_premium:
            # When RLUSD premium detected, be more aggressive on XRP buys
            volume_spike = volume[i] > np.mean(volume[max(0, i-50):i]) * 1.5

        # Buy/hold signal if LSTM bullish + volume support + volatility
        signal = lstm_bullish and volume_spike and volatility_ok

        # Extra boost: if RLUSD at premium, also trigger on just LSTM + volatility
        if rlusd_at_premium and lstm_bullish and volatility_ok:
            signal = True

        signals.append(signal)

    signal_series = pd.Series([False] * train_end + signals, index=df.index)
    return signal_series
