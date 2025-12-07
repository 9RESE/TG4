from models.lstm_predictor import LSTMPredictor
import pandas as pd
import numpy as np


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range for volatility filtering"""
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=period).mean().values
    return atr


def calculate_momentum(close, period=10):
    """Calculate price momentum (rate of change)"""
    roc = (close - np.roll(close, period)) / np.roll(close, period)
    roc[:period] = 0
    return roc


def detect_dip(close, lookback=20, threshold=-0.05):
    """
    Detect if price is in a dip (down > threshold from recent high).
    Returns True if current price is significantly below recent high.
    """
    if len(close) < lookback:
        return False
    recent_high = np.max(close[-lookback:])
    current = close[-1]
    drawdown = (current - recent_high) / recent_high
    return drawdown < threshold


def generate_xrp_signals(data: dict, symbol: str = 'XRP/USDT'):
    """
    Phase 7: USDT-based XRP momentum strategy with dip detection.
    Optimized for 10x leverage entries on dips.

    Args:
        data: Dict of symbol -> DataFrame with OHLCV data
        symbol: Primary trading pair (default XRP/USDT for deepest liquidity)

    Returns:
        dict with 'signal' (buy/sell/hold), 'confidence', 'leverage_ok'
    """
    if symbol not in data:
        return {'signal': 'hold', 'confidence': 0.0, 'leverage_ok': False}

    df = data[symbol]
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # Calculate indicators
    atr = calculate_atr(high, low, close, period=14)
    momentum = calculate_momentum(close, period=10)

    # Current state
    current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
    current_momentum = momentum[-1]
    avg_atr = np.nanmean(atr)

    # Volume analysis
    avg_volume = np.mean(volume[-50:]) if len(volume) >= 50 else np.mean(volume)
    volume_spike = volume[-1] > avg_volume * 2.0

    # Dip detection for leverage entries
    is_dip = detect_dip(close, lookback=24, threshold=-0.03)  # 3% dip in last 24 hours

    # LSTM prediction
    predictor = LSTMPredictor()
    train_end = int(len(close) * 0.8)
    if train_end > 60:
        predictor.train(close[:train_end])
        window = close[max(0, len(close)-100):]
        lstm_bullish = predictor.predict_signal(window)
    else:
        lstm_bullish = False

    # Signal generation
    signal = 'hold'
    confidence = 0.0
    leverage_ok = False

    # Strong buy: LSTM bullish + volume spike + volatility
    if lstm_bullish and volume_spike and current_atr > avg_atr * 0.5:
        signal = 'buy'
        confidence = 0.7

    # Dip buy with leverage: Price dipped but momentum turning positive
    if is_dip and current_momentum > -0.02 and lstm_bullish:
        signal = 'buy'
        confidence = 0.85
        leverage_ok = True  # 10x leverage appropriate on dips

    # Strong momentum buy
    if current_momentum > 0.05 and volume_spike:
        signal = 'buy'
        confidence = 0.75
        leverage_ok = current_atr < avg_atr * 1.5  # Only leverage if volatility not extreme

    # Sell signals: Negative momentum + high volatility
    if current_momentum < -0.05 and current_atr > avg_atr * 1.5:
        signal = 'sell'
        confidence = 0.6

    return {
        'signal': signal,
        'confidence': confidence,
        'leverage_ok': leverage_ok,
        'momentum': current_momentum,
        'is_dip': is_dip,
        'atr': current_atr,
        'volume_spike': volume_spike
    }


def generate_btc_signals(data: dict, symbol: str = 'BTC/USDT'):
    """
    Phase 7: BTC momentum strategy for accumulation.
    Similar to XRP but with more conservative leverage thresholds.
    """
    if symbol not in data:
        return {'signal': 'hold', 'confidence': 0.0, 'leverage_ok': False}

    df = data[symbol]
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # Calculate indicators
    atr = calculate_atr(high, low, close, period=14)
    momentum = calculate_momentum(close, period=10)

    current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
    current_momentum = momentum[-1]
    avg_atr = np.nanmean(atr)

    avg_volume = np.mean(volume[-50:]) if len(volume) >= 50 else np.mean(volume)
    volume_spike = volume[-1] > avg_volume * 1.5  # Lower threshold for BTC

    # Dip detection - BTC moves slower, use 5% threshold
    is_dip = detect_dip(close, lookback=48, threshold=-0.05)

    signal = 'hold'
    confidence = 0.0
    leverage_ok = False

    # BTC accumulation on dips
    if is_dip and current_momentum > -0.03:
        signal = 'buy'
        confidence = 0.8
        leverage_ok = True

    # Momentum buy
    if current_momentum > 0.03 and volume_spike:
        signal = 'buy'
        confidence = 0.7
        leverage_ok = current_atr < avg_atr  # Conservative leverage for BTC

    # Sell on strong negative momentum
    if current_momentum < -0.07:
        signal = 'sell'
        confidence = 0.65

    return {
        'signal': signal,
        'confidence': confidence,
        'leverage_ok': leverage_ok,
        'momentum': current_momentum,
        'is_dip': is_dip,
        'atr': current_atr,
        'volume_spike': volume_spike
    }


# Legacy function for backwards compatibility with backtester
def generate_ripple_signals(data: dict, symbol: str = 'XRP/USDT'):
    """
    Generate boolean signal series for backtesting.
    Wraps new signal generator for compatibility.
    """
    if symbol not in data:
        # Try fallback symbols
        for fallback in ['XRP/USDT', 'XRP/BTC']:
            if fallback in data:
                symbol = fallback
                break
        else:
            return pd.Series([False] * 100)

    df = data[symbol]
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    atr = calculate_atr(high, low, close, period=14)
    atr_threshold = np.nanmean(atr) * 0.5

    predictor = LSTMPredictor()
    train_end = int(len(close) * 0.8)
    if train_end > 60:
        predictor.train(close[:train_end])

    signals = []
    for i in range(train_end, len(close)):
        window = close[max(0, i-100):i]
        lstm_bullish = predictor.predict_signal(window) if train_end > 60 else False

        volume_spike = volume[i] > np.mean(volume[max(0, i-50):i]) * 2.0
        volatility_ok = atr[i] > atr_threshold if not np.isnan(atr[i]) else True

        # Dip detection
        is_dip = False
        if i >= 24:
            recent_high = np.max(close[i-24:i])
            drawdown = (close[i] - recent_high) / recent_high
            is_dip = drawdown < -0.03

        # Buy signal
        signal = (lstm_bullish and volume_spike and volatility_ok) or (is_dip and lstm_bullish)
        signals.append(signal)

    signal_series = pd.Series([False] * train_end + signals, index=df.index)
    return signal_series
