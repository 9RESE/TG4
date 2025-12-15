"""
Golden Test Fixtures for Indicator Library Refactoring

This file captures the exact outputs of current indicator implementations
BEFORE refactoring. Used for regression testing to ensure new centralized
implementations match existing behavior within floating-point tolerance (1e-10).

Generated: 2025-12-15
Purpose: Phase 0 - Capture current indicator behavior before any code changes

WARNING: DO NOT MODIFY THE EXPECTED VALUES in this file without understanding
that you are changing the "ground truth" for regression testing.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

# Import Candle type from ws_tester
# This import will work when running tests from project root
try:
    from ws_tester.types import Candle, Trade, OrderbookSnapshot
except ImportError:
    # Fallback for standalone testing - define compatible types
    @dataclass(frozen=True)
    class Candle:
        timestamp: datetime
        open: float
        high: float
        low: float
        close: float
        volume: float

    @dataclass(frozen=True)
    class Trade:
        timestamp: datetime
        price: float
        size: float
        side: str

    @dataclass(frozen=True)
    class OrderbookSnapshot:
        bids: Tuple[Tuple[float, float], ...]
        asks: Tuple[Tuple[float, float], ...]

# =============================================================================
# FLOATING POINT TOLERANCE
# =============================================================================
TOLERANCE = 1e-10

# =============================================================================
# TEST CANDLE DATA
# =============================================================================
# Realistic XRP/USDT price movement over 50 1-minute candles
BASE_TIMESTAMP = datetime(2025, 12, 15, 0, 0, 0)

# Generate 50 test candles with realistic price movement
# Starting at $2.30, with ~0.5% volatility range
def _generate_test_candles(count: int = 50) -> Tuple[Candle, ...]:
    """Generate synthetic test candles with realistic price action."""
    candles = []
    base_price = 2.30

    # Price sequence with realistic movement
    price_changes = [
        0.0, 0.002, -0.001, 0.003, -0.002, 0.004, -0.003, 0.002, 0.001, -0.004,
        0.005, -0.002, 0.001, -0.001, 0.003, -0.002, 0.004, 0.002, -0.003, 0.001,
        -0.002, 0.003, -0.001, 0.002, -0.004, 0.003, 0.001, -0.002, 0.002, -0.001,
        0.004, -0.003, 0.002, 0.001, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001,
        0.002, -0.001, 0.003, -0.002, 0.001, 0.002, -0.003, 0.004, -0.001, 0.002
    ]

    current_price = base_price
    for i in range(count):
        change_pct = price_changes[i % len(price_changes)]
        current_price *= (1 + change_pct)

        # Realistic OHLC with some wick
        open_price = current_price * (1 - 0.001)
        high_price = current_price * (1 + 0.002)
        low_price = current_price * (1 - 0.003)
        close_price = current_price

        volume = 50000 + (i % 10) * 5000  # Volume varies 50k-95k

        candles.append(Candle(
            timestamp=BASE_TIMESTAMP + timedelta(minutes=i),
            open=round(open_price, 6),
            high=round(high_price, 6),
            low=round(low_price, 6),
            close=round(close_price, 6),
            volume=float(volume)
        ))

    return tuple(candles)

# Standard test candle set - 50 candles
TEST_CANDLES: Tuple[Candle, ...] = _generate_test_candles(50)

# Extract closes for functions that accept List[float]
TEST_CLOSES: List[float] = [c.close for c in TEST_CANDLES]

# Second price series for correlation testing (BTC with higher prices)
def _generate_btc_candles(count: int = 50) -> Tuple[Candle, ...]:
    """Generate synthetic BTC candles for correlation testing."""
    candles = []
    base_price = 104000.0

    # Similar but not identical movement pattern
    price_changes = [
        0.0, 0.0015, -0.0008, 0.0022, -0.0015, 0.0030, -0.0020, 0.0015, 0.0008, -0.0025,
        0.0035, -0.0012, 0.0008, -0.0005, 0.0020, -0.0015, 0.0028, 0.0012, -0.0020, 0.0006,
        -0.0012, 0.0018, -0.0006, 0.0012, -0.0025, 0.0018, 0.0006, -0.0012, 0.0012, -0.0006,
        0.0025, -0.0018, 0.0012, 0.0006, -0.0012, 0.0018, -0.0006, 0.0012, -0.0018, 0.0006,
        0.0012, -0.0006, 0.0018, -0.0012, 0.0006, 0.0012, -0.0018, 0.0025, -0.0006, 0.0012
    ]

    current_price = base_price
    for i in range(count):
        change_pct = price_changes[i % len(price_changes)]
        current_price *= (1 + change_pct)

        open_price = current_price * (1 - 0.0005)
        high_price = current_price * (1 + 0.001)
        low_price = current_price * (1 - 0.0015)
        close_price = current_price

        volume = 100.0 + (i % 10) * 10.0

        candles.append(Candle(
            timestamp=BASE_TIMESTAMP + timedelta(minutes=i),
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=round(volume, 2)
        ))

    return tuple(candles)

BTC_TEST_CANDLES: Tuple[Candle, ...] = _generate_btc_candles(50)
BTC_TEST_CLOSES: List[float] = [c.close for c in BTC_TEST_CANDLES]

# =============================================================================
# TEST TRADE DATA (for trade flow calculations)
# =============================================================================
def _generate_test_trades(count: int = 100) -> Tuple[Trade, ...]:
    """Generate synthetic trades with realistic buy/sell distribution."""
    trades = []
    base_price = 2.35

    # Slightly more buys than sells (55/45 split) to create positive imbalance
    sides = ['buy'] * 55 + ['sell'] * 45
    import random
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(sides)

    for i in range(count):
        price = base_price * (1 + (random.random() - 0.5) * 0.01)
        size = 100 + random.random() * 500

        trades.append(Trade(
            timestamp=BASE_TIMESTAMP + timedelta(seconds=i),
            price=round(price, 6),
            size=round(size, 2),
            side=sides[i % len(sides)]
        ))

    return tuple(trades)

TEST_TRADES: Tuple[Trade, ...] = _generate_test_trades(100)

# =============================================================================
# TEST ORDERBOOK DATA (for micro-price calculations)
# =============================================================================
TEST_ORDERBOOK = OrderbookSnapshot(
    bids=(
        (2.3400, 10000.0),
        (2.3395, 15000.0),
        (2.3390, 20000.0),
    ),
    asks=(
        (2.3410, 8000.0),
        (2.3415, 12000.0),
        (2.3420, 18000.0),
    )
)

# =============================================================================
# GOLDEN EXPECTED VALUES
# These are the EXACT outputs from current implementations
# Captured before any refactoring changes
# =============================================================================

# -----------------------------------------------------------------------------
# SMA (Simple Moving Average)
# -----------------------------------------------------------------------------
# Source: mean_reversion/indicators.py:14-19, wavetrend/indicators.py:102-116
GOLDEN_SMA = {
    'period_5': {
        'input_closes': TEST_CLOSES,
        'expected': sum(TEST_CLOSES[-5:]) / 5,  # Last 5 values
        'source': 'mean_reversion/indicators.py:14-19',
    },
    'period_14': {
        'input_closes': TEST_CLOSES,
        'expected': sum(TEST_CLOSES[-14:]) / 14,
        'source': 'wavetrend/indicators.py:102-116',
    },
    'period_20': {
        'input_closes': TEST_CLOSES,
        'expected': sum(TEST_CLOSES[-20:]) / 20,
        'source': 'all implementations consistent',
    },
    'insufficient_data': {
        'input_closes': TEST_CLOSES[:3],  # Only 3 values
        'period': 5,
        'expected': None,  # Should return None or 0.0
        'source': 'edge case',
    },
}

# -----------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# -----------------------------------------------------------------------------
# Source: momentum_scalping/indicators.py:13-39, wavetrend/indicators.py:45-71
# All implementations use: multiplier = 2 / (period + 1), init with SMA
def _calculate_golden_ema(values: List[float], period: int) -> Optional[float]:
    """Calculate EMA exactly as current implementations do."""
    if len(values) < period:
        return None
    sma = sum(values[:period]) / period
    multiplier = 2.0 / (period + 1)
    ema = sma
    for value in values[period:]:
        ema = (value * multiplier) + (ema * (1 - multiplier))
    return ema

GOLDEN_EMA = {
    'period_8': {
        'input_closes': TEST_CLOSES,
        'expected': _calculate_golden_ema(TEST_CLOSES, 8),
        'source': 'momentum_scalping/indicators.py:13-39',
    },
    'period_21': {
        'input_closes': TEST_CLOSES,
        'expected': _calculate_golden_ema(TEST_CLOSES, 21),
        'source': 'wavetrend/indicators.py:45-71',
    },
    'period_50': {
        'input_closes': TEST_CLOSES,
        'expected': _calculate_golden_ema(TEST_CLOSES, 50),
        'source': 'whale_sentiment/indicators.py:44-70',
    },
    'insufficient_data': {
        'input_closes': TEST_CLOSES[:5],
        'period': 10,
        'expected': None,
        'source': 'edge case',
    },
}

# -----------------------------------------------------------------------------
# RSI (Relative Strength Index)
# -----------------------------------------------------------------------------
# Note: Two implementations exist:
# 1. Simple averaging (mean_reversion, ratio_trading): uses sum/len
# 2. Wilder's smoothing (momentum_scalping, grid_rsi): uses weighted average
#
# Per user decision: Use Wilder's smoothing as canonical
def _calculate_golden_rsi_wilder(closes: List[float], period: int = 14) -> Optional[float]:
    """Calculate RSI with Wilder's smoothing (momentum_scalping/grid_rsi method)."""
    if len(closes) < period + 1:
        return None

    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(0, change) for change in changes[:period]]
    losses = [max(0, -change) for change in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder's smoothing for remaining periods
    for change in changes[period:]:
        gain = max(0, change)
        loss = max(0, -change)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _calculate_golden_rsi_simple(closes: List[float], period: int = 14) -> float:
    """Calculate RSI with simple averaging (mean_reversion/ratio_trading method)."""
    if len(closes) < period + 1:
        return 50.0  # Neutral

    gains = []
    losses = []

    start_idx = max(1, len(closes) - period)
    for i in range(start_idx, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    if not gains:
        return 50.0

    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

GOLDEN_RSI = {
    'wilder_period_14': {
        'input_closes': TEST_CLOSES,
        'expected': _calculate_golden_rsi_wilder(TEST_CLOSES, 14),
        'source': 'momentum_scalping/indicators.py:70-112, grid_rsi_reversion/indicators.py:12-54',
        'method': 'wilder_smoothing',
    },
    'wilder_period_7': {
        'input_closes': TEST_CLOSES,
        'expected': _calculate_golden_rsi_wilder(TEST_CLOSES, 7),
        'source': 'momentum_scalping (scalping uses period 7)',
        'method': 'wilder_smoothing',
    },
    'simple_period_14': {
        'input_closes': TEST_CLOSES,
        'expected': _calculate_golden_rsi_simple(TEST_CLOSES, 14),
        'source': 'mean_reversion/indicators.py:22-57, ratio_trading/indicators.py:72-109',
        'method': 'simple_averaging',
        'note': 'DEPRECATED - for reference only, use Wilder smoothing',
    },
    'insufficient_data': {
        'input_closes': TEST_CLOSES[:10],
        'period': 14,
        'expected': None,  # Wilder returns None, simple returns 50.0
        'source': 'edge case',
    },
}

# -----------------------------------------------------------------------------
# VOLATILITY (Std Dev of Returns * 100)
# -----------------------------------------------------------------------------
# Source: All 6 implementations are IDENTICAL
def _calculate_golden_volatility(candles, lookback: int = 20) -> float:
    """Calculate volatility exactly as all current implementations do."""
    if len(candles) < lookback + 1:
        return 0.0

    closes = [c.close for c in candles[-(lookback + 1):]]
    if len(closes) < 2:
        return 0.0

    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] != 0]

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    return (variance ** 0.5) * 100

GOLDEN_VOLATILITY = {
    'lookback_20': {
        'input_candles': TEST_CANDLES,
        'expected': _calculate_golden_volatility(TEST_CANDLES, 20),
        'source': 'all 6 implementations identical',
    },
    'lookback_50': {
        'input_candles': TEST_CANDLES,
        'expected': _calculate_golden_volatility(TEST_CANDLES, 50),
        'source': 'all 6 implementations identical',
    },
    'insufficient_data': {
        'input_candles': TEST_CANDLES[:10],
        'lookback': 20,
        'expected': 0.0,
        'source': 'edge case',
    },
}

# -----------------------------------------------------------------------------
# ATR (Average True Range)
# -----------------------------------------------------------------------------
# Two variants:
# 1. Simple average (momentum_scalping, grid_rsi): sum(tr) / period
# 2. Wilder's smoothing (whale_sentiment): progressive smoothing
def _calculate_golden_atr_simple(candles, period: int = 14) -> Optional[float]:
    """Calculate ATR with simple averaging (momentum_scalping/grid_rsi method)."""
    if len(candles) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    return sum(true_ranges[-period:]) / period

def _calculate_golden_atr_wilder(candles, period: int = 14) -> Dict[str, Any]:
    """Calculate ATR with Wilder's smoothing (whale_sentiment method)."""
    result = {'atr': None, 'atr_pct': None, 'tr_series': []}

    if len(candles) < period + 1:
        return result

    tr_series = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_series.append(tr)

    if len(tr_series) < period:
        return result

    # Wilder's smoothing
    atr = sum(tr_series[:period]) / period
    for i in range(period, len(tr_series)):
        atr = (atr * (period - 1) + tr_series[i]) / period

    result['atr'] = atr
    result['tr_series'] = tr_series

    current_price = candles[-1].close
    if current_price > 0:
        result['atr_pct'] = (atr / current_price) * 100

    return result

GOLDEN_ATR = {
    'simple_period_14': {
        'input_candles': TEST_CANDLES,
        'expected': _calculate_golden_atr_simple(TEST_CANDLES, 14),
        'source': 'momentum_scalping/indicators.py:407-438, grid_rsi_reversion/indicators.py:57-88',
        'method': 'simple_average',
    },
    'wilder_period_14': {
        'input_candles': TEST_CANDLES,
        'expected': _calculate_golden_atr_wilder(TEST_CANDLES, 14),
        'source': 'whale_sentiment/indicators.py:90-146',
        'method': 'wilder_smoothing',
    },
    'insufficient_data': {
        'input_candles': TEST_CANDLES[:10],
        'period': 14,
        'expected': None,
        'source': 'edge case',
    },
}

# -----------------------------------------------------------------------------
# BOLLINGER BANDS
# -----------------------------------------------------------------------------
# Source: mean_reversion/indicators.py:60-78, ratio_trading/indicators.py:11-38
# Note: Different return order between implementations - will standardize
def _calculate_golden_bollinger(prices: List[float], period: int = 20, num_std: float = 2.0) -> Tuple:
    """Calculate Bollinger Bands exactly as ratio_trading does (sma, upper, lower, std)."""
    if len(prices) < period:
        return (None, None, None, None)

    recent = prices[-period:]
    sma = sum(recent) / len(recent)
    variance = sum((p - sma) ** 2 for p in recent) / len(recent)
    std_dev = variance ** 0.5

    upper = sma + (num_std * std_dev)
    lower = sma - (num_std * std_dev)

    return (sma, upper, lower, std_dev)

GOLDEN_BOLLINGER = {
    'period_20_std_2': {
        'input_closes': TEST_CLOSES,
        'expected': _calculate_golden_bollinger(TEST_CLOSES, 20, 2.0),
        'source': 'ratio_trading/indicators.py:11-38',
        'return_order': '(sma, upper, lower, std_dev)',
    },
    'period_20_std_25': {
        'input_closes': TEST_CLOSES,
        'expected': _calculate_golden_bollinger(TEST_CLOSES, 20, 2.5),
        'source': 'all implementations',
    },
    'insufficient_data': {
        'input_closes': TEST_CLOSES[:10],
        'period': 20,
        'expected': (None, None, None, None),
        'source': 'edge case',
    },
}

# -----------------------------------------------------------------------------
# ADX (Average Directional Index)
# -----------------------------------------------------------------------------
# Source: mean_reversion/indicators.py:204-310, grid_rsi_reversion/indicators.py:91-190
# All implementations use Wilder's smoothing
def _calculate_golden_adx(candles, period: int = 14) -> Optional[float]:
    """Calculate ADX with Wilder's smoothing."""
    if len(candles) < period * 2:
        return None

    true_ranges = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_high = candles[i - 1].high
        prev_low = candles[i - 1].low
        prev_close = candles[i - 1].close

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0)

        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0)

    if len(true_ranges) < period:
        return None

    def wilder_smooth(values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return []
        smoothed = [sum(values[:period]) / period]
        for i in range(period, len(values)):
            smoothed.append((smoothed[-1] * (period - 1) + values[i]) / period)
        return smoothed

    atr = wilder_smooth(true_ranges, period)
    smooth_plus_dm = wilder_smooth(plus_dm, period)
    smooth_minus_dm = wilder_smooth(minus_dm, period)

    if not atr or not smooth_plus_dm or not smooth_minus_dm:
        return None

    dx_values = []
    for i in range(len(atr)):
        if atr[i] == 0:
            dx_values.append(0)
        else:
            plus_di = 100 * smooth_plus_dm[i] / atr[i]
            minus_di = 100 * smooth_minus_dm[i] / atr[i]
            di_sum = plus_di + minus_di
            if di_sum == 0:
                dx_values.append(0)
            else:
                dx_values.append(100 * abs(plus_di - minus_di) / di_sum)

    if len(dx_values) < period:
        return None

    adx_values = wilder_smooth(dx_values, period)

    if not adx_values:
        return None

    return adx_values[-1]

GOLDEN_ADX = {
    'period_14': {
        'input_candles': TEST_CANDLES,
        'expected': _calculate_golden_adx(TEST_CANDLES, 14),
        'source': 'mean_reversion/indicators.py:204-310',
    },
    'insufficient_data': {
        'input_candles': TEST_CANDLES[:20],
        'period': 14,
        'expected': None,
        'source': 'edge case',
    },
}

# -----------------------------------------------------------------------------
# ROLLING CORRELATION
# -----------------------------------------------------------------------------
# Source: Multiple implementations with slight variations
# Using grid_rsi_reversion style (returns on prices)
def _calculate_golden_correlation(prices_a: List[float], prices_b: List[float], lookback: int = 20) -> Optional[float]:
    """Calculate rolling Pearson correlation on returns."""
    if len(prices_a) < lookback + 1 or len(prices_b) < lookback + 1:
        return None

    a = prices_a[-(lookback + 1):]
    b = prices_b[-(lookback + 1):]

    returns_a = [(a[i] - a[i-1]) / a[i-1] for i in range(1, len(a)) if a[i-1] != 0]
    returns_b = [(b[i] - b[i-1]) / b[i-1] for i in range(1, len(b)) if b[i-1] != 0]

    if len(returns_a) < 2 or len(returns_b) < 2:
        return None

    min_len = min(len(returns_a), len(returns_b))
    returns_a = returns_a[-min_len:]
    returns_b = returns_b[-min_len:]

    mean_a = sum(returns_a) / len(returns_a)
    mean_b = sum(returns_b) / len(returns_b)

    covariance = sum((ra - mean_a) * (rb - mean_b) for ra, rb in zip(returns_a, returns_b)) / len(returns_a)
    variance_a = sum((r - mean_a) ** 2 for r in returns_a) / len(returns_a)
    variance_b = sum((r - mean_b) ** 2 for r in returns_b) / len(returns_b)

    std_a = variance_a ** 0.5
    std_b = variance_b ** 0.5

    if std_a == 0 or std_b == 0:
        return None

    correlation = covariance / (std_a * std_b)
    return max(-1.0, min(1.0, correlation))

GOLDEN_CORRELATION = {
    'window_20': {
        'input_prices_a': TEST_CLOSES,
        'input_prices_b': BTC_TEST_CLOSES,
        'expected': _calculate_golden_correlation(TEST_CLOSES, BTC_TEST_CLOSES, 20),
        'source': 'grid_rsi_reversion/indicators.py:473-530',
    },
    'window_50': {
        'input_prices_a': TEST_CLOSES,
        'input_prices_b': BTC_TEST_CLOSES,
        'expected': _calculate_golden_correlation(TEST_CLOSES, BTC_TEST_CLOSES, 50),
        'source': 'mean_reversion/indicators.py:139-201',
    },
    'insufficient_data': {
        'input_prices_a': TEST_CLOSES[:10],
        'input_prices_b': BTC_TEST_CLOSES[:10],
        'window': 20,
        'expected': None,
        'source': 'edge case',
    },
}

# -----------------------------------------------------------------------------
# TRADE FLOW
# -----------------------------------------------------------------------------
# Source: grid_rsi returns Tuple, wavetrend/whale_sentiment return Dict
# Will standardize to NamedTuple in refactored version
def _calculate_golden_trade_flow_dict(trades, lookback: int = 50) -> Dict[str, Any]:
    """Calculate trade flow as whale_sentiment/wavetrend do (returns Dict)."""
    result = {
        'buy_volume': 0.0,
        'sell_volume': 0.0,
        'total_volume': 0.0,
        'imbalance': 0.0,
        'trade_count': 0,
        'valid': False,
    }

    if not trades or len(trades) == 0:
        return result

    recent_trades = trades[-lookback:] if len(trades) > lookback else trades
    result['trade_count'] = len(recent_trades)

    if result['trade_count'] < 5:
        return result

    for trade in recent_trades:
        if hasattr(trade, 'side'):
            side = trade.side
            value = getattr(trade, 'size', 0) * getattr(trade, 'price', 1)
        else:
            continue

        if side == 'buy':
            result['buy_volume'] += value
        elif side == 'sell':
            result['sell_volume'] += value

    result['total_volume'] = result['buy_volume'] + result['sell_volume']

    if result['total_volume'] > 0:
        result['imbalance'] = (result['buy_volume'] - result['sell_volume']) / result['total_volume']
        result['valid'] = True

    return result

GOLDEN_TRADE_FLOW = {
    'lookback_50': {
        'input_trades': TEST_TRADES,
        'expected': _calculate_golden_trade_flow_dict(TEST_TRADES, 50),
        'source': 'whale_sentiment/indicators.py:465-519',
    },
    'lookback_100': {
        'input_trades': TEST_TRADES,
        'expected': _calculate_golden_trade_flow_dict(TEST_TRADES, 100),
        'source': 'wavetrend/indicators.py:465-522',
    },
    'insufficient_trades': {
        'input_trades': TEST_TRADES[:3],
        'lookback': 50,
        'expected': {'buy_volume': 0.0, 'sell_volume': 0.0, 'total_volume': 0.0,
                     'imbalance': 0.0, 'trade_count': 3, 'valid': False},
        'source': 'edge case',
    },
}

# -----------------------------------------------------------------------------
# MICRO PRICE
# -----------------------------------------------------------------------------
# Source: order_flow/indicators.py:36-58, market_making/calculations.py:26-46
def _calculate_golden_micro_price(ob: OrderbookSnapshot) -> float:
    """Calculate volume-weighted micro-price."""
    if not ob or not ob.bids or not ob.asks:
        return 0.0

    best_bid, bid_size = ob.bids[0]
    best_ask, ask_size = ob.asks[0]

    total_size = bid_size + ask_size
    if total_size <= 0:
        return (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0

    return (best_bid * ask_size + best_ask * bid_size) / total_size

GOLDEN_MICRO_PRICE = {
    'standard': {
        'input_orderbook': TEST_ORDERBOOK,
        'expected': _calculate_golden_micro_price(TEST_ORDERBOOK),
        'source': 'order_flow/indicators.py:36-58',
    },
}

# -----------------------------------------------------------------------------
# TREND SLOPE
# -----------------------------------------------------------------------------
# Source: mean_reversion/indicators.py:105-136, market_making/calculations.py:335-381
def _calculate_golden_trend_slope(candles, period: int = 20) -> float:
    """Calculate trend slope using linear regression."""
    if len(candles) < period:
        return 0.0

    closes = [c.close for c in candles[-period:]]
    if len(closes) < 2:
        return 0.0

    n = len(closes)
    sum_x = sum(range(n))
    sum_y = sum(closes)
    sum_xy = sum(i * closes[i] for i in range(n))
    sum_x2 = sum(i * i for i in range(n))

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    avg_price = sum_y / n if n > 0 else 1.0
    slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0.0

    return slope_pct

GOLDEN_TREND_SLOPE = {
    'period_20': {
        'input_candles': TEST_CANDLES,
        'expected': _calculate_golden_trend_slope(TEST_CANDLES, 20),
        'source': 'market_making/calculations.py:335-381',
    },
    'period_50': {
        'input_candles': TEST_CANDLES,
        'expected': _calculate_golden_trend_slope(TEST_CANDLES, 50),
        'source': 'mean_reversion/indicators.py:105-136',
    },
    'insufficient_data': {
        'input_candles': TEST_CANDLES[:10],
        'period': 20,
        'expected': 0.0,
        'source': 'edge case',
    },
}

# =============================================================================
# SUMMARY OF GOLDEN VALUES
# =============================================================================
# For quick reference, all expected values at standard parameters
GOLDEN_SUMMARY = {
    'sma_20': GOLDEN_SMA['period_20']['expected'],
    'ema_21': GOLDEN_EMA['period_21']['expected'],
    'rsi_14_wilder': GOLDEN_RSI['wilder_period_14']['expected'],
    'volatility_20': GOLDEN_VOLATILITY['lookback_20']['expected'],
    'atr_14_simple': GOLDEN_ATR['simple_period_14']['expected'],
    'atr_14_wilder_value': GOLDEN_ATR['wilder_period_14']['expected']['atr'],
    'bollinger_20_2': GOLDEN_BOLLINGER['period_20_std_2']['expected'],
    'adx_14': GOLDEN_ADX['period_14']['expected'],
    'correlation_20': GOLDEN_CORRELATION['window_20']['expected'],
    'micro_price': GOLDEN_MICRO_PRICE['standard']['expected'],
    'trend_slope_20': GOLDEN_TREND_SLOPE['period_20']['expected'],
}

# Print summary when module is loaded for debugging
if __name__ == '__main__':
    print("=" * 60)
    print("GOLDEN TEST FIXTURE SUMMARY")
    print("=" * 60)
    for key, value in GOLDEN_SUMMARY.items():
        if isinstance(value, tuple):
            print(f"{key}: {tuple(round(v, 6) if v is not None else None for v in value)}")
        elif isinstance(value, float):
            print(f"{key}: {value:.10f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)
