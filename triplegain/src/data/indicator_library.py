"""
Indicator Library - Technical indicator calculations for TripleGain.

This module provides centralized, pre-computed technical indicators
for use by LLM agents. All calculations use numpy for performance.

Indicators are NOT calculated by LLMs - they use these pre-computed values.
"""

import logging
import time
from decimal import Decimal
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class IndicatorLibrary:
    """
    Central library for technical indicator calculations.

    All calculations use numpy for performance.
    Results can be cached to indicator_cache table.

    Warmup Periods:
        Different indicators require different amounts of historical data
        before producing valid values. Values before the warmup period are NaN.

        | Indicator | First Valid Index | Notes |
        |-----------|-------------------|-------|
        | SMA       | period - 1        | Simple moving average |
        | EMA       | period - 1        | Starts with SMA seed |
        | RSI       | period            | Needs period+1 price changes |
        | ATR       | period            | Needs period true ranges |
        | ADX       | period * 2 - 1    | DI smoothing + ADX smoothing |
        | MACD      | slow + signal - 1 | Depends on slow EMA + signal |
        | Bollinger | period - 1        | Same as SMA |
        | Choppiness| period            | Needs period of TR and range |
        | Supertrend| period            | Depends on ATR warmup |
        | Stoch RSI | rsi_period + stoch_period + k_period + d_period - 3 | Multiple smoothing |
        | ROC       | period            | Needs period lookback |
        | VWAP      | 0                 | Cumulative, valid from start |
        | OBV       | 0                 | Cumulative, valid from start |

        The calculate_all() method returns the most recent valid value for each
        indicator, handling NaN values appropriately.
    """

    def __init__(self, config: dict, db_pool=None):
        """
        Initialize IndicatorLibrary.

        Args:
            config: Indicator configuration dictionary
            db_pool: Optional database pool for caching
        """
        self.config = config
        self.db = db_pool
        self._cache_enabled = db_pool is not None

    def calculate_all(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict]
    ) -> dict[str, any]:
        """
        Calculate all configured indicators for a symbol/timeframe.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1h")
            candles: List of OHLCV candles (oldest first)

        Returns:
            Dictionary of indicator_name -> value
        """
        start_time = time.perf_counter()

        if not candles:
            logger.debug(f"No candles provided for {symbol}/{timeframe}")
            return {}

        # Extract OHLCV arrays
        opens = [float(c.get('open', 0)) for c in candles]
        highs = [float(c.get('high', 0)) for c in candles]
        lows = [float(c.get('low', 0)) for c in candles]
        closes = [float(c.get('close', 0)) for c in candles]
        volumes = [float(c.get('volume', 0)) for c in candles]

        results = {}

        # EMA calculations
        ema_config = self.config.get('ema', {})
        for period in ema_config.get('periods', [9, 21, 50, 200]):
            ema = self.calculate_ema(closes, period)
            results[f'ema_{period}'] = float(ema[-1]) if not np.isnan(ema[-1]) else None

        # SMA calculations
        sma_config = self.config.get('sma', {})
        for period in sma_config.get('periods', [20, 50, 200]):
            sma = self.calculate_sma(closes, period)
            results[f'sma_{period}'] = float(sma[-1]) if not np.isnan(sma[-1]) else None

        # RSI
        rsi_period = self.config.get('rsi', {}).get('period', 14)
        rsi = self.calculate_rsi(closes, rsi_period)
        results[f'rsi_{rsi_period}'] = float(rsi[-1]) if not np.isnan(rsi[-1]) else None

        # MACD
        macd_config = self.config.get('macd', {})
        macd = self.calculate_macd(
            closes,
            macd_config.get('fast_period', 12),
            macd_config.get('slow_period', 26),
            macd_config.get('signal_period', 9)
        )
        results['macd'] = {
            'line': float(macd['line'][-1]) if not np.isnan(macd['line'][-1]) else None,
            'signal': float(macd['signal'][-1]) if not np.isnan(macd['signal'][-1]) else None,
            'histogram': float(macd['histogram'][-1]) if not np.isnan(macd['histogram'][-1]) else None,
        }

        # ATR
        atr_period = self.config.get('atr', {}).get('period', 14)
        atr = self.calculate_atr(highs, lows, closes, atr_period)
        results[f'atr_{atr_period}'] = float(atr[-1]) if not np.isnan(atr[-1]) else None

        # ADX
        adx_period = self.config.get('adx', {}).get('period', 14)
        adx = self.calculate_adx(highs, lows, closes, adx_period)
        results[f'adx_{adx_period}'] = float(adx[-1]) if not np.isnan(adx[-1]) else None

        # Bollinger Bands
        bb_config = self.config.get('bollinger_bands', {})
        bb = self.calculate_bollinger_bands(
            closes,
            bb_config.get('period', 20),
            bb_config.get('std_dev', 2.0)
        )
        results['bollinger_bands'] = {
            'upper': float(bb['upper'][-1]) if not np.isnan(bb['upper'][-1]) else None,
            'middle': float(bb['middle'][-1]) if not np.isnan(bb['middle'][-1]) else None,
            'lower': float(bb['lower'][-1]) if not np.isnan(bb['lower'][-1]) else None,
            'width': float(bb['width'][-1]) if not np.isnan(bb['width'][-1]) else None,
            'position': float(bb['position'][-1]) if not np.isnan(bb['position'][-1]) else None,
        }

        # OBV
        obv = self.calculate_obv(closes, volumes)
        results['obv'] = float(obv[-1]) if not np.isnan(obv[-1]) else None

        # Choppiness
        chop_period = self.config.get('choppiness', {}).get('period', 14)
        chop = self.calculate_choppiness(highs, lows, closes, chop_period)
        results[f'choppiness_{chop_period}'] = float(chop[-1]) if not np.isnan(chop[-1]) else None

        # Squeeze detection
        squeeze = self.detect_squeeze(
            closes, highs, lows,
            bb_config={'period': 20, 'std_dev': 2.0},
            kc_config={'period': 20, 'mult': 1.5}
        )
        results['squeeze_detected'] = squeeze

        # VWAP
        vwap = self.calculate_vwap(highs, lows, closes, volumes)
        results['vwap'] = float(vwap[-1]) if not np.isnan(vwap[-1]) else None

        # Supertrend
        st_config = self.config.get('supertrend', {})
        supertrend = self.calculate_supertrend(
            highs, lows, closes,
            st_config.get('period', 10),
            st_config.get('multiplier', 3.0)
        )
        if not np.isnan(supertrend['supertrend'][-1]):
            results['supertrend'] = {
                'value': float(supertrend['supertrend'][-1]),
                'direction': int(supertrend['direction'][-1])
            }
        else:
            results['supertrend'] = None

        # Stochastic RSI
        stoch_config = self.config.get('stochastic_rsi', {})
        stoch_rsi = self.calculate_stochastic_rsi(
            closes,
            stoch_config.get('rsi_period', 14),
            stoch_config.get('stoch_period', 14),
            stoch_config.get('k_period', 3),
            stoch_config.get('d_period', 3)
        )
        results['stochastic_rsi'] = {
            'k': float(stoch_rsi['k'][-1]) if not np.isnan(stoch_rsi['k'][-1]) else None,
            'd': float(stoch_rsi['d'][-1]) if not np.isnan(stoch_rsi['d'][-1]) else None,
        }

        # ROC (Rate of Change)
        roc_period = self.config.get('roc', {}).get('period', 10)
        roc = self.calculate_roc(closes, roc_period)
        results[f'roc_{roc_period}'] = float(roc[-1]) if not np.isnan(roc[-1]) else None

        # Volume SMA
        vol_sma_config = self.config.get('volume_sma', {})
        vol_sma_period = vol_sma_config.get('period', 20)
        vol_sma = self.calculate_sma(volumes, vol_sma_period)
        results[f'volume_sma_{vol_sma_period}'] = float(vol_sma[-1]) if not np.isnan(vol_sma[-1]) else None

        # Volume vs average ratio
        if vol_sma[-1] and not np.isnan(vol_sma[-1]) and vol_sma[-1] != 0:
            results['volume_vs_avg'] = float(volumes[-1] / vol_sma[-1])
        else:
            results['volume_vs_avg'] = None

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Calculated {len(results)} indicators for {symbol}/{timeframe} "
            f"({len(candles)} candles) in {elapsed_ms:.2f}ms"
        )

        return results

    def calculate_ema(self, closes: list, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.

        Args:
            closes: List of closing prices
            period: EMA period

        Returns:
            numpy array of EMA values
        """
        if not closes:
            raise ValueError("Input data cannot be empty")
        if period <= 0:
            raise ValueError("Period must be positive")

        closes = np.array(closes, dtype=float)
        n = len(closes)

        if n < period:
            return np.full(n, np.nan)

        result = np.full(n, np.nan)
        multiplier = 2.0 / (period + 1)

        # Initialize with SMA
        result[period - 1] = np.mean(closes[:period])

        # Calculate EMA
        for i in range(period, n):
            result[i] = (closes[i] - result[i - 1]) * multiplier + result[i - 1]

        return result

    def calculate_sma(self, closes: list, period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average.

        Args:
            closes: List of closing prices
            period: SMA period

        Returns:
            numpy array of SMA values
        """
        if not closes:
            raise ValueError("Input data cannot be empty")
        if period <= 0:
            raise ValueError("Period must be positive")

        closes = np.array(closes, dtype=float)
        n = len(closes)

        if n < period:
            return np.full(n, np.nan)

        result = np.full(n, np.nan)

        for i in range(period - 1, n):
            result[i] = np.mean(closes[i - period + 1:i + 1])

        return result

    def calculate_rsi(self, closes: list, period: int) -> np.ndarray:
        """
        Calculate Relative Strength Index.

        Args:
            closes: List of closing prices
            period: RSI period

        Returns:
            numpy array of RSI values (0-100)
        """
        if not closes:
            raise ValueError("Input data cannot be empty")
        if period <= 0:
            raise ValueError("Period must be positive")

        closes = np.array(closes, dtype=float)
        n = len(closes)

        if n < period + 1:
            return np.full(n, np.nan)

        result = np.full(n, np.nan)

        # Calculate price changes
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Initial average gain/loss
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))

        # Calculate subsequent RSI values using smoothed average
        for i in range(period, n - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                result[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return result

    def calculate_macd(
        self,
        closes: list,
        fast: int,
        slow: int,
        signal: int
    ) -> dict:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            closes: List of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period

        Returns:
            Dictionary with 'line', 'signal', and 'histogram' arrays
        """
        if not closes:
            raise ValueError("Input data cannot be empty")

        closes = np.array(closes, dtype=float)
        n = len(closes)

        # Calculate fast and slow EMAs
        fast_ema = self.calculate_ema(closes.tolist(), fast)
        slow_ema = self.calculate_ema(closes.tolist(), slow)

        # MACD line
        macd_line = fast_ema - slow_ema

        # Signal line (EMA of MACD line)
        signal_line = np.full(n, np.nan)

        # Find first valid MACD value
        first_valid = slow - 1
        if first_valid + signal <= n:
            signal_ema_start = first_valid + signal - 1
            multiplier = 2.0 / (signal + 1)

            # Initialize signal line with SMA of MACD
            valid_macd = macd_line[first_valid:first_valid + signal]
            signal_line[signal_ema_start] = np.nanmean(valid_macd)

            for i in range(signal_ema_start + 1, n):
                if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i - 1]):
                    signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]

        # Histogram
        histogram = macd_line - signal_line

        return {
            'line': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_atr(
        self,
        highs: list,
        lows: list,
        closes: list,
        period: int
    ) -> np.ndarray:
        """
        Calculate Average True Range.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ATR period

        Returns:
            numpy array of ATR values
        """
        if not closes:
            raise ValueError("Input data cannot be empty")
        if period <= 0:
            raise ValueError("Period must be positive")

        highs = np.array(highs, dtype=float)
        lows = np.array(lows, dtype=float)
        closes = np.array(closes, dtype=float)
        n = len(closes)

        if n < period + 1:
            return np.full(n, np.nan)

        result = np.full(n, np.nan)

        # Calculate True Range
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]

        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        # Initial ATR
        result[period] = np.mean(tr[1:period + 1])

        # Smoothed ATR
        for i in range(period + 1, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

        return result

    def calculate_bollinger_bands(
        self,
        closes: list,
        period: int,
        std_dev: float
    ) -> dict:
        """
        Calculate Bollinger Bands.

        Args:
            closes: List of closing prices
            period: Period for SMA and std calculation
            std_dev: Standard deviation multiplier

        Returns:
            Dictionary with 'upper', 'middle', 'lower', 'width', 'position'
        """
        if not closes:
            raise ValueError("Input data cannot be empty")
        if period <= 0:
            raise ValueError("Period must be positive")

        closes = np.array(closes, dtype=float)
        n = len(closes)

        middle = self.calculate_sma(closes.tolist(), period)

        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        width = np.full(n, np.nan)
        position = np.full(n, np.nan)

        for i in range(period - 1, n):
            std = np.std(closes[i - period + 1:i + 1], ddof=0)
            upper[i] = middle[i] + std_dev * std
            lower[i] = middle[i] - std_dev * std

            band_width = upper[i] - lower[i]
            if middle[i] != 0:
                width[i] = band_width / middle[i]

            if band_width != 0:
                position[i] = (closes[i] - lower[i]) / band_width

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'position': position
        }

    def calculate_adx(
        self,
        highs: list,
        lows: list,
        closes: list,
        period: int
    ) -> np.ndarray:
        """
        Calculate Average Directional Index.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ADX period

        Returns:
            numpy array of ADX values (0-100)
        """
        if not closes:
            raise ValueError("Input data cannot be empty")
        if period <= 0:
            raise ValueError("Period must be positive")

        highs = np.array(highs, dtype=float)
        lows = np.array(lows, dtype=float)
        closes = np.array(closes, dtype=float)
        n = len(closes)

        if n < period * 2:
            return np.full(n, np.nan)

        result = np.full(n, np.nan)

        # Calculate True Range
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]

        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        # Calculate +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smooth TR, +DM, -DM
        smoothed_tr = np.zeros(n)
        smoothed_plus_dm = np.zeros(n)
        smoothed_minus_dm = np.zeros(n)

        # Initial smoothed values
        smoothed_tr[period] = np.sum(tr[1:period + 1])
        smoothed_plus_dm[period] = np.sum(plus_dm[1:period + 1])
        smoothed_minus_dm[period] = np.sum(minus_dm[1:period + 1])

        for i in range(period + 1, n):
            smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / period) + tr[i]
            smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / period) + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / period) + minus_dm[i]

        # Calculate +DI and -DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)

        for i in range(period, n):
            if smoothed_tr[i] != 0:
                plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
                minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]

            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        # Calculate ADX
        adx_start = period * 2 - 1
        if adx_start < n:
            result[adx_start] = np.mean(dx[period:adx_start + 1])

            for i in range(adx_start + 1, n):
                result[i] = (result[i - 1] * (period - 1) + dx[i]) / period

        return result

    def calculate_obv(self, closes: list, volumes: list) -> np.ndarray:
        """
        Calculate On-Balance Volume.

        Args:
            closes: List of closing prices
            volumes: List of volumes

        Returns:
            numpy array of OBV values
        """
        if not closes:
            raise ValueError("Input data cannot be empty")

        closes = np.array(closes, dtype=float)
        volumes = np.array(volumes, dtype=float)
        n = len(closes)

        result = np.zeros(n)

        for i in range(1, n):
            if closes[i] > closes[i - 1]:
                result[i] = result[i - 1] + volumes[i]
            elif closes[i] < closes[i - 1]:
                result[i] = result[i - 1] - volumes[i]
            else:
                result[i] = result[i - 1]

        return result

    def calculate_vwap(
        self,
        highs: list,
        lows: list,
        closes: list,
        volumes: list
    ) -> np.ndarray:
        """
        Calculate Volume Weighted Average Price.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            volumes: List of volumes

        Returns:
            numpy array of VWAP values
        """
        if not closes:
            raise ValueError("Input data cannot be empty")

        highs = np.array(highs, dtype=float)
        lows = np.array(lows, dtype=float)
        closes = np.array(closes, dtype=float)
        volumes = np.array(volumes, dtype=float)
        n = len(closes)

        typical_price = (highs + lows + closes) / 3
        cumulative_tpv = np.cumsum(typical_price * volumes)
        cumulative_vol = np.cumsum(volumes)

        result = np.zeros(n)
        for i in range(n):
            if cumulative_vol[i] != 0:
                result[i] = cumulative_tpv[i] / cumulative_vol[i]

        return result

    def calculate_choppiness(
        self,
        highs: list,
        lows: list,
        closes: list,
        period: int
    ) -> np.ndarray:
        """
        Calculate Choppiness Index.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: Choppiness period

        Returns:
            numpy array of Choppiness values (0-100)
        """
        if not closes:
            raise ValueError("Input data cannot be empty")
        if period <= 0:
            raise ValueError("Period must be positive")

        highs = np.array(highs, dtype=float)
        lows = np.array(lows, dtype=float)
        closes = np.array(closes, dtype=float)
        n = len(closes)

        if n < period + 1:
            return np.full(n, np.nan)

        result = np.full(n, np.nan)

        # Calculate True Range
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]

        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        # Calculate Choppiness
        for i in range(period, n):
            atr_sum = np.sum(tr[i - period + 1:i + 1])
            highest_high = np.max(highs[i - period + 1:i + 1])
            lowest_low = np.min(lows[i - period + 1:i + 1])
            hl_range = highest_high - lowest_low

            if hl_range != 0 and atr_sum != 0:
                result[i] = 100 * np.log10(atr_sum / hl_range) / np.log10(period)

        return result

    def calculate_keltner_channels(
        self,
        highs: list,
        lows: list,
        closes: list,
        ema_period: int,
        atr_period: int,
        multiplier: float
    ) -> dict:
        """
        Calculate Keltner Channels.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            ema_period: EMA period for middle line
            atr_period: ATR period
            multiplier: ATR multiplier for bands

        Returns:
            Dictionary with 'upper', 'middle', 'lower'
        """
        middle = self.calculate_ema(closes, ema_period)
        atr = self.calculate_atr(highs, lows, closes, atr_period)

        upper = middle + multiplier * atr
        lower = middle - multiplier * atr

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    def detect_squeeze(
        self,
        closes: list,
        highs: list,
        lows: list,
        bb_config: dict,
        kc_config: dict
    ) -> bool:
        """
        Detect Bollinger Band / Keltner Channel squeeze.

        Squeeze occurs when BB is inside KC (low volatility, potential breakout).

        Args:
            closes: List of closing prices
            highs: List of high prices
            lows: List of low prices
            bb_config: Bollinger Bands config {'period': 20, 'std_dev': 2.0}
            kc_config: Keltner Channels config {'period': 20, 'mult': 1.5}

        Returns:
            True if squeeze is detected
        """
        if len(closes) < max(bb_config.get('period', 20), kc_config.get('period', 20)):
            return False

        bb = self.calculate_bollinger_bands(
            closes,
            bb_config.get('period', 20),
            bb_config.get('std_dev', 2.0)
        )

        kc = self.calculate_keltner_channels(
            highs, lows, closes,
            kc_config.get('period', 20),
            kc_config.get('period', 20),
            kc_config.get('mult', 1.5)
        )

        # Check if BB is inside KC (squeeze)
        bb_lower = bb['lower'][-1]
        bb_upper = bb['upper'][-1]
        kc_lower = kc['lower'][-1]
        kc_upper = kc['upper'][-1]

        if np.isnan(bb_lower) or np.isnan(kc_lower):
            return False

        return bool(bb_lower > kc_lower and bb_upper < kc_upper)

    def calculate_stochastic_rsi(
        self,
        closes: list,
        rsi_period: int,
        stoch_period: int,
        k_period: int,
        d_period: int
    ) -> dict:
        """
        Calculate Stochastic RSI.

        Args:
            closes: List of closing prices
            rsi_period: RSI period
            stoch_period: Stochastic period
            k_period: %K smoothing period
            d_period: %D smoothing period

        Returns:
            Dictionary with 'k' and 'd' arrays
        """
        rsi = self.calculate_rsi(closes, rsi_period)
        n = len(closes)

        k = np.full(n, np.nan)
        d = np.full(n, np.nan)

        for i in range(rsi_period + stoch_period - 1, n):
            rsi_window = rsi[i - stoch_period + 1:i + 1]
            rsi_min = np.nanmin(rsi_window)
            rsi_max = np.nanmax(rsi_window)

            if rsi_max - rsi_min != 0:
                k[i] = 100 * (rsi[i] - rsi_min) / (rsi_max - rsi_min)
            else:
                k[i] = 50.0

        # Smooth K to get %K (using SMA)
        for i in range(rsi_period + stoch_period + k_period - 2, n):
            k_window = k[i - k_period + 1:i + 1]
            k[i] = np.nanmean(k_window)

        # Calculate %D (SMA of %K)
        for i in range(rsi_period + stoch_period + k_period + d_period - 3, n):
            d_window = k[i - d_period + 1:i + 1]
            d[i] = np.nanmean(d_window)

        return {'k': k, 'd': d}

    def calculate_roc(self, closes: list, period: int) -> np.ndarray:
        """
        Calculate Rate of Change.

        Args:
            closes: List of closing prices
            period: ROC period

        Returns:
            numpy array of ROC values
        """
        if not closes:
            raise ValueError("Input data cannot be empty")
        if period <= 0:
            raise ValueError("Period must be positive")

        closes = np.array(closes, dtype=float)
        n = len(closes)

        result = np.full(n, np.nan)

        for i in range(period, n):
            if closes[i - period] != 0:
                result[i] = ((closes[i] - closes[i - period]) / closes[i - period]) * 100

        return result

    def calculate_supertrend(
        self,
        highs: list,
        lows: list,
        closes: list,
        period: int,
        multiplier: float
    ) -> dict:
        """
        Calculate Supertrend indicator.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ATR period
            multiplier: ATR multiplier

        Returns:
            Dictionary with 'supertrend' and 'direction' arrays
        """
        if not closes:
            raise ValueError("Input data cannot be empty")

        highs = np.array(highs, dtype=float)
        lows = np.array(lows, dtype=float)
        closes = np.array(closes, dtype=float)
        n = len(closes)

        # Need at least period + 1 candles for supertrend
        if n <= period:
            return {
                'supertrend': np.full(n, np.nan),
                'direction': np.zeros(n)
            }

        atr = self.calculate_atr(highs.tolist(), lows.tolist(), closes.tolist(), period)

        # Calculate basic bands
        hl2 = (highs + lows) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        supertrend = np.zeros(n)
        direction = np.zeros(n)  # 1 = uptrend, -1 = downtrend

        # Initialize based on price position relative to midpoint (hl2)
        # Using midpoint gives a more accurate initial trend direction
        if closes[period] > hl2[period]:
            supertrend[period] = lower_band[period]
            direction[period] = 1
        else:
            supertrend[period] = upper_band[period]
            direction[period] = -1

        for i in range(period + 1, n):
            if closes[i] > supertrend[i - 1]:
                # Uptrend
                supertrend[i] = max(lower_band[i], supertrend[i - 1]) if direction[i - 1] == 1 else lower_band[i]
                direction[i] = 1
            else:
                # Downtrend
                supertrend[i] = min(upper_band[i], supertrend[i - 1]) if direction[i - 1] == -1 else upper_band[i]
                direction[i] = -1

        return {
            'supertrend': supertrend,
            'direction': direction
        }
