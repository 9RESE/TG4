"""
Feature Extractor for ML Models

Extracts technical indicators and derived features from OHLCV data
for use in ML model training and inference.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None
    print("[Warning] pandas-ta not installed. Some features will be limited.")

from ..config import FeatureConfig, default_config


@dataclass
class FeatureSet:
    """Container for extracted features."""
    features: pd.DataFrame
    feature_names: List[str]
    timestamp_col: str = "timestamp"


class FeatureExtractor:
    """
    Extract ML features from OHLCV data.

    Supports:
    - Trend indicators (EMA, SMA, price vs MA)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (ATR, Bollinger Bands, ADX)
    - Volume features (volume ratio, z-score)
    - Temporal features (hour, day of week with cyclical encoding)
    - Derived features (returns, regime classification)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature extractor.

        Args:
            config: Feature extraction configuration. Uses default if None.
        """
        self.config = config or default_config.features

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from OHLCV DataFrame.

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            DataFrame with original columns plus all extracted features
        """
        # Make a copy to avoid modifying original
        features = df.copy()

        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in features.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Extract features by category
        features = self._add_returns(features)
        features = self._add_trend_indicators(features)
        features = self._add_momentum_indicators(features)
        features = self._add_volatility_indicators(features)
        features = self._add_volume_features(features)
        features = self._add_temporal_features(features)
        features = self._add_regime_features(features)

        return features

    def extract_for_inference(
        self,
        candles: List[Dict],
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Extract features for real-time inference from candle list.

        Args:
            candles: List of candle dicts with OHLCV data
            feature_names: List of features to extract

        Returns:
            numpy array of shape (1, num_features) for single prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame(candles)

        # Extract all features
        features_df = self.extract_features(df)

        # Get last row with specified features
        available_features = [f for f in feature_names if f in features_df.columns]
        missing_features = [f for f in feature_names if f not in features_df.columns]

        if missing_features:
            print(f"[Warning] Missing features: {missing_features}")

        # Get last row
        last_row = features_df[available_features].iloc[-1].values

        # Handle NaN by filling with 0
        last_row = np.nan_to_num(last_row, nan=0.0)

        return last_row.reshape(1, -1)

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return features."""
        for period in self.config.return_periods:
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'log_returns_{period}'] = np.log(
                df['close'] / df['close'].shift(period)
            )

        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicator features."""
        # EMAs
        for period in self.config.ema_periods:
            if ta is not None:
                df[f'ema_{period}'] = ta.ema(df['close'], length=period)
            else:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

            # Price vs EMA (normalized)
            df[f'price_vs_ema_{period}'] = (
                (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
            )

        # SMAs
        for period in self.config.sma_periods:
            if ta is not None:
                df[f'sma_{period}'] = ta.sma(df['close'], length=period)
            else:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()

        # EMA alignment score (-2 to +2)
        if all(f'ema_{p}' in df.columns for p in [9, 21, 50]):
            df['ema_alignment'] = (
                np.sign(df['ema_9'] - df['ema_21']) +
                np.sign(df['ema_21'] - df['ema_50'])
            )
        else:
            df['ema_alignment'] = 0

        # Consecutive bars above/below EMA
        if 'ema_9' in df.columns:
            above_ema = (df['close'] > df['ema_9']).astype(int)
            df['consecutive_above_ema'] = self._count_consecutive(above_ema)
            df['consecutive_below_ema'] = self._count_consecutive(1 - above_ema)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicator features."""
        if ta is not None:
            # RSI
            df['rsi_14'] = ta.rsi(df['close'], length=self.config.rsi_period)
            df['rsi_7'] = ta.rsi(df['close'], length=self.config.rsi_fast_period)

            # MACD
            macd = ta.macd(
                df['close'],
                fast=self.config.macd_fast,
                slow=self.config.macd_slow,
                signal=self.config.macd_signal
            )
            if macd is not None and not macd.empty:
                # Search for actual column names
                macd_cols = macd.columns.tolist()
                macd_col = [c for c in macd_cols if c.startswith('MACD_')]
                signal_col = [c for c in macd_cols if c.startswith('MACDs_')]
                hist_col = [c for c in macd_cols if c.startswith('MACDh_')]

                if macd_col:
                    df['macd'] = macd[macd_col[0]]
                if signal_col:
                    df['macd_signal'] = macd[signal_col[0]]
                if hist_col:
                    df['macd_histogram'] = macd[hist_col[0]]

            # Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'],
                           k=self.config.stoch_k_period, d=self.config.stoch_d_period)
            if stoch is not None and not stoch.empty:
                # Search for actual column names
                stoch_cols = stoch.columns.tolist()
                stoch_k_col = [c for c in stoch_cols if c.startswith('STOCHk_')]
                stoch_d_col = [c for c in stoch_cols if c.startswith('STOCHd_')]

                if stoch_k_col:
                    df['stoch_k'] = stoch[stoch_k_col[0]]
                if stoch_d_col:
                    df['stoch_d'] = stoch[stoch_d_col[0]]

            # Momentum
            df['momentum_10'] = ta.mom(df['close'], length=10)

        else:
            # Fallback RSI calculation
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
            df['rsi_7'] = self._calculate_rsi(df['close'], 7)

            # Fallback MACD
            ema_fast = df['close'].ewm(span=self.config.macd_fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.config.macd_slow, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # Momentum
            df['momentum_10'] = df['close'].pct_change(10) * 100

        # Derived momentum features
        df['rsi_zone'] = pd.cut(
            df['rsi_14'],
            bins=[0, 30, 70, 100],
            labels=['oversold', 'neutral', 'overbought']
        )

        # MACD crossover (1 = bullish cross, -1 = bearish cross, 0 = none)
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_crossover'] = np.where(
                (df['macd'] > df['macd_signal']) &
                (df['macd'].shift(1) <= df['macd_signal'].shift(1)),
                1,
                np.where(
                    (df['macd'] < df['macd_signal']) &
                    (df['macd'].shift(1) >= df['macd_signal'].shift(1)),
                    -1,
                    0
                )
            )

        # RSI divergence (simplified)
        if 'rsi_14' in df.columns:
            price_change = df['close'].pct_change(5)
            rsi_change = df['rsi_14'].diff(5)
            df['rsi_divergence'] = np.where(
                (price_change > 0) & (rsi_change < 0), -1,  # Bearish divergence
                np.where(
                    (price_change < 0) & (rsi_change > 0), 1,  # Bullish divergence
                    0
                )
            )

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicator features."""
        if ta is not None:
            # ATR
            df['atr_14'] = ta.atr(
                df['high'], df['low'], df['close'],
                length=self.config.atr_period
            )

            # Bollinger Bands
            bb = ta.bbands(
                df['close'],
                length=self.config.bb_period,
                std=self.config.bb_std
            )
            if bb is not None and not bb.empty:
                # pandas-ta uses different key formats depending on version
                # Search for the actual column names instead of hardcoding
                bb_cols = bb.columns.tolist()
                bb_upper_col = [c for c in bb_cols if c.startswith('BBU_')]
                bb_lower_col = [c for c in bb_cols if c.startswith('BBL_')]
                bb_mid_col = [c for c in bb_cols if c.startswith('BBM_')]

                if bb_upper_col:
                    df['bb_upper'] = bb[bb_upper_col[0]]
                if bb_lower_col:
                    df['bb_lower'] = bb[bb_lower_col[0]]
                if bb_mid_col:
                    df['bb_mid'] = bb[bb_mid_col[0]]

            # ADX
            adx = ta.adx(
                df['high'], df['low'], df['close'],
                length=self.config.adx_period
            )
            if adx is not None and not adx.empty:
                # Search for actual column names
                adx_cols = adx.columns.tolist()
                adx_col = [c for c in adx_cols if c.startswith('ADX_')]
                dmp_col = [c for c in adx_cols if c.startswith('DMP_')]
                dmn_col = [c for c in adx_cols if c.startswith('DMN_')]

                if adx_col:
                    df['adx_14'] = adx[adx_col[0]]
                if dmp_col:
                    df['di_plus'] = adx[dmp_col[0]]
                if dmn_col:
                    df['di_minus'] = adx[dmn_col[0]]

        else:
            # Fallback ATR
            df['atr_14'] = self._calculate_atr(df, 14)

            # Fallback Bollinger Bands
            sma = df['close'].rolling(self.config.bb_period).mean()
            std = df['close'].rolling(self.config.bb_period).std()
            df['bb_upper'] = sma + self.config.bb_std * std
            df['bb_lower'] = sma - self.config.bb_std * std
            df['bb_mid'] = sma

            # ADX placeholder
            df['adx_14'] = 25.0  # Neutral value
            df['di_plus'] = 25.0
            df['di_minus'] = 25.0

        # Derived volatility features
        df['atr_pct'] = df['atr_14'] / df['close'] * 100

        # Bollinger Band position (0-1)
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_range = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = np.where(
                bb_range > 0,
                (df['close'] - df['bb_lower']) / bb_range,
                0.5
            )
            df['bb_width'] = bb_range / df['bb_mid']

        # Rolling volatility
        df['volatility_20'] = df['returns_1'].rolling(20).std() * np.sqrt(365 * 24 * 60)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume features."""
        lookback = self.config.volume_lookback

        # Volume SMA and ratio
        df['volume_sma'] = df['volume'].rolling(lookback).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Volume z-score
        volume_std = df['volume'].rolling(lookback).std()
        df['volume_zscore'] = (df['volume'] - df['volume_sma']) / volume_std

        # Volume anomaly flag
        df['volume_anomaly'] = (df['volume_zscore'].abs() > 2).astype(int)

        # Placeholder for trade imbalance (would need trade data)
        # In real implementation, this would come from order flow
        df['trade_imbalance'] = 0.0

        # Placeholder for VPIN
        df['vpin'] = 0.5

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features with cyclical encoding."""
        if 'timestamp' in df.columns:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

            # Session classification
            df['session'] = df['hour'].apply(self._classify_session)
        else:
            # No timestamp - use placeholders
            df['hour_sin'] = 0.0
            df['hour_cos'] = 1.0
            df['dow_sin'] = 0.0
            df['dow_cos'] = 1.0

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification features."""
        # Volatility regime
        if 'atr_pct' in df.columns:
            df['volatility_regime'] = pd.cut(
                df['atr_pct'],
                bins=[0, 1.5, 3.5, 7.0, float('inf')],
                labels=[0, 1, 2, 3]  # LOW, MEDIUM, HIGH, EXTREME
            ).astype(float)
        else:
            df['volatility_regime'] = 1.0

        # Trend strength (based on ADX)
        if 'adx_14' in df.columns:
            df['trend_strength'] = pd.cut(
                df['adx_14'],
                bins=[0, 20, 40, float('inf')],
                labels=[0, 1, 2]  # WEAK, MODERATE, STRONG
            ).astype(float)
        else:
            df['trend_strength'] = 1.0

        # Market regime (simplified)
        # BULL=2, SIDEWAYS=1, BEAR=0
        if 'ema_alignment' in df.columns and 'adx_14' in df.columns:
            df['market_regime'] = np.where(
                (df['ema_alignment'] > 0) & (df['adx_14'] > 20), 2,  # BULL
                np.where(
                    (df['ema_alignment'] < 0) & (df['adx_14'] > 20), 0,  # BEAR
                    1  # SIDEWAYS
                )
            )
        else:
            df['market_regime'] = 1

        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI without pandas-ta."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR without pandas-ta."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _count_consecutive(series: pd.Series) -> pd.Series:
        """Count consecutive True values."""
        # Reset count when value changes
        cumsum = series.cumsum()
        reset = cumsum - cumsum.where(series == 0).ffill().fillna(0)
        return reset

    @staticmethod
    def _classify_session(hour: int) -> str:
        """Classify trading session by hour (UTC)."""
        if 8 <= hour < 16:
            return 'EU'
        elif 14 <= hour < 22:
            return 'US'
        else:
            return 'ASIA'


def extract_features_from_snapshot(
    snapshot,
    symbol: str,
    feature_names: List[str],
    config: Optional[FeatureConfig] = None
) -> np.ndarray:
    """
    Extract features from a DataSnapshot for real-time inference.

    Args:
        snapshot: DataSnapshot object from data_layer
        symbol: Symbol to extract features for
        feature_names: List of feature names to extract
        config: Feature config (optional)

    Returns:
        numpy array of shape (1, num_features)
    """
    extractor = FeatureExtractor(config)

    # Get candles from snapshot
    candles_tuple = snapshot.candles_1m.get(symbol, ())

    if len(candles_tuple) < 60:
        # Not enough data for feature extraction
        return np.zeros((1, len(feature_names)))

    # Convert to list of dicts
    candles = []
    for c in candles_tuple:
        candles.append({
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        })

    return extractor.extract_for_inference(candles, feature_names)
