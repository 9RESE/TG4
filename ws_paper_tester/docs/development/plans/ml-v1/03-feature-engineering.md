# Feature Engineering - ML Features from Trading Strategies

**Document Version**: 1.0
**Created**: 2025-12-16
**Status**: Research Complete

---

## Overview

This document catalogs all features extractable from the existing strategy implementations, organized by category and ML applicability.

## Feature Categories

### Tier 1: Raw Price Features (Base)

Direct OHLCV data from candles:

| Feature | Description | Normalization | ML Use |
|---------|-------------|---------------|--------|
| `open` | Opening price | Z-score or MinMax | Baseline |
| `high` | Highest price | Z-score or MinMax | Range analysis |
| `low` | Lowest price | Z-score or MinMax | Range analysis |
| `close` | Closing price | Z-score or MinMax | Primary target |
| `volume` | Trading volume | Log transform + Z-score | Activity level |
| `vwap` | Volume-weighted avg price | Z-score | Fair value |
| `trade_count` | Number of trades | Log transform | Liquidity |

**Returns Features**:
| Feature | Formula | Description |
|---------|---------|-------------|
| `returns_1` | `(close[t] - close[t-1]) / close[t-1]` | 1-bar return |
| `returns_5` | `(close[t] - close[t-5]) / close[t-5]` | 5-bar return |
| `returns_20` | `(close[t] - close[t-20]) / close[t-20]` | 20-bar return |
| `log_returns` | `log(close[t] / close[t-1])` | Log return (better for sums) |

### Tier 2: Trend Indicators

From EMA-9 Trend Flip and Momentum Scalping strategies:

| Feature | Formula | Params | Source Strategy |
|---------|---------|--------|-----------------|
| `ema_9` | Exponential MA | period=9 | EMA9 Trend Flip |
| `ema_21` | Exponential MA | period=21 | Momentum Scalping |
| `ema_50` | Exponential MA | period=50 | All |
| `ema_200` | Exponential MA | period=200 | Long-term trend |
| `sma_20` | Simple MA | period=20 | Mean Reversion |
| `sma_50` | Simple MA | period=50 | Trend baseline |

**Derived Trend Features**:
| Feature | Formula | Description |
|---------|---------|-------------|
| `price_vs_ema9` | `(close - ema_9) / ema_9` | Distance from EMA-9 |
| `price_vs_ema50` | `(close - ema_50) / ema_50` | Distance from EMA-50 |
| `ema_alignment` | `sign(ema_9 - ema_21) + sign(ema_21 - ema_50)` | Trend alignment score |
| `ema_spread` | `(ema_9 - ema_50) / ema_50` | EMA spread percentage |
| `consecutive_above_ema` | Count of bars above EMA | Trend persistence |
| `consecutive_below_ema` | Count of bars below EMA | Trend persistence |

### Tier 3: Momentum Indicators

From Momentum Scalping, Grid RSI, Mean Reversion strategies:

| Feature | Formula | Params | Range |
|---------|---------|--------|-------|
| `rsi_14` | Relative Strength Index | period=14 | 0-100 |
| `rsi_7` | Fast RSI | period=7 | 0-100 |
| `macd` | MACD line | 12, 26, 9 | Unbounded |
| `macd_signal` | Signal line | 12, 26, 9 | Unbounded |
| `macd_histogram` | MACD - Signal | - | Unbounded |
| `momentum_10` | Rate of Change | period=10 | Percentage |
| `stoch_k` | Stochastic %K | period=14 | 0-100 |
| `stoch_d` | Stochastic %D | period=3 | 0-100 |

**Derived Momentum Features**:
| Feature | Formula | Description |
|---------|---------|-------------|
| `rsi_zone` | Categorical: oversold/neutral/overbought | RSI classification |
| `rsi_crossover` | RSI crosses 50 from below/above | Momentum shift |
| `macd_crossover` | MACD crosses signal line | Momentum shift |
| `divergence_rsi` | Price up + RSI down (or vice versa) | Hidden divergence |

### Tier 4: Volatility Indicators

From Mean Reversion, EMA9 Trend Flip (ATR stops):

| Feature | Formula | Params | Description |
|---------|---------|--------|-------------|
| `atr_14` | Average True Range | period=14 | Volatility measure |
| `atr_pct` | ATR / close * 100 | - | Normalized volatility |
| `bb_upper` | SMA + 2*std | period=20 | Upper Bollinger Band |
| `bb_lower` | SMA - 2*std | period=20 | Lower Bollinger Band |
| `bb_width` | (upper - lower) / sma | - | Band width % |
| `bb_position` | (close - lower) / (upper - lower) | - | Position in bands (0-1) |
| `volatility_20` | std(returns) * sqrt(20) | period=20 | 20-bar volatility |
| `adx_14` | Average Directional Index | period=14 | Trend strength |

**Regime Classification Features**:
| Feature | Values | Thresholds |
|---------|--------|------------|
| `volatility_regime` | LOW, MEDIUM, HIGH, EXTREME | <1.5%, <3.5%, <7%, >=7% |
| `trend_strength` | WEAK, MODERATE, STRONG | ADX <20, <40, >=40 |
| `market_regime` | BULL, BEAR, SIDEWAYS | Based on MA alignment + ADX |

### Tier 5: Volume & Flow Indicators

From Order Flow, Momentum Scalping, Whale Sentiment strategies:

| Feature | Formula | Description |
|---------|---------|-------------|
| `volume_ratio` | volume / avg_volume(20) | Volume spike detection |
| `buy_volume` | Sum of buy-side volume | Recent buy pressure |
| `sell_volume` | Sum of sell-side volume | Recent sell pressure |
| `trade_imbalance` | (buy_vol - sell_vol) / total_vol | Order flow imbalance (-1 to 1) |
| `vpin` | Prob. of Informed Trading | Toxicity measure (0-1) |
| `volume_anomaly` | Z-score of volume | Unusual activity flag |

**Order Book Features** (from Order Flow strategy):
| Feature | Description |
|---------|-------------|
| `bid_depth` | Total bid volume (L10) |
| `ask_depth` | Total ask volume (L10) |
| `book_imbalance` | (bid - ask) / (bid + ask) |
| `micro_price` | Weighted mid-price |
| `spread_bps` | Spread in basis points |

### Tier 6: Multi-Timeframe Features

Aligned features from different timeframes:

| Feature | Timeframes | Alignment Method |
|---------|------------|------------------|
| `ema_9_[tf]` | 1m, 5m, 15m, 1h, 4h | As-of join |
| `rsi_14_[tf]` | 1m, 5m, 15m, 1h | As-of join |
| `trend_[tf]` | 5m, 1h, 4h, 1d | Categorical (up/down/sideways) |
| `volatility_[tf]` | 5m, 1h, 4h | ATR-based |

**MTF Confluence Features**:
| Feature | Formula | Description |
|---------|---------|-------------|
| `mtf_trend_alignment` | Sum of trend signals across TFs | Multi-TF trend confluence |
| `mtf_momentum_alignment` | Sum of RSI zones across TFs | Multi-TF momentum confluence |
| `higher_tf_support` | 1h trend matches 1m signal | Higher TF confirmation |

### Tier 7: Temporal Features

Time-based features:

| Feature | Values | Description |
|---------|--------|-------------|
| `hour_of_day` | 0-23 | Trading hour (UTC) |
| `day_of_week` | 0-6 | Day of week |
| `session` | US, EU, ASIA | Trading session |
| `is_weekend` | 0, 1 | Weekend flag |
| `minutes_since_session_open` | 0-N | Session progress |

**Cyclical Encoding** (for neural networks):
```python
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
dow_sin = sin(2 * pi * day_of_week / 7)
dow_cos = cos(2 * pi * day_of_week / 7)
```

### Tier 8: Position State Features

From portfolio and position tracking:

| Feature | Description |
|---------|-------------|
| `has_position` | Binary: position exists |
| `position_side` | 1 (long), -1 (short), 0 (flat) |
| `position_size_pct` | Current size / max size |
| `position_duration` | Bars since entry |
| `unrealized_pnl_pct` | Current unrealized P&L % |
| `distance_to_stop` | (price - stop) / price |
| `distance_to_target` | (target - price) / price |

### Tier 9: Strategy-Specific Features

**EMA-9 Trend Flip**:
| Feature | Description |
|---------|-------------|
| `consecutive_candles_position` | Count of candles on same side of EMA |
| `flip_detected` | Binary: EMA flip occurred |
| `candles_since_flip` | Bars since last flip |

**Mean Reversion**:
| Feature | Description |
|---------|-------------|
| `deviation_from_sma` | Price deviation from SMA % |
| `mean_reversion_score` | Composite reversion signal strength |

**Grid RSI Reversion**:
| Feature | Description |
|---------|-------------|
| `nearest_grid_level` | Distance to nearest grid level |
| `grid_cycles_completed` | Count of completed buy-sell cycles |

**Order Flow**:
| Feature | Description |
|---------|-------------|
| `vpin_regime` | LOW, MEDIUM, HIGH toxicity |
| `anomaly_type` | NONE, SPOOFING, WASH_TRADING |

## Feature Engineering Pipeline

### Proposed Implementation

```python
# ws_paper_tester/ml/features.py

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import pandas_ta as ta
import numpy as np

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    # Trend indicators
    ema_periods: List[int] = (9, 21, 50, 200)
    sma_periods: List[int] = (20, 50)

    # Momentum indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Volatility indicators
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14

    # Volume indicators
    volume_lookback: int = 20

    # Returns
    return_periods: List[int] = (1, 5, 10, 20)

    # Multi-timeframe
    timeframes: List[int] = (1, 5, 15, 60, 240)


class FeatureExtractor:
    """Extract ML features from OHLCV data"""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from OHLCV DataFrame"""
        features = df.copy()

        # Tier 1: Raw price features
        features = self._add_returns(features)

        # Tier 2: Trend indicators
        features = self._add_trend_indicators(features)

        # Tier 3: Momentum indicators
        features = self._add_momentum_indicators(features)

        # Tier 4: Volatility indicators
        features = self._add_volatility_indicators(features)

        # Tier 5: Volume features
        features = self._add_volume_features(features)

        # Tier 7: Temporal features
        features = self._add_temporal_features(features)

        return features

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return features"""
        for period in self.config.return_periods:
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'log_returns_{period}'] = np.log(df['close'] / df['close'].shift(period))
        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicator features"""
        # EMAs
        for period in self.config.ema_periods:
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']

        # SMAs
        for period in self.config.sma_periods:
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)

        # EMA alignment score
        df['ema_alignment'] = (
            np.sign(df['ema_9'] - df['ema_21']) +
            np.sign(df['ema_21'] - df['ema_50'])
        )

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicator features"""
        # RSI
        df['rsi_14'] = ta.rsi(df['close'], length=self.config.rsi_period)
        df['rsi_7'] = ta.rsi(df['close'], length=7)

        # MACD
        macd = ta.macd(
            df['close'],
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal
        )
        df['macd'] = macd[f'MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}']
        df['macd_signal'] = macd[f'MACDs_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}']
        df['macd_histogram'] = macd[f'MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}']

        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicator features"""
        # ATR
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=self.config.atr_period)
        df['atr_pct'] = df['atr_14'] / df['close'] * 100

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=self.config.bb_period, std=self.config.bb_std)
        df['bb_upper'] = bb[f'BBU_{self.config.bb_period}_{self.config.bb_std}']
        df['bb_lower'] = bb[f'BBL_{self.config.bb_period}_{self.config.bb_std}']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df[f'sma_{self.config.bb_period}']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ADX
        adx = ta.adx(df['high'], df['low'], df['close'], length=self.config.adx_period)
        df['adx_14'] = adx[f'ADX_{self.config.adx_period}']
        df['di_plus'] = adx[f'DMP_{self.config.adx_period}']
        df['di_minus'] = adx[f'DMN_{self.config.adx_period}']

        # Volatility (annualized)
        df['volatility_20'] = df['returns_1'].rolling(20).std() * np.sqrt(365 * 24 * 60)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume features"""
        # Volume ratio
        df['volume_sma'] = ta.sma(df['volume'], length=self.config.volume_lookback)
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Volume anomaly (z-score)
        df['volume_zscore'] = (
            (df['volume'] - df['volume'].rolling(self.config.volume_lookback).mean()) /
            df['volume'].rolling(self.config.volume_lookback).std()
        )

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""
        if 'timestamp' in df.columns:
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

        return df

    @staticmethod
    def _classify_session(hour: int) -> str:
        """Classify trading session by hour (UTC)"""
        if 8 <= hour < 16:
            return 'EU'
        elif 14 <= hour < 22:
            return 'US'
        else:
            return 'ASIA'
```

## Label Generation

### Classification Labels

```python
def generate_classification_labels(
    df: pd.DataFrame,
    future_bars: int = 5,
    threshold_pct: float = 0.5
) -> pd.DataFrame:
    """Generate classification labels for supervised learning"""

    # Future return
    df['future_return'] = df['close'].pct_change(future_bars).shift(-future_bars) * 100

    # Direction labels
    df['label_direction'] = 0  # HOLD
    df.loc[df['future_return'] > threshold_pct, 'label_direction'] = 1   # BUY
    df.loc[df['future_return'] < -threshold_pct, 'label_direction'] = -1  # SELL

    # Probability-style labels (for soft targets)
    df['label_buy_prob'] = (df['future_return'] > 0).astype(float)
    df['label_sell_prob'] = (df['future_return'] < 0).astype(float)

    return df
```

### Regression Labels

```python
def generate_regression_labels(
    df: pd.DataFrame,
    horizons: List[int] = [5, 10, 20, 60]
) -> pd.DataFrame:
    """Generate regression labels for price prediction"""

    for horizon in horizons:
        # Future return (percentage)
        df[f'future_return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon) * 100

        # Future price level
        df[f'future_close_{horizon}'] = df['close'].shift(-horizon)

        # Future volatility
        df[f'future_volatility_{horizon}'] = (
            df['returns_1'].shift(-horizon).rolling(horizon).std() * np.sqrt(365 * 24 * 60)
        )

    return df
```

## Feature Selection

### Recommended Feature Sets

**For XGBoost/LightGBM Signal Classifier**:
```python
XGBOOST_FEATURES = [
    # Trend
    'price_vs_ema_9', 'price_vs_ema_21', 'price_vs_ema_50',
    'ema_alignment',

    # Momentum
    'rsi_14', 'macd_histogram', 'stoch_k',

    # Volatility
    'atr_pct', 'bb_position', 'adx_14', 'volatility_20',

    # Volume
    'volume_ratio', 'volume_zscore',

    # Returns
    'returns_1', 'returns_5', 'returns_20',

    # Temporal
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]
```

**For LSTM/Transformer Time Series**:
```python
SEQUENCE_FEATURES = [
    # Price (normalized per-sequence)
    'open', 'high', 'low', 'close', 'volume',

    # Technical (already normalized)
    'rsi_14', 'bb_position', 'adx_14',

    # Returns (inherently normalized)
    'returns_1', 'log_returns_1'
]
```

**For Reinforcement Learning**:
```python
RL_OBSERVATION_FEATURES = [
    # Market state
    'price_vs_ema_9', 'rsi_14', 'atr_pct', 'volume_ratio',

    # Position state
    'has_position', 'position_side', 'unrealized_pnl_pct',
    'distance_to_stop', 'distance_to_target',

    # Account state
    'equity_pct_change', 'drawdown_pct'
]
```

---

**Next Document**: [AMD GPU Setup](./04-amd-gpu-setup.md)
