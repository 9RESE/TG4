"""
Label Generation for ML Models

Generates classification and regression labels from OHLCV data
for supervised learning.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def generate_classification_labels(
    df: pd.DataFrame,
    future_bars: int = 5,
    threshold_pct: float = 0.5,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Generate classification labels for supervised learning.

    Labels:
    - 1 (BUY): Future return > threshold
    - 0 (HOLD): Future return between -threshold and +threshold
    - -1 (SELL): Future return < -threshold

    Args:
        df: DataFrame with price data
        future_bars: Prediction horizon in bars
        threshold_pct: Threshold percentage for buy/sell classification
        price_col: Column name for price data

    Returns:
        DataFrame with added label columns:
        - future_return: Percentage return over future_bars
        - label_direction: -1 (sell), 0 (hold), 1 (buy)
        - label_class: 0 (sell), 1 (hold), 2 (buy) for neural networks
    """
    result = df.copy()

    # Calculate future return (percentage)
    result['future_return'] = (
        result[price_col].shift(-future_bars) / result[price_col] - 1
    ) * 100

    # Direction labels (-1, 0, 1)
    result['label_direction'] = 0  # HOLD by default

    result.loc[result['future_return'] > threshold_pct, 'label_direction'] = 1   # BUY
    result.loc[result['future_return'] < -threshold_pct, 'label_direction'] = -1  # SELL

    # Class labels (0, 1, 2) for neural network cross-entropy
    result['label_class'] = result['label_direction'] + 1  # Map -1,0,1 to 0,1,2

    # Probability-style labels for soft targets
    result['label_buy_prob'] = (result['future_return'] > threshold_pct).astype(float)
    result['label_sell_prob'] = (result['future_return'] < -threshold_pct).astype(float)
    result['label_hold_prob'] = (
        (result['future_return'] >= -threshold_pct) &
        (result['future_return'] <= threshold_pct)
    ).astype(float)

    return result


def generate_regression_labels(
    df: pd.DataFrame,
    horizons: List[int] = None,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Generate regression labels for price prediction.

    Args:
        df: DataFrame with price data
        horizons: List of prediction horizons in bars (default: [5, 10, 20, 60])
        price_col: Column name for price data

    Returns:
        DataFrame with added label columns:
        - future_return_{horizon}: Percentage return
        - future_close_{horizon}: Actual future price
        - future_volatility_{horizon}: Future realized volatility
    """
    if horizons is None:
        horizons = [5, 10, 20, 60]

    result = df.copy()

    for horizon in horizons:
        # Future return (percentage)
        result[f'future_return_{horizon}'] = (
            result[price_col].shift(-horizon) / result[price_col] - 1
        ) * 100

        # Future price level
        result[f'future_close_{horizon}'] = result[price_col].shift(-horizon)

        # Future log return (for statistical properties)
        result[f'future_log_return_{horizon}'] = np.log(
            result[price_col].shift(-horizon) / result[price_col]
        )

        # Future volatility (realized volatility over next horizon bars)
        if 'returns_1' in result.columns:
            # Calculate rolling std of returns shifted back
            future_vol = result['returns_1'].shift(-horizon).rolling(horizon).std()
            result[f'future_volatility_{horizon}'] = future_vol * np.sqrt(365 * 24 * 60)
        else:
            # Calculate returns first
            returns = result[price_col].pct_change()
            future_vol = returns.shift(-horizon).rolling(horizon).std()
            result[f'future_volatility_{horizon}'] = future_vol * np.sqrt(365 * 24 * 60)

        # Future direction (binary: 1 if up, 0 if down)
        result[f'future_direction_{horizon}'] = (
            result[f'future_return_{horizon}'] > 0
        ).astype(int)

    return result


def generate_multi_horizon_labels(
    df: pd.DataFrame,
    horizons: List[int] = None,
    threshold_pct: float = 0.5,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Generate labels for multiple prediction horizons.

    Useful for multi-task learning where model predicts
    multiple horizons simultaneously.

    Args:
        df: DataFrame with price data
        horizons: List of prediction horizons
        threshold_pct: Threshold for classification
        price_col: Column name for price data

    Returns:
        DataFrame with labels for all horizons
    """
    if horizons is None:
        horizons = [5, 10, 20]

    result = df.copy()

    for horizon in horizons:
        # Future return
        future_return = (
            result[price_col].shift(-horizon) / result[price_col] - 1
        ) * 100

        # Classification
        result[f'label_class_{horizon}'] = 1  # HOLD
        result.loc[future_return > threshold_pct, f'label_class_{horizon}'] = 2  # BUY
        result.loc[future_return < -threshold_pct, f'label_class_{horizon}'] = 0  # SELL

        # Regression
        result[f'future_return_{horizon}'] = future_return

    return result


def create_sequence_labels(
    df: pd.DataFrame,
    sequence_length: int = 60,
    future_bars: int = 5,
    threshold_pct: float = 0.5,
    feature_cols: List[str] = None,
    label_col: str = 'label_class'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequence data with labels for LSTM/Transformer training.

    Args:
        df: DataFrame with features and labels
        sequence_length: Number of past bars to include in sequence
        future_bars: Prediction horizon (for removing look-ahead data)
        threshold_pct: Classification threshold (if labels need to be generated)
        feature_cols: Feature columns to include (default: all numeric except labels)
        label_col: Label column name

    Returns:
        Tuple of (X, y) where:
        - X: shape (num_samples, sequence_length, num_features)
        - y: shape (num_samples,)
    """
    # Generate labels if not present
    if label_col not in df.columns:
        df = generate_classification_labels(df, future_bars, threshold_pct)

    # Determine feature columns
    if feature_cols is None:
        # Use all numeric columns except label columns
        label_cols = ['label_direction', 'label_class', 'label_buy_prob',
                     'label_sell_prob', 'label_hold_prob', 'future_return']
        future_cols = [c for c in df.columns if c.startswith('future_')]

        feature_cols = [
            c for c in df.columns
            if df[c].dtype in ['float64', 'float32', 'int64', 'int32']
            and c not in label_cols + future_cols
        ]

    # Remove rows with NaN values
    df_clean = df.dropna(subset=feature_cols + [label_col])

    if len(df_clean) < sequence_length + future_bars:
        raise ValueError(
            f"Not enough data: {len(df_clean)} rows, need {sequence_length + future_bars}"
        )

    # Create sequences
    X_list = []
    y_list = []

    features_data = df_clean[feature_cols].values
    labels_data = df_clean[label_col].values

    # Leave room at the end for future_bars (label lookahead)
    for i in range(sequence_length, len(df_clean) - future_bars):
        X_list.append(features_data[i - sequence_length:i])
        y_list.append(labels_data[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by time for proper train/val/test sets.

    Important: Always split time series by time, not randomly,
    to avoid look-ahead bias.

    Args:
        df: DataFrame with timestamp column
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        timestamp_col: Column containing timestamps

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("Ratios must sum to 1.0")

    # Sort by timestamp
    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def calculate_class_weights(
    labels: np.ndarray,
    num_classes: int = 3
) -> np.ndarray:
    """
    Calculate class weights for imbalanced classification.

    Uses inverse frequency weighting to balance classes.

    Args:
        labels: Array of class labels (0, 1, 2)
        num_classes: Number of classes

    Returns:
        Array of class weights
    """
    from collections import Counter

    counts = Counter(labels)
    total = len(labels)

    # Inverse frequency weighting
    weights = np.zeros(num_classes)
    for cls in range(num_classes):
        if counts[cls] > 0:
            weights[cls] = total / (num_classes * counts[cls])
        else:
            weights[cls] = 1.0

    return weights


def validate_labels(
    df: pd.DataFrame,
    label_col: str = 'label_class'
) -> dict:
    """
    Validate label distribution and quality.

    Args:
        df: DataFrame with labels
        label_col: Label column name

    Returns:
        Dictionary with validation statistics
    """
    if label_col not in df.columns:
        return {"error": f"Label column '{label_col}' not found"}

    labels = df[label_col].dropna()

    # Class distribution
    distribution = labels.value_counts(normalize=True).to_dict()

    # Imbalance ratio
    counts = labels.value_counts()
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    # Consecutive same labels (potential data quality issue)
    same_as_prev = (labels == labels.shift(1)).sum()
    consecutive_ratio = same_as_prev / len(labels)

    # NaN count
    nan_count = df[label_col].isna().sum()

    return {
        "total_samples": len(df),
        "valid_samples": len(labels),
        "nan_count": nan_count,
        "distribution": distribution,
        "imbalance_ratio": imbalance_ratio,
        "consecutive_ratio": consecutive_ratio,
        "is_balanced": imbalance_ratio < 3.0,
        "warnings": []
    }
