"""
Data Preprocessing Utilities

Provides normalization, scaling, and data cleaning functions
for ML model training.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataPreprocessor:
    """
    Preprocessor for trading ML data.

    Handles normalization, NaN filling, and feature scaling
    with proper handling for time series data.
    """

    def __init__(
        self,
        normalization_method: str = 'zscore',
        handle_nan: str = 'fill_zero',
        clip_outliers: bool = True,
        outlier_std: float = 5.0
    ):
        """
        Initialize preprocessor.

        Args:
            normalization_method: 'zscore', 'minmax', or 'robust'
            handle_nan: 'fill_zero', 'fill_mean', 'fill_forward', or 'drop'
            clip_outliers: Whether to clip outliers
            outlier_std: Number of standard deviations for outlier clipping
        """
        self.normalization_method = normalization_method
        self.handle_nan = handle_nan
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std

        # Scaler instances (fitted during fit_transform)
        self._scalers: Dict[str, StandardScaler] = {}
        self._stats: Dict[str, dict] = {}

        # Features that should not be normalized
        self.skip_normalize = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',  # Already bounded
            'label_class', 'label_direction',  # Labels
            'volatility_regime', 'trend_strength', 'market_regime',  # Categorical
        ]

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit preprocessor on data and transform.

        Args:
            df: DataFrame with features
            feature_cols: Columns to preprocess (default: all numeric)

        Returns:
            Preprocessed DataFrame
        """
        result = df.copy()

        if feature_cols is None:
            feature_cols = [
                c for c in df.columns
                if df[c].dtype in ['float64', 'float32', 'int64', 'int32']
            ]

        # Handle NaN values
        result = self._handle_nan(result, feature_cols)

        # Clip outliers before fitting scaler
        if self.clip_outliers:
            result = self._clip_outliers(result, feature_cols)

        # Fit and transform
        for col in feature_cols:
            if col in self.skip_normalize:
                continue

            # Store statistics
            self._stats[col] = {
                'mean': result[col].mean(),
                'std': result[col].std(),
                'min': result[col].min(),
                'max': result[col].max()
            }

            # Create and fit scaler
            if self.normalization_method == 'zscore':
                scaler = StandardScaler()
            elif self.normalization_method == 'minmax':
                scaler = MinMaxScaler()
            elif self.normalization_method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            # Fit and transform
            values = result[col].values.reshape(-1, 1)
            result[col] = scaler.fit_transform(values).flatten()

            self._scalers[col] = scaler

        return result

    def transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        Args:
            df: DataFrame with features
            feature_cols: Columns to preprocess

        Returns:
            Preprocessed DataFrame
        """
        result = df.copy()

        if feature_cols is None:
            feature_cols = list(self._scalers.keys())

        # Handle NaN values
        result = self._handle_nan(result, feature_cols)

        # Clip outliers
        if self.clip_outliers:
            result = self._clip_outliers(result, feature_cols)

        # Transform using fitted scalers
        for col in feature_cols:
            if col in self.skip_normalize or col not in self._scalers:
                continue

            values = result[col].values.reshape(-1, 1)
            result[col] = self._scalers[col].transform(values).flatten()

        return result

    def inverse_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Inverse transform data back to original scale.

        Args:
            df: Preprocessed DataFrame
            feature_cols: Columns to inverse transform

        Returns:
            DataFrame in original scale
        """
        result = df.copy()

        if feature_cols is None:
            feature_cols = list(self._scalers.keys())

        for col in feature_cols:
            if col not in self._scalers:
                continue

            values = result[col].values.reshape(-1, 1)
            result[col] = self._scalers[col].inverse_transform(values).flatten()

        return result

    def _handle_nan(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """Handle NaN values according to strategy."""
        for col in feature_cols:
            if df[col].isna().any():
                if self.handle_nan == 'fill_zero':
                    df[col] = df[col].fillna(0)
                elif self.handle_nan == 'fill_mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif self.handle_nan == 'fill_forward':
                    df[col] = df[col].ffill().bfill()
                elif self.handle_nan == 'drop':
                    pass  # Handled later by dropna

        return df

    def _clip_outliers(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """Clip outliers beyond outlier_std standard deviations."""
        for col in feature_cols:
            if col in self.skip_normalize:
                continue

            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                lower = mean - self.outlier_std * std
                upper = mean + self.outlier_std * std
                df[col] = df[col].clip(lower, upper)

        return df


def normalize_features(
    features: np.ndarray,
    method: str = 'zscore',
    per_feature: bool = True,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, dict]:
    """
    Normalize feature array.

    Args:
        features: Array of shape (n_samples, n_features) or (n_samples, seq_len, n_features)
        method: 'zscore', 'minmax', or 'robust'
        per_feature: If True, normalize each feature independently
        epsilon: Small constant to avoid division by zero

    Returns:
        Tuple of (normalized_features, normalization_params)
    """
    original_shape = features.shape

    # Reshape to 2D for processing
    if features.ndim == 3:
        n_samples, seq_len, n_features = features.shape
        features = features.reshape(-1, n_features)
    elif features.ndim == 2:
        pass
    else:
        raise ValueError(f"Expected 2D or 3D array, got {features.ndim}D")

    params = {}

    if per_feature:
        # Normalize each feature column
        if method == 'zscore':
            mean = features.mean(axis=0)
            std = features.std(axis=0) + epsilon
            normalized = (features - mean) / std
            params = {'mean': mean, 'std': std}

        elif method == 'minmax':
            min_val = features.min(axis=0)
            max_val = features.max(axis=0)
            range_val = max_val - min_val + epsilon
            normalized = (features - min_val) / range_val
            params = {'min': min_val, 'max': max_val}

        elif method == 'robust':
            median = np.median(features, axis=0)
            q75, q25 = np.percentile(features, [75, 25], axis=0)
            iqr = q75 - q25 + epsilon
            normalized = (features - median) / iqr
            params = {'median': median, 'iqr': iqr}

        else:
            normalized = features
    else:
        # Global normalization
        if method == 'zscore':
            mean = features.mean()
            std = features.std() + epsilon
            normalized = (features - mean) / std
            params = {'mean': mean, 'std': std}
        else:
            normalized = features

    # Reshape back to original
    if len(original_shape) == 3:
        normalized = normalized.reshape(original_shape)

    return normalized.astype(np.float32), params


def inverse_normalize(
    features: np.ndarray,
    params: dict,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Inverse normalization to recover original scale.

    Args:
        features: Normalized features
        params: Normalization parameters from normalize_features
        method: Normalization method used

    Returns:
        Features in original scale
    """
    if method == 'zscore':
        return features * params['std'] + params['mean']
    elif method == 'minmax':
        range_val = params['max'] - params['min']
        return features * range_val + params['min']
    elif method == 'robust':
        return features * params['iqr'] + params['median']
    else:
        return features


def remove_warmup_rows(
    df: pd.DataFrame,
    warmup_periods: int = 200
) -> pd.DataFrame:
    """
    Remove initial rows that may have NaN due to indicator warmup.

    Args:
        df: DataFrame with features
        warmup_periods: Number of rows to remove from start

    Returns:
        DataFrame with warmup rows removed
    """
    return df.iloc[warmup_periods:].reset_index(drop=True)


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate data quality for ML training.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    results = {
        'total_rows': len(df),
        'issues': []
    }

    # Check for NaN
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        results['nan_columns'] = nan_cols.to_dict()
        results['issues'].append(f"Found NaN in {len(nan_cols)} columns")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count

    if inf_counts:
        results['inf_columns'] = inf_counts
        results['issues'].append(f"Found infinite values in {len(inf_counts)} columns")

    # Check for constant columns
    constant_cols = [c for c in numeric_cols if df[c].std() == 0]
    if constant_cols:
        results['constant_columns'] = constant_cols
        results['issues'].append(f"Found {len(constant_cols)} constant columns")

    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        results['duplicate_rows'] = dup_count
        results['issues'].append(f"Found {dup_count} duplicate rows")

    results['is_valid'] = len(results['issues']) == 0

    return results
