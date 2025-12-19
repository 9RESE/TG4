"""
PyTorch Dataset classes for ML training.

Provides Dataset implementations for:
- Tabular data (XGBoost/LightGBM features)
- Sequence data (LSTM/Transformer)
- RL environment data
"""

from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading ML models.

    Supports both tabular and sequence data formats.
    """

    def __init__(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        sequence_mode: bool = False,
        sequence_length: int = 60,
        transform=None
    ):
        """
        Initialize trading dataset.

        Args:
            features: Feature data (numpy array or DataFrame)
            labels: Target labels (optional for inference)
            feature_names: Names of feature columns
            sequence_mode: If True, return sequences for LSTM
            sequence_length: Length of sequences (if sequence_mode=True)
            transform: Optional transform to apply to features
        """
        # Convert DataFrame to numpy
        if isinstance(features, pd.DataFrame):
            self.feature_names = list(features.columns)
            self.features = features.values.astype(np.float32)
        else:
            self.feature_names = feature_names or [f'feature_{i}' for i in range(features.shape[-1])]
            self.features = features.astype(np.float32)

        self.labels = labels
        self.sequence_mode = sequence_mode
        self.sequence_length = sequence_length
        self.transform = transform

        # Validate shapes
        if sequence_mode:
            if self.features.ndim == 2:
                # Need to create sequences from 2D data
                self._create_sequences()
            elif self.features.ndim != 3:
                raise ValueError(f"For sequence_mode, features must be 2D or 3D, got {self.features.ndim}D")
        else:
            if self.features.ndim != 2:
                raise ValueError(f"For tabular mode, features must be 2D, got {self.features.ndim}D")

        # Handle NaN values
        self.features = np.nan_to_num(self.features, nan=0.0)

    def _create_sequences(self):
        """Create sequences from 2D data."""
        if self.features.ndim != 2:
            return

        n_samples = len(self.features)
        n_features = self.features.shape[1]

        if n_samples < self.sequence_length:
            raise ValueError(
                f"Not enough samples ({n_samples}) for sequence length ({self.sequence_length})"
            )

        # Create sliding window sequences
        sequences = []
        new_labels = []

        for i in range(self.sequence_length, n_samples):
            seq = self.features[i - self.sequence_length:i]
            sequences.append(seq)

            if self.labels is not None:
                new_labels.append(self.labels[i])

        self.features = np.array(sequences, dtype=np.float32)

        if self.labels is not None:
            self.labels = np.array(new_labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = torch.from_numpy(self.features[idx])

        if self.transform:
            x = self.transform(x)

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y

        return x

    @property
    def num_features(self) -> int:
        """Number of features per sample."""
        if self.sequence_mode:
            return self.features.shape[2]
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        """Number of unique classes (for classification)."""
        if self.labels is None:
            return 0
        return len(np.unique(self.labels))


class SequenceDataset(Dataset):
    """
    Dataset for sequence models (LSTM, Transformer).

    Pre-computed sequences with proper memory layout.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        normalize_per_sequence: bool = True
    ):
        """
        Initialize sequence dataset.

        Args:
            sequences: Array of shape (n_samples, seq_len, n_features)
            labels: Array of shape (n_samples,)
            normalize_per_sequence: If True, normalize each sequence independently
        """
        self.sequences = sequences.astype(np.float32)
        self.labels = labels

        if normalize_per_sequence:
            self._normalize_sequences()

    def _normalize_sequences(self):
        """Normalize each sequence to zero mean, unit variance."""
        # Avoid modifying original array
        normalized = np.zeros_like(self.sequences)

        for i in range(len(self.sequences)):
            seq = self.sequences[i]
            mean = seq.mean(axis=0, keepdims=True)
            std = seq.std(axis=0, keepdims=True) + 1e-8
            normalized[i] = (seq - mean) / std

        self.sequences = normalized

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.sequences[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def create_dataloaders(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    test_features: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    batch_size: int = 128,
    sequence_mode: bool = False,
    sequence_length: int = 60,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features (optional)
        val_labels: Validation labels (optional)
        test_features: Test features (optional)
        test_labels: Test labels (optional)
        batch_size: Batch size
        sequence_mode: Whether data is sequential
        sequence_length: Length of sequences
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}

    # Training loader
    train_dataset = TradingDataset(
        train_features, train_labels,
        sequence_mode=sequence_mode,
        sequence_length=sequence_length
    )
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    # Validation loader
    if val_features is not None and val_labels is not None:
        val_dataset = TradingDataset(
            val_features, val_labels,
            sequence_mode=sequence_mode,
            sequence_length=sequence_length
        )
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # Test loader
    if test_features is not None and test_labels is not None:
        test_dataset = TradingDataset(
            test_features, test_labels,
            sequence_mode=sequence_mode,
            sequence_length=sequence_length
        )
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    return loaders


def load_parquet_dataset(
    path: Union[str, Path],
    feature_cols: Optional[List[str]] = None,
    label_col: str = 'label_class',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    sequence_mode: bool = False,
    sequence_length: int = 60
) -> Tuple[Dict[str, DataLoader], List[str]]:
    """
    Load dataset from Parquet file and create DataLoaders.

    Args:
        path: Path to Parquet file
        feature_cols: List of feature columns to use
        label_col: Name of label column
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        sequence_mode: Whether to create sequences
        sequence_length: Sequence length for LSTM

    Returns:
        Tuple of (dataloaders_dict, feature_names)
    """
    # Load data
    df = pd.read_parquet(path)

    # Determine feature columns
    if feature_cols is None:
        # Exclude label and timestamp columns
        exclude_cols = ['timestamp', 'symbol', label_col]
        exclude_patterns = ['label_', 'future_']

        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and not any(c.startswith(p) for p in exclude_patterns)
            and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
        ]

    # Extract features and labels
    features = df[feature_cols].values
    labels = df[label_col].values

    # Split by time
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_features = features[:train_end]
    train_labels = labels[:train_end]

    val_features = features[train_end:val_end]
    val_labels = labels[train_end:val_end]

    test_features = features[val_end:]
    test_labels = labels[val_end:]

    # Create DataLoaders
    loaders = create_dataloaders(
        train_features, train_labels,
        val_features, val_labels,
        test_features, test_labels,
        sequence_mode=sequence_mode,
        sequence_length=sequence_length
    )

    return loaders, feature_cols


class StreamingDataset(Dataset):
    """
    Dataset that loads data on-demand from disk.

    Useful for very large datasets that don't fit in memory.
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        feature_cols: List[str],
        label_col: str,
        chunk_size: int = 10000
    ):
        """
        Initialize streaming dataset.

        Args:
            parquet_path: Path to Parquet file
            feature_cols: Feature column names
            label_col: Label column name
            chunk_size: Number of rows to load at once
        """
        self.parquet_path = Path(parquet_path)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.chunk_size = chunk_size

        # Get total length without loading all data
        self._length = len(pd.read_parquet(parquet_path, columns=[label_col]))

        # Cache for loaded chunks
        self._cache = {}
        self._cache_idx = -1

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_idx = idx // self.chunk_size

        if chunk_idx != self._cache_idx:
            # Load new chunk
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, self._length)

            df = pd.read_parquet(
                self.parquet_path,
                columns=self.feature_cols + [self.label_col]
            ).iloc[start:end]

            self._cache = {
                'features': df[self.feature_cols].values.astype(np.float32),
                'labels': df[self.label_col].values
            }
            self._cache_idx = chunk_idx

        local_idx = idx % self.chunk_size
        x = torch.from_numpy(self._cache['features'][local_idx])
        y = torch.tensor(self._cache['labels'][local_idx], dtype=torch.long)

        return x, y
