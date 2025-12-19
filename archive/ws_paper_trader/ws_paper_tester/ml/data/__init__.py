"""
Data module for ML training.

Provides data loading, preprocessing, and PyTorch Dataset classes
for training signal classifiers and price predictors.
"""

from .dataset import TradingDataset, SequenceDataset, create_dataloaders
from .preprocessing import DataPreprocessor, normalize_features

__all__ = [
    "TradingDataset",
    "SequenceDataset",
    "create_dataloaders",
    "DataPreprocessor",
    "normalize_features",
]
