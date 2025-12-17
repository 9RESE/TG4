"""
Feature engineering module for ML models.

Provides feature extraction from OHLCV data and DataSnapshots,
including technical indicators, temporal features, and regime classification.

New in v2.0:
- order_flow_features: Real VPIN and trade imbalance from historic trades
- multi_timeframe: Features across multiple timeframes (MTF analysis)
"""

from .extractor import FeatureExtractor
from .labels import generate_classification_labels, generate_regression_labels
from .order_flow_features import (
    OrderFlowFeatureProvider,
    OrderFlowFeatures,
    enrich_features_with_order_flow,
    enrich_features_with_order_flow_sync,
)
from .multi_timeframe import (
    MultiTimeframeFeatureProvider,
    MTFFeatures,
    TimeframeFeatures,
    enrich_features_with_mtf,
    enrich_features_with_mtf_sync,
)

__all__ = [
    "FeatureExtractor",
    "generate_classification_labels",
    "generate_regression_labels",
    # Order flow features from historic trades
    "OrderFlowFeatureProvider",
    "OrderFlowFeatures",
    "enrich_features_with_order_flow",
    "enrich_features_with_order_flow_sync",
    # Multi-timeframe features
    "MultiTimeframeFeatureProvider",
    "MTFFeatures",
    "TimeframeFeatures",
    "enrich_features_with_mtf",
    "enrich_features_with_mtf_sync",
]
