"""
Signal Classifier Models

XGBoost and LightGBM-based signal classifiers for predicting
buy/sell/hold signals from technical features.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import numpy as np
import joblib

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.metrics import accuracy_score, f1_score, classification_report


class SignalClassifier(ABC):
    """
    Abstract base class for signal classifiers.

    All signal classifiers must implement:
    - fit(): Train the model
    - predict(): Get class predictions
    - predict_proba(): Get class probabilities
    - save(): Save model to disk
    - load(): Load model from disk
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> 'SignalClassifier':
        """Load model from disk."""
        pass

    def get_signal(
        self,
        X: np.ndarray,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Get trading signal with confidence.

        Args:
            X: Feature array of shape (1, n_features)
            confidence_threshold: Minimum confidence for signal

        Returns:
            Dictionary with:
            - action: 'buy', 'sell', or 'hold'
            - confidence: Probability of predicted class
            - probabilities: All class probabilities
        """
        probs = self.predict_proba(X)[0]

        # Map indices to actions (0=sell, 1=hold, 2=buy)
        action_map = {0: 'sell', 1: 'hold', 2: 'buy'}

        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]

        if confidence < confidence_threshold:
            action = 'hold'
        else:
            action = action_map[predicted_class]

        return {
            'action': action,
            'confidence': float(confidence),
            'probabilities': {
                'sell': float(probs[0]),
                'hold': float(probs[1]),
                'buy': float(probs[2])
            }
        }


class XGBoostClassifier(SignalClassifier):
    """
    XGBoost-based signal classifier.

    Features:
    - GPU acceleration via ROCm (device='cuda')
    - Early stopping with validation set
    - Built-in feature importance
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        device: str = 'cuda',
        random_state: int = 42
    ):
        """
        Initialize XGBoost classifier.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio for training
            colsample_bytree: Feature subsample ratio
            min_child_weight: Minimum sum of instance weight in leaf
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            early_stopping_rounds: Rounds without improvement to stop
            device: 'cuda' or 'cpu'
            random_state: Random seed
        """
        if xgb is None:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'early_stopping_rounds': early_stopping_rounds,
            'objective': 'multi:softprob',
            'num_class': 3,
            'tree_method': 'hist',
            'device': device,
            'random_state': random_state,
            'eval_metric': 'mlogloss'
        }

        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.feature_importance_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Override early stopping rounds (uses init value if None)
            verbose: Print training progress

        Returns:
            Dictionary with training metrics
        """
        # Override early_stopping_rounds if provided
        params = self.params.copy()
        if early_stopping_rounds is not None:
            params['early_stopping_rounds'] = early_stopping_rounds

        # Disable early stopping if no validation set
        if X_val is None:
            params.pop('early_stopping_rounds', None)

        # Create model
        self.model = xgb.XGBClassifier(**params)

        # Prepare eval set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        # Store feature importance
        self.feature_importance_ = self.model.feature_importances_

        # Calculate metrics
        train_pred = self.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')

        metrics = {
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.params['n_estimators']
        }

        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics['val_accuracy'] = accuracy_score(y_val, val_pred)
            metrics['val_f1'] = f1_score(y_val, val_pred, average='weighted')

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        importance_type: str = 'weight'
    ) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            feature_names: Names of features
            importance_type: Type of importance ('weight', 'gain', 'cover')

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        importance = self.model.get_booster().get_score(importance_type=importance_type)

        if feature_names is None:
            return importance

        # Map to provided feature names
        result = {}
        for key, value in importance.items():
            # XGBoost uses f0, f1, f2... as default names
            idx = int(key.replace('f', ''))
            if idx < len(feature_names):
                result[feature_names[idx]] = value

        return result

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        save_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_
        }
        joblib.dump(save_data, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'XGBoostClassifier':
        """Load model from disk."""
        save_data = joblib.load(path)

        classifier = cls()
        classifier.model = save_data['model']
        classifier.params = save_data['params']
        classifier.feature_names = save_data.get('feature_names')
        classifier.feature_importance_ = save_data.get('feature_importance')

        return classifier


class LightGBMClassifier(SignalClassifier):
    """
    LightGBM-based signal classifier.

    Features:
    - Fast training with histogram-based algorithm
    - Handles categorical features natively
    - Built-in feature importance
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        device: str = 'cpu',  # LightGBM GPU support is different
        random_state: int = 42
    ):
        """
        Initialize LightGBM classifier.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            num_leaves: Maximum number of leaves in one tree
            subsample: Subsample ratio for training
            colsample_bytree: Feature subsample ratio
            min_child_samples: Minimum data in leaf
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            device: 'cpu' or 'gpu'
            random_state: Random seed
        """
        if lgb is None:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")

        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_samples': min_child_samples,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'objective': 'multiclass',
            'num_class': 3,
            'device': device,
            'random_state': random_state,
            'metric': 'multi_logloss',
            'verbose': -1
        }

        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.feature_importance_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Rounds for early stopping
            verbose: Print training progress

        Returns:
            Dictionary with training metrics
        """
        # Create model
        self.model = lgb.LGBMClassifier(**self.params)

        # Prepare callbacks
        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=verbose))
            callbacks.append(lgb.log_evaluation(period=100 if verbose else 0))

        # Train
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None
        )

        # Store feature importance
        self.feature_importance_ = self.model.feature_importances_

        # Calculate metrics
        train_pred = self.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')

        metrics = {
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'best_iteration': self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else self.params['n_estimators']
        }

        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics['val_accuracy'] = accuracy_score(y_val, val_pred)
            metrics['val_f1'] = f1_score(y_val, val_pred, average='weighted')

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_
        }
        joblib.dump(save_data, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LightGBMClassifier':
        """Load model from disk."""
        save_data = joblib.load(path)

        classifier = cls()
        classifier.model = save_data['model']
        classifier.params = save_data['params']
        classifier.feature_names = save_data.get('feature_names')
        classifier.feature_importance_ = save_data.get('feature_importance')

        return classifier


def create_classifier(
    model_type: str = 'xgboost',
    **kwargs
) -> SignalClassifier:
    """
    Factory function to create a signal classifier.

    Args:
        model_type: 'xgboost' or 'lightgbm'
        **kwargs: Model-specific parameters

    Returns:
        SignalClassifier instance
    """
    if model_type.lower() == 'xgboost':
        return XGBoostClassifier(**kwargs)
    elif model_type.lower() in ['lightgbm', 'lgb']:
        return LightGBMClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
