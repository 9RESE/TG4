"""
Hyperparameter Optimization with Optuna

Provides Bayesian optimization for finding optimal model hyperparameters.
"""

from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None

from ..models.classifier import XGBoostClassifier, LightGBMClassifier
from ..config import OptunaConfig, default_config


def create_study(
    study_name: str = "signal_classifier",
    storage: Optional[str] = None,
    direction: str = "maximize",
    load_if_exists: bool = True
) -> 'optuna.Study':
    """
    Create or load an Optuna study.

    Args:
        study_name: Name of the study
        storage: Database URL for persistence
        direction: "maximize" or "minimize"
        load_if_exists: Whether to load existing study

    Returns:
        Optuna Study object
    """
    if optuna is None:
        raise ImportError("optuna is required. Install with: pip install optuna")

    sampler = TPESampler(seed=42)

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=load_if_exists
    )


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    study_name: str = "xgboost_optimization",
    storage: Optional[str] = None,
    metric: str = "accuracy",
    timeout: Optional[int] = None,
    show_progress_bar: bool = True
) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials
        study_name: Name for the Optuna study
        storage: Database URL for persistence
        metric: Metric to optimize ("accuracy" or "f1")
        timeout: Optional timeout in seconds
        show_progress_bar: Show progress bar

    Returns:
        Dictionary with best parameters and results
    """
    if optuna is None:
        raise ImportError("optuna is required. Install with: pip install optuna")

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        }

        # Create and train model
        model = XGBoostClassifier(**params)
        metrics = model.fit(X_train, y_train, X_val, y_val, verbose=False)

        # Return metric to optimize
        if metric == "accuracy":
            return metrics['val_accuracy']
        elif metric == "f1":
            return metrics['val_f1']
        else:
            return metrics['val_accuracy']

    # Create study
    study = create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize"
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        n_jobs=1  # GPU training is not parallelizable
    )

    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    return {
        'best_params': best_params,
        'best_value': best_value,
        'best_trial_number': best_trial.number,
        'n_trials': len(study.trials),
        'study': study
    }


def optimize_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    study_name: str = "lightgbm_optimization",
    storage: Optional[str] = None,
    metric: str = "accuracy",
    timeout: Optional[int] = None,
    show_progress_bar: bool = True
) -> Dict[str, Any]:
    """
    Optimize LightGBM hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials
        study_name: Name for the Optuna study
        storage: Database URL for persistence
        metric: Metric to optimize
        timeout: Optional timeout in seconds
        show_progress_bar: Show progress bar

    Returns:
        Dictionary with best parameters and results
    """
    if optuna is None:
        raise ImportError("optuna is required. Install with: pip install optuna")

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        }

        model = LightGBMClassifier(**params)
        metrics = model.fit(X_train, y_train, X_val, y_val, verbose=False)

        if metric == "accuracy":
            return metrics['val_accuracy']
        elif metric == "f1":
            return metrics['val_f1']
        else:
            return metrics['val_accuracy']

    study = create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar
    )

    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_trial_number': study.best_trial.number,
        'n_trials': len(study.trials),
        'study': study
    }


def optimize_lstm(
    train_loader,
    val_loader,
    input_size: int,
    n_trials: int = 50,
    study_name: str = "lstm_optimization",
    storage: Optional[str] = None,
    device: str = 'cuda',
    max_epochs: int = 50,
    timeout: Optional[int] = None,
    show_progress_bar: bool = True
) -> Dict[str, Any]:
    """
    Optimize LSTM hyperparameters using Optuna.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_size: Number of input features
        n_trials: Number of optimization trials
        study_name: Name for the Optuna study
        storage: Database URL for persistence
        device: Training device
        max_epochs: Maximum training epochs per trial
        timeout: Optional timeout in seconds
        show_progress_bar: Show progress bar

    Returns:
        Dictionary with best parameters and results
    """
    if optuna is None:
        raise ImportError("optuna is required. Install with: pip install optuna")

    from ..models.predictor import PriceDirectionLSTM
    from .trainer import Trainer

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Suggest hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        use_attention = trial.suggest_categorical('use_attention', [True, False])

        # Create model
        model = PriceDirectionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention
        )

        # Train
        trainer = Trainer(model, device=device)
        metrics = trainer.train(
            train_loader,
            val_loader,
            epochs=max_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=10,
            verbose=False
        )

        # Report intermediate values for pruning
        trial.report(metrics['final_val_acc'], step=metrics['total_epochs'])

        # Check for pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        return metrics['final_val_acc']

    # Create study with pruning
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        n_jobs=1
    )

    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_trial_number': study.best_trial.number,
        'n_trials': len(study.trials),
        'study': study
    }


def run_optimization(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[OptunaConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for any model type.

    Args:
        model_type: 'xgboost', 'lightgbm', or 'lstm'
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Optuna configuration
        **kwargs: Additional arguments for specific optimizer

    Returns:
        Dictionary with optimization results
    """
    if config is None:
        config = default_config.optuna

    if model_type.lower() == 'xgboost':
        return optimize_xgboost(
            X_train, y_train, X_val, y_val,
            n_trials=config.n_trials,
            study_name=config.study_name,
            storage=config.storage,
            **kwargs
        )
    elif model_type.lower() in ['lightgbm', 'lgb']:
        return optimize_lightgbm(
            X_train, y_train, X_val, y_val,
            n_trials=config.n_trials,
            study_name=config.study_name,
            storage=config.storage,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
