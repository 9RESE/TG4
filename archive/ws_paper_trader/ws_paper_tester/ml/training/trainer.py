"""
Training Pipeline for ML Models

Provides training loops, callbacks, and utilities for training
XGBoost, LSTM, and RL models.
"""

from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import time
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from ..models.classifier import XGBoostClassifier, LightGBMClassifier, SignalClassifier
from ..models.predictor import PriceDirectionLSTM, save_lstm_model


class Trainer:
    """
    Generic trainer for neural network models.

    Features:
    - Mixed precision training (AMP)
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Metrics logging
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_amp: bool = True
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            device: Training device
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device == 'cuda'

        self.scaler = GradScaler() if self.use_amp else None
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        early_stopping_patience: int = 20,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            scheduler_patience: LR scheduler patience
            scheduler_factor: LR scheduler factor
            early_stopping_patience: Early stopping patience
            checkpoint_dir: Directory for saving checkpoints
            verbose: Print training progress

        Returns:
            Dictionary with training metrics
        """
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=scheduler_patience,
            factor=scheduler_factor,
            verbose=verbose
        )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Early stopping tracking
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        # Checkpoint directory
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            train_loss /= train_total
            train_acc = 100.0 * train_correct / train_total

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0

            if val_loader is not None:
                self.model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        if self.use_amp:
                            with autocast():
                                outputs = self.model(inputs)
                                loss = criterion(outputs, targets)
                        else:
                            outputs = self.model(inputs)
                            loss = criterion(outputs, targets)

                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

                val_loss /= val_total
                val_acc = 100.0 * val_correct / val_total

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch

                    # Save best model
                    if checkpoint_dir:
                        save_lstm_model(
                            self.model,
                            checkpoint_path / 'best_model.pt',
                            {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
                        )
                else:
                    patience_counter += 1

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                lr = optimizer.param_groups[0]['lr']
                msg = f"Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%"
                if val_loader:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.1f}%"
                msg += f", lr={lr:.6f}"
                print(msg)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - start_time

        return {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_train_loss': self.history['train_loss'][-1],
            'final_train_acc': self.history['train_acc'][-1],
            'final_val_loss': self.history['val_loss'][-1] if val_loader else None,
            'final_val_acc': self.history['val_acc'][-1] if val_loader else None,
            'elapsed_time': elapsed,
            'total_epochs': len(self.history['train_loss'])
        }

    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        test_loss /= total
        test_acc = 100.0 * correct / total

        # Calculate per-class accuracy
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        class_acc = {}
        for cls in range(3):
            mask = all_targets == cls
            if mask.sum() > 0:
                class_acc[cls] = (all_preds[mask] == cls).mean() * 100

        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'class_accuracy': class_acc,
            'predictions': all_preds,
            'targets': all_targets
        }


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model_type: str = 'xgboost',
    params: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> tuple:
    """
    Train XGBoost or LightGBM classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: 'xgboost' or 'lightgbm'
        params: Model hyperparameters
        save_path: Path to save trained model
        verbose: Print training progress

    Returns:
        Tuple of (trained_model, metrics)
    """
    # Create model
    if params is None:
        params = {}

    if model_type.lower() == 'xgboost':
        model = XGBoostClassifier(**params)
    else:
        model = LightGBMClassifier(**params)

    # Train
    metrics = model.fit(
        X_train, y_train,
        X_val, y_val,
        verbose=verbose
    )

    # Save if path provided
    if save_path:
        model.save(save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    return model, metrics


def train_lstm(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    input_size: int = 10,
    hidden_size: int = 128,
    num_layers: int = 2,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cuda',
    save_path: Optional[str] = None,
    verbose: bool = True
) -> tuple:
    """
    Train LSTM predictor.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_size: Number of input features
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        epochs: Training epochs
        learning_rate: Learning rate
        device: Training device
        save_path: Path to save trained model
        verbose: Print training progress

    Returns:
        Tuple of (trained_model, metrics)
    """
    # Create model
    model = PriceDirectionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    # Create trainer
    trainer = Trainer(model, device=device)

    # Train
    metrics = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=verbose,
        checkpoint_dir=str(Path(save_path).parent) if save_path else None
    )

    # Save if path provided
    if save_path:
        save_lstm_model(model, save_path, metrics)
        if verbose:
            print(f"Model saved to {save_path}")

    return model, metrics
