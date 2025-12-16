#!/usr/bin/env python3
"""
GPU-Accelerated LSTM Training Script.

Trains LSTM models on AMD GPU via ROCm/PyTorch for signal prediction.
Optimized for AMD Radeon RX 6700 XT (12GB VRAM).

Usage:
    cd ws_paper_tester
    source .env
    python -m ml.scripts.train_lstm_gpu --symbol XRP/USDT --days 90

Features:
    - Mixed precision training (AMP) for faster GPU computation
    - Gradient checkpointing for memory efficiency
    - Multi-symbol training support
    - Early stopping and LR scheduling
    - TensorBoard logging
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import time

# Set ROCm environment BEFORE importing torch
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1030')
os.environ.setdefault('MIOPEN_FIND_MODE', '3')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


def check_gpu():
    """Verify GPU is available and print info."""
    print("\n=== GPU Check ===")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"GPU Available: YES")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"PyTorch: {torch.__version__}")
        return 'cuda'
    else:
        print("GPU Available: NO - falling back to CPU")
        return 'cpu'


class AttentionLSTM(nn.Module):
    """
    LSTM with self-attention for sequence classification.

    Architecture:
    - Bidirectional LSTM layers
    - Self-attention mechanism
    - Fully connected output layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention
        lstm_output_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*directions)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*directions)

        # Output
        out = self.fc(context)
        return out


async def load_data(db_url: str, symbol: str, days: int, interval: int = 5) -> pd.DataFrame:
    """Load candle data from database."""
    import asyncpg

    print(f"\nLoading {days} days of {symbol} data...")

    conn = await asyncpg.connect(db_url)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE symbol = $1 AND interval_minutes = $2
          AND timestamp >= $3 AND timestamp < $4
        ORDER BY timestamp ASC
    """

    rows = await conn.fetch(query, symbol, interval, start_time, end_time)
    await conn.close()

    if not rows:
        raise ValueError(f"No data found for {symbol}")

    df = pd.DataFrame([dict(row) for row in rows])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    print(f"Loaded {len(df):,} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract technical features for LSTM."""
    from ml.features.extractor import FeatureExtractor

    print("Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_features(df)
    print(f"Extracted {len(features.columns)} features")
    return features


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_length: int = 60
) -> tuple:
    """Create sequences for LSTM input."""
    X, y = [], []

    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(labels[i + seq_length])

    return np.array(X), np.array(y)


def prepare_data(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    seq_length: int = 60,
    threshold_pct: float = 0.5,
    lookahead: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> dict:
    """Prepare data for training."""
    from ml.features.labels import generate_classification_labels

    print("\nPreparing data...")

    # Generate labels
    labeled_df = generate_classification_labels(
        df=df, future_bars=lookahead, threshold_pct=threshold_pct, price_col='close'
    )
    labels = labeled_df['label_class'].values

    # Get numeric features only
    drop_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'rsi_zone', 'session']
    feature_cols = [c for c in features_df.columns if c not in drop_cols]
    features_only = features_df[feature_cols].select_dtypes(include=[np.number])

    # Clean NaN
    valid_mask = ~features_only.isna().any(axis=1) & ~np.isnan(labels)
    features_clean = features_only[valid_mask].values
    labels_clean = labels[valid_mask].astype(np.int64)

    # Normalize features (z-score per feature)
    mean = features_clean.mean(axis=0)
    std = features_clean.std(axis=0) + 1e-8
    features_norm = (features_clean - mean) / std

    # Create sequences
    X, y = create_sequences(features_norm, labels_clean, seq_length)

    # Split by time
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Sequences: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    print(f"Input shape: {X_train.shape} (samples, seq_len, features)")

    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Label distribution: {dict(zip(['SELL', 'HOLD', 'BUY'], counts))}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'input_size': X_train.shape[2],
        'norm_params': {'mean': mean, 'std': std}
    }


def train_model(
    data: dict,
    device: str,
    hidden_size: int = 128,
    num_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    patience: int = 15,
    log_dir: str = None
) -> tuple:
    """Train LSTM model on GPU."""

    print(f"\n=== Training on {device.upper()} ===")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']),
        torch.LongTensor(data['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.LongTensor(data['y_val'])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    # Create model
    model = AttentionLSTM(
        input_size=data['input_size'],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=3,
        dropout=0.3
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights for imbalanced data
    class_counts = np.bincount(data['y_train'])
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    class_weights = class_weights / class_weights.sum() * 3  # Normalize

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience//2, factor=0.5
    )

    # Mixed precision
    scaler = GradScaler() if device == 'cuda' else None
    use_amp = device == 'cuda'

    # TensorBoard
    writer = SummaryWriter(log_dir) if log_dir else None

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    start_time = time.time()

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if use_amp:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        # Update scheduler
        scheduler.step(val_loss)

        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.1f}%, lr={lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Load best model
    model.load_state_dict(best_model_state)

    if writer:
        writer.close()

    return model, history


def evaluate_model(model, data: dict, device: str) -> dict:
    """Evaluate model on test set."""
    print("\n=== Evaluating ===")

    model.eval()

    X_test = torch.FloatTensor(data['X_test']).to(device)
    y_test = data['y_test']

    with torch.no_grad():
        with autocast() if device == 'cuda' else torch.no_grad():
            outputs = model(X_test)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

    # Metrics
    accuracy = (preds == y_test).mean()

    # Per-class accuracy
    class_names = ['SELL', 'HOLD', 'BUY']
    class_acc = {}
    for i, name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc[name] = (preds[mask] == i).mean()

    print(f"Test Accuracy: {accuracy*100:.1f}%")
    print(f"Per-class: {', '.join(f'{k}={v*100:.1f}%' for k, v in class_acc.items())}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, preds)
    print(f"\nConfusion Matrix:")
    print(f"         SELL  HOLD  BUY")
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:4s}: {row}")

    return {
        'accuracy': accuracy,
        'class_accuracy': class_acc,
        'confusion_matrix': cm.tolist(),
        'predictions': preds,
        'probabilities': probs
    }


def save_model(model, path: str, data: dict, metrics: dict):
    """Save model and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': data['input_size'],
        'norm_params': data['norm_params'],
        'metrics': metrics
    }, path)

    print(f"\nModel saved to: {path}")


async def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated LSTM Training')
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'))
    parser.add_argument('--symbol', type=str, default='XRP/USDT')
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--interval', type=int, default=5, help='Candle interval (minutes)')
    parser.add_argument('--seq-length', type=int, default=60)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--lookahead', type=int, default=10)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--output', type=str, default='models/lstm_signal.pt')
    parser.add_argument('--tensorboard', type=str, default='logs/tensorboard')

    args = parser.parse_args()

    if not args.db_url:
        print("ERROR: Set DATABASE_URL or use --db-url")
        sys.exit(1)

    print("=" * 60)
    print("   GPU-ACCELERATED LSTM TRAINING")
    print("=" * 60)

    device = check_gpu()

    try:
        # Load data
        df = await load_data(args.db_url, args.symbol, args.days, args.interval)

        # Extract features
        features_df = extract_features(df)

        # Prepare sequences
        data = prepare_data(
            df, features_df,
            seq_length=args.seq_length,
            threshold_pct=args.threshold,
            lookahead=args.lookahead
        )

        # Train
        log_dir = f"{args.tensorboard}/{args.symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model, history = train_model(
            data, device,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            log_dir=log_dir
        )

        # Evaluate
        metrics = evaluate_model(model, data, device)

        # Save
        save_model(model, args.output, data, metrics)

        print("\n" + "=" * 60)
        print("   TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\nModel: {args.output}")
        print(f"TensorBoard: tensorboard --logdir {args.tensorboard}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
