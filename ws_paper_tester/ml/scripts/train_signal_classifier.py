#!/usr/bin/env python3
"""
ML Training Script - Signal Classifier

This script trains a machine learning model to predict trading signals
(buy/hold/sell) from historical price data.

=== WHAT IS ML? (For Beginners) ===

Machine Learning is teaching a computer to find patterns in data.
Instead of writing rules like "if RSI < 30, buy", we:
1. Give the computer lots of historical examples
2. Tell it what happened after (price went up = buy was good)
3. Let it figure out the patterns itself

=== THE WORKFLOW ===

1. LOAD DATA: Get historical candles (OHLCV = Open, High, Low, Close, Volume)

2. FEATURES: Transform raw data into meaningful signals
   - Raw: [open=2.10, high=2.15, low=2.08, close=2.12]
   - Features: [rsi=45, macd=0.02, volatility=1.2%, trend=up]

3. LABELS: Define what we're predicting
   - Look at future price: did it go up >0.5%? That's a "buy" label
   - Did it go down >0.5%? That's a "sell" label
   - Stayed flat? That's a "hold" label

4. TRAIN: Algorithm finds patterns in features that predict labels
   - "When RSI < 30 AND volatility > 2%, price often goes up"

5. PREDICT: Given new features, predict the likely label

Usage:
    # Make sure DATABASE_URL is set
    export DATABASE_URL=postgresql://trading:password@localhost:5432/kraken_data

    # Run training
    python -m ml.scripts.train_signal_classifier --symbol XRP/USDT --days 30
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

# Set GPU environment before importing torch
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')


async def load_data_from_db(
    db_url: str,
    symbol: str,
    interval_minutes: int = 1,
    days: int = 30
) -> pd.DataFrame:
    """
    STEP 1: Load historical candle data from TimescaleDB.

    This connects to your database and fetches OHLCV candles.
    Think of this as getting your "textbook" for the ML to learn from.

    Args:
        db_url: Database connection string
        symbol: Trading pair (e.g., 'XRP/USDT')
        interval_minutes: Candle interval (1 = 1-minute candles)
        days: How many days of history to load

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    import asyncpg

    print(f"\n=== STEP 1: Loading Data ===")
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval_minutes}m candles")
    print(f"History: {days} days")

    # Connect to database
    conn = await asyncpg.connect(db_url)

    # Calculate time range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    # Query candles
    query = """
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM candles
        WHERE symbol = $1
          AND interval_minutes = $2
          AND timestamp >= $3
          AND timestamp < $4
        ORDER BY timestamp ASC
    """

    rows = await conn.fetch(query, symbol, interval_minutes, start_time, end_time)
    await conn.close()

    if not rows:
        raise ValueError(f"No data found for {symbol}")

    # Convert to DataFrame
    df = pd.DataFrame([dict(row) for row in rows])

    # Convert Decimal columns to float (for ML compatibility)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    print(f"Loaded {len(df):,} candles")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():.4f} - ${df['close'].max():.4f}")

    return df


def extract_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    STEP 2: Extract features from raw price data.

    Features are the "inputs" to our ML model. We transform raw OHLCV
    into meaningful indicators that might predict future price movement.

    Why features matter:
    - Raw price (2.15) isn't very useful - is that high or low?
    - RSI (30) tells us "price has been falling, might bounce"
    - MACD crossing tells us "momentum is shifting"

    We extract ~50 features including:
    - Price returns (how much price changed)
    - RSI (momentum - overbought/oversold)
    - MACD (trend momentum)
    - Bollinger Bands (volatility + price position)
    - Volume patterns
    - And more...

    Args:
        df: DataFrame with OHLCV columns
        verbose: Print progress

    Returns:
        DataFrame with feature columns added
    """
    if verbose:
        print(f"\n=== STEP 2: Extracting Features ===")

    # Import our feature extractor
    from ml.features.extractor import FeatureExtractor

    # Create extractor with default config
    # (includes technical indicators, volume features, and temporal features)
    extractor = FeatureExtractor()

    # Extract features
    features_df = extractor.extract_features(df)

    if verbose:
        print(f"Extracted {len(features_df.columns)} features")
        print(f"Sample features: {list(features_df.columns[:10])}")

        # Show feature statistics
        print(f"\nFeature statistics:")
        print(features_df.describe().loc[['mean', 'std', 'min', 'max']].T.head(10))

    return features_df


def generate_labels(
    df: pd.DataFrame,
    threshold_pct: float = 0.5,
    lookahead: int = 10,
    verbose: bool = True
) -> np.ndarray:
    """
    STEP 3: Generate labels (what we're predicting).

    Labels tell the ML what the "correct answer" is for each data point.
    We look at FUTURE price to create labels:

    - If price goes UP by >threshold_pct in next N bars -> Label = BUY (2)
    - If price goes DOWN by >threshold_pct in next N bars -> Label = SELL (0)
    - Otherwise -> Label = HOLD (1)

    Example with threshold=0.5%, lookahead=10:
    - Current price: $2.00
    - Price in 10 bars: $2.02 (+1%) -> BUY signal was correct!
    - Price in 10 bars: $1.98 (-1%) -> SELL signal was correct!
    - Price in 10 bars: $2.00 (0%) -> HOLD was correct

    IMPORTANT: We use FUTURE data for labels, but the model only sees PAST features.
    This is how we train it to predict the future!

    Args:
        df: DataFrame with at least 'close' column
        threshold_pct: Minimum price change to trigger buy/sell
        lookahead: How many bars forward to check
        verbose: Print progress

    Returns:
        Array of labels: 0=sell, 1=hold, 2=buy
    """
    if verbose:
        print(f"\n=== STEP 3: Generating Labels ===")
        print(f"Threshold: {threshold_pct}% price change")
        print(f"Lookahead: {lookahead} bars")

    from ml.features.labels import generate_classification_labels

    # generate_classification_labels adds label columns to the DataFrame
    labeled_df = generate_classification_labels(
        df=df,
        future_bars=lookahead,
        threshold_pct=threshold_pct,
        price_col='close'
    )

    # Extract the label_class column (0=sell, 1=hold, 2=buy)
    labels = labeled_df['label_class'].values

    if verbose:
        # Count label distribution
        unique, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
        label_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

        print(f"\nLabel distribution:")
        for label, count in zip(unique, counts):
            pct = count / len(labels) * 100
            print(f"  {label_names.get(int(label), label)}: {count:,} ({pct:.1f}%)")

        # Warn about class imbalance
        if counts.min() / counts.max() < 0.3:
            print("\n[WARNING] Labels are imbalanced!")
            print("  This is normal - markets don't always trend.")
            print("  The model will learn to handle this.")

    return labels


def prepare_train_test_split(
    features: pd.DataFrame,
    labels: np.ndarray,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    verbose: bool = True
) -> tuple:
    """
    STEP 4: Split data into training, validation, and test sets.

    Why split the data?
    - TRAINING (70%): The model learns patterns from this data
    - VALIDATION (15%): We check if model is learning correctly
    - TEST (15%): Final evaluation on data model never saw

    CRITICAL: We split by TIME, not randomly!
    - We train on older data, test on newer data
    - This simulates real trading (can't train on future data)

    Args:
        features: Feature DataFrame
        labels: Label array
        train_pct: Percentage for training
        val_pct: Percentage for validation
        verbose: Print progress

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if verbose:
        print(f"\n=== STEP 4: Splitting Data ===")

    # Drop non-feature columns (OHLCV, timestamp) and categorical columns
    drop_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    # Also exclude categorical columns (strings) like rsi_zone, session
    cat_cols = ['rsi_zone', 'session']
    drop_cols.extend(cat_cols)

    feature_cols = [c for c in features.columns if c not in drop_cols]
    features_only = features[feature_cols]

    # Ensure all columns are numeric (drop any remaining categorical)
    numeric_cols = features_only.select_dtypes(include=[np.number]).columns.tolist()
    features_only = features_only[numeric_cols]

    if verbose:
        print(f"Using {len(numeric_cols)} feature columns")

    # Remove rows with NaN values (from indicator warmup period)
    valid_mask = ~features_only.isna().any(axis=1) & ~np.isnan(labels)
    features_clean = features_only[valid_mask].values
    labels_clean = labels[valid_mask]

    if verbose:
        print(f"Samples after removing NaN: {len(features_clean):,}")

    # Calculate split points (TIME-ORDERED, not random!)
    n = len(features_clean)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    # Split
    X_train = features_clean[:train_end]
    X_val = features_clean[train_end:val_end]
    X_test = features_clean[val_end:]

    y_train = labels_clean[:train_end]
    y_val = labels_clean[train_end:val_end]
    y_test = labels_clean[val_end:]

    if verbose:
        print(f"\nSplit sizes:")
        print(f"  Training:   {len(X_train):,} samples ({len(X_train)/n*100:.0f}%)")
        print(f"  Validation: {len(X_val):,} samples ({len(X_val)/n*100:.0f}%)")
        print(f"  Test:       {len(X_test):,} samples ({len(X_test)/n*100:.0f}%)")
        print(f"\nRemember: Test data is from AFTER training data (time-ordered)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True
) -> 'XGBoostClassifier':
    """
    STEP 5: Train the XGBoost model.

    XGBoost is an "ensemble" of decision trees:
    - Decision tree: "If RSI < 30 AND volume > avg, then BUY"
    - XGBoost: Combines 100s of trees, each learning from others' mistakes

    Why XGBoost?
    - Fast training (especially on GPU)
    - Handles imbalanced data well
    - Works great with tabular data (like our features)
    - Built-in feature importance (we can see what it learned)

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        verbose: Print progress

    Returns:
        Trained model
    """
    if verbose:
        print(f"\n=== STEP 5: Training Model ===")
        print("Using XGBoost (gradient boosting)")
        print("This builds many decision trees that learn from each other")

    from ml.models.classifier import XGBoostClassifier

    # Create model with good defaults
    model = XGBoostClassifier(
        n_estimators=200,      # Number of trees (more = better, but slower)
        max_depth=6,           # Tree depth (deeper = more complex patterns)
        learning_rate=0.1,     # How fast to learn (slower = more stable)
        subsample=0.8,         # Use 80% of data per tree (prevents overfitting)
        colsample_bytree=0.8,  # Use 80% of features per tree
        device='cuda'          # Use GPU if available ('cuda' or 'cpu')
    )

    if verbose:
        print(f"\nModel configuration:")
        print(f"  Trees: {model.params['n_estimators']}")
        print(f"  Max depth: {model.params['max_depth']}")
        print(f"  Learning rate: {model.params['learning_rate']}")

    # Train!
    print(f"\nTraining... (this may take a minute)")
    metrics = model.fit(
        X_train, y_train,
        X_val, y_val,
        verbose=verbose
    )

    if verbose:
        print(f"\n=== Training Complete ===")
        # Metrics are in 0-1 range, convert to percentage
        train_acc = metrics['train_accuracy'] * 100
        val_acc = metrics['val_accuracy'] * 100
        print(f"Training accuracy: {train_acc:.1f}%")
        print(f"Validation accuracy: {val_acc:.1f}%")

        # Explain accuracy
        print(f"\nWhat does this mean?")
        print(f"  Random guessing (3 classes) = 33.3% accuracy")
        print(f"  Your model achieved {val_acc:.1f}%")

        improvement = (val_acc - 33.3) / 33.3 * 100
        print(f"  That's {improvement:.0f}% better than random!")

    return model


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prices_test: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    STEP 6: Evaluate model on test data.

    We test on data the model NEVER saw during training.
    This tells us how it will perform on real, new data.

    Metrics we look at:
    - Accuracy: % of correct predictions
    - Precision: When we say "buy", how often are we right?
    - Recall: Of all actual "buy" opportunities, how many did we catch?
    - Confusion Matrix: Visual breakdown of predictions vs reality

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        prices_test: Test prices (for backtest)
        verbose: Print progress

    Returns:
        Dictionary of evaluation metrics
    """
    if verbose:
        print(f"\n=== STEP 6: Evaluating Model ===")
        print("Testing on data the model never saw during training")

    from ml.evaluation.metrics import (
        calculate_metrics,
        confusion_matrix_report,
        calculate_trading_metrics
    )
    from ml.evaluation.backtest import backtest_model, BacktestConfig

    # Get predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    confidences = probabilities.max(axis=1)

    # Calculate ML metrics
    metrics = calculate_metrics(y_test, predictions)
    cm = confusion_matrix_report(y_test, predictions)

    if verbose:
        print(f"\n--- Classification Metrics ---")
        print(f"Accuracy:  {metrics['accuracy']*100:.1f}%")
        print(f"Precision: {metrics['precision']*100:.1f}%")
        print(f"Recall:    {metrics['recall']*100:.1f}%")
        print(f"F1 Score:  {metrics['f1_score']*100:.1f}%")

        # Confusion matrix
        print(f"\n--- Confusion Matrix ---")
        print("(Rows = Actual, Columns = Predicted)")
        print("         SELL  HOLD  BUY")
        for i, row in enumerate(cm['confusion_matrix']):
            label = ['SELL', 'HOLD', 'BUY'][i]
            print(f"  {label}: {row}")

    # Run backtest (simulate trading)
    if verbose:
        print(f"\n--- Backtest Results ---")
        print("Simulating trades based on model predictions...")

    backtest_result = backtest_model(
        model=model,
        features=X_test,
        prices=prices_test,
        config=BacktestConfig(
            initial_capital=1000.0,
            position_size_pct=0.1,      # 10% per trade
            confidence_threshold=0.6,   # Only trade when >60% confident
            stop_loss_pct=2.0,
            take_profit_pct=4.0
        )
    )

    if verbose:
        tm = backtest_result.metrics
        print(f"\n  Initial capital: $1,000")
        print(f"  Total return:    {tm.total_return:.2f}%")
        print(f"  Sharpe ratio:    {tm.sharpe_ratio:.2f}")
        print(f"  Max drawdown:    {tm.max_drawdown:.2f}%")
        print(f"  Win rate:        {tm.win_rate:.1f}%")
        print(f"  Total trades:    {tm.num_trades}")

        print(f"\n--- What These Metrics Mean ---")
        print(f"  Sharpe ratio > 1.0 is considered good")
        print(f"  Win rate > 50% means more winners than losers")
        print(f"  Max drawdown is the biggest drop from peak")

    return {
        'classification': metrics,
        'backtest': backtest_result.metrics,
        'confusion_matrix': cm
    }


def save_model(model, path: str, metrics: dict, verbose: bool = True):
    """
    STEP 7: Save the trained model.

    We save the model so we can use it later without retraining.

    Args:
        model: Trained model
        path: Where to save
        metrics: Training/evaluation metrics to save with model
        verbose: Print progress
    """
    if verbose:
        print(f"\n=== STEP 7: Saving Model ===")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(path))

    # Save metrics too
    import json
    metrics_path = path.parent / f"{path.stem}_metrics.json"

    # Convert metrics to serializable format
    metrics_dict = {
        'classification': metrics['classification'],
        'backtest': {
            'total_return': metrics['backtest'].total_return,
            'sharpe_ratio': metrics['backtest'].sharpe_ratio,
            'max_drawdown': metrics['backtest'].max_drawdown,
            'win_rate': metrics['backtest'].win_rate,
            'num_trades': metrics['backtest'].num_trades
        }
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    if verbose:
        print(f"Model saved to: {path}")
        print(f"Metrics saved to: {metrics_path}")


async def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train ML Signal Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on 30 days of XRP data
    python -m ml.scripts.train_signal_classifier --symbol XRP/USDT --days 30

    # Train with custom threshold
    python -m ml.scripts.train_signal_classifier --symbol XRP/USDT --threshold 0.3

    # Use specific database
    python -m ml.scripts.train_signal_classifier --db-url postgresql://user:pass@host/db
        """
    )

    parser.add_argument(
        '--db-url',
        type=str,
        default=os.getenv('DATABASE_URL'),
        help='PostgreSQL connection URL (or set DATABASE_URL env var)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='XRP/USDT',
        help='Trading symbol (default: XRP/USDT)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='Candle interval in minutes (default: 1)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Days of history to load (default: 30)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Price change threshold for labels in %% (default: 0.5)'
    )
    parser.add_argument(
        '--lookahead',
        type=int,
        default=10,
        help='Bars to look ahead for labels (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/signal_classifier.json',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Validate database URL
    if not args.db_url:
        print("ERROR: Database URL required.")
        print("Set DATABASE_URL environment variable or use --db-url")
        print("\nExample:")
        print("  export DATABASE_URL=postgresql://trading:password@localhost:5432/kraken_data")
        sys.exit(1)

    print("=" * 60)
    print("   ML SIGNAL CLASSIFIER TRAINING")
    print("=" * 60)

    try:
        # Step 1: Load data
        df = await load_data_from_db(
            args.db_url,
            args.symbol,
            args.interval,
            args.days
        )

        # Step 2: Extract features
        features = extract_features(df, verbose=verbose)

        # Step 3: Generate labels
        labels = generate_labels(
            df,  # Pass full DataFrame, not just prices
            threshold_pct=args.threshold,
            lookahead=args.lookahead,
            verbose=verbose
        )

        # Step 4: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(
            features, labels, verbose=verbose
        )

        # Get test prices for backtest
        # Use same mask as prepare_train_test_split
        drop_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in features.columns if c not in drop_cols]
        valid_mask = ~features[feature_cols].isna().any(axis=1) & ~np.isnan(labels)
        prices_clean = df['close'].values[valid_mask]
        n = len(prices_clean)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        prices_test = prices_clean[val_end:]

        # Step 5: Train model
        model = train_model(X_train, y_train, X_val, y_val, verbose=verbose)

        # Step 6: Evaluate
        metrics = evaluate_model(model, X_test, y_test, prices_test, verbose=verbose)

        # Step 7: Save
        save_model(model, args.output, metrics, verbose=verbose)

        print("\n" + "=" * 60)
        print("   TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\nYour trained model is saved at: {args.output}")
        print("\nNext steps:")
        print("  1. Review the metrics above")
        print("  2. Try different --threshold and --lookahead values")
        print("  3. Integrate with paper trading using MLStrategy")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
