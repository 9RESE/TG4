"""
ML Signal Strategy - signal.py

Machine learning-based trading strategy that integrates with the
backtest runner and paper trading system.

Uses:
- Trained models from ModelRegistry
- Order flow features (VPIN, trade imbalance)
- Multi-timeframe features (1m, 5m, 15m, 1h, 4h)
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Set GPU environment
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')

logger = logging.getLogger(__name__)


# ==============================================================================
# Strategy Metadata (Required by strategy_loader)
# ==============================================================================

STRATEGY_NAME = 'ml_signal'
STRATEGY_VERSION = '2.0.0'
SYMBOLS = ['XRP/USDT', 'BTC/USDT']


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class MLSignalConfig:
    """Configuration for ML signal strategy."""
    # Model settings
    model_name: str = 'signal_classifier'
    model_version: Optional[str] = None  # None = deployed version
    registry_path: str = 'models/registry'

    # Timeframe
    candle_timeframe_minutes: int = 1

    # Feature settings
    lookback_bars: int = 60
    use_order_flow_features: bool = True
    use_mtf_features: bool = True

    # Signal thresholds
    confidence_threshold: float = 0.6
    min_signal_strength: float = 0.2

    # Position sizing
    position_size_usd: float = 10.0

    # Risk management
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    max_positions: int = 1
    min_bars_between_trades: int = 5
    max_daily_trades: int = 20

    # Feature columns to use (extended with new features)
    feature_columns: List[str] = field(default_factory=lambda: [
        # Basic
        'returns_1', 'returns_5', 'returns_10',
        # Trend
        'price_vs_ema_9', 'price_vs_ema_21', 'ema_alignment',
        # Momentum
        'rsi_14', 'macd_histogram', 'stoch_k',
        # Volatility
        'atr_pct', 'bb_position', 'adx_14',
        # Volume
        'volume_ratio', 'volume_zscore',
        # Order flow (from trades table)
        'trade_imbalance', 'vpin', 'order_flow_toxicity',
        # MTF
        'mtf_trend_alignment', 'dominant_trend', 'momentum_confluence',
    ])


CONFIG = MLSignalConfig()


# ==============================================================================
# State
# ==============================================================================

_state: Dict[str, Any] = {}


def initialize_state() -> Dict[str, Any]:
    """Initialize strategy state."""
    return {
        'model': None,
        'feature_extractor': None,
        'last_signal_bar': -999,
        'daily_trades': 0,
        'last_trade_date': None,
        'positions': {},
        'initialized': False,
        'bar_count': 0,
    }


# ==============================================================================
# Core Strategy Functions
# ==============================================================================

def on_start() -> None:
    """Initialize strategy on startup."""
    global _state
    _state = initialize_state()

    # Load model from registry
    _load_model()

    # Initialize feature extractor
    _init_feature_extractor()

    _state['initialized'] = True
    logger.info(f"ML Signal strategy initialized (v{STRATEGY_VERSION})")


def _load_model() -> None:
    """Load model from registry."""
    global _state

    registry_path = Path(CONFIG.registry_path)
    if not registry_path.exists():
        logger.warning(f"Registry path not found: {registry_path}")
        return

    try:
        from ml.integration.registry import ModelRegistry

        registry = ModelRegistry(registry_path)
        _state['model'] = registry.load(
            CONFIG.model_name,
            version=CONFIG.model_version
        )
        logger.info(f"Loaded model: {CONFIG.model_name}")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        _state['model'] = None


def _init_feature_extractor() -> None:
    """Initialize feature extractor."""
    global _state

    try:
        from ml.features.extractor import FeatureExtractor
        _state['feature_extractor'] = FeatureExtractor()
    except ImportError as e:
        logger.warning(f"Could not import FeatureExtractor: {e}")
        _state['feature_extractor'] = None


def generate_signal(snapshot: Any) -> Optional[Any]:
    """
    Generate trading signal from market snapshot.

    Args:
        snapshot: DataSnapshot with current market data

    Returns:
        Signal if conditions are met, None otherwise
    """
    global _state
    from ws_tester.types import Signal

    if not _state.get('initialized'):
        on_start()

    if _state.get('model') is None:
        return None

    _state['bar_count'] += 1

    # Check daily trade limit
    current_date = snapshot.timestamp.date()
    if _state.get('last_trade_date') != current_date:
        _state['daily_trades'] = 0
        _state['last_trade_date'] = current_date

    if _state['daily_trades'] >= CONFIG.max_daily_trades:
        return None

    # Check cooldown
    bars_since_signal = _state['bar_count'] - _state.get('last_signal_bar', -999)
    if bars_since_signal < CONFIG.min_bars_between_trades:
        return None

    # Process each symbol
    for symbol in SYMBOLS:
        if symbol not in snapshot.candles_1m:
            continue

        # Skip if we already have a position
        if symbol in _state.get('positions', {}):
            signal = _check_exit(snapshot, symbol)
            if signal:
                return signal
            continue

        # Check position limit
        if len(_state.get('positions', {})) >= CONFIG.max_positions:
            continue

        # Extract features
        features = _extract_features(snapshot, symbol)
        if features is None:
            continue

        # Get prediction
        prediction = _get_prediction(features)
        if prediction is None:
            continue

        # Check for entry signal
        signal = _check_entry(snapshot, symbol, prediction)
        if signal:
            _state['last_signal_bar'] = _state['bar_count']
            _state['daily_trades'] += 1
            return signal

    return None


def _extract_features(snapshot: Any, symbol: str) -> Optional[np.ndarray]:
    """Extract features from snapshot for ML prediction."""
    global _state

    candles = snapshot.candles_1m.get(symbol, ())
    if len(candles) < CONFIG.lookback_bars:
        return None

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        }
        for c in candles[-CONFIG.lookback_bars:]
    ])

    # Extract base features
    extractor = _state.get('feature_extractor')
    if extractor is not None:
        try:
            features_df = extractor.extract_features(df)
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            features_df = df
    else:
        features_df = df

    # Get available feature columns
    available_cols = [c for c in CONFIG.feature_columns if c in features_df.columns]

    if not available_cols:
        # Fallback to basic features
        return _extract_basic_features(df)

    try:
        feature_values = features_df[available_cols].iloc[-1].values
        feature_values = np.nan_to_num(feature_values, nan=0.0)
        return feature_values.reshape(1, -1).astype(np.float32)
    except Exception as e:
        logger.debug(f"Feature value extraction error: {e}")
        return _extract_basic_features(df)


def _extract_basic_features(df: pd.DataFrame) -> np.ndarray:
    """Extract basic features without full extractor."""
    close = df['close'].values
    volume = df['volume'].values

    # Returns
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.append(0, returns)

    # Volatility
    volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02

    # Simple RSI
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.01
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume ratio
    vol_mean = np.mean(volume[-20:]) if len(volume) >= 20 else 1
    vol_ratio = volume[-1] / (vol_mean + 1e-10)

    features = np.array([
        returns[-1], returns[-5] if len(returns) > 5 else 0, returns[-10] if len(returns) > 10 else 0,
        0.0, 0.0, 0.0,  # EMA features
        rsi / 100, 0.0, 0.0,  # Momentum
        volatility, 0.5, 25.0,  # Volatility
        vol_ratio, 0.0,  # Volume
        0.0, 0.5, 0.5,  # Order flow (placeholders)
        0.0, 0.0, 0.0  # MTF (placeholders)
    ])

    return features.reshape(1, -1).astype(np.float32)


def _get_prediction(features: np.ndarray) -> Optional[Dict[str, Any]]:
    """Get prediction from model."""
    global _state

    model = _state.get('model')
    if model is None:
        return None

    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)
            if probs.ndim == 2:
                probs = probs[0]

            action_idx = np.argmax(probs)
            confidence = probs[action_idx]

            action_map = {0: 'sell', 1: 'hold', 2: 'buy'}

            return {
                'action': action_map.get(action_idx, 'hold'),
                'confidence': float(confidence),
                'probabilities': {
                    'sell': float(probs[0]),
                    'hold': float(probs[1]) if len(probs) > 1 else 0.0,
                    'buy': float(probs[2]) if len(probs) > 2 else 0.0
                }
            }
        else:
            pred = model.predict(features)
            return {
                'action': 'buy' if pred[0] == 2 else ('sell' if pred[0] == 0 else 'hold'),
                'confidence': 0.7,
                'probabilities': {}
            }

    except Exception as e:
        logger.debug(f"Prediction error: {e}")
        return None


def _check_entry(
    snapshot: Any,
    symbol: str,
    prediction: Dict[str, Any]
) -> Optional[Any]:
    """Check if entry conditions are met."""
    from ws_tester.types import Signal

    action = prediction.get('action', 'hold')
    confidence = prediction.get('confidence', 0)

    # Check confidence threshold
    if confidence < CONFIG.confidence_threshold:
        return None

    # Check signal strength
    random_prob = 1.0 / 3
    signal_strength = confidence - random_prob
    if signal_strength < CONFIG.min_signal_strength:
        return None

    # Only act on buy/sell signals
    if action == 'hold':
        return None

    # Get current price
    price = snapshot.prices.get(symbol, 0)
    if price <= 0:
        return None

    # Calculate position size
    position_size = CONFIG.position_size_usd / price

    # Calculate stops
    if action == 'buy':
        stop_loss = price * (1 - CONFIG.stop_loss_pct / 100)
        take_profit = price * (1 + CONFIG.take_profit_pct / 100)
        signal_action = 'buy'
    else:  # sell -> short
        stop_loss = price * (1 + CONFIG.stop_loss_pct / 100)
        take_profit = price * (1 - CONFIG.take_profit_pct / 100)
        signal_action = 'short'

    logger.debug(f"ML signal: {action} {symbol} @ {price:.4f} (conf={confidence:.2f})")

    return Signal(
        action=signal_action,
        symbol=symbol,
        size=position_size,
        price=price,
        reason=f"ML {action} (conf={confidence:.2f})",
        stop_loss=stop_loss,
        take_profit=take_profit,
        metadata={
            'strategy': STRATEGY_NAME,
            'version': STRATEGY_VERSION,
            'confidence': confidence,
            'probabilities': prediction.get('probabilities', {}),
            'model': CONFIG.model_name,
            'entry_price': price,
        }
    )


def _check_exit(snapshot: Any, symbol: str) -> Optional[Any]:
    """Check if exit conditions are met for existing position."""
    from ws_tester.types import Signal
    global _state

    position = _state.get('positions', {}).get(symbol)
    if position is None:
        return None

    price = snapshot.prices.get(symbol, 0)
    if price <= 0:
        return None

    # Get features and prediction
    features = _extract_features(snapshot, symbol)
    if features is None:
        return None

    prediction = _get_prediction(features)
    if prediction is None:
        return None

    action = prediction.get('action', 'hold')
    confidence = prediction.get('confidence', 0)

    # Exit long on sell signal
    if position.get('side') == 'long' and action == 'sell':
        if confidence >= CONFIG.confidence_threshold:
            del _state['positions'][symbol]
            return Signal(
                action='sell',
                symbol=symbol,
                size=position.get('size', 0),
                price=price,
                reason=f"ML exit: sell signal (conf={confidence:.2f})",
                metadata={
                    'strategy': STRATEGY_NAME,
                    'exit_confidence': confidence,
                    'entry_price': position.get('entry_price'),
                }
            )

    # Exit short on buy signal
    elif position.get('side') == 'short' and action == 'buy':
        if confidence >= CONFIG.confidence_threshold:
            del _state['positions'][symbol]
            return Signal(
                action='cover',
                symbol=symbol,
                size=position.get('size', 0),
                price=price,
                reason=f"ML exit: buy signal (conf={confidence:.2f})",
                metadata={
                    'strategy': STRATEGY_NAME,
                    'exit_confidence': confidence,
                    'entry_price': position.get('entry_price'),
                }
            )

    return None


def on_fill(fill: Dict[str, Any]) -> None:
    """Handle fill notification."""
    global _state

    symbol = fill.get('symbol')
    if not symbol:
        return

    action = fill.get('side', '')

    if action in ('buy', 'long'):
        _state['positions'][symbol] = {
            'side': 'long',
            'size': fill.get('size', 0),
            'entry_price': fill.get('price', 0),
            'timestamp': fill.get('timestamp'),
        }
    elif action == 'short':
        _state['positions'][symbol] = {
            'side': 'short',
            'size': fill.get('size', 0),
            'entry_price': fill.get('price', 0),
            'timestamp': fill.get('timestamp'),
        }
    elif action in ('sell', 'cover'):
        if symbol in _state.get('positions', {}):
            del _state['positions'][symbol]


def on_stop() -> None:
    """Cleanup strategy on shutdown."""
    global _state
    _state['initialized'] = False
    logger.info("ML Signal strategy stopped")
