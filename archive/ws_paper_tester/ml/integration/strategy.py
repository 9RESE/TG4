"""
ML Strategy Integration

Provides integration between ML models and the paper trading system.
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import os
import numpy as np
import pandas as pd

# Set HSA override before any torch import
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')


@dataclass
class MLStrategyConfig:
    """Configuration for ML strategy."""
    # Model settings
    model_name: str = 'signal_classifier'
    model_version: Optional[str] = None  # None = use deployed version
    registry_path: str = 'models/registry'

    # Feature settings
    lookback_bars: int = 60  # Number of 1m bars for feature calculation
    use_1m_candles: bool = True
    use_5m_candles: bool = False

    # Signal settings
    confidence_threshold: float = 0.6
    min_signal_strength: float = 0.55  # Minimum probability difference from random

    # Position sizing
    position_size_usd: float = 100.0  # Fixed position size
    position_size_pct: float = 0.1  # Alternative: percentage of capital

    # Risk management
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    max_positions: int = 1

    # Cooldown
    min_bars_between_trades: int = 5  # Minimum bars between signals
    max_daily_trades: int = 20

    # Feature columns
    feature_columns: List[str] = field(default_factory=lambda: [
        'returns', 'log_returns', 'volatility', 'rsi', 'macd',
        'bb_position', 'atr_pct', 'volume_ratio'
    ])

    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: ['XRP/USD'])


# Strategy metadata
STRATEGY_NAME = 'ml_signal'
STRATEGY_VERSION = '1.0.0'


class MLStrategy:
    """
    ML-based trading strategy.

    Uses trained ML models to generate buy/sell signals from market data.
    Supports ensemble of models and integrates with ModelRegistry.
    """

    def __init__(
        self,
        config: Optional[MLStrategyConfig] = None,
        model: Optional[Any] = None
    ):
        """
        Initialize ML strategy.

        Args:
            config: Strategy configuration
            model: Pre-loaded model (optional, otherwise loads from registry)
        """
        self.config = config or MLStrategyConfig()
        self.model = model
        self.ensemble = None

        # State tracking
        self.state = {
            'last_signal_bar': -999,
            'daily_trades': 0,
            'last_trade_date': None,
            'positions': {},
            'feature_buffer': {},
            'initialized': False
        }

        # Feature extractor
        self._feature_extractor = None

    def initialize(self) -> None:
        """Initialize strategy (called on_start)."""
        if self.state['initialized']:
            return

        # Load model if not provided
        if self.model is None:
            self._load_model()

        # Initialize feature extractor
        self._init_feature_extractor()

        self.state['initialized'] = True

    def _load_model(self) -> None:
        """Load model from registry."""
        from .registry import ModelRegistry

        registry_path = Path(self.config.registry_path)

        if registry_path.exists():
            registry = ModelRegistry(registry_path)

            try:
                self.model = registry.load(
                    self.config.model_name,
                    version=self.config.model_version
                )
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                self.model = None

    def _init_feature_extractor(self) -> None:
        """Initialize feature extractor."""
        try:
            from ..features.extractor import FeatureExtractor

            self._feature_extractor = FeatureExtractor(
                include_ta=True,
                include_volume=True,
                include_temporal=True
            )
        except ImportError:
            self._feature_extractor = None

    def generate_signal(
        self,
        snapshot: Any,  # DataSnapshot
        positions: Optional[Dict] = None,
        capital: float = 10000.0
    ) -> Optional[Any]:  # Optional[Signal]
        """
        Generate trading signal from market data.

        Args:
            snapshot: DataSnapshot with current market data
            positions: Current open positions
            capital: Available capital

        Returns:
            Signal if conditions are met, None otherwise
        """
        # Import here to avoid circular imports
        from ...ws_tester.types import Signal

        if not self.state['initialized']:
            self.initialize()

        if self.model is None:
            return None

        # Check daily trade limit
        current_date = snapshot.timestamp.date()
        if self.state['last_trade_date'] != current_date:
            self.state['daily_trades'] = 0
            self.state['last_trade_date'] = current_date

        if self.state['daily_trades'] >= self.config.max_daily_trades:
            return None

        # Process each symbol
        for symbol in self.config.symbols:
            if symbol not in snapshot.candles_1m:
                continue

            # Check position limits
            current_positions = positions or {}
            if symbol in current_positions:
                # Already have position, check for exit
                signal = self._check_exit(snapshot, symbol, current_positions[symbol])
                if signal:
                    return signal
                continue

            if len(current_positions) >= self.config.max_positions:
                continue

            # Get features
            features = self._extract_features(snapshot, symbol)
            if features is None:
                continue

            # Get prediction
            prediction = self._get_prediction(features)
            if prediction is None:
                continue

            # Check for entry signal
            signal = self._check_entry(snapshot, symbol, prediction, capital)
            if signal:
                return signal

        return None

    def _extract_features(
        self,
        snapshot: Any,
        symbol: str
    ) -> Optional[np.ndarray]:
        """Extract features from snapshot."""
        candles = snapshot.candles_1m.get(symbol, ())

        if len(candles) < self.config.lookback_bars:
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
            for c in candles[-self.config.lookback_bars:]
        ])

        if self._feature_extractor is not None:
            features_df = self._feature_extractor.extract_features(df)
            # Get the last row of features
            feature_values = features_df[self.config.feature_columns].iloc[-1].values
        else:
            # Basic features
            feature_values = self._compute_basic_features(df)

        return feature_values.reshape(1, -1).astype(np.float32)

    def _compute_basic_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute basic features without full extractor."""
        close = df['close'].values
        volume = df['volume'].values

        # Returns
        returns = np.diff(close) / close[:-1]
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

        return np.array([
            returns[-1],  # Latest return
            np.log(1 + returns[-1] + 1e-10),  # Log return
            volatility,
            rsi / 100,  # Normalized RSI
            0.0,  # MACD placeholder
            0.5,  # BB position placeholder
            volatility,  # ATR placeholder
            vol_ratio
        ])

    def _get_prediction(
        self,
        features: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Get prediction from model."""
        try:
            if hasattr(self.model, 'get_signal'):
                return self.model.get_signal(features, self.config.confidence_threshold)
            elif hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(features)
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
                return None

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def _check_entry(
        self,
        snapshot: Any,
        symbol: str,
        prediction: Dict[str, Any],
        capital: float
    ) -> Optional[Any]:
        """Check if entry conditions are met."""
        from ...ws_tester.types import Signal

        action = prediction.get('action', 'hold')
        confidence = prediction.get('confidence', 0)
        probs = prediction.get('probabilities', {})

        # Check confidence threshold
        if confidence < self.config.confidence_threshold:
            return None

        # Check signal strength (difference from random)
        random_prob = 1.0 / 3  # For 3-class classification
        signal_strength = confidence - random_prob
        if signal_strength < self.config.min_signal_strength:
            return None

        # Only act on buy/sell signals
        if action == 'hold':
            return None

        # Get current price
        price = snapshot.prices.get(symbol, 0)
        if price <= 0:
            return None

        # Calculate position size
        position_size = self.config.position_size_usd / price

        # Calculate stops
        if action == 'buy':
            stop_loss = price * (1 - self.config.stop_loss_pct / 100)
            take_profit = price * (1 + self.config.take_profit_pct / 100)
            signal_action = 'buy'
        else:  # sell -> short
            stop_loss = price * (1 + self.config.stop_loss_pct / 100)
            take_profit = price * (1 - self.config.take_profit_pct / 100)
            signal_action = 'short'

        # Update state
        self.state['daily_trades'] += 1

        return Signal(
            action=signal_action,
            symbol=symbol,
            size=position_size,
            price=price,
            reason=f"ML {action} signal (conf={confidence:.2f})",
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'strategy': STRATEGY_NAME,
                'confidence': confidence,
                'probabilities': probs,
                'model': self.config.model_name
            }
        )

    def _check_exit(
        self,
        snapshot: Any,
        symbol: str,
        position: Any
    ) -> Optional[Any]:
        """Check if exit conditions are met."""
        from ...ws_tester.types import Signal

        price = snapshot.prices.get(symbol, 0)
        if price <= 0:
            return None

        # Get features and prediction
        features = self._extract_features(snapshot, symbol)
        if features is None:
            return None

        prediction = self._get_prediction(features)
        if prediction is None:
            return None

        action = prediction.get('action', 'hold')
        confidence = prediction.get('confidence', 0)

        # Exit long on sell signal
        if position.side == 'long' and action == 'sell':
            if confidence >= self.config.confidence_threshold:
                return Signal(
                    action='sell',
                    symbol=symbol,
                    size=position.size,
                    price=price,
                    reason=f"ML exit: sell signal (conf={confidence:.2f})",
                    metadata={
                        'strategy': STRATEGY_NAME,
                        'exit_confidence': confidence
                    }
                )

        # Exit short on buy signal
        elif position.side == 'short' and action == 'buy':
            if confidence >= self.config.confidence_threshold:
                return Signal(
                    action='cover',
                    symbol=symbol,
                    size=position.size,
                    price=price,
                    reason=f"ML exit: buy signal (conf={confidence:.2f})",
                    metadata={
                        'strategy': STRATEGY_NAME,
                        'exit_confidence': confidence
                    }
                )

        return None

    def on_start(self) -> None:
        """Called when strategy starts."""
        self.initialize()

    def on_fill(self, fill: Any) -> None:
        """Called when a fill is received."""
        pass

    def on_stop(self) -> None:
        """Called when strategy stops."""
        self.state['initialized'] = False


# Module-level exports for strategy interface compatibility
SYMBOLS = ['XRP/USD']
CONFIG = MLStrategyConfig()

# State initialization
_strategy_instance: Optional[MLStrategy] = None


def get_strategy() -> MLStrategy:
    """Get or create strategy instance."""
    global _strategy_instance
    if _strategy_instance is None:
        _strategy_instance = MLStrategy(CONFIG)
    return _strategy_instance


def generate_signal(
    snapshot: Any,
    positions: Optional[Dict] = None,
    capital: float = 10000.0
) -> Optional[Any]:
    """Generate signal using ML strategy."""
    strategy = get_strategy()
    return strategy.generate_signal(snapshot, positions, capital)


def on_start() -> None:
    """Initialize strategy."""
    get_strategy().on_start()


def on_fill(fill: Any) -> None:
    """Handle fill notification."""
    get_strategy().on_fill(fill)


def on_stop() -> None:
    """Cleanup strategy."""
    get_strategy().on_stop()


def initialize_state() -> Dict[str, Any]:
    """Initialize strategy state."""
    return {
        'last_signal_bar': -999,
        'daily_trades': 0,
        'last_trade_date': None,
        'positions': {},
        'feature_buffer': {},
        'initialized': False
    }


def validate_config(config: MLStrategyConfig) -> bool:
    """Validate strategy configuration."""
    if config.confidence_threshold < 0 or config.confidence_threshold > 1:
        return False
    if config.stop_loss_pct <= 0:
        return False
    if config.take_profit_pct <= 0:
        return False
    return True
