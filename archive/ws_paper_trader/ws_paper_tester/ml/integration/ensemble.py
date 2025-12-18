"""
Signal Ensemble for Combining Multiple Models

Provides methods to combine signals from multiple ML models
for improved prediction quality.
"""

from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class EnsembleMethod(Enum):
    """Ensemble combination methods."""
    VOTING = 'voting'  # Majority voting
    WEIGHTED_VOTING = 'weighted_voting'  # Weighted majority voting
    AVERAGE = 'average'  # Average probabilities
    WEIGHTED_AVERAGE = 'weighted_average'  # Weighted average
    CONFIDENCE_WEIGHTED = 'confidence_weighted'  # Weight by model confidence
    STACKING = 'stacking'  # Meta-learner combination
    MAX_CONFIDENCE = 'max_confidence'  # Take highest confidence prediction


@dataclass
class EnsembleConfig:
    """Configuration for signal ensemble."""
    method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE
    confidence_threshold: float = 0.6
    min_agreement: float = 0.5  # Minimum proportion of models that must agree
    weights: Optional[Dict[str, float]] = None  # Model weights
    fallback_action: str = 'hold'  # Action when no consensus


@dataclass
class EnsembleSignal:
    """Result from ensemble prediction."""
    action: str  # 'buy', 'hold', 'sell'
    confidence: float
    probabilities: Dict[str, float]  # Class probabilities
    agreement: float  # Proportion of models that agree
    model_signals: Dict[str, Dict]  # Individual model signals
    method: str


class SignalEnsemble:
    """
    Combine signals from multiple ML models.

    Features:
    - Multiple ensemble methods
    - Dynamic weight adjustment
    - Confidence-based filtering
    - Agreement tracking
    """

    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        config: Optional[EnsembleConfig] = None
    ):
        """
        Initialize signal ensemble.

        Args:
            models: Dictionary mapping model name to model object
            config: Ensemble configuration
        """
        self.models = models or {}
        self.config = config or EnsembleConfig()

        # Initialize weights
        if self.config.weights is None:
            # Equal weights by default
            self.weights = {name: 1.0 / len(self.models) for name in self.models}
        else:
            self.weights = self.config.weights.copy()

        # Performance tracking for dynamic weight adjustment
        self.model_performance: Dict[str, List[float]] = {
            name: [] for name in self.models
        }

    def add_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0
    ) -> None:
        """
        Add a model to the ensemble.

        Args:
            name: Model identifier
            model: Model object (must have predict_proba method)
            weight: Model weight
        """
        self.models[name] = model
        self.weights[name] = weight
        self.model_performance[name] = []

        # Renormalize weights
        self._normalize_weights()

    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            del self.model_performance[name]
            self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def predict(
        self,
        features: Union[np.ndarray, Dict[str, np.ndarray]],
        method: Optional[EnsembleMethod] = None
    ) -> EnsembleSignal:
        """
        Get ensemble prediction.

        Args:
            features: Input features (array or dict mapping model name to features)
            method: Override ensemble method

        Returns:
            EnsembleSignal with combined prediction
        """
        if not self.models:
            return EnsembleSignal(
                action=self.config.fallback_action,
                confidence=0.0,
                probabilities={'sell': 0.33, 'hold': 0.34, 'buy': 0.33},
                agreement=0.0,
                model_signals={},
                method='none'
            )

        method = method or self.config.method

        # Get predictions from all models
        model_signals = self._get_model_signals(features)

        # Combine based on method
        if method == EnsembleMethod.VOTING:
            return self._voting(model_signals)
        elif method == EnsembleMethod.WEIGHTED_VOTING:
            return self._weighted_voting(model_signals)
        elif method == EnsembleMethod.AVERAGE:
            return self._average(model_signals)
        elif method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(model_signals)
        elif method == EnsembleMethod.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted(model_signals)
        elif method == EnsembleMethod.MAX_CONFIDENCE:
            return self._max_confidence(model_signals)
        else:
            return self._weighted_average(model_signals)

    def _get_model_signals(
        self,
        features: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict]:
        """Get signals from all models."""
        signals = {}

        for name, model in self.models.items():
            try:
                # Get features for this model
                if isinstance(features, dict):
                    model_features = features.get(name, features.get('default', None))
                    if model_features is None:
                        continue
                else:
                    model_features = features

                # Get prediction
                if hasattr(model, 'get_signal'):
                    signal = model.get_signal(model_features)
                elif hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(model_features)
                    if probs.ndim == 2:
                        probs = probs[0]  # Single sample

                    action_idx = np.argmax(probs)
                    action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
                    signal = {
                        'action': action_map.get(action_idx, 'hold'),
                        'confidence': float(probs[action_idx]),
                        'probabilities': {
                            'sell': float(probs[0]),
                            'hold': float(probs[1]) if len(probs) > 1 else 0.0,
                            'buy': float(probs[2]) if len(probs) > 2 else 0.0
                        }
                    }
                else:
                    continue

                signals[name] = signal

            except Exception as e:
                # Log error but continue with other models
                print(f"Error getting signal from {name}: {e}")
                continue

        return signals

    def _voting(self, signals: Dict[str, Dict]) -> EnsembleSignal:
        """Simple majority voting."""
        if not signals:
            return self._create_fallback_signal(signals, 'voting')

        votes = {'sell': 0, 'hold': 0, 'buy': 0}

        for signal in signals.values():
            action = signal.get('action', 'hold')
            if signal.get('confidence', 0) >= self.config.confidence_threshold:
                votes[action] += 1

        # Get winning action
        total_votes = sum(votes.values())
        if total_votes == 0:
            return self._create_fallback_signal(signals, 'voting')

        winner = max(votes.keys(), key=lambda k: votes[k])
        agreement = votes[winner] / len(signals)

        if agreement < self.config.min_agreement:
            return self._create_fallback_signal(signals, 'voting')

        # Calculate probabilities from votes
        probs = {k: v / total_votes for k, v in votes.items()}

        return EnsembleSignal(
            action=winner,
            confidence=votes[winner] / len(signals),
            probabilities=probs,
            agreement=agreement,
            model_signals=signals,
            method='voting'
        )

    def _weighted_voting(self, signals: Dict[str, Dict]) -> EnsembleSignal:
        """Weighted majority voting."""
        if not signals:
            return self._create_fallback_signal(signals, 'weighted_voting')

        weighted_votes = {'sell': 0.0, 'hold': 0.0, 'buy': 0.0}

        for name, signal in signals.items():
            action = signal.get('action', 'hold')
            weight = self.weights.get(name, 1.0 / len(signals))

            if signal.get('confidence', 0) >= self.config.confidence_threshold:
                weighted_votes[action] += weight

        # Get winning action
        total = sum(weighted_votes.values())
        if total == 0:
            return self._create_fallback_signal(signals, 'weighted_voting')

        winner = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
        confidence = weighted_votes[winner]

        # Calculate agreement
        agreeing_weight = sum(
            self.weights.get(name, 1.0 / len(signals))
            for name, sig in signals.items()
            if sig.get('action') == winner
        )

        return EnsembleSignal(
            action=winner,
            confidence=confidence,
            probabilities={k: v / total if total > 0 else 0.33 for k, v in weighted_votes.items()},
            agreement=agreeing_weight,
            model_signals=signals,
            method='weighted_voting'
        )

    def _average(self, signals: Dict[str, Dict]) -> EnsembleSignal:
        """Average probabilities."""
        if not signals:
            return self._create_fallback_signal(signals, 'average')

        avg_probs = {'sell': 0.0, 'hold': 0.0, 'buy': 0.0}

        for signal in signals.values():
            probs = signal.get('probabilities', {})
            for action in avg_probs:
                avg_probs[action] += probs.get(action, 0.33 if action == 'hold' else 0.33)

        # Normalize
        n = len(signals)
        avg_probs = {k: v / n for k, v in avg_probs.items()}

        # Get winning action
        winner = max(avg_probs.keys(), key=lambda k: avg_probs[k])
        confidence = avg_probs[winner]

        # Calculate agreement
        agreeing = sum(1 for sig in signals.values() if sig.get('action') == winner)
        agreement = agreeing / len(signals)

        return EnsembleSignal(
            action=winner,
            confidence=confidence,
            probabilities=avg_probs,
            agreement=agreement,
            model_signals=signals,
            method='average'
        )

    def _weighted_average(self, signals: Dict[str, Dict]) -> EnsembleSignal:
        """Weighted average of probabilities."""
        if not signals:
            return self._create_fallback_signal(signals, 'weighted_average')

        weighted_probs = {'sell': 0.0, 'hold': 0.0, 'buy': 0.0}
        total_weight = 0.0

        for name, signal in signals.items():
            weight = self.weights.get(name, 1.0 / len(signals))
            probs = signal.get('probabilities', {})

            for action in weighted_probs:
                weighted_probs[action] += weight * probs.get(action, 0.33)

            total_weight += weight

        # Normalize
        if total_weight > 0:
            weighted_probs = {k: v / total_weight for k, v in weighted_probs.items()}

        # Get winning action
        winner = max(weighted_probs.keys(), key=lambda k: weighted_probs[k])
        confidence = weighted_probs[winner]

        # Calculate agreement
        agreeing_weight = sum(
            self.weights.get(name, 1.0 / len(signals))
            for name, sig in signals.items()
            if sig.get('action') == winner
        )

        # Apply confidence threshold
        if confidence < self.config.confidence_threshold:
            winner = self.config.fallback_action

        return EnsembleSignal(
            action=winner,
            confidence=confidence,
            probabilities=weighted_probs,
            agreement=agreeing_weight,
            model_signals=signals,
            method='weighted_average'
        )

    def _confidence_weighted(self, signals: Dict[str, Dict]) -> EnsembleSignal:
        """Weight by each model's confidence."""
        if not signals:
            return self._create_fallback_signal(signals, 'confidence_weighted')

        weighted_probs = {'sell': 0.0, 'hold': 0.0, 'buy': 0.0}
        total_weight = 0.0

        for name, signal in signals.items():
            # Use confidence as additional weight
            model_weight = self.weights.get(name, 1.0 / len(signals))
            conf_weight = signal.get('confidence', 0.5)
            combined_weight = model_weight * conf_weight

            probs = signal.get('probabilities', {})
            for action in weighted_probs:
                weighted_probs[action] += combined_weight * probs.get(action, 0.33)

            total_weight += combined_weight

        # Normalize
        if total_weight > 0:
            weighted_probs = {k: v / total_weight for k, v in weighted_probs.items()}

        winner = max(weighted_probs.keys(), key=lambda k: weighted_probs[k])

        # Calculate agreement
        agreeing = sum(1 for sig in signals.values() if sig.get('action') == winner)

        return EnsembleSignal(
            action=winner,
            confidence=weighted_probs[winner],
            probabilities=weighted_probs,
            agreement=agreeing / len(signals),
            model_signals=signals,
            method='confidence_weighted'
        )

    def _max_confidence(self, signals: Dict[str, Dict]) -> EnsembleSignal:
        """Take the prediction with highest confidence."""
        if not signals:
            return self._create_fallback_signal(signals, 'max_confidence')

        best_name = max(signals.keys(), key=lambda k: signals[k].get('confidence', 0))
        best_signal = signals[best_name]

        return EnsembleSignal(
            action=best_signal.get('action', 'hold'),
            confidence=best_signal.get('confidence', 0.5),
            probabilities=best_signal.get('probabilities', {'sell': 0.33, 'hold': 0.34, 'buy': 0.33}),
            agreement=1.0 / len(signals),  # Only one model's opinion
            model_signals=signals,
            method='max_confidence'
        )

    def _create_fallback_signal(
        self,
        signals: Dict[str, Dict],
        method: str
    ) -> EnsembleSignal:
        """Create fallback signal when no consensus."""
        return EnsembleSignal(
            action=self.config.fallback_action,
            confidence=0.0,
            probabilities={'sell': 0.33, 'hold': 0.34, 'buy': 0.33},
            agreement=0.0,
            model_signals=signals,
            method=method
        )

    def update_weights(
        self,
        performance: Dict[str, float]
    ) -> None:
        """
        Update model weights based on performance.

        Args:
            performance: Dictionary mapping model name to performance score
        """
        for name, score in performance.items():
            if name in self.weights:
                self.model_performance[name].append(score)

        # Update weights based on recent performance
        for name in self.weights:
            if self.model_performance[name]:
                # Use exponential moving average of recent performance
                recent = self.model_performance[name][-20:]  # Last 20 observations
                weights = np.exp(np.linspace(-1, 0, len(recent)))
                self.weights[name] = np.average(recent, weights=weights)

        self._normalize_weights()

    def get_model_contributions(self) -> Dict[str, float]:
        """Get current model weight contributions."""
        return self.weights.copy()

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Manually set model weights."""
        self.weights = weights.copy()
        self._normalize_weights()
