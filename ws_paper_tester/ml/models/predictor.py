"""
Price Direction Predictor Models

LSTM and Transformer-based models for predicting price direction
from sequential market data.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class PriceDirectionLSTM(nn.Module):
    """
    LSTM model for price direction prediction.

    Architecture:
    - Bidirectional LSTM layers
    - Multi-head self-attention
    - Dense classification head

    Input: (batch_size, sequence_length, num_features)
    Output: (batch_size, num_classes) logits
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True,
        num_attention_heads: int = 4
    ):
        """
        Initialize LSTM predictor.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (3 for buy/hold/sell)
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
            num_attention_heads: Number of attention heads
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Calculate direction multiplier
        self.num_directions = 2 if bidirectional else 1

        # Input layer normalization
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        lstm_output_size = hidden_size * self.num_directions
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=num_attention_heads,
                batch_first=True,
                dropout=dropout
            )
            self.attention_norm = nn.LayerNorm(lstm_output_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            return_attention: Whether to return attention weights

        Returns:
            Logits tensor of shape (batch_size, num_classes)
            If return_attention: tuple of (logits, attention_weights)
        """
        # Input normalization
        x = self.input_norm(x)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Attention
        if self.use_attention:
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)
        else:
            attn_weights = None

        # Use last timestep for classification
        last_hidden = lstm_out[:, -1, :]

        # Classification
        logits = self.classifier(last_hidden)

        if return_attention and attn_weights is not None:
            return logits, attn_weights

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over classes."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'bidirectional': self.bidirectional,
            'use_attention': self.use_attention
        }


class PositionSizer(nn.Module):
    """
    Neural network for dynamic position sizing.

    Predicts optimal position size (0-1) based on market conditions
    and signal characteristics.
    """

    def __init__(
        self,
        input_size: int = 15,
        hidden_sizes: List[int] = None,
        dropout: float = 0.2
    ):
        """
        Initialize position sizer.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Output layer with sigmoid for 0-1 range
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch_size, input_size)

        Returns:
            Position size in range [0, 1]
        """
        return self.network(x).squeeze(-1)


def load_lstm_model(
    path: Union[str, Path],
    device: str = 'cuda'
) -> Tuple[PriceDirectionLSTM, Dict[str, Any]]:
    """
    Load LSTM model from disk.

    Args:
        path: Path to saved model
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(path, map_location=device)

    config = checkpoint['config']
    model = PriceDirectionLSTM(**config)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def save_lstm_model(
    model: PriceDirectionLSTM,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save LSTM model to disk.

    Args:
        model: Model to save
        path: Save path
        metadata: Additional metadata to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'state_dict': model.state_dict(),
        'config': model.get_config(),
        'metadata': metadata or {}
    }

    torch.save(checkpoint, path)


class LSTMPredictor:
    """
    High-level wrapper for LSTM prediction.

    Handles model loading, preprocessing, and inference.
    """

    _device_printed = False

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = None,
        verbose: bool = False
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to saved model (optional)
            device: Device to use ('cuda' or 'cpu')
            verbose: Print device info
        """
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Print device info once
        if verbose or not LSTMPredictor._device_printed:
            if verbose:
                print(f"LSTMPredictor using device: {self.device}")
            LSTMPredictor._device_printed = True

        self.model = None
        self.config = None

        # Load model if path provided
        if model_path is not None:
            self.load(model_path)

    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk."""
        self.model, self.config = load_lstm_model(path, self.device)

    def predict(
        self,
        sequences: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict class labels.

        Args:
            sequences: Input sequences of shape (batch, seq_len, features)

        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(sequences)
        return np.argmax(probs, axis=1)

    def predict_proba(
        self,
        sequences: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            sequences: Input sequences

        Returns:
            Class probabilities of shape (batch, num_classes)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if isinstance(sequences, np.ndarray):
            sequences = torch.from_numpy(sequences).float()

        sequences = sequences.to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = self.model.predict_proba(sequences)

        return probs.cpu().numpy()

    def get_signal(
        self,
        sequence: np.ndarray,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Get trading signal from sequence.

        Args:
            sequence: Input sequence of shape (seq_len, features) or (1, seq_len, features)
            confidence_threshold: Minimum confidence for signal

        Returns:
            Dictionary with action, confidence, and probabilities
        """
        # Ensure batch dimension
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, ...]

        probs = self.predict_proba(sequence)[0]

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


class TransformerPredictor(nn.Module):
    """
    Transformer-based price direction predictor.

    Uses encoder-only transformer architecture for time series classification.
    """

    def __init__(
        self,
        input_size: int = 10,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 3,
        dropout: float = 0.1,
        max_seq_length: int = 200
    ):
        """
        Initialize Transformer predictor.

        Args:
            input_size: Number of input features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_length, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Use CLS token (last position) for classification
        cls_output = x[:, -1, :]

        # Classification
        return self.classifier(cls_output)
