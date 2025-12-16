"""
Model Registry for ML Models

Provides version control, deployment management, and model artifact storage.
"""

from typing import Optional, Dict, Any, List, Union, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import json
import shutil
import hashlib
import os

# Set HSA override before any torch import
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')

import numpy as np


@dataclass
class ModelInfo:
    """Metadata for a registered model."""
    name: str
    version: str
    model_type: str  # 'xgboost', 'lightgbm', 'lstm', 'ppo', 'sac'
    created_at: str
    path: str

    # Training info
    training_data_hash: Optional[str] = None
    training_samples: int = 0
    training_features: List[str] = field(default_factory=list)

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = 'registered'  # 'registered', 'deployed', 'deprecated'
    deployed_at: Optional[str] = None

    # Additional metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Model Registry for managing ML models.

    Features:
    - Version control for models
    - Model artifact storage
    - Deployment management
    - Model comparison
    - Rollback capability
    """

    def __init__(
        self,
        registry_path: Union[str, Path],
        auto_create: bool = True
    ):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry directory
            auto_create: Create directory if it doesn't exist
        """
        self.registry_path = Path(registry_path)

        if auto_create:
            self.registry_path.mkdir(parents=True, exist_ok=True)

        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.registry_path / "registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry metadata from disk."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self._models = {
                    k: ModelInfo.from_dict(v) for k, v in data.get('models', {}).items()
                }
                self._deployed = data.get('deployed', {})
        else:
            self._models: Dict[str, ModelInfo] = {}
            self._deployed: Dict[str, str] = {}  # model_name -> version

    def _save_registry(self) -> None:
        """Save registry metadata to disk."""
        data = {
            'models': {k: v.to_dict() for k, v in self._models.items()},
            'deployed': self._deployed,
            'updated_at': datetime.now().isoformat()
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_model_key(self, name: str, version: str) -> str:
        """Get unique key for model."""
        return f"{name}:{version}"

    def _next_version(self, name: str) -> str:
        """Generate next version number for a model."""
        existing_versions = [
            info.version for key, info in self._models.items()
            if info.name == name
        ]

        if not existing_versions:
            return "1.0.0"

        # Parse and increment
        latest = max(existing_versions, key=lambda v: [int(x) for x in v.split('.')])
        parts = [int(x) for x in latest.split('.')]
        parts[-1] += 1

        return '.'.join(str(x) for x in parts)

    def register(
        self,
        model: Any,
        name: str,
        model_type: str,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        training_features: Optional[List[str]] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> ModelInfo:
        """
        Register a new model in the registry.

        Args:
            model: Trained model object
            name: Model name
            model_type: Type of model
            version: Optional version (auto-generated if not provided)
            metrics: Performance metrics
            config: Training configuration
            training_features: List of feature names
            description: Model description
            tags: Model tags

        Returns:
            ModelInfo for registered model
        """
        if version is None:
            version = self._next_version(name)

        key = self._get_model_key(name, version)

        if key in self._models:
            raise ValueError(f"Model {name}:{version} already exists")

        # Create model directory
        model_dir = self.models_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = self._save_model(model, model_type, model_dir)

        # Create model info
        info = ModelInfo(
            name=name,
            version=version,
            model_type=model_type,
            created_at=datetime.now().isoformat(),
            path=str(model_path),
            metrics=metrics or {},
            config=config or {},
            training_features=training_features or [],
            description=description,
            tags=tags or []
        )

        # Register
        self._models[key] = info
        self._save_registry()

        return info

    def _save_model(
        self,
        model: Any,
        model_type: str,
        model_dir: Path
    ) -> Path:
        """Save model to disk."""
        if model_type in ['xgboost', 'lightgbm']:
            path = model_dir / "model.json"
            if hasattr(model, 'save'):
                model.save(str(path))
            elif hasattr(model, 'model') and hasattr(model.model, 'save_model'):
                model.model.save_model(str(path))
            else:
                # Fallback to pickle
                import pickle
                path = model_dir / "model.pkl"
                with open(path, 'wb') as f:
                    pickle.dump(model, f)

        elif model_type == 'lstm':
            import torch
            path = model_dir / "model.pt"
            if hasattr(model, 'state_dict'):
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'config': model.get_config() if hasattr(model, 'get_config') else {}
                }
                torch.save(checkpoint, path)
            else:
                torch.save(model, path)

        elif model_type in ['ppo', 'sac']:
            path = model_dir / f"{model_type}_model"
            if hasattr(model, 'save'):
                model.save(str(path))
            else:
                import pickle
                path = model_dir / "model.pkl"
                with open(path, 'wb') as f:
                    pickle.dump(model, f)

        else:
            # Generic pickle save
            import pickle
            path = model_dir / "model.pkl"
            with open(path, 'wb') as f:
                pickle.dump(model, f)

        return path

    def load(
        self,
        name: str,
        version: Optional[str] = None,
        device: str = 'auto'
    ) -> Any:
        """
        Load a model from the registry.

        Args:
            name: Model name
            version: Version to load (latest if not specified)
            device: Device to load model on

        Returns:
            Loaded model
        """
        if version is None:
            # Get deployed version or latest
            if name in self._deployed:
                version = self._deployed[name]
            else:
                version = self._get_latest_version(name)

        key = self._get_model_key(name, version)

        if key not in self._models:
            raise ValueError(f"Model {name}:{version} not found")

        info = self._models[key]
        return self._load_model(info, device)

    def _load_model(self, info: ModelInfo, device: str) -> Any:
        """Load model from disk."""
        path = Path(info.path)

        if info.model_type in ['xgboost', 'lightgbm']:
            from ..models.classifier import XGBoostClassifier, LightGBMClassifier

            if info.model_type == 'xgboost':
                model = XGBoostClassifier()
            else:
                model = LightGBMClassifier()

            if path.suffix == '.json':
                model.load(str(path))
            else:
                import pickle
                with open(path, 'rb') as f:
                    model = pickle.load(f)

        elif info.model_type == 'lstm':
            import torch
            from ..models.predictor import PriceDirectionLSTM

            checkpoint = torch.load(path, map_location=device)

            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                config = checkpoint.get('config', {})
                model = PriceDirectionLSTM(**config)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model = checkpoint

            if device != 'cpu':
                model = model.to(device)
            model.eval()

        elif info.model_type in ['ppo', 'sac']:
            from ..rl.agents import load_agent
            model = load_agent(path, info.model_type, device)

        else:
            import pickle
            with open(path, 'rb') as f:
                model = pickle.load(f)

        return model

    def _get_latest_version(self, name: str) -> str:
        """Get latest version of a model."""
        versions = [
            info.version for key, info in self._models.items()
            if info.name == name
        ]

        if not versions:
            raise ValueError(f"No versions found for model {name}")

        return max(versions, key=lambda v: [int(x) for x in v.split('.')])

    def deploy(self, name: str, version: str) -> None:
        """
        Deploy a specific model version.

        Args:
            name: Model name
            version: Version to deploy
        """
        key = self._get_model_key(name, version)

        if key not in self._models:
            raise ValueError(f"Model {name}:{version} not found")

        # Update deployed status
        self._deployed[name] = version
        self._models[key].status = 'deployed'
        self._models[key].deployed_at = datetime.now().isoformat()

        self._save_registry()

    def rollback(self, name: str, to_version: str) -> None:
        """
        Rollback to a previous version.

        Args:
            name: Model name
            to_version: Version to rollback to
        """
        self.deploy(name, to_version)

    def get_deployed_version(self, name: str) -> Optional[str]:
        """Get currently deployed version."""
        return self._deployed.get(name)

    def list_models(self, name: Optional[str] = None) -> List[ModelInfo]:
        """
        List registered models.

        Args:
            name: Filter by model name

        Returns:
            List of ModelInfo
        """
        if name is None:
            return list(self._models.values())

        return [
            info for info in self._models.values()
            if info.name == name
        ]

    def get_model_info(self, name: str, version: str) -> Optional[ModelInfo]:
        """Get info for specific model version."""
        key = self._get_model_key(name, version)
        return self._models.get(key)

    def compare_models(
        self,
        models: List[tuple],  # List of (name, version)
        metric: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Compare models by a specific metric.

        Args:
            models: List of (name, version) tuples
            metric: Metric to compare

        Returns:
            Dictionary mapping model key to metric value
        """
        results = {}

        for name, version in models:
            key = self._get_model_key(name, version)
            info = self._models.get(key)

            if info and metric in info.metrics:
                results[key] = info.metrics[metric]

        return results

    def delete(self, name: str, version: str) -> None:
        """
        Delete a model version.

        Args:
            name: Model name
            version: Version to delete
        """
        key = self._get_model_key(name, version)

        if key not in self._models:
            raise ValueError(f"Model {name}:{version} not found")

        info = self._models[key]

        # Check if deployed
        if info.status == 'deployed':
            raise ValueError("Cannot delete deployed model. Rollback first.")

        # Remove files
        model_dir = self.models_path / name / version
        if model_dir.exists():
            shutil.rmtree(model_dir)

        # Remove from registry
        del self._models[key]
        self._save_registry()

    def deprecate(self, name: str, version: str) -> None:
        """Mark a model version as deprecated."""
        key = self._get_model_key(name, version)

        if key not in self._models:
            raise ValueError(f"Model {name}:{version} not found")

        self._models[key].status = 'deprecated'
        self._save_registry()

    def export(self, name: str, version: str, export_path: Union[str, Path]) -> None:
        """
        Export a model to a different location.

        Args:
            name: Model name
            version: Version to export
            export_path: Destination path
        """
        key = self._get_model_key(name, version)

        if key not in self._models:
            raise ValueError(f"Model {name}:{version} not found")

        info = self._models[key]
        source = Path(info.path)
        dest = Path(export_path)

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

        # Also export metadata
        metadata_dest = dest.parent / f"{dest.stem}_metadata.json"
        with open(metadata_dest, 'w') as f:
            json.dump(info.to_dict(), f, indent=2)
