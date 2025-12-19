"""
Configuration Loading Utility - Centralized config management for TripleGain.

This module provides:
- YAML configuration loading with validation
- Environment variable substitution
- Config validation and schema enforcement
- Cached config access
- Thread-safe global config instance
"""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Thread lock for global config loader access
_config_lock = threading.Lock()


class ConfigError(Exception):
    """Configuration loading or validation error."""
    pass


class ConfigLoader:
    """
    Centralized configuration loader for TripleGain.

    Loads YAML configs with environment variable substitution
    and validation.
    """

    # Environment variable pattern: ${VAR_NAME:-default_value}
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::-([^}]*))?\}')

    def __init__(self, config_dir: str | Path):
        """
        Initialize ConfigLoader.

        Args:
            config_dir: Path to configuration directory
        """
        self.config_dir = Path(config_dir)
        self._cache: dict[str, dict] = {}

        if not self.config_dir.exists():
            raise ConfigError(f"Config directory not found: {self.config_dir}")

    def load(self, config_name: str, validate: bool = True) -> dict:
        """
        Load a configuration file.

        Args:
            config_name: Config file name (without .yaml extension)
            validate: Whether to validate the config

        Returns:
            Configuration dictionary
        """
        if config_name in self._cache:
            return self._cache[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                raw_content = f.read()

            # Substitute environment variables
            substituted_content = self._substitute_env_vars(raw_content)

            config = yaml.safe_load(substituted_content)

            # Coerce types (env vars come as strings)
            config = self._coerce_types(config)

            if validate:
                self._validate_config(config_name, config)

            self._cache[config_name] = config
            logger.info(f"Loaded config: {config_name}")
            return config

        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_path}: {e}")

    def load_all(self) -> dict[str, dict]:
        """Load all configuration files in the config directory."""
        configs = {}

        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            try:
                configs[config_name] = self.load(config_name)
            except ConfigError as e:
                logger.warning(f"Failed to load {config_name}: {e}")

        return configs

    def get_indicators_config(self) -> dict:
        """Get the indicators configuration."""
        config = self.load('indicators')
        return config.get('indicators', {})

    def get_snapshot_config(self) -> dict:
        """Get the snapshot builder configuration."""
        config = self.load('snapshot')
        return config.get('snapshot_builder', {})

    def get_database_config(self) -> dict:
        """Get the database configuration."""
        config = self.load('database')
        return config.get('database', {})

    def get_prompts_config(self) -> dict:
        """Get the prompts configuration."""
        return self.load('prompts')

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()

    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in config content.

        Supports ${VAR_NAME} and ${VAR_NAME:-default_value} syntax.
        """
        def replace_match(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ''
            return os.environ.get(var_name, default_value)

        return self.ENV_VAR_PATTERN.sub(replace_match, content)

    def _coerce_types(self, value: Any) -> Any:
        """
        Recursively coerce string values to appropriate types.

        Handles:
        - Numeric strings to int/float
        - Boolean strings to bool
        - Nested dicts and lists

        Args:
            value: Value to coerce

        Returns:
            Coerced value
        """
        if isinstance(value, dict):
            return {k: self._coerce_types(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._coerce_types(item) for item in value]
        elif isinstance(value, str):
            # Try boolean first (before numeric to avoid "true" -> error)
            if value.lower() in ('true', 'yes', 'on'):
                return True
            if value.lower() in ('false', 'no', 'off'):
                return False

            # Skip special float values that shouldn't be converted
            if value.lower() in ('inf', '-inf', 'nan', 'infinity', '-infinity'):
                return value

            # Try integer
            if value.lstrip('-').isdigit():
                try:
                    return int(value)
                except ValueError:
                    pass

            # Try float (including scientific notation like 1e-5, 2.5E10)
            try:
                float_val = float(value)
                # Verify it's a valid number (not inf/nan from string)
                import math
                if math.isfinite(float_val):
                    return float_val
            except ValueError:
                pass

            return value
        return value

    def _validate_config(self, config_name: str, config: dict) -> None:
        """
        Validate configuration based on schema.

        Args:
            config_name: Name of the config file
            config: Loaded configuration dict
        """
        validators = {
            'indicators': self._validate_indicators_config,
            'snapshot': self._validate_snapshot_config,
            'database': self._validate_database_config,
            'prompts': self._validate_prompts_config,
        }

        validator = validators.get(config_name)
        if validator:
            validator(config)

    def _validate_indicators_config(self, config: dict) -> None:
        """Validate indicators configuration."""
        indicators = config.get('indicators', {})

        # Check required indicators
        required = ['ema', 'sma', 'rsi', 'macd', 'atr', 'bollinger_bands']
        for ind in required:
            if ind not in indicators:
                raise ConfigError(f"Missing required indicator config: {ind}")

        # Validate EMA periods
        ema = indicators.get('ema', {})
        periods = ema.get('periods', [])
        if not periods or not all(isinstance(p, int) and p > 0 for p in periods):
            raise ConfigError("Invalid EMA periods configuration")

        # Validate RSI period
        rsi = indicators.get('rsi', {})
        rsi_period = rsi.get('period', 14)
        if not isinstance(rsi_period, int) or rsi_period <= 0:
            raise ConfigError("Invalid RSI period")

        logger.debug("Indicators config validated successfully")

    def _validate_snapshot_config(self, config: dict) -> None:
        """Validate snapshot configuration."""
        builder = config.get('snapshot_builder', {})

        # Check candle lookback
        lookback = builder.get('candle_lookback', {})
        if not lookback:
            raise ConfigError("Missing candle_lookback configuration")

        # Check data quality settings
        quality = builder.get('data_quality', {})
        max_age = quality.get('max_age_seconds', 60)
        if not isinstance(max_age, int) or max_age <= 0:
            raise ConfigError("Invalid max_age_seconds")

        logger.debug("Snapshot config validated successfully")

    def _validate_database_config(self, config: dict) -> None:
        """Validate database configuration."""
        db = config.get('database', {})

        # Check connection settings
        conn = db.get('connection', {})
        required_conn = ['host', 'port', 'database', 'user']
        for field in required_conn:
            if field not in conn:
                raise ConfigError(f"Missing database connection field: {field}")

        # Check retention settings
        retention = db.get('retention', {})
        if not retention:
            logger.warning("No retention policies configured")

        logger.debug("Database config validated successfully")

    def _validate_prompts_config(self, config: dict) -> None:
        """Validate prompts configuration."""
        agents = config.get('agents', {})
        if not agents:
            raise ConfigError("No agents configured in prompts config")

        token_budgets = config.get('token_budgets', {})
        if not token_budgets:
            raise ConfigError("No token budgets configured")

        logger.debug("Prompts config validated successfully")


# Global config instance (lazy-loaded, thread-safe)
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: str | Path | None = None) -> ConfigLoader:
    """
    Get or create the global ConfigLoader instance (thread-safe).

    Args:
        config_dir: Path to config directory (uses default if not provided)

    Returns:
        ConfigLoader instance
    """
    global _config_loader

    # Double-checked locking pattern for thread safety
    if _config_loader is None:
        with _config_lock:
            # Check again inside lock
            if _config_loader is None:
                if config_dir is None:
                    # Default to config/ relative to project root
                    project_root = Path(__file__).parent.parent.parent.parent
                    config_dir = project_root / 'config'

                _config_loader = ConfigLoader(config_dir)

    return _config_loader


def reset_config_loader() -> None:
    """
    Reset the global ConfigLoader instance (for testing).

    This allows tests to re-initialize the config loader with
    different settings. Thread-safe.
    """
    global _config_loader
    with _config_lock:
        _config_loader = None
        logger.debug("Global config loader reset")


def load_config(config_name: str) -> dict:
    """
    Convenience function to load a config file.

    Args:
        config_name: Config file name (without .yaml extension)

    Returns:
        Configuration dictionary
    """
    return get_config_loader().load(config_name)
