"""
Unit tests for the Config Loading Utility.

Tests validate:
- YAML loading and parsing
- Environment variable substitution
- Config validation
- Error handling
"""

import pytest
import os
import tempfile
from pathlib import Path

from triplegain.src.utils.config import (
    ConfigLoader,
    ConfigError,
    get_config_loader,
    load_config,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create indicators.yaml
        indicators_content = """
indicators:
  ema:
    periods: [9, 21, 50, 200]
    source: close
  sma:
    periods: [20, 50, 200]
    source: close
  rsi:
    period: 14
    source: close
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
  atr:
    period: 14
  bollinger_bands:
    period: 20
    std_dev: 2.0
"""
        (Path(tmpdir) / "indicators.yaml").write_text(indicators_content)

        # Create snapshot.yaml
        snapshot_content = """
snapshot_builder:
  candle_lookback:
    1m: 60
    5m: 48
    15m: 32
    1h: 48
  data_quality:
    max_age_seconds: 60
    min_candles_required: 20
"""
        (Path(tmpdir) / "snapshot.yaml").write_text(snapshot_content)

        # Create database.yaml
        database_content = """
database:
  connection:
    host: localhost
    port: 5432
    database: test_db
    user: test_user
    password: test_pass
  retention:
    agent_outputs_days: 90
"""
        (Path(tmpdir) / "database.yaml").write_text(database_content)

        # Create prompts.yaml
        prompts_content = """
agents:
  technical_analysis:
    tier: tier1_local
    template: ta_agent.txt
  risk_manager:
    tier: tier2_api
    template: risk_agent.txt
token_budgets:
  tier1_local:
    total: 4096
    buffer: 1000
  tier2_api:
    total: 8192
    buffer: 2000
"""
        (Path(tmpdir) / "prompts.yaml").write_text(prompts_content)

        yield tmpdir


@pytest.fixture
def config_loader(temp_config_dir):
    """Create a ConfigLoader with the temp config dir."""
    return ConfigLoader(temp_config_dir)


# =============================================================================
# ConfigLoader Tests
# =============================================================================

class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_init_with_valid_dir(self, temp_config_dir):
        """Test initialization with valid directory."""
        loader = ConfigLoader(temp_config_dir)
        assert loader.config_dir == Path(temp_config_dir)

    def test_init_with_invalid_dir(self):
        """Test initialization with invalid directory."""
        with pytest.raises(ConfigError, match="Config directory not found"):
            ConfigLoader("/nonexistent/path")

    def test_load_indicators_config(self, config_loader):
        """Test loading indicators config."""
        config = config_loader.load('indicators')

        assert 'indicators' in config
        assert 'ema' in config['indicators']
        assert config['indicators']['ema']['periods'] == [9, 21, 50, 200]

    def test_load_snapshot_config(self, config_loader):
        """Test loading snapshot config."""
        config = config_loader.load('snapshot')

        assert 'snapshot_builder' in config
        assert 'candle_lookback' in config['snapshot_builder']

    def test_load_database_config(self, config_loader):
        """Test loading database config."""
        config = config_loader.load('database')

        assert 'database' in config
        assert config['database']['connection']['host'] == 'localhost'

    def test_load_prompts_config(self, config_loader):
        """Test loading prompts config."""
        config = config_loader.load('prompts')

        assert 'agents' in config
        assert 'technical_analysis' in config['agents']

    def test_load_nonexistent_config(self, config_loader):
        """Test loading non-existent config file."""
        with pytest.raises(ConfigError, match="Config file not found"):
            config_loader.load('nonexistent')

    def test_config_caching(self, config_loader):
        """Test that configs are cached."""
        config1 = config_loader.load('indicators')
        config2 = config_loader.load('indicators')

        # Should be the same object (cached)
        assert config1 is config2

    def test_clear_cache(self, config_loader):
        """Test cache clearing."""
        config1 = config_loader.load('indicators')
        config_loader.clear_cache()
        config2 = config_loader.load('indicators')

        # Should be different objects after cache clear
        assert config1 is not config2

    def test_load_all(self, config_loader):
        """Test loading all configs."""
        configs = config_loader.load_all()

        assert 'indicators' in configs
        assert 'snapshot' in configs
        assert 'database' in configs
        assert 'prompts' in configs

    def test_get_indicators_config(self, config_loader):
        """Test get_indicators_config helper."""
        indicators = config_loader.get_indicators_config()

        assert 'ema' in indicators
        assert 'rsi' in indicators

    def test_get_snapshot_config(self, config_loader):
        """Test get_snapshot_config helper."""
        snapshot = config_loader.get_snapshot_config()

        assert 'candle_lookback' in snapshot

    def test_get_database_config(self, config_loader):
        """Test get_database_config helper."""
        db = config_loader.get_database_config()

        assert 'connection' in db


# =============================================================================
# Environment Variable Substitution Tests
# =============================================================================

class TestEnvVarSubstitution:
    """Tests for environment variable substitution."""

    def test_env_var_substitution(self):
        """Test ${VAR} substitution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set environment variable
            os.environ['TEST_DB_HOST'] = 'myhost.example.com'

            content = """
database:
  connection:
    host: ${TEST_DB_HOST}
    port: 5432
    database: test
    user: test
"""
            (Path(tmpdir) / "database.yaml").write_text(content)

            loader = ConfigLoader(tmpdir)
            config = loader.load('database', validate=False)

            assert config['database']['connection']['host'] == 'myhost.example.com'

            # Cleanup
            del os.environ['TEST_DB_HOST']

    def test_env_var_with_default(self):
        """Test ${VAR:-default} substitution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't set the env var, should use default
            content = """
database:
  connection:
    host: ${NONEXISTENT_VAR:-localhost}
    port: 5432
    database: test
    user: test
"""
            (Path(tmpdir) / "database.yaml").write_text(content)

            loader = ConfigLoader(tmpdir)
            config = loader.load('database', validate=False)

            assert config['database']['connection']['host'] == 'localhost'

    def test_env_var_missing_without_default(self):
        """Test ${VAR} without default when var is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """
database:
  connection:
    host: ${NONEXISTENT_VAR}
    port: 5432
    database: test
    user: test
"""
            (Path(tmpdir) / "database.yaml").write_text(content)

            loader = ConfigLoader(tmpdir)
            config = loader.load('database', validate=False)

            # Should be empty string (substituted to empty)
            # YAML parses empty string as None
            assert config['database']['connection']['host'] in ('', None)


# =============================================================================
# Validation Tests
# =============================================================================

class TestConfigValidation:
    """Tests for config validation."""

    def test_valid_indicators_config(self, config_loader):
        """Test validation passes for valid indicators config."""
        # Should not raise
        config = config_loader.load('indicators')
        assert config is not None

    def test_invalid_indicators_config(self):
        """Test validation fails for invalid indicators config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Missing required indicators
            content = """
indicators:
  custom_indicator:
    period: 14
"""
            (Path(tmpdir) / "indicators.yaml").write_text(content)

            loader = ConfigLoader(tmpdir)
            with pytest.raises(ConfigError, match="Missing required indicator"):
                loader.load('indicators')

    def test_invalid_ema_periods(self):
        """Test validation fails for invalid EMA periods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """
indicators:
  ema:
    periods: []  # Empty periods
    source: close
  sma:
    periods: [20]
  rsi:
    period: 14
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
  atr:
    period: 14
  bollinger_bands:
    period: 20
    std_dev: 2.0
"""
            (Path(tmpdir) / "indicators.yaml").write_text(content)

            loader = ConfigLoader(tmpdir)
            with pytest.raises(ConfigError, match="Invalid EMA periods"):
                loader.load('indicators')

    def test_invalid_snapshot_config(self):
        """Test validation fails for invalid snapshot config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """
snapshot_builder:
  candle_lookback:
    1h: 48
  data_quality:
    max_age_seconds: -1  # Invalid negative value
"""
            (Path(tmpdir) / "snapshot.yaml").write_text(content)

            loader = ConfigLoader(tmpdir)
            with pytest.raises(ConfigError, match="Invalid max_age_seconds"):
                loader.load('snapshot')

    def test_skip_validation(self):
        """Test that validation can be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Invalid config
            content = """
indicators:
  only_one_indicator:
    period: 14
"""
            (Path(tmpdir) / "indicators.yaml").write_text(content)

            loader = ConfigLoader(tmpdir)
            # Should not raise when validation is disabled
            config = loader.load('indicators', validate=False)
            assert config is not None


# =============================================================================
# Invalid YAML Tests
# =============================================================================

class TestInvalidYAML:
    """Tests for invalid YAML handling."""

    def test_invalid_yaml_syntax(self):
        """Test handling of invalid YAML syntax."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """
indicators:
  ema:
    periods: [1, 2, 3
  # Missing closing bracket - invalid YAML
"""
            (Path(tmpdir) / "indicators.yaml").write_text(content)

            loader = ConfigLoader(tmpdir)
            with pytest.raises(ConfigError, match="Invalid YAML"):
                loader.load('indicators')


# =============================================================================
# Real Config File Tests
# =============================================================================

class TestRealConfigFiles:
    """Tests that load from actual config files in the project."""

    @pytest.fixture
    def project_config_dir(self):
        """Get the path to the actual project config directory."""
        # Navigate from tests to config dir
        project_root = Path(__file__).parent.parent.parent.parent
        config_dir = project_root / "config"
        if config_dir.exists():
            return config_dir
        pytest.skip("Project config directory not found")

    def test_load_real_indicators_config(self, project_config_dir):
        """Test loading actual indicators.yaml from project."""
        loader = ConfigLoader(project_config_dir)
        config = loader.get_indicators_config()

        # Verify expected indicators exist
        assert 'ema' in config
        assert 'sma' in config
        assert 'rsi' in config
        assert 'macd' in config
        assert 'atr' in config
        assert 'bollinger_bands' in config

    def test_load_real_snapshot_config(self, project_config_dir):
        """Test loading actual snapshot.yaml from project."""
        loader = ConfigLoader(project_config_dir)
        config = loader.get_snapshot_config()

        # Verify expected structure
        assert 'candle_lookback' in config
        assert '1h' in config['candle_lookback']

    def test_load_real_database_config(self, project_config_dir):
        """Test loading actual database.yaml from project."""
        loader = ConfigLoader(project_config_dir)
        # Skip validation since env vars might not be set
        raw_config = loader.load('database', validate=False)

        # Verify expected structure
        assert 'database' in raw_config
        assert 'connection' in raw_config['database']

    def test_load_real_prompts_config(self, project_config_dir):
        """Test loading actual prompts.yaml from project."""
        loader = ConfigLoader(project_config_dir)
        # Skip validation since prompts config may not have agents yet
        raw_config = loader.load('prompts', validate=False)

        # Verify file can be loaded (structure may vary)
        assert raw_config is not None
        assert isinstance(raw_config, dict)

    def test_indicator_library_with_real_config(self, project_config_dir):
        """Test that IndicatorLibrary works with real config files."""
        from triplegain.src.data.indicator_library import IndicatorLibrary

        loader = ConfigLoader(project_config_dir)
        config = loader.get_indicators_config()

        # Create indicator library with real config
        library = IndicatorLibrary(config)

        # Test with sample data
        candles = [
            {'open': 100, 'high': 105, 'low': 98, 'close': 103, 'volume': 1000},
            {'open': 103, 'high': 108, 'low': 101, 'close': 106, 'volume': 1200},
            {'open': 106, 'high': 110, 'low': 104, 'close': 108, 'volume': 800},
        ] * 100  # 300 candles

        indicators = library.calculate_all("TEST/USDT", "1h", candles)

        # Verify indicators were calculated
        assert 'ema_9' in indicators or 'ema_21' in indicators
        assert 'rsi_14' in indicators
