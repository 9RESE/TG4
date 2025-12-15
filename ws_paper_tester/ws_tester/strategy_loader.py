"""
Strategy auto-discovery and loading for WebSocket Paper Tester.
Automatically discovers and loads strategies from the strategies/ directory.

Security Features:
- Whitelist of approved strategy files
- SHA256 hash verification for known good strategies
- Validation before loading
"""

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Set

from .types import DataSnapshot, Signal


# Security configuration
STRATEGY_SECURITY_FILE = "strategy_hashes.json"  # File containing approved hashes
ALLOW_UNSIGNED_STRATEGIES = True  # Set to False in production to require hash verification


class StrategyWrapper:
    """Wrapper for a loaded strategy module."""

    def __init__(
        self,
        name: str,
        version: str,
        symbols: List[str],
        config: dict,
        generate_signal: Callable,
        on_fill: Optional[Callable] = None,
        on_start: Optional[Callable] = None,
        on_stop: Optional[Callable] = None,
    ):
        self.name = name
        self.version = version
        self.symbols = symbols
        self.config = config
        self.state: Dict[str, Any] = {}  # Mutable strategy state

        self._generate_signal = generate_signal
        self._on_fill = on_fill
        self._on_start = on_start
        self._on_stop = on_stop

        # Stats
        self.signals_generated = 0
        self.errors = 0
        self.last_signal_time = None

    def generate_signal(self, data: DataSnapshot) -> Optional[Signal]:
        """Generate trading signal from market data."""
        try:
            signal = self._generate_signal(data, self.config, self.state)
            if signal:
                self.signals_generated += 1
                self.last_signal_time = data.timestamp
            return signal
        except Exception as e:
            self.errors += 1
            raise

    def on_fill(self, fill: dict):
        """Notify strategy of a fill."""
        if self._on_fill:
            try:
                self._on_fill(fill, self.state)
            except Exception as e:
                # Catch all exceptions from user-provided strategy code
                self.errors += 1
                print(f"[Strategy:{self.name}] on_fill error: {e}")

    def on_start(self):
        """Called when strategy starts."""
        if self._on_start:
            try:
                self._on_start(self.config, self.state)
            except Exception as e:
                # Catch all exceptions from user-provided strategy code
                self.errors += 1
                print(f"[Strategy:{self.name}] on_start error: {e}")

    def on_stop(self):
        """Called when strategy stops."""
        if self._on_stop:
            try:
                self._on_stop(self.state)
            except Exception as e:
                # Catch all exceptions from user-provided strategy code
                self.errors += 1
                print(f"[Strategy:{self.name}] on_stop error: {e}")

    def to_dict(self) -> dict:
        """Serialize strategy info."""
        return {
            'name': self.name,
            'version': self.version,
            'symbols': self.symbols,
            'config': self.config,
            'signals_generated': self.signals_generated,
            'errors': self.errors,
        }


def _calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _load_strategy_hashes(strategies_dir: Path) -> Dict[str, str]:
    """Load approved strategy hashes from security file."""
    hash_file = strategies_dir / STRATEGY_SECURITY_FILE
    if hash_file.exists():
        try:
            with open(hash_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Loader] Warning: Could not load strategy hashes: {e}")
    return {}


def _save_strategy_hashes(strategies_dir: Path, hashes: Dict[str, str]) -> None:
    """Save strategy hashes to security file."""
    hash_file = strategies_dir / STRATEGY_SECURITY_FILE
    try:
        with open(hash_file, 'w') as f:
            json.dump(hashes, f, indent=2)
        print(f"[Loader] Strategy hashes saved to {hash_file}")
    except IOError as e:
        print(f"[Loader] Warning: Could not save strategy hashes: {e}")


def verify_strategy_file(
    filepath: Path,
    approved_hashes: Dict[str, str],
    allow_unsigned: bool = ALLOW_UNSIGNED_STRATEGIES
) -> tuple[bool, str]:
    """
    Verify a strategy file against approved hashes.

    Returns:
        (is_valid, reason) tuple
    """
    filename = filepath.name
    file_hash = _calculate_file_hash(filepath)

    if filename in approved_hashes:
        expected_hash = approved_hashes[filename]
        if file_hash == expected_hash:
            return True, "Hash verified"
        else:
            return False, f"Hash mismatch: expected {expected_hash[:16]}..., got {file_hash[:16]}..."

    if allow_unsigned:
        return True, f"Unsigned strategy (hash: {file_hash[:16]}...)"
    else:
        return False, "Strategy not in approved list (unsigned strategies disabled)"


def generate_strategy_hashes(strategies_dir: str = "strategies") -> Dict[str, str]:
    """
    Generate hashes for all strategy files in a directory.
    Useful for creating the initial strategy_hashes.json file.
    """
    strategies_path = Path(strategies_dir)
    hashes = {}

    if not strategies_path.exists():
        return hashes

    for file in strategies_path.glob("*.py"):
        if file.name.startswith("_"):
            continue
        hashes[file.name] = _calculate_file_hash(file)

    return hashes


def discover_strategies(
    strategies_dir: str = "strategies",
    whitelist: Optional[Set[str]] = None,
    verify_hashes: bool = True
) -> Dict[str, StrategyWrapper]:
    """
    Auto-discover strategies from directory.

    Each .py file in strategies/ with STRATEGY_NAME is loaded.

    Security:
        - Optional whitelist of allowed strategy files
        - Hash verification against approved_hashes file
        - Set ALLOW_UNSIGNED_STRATEGIES=False to require verification

    Required module attributes:
        - STRATEGY_NAME: str - Unique strategy identifier
        - generate_signal(data, config, state) -> Optional[Signal]

    Optional module attributes:
        - STRATEGY_VERSION: str - Version string (default: "0.0.0")
        - SYMBOLS: List[str] - Symbols this strategy trades (default: [])
        - CONFIG: dict - Default configuration (default: {})
        - on_fill(fill, state) - Called when an order is filled
        - on_start(config, state) - Called when strategy starts
        - on_stop(state) - Called when strategy stops

    Args:
        strategies_dir: Directory containing strategy files
        whitelist: Optional set of allowed strategy filenames (e.g., {"market_making.py"})
        verify_hashes: Whether to verify file hashes against approved list
    """
    strategies = {}
    strategies_path = Path(strategies_dir)

    if not strategies_path.exists():
        print(f"[Loader] Strategies directory '{strategies_dir}' not found")
        return strategies

    # Load approved hashes for verification
    approved_hashes = {}
    if verify_hashes:
        approved_hashes = _load_strategy_hashes(strategies_path)
        if approved_hashes:
            print(f"[Loader] Loaded {len(approved_hashes)} approved strategy hashes")

    print(f"[Loader] Discovering strategies in {strategies_path.absolute()}")

    for file in strategies_path.glob("*.py"):
        if file.name.startswith("_"):
            continue

        # Whitelist check
        if whitelist is not None and file.name not in whitelist:
            print(f"  - Skipping {file.name}: not in whitelist")
            continue

        # Security verification
        if verify_hashes:
            is_valid, reason = verify_strategy_file(file, approved_hashes)
            if not is_valid:
                print(f"  ! BLOCKED {file.name}: {reason}")
                continue
            else:
                print(f"  * Verified {file.name}: {reason}")

        try:
            # Add strategies directory parent to sys.path for proper package imports
            strategies_parent = str(strategies_path.parent)
            if strategies_parent not in sys.path:
                sys.path.insert(0, strategies_parent)

            # Use package import to support relative imports in shim files
            # This treats strategies/ as a package and imports e.g. strategies.momentum_scalping
            package_name = strategies_path.name  # "strategies"
            module_name = f"{package_name}.{file.stem}"  # e.g. "strategies.momentum_scalping"

            # Remove from sys.modules if already loaded (for reloading)
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Import the module properly as part of the strategies package
            module = importlib.import_module(module_name)

            # Check for required attributes
            if not hasattr(module, 'STRATEGY_NAME'):
                print(f"  - Skipping {file.name}: missing STRATEGY_NAME")
                continue

            if not hasattr(module, 'generate_signal'):
                print(f"  - Skipping {file.name}: missing generate_signal function")
                continue

            # Create wrapper
            strategy = StrategyWrapper(
                name=module.STRATEGY_NAME,
                version=getattr(module, 'STRATEGY_VERSION', '0.0.0'),
                symbols=getattr(module, 'SYMBOLS', []),
                config=getattr(module, 'CONFIG', {}).copy(),
                generate_signal=module.generate_signal,
                on_fill=getattr(module, 'on_fill', None),
                on_start=getattr(module, 'on_start', None),
                on_stop=getattr(module, 'on_stop', None),
            )

            strategies[strategy.name] = strategy
            print(f"  + Loaded: {strategy.name} v{strategy.version} ({len(strategy.symbols)} symbols)")

        except Exception as e:
            print(f"  ! Failed to load {file.name}: {e}")

    print(f"[Loader] Loaded {len(strategies)} strategies")
    return strategies


def load_strategy_from_file(filepath: str) -> Optional[StrategyWrapper]:
    """Load a single strategy from a file path."""
    path = Path(filepath)

    if not path.exists():
        print(f"[Loader] Strategy file not found: {filepath}")
        return None

    try:
        module_name = f"strategy_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, 'STRATEGY_NAME') or not hasattr(module, 'generate_signal'):
            print(f"[Loader] Invalid strategy file: missing required attributes")
            return None

        strategy = StrategyWrapper(
            name=module.STRATEGY_NAME,
            version=getattr(module, 'STRATEGY_VERSION', '0.0.0'),
            symbols=getattr(module, 'SYMBOLS', []),
            config=getattr(module, 'CONFIG', {}).copy(),
            generate_signal=module.generate_signal,
            on_fill=getattr(module, 'on_fill', None),
            on_start=getattr(module, 'on_start', None),
            on_stop=getattr(module, 'on_stop', None),
        )

        print(f"[Loader] Loaded: {strategy.name} v{strategy.version}")
        return strategy

    except Exception as e:
        print(f"[Loader] Failed to load {filepath}: {e}")
        return None


def validate_strategy(strategy: StrategyWrapper) -> List[str]:
    """
    Validate a strategy and return list of issues.
    Empty list means strategy is valid.
    """
    issues = []

    if not strategy.name:
        issues.append("STRATEGY_NAME is empty")

    if not strategy.symbols:
        issues.append("SYMBOLS list is empty - strategy won't trade any symbols")

    if not callable(strategy._generate_signal):
        issues.append("generate_signal is not callable")

    # Test generate_signal signature
    import inspect
    sig = inspect.signature(strategy._generate_signal)
    params = list(sig.parameters.keys())

    if len(params) < 3:
        issues.append(f"generate_signal should take (data, config, state), has {len(params)} params")

    return issues


def get_all_symbols(strategies: Dict[str, StrategyWrapper]) -> List[str]:
    """Get unique list of all symbols needed by loaded strategies."""
    symbols = set()
    for strategy in strategies.values():
        symbols.update(strategy.symbols)
    return sorted(list(symbols))
