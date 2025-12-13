"""
Strategy auto-discovery and loading for WebSocket Paper Tester.
Automatically discovers and loads strategies from the strategies/ directory.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

from .types import DataSnapshot, Signal


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
            except Exception:
                self.errors += 1

    def on_start(self):
        """Called when strategy starts."""
        if self._on_start:
            try:
                self._on_start(self.config, self.state)
            except Exception:
                self.errors += 1

    def on_stop(self):
        """Called when strategy stops."""
        if self._on_stop:
            try:
                self._on_stop(self.state)
            except Exception:
                self.errors += 1

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


def discover_strategies(strategies_dir: str = "strategies") -> Dict[str, StrategyWrapper]:
    """
    Auto-discover strategies from directory.

    Each .py file in strategies/ with STRATEGY_NAME is loaded.

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
    """
    strategies = {}
    strategies_path = Path(strategies_dir)

    if not strategies_path.exists():
        print(f"[Loader] Strategies directory '{strategies_dir}' not found")
        return strategies

    # Add parent to Python path for imports
    parent_path = str(strategies_path.parent.absolute())
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)

    print(f"[Loader] Discovering strategies in {strategies_path.absolute()}")

    for file in strategies_path.glob("*.py"):
        if file.name.startswith("_"):
            continue

        try:
            # Load module
            module_name = f"{strategies_dir}.{file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file)
            if not spec or not spec.loader:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

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
