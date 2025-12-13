"""
Phase 24: Enhanced Strategy Logger
Per-strategy logging with experiment tracking for parameter tuning.

Each strategy gets its own log file + shared experiment log.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib


class StrategyLogger:
    """
    Per-strategy logger with experiment tracking.

    Creates:
    - logs/strategies/{strategy_name}_{timestamp}.jsonl - Per-strategy log
    - logs/experiments/{experiment_id}.jsonl - Experiment-level tracking
    - logs/performance/{strategy_name}_metrics.json - Rolling performance metrics
    """

    def __init__(self, strategy_name: str, experiment_id: str = None, log_dir: str = "logs"):
        self.strategy_name = strategy_name
        self.log_dir = Path(log_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create subdirectories
        (self.log_dir / "strategies").mkdir(parents=True, exist_ok=True)
        (self.log_dir / "experiments").mkdir(parents=True, exist_ok=True)
        (self.log_dir / "performance").mkdir(parents=True, exist_ok=True)

        # Strategy-specific log
        self.strategy_log = self.log_dir / "strategies" / f"{strategy_name}_{self.timestamp}.jsonl"

        # Experiment tracking
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.experiment_log = self.log_dir / "experiments" / f"{self.experiment_id}.jsonl"

        # Performance metrics file
        self.metrics_file = self.log_dir / "performance" / f"{strategy_name}_metrics.json"

        # Running stats
        self.stats = {
            'strategy': strategy_name,
            'experiment_id': self.experiment_id,
            'start_time': datetime.now().isoformat(),
            'signals': {'buy': 0, 'sell': 0, 'short': 0, 'hold': 0, 'close': 0},
            'executions': {'success': 0, 'rejected': 0},
            'pnl': {'realized': 0.0, 'unrealized': 0.0, 'fees': 0.0},
            'trades': {'total': 0, 'wins': 0, 'losses': 0},
            'confidence_history': [],
            'config_params': {},
        }

        self._write_header()

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID based on timestamp + random."""
        return f"exp_{self.timestamp}_{hashlib.md5(os.urandom(8)).hexdigest()[:6]}"

    def _write_log(self, log_file: Path, data: Dict[str, Any]):
        """Write a log entry as JSON line."""
        with open(log_file, 'a') as f:
            f.write(json.dumps(data, default=str) + '\n')

    def _write_header(self):
        """Write session start header."""
        header = {
            'type': 'session_start',
            'strategy': self.strategy_name,
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'log_file': str(self.strategy_log)
        }
        self._write_log(self.strategy_log, header)

    def log_config(self, config: Dict[str, Any]):
        """Log strategy configuration for experiment tracking."""
        self.stats['config_params'] = config

        entry = {
            'type': 'config',
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy_name,
            'config': config
        }
        self._write_log(self.strategy_log, entry)
        self._write_log(self.experiment_log, entry)

    def log_signal(self,
                   action: str,
                   symbol: str,
                   confidence: float,
                   leverage: int = 1,
                   size: float = 0.0,
                   reason: str = "",
                   indicators: Dict[str, float] = None,
                   price: float = 0.0):
        """Log a generated signal."""

        # Update stats
        if action in self.stats['signals']:
            self.stats['signals'][action] += 1
        self.stats['confidence_history'].append(confidence)

        entry = {
            'type': 'signal',
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy_name,
            'action': action,
            'symbol': symbol,
            'confidence': confidence,
            'leverage': leverage,
            'size': size,
            'reason': reason,
            'price': price,
            'indicators': indicators or {}
        }
        self._write_log(self.strategy_log, entry)

    def log_execution(self,
                      action: str,
                      executed: bool,
                      price: float,
                      size: float,
                      reason: str = "",
                      result: Dict[str, Any] = None):
        """Log trade execution."""

        if executed:
            self.stats['executions']['success'] += 1
            self.stats['trades']['total'] += 1
        else:
            self.stats['executions']['rejected'] += 1

        entry = {
            'type': 'execution',
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy_name,
            'action': action,
            'executed': executed,
            'price': price,
            'size': size,
            'reason': reason,
            'result': result or {}
        }
        self._write_log(self.strategy_log, entry)

    def log_trade_close(self,
                        entry_price: float,
                        exit_price: float,
                        size: float,
                        leverage: int,
                        side: str,
                        fee: float = 0.0):
        """Log a closed trade with PnL calculation."""

        if side == 'long':
            pnl = (exit_price - entry_price) * size * leverage
        else:
            pnl = (entry_price - exit_price) * size * leverage

        pnl_after_fee = pnl - fee

        # Update stats
        self.stats['pnl']['realized'] += pnl_after_fee
        self.stats['pnl']['fees'] += fee

        if pnl_after_fee > 0:
            self.stats['trades']['wins'] += 1
        else:
            self.stats['trades']['losses'] += 1

        entry = {
            'type': 'trade_close',
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy_name,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'leverage': leverage,
            'gross_pnl': pnl,
            'fee': fee,
            'net_pnl': pnl_after_fee,
            'pnl_pct': ((exit_price / entry_price) - 1) * 100 * leverage if side == 'long' else ((entry_price / exit_price) - 1) * 100 * leverage
        }
        self._write_log(self.strategy_log, entry)

        return pnl_after_fee

    def log_market_state(self,
                         prices: Dict[str, float],
                         indicators: Dict[str, float],
                         regime: str = ""):
        """Log market state snapshot."""
        entry = {
            'type': 'market_state',
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy_name,
            'prices': prices,
            'indicators': indicators,
            'regime': regime
        }
        self._write_log(self.strategy_log, entry)

    def log_experiment_param(self, param_name: str, param_value: Any, description: str = ""):
        """Log an experiment parameter variation."""
        entry = {
            'type': 'experiment_param',
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id,
            'strategy': self.strategy_name,
            'param_name': param_name,
            'param_value': param_value,
            'description': description
        }
        self._write_log(self.experiment_log, entry)

    def log_insight(self, category: str, insight: str, data: Dict[str, Any] = None):
        """Log a tuning insight or observation."""
        entry = {
            'type': 'insight',
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy_name,
            'category': category,
            'insight': insight,
            'data': data or {}
        }
        self._write_log(self.strategy_log, entry)

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_trades = self.stats['trades']['total']
        wins = self.stats['trades']['wins']

        avg_confidence = (
            sum(self.stats['confidence_history']) / len(self.stats['confidence_history'])
            if self.stats['confidence_history'] else 0
        )

        return {
            'strategy': self.strategy_name,
            'experiment_id': self.experiment_id,
            'duration': str(datetime.now() - datetime.fromisoformat(self.stats['start_time'])),
            'signals': self.stats['signals'],
            'signal_rate': {
                k: v / max(sum(self.stats['signals'].values()), 1) * 100
                for k, v in self.stats['signals'].items()
            },
            'executions': self.stats['executions'],
            'execution_rate': (
                self.stats['executions']['success'] /
                max(self.stats['executions']['success'] + self.stats['executions']['rejected'], 1) * 100
            ),
            'trades': {
                'total': total_trades,
                'wins': wins,
                'losses': self.stats['trades']['losses'],
                'win_rate': wins / max(total_trades, 1) * 100
            },
            'pnl': {
                'realized': round(self.stats['pnl']['realized'], 2),
                'fees': round(self.stats['pnl']['fees'], 2),
                'net': round(self.stats['pnl']['realized'] - self.stats['pnl']['fees'], 2)
            },
            'confidence': {
                'avg': round(avg_confidence, 3),
                'min': round(min(self.stats['confidence_history']) if self.stats['confidence_history'] else 0, 3),
                'max': round(max(self.stats['confidence_history']) if self.stats['confidence_history'] else 0, 3)
            },
            'config_params': self.stats['config_params']
        }

    def close(self) -> Dict[str, Any]:
        """Close logger and generate final summary."""
        summary = self.get_summary()

        # Write session end
        end_entry = {
            'type': 'session_end',
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy_name,
            'summary': summary
        }
        self._write_log(self.strategy_log, end_entry)
        self._write_log(self.experiment_log, end_entry)

        # Update rolling metrics file
        self._update_rolling_metrics(summary)

        print(f"\n[{self.strategy_name}] Session Summary:")
        print(f"  Signals: buy={summary['signals']['buy']}, sell={summary['signals']['sell']}, hold={summary['signals']['hold']}")
        print(f"  Trades: {summary['trades']['total']} (Win rate: {summary['trades']['win_rate']:.1f}%)")
        print(f"  PnL: ${summary['pnl']['net']:.2f}")
        print(f"  Avg Confidence: {summary['confidence']['avg']:.2f}")
        print(f"  Log: {self.strategy_log}")

        return summary

    def _update_rolling_metrics(self, summary: Dict[str, Any]):
        """Update rolling metrics file for historical comparison."""
        metrics_history = []

        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics_history = json.load(f)
            except:
                metrics_history = []

        # Add current session
        metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id,
            'summary': summary
        })

        # Keep last 100 sessions
        metrics_history = metrics_history[-100:]

        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=2, default=str)


class StrategyLoggerManager:
    """
    Manages multiple strategy loggers for unified orchestration.
    """

    def __init__(self, experiment_id: str = None, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = experiment_id or f"unified_{self.timestamp}"
        self.loggers: Dict[str, StrategyLogger] = {}

        # Master log for orchestrator-level events
        (self.log_dir / "orchestrator").mkdir(parents=True, exist_ok=True)
        self.master_log = self.log_dir / "orchestrator" / f"unified_{self.timestamp}.jsonl"

        self._write_master_header()

    def _write_master_header(self):
        """Write master log header."""
        header = {
            'type': 'unified_session_start',
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.master_log, 'a') as f:
            f.write(json.dumps(header, default=str) + '\n')

    def get_logger(self, strategy_name: str) -> StrategyLogger:
        """Get or create a logger for a strategy."""
        if strategy_name not in self.loggers:
            self.loggers[strategy_name] = StrategyLogger(
                strategy_name=strategy_name,
                experiment_id=self.experiment_id,
                log_dir=str(self.log_dir)
            )
        return self.loggers[strategy_name]

    def log_orchestrator_decision(self,
                                   strategy_signals: Dict[str, Dict],
                                   final_action: str,
                                   final_confidence: float,
                                   weights: Dict[str, float],
                                   regime: str):
        """Log the orchestrator's combined decision."""
        entry = {
            'type': 'orchestrator_decision',
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id,
            'strategy_signals': {
                name: {
                    'action': sig.get('action', 'hold'),
                    'confidence': sig.get('confidence', 0),
                    'reason': sig.get('reason', '')
                }
                for name, sig in strategy_signals.items()
            },
            'weights': weights,
            'final_action': final_action,
            'final_confidence': final_confidence,
            'regime': regime
        }
        with open(self.master_log, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def log_experiment_config(self, config: Dict[str, Any]):
        """Log the full experiment configuration."""
        entry = {
            'type': 'experiment_config',
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id,
            'config': config
        }
        with open(self.master_log, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def close_all(self) -> Dict[str, Any]:
        """Close all loggers and generate combined summary."""
        summaries = {}

        for name, logger in self.loggers.items():
            summaries[name] = logger.close()

        # Write master summary
        master_summary = {
            'type': 'unified_session_end',
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id,
            'strategy_summaries': summaries,
            'total_strategies': len(summaries),
            'combined_pnl': sum(s['pnl']['net'] for s in summaries.values()),
            'combined_trades': sum(s['trades']['total'] for s in summaries.values())
        }

        with open(self.master_log, 'a') as f:
            f.write(json.dumps(master_summary, default=str) + '\n')

        print(f"\n{'='*60}")
        print(f"UNIFIED SESSION SUMMARY")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"{'='*60}")
        print(f"Strategies run: {len(summaries)}")
        print(f"Combined PnL: ${master_summary['combined_pnl']:.2f}")
        print(f"Total trades: {master_summary['combined_trades']}")
        print(f"Master log: {self.master_log}")
        print(f"{'='*60}")

        return master_summary
