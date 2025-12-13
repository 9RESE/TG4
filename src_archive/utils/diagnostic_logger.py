"""
Phase 23: Diagnostic Logger for Strategy Tuning
Captures detailed metrics for post-run analysis and optimization.

Log file: logs/diagnostic_{timestamp}.jsonl
Format: JSON Lines (one JSON object per line for easy parsing)
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class DiagnosticLogger:
    """
    Comprehensive diagnostic logger for ensemble trading analysis.

    Captures:
    - Individual strategy signals and confidence levels
    - Regime detection and feature values
    - Weighted vote calculations
    - Execution decisions and rejections
    - Market conditions at decision time
    - Performance metrics
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"diagnostic_{timestamp}.jsonl"
        self.summary_file = self.log_dir / f"summary_{timestamp}.json"

        # Running statistics
        self.stats = {
            'total_decisions': 0,
            'signals_by_strategy': {},
            'signals_by_action': {'buy': 0, 'sell': 0, 'short': 0, 'hold': 0},
            'executions': {'executed': 0, 'rejected': 0},
            'rejection_reasons': {},
            'regimes': {},
            'confidence_distribution': [],
            'strategy_accuracy': {},  # Track which strategies led to profitable trades
            'missed_opportunities': [],  # Hold when price moved significantly
            'false_signals': [],  # Action that lost money
        }

        # Write header
        self._write_log({
            'type': 'session_start',
            'timestamp': datetime.now().isoformat(),
            'log_file': str(self.log_file)
        })

        print(f"DiagnosticLogger: Writing to {self.log_file}")

    def _write_log(self, data: Dict[str, Any]):
        """Write a log entry as JSON line."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data, default=str) + '\n')

    def log_market_state(self,
                         prices: Dict[str, float],
                         features: Dict[str, float],
                         regime: str,
                         rsi: Dict[str, float],
                         volatility: float):
        """Log current market conditions."""
        entry = {
            'type': 'market_state',
            'timestamp': datetime.now().isoformat(),
            'prices': prices,
            'features': {
                'atr_xrp': features.get('atr_xrp', 0),
                'atr_btc': features.get('atr_btc', 0),
                'correlation': features.get('correlation', 0),
                'vwap_dev_xrp': features.get('vwap_dev_xrp', 0),
                'xrp_rsi': features.get('xrp_rsi', 50),
                'btc_rsi': features.get('btc_rsi', 50),
            },
            'regime': regime,
            'rsi': rsi,
            'volatility': volatility,
            'daily_atr_equiv': features.get('atr_xrp', 0) * 4.9  # sqrt(24) approximation
        }
        self._write_log(entry)

        # Update stats
        self.stats['regimes'][regime] = self.stats['regimes'].get(regime, 0) + 1

    def log_strategy_signals(self,
                             signals: Dict[str, Dict[str, Any]],
                             weights: Dict[str, float]):
        """Log individual strategy signals before voting."""
        entry = {
            'type': 'strategy_signals',
            'timestamp': datetime.now().isoformat(),
            'signals': {},
            'weights': weights
        }

        for name, signal in signals.items():
            action = signal.get('action', 'hold')
            confidence = signal.get('confidence', 0)

            entry['signals'][name] = {
                'action': action,
                'confidence': confidence,
                'reason': signal.get('reason', ''),
                'symbol': signal.get('symbol', ''),
                'leverage': signal.get('leverage', 1),
                'size': signal.get('size', 0),
                'indicators': signal.get('indicators', {})
            }

            # Update stats
            if name not in self.stats['signals_by_strategy']:
                self.stats['signals_by_strategy'][name] = {
                    'buy': 0, 'sell': 0, 'short': 0, 'hold': 0, 'close': 0,
                    'close_all': 0, 'close_hedge': 0,
                    'avg_confidence': [], 'triggered': 0
                }

            # Handle unknown action types gracefully
            if action not in self.stats['signals_by_strategy'][name]:
                self.stats['signals_by_strategy'][name][action] = 0
            self.stats['signals_by_strategy'][name][action] += 1
            if action != 'hold':
                self.stats['signals_by_strategy'][name]['triggered'] += 1
                self.stats['signals_by_strategy'][name]['avg_confidence'].append(confidence)

        self._write_log(entry)

    def log_weighted_vote(self,
                          action_scores: Dict[str, Dict[str, Any]],
                          final_action: str,
                          final_confidence: float,
                          btc_momentum_bias: float):
        """Log weighted vote calculation details."""
        entry = {
            'type': 'weighted_vote',
            'timestamp': datetime.now().isoformat(),
            'action_scores': {
                action: {
                    'score': info.get('score', 0),
                    'num_signals': len(info.get('signals', [])),
                    'total_confidence': info.get('total_confidence', 0)
                }
                for action, info in action_scores.items()
            },
            'final_action': final_action,
            'final_confidence': final_confidence,
            'btc_momentum_bias': btc_momentum_bias
        }
        self._write_log(entry)

        # Update stats
        self.stats['signals_by_action'][final_action] += 1
        self.stats['confidence_distribution'].append(final_confidence)

    def log_decision(self,
                     signal: Dict[str, Any],
                     regime: str,
                     portfolio_state: Dict[str, float]):
        """Log final ensemble decision."""
        self.stats['total_decisions'] += 1

        entry = {
            'type': 'decision',
            'timestamp': datetime.now().isoformat(),
            'decision_id': self.stats['total_decisions'],
            'action': signal.get('action', 'hold'),
            'symbol': signal.get('symbol', ''),
            'confidence': signal.get('confidence', 0),
            'leverage': signal.get('leverage', 1),
            'size': signal.get('size', 0),
            'reason': signal.get('reason', ''),
            'strategy': signal.get('strategy', 'ensemble'),
            'contributing_strategies': signal.get('contributing_strategies', []),
            'regime': regime,
            'portfolio': portfolio_state,
            'momentum_seed': signal.get('momentum_seed', False),
            'privileged_override': signal.get('privileged_override', False)
        }
        self._write_log(entry)

    def log_execution(self,
                      signal: Dict[str, Any],
                      executed: bool,
                      reason: str,
                      result: Dict[str, Any] = None):
        """Log execution attempt and result."""
        entry = {
            'type': 'execution',
            'timestamp': datetime.now().isoformat(),
            'action': signal.get('action', 'hold'),
            'executed': executed,
            'reason': reason,
            'confidence': signal.get('confidence', 0),
            'confidence_threshold': result.get('confidence_threshold', 0) if result else 0,
            'btc_atr': result.get('btc_atr', 0) if result else 0,
            'result': result
        }
        self._write_log(entry)

        # Update stats
        if executed:
            self.stats['executions']['executed'] += 1
        else:
            self.stats['executions']['rejected'] += 1
            self.stats['rejection_reasons'][reason] = \
                self.stats['rejection_reasons'].get(reason, 0) + 1

    def log_price_change(self,
                         symbol: str,
                         price_before: float,
                         price_after: float,
                         time_delta_minutes: int,
                         last_action: str):
        """Log price changes to detect missed opportunities."""
        pct_change = (price_after - price_before) / price_before * 100

        entry = {
            'type': 'price_change',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'price_before': price_before,
            'price_after': price_after,
            'pct_change': pct_change,
            'time_delta_minutes': time_delta_minutes,
            'last_action': last_action
        }
        self._write_log(entry)

        # Track missed opportunities (held while price moved >1%)
        if last_action == 'hold' and abs(pct_change) > 1.0:
            self.stats['missed_opportunities'].append({
                'symbol': symbol,
                'pct_change': pct_change,
                'direction': 'up' if pct_change > 0 else 'down'
            })

    def log_trade_result(self,
                         strategy: str,
                         action: str,
                         entry_price: float,
                         exit_price: float,
                         pnl_pct: float):
        """Log trade result for strategy accuracy tracking."""
        entry = {
            'type': 'trade_result',
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'profitable': pnl_pct > 0
        }
        self._write_log(entry)

        # Update strategy accuracy
        if strategy not in self.stats['strategy_accuracy']:
            self.stats['strategy_accuracy'][strategy] = {'wins': 0, 'losses': 0, 'total_pnl': 0}

        if pnl_pct > 0:
            self.stats['strategy_accuracy'][strategy]['wins'] += 1
        else:
            self.stats['strategy_accuracy'][strategy]['losses'] += 1
        self.stats['strategy_accuracy'][strategy]['total_pnl'] += pnl_pct

    def log_tuning_insight(self, category: str, insight: str, data: Dict[str, Any] = None):
        """Log a tuning insight or anomaly detected."""
        entry = {
            'type': 'tuning_insight',
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'insight': insight,
            'data': data or {}
        }
        self._write_log(entry)

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for analysis."""
        summary = {
            'session_end': datetime.now().isoformat(),
            'total_decisions': self.stats['total_decisions'],
            'executions': self.stats['executions'],
            'execution_rate': (
                self.stats['executions']['executed'] /
                max(self.stats['executions']['executed'] + self.stats['executions']['rejected'], 1)
            ) * 100,

            # Action distribution
            'action_distribution': self.stats['signals_by_action'],
            'action_percentages': {
                action: count / max(self.stats['total_decisions'], 1) * 100
                for action, count in self.stats['signals_by_action'].items()
            },

            # Regime distribution
            'regime_distribution': self.stats['regimes'],

            # Rejection analysis
            'rejection_reasons': self.stats['rejection_reasons'],

            # Strategy performance
            'strategy_signals': {},

            # Confidence analysis
            'confidence_stats': {
                'mean': sum(self.stats['confidence_distribution']) / max(len(self.stats['confidence_distribution']), 1),
                'min': min(self.stats['confidence_distribution']) if self.stats['confidence_distribution'] else 0,
                'max': max(self.stats['confidence_distribution']) if self.stats['confidence_distribution'] else 0,
                'below_threshold': len([c for c in self.stats['confidence_distribution'] if c < 0.35])
            },

            # Missed opportunities
            'missed_opportunities': {
                'count': len(self.stats['missed_opportunities']),
                'up_moves': len([m for m in self.stats['missed_opportunities'] if m['direction'] == 'up']),
                'down_moves': len([m for m in self.stats['missed_opportunities'] if m['direction'] == 'down']),
                'avg_missed_pct': (
                    sum(abs(m['pct_change']) for m in self.stats['missed_opportunities']) /
                    max(len(self.stats['missed_opportunities']), 1)
                )
            },

            # Strategy accuracy
            'strategy_accuracy': self.stats['strategy_accuracy'],

            # Tuning recommendations
            'tuning_recommendations': []
        }

        # Per-strategy analysis
        for name, data in self.stats['signals_by_strategy'].items():
            total_signals = sum(data[a] for a in ['buy', 'sell', 'short', 'hold'])
            action_signals = data['triggered']
            avg_conf = sum(data['avg_confidence']) / max(len(data['avg_confidence']), 1)

            summary['strategy_signals'][name] = {
                'total_signals': total_signals,
                'action_signals': action_signals,
                'action_rate': action_signals / max(total_signals, 1) * 100,
                'avg_confidence': avg_conf,
                'breakdown': {a: data[a] for a in ['buy', 'sell', 'short', 'hold']}
            }

            # Generate tuning recommendations
            if action_signals == 0 and total_signals > 10:
                summary['tuning_recommendations'].append(
                    f"{name}: NEVER triggered - thresholds too strict"
                )
            elif avg_conf < 0.3 and action_signals > 0:
                summary['tuning_recommendations'].append(
                    f"{name}: Low avg confidence ({avg_conf:.2f}) - review signal strength calc"
                )

        # Execution analysis
        if self.stats['executions']['rejected'] > self.stats['executions']['executed']:
            summary['tuning_recommendations'].append(
                f"High rejection rate ({summary['execution_rate']:.1f}% executed) - lower confidence thresholds"
            )

        # Regime analysis
        if 'chop' in self.stats['regimes'] and self.stats['regimes'].get('chop', 0) > self.stats['total_decisions'] * 0.5:
            summary['tuning_recommendations'].append(
                "Chop regime >50% of time - verify ATR scaling is working"
            )

        # Missed opportunity analysis
        if summary['missed_opportunities']['up_moves'] > 5:
            summary['tuning_recommendations'].append(
                f"Missed {summary['missed_opportunities']['up_moves']} upward moves - lower buy thresholds"
            )

        return summary

    def close(self):
        """Close logger and write summary."""
        summary = self.generate_summary()

        # Write summary to separate file
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Write final log entry
        self._write_log({
            'type': 'session_end',
            'timestamp': datetime.now().isoformat(),
            'summary': summary
        })

        print(f"\nDiagnosticLogger: Summary written to {self.summary_file}")
        print(f"Total decisions: {summary['total_decisions']}")
        print(f"Execution rate: {summary['execution_rate']:.1f}%")
        print(f"Missed opportunities: {summary['missed_opportunities']['count']}")

        if summary['tuning_recommendations']:
            print("\nTuning Recommendations:")
            for rec in summary['tuning_recommendations']:
                print(f"  - {rec}")

        return summary


# Singleton instance for easy access
_logger_instance: Optional[DiagnosticLogger] = None


def get_diagnostic_logger(log_dir: str = "logs") -> DiagnosticLogger:
    """Get or create the diagnostic logger singleton."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DiagnosticLogger(log_dir)
    return _logger_instance


def close_diagnostic_logger():
    """Close the diagnostic logger and write summary."""
    global _logger_instance
    if _logger_instance is not None:
        summary = _logger_instance.close()
        _logger_instance = None
        return summary
    return None
