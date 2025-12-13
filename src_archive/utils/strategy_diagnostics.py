"""
Phase 32: Strategy Diagnostics System
Provides detailed analysis of why strategies aren't trading.

Features:
- Intercepts strategy signals and adds diagnostic info
- Always captures indicator values, even on HOLD
- Tracks filter blocking reasons
- Provides threshold comparison info
- Logs data availability issues
- Generates actionable tuning recommendations
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class StrategyDiagnostic:
    """Diagnostic info for a single strategy evaluation."""
    strategy_name: str
    timestamp: datetime
    action: str
    confidence: float
    reason: str

    # Why no trade?
    blocked_by: Optional[str] = None  # 'cooldown', 'max_positions', 'low_confidence', 'threshold', 'data'

    # Data availability
    data_status: Dict[str, Any] = field(default_factory=dict)

    # Indicator values (always captured)
    indicators: Dict[str, float] = field(default_factory=dict)

    # Threshold comparisons
    threshold_checks: List[Dict[str, Any]] = field(default_factory=list)

    # Strategy state
    strategy_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy_name,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'confidence': self.confidence,
            'reason': self.reason,
            'blocked_by': self.blocked_by,
            'data_status': self.data_status,
            'indicators': self.indicators,
            'threshold_checks': self.threshold_checks,
            'strategy_state': self.strategy_state
        }


class StrategyDiagnostics:
    """
    Diagnostic wrapper for trading strategies.

    Intercepts signal generation and adds detailed diagnostic info
    to help understand why strategies aren't trading.
    """

    # Common indicator extractors for different strategy types
    INDICATOR_EXTRACTORS = {
        'rsi': ['rsi', 'current_rsi', 'xrp_rsi', 'btc_rsi'],
        'volume': ['volume_surge', 'volume_ratio', 'current_volume_ratio'],
        'atr': ['atr', 'atr_pct', 'current_atr', 'current_atr_pct'],
        'drawdown': ['drawdown', 'max_drawdown'],
        'price': ['current_price', 'last_price', 'entry_price'],
        'trend': ['trend', 'trend_direction', 'ma_trend'],
        'volatility': ['volatility', 'bb_width', 'squeeze_level'],
        'positions': ['position_count', 'open_positions', 'active_positions'],
    }

    # Known thresholds for common strategies
    STRATEGY_THRESHOLDS = {
        'dip_detector': {
            'rsi_oversold': ('rsi', '<', 38, 'RSI must be below {threshold} for oversold'),
            'volume_surge': ('volume_surge', '>', 1.5, 'Volume must be {threshold}x average'),
            'drawdown': ('drawdown', '<', -0.05, 'Drawdown must be below {threshold}%'),
        },
        'mean_reversion_vwap': {
            'vwap_deviation': ('vwap_dev', '<', -0.02, 'Price must be {threshold}% below VWAP'),
            'rsi_oversold': ('rsi', '<', 35, 'RSI must be below {threshold}'),
        },
        'mean_reversion_short': {
            'rsi_overbought': ('rsi', '>', 65, 'RSI must be above {threshold} for overbought'),
            'price_extended': ('price_extension', '>', 0.02, 'Price must be {threshold}% extended'),
        },
        'ma_trend_follow': {
            'trend_strength': ('trend_strength', '>', 0.5, 'Trend strength must exceed {threshold}'),
            'price_above_ma': ('price_vs_ma', '>', 0, 'Price must be above MA'),
        },
        'intraday_scalper': {
            'volatility': ('atr_pct', '>', 0.01, 'ATR must exceed {threshold}%'),
            'momentum': ('momentum', '>', 0, 'Momentum must be positive'),
        },
        'scalping_1m5m': {
            'volatility': ('atr_pct', '>', 0.005, 'ATR must exceed {threshold}%'),
            'spread_ok': ('spread', '<', 0.002, 'Spread must be below {threshold}%'),
        },
        'defensive_yield': {
            'market_stable': ('volatility', '<', 0.03, 'Volatility must be below {threshold}'),
            'position_open': ('has_position', '==', True, 'Must have open position to manage'),
        },
        'grid_arithmetic': {
            'price_in_range': ('price_in_range', '==', True, 'Price must be within grid range'),
            'grid_level_hit': ('grid_level_distance', '<', 0.002, 'Price must be within {threshold}% of grid level'),
        },
        'grid_geometric': {
            'price_in_range': ('price_in_range', '==', True, 'Price must be within grid range'),
        },
        'grid_rsi_reversion': {
            'rsi_zone': ('rsi', 'between', (30, 70), 'RSI should trigger in zones'),
            'price_in_range': ('price_in_range', '==', True, 'Price must be within grid range'),
        },
        'xrp_btc_pair_trading': {
            'spread_deviation': ('spread_zscore', '>', 1.5, 'Spread z-score must exceed {threshold}'),
            'correlation': ('correlation', '>', 0.5, 'Correlation must be above {threshold}'),
        },
        'xrp_btc_leadlag': {
            'btc_momentum': ('btc_momentum', '!=', 0, 'BTC must show momentum'),
            'lag_confirmed': ('lag_bars', '>', 2, 'Lag must be at least {threshold} bars'),
        },
        'enhanced_dca': {
            'dip_detected': ('price_below_avg', '==', True, 'Price must be below DCA average'),
            'interval_passed': ('time_since_last', '>', 3600, 'Must wait {threshold}s between buys'),
        },
        'twap_accumulator': {
            'interval_passed': ('time_since_last', '>', 300, 'Must wait {threshold}s between slices'),
        },
        'funding_rate_arb': {
            'funding_rate': ('funding_rate', '>', 0.0001, 'Funding rate must exceed {threshold}'),
            'spread_ok': ('spot_perp_spread', '<', 0.005, 'Spread must be below {threshold}'),
        },
        'triangular_arb': {
            'profit_threshold': ('arb_profit', '>', 0.001, 'Arbitrage profit must exceed {threshold}'),
        },
        'wavetrend': {
            'wt_oversold': ('wavetrend', '<', -50, 'WaveTrend must be below {threshold}'),
            'wt_cross': ('wt_cross', '==', True, 'WaveTrend must cross'),
        },
        'supertrend': {
            'trend_change': ('trend_changed', '==', True, 'Trend must change'),
        },
        'ichimoku_cloud': {
            'price_vs_cloud': ('price_vs_cloud', '!=', 0, 'Price must be outside cloud'),
            'tk_cross': ('tk_cross', '==', True, 'TK cross must occur'),
        },
        'volume_profile': {
            'volume_node': ('near_volume_node', '==', True, 'Price must be near volume node'),
        },
        'volatility_breakout': {
            'squeeze': ('bb_squeeze', '==', True, 'Bollinger squeeze must be active'),
            'breakout': ('breakout_detected', '==', True, 'Breakout must be detected'),
        },
        'whale_sentiment': {
            'whale_activity': ('whale_score', '>', 0.5, 'Whale score must exceed {threshold}'),
        },
        'multi_indicator_confluence': {
            'confluence_score': ('confluence', '>', 3, 'At least {threshold} indicators must align'),
        },
        'portfolio_rebalancer': {
            'drift': ('portfolio_drift', '>', 0.05, 'Portfolio drift must exceed {threshold}'),
        },
        'ema9_scalper': {
            'volatility': ('atr_pct', '>', 0.01, 'ATR must exceed {threshold}%'),
            'adx_max': ('adx', '<', 25, 'ADX must be below {threshold}'),
            'ema_crossover': ('waiting_for_crossover', '==', False, 'EMA crossover required'),
        },
        'xrp_momentum_lstm': {
            'momentum': ('momentum', '!=', 0, 'Momentum must be non-zero'),
            'dip_detected': ('is_dip', '==', True, 'Dip must be detected for leveraged entry'),
            'rsi_zone': ('rsi', 'between', (25, 75), 'RSI should be in tradeable zone'),
        },
        'scalper': {  # Alias for intraday_scalper
            'volatility': ('atr_pct', '>', 0.01, 'ATR must exceed {threshold}%'),
            'adx_max': ('adx', '<', 25, 'ADX must be below {threshold}'),
        },
        'leadlag': {  # Alias for xrp_btc_leadlag
            'btc_momentum': ('btc_trend', '!=', 'none', 'BTC must show trend'),
            'correlation': ('correlation', '>', 0.75, 'Correlation must exceed {threshold}'),
        },
    }

    def __init__(self, log_dir: str = "logs/diagnostics", enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

        self.session_start = datetime.now()
        self.log_file = self.log_dir / f"strategy_diagnostics_{self.session_start.strftime('%Y%m%d_%H%M%S')}.jsonl"

        # Track diagnostics per strategy
        self.diagnostics: Dict[str, List[StrategyDiagnostic]] = {}
        self.hold_streaks: Dict[str, int] = {}  # Consecutive holds per strategy
        self.last_trade_time: Dict[str, datetime] = {}

        # Summary stats
        self.stats = {
            'total_evaluations': 0,
            'total_signals': 0,
            'total_holds': 0,
            'blocked_reasons': {},
            'strategies_never_traded': set(),
        }

    def diagnose_signal(self,
                        strategy_name: str,
                        signal: Dict[str, Any],
                        data: Dict[str, pd.DataFrame],
                        strategy_instance: Any = None) -> Dict[str, Any]:
        """
        Enhance a strategy signal with diagnostic information.

        Args:
            strategy_name: Name of the strategy
            signal: The signal dict returned by generate_signals()
            data: The data dict passed to generate_signals()
            strategy_instance: Optional strategy object for state extraction

        Returns:
            Enhanced signal dict with diagnostic info
        """
        if not self.enabled:
            return signal

        self.stats['total_evaluations'] += 1

        action = signal.get('action', 'hold')
        confidence = signal.get('confidence', 0.0)
        reason = signal.get('reason', '')

        # Create diagnostic record
        diag = StrategyDiagnostic(
            strategy_name=strategy_name,
            timestamp=datetime.now(),
            action=action,
            confidence=confidence,
            reason=reason
        )

        # Check data availability
        diag.data_status = self._check_data_availability(data, strategy_name)

        # Extract indicators (always, even on hold)
        diag.indicators = self._extract_indicators(signal, data, strategy_instance)

        # Check thresholds
        diag.threshold_checks = self._check_thresholds(strategy_name, diag.indicators)

        # Get strategy state
        if strategy_instance and hasattr(strategy_instance, 'get_status'):
            try:
                status = strategy_instance.get_status()
                diag.strategy_state = self._sanitize_state(status)
            except Exception as e:
                diag.strategy_state = {'error': str(e)}

        # Determine blocking reason
        diag.blocked_by = self._determine_blocking_reason(
            action, confidence, diag.data_status, diag.threshold_checks, strategy_instance, signal
        )

        # Track stats
        if action == 'hold':
            self.stats['total_holds'] += 1
            self.hold_streaks[strategy_name] = self.hold_streaks.get(strategy_name, 0) + 1

            if diag.blocked_by:
                self.stats['blocked_reasons'][diag.blocked_by] = \
                    self.stats['blocked_reasons'].get(diag.blocked_by, 0) + 1
        else:
            self.stats['total_signals'] += 1
            self.hold_streaks[strategy_name] = 0
            self.last_trade_time[strategy_name] = datetime.now()
            self.stats['strategies_never_traded'].discard(strategy_name)

        # Track strategies that never trade
        if strategy_name not in self.last_trade_time:
            self.stats['strategies_never_traded'].add(strategy_name)

        # Store diagnostic
        if strategy_name not in self.diagnostics:
            self.diagnostics[strategy_name] = []
        self.diagnostics[strategy_name].append(diag)

        # Keep only last 100 per strategy
        if len(self.diagnostics[strategy_name]) > 100:
            self.diagnostics[strategy_name] = self.diagnostics[strategy_name][-100:]

        # Log to file
        self._write_diagnostic(diag)

        # Enhance the original signal
        enhanced_signal = signal.copy()
        enhanced_signal['_diagnostic'] = {
            'blocked_by': diag.blocked_by,
            'indicators': diag.indicators,
            'threshold_checks': diag.threshold_checks,
            'data_status': diag.data_status,
            'hold_streak': self.hold_streaks.get(strategy_name, 0),
        }

        return enhanced_signal

    def _check_data_availability(self, data: Dict[str, pd.DataFrame], strategy_name: str) -> Dict[str, Any]:
        """Check data availability and quality."""
        status = {
            'symbols_available': list(data.keys()),
            'symbol_bars': {},
            'issues': []
        }

        for symbol, df in data.items():
            if df is None:
                status['issues'].append(f"{symbol}: None")
                status['symbol_bars'][symbol] = 0
            elif isinstance(df, pd.DataFrame):
                bars = len(df)
                status['symbol_bars'][symbol] = bars

                if df.empty:
                    status['issues'].append(f"{symbol}: Empty DataFrame")
                elif bars < 50:
                    status['issues'].append(f"{symbol}: Only {bars} bars (need 50+)")

                # Check for required columns
                required = ['open', 'high', 'low', 'close', 'volume']
                missing = [c for c in required if c not in df.columns]
                if missing:
                    status['issues'].append(f"{symbol}: Missing columns {missing}")
            else:
                status['issues'].append(f"{symbol}: Not a DataFrame ({type(df).__name__})")

        status['has_issues'] = len(status['issues']) > 0
        return status

    def _extract_indicators(self,
                           signal: Dict[str, Any],
                           data: Dict[str, pd.DataFrame],
                           strategy_instance: Any) -> Dict[str, float]:
        """Extract indicator values from signal, data, and strategy state."""
        indicators = {}

        # First, get indicators from the signal itself
        if 'indicators' in signal and isinstance(signal['indicators'], dict):
            indicators.update(signal['indicators'])

        # Extract from strategy state
        if strategy_instance:
            # Try common indicator attributes
            for category, names in self.INDICATOR_EXTRACTORS.items():
                for name in names:
                    if hasattr(strategy_instance, name):
                        val = getattr(strategy_instance, name)
                        if isinstance(val, (int, float)):
                            indicators[name] = val
                        elif isinstance(val, dict):
                            for k, v in val.items():
                                if isinstance(v, (int, float)):
                                    indicators[f"{name}_{k}"] = v

            # Try get_status()
            if hasattr(strategy_instance, 'get_status'):
                try:
                    status = strategy_instance.get_status()
                    for key, val in status.items():
                        if isinstance(val, (int, float)) and key not in indicators:
                            indicators[key] = val
                except:
                    pass

        # Calculate basic indicators from data if not present
        for symbol, df in data.items():
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue

            prefix = symbol.split('/')[0].lower() if '/' in symbol else symbol.lower()

            try:
                if 'close' in df.columns and len(df) > 0:
                    close = df['close'].values

                    # Current price
                    if f'{prefix}_price' not in indicators:
                        indicators[f'{prefix}_price'] = float(close[-1])

                    # Simple RSI if not present
                    if f'{prefix}_rsi' not in indicators and len(close) > 14:
                        indicators[f'{prefix}_rsi'] = self._quick_rsi(close)

                    # Drawdown
                    if f'{prefix}_drawdown' not in indicators and len(close) > 20:
                        high = max(close[-20:])
                        indicators[f'{prefix}_drawdown'] = (close[-1] - high) / high

                # Volume surge
                if 'volume' in df.columns and len(df) > 20:
                    vol = df['volume'].values
                    avg_vol = np.mean(vol[-20:-1]) if len(vol) > 20 else np.mean(vol[:-1])
                    if avg_vol > 0 and f'{prefix}_volume_surge' not in indicators:
                        indicators[f'{prefix}_volume_surge'] = vol[-1] / avg_vol
            except Exception as e:
                indicators[f'{prefix}_calc_error'] = str(e)

        return indicators

    def _quick_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Quick RSI calculation."""
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _check_thresholds(self, strategy_name: str, indicators: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check if indicator values meet strategy thresholds."""
        checks = []

        thresholds = self.STRATEGY_THRESHOLDS.get(strategy_name, {})

        for check_name, (indicator, op, threshold, desc) in thresholds.items():
            # Find matching indicator
            indicator_val = None
            for key, val in indicators.items():
                if indicator in key.lower():
                    indicator_val = val
                    break

            if indicator_val is None:
                checks.append({
                    'name': check_name,
                    'indicator': indicator,
                    'value': None,
                    'threshold': threshold,
                    'operator': op,
                    'passed': None,
                    'description': f"Indicator '{indicator}' not found"
                })
                continue

            # Evaluate condition
            passed = False
            if op == '<':
                passed = indicator_val < threshold
            elif op == '>':
                passed = indicator_val > threshold
            elif op == '<=':
                passed = indicator_val <= threshold
            elif op == '>=':
                passed = indicator_val >= threshold
            elif op == '==':
                passed = indicator_val == threshold
            elif op == 'between':
                passed = threshold[0] <= indicator_val <= threshold[1]

            checks.append({
                'name': check_name,
                'indicator': indicator,
                'value': indicator_val,
                'threshold': threshold,
                'operator': op,
                'passed': passed,
                'description': desc.format(threshold=threshold)
            })

        return checks

    def _determine_blocking_reason(self,
                                   action: str,
                                   confidence: float,
                                   data_status: Dict[str, Any],
                                   threshold_checks: List[Dict[str, Any]],
                                   strategy_instance: Any,
                                   signal: Dict[str, Any] = None) -> Optional[str]:
        """Determine why a strategy didn't generate a signal."""
        if action != 'hold':
            return None

        # Check data issues first
        if data_status.get('has_issues'):
            return 'data_unavailable'

        # Check if any symbol has insufficient bars
        for symbol, bars in data_status.get('symbol_bars', {}).items():
            if bars < 50:
                return 'insufficient_data'

        # Check cooldown - improved detection
        if strategy_instance:
            # Check for active cooldown
            if hasattr(strategy_instance, 'last_signal_time') and hasattr(strategy_instance, 'cooldown_minutes'):
                from datetime import datetime, timedelta
                last_times = strategy_instance.last_signal_time
                if isinstance(last_times, dict) and last_times:
                    cooldown_mins = getattr(strategy_instance, 'cooldown_minutes', 30)
                    for symbol, last_time in last_times.items():
                        if isinstance(last_time, datetime):
                            if datetime.now() - last_time < timedelta(minutes=cooldown_mins):
                                return 'cooldown'

            # Check max positions
            if hasattr(strategy_instance, 'position_count') and hasattr(strategy_instance, 'max_positions'):
                if strategy_instance.position_count >= strategy_instance.max_positions:
                    return 'max_positions'

            # Check for active positions blocking new entries
            if hasattr(strategy_instance, 'active_positions') and strategy_instance.active_positions:
                return 'has_position'

            if hasattr(strategy_instance, 'entry_prices') and strategy_instance.entry_prices:
                return 'has_position'

        # Parse reason string for clues
        if signal:
            reason = signal.get('reason', '').lower()

            # Parse common reason patterns
            if 'rsi' in reason:
                if 'not oversold' in reason or 'rsi=' in reason:
                    return 'rsi_not_extreme'
            if 'trend' in reason:
                if 'no trend' in reason or 'sideways' in reason or 'neutral' in reason:
                    return 'no_trend'
            if 'volume' in reason:
                if 'low volume' in reason or 'no volume' in reason:
                    return 'low_volume'
            if 'volatility' in reason or 'atr' in reason:
                if 'low' in reason:
                    return 'low_volatility'
            if 'squeeze' in reason:
                return 'no_squeeze'
            if 'cloud' in reason or 'kumo' in reason:
                return 'in_cloud'
            if 'cross' in reason:
                return 'no_cross'
            if 'divergence' in reason:
                return 'no_divergence'
            if 'spread' in reason:
                return 'spread_issue'
            if 'correlation' in reason:
                return 'correlation_issue'
            if 'funding' in reason:
                return 'funding_unfavorable'
            if 'whale' in reason:
                return 'no_whale_activity'
            if 'confluence' in reason:
                return 'low_confluence'
            if 'breakout' in reason:
                return 'no_breakout'
            if 'waiting' in reason or 'accumulating' in reason:
                return 'accumulating'
            if 'rebalance' in reason or 'drift' in reason:
                return 'no_drift'

        # Check threshold failures
        failed_thresholds = [c for c in threshold_checks if c.get('passed') == False]
        if failed_thresholds:
            return f"threshold:{failed_thresholds[0]['name']}"

        # Low confidence but some signal attempted
        if 0 < confidence < 0.25:
            return 'low_confidence'

        # Check indicators for specific issues
        if signal and signal.get('indicators'):
            indicators = signal['indicators']

            # RSI in neutral zone
            for key, val in indicators.items():
                if 'rsi' in key.lower() and isinstance(val, (int, float)):
                    if 40 <= val <= 60:
                        return 'rsi_neutral'
                    elif val > 60:
                        return 'rsi_high'

            # Check for specific indicator states
            if indicators.get('in_cloud') == True:
                return 'in_cloud'
            if indicators.get('squeeze_active') == False:
                return 'no_squeeze'
            if indicators.get('trend_strength', 1) < 0.3:
                return 'weak_trend'

        # Default - no conditions met
        return 'no_setup'

    def _sanitize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize strategy state for logging (remove large/complex objects)."""
        sanitized = {}
        for key, val in state.items():
            if isinstance(val, (int, float, str, bool)):
                sanitized[key] = val
            elif isinstance(val, dict) and len(val) < 10:
                sanitized[key] = val
            elif isinstance(val, list) and len(val) < 10:
                sanitized[key] = val
            elif val is None:
                sanitized[key] = None
        return sanitized

    def _write_diagnostic(self, diag: StrategyDiagnostic):
        """Write diagnostic to log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(diag.to_dict(), default=str) + '\n')
        except Exception as e:
            pass  # Don't let logging errors affect trading

    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Get diagnostic summary for a specific strategy."""
        diags = self.diagnostics.get(strategy_name, [])

        if not diags:
            return {'strategy': strategy_name, 'evaluations': 0}

        holds = [d for d in diags if d.action == 'hold']
        signals = [d for d in diags if d.action != 'hold']

        # Analyze blocking reasons
        blocking_reasons = {}
        for d in holds:
            if d.blocked_by:
                blocking_reasons[d.blocked_by] = blocking_reasons.get(d.blocked_by, 0) + 1

        # Analyze threshold failures
        threshold_failures = {}
        for d in holds:
            for check in d.threshold_checks:
                if check.get('passed') == False:
                    name = check['name']
                    threshold_failures[name] = threshold_failures.get(name, 0) + 1

        # Get latest indicators
        latest_indicators = diags[-1].indicators if diags else {}

        return {
            'strategy': strategy_name,
            'evaluations': len(diags),
            'signals': len(signals),
            'holds': len(holds),
            'hold_rate': len(holds) / len(diags) * 100 if diags else 0,
            'current_hold_streak': self.hold_streaks.get(strategy_name, 0),
            'blocking_reasons': blocking_reasons,
            'threshold_failures': threshold_failures,
            'latest_indicators': latest_indicators,
            'last_signal_time': self.last_trade_time.get(strategy_name),
            'recommendations': self._get_recommendations(strategy_name, blocking_reasons, threshold_failures)
        }

    def _get_recommendations(self,
                            strategy_name: str,
                            blocking_reasons: Dict[str, int],
                            threshold_failures: Dict[str, int]) -> List[str]:
        """Generate tuning recommendations based on diagnostic data."""
        recs = []

        # Check for data issues
        if blocking_reasons.get('data_unavailable', 0) > 5:
            recs.append("Check data feed - frequent data availability issues")

        if blocking_reasons.get('insufficient_data', 0) > 5:
            recs.append("Increase data lookback - not enough historical bars")

        # Check for threshold issues
        for threshold, count in threshold_failures.items():
            if count > 10:
                recs.append(f"Consider relaxing '{threshold}' - failed {count} times")

        # Check for no setup
        if blocking_reasons.get('no_setup', 0) > 20:
            recs.append("Market conditions may not suit this strategy - review parameters")

        # Check for cooldown blocking
        if blocking_reasons.get('cooldown', 0) > 5:
            recs.append("Consider reducing cooldown period")

        # Check for max positions
        if blocking_reasons.get('max_positions', 0) > 5:
            recs.append("Strategy hitting max positions limit - consider increasing or adding exit logic")

        return recs

    def get_all_summaries(self) -> Dict[str, Any]:
        """Get diagnostic summaries for all strategies."""
        summaries = {}
        for strategy_name in self.diagnostics.keys():
            summaries[strategy_name] = self.get_strategy_summary(strategy_name)

        return {
            'session_start': self.session_start.isoformat(),
            'total_evaluations': self.stats['total_evaluations'],
            'total_signals': self.stats['total_signals'],
            'total_holds': self.stats['total_holds'],
            'signal_rate': self.stats['total_signals'] / max(self.stats['total_evaluations'], 1) * 100,
            'strategies_never_traded': list(self.stats['strategies_never_traded']),
            'blocked_reasons_summary': self.stats['blocked_reasons'],
            'strategy_summaries': summaries
        }

    def print_diagnostic_report(self):
        """Print a human-readable diagnostic report."""
        summaries = self.get_all_summaries()

        print("\n" + "=" * 70)
        print("STRATEGY DIAGNOSTICS REPORT")
        print("=" * 70)
        print(f"Session: {summaries['session_start']}")
        print(f"Total evaluations: {summaries['total_evaluations']}")
        print(f"Signal rate: {summaries['signal_rate']:.1f}%")
        print()

        # Strategies that never traded
        never_traded = summaries['strategies_never_traded']
        if never_traded:
            print("STRATEGIES THAT NEVER TRADED:")
            for s in never_traded:
                summary = summaries['strategy_summaries'].get(s, {})
                print(f"  - {s}")
                if summary.get('blocking_reasons'):
                    top_reason = max(summary['blocking_reasons'].items(), key=lambda x: x[1])
                    print(f"    Primary blocker: {top_reason[0]} ({top_reason[1]} times)")
                if summary.get('recommendations'):
                    print(f"    Recommendation: {summary['recommendations'][0]}")
            print()

        # Per-strategy details
        print("PER-STRATEGY DIAGNOSTICS:")
        print("-" * 70)

        for name, summary in summaries['strategy_summaries'].items():
            status = "ACTIVE" if summary['signals'] > 0 else "INACTIVE"
            print(f"\n[{status}] {name}")
            print(f"  Evaluations: {summary['evaluations']} | Signals: {summary['signals']} | Hold rate: {summary['hold_rate']:.0f}%")

            if summary['current_hold_streak'] > 10:
                print(f"  WARNING: {summary['current_hold_streak']} consecutive holds")

            if summary['blocking_reasons']:
                print(f"  Blocking reasons: {summary['blocking_reasons']}")

            if summary['threshold_failures']:
                print(f"  Failed thresholds: {summary['threshold_failures']}")

            if summary.get('latest_indicators'):
                # Show key indicators
                key_indicators = {k: v for k, v in summary['latest_indicators'].items()
                                 if any(x in k.lower() for x in ['rsi', 'price', 'atr', 'volume'])}
                if key_indicators:
                    formatted = ', '.join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                         for k, v in list(key_indicators.items())[:5])
                    print(f"  Latest indicators: {formatted}")

            if summary['recommendations']:
                print(f"  Recommendations:")
                for rec in summary['recommendations'][:2]:
                    print(f"    - {rec}")

        print("\n" + "=" * 70)

    def close(self) -> Dict[str, Any]:
        """Close diagnostics and write final summary."""
        summary = self.get_all_summaries()

        # Write summary file
        summary_file = self.log_dir / f"diagnostics_summary_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nDiagnostics written to: {self.log_file}")
        print(f"Summary written to: {summary_file}")

        return summary


# Singleton instance
_diagnostics_instance: Optional[StrategyDiagnostics] = None


def get_strategy_diagnostics(log_dir: str = "logs/diagnostics", enabled: bool = True) -> StrategyDiagnostics:
    """Get or create the strategy diagnostics singleton."""
    global _diagnostics_instance
    if _diagnostics_instance is None:
        _diagnostics_instance = StrategyDiagnostics(log_dir, enabled)
    return _diagnostics_instance


def close_strategy_diagnostics() -> Optional[Dict[str, Any]]:
    """Close and get final summary."""
    global _diagnostics_instance
    if _diagnostics_instance is not None:
        summary = _diagnostics_instance.close()
        _diagnostics_instance = None
        return summary
    return None
