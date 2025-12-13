"""
XRP/BTC Pair Trading Strategy
Phase 25: Complete Rewrite with Critical Fixes

Statistical arbitrage exploiting cointegration between XRP and BTC.
Uses log prices, proper cointegration enforcement, half-life filtering,
and standard action types for orchestrator compatibility.

Phase 25 Improvements:
- CRITICAL: Maps pair signals to standard actions (buy/sell/short)
- CRITICAL: Enforces cointegration - blocks trades when p-value > threshold
- Uses log prices for cointegration testing (more stationary)
- Adds half-life filter via Ornstein-Uhlenbeck estimation
- Adds ADF test on spread itself (not just Engle-Granger)
- Separates hedge lookback from z-score lookback (avoids look-ahead bias)
- Faster hedge recalculation (6h instead of 24h)
- Regime-based dynamic entry thresholds
- Normalized hedge ratio using log returns
- Transaction cost consideration in sizing

Target: Only trade when statistical edge exists (cointegration + reasonable half-life)
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Pair trading will use fallback methods.")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy


class XRPBTCPairTrading(BaseStrategy):
    """
    Pair Trading Strategy using XRP and BTC cointegration.

    Phase 25 Complete Rewrite:
    - Uses log prices for more reliable cointegration testing
    - Strictly enforces cointegration requirement (no trading without it)
    - Estimates half-life to ensure mean reversion is tradeable
    - Maps pair actions to standard buy/sell for orchestrator compatibility
    - Separates hedge estimation window from z-score window

    Entry:
    - Z-score > entry_z AND cointegration valid AND half-life reasonable: Short XRP
    - Z-score < -entry_z AND cointegration valid AND half-life reasonable: Long XRP

    Exit:
    - |Z-score| < exit_z (mean reverted)
    - |Z-score| > stop_z (divergence stop)
    - Half-life exceeded (trade not working)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.name = 'xrp_btc_pair_trading'
        self.xrp_symbol = config.get('xrp_symbol', 'XRP/USDT')
        self.btc_symbol = config.get('btc_symbol', 'BTC/USDT')

        # Phase 25: Separate lookback windows to avoid look-ahead bias
        self.hedge_lookback = config.get('hedge_lookback', 168)  # 1 week for hedge estimation
        self.zscore_lookback = config.get('zscore_lookback', 72)  # 3 days for z-score calculation

        # Legacy support
        if 'lookback' in config and 'hedge_lookback' not in config:
            self.hedge_lookback = config['lookback']
            self.zscore_lookback = min(config['lookback'] // 2, 72)

        # Entry/exit thresholds (can be overridden by regime)
        self.base_entry_z = config.get('entry_z', 2.0)
        self.exit_z = config.get('exit_z', 0.5)
        self.stop_z = config.get('stop_z', 3.5)

        # Dynamic entry based on regime
        self.entry_z = self.base_entry_z
        self.use_regime_adjustment = config.get('use_regime_adjustment', True)

        self.max_leverage = config.get('max_leverage', 10)
        self.position_size_pct = config.get('position_size_pct', 0.10)

        # Phase 25: Strict cointegration enforcement
        self.coint_pvalue_threshold = config.get('coint_pvalue_threshold', 0.05)  # Strict 5%
        self.spread_adf_threshold = config.get('spread_adf_threshold', 0.10)  # ADF on spread
        self.require_cointegration = config.get('require_cointegration', True)

        # Phase 25: Half-life filtering
        self.max_half_life = config.get('max_half_life', 48)  # Max 48 hours
        self.min_half_life = config.get('min_half_life', 2)   # Min 2 hours (avoid noise)
        self.use_half_life_filter = config.get('use_half_life_filter', True)

        # Phase 25: Faster hedge recalculation
        self.hedge_recalc_period = config.get('hedge_recalc_period', 6)  # 6 hours (was 24)
        self.candles_since_recalc = 0

        # Phase 25: Transaction costs
        self.fee_rate = config.get('fee_rate', 0.001)  # 0.1% per leg
        self.min_expected_profit = config.get('min_expected_profit', 0.01)  # 1% min profit target

        # BTC momentum filter (optional)
        self.use_btc_filter = config.get('use_btc_filter', False)  # Disabled by default now
        self.btc_rsi_window = config.get('btc_rsi_window', 14)

        # State
        self.hedge_ratio = None
        self.intercept = None
        self.spread_mean = None
        self.spread_std = None
        self.current_position = None
        self.entry_zscore = None
        self.entry_price = None
        self.position_bars = 0
        self.last_zscore = 0.0
        self.cointegration_pvalue = None
        self.spread_adf_pvalue = None
        self.half_life = None
        self.btc_rsi = 50.0
        self.is_cointegrated = False
        self.spread_is_stationary = False

        # Regime state
        self.current_regime = 'normal'
        self.regime_volatility = 0.0

    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> float:
        """Calculate RSI from price array."""
        if len(prices) < window + 1:
            return 50.0

        deltas = np.diff(prices[-(window + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _estimate_half_life(self, spread: np.ndarray) -> float:
        """
        Estimate half-life of mean reversion using Ornstein-Uhlenbeck process.

        The spread follows: dS = theta * (mu - S) * dt + sigma * dW
        Half-life = ln(2) / theta

        We estimate theta via AR(1) regression: S(t) - S(t-1) = alpha + beta * S(t-1) + epsilon
        Then theta = -ln(1 + beta)
        """
        if len(spread) < 20:
            return float('inf')

        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)

        # Remove any NaN/inf
        valid = np.isfinite(spread_lag) & np.isfinite(spread_diff)
        spread_lag = spread_lag[valid]
        spread_diff = spread_diff[valid]

        if len(spread_lag) < 10:
            return float('inf')

        try:
            if STATSMODELS_AVAILABLE:
                X = sm.add_constant(spread_lag)
                model = sm.OLS(spread_diff, X).fit()
                beta = model.params[1]
            else:
                # Simple OLS fallback
                mean_lag = np.mean(spread_lag)
                mean_diff = np.mean(spread_diff)
                cov = np.sum((spread_lag - mean_lag) * (spread_diff - mean_diff))
                var = np.sum((spread_lag - mean_lag) ** 2)
                beta = cov / var if var > 0 else 0

            # Beta should be negative for mean reversion
            if beta >= 0:
                return float('inf')  # No mean reversion

            # Theta = -ln(1 + beta), but we need beta in (-1, 0)
            if beta <= -1:
                return float('inf')

            theta = -np.log(1 + beta)
            if theta <= 0:
                return float('inf')

            half_life = np.log(2) / theta
            return half_life

        except Exception:
            return float('inf')

    def _test_spread_stationarity(self, spread: np.ndarray) -> Tuple[bool, float]:
        """
        Test if spread is stationary using ADF test.
        Returns (is_stationary, p_value).
        """
        if not STATSMODELS_AVAILABLE or len(spread) < 20:
            return True, 0.0  # Assume stationary if can't test

        try:
            # ADF test - null hypothesis is non-stationary
            result = adfuller(spread, maxlag=int(len(spread) / 4))
            pvalue = result[1]
            is_stationary = pvalue < self.spread_adf_threshold
            return is_stationary, pvalue
        except Exception:
            return True, 0.0

    def _detect_regime(self, data: Dict[str, pd.DataFrame]) -> str:
        """
        Detect market regime for dynamic threshold adjustment.
        Returns: 'low_vol', 'normal', 'high_vol'
        """
        if self.xrp_symbol not in data:
            return 'normal'

        df = data[self.xrp_symbol]
        if len(df) < 20:
            return 'normal'

        # Calculate ATR as volatility proxy
        high = df['high'].iloc[-15:].values
        low = df['low'].iloc[-15:].values
        close = df['close'].iloc[-15:].values

        # True Range calculation
        tr1 = high[1:] - low[1:]  # High - Low
        tr2 = np.abs(high[1:] - close[:-1])  # High - Previous Close
        tr3 = np.abs(low[1:] - close[:-1])   # Low - Previous Close

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr) / close[-1] if len(tr) > 0 else 0

        self.regime_volatility = atr

        if atr < 0.015:
            self.current_regime = 'low_vol'
        elif atr > 0.04:
            self.current_regime = 'high_vol'
        else:
            self.current_regime = 'normal'

        return self.current_regime

    def _adjust_entry_threshold(self):
        """Adjust entry z-score threshold based on regime."""
        if not self.use_regime_adjustment:
            self.entry_z = self.base_entry_z
            return

        if self.current_regime == 'low_vol':
            # In low vol, z-scores are more meaningful - can use tighter threshold
            self.entry_z = self.base_entry_z * 0.85
        elif self.current_regime == 'high_vol':
            # In high vol, need wider threshold to avoid noise
            self.entry_z = self.base_entry_z * 1.25
        else:
            self.entry_z = self.base_entry_z

    def update_hedge_ratio(self, data: Dict[str, pd.DataFrame], force: bool = False) -> bool:
        """
        Update hedge ratio using log prices for more reliable estimation.
        Phase 25: Uses log prices, separate windows, strict cointegration.
        """
        self.candles_since_recalc += 1
        if not force and self.hedge_ratio is not None and self.candles_since_recalc < self.hedge_recalc_period:
            return True

        if self.xrp_symbol not in data or self.btc_symbol not in data:
            return False

        xrp_df = data[self.xrp_symbol]
        btc_df = data[self.btc_symbol]

        if len(xrp_df) < self.hedge_lookback or len(btc_df) < self.hedge_lookback:
            return False

        # Use log prices for cointegration testing
        xrp_prices = xrp_df['close'].iloc[-self.hedge_lookback:].values
        btc_prices = btc_df['close'].iloc[-self.hedge_lookback:].values

        log_xrp = np.log(xrp_prices)
        log_btc = np.log(btc_prices)

        # Calculate BTC RSI for optional filter
        self.btc_rsi = self._calculate_rsi(btc_prices, self.btc_rsi_window)

        if STATSMODELS_AVAILABLE:
            # OLS on log prices: log(XRP) = alpha + beta * log(BTC) + epsilon
            btc_with_const = sm.add_constant(log_btc)
            model = sm.OLS(log_xrp, btc_with_const).fit()
            self.intercept = model.params[0]
            self.hedge_ratio = model.params[1]  # This is now the log-price beta

            # Test cointegration on log prices
            try:
                _, pvalue, _ = coint(log_xrp, log_btc)
                self.cointegration_pvalue = pvalue
                self.is_cointegrated = pvalue < self.coint_pvalue_threshold
            except Exception:
                self.cointegration_pvalue = 1.0
                self.is_cointegrated = False

            # Calculate spread and test its stationarity
            spread = log_xrp - self.hedge_ratio * log_btc - self.intercept
            self.spread_is_stationary, self.spread_adf_pvalue = self._test_spread_stationarity(spread)

            # Estimate half-life
            self.half_life = self._estimate_half_life(spread)

        else:
            # Fallback: Simple OLS on log prices
            log_btc_mean = np.mean(log_btc)
            log_xrp_mean = np.mean(log_xrp)
            covariance = np.sum((log_btc - log_btc_mean) * (log_xrp - log_xrp_mean))
            variance = np.sum((log_btc - log_btc_mean) ** 2)
            self.hedge_ratio = covariance / variance if variance > 0 else 0
            self.intercept = log_xrp_mean - self.hedge_ratio * log_btc_mean

            # Can't test cointegration without statsmodels - be conservative
            self.is_cointegrated = False
            self.cointegration_pvalue = 1.0
            self.spread_is_stationary = False
            self.spread_adf_pvalue = 1.0
            self.half_life = float('inf')

        self.candles_since_recalc = 0
        return self.hedge_ratio is not None

    def _calculate_spread_zscore(self, data: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
        """
        Calculate current spread and z-score using log prices.
        Uses separate window for z-score to avoid look-ahead bias.
        """
        xrp_df = data[self.xrp_symbol]
        btc_df = data[self.btc_symbol]

        # Current log prices
        log_xrp_current = np.log(xrp_df['close'].iloc[-1])
        log_btc_current = np.log(btc_df['close'].iloc[-1])

        # Current spread
        current_spread = log_xrp_current - self.hedge_ratio * log_btc_current - self.intercept

        # Historical spread for z-score (shorter window to avoid look-ahead bias)
        lookback = min(self.zscore_lookback, len(xrp_df) - 1)
        log_xrp_hist = np.log(xrp_df['close'].iloc[-lookback:].values)
        log_btc_hist = np.log(btc_df['close'].iloc[-lookback:].values)
        historical_spread = log_xrp_hist - self.hedge_ratio * log_btc_hist - self.intercept

        self.spread_mean = np.mean(historical_spread)
        self.spread_std = np.std(historical_spread)

        if self.spread_std == 0 or self.spread_std < 1e-10:
            return current_spread, 0.0

        zscore = (current_spread - self.spread_mean) / self.spread_std

        return current_spread, zscore

    def _check_entry_conditions(self, zscore: float, direction: str) -> Tuple[bool, str]:
        """
        Check all entry conditions with detailed reason.
        Returns (can_enter, reason).
        """
        reasons = []

        # 1. Cointegration check
        if self.require_cointegration and not self.is_cointegrated:
            return False, f"No cointegration (p={self.cointegration_pvalue:.3f} > {self.coint_pvalue_threshold})"

        # 2. Spread stationarity check
        if self.require_cointegration and not self.spread_is_stationary:
            return False, f"Spread not stationary (ADF p={self.spread_adf_pvalue:.3f})"

        # 3. Half-life check
        if self.use_half_life_filter:
            if self.half_life > self.max_half_life:
                return False, f"Half-life too long ({self.half_life:.1f}h > {self.max_half_life}h)"
            if self.half_life < self.min_half_life:
                return False, f"Half-life too short ({self.half_life:.1f}h < {self.min_half_life}h)"

        # 4. BTC RSI filter (optional)
        if self.use_btc_filter:
            if direction == 'short_xrp' and self.btc_rsi <= 50:
                return False, f"BTC RSI filter: {self.btc_rsi:.1f} <= 50 (need BTC strength)"
            if direction == 'long_xrp' and self.btc_rsi >= 50:
                return False, f"BTC RSI filter: {self.btc_rsi:.1f} >= 50 (need BTC weakness)"

        # 5. Transaction cost check
        expected_move = abs(zscore) * self.spread_std  # Expected spread change
        total_fees = 4 * self.fee_rate  # Entry + exit for both legs
        if expected_move < total_fees + self.min_expected_profit:
            return False, f"Insufficient edge after costs ({expected_move:.4f} < {total_fees + self.min_expected_profit:.4f})"

        return True, "All conditions met"

    def _check_btc_filter(self, action: str) -> bool:
        """Check if BTC momentum supports the trade direction."""
        if not self.use_btc_filter:
            return True

        if action == 'short_xrp':
            return self.btc_rsi > 50
        elif action == 'long_xrp':
            return self.btc_rsi < 50

        return True

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate pair trading signals.

        Phase 25: Returns STANDARD actions (buy/sell/short) for orchestrator compatibility.
        The XRP leg is the primary action, BTC hedge is informational.
        """
        # Check both symbols are available
        if self.xrp_symbol not in data or self.btc_symbol not in data:
            return {
                'action': 'hold',
                'symbol': self.xrp_symbol,
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': 'Missing XRP or BTC data'
            }

        xrp_df = data[self.xrp_symbol]
        btc_df = data[self.btc_symbol]

        min_lookback = max(self.hedge_lookback, self.zscore_lookback)
        if len(xrp_df) < min_lookback or len(btc_df) < min_lookback:
            return {
                'action': 'hold',
                'symbol': self.xrp_symbol,
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': f'Insufficient data ({len(xrp_df)} < {min_lookback})'
            }

        # Detect regime and adjust thresholds
        self._detect_regime(data)
        self._adjust_entry_threshold()

        # Update hedge ratio (recalculates if needed)
        if not self.update_hedge_ratio(data):
            return {
                'action': 'hold',
                'symbol': self.xrp_symbol,
                'size': 0.0,
                'leverage': 1,
                'confidence': 0.0,
                'reason': 'Could not calculate hedge ratio'
            }

        # Calculate spread and z-score
        spread, zscore = self._calculate_spread_zscore(data)
        self.last_zscore = zscore

        xrp_price = xrp_df['close'].iloc[-1]
        btc_price = btc_df['close'].iloc[-1]

        # Build indicator dict for logging
        indicators = {
            'hedge_ratio': self.hedge_ratio,
            'intercept': self.intercept,
            'zscore': zscore,
            'spread': spread,
            'spread_mean': self.spread_mean,
            'spread_std': self.spread_std,
            'xrp_price': xrp_price,
            'btc_price': btc_price,
            'btc_rsi': self.btc_rsi,
            'cointegration_pvalue': self.cointegration_pvalue,
            'is_cointegrated': self.is_cointegrated,
            'spread_adf_pvalue': self.spread_adf_pvalue,
            'spread_is_stationary': self.spread_is_stationary,
            'half_life': self.half_life,
            'current_position': self.current_position,
            'regime': self.current_regime,
            'regime_volatility': self.regime_volatility,
            'entry_z_threshold': self.entry_z,
            'position_bars': self.position_bars
        }

        # Default hold signal
        signal = {
            'action': 'hold',
            'symbol': self.xrp_symbol,
            'size': 0.0,
            'leverage': 1,
            'confidence': 0.0,
            'reason': f'Z-score {zscore:.2f} within range [{-self.entry_z:.2f}, {self.entry_z:.2f}]',
            'indicators': indicators
        }

        # STOP LOSS: Close on extreme divergence
        if self.current_position and abs(zscore) > self.stop_z:
            # Map to standard action: close XRP position
            if self.current_position == 'short_xrp':
                action = 'buy'  # Cover short by buying
                reason = f'STOP LOSS: Z-score {zscore:.2f} > {self.stop_z} (cover short XRP)'
            else:  # long_xrp
                action = 'sell'  # Close long by selling
                reason = f'STOP LOSS: Z-score {zscore:.2f} < -{self.stop_z} (sell long XRP)'

            signal = {
                'action': action,
                'symbol': self.xrp_symbol,
                'size': 1.0,  # Close full position
                'leverage': 1,
                'confidence': 0.9,
                'reason': reason,
                'indicators': indicators,
                'exit_type': 'stop_loss',
                'pair_trade': True,
                'btc_hedge_action': 'close'  # Informational
            }
            self.current_position = None
            self.entry_zscore = None
            self.entry_price = None
            self.position_bars = 0
            return signal

        # EXIT: Mean reversion complete or time exit
        if self.current_position:
            self.position_bars += 1

            should_exit_mean_reversion = abs(zscore) < self.exit_z
            should_exit_time = self.half_life and self.position_bars > self.half_life * 3

            if should_exit_mean_reversion or should_exit_time:
                if self.current_position == 'short_xrp':
                    action = 'buy'  # Cover short
                    exit_reason = 'mean reversion' if should_exit_mean_reversion else 'time exit'
                    reason = f'EXIT ({exit_reason}): Z-score {zscore:.2f} (cover short XRP)'
                else:  # long_xrp
                    action = 'sell'  # Close long
                    exit_reason = 'mean reversion' if should_exit_mean_reversion else 'time exit'
                    reason = f'EXIT ({exit_reason}): Z-score {zscore:.2f} (sell long XRP)'

                signal = {
                    'action': action,
                    'symbol': self.xrp_symbol,
                    'size': 1.0,  # Close full position
                    'leverage': 1,
                    'confidence': 0.85 if should_exit_mean_reversion else 0.7,
                    'reason': reason,
                    'indicators': indicators,
                    'exit_type': 'mean_reversion' if should_exit_mean_reversion else 'time_exit',
                    'pair_trade': True,
                    'btc_hedge_action': 'close'
                }
                self.current_position = None
                self.entry_zscore = None
                self.entry_price = None
                self.position_bars = 0
                return signal

            # Still in position - update reason
            signal['reason'] = f'In {self.current_position} position (bars={self.position_bars}, z={zscore:.2f})'
            return signal

        # ENTRY signals (only if no current position)
        if self.current_position is None:

            # Z-score > entry_z: Spread too high, short XRP (expect XRP to fall vs BTC)
            if zscore > self.entry_z:
                can_enter, entry_reason = self._check_entry_conditions(zscore, 'short_xrp')

                if can_enter:
                    confidence = min(0.5 + (zscore - self.entry_z) * 0.1, 0.85)

                    # Account for transaction costs in sizing
                    adjusted_size = self.position_size_pct * (1 - 2 * self.fee_rate)

                    signal = {
                        'action': 'short',  # Standard action: short XRP
                        'symbol': self.xrp_symbol,
                        'size': adjusted_size,
                        'leverage': min(self.max_leverage, 5),  # Conservative leverage
                        'confidence': confidence,
                        'reason': f'ENTRY: Z-score {zscore:.2f} > {self.entry_z:.2f}, half-life={self.half_life:.1f}h (short XRP)',
                        'indicators': indicators,
                        'entry_type': 'pair_trade',
                        'pair_trade': True,
                        'btc_hedge_action': 'long',  # Informational: should long BTC
                        'hedge_ratio': self.hedge_ratio
                    }
                    self.current_position = 'short_xrp'
                    self.entry_zscore = zscore
                    self.entry_price = xrp_price
                    self.position_bars = 0
                else:
                    signal['reason'] = f'Z-score {zscore:.2f} > {self.entry_z:.2f} but blocked: {entry_reason}'

            # Z-score < -entry_z: Spread too low, long XRP (expect XRP to rise vs BTC)
            elif zscore < -self.entry_z:
                can_enter, entry_reason = self._check_entry_conditions(zscore, 'long_xrp')

                if can_enter:
                    confidence = min(0.5 + (abs(zscore) - self.entry_z) * 0.1, 0.85)

                    # Account for transaction costs in sizing
                    adjusted_size = self.position_size_pct * (1 - 2 * self.fee_rate)

                    signal = {
                        'action': 'buy',  # Standard action: buy XRP
                        'symbol': self.xrp_symbol,
                        'size': adjusted_size,
                        'leverage': min(self.max_leverage, 5),
                        'confidence': confidence,
                        'reason': f'ENTRY: Z-score {zscore:.2f} < -{self.entry_z:.2f}, half-life={self.half_life:.1f}h (long XRP)',
                        'indicators': indicators,
                        'entry_type': 'pair_trade',
                        'pair_trade': True,
                        'btc_hedge_action': 'short',  # Informational: should short BTC
                        'hedge_ratio': self.hedge_ratio
                    }
                    self.current_position = 'long_xrp'
                    self.entry_zscore = zscore
                    self.entry_price = xrp_price
                    self.position_bars = 0
                else:
                    signal['reason'] = f'Z-score {zscore:.2f} < -{self.entry_z:.2f} but blocked: {entry_reason}'

        return signal

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Force recalculation of hedge ratio."""
        if data is None:
            return True
        return self.update_hedge_ratio(data, force=True)

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            'xrp_symbol': self.xrp_symbol,
            'btc_symbol': self.btc_symbol,
            'hedge_ratio': self.hedge_ratio,
            'intercept': self.intercept,
            'current_position': self.current_position,
            'entry_zscore': self.entry_zscore,
            'position_bars': self.position_bars,
            'last_zscore': self.last_zscore,
            'btc_rsi': self.btc_rsi,
            'cointegration_pvalue': self.cointegration_pvalue,
            'is_cointegrated': self.is_cointegrated,
            'spread_adf_pvalue': self.spread_adf_pvalue,
            'spread_is_stationary': self.spread_is_stationary,
            'half_life': self.half_life,
            'regime': self.current_regime,
            'regime_volatility': self.regime_volatility,
            'hedge_lookback': self.hedge_lookback,
            'zscore_lookback': self.zscore_lookback,
            'use_btc_filter': self.use_btc_filter,
            'use_half_life_filter': self.use_half_life_filter,
            'use_regime_adjustment': self.use_regime_adjustment,
            'require_cointegration': self.require_cointegration,
            'coint_pvalue_threshold': self.coint_pvalue_threshold,
            'statsmodels_available': STATSMODELS_AVAILABLE,
            'thresholds': {
                'base_entry_z': self.base_entry_z,
                'current_entry_z': self.entry_z,
                'exit_z': self.exit_z,
                'stop_z': self.stop_z,
                'max_half_life': self.max_half_life,
                'min_half_life': self.min_half_life
            }
        })
        return base_status
