"""
Phase 26: XRP/BTC Lead-Lag Strategy - Complete Rewrite with Advanced Features

Correlation-aware trading with dynamic lag detection, cointegration validation,
regime-aware thresholds, trailing stops, and multi-timeframe confirmation.

Key Improvements (Phase 26):
1. Dynamic lag detection via cross-correlation (finds optimal BTC lead time)
2. Cointegration gate (statistical validation before trading)
3. Lead-Lag Ratio (LLR) metric (confirms BTC leads XRP)
4. Regime-aware thresholds (adjust for volatility)
5. Trailing stops (lock in profits)
6. Volume-adjusted confidence
7. Position state sync via on_order_filled()
8. Multi-timeframe confirmation
9. Correlation decay warning
10. Market session awareness
11. Seesaw effect (inverse lead-lag) option

Rules:
- High correlation (>0.75) + BTC leads (LLR > 1.2): Trade XRP in BTC direction
- Cointegration required (p-value < 0.10) for statistical validity
- Dynamic lag period (1-15 bars) determined by cross-correlation
- Trailing stops replace fixed take-profit for trend-following
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Optional statsmodels for cointegration testing
try:
    from statsmodels.tsa.stattools import coint, adfuller
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Cointegration tests will use fallback.")


class XRPBTCLeadLag(BaseStrategy):
    """
    Advanced Correlation-aware XRP/BTC lead-lag strategy.

    Phase 26 Complete Rewrite:
    - Dynamic lag detection via cross-correlation
    - Statistical validation via cointegration testing
    - Lead-Lag Ratio (LLR) to confirm BTC leads
    - Regime-aware thresholds
    - Trailing stops
    - Multi-timeframe confirmation
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Core correlation parameters (updated defaults)
        self.corr_high = config.get('corr_high', 0.75)  # Lowered from 0.8 for 2025 market
        self.corr_low = config.get('corr_low', 0.55)    # Lowered from 0.6
        self.lookback = config.get('lookback', 100)     # Increased from 50 for robust correlation

        # Dynamic lag detection parameters
        self.use_dynamic_lag = config.get('use_dynamic_lag', True)
        self.max_lag_search = config.get('max_lag_search', 15)  # Search up to 15 bars lag
        self.min_lag_correlation = config.get('min_lag_correlation', 0.5)  # Min correlation at optimal lag
        self.btc_lead_bars = config.get('btc_lead_bars', 3)  # Fallback if dynamic disabled
        self.optimal_lag = self.btc_lead_bars  # Will be updated dynamically
        self.optimal_lag_correlation = 0.0

        # Cointegration parameters (new)
        self.use_cointegration = config.get('use_cointegration', True)
        self.coint_pvalue_threshold = config.get('coint_pvalue_threshold', 0.10)  # Lenient for lead-lag
        self.cointegration_pvalue = None
        self.is_cointegrated = False

        # Lead-Lag Ratio (LLR) parameters (new)
        self.use_llr = config.get('use_llr', True)
        self.min_llr = config.get('min_llr', 1.2)  # BTC must lead by at least 20%
        self.llr_lookback = config.get('llr_lookback', 10)  # Lags to consider for LLR
        self.current_llr = 1.0

        # Regime-aware thresholds (new)
        self.use_regime_adjustment = config.get('use_regime_adjustment', True)
        self.base_min_btc_move = config.get('min_btc_move', 0.005)  # 0.5% base threshold
        self.min_btc_move = self.base_min_btc_move  # Will be adjusted by regime
        self.current_regime = 'normal'
        self.regime_volatility = 0.0

        # BTC breakout detection params
        self.btc_high_lookback = config.get('btc_high_lookback', 24)
        self.breakout_vol_mult = config.get('breakout_vol_mult', 1.5)
        self.breakout_leverage = config.get('breakout_leverage', 7)

        # Volume-adjusted confidence (new)
        self.use_volume_confirmation = config.get('use_volume_confirmation', True)
        self.volume_lookback = config.get('volume_lookback', 20)
        self.volume_confidence_mult = 1.0  # Updated each signal

        # Multi-timeframe confirmation (new)
        self.use_mtf_confirmation = config.get('use_mtf_confirmation', True)
        self.mtf_agreement = False

        # Market session awareness (new)
        self.use_session_filter = config.get('use_session_filter', False)  # Disabled by default
        self.preferred_sessions = config.get('preferred_sessions', ['us', 'europe'])
        self.current_session = 'unknown'

        # Correlation trend tracking (new)
        self.correlation_trend = 'stable'  # 'strengthening', 'weakening', 'stable'
        self.previous_correlations = []  # Rolling window of correlations
        self.corr_trend_lookback = config.get('corr_trend_lookback', 10)

        # Seesaw effect / inverse lead-lag (new)
        self.allow_inverse_leadlag = config.get('allow_inverse_leadlag', False)
        self.inverse_llr_threshold = config.get('inverse_llr_threshold', 0.7)  # LLR < 0.7 = XRP leads

        # Track state
        self.last_correlation = 0.0
        self.btc_trend = 'none'
        self.position_taken = False
        self.last_btc_breakout = None

        # Position tracking for exit logic
        self.current_position = None  # 'long', 'short', or None
        self.entry_price = 0.0
        self.entry_correlation = 0.0
        self.position_bars = 0

        # Trailing stop parameters (new - replaces fixed take profit)
        self.use_trailing_stop = config.get('use_trailing_stop', True)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.012)  # 1.2% trailing
        self.trailing_stop_activation = config.get('trailing_stop_activation', 0.008)  # Activate after 0.8% profit
        self.trailing_stop_price = 0.0
        self.trailing_stop_active = False
        self.peak_price = 0.0  # Track highest/lowest price since entry

        # Fixed exit parameters (used when trailing disabled or as fallbacks)
        self.take_profit_pct = config.get('take_profit_pct', 0.025)  # 2.5% take profit (increased)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.015)  # 1.5% stop loss
        self.max_hold_bars = config.get('max_hold_bars', 18)  # Increased from 12
        self.corr_breakdown_threshold = config.get('corr_breakdown', 0.15)

    # ============================================================
    # DYNAMIC LAG DETECTION
    # ============================================================

    def _find_optimal_lag(self, btc_df: pd.DataFrame, xrp_df: pd.DataFrame) -> Tuple[int, float]:
        """
        Find optimal lag period using cross-correlation.

        Returns:
            Tuple of (optimal_lag, correlation_at_optimal_lag)
        """
        if len(btc_df) < self.lookback or len(xrp_df) < self.lookback:
            return self.btc_lead_bars, 0.0

        btc_returns = self._calculate_returns(btc_df).dropna().tail(self.lookback)
        xrp_returns = self._calculate_returns(xrp_df).dropna().tail(self.lookback)

        if len(btc_returns) < 20 or len(xrp_returns) < 20:
            return self.btc_lead_bars, 0.0

        best_corr = -1.0
        best_lag = 1

        for lag in range(1, min(self.max_lag_search + 1, len(xrp_returns) - 10)):
            # Shift XRP returns forward (BTC at t predicts XRP at t+lag)
            shifted_xrp = xrp_returns.shift(-lag).dropna()
            aligned_btc = btc_returns.iloc[:len(shifted_xrp)]

            if len(aligned_btc) < 10:
                continue

            try:
                corr = aligned_btc.corr(shifted_xrp)
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            except:
                continue

        self.optimal_lag = best_lag
        self.optimal_lag_correlation = best_corr if best_corr > 0 else 0.0

        return best_lag, self.optimal_lag_correlation

    # ============================================================
    # LEAD-LAG RATIO (LLR) CALCULATION
    # ============================================================

    def _calculate_llr(self, btc_df: pd.DataFrame, xrp_df: pd.DataFrame) -> float:
        """
        Calculate Lead-Lag Ratio (LLR).
        LLR > 1 means BTC leads XRP, LLR < 1 means XRP leads BTC.
        """
        if len(btc_df) < self.lookback or len(xrp_df) < self.lookback:
            return 1.0

        btc_returns = self._calculate_returns(btc_df).dropna().tail(self.lookback)
        xrp_returns = self._calculate_returns(xrp_df).dropna().tail(self.lookback)

        if len(btc_returns) < 20 or len(xrp_returns) < 20:
            return 1.0

        positive_lags_sum = 0.0  # BTC leads XRP
        negative_lags_sum = 0.0  # XRP leads BTC

        for lag in range(1, self.llr_lookback + 1):
            # Positive lag: BTC at t correlates with XRP at t+lag (BTC leads)
            try:
                shifted_xrp = xrp_returns.shift(-lag).dropna()
                aligned_btc = btc_returns.iloc[:len(shifted_xrp)]
                if len(aligned_btc) >= 10:
                    corr_pos = aligned_btc.corr(shifted_xrp)
                    if not np.isnan(corr_pos):
                        positive_lags_sum += abs(corr_pos)
            except:
                pass

            # Negative lag: XRP at t correlates with BTC at t+lag (XRP leads)
            try:
                shifted_btc = btc_returns.shift(-lag).dropna()
                aligned_xrp = xrp_returns.iloc[:len(shifted_btc)]
                if len(aligned_xrp) >= 10:
                    corr_neg = aligned_xrp.corr(shifted_btc)
                    if not np.isnan(corr_neg):
                        negative_lags_sum += abs(corr_neg)
            except:
                pass

        if negative_lags_sum == 0:
            self.current_llr = float('inf') if positive_lags_sum > 0 else 1.0
        else:
            self.current_llr = positive_lags_sum / negative_lags_sum

        return self.current_llr

    # ============================================================
    # COINTEGRATION TESTING
    # ============================================================

    def _test_cointegration(self, btc_df: pd.DataFrame, xrp_df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Test for cointegration between BTC and XRP prices.
        Returns (is_cointegrated, p_value).
        """
        if not STATSMODELS_AVAILABLE:
            # Fallback: use correlation as proxy (less rigorous)
            corr = self._get_correlation(btc_df, xrp_df)
            return corr > 0.7, 0.05 if corr > 0.7 else 0.5

        if len(btc_df) < self.lookback or len(xrp_df) < self.lookback:
            return False, 1.0

        try:
            btc_prices = btc_df['close'].iloc[-self.lookback:].values
            xrp_prices = xrp_df['close'].iloc[-self.lookback:].values

            # Use log prices for better stationarity
            log_btc = np.log(btc_prices)
            log_xrp = np.log(xrp_prices)

            _, pvalue, _ = coint(log_xrp, log_btc)

            self.cointegration_pvalue = pvalue
            self.is_cointegrated = pvalue < self.coint_pvalue_threshold

            return self.is_cointegrated, pvalue

        except Exception as e:
            self.cointegration_pvalue = 1.0
            self.is_cointegrated = False
            return False, 1.0

    # ============================================================
    # REGIME DETECTION AND THRESHOLD ADJUSTMENT
    # ============================================================

    def _detect_regime(self, btc_df: pd.DataFrame) -> str:
        """
        Detect market regime based on volatility.
        Returns: 'low_vol', 'normal', 'high_vol'
        """
        if len(btc_df) < 15:
            return 'normal'

        # Calculate ATR as volatility proxy
        high = btc_df['high'].iloc[-15:].values
        low = btc_df['low'].iloc[-15:].values
        close = btc_df['close'].iloc[-15:].values

        # True Range calculation
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])

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

    def _adjust_thresholds_for_regime(self):
        """Adjust min_btc_move threshold based on current regime."""
        if not self.use_regime_adjustment:
            self.min_btc_move = self.base_min_btc_move
            return

        if self.current_regime == 'low_vol':
            # In low vol, smaller moves are meaningful
            self.min_btc_move = self.base_min_btc_move * 0.7
        elif self.current_regime == 'high_vol':
            # In high vol, need larger moves to filter noise
            self.min_btc_move = self.base_min_btc_move * 1.5
        else:
            self.min_btc_move = self.base_min_btc_move

    # ============================================================
    # VOLUME ANALYSIS
    # ============================================================

    def _calculate_volume_confidence(self, btc_df: pd.DataFrame) -> float:
        """
        Calculate confidence multiplier based on volume.
        Returns multiplier: 0.7 (low vol) to 1.3 (high vol confirmation).
        """
        if len(btc_df) < self.volume_lookback + 1:
            return 1.0

        current_vol = btc_df['volume'].iloc[-1]
        avg_vol = btc_df['volume'].iloc[-self.volume_lookback:-1].mean()

        if avg_vol == 0:
            return 1.0

        vol_ratio = current_vol / avg_vol

        if vol_ratio > 2.0:
            self.volume_confidence_mult = 1.3  # Strong volume confirmation
        elif vol_ratio > 1.5:
            self.volume_confidence_mult = 1.15
        elif vol_ratio < 0.5:
            self.volume_confidence_mult = 0.7  # Low volume = less confidence
        elif vol_ratio < 0.75:
            self.volume_confidence_mult = 0.85
        else:
            self.volume_confidence_mult = 1.0

        return self.volume_confidence_mult

    def _get_volume_percentile(self, btc_df: pd.DataFrame) -> float:
        """Get current volume as percentile of recent volume."""
        if len(btc_df) < self.volume_lookback + 1:
            return 50.0

        current_vol = btc_df['volume'].iloc[-1]
        recent_vols = btc_df['volume'].iloc[-self.volume_lookback:-1].values

        percentile = (np.sum(recent_vols < current_vol) / len(recent_vols)) * 100
        return percentile

    # ============================================================
    # MULTI-TIMEFRAME CONFIRMATION
    # ============================================================

    def _check_mtf_confirmation(self, data: Dict[str, pd.DataFrame], trend: str) -> bool:
        """
        Check if multiple timeframes agree on the trend direction.
        """
        if not self.use_mtf_confirmation:
            return True

        # Look for 15m BTC data
        btc_15m_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '15m' in key.lower():
                btc_15m_key = key
                break

        if not btc_15m_key or btc_15m_key not in data:
            return True  # No MTF data available, allow trade

        btc_15m = data[btc_15m_key]
        if len(btc_15m) < 10:
            return True

        # Check 15m trend
        mtf_trend = self._detect_btc_trend_from_df(btc_15m, bars=5)

        self.mtf_agreement = (mtf_trend == trend) or (mtf_trend == 'none')
        return self.mtf_agreement

    def _detect_btc_trend_from_df(self, df: pd.DataFrame, bars: int = 3) -> str:
        """Detect trend from any dataframe."""
        if len(df) < bars + 1:
            return 'none'

        recent = df.tail(bars + 1)
        start_price = recent['close'].iloc[0]
        end_price = recent['close'].iloc[-1]
        pct_change = (end_price - start_price) / start_price

        if pct_change > self.min_btc_move:
            return 'up'
        elif pct_change < -self.min_btc_move:
            return 'down'
        return 'none'

    # ============================================================
    # CORRELATION TREND TRACKING
    # ============================================================

    def _update_correlation_trend(self, current_corr: float):
        """
        Track correlation trend (strengthening, weakening, stable).
        """
        self.previous_correlations.append(current_corr)

        # Keep only recent correlations
        if len(self.previous_correlations) > self.corr_trend_lookback:
            self.previous_correlations = self.previous_correlations[-self.corr_trend_lookback:]

        if len(self.previous_correlations) < 3:
            self.correlation_trend = 'stable'
            return

        recent_avg = np.mean(self.previous_correlations[-3:])
        older_avg = np.mean(self.previous_correlations[:-3]) if len(self.previous_correlations) > 3 else recent_avg

        diff = recent_avg - older_avg

        if diff > 0.05:
            self.correlation_trend = 'strengthening'
        elif diff < -0.05:
            self.correlation_trend = 'weakening'
        else:
            self.correlation_trend = 'stable'

    # ============================================================
    # MARKET SESSION DETECTION
    # ============================================================

    def _get_market_session(self) -> str:
        """Determine current market session based on UTC time."""
        hour = datetime.utcnow().hour

        if 13 <= hour <= 21:  # US market hours (UTC)
            self.current_session = 'us'
        elif 7 <= hour <= 16:  # Europe hours (UTC)
            self.current_session = 'europe'
        elif 0 <= hour <= 9:  # Asia hours (UTC)
            self.current_session = 'asia'
        else:
            self.current_session = 'transition'

        return self.current_session

    def _session_filter_allows_trade(self) -> bool:
        """Check if current session is preferred for trading."""
        if not self.use_session_filter:
            return True

        self._get_market_session()
        return self.current_session in self.preferred_sessions

    # ============================================================
    # CORE CALCULATION METHODS
    # ============================================================

    def _calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate log returns."""
        return np.log(df['close'] / df['close'].shift(1))

    def _get_correlation(self, btc_df: pd.DataFrame, xrp_df: pd.DataFrame) -> float:
        """Calculate rolling correlation between BTC and XRP returns."""
        if len(btc_df) < self.lookback or len(xrp_df) < self.lookback:
            return 0.5

        btc_returns = self._calculate_returns(btc_df).tail(self.lookback)
        xrp_returns = self._calculate_returns(xrp_df).tail(self.lookback)

        if len(btc_returns) != len(xrp_returns):
            min_len = min(len(btc_returns), len(xrp_returns))
            btc_returns = btc_returns.tail(min_len)
            xrp_returns = xrp_returns.tail(min_len)

        corr = btc_returns.corr(xrp_returns)
        return corr if not np.isnan(corr) else 0.5

    def _detect_btc_trend(self, btc_df: pd.DataFrame) -> str:
        """Detect recent BTC trend using optimal lag period."""
        lag_to_use = self.optimal_lag if self.use_dynamic_lag else self.btc_lead_bars

        if len(btc_df) < lag_to_use + 1:
            return 'none'

        recent = btc_df.tail(lag_to_use + 1)
        start_price = recent['close'].iloc[0]
        end_price = recent['close'].iloc[-1]
        pct_change = (end_price - start_price) / start_price

        if pct_change > self.min_btc_move:
            return 'up'
        elif pct_change < -self.min_btc_move:
            return 'down'
        return 'none'

    def _check_divergence(self, btc_df: pd.DataFrame, xrp_df: pd.DataFrame) -> Optional[str]:
        """Check for BTC/XRP divergence when correlation is low."""
        if len(btc_df) < 10 or len(xrp_df) < 10:
            return None

        btc_change = (btc_df['close'].iloc[-1] - btc_df['close'].iloc[-10]) / btc_df['close'].iloc[-10]
        xrp_change = (xrp_df['close'].iloc[-1] - xrp_df['close'].iloc[-10]) / xrp_df['close'].iloc[-10]

        # Adjust divergence threshold based on regime
        div_threshold = 0.02 if self.current_regime == 'normal' else (0.015 if self.current_regime == 'low_vol' else 0.03)

        if btc_change > div_threshold and xrp_change < -div_threshold:
            return 'long_xrp'
        if btc_change < -div_threshold and xrp_change > div_threshold:
            return 'short_xrp'

        return None

    # ============================================================
    # TRAILING STOP MANAGEMENT
    # ============================================================

    def _update_trailing_stop(self, current_price: float):
        """Update trailing stop based on current price movement."""
        if self.current_position is None:
            return

        if self.current_position == 'long':
            # Track peak price
            if current_price > self.peak_price:
                self.peak_price = current_price

            # Calculate profit from entry
            profit_pct = (current_price - self.entry_price) / self.entry_price

            # Activate trailing stop after minimum profit
            if profit_pct >= self.trailing_stop_activation and not self.trailing_stop_active:
                self.trailing_stop_active = True
                self.trailing_stop_price = current_price * (1 - self.trailing_stop_pct)

            # Update trailing stop if active
            if self.trailing_stop_active:
                new_stop = self.peak_price * (1 - self.trailing_stop_pct)
                self.trailing_stop_price = max(self.trailing_stop_price, new_stop)

        elif self.current_position == 'short':
            # Track trough price (lowest)
            if self.peak_price == 0 or current_price < self.peak_price:
                self.peak_price = current_price

            # Calculate profit from entry (inverted for short)
            profit_pct = (self.entry_price - current_price) / self.entry_price

            if profit_pct >= self.trailing_stop_activation and not self.trailing_stop_active:
                self.trailing_stop_active = True
                self.trailing_stop_price = current_price * (1 + self.trailing_stop_pct)

            if self.trailing_stop_active:
                new_stop = self.peak_price * (1 + self.trailing_stop_pct)
                self.trailing_stop_price = min(self.trailing_stop_price, new_stop) if self.trailing_stop_price > 0 else new_stop

    def _check_trailing_stop_hit(self, current_price: float) -> bool:
        """Check if trailing stop has been hit."""
        if not self.trailing_stop_active or self.trailing_stop_price == 0:
            return False

        if self.current_position == 'long':
            return current_price <= self.trailing_stop_price
        elif self.current_position == 'short':
            return current_price >= self.trailing_stop_price

        return False

    # ============================================================
    # EXIT CONDITIONS
    # ============================================================

    def _check_exit_conditions(self, xrp_df: pd.DataFrame, corr: float) -> Optional[Dict[str, Any]]:
        """Check if current position should be exited."""
        if self.current_position is None:
            return None

        current_price = xrp_df['close'].iloc[-1]
        self.position_bars += 1

        # Update trailing stop
        if self.use_trailing_stop:
            self._update_trailing_stop(current_price)

        # Calculate unrealized P&L
        if self.current_position == 'long':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        xrp_key = 'XRP/USDT'

        # Check trailing stop hit first
        if self.use_trailing_stop and self._check_trailing_stop_hit(current_price):
            action = 'sell' if self.current_position == 'long' else 'cover'
            reason = f'TRAILING STOP: {pnl_pct*100:.2f}% gain locked (trail={self.trailing_stop_price:.4f}, now={current_price:.4f})'
            self._reset_position_state()
            return {
                'action': action,
                'symbol': xrp_key,
                'size': 1.0,
                'confidence': 0.9,
                'reason': reason,
                'strategy': 'leadlag',
                'exit_type': 'trailing_stop'
            }

        # Take profit (only if trailing stop disabled)
        if not self.use_trailing_stop and pnl_pct >= self.take_profit_pct:
            action = 'sell' if self.current_position == 'long' else 'cover'
            reason = f'TAKE PROFIT: {pnl_pct*100:.2f}% gain (entry={self.entry_price:.4f}, now={current_price:.4f})'
            self._reset_position_state()
            return {
                'action': action,
                'symbol': xrp_key,
                'size': 1.0,
                'confidence': 0.9,
                'reason': reason,
                'strategy': 'leadlag',
                'exit_type': 'take_profit'
            }

        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            action = 'sell' if self.current_position == 'long' else 'cover'
            reason = f'STOP LOSS: {pnl_pct*100:.2f}% loss (entry={self.entry_price:.4f}, now={current_price:.4f})'
            self._reset_position_state()
            return {
                'action': action,
                'symbol': xrp_key,
                'size': 1.0,
                'confidence': 0.85,
                'reason': reason,
                'strategy': 'leadlag',
                'exit_type': 'stop_loss'
            }

        # Max hold time
        if self.position_bars >= self.max_hold_bars:
            action = 'sell' if self.current_position == 'long' else 'cover'
            reason = f'TIME EXIT: held {self.position_bars} bars, P&L={pnl_pct*100:.2f}%'
            self._reset_position_state()
            return {
                'action': action,
                'symbol': xrp_key,
                'size': 1.0,
                'confidence': 0.7,
                'reason': reason,
                'strategy': 'leadlag',
                'exit_type': 'time_exit'
            }

        # Correlation breakdown
        corr_drop = self.entry_correlation - corr
        if corr_drop > self.corr_breakdown_threshold:
            action = 'sell' if self.current_position == 'long' else 'cover'
            reason = f'CORR BREAKDOWN: corr dropped {corr_drop:.2f} (entry={self.entry_correlation:.2f}, now={corr:.2f})'
            self._reset_position_state()
            return {
                'action': action,
                'symbol': xrp_key,
                'size': 1.0,
                'confidence': 0.75,
                'reason': reason,
                'strategy': 'leadlag',
                'exit_type': 'corr_breakdown'
            }

        return None

    def _reset_position_state(self):
        """Reset all position-related state variables."""
        self.current_position = None
        self.entry_price = 0.0
        self.entry_correlation = 0.0
        self.position_bars = 0
        self.trailing_stop_price = 0.0
        self.trailing_stop_active = False
        self.peak_price = 0.0

    def _record_entry(self, position_type: str, price: float, corr: float):
        """Record a new position entry."""
        self.current_position = position_type
        self.entry_price = price
        self.entry_correlation = corr
        self.position_bars = 0
        self.trailing_stop_price = 0.0
        self.trailing_stop_active = False
        self.peak_price = price

    # ============================================================
    # BREAKOUT DETECTION
    # ============================================================

    def _detect_btc_breakout(self, btc_df: pd.DataFrame) -> dict:
        """Detect BTC breakout for immediate XRP long."""
        if len(btc_df) < self.btc_high_lookback + 1:
            return {'is_breakout': False, 'strength': 0.0, 'leverage': 5}

        current_price = btc_df['close'].iloc[-1]
        current_vol = btc_df['volume'].iloc[-1]

        recent_high = btc_df['high'].iloc[-self.btc_high_lookback:-1].max()
        avg_volume = btc_df['volume'].iloc[-self.btc_high_lookback:-1].mean()

        # Use volume percentile instead of raw average for robustness
        vol_percentile = self._get_volume_percentile(btc_df)

        price_break = current_price > recent_high
        vol_ratio = current_vol / avg_volume if avg_volume > 0 else 1.0
        volume_spike = vol_percentile > 80  # Top 20% volume

        if price_break and volume_spike:
            strength = vol_ratio * (current_price / recent_high - 1) * 100
            leverage = self.breakout_leverage
            self.last_btc_breakout = {'price': current_price, 'vol_ratio': vol_ratio, 'percentile': vol_percentile}
            return {'is_breakout': True, 'strength': strength, 'leverage': leverage}
        elif price_break:
            return {'is_breakout': True, 'strength': 0.2, 'leverage': 5}

        return {'is_breakout': False, 'strength': 0.0, 'leverage': 5}

    # ============================================================
    # ENTRY VALIDATION
    # ============================================================

    def _validate_entry_conditions(self, btc_df: pd.DataFrame, xrp_df: pd.DataFrame,
                                   direction: str, corr: float) -> Tuple[bool, str]:
        """
        Validate all entry conditions before taking a position.
        Returns (can_enter, reason_if_blocked).
        """
        # 1. Cointegration check
        if self.use_cointegration:
            is_coint, pvalue = self._test_cointegration(btc_df, xrp_df)
            if not is_coint:
                return False, f"No cointegration (p={pvalue:.3f} > {self.coint_pvalue_threshold})"

        # 2. Lead-Lag Ratio check
        if self.use_llr:
            llr = self._calculate_llr(btc_df, xrp_df)

            # Check for normal lead-lag (BTC leads)
            if direction in ['long', 'short'] and llr < self.min_llr:
                # Check if inverse lead-lag is allowed and applicable
                if self.allow_inverse_leadlag and llr < self.inverse_llr_threshold:
                    pass  # Allow inverse lead-lag trades
                else:
                    return False, f"BTC not leading (LLR={llr:.2f} < {self.min_llr})"

        # 3. Dynamic lag correlation check
        if self.use_dynamic_lag:
            self._find_optimal_lag(btc_df, xrp_df)
            if self.optimal_lag_correlation < self.min_lag_correlation:
                return False, f"Weak lag correlation ({self.optimal_lag_correlation:.2f} < {self.min_lag_correlation})"

        # 4. Correlation trend check (warn on weakening)
        if self.correlation_trend == 'weakening' and corr < 0.8:
            return False, f"Correlation weakening (trend={self.correlation_trend}, corr={corr:.2f})"

        # 5. Session filter
        if not self._session_filter_allows_trade():
            return False, f"Session filter: {self.current_session} not in {self.preferred_sessions}"

        return True, "All conditions met"

    # ============================================================
    # MAIN SIGNAL GENERATION
    # ============================================================

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate trading signals based on BTC/XRP correlation and lead-lag."""
        # Find BTC and XRP data keys
        btc_key = None
        xrp_key = None
        for key in data.keys():
            if 'BTC' in key.upper() and '5m' not in key.lower() and '15m' not in key.lower():
                btc_key = key
            if 'XRP' in key.upper() and '5m' not in key.lower() and '15m' not in key.lower():
                xrp_key = key

        if not btc_key or not xrp_key:
            return self._hold_signal('XRP/USDT', 'Missing BTC or XRP data')

        btc_df = data[btc_key]
        xrp_df = data[xrp_key]

        if len(btc_df) < self.lookback + 5 or len(xrp_df) < self.lookback + 5:
            return self._hold_signal(xrp_key, 'Insufficient data for correlation')

        # Update regime and adjust thresholds
        self._detect_regime(btc_df)
        self._adjust_thresholds_for_regime()

        # Calculate core metrics
        corr = self._get_correlation(btc_df, xrp_df)
        self.last_correlation = corr
        self._update_correlation_trend(corr)

        # Dynamic lag detection
        if self.use_dynamic_lag:
            self._find_optimal_lag(btc_df, xrp_df)

        # Calculate LLR
        if self.use_llr:
            self._calculate_llr(btc_df, xrp_df)

        # Detect BTC trend
        btc_trend = self._detect_btc_trend(btc_df)
        self.btc_trend = btc_trend

        # Calculate volume confidence
        if self.use_volume_confirmation:
            self._calculate_volume_confidence(btc_df)

        # Build indicators dict for logging
        indicators = self._build_indicators_dict(btc_df, xrp_df, corr)

        # CHECK EXIT CONDITIONS FIRST
        exit_signal = self._check_exit_conditions(xrp_df, corr)
        if exit_signal:
            exit_signal['indicators'] = indicators
            return exit_signal

        # If already in position, hold
        if self.current_position is not None:
            current_price = xrp_df['close'].iloc[-1]
            pnl_pct = self._calculate_pnl_pct(current_price)
            return self._hold_signal(
                xrp_key,
                f'In {self.current_position} position, bars={self.position_bars}, P&L={pnl_pct*100:.2f}%',
                indicators
            )

        # BTC BREAKOUT - Check first for immediate action
        btc_breakout = self._detect_btc_breakout(btc_df)
        if btc_breakout['is_breakout'] and corr > 0.65:
            can_enter, block_reason = self._validate_entry_conditions(btc_df, xrp_df, 'long', corr)
            if can_enter:
                # Check MTF confirmation
                if self._check_mtf_confirmation(data, 'up'):
                    confidence = 0.85 + btc_breakout['strength'] * 0.05
                    confidence *= self.volume_confidence_mult
                    entry_price = xrp_df['close'].iloc[-1]
                    self._record_entry('long', entry_price, corr)
                    return {
                        'action': 'buy',
                        'symbol': xrp_key,
                        'size': self.position_size_pct * 1.2,
                        'leverage': min(self.max_leverage, btc_breakout['leverage']),
                        'confidence': min(confidence, 0.95),
                        'order_type': 'limit',  # Phase 32: Limit orders for better fills
                        'limit_price': entry_price,
                        'reason': f'BTC BREAKOUT: new high + vol {btc_breakout["strength"]:.1f}x → XRP follow (corr={corr:.2f}, lag={self.optimal_lag})',
                        'strategy': 'leadlag',
                        'breakout': True,
                        'indicators': indicators
                    }

        # HIGH CORRELATION MODE - Follow BTC
        if corr > self.corr_high and btc_trend != 'none':
            can_enter, block_reason = self._validate_entry_conditions(
                btc_df, xrp_df, 'long' if btc_trend == 'up' else 'short', corr
            )

            if can_enter:
                # Check MTF confirmation
                if self._check_mtf_confirmation(data, btc_trend):
                    entry_price = xrp_df['close'].iloc[-1]
                    base_confidence = 0.7 + (corr - 0.75) * 0.5
                    confidence = base_confidence * self.volume_confidence_mult

                    if btc_trend == 'up':
                        self._record_entry('long', entry_price, corr)
                        return {
                            'action': 'buy',
                            'symbol': xrp_key,
                            'size': self.position_size_pct,
                            'leverage': min(self.max_leverage, 5),
                            'confidence': min(confidence, 0.9),
                            'order_type': 'limit',  # Phase 32: Limit orders
                            'limit_price': entry_price,
                            'reason': f'Lead-lag: BTC up +{self.min_btc_move*100:.1f}%, corr={corr:.2f}, lag={self.optimal_lag}, LLR={self.current_llr:.2f}',
                            'strategy': 'leadlag',
                            'indicators': indicators
                        }
                    else:  # btc_trend == 'down'
                        self._record_entry('short', entry_price, corr)
                        return {
                            'action': 'short',
                            'symbol': xrp_key,
                            'size': self.position_size_pct * 0.8,
                            'leverage': min(self.max_leverage, 5),
                            'confidence': min(confidence * 0.95, 0.85),
                            'order_type': 'limit',  # Phase 32: Limit orders
                            'limit_price': entry_price,
                            'reason': f'Lead-lag: BTC down -{self.min_btc_move*100:.1f}%, corr={corr:.2f}, lag={self.optimal_lag}, LLR={self.current_llr:.2f}',
                            'strategy': 'leadlag',
                            'indicators': indicators
                        }
                else:
                    return self._hold_signal(xrp_key, f'MTF disagreement (1h={btc_trend}, 15m different)', indicators)
            else:
                return self._hold_signal(xrp_key, f'Entry blocked: {block_reason}', indicators)

        # LOW CORRELATION MODE - Check for divergence
        if corr < self.corr_low:
            divergence = self._check_divergence(btc_df, xrp_df)
            if divergence:
                entry_price = xrp_df['close'].iloc[-1]

                if divergence == 'long_xrp':
                    self._record_entry('long', entry_price, corr)
                    return {
                        'action': 'buy',
                        'symbol': xrp_key,
                        'size': self.position_size_pct * 0.6,
                        'leverage': min(self.max_leverage, 3),
                        'confidence': 0.55 * self.volume_confidence_mult,
                        'order_type': 'limit',  # Phase 32: Limit orders
                        'limit_price': entry_price,
                        'reason': f'Divergence: XRP lagging BTC, corr={corr:.2f}, expect catch-up',
                        'strategy': 'leadlag',
                        'indicators': indicators
                    }
                elif divergence == 'short_xrp':
                    self._record_entry('short', entry_price, corr)
                    return {
                        'action': 'short',
                        'symbol': xrp_key,
                        'size': self.position_size_pct * 0.5,
                        'leverage': min(self.max_leverage, 3),
                        'confidence': 0.50 * self.volume_confidence_mult,
                        'order_type': 'limit',  # Phase 32: Limit orders
                        'limit_price': entry_price,
                        'reason': f'Divergence: XRP leading BTC, corr={corr:.2f}, expect pullback',
                        'strategy': 'leadlag',
                        'indicators': indicators
                    }

        # No clear signal
        return self._hold_signal(
            xrp_key,
            f'No signal (corr={corr:.2f}, btc_trend={btc_trend}, lag={self.optimal_lag}, LLR={self.current_llr:.2f})',
            indicators
        )

    def _hold_signal(self, symbol: str, reason: str, indicators: Dict = None) -> Dict[str, Any]:
        """Generate a hold signal."""
        signal = {
            'action': 'hold',
            'symbol': symbol,
            'confidence': 0.0,
            'reason': reason,
            'strategy': 'leadlag'
        }
        if indicators:
            signal['indicators'] = indicators
        return signal

    def _calculate_pnl_pct(self, current_price: float) -> float:
        """Calculate current P&L percentage."""
        if self.current_position == 'long':
            return (current_price - self.entry_price) / self.entry_price
        elif self.current_position == 'short':
            return (self.entry_price - current_price) / self.entry_price
        return 0.0

    def _build_indicators_dict(self, btc_df: pd.DataFrame, xrp_df: pd.DataFrame, corr: float) -> Dict:
        """Build comprehensive indicators dict for logging."""
        return {
            'correlation': corr,
            'correlation_trend': self.correlation_trend,
            'btc_trend': self.btc_trend,
            'optimal_lag': self.optimal_lag,
            'optimal_lag_correlation': self.optimal_lag_correlation,
            'llr': self.current_llr,
            'regime': self.current_regime,
            'regime_volatility': self.regime_volatility,
            'min_btc_move_adjusted': self.min_btc_move,
            'volume_confidence_mult': self.volume_confidence_mult,
            'mtf_agreement': self.mtf_agreement,
            'session': self.current_session,
            'cointegration_pvalue': self.cointegration_pvalue,
            'is_cointegrated': self.is_cointegrated,
            'current_position': self.current_position,
            'entry_price': self.entry_price,
            'position_bars': self.position_bars,
            'trailing_stop_active': self.trailing_stop_active,
            'trailing_stop_price': self.trailing_stop_price,
            'btc_price': btc_df['close'].iloc[-1] if len(btc_df) > 0 else 0,
            'xrp_price': xrp_df['close'].iloc[-1] if len(xrp_df) > 0 else 0
        }

    # ============================================================
    # POSITION SYNC (on_order_filled)
    # ============================================================

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """
        Sync internal state when orders are filled by orchestrator.
        This ensures position state stays in sync with actual executions.
        """
        action = order.get('action', '')
        symbol = order.get('symbol', '')
        price = order.get('price', 0)

        if 'XRP' not in symbol.upper():
            return

        if action in ['sell', 'cover', 'close']:
            # Position closed
            self._reset_position_state()
        elif action == 'buy':
            # Long position opened
            if self.current_position is None:
                self._record_entry('long', price, self.last_correlation)
        elif action == 'short':
            # Short position opened
            if self.current_position is None:
                self._record_entry('short', price, self.last_correlation)

    # ============================================================
    # MODEL UPDATE AND STATUS
    # ============================================================

    def update_model(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """Lead-lag is rule-based, no model to update."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive strategy status."""
        base_status = super().get_status()
        base_status.update({
            # Core metrics
            'correlation': self.last_correlation,
            'correlation_trend': self.correlation_trend,
            'btc_trend': self.btc_trend,

            # Dynamic lag
            'optimal_lag': self.optimal_lag,
            'optimal_lag_correlation': self.optimal_lag_correlation,
            'use_dynamic_lag': self.use_dynamic_lag,

            # Lead-Lag Ratio
            'llr': self.current_llr,
            'min_llr': self.min_llr,
            'use_llr': self.use_llr,

            # Cointegration
            'cointegration_pvalue': self.cointegration_pvalue,
            'is_cointegrated': self.is_cointegrated,
            'use_cointegration': self.use_cointegration,

            # Regime
            'regime': self.current_regime,
            'regime_volatility': self.regime_volatility,
            'min_btc_move_adjusted': self.min_btc_move,

            # Volume
            'volume_confidence_mult': self.volume_confidence_mult,

            # Multi-timeframe
            'mtf_agreement': self.mtf_agreement,
            'use_mtf_confirmation': self.use_mtf_confirmation,

            # Session
            'current_session': self.current_session,
            'use_session_filter': self.use_session_filter,

            # Position
            'current_position': self.current_position,
            'entry_price': self.entry_price,
            'position_bars': self.position_bars,

            # Trailing stop
            'trailing_stop_active': self.trailing_stop_active,
            'trailing_stop_price': self.trailing_stop_price,
            'use_trailing_stop': self.use_trailing_stop,

            # Thresholds
            'corr_high_threshold': self.corr_high,
            'corr_low_threshold': self.corr_low,
            'lookback': self.lookback,
            'take_profit_pct': self.take_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,

            # Features enabled
            'features': {
                'dynamic_lag': self.use_dynamic_lag,
                'cointegration': self.use_cointegration,
                'llr': self.use_llr,
                'regime_adjustment': self.use_regime_adjustment,
                'trailing_stop': self.use_trailing_stop,
                'volume_confirmation': self.use_volume_confirmation,
                'mtf_confirmation': self.use_mtf_confirmation,
                'session_filter': self.use_session_filter,
                'inverse_leadlag': self.allow_inverse_leadlag
            }
        })
        return base_status
