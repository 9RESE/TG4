"""
Centralized Indicator Library for Trading Strategies

This library consolidates common technical indicator calculations
that were previously duplicated across 8+ strategy modules.

Usage:
    from ws_tester.indicators import (
        calculate_sma, calculate_ema,
        calculate_rsi, calculate_adx,
        calculate_volatility, calculate_atr,
        calculate_rolling_correlation,
    )

Modules:
    - moving_averages: SMA, EMA (single value and series)
    - oscillators: RSI, ADX, MACD
    - volatility: ATR, Bollinger Bands, volatility percentage
    - correlation: Rolling Pearson correlation
    - volume: Volume ratio, VPIN, micro-price
    - flow: Trade flow analysis
    - trend: Trend slope, trailing stops
"""

# Type definitions and public helpers
from ._types import (
    PriceInput,
    BollingerResult,
    ATRResult,
    ADXResult,
    MAAlignmentResult,
    TradeFlowResult,
    TrendResult,
    CorrelationTrendResult,
    extract_closes,
    extract_hlc,
    is_candle_data,
)
# Note: extract_volumes is intentionally not exported - it's an internal helper

# Moving Averages
from .moving_averages import (
    calculate_sma,
    calculate_sma_series,
    calculate_ema,
    calculate_ema_series,
    calculate_ma_alignment,
    calculate_ma_alignment_from_data,
)

# Oscillators
from .oscillators import (
    calculate_rsi,
    calculate_rsi_series,
    calculate_adx,
    calculate_adx_with_di,
    calculate_macd,
    calculate_macd_with_history,
)

# Choppiness (consolidation/trending detection)
from .choppiness import (
    calculate_choppiness,
    calculate_choppiness_series,
    is_choppy,
    is_trending_chop,
    get_choppiness_state,
)

# Volatility
from .volatility import (
    calculate_volatility,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_z_score,
    get_volatility_regime,
)

# Correlation
from .correlation import (
    calculate_rolling_correlation,
    calculate_correlation_trend,
)

# Volume
from .volume import (
    calculate_volume_ratio,
    calculate_volume_spike,
    calculate_micro_price,
    calculate_vpin,
)

# Flow
from .flow import (
    calculate_trade_flow,
    check_trade_flow_confirmation,
)

# Trend
from .trend import (
    calculate_trend_slope,
    detect_trend_strength,
    calculate_trailing_stop,
)

__all__ = [
    # Types and helpers
    'PriceInput',
    'BollingerResult',
    'ATRResult',
    'ADXResult',
    'MAAlignmentResult',
    'TradeFlowResult',
    'TrendResult',
    'CorrelationTrendResult',
    'extract_closes',
    'extract_hlc',
    'is_candle_data',
    # Moving Averages
    'calculate_sma',
    'calculate_sma_series',
    'calculate_ema',
    'calculate_ema_series',
    'calculate_ma_alignment',
    'calculate_ma_alignment_from_data',
    # Oscillators
    'calculate_rsi',
    'calculate_rsi_series',
    'calculate_adx',
    'calculate_adx_with_di',
    'calculate_macd',
    'calculate_macd_with_history',
    # Choppiness
    'calculate_choppiness',
    'calculate_choppiness_series',
    'is_choppy',
    'is_trending_chop',
    'get_choppiness_state',
    # Volatility
    'calculate_volatility',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_z_score',
    'get_volatility_regime',
    # Correlation
    'calculate_rolling_correlation',
    'calculate_correlation_trend',
    # Volume
    'calculate_volume_ratio',
    'calculate_volume_spike',
    'calculate_micro_price',
    'calculate_vpin',
    # Flow
    'calculate_trade_flow',
    'check_trade_flow_confirmation',
    # Trend
    'calculate_trend_slope',
    'detect_trend_strength',
    'calculate_trailing_stop',
]
