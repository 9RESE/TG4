"""
Momentum Scalping Strategy - Monitoring Module

REC-012 (v2.1.0): XRP Independence Monitoring
- Track XRP-BTC correlation over time for trend analysis
- Weekly review capability with correlation history
- Escalation triggers for sustained low correlation

REC-013 (v2.1.0): Market Sentiment Monitoring
- Fetch Fear & Greed Index for volatility expansion signals
- Daily monitoring with regime alignment
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field

from .config import STRATEGY_NAME

# Configure logger
logger = logging.getLogger(STRATEGY_NAME)


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class CorrelationRecord:
    """Single correlation data point."""
    timestamp: str
    correlation: float
    xrp_btc_paused: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorrelationRecord':
        """Create from dictionary."""
        return cls(
            timestamp=data.get('timestamp', ''),
            correlation=data.get('correlation', 0.0),
            xrp_btc_paused=data.get('xrp_btc_paused', False)
        )


@dataclass
class SentimentRecord:
    """Market sentiment data point."""
    timestamp: str
    fear_greed_index: int
    classification: str  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentRecord':
        """Create from dictionary."""
        return cls(
            timestamp=data.get('timestamp', ''),
            fear_greed_index=data.get('fear_greed_index', 50),
            classification=data.get('classification', 'Neutral')
        )


@dataclass
class MonitoringState:
    """Persistent monitoring state."""
    correlation_history: List[CorrelationRecord] = field(default_factory=list)
    sentiment_history: List[SentimentRecord] = field(default_factory=list)
    last_weekly_review: Optional[str] = None
    last_sentiment_fetch: Optional[str] = None
    consecutive_low_correlation_days: int = 0
    pause_session_count: int = 0
    total_session_count: int = 0


# =============================================================================
# REC-012: XRP Independence Monitoring
# =============================================================================
class CorrelationMonitor:
    """
    Monitor XRP-BTC correlation for independence trends.

    REC-012 Action Items:
    1. Review correlation status weekly
    2. Log correlation values for trend analysis
    3. Consider raising threshold to 0.65 if decline continues

    Trigger for Escalation:
    - XRP-BTC correlation consistently below 0.70 for 30 days
    - XRP/BTC pair paused more than 50% of trading sessions
    """

    # Thresholds
    LOW_CORRELATION_THRESHOLD = 0.70  # For escalation tracking
    ESCALATION_THRESHOLD_DAYS = 30    # Days of low correlation for escalation
    PAUSE_RATE_THRESHOLD = 0.50       # 50% pause rate triggers escalation

    def __init__(self, state: MonitoringState, log_dir: Optional[Path] = None):
        self.state = state
        self.log_dir = log_dir or Path("logs/monitoring")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record_correlation(
        self,
        correlation: float,
        xrp_btc_paused: bool,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Record a correlation observation.

        Args:
            correlation: Current XRP-BTC correlation coefficient
            xrp_btc_paused: Whether XRP/BTC trading is currently paused
            timestamp: Observation timestamp (default: now)

        Returns:
            Dict with recording status and any alerts
        """
        ts = timestamp or datetime.now()

        record = CorrelationRecord(
            timestamp=ts.isoformat(),
            correlation=correlation,
            xrp_btc_paused=xrp_btc_paused
        )

        self.state.correlation_history.append(record)
        self.state.total_session_count += 1

        if xrp_btc_paused:
            self.state.pause_session_count += 1

        # Track consecutive low correlation days
        if correlation < self.LOW_CORRELATION_THRESHOLD:
            self.state.consecutive_low_correlation_days += 1
        else:
            self.state.consecutive_low_correlation_days = 0

        # Check for escalation triggers
        alerts = self._check_escalation_triggers()

        # Keep history bounded (90 days of hourly data max)
        max_records = 90 * 24
        if len(self.state.correlation_history) > max_records:
            self.state.correlation_history = self.state.correlation_history[-max_records:]

        return {
            'recorded': True,
            'correlation': correlation,
            'alerts': alerts,
            'consecutive_low_days': self.state.consecutive_low_correlation_days,
            'pause_rate': self._calculate_pause_rate()
        }

    def _check_escalation_triggers(self) -> List[str]:
        """Check if any escalation triggers are met."""
        alerts = []

        # Trigger 1: Sustained low correlation
        if self.state.consecutive_low_correlation_days >= self.ESCALATION_THRESHOLD_DAYS:
            alerts.append(
                f"ESCALATION: XRP-BTC correlation below {self.LOW_CORRELATION_THRESHOLD} "
                f"for {self.state.consecutive_low_correlation_days} consecutive days"
            )
            logger.warning("Correlation escalation trigger", extra={
                'trigger': 'sustained_low_correlation',
                'days': self.state.consecutive_low_correlation_days,
                'threshold': self.LOW_CORRELATION_THRESHOLD
            })

        # Trigger 2: High pause rate
        pause_rate = self._calculate_pause_rate()
        if pause_rate > self.PAUSE_RATE_THRESHOLD and self.state.total_session_count >= 100:
            alerts.append(
                f"ESCALATION: XRP/BTC paused {pause_rate:.1%} of sessions "
                f"(threshold: {self.PAUSE_RATE_THRESHOLD:.0%})"
            )
            logger.warning("Pause rate escalation trigger", extra={
                'trigger': 'high_pause_rate',
                'pause_rate': pause_rate,
                'threshold': self.PAUSE_RATE_THRESHOLD
            })

        return alerts

    def _calculate_pause_rate(self) -> float:
        """Calculate the percentage of sessions where XRP/BTC was paused."""
        if self.state.total_session_count == 0:
            return 0.0
        return self.state.pause_session_count / self.state.total_session_count

    def generate_weekly_report(self) -> Dict[str, Any]:
        """
        Generate weekly correlation review report.

        REC-012 Action Item 1: Review correlation status weekly
        """
        now = datetime.now()
        week_ago = now - timedelta(days=7)

        # Filter records from the last week
        weekly_records = [
            r for r in self.state.correlation_history
            if datetime.fromisoformat(r.timestamp) > week_ago
        ]

        if not weekly_records:
            return {
                'status': 'no_data',
                'message': 'No correlation data from the past week'
            }

        correlations = [r.correlation for r in weekly_records]
        paused_count = sum(1 for r in weekly_records if r.xrp_btc_paused)

        report = {
            'report_date': now.isoformat(),
            'period_start': week_ago.isoformat(),
            'period_end': now.isoformat(),
            'sample_count': len(weekly_records),
            'correlation_stats': {
                'mean': sum(correlations) / len(correlations),
                'min': min(correlations),
                'max': max(correlations),
                'current': correlations[-1],
                'trend': self._calculate_trend(correlations)
            },
            'pause_stats': {
                'paused_sessions': paused_count,
                'total_sessions': len(weekly_records),
                'pause_rate': paused_count / len(weekly_records) if weekly_records else 0
            },
            'escalation_status': {
                'consecutive_low_days': self.state.consecutive_low_correlation_days,
                'days_until_escalation': max(
                    0,
                    self.ESCALATION_THRESHOLD_DAYS - self.state.consecutive_low_correlation_days
                ),
                'overall_pause_rate': self._calculate_pause_rate()
            },
            'recommendation': self._generate_recommendation(correlations)
        }

        # Update last review time
        self.state.last_weekly_review = now.isoformat()

        # Log the report
        logger.info("Weekly correlation report generated", extra=report)

        # Save report to file
        self._save_weekly_report(report)

        return report

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return 'insufficient_data'

        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

        diff = second_half - first_half
        if diff > 0.05:
            return 'increasing'
        elif diff < -0.05:
            return 'decreasing'
        return 'stable'

    def _generate_recommendation(self, correlations: List[float]) -> str:
        """Generate actionable recommendation based on correlation data."""
        if not correlations:
            return "Insufficient data for recommendation"

        avg_correlation = sum(correlations) / len(correlations)
        trend = self._calculate_trend(correlations)

        if avg_correlation < 0.60:
            if trend == 'decreasing':
                return (
                    "CRITICAL: Consider pausing XRP/BTC trading. "
                    "Correlation is low and declining. "
                    "Review pair viability."
                )
            return (
                "WARNING: Correlation is low. "
                "Monitor closely and consider raising pause threshold to 0.65."
            )
        elif avg_correlation < 0.70:
            if trend == 'decreasing':
                return (
                    "CAUTION: Correlation approaching critical levels. "
                    "Consider raising pause threshold."
                )
            return "Monitor: Correlation is moderate. Continue weekly reviews."
        else:
            return "HEALTHY: Correlation is adequate for momentum scalping."

    def _save_weekly_report(self, report: Dict[str, Any]) -> None:
        """Save weekly report to file."""
        report_date = datetime.now().strftime("%Y%m%d")
        report_path = self.log_dir / f"correlation_report_{report_date}.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Weekly report saved to {report_path}")

    def get_correlation_trend_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Get correlation trend data for analysis.

        REC-012 Action Item 2: Log correlation values for trend analysis
        """
        cutoff = datetime.now() - timedelta(days=days)

        recent_records = [
            r for r in self.state.correlation_history
            if datetime.fromisoformat(r.timestamp) > cutoff
        ]

        if not recent_records:
            return {'status': 'no_data', 'days': days}

        correlations = [r.correlation for r in recent_records]
        timestamps = [r.timestamp for r in recent_records]

        return {
            'days': days,
            'sample_count': len(recent_records),
            'timestamps': timestamps,
            'correlations': correlations,
            'mean': sum(correlations) / len(correlations),
            'trend': self._calculate_trend(correlations),
            'days_below_070': sum(1 for c in correlations if c < 0.70),
            'days_below_065': sum(1 for c in correlations if c < 0.65),
            'days_below_060': sum(1 for c in correlations if c < 0.60)
        }


# =============================================================================
# REC-013: Market Sentiment Monitoring
# =============================================================================
class SentimentMonitor:
    """
    Monitor Fear & Greed Index for market sentiment.

    REC-013 Action Items:
    1. Monitor Fear & Greed Index daily
    2. Regime classification will auto-adjust
    3. No manual intervention needed unless prolonged extremes

    Fear & Greed Index Levels:
    - 0-24: Extreme Fear (historically precedes volatility expansion)
    - 25-44: Fear
    - 45-55: Neutral
    - 56-75: Greed
    - 76-100: Extreme Greed
    """

    # Classification thresholds
    EXTREME_FEAR_THRESHOLD = 24
    FEAR_THRESHOLD = 44
    NEUTRAL_THRESHOLD = 55
    GREED_THRESHOLD = 75

    def __init__(self, state: MonitoringState, log_dir: Optional[Path] = None):
        self.state = state
        self.log_dir = log_dir or Path("logs/monitoring")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def classify_sentiment(self, index: int) -> str:
        """Classify Fear & Greed Index into sentiment category."""
        if index <= self.EXTREME_FEAR_THRESHOLD:
            return "Extreme Fear"
        elif index <= self.FEAR_THRESHOLD:
            return "Fear"
        elif index <= self.NEUTRAL_THRESHOLD:
            return "Neutral"
        elif index <= self.GREED_THRESHOLD:
            return "Greed"
        return "Extreme Greed"

    def record_sentiment(
        self,
        fear_greed_index: int,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Record a sentiment observation.

        Args:
            fear_greed_index: Fear & Greed Index value (0-100)
            timestamp: Observation timestamp (default: now)

        Returns:
            Dict with recording status and analysis
        """
        ts = timestamp or datetime.now()
        classification = self.classify_sentiment(fear_greed_index)

        record = SentimentRecord(
            timestamp=ts.isoformat(),
            fear_greed_index=fear_greed_index,
            classification=classification
        )

        self.state.sentiment_history.append(record)
        self.state.last_sentiment_fetch = ts.isoformat()

        # Analyze current conditions
        analysis = self._analyze_sentiment(fear_greed_index, classification)

        # Keep history bounded (365 days of daily data)
        max_records = 365
        if len(self.state.sentiment_history) > max_records:
            self.state.sentiment_history = self.state.sentiment_history[-max_records:]

        logger.info("Sentiment recorded", extra={
            'fear_greed_index': fear_greed_index,
            'classification': classification,
            'analysis': analysis
        })

        return {
            'recorded': True,
            'fear_greed_index': fear_greed_index,
            'classification': classification,
            'analysis': analysis
        }

    def _analyze_sentiment(self, index: int, classification: str) -> Dict[str, Any]:
        """Analyze current sentiment conditions."""
        analysis = {
            'volatility_signal': 'neutral',
            'regime_impact': 'none',
            'recommendation': ''
        }

        if classification == "Extreme Fear":
            analysis['volatility_signal'] = 'high_expansion_expected'
            analysis['regime_impact'] = 'potential_regime_shift'
            analysis['recommendation'] = (
                "Extreme Fear detected. Historically precedes volatility expansion. "
                "Current MEDIUM regime favorable for strategy. "
                "Monitor for regime shift to HIGH/EXTREME."
            )
        elif classification == "Fear":
            analysis['volatility_signal'] = 'moderate_expansion_possible'
            analysis['recommendation'] = (
                "Fear sentiment. Stay alert for increased volatility. "
                "Strategy can continue normal operation."
            )
        elif classification == "Extreme Greed":
            analysis['volatility_signal'] = 'reversal_risk'
            analysis['regime_impact'] = 'potential_regime_shift'
            analysis['recommendation'] = (
                "Extreme Greed detected. Market may be overextended. "
                "Watch for sharp reversals and regime shifts."
            )
        else:
            analysis['recommendation'] = "Normal market sentiment. Continue standard operation."

        return analysis

    def get_sentiment_trend(self, days: int = 30) -> Dict[str, Any]:
        """Get sentiment trend over specified period."""
        cutoff = datetime.now() - timedelta(days=days)

        recent_records = [
            r for r in self.state.sentiment_history
            if datetime.fromisoformat(r.timestamp) > cutoff
        ]

        if not recent_records:
            return {'status': 'no_data', 'days': days}

        indices = [r.fear_greed_index for r in recent_records]
        classifications = [r.classification for r in recent_records]

        # Count days in each sentiment category
        sentiment_counts = {}
        for c in classifications:
            sentiment_counts[c] = sentiment_counts.get(c, 0) + 1

        return {
            'days': days,
            'sample_count': len(recent_records),
            'current_index': indices[-1],
            'current_classification': classifications[-1],
            'mean_index': sum(indices) / len(indices),
            'min_index': min(indices),
            'max_index': max(indices),
            'sentiment_distribution': sentiment_counts,
            'extreme_fear_days': sentiment_counts.get('Extreme Fear', 0),
            'extreme_greed_days': sentiment_counts.get('Extreme Greed', 0)
        }

    def should_alert_prolonged_extreme(self, threshold_days: int = 7) -> Optional[str]:
        """
        Check for prolonged extreme sentiment conditions.

        REC-013: No manual intervention needed unless prolonged extremes
        """
        if len(self.state.sentiment_history) < threshold_days:
            return None

        recent = self.state.sentiment_history[-threshold_days:]
        classifications = [r.classification for r in recent]

        # Check for prolonged extreme fear
        if all(c == "Extreme Fear" for c in classifications):
            return (
                f"ALERT: Extreme Fear for {threshold_days}+ consecutive days. "
                "Consider defensive position sizing."
            )

        # Check for prolonged extreme greed
        if all(c == "Extreme Greed" for c in classifications):
            return (
                f"ALERT: Extreme Greed for {threshold_days}+ consecutive days. "
                "Watch for market reversal."
            )

        return None


# =============================================================================
# Combined Monitoring Manager
# =============================================================================
class MonitoringManager:
    """
    Combined manager for REC-012 and REC-013 monitoring.

    Provides unified interface for:
    - XRP Independence Monitoring (REC-012)
    - Market Sentiment Monitoring (REC-013)
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/monitoring")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()
        self.correlation_monitor = CorrelationMonitor(self.state, self.log_dir)
        self.sentiment_monitor = SentimentMonitor(self.state, self.log_dir)

    def _get_state_path(self) -> Path:
        """Get path to state file."""
        return self.log_dir / "monitoring_state.json"

    def _load_state(self) -> MonitoringState:
        """Load monitoring state from file."""
        state_path = self._get_state_path()

        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    data = json.load(f)

                state = MonitoringState()
                state.correlation_history = [
                    CorrelationRecord.from_dict(r)
                    for r in data.get('correlation_history', [])
                ]
                state.sentiment_history = [
                    SentimentRecord.from_dict(r)
                    for r in data.get('sentiment_history', [])
                ]
                state.last_weekly_review = data.get('last_weekly_review')
                state.last_sentiment_fetch = data.get('last_sentiment_fetch')
                state.consecutive_low_correlation_days = data.get(
                    'consecutive_low_correlation_days', 0
                )
                state.pause_session_count = data.get('pause_session_count', 0)
                state.total_session_count = data.get('total_session_count', 0)

                logger.info("Monitoring state loaded", extra={
                    'correlation_records': len(state.correlation_history),
                    'sentiment_records': len(state.sentiment_history)
                })

                return state

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load monitoring state: {e}")

        return MonitoringState()

    def save_state(self) -> None:
        """Save monitoring state to file."""
        state_path = self._get_state_path()

        data = {
            'correlation_history': [asdict(r) for r in self.state.correlation_history],
            'sentiment_history': [asdict(r) for r in self.state.sentiment_history],
            'last_weekly_review': self.state.last_weekly_review,
            'last_sentiment_fetch': self.state.last_sentiment_fetch,
            'consecutive_low_correlation_days': self.state.consecutive_low_correlation_days,
            'pause_session_count': self.state.pause_session_count,
            'total_session_count': self.state.total_session_count
        }

        with open(state_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"Monitoring state saved to {state_path}")

    def record_session_data(
        self,
        correlation: float,
        xrp_btc_paused: bool,
        fear_greed_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Record session monitoring data.

        Args:
            correlation: Current XRP-BTC correlation
            xrp_btc_paused: Whether XRP/BTC trading is paused
            fear_greed_index: Optional Fear & Greed Index (if fetched)

        Returns:
            Combined monitoring status
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'correlation': None,
            'sentiment': None,
            'alerts': []
        }

        # Record correlation (REC-012)
        corr_result = self.correlation_monitor.record_correlation(
            correlation, xrp_btc_paused
        )
        result['correlation'] = corr_result
        result['alerts'].extend(corr_result.get('alerts', []))

        # Record sentiment if provided (REC-013)
        if fear_greed_index is not None:
            sent_result = self.sentiment_monitor.record_sentiment(fear_greed_index)
            result['sentiment'] = sent_result

            # Check for prolonged extreme alert
            extreme_alert = self.sentiment_monitor.should_alert_prolonged_extreme()
            if extreme_alert:
                result['alerts'].append(extreme_alert)

        # Save state after recording
        self.save_state()

        return result

    def needs_weekly_review(self) -> bool:
        """Check if weekly correlation review is due."""
        if self.state.last_weekly_review is None:
            return True

        last_review = datetime.fromisoformat(self.state.last_weekly_review)
        return datetime.now() - last_review > timedelta(days=7)

    def needs_sentiment_update(self) -> bool:
        """Check if daily sentiment update is due."""
        if self.state.last_sentiment_fetch is None:
            return True

        last_fetch = datetime.fromisoformat(self.state.last_sentiment_fetch)
        return datetime.now() - last_fetch > timedelta(hours=24)

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            'timestamp': datetime.now().isoformat(),
            'correlation': {
                'trend_30d': self.correlation_monitor.get_correlation_trend_data(30),
                'consecutive_low_days': self.state.consecutive_low_correlation_days,
                'pause_rate': self.correlation_monitor._calculate_pause_rate(),
                'needs_weekly_review': self.needs_weekly_review()
            },
            'sentiment': {
                'trend_30d': self.sentiment_monitor.get_sentiment_trend(30),
                'needs_update': self.needs_sentiment_update(),
                'prolonged_extreme_alert': self.sentiment_monitor.should_alert_prolonged_extreme()
            },
            'state_summary': {
                'correlation_records': len(self.state.correlation_history),
                'sentiment_records': len(self.state.sentiment_history),
                'total_sessions_tracked': self.state.total_session_count
            }
        }


# =============================================================================
# Helper Functions for Integration
# =============================================================================
def integrate_monitoring_on_tick(
    state: Dict[str, Any],
    config: Dict[str, Any],
    correlation: Optional[float],
    xrp_btc_paused: bool,
    manager: Optional[MonitoringManager] = None
) -> Dict[str, Any]:
    """
    Integrate monitoring into strategy tick processing.

    Call this during each strategy tick to track correlation.

    Args:
        state: Strategy state dict
        config: Strategy config
        correlation: Current XRP-BTC correlation
        xrp_btc_paused: Whether XRP/BTC trading is paused
        manager: Optional MonitoringManager instance

    Returns:
        Monitoring status dict
    """
    if manager is None:
        # Create manager if not provided (will load existing state)
        manager = MonitoringManager()

    # Only record if correlation monitoring is enabled
    if not config.get('use_correlation_monitoring', True):
        return {'monitoring_enabled': False}

    if correlation is None:
        return {'correlation_available': False}

    # Record the data point
    result = manager.record_session_data(
        correlation=correlation,
        xrp_btc_paused=xrp_btc_paused
    )

    # Store manager reference in state for reuse
    state['_monitoring_manager'] = manager

    # Log any alerts
    for alert in result.get('alerts', []):
        logger.warning(alert)

    return result


def get_or_create_monitoring_manager(state: Dict[str, Any]) -> MonitoringManager:
    """Get existing monitoring manager or create new one."""
    if '_monitoring_manager' not in state:
        state['_monitoring_manager'] = MonitoringManager()
    return state['_monitoring_manager']
