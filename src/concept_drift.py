"""
Concept drift detection for payroll data.

Data changes over time (raises, new hires, etc) so we need to detect
when the model should be retrained. Using Page-Hinkley and ADWIN.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from collections import deque
from scipy import stats
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DriftResult:
    """Result of drift detection"""
    is_drift_detected: bool
    drift_score: float
    affected_features: List[str]
    timestamp: datetime
    recommendation: str


class PageHinkleyTest:
    """
    Page-Hinkley Test for detecting abrupt changes in data stream.

    This is a sequential analysis technique that accumulates the
    difference between observed values and their mean. When the
    cumulative sum exceeds a threshold, drift is detected.

    Good for detecting:
    - Sudden shifts in salary distributions
    - Abrupt changes in overtime patterns
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50,
        alpha: float = 0.9999
    ):
        """
        Args:
            delta: Minimum magnitude of change to detect
            lambda_: Detection threshold
            alpha: Forgetting factor for old observations
        """
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.reset()

    def reset(self):
        """Reset the detector state"""
        self.sum = 0
        self.x_mean = 0
        self.sample_count = 0
        self.min_sum = float('inf')

    def update(self, x: float) -> bool:
        """
        Update detector with new observation.

        Returns True if drift is detected.
        """
        self.sample_count += 1

        # Update running mean
        self.x_mean = self.x_mean + (x - self.x_mean) / self.sample_count

        # Update cumulative sum
        self.sum = self.alpha * self.sum + (x - self.x_mean - self.delta)

        # Track minimum
        self.min_sum = min(self.min_sum, self.sum)

        # Check for drift
        if self.sum - self.min_sum > self.lambda_:
            return True

        return False

    def get_drift_score(self) -> float:
        """Get current drift score (0-1 normalized)"""
        if self.min_sum == float('inf'):
            return 0
        diff = self.sum - self.min_sum
        return min(diff / self.lambda_, 1.0)


class ADWIN:
    """
    Adaptive Windowing (ADWIN) for detecting distribution changes.

    ADWIN maintains a variable-length window of recent observations.
    When the means of two sub-windows differ significantly, it
    indicates concept drift and shrinks the window.

    Advantages:
    - Automatically adjusts window size
    - No need to specify fixed window length
    - Provides theoretical guarantees on false positive rate
    """

    def __init__(self, delta: float = 0.002):
        """
        Args:
            delta: Confidence parameter (lower = more sensitive)
        """
        self.delta = delta
        self.window = deque()
        self.width = 0
        self.total = 0

    def update(self, x: float) -> bool:
        """
        Add new element and check for drift.

        Returns True if drift is detected.
        """
        self.window.append(x)
        self.total += x
        self.width += 1

        return self._detect_change()

    def _detect_change(self) -> bool:
        """Check if change occurred using ADWIN algorithm"""
        if self.width < 10:  # Minimum samples needed
            return False

        drift_detected = False

        # Try different split points
        for split in range(1, self.width):
            n0 = split
            n1 = self.width - split

            if n0 < 5 or n1 < 5:
                continue

            # Calculate means of sub-windows
            left_sum = sum(list(self.window)[:split])
            right_sum = self.total - left_sum

            mean0 = left_sum / n0
            mean1 = right_sum / n1

            # ADWIN cut condition
            m = 1.0 / (1.0/n0 + 1.0/n1)
            epsilon = np.sqrt((1.0/(2*m)) * np.log(4.0/self.delta))

            if abs(mean0 - mean1) > epsilon:
                # Drift detected, shrink window
                for _ in range(split):
                    removed = self.window.popleft()
                    self.total -= removed
                    self.width -= 1
                drift_detected = True
                break

        return drift_detected

    def get_window_size(self) -> int:
        """Get current window size"""
        return self.width


class ConceptDriftMonitor:
    """
    Monitors multiple features for concept drift.

    Combines multiple drift detection methods and provides
    actionable recommendations for model maintenance.
    """

    def __init__(
        self,
        feature_names: List[str],
        sensitivity: str = 'medium'
    ):
        """
        Args:
            feature_names: Names of features to monitor
            sensitivity: 'low', 'medium', or 'high'
        """
        self.feature_names = feature_names
        self.sensitivity = sensitivity

        # Set parameters based on sensitivity
        params = {
            'low': {'delta': 0.01, 'lambda': 100},
            'medium': {'delta': 0.005, 'lambda': 50},
            'high': {'delta': 0.001, 'lambda': 25}
        }
        self.params = params[sensitivity]

        # Initialize detectors for each feature
        self.ph_detectors = {
            name: PageHinkleyTest(
                delta=self.params['delta'],
                lambda_=self.params['lambda']
            )
            for name in feature_names
        }

        self.adwin_detectors = {
            name: ADWIN(delta=0.002)
            for name in feature_names
        }

        self.drift_history = []
        self.last_retrain_timestamp = None

    def update(self, features: pd.Series) -> DriftResult:
        """
        Update monitors with new observation.

        Args:
            features: Single observation (row) of features

        Returns:
            DriftResult with detection status and recommendations
        """
        drift_detected = False
        affected_features = []
        total_drift_score = 0

        for name in self.feature_names:
            if name not in features.index:
                continue

            value = features[name]
            if pd.isna(value):
                continue

            # Update both detectors
            ph_drift = self.ph_detectors[name].update(value)
            adwin_drift = self.adwin_detectors[name].update(value)

            # Get drift score
            drift_score = self.ph_detectors[name].get_drift_score()
            total_drift_score += drift_score

            if ph_drift or adwin_drift:
                drift_detected = True
                affected_features.append(name)

        avg_drift_score = total_drift_score / len(self.feature_names) if self.feature_names else 0

        # Generate recommendation
        if drift_detected and len(affected_features) > len(self.feature_names) * 0.3:
            recommendation = "RETRAIN: Significant drift detected in >30% of features"
        elif drift_detected:
            recommendation = f"MONITOR: Drift detected in {affected_features}"
        elif avg_drift_score > 0.5:
            recommendation = "REVIEW: Elevated drift scores, consider retraining soon"
        else:
            recommendation = "OK: No significant drift detected"

        result = DriftResult(
            is_drift_detected=drift_detected,
            drift_score=avg_drift_score,
            affected_features=affected_features,
            timestamp=datetime.now(),
            recommendation=recommendation
        )

        self.drift_history.append(result)

        return result

    def batch_update(self, features: pd.DataFrame) -> List[DriftResult]:
        """Update with batch of observations"""
        results = []
        for idx, row in features.iterrows():
            result = self.update(row)
            results.append(result)
        return results

    def should_retrain(self, min_interval_hours: int = 24) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.

        Returns:
            - should_retrain: Boolean
            - reason: Explanation
        """
        if not self.drift_history:
            return False, "No drift history available"

        # Check recent drift detections
        recent_results = self.drift_history[-100:]  # Last 100 observations
        drift_rate = sum(1 for r in recent_results if r.is_drift_detected) / len(recent_results)

        # Check time since last retrain
        if self.last_retrain_timestamp:
            hours_since_retrain = (datetime.now() - self.last_retrain_timestamp).total_seconds() / 3600
            if hours_since_retrain < min_interval_hours:
                return False, f"Recently retrained {hours_since_retrain:.1f}h ago"

        # Decision logic
        if drift_rate > 0.1:  # More than 10% of recent observations showed drift
            return True, f"High drift rate: {drift_rate:.1%}"

        avg_score = np.mean([r.drift_score for r in recent_results])
        if avg_score > 0.6:
            return True, f"Elevated average drift score: {avg_score:.2f}"

        return False, "Drift within acceptable limits"

    def reset_detectors(self):
        """Reset all detectors after retraining"""
        for name in self.feature_names:
            self.ph_detectors[name].reset()
            self.adwin_detectors[name] = ADWIN(delta=0.002)
        self.last_retrain_timestamp = datetime.now()

    def get_drift_summary(self) -> Dict:
        """Get summary statistics of drift detection"""
        if not self.drift_history:
            return {"status": "No data"}

        recent = self.drift_history[-100:]

        return {
            "total_observations": len(self.drift_history),
            "recent_drift_rate": sum(1 for r in recent if r.is_drift_detected) / len(recent),
            "avg_drift_score": np.mean([r.drift_score for r in recent]),
            "most_affected_features": self._get_most_affected_features(recent),
            "last_drift_timestamp": next(
                (r.timestamp for r in reversed(recent) if r.is_drift_detected),
                None
            )
        }

    def _get_most_affected_features(self, results: List[DriftResult]) -> List[str]:
        """Find features most frequently affected by drift"""
        feature_counts = {}
        for r in results:
            for f in r.affected_features:
                feature_counts[f] = feature_counts.get(f, 0) + 1

        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_features[:5]]


# ============================================================
# PSEUDOCODE FOR CONCEPT DRIFT HANDLING
# ============================================================
"""
PSEUDOCODE: Concept Drift Detection and Handling

ALGORITHM: AdaptiveModelMaintenance

INPUT:
    - stream: Continuous stream of payroll transactions
    - model: Trained anomaly detection model
    - feature_engineer: Fitted feature transformer
    - drift_threshold: Threshold for triggering retraining

OUTPUT:
    - predictions: Anomaly predictions (continuously updated)
    - alerts: Drift alerts and retraining triggers

GLOBAL STATE:
    - drift_monitors: Dict of DriftMonitor for each feature
    - retraining_queue: Queue of batches for retraining
    - model_version: Current model version

PROCEDURE:

1. INITIALIZATION
   FOR each feature in feature_names:
       drift_monitors[feature] = PageHinkleyTest(delta=0.005, lambda=50)
   END FOR
   window_buffer = SlidingWindow(size=1000)

2. STREAMING LOOP
   WHILE stream has data:
       transaction = stream.next()

       # Extract features
       features = feature_engineer.transform(transaction)

       # Score with current model
       score = model.score(features)
       prediction = score > threshold

       # Update drift monitors
       FOR each feature in features:
           is_drift = drift_monitors[feature].update(feature_value)

           IF is_drift:
               log_drift_event(feature, timestamp)
           END IF
       END FOR

       # Add to window buffer
       window_buffer.add(transaction)

       # Check if retraining needed
       drift_rate = calculate_drift_rate(drift_monitors)

       IF drift_rate > drift_threshold:
           # Trigger retraining
           CALL trigger_retraining(window_buffer.get_data())

           # Reset monitors
           FOR each monitor in drift_monitors:
               monitor.reset()
           END FOR
       END IF

       YIELD prediction, score
   END WHILE

3. RETRAINING PROCEDURE
   FUNCTION trigger_retraining(recent_data):
       # Combine with historical data
       training_data = merge(historical_sample, recent_data)

       # Retrain model
       new_model = train_model(training_data)

       # Validate new model
       validation_score = evaluate(new_model, validation_set)

       IF validation_score > minimum_threshold:
           # Deploy new model
           model = new_model
           model_version += 1
           log("Model updated to version " + model_version)
       ELSE:
           log("New model failed validation, keeping current")
       END IF
   END FUNCTION

4. ADAPTIVE WINDOW MANAGEMENT
   - Window size automatically adjusts based on drift rate
   - High drift rate -> smaller window (more recent data)
   - Low drift rate -> larger window (more stable baseline)

END ALGORITHM
"""
