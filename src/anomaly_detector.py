"""
Anomaly Detection for Payroll Data

Using Isolation Forest as main algorithm + some statistical rules.
I chose IF because it works well for anomaly detection without needing labels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')


class IsolationForestDetector:
    """
    Main detector using Isolation Forest.
    Chose this over LOF because its faster and scales better.
    """

    def __init__(self, contamination=0.02, n_estimators=50, max_samples=256, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, features: pd.DataFrame) -> 'IsolationForestDetector':
        """Train the isolation forest on normal data patterns"""
        # Scale features for consistent splitting
        X_scaled = self.scaler.fit_transform(features.fillna(0))

        self.model.fit(X_scaled)
        self.feature_names = features.columns.tolist()
        self.fitted = True

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly labels.
        Returns: 1 for normal, -1 for anomaly
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(features.fillna(0))
        return self.model.predict(X_scaled)

    def score_samples(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores for samples.
        Lower (more negative) scores indicate more anomalous samples.

        Returns scores normalized to [0, 1] where 1 = most anomalous
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before scoring")

        X_scaled = self.scaler.transform(features.fillna(0))

        # Raw scores (negative, lower = more anomalous)
        raw_scores = self.model.score_samples(X_scaled)

        # Normalize to [0, 1] where 1 = anomaly
        # More negative raw score -> higher anomaly score
        normalized = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)

        return normalized

    def save(self, path: str):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)

    def load(self, path: str) -> 'IsolationForestDetector':
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.fitted = True
        return self


class StatisticalDetector:
    """
    Rule-based statistical anomaly detection.

    Provides interpretable thresholds that can be explained to
    non-technical stakeholders (e.g., "salary is 3 standard
    deviations above department average").
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5
    ):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.thresholds = {}
        self.fitted = False

    def fit(self, transactions: pd.DataFrame) -> 'StatisticalDetector':
        """Calculate statistical thresholds from training data"""

        # Per-department thresholds
        self.thresholds['dept_salary'] = transactions.groupby('department').agg({
            'base_amount': ['mean', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
            'overtime_hours': ['mean', 'std', lambda x: x.quantile(0.95)]
        })

        # Per-role thresholds
        self.thresholds['role_salary'] = transactions.groupby('role').agg({
            'base_amount': ['mean', 'std', 'min', 'max']
        })

        # Global thresholds
        self.thresholds['global'] = {
            'max_ot_hours': transactions['overtime_hours'].quantile(0.99),
            'max_ot_ratio': 0.4,  # OT should not exceed 40% of total pay
            'max_salary_change': 0.15  # 15% max monthly change
        }

        self.fitted = True
        return self

    def detect(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply statistical rules to detect anomalies.

        Returns DataFrame with boolean flags for each rule violation.
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before detection")

        results = pd.DataFrame(index=transactions.index)

        # Rule 1: Salary Z-score exceeds threshold
        results['rule_salary_zscore'] = transactions.apply(
            lambda x: self._check_salary_zscore(x), axis=1
        )

        # Rule 2: Overtime exceeds 99th percentile
        results['rule_ot_excessive'] = (
            transactions['overtime_hours'] > self.thresholds['global']['max_ot_hours']
        )

        # Rule 3: OT ratio too high
        ot_ratio = transactions['overtime_amount'] / transactions['total_amount'].replace(0, 1)
        results['rule_ot_ratio'] = ot_ratio > self.thresholds['global']['max_ot_ratio']

        # Rule 4: Salary outside role band
        results['rule_salary_band'] = transactions.apply(
            lambda x: self._check_salary_band(x), axis=1
        )

        # Combined flag
        results['any_rule_violation'] = results.any(axis=1)

        # Count of violations
        results['violation_count'] = results.iloc[:, :-1].sum(axis=1)

        return results

    def _check_salary_zscore(self, row) -> bool:
        """Check if salary z-score exceeds threshold"""
        try:
            dept_stats = self.thresholds['dept_salary'].loc[row['department'], 'base_amount']
            mean = dept_stats['mean']
            std = dept_stats['std']
            if std == 0:
                return False
            z_score = abs((row['base_amount'] - mean) / std)
            return z_score > self.z_threshold
        except (KeyError, ZeroDivisionError):
            return False

    def _check_salary_band(self, row) -> bool:
        """Check if salary is outside role band"""
        try:
            role_stats = self.thresholds['role_salary'].loc[row['role'], 'base_amount']
            min_salary = role_stats['min'] * 0.8  # 20% tolerance
            max_salary = role_stats['max'] * 1.2
            return not (min_salary <= row['base_amount'] <= max_salary)
        except KeyError:
            return False


class EnsembleDetector:
    """
    Combines multiple detection methods for robust anomaly detection.

    Ensemble approach benefits:
    - Reduces false positives from any single method
    - Captures different types of anomalies
    - Provides confidence through agreement
    """

    def __init__(
        self,
        contamination: float = 0.02,
        weights: Optional[Dict[str, float]] = None
    ):
        self.isolation_forest = IsolationForestDetector(contamination=contamination)
        self.statistical = StatisticalDetector()

        # Default weights for combining scores
        self.weights = weights or {
            'isolation_forest': 0.6,
            'statistical': 0.4
        }

    def fit(
        self,
        features: pd.DataFrame,
        transactions: pd.DataFrame
    ) -> 'EnsembleDetector':
        """Fit all component detectors"""
        self.isolation_forest.fit(features)
        self.statistical.fit(transactions)
        return self

    def predict(
        self,
        features: pd.DataFrame,
        transactions: pd.DataFrame,
        threshold: float = 0.7
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predict anomalies using ensemble.

        Returns:
            - predictions: Binary array (1 = anomaly)
            - details: DataFrame with individual detector outputs
        """
        # Get Isolation Forest scores
        if_scores = self.isolation_forest.score_samples(features)

        # Get statistical rule violations
        stat_results = self.statistical.detect(transactions)
        stat_scores = stat_results['violation_count'] / 4  # Normalize to [0, 1]

        # Weighted combination
        ensemble_scores = (
            self.weights['isolation_forest'] * if_scores +
            self.weights['statistical'] * stat_scores.values
        )

        # Create detailed output
        details = pd.DataFrame({
            'isolation_forest_score': if_scores,
            'statistical_score': stat_scores.values,
            'ensemble_score': ensemble_scores,
            'is_anomaly': ensemble_scores >= threshold
        })

        # Add rule violation details
        for col in stat_results.columns:
            details[f'stat_{col}'] = stat_results[col].values

        predictions = (ensemble_scores >= threshold).astype(int)

        return predictions, details

    def save(self, path: str):
        """Save ensemble model"""
        self.isolation_forest.save(f"{path}_isolation_forest.joblib")
        joblib.dump({
            'statistical_thresholds': self.statistical.thresholds,
            'weights': self.weights
        }, f"{path}_ensemble_meta.joblib")

    def load(self, path: str) -> 'EnsembleDetector':
        """Load ensemble model"""
        self.isolation_forest.load(f"{path}_isolation_forest.joblib")
        meta = joblib.load(f"{path}_ensemble_meta.joblib")
        self.statistical.thresholds = meta['statistical_thresholds']
        self.statistical.fitted = True
        self.weights = meta['weights']
        return self


# ============================================================
# PSEUDOCODE FOR ALGORITHM
# ============================================================
"""
PSEUDOCODE: Payroll Anomaly Detection Engine

ALGORITHM: EnsembleAnomalyDetection

INPUT:
    - transactions: DataFrame of payroll transactions
    - features: Engineered features from FeatureEngineer
    - contamination_rate: Expected proportion of anomalies (default 0.02)
    - threshold: Decision threshold for ensemble (default 0.7)

OUTPUT:
    - predictions: Binary labels (1 = anomaly, 0 = normal)
    - scores: Continuous anomaly scores [0, 1]
    - explanations: Human-readable explanation of detection

PROCEDURE:

1. PREPROCESSING
   FOR each feature in features:
       - Handle missing values (fill with 0 or median)
       - Standardize using Z-score normalization
   END FOR

2. ISOLATION FOREST DETECTION
   - Build forest of 100 isolation trees
   - FOR each tree:
       - Randomly select feature
       - Randomly select split value within feature range
       - Recursively partition data
   - END FOR
   - FOR each sample:
       - Calculate average path length across all trees
       - Anomaly score = 2^(-avg_path_length / expected_path_length)
   - END FOR
   - Normalize scores to [0, 1]

3. STATISTICAL RULE DETECTION
   FOR each transaction:
       rule_violations = 0

       IF salary Z-score > 3.0:
           rule_violations += 1
       END IF

       IF overtime_hours > 99th_percentile:
           rule_violations += 1
       END IF

       IF overtime_ratio > 0.4:
           rule_violations += 1
       END IF

       IF salary outside role_band * 1.2:
           rule_violations += 1
       END IF

       statistical_score = rule_violations / 4
   END FOR

4. ENSEMBLE COMBINATION
   FOR each sample:
       ensemble_score = 0.6 * isolation_forest_score + 0.4 * statistical_score

       IF ensemble_score >= threshold:
           prediction = ANOMALY
       ELSE:
           prediction = NORMAL
       END IF
   END FOR

5. GENERATE EXPLANATIONS
   FOR each anomaly:
       explanation = []
       IF isolation_forest_score > 0.7:
           explanation.append("Unusual pattern detected by ML model")
       IF any statistical rule violated:
           explanation.append(violated_rule_description)
   END FOR

RETURN predictions, scores, explanations

END ALGORITHM
"""


def detect_anomalies(
    transactions: pd.DataFrame,
    features: pd.DataFrame,
    contamination: float = 0.02,
    threshold: float = 0.7
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Main entry point for anomaly detection.

    Args:
        transactions: Raw transaction data
        features: Engineered features
        contamination: Expected anomaly rate
        threshold: Decision threshold

    Returns:
        predictions: Binary anomaly predictions
        details: Detailed scoring breakdown
    """
    detector = EnsembleDetector(contamination=contamination)
    detector.fit(features, transactions)
    return detector.predict(features, transactions, threshold=threshold)
