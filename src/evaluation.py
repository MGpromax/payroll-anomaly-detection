"""
Evaluation for anomaly detection without labels.

The hard part is we dont know whats actually fraud, so I use
proxy metrics like silhouette score and synthetic injection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    metric_name: str
    value: float
    interpretation: str
    confidence: str  # 'high', 'medium', 'low'


class UnsupervisedEvaluator:
    """
    Evaluates anomaly detection performance without ground truth labels.

    Key insight: We can't measure precision/recall directly, but we can:
    - Measure how "separated" anomalies are from normal data
    - Inject known anomalies and measure detection
    - Check if different runs produce consistent results
    - Have experts review samples for subjective validation
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.evaluation_history = []

    def evaluate_all(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        scores: np.ndarray,
        transactions: Optional[pd.DataFrame] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Run all applicable evaluation metrics.

        Args:
            features: Feature matrix
            predictions: Binary predictions (1 = anomaly)
            scores: Continuous anomaly scores
            transactions: Original transaction data (for business metrics)

        Returns:
            Dictionary of evaluation results
        """
        results = {}

        # 1. Cluster separation metrics
        results['silhouette'] = self._evaluate_silhouette(features, predictions)
        results['calinski_harabasz'] = self._evaluate_calinski_harabasz(features, predictions)

        # 2. Score distribution analysis
        results['score_separation'] = self._evaluate_score_separation(scores, predictions)
        results['contamination_ratio'] = self._evaluate_contamination_ratio(predictions)

        # 3. Agreement with LOF (ensemble validation)
        results['lof_agreement'] = self._evaluate_lof_agreement(features, predictions)

        # 4. Stability metrics
        results['score_stability'] = self._evaluate_score_stability(features)

        # 5. Business reasonability (if transaction data available)
        if transactions is not None:
            results['dept_distribution'] = self._evaluate_dept_distribution(
                transactions, predictions
            )
            results['amount_correlation'] = self._evaluate_amount_correlation(
                transactions, scores
            )

        return results

    def _evaluate_silhouette(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray
    ) -> EvaluationResult:
        """
        Silhouette Score: Measures how similar points are to their own cluster
        vs other clusters. Higher is better (-1 to 1).

        For anomaly detection: measures separation between normal and anomaly points.
        """
        try:
            # Need at least 2 clusters (normal + anomaly)
            if len(np.unique(predictions)) < 2:
                return EvaluationResult(
                    metric_name="Silhouette Score",
                    value=0.0,
                    interpretation="Cannot compute - only one class predicted",
                    confidence="low"
                )

            score = silhouette_score(features.fillna(0), predictions)

            if score > 0.5:
                interpretation = "Excellent separation between normal and anomaly clusters"
                confidence = "high"
            elif score > 0.25:
                interpretation = "Moderate separation - anomalies somewhat distinct"
                confidence = "medium"
            elif score > 0:
                interpretation = "Weak separation - some overlap between clusters"
                confidence = "low"
            else:
                interpretation = "Poor separation - clusters may be misassigned"
                confidence = "low"

            return EvaluationResult(
                metric_name="Silhouette Score",
                value=round(score, 4),
                interpretation=interpretation,
                confidence=confidence
            )

        except Exception as e:
            return EvaluationResult(
                metric_name="Silhouette Score",
                value=0.0,
                interpretation=f"Error computing: {str(e)}",
                confidence="low"
            )

    def _evaluate_calinski_harabasz(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray
    ) -> EvaluationResult:
        """
        Calinski-Harabasz Index: Ratio of between-cluster to within-cluster variance.
        Higher is better (no upper bound).
        """
        try:
            if len(np.unique(predictions)) < 2:
                return EvaluationResult(
                    metric_name="Calinski-Harabasz Index",
                    value=0.0,
                    interpretation="Cannot compute - only one class predicted",
                    confidence="low"
                )

            score = calinski_harabasz_score(features.fillna(0), predictions)

            # Interpretation depends on data size and dimensionality
            if score > 100:
                interpretation = "Strong cluster definition"
                confidence = "high"
            elif score > 50:
                interpretation = "Moderate cluster definition"
                confidence = "medium"
            else:
                interpretation = "Weak cluster definition"
                confidence = "low"

            return EvaluationResult(
                metric_name="Calinski-Harabasz Index",
                value=round(score, 2),
                interpretation=interpretation,
                confidence=confidence
            )

        except Exception as e:
            return EvaluationResult(
                metric_name="Calinski-Harabasz Index",
                value=0.0,
                interpretation=f"Error computing: {str(e)}",
                confidence="low"
            )

    def _evaluate_score_separation(
        self,
        scores: np.ndarray,
        predictions: np.ndarray
    ) -> EvaluationResult:
        """
        Measure how well anomaly scores separate normal from anomalous points.
        Uses Mann-Whitney U test for statistical significance.
        """
        normal_scores = scores[predictions == 0]
        anomaly_scores = scores[predictions == 1]

        if len(anomaly_scores) == 0:
            return EvaluationResult(
                metric_name="Score Separation",
                value=0.0,
                interpretation="No anomalies detected to compare",
                confidence="low"
            )

        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            normal_scores, anomaly_scores, alternative='less'
        )

        # Effect size (rank-biserial correlation)
        n1, n2 = len(normal_scores), len(anomaly_scores)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

        if p_value < 0.001 and effect_size > 0.5:
            interpretation = "Scores strongly discriminate anomalies (p < 0.001)"
            confidence = "high"
        elif p_value < 0.05:
            interpretation = "Scores moderately discriminate anomalies (p < 0.05)"
            confidence = "medium"
        else:
            interpretation = "Weak discrimination between normal and anomaly scores"
            confidence = "low"

        return EvaluationResult(
            metric_name="Score Separation (Effect Size)",
            value=round(effect_size, 4),
            interpretation=interpretation,
            confidence=confidence
        )

    def _evaluate_contamination_ratio(
        self,
        predictions: np.ndarray
    ) -> EvaluationResult:
        """
        Check if detected anomaly rate is reasonable.
        Typical payroll fraud rate: 0.5% - 5%
        """
        anomaly_rate = predictions.mean()

        if 0.005 <= anomaly_rate <= 0.05:
            interpretation = "Anomaly rate within expected range (0.5% - 5%)"
            confidence = "high"
        elif 0.001 <= anomaly_rate < 0.005:
            interpretation = "Lower than typical - may be missing anomalies"
            confidence = "medium"
        elif 0.05 < anomaly_rate <= 0.10:
            interpretation = "Higher than typical - may have false positives"
            confidence = "medium"
        else:
            interpretation = "Outside expected range - review model configuration"
            confidence = "low"

        return EvaluationResult(
            metric_name="Contamination Ratio",
            value=round(anomaly_rate, 4),
            interpretation=interpretation,
            confidence=confidence
        )

    def _evaluate_lof_agreement(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray
    ) -> EvaluationResult:
        """
        Compare predictions with LOF on sampled data for speed.
        """
        # Sample for speed - LOF is O(n^2)
        sample_size = min(1000, len(features))
        indices = np.random.choice(len(features), sample_size, replace=False)
        sample_features = features.iloc[indices].fillna(0)
        sample_predictions = predictions[indices]

        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02, n_jobs=-1)
        lof_predictions = lof.fit_predict(sample_features)
        lof_binary = (lof_predictions == -1).astype(int)

        agreement = (sample_predictions == lof_binary).mean()

        if agreement > 0.9:
            interpretation = "High agreement with LOF - robust detection"
            confidence = "high"
        elif agreement > 0.7:
            interpretation = "Moderate agreement with LOF"
            confidence = "medium"
        else:
            interpretation = "Low agreement - methods detecting different patterns"
            confidence = "low"

        return EvaluationResult(
            metric_name="LOF Agreement",
            value=round(agreement, 4),
            interpretation=interpretation,
            confidence=confidence
        )

    def _evaluate_score_stability(
        self,
        features: pd.DataFrame,
        n_runs: int = 2
    ) -> EvaluationResult:
        """
        Quick stability check using 2 runs on sampled data.
        """
        from .anomaly_detector import IsolationForestDetector

        # Sample for speed - use max 500 rows
        sample_size = min(500, len(features))
        sample_features = features.sample(n=sample_size, random_state=42)

        all_scores = []
        for seed in range(n_runs):
            detector = IsolationForestDetector(random_state=seed, n_estimators=50)
            detector.fit(sample_features)
            scores = detector.score_samples(sample_features)
            all_scores.append(scores)

        corr = np.corrcoef(all_scores[0], all_scores[1])[0, 1]

        if corr > 0.95:
            interpretation = "Highly stable scores"
            confidence = "high"
        elif corr > 0.85:
            interpretation = "Moderately stable scores"
            confidence = "medium"
        else:
            interpretation = "Unstable scores - results may vary"
            confidence = "low"

        return EvaluationResult(
            metric_name="Score Stability (Correlation)",
            value=round(corr, 4),
            interpretation=interpretation,
            confidence=confidence
        )

    def _evaluate_dept_distribution(
        self,
        transactions: pd.DataFrame,
        predictions: np.ndarray
    ) -> EvaluationResult:
        """
        Check if anomalies are reasonably distributed across departments.
        All anomalies from one department may indicate a problem.
        """
        transactions = transactions.copy()
        transactions['is_anomaly'] = predictions

        dept_anomaly_rates = transactions.groupby('department')['is_anomaly'].mean()
        dept_counts = transactions.groupby('department')['is_anomaly'].sum()

        # Check for concentration
        total_anomalies = predictions.sum()
        if total_anomalies == 0:
            return EvaluationResult(
                metric_name="Department Distribution",
                value=0.0,
                interpretation="No anomalies to analyze",
                confidence="low"
            )

        max_dept_share = dept_counts.max() / total_anomalies
        n_depts_with_anomalies = (dept_counts > 0).sum()

        if max_dept_share > 0.7:
            interpretation = f"Concentrated in one department ({max_dept_share:.0%})"
            confidence = "low"
        elif n_depts_with_anomalies >= 3:
            interpretation = "Distributed across multiple departments"
            confidence = "high"
        else:
            interpretation = "Moderate department distribution"
            confidence = "medium"

        return EvaluationResult(
            metric_name="Department Distribution (Max Share)",
            value=round(max_dept_share, 4),
            interpretation=interpretation,
            confidence=confidence
        )

    def _evaluate_amount_correlation(
        self,
        transactions: pd.DataFrame,
        scores: np.ndarray
    ) -> EvaluationResult:
        """
        Check if anomaly scores correlate reasonably with transaction amounts.
        Some correlation expected, but not too high (would just flag large salaries).
        """
        correlation = np.corrcoef(transactions['total_amount'], scores)[0, 1]

        if 0.1 <= abs(correlation) <= 0.5:
            interpretation = "Healthy correlation with amount (not just flagging high salaries)"
            confidence = "high"
        elif abs(correlation) < 0.1:
            interpretation = "Low correlation - may be missing amount-based anomalies"
            confidence = "medium"
        else:
            interpretation = "High correlation - may just be flagging high amounts"
            confidence = "low"

        return EvaluationResult(
            metric_name="Amount Correlation",
            value=round(correlation, 4),
            interpretation=interpretation,
            confidence=confidence
        )

    def synthetic_injection_test(
        self,
        features: pd.DataFrame,
        detector,
        n_synthetic: int = 100
    ) -> Dict[str, float]:
        """
        Inject known synthetic anomalies and measure detection rate.

        This is the closest we can get to measuring recall without labels.
        """
        # Create synthetic anomalies by perturbing normal data
        synthetic_anomalies = features.sample(n_synthetic, random_state=self.random_state).copy()

        # Apply various perturbations
        for col in synthetic_anomalies.columns:
            if synthetic_anomalies[col].dtype in ['float64', 'int64']:
                # Add 3-5 standard deviations of noise
                std = features[col].std()
                perturbation = np.random.uniform(3, 5, n_synthetic) * std
                synthetic_anomalies[col] += perturbation * np.random.choice([-1, 1], n_synthetic)

        # Score synthetic anomalies
        synthetic_scores = detector.score_samples(synthetic_anomalies)

        # Measure detection at various thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        detection_rates = {}

        for t in thresholds:
            detection_rate = (synthetic_scores >= t).mean()
            detection_rates[f"detection_rate_at_{t}"] = round(detection_rate, 4)

        return detection_rates

    def generate_report(
        self,
        results: Dict[str, EvaluationResult]
    ) -> str:
        """Generate human-readable evaluation report"""
        lines = [
            "=" * 60,
            "ANOMALY DETECTION EVALUATION REPORT",
            "=" * 60,
            ""
        ]

        for name, result in results.items():
            lines.extend([
                f"Metric: {result.metric_name}",
                f"Value: {result.value}",
                f"Interpretation: {result.interpretation}",
                f"Confidence: {result.confidence.upper()}",
                "-" * 40
            ])

        # Overall assessment
        high_conf_count = sum(1 for r in results.values() if r.confidence == 'high')
        total = len(results)

        lines.extend([
            "",
            "OVERALL ASSESSMENT",
            f"High confidence metrics: {high_conf_count}/{total}",
        ])

        if high_conf_count >= total * 0.7:
            lines.append("Recommendation: Model appears robust for deployment")
        elif high_conf_count >= total * 0.5:
            lines.append("Recommendation: Model acceptable, monitor closely")
        else:
            lines.append("Recommendation: Consider retraining or parameter tuning")

        return "\n".join(lines)


# ============================================================
# EVALUATION STRATEGY DOCUMENTATION
# ============================================================
"""
EVALUATION STRATEGY FOR UNSUPERVISED PAYROLL ANOMALY DETECTION

=== THE CHALLENGE ===

In unsupervised anomaly detection, we face a fundamental problem:
we don't have ground truth labels to measure traditional metrics
like precision, recall, and F1-score.

However, we can still evaluate our model using several strategies:


=== STRATEGY 1: INTERNAL VALIDATION METRICS ===

These metrics don't require labels - they measure properties of the
predictions themselves.

1. Silhouette Score [-1, 1]
   - Measures how similar points are to their own cluster vs. other clusters
   - Higher = better separation between normal and anomaly groups
   - Target: > 0.25

2. Calinski-Harabasz Index [0, inf]
   - Ratio of between-cluster to within-cluster variance
   - Higher = better defined clusters
   - No fixed target, compare across model versions

3. Score Separation
   - Statistical test (Mann-Whitney U) comparing scores of predicted
     normal vs. anomaly groups
   - Should show significant difference (p < 0.05)


=== STRATEGY 2: SYNTHETIC ANOMALY INJECTION ===

Since we don't know the real anomalies, we create fake ones:

1. Take random normal samples
2. Perturb them significantly (e.g., multiply salary by 1.5)
3. Score them with the model
4. Measure what percentage are detected

This gives us a proxy for recall:
- If model catches 90% of obvious fake anomalies, it's working
- If it misses obvious fakes, something is wrong


=== STRATEGY 3: ENSEMBLE AGREEMENT ===

Train multiple different algorithms and compare predictions:

1. Train Isolation Forest, LOF, and Statistical rules
2. Compare which transactions each flags
3. High agreement = more confidence in detection

Transactions flagged by all methods are highest confidence.
Transactions flagged by only one method need review.


=== STRATEGY 4: STABILITY ANALYSIS ===

Check if model produces consistent results:

1. Train model multiple times with different random seeds
2. Compare anomaly scores across runs
3. High correlation = stable model

Unstable models may be overfitting to noise.


=== STRATEGY 5: DOMAIN EXPERT VALIDATION ===

Ultimately, humans must validate the results:

1. Sample N flagged transactions (stratified by score)
2. Present to payroll experts for review
3. Track True Positive Rate (of those reviewed, how many are real issues)

This is the gold standard but expensive and slow.
Use it to calibrate thresholds and build confidence.


=== STRATEGY 6: BUSINESS OUTCOME TRACKING ===

In production, track long-term outcomes:

1. Recovered Fraud Amount: Money saved from catching fraud
2. False Positive Rate: Wasted investigation time
3. Investigation Outcomes: What happened after alerts

Over time, these metrics show real-world effectiveness.


=== RECOMMENDED EVALUATION WORKFLOW ===

BEFORE DEPLOYMENT:
1. Run internal validation metrics
2. Perform synthetic injection test (target: >80% detection)
3. Check stability analysis (target: correlation >0.9)
4. Have experts review top 100 flagged transactions

DURING PRODUCTION:
1. Monitor alert volume and distribution
2. Track investigation outcomes
3. Check for concept drift weekly
4. Re-evaluate quarterly with full workflow

RETRAINING TRIGGERS:
1. Drift detected by monitoring system
2. False positive rate exceeds threshold
3. Quarterly scheduled refresh
4. Significant business changes (new departments, policies)
"""


def evaluate_model(
    features: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
    transactions: Optional[pd.DataFrame] = None,
    print_report: bool = True
) -> Dict[str, EvaluationResult]:
    """
    Convenience function to run full evaluation.

    Args:
        features: Feature matrix
        predictions: Binary predictions
        scores: Continuous anomaly scores
        transactions: Original transaction data
        print_report: Whether to print human-readable report

    Returns:
        Dictionary of evaluation results
    """
    evaluator = UnsupervisedEvaluator()
    results = evaluator.evaluate_all(features, predictions, scores, transactions)

    if print_report:
        report = evaluator.generate_report(results)
        print(report)

    return results
