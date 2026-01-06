"""
Processing pipelines - batch and realtime.

Batch: runs daily/weekly for training and bulk scoring
Realtime: scores individual transactions as they come in
"""

import numpy as np
import pandas as pd
from typing import Optional, Generator, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging
from abc import ABC, abstractmethod

from .feature_engineering import PayrollFeatureEngineer
from .anomaly_detector import EnsembleDetector
from .concept_drift import ConceptDriftMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    model_path: str = "models/ensemble"
    contamination: float = 0.02
    threshold: float = 0.35  # Lower threshold for better recall
    batch_size: int = 1000
    enable_drift_monitoring: bool = True
    alert_callback: Optional[Callable] = None


@dataclass
class ScoringResult:
    """Result of scoring a transaction"""
    transaction_id: str
    is_anomaly: bool
    anomaly_score: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    explanation: str
    timestamp: datetime


class BasePipeline(ABC):
    """Base class for processing pipelines"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_engineer = None
        self.detector = None
        self.drift_monitor = None
        self.is_initialized = False

    @abstractmethod
    def process(self, data):
        """Process data through the pipeline"""
        pass

    def _get_risk_level(self, score: float) -> str:
        """Convert score to risk level"""
        if score >= 0.9:
            return 'critical'
        elif score >= 0.7:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        return 'low'

    def _generate_explanation(self, details: Dict) -> str:
        """Generate human-readable explanation"""
        explanations = []

        if details.get('stat_rule_salary_zscore', False):
            explanations.append("Salary deviates significantly from department average")
        if details.get('stat_rule_ot_excessive', False):
            explanations.append("Overtime hours exceed normal limits")
        if details.get('stat_rule_ot_ratio', False):
            explanations.append("Overtime pay ratio unusually high")
        if details.get('stat_rule_salary_band', False):
            explanations.append("Salary outside expected range for role")

        if not explanations and details.get('isolation_forest_score', 0) > 0.7:
            explanations.append("Unusual pattern detected by ML model")

        return "; ".join(explanations) if explanations else "No specific rule triggered"


class BatchPipeline(BasePipeline):
    """
    Batch processing pipeline for historical analysis and model training.

    Runs on schedule (typically daily or weekly) to:
    - Train/retrain models on recent data
    - Score historical transactions for analysis
    - Generate reports for compliance review
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    def train(
        self,
        transactions: pd.DataFrame,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train the anomaly detection model on historical data.

        Args:
            transactions: Historical payroll transactions
            save_model: Whether to persist the trained model

        Returns:
            Training metrics and statistics
        """
        logger.info(f"Training on {len(transactions)} transactions...")

        # Initialize and fit feature engineer
        self.feature_engineer = PayrollFeatureEngineer()
        features = self.feature_engineer.fit_transform(transactions)

        # Initialize and fit detector
        self.detector = EnsembleDetector(contamination=self.config.contamination)
        self.detector.fit(features, transactions)

        # Initialize drift monitor
        if self.config.enable_drift_monitoring:
            self.drift_monitor = ConceptDriftMonitor(
                feature_names=self.feature_engineer.get_feature_names(),
                sensitivity='medium'
            )

        self.is_initialized = True

        # Calculate training statistics
        predictions, details = self.detector.predict(
            features, transactions, self.config.threshold
        )

        stats = {
            'training_samples': len(transactions),
            'feature_count': len(features.columns),
            'detected_anomalies': predictions.sum(),
            'anomaly_rate': predictions.mean(),
            'score_distribution': {
                'mean': details['ensemble_score'].mean(),
                'std': details['ensemble_score'].std(),
                'percentiles': {
                    '50': details['ensemble_score'].quantile(0.5),
                    '90': details['ensemble_score'].quantile(0.9),
                    '95': details['ensemble_score'].quantile(0.95),
                    '99': details['ensemble_score'].quantile(0.99)
                }
            },
            'trained_at': datetime.now().isoformat()
        }

        if save_model:
            self.detector.save(self.config.model_path)
            logger.info(f"Model saved to {self.config.model_path}")

        logger.info(f"Training complete. Detected {stats['detected_anomalies']} anomalies")

        return stats

    def process(
        self,
        transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Score batch of transactions.

        Returns DataFrame with original data plus anomaly scores and flags.
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call train() first.")

        logger.info(f"Processing batch of {len(transactions)} transactions...")

        # Extract features
        features = self.feature_engineer.transform(transactions)

        # Get predictions
        predictions, details = self.detector.predict(
            features, transactions, self.config.threshold
        )

        # Combine results
        result = transactions.copy()
        result['anomaly_score'] = details['ensemble_score']
        result['is_anomaly'] = details['is_anomaly']
        result['risk_level'] = details['ensemble_score'].apply(self._get_risk_level)

        # Generate explanations for anomalies
        result['explanation'] = ''
        for idx in result[result['is_anomaly']].index:
            row_details = details.loc[idx].to_dict()
            result.loc[idx, 'explanation'] = self._generate_explanation(row_details)

        # Update drift monitor (sample only for speed)
        if self.config.enable_drift_monitoring and self.drift_monitor:
            sample_size = min(100, len(features))
            sample_features = features.sample(n=sample_size, random_state=42)
            self.drift_monitor.batch_update(sample_features)
            logger.info("Drift monitoring updated")

        logger.info(f"Found {result['is_anomaly'].sum()} anomalies in batch")

        return result

    def generate_report(
        self,
        scored_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Dict:
        """Generate summary report of batch results"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_transactions': len(scored_data),
            'anomalies_detected': scored_data['is_anomaly'].sum(),
            'anomaly_rate': scored_data['is_anomaly'].mean(),
            'risk_breakdown': scored_data['risk_level'].value_counts().to_dict(),
            'by_department': scored_data.groupby('department').agg({
                'is_anomaly': 'sum',
                'anomaly_score': 'mean'
            }).to_dict(),
            'top_anomalies': scored_data.nlargest(10, 'anomaly_score')[[
                'transaction_id', 'employee_id', 'department',
                'total_amount', 'anomaly_score', 'explanation'
            ]].to_dict('records')
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_path}")

        return report


class RealTimePipeline(BasePipeline):
    """
    Real-time scoring pipeline for streaming transactions.

    Scores transactions as they arrive with low latency.
    Uses pre-trained model loaded from disk.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.buffer = []
        self.buffer_size = config.batch_size

    def load_model(self, model_path: Optional[str] = None):
        """Load pre-trained model for scoring"""
        path = model_path or self.config.model_path

        logger.info(f"Loading model from {path}...")

        self.detector = EnsembleDetector()
        self.detector.load(path)

        # Feature engineer needs to be trained on reference data
        # In production, this would be loaded from saved state
        self.feature_engineer = PayrollFeatureEngineer()

        if self.config.enable_drift_monitoring:
            self.drift_monitor = ConceptDriftMonitor(
                feature_names=self.feature_engineer.get_feature_names(),
                sensitivity='medium'
            )

        self.is_initialized = True
        logger.info("Model loaded successfully")

    def process(self, transaction: Dict) -> ScoringResult:
        """
        Score a single transaction in real-time.

        Args:
            transaction: Single payroll transaction as dict

        Returns:
            ScoringResult with anomaly score and risk level
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call load_model() first.")

        # Convert to DataFrame for processing
        df = pd.DataFrame([transaction])

        # Extract features
        features = self.feature_engineer.transform(df)

        # Get prediction
        predictions, details = self.detector.predict(
            features, df, self.config.threshold
        )

        # Create result
        row_details = details.iloc[0].to_dict()
        result = ScoringResult(
            transaction_id=transaction.get('transaction_id', 'unknown'),
            is_anomaly=bool(predictions[0]),
            anomaly_score=float(row_details['ensemble_score']),
            risk_level=self._get_risk_level(row_details['ensemble_score']),
            explanation=self._generate_explanation(row_details),
            timestamp=datetime.now()
        )

        # Trigger alert for high-risk anomalies
        if result.is_anomaly and result.risk_level in ['high', 'critical']:
            self._send_alert(result)

        # Update drift monitor
        if self.config.enable_drift_monitoring and self.drift_monitor:
            drift_result = self.drift_monitor.update(features.iloc[0])
            if drift_result.is_drift_detected:
                logger.warning(f"Drift detected: {drift_result.recommendation}")

        return result

    def process_stream(
        self,
        transaction_stream: Generator
    ) -> Generator[ScoringResult, None, None]:
        """
        Process continuous stream of transactions.

        Args:
            transaction_stream: Generator yielding transactions

        Yields:
            ScoringResult for each transaction
        """
        for transaction in transaction_stream:
            yield self.process(transaction)

    def _send_alert(self, result: ScoringResult):
        """Send alert for high-risk anomaly"""
        if self.config.alert_callback:
            self.config.alert_callback(result)
        else:
            logger.warning(
                f"ALERT: High-risk anomaly detected - "
                f"Transaction: {result.transaction_id}, "
                f"Score: {result.anomaly_score:.3f}, "
                f"Reason: {result.explanation}"
            )


# ============================================================
# PIPELINE PSEUDOCODE
# ============================================================
"""
PSEUDOCODE: Batch and Real-Time Pipelines

=== BATCH PIPELINE ===

ALGORITHM: BatchProcessingPipeline

SCHEDULE: Daily at 2:00 AM

INPUT:
    - date_range: Date range to process
    - model_path: Path to trained model

PROCEDURE:

1. DATA EXTRACTION
   transactions = query_payroll_database(
       start_date = date_range.start,
       end_date = date_range.end
   )

2. FEATURE ENGINEERING
   feature_engineer = load_or_create_feature_engineer()
   features = feature_engineer.transform(transactions)

3. MODEL SCORING
   model = load_model(model_path)
   scores = model.score_batch(features)

   FOR each transaction, score in zip(transactions, scores):
       IF score > THRESHOLD:
           flag_as_anomaly(transaction)
           generate_explanation(transaction, score)
       END IF
   END FOR

4. REPORTING
   report = {
       date: today,
       total_processed: count(transactions),
       anomalies_found: count(flagged),
       by_department: group_and_count(flagged, 'department'),
       high_risk_cases: filter(flagged, score > 0.9)
   }
   save_report(report)
   send_summary_email(report)

5. MODEL MAINTENANCE
   drift_score = check_concept_drift(features)

   IF drift_score > DRIFT_THRESHOLD:
       trigger_model_retraining()
   END IF

END ALGORITHM


=== REAL-TIME PIPELINE ===

ALGORITHM: RealTimeStreamingPipeline

INPUT:
    - transaction_stream: Kafka/event stream of transactions
    - model: Pre-loaded anomaly detection model

PROCEDURE:

1. INITIALIZATION
   model = load_model(MODEL_PATH)
   feature_cache = LRUCache(size=10000)  # Cache recent features
   drift_monitor = initialize_drift_monitor()

2. STREAM PROCESSING LOOP
   FOR each transaction in transaction_stream:

       # Feature extraction (with caching)
       employee_id = transaction.employee_id
       IF employee_id in feature_cache:
           base_features = feature_cache.get(employee_id)
       ELSE:
           base_features = query_employee_baseline(employee_id)
           feature_cache.put(employee_id, base_features)
       END IF

       features = compute_features(transaction, base_features)

       # Real-time scoring
       score = model.score(features)
       latency = measure_latency()

       IF latency > SLA_THRESHOLD:
           log_latency_warning(latency)
       END IF

       # Classification
       IF score > CRITICAL_THRESHOLD:
           risk_level = "CRITICAL"
           send_immediate_alert(transaction, score)
       ELSE IF score > HIGH_THRESHOLD:
           risk_level = "HIGH"
           queue_for_review(transaction, score)
       ELSE IF score > MEDIUM_THRESHOLD:
           risk_level = "MEDIUM"
           log_for_analysis(transaction, score)
       ELSE:
           risk_level = "LOW"
       END IF

       # Emit result
       emit_result({
           transaction_id: transaction.id,
           score: score,
           risk_level: risk_level,
           timestamp: now()
       })

       # Update drift monitor
       drift_result = drift_monitor.update(features)
       IF drift_result.should_retrain:
           notify_ml_team("Drift detected, retraining recommended")
       END IF

   END FOR

END ALGORITHM
"""


def create_batch_pipeline(config: Optional[PipelineConfig] = None) -> BatchPipeline:
    """Factory function to create batch pipeline"""
    config = config or PipelineConfig()
    return BatchPipeline(config)


def create_realtime_pipeline(config: Optional[PipelineConfig] = None) -> RealTimePipeline:
    """Factory function to create real-time pipeline"""
    config = config or PipelineConfig()
    return RealTimePipeline(config)
