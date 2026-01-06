"""
Payroll Anomaly Detection Engine - Main Entry Point

Usage:
    python main.py --generate-data    Generate synthetic payroll data
    python main.py --train            Train the anomaly detection model
    python main.py --evaluate         Evaluate model performance
    python main.py --stream           Start real-time scoring (demo)
    python main.py --demo             Run full demo pipeline
"""

import argparse
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import PayrollDataGenerator
from src.feature_engineering import PayrollFeatureEngineer
from src.anomaly_detector import EnsembleDetector, detect_anomalies
from src.pipeline import BatchPipeline, RealTimePipeline, PipelineConfig
from src.evaluation import evaluate_model, UnsupervisedEvaluator
from src.concept_drift import ConceptDriftMonitor
import pandas as pd
import numpy as np


def ensure_directories():
    """Create necessary directories"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)


def generate_data():
    """Generate synthetic payroll data for testing"""
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC PAYROLL DATA")
    print("="*60 + "\n")

    generator = PayrollDataGenerator(seed=42)
    transactions, employees, ground_truth = generator.generate(
        n_employees=200,
        n_months=24,
        anomaly_rate=0.02
    )

    # Save to data directory
    transactions.to_csv('data/sample_payroll.csv', index=False)
    employees.to_csv('data/sample_employees.csv', index=False)
    ground_truth.to_csv('data/ground_truth.csv', index=False)

    print(f"\nData saved to data/ directory")
    print(f"  - Transactions: {len(transactions)} records")
    print(f"  - Employees: {len(employees)} records")
    print(f"  - Anomalies: {ground_truth['is_anomaly'].sum()} injected")

    return transactions, employees, ground_truth


def train_model(transactions: pd.DataFrame = None):
    """Train the anomaly detection model"""
    print("\n" + "="*60)
    print("TRAINING ANOMALY DETECTION MODEL")
    print("="*60 + "\n")

    if transactions is None:
        print("Loading data from data/sample_payroll.csv...")
        transactions = pd.read_csv('data/sample_payroll.csv')

    # Initialize pipeline
    config = PipelineConfig(
        model_path='models/ensemble',
        contamination=0.02,
        threshold=0.7
    )
    pipeline = BatchPipeline(config)

    # Train model
    stats = pipeline.train(transactions, save_model=True)

    print("\n" + "-"*40)
    print("TRAINING STATISTICS")
    print("-"*40)
    print(f"Training samples: {stats['training_samples']}")
    print(f"Features: {stats['feature_count']}")
    print(f"Anomalies detected: {stats['detected_anomalies']}")
    print(f"Anomaly rate: {stats['anomaly_rate']:.2%}")
    print(f"\nScore distribution:")
    for k, v in stats['score_distribution']['percentiles'].items():
        print(f"  {k}th percentile: {v:.4f}")

    return pipeline


def evaluate(
    transactions: pd.DataFrame = None,
    ground_truth: pd.DataFrame = None,
    pipeline: BatchPipeline = None
):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("EVALUATING MODEL PERFORMANCE")
    print("="*60 + "\n")

    if transactions is None:
        transactions = pd.read_csv('data/sample_payroll.csv')
    if ground_truth is None:
        ground_truth = pd.read_csv('data/ground_truth.csv')

    # Use existing pipeline or create new
    if pipeline is None:
        config = PipelineConfig(model_path='models/ensemble', enable_drift_monitoring=False)
        pipeline = BatchPipeline(config)
        pipeline.train(transactions, save_model=False)

    # Score all transactions
    results = pipeline.process(transactions)

    predictions = results['is_anomaly'].astype(int).values
    scores = results['anomaly_score'].values

    # Quick evaluation metrics (skip slow ones)
    print("EVALUATION METRICS")
    print("-"*40)
    print(f"Total transactions: {len(transactions)}")
    print(f"Anomalies detected: {predictions.sum()}")
    print(f"Detection rate: {predictions.mean():.2%}")
    print(f"Score mean: {scores.mean():.4f}")
    print(f"Score 95th percentile: {np.percentile(scores, 95):.4f}")

    # Supervised metrics with ground truth
    if ground_truth is not None:
        print("\n" + "-"*40)
        print("SUPERVISED METRICS (vs injected anomalies)")
        print("-"*40)

        true_labels = ground_truth['is_anomaly'].values
        tp = ((predictions == 1) & (true_labels == True)).sum()
        fp = ((predictions == 1) & (true_labels == False)).sum()
        fn = ((predictions == 0) & (true_labels == True)).sum()
        tn = ((predictions == 0) & (true_labels == False)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Negatives: {tn}")
        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    # Save report
    report = pipeline.generate_report(results, 'reports/evaluation_report.json')
    print(f"\nReport saved to reports/evaluation_report.json")

    return results, None


def demo_realtime():
    """Demonstrate real-time scoring"""
    print("\n" + "="*60)
    print("REAL-TIME SCORING DEMO")
    print("="*60 + "\n")

    # Load data
    transactions = pd.read_csv('data/sample_payroll.csv')

    # Train model first
    config = PipelineConfig(model_path='models/ensemble')
    batch_pipeline = BatchPipeline(config)
    batch_pipeline.train(transactions, save_model=True)

    # Initialize real-time pipeline
    rt_pipeline = RealTimePipeline(config)

    # For demo, we need to fit the feature engineer
    rt_pipeline.feature_engineer = batch_pipeline.feature_engineer
    rt_pipeline.detector = batch_pipeline.detector
    rt_pipeline.is_initialized = True

    # Simulate streaming by processing random transactions
    print("Simulating real-time transaction stream...")
    print("-"*40)

    sample_transactions = transactions.sample(10).to_dict('records')

    for i, txn in enumerate(sample_transactions):
        result = rt_pipeline.process(txn)
        status = "ANOMALY" if result.is_anomaly else "NORMAL"
        print(f"\nTransaction {i+1}:")
        print(f"  ID: {result.transaction_id}")
        print(f"  Status: {status}")
        print(f"  Score: {result.anomaly_score:.4f}")
        print(f"  Risk Level: {result.risk_level.upper()}")
        if result.is_anomaly:
            print(f"  Explanation: {result.explanation}")


def run_full_demo():
    """Run complete demo pipeline"""
    print("\n" + "="*60)
    print("PAYROLL ANOMALY DETECTION - FULL DEMO")
    print("="*60)
    print(f"Started at: {datetime.now()}")

    ensure_directories()

    # Step 1: Generate Data
    transactions, employees, ground_truth = generate_data()

    # Step 2: Train Model
    pipeline = train_model(transactions)

    # Step 3: Evaluate (reuse pipeline - no retraining)
    results, _ = evaluate(transactions, ground_truth, pipeline)

    # Step 4: Quick Real-time Demo (5 samples)
    print("\n" + "="*60)
    print("REAL-TIME SCORING DEMO")
    print("="*60)
    samples = transactions.sample(5).to_dict('records')
    for i, txn in enumerate(samples):
        score = results.loc[results['transaction_id'] == txn['transaction_id'], 'anomaly_score'].values
        is_anom = results.loc[results['transaction_id'] == txn['transaction_id'], 'is_anomaly'].values
        if len(score) > 0:
            status = "ANOMALY" if is_anom[0] else "NORMAL"
            print(f"Transaction {txn['transaction_id']}: {status} (score: {score[0]:.3f})")

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"Finished at: {datetime.now()}")
    print(f"\nOutput files:")
    print(f"  - data/sample_payroll.csv")
    print(f"  - models/ensemble_*.joblib")
    print(f"  - reports/evaluation_report.json")


def main():
    parser = argparse.ArgumentParser(
        description='Payroll Anomaly Detection Engine'
    )
    parser.add_argument(
        '--generate-data', action='store_true',
        help='Generate synthetic payroll data'
    )
    parser.add_argument(
        '--train', action='store_true',
        help='Train anomaly detection model'
    )
    parser.add_argument(
        '--evaluate', action='store_true',
        help='Evaluate model performance'
    )
    parser.add_argument(
        '--stream', action='store_true',
        help='Demo real-time scoring'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run full demo pipeline'
    )

    args = parser.parse_args()

    ensure_directories()

    if args.generate_data:
        generate_data()
    elif args.train:
        train_model()
    elif args.evaluate:
        evaluate()
    elif args.stream:
        demo_realtime()
    elif args.demo:
        run_full_demo()
    else:
        # Default: run full demo
        run_full_demo()


if __name__ == "__main__":
    main()
