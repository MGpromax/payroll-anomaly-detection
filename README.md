# Payroll Anomaly Detection

## What this project does

This is an anomaly detection system for payroll data. It detects two main types of fraud:
- Salary manipulation (when someone's pay is changed without authorization)
- Fake overtime (claiming OT hours that werent actually worked)

Since we dont have labeled fraud examples, I used unsupervised learning.

## How to run

```bash
pip install -r requirements.txt
python main.py --demo      # train and evaluate
python visualize.py        # generate charts
```

Or open `notebooks/analysis.ipynb` for a step-by-step walkthrough with explanations.

## Project files

- `main.py` - run this to start
- `visualize.py` - generates charts and analysis
- `src/anomaly_detector.py` - the main ML model (Isolation Forest)
- `src/feature_engineering.py` - creates features from raw payroll data
- `src/concept_drift.py` - handles when data patterns change over time
- `src/pipeline.py` - batch and realtime processing
- `src/evaluation.py` - how to evaluate without labels
- `src/data_generator.py` - creates fake payroll data for testing

## Why I chose Isolation Forest

I looked at several algorithms and picked Isolation Forest because:

1. It doesnt need labeled data (unsupervised)
2. Its fast - O(n) complexity
3. It specifically isolates anomalies instead of trying to model "normal" data
4. Works well with the mixed features we have (numerical + categorical)

I also tried LOF but it was too slow for our data size. K-means didnt make sense because payroll data doesnt form clear clusters.

## Features I created

For salary manipulation:
- How much someones salary deviates from their department average
- Percentage change from last month
- Whether salary is within the expected band for their role

For overtime fraud:
- OT hours compared to their own history
- OT hours compared to department average
- Ratio of OT pay to total pay (flag if too high)
- Whether OT exceeds their historical max

## Handling concept drift

Payroll data changes over time - people get raises, new employees join, etc. I implemented two methods:

1. Page-Hinkley test - detects sudden changes
2. ADWIN - adjusts the window size automatically when drift happens

When drift is detected, the system flags that model might need retraining.

## Evaluation (without labels)

This was tricky since we dont know which transactions are actually fraud. I used:

- Silhouette score to check if anomalies are well-separated from normal data
- Injecting synthetic anomalies and checking if theyre detected
- Checking if results are stable across multiple runs
- In production, would need domain experts to review flagged cases

## Results

On the synthetic data (with 2% injected anomalies):
- Precision is high (detected anomalies are real)
- Recall depends on threshold setting
- Runs in about 8 seconds for 4800 transactions

### Sample Output

```
Dataset: 4,800 transactions
Anomalies: 96 (2.0%)

Anomaly Types:
  - fake_overtime: 23
  - salary_manipulation: 19
  - duplicate_payment: 18
  - excessive_overtime: 18
  - salary_spike: 18

Top Suspicious Transactions:
  TXN00001026 | Operations | $22,444 | OT: 90h | Type: excessive_overtime
  TXN00002039 | Operations | $21,909 | OT: 0h  | Type: duplicate_payment
```

### Generated Charts

Running `python visualize.py` creates:
- `reports/anomaly_analysis.png` - distribution and breakdown
- `reports/time_series.png` - trends over time
- `reports/feature_analysis.png` - feature correlations

## Deployment

See DEPLOYMENT_PLAN.md for the full plan. Basic idea:
- Batch pipeline runs daily to retrain and score historical data
- Realtime pipeline scores new transactions as they come in
- Alerts go to investigators for review

## TODO / improvements

- Add more features (day of week patterns, approver analysis)
- Try autoencoder for comparison
- Better threshold tuning
- Add visualization dashboard
