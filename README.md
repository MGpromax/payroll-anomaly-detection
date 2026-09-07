# Payroll Anomaly Detection

A Python prototype for investigating unusual payroll transactions through unsupervised learning, interpretable statistical rules, and synthetic evaluation data.

The project explores how to turn raw salary and overtime records into reviewable signals: learn department and employee baselines, score transactions, explain rule violations, and inspect changes in feature distributions. Its primary workflow is a local batch demonstration.

**Status:** research and portfolio prototype. The supplied data is synthetic. An anomaly flag identifies a transaction for review; it does not establish fraud.

## At a glance

| Area | Implementation |
| --- | --- |
| Detection | scikit-learn Isolation Forest combined with salary and overtime rules |
| Features | 15 numerical features covering peer baselines, employee history, overtime, and time |
| Evaluation | Injected anomaly labels, confusion counts, precision, recall, and F1 |
| Drift analysis | Local Page-Hinkley and adaptive-window implementations |
| Outputs | Saved model components, JSON summary, and Matplotlib/Seaborn charts |
| Interfaces | Python CLI, batch pipeline, and experimental transaction-stream interface |

## Quick start

Run from the repository root with Python 3 and a virtual environment:

```bash
git clone https://github.com/MGpromax/payroll-anomaly-detection.git
cd payroll-anomaly-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONHASHSEED=0 python main.py --demo
MPLBACKEND=Agg python visualize.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell and set environment variables with `$env:PYTHONHASHSEED = "0"` and `$env:MPLBACKEND = "Agg"` before the Python commands.

The demo generates 4,800 transactions for 200 synthetic employees across 24 periods, including 96 injected anomalies. It then fits the ensemble, scores the same dataset, prints evaluation metrics, and writes artifacts under `data/`, `models/`, and `reports/`. Running it replaces generated files at those paths. `PYTHONHASHSEED` stabilizes the generator's hash-derived approver identifiers.

The demo and chart commands were smoke-checked with Python 3.9.6 and the dependency versions recorded in [evaluation notes](docs/EVALUATION.md). The requirements use version ranges; a clean install may resolve different versions.

## How it works

```text
Synthetic payroll records
        |
        v
Department, role, employee, and temporal features
        |
        +----> Isolation Forest score ----+
        |                                |
        +----> Statistical rule flags ---+--> Weighted score + threshold
                                                  |
                                                  v
                                       Flags, explanations, JSON report

Feature observations --> Drift monitors --> Review/retraining recommendation
Injected labels -------> Evaluation metrics
```

Isolation Forest learns unusual combinations of features without using the injected labels. Statistical rules add explanations for salary deviations, overtime volume, overtime compensation ratios, and role-band violations. Labels are used to inspect the resulting predictions.

## Commands and artifacts

| Command | Behavior |
| --- | --- |
| `python main.py --help` | Show CLI options |
| `python main.py --generate-data` | Regenerate the synthetic CSV files |
| `python main.py --train` | Train on `data/sample_payroll.csv` and save detector components |
| `python main.py --evaluate` | Train and evaluate on the CSV data; does not load the saved ensemble |
| `python main.py --demo` | Generate, train, evaluate, and display sampled batch results |
| `python main.py --stream` | Demonstrate individual-transaction scoring; see the streaming limitations below |
| `MPLBACKEND=Agg python visualize.py` | Save charts without an interactive display |

`reports/evaluation_report.json` contains aggregate counts, department summaries, and the highest-scoring transactions. Precision, recall, and F1 are printed by the CLI. The visualization script uses the **injected labels**, so its anomaly counts describe the dataset, not the model's detections.

## Reading the results

The default full demo uses a decision threshold of `0.7`. A documented smoke run flagged 7 of the 96 injected anomalies, with 7 true positives and 89 false negatives. This illustrates the precision/recall tradeoff at a conservative threshold, not validated performance on real payroll data.

The full demo evaluates its training data, and the score normalization depends on the batch being scored. A proper generalization study therefore needs a time-based holdout, training-only feature statistics, and a stable scoring contract. See [evaluation notes and known limitations](docs/EVALUATION.md) for the complete interpretation.

## Repository guide

```text
main.py                       CLI and demonstration workflow
visualize.py                  Charts based on synthetic labels
src/
  data_generator.py           Synthetic employees, transactions, and anomalies
  feature_engineering.py      Baseline and behavioral features
  anomaly_detector.py         Isolation Forest, rules, and ensemble
  concept_drift.py            Drift monitors and recommendations
  pipeline.py                 Batch and experimental streaming interfaces
  evaluation.py               Additional evaluation utilities
notebooks/analysis.ipynb      Exploratory walkthrough
config/config.yaml           Configuration sketch; not loaded by the CLI
data/                        Sample synthetic inputs and labels
models/                      Sample serialized detector components
reports/                     Example report and charts
docs/EVALUATION.md            Reproduction details and limitations
DEPLOYMENT_PLAN.md            Proposed path toward a reviewed deployment
```

## Development priorities

- Establish chronological evaluation with frozen preprocessing and threshold calibration.
- Make individual and batch scoring consistent, and persist feature-engineering state alongside the detector.
- Add regression tests for score bounds, rule aggregation, schema validation, and save/load behavior.
- Validate drift behavior on ordered streams and review alerts with payroll domain experts.

There is currently no automated test suite or deployed service in this repository. [The deployment plan](DEPLOYMENT_PLAN.md) describes proposed work and acceptance criteria.
