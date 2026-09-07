# Evaluation and reproducibility

## Scope of the current experiment

The primary experiment is `python main.py --demo`. It generates synthetic records, fits the feature engineer and ensemble, and evaluates those same records. The generator injects five scenario labels: salary spike, salary manipulation, excessive overtime, fake overtime, and duplicate payment.

These are synthetic perturbations, not adjudicated fraud cases. For example, the duplicate-payment scenario doubles an amount rather than creating a second transaction record. Results measure recovery of these particular perturbations under the implemented scoring rules.

## Smoke-check record

The CLI help, full demo, and non-interactive chart generation were checked on macOS using Python 3.9.6. Generated artifacts were written to a separate working directory so that checked-in examples remained unchanged.

```bash
python main.py --help
PYTHONHASHSEED=0 python main.py --demo
MPLBACKEND=Agg python visualize.py
```

All three commands completed successfully. Matplotlib emitted a deprecation warning for the boxplot `labels` argument and expected warnings that `Agg` cannot open interactive windows; all three PNG files were created.

| Dependency | Version used |
| --- | --- |
| NumPy | 1.23.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| SciPy | 1.13.1 |
| Matplotlib | 3.9.4 |
| Seaborn | 0.13.2 |
| joblib | 1.5.3 |

This records one existing environment, not a lockfile or a clean-install compatibility guarantee. The notebook, additional evaluation utilities, and streaming command were not included in this smoke check.

### Observed full-demo results

| Measure | Value |
| --- | --- |
| Transactions / engineered features | 4,800 / 15 |
| Injected anomalies | 96 |
| Decision threshold | 0.7 |
| True positives / false positives | 7 / 0 |
| False negatives / true negatives | 89 / 4,704 |
| Precision / recall / F1 | 1.0000 / 0.0729 / 0.1359 |

These results are an in-sample smoke check. They are not a holdout benchmark, a production detection rate, or evidence of performance across different payroll distributions. The report's `top_anomalies` field is a ranking of ten records and can include records below the decision threshold.

## Interpretation and implementation limits

1. **Evaluation split.** The full demo trains and evaluates on the same records. Department, role, and employee history statistics therefore include the evaluation period and its injected anomalies. A temporal holdout is needed before making a generalization claim.
2. **Thresholds differ by entry point.** `train_model()` explicitly sets `0.7`. `PipelineConfig` defaults to `0.35`, which is used when `--evaluate` creates its own pipeline. Comparisons must record the entry point and threshold, not only the contamination parameter.
3. **Scores depend on the scoring batch.** Isolation Forest scores are min-max normalized using the current batch. For a single record, the normalization yields an Isolation Forest contribution of `1.0`. Batch and individual-transaction scores therefore do not have a common calibration.
4. **Preprocessing state is incomplete for serving.** The saved ensemble does not include fitted employee/department statistics. `RealTimePipeline.load_model()` creates a fresh, unfitted feature engineer. The stream demo works around this by reusing an in-memory fitted object; independent model loading is incomplete.
5. **Some features are batch-dependent.** Approver IDs are factorized per transform call, overtime percentiles use the current batch, and salary changes use records present in that call. Serving needs a stable encoding and historical feature contract.
6. **Rule aggregation needs a regression test.** `StatisticalDetector.detect()` includes the aggregate `any_rule_violation` flag in `violation_count`, which is later divided by four. This can produce a statistical score above one. Scores should be treated as implementation-specific ranking values, not calibrated probabilities.
7. **Drift is exploratory.** Batch processing updates monitors with a random sample rather than a chronological stream. The adaptive-window implementation is local, and no conformance or false-alarm study is supplied. Recommendations do not schedule retraining.
8. **Configuration and artifacts.** `config/config.yaml` is a design sketch and is not read by the CLI. The JSON writer uses `default=str`, so some NumPy-backed counts are serialized as strings. There is no automated schema test or dependency lockfile.

The repository has no automated test suite. Successful command execution establishes that the demonstration runs in the recorded environment; it does not resolve these modeling and serving limitations.

## Next evaluation design

- Split chronologically into training, calibration, and held-out evaluation periods; keep labels out of feature fitting and threshold selection on the holdout.
- Freeze preprocessing, categorical mappings, normalization, and thresholds before scoring the holdout.
- Measure precision/recall by anomaly type and department, alongside investigator workload and false-positive review costs.
- Compare statistical-only, Isolation-Forest-only, and ensemble baselines on identical records and splits.
- Record data seed, Python hash seed, dependency versions, feature schema, threshold, and model revision with each result.
- Add regression checks for individual-versus-batch consistency, score bounds, missing/unseen categories, index alignment, and artifact round trips.
