# Deployment roadmap

This is a proposed roadmap for moving the local prototype toward a reviewed payroll decision-support service. The repository currently provides CLI and Python pipeline interfaces. It does not implement an API server, scheduled jobs, a message broker integration, authentication, or production infrastructure.

## 1. Establish a reliable scoring contract

- Resolve the [documented scoring and preprocessing limitations](docs/EVALUATION.md).
- Persist the feature engineer, categorical mappings, score calibration, threshold, and detector as one versioned artifact.
- Validate input schemas and define behavior for missing values, new employees, unknown departments, and malformed dates.
- Add regression tests for batch/individual consistency, bounded rule aggregation, and save/load equivalence.

**Acceptance evidence:** passing regression tests, versioned artifact metadata, and reproducible predictions from a fresh process.

## 2. Evaluate on held-out data

- Obtain an authorized dataset and agree on handling and retention requirements.
- Use chronological splits and fit every preprocessing step on training data only.
- Compare model-only, rule-only, and ensemble baselines.
- Calibrate thresholds against the review capacity and error costs agreed with payroll specialists.
- Report metrics by anomaly type and relevant operational groups.

**Acceptance evidence:** a documented holdout study, error analysis, and an agreed review workflow. Numerical acceptance thresholds should follow that study rather than be treated as existing service guarantees.

## 3. Run in shadow mode

- Score an approved feed without changing payment decisions.
- Log model version, input schema version, threshold, scores, and explanations.
- Have authorized reviewers assess a sample of flagged and unflagged records.
- Track latency, failures, score distributions, alert volume, and investigator outcomes.

**Acceptance evidence:** stable operation over an agreed observation period and evidence that alerts are actionable. Measure latency on the actual workload before setting a service objective.

## 4. Introduce a controlled review service

- Add authenticated access and restrict payroll records and model artifacts to authorized roles.
- Define encryption, audit logging, retention, backup, and deletion controls for the chosen infrastructure.
- Keep human review between model flags and decisions affecting compensation.
- Add a versioned release process with rollback to a previously validated model and feature pipeline.
- Retrain only after investigating drift, data-quality issues, and changes in payroll policy.

**Acceptance evidence:** reviewed access controls, operational runbooks, a tested rollback procedure, and clear ownership of model behavior, data handling, and payroll review.

## Operational questions to resolve

| Area | Decision needed |
| --- | --- |
| Data | Authorized sources, schema, retention, and data-quality checks |
| Modeling | Holdout design, review thresholds, calibration, and drift validation |
| Serving | Batch cadence, throughput, latency, and artifact storage |
| Operations | Monitoring, incident response, model rollback, and retraining approval |
| Review | Responsible reviewers, escalation criteria, and outcome feedback |

No infrastructure sizing, service-level objective, fraud-recovery figure, or security control in this roadmap is a measured or deployed property of the current prototype.
