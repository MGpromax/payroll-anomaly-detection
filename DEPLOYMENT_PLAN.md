# Deployment Plan

## System Overview

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Payroll   │ ──── │   Feature   │ ──── │   Anomaly   │
│   Database  │      │   Pipeline  │      │   Scorer    │
└─────────────┘      └─────────────┘      └─────────────┘
                                                 │
                            ┌────────────────────┼────────────────────┐
                            ▼                    ▼                    ▼
                     ┌───────────┐        ┌───────────┐        ┌───────────┐
                     │   Alert   │        │   Model   │        │  Dashboard │
                     │   Queue   │        │  Registry │        │  (Grafana) │
                     └───────────┘        └───────────┘        └───────────┘
```

## Deployment Phases

### Phase 1: Offline Validation
**Duration: 1-2 weeks**

- [ ] Run model on 6+ months historical data
- [ ] Validate with finance/HR team on sample flagged transactions
- [ ] Establish baseline metrics (expected alert volume, false positive rate)
- [ ] Document edge cases and known limitations

**Exit criteria:** Domain experts confirm >60% of flagged transactions warrant review

### Phase 2: Shadow Mode
**Duration: 2 weeks**

- [ ] Deploy model alongside existing process
- [ ] Score all transactions but don't trigger alerts
- [ ] Log predictions to separate table for analysis
- [ ] Compare against any known fraud cases from this period

```sql
-- shadow mode logging
CREATE TABLE anomaly_scores_shadow (
    transaction_id VARCHAR(20),
    score FLOAT,
    is_flagged BOOLEAN,
    model_version VARCHAR(10),
    scored_at TIMESTAMP
);
```

**Exit criteria:** System stable, latency <100ms p99, no data pipeline issues

### Phase 3: Limited Rollout
**Duration: 2-4 weeks**

- [ ] Enable alerts for high-confidence only (score > 0.85)
- [ ] Route to small group of investigators (2-3 people)
- [ ] Daily review of flagged vs investigated outcomes
- [ ] Adjust threshold based on feedback

**Exit criteria:** Investigators find alerts actionable, false positive rate <40%

### Phase 4: Full Production
**Duration: Ongoing**

- [ ] Lower threshold gradually (0.85 → 0.7 → 0.5)
- [ ] Expand to all investigators
- [ ] Set up automated retraining pipeline
- [ ] Monitor drift metrics

---

## Infrastructure

### Compute Requirements

| Component | Spec | Notes |
|-----------|------|-------|
| API Server | 4 vCPU, 8GB RAM | Handles real-time scoring |
| Batch Job | 8 vCPU, 16GB RAM | Nightly retraining + bulk scoring |
| Redis | 2GB | Feature caching for low latency |

### Data Flow

**Batch (nightly):**
```
1. Extract transactions from last 24h
2. Generate features
3. Score with current model
4. Write results to anomaly_scores table
5. High-risk scores → alert queue
```

**Real-time (per transaction):**
```
1. Transaction hits API
2. Fetch employee baseline from cache
3. Compute features
4. Score (target: <50ms)
5. Return score + risk level
6. If high risk → async alert
```

---

## Model Management

### Versioning
```
models/
├── v1.0.0/
│   ├── isolation_forest.joblib
│   ├── feature_config.yaml
│   └── metrics.json
├── v1.0.1/
│   └── ...
└── current -> v1.0.1/
```

### Retraining Triggers

1. **Scheduled:** Weekly, Sunday 2 AM
2. **Drift-based:** When drift score > 0.3 for 3 consecutive days
3. **Performance-based:** When false positive rate exceeds 50%
4. **Manual:** After significant business changes

### Rollback Process

```bash
# if new model performs worse
./scripts/rollback.sh --to-version v1.0.0

# this does:
# 1. update current symlink
# 2. restart scoring service
# 3. alert on-call
```

---

## Monitoring

### Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Scoring latency p99 | <100ms | >500ms |
| Daily alert volume | 10-50 | >100 or <2 |
| Model drift score | <0.3 | >0.5 |
| API error rate | <0.1% | >1% |

### Dashboards

1. **Operational:** Latency, throughput, errors
2. **Model Health:** Score distribution, drift metrics, feature stats
3. **Business:** Alerts by department, investigation outcomes, fraud recovered

### Alerting

```yaml
# PagerDuty/Opsgenie rules
critical:
  - api_error_rate > 5%
  - scoring_service_down

warning:
  - drift_score > 0.4
  - latency_p99 > 200ms
  - zero_alerts_24h  # model might be broken
```

---

## Security

- All payroll data encrypted at rest (AES-256)
- TLS 1.3 for data in transit
- Model artifacts in private S3 bucket
- API authentication via internal service mesh
- Audit logs for all score queries
- No PII stored in model features (only employee_id)

---

## Runbook

### High Alert Volume

```
1. Check if threshold was accidentally lowered
2. Look for data quality issues (bulk imports, system migrations)
3. Check if model was recently retrained
4. If legitimate spike, notify investigators + temporarily raise threshold
```

### Model Serving Errors

```
1. Check model file integrity (checksum)
2. Verify feature pipeline is producing expected columns
3. Check for schema changes in source data
4. Rollback to previous model version if needed
```

### Drift Detected

```
1. Don't panic - some drift is normal
2. Check which features are drifting (salary bands? overtime patterns?)
3. If business-driven (new policy, annual raises), retrain
4. If data quality issue, fix upstream first
```

---

## Success Metrics (90-day review)

| Metric | Target |
|--------|--------|
| System uptime | >99.5% |
| Fraud cases identified | >5 |
| Investigation time saved | measurable reduction |
| False positive rate | <40% |
| Stakeholder satisfaction | positive feedback from investigators |

---

## Ownership

| Area | Owner |
|------|-------|
| Model performance | ML team |
| Data pipeline | Data engineering |
| Infrastructure | Platform/DevOps |
| Alert review process | Finance/HR ops |
| Escalation | Security team |
