# Deployment Plan

## Overview

This doc covers how to deploy the anomaly detection system to production.

## Architecture

```
Payroll DB --> Kafka Queue --> Anomaly Detector --> Alert System --> Dashboard
                                    |
                              Model Registry
```

Theres two main flows:
1. Batch - runs every night, retrains model, scores all transactions from that day
2. Realtime - scores each transaction as it happens (for immediate alerts)

## Deployment Steps

### Phase 1: Dev/Testing (Week 1-2)
- Set up dev environment
- Test with synthetic data
- Make sure model metrics look good
- Write unit tests

### Phase 2: Integration (Week 3-4)
- Connect to actual payroll database (read-only)
- Run on 6 months of historical data
- Have HR/finance review some flagged transactions
- Tune thresholds based on feedback

### Phase 3: Shadow Mode (Week 5-6)
- Deploy to production but dont send any alerts
- Just log predictions
- Monitor performance and latency
- Compare against any known fraud cases

### Phase 4: Gradual Rollout (Week 7+)
- Start with only high-confidence alerts (score > 0.9)
- Gradually lower threshold as we build trust
- Set up on-call rotation for issues
- Monitor false positive rate

## Infrastructure

What we need:
- 2-4 servers for the API (8 CPU, 16GB RAM each)
- Kafka cluster for streaming
- PostgreSQL for storing results
- Grafana for monitoring dashboard

Estimated cost: ~$1500/month

## Monitoring

Key things to watch:
- Latency (should be < 100ms for realtime)
- Error rate
- Number of alerts per day (should be manageable)
- Drift score (retrain if it gets too high)

## Retraining

Model should be retrained when:
- Drift is detected
- False positive rate goes up
- New types of fraud emerge
- At least once a month anyway

## Rollback Plan

If something goes wrong:
1. Switch traffic to previous model version
2. Investigate issue
3. Fix and redeploy

Keep last 5 model versions in registry.

## Security

- Encrypt all data
- No PII in model features (just IDs)
- Audit logging for all access
- Role-based permissions

## Success Metrics

After 90 days we should see:
- System uptime > 99%
- At least a few real fraud cases caught
- False positive rate under control
- Investigators finding the alerts useful
