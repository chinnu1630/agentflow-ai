---
document_id: payment-service-runbook-v1
title: Payment Service Production Runbook
document_type: RUNBOOK
service: payment-service
owner: Payments Platform Team
environment: production
version: 1.0
status: active
last_reviewed: 2026-07-01
review_frequency: quarterly
tags:
  - payment-service
  - runbook
  - incident-response
  - rollback
  - atlaspay
  - postgresql
  - redis
  - deployment-blocker
  - release-risk
---

# Payment Service Production Runbook

## 1. Purpose

This runbook defines how engineers detect, triage, escalate, and resolve production incidents affecting `payment-service`. This runbook also defines the rollback criteria and rollback procedure that engineers must follow when `payment-service` is unhealthy in `production`. This runbook is the authoritative source for on-call decision-making for `payment-service` and is referenced during release-risk review by the Payments Platform Team and by AgentFlow AI's automated release-risk assessment.

## 2. Scope

This runbook applies to the `payment-service` repository component within `acme-backend-services`, deployed as a Docker container to AWS EC2 in both the `staging` and `production` environments. This runbook does not cover other services in `acme-backend-services` unless those services directly affect `payment-service` availability (for example, shared PostgreSQL infrastructure or shared Redis infrastructure).

## 3. Service overview

`payment-service` is a Python 3.11 / FastAPI backend service owned by the Payments Platform Team. `payment-service` is responsible for the following functions:

- Payment authorization
- Payment capture
- Refund processing
- Payment status updates
- Communication with the external payment provider `AtlasPay`
- Publishing payment events to the `payment-events` topic
- Storing payment transaction metadata in PostgreSQL

The critical user journey for `payment-service` is: **authorize payment → capture payment → publish payment event**. Any failure in this journey is treated as a payment-critical failure and is evaluated against the rollback decision criteria in Section 22.

## 4. Architecture and dependencies

`payment-service` has the following runtime dependencies:

| Dependency | Purpose | Failure impact |
|---|---|---|
| PostgreSQL (primary) | Stores payment transaction metadata, idempotency records, and audit history | Authorization and capture cannot be persisted; high risk of duplicate processing |
| Redis | Cache and idempotency store for in-flight payment requests | Idempotency keys cannot be verified; duplicate payment risk increases significantly |
| AtlasPay | External payment provider; performs actual authorization and capture with card networks | Payment authorization and capture cannot complete; `payment-service` cannot process new payments |
| `payment-events` topic | Downstream event bus consumed by ledger, notification, and reporting services | Downstream services do not receive timely payment state updates |
| GitHub Actions | CI/CD pipeline that builds and deploys the `payment-service` Docker image | New deployments cannot proceed; does not affect already-running production traffic |

`payment-service` is deployed as a Docker container on AWS EC2 instances behind a load balancer in both `staging` and `production`.

## 5. Ownership and escalation

`payment-service` is owned by the **Payments Platform Team**. Primary on-call responsibility belongs to the **Payments Platform On-Call** rotation, paged via PagerDuty.

| Escalation level | Contact | Trigger |
|---|---|---|
| Primary | Payments Platform On-Call (PagerDuty schedule: `payments-platform-primary`) | Any PagerDuty alert defined in Section 10 |
| Secondary | Payments Platform Engineering Manager | Primary on-call does not acknowledge PagerDuty alert within 5 minutes |
| Incident Commander | Assigned from the on-call Incident Commander rotation | Any SEV1 incident, or any SEV2 incident lasting longer than 30 minutes |
| Executive notification | VP of Engineering, Head of Payments | Any SEV1 incident lasting longer than 30 minutes, or any confirmed duplicate customer charge |

All incidents must be communicated in `#payments-operations`. Release coordination and deployment-related incidents must additionally be communicated in `#release-management`.

## 6. Service-level indicators

The following service-level indicators (SLIs) are tracked for `payment-service` in Grafana:

- Payment authorization success rate
- API error rate (5xx responses)
- p95 and p99 latency for the authorize endpoint
- p95 and p99 latency for the capture endpoint
- AtlasPay call timeout rate
- PostgreSQL connection-pool saturation
- Redis error rate
- Payment event publication lag
- Duplicate transaction detection rate

## 7. Normal operating thresholds

The following table defines normal, warning, and critical operating thresholds for `payment-service`. These thresholds are the basis for all alerting, release-hold, and rollback decisions referenced throughout this runbook.

| Metric | Normal | Warning (investigate) | Critical (rollback / escalate) |
|---|---|---|---|
| Payment authorization failure rate | < 0.5% | > 1% for 5 minutes | > 3% for 3 consecutive minutes |
| API error rate (5xx) | < 0.1% | > 1% for 5 minutes | > 5% for 5 minutes |
| p95 latency, authorize endpoint | < 400 ms | > 600 ms for 5 minutes | > 900 ms for 3 consecutive minutes |
| AtlasPay timeout rate | < 0.2% | > 2% for 10 minutes | > 8% for 10 minutes |
| PostgreSQL connection-pool saturation | < 70% | > 80% for 5 minutes | > 95% for 2 minutes |
| Redis error rate | < 0.1% | > 1% for 5 minutes | > 5% for 5 minutes |
| Payment event publication lag (p95) | < 10 s | > 30 s for 10 minutes | > 120 s or backlog > 5,000 messages |
| Duplicate transaction detection rate | 0% | > 0.1% detected duplicates | Any confirmed duplicate customer charge |

## 8. Production dashboards

The following Grafana dashboards must be reviewed during triage of any `payment-service` incident:

- `Payment Service - Overview` — authorization success rate, error rate, latency
- `Payment Service - AtlasPay Integration` — AtlasPay call latency, timeout rate, error codes
- `Payment Service - Database` — PostgreSQL connection-pool saturation, query latency, replication lag
- `Payment Service - Redis` — Redis error rate, cache hit rate, idempotency key conflicts
- `Payment Service - Event Publication` — publication lag, backlog depth, publish failure rate

## 9. Required logs and traces

Engineers must collect the following evidence during triage:

- Structured JSON application logs from the affected `payment-service` container, filtered by `trace_id` and `severity=ERROR`
- OpenTelemetry distributed traces for the affected time window, filtered by the `authorize` or `capture` span
- PostgreSQL slow-query log entries during the affected time window
- Redis `INFO` output and `SLOWLOG` output during the affected time window
- AtlasPay response codes and correlation IDs for failed authorization or capture attempts
- The deployed container image tag at the time of the incident

## 10. Common alerts

| Alert name | Trigger condition | Severity | Action |
|---|---|---|---|
| `PaymentAuthFailureRateHigh` | Payment authorization failure rate > 1% for 5 minutes | Warning | Investigate per Section 12 |
| `PaymentAuthFailureRateCritical` | Payment authorization failure rate > 3% for 3 minutes | Critical | Page on-call, follow Section 12, evaluate rollback per Section 22 |
| `PaymentServiceLatencyHigh` | p95 authorize latency > 600 ms for 5 minutes | Warning | Investigate per Section 13 |
| `PaymentServiceLatencyCritical` | p95 authorize latency > 900 ms for 3 minutes | Critical | Page on-call, follow Section 13, evaluate rollback per Section 22 |
| `AtlasPayTimeoutRateHigh` | AtlasPay timeout rate > 8% for 10 minutes | Critical | Page on-call, follow Section 14 |
| `PostgresConnectionPoolSaturated` | Connection-pool saturation > 95% for 2 minutes | Critical | Page on-call, follow Section 15 |
| `RedisErrorRateHigh` | Redis error rate > 5% for 5 minutes | Critical | Page on-call, follow Section 16, treat as duplicate-payment risk |
| `PaymentEventPublicationLagHigh` | Publication lag > 120 s or backlog > 5,000 messages | Critical | Page on-call, follow Section 17 |
| `DuplicateTransactionDetected` | Any confirmed duplicate customer charge | Critical (SEV1) | Page on-call and Incident Commander immediately, follow Section 18 |

## 11. Initial incident triage

When a `payment-service` alert fires, the on-call engineer must perform the following steps, in order:

1. Acknowledge the PagerDuty alert within 5 minutes.
2. Open the `Payment Service - Overview` Grafana dashboard and confirm whether the alert reflects a real degradation or a monitoring false positive.
3. Check service health directly:
   ```
   curl -s https://payment-service.internal.acme-commerce.example/healthz
   ```
4. Confirm the currently deployed container image:
   ```
   docker inspect --format='{{.Config.Image}}' payment-service
   ```
5. Post an initial status message in `#payments-operations`, including the alert name, affected metric, and current value.
6. Determine whether the failure correlates with a recent deployment by checking the GitHub Actions deployment history for `acme-backend-services`.
7. Proceed to the relevant failure-scenario section (Sections 12–19) based on the affected metric.

## 12. High payment failure rate

**Symptom:** `PaymentAuthFailureRateHigh` or `PaymentAuthFailureRateCritical` has fired.

**Evidence to collect:**

- Query recent payment failures:
  ```
  psql -h payment-db.internal -U payment_service_readonly -d payments \
    -c "SELECT status, failure_reason, count(*) FROM payment_transactions \
        WHERE created_at > now() - interval '15 minutes' \
        GROUP BY status, failure_reason ORDER BY count(*) DESC;"
  ```
- Determine whether failures are concentrated on a single failure reason (for example, `atlaspay_timeout`, `card_declined`, `idempotency_conflict`).
- Cross-reference the failure spike against the AtlasPay integration dashboard (Section 8) to rule out an external provider failure.

**Decision rule:** If the payment authorization failure rate exceeds 3% for 3 consecutive minutes and correlates with a recent deployment, this is a **deployment blocker** and the on-call engineer must proceed to Section 22 (rollback decision criteria). If the failure rate is driven by AtlasPay-side errors and not by `payment-service` itself, proceed to Section 14.

## 13. Increased payment latency

**Symptom:** `PaymentServiceLatencyHigh` or `PaymentServiceLatencyCritical` has fired.

**Evidence to collect:**

- Review OpenTelemetry traces for the `authorize` span during the affected window, identifying whether latency is concentrated in the `payment-service` application layer, the PostgreSQL query layer, or the AtlasPay call layer.
- Check PostgreSQL connection-pool saturation (Section 15) as a possible root cause.
- Check AtlasPay call latency on the `Payment Service - AtlasPay Integration` dashboard.

**Decision rule:** If p95 authorize latency exceeds 900 ms for 3 consecutive minutes, this is a critical incident. If the latency increase correlates with a recent deployment, this is a **deployment blocker** and the on-call engineer must proceed to Section 22.

## 14. AtlasPay provider failures

**Symptom:** `AtlasPayTimeoutRateHigh` has fired, or AtlasPay is returning elevated error rates.

**Evidence to collect:**

- Confirm the AtlasPay timeout rate on the `Payment Service - AtlasPay Integration` dashboard.
- Check the AtlasPay public status page for a known provider-side incident.
- Sample recent AtlasPay error responses and correlation IDs from the application logs.

**Decision rule:** AtlasPay provider failures are an **external dependency incident**, not a `payment-service` code defect. Rollback of `payment-service` does not resolve an AtlasPay-side outage and must not be performed solely on the basis of AtlasPay failures unless a recent `payment-service` deployment changed AtlasPay client configuration (timeout values, retry policy, endpoint URL). Declare an incident per Section 27 and notify `#payments-operations` with the AtlasPay status page link.

## 15. PostgreSQL connectivity failures

**Symptom:** `PostgresConnectionPoolSaturated` has fired, or `payment-service` logs show database connection errors.

**Evidence to collect:**

- Check connectivity directly:
  ```
  psql -h payment-db.internal -U payment_service_readonly -d payments -c "SELECT 1;"
  ```
- Check current connection-pool saturation on the `Payment Service - Database` dashboard.
- Check for long-running or blocking queries:
  ```
  psql -h payment-db.internal -U payment_service_readonly -d payments \
    -c "SELECT pid, now() - query_start AS duration, state, query \
        FROM pg_stat_activity WHERE state != 'idle' ORDER BY duration DESC LIMIT 20;"
  ```

**Decision rule:** If connection-pool saturation exceeds 95% for 2 minutes, this is a **deployment blocker**, and any in-flight deployment must be halted immediately. If saturation is caused by a recent connection-pool configuration change, proceed to Section 22 for rollback evaluation.

## 16. Redis connectivity or idempotency failures

**Symptom:** `RedisErrorRateHigh` has fired, or idempotency key conflicts are increasing.

**Evidence to collect:**

- Check Redis availability directly:
  ```
  redis-cli -h payment-cache.internal PING
  ```
- Check Redis error rate and slow log:
  ```
  redis-cli -h payment-cache.internal SLOWLOG GET 20
  ```
- Check idempotency key conflict rate on the `Payment Service - Redis` dashboard.

**Decision rule:** Redis unavailability directly increases duplicate-payment risk because `payment-service` cannot reliably verify idempotency keys before submitting a request to AtlasPay. If Redis error rate exceeds 5% for 5 minutes, this is a **critical incident** and must be treated as an elevated duplicate-payment risk per Section 18, regardless of whether a duplicate has been confirmed yet.

## 17. Payment event publishing failures

**Symptom:** `PaymentEventPublicationLagHigh` has fired.

**Evidence to collect:**

- Check publication backlog depth and publish failure rate on the `Payment Service - Event Publication` dashboard.
- Confirm whether downstream consumers (ledger, notification, reporting services) are experiencing delayed state updates.

**Decision rule:** Event publication lag does not block payment authorization or capture directly, but it does create downstream data-consistency risk. If backlog exceeds 5,000 messages or lag exceeds 120 seconds, declare a SEV2 incident per Section 27 and notify downstream service owners.

## 18. Duplicate payment risk

**Symptom:** `DuplicateTransactionDetected` has fired, or the duplicate transaction detection rate exceeds 0.1%.

**Evidence to collect:**

- Query for duplicate transactions:
  ```
  psql -h payment-db.internal -U payment_service_readonly -d payments \
    -c "SELECT idempotency_key, count(*) FROM payment_transactions \
        WHERE created_at > now() - interval '1 hour' \
        GROUP BY idempotency_key HAVING count(*) > 1;"
  ```
- Confirm whether any duplicate resulted in an actual duplicate charge with AtlasPay, versus a duplicate database row without a duplicate provider-side charge.

**Decision rule:** Any confirmed duplicate customer charge is automatically classified as **SEV1** per Section 27 and requires immediate Incident Commander and executive notification, regardless of overall service health. Elevated duplicate detection without a confirmed customer charge is classified as SEV2 and requires immediate investigation of Redis idempotency handling (Section 16).

## 19. Partial payment-processing failures

**Symptom:** Payments are authorized successfully but fail during capture, or captures succeed but event publication fails.

**Evidence to collect:**

- Identify the specific stage of the critical user journey (authorize → capture → publish) where failures are occurring, using OpenTelemetry traces.
- Determine whether failed captures leave transactions in an inconsistent state requiring manual reconciliation.

**Decision rule:** Partial failures affecting the critical user journey are treated with the same severity as full failures of that stage. A capture-stage failure rate exceeding the thresholds in Section 7 is a deployment blocker if it correlates with a recent deployment.

## 20. Deployment-related incidents

Any incident that begins within 30 minutes of a `payment-service` deployment to `production` must be treated as a **potential deployment-caused incident** until ruled out. The on-call engineer must:

1. Confirm the deployment timestamp and the deployed container image tag.
2. Compare the incident start time to the deployment timestamp.
3. If the incident correlates with the deployment, proceed directly to Section 22 (rollback decision criteria) without waiting for root-cause confirmation.

## 21. Database migration failures

**Symptom:** A database migration associated with a `payment-service` release fails to apply, or applies but causes downstream query errors.

**Evidence to collect:**

- Review the GitHub Actions migration job logs for the failed migration.
- Confirm whether the migration was applied partially.

**Decision rule:** A failed or partially applied database migration is an automatic **deployment blocker**. Deployment must stop immediately, and the migration must be rolled back using the migration tool's `downgrade` command before any further deployment activity resumes. Migrations must never be manually edited in production to work around a failure.

## 22. Rollback decision criteria

A rollback of `payment-service` in `production` is **required** when any of the following conditions is true:

- Payment authorization failure rate exceeds 3% for 3 consecutive minutes and correlates with a recent deployment.
- p95 authorize latency exceeds 900 ms for 3 consecutive minutes and correlates with a recent deployment.
- PostgreSQL connection-pool saturation exceeds 95% for 2 minutes following a deployment-related configuration change.
- A database migration associated with the release has failed (Section 21).
- Any confirmed duplicate customer charge has occurred following a recent deployment (Section 18).
- API error rate (5xx) exceeds 5% for 5 minutes and correlates with a recent deployment.

A rollback is **not** automatically required for AtlasPay-side provider failures (Section 14) unless the deployment altered AtlasPay client configuration.

## 23. Rollback procedure

1. Declare a deployment-related incident in `#release-management` and `#payments-operations`.
2. Identify the most recent known-good container image tag from the GitHub Actions deployment history.
3. Roll back the direct EC2 container deployment:
   ```
   docker pull acme-commerce/payment-service:<last-known-good-tag>
   docker stop payment-service
   docker rm payment-service
   docker run -d --name payment-service \
     acme-commerce/payment-service:<last-known-good-tag>
   ```
4. Confirm the rollback deployed successfully by re-checking the deployed image tag (Section 11, step 4).
5. Monitor the metrics in Section 7 for 15 minutes to confirm recovery.

## 24. Roll-forward procedure

A roll-forward (deploying a fix rather than reverting) is permitted only when all of the following are true:

- The root cause of the incident is fully understood and documented.
- A fix has passed CI and staging validation per the Release Readiness Checklist.
- The Payments Platform Engineering Manager has approved the roll-forward in place of a rollback.

If any of these conditions is not met, the on-call engineer must roll back per Section 23 rather than attempt a roll-forward.

## 25. Validation after rollback

After a rollback, the on-call engineer must confirm the following before closing the incident:

- Payment authorization failure rate has returned below 0.5%.
- p95 authorize latency has returned below 400 ms.
- No new duplicate transactions have been detected in the 15 minutes following rollback.
- PostgreSQL connection-pool saturation has returned below 70%.
- A rollback record has been created linking the incident, the reverted deployment, and the rollback container image tag.

## 26. Communication requirements

All `payment-service` production incidents must be communicated in `#payments-operations`. Deployment-related incidents must additionally be communicated in `#release-management`. A production Slack release notification must never be sent, and a deployment must never be marked as approved, until the authorized Payments Platform Engineering Manager has explicitly approved the release or the rollback outcome.

## 27. Incident severity classification

| Severity | Definition | Response time | Escalation |
|---|---|---|---|
| SEV1 | Full outage of the critical user journey, or any confirmed duplicate customer charge | 15 minutes | Incident Commander and executive notification |
| SEV2 | Major degradation (thresholds in Section 7 breached) without confirmed duplicate charge | 30 minutes | On-call plus Payments Platform Engineering Manager |
| SEV3 | Minor degradation with an available workaround | Next business day | Owning team backlog |
| SEV4 | Cosmetic or low-impact issue | Best effort | Backlog only |

## 28. Evidence collection

Every `payment-service` incident must have the following evidence attached to the incident record before closure:

- Grafana dashboard screenshots or exported panels for the affected time window.
- The relevant OpenTelemetry trace IDs.
- The PostgreSQL query results used during triage.
- The deployed container image tag before and after any rollback.
- A timeline of alert firing, acknowledgment, mitigation, and resolution timestamps.

## 29. Post-incident actions

Within 3 business days of any SEV1 or SEV2 incident, the owning team must:

- Publish an incident postmortem using the `INCIDENT_POSTMORTEM` document type.
- File corrective-action tickets in Jira for each identified root cause.
- Review whether the Section 7 thresholds require adjustment.

## 30. Known failure modes

The following failure modes have been observed in `payment-service` and should be considered during triage:

- AtlasPay intermittent timeout spikes during AtlasPay-side deployments, typically resolving within 30 minutes without `payment-service` changes.
- PostgreSQL connection-pool exhaustion following connection-pool size misconfiguration in deployment environment variables.
- Redis idempotency key conflicts following Redis failover events, which temporarily increase duplicate-detection alerts.
- Elevated event publication lag during downstream consumer maintenance windows.

## 31. Runbook review checklist

This runbook must be reviewed quarterly by the Payments Platform Team. Each review must confirm:

- All thresholds in Section 7 still reflect current production baselines.
- All dashboard links in Section 8 are valid.
- All example commands in this runbook execute successfully against `staging`.
- The escalation contacts in Section 5 are current.
- Any new known failure modes from the review period have been added to Section 30.
