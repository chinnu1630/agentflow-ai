---
document_id: payment-service-release-readiness-checklist-v1
title: Payment Service Release Readiness Checklist
document_type: CHECKLIST
service: payment-service
owner: Payments Platform Team
environment: production
version: 1.0
status: active
last_reviewed: 2026-07-01
review_frequency: quarterly
tags:
  - payment-service
  - checklist
  - release-readiness
  - deployment-blocker
  - approval
  - rollback
  - release-risk
---

# Payment Service Release Readiness Checklist

## 1. Purpose

This checklist defines the required conditions for approving a `payment-service` release to `production`. This checklist is the authoritative source for release-blocking conditions, conditional-approval conditions, and the go/no-go decision for `payment-service` deployments, and is referenced by AgentFlow AI's automated release-risk assessment.

## 2. Scope

This checklist applies to every release of `payment-service` within `acme-backend-services` targeting the `production` environment. Releases targeting only `staging` require the Staging Validation section (Section 22) but do not require the full go/no-go decision defined in Section 36.

## 3. Roles and responsibilities

| Role | Responsibility |
|---|---|
| Release engineer | Executes the checklist and prepares the release for approval |
| Code reviewer | Confirms code-review requirements (Section 6) |
| Payments Platform Engineering Manager | Provides final human approval for the release (Section 32) |
| Release Manager | Coordinates the deployment window and confirms no active change freeze applies |
| On-call engineer | Confirms rollback-plan validity (Section 21) and monitors post-deployment (Section 37) |

## 4. Required release artifacts

Every `payment-service` release must include the following artifacts before review begins:

- A merged pull request into the `main` branch of `acme-backend-services`.
- A GitHub Actions CI run associated with the merge commit, with a passing status.
- A release notes entry describing the change, the affected components, and the rollback plan.
- A linked Jira release ticket referencing all included pull requests.

## 5. Pull-request readiness

**Check:** All pull requests included in the release have been merged into `main` with no unresolved review comments.
**Owner:** Release engineer.
**Required evidence:** Link to each merged pull request in the release Jira ticket.
**Pass criteria:** Every pull request shows status `merged` with all review conversations resolved.
**Failure action:** Remove the unresolved pull request from the release scope, or resolve the conversation before proceeding.

## 6. Code-review requirements

**Check:** Every pull request has at least one approving review from an engineer other than the author.
**Owner:** Code reviewer.
**Required evidence:** GitHub pull-request approval record.
**Pass criteria:** At least one approval is present and no reviewer has requested changes that remain unaddressed.
**Failure action:** Deployment blocker. The release cannot proceed until an approval is obtained.

## 7. CI requirements

**Check:** All required GitHub Actions checks pass on the release commit, including unit tests, integration tests, lint, and build.
**Owner:** Release engineer.
**Required evidence:** Link to the passing GitHub Actions run.
**Pass criteria:** All required checks report `success`.
**Failure action:** **Mandatory release blocker.** Failing required CI checks must never be bypassed. The release cannot proceed until all required checks pass.

## 8. Test requirements

**Check:** Unit test coverage for changed files meets or exceeds 80%, and all integration tests covering the authorize, capture, and refund flows pass.
**Owner:** Code reviewer.
**Required evidence:** Coverage report from the CI run.
**Pass criteria:** Coverage threshold met and all payment-flow integration tests pass.
**Failure action:** Deployment blocker if coverage falls below 80% on payment-critical files (authorization, capture, idempotency handling).

## 9. Security checks

**Check:** Static analysis security scanning and dependency vulnerability scanning have completed on the release commit with no unresolved critical findings.
**Owner:** Release engineer.
**Required evidence:** Security scan report from the CI run.
**Pass criteria:** Zero unresolved critical or high-severity findings.
**Failure action:** **Mandatory release blocker** for any critical security finding. High-severity findings require Payments Platform Engineering Manager sign-off to proceed.

## 10. Dependency checks

**Check:** All new or updated third-party dependencies have been reviewed for license compatibility and known vulnerabilities.
**Owner:** Code reviewer.
**Required evidence:** Dependency scan output attached to the pull request.
**Pass criteria:** No dependency introduces a known critical vulnerability without a documented mitigation.
**Failure action:** Deployment blocker until the dependency is updated, replaced, or the vulnerability is confirmed non-exploitable in this context and documented.

## 11. Database migration checks

**Check:** Any database migration included in the release has been reviewed for backward compatibility and tested in `staging`.
**Owner:** Release engineer.
**Required evidence:** Migration file, staging migration run log, and rollback (`downgrade`) script.
**Pass criteria:** Migration applies cleanly in `staging`, includes a working `downgrade` script, and does not lock payment-critical tables for longer than 5 seconds.
**Failure action:** **Mandatory release blocker.** Unapproved or untested database migrations must never be deployed to `production`.

## 12. Backward-compatibility checks

**Check:** The release does not remove or change the meaning of any existing database column, API field, or event schema field still in use by another service.
**Owner:** Code reviewer.
**Required evidence:** Schema diff and a list of consuming services checked against the change.
**Pass criteria:** No breaking change to shared schema, or a documented deprecation window has been communicated to all consumers.
**Failure action:** Deployment blocker for any undocumented breaking change.

## 13. API compatibility checks

**Check:** Changes to `payment-service` REST endpoints preserve backward compatibility for existing API consumers.
**Owner:** Code reviewer.
**Required evidence:** API diff report comparing the release branch to the currently deployed `production` API contract.
**Pass criteria:** No required field removed, no response field type changed, no endpoint removed without a documented deprecation period.
**Failure action:** **Mandatory release blocker** for incompatible API changes without a coordinated consumer migration plan.

## 14. Feature-flag checks

**Check:** Any new payment-processing behavior is gated behind a feature flag defaulting to disabled in `production`.
**Owner:** Release engineer.
**Required evidence:** Feature-flag configuration reference in the release notes.
**Pass criteria:** New behavior affecting the authorize, capture, or refund flow is flag-gated and the flag defaults to off.
**Failure action:** Deployment blocker if payment-critical behavior is not flag-gated.

## 15. Configuration validation

**Check:** All required environment variables and configuration values for `production` are present and validated against the `staging` configuration.
**Owner:** Release engineer.
**Required evidence:** Configuration diff between `staging` and `production`.
**Pass criteria:** No missing required configuration value; any new configuration value has a documented default.
**Failure action:** Deployment blocker for any missing required configuration value.

## 16. Secrets validation

**Check:** No secrets, API keys, or credentials are present in the release commit, configuration files, or container image.
**Owner:** Release engineer.
**Required evidence:** Secret-scanning report from CI.
**Pass criteria:** Zero secrets detected in the release artifacts.
**Failure action:** **Mandatory release blocker.** Any detected secret must be rotated and removed before the release proceeds.

## 17. Observability validation

**Check:** New code paths emit structured logs and OpenTelemetry spans consistent with existing `payment-service` observability standards.
**Owner:** Code reviewer.
**Required evidence:** Sample log output and trace from a `staging` test run.
**Pass criteria:** New authorize, capture, or refund code paths are traced and logged with `trace_id`.
**Failure action:** Deployment blocker if payment-critical code paths lack tracing or structured logging.

## 18. Dashboard readiness

**Check:** Grafana dashboards referenced in the Payment Service Production Runbook reflect any new metrics introduced by this release.
**Owner:** Release engineer.
**Required evidence:** Updated dashboard link or confirmation that no dashboard changes are required.
**Pass criteria:** All new metrics relevant to payment success rate, latency, or error rate are visible on the `Payment Service - Overview` dashboard before deployment.
**Failure action:** Deployment blocker if a new payment-critical metric is not visible on a dashboard.

## 19. Alert readiness

**Check:** PagerDuty alerts exist for any new failure mode introduced by this release, consistent with the thresholds in the Payment Service Production Runbook.
**Owner:** Release engineer.
**Required evidence:** Alert configuration reference.
**Pass criteria:** New payment-critical failure modes have a corresponding alert with a defined threshold.
**Failure action:** Deployment blocker if a new payment-critical failure mode has no corresponding alert.

## 20. Runbook readiness

**Check:** The Payment Service Production Runbook has been updated to reflect any new failure modes, thresholds, or rollback considerations introduced by this release.
**Owner:** Release engineer.
**Required evidence:** Link to the runbook update pull request, or confirmation that no runbook change is required.
**Pass criteria:** Runbook accurately reflects the post-release system behavior.
**Failure action:** Conditional approval only; runbook update must be completed within 2 business days of deployment.

## 21. Rollback-plan validation

**Check:** A rollback plan is documented, including the last known-good container image tag and any required migration downgrade steps.
**Owner:** On-call engineer.
**Required evidence:** Rollback plan section in the release notes.
**Pass criteria:** Rollback plan references a specific, verified known-good image tag and, if applicable, a tested migration downgrade script.
**Failure action:** **Mandatory release blocker.** A release must never proceed without a documented and validated rollback plan.

## 22. Staging validation

**Check:** The release has been deployed to `staging` and has run without payment-flow errors for at least 2 hours or 500 test transactions, whichever occurs first.
**Owner:** Release engineer.
**Required evidence:** Staging deployment timestamp and staging payment-flow test results.
**Pass criteria:** Zero authorization or capture failures attributable to the release during the staging soak period.
**Failure action:** **Mandatory release blocker.** Failed staging payment-flow validation must never be bypassed for a `production` deployment.

## 23. Performance validation

**Check:** p95 authorize latency in `staging` under load testing does not exceed the `production` warning threshold of 600 ms defined in the Payment Service Production Runbook.
**Owner:** Release engineer.
**Required evidence:** Load test report from `staging`.
**Pass criteria:** p95 latency remains below 600 ms under representative load.
**Failure action:** Deployment blocker if p95 latency exceeds 600 ms in staging load testing.

## 24. Payment-flow validation

**Check:** The full critical user journey (authorize → capture → publish payment event) has been manually or automatically verified in `staging`.
**Owner:** Release engineer.
**Required evidence:** Staging test transaction IDs and corresponding event-publication confirmation.
**Pass criteria:** All three stages of the critical user journey complete successfully in `staging`.
**Failure action:** **Mandatory release blocker.**

## 25. AtlasPay integration validation

**Check:** The release has been validated against the AtlasPay `staging` sandbox environment, including at least one successful authorization, one successful capture, and one successful refund.
**Owner:** Release engineer.
**Required evidence:** AtlasPay sandbox transaction references.
**Pass criteria:** All three AtlasPay sandbox transaction types succeed without unexpected error codes.
**Failure action:** Deployment blocker if any AtlasPay sandbox transaction type fails.

## 26. Redis and idempotency validation

**Check:** Idempotency key handling has been tested in `staging`, including a duplicate-request scenario that must not result in a duplicate AtlasPay charge.
**Owner:** Release engineer.
**Required evidence:** Staging test log showing a duplicate request correctly deduplicated via Redis.
**Pass criteria:** Duplicate request test in staging results in exactly one AtlasPay charge.
**Failure action:** **Mandatory release blocker.** Unresolved payment duplication risk must never be deployed to `production`.

## 27. PostgreSQL validation

**Check:** Any schema changes have been validated for query-plan impact on payment-critical queries in `staging`.
**Owner:** Release engineer.
**Required evidence:** `EXPLAIN ANALYZE` output for payment-critical queries before and after the schema change.
**Pass criteria:** No payment-critical query regresses beyond the p95 latency thresholds defined in the runbook.
**Failure action:** Deployment blocker for any confirmed query-plan regression on payment-critical queries.

## 28. Event-publication validation

**Check:** Payment events published by the release have been validated against the current `payment-events` schema consumed by downstream services.
**Owner:** Release engineer.
**Required evidence:** Sample published event payload from `staging`.
**Pass criteria:** Event payload matches the schema expected by downstream consumers with no breaking field changes.
**Failure action:** Deployment blocker for any breaking change to the published event schema without a coordinated consumer migration.

## 29. Open Jira blocker review

**Check:** No open Jira ticket labeled `release-blocker` or with priority `P1` exists for `payment-service` at the time of release review.
**Owner:** Release engineer.
**Required evidence:** Jira query result for open `payment-service` tickets filtered by `release-blocker` label or `P1` priority.
**Pass criteria:** Zero matching open tickets.
**Failure action:** **Mandatory release blocker.** Any open `release-blocker` or P1 Jira ticket for `payment-service` prevents release approval until resolved or explicitly waived by the Payments Platform Engineering Manager.

## 30. Open GitHub pull-request review

**Check:** No other open pull request targeting `payment-service` modifies the same files as the release under review, in a way that could conflict post-deployment.
**Owner:** Code reviewer.
**Required evidence:** List of open pull requests touching `payment-service` files.
**Pass criteria:** No unresolved file-level conflict with an open pull request intended for a near-term separate release.
**Failure action:** Conditional approval; coordinate merge order with the other pull request's author before deployment.

## 31. Incident-history review

**Check:** Review open and recently resolved `payment-service` incidents (previous 14 days) for any unresolved root cause relevant to this release.
**Owner:** Release engineer.
**Required evidence:** List of incidents from the incident tracker for the previous 14 days.
**Pass criteria:** No unresolved root cause from a recent incident overlaps with the code paths changed in this release.
**Failure action:** Conditional approval requiring explicit acknowledgment from the Payments Platform Engineering Manager if overlap exists.

## 32. Human approval requirements

Every `payment-service` production release requires explicit human approval from the **Payments Platform Engineering Manager** before deployment. High-risk releases additionally require approval from the **Release Manager**. A release is classified as high-risk if it includes any of the following:

- A database migration
- A change to AtlasPay integration configuration
- A change to idempotency or Redis key handling
- A change to the payment event schema

No production Slack release notification is sent, and no deployment is finalized, until the authorized Payments Platform Engineering Manager has explicitly approved the release.

## 33. Release-blocking conditions

The following conditions are **mandatory release blockers** and must never be waived:

- Failing required CI checks (Section 7)
- Any open P1 defect or `release-blocker` Jira ticket for `payment-service` (Section 29)
- An unapproved or untested database migration (Section 11)
- A missing or unvalidated rollback plan (Section 21)
- Unresolved payment duplication risk (Section 26)
- Missing production monitoring for a new payment-critical failure mode (Section 19)
- Any unresolved critical security finding (Section 9)
- An incompatible API change without a coordinated consumer migration plan (Section 13)
- Failed staging payment-flow validation (Section 22, Section 24)

## 34. Conditional approval conditions

A release may proceed under **conditional approval** only when the outstanding item is one of the following, and the Payments Platform Engineering Manager has explicitly documented the condition and its required follow-up timeline:

- Runbook documentation update pending (Section 20), with a required completion date within 2 business days
- A non-conflicting open pull request exists (Section 30), with merge-order coordination documented
- A recent incident's root cause partially overlaps this release's code paths (Section 31), with explicit manager acknowledgment
- A high-severity (non-critical) security finding exists (Section 9), with a documented mitigation and remediation date

## 35. Deployment window decision

`payment-service` production deployments are permitted only during the standard deployment window: Monday through Thursday, 09:00–16:00 ET. Deployments outside this window require Release Manager approval and are limited to SEV1/SEV2 hotfixes. No deployment is permitted during an active change freeze declared by the Release Manager.

## 36. Go/no-go decision

| Decision | Permitted when |
|---|---|
| **GO** | All mandatory release-blocking conditions (Section 33) pass, no conditional-approval items are outstanding, and the Payments Platform Engineering Manager has approved the release |
| **CONDITIONAL GO** | All mandatory release-blocking conditions (Section 33) pass, one or more conditional-approval conditions (Section 34) are outstanding with documented follow-up, and the Payments Platform Engineering Manager has explicitly approved proceeding under those conditions |
| **NO-GO** | Any mandatory release-blocking condition (Section 33) fails, or the Payments Platform Engineering Manager has not provided approval |

## 37. Post-deployment validation

Within 30 minutes of deployment to `production`, the on-call engineer must confirm:

- Payment authorization failure rate remains below 0.5%.
- p95 authorize latency remains below 400 ms.
- No new duplicate transactions have been detected.
- All alerts listed in the Payment Service Production Runbook (Section 10) remain in a normal state.

If any threshold from Section 7 of the Payment Service Production Runbook is breached within this window, the on-call engineer must proceed directly to the rollback decision criteria in that runbook.

## 38. Release evidence retention

All release-readiness checklist results, CI run links, staging validation evidence, and approval records must be retained in the release Jira ticket for a minimum of 1 year to satisfy audit and compliance review requirements.
