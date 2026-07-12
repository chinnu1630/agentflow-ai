# OpenTelemetry Observability Milestone

## Purpose

This milestone adds production-grade observability to AgentFlow AI.

AgentFlow AI answers the release manager question:

"What are the biggest release risks this week?"

The workflow touches GitHub, Jira, engineering documents, risk scoring, HITL approval, and Slack alerts. In production, failures can happen at any step. OpenTelemetry helps engineers trace where latency or failure happened without leaking sensitive enterprise data.

## Architecture Placement

AgentFlow architecture:

Streamlit UI
-> FastAPI
-> LangGraph Orchestrator
-> EngOps Agent, Knowledge Agent, ML Scoring, Risk Synthesis, HITL Gate, Slack Agent
-> PostgreSQL, pgvector, Redis
-> OpenTelemetry, Structured Logs, Audit Events
-> Docker
-> AWS EC2

OpenTelemetry sits across the FastAPI and LangGraph execution path.

## Observability Layers

AgentFlow now has three observability layers:

1. Audit Events

Durable business history: who ran what, when, what happened, and who approved.

2. Structured Logs

Runtime details: request lifecycle, errors, and service execution.

3. OpenTelemetry Traces

Request and workflow path: latency, failure location, and span-level metadata.

## Implemented Business Spans

Current business spans:

- release_run.risks_endpoint
- release_risk.workflow
- risk.scoring_audit
- approval.ensure_pending
- snapshot.persist
- approval.decision
- knowledge.retrieve
- slack.release_alert.route
- slack.release_alert.duplicate_check

## Trace Correlation

Structured logs now include OpenTelemetry correlation fields when an active span exists:

- trace_id
- span_id

Request middleware also attaches AgentFlow business IDs to the current request span:

- agentflow.run_id
- agentflow.request_id

This allows engineers to move from trace to logs to audit events to release_run_id.

## Safe Span Attributes

Only safe metadata should be attached to spans:

- IDs
- counts
- booleans
- statuses
- route patterns
- policy versions

Unsafe data must not be traced:

- GitHub PR body
- Jira description
- Knowledge document chunks
- Slack message body
- manager query text
- API tokens
- secrets
- raw risk payloads

## Security Reasoning

Trace data often leaves the application boundary and may be exported to an OpenTelemetry collector, dashboard, or external observability vendor. For AgentFlow, this matters because the system handles internal engineering data.

The implementation avoids putting raw enterprise content into traces. Instead, it uses safe metadata such as IDs, counts, booleans, and status fields.

## Tests Added

Observability test coverage includes:

- tests/observability/test_tracing.py
- tests/observability/test_business_span_wiring.py
- tests/observability/test_safe_span_attributes.py
- tests/observability/test_logging_trace_context.py
- tests/observability/test_request_context_span_attributes.py

These tests verify:

- tracing can be disabled safely
- safe span attributes are filtered
- known business spans exist
- sensitive terms are not added to span blocks
- logs include trace_id and span_id when available
- request middleware attaches run_id to spans

## Production Benefit

With this milestone, AgentFlow can answer:

- Which release run failed?
- Which workflow stage was slow?
- Did Knowledge retrieval run?
- Was approval required?
- Was snapshot persistence successful?
- Was Slack delivery attempted?
- Was Slack blocked because of duplicate-send protection?
- Which logs belong to the failed trace?

## Scalability Considerations

At 10x load, trace volume can become expensive and slow if exported directly from the app.

Recommended production hardening:

- Use BatchSpanProcessor
- Use sampling
- Export to an OpenTelemetry Collector
- Avoid high-cardinality attributes where possible
- Do not trace raw payloads
- Alert on failure rate and latency by span name

## Interview Explanation

I added OpenTelemetry observability to AgentFlow AI so every release-risk workflow can be traced across FastAPI, LangGraph, Knowledge retrieval, HITL approval, risk snapshot persistence, and Slack delivery. I also connected traces to structured logs using trace_id and span_id, and attached run_id to request spans for business-level correlation. To prevent data leakage, I added guardrail tests that block secrets or raw enterprise content from being attached to spans.

## Status

Completed and tested.
