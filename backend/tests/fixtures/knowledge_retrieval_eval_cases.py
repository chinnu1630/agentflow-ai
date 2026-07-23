"""Reusable deterministic Knowledge Agent retrieval evaluation fixtures.

These fixtures define a small benchmark dataset for AgentFlow AI retrieval.
They are intentionally test-only and deterministic so BM25, pgvector,
reranker, and future RAGAS evaluations can compare against the same cases.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.models.engineering_document import EngineeringDocumentSourceType
from app.services.document_chunker import DocumentChunkingConfig
from app.services.engineering_document_ingestion_service import (
    EngineeringDocumentIngestionRequest,
    EngineeringDocumentIngestionService,
)
from app.services.knowledge_retrieval_evaluation_service import (
    KnowledgeRetrievalEvalCase,
)


class KnowledgeRetrievalEvalDocumentFixture(BaseModel):
    """Reusable engineering document fixture for retrieval evaluation."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., min_length=1)
    source_type: EngineeringDocumentSourceType
    source_uri: str = Field(..., min_length=1)
    raw_content: str = Field(..., min_length=1)
    metadata_json: dict[str, str] = Field(default_factory=dict)

    def to_ingestion_request(self) -> EngineeringDocumentIngestionRequest:
        """Convert this fixture into an ingestion request."""
        return EngineeringDocumentIngestionRequest(
            title=self.title,
            source_type=self.source_type,
            source_uri=self.source_uri,
            raw_content=self.raw_content,
            metadata_json=self.metadata_json,
            chunking_config=DocumentChunkingConfig(
                max_tokens_per_chunk=40,
                overlap_tokens=5,
            ),
        )


def build_knowledge_retrieval_eval_documents() -> list[KnowledgeRetrievalEvalDocumentFixture]:
    """Build deterministic engineering document fixtures for retrieval evals."""
    return [
        KnowledgeRetrievalEvalDocumentFixture(
            title="Payment Redis Incident Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-redis-incident-runbook.md",
            raw_content=(
                "Redis checkout failure caused payment release risk. "
                "Redis latency increased during checkout. "
                "Use the payment rollback procedure when checkout failures spike."
            ),
            metadata_json={"team": "payments"},
        ),
        KnowledgeRetrievalEvalDocumentFixture(
            title="Release Readiness Checklist",
            source_type=EngineeringDocumentSourceType.RELEASE_CHECKLIST,
            source_uri="docs/release-readiness-checklist.md",
            raw_content=(
                "Release approval checklist requires Jira P1 review, "
                "GitHub pull request approval, CI validation, rollback readiness, "
                "and engineering manager sign off."
            ),
            metadata_json={"team": "release-management"},
        ),
        KnowledgeRetrievalEvalDocumentFixture(
            title="Payment Rollback Procedure",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-rollback-procedure.md",
            raw_content=(
                "Payment outage rollback procedure explains how to revert "
                "payment service deployments after checkout errors, failed releases, "
                "or customer-impacting payment incidents."
            ),
            metadata_json={"team": "payments"},
        ),
        KnowledgeRetrievalEvalDocumentFixture(
            title="Checkout Retry Storm Postmortem",
            source_type=EngineeringDocumentSourceType.INCIDENT_POSTMORTEM,
            source_uri="docs/checkout-retry-storm-postmortem.md",
            raw_content=(
                "Cache saturation caused a checkout retry storm and elevated latency. "
                "Missing retry jitter amplified traffic against the payment gateway. "
                "The corrective action added bounded exponential backoff."
            ),
            metadata_json={"team": "site-reliability"},
        ),
        KnowledgeRetrievalEvalDocumentFixture(
            title="Payment Service Architecture",
            source_type=EngineeringDocumentSourceType.ARCHITECTURE_DOC,
            source_uri="docs/payment-service-architecture.md",
            raw_content=(
                "The payment orchestration service owns idempotency keys. "
                "It coordinates authorization, ledger writes, and gateway callbacks. "
                "Redis stores short-lived request deduplication records."
            ),
            metadata_json={"team": "payments-platform"},
        ),
        KnowledgeRetrievalEvalDocumentFixture(
            title="Emergency Change Freeze Policy",
            source_type=EngineeringDocumentSourceType.OTHER,
            source_uri="docs/emergency-change-freeze-policy.md",
            raw_content=(
                "An emergency change freeze exception requires approval from "
                "the release manager and incident commander. "
                "The approval record must include impact and rollback evidence."
            ),
            metadata_json={"team": "release-management"},
        ),
    ]


def build_knowledge_retrieval_eval_cases(
    *,
    top_k: int = 3,
) -> list[KnowledgeRetrievalEvalCase]:
    """Build deterministic retrieval eval cases for AgentFlow AI."""
    return [
        KnowledgeRetrievalEvalCase(
            name="redis_checkout_failure",
            query="Redis checkout failure",
            expected_document_title="Payment Redis Incident Runbook",
            top_k=top_k,
        ),
        KnowledgeRetrievalEvalCase(
            name="release_approval_checklist",
            query="release approval checklist",
            expected_document_title="Release Readiness Checklist",
            top_k=top_k,
        ),
        KnowledgeRetrievalEvalCase(
            name="rollback_after_payment_outage",
            query="rollback after payment outage",
            expected_document_title="Payment Rollback Procedure",
            top_k=top_k,
        ),
        KnowledgeRetrievalEvalCase(
            name="checkout_retry_storm_cause",
            query="cache saturation retry storm",
            expected_document_title="Checkout Retry Storm Postmortem",
            top_k=top_k,
            source_types=[EngineeringDocumentSourceType.INCIDENT_POSTMORTEM.value],
        ),
        KnowledgeRetrievalEvalCase(
            name="payment_idempotency_owner",
            query="which service owns idempotency keys",
            expected_document_title="Payment Service Architecture",
            top_k=top_k,
            source_types=[EngineeringDocumentSourceType.ARCHITECTURE_DOC.value],
        ),
        KnowledgeRetrievalEvalCase(
            name="emergency_freeze_exception",
            query="emergency change freeze exception approval",
            expected_document_title="Emergency Change Freeze Policy",
            top_k=top_k,
            source_types=[EngineeringDocumentSourceType.OTHER.value],
        ),
    ]


async def seed_knowledge_retrieval_eval_documents(
    ingestion_service: EngineeringDocumentIngestionService,
    *,
    documents: Sequence[KnowledgeRetrievalEvalDocumentFixture] | None = None,
) -> None:
    """Seed reusable retrieval eval documents through the real ingestion service."""
    documents_to_seed = (
        list(documents) if documents is not None else build_knowledge_retrieval_eval_documents()
    )

    for document in documents_to_seed:
        await ingestion_service.ingest_document(document.to_ingestion_request())
