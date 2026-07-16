"""Ingest AgentFlow engineering Markdown documents into the knowledge store."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import get_session_factory
from app.models.engineering_document import EngineeringDocumentSourceType
from app.repositories.engineering_document_repository import (
    EngineeringDocumentRepository,
    EngineeringDocumentRepositoryError,
)
from app.services.engineering_document_ingestion_service import (
    EngineeringDocumentIngestionRequest,
    EngineeringDocumentIngestionResult,
    EngineeringDocumentIngestionService,
)

logger = structlog.get_logger(__name__)

DEFAULT_DOCUMENT_PATHS = (
    Path("data/engineering_documents/payment-service-runbook.md"),
    Path("data/engineering_documents/release-readiness-checklist.md"),
)


def parse_front_matter(raw_content: str) -> tuple[dict[str, Any], str]:
    """Parse the supported flat YAML front matter and return the Markdown body."""

    lines = raw_content.splitlines()

    if not lines or lines[0].strip() != "---":
        raise ValueError("Engineering document must begin with YAML front matter.")

    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("Engineering document front matter is not closed.") from exc

    metadata: dict[str, Any] = {}
    active_list_key: str | None = None

    for line in lines[1:closing_index]:
        stripped_line = line.strip()

        if not stripped_line:
            continue

        if stripped_line.startswith("- "):
            if active_list_key is None:
                raise ValueError("Front-matter list item has no parent key.")

            list_value = metadata.get(active_list_key)

            if not isinstance(list_value, list):
                raise ValueError("Front-matter list structure is invalid.")

            list_value.append(stripped_line[2:].strip())
            continue

        if ":" not in line:
            raise ValueError(f"Invalid front-matter line: {line}")

        key, raw_value = line.split(":", 1)
        normalized_key = key.strip()
        normalized_value = raw_value.strip()

        if not normalized_key:
            raise ValueError("Front-matter key must not be blank.")

        if normalized_value:
            metadata[normalized_key] = normalized_value
            active_list_key = None
        else:
            metadata[normalized_key] = []
            active_list_key = normalized_key

    body = "\n".join(lines[closing_index + 1 :]).strip()

    if not body:
        raise ValueError("Engineering document body must not be empty.")

    return metadata, body


def resolve_source_type(document_type: str) -> EngineeringDocumentSourceType:
    """Map document front-matter types to AgentFlow source types."""

    normalized_type = document_type.strip().upper()

    mapping = {
        "RUNBOOK": EngineeringDocumentSourceType.RUNBOOK,
        "CHECKLIST": EngineeringDocumentSourceType.RELEASE_CHECKLIST,
    }

    try:
        return mapping[normalized_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported engineering document type: {document_type}"
        ) from exc


def build_ingestion_request(
    document_path: Path,
) -> EngineeringDocumentIngestionRequest:
    """Load one Markdown file and build a validated ingestion request."""

    resolved_path = document_path.resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Engineering document not found: {document_path}")

    metadata, body = parse_front_matter(
        resolved_path.read_text(encoding="utf-8")
    )

    title = metadata.get("title")
    document_type = metadata.get("document_type")

    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"Document title is missing: {document_path}")

    if not isinstance(document_type, str) or not document_type.strip():
        raise ValueError(f"Document type is missing: {document_path}")

    metadata["source_filename"] = resolved_path.name

    return EngineeringDocumentIngestionRequest(
        title=title,
        source_type=resolve_source_type(document_type),
        source_uri=str(document_path),
        raw_content=body,
        metadata_json=metadata,
    )


async def ingest_documents(
    document_paths: Sequence[Path],
) -> list[EngineeringDocumentIngestionResult]:
    """Ingest all supplied documents within one database transaction."""

    session_factory = get_session_factory()

    async with session_factory() as session:
        repository = EngineeringDocumentRepository(session=session)
        service = EngineeringDocumentIngestionService(repository=repository)
        results: list[EngineeringDocumentIngestionResult] = []

        try:
            for document_path in document_paths:
                request = build_ingestion_request(document_path)
                result = await service.ingest_document(
                    request,
                    run_id="engineering-document-seed",
                )
                results.append(result)

                logger.info(
                    "engineering_document_seeded",
                    source_uri=request.source_uri,
                    document_id=str(result.document_id),
                    chunk_count=result.chunk_count,
                    duplicate_document=result.duplicate_document,
                )

            await session.commit()
            return results

        except (
            FileNotFoundError,
            ValueError,
            EngineeringDocumentRepositoryError,
            SQLAlchemyError,
        ):
            await session.rollback()
            logger.exception("engineering_document_seed_failed")
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line document paths."""

    parser = argparse.ArgumentParser(
        description="Ingest AgentFlow engineering Markdown documents."
    )
    parser.add_argument(
        "documents",
        nargs="*",
        type=Path,
        default=list(DEFAULT_DOCUMENT_PATHS),
        help="Markdown document paths. Defaults to the two payment documents.",
    )

    return parser.parse_args()


def main() -> int:
    """Run engineering-document ingestion."""

    arguments = parse_arguments()
    asyncio.run(ingest_documents(arguments.documents))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
