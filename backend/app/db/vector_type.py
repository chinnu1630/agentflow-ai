"""Runtime-loaded PostgreSQL vector type for SQLAlchemy models."""

from __future__ import annotations

import importlib
from typing import Protocol, cast

from sqlalchemy.sql.type_api import TypeEngine


class VectorTypeFactory(Protocol):
    """Callable interface exposed by pgvector's SQLAlchemy Vector type."""

    def __call__(self, dimension: int) -> TypeEngine[object]:
        """Create a vector column type with the requested dimension."""
        ...


def create_vector_type(dimension: int) -> TypeEngine[object]:
    """Return pgvector's SQLAlchemy type without static import traversal.

    Loading pgvector dynamically keeps mypy's Python 3.11 analysis from
    traversing NumPy stubs that currently require Python 3.12 syntax.
    """
    if dimension < 1:
        raise ValueError("dimension must be greater than zero")

    module = importlib.import_module("pgvector.sqlalchemy")
    vector_factory = cast(VectorTypeFactory, vars(module)["Vector"])

    return vector_factory(dimension)
