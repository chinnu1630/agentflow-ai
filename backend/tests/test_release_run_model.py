from sqlalchemy import inspect

from app.models.release_run import ReleaseRun


def test_release_run_table_name_is_correct() -> None:
    """ReleaseRun should map to the release_runs database table."""
    assert ReleaseRun.__tablename__ == "release_runs"


def test_release_run_model_has_required_columns() -> None:
    """ReleaseRun should contain the required audit and lifecycle columns."""
    mapper = inspect(ReleaseRun)

    column_names = {column.name for column in mapper.columns}

    expected_columns = {
        "id",
        "run_id",
        "query",
        "requested_by",
        "status",
        "created_at",
        "completed_at",
    }

    assert expected_columns.issubset(column_names)


def test_release_run_primary_key_is_id() -> None:
    """ReleaseRun should use id as the primary key."""
    mapper = inspect(ReleaseRun)

    primary_key_columns = {column.name for column in mapper.primary_key}

    assert primary_key_columns == {"id"}