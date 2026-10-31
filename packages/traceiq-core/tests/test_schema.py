from __future__ import annotations

import polars as pl
import pytest

from traceiq.comparison.schema import SchemaComparisonError, compare_schemas
from traceiq.models import (
    ColumnProfile,
    DatasetProfile,
    DateToDatetimePolicy,
    FindingKind,
    FindingSeverity,
    LogicalDataType,
    SchemaChangeKind,
    SchemaComparisonInput,
    SchemaComparisonResult,
    TypeCompatibility,
)
from traceiq.types import normalize_polars_dtype


@pytest.mark.parametrize(
    ("polars_dtype", "logical_type"),
    [
        (pl.Null, LogicalDataType.NULL),
        (pl.Boolean, LogicalDataType.BOOLEAN),
        (pl.Int8, LogicalDataType.INTEGER),
        (pl.UInt64, LogicalDataType.INTEGER),
        (pl.Float32, LogicalDataType.FLOAT),
        (pl.Decimal(precision=12, scale=3), LogicalDataType.DECIMAL),
        (pl.String, LogicalDataType.STRING),
        (pl.Date, LogicalDataType.DATE),
        (pl.Datetime("us"), LogicalDataType.DATETIME),
        (pl.Time, LogicalDataType.TIME),
        (pl.Duration("ms"), LogicalDataType.DURATION),
        (pl.Binary, LogicalDataType.BINARY),
        (pl.Categorical, LogicalDataType.CATEGORICAL),
        (pl.Enum(["a", "b"]), LogicalDataType.CATEGORICAL),
        (pl.List(pl.Int64), LogicalDataType.LIST),
        (pl.Struct({"x": pl.Int64}), LogicalDataType.STRUCT),
        (pl.Object, LogicalDataType.UNKNOWN),
    ],
)
def test_normalize_polars_dtype(polars_dtype: pl.DataType, logical_type: LogicalDataType) -> None:
    assert normalize_polars_dtype(polars_dtype) == logical_type


def _profile(columns: list[tuple[str, LogicalDataType]]) -> DatasetProfile:
    return DatasetProfile(
        row_count=0,
        column_count=len(columns),
        columns=tuple(
            ColumnProfile(name=name, logical_type=logical_type) for name, logical_type in columns
        ),
    )


def _compare(
    before: list[tuple[str, LogicalDataType]],
    after: list[tuple[str, LogicalDataType]],
    policy: DateToDatetimePolicy = DateToDatetimePolicy.DISALLOW,
) -> SchemaComparisonResult:
    return compare_schemas(
        SchemaComparisonInput(
            baseline=_profile(before),
            candidate=_profile(after),
            date_to_datetime_policy=policy,
        )
    )


def test_schema_diff_is_case_sensitive_order_independent_and_deterministic() -> None:
    baseline = [
        ("z_same", LogicalDataType.STRING),
        ("Old", LogicalDataType.DATE),
        ("amount", LogicalDataType.INTEGER),
        ("case", LogicalDataType.BOOLEAN),
    ]
    candidate = [
        ("new", LogicalDataType.BINARY),
        ("case", LogicalDataType.BOOLEAN),
        ("amount", LogicalDataType.FLOAT),
        ("z_same", LogicalDataType.STRING),
        ("old", LogicalDataType.DATE),
    ]
    first = _compare(baseline, candidate)
    second = _compare(list(reversed(baseline)), list(reversed(candidate)))

    assert first == second
    assert [change.column_name for change in first.changes] == [
        "Old",
        "amount",
        "new",
        "old",
    ]
    assert [change.kind for change in first.changes] == [
        SchemaChangeKind.REMOVED,
        SchemaChangeKind.TYPE_CHANGED,
        SchemaChangeKind.ADDED,
        SchemaChangeKind.ADDED,
    ]
    assert first.counts.unchanged == 2
    assert first.counts.added == 2
    assert first.counts.removed == 1
    assert first.counts.type_changed == 1
    assert first.counts.potentially_lossy == 1
    assert first.counts.incompatible == 0
    assert first.findings == second.findings
    assert [finding.finding_id for finding in first.findings] == [
        finding.finding_id for finding in second.findings
    ]


def test_findings_include_types_severity_columns_counts_and_stable_ids() -> None:
    result = _compare(
        [("added_later", LogicalDataType.INTEGER), ("removed_later", LogicalDataType.STRING)],
        [("added_later", LogicalDataType.DECIMAL), ("added_now", LogicalDataType.DATE)],
    )
    by_column = {finding.affected_columns[0]: finding for finding in result.findings}

    type_finding = by_column["added_later"]
    assert type_finding.kind == FindingKind.SCHEMA_TYPE_CHANGED
    assert type_finding.severity == FindingSeverity.WARNING
    assert type_finding.counts == {"columns": 1}
    assert type_finding.evidence.counts == {"columns": 1}
    assert type_finding.evidence.details["before_type"] == "integer"
    assert type_finding.evidence.details["after_type"] == "decimal"
    assert type_finding.evidence.details["compatibility"] == "potentially_lossy"

    added_finding = by_column["added_now"]
    assert added_finding.kind == FindingKind.SCHEMA_ADDED_COLUMN
    assert added_finding.evidence.details["before_type"] is None
    assert added_finding.evidence.details["after_type"] == "date"

    removed_finding = by_column["removed_later"]
    assert removed_finding.kind == FindingKind.SCHEMA_REMOVED_COLUMN
    assert removed_finding.evidence.details["before_type"] == "string"
    assert removed_finding.evidence.details["after_type"] is None

    second = _compare(
        [("added_later", LogicalDataType.INTEGER), ("removed_later", LogicalDataType.STRING)],
        [("added_later", LogicalDataType.DECIMAL), ("added_now", LogicalDataType.DATE)],
    )
    assert [finding.finding_id for finding in result.findings] == [
        finding.finding_id for finding in second.findings
    ]


@pytest.mark.parametrize(
    ("before", "after", "policy", "expected"),
    [
        (
            LogicalDataType.INTEGER,
            LogicalDataType.DECIMAL,
            DateToDatetimePolicy.DISALLOW,
            TypeCompatibility.POTENTIALLY_LOSSY,
        ),
        (
            LogicalDataType.INTEGER,
            LogicalDataType.FLOAT,
            DateToDatetimePolicy.DISALLOW,
            TypeCompatibility.POTENTIALLY_LOSSY,
        ),
        (
            LogicalDataType.CATEGORICAL,
            LogicalDataType.STRING,
            DateToDatetimePolicy.DISALLOW,
            TypeCompatibility.COMPATIBLE,
        ),
        (
            LogicalDataType.STRING,
            LogicalDataType.BOOLEAN,
            DateToDatetimePolicy.DISALLOW,
            TypeCompatibility.INCOMPATIBLE,
        ),
        (
            LogicalDataType.DATE,
            LogicalDataType.DATETIME,
            DateToDatetimePolicy.DISALLOW,
            TypeCompatibility.INCOMPATIBLE,
        ),
        (
            LogicalDataType.DATE,
            LogicalDataType.DATETIME,
            DateToDatetimePolicy.MIDNIGHT,
            TypeCompatibility.POTENTIALLY_LOSSY,
        ),
        (
            LogicalDataType.DATETIME,
            LogicalDataType.DATE,
            DateToDatetimePolicy.MIDNIGHT,
            TypeCompatibility.INCOMPATIBLE,
        ),
    ],
)
def test_type_compatibility_matrix_is_conservative(
    before: LogicalDataType,
    after: LogicalDataType,
    policy: DateToDatetimePolicy,
    expected: TypeCompatibility,
) -> None:
    result = _compare([("value", before)], [("value", after)], policy)
    assert result.counts.type_changed == 1
    assert result.changes[0].compatibility == expected
    assert result.changes[0].before_type == before
    assert result.changes[0].after_type == after


def test_duplicate_column_names_are_rejected_instead_of_silently_overwritten() -> None:
    duplicate_profile = DatasetProfile(
        row_count=0,
        column_count=2,
        columns=(
            ColumnProfile(name="x", logical_type=LogicalDataType.INTEGER),
            ColumnProfile(name="x", logical_type=LogicalDataType.STRING),
        ),
    )
    with pytest.raises(SchemaComparisonError, match="duplicate column names"):
        compare_schemas(
            SchemaComparisonInput(
                baseline=duplicate_profile,
                candidate=_profile([]),
            )
        )
