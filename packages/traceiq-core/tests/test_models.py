from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from traceiq.models import (
    ComparisonOptions,
    DatasetProfile,
    DatasetReference,
    Finding,
    FindingEvidence,
    FindingKind,
    FindingSeverity,
    InvestigationReport,
    InvestigationSummary,
)

HASH = "a" * 64


def test_comparison_options_have_deterministic_policies_and_bounded_examples() -> None:
    options = ComparisonOptions(key_columns=("id",))
    assert options.null_key_policy == "report_invalid"
    assert options.duplicate_key_policy == "report_invalid"
    assert options.row_order_policy == "ignored"
    assert options.type_normalization == "logical"
    assert options.example_limit == 100
    with pytest.raises(ValidationError):
        ComparisonOptions(example_limit=10_001)


def test_dataset_profile_requires_consistent_column_count() -> None:
    with pytest.raises(ValidationError, match="column_count"):
        DatasetProfile(row_count=2, column_count=1, columns=())


def test_report_model_round_trips_structured_finding_without_presentation_fields() -> None:
    finding = Finding(
        finding_id="f-1",
        kind=FindingKind.SCHEMA_ADDED_COLUMN,
        severity=FindingSeverity.WARNING,
        message="Column added",
        affected_columns=("region",),
        counts={"columns": 1},
        evidence=FindingEvidence(
            method="schema_name_set_difference",
            counts={"added": 1},
            examples=({"column": "region", "type": "String"},),
        ),
    )
    report = InvestigationReport(
        report_id="r-1",
        operation="compare",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        datasets=(
            DatasetReference(dataset_id="d1", name="old.csv", content_sha256=HASH),
            DatasetReference(dataset_id="d2", name="new.csv", content_sha256=HASH),
        ),
        summary=InvestigationSummary(finding_count=1),
        findings=(finding,),
        example_limit=1,
    )
    encoded = report.model_dump(mode="json")
    decoded = InvestigationReport.model_validate(encoded)
    assert decoded == report
    assert "table" not in encoded and "color" not in encoded


def test_inspection_report_accepts_exactly_one_dataset() -> None:
    report = InvestigationReport(
        report_id="inspect-1",
        operation="inspect",
        datasets=(DatasetReference(dataset_id="d1", name="data.csv", content_sha256=HASH),),
        summary=InvestigationSummary(finding_count=0),
    )
    assert len(report.datasets) == 1


def test_comparison_report_accepts_exactly_two_datasets() -> None:
    report = InvestigationReport(
        report_id="compare-1",
        operation="compare",
        datasets=(
            DatasetReference(dataset_id="d1", name="old.csv", content_sha256=HASH),
            DatasetReference(dataset_id="d2", name="new.csv", content_sha256=HASH),
        ),
        summary=InvestigationSummary(finding_count=0),
    )
    assert len(report.datasets) == 2


@pytest.mark.parametrize(
    ("operation", "datasets"),
    [
        (
            "inspect",
            (
                DatasetReference(dataset_id="d1", name="a.csv", content_sha256=HASH),
                DatasetReference(dataset_id="d2", name="b.csv", content_sha256=HASH),
            ),
        ),
        ("compare", (DatasetReference(dataset_id="d1", name="a.csv", content_sha256=HASH),)),
    ],
)
def test_report_rejects_invalid_dataset_counts(
    operation: str, datasets: tuple[DatasetReference, ...]
) -> None:
    with pytest.raises(ValidationError, match="must reference exactly"):
        InvestigationReport(
            report_id="invalid-count",
            operation=operation,
            datasets=datasets,
            summary=InvestigationSummary(finding_count=0),
        )


def test_report_rejects_examples_over_limit_and_incorrect_summary() -> None:
    finding = Finding(
        finding_id="f-1",
        kind=FindingKind.MISSING_VALUES,
        severity=FindingSeverity.INFO,
        message="Missing values found",
        affected_columns=("x",),
        counts={"missing": 2},
        evidence=FindingEvidence(method="null_count", examples=({"row": 1}, {"row": 2})),
    )
    base = dict(
        report_id="r-1",
        operation="inspect",
        datasets=(DatasetReference(dataset_id="d1", name="x.csv", content_sha256=HASH),),
        summary=InvestigationSummary(finding_count=1),
        findings=(finding,),
        example_limit=1,
    )
    with pytest.raises(ValidationError, match="example"):
        InvestigationReport(**base)
    with pytest.raises(ValidationError, match="finding_count"):
        invalid_summary = InvestigationSummary(finding_count=1)
        InvestigationReport(**{**base, "findings": (), "summary": invalid_summary})
