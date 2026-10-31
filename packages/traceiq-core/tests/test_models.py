from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from traceiq.models import (
    AnalysisStage,
    AnalysisStageResult,
    AnalysisStageStatus,
    ColumnProfile,
    ComparisonOptions,
    DatasetMetadata,
    DatasetProfile,
    DatasetProfileResult,
    DatasetReference,
    DatasetSource,
    DateToDatetimePolicy,
    Finding,
    FindingEvidence,
    FindingKind,
    FindingProvenance,
    FindingSeverity,
    InvestigationReport,
    InvestigationRequest,
    InvestigationStatus,
    InvestigationSummary,
    LogicalDataType,
    RecordComparisonMode,
    RecordComparisonResult,
    RecordComparisonStatus,
    SchemaChange,
    SchemaChangeKind,
    SchemaComparisonCounts,
    SchemaComparisonInput,
    SchemaComparisonResult,
    SourceInvestigationRequest,
    TypeCompatibility,
)

HASH = "a" * 64


def _metadata(dataset_id: str, name: str) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id=dataset_id,
        name=name,
        snapshot_path=f"{name}",
        size_bytes=1,
        content_sha256=HASH,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_comparison_options_have_deterministic_policies_and_bounded_examples() -> None:
    options = ComparisonOptions(key_columns=("id",))
    assert options.null_key_policy == "report_invalid"
    assert options.duplicate_key_policy == "report_invalid"
    assert options.row_order_policy == "ignored"
    assert options.type_normalization == "logical"
    assert options.example_limit == 100
    with pytest.raises(ValidationError):
        ComparisonOptions(example_limit=10_001)


def test_investigation_request_enforces_dataset_count_and_unique_ids() -> None:
    with pytest.raises(ValidationError, match="requires exactly"):
        InvestigationRequest(operation="compare", datasets=(_metadata("d1", "one.csv"),))
    with pytest.raises(ValidationError, match="unique"):
        InvestigationRequest(
            operation="compare",
            datasets=(_metadata("d1", "one.csv"), _metadata("d1", "two.csv")),
        )


def test_source_investigation_request_validates_operation_counts_and_ids(
    tmp_path: Path,
) -> None:
    first = DatasetSource(path=tmp_path / "first.csv", dataset_id="first")
    second = DatasetSource(path=tmp_path / "second.csv", dataset_id="second")
    inspect = SourceInvestigationRequest(operation="inspect", datasets=(first,))
    compare = SourceInvestigationRequest(operation="compare", datasets=(first, second))

    assert inspect.datasets[0].path == tmp_path / "first.csv"
    assert len(compare.datasets) == 2
    with pytest.raises(ValidationError, match="requires exactly"):
        SourceInvestigationRequest(operation="compare", datasets=(first,))
    with pytest.raises(ValidationError, match="requires exactly"):
        SourceInvestigationRequest(operation="inspect", datasets=(first, second))
    with pytest.raises(ValidationError, match="unique"):
        SourceInvestigationRequest(
            operation="compare",
            datasets=(first, DatasetSource(path=tmp_path / "other.csv", dataset_id="first")),
        )
    with pytest.raises(ValidationError):
        DatasetSource(path=tmp_path / "bad.csv", unexpected="field")


def test_dataset_profile_requires_consistent_column_count() -> None:
    with pytest.raises(ValidationError, match="column_count"):
        DatasetProfile(row_count=2, column_count=1, columns=())


def test_dataset_profile_validates_normalized_types_and_null_percentage() -> None:
    profile = DatasetProfile(
        row_count=4,
        column_count=1,
        columns=(
            ColumnProfile(
                name="amount",
                logical_type="decimal",
                null_count=1,
                null_percentage=25.0,
            ),
        ),
        duplicate_row_count=2,
    )
    assert profile.columns[0].logical_type == LogicalDataType.DECIMAL
    with pytest.raises(ValidationError, match="null_percentage"):
        DatasetProfile(
            row_count=4,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="amount",
                    logical_type="decimal",
                    null_count=1,
                    null_percentage=20.0,
                ),
            ),
        )
    with pytest.raises(ValidationError):
        ColumnProfile(name="x", logical_type="unrecognized")


def test_schema_change_validates_before_after_types_and_compatibility() -> None:
    change = SchemaChange(
        column_name="amount",
        kind=SchemaChangeKind.TYPE_CHANGED,
        before_type=LogicalDataType.INTEGER,
        after_type=LogicalDataType.DECIMAL,
        compatibility=TypeCompatibility.POTENTIALLY_LOSSY,
    )
    assert change.before_type == LogicalDataType.INTEGER
    assert change.after_type == LogicalDataType.DECIMAL
    with pytest.raises(ValidationError):
        SchemaChange(
            column_name="amount",
            kind=SchemaChangeKind.TYPE_CHANGED,
            before_type=LogicalDataType.INTEGER,
            after_type=LogicalDataType.INTEGER,
            compatibility=TypeCompatibility.UNCHANGED,
        )
    comparison_input = SchemaComparisonInput(
        baseline=DatasetProfile(row_count=0, column_count=0),
        candidate=DatasetProfile(row_count=0, column_count=0),
        date_to_datetime_policy=DateToDatetimePolicy.MIDNIGHT,
    )
    assert comparison_input.date_to_datetime_policy == DateToDatetimePolicy.MIDNIGHT
    result = SchemaComparisonResult(
        counts=SchemaComparisonCounts(
            added=0,
            removed=0,
            type_changed=0,
            unchanged=0,
            compatible=0,
            potentially_lossy=0,
            incompatible=0,
        )
    )
    assert result.findings == ()


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
        provenance=FindingProvenance(
            stage=AnalysisStage.SCHEMA,
            method="schema_name_set_difference",
            baseline_dataset_id="d1",
            candidate_dataset_id="d2",
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
        dataset_profiles=(
            DatasetProfileResult(
                dataset_id="d1", profile=DatasetProfile(row_count=0, column_count=0)
            ),
            DatasetProfileResult(
                dataset_id="d2", profile=DatasetProfile(row_count=0, column_count=0)
            ),
        ),
        summary=InvestigationSummary(finding_count=1),
        findings=(finding,),
        stages=(
            AnalysisStageResult(
                stage=AnalysisStage.PROFILE, status=AnalysisStageStatus.COMPLETED, dataset_id="d1"
            ),
            AnalysisStageResult(
                stage=AnalysisStage.PROFILE, status=AnalysisStageStatus.COMPLETED, dataset_id="d2"
            ),
            AnalysisStageResult(stage=AnalysisStage.SCHEMA, status=AnalysisStageStatus.COMPLETED),
            AnalysisStageResult(stage=AnalysisStage.RECORDS, status=AnalysisStageStatus.COMPLETED),
        ),
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
        dataset_profiles=(
            DatasetProfileResult(
                dataset_id="d1", profile=DatasetProfile(row_count=0, column_count=0)
            ),
        ),
        summary=InvestigationSummary(finding_count=0),
        stages=(
            AnalysisStageResult(
                stage=AnalysisStage.PROFILE,
                status=AnalysisStageStatus.COMPLETED,
                dataset_id="d1",
            ),
        ),
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
        dataset_profiles=(
            DatasetProfileResult(
                dataset_id="d1", profile=DatasetProfile(row_count=0, column_count=0)
            ),
            DatasetProfileResult(
                dataset_id="d2", profile=DatasetProfile(row_count=0, column_count=0)
            ),
        ),
        summary=InvestigationSummary(finding_count=0),
        stages=(
            AnalysisStageResult(
                stage=AnalysisStage.PROFILE,
                status=AnalysisStageStatus.COMPLETED,
                dataset_id="d1",
            ),
            AnalysisStageResult(
                stage=AnalysisStage.PROFILE,
                status=AnalysisStageStatus.COMPLETED,
                dataset_id="d2",
            ),
            AnalysisStageResult(stage=AnalysisStage.SCHEMA, status=AnalysisStageStatus.COMPLETED),
            AnalysisStageResult(stage=AnalysisStage.RECORDS, status=AnalysisStageStatus.COMPLETED),
        ),
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
        dataset_profiles=(
            DatasetProfileResult(
                dataset_id="d1", profile=DatasetProfile(row_count=0, column_count=0)
            ),
        ),
        summary=InvestigationSummary(finding_count=1),
        findings=(finding,),
        stages=(
            AnalysisStageResult(
                stage=AnalysisStage.PROFILE, status=AnalysisStageStatus.COMPLETED, dataset_id="d1"
            ),
        ),
        example_limit=1,
    )
    with pytest.raises(ValidationError, match="example"):
        InvestigationReport(**base)
    with pytest.raises(ValidationError, match="finding_count"):
        invalid_summary = InvestigationSummary(finding_count=1)
        InvestigationReport(**{**base, "findings": (), "summary": invalid_summary})


def test_report_requires_provenance_and_consistent_stage_status() -> None:
    finding = Finding(
        finding_id="f-1",
        kind=FindingKind.MISSING_VALUES,
        severity=FindingSeverity.WARNING,
        message="Missing values",
        counts={"missing": 1},
        evidence=FindingEvidence(method="profile_null_count"),
        provenance=FindingProvenance(
            stage=AnalysisStage.PROFILE,
            method="profile_null_count",
            dataset_id="d1",
        ),
    )
    dataset = DatasetReference(dataset_id="d1", name="data.csv", content_sha256=HASH)
    common = dict(
        report_id="r-1",
        operation="inspect",
        datasets=(dataset,),
        summary=InvestigationSummary(finding_count=1),
        findings=(finding,),
        stages=(
            AnalysisStageResult(
                stage=AnalysisStage.PROFILE,
                status=AnalysisStageStatus.FAILED,
                dataset_id="d1",
                error_code="CSV_INVALID",
                message="could not parse source",
            ),
        ),
    )
    with pytest.raises(ValidationError, match="analysis_status"):
        InvestigationReport(**common)
    partial = InvestigationReport(**{**common, "analysis_status": InvestigationStatus.PARTIAL})
    assert partial.analysis_status == InvestigationStatus.PARTIAL


def test_keyless_comparison_without_common_columns_is_explicitly_non_comparable() -> None:
    finding = Finding(
        finding_id="f-1",
        kind=FindingKind.RECORDS_NOT_COMPARABLE,
        severity=FindingSeverity.ERROR,
        message="No common columns",
        counts={"columns": 0},
        evidence=FindingEvidence(method="keyless_multiset_comparison"),
        provenance=FindingProvenance(
            stage=AnalysisStage.RECORDS,
            method="keyless_multiset_comparison",
            baseline_dataset_id="d1",
            candidate_dataset_id="d2",
        ),
    )
    result = RecordComparisonResult(
        mode=RecordComparisonMode.MULTISET,
        status=RecordComparisonStatus.NON_COMPARABLE,
        findings=(finding,),
    )
    assert result.status == RecordComparisonStatus.NON_COMPARABLE
