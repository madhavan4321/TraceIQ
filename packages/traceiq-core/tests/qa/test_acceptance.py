"""Independent acceptance checks spanning record comparison and report assembly."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime

import polars as pl
import pytest

from traceiq.comparison.records import compare_records
from traceiq.models import (
    AnalysisStage,
    ComparisonOptions,
    DatasetMetadata,
    FindingKind,
    FindingSeverity,
    InvestigationRequest,
    InvestigationStatus,
    RecordComparisonMode,
    RecordComparisonStatus,
)
from traceiq.reporting.assembly import run_investigation
from traceiq.reporting.json import report_from_json, report_to_json
from traceiq.storage.reports import (
    DuplicateReportIdError,
    ReportNotFoundError,
    ReportStore,
)


def _compare(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    keys: tuple[str, ...] = (),
    example_limit: int = 10,
    excluded_columns: tuple[str, ...] = (),
):
    return compare_records(
        baseline.lazy(),
        candidate.lazy(),
        ComparisonOptions(key_columns=keys, example_limit=example_limit),
        baseline_dataset_id="baseline-id",
        candidate_dataset_id="candidate-id",
        excluded_columns=excluded_columns,
    )


def _metadata(path, dataset_id: str, name: str) -> DatasetMetadata:
    payload = path.read_bytes()
    return DatasetMetadata(
        dataset_id=dataset_id,
        name=name,
        snapshot_path=str(path),
        size_bytes=len(payload),
        content_sha256=hashlib.sha256(payload).hexdigest(),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_keyed_comparison_reports_added_removed_modified_and_null_semantics() -> None:
    baseline = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [None, "before", "gone"],
            "schema_changed": ["old-a", "old-b", "old-c"],
        }
    )
    candidate = pl.DataFrame(
        {
            "id": [4, 2, 1],
            "value": ["new", None, None],
            "schema_changed": ["new-d", "new-b", "new-a"],
        }
    )

    result = _compare(
        baseline,
        candidate,
        keys=("id",),
        excluded_columns=("schema_changed",),
    )

    assert result.mode == RecordComparisonMode.KEYED
    assert result.status == RecordComparisonStatus.COMPARABLE
    assert result.counts.added == 1
    assert result.counts.removed == 1
    assert result.counts.modified == 1
    assert {finding.kind for finding in result.findings} == {
        FindingKind.RECORDS_ADDED,
        FindingKind.RECORDS_REMOVED,
        FindingKind.RECORD_MODIFIED,
    }
    modified = next(f for f in result.findings if f.kind == FindingKind.RECORD_MODIFIED)
    assert modified.affected_columns == ("value",)
    provenance = modified.provenance
    assert provenance is not None
    assert provenance.stage == AnalysisStage.RECORDS
    assert provenance.baseline_dataset_id == "baseline-id"
    assert provenance.candidate_dataset_id == "candidate-id"
    assert provenance.method


@pytest.mark.parametrize(
    "baseline,candidate",
    [
        (
            pl.DataFrame({"id": [1, None], "value": ["a", "b"]}),
            pl.DataFrame({"id": [1, 2], "value": ["a", "b"]}),
        ),
        (
            pl.DataFrame({"id": [1, 1], "value": ["a", "b"]}),
            pl.DataFrame({"id": [1, 2], "value": ["a", "b"]}),
        ),
    ],
    ids=["null-key", "duplicate-key"],
)
def test_invalid_keys_are_reported_without_guessing_diffs(baseline, candidate) -> None:
    result = _compare(baseline, candidate, keys=("id",))

    assert result.status == RecordComparisonStatus.INVALID_KEYS
    assert result.counts.added == result.counts.removed == result.counts.modified == 0
    assert any(f.kind == FindingKind.INVALID_COMPARISON_KEY for f in result.findings)


def test_key_type_mismatch_is_not_silently_coerced() -> None:
    result = _compare(
        pl.DataFrame({"id": [1], "value": ["a"]}),
        pl.DataFrame({"id": ["1"], "value": ["a"]}),
        keys=("id",),
    )

    assert result.status == RecordComparisonStatus.INVALID_KEYS
    assert any(f.kind == FindingKind.INVALID_COMPARISON_KEY for f in result.findings)


def test_missing_key_column_is_invalid() -> None:
    result = _compare(
        pl.DataFrame({"id": [1], "value": ["a"]}),
        pl.DataFrame({"other_id": [1], "value": ["a"]}),
        keys=("id",),
    )
    assert result.status == RecordComparisonStatus.INVALID_KEYS
    assert result.counts.added == result.counts.removed == result.counts.modified == 0


def test_keyless_multiset_counts_multiplicity_and_ignores_row_order() -> None:
    baseline = pl.DataFrame({"name": ["A", "A", "A", "B"], "value": [None, None, None, 2]})
    candidate = pl.DataFrame({"name": ["B", "A", "C", "C"], "value": [2, None, 3, 3]})

    result = _compare(baseline, candidate)
    permuted = _compare(baseline.reverse(), candidate.reverse())

    assert result.mode == RecordComparisonMode.MULTISET
    assert result.status == RecordComparisonStatus.COMPARABLE
    assert result.counts.added == 2
    assert result.counts.removed == 2
    assert result.counts.modified == 0
    assert result == permuted


def test_keyless_without_common_columns_is_non_comparable() -> None:
    result = _compare(
        pl.DataFrame({"old": [1, 2]}),
        pl.DataFrame({"new": [1, 2]}),
    )

    assert result.status == RecordComparisonStatus.NON_COMPARABLE
    assert any(f.kind == FindingKind.RECORDS_NOT_COMPARABLE for f in result.findings)
    assert result.counts.added == result.counts.removed == 0


def test_key_columns_are_not_reported_as_modified_attributes() -> None:
    result = _compare(
        pl.DataFrame({"id": [1], "Name": ["alpha"]}),
        pl.DataFrame({"id": [1], "Name": ["ALPHA"]}),
        keys=("id",),
    )
    assert result.counts.modified == 1
    finding = next(f for f in result.findings if f.kind == FindingKind.RECORD_MODIFIED)
    assert finding.affected_columns == ("Name",)


def test_examples_are_bounded_and_deterministic_under_row_permutation() -> None:
    baseline = pl.DataFrame({"id": [], "value": []}, schema={"id": pl.Int64, "value": pl.String})
    candidate = pl.DataFrame({"id": list(range(10)), "value": [f"v{i}" for i in range(10)]})
    first = _compare(baseline, candidate, keys=("id",), example_limit=2)
    second = _compare(baseline, candidate.reverse(), keys=("id",), example_limit=2)

    assert first == second
    assert all(len(f.evidence.examples) <= 2 for f in first.findings)


def test_investigation_assembles_profiles_schema_record_findings_and_provenance(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    baseline_path.write_text("id,name\n1,Ada\n2,Lin\n", encoding="utf-8")
    candidate_path.write_text("id,name\n2,Lin changed\n3,Sam\n", encoding="utf-8")
    request = InvestigationRequest(
        operation="compare",
        datasets=(
            _metadata(baseline_path, "base", "baseline.csv"),
            _metadata(candidate_path, "cand", "candidate.csv"),
        ),
        comparison_options=ComparisonOptions(key_columns=("id",), example_limit=3),
    )

    report = run_investigation(request)

    assert report.analysis_status == InvestigationStatus.COMPLETE
    assert {profile.dataset_id for profile in report.dataset_profiles} == {"base", "cand"}
    assert {stage.stage for stage in report.stages} == {
        AnalysisStage.PROFILE,
        AnalysisStage.SCHEMA,
        AnalysisStage.RECORDS,
    }
    assert all(f.provenance is not None for f in report.findings)
    assert report.summary.finding_count == len(report.findings)
    severity_counts = Counter(f.severity for f in report.findings)
    assert report.summary.counts_by_severity == {
        severity: severity_counts.get(severity, 0) for severity in FindingSeverity
    }


def test_failed_profile_marks_report_partial_and_dependents_skipped(tmp_path) -> None:
    malformed_path = tmp_path / "bad.csv"
    malformed_path.write_bytes(b"a,b\n1,2\n3,4,5\n")
    other_path = tmp_path / "other.csv"
    other_path.write_text("id\n1\n", encoding="utf-8")
    request = InvestigationRequest(
        operation="compare",
        datasets=(
            _metadata(malformed_path, "base", "bad.csv"),
            _metadata(other_path, "cand", "other.csv"),
        ),
        comparison_options=ComparisonOptions(key_columns=("id",)),
    )

    report = run_investigation(request)

    assert report.analysis_status == InvestigationStatus.PARTIAL
    status_by_stage = {(s.stage, s.dataset_id): s.status for s in report.stages}
    assert status_by_stage[(AnalysisStage.PROFILE, "base")].value == "failed"
    assert status_by_stage[(AnalysisStage.SCHEMA, None)].value == "skipped"
    assert status_by_stage[(AnalysisStage.RECORDS, None)].value == "skipped"


def test_json_is_deterministic_and_sqlite_store_roundtrips_and_rejects_duplicate_ids(
    tmp_path,
) -> None:
    path = tmp_path / "dataset.csv"
    path.write_text("id,value\n1,a\n", encoding="utf-8")
    report = run_investigation(
        InvestigationRequest(
            operation="inspect",
            datasets=(_metadata(path, "only", "dataset.csv"),),
        )
    )
    encoded = report_to_json(report)
    assert report_to_json(report) == encoded
    assert report_from_json(encoded) == report

    store = ReportStore(tmp_path / "reports.sqlite3")
    store.save(report)
    assert store.get(report.report_id) == report
    with pytest.raises(ReportNotFoundError):
        store.get("missing")
    duplicate = report.model_copy(update={"warnings": ("different payload",)})
    with pytest.raises(DuplicateReportIdError):
        store.save(duplicate)
    assert store.get(report.report_id) == report
