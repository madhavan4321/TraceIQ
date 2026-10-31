from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from traceiq.models import (
    AnalysisStage,
    AnalysisStageResult,
    AnalysisStageStatus,
    DatasetMetadata,
    DatasetProfile,
    DatasetProfileResult,
    DatasetReference,
    DatasetSource,
    Finding,
    FindingEvidence,
    FindingKind,
    FindingProvenance,
    FindingSeverity,
    InvestigationReport,
    InvestigationRequest,
    InvestigationStatus,
    InvestigationSummary,
    SourceInvestigationRequest,
)
from traceiq.reporting.assembly import (
    _profile_finding_id,
    run_investigation,
    run_investigation_from_sources,
)
from traceiq.reporting.json import report_from_json, report_to_json
from traceiq.storage.reports import (
    DuplicateReportIdError,
    ReportNotFoundError,
    ReportStore,
)


def _write_dataset(tmp_path: Path, name: str, content: str) -> DatasetMetadata:
    path = tmp_path / name
    data = content.encode("utf-8")
    path.write_bytes(data)
    return DatasetMetadata(
        dataset_id=name.removesuffix(".csv"),
        name=name,
        snapshot_path=str(path),
        size_bytes=len(data),
        content_sha256=hashlib.sha256(data).hexdigest(),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _minimal_report(report_id: str = "report-1") -> InvestigationReport:
    dataset = DatasetReference(dataset_id="data", name="data.csv", content_sha256="a" * 64)
    profile = DatasetProfile(row_count=0, column_count=0)
    return InvestigationReport(
        report_id=report_id,
        operation="inspect",
        datasets=(dataset,),
        dataset_profiles=(DatasetProfileResult(dataset_id="data", profile=profile),),
        findings=(
            Finding(
                finding_id="profile-null-data-value",
                kind=FindingKind.MISSING_VALUES,
                severity=FindingSeverity.WARNING,
                message="Value is missing",
                affected_columns=("value",),
                counts={"missing_values": 1},
                evidence=FindingEvidence(
                    method="profile_null_count",
                    counts={"missing_values": 1},
                    details={"null_percentage": 50.0},
                ),
                provenance=FindingProvenance(
                    stage=AnalysisStage.PROFILE,
                    method="profile_null_count",
                    dataset_id="data",
                ),
            ),
        ),
        stages=(
            AnalysisStageResult(
                stage=AnalysisStage.PROFILE,
                status=AnalysisStageStatus.COMPLETED,
                dataset_id="data",
            ),
        ),
        summary=InvestigationSummary(
            finding_count=1,
            counts_by_severity={FindingSeverity.WARNING: 1},
            totals={"rows:data": 2},
        ),
    )


def test_inspect_assembles_profiles_findings_and_provenance(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, "profile.csv", "name,value\nAda,1\nLin,\n")

    report = run_investigation(InvestigationRequest(operation="inspect", datasets=(dataset,)))

    assert report.analysis_status == InvestigationStatus.COMPLETE
    assert report.dataset_profiles[0].dataset_id == dataset.dataset_id
    assert report.dataset_profiles[0].profile.row_count == 2
    assert report.summary.counts_by_severity[FindingSeverity.WARNING] == 1
    finding = next(item for item in report.findings if item.kind == FindingKind.MISSING_VALUES)
    assert finding.provenance is not None
    assert finding.provenance.stage == AnalysisStage.PROFILE
    assert finding.provenance.dataset_id == dataset.dataset_id


def test_source_inspection_computes_metadata_and_round_trips(tmp_path: Path) -> None:
    content = b"name,value\nAda,1\nLin,\n"
    source = tmp_path / "profile.csv"
    source.write_bytes(content)
    request = SourceInvestigationRequest(
        operation="inspect",
        datasets=(DatasetSource(path=source, dataset_id="source-profile"),),
    )

    report = run_investigation_from_sources(request)

    assert report.datasets[0].dataset_id == "source-profile"
    assert report.datasets[0].content_sha256 == hashlib.sha256(content).hexdigest()
    assert report.analysis_status == InvestigationStatus.COMPLETE
    encoded = report_to_json(report)
    assert report_from_json(encoded) == report
    store = ReportStore(tmp_path / "source-reports.sqlite3")
    store.save(report)
    assert store.get(report.report_id) == report


def test_source_comparison_stages_each_csv_once(tmp_path: Path, monkeypatch) -> None:
    baseline_bytes = b"id,value\n1,old\n"
    candidate_bytes = b"id,value\n1,new\n2,added\n"
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    baseline.write_bytes(baseline_bytes)
    candidate.write_bytes(candidate_bytes)
    request = SourceInvestigationRequest(
        operation="compare",
        datasets=(
            DatasetSource(path=baseline, dataset_id="base"),
            DatasetSource(path=candidate, dataset_id="candidate"),
        ),
        comparison_options={"key_columns": ("id",)},
    )
    from traceiq.ingestion import csv as ingestion_csv
    from traceiq.reporting import assembly

    real_stage = assembly.stage_csv_to_parquet_with_metadata
    real_scan = ingestion_csv.pl.scan_csv
    staged_sources: list[str] = []
    scanned_sources: list[str] = []

    def count_staging(source, *args, **kwargs):
        staged_sources.append(str(source))
        return real_stage(source, *args, **kwargs)

    def count_csv_scan(source, *args, **kwargs):
        scanned_sources.append(str(source))
        return real_scan(source, *args, **kwargs)

    monkeypatch.setattr(assembly, "stage_csv_to_parquet_with_metadata", count_staging)
    monkeypatch.setattr(ingestion_csv.pl, "scan_csv", count_csv_scan)
    report = run_investigation_from_sources(request)

    assert staged_sources == [str(baseline), str(candidate)]
    assert len(scanned_sources) == len(request.datasets)
    assert len(set(scanned_sources)) == len(request.datasets)
    assert all(Path(source).name == "source.csv" for source in scanned_sources)
    assert [dataset.content_sha256 for dataset in report.datasets] == [
        hashlib.sha256(baseline_bytes).hexdigest(),
        hashlib.sha256(candidate_bytes).hexdigest(),
    ]
    assert report.summary.totals["records_added"] == 1
    assert report.summary.totals["records_modified"] == 1
    assert report.analysis_status == InvestigationStatus.COMPLETE


def test_source_compare_missing_file_preserves_partial_stage_outcomes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    baseline.write_text("id,value\n1,a\n", encoding="utf-8")
    missing = tmp_path / "missing.csv"
    request = SourceInvestigationRequest(
        operation="compare",
        datasets=(DatasetSource(path=baseline), DatasetSource(path=missing)),
    )

    report = run_investigation_from_sources(request)

    assert report.analysis_status == InvestigationStatus.PARTIAL
    assert report.datasets[0].content_sha256 == hashlib.sha256(baseline.read_bytes()).hexdigest()
    assert report.datasets[1].content_sha256 is None
    status_by_key = {(item.stage, item.dataset_id): item.status for item in report.stages}
    assert status_by_key[(AnalysisStage.PROFILE, report.datasets[1].dataset_id)] == (
        AnalysisStageStatus.FAILED
    )
    assert status_by_key[(AnalysisStage.SCHEMA, None)] == AnalysisStageStatus.SKIPPED
    assert status_by_key[(AnalysisStage.RECORDS, None)] == AnalysisStageStatus.SKIPPED


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("malformed.csv", b"a,b\n1,2\n3,4,5\n"),
        ("invalid-encoding.csv", b"name\nAda\xff\n"),
    ],
)
def test_source_failures_keep_hash_when_validation_read_completes(
    tmp_path: Path,
    name: str,
    content: bytes,
) -> None:
    source = tmp_path / name
    source.write_bytes(content)
    report = run_investigation_from_sources(
        SourceInvestigationRequest(
            operation="inspect",
            datasets=(DatasetSource(path=source),),
        )
    )

    assert report.analysis_status == InvestigationStatus.PARTIAL
    assert report.datasets[0].content_sha256 == hashlib.sha256(content).hexdigest()
    assert report.stages[0].status == AnalysisStageStatus.FAILED


def test_compare_assembles_schema_records_and_dataset_provenance(tmp_path: Path) -> None:
    baseline = _write_dataset(tmp_path, "baseline.csv", "id,value\n1,a\n2,b\n")
    candidate = _write_dataset(tmp_path, "candidate.csv", "id,value\n1,a\n3,c\n")
    request = InvestigationRequest(
        operation="compare",
        datasets=(baseline, candidate),
        comparison_options={"key_columns": ("id",), "example_limit": 2},
    )

    report = run_investigation(request)

    assert report.analysis_status == InvestigationStatus.COMPLETE
    assert {stage.stage for stage in report.stages} == {
        AnalysisStage.PROFILE,
        AnalysisStage.SCHEMA,
        AnalysisStage.RECORDS,
    }
    assert report.summary.totals["records_added"] == 1
    assert report.summary.totals["records_removed"] == 1
    record_findings = [
        finding
        for finding in report.findings
        if finding.kind in (FindingKind.RECORDS_ADDED, FindingKind.RECORDS_REMOVED)
    ]
    assert record_findings
    for finding in record_findings:
        assert finding.provenance is not None
        assert finding.provenance.stage == AnalysisStage.RECORDS
        assert finding.provenance.baseline_dataset_id == baseline.dataset_id
        assert finding.provenance.candidate_dataset_id == candidate.dataset_id


def test_failed_profile_marks_dependent_stages_skipped(tmp_path: Path) -> None:
    baseline = _write_dataset(tmp_path, "baseline.csv", "id,value\n1,a\n")
    invalid = _write_dataset(tmp_path, "invalid.csv", "id,value\n1,a,b\n")

    report = run_investigation(
        InvestigationRequest(operation="compare", datasets=(baseline, invalid))
    )

    assert report.analysis_status == InvestigationStatus.PARTIAL
    status_by_key = {(item.stage, item.dataset_id): item.status for item in report.stages}
    assert (
        status_by_key[(AnalysisStage.PROFILE, baseline.dataset_id)] == AnalysisStageStatus.COMPLETED
    )
    assert status_by_key[(AnalysisStage.PROFILE, invalid.dataset_id)] == AnalysisStageStatus.FAILED
    assert status_by_key[(AnalysisStage.SCHEMA, None)] == AnalysisStageStatus.SKIPPED
    assert status_by_key[(AnalysisStage.RECORDS, None)] == AnalysisStageStatus.SKIPPED


def test_snapshot_hash_mismatch_becomes_an_explicit_failed_profile_stage(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path, "changed.csv", "value\n1\n")
    request = InvestigationRequest(
        operation="inspect",
        datasets=(dataset.model_copy(update={"content_sha256": "0" * 64}),),
    )

    report = run_investigation(request)

    assert report.analysis_status == InvestigationStatus.PARTIAL
    stage = report.stages[0]
    assert stage.stage == AnalysisStage.PROFILE
    assert stage.status == AnalysisStageStatus.FAILED
    assert stage.error_code == "CsvIngestionError"
    assert "hash does not match" in stage.message


def test_report_json_is_deterministic_and_round_trips() -> None:
    report = _minimal_report()

    first = report_to_json(report)
    second = report_to_json(report)

    assert first == second
    assert first.index('"analysis_status"') < first.index('"created_at"')
    assert report_from_json(first) == report


def test_profile_finding_ids_are_unambiguous_for_delimiter_values() -> None:
    assert _profile_finding_id("nulls", "dataset-a", "b-c") != _profile_finding_id(
        "nulls", "dataset-a-b", "c"
    )


def test_sqlite_report_store_round_trips_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    store = ReportStore(tmp_path / "reports.sqlite3")
    report = _minimal_report()

    store.save(report)
    assert store.get(report.report_id) == report
    with pytest.raises(DuplicateReportIdError, match="already exists"):
        store.save(report)
    assert store.get(report.report_id) == report
    with pytest.raises(ReportNotFoundError, match="not found"):
        store.get("missing")
