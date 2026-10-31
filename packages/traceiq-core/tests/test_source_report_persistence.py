"""Report hash-state validation and persistence compatibility tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from traceiq.models import (
    AnalysisStage,
    AnalysisStageResult,
    AnalysisStageStatus,
    ContentHashStatus,
    DatasetProfile,
    DatasetProfileResult,
    DatasetReference,
    InvestigationReport,
    InvestigationStatus,
    InvestigationSummary,
)
from traceiq.reporting.json import report_from_json, report_to_json
from traceiq.storage.reports import ReportStore

HASH = "c" * 64


def _partial_report(report_id: str = "partial-no-hash") -> InvestigationReport:
    return InvestigationReport(
        report_id=report_id,
        operation="inspect",
        datasets=(
            DatasetReference(
                dataset_id="missing-source",
                name="missing.csv",
                content_sha256=None,
                content_hash_status=ContentHashStatus.INCOMPLETE,
            ),
        ),
        summary=InvestigationSummary(finding_count=0),
        stages=(
            AnalysisStageResult(
                stage=AnalysisStage.PROFILE,
                status=AnalysisStageStatus.FAILED,
                dataset_id="missing-source",
                error_code="CsvIngestionError",
                message="Source could not be opened before hashing completed",
            ),
        ),
        analysis_status=InvestigationStatus.PARTIAL,
    )


def _complete_report() -> InvestigationReport:
    return InvestigationReport(
        report_id="complete-hash",
        operation="inspect",
        datasets=(DatasetReference(dataset_id="source", name="source.csv", content_sha256=HASH),),
        dataset_profiles=(
            DatasetProfileResult(
                dataset_id="source",
                profile=DatasetProfile(row_count=0, column_count=0),
            ),
        ),
        summary=InvestigationSummary(finding_count=0),
        stages=(
            AnalysisStageResult(
                stage=AnalysisStage.PROFILE,
                status=AnalysisStageStatus.COMPLETED,
                dataset_id="source",
            ),
        ),
    )


def test_hash_status_requires_digest_and_complete_profile_hash() -> None:
    with pytest.raises(ValidationError, match="content_hash_status"):
        DatasetReference(dataset_id="source", name="source.csv", content_sha256=None)
    with pytest.raises(ValidationError, match="content_hash_status"):
        DatasetReference(
            dataset_id="source",
            name="source.csv",
            content_sha256=HASH,
            content_hash_status=ContentHashStatus.INCOMPLETE,
        )

    valid_complete = _complete_report()
    assert valid_complete.datasets[0].content_hash_status == ContentHashStatus.AVAILABLE


def test_incomplete_hash_requires_failed_profile_stage() -> None:
    dataset = DatasetReference(
        dataset_id="source",
        name="source.csv",
        content_sha256=None,
        content_hash_status=ContentHashStatus.INCOMPLETE,
    )
    with pytest.raises(ValidationError, match="requires a failed profile stage"):
        InvestigationReport(
            report_id="invalid-hash-state",
            operation="inspect",
            datasets=(dataset,),
            dataset_profiles=(
                DatasetProfileResult(
                    dataset_id="source",
                    profile=DatasetProfile(row_count=0, column_count=0),
                ),
            ),
            summary=InvestigationSummary(finding_count=0),
            stages=(
                AnalysisStageResult(
                    stage=AnalysisStage.PROFILE,
                    status=AnalysisStageStatus.COMPLETED,
                    dataset_id="source",
                ),
            ),
        )


def test_incomplete_hash_partial_report_round_trips_through_json_and_sqlite(
    tmp_path,
) -> None:
    report = _partial_report()

    encoded = report_to_json(report)
    decoded = report_from_json(encoded)
    assert decoded == report
    assert decoded.datasets[0].content_sha256 is None
    assert decoded.datasets[0].content_hash_status == ContentHashStatus.INCOMPLETE

    store = ReportStore(tmp_path / "partial-reports.sqlite3")
    store.save(report)
    assert store.get(report.report_id) == report


def test_hash_present_legacy_json_without_status_remains_readable() -> None:
    legacy = json.loads(report_to_json(_complete_report()))
    del legacy["datasets"][0]["content_hash_status"]

    decoded = report_from_json(json.dumps(legacy))

    assert decoded.datasets[0].content_sha256 == HASH
    assert decoded.datasets[0].content_hash_status == ContentHashStatus.AVAILABLE
