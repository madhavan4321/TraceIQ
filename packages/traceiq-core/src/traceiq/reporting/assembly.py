"""Orchestrate shared core stages and assemble presentation-neutral reports."""

from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from traceiq.comparison.records import compare_records
from traceiq.comparison.schema import compare_schemas
from traceiq.ingestion.csv import (
    StagedCsv,
    profile_lazy_frame,
    stage_csv_to_parquet_with_metadata,
)
from traceiq.models import (
    AnalysisStage,
    AnalysisStageResult,
    AnalysisStageStatus,
    ComparisonOptions,
    ContentHashStatus,
    DatasetMetadata,
    DatasetProfile,
    DatasetProfileResult,
    DatasetReference,
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
    SchemaChangeKind,
    SchemaComparisonInput,
    SourceInvestigationRequest,
)


@dataclass(frozen=True, slots=True)
class _DatasetInput:
    dataset_id: str
    name: str
    snapshot_path: str
    expected_sha256: str | None
    metadata: DatasetMetadata | None


def run_investigation(request: InvestigationRequest) -> InvestigationReport:
    """Run an investigation from metadata-bearing dataset snapshots.

    Existing callers retain expected-hash verification. All analysis is delegated to the
    same internal runner used by :func:`run_investigation_from_sources`.
    """
    datasets = tuple(
        _DatasetInput(
            dataset_id=item.dataset_id,
            name=item.name,
            snapshot_path=item.snapshot_path,
            expected_sha256=item.content_sha256,
            metadata=item,
        )
        for item in request.datasets
    )
    return _run_investigation(
        request.operation,
        datasets,
        request.comparison_options,
        request.date_to_datetime_policy,
    )


def run_investigation_from_sources(
    request: SourceInvestigationRequest,
) -> InvestigationReport:
    """Run an investigation from CSV paths; the core creates all dataset metadata.

    Each source is UTF-8 validated and hashed in a bounded byte pass, then parsed and
    staged to temporary Parquet once. Profiling and comparison read the staged Parquet.
    Hashing and CSV parsing are separate physical reads, but no extra pre-hash or repeated
    CSV parsing pass is added by this API.
    """
    datasets = tuple(
        _DatasetInput(
            dataset_id=source.dataset_id or uuid.uuid4().hex,
            name=source.name or source.path.name or str(source.path),
            snapshot_path=str(source.path),
            expected_sha256=None,
            metadata=None,
        )
        for source in request.datasets
    )
    return _run_investigation(
        request.operation,
        datasets,
        request.comparison_options,
        request.date_to_datetime_policy,
    )


def _run_investigation(
    operation: str,
    datasets: tuple[_DatasetInput, ...],
    comparison_options: ComparisonOptions,
    date_to_datetime_policy: DateToDatetimePolicy,
) -> InvestigationReport:
    """Shared report orchestration for metadata and source-path entry points."""
    metadata_by_id: dict[str, DatasetMetadata] = {
        item.dataset_id: item.metadata
        for item in datasets
        if item.metadata is not None
    }
    profiles: dict[str, DatasetProfile] = {}
    frames: dict[str, pl.LazyFrame] = {}
    stage_results: list[AnalysisStageResult] = []
    findings: list[Finding] = []
    warnings: list[str] = []
    totals: dict[str, int | float] = {}

    with tempfile.TemporaryDirectory(prefix="traceiq-investigation-") as temp_dir:
        for index, item in enumerate(datasets):
            parquet_path = Path(temp_dir) / f"dataset-{index}.parquet"
            try:
                staged = stage_csv_to_parquet_with_metadata(
                    item.snapshot_path,
                    parquet_path,
                    expected_sha256=item.expected_sha256,
                )
                _record_source_metadata(metadata_by_id, item, staged)
                frame = pl.scan_parquet(staged.path)
                profile = profile_lazy_frame(frame, source=item.snapshot_path)
                frames[item.dataset_id] = frame
                profiles[item.dataset_id] = profile
                stage_results.append(
                    AnalysisStageResult(
                        stage=AnalysisStage.PROFILE,
                        status=AnalysisStageStatus.COMPLETED,
                        dataset_id=item.dataset_id,
                    )
                )
                findings.extend(
                    _profile_findings(
                        profile,
                        item.dataset_id,
                        baseline_id=datasets[0].dataset_id
                        if operation == "compare"
                        else None,
                        candidate_id=datasets[1].dataset_id
                        if operation == "compare"
                        else None,
                    )
                )
                totals[f"rows:{item.dataset_id}"] = profile.row_count
                totals[f"columns:{item.dataset_id}"] = profile.column_count
                totals[f"missing_values:{item.dataset_id}"] = sum(
                    column.null_count for column in profile.columns
                )
                totals[f"duplicate_rows:{item.dataset_id}"] = profile.duplicate_row_count
            except Exception as exc:  # Convert a failed stage into a partial report.
                _record_failure_metadata(metadata_by_id, item, exc)
                message = str(exc) or type(exc).__name__
                warnings.append(f"Profiling failed for {item.name}: {message}")
                stage_results.append(
                    AnalysisStageResult(
                        stage=AnalysisStage.PROFILE,
                        status=AnalysisStageStatus.FAILED,
                        dataset_id=item.dataset_id,
                        error_code=type(exc).__name__,
                        message=message,
                    )
                )

        if operation == "compare":
            baseline, candidate = datasets
            if len(profiles) != 2:
                reason = (
                    "Schema and record comparison skipped because profiling did not complete "
                    "for both datasets."
                )
                warnings.append(reason)
                stage_results.extend(
                    (
                        AnalysisStageResult(
                            stage=AnalysisStage.SCHEMA,
                            status=AnalysisStageStatus.SKIPPED,
                            message=reason,
                        ),
                        AnalysisStageResult(
                            stage=AnalysisStage.RECORDS,
                            status=AnalysisStageStatus.SKIPPED,
                            message=reason,
                        ),
                    )
                )
            else:
                schema_result = None
                try:
                    schema_result = compare_schemas(
                        _schema_input(
                            profiles[baseline.dataset_id],
                            profiles[candidate.dataset_id],
                            date_to_datetime_policy,
                        )
                    )
                    findings.extend(
                        _with_comparison_provenance(
                            schema_result.findings,
                            stage=AnalysisStage.SCHEMA,
                            baseline_id=baseline.dataset_id,
                            candidate_id=candidate.dataset_id,
                        )
                    )
                    stage_results.append(
                        AnalysisStageResult(
                            stage=AnalysisStage.SCHEMA,
                            status=AnalysisStageStatus.COMPLETED,
                        )
                    )
                    totals["schema_added_columns"] = schema_result.counts.added
                    totals["schema_removed_columns"] = schema_result.counts.removed
                    totals["schema_type_changes"] = schema_result.counts.type_changed
                except Exception as exc:
                    reason = str(exc) or type(exc).__name__
                    warnings.append(f"Schema comparison failed: {reason}")
                    stage_results.append(
                        AnalysisStageResult(
                            stage=AnalysisStage.SCHEMA,
                            status=AnalysisStageStatus.FAILED,
                            error_code=type(exc).__name__,
                            message=reason,
                        )
                    )

                if schema_result is None:
                    reason = "Record comparison skipped because schema comparison failed."
                    stage_results.append(
                        AnalysisStageResult(
                            stage=AnalysisStage.RECORDS,
                            status=AnalysisStageStatus.SKIPPED,
                            message=reason,
                        )
                    )
                else:
                    excluded_columns = tuple(
                        change.column_name
                        for change in schema_result.changes
                        if change.kind == SchemaChangeKind.TYPE_CHANGED
                    )
                    try:
                        record_result = compare_records(
                            frames[baseline.dataset_id],
                            frames[candidate.dataset_id],
                            comparison_options,
                            baseline_dataset_id=baseline.dataset_id,
                            candidate_dataset_id=candidate.dataset_id,
                            excluded_columns=excluded_columns,
                        )
                        findings.extend(
                            _with_comparison_provenance(
                                record_result.findings,
                                stage=AnalysisStage.RECORDS,
                                baseline_id=baseline.dataset_id,
                                candidate_id=candidate.dataset_id,
                            )
                        )
                        stage_results.append(
                            AnalysisStageResult(
                                stage=AnalysisStage.RECORDS,
                                status=AnalysisStageStatus.COMPLETED,
                            )
                        )
                        totals["records_added"] = record_result.counts.added
                        totals["records_removed"] = record_result.counts.removed
                        totals["records_modified"] = record_result.counts.modified
                    except Exception as exc:
                        reason = str(exc) or type(exc).__name__
                        warnings.append(f"Record comparison failed: {reason}")
                        stage_results.append(
                            AnalysisStageResult(
                                stage=AnalysisStage.RECORDS,
                                status=AnalysisStageStatus.FAILED,
                                error_code=type(exc).__name__,
                                message=reason,
                            )
                        )

    severity_counts = Counter(finding.severity for finding in findings)
    report_findings = tuple(findings)
    partial = any(stage.status != AnalysisStageStatus.COMPLETED for stage in stage_results)
    return InvestigationReport(
        report_id=uuid.uuid4().hex,
        operation=operation,
        datasets=tuple(
            DatasetReference(
                dataset_id=item.dataset_id,
                name=item.name,
                content_sha256=(
                    metadata_by_id[item.dataset_id].content_sha256
                    if item.dataset_id in metadata_by_id
                    else None
                ),
                content_hash_status=(
                    ContentHashStatus.AVAILABLE
                    if item.dataset_id in metadata_by_id
                    else ContentHashStatus.INCOMPLETE
                ),
            )
            for item in datasets
        ),
        dataset_profiles=tuple(
            DatasetProfileResult(dataset_id=dataset_id, profile=profiles[dataset_id])
            for dataset_id in sorted(profiles)
        ),
        summary=InvestigationSummary(
            finding_count=len(report_findings),
            counts_by_severity={
                severity: severity_counts[severity] for severity in FindingSeverity
            },
            totals=totals,
        ),
        findings=report_findings,
        warnings=tuple(warnings),
        stages=tuple(stage_results),
        analysis_status=InvestigationStatus.PARTIAL if partial else InvestigationStatus.COMPLETE,
        example_limit=comparison_options.example_limit,
    )


def _record_source_metadata(
    metadata_by_id: dict[str, DatasetMetadata],
    item: _DatasetInput,
    staged: StagedCsv,
) -> None:
    if item.metadata is None:
        metadata_by_id[item.dataset_id] = DatasetMetadata(
            dataset_id=item.dataset_id,
            name=item.name,
            snapshot_path=item.snapshot_path,
            size_bytes=staged.size_bytes,
            content_sha256=staged.content_sha256,
            created_at=datetime.now(UTC),
        )


def _record_failure_metadata(
    metadata_by_id: dict[str, DatasetMetadata],
    item: _DatasetInput,
    error: Exception,
) -> None:
    if item.metadata is not None or item.dataset_id in metadata_by_id:
        return
    content_sha256 = getattr(error, "content_sha256", None)
    size_bytes = getattr(error, "size_bytes", None)
    if content_sha256 is not None and size_bytes is not None:
        metadata_by_id[item.dataset_id] = DatasetMetadata(
            dataset_id=item.dataset_id,
            name=item.name,
            snapshot_path=item.snapshot_path,
            size_bytes=size_bytes,
            content_sha256=content_sha256,
            created_at=datetime.now(UTC),
        )


def _schema_input(
    baseline: DatasetProfile,
    candidate: DatasetProfile,
    date_to_datetime_policy: DateToDatetimePolicy,
) -> SchemaComparisonInput:
    return SchemaComparisonInput(
        baseline=baseline,
        candidate=candidate,
        date_to_datetime_policy=date_to_datetime_policy,
    )


def _profile_findings(
    profile: DatasetProfile,
    dataset_id: str,
    *,
    baseline_id: str | None = None,
    candidate_id: str | None = None,
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    for column in sorted(profile.columns, key=lambda item: item.name):
        if column.null_count:
            counts = {"missing_values": column.null_count}
            findings.append(
                Finding(
                    finding_id=_profile_finding_id("nulls", dataset_id, column.name),
                    kind=FindingKind.MISSING_VALUES,
                    severity=FindingSeverity.WARNING,
                    message=f"Column {column.name!r} contains missing values",
                    affected_columns=(column.name,),
                    counts=counts,
                    evidence=FindingEvidence(
                        method="profile_null_count",
                        counts=counts,
                        details={
                            "null_count": column.null_count,
                            "null_percentage": column.null_percentage,
                        },
                    ),
                    provenance=FindingProvenance(
                        stage=AnalysisStage.PROFILE,
                        dataset_id=dataset_id,
                        baseline_dataset_id=baseline_id,
                        candidate_dataset_id=candidate_id,
                        method="profile_null_count",
                    ),
                )
            )
    if profile.duplicate_row_count:
        counts = {"duplicate_rows": profile.duplicate_row_count}
        findings.append(
            Finding(
                finding_id=_profile_finding_id("duplicates", dataset_id),
                kind=FindingKind.DUPLICATE_RECORDS,
                severity=FindingSeverity.WARNING,
                message="Dataset contains duplicate rows",
                affected_columns=tuple(column.name for column in profile.columns),
                counts=counts,
                evidence=FindingEvidence(
                    method="profile_duplicate_row_count",
                    counts=counts,
                    details={"duplicate_semantics": "excess_occurrences"},
                ),
                provenance=FindingProvenance(
                    stage=AnalysisStage.PROFILE,
                    dataset_id=dataset_id,
                    baseline_dataset_id=baseline_id,
                    candidate_dataset_id=candidate_id,
                    method="profile_duplicate_row_count",
                ),
            )
        )
    return tuple(findings)


def _profile_finding_id(kind: str, dataset_id: str, column_name: str | None = None) -> str:
    identity = json.dumps(
        (kind, dataset_id, column_name), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:20]
    return f"profile-{kind}-{digest}"


def _with_comparison_provenance(
    findings: tuple[Finding, ...],
    *,
    stage: AnalysisStage,
    baseline_id: str,
    candidate_id: str,
) -> tuple[Finding, ...]:
    enriched: list[Finding] = []
    for finding in findings:
        expected = FindingProvenance(
            stage=stage,
            method=finding.evidence.method,
            baseline_dataset_id=baseline_id,
            candidate_dataset_id=candidate_id,
        )
        if finding.provenance is not None and finding.provenance != expected:
            raise ValueError(
                f"finding {finding.finding_id!r} has provenance inconsistent with its report stage"
            )
        enriched.append(finding.model_copy(update={"provenance": expected}))
    return tuple(enriched)
