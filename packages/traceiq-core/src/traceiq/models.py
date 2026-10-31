"""Stable, presentation-independent Pydantic contracts for TraceIQ investigations."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TraceIQModel(BaseModel):
    """Base for strict, serializable public data models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetFormat(StrEnum):
    CSV = "csv"


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FindingKind(StrEnum):
    SCHEMA_ADDED_COLUMN = "schema.added_column"
    SCHEMA_REMOVED_COLUMN = "schema.removed_column"
    SCHEMA_TYPE_CHANGED = "schema.type_changed"
    RECORDS_ADDED = "records.added"
    RECORDS_REMOVED = "records.removed"
    RECORD_MODIFIED = "records.modified"
    DUPLICATE_RECORDS = "data.duplicate_records"
    MISSING_VALUES = "data.missing_values"
    INVALID_COMPARISON_KEY = "comparison.invalid_key"
    RECORDS_NOT_COMPARABLE = "comparison.records_not_comparable"
    RECORD_COLUMNS_UNCOMPARABLE = "comparison.record_columns_uncomparable"


class LogicalDataType(StrEnum):
    """Stable logical types shared by profiling and schema comparison."""

    NULL = "null"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    STRING = "string"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    DURATION = "duration"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    LIST = "list"
    STRUCT = "struct"
    UNKNOWN = "unknown"


class SchemaChangeKind(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    TYPE_CHANGED = "type_changed"


class TypeCompatibility(StrEnum):
    UNCHANGED = "unchanged"
    COMPATIBLE = "compatible"
    POTENTIALLY_LOSSY = "potentially_lossy"
    INCOMPATIBLE = "incompatible"


class DateToDatetimePolicy(StrEnum):
    DISALLOW = "disallow"
    MIDNIGHT = "midnight"


class AnalysisStage(StrEnum):
    PROFILE = "profile"
    SCHEMA = "schema"
    RECORDS = "records"


class AnalysisStageStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class InvestigationStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class ContentHashStatus(StrEnum):
    AVAILABLE = "available"
    INCOMPLETE = "incomplete"


class RecordComparisonMode(StrEnum):
    KEYED = "keyed"
    MULTISET = "multiset"


class RecordComparisonStatus(StrEnum):
    COMPARABLE = "comparable"
    NON_COMPARABLE = "non_comparable"
    INVALID_KEYS = "invalid_keys"


class DatasetMetadata(TraceIQModel):
    """Reference to a persisted dataset snapshot; paths are local storage details."""

    dataset_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    format: DatasetFormat = DatasetFormat.CSV
    snapshot_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    created_at: datetime


class ColumnProfile(TraceIQModel):
    name: str = Field(min_length=1)
    logical_type: LogicalDataType
    null_count: int = Field(default=0, ge=0)
    null_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    distinct_count: int | None = Field(default=None, ge=0)


class DatasetProfile(TraceIQModel):
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: tuple[ColumnProfile, ...] = ()
    duplicate_row_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def column_count_matches_profiles(self) -> DatasetProfile:
        if self.column_count != len(self.columns):
            raise ValueError("column_count must equal the number of column profiles")
        if self.duplicate_row_count > self.row_count:
            raise ValueError("duplicate_row_count cannot exceed row_count")
        for column in self.columns:
            expected_percentage = (
                100.0 * column.null_count / self.row_count if self.row_count else 0.0
            )
            if (
                column.null_count > self.row_count
                or abs(column.null_percentage - expected_percentage) > 1e-6
            ):
                raise ValueError(
                    f"null_percentage for column {column.name!r} must match "
                    "null_count and row_count"
                )
        return self


class DatasetReference(TraceIQModel):
    dataset_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    content_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    content_hash_status: ContentHashStatus = ContentHashStatus.AVAILABLE

    @model_validator(mode="after")
    def hash_status_matches_digest(self) -> DatasetReference:
        has_hash = self.content_sha256 is not None
        if (self.content_hash_status == ContentHashStatus.AVAILABLE) != has_hash:
            raise ValueError("content_hash_status must match content_sha256 availability")
        return self


class ComparisonOptions(TraceIQModel):
    key_columns: tuple[str, ...] = ()
    example_limit: int = Field(default=100, ge=0, le=10_000)
    # These policies are part of the contract to make subsequent comparison behavior explicit.
    null_key_policy: str = Field(default="report_invalid", pattern=r"^report_invalid$")
    duplicate_key_policy: str = Field(default="report_invalid", pattern=r"^report_invalid$")
    row_order_policy: str = Field(default="ignored", pattern=r"^ignored$")
    type_normalization: str = Field(default="logical", pattern=r"^logical$")


class InvestigationRequest(TraceIQModel):
    """Inputs for a local inspection or two-snapshot investigation."""

    operation: str = Field(pattern=r"^(inspect|compare)$")
    datasets: tuple[DatasetMetadata, ...] = Field(min_length=1, max_length=2)
    comparison_options: ComparisonOptions = Field(default_factory=ComparisonOptions)
    date_to_datetime_policy: DateToDatetimePolicy = DateToDatetimePolicy.DISALLOW

    @model_validator(mode="after")
    def operation_matches_dataset_count(self) -> InvestigationRequest:
        expected = 1 if self.operation == "inspect" else 2
        if len(self.datasets) != expected:
            raise ValueError(f"{self.operation} requires exactly {expected} dataset(s)")
        if len({dataset.dataset_id for dataset in self.datasets}) != len(self.datasets):
            raise ValueError("dataset IDs must be unique within an investigation")
        return self


class DatasetSource(TraceIQModel):
    """A local CSV path for core-owned ingestion and metadata creation."""

    path: Path
    dataset_id: str | None = Field(default=None, min_length=1)
    name: str | None = Field(default=None, min_length=1)


class SourceInvestigationRequest(TraceIQModel):
    """Inputs for an investigation where the core computes dataset metadata."""

    operation: str = Field(pattern=r"^(inspect|compare)$")
    datasets: tuple[DatasetSource, ...] = Field(min_length=1, max_length=2)
    comparison_options: ComparisonOptions = Field(default_factory=ComparisonOptions)
    date_to_datetime_policy: DateToDatetimePolicy = DateToDatetimePolicy.DISALLOW

    @model_validator(mode="after")
    def operation_matches_source_count(self) -> SourceInvestigationRequest:
        expected = 1 if self.operation == "inspect" else 2
        if len(self.datasets) != expected:
            raise ValueError(f"{self.operation} requires exactly {expected} dataset(s)")
        dataset_ids = [source.dataset_id for source in self.datasets if source.dataset_id]
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("provided dataset IDs must be unique within an investigation")
        return self


class SchemaComparisonInput(TraceIQModel):
    baseline: DatasetProfile
    candidate: DatasetProfile
    date_to_datetime_policy: DateToDatetimePolicy = DateToDatetimePolicy.DISALLOW


class SchemaChange(TraceIQModel):
    column_name: str = Field(min_length=1)
    kind: SchemaChangeKind
    before_type: LogicalDataType | None = None
    after_type: LogicalDataType | None = None
    compatibility: TypeCompatibility | None = None

    @model_validator(mode="after")
    def types_match_change_kind(self) -> SchemaChange:
        if self.kind == SchemaChangeKind.ADDED:
            valid = self.before_type is None and self.after_type is not None
        elif self.kind == SchemaChangeKind.REMOVED:
            valid = self.before_type is not None and self.after_type is None
        else:
            valid = (
                self.before_type is not None
                and self.after_type is not None
                and self.before_type != self.after_type
                and self.compatibility is not None
                and self.compatibility != TypeCompatibility.UNCHANGED
            )
        if not valid:
            raise ValueError("before/after types and compatibility must match the change kind")
        if self.kind != SchemaChangeKind.TYPE_CHANGED and self.compatibility is not None:
            raise ValueError("compatibility applies only to type changes")
        return self


class SchemaComparisonCounts(TraceIQModel):
    added: int = Field(ge=0)
    removed: int = Field(ge=0)
    type_changed: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    compatible: int = Field(ge=0)
    potentially_lossy: int = Field(ge=0)
    incompatible: int = Field(ge=0)


class FindingEvidence(TraceIQModel):
    """Structured, machine-readable evidence; values are JSON-compatible scalars/containers."""

    method: str = Field(min_length=1)
    counts: dict[str, int | float] = Field(default_factory=dict)
    examples: tuple[dict[str, Any], ...] = ()
    details: dict[str, Any] = Field(default_factory=dict)


class FindingProvenance(TraceIQModel):
    """Machine-readable source stage and dataset context for a finding."""

    stage: AnalysisStage
    method: str = Field(min_length=1)
    dataset_id: str | None = None
    baseline_dataset_id: str | None = None
    candidate_dataset_id: str | None = None

    @model_validator(mode="after")
    def provenance_has_relevant_dataset_ids(self) -> FindingProvenance:
        if self.stage == AnalysisStage.PROFILE and self.dataset_id is None:
            raise ValueError("profile provenance requires dataset_id")
        if (self.baseline_dataset_id is None) != (self.candidate_dataset_id is None):
            raise ValueError("baseline and candidate IDs must be provided together")
        if self.stage in (AnalysisStage.SCHEMA, AnalysisStage.RECORDS) and (
            self.baseline_dataset_id is None or self.candidate_dataset_id is None
        ):
            raise ValueError("comparison provenance requires baseline and candidate dataset IDs")
        return self


class Finding(TraceIQModel):
    finding_id: str = Field(min_length=1)
    kind: FindingKind
    severity: FindingSeverity
    message: str = Field(min_length=1)
    affected_columns: tuple[str, ...] = ()
    counts: dict[str, int | float]
    evidence: FindingEvidence
    provenance: FindingProvenance | None = None

    @model_validator(mode="after")
    def finding_has_counts(self) -> Finding:
        if not self.counts:
            raise ValueError("each finding must include relevant counts")
        if self.provenance is not None:
            expected_stages = {
                FindingKind.SCHEMA_ADDED_COLUMN: AnalysisStage.SCHEMA,
                FindingKind.SCHEMA_REMOVED_COLUMN: AnalysisStage.SCHEMA,
                FindingKind.SCHEMA_TYPE_CHANGED: AnalysisStage.SCHEMA,
                FindingKind.RECORDS_ADDED: AnalysisStage.RECORDS,
                FindingKind.RECORDS_REMOVED: AnalysisStage.RECORDS,
                FindingKind.RECORD_MODIFIED: AnalysisStage.RECORDS,
                FindingKind.INVALID_COMPARISON_KEY: AnalysisStage.RECORDS,
                FindingKind.RECORDS_NOT_COMPARABLE: AnalysisStage.RECORDS,
                FindingKind.RECORD_COLUMNS_UNCOMPARABLE: AnalysisStage.RECORDS,
                FindingKind.DUPLICATE_RECORDS: AnalysisStage.PROFILE,
                FindingKind.MISSING_VALUES: AnalysisStage.PROFILE,
            }
            expected_stage = expected_stages.get(self.kind)
            if expected_stage is not None and self.provenance.stage != expected_stage:
                raise ValueError("finding provenance stage does not match finding kind")
            if self.evidence.method != self.provenance.method:
                raise ValueError("finding provenance method must match evidence method")
        return self


class SchemaComparisonResult(TraceIQModel):
    changes: tuple[SchemaChange, ...] = ()
    findings: tuple[Finding, ...] = ()
    counts: SchemaComparisonCounts


class RecordComparisonCounts(TraceIQModel):
    """Comparison totals; invalid_key_rows counts null rows and duplicate-key group rows."""

    added: int = Field(default=0, ge=0)
    removed: int = Field(default=0, ge=0)
    modified: int = Field(default=0, ge=0)
    invalid_key_rows: int = Field(default=0, ge=0)


class RecordComparisonResult(TraceIQModel):
    mode: RecordComparisonMode
    status: RecordComparisonStatus
    comparable_columns: tuple[str, ...] = ()
    counts: RecordComparisonCounts = Field(default_factory=RecordComparisonCounts)
    findings: tuple[Finding, ...] = ()
    example_limit: int = Field(default=100, ge=0, le=10_000)

    @model_validator(mode="after")
    def result_matches_comparison_status(self) -> RecordComparisonResult:
        if self.status == RecordComparisonStatus.NON_COMPARABLE and not any(
            finding.kind == FindingKind.RECORDS_NOT_COMPARABLE for finding in self.findings
        ):
            raise ValueError("non-comparable results require a structured finding")
        if self.status == RecordComparisonStatus.INVALID_KEYS and not any(
            finding.kind == FindingKind.INVALID_COMPARISON_KEY for finding in self.findings
        ):
            raise ValueError("invalid-key results require a structured finding")
        if any(
            finding.provenance is None or finding.provenance.stage != AnalysisStage.RECORDS
            for finding in self.findings
        ):
            raise ValueError("record comparison findings require record-stage provenance")
        if any(len(finding.evidence.examples) > self.example_limit for finding in self.findings):
            raise ValueError("a finding exceeds the configured per-finding example limit")
        return self


class DatasetProfileResult(TraceIQModel):
    dataset_id: str = Field(min_length=1)
    profile: DatasetProfile


class AnalysisStageResult(TraceIQModel):
    stage: AnalysisStage
    status: AnalysisStageStatus
    dataset_id: str | None = None
    error_code: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def status_has_context(self) -> AnalysisStageResult:
        if self.stage == AnalysisStage.PROFILE and self.dataset_id is None:
            raise ValueError("profile stage results require dataset_id")
        if self.stage != AnalysisStage.PROFILE and self.dataset_id is not None:
            raise ValueError("comparison stage results cannot have a single dataset_id")
        if (
            self.status in (AnalysisStageStatus.FAILED, AnalysisStageStatus.SKIPPED)
            and not self.message
        ):
            raise ValueError("failed and skipped stages require a message")
        if self.status != AnalysisStageStatus.FAILED and self.error_code is not None:
            raise ValueError("error_code is valid only for failed stages")
        return self


class InvestigationSummary(TraceIQModel):
    finding_count: int = Field(ge=0)
    counts_by_severity: dict[FindingSeverity, int] = Field(default_factory=dict)
    totals: dict[str, int | float] = Field(default_factory=dict)


class InvestigationReport(TraceIQModel):
    """Analysis result only; terminal/web formatting belongs to each interface."""

    report_id: str = Field(min_length=1)
    operation: str = Field(pattern=r"^(inspect|compare)$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    datasets: tuple[DatasetReference, ...] = Field(min_length=1, max_length=2)
    dataset_profiles: tuple[DatasetProfileResult, ...] = ()
    summary: InvestigationSummary
    findings: tuple[Finding, ...] = ()
    warnings: tuple[str, ...] = ()
    stages: tuple[AnalysisStageResult, ...] = ()
    analysis_status: InvestigationStatus = InvestigationStatus.COMPLETE
    example_limit: int = Field(default=100, ge=0, le=10_000)

    @model_validator(mode="after")
    def findings_respect_example_limit(self) -> InvestigationReport:
        expected_dataset_count = 1 if self.operation == "inspect" else 2
        if len(self.datasets) != expected_dataset_count:
            raise ValueError(
                f"{self.operation} reports must reference exactly "
                f"{expected_dataset_count} dataset(s)"
            )
        if any(len(f.evidence.examples) > self.example_limit for f in self.findings):
            raise ValueError("a finding exceeds the configured per-finding example limit")
        if self.summary.finding_count != len(self.findings):
            raise ValueError("summary.finding_count must equal the number of findings")
        if any(finding.provenance is None for finding in self.findings):
            raise ValueError("report findings require structured provenance")
        dataset_id_set = {dataset.dataset_id for dataset in self.datasets}
        for finding in self.findings:
            provenance = finding.provenance
            referenced_ids = {
                dataset_id
                for dataset_id in (
                    provenance.dataset_id,
                    provenance.baseline_dataset_id,
                    provenance.candidate_dataset_id,
                )
                if dataset_id is not None
            }
            if not referenced_ids.issubset(dataset_id_set):
                raise ValueError("finding provenance must reference report datasets")
        incomplete = any(stage.status != AnalysisStageStatus.COMPLETED for stage in self.stages)
        if incomplete != (self.analysis_status == InvestigationStatus.PARTIAL):
            raise ValueError("analysis_status must reflect failed or skipped stages")
        stage_keys = [(stage.stage, stage.dataset_id) for stage in self.stages]
        if len(stage_keys) != len(set(stage_keys)):
            raise ValueError("analysis stage outcomes must be unique")
        dataset_ids = dataset_id_set
        if len(dataset_ids) != len(self.datasets):
            raise ValueError("report dataset IDs must be unique")
        if self.operation == "inspect":
            expected_keys = {(AnalysisStage.PROFILE, next(iter(dataset_ids)))}
        else:
            expected_keys = {
                *((AnalysisStage.PROFILE, dataset_id) for dataset_id in dataset_ids),
                (AnalysisStage.SCHEMA, None),
                (AnalysisStage.RECORDS, None),
            }
        if set(stage_keys) != expected_keys:
            raise ValueError("stage outcomes must cover all stages required by the operation")
        failed_profile_ids = {
            stage.dataset_id
            for stage in self.stages
            if stage.stage == AnalysisStage.PROFILE
            and stage.status == AnalysisStageStatus.FAILED
        }
        incomplete_hash_ids = {
            dataset.dataset_id
            for dataset in self.datasets
            if dataset.content_hash_status == ContentHashStatus.INCOMPLETE
        }
        if not incomplete_hash_ids.issubset(failed_profile_ids):
            raise ValueError("an incomplete content hash requires a failed profile stage")
        profile_ids = [profile.dataset_id for profile in self.dataset_profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("dataset profile IDs must be unique")
        if not set(profile_ids).issubset({dataset.dataset_id for dataset in self.datasets}):
            raise ValueError("dataset profiles must reference report datasets")
        completed_profile_ids = {
            stage.dataset_id
            for stage in self.stages
            if stage.stage == AnalysisStage.PROFILE
            and stage.status == AnalysisStageStatus.COMPLETED
        }
        if not completed_profile_ids.issubset(set(profile_ids)):
            raise ValueError("completed profile stages require a dataset profile")
        return self
