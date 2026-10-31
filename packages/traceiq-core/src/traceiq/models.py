"""Stable, presentation-independent Pydantic contracts for TraceIQ investigations."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
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
    logical_type: str = Field(min_length=1)
    null_count: int = Field(default=0, ge=0)
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
        return self


class DatasetReference(TraceIQModel):
    dataset_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")


class ComparisonOptions(TraceIQModel):
    key_columns: tuple[str, ...] = ()
    example_limit: int = Field(default=100, ge=0, le=10_000)
    # These policies are part of the contract to make subsequent comparison behavior explicit.
    null_key_policy: str = Field(default="report_invalid", pattern=r"^report_invalid$")
    duplicate_key_policy: str = Field(default="report_invalid", pattern=r"^report_invalid$")
    row_order_policy: str = Field(default="ignored", pattern=r"^ignored$")
    type_normalization: str = Field(default="logical", pattern=r"^logical$")


class FindingEvidence(TraceIQModel):
    """Structured, machine-readable evidence; values are JSON-compatible scalars/containers."""

    method: str = Field(min_length=1)
    counts: dict[str, int | float] = Field(default_factory=dict)
    examples: tuple[dict[str, Any], ...] = ()
    details: dict[str, Any] = Field(default_factory=dict)


class Finding(TraceIQModel):
    finding_id: str = Field(min_length=1)
    kind: FindingKind
    severity: FindingSeverity
    message: str = Field(min_length=1)
    affected_columns: tuple[str, ...] = ()
    counts: dict[str, int | float]
    evidence: FindingEvidence

    @model_validator(mode="after")
    def finding_has_counts(self) -> Finding:
        if not self.counts:
            raise ValueError("each finding must include relevant counts")
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
    summary: InvestigationSummary
    findings: tuple[Finding, ...] = ()
    warnings: tuple[str, ...] = ()
    example_limit: int = Field(default=100, ge=0, le=10_000)

    @model_validator(mode="after")
    def findings_respect_example_limit(self) -> InvestigationReport:
        expected_dataset_count = 1 if self.operation == "inspect" else 2
        if len(self.datasets) != expected_dataset_count:
            raise ValueError(
                f"{self.operation} reports must reference exactly {expected_dataset_count} dataset(s)"
            )
        if any(len(f.evidence.examples) > self.example_limit for f in self.findings):
            raise ValueError("a finding exceeds the configured per-finding example limit")
        if self.summary.finding_count != len(self.findings):
            raise ValueError("summary.finding_count must equal the number of findings")
        return self
