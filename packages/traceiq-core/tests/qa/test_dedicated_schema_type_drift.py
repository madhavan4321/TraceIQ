"""Deterministic logical-type drift fixture verification."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from traceiq.ingestion import profile_csv
from traceiq.models import (
    ComparisonOptions,
    DatasetMetadata,
    FindingKind,
    InvestigationRequest,
    LogicalDataType,
)
from traceiq.reporting.assembly import run_investigation

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[4] / "examples" / "test-data" / "traceiq_schema_type_drift"
)


def _metadata(path: Path, dataset_id: str) -> DatasetMetadata:
    data = path.read_bytes()
    return DatasetMetadata(
        dataset_id=dataset_id,
        name=path.name,
        snapshot_path=str(path),
        size_bytes=len(data),
        content_sha256=hashlib.sha256(data).hexdigest(),
        created_at=datetime(2026, 10, 31, tzinfo=UTC),
    )


def test_amount_type_drift_is_deterministic_under_current_inference() -> None:
    baseline_path = FIXTURE_ROOT / "baseline.csv"
    candidate_path = FIXTURE_ROOT / "candidate.csv"

    # Verify the actual inferred input types before asserting the comparison result.
    baseline_profile = profile_csv(baseline_path)
    candidate_profile = profile_csv(candidate_path)
    assert baseline_profile.columns[1].name == "amount"
    assert candidate_profile.columns[1].name == "amount"
    assert baseline_profile.columns[1].logical_type == LogicalDataType.FLOAT
    assert candidate_profile.columns[1].logical_type == LogicalDataType.STRING

    report = run_investigation(
        InvestigationRequest(
            operation="compare",
            datasets=(
                _metadata(baseline_path, "type-drift-baseline"),
                _metadata(candidate_path, "type-drift-candidate"),
            ),
            comparison_options=ComparisonOptions(key_columns=("transaction_id",)),
        )
    )

    assert report.summary.totals["schema_type_changes"] == 1
    finding = next(
        finding for finding in report.findings if finding.kind == FindingKind.SCHEMA_TYPE_CHANGED
    )
    assert finding.affected_columns == ("amount",)
    assert finding.evidence.details["before_type"] == "float"
    assert finding.evidence.details["after_type"] == "string"
