"""Deterministic JSON serialization for investigation reports."""

from __future__ import annotations

import json

from traceiq.models import InvestigationReport


def report_to_json(report: InvestigationReport, *, indent: int | None = 2) -> str:
    """Serialize a report with stable object-key ordering and no non-JSON values."""
    if indent is not None and indent < 0:
        raise ValueError("indent must be non-negative or None")
    return json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        allow_nan=False,
    )


def report_from_json(payload: str | bytes | bytearray) -> InvestigationReport:
    """Validate serialized JSON into the public report contract."""
    return InvestigationReport.model_validate_json(payload)
