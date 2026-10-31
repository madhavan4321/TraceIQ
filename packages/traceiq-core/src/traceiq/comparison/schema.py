"""Deterministic schema drift detection over shared dataset profile contracts."""

from __future__ import annotations

import hashlib

from traceiq.models import (
    ColumnProfile,
    DateToDatetimePolicy,
    Finding,
    FindingEvidence,
    FindingKind,
    FindingSeverity,
    LogicalDataType,
    SchemaChange,
    SchemaChangeKind,
    SchemaComparisonCounts,
    SchemaComparisonInput,
    SchemaComparisonResult,
    TypeCompatibility,
)


class SchemaComparisonError(ValueError):
    """Raised when a dataset profile cannot describe a unique column schema."""


def compare_schemas(comparison: SchemaComparisonInput) -> SchemaComparisonResult:
    """Compare column names/types without depending on source column ordering.

    Compatibility is deliberately conservative:

    * Same logical types are unchanged.
    * Categorical to string is compatible because category labels map directly to strings.
    * Integer/decimal/float transitions are potentially lossy: profile contracts do not
      include value ranges, decimal precision, or scale needed to prove safety.
    * Date to datetime is incompatible unless the caller explicitly selects the midnight
      conversion policy; even then it remains potentially lossy because timezone semantics
      are not represented in the current profile.
    * Other changes, including string to boolean, are incompatible absent a validated
      conversion rule and value-level evidence.

    Findings and changes are emitted in case-sensitive Unicode lexical column-name order.
    """
    baseline = _columns_by_name(comparison.baseline.columns, "baseline")
    candidate = _columns_by_name(comparison.candidate.columns, "candidate")

    changes: list[SchemaChange] = []
    findings: list[Finding] = []
    added = removed = type_changed = unchanged = 0
    compatible = potentially_lossy = incompatible = 0

    for column_name in sorted(baseline.keys() | candidate.keys()):
        before = baseline.get(column_name)
        after = candidate.get(column_name)
        if before is None:
            change = SchemaChange(
                column_name=column_name,
                kind=SchemaChangeKind.ADDED,
                after_type=after.logical_type,
            )
            added += 1
        elif after is None:
            change = SchemaChange(
                column_name=column_name,
                kind=SchemaChangeKind.REMOVED,
                before_type=before.logical_type,
            )
            removed += 1
        elif before.logical_type == after.logical_type:
            unchanged += 1
            continue
        else:
            compatibility = _classify_type_change(
                before.logical_type,
                after.logical_type,
                comparison.date_to_datetime_policy,
            )
            change = SchemaChange(
                column_name=column_name,
                kind=SchemaChangeKind.TYPE_CHANGED,
                before_type=before.logical_type,
                after_type=after.logical_type,
                compatibility=compatibility,
            )
            type_changed += 1
            if compatibility == TypeCompatibility.COMPATIBLE:
                compatible += 1
            elif compatibility == TypeCompatibility.POTENTIALLY_LOSSY:
                potentially_lossy += 1
            else:
                incompatible += 1

        changes.append(change)
        findings.append(_finding_for_change(change))

    return SchemaComparisonResult(
        changes=tuple(changes),
        findings=tuple(findings),
        counts=SchemaComparisonCounts(
            added=added,
            removed=removed,
            type_changed=type_changed,
            unchanged=unchanged,
            compatible=compatible,
            potentially_lossy=potentially_lossy,
            incompatible=incompatible,
        ),
    )


def _columns_by_name(columns: tuple[ColumnProfile, ...], side: str) -> dict[str, ColumnProfile]:
    column_map = {column.name: column for column in columns}
    if len(column_map) != len(columns):
        raise SchemaComparisonError(f"{side} profile contains duplicate column names")
    return column_map


def _classify_type_change(
    before: LogicalDataType,
    after: LogicalDataType,
    date_policy: DateToDatetimePolicy,
) -> TypeCompatibility:
    if before == after:
        return TypeCompatibility.UNCHANGED
    if before == LogicalDataType.CATEGORICAL and after == LogicalDataType.STRING:
        return TypeCompatibility.COMPATIBLE
    numeric = {LogicalDataType.INTEGER, LogicalDataType.DECIMAL, LogicalDataType.FLOAT}
    if before in numeric and after in numeric:
        return TypeCompatibility.POTENTIALLY_LOSSY
    if before == LogicalDataType.DATE and after == LogicalDataType.DATETIME:
        if date_policy == DateToDatetimePolicy.MIDNIGHT:
            return TypeCompatibility.POTENTIALLY_LOSSY
        return TypeCompatibility.INCOMPATIBLE
    return TypeCompatibility.INCOMPATIBLE


def _finding_for_change(change: SchemaChange) -> Finding:
    if change.kind == SchemaChangeKind.ADDED:
        finding_kind = FindingKind.SCHEMA_ADDED_COLUMN
        severity = FindingSeverity.WARNING
        message = f"Column {change.column_name!r} was added"
        compatibility_reason = None
    elif change.kind == SchemaChangeKind.REMOVED:
        finding_kind = FindingKind.SCHEMA_REMOVED_COLUMN
        severity = FindingSeverity.WARNING
        message = f"Column {change.column_name!r} was removed"
        compatibility_reason = None
    else:
        finding_kind = FindingKind.SCHEMA_TYPE_CHANGED
        severity = {
            TypeCompatibility.COMPATIBLE: FindingSeverity.INFO,
            TypeCompatibility.POTENTIALLY_LOSSY: FindingSeverity.WARNING,
            TypeCompatibility.INCOMPATIBLE: FindingSeverity.ERROR,
        }[change.compatibility]
        message = (
            f"Column {change.column_name!r} changed from {change.before_type.value} "
            f"to {change.after_type.value} ({change.compatibility.value})"
        )
        compatibility_reason = _compatibility_reason(change)

    evidence_details = {
        "column": change.column_name,
        "change": change.kind.value,
        "before_type": change.before_type.value if change.before_type else None,
        "after_type": change.after_type.value if change.after_type else None,
    }
    if change.compatibility is not None:
        evidence_details["compatibility"] = change.compatibility.value
        evidence_details["compatibility_reason"] = compatibility_reason
    evidence_counts = {"columns": 1}
    stable_key = f"{finding_kind.value}\0{change.column_name}".encode()
    finding_id = f"schema-{hashlib.sha256(stable_key).hexdigest()[:16]}"
    return Finding(
        finding_id=finding_id,
        kind=finding_kind,
        severity=severity,
        message=message,
        affected_columns=(change.column_name,),
        counts=evidence_counts,
        evidence=FindingEvidence(
            method="schema_profile_comparison",
            counts=evidence_counts,
            details=evidence_details,
        ),
    )


def _compatibility_reason(change: SchemaChange) -> str:
    before, after = change.before_type, change.after_type
    if before == LogicalDataType.CATEGORICAL and after == LogicalDataType.STRING:
        return "category labels map directly to strings"
    if before == LogicalDataType.DATE and after == LogicalDataType.DATETIME:
        return (
            "explicit midnight policy selected; timezone semantics are absent from the "
            "profile, so losslessness cannot be guaranteed"
            if change.compatibility == TypeCompatibility.POTENTIALLY_LOSSY
            else "date-to-datetime conversion requires an explicit midnight policy"
        )
    numeric_types = {LogicalDataType.INTEGER, LogicalDataType.DECIMAL, LogicalDataType.FLOAT}
    if before in numeric_types and after in numeric_types:
        return "value ranges and decimal precision/scale are unavailable to prove losslessness"
    return "no validated conversion rule or value-level evidence is available"
