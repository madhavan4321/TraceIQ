"""Deterministic, bounded-evidence record comparison over Polars lazy frames."""

from __future__ import annotations

import base64
import hashlib
import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import polars as pl

from traceiq.models import (
    AnalysisStage,
    ComparisonOptions,
    Finding,
    FindingEvidence,
    FindingKind,
    FindingProvenance,
    FindingSeverity,
    LogicalDataType,
    RecordComparisonCounts,
    RecordComparisonMode,
    RecordComparisonResult,
    RecordComparisonStatus,
)
from traceiq.types import normalize_polars_dtype


class RecordComparisonError(ValueError):
    """Raised when record comparison cannot safely execute on the input schemas."""


def compare_records(
    baseline: pl.LazyFrame,
    candidate: pl.LazyFrame,
    options: ComparisonOptions,
    *,
    baseline_dataset_id: str,
    candidate_dataset_id: str,
    excluded_columns: tuple[str, ...] = (),
) -> RecordComparisonResult:
    """Compare keyed records or keyless row multiplicities without relying on order.

    Inputs remain lazy. Each result query is executed with Polars' streaming engine;
    only aggregate summaries and up to ``example_limit`` examples per finding are
    collected into Python. Callers that need to reuse a CSV scan across several analysis
    stages should stage it once (for example as temporary Parquet) and pass lazy scans of
    that staged file. A LazyFrame alone does not cache its source between collect calls.
    """
    baseline_schema = baseline.collect_schema()
    candidate_schema = candidate.collect_schema()
    baseline_names = tuple(baseline_schema.names())
    candidate_names = tuple(candidate_schema.names())
    if len(set(baseline_names)) != len(baseline_names):
        raise RecordComparisonError("baseline schema contains duplicate column names")
    if len(set(candidate_names)) != len(candidate_names):
        raise RecordComparisonError("candidate schema contains duplicate column names")

    common = set(baseline_names) & set(candidate_names)
    common_columns = tuple(sorted(common))
    changed_type_columns = {
        name
        for name in common_columns
        if normalize_polars_dtype(baseline_schema[name])
        != normalize_polars_dtype(candidate_schema[name])
    }
    excluded = set(excluded_columns) | changed_type_columns
    baseline, candidate, uncanonicalizable = _canonicalize_shared_types(
        baseline, candidate, baseline_schema, candidate_schema, common_columns
    )
    excluded.update(uncanonicalizable)

    if options.key_columns:
        prefix = _internal_prefix((*baseline_names, *candidate_names))
        return _compare_keyed(
            baseline,
            candidate,
            baseline_schema,
            candidate_schema,
            baseline_dataset_id,
            candidate_dataset_id,
            options,
            excluded,
            common_columns,
            prefix,
            uncanonicalizable,
        )
    comparable = tuple(name for name in common_columns if name not in excluded)
    if not comparable:
        finding = _not_comparable_finding(
            baseline_dataset_id,
            candidate_dataset_id,
            options.example_limit,
            uncanonicalizable,
            baseline_schema,
            candidate_schema,
        )
        return RecordComparisonResult(
            mode=RecordComparisonMode.MULTISET,
            status=RecordComparisonStatus.NON_COMPARABLE,
            comparable_columns=(),
            findings=(finding,),
            example_limit=options.example_limit,
        )
    pre_findings = (
        (
            _uncomparable_columns_finding(
                uncanonicalizable,
                baseline_schema,
                candidate_schema,
                baseline_dataset_id,
                candidate_dataset_id,
            ),
        )
        if uncanonicalizable
        else ()
    )
    return _compare_multiset(
        baseline,
        candidate,
        comparable,
        baseline_dataset_id,
        candidate_dataset_id,
        options.example_limit,
        pre_findings,
    )


def _canonicalize_shared_types(
    baseline: pl.LazyFrame,
    candidate: pl.LazyFrame,
    baseline_schema: pl.Schema,
    candidate_schema: pl.Schema,
    common_columns: tuple[str, ...],
) -> tuple[pl.LazyFrame, pl.LazyFrame, set[str]]:
    """Align same-logical-type physical widths before joins and multiset grouping."""
    baseline_casts: list[pl.Expr] = []
    candidate_casts: list[pl.Expr] = []
    uncanonicalizable: set[str] = set()
    for name in common_columns:
        before_type, after_type = baseline_schema[name], candidate_schema[name]
        if normalize_polars_dtype(before_type) != normalize_polars_dtype(after_type):
            continue
        if before_type == after_type:
            continue
        logical_type = normalize_polars_dtype(before_type)
        target: pl.DataType | None = None
        if logical_type == LogicalDataType.INTEGER:
            # Int128 safely contains the full UInt64 domain as well as signed widths.
            target = pl.Int128
        elif logical_type == LogicalDataType.FLOAT:
            target = pl.Float64
        elif logical_type == LogicalDataType.DECIMAL:
            scale = max(int(before_type.scale or 0), int(after_type.scale or 0))
            if scale <= 38:
                target = pl.Decimal(precision=38, scale=scale)
        elif logical_type == LogicalDataType.CATEGORICAL:
            target = pl.String
        elif logical_type == LogicalDataType.DATETIME:
            before_zone = before_type.time_zone
            after_zone = after_type.time_zone
            if before_zone == after_zone:
                target = pl.Datetime(time_unit="ns", time_zone=before_zone)
        elif logical_type == LogicalDataType.DURATION:
            target = pl.Duration(time_unit="ns")
        if target is None:
            uncanonicalizable.add(name)
        else:
            baseline_casts.append(pl.col(name).cast(target, strict=True).alias(name))
            candidate_casts.append(pl.col(name).cast(target, strict=True).alias(name))
    if baseline_casts:
        baseline = baseline.with_columns(baseline_casts)
        candidate = candidate.with_columns(candidate_casts)
    return baseline, candidate, uncanonicalizable


def _compare_keyed(
    baseline: pl.LazyFrame,
    candidate: pl.LazyFrame,
    baseline_schema: pl.Schema,
    candidate_schema: pl.Schema,
    baseline_dataset_id: str,
    candidate_dataset_id: str,
    options: ComparisonOptions,
    excluded: set[str],
    common_columns: tuple[str, ...],
    prefix: str,
    uncanonicalizable: set[str],
) -> RecordComparisonResult:
    keys = options.key_columns
    invalid_reasons: list[dict[str, Any]] = []
    if len(set(keys)) != len(keys):
        invalid_reasons.append({"reason": "duplicate_key_column_names", "columns": list(keys)})
    missing_baseline = [name for name in keys if name not in baseline_schema]
    missing_candidate = [name for name in keys if name not in candidate_schema]
    if missing_baseline or missing_candidate:
        invalid_reasons.append(
            {
                "reason": "missing_key_columns",
                "baseline": missing_baseline,
                "candidate": missing_candidate,
            }
        )
    mismatched_types = [
        {
            "column": name,
            "baseline_type": normalize_polars_dtype(baseline_schema[name]).value,
            "candidate_type": normalize_polars_dtype(candidate_schema[name]).value,
        }
        for name in keys
        if name in baseline_schema
        and name in candidate_schema
        and normalize_polars_dtype(baseline_schema[name])
        != normalize_polars_dtype(candidate_schema[name])
    ]
    if mismatched_types:
        invalid_reasons.append({"reason": "mismatched_key_types", "columns": mismatched_types})
    physical_mismatches = [
        {
            "column": name,
            "baseline_type": str(baseline_schema[name]),
            "candidate_type": str(candidate_schema[name]),
        }
        for name in keys
        if name in uncanonicalizable
    ]
    if physical_mismatches:
        invalid_reasons.append(
            {"reason": "uncanonicalizable_key_physical_types", "columns": physical_mismatches}
        )
    if invalid_reasons:
        return _invalid_keys_result(
            keys,
            invalid_reasons,
            baseline_dataset_id,
            candidate_dataset_id,
            options.example_limit,
            invalid_rows=0,
        )

    try:
        baseline_key_summary, baseline_invalid_examples = _key_validation(
            baseline, keys, options.example_limit, "baseline"
        )
        remaining_examples = max(0, options.example_limit - len(baseline_invalid_examples))
        candidate_key_summary, candidate_invalid_examples = _key_validation(
            candidate, keys, remaining_examples, "candidate"
        )
    except (pl.exceptions.PolarsError, OSError) as exc:
        raise RecordComparisonError(f"could not validate comparison keys: {exc}") from exc

    invalid_rows = (
        baseline_key_summary["invalid_key_rows"] + candidate_key_summary["invalid_key_rows"]
    )
    if invalid_rows:
        return _invalid_keys_result(
            keys,
            [],
            baseline_dataset_id,
            candidate_dataset_id,
            options.example_limit,
            invalid_rows=invalid_rows,
            side_summaries={
                "baseline": baseline_key_summary,
                "candidate": candidate_key_summary,
            },
            examples=baseline_invalid_examples + candidate_invalid_examples,
        )

    attributes = tuple(
        name for name in common_columns if name not in set(keys) and name not in excluded
    )
    try:
        joined = _keyed_join(baseline, candidate, keys, attributes, prefix)
        counts = _keyed_change_counts(joined, attributes, prefix)
        unsupported_attributes = tuple(
            sorted(name for name in uncanonicalizable if name not in keys)
        )
        findings: list[Finding] = []
        if unsupported_attributes:
            findings.append(
                _uncomparable_columns_finding(
                    unsupported_attributes,
                    baseline_schema,
                    candidate_schema,
                    baseline_dataset_id,
                    candidate_dataset_id,
                )
            )
        if counts["added"]:
            findings.append(
                _keyed_finding(
                    joined,
                    FindingKind.RECORDS_ADDED,
                    "candidate_only",
                    keys,
                    attributes,
                    counts["added"],
                    options.example_limit,
                    baseline_dataset_id,
                    candidate_dataset_id,
                    prefix,
                )
            )
        if counts["removed"]:
            findings.append(
                _keyed_finding(
                    joined,
                    FindingKind.RECORDS_REMOVED,
                    "baseline_only",
                    keys,
                    attributes,
                    counts["removed"],
                    options.example_limit,
                    baseline_dataset_id,
                    candidate_dataset_id,
                    prefix,
                )
            )
        if counts["modified"]:
            findings.append(
                _modified_finding(
                    joined,
                    keys,
                    attributes,
                    counts["modified"],
                    options.example_limit,
                    baseline_dataset_id,
                    candidate_dataset_id,
                    prefix,
                )
            )
    except (pl.exceptions.PolarsError, OSError) as exc:
        raise RecordComparisonError(f"could not compare keyed records: {exc}") from exc

    return RecordComparisonResult(
        mode=RecordComparisonMode.KEYED,
        status=RecordComparisonStatus.COMPARABLE,
        comparable_columns=attributes,
        counts=RecordComparisonCounts(
            added=counts["added"], removed=counts["removed"], modified=counts["modified"]
        ),
        findings=tuple(findings),
        example_limit=options.example_limit,
    )


def _key_validation(
    frame: pl.LazyFrame,
    keys: tuple[str, ...],
    example_limit: int,
    side: str,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    prefix = _internal_prefix(keys)
    multiplicity_column = f"{prefix}multiplicity"
    null_column = f"{prefix}has_null_key"
    null_key = pl.any_horizontal([pl.col(name).is_null() for name in keys])
    grouped = frame.group_by(list(keys)).agg(
        pl.len().alias(multiplicity_column),
        null_key.any().alias(null_column),
    )
    summary = _collect(
        grouped.select(
            pl.when(pl.col(null_column))
            .then(pl.col(multiplicity_column))
            .otherwise(0)
            .sum()
            .alias("null_key_rows"),
            pl.when((~pl.col(null_column)) & (pl.col(multiplicity_column) > 1))
            .then(pl.col(multiplicity_column))
            .otherwise(0)
            .sum()
            .alias("duplicate_key_rows"),
        )
    )
    null_rows = int(summary.item(0, "null_key_rows") or 0)
    duplicate_rows = int(summary.item(0, "duplicate_key_rows") or 0)
    invalid_examples: list[dict[str, Any]] = []
    if example_limit and (null_rows or duplicate_rows):
        invalid_groups = grouped.filter(
            pl.col(null_column) | (pl.col(multiplicity_column) > 1)
        ).sort(list(keys), nulls_last=True)
        for row in _collect(invalid_groups.head(example_limit)).to_dicts():
            key = {name: _json_value(row[name]) for name in keys}
            issue = "null_key" if row[null_column] else "duplicate_key"
            invalid_examples.append(
                {
                    "side": side,
                    "issue": issue,
                    "key": key,
                    "row_count": int(row[multiplicity_column]),
                }
            )
    return (
        {
            "null_key_rows": null_rows,
            "duplicate_key_rows": duplicate_rows,
            "invalid_key_rows": null_rows + duplicate_rows,
        },
        invalid_examples,
    )


def _invalid_keys_result(
    keys: tuple[str, ...],
    reasons: list[dict[str, Any]],
    baseline_dataset_id: str,
    candidate_dataset_id: str,
    example_limit: int,
    *,
    invalid_rows: int,
    side_summaries: dict[str, dict[str, int]] | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> RecordComparisonResult:
    counts: dict[str, int | float] = {"invalid_key_rows": invalid_rows}
    structured_reasons = list(reasons)
    if side_summaries is not None:
        for side, values in side_summaries.items():
            counts.update({f"{side}_{key}": value for key, value in values.items()})
            if values["null_key_rows"]:
                structured_reasons.append(
                    {
                        "reason": "null_key_rows",
                        "side": side,
                        "count": values["null_key_rows"],
                    }
                )
            if values["duplicate_key_rows"]:
                structured_reasons.append(
                    {
                        "reason": "duplicate_key_rows",
                        "side": side,
                        "count": values["duplicate_key_rows"],
                    }
                )
    finding = _finding(
        FindingKind.INVALID_COMPARISON_KEY,
        FindingSeverity.ERROR,
        f"Comparison key validation failed for {', '.join(keys) or '(empty key)'}",
        tuple(sorted(set(keys))),
        counts,
        "key_validation",
        examples=examples or [],
        details={"key_columns": list(keys), "reasons": structured_reasons},
        baseline_dataset_id=baseline_dataset_id,
        candidate_dataset_id=candidate_dataset_id,
    )
    return RecordComparisonResult(
        mode=RecordComparisonMode.KEYED,
        status=RecordComparisonStatus.INVALID_KEYS,
        counts=RecordComparisonCounts(invalid_key_rows=invalid_rows),
        findings=(finding,),
        example_limit=example_limit,
    )


def _keyed_join(
    baseline: pl.LazyFrame,
    candidate: pl.LazyFrame,
    keys: tuple[str, ...],
    attributes: tuple[str, ...],
    prefix: str,
) -> pl.LazyFrame:
    before_marker = f"{prefix}before_present"
    after_marker = f"{prefix}after_present"
    before_names = {name: f"{prefix}before_{index}" for index, name in enumerate(attributes)}
    after_names = {name: f"{prefix}after_{index}" for index, name in enumerate(attributes)}
    left = baseline.select(
        [pl.col(name) for name in keys]
        + [pl.col(name).alias(before_names[name]) for name in attributes]
        + [pl.lit(True).alias(before_marker)]
    )
    right = candidate.select(
        [pl.col(name) for name in keys]
        + [pl.col(name).alias(after_names[name]) for name in attributes]
        + [pl.lit(True).alias(after_marker)]
    )
    return left.join(
        right,
        on=list(keys),
        how="full",
        coalesce=True,
        nulls_equal=False,
    ).with_columns(pl.col(before_marker).fill_null(False), pl.col(after_marker).fill_null(False))


def _keyed_change_counts(
    joined: pl.LazyFrame, attributes: tuple[str, ...], prefix: str
) -> dict[str, int]:
    before_present = pl.col(f"{prefix}before_present")
    after_present = pl.col(f"{prefix}after_present")
    added_expr = ~before_present & after_present
    removed_expr = before_present & ~after_present
    changed_exprs = [expression for _, expression in _modified_exprs(attributes, prefix)]
    modified_expr = (
        (pl.any_horizontal(changed_exprs) if changed_exprs else pl.lit(False))
        & before_present
        & after_present
    )
    stats = _collect(
        joined.select(
            added_expr.sum().alias("added"),
            removed_expr.sum().alias("removed"),
            modified_expr.sum().alias("modified"),
        )
    )
    return {name: int(stats.item(0, name) or 0) for name in ("added", "removed", "modified")}


def _modified_exprs(attributes: tuple[str, ...], prefix: str) -> list[tuple[str, pl.Expr]]:
    return [
        (
            name,
            pl.col(f"{prefix}before_{index}").ne_missing(pl.col(f"{prefix}after_{index}")),
        )
        for index, name in enumerate(attributes)
    ]


def _keyed_finding(
    joined: pl.LazyFrame,
    kind: FindingKind,
    side_filter: str,
    keys: tuple[str, ...],
    attributes: tuple[str, ...],
    count: int,
    example_limit: int,
    baseline_dataset_id: str,
    candidate_dataset_id: str,
    prefix: str,
) -> Finding:
    if side_filter == "candidate_only":
        filtered = joined.filter(
            ~pl.col(f"{prefix}before_present") & pl.col(f"{prefix}after_present")
        )
        record_names = {name: f"{prefix}after_{index}" for index, name in enumerate(attributes)}
    else:
        filtered = joined.filter(
            pl.col(f"{prefix}before_present") & ~pl.col(f"{prefix}after_present")
        )
        record_names = {name: f"{prefix}before_{index}" for index, name in enumerate(attributes)}
    examples: list[dict[str, Any]] = []
    if example_limit:
        selected = (
            filtered.sort(list(keys), nulls_last=True)
            .head(example_limit)
            .select(
                [pl.col(name) for name in keys]
                + [pl.col(record_names[name]) for name in attributes]
            )
        )
        for row in _collect(selected).to_dicts():
            examples.append(
                {
                    "key": {name: _json_value(row[name]) for name in keys},
                    "record": {name: _json_value(row[record_names[name]]) for name in attributes},
                }
            )
    affected = tuple(sorted(set(keys) | set(attributes)))
    return _finding(
        kind,
        FindingSeverity.INFO,
        f"{count} {kind.value.split('.')[1]} record(s)",
        affected,
        {"records": count},
        "keyed_full_outer_join",
        examples=examples,
        baseline_dataset_id=baseline_dataset_id,
        candidate_dataset_id=candidate_dataset_id,
    )


def _modified_finding(
    joined: pl.LazyFrame,
    keys: tuple[str, ...],
    attributes: tuple[str, ...],
    count: int,
    example_limit: int,
    baseline_dataset_id: str,
    candidate_dataset_id: str,
    prefix: str,
) -> Finding:
    named_changed = _modified_exprs(attributes, prefix)
    changed = pl.any_horizontal([expression for _, expression in named_changed])
    modified = joined.filter(
        pl.col(f"{prefix}before_present") & pl.col(f"{prefix}after_present") & changed
    )
    changed_column_counts: dict[str, int] = {}
    if named_changed:
        tally = _collect(
            modified.select(
                [
                    expression.sum().alias(f"{prefix}change_{index}")
                    for index, (_, expression) in enumerate(named_changed)
                ]
            )
        )
        changed_column_counts = {
            name: int(tally.item(0, f"{prefix}change_{index}") or 0)
            for index, (name, _) in enumerate(named_changed)
            if int(tally.item(0, f"{prefix}change_{index}") or 0) > 0
        }
    changed_columns = tuple(
        name for name, changed_count in changed_column_counts.items() if changed_count
    )
    examples: list[dict[str, Any]] = []
    if example_limit:
        example_fields = [pl.col(name) for name in keys]
        example_fields += [pl.col(f"{prefix}before_{index}") for index, _ in enumerate(attributes)]
        example_fields += [pl.col(f"{prefix}after_{index}") for index, _ in enumerate(attributes)]
        sample = (
            modified.sort(list(keys), nulls_last=True).head(example_limit).select(example_fields)
        )
        for row in _collect(sample).to_dicts():
            before = {
                name: _json_value(row[f"{prefix}before_{index}"])
                for index, name in enumerate(attributes)
            }
            after = {
                name: _json_value(row[f"{prefix}after_{index}"])
                for index, name in enumerate(attributes)
            }
            row_changed = [name for name in attributes if before[name] != after[name]]
            examples.append(
                {
                    "key": {name: _json_value(row[name]) for name in keys},
                    "changed_columns": row_changed,
                    "before": before,
                    "after": after,
                }
            )
    return _finding(
        FindingKind.RECORD_MODIFIED,
        FindingSeverity.WARNING,
        f"{count} keyed record(s) have modified shared attributes",
        changed_columns,
        {"records": count},
        "keyed_full_outer_join",
        examples=examples,
        details={"column_counts": changed_column_counts},
        baseline_dataset_id=baseline_dataset_id,
        candidate_dataset_id=candidate_dataset_id,
    )


def _compare_multiset(
    baseline: pl.LazyFrame,
    candidate: pl.LazyFrame,
    columns: tuple[str, ...],
    baseline_dataset_id: str,
    candidate_dataset_id: str,
    example_limit: int,
    pre_findings: tuple[Finding, ...] = (),
) -> RecordComparisonResult:
    prefix = _internal_prefix(columns)
    before_count = f"{prefix}before_count"
    after_count = f"{prefix}after_count"
    added_count = f"{prefix}added_count"
    removed_count = f"{prefix}removed_count"
    left = baseline.group_by(list(columns)).agg(pl.len().alias(before_count))
    right = candidate.group_by(list(columns)).agg(pl.len().alias(after_count))
    joined = (
        left.join(
            right,
            on=list(columns),
            how="full",
            coalesce=True,
            nulls_equal=True,
        )
        .with_columns(
            pl.col(before_count).fill_null(0),
            pl.col(after_count).fill_null(0),
        )
        .with_columns(
            (pl.col(after_count).cast(pl.Int64) - pl.col(before_count).cast(pl.Int64))
            .clip(lower_bound=0)
            .alias(added_count),
            (pl.col(before_count).cast(pl.Int64) - pl.col(after_count).cast(pl.Int64))
            .clip(lower_bound=0)
            .alias(removed_count),
        )
    )
    try:
        stats = _collect(
            joined.select(
                pl.col(added_count).sum().alias("added"),
                pl.col(removed_count).sum().alias("removed"),
            )
        )
        added = int(stats.item(0, "added") or 0)
        removed = int(stats.item(0, "removed") or 0)
        findings: list[Finding] = list(pre_findings)
        if added:
            findings.append(
                _multiset_finding(
                    joined,
                    columns,
                    added_count,
                    FindingKind.RECORDS_ADDED,
                    added,
                    example_limit,
                    baseline_dataset_id,
                    candidate_dataset_id,
                )
            )
        if removed:
            findings.append(
                _multiset_finding(
                    joined,
                    columns,
                    removed_count,
                    FindingKind.RECORDS_REMOVED,
                    removed,
                    example_limit,
                    baseline_dataset_id,
                    candidate_dataset_id,
                )
            )
    except (pl.exceptions.PolarsError, OSError) as exc:
        raise RecordComparisonError(f"could not compare keyless record multisets: {exc}") from exc
    return RecordComparisonResult(
        mode=RecordComparisonMode.MULTISET,
        status=RecordComparisonStatus.COMPARABLE,
        comparable_columns=columns,
        counts=RecordComparisonCounts(added=added, removed=removed),
        findings=tuple(findings),
        example_limit=example_limit,
    )


def _multiset_finding(
    joined: pl.LazyFrame,
    columns: tuple[str, ...],
    count_column: str,
    kind: FindingKind,
    count: int,
    example_limit: int,
    baseline_dataset_id: str,
    candidate_dataset_id: str,
) -> Finding:
    examples: list[dict[str, Any]] = []
    if example_limit:
        sample = (
            joined.filter(pl.col(count_column) > 0)
            .sort(list(columns), nulls_last=True)
            .head(example_limit)
            .select([pl.col(name) for name in columns] + [pl.col(count_column)])
        )
        for row in _collect(sample).to_dicts():
            examples.append(
                {
                    "row": {name: _json_value(row[name]) for name in columns},
                    "multiplicity": int(row[count_column]),
                }
            )
    return _finding(
        kind,
        FindingSeverity.INFO,
        f"{count} row occurrence(s) are {kind.value.split('.')[1]} in the multiset",
        columns,
        {"records": count},
        "keyless_multiset_frequency_diff",
        examples=examples,
        baseline_dataset_id=baseline_dataset_id,
        candidate_dataset_id=candidate_dataset_id,
    )


def _not_comparable_finding(
    baseline_dataset_id: str,
    candidate_dataset_id: str,
    example_limit: int,
    uncanonicalizable: set[str],
    baseline_schema: pl.Schema,
    candidate_schema: pl.Schema,
) -> Finding:
    return _finding(
        FindingKind.RECORDS_NOT_COMPARABLE,
        FindingSeverity.ERROR,
        "Keyless record comparison has no shared comparable columns",
        tuple(sorted(uncanonicalizable)),
        {"columns": 0},
        "keyless_multiset_frequency_diff",
        details={
            "reason": "no_common_comparable_columns",
            "unsupported_physical_types": [
                {
                    "column": name,
                    "baseline_type": str(baseline_schema[name]),
                    "candidate_type": str(candidate_schema[name]),
                    "reason": "same logical type but physical details cannot be safely normalized",
                }
                for name in sorted(uncanonicalizable)
            ],
        },
        baseline_dataset_id=baseline_dataset_id,
        candidate_dataset_id=candidate_dataset_id,
    )


def _uncomparable_columns_finding(
    columns: tuple[str, ...],
    baseline_schema: pl.Schema,
    candidate_schema: pl.Schema,
    baseline_dataset_id: str,
    candidate_dataset_id: str,
) -> Finding:
    details = {
        "reason": "same logical types have physical details that cannot be safely normalized",
        "columns": [
            {
                "column": name,
                "baseline_type": str(baseline_schema[name]),
                "candidate_type": str(candidate_schema[name]),
            }
            for name in columns
        ],
    }
    return _finding(
        FindingKind.RECORD_COLUMNS_UNCOMPARABLE,
        FindingSeverity.WARNING,
        f"Values in {len(columns)} shared column(s) were excluded from record comparison",
        columns,
        {"columns": len(columns)},
        "record_type_alignment",
        details=details,
        baseline_dataset_id=baseline_dataset_id,
        candidate_dataset_id=candidate_dataset_id,
    )


def _finding(
    kind: FindingKind,
    severity: FindingSeverity,
    message: str,
    affected_columns: tuple[str, ...],
    counts: dict[str, int | float],
    method: str,
    *,
    examples: list[dict[str, Any]] | None = None,
    details: dict[str, Any] | None = None,
    baseline_dataset_id: str,
    candidate_dataset_id: str,
) -> Finding:
    identity = "\0".join((kind.value, baseline_dataset_id, candidate_dataset_id))
    finding_id = f"record-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"
    return Finding(
        finding_id=finding_id,
        kind=kind,
        severity=severity,
        message=message,
        affected_columns=affected_columns,
        counts=counts,
        evidence=FindingEvidence(
            method=method,
            counts=counts,
            examples=tuple(examples or ()),
            details=details or {},
        ),
        provenance=FindingProvenance(
            stage=AnalysisStage.RECORDS,
            method=method,
            baseline_dataset_id=baseline_dataset_id,
            candidate_dataset_id=candidate_dataset_id,
        ),
    )


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    return frame.collect(engine="streaming")


def _internal_prefix(columns: tuple[str, ...]) -> str:
    """Choose an internal alias namespace that cannot collide with user columns."""
    prefix = "__traceiq_internal__"
    while any(name.startswith(prefix) for name in columns):
        prefix = f"_{prefix}"
    return prefix


def _json_value(value: Any) -> Any:
    """Convert Polars/Python scalars into deterministic JSON-compatible evidence."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"$traceiq_float": "NaN"}
        if math.isinf(value):
            return {"$traceiq_float": "Infinity" if value > 0 else "-Infinity"}
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise RecordComparisonError(f"cannot serialize evidence value of type {type(value).__name__}")
