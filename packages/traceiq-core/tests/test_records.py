from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import polars as pl
import pytest

from traceiq.comparison.records import compare_records
from traceiq.models import (
    AnalysisStage,
    ComparisonOptions,
    FindingKind,
    RecordComparisonMode,
    RecordComparisonStatus,
)

BASELINE_ID = "baseline-dataset"
CANDIDATE_ID = "candidate-dataset"


def compare(
    before: pl.DataFrame,
    after: pl.DataFrame,
    *,
    keys: tuple[str, ...] = (),
    example_limit: int = 100,
    excluded_columns: tuple[str, ...] = (),
):
    return compare_records(
        before.lazy(),
        after.lazy(),
        ComparisonOptions(key_columns=keys, example_limit=example_limit),
        baseline_dataset_id=BASELINE_ID,
        candidate_dataset_id=CANDIDATE_ID,
        excluded_columns=excluded_columns,
    )


def test_keyed_comparison_detects_added_removed_and_modified_rows() -> None:
    before = pl.DataFrame(
        {"id": [1, 2, 3, 4], "name": ["A", None, "gone", "same"], "score": [1, 2, 3, 4]}
    )
    after = pl.DataFrame(
        {"score": [1, 2, 4, 5], "name": ["a", "B", "same", "new"], "id": [1, 2, 4, 5]}
    )

    result = compare(before, after, keys=("id",))

    assert result.mode == RecordComparisonMode.KEYED
    assert result.status == RecordComparisonStatus.COMPARABLE
    assert result.comparable_columns == ("name", "score")
    assert result.counts.added == 1
    assert result.counts.removed == 1
    assert result.counts.modified == 2
    by_kind = {finding.kind: finding for finding in result.findings}
    assert set(by_kind) == {
        FindingKind.RECORDS_ADDED,
        FindingKind.RECORDS_REMOVED,
        FindingKind.RECORD_MODIFIED,
    }
    modified = by_kind[FindingKind.RECORD_MODIFIED]
    assert modified.counts["records"] == 2
    assert modified.evidence.details["column_counts"] == {"name": 2}
    assert modified.provenance.stage == AnalysisStage.RECORDS
    assert modified.provenance.baseline_dataset_id == BASELINE_ID
    assert modified.provenance.candidate_dataset_id == CANDIDATE_ID
    assert modified.provenance.method == modified.evidence.method


def test_keyed_comparison_treats_null_as_equal_but_null_to_value_as_modified() -> None:
    before = pl.DataFrame(
        {"id": [1, 2], "value": [None, None]}, schema={"id": pl.Int64, "value": pl.String}
    )
    after = pl.DataFrame(
        {"id": [1, 2], "value": [None, "present"]}, schema={"id": pl.Int64, "value": pl.String}
    )

    result = compare(before, after, keys=("id",))

    assert result.counts.modified == 1
    finding = result.findings[0]
    assert finding.kind == FindingKind.RECORD_MODIFIED
    assert finding.evidence.examples == (
        {
            "key": {"id": 2},
            "changed_columns": ["value"],
            "before": {"value": None},
            "after": {"value": "present"},
        },
    )


def test_keyed_comparison_excludes_keys_and_schema_changed_columns() -> None:
    before = pl.DataFrame({"id": [1, 2], "amount": [10, 20], "flag": [True, True]})
    after = pl.DataFrame(
        {"id": [1, 3], "amount": [10.0, 30.0], "flag": [False, True]},
        schema={"id": pl.Int64, "amount": pl.Float64, "flag": pl.Boolean},
    )

    result = compare(before, after, keys=("id",))

    assert result.comparable_columns == ("flag",)
    assert result.counts.added == 1
    assert result.counts.removed == 1
    assert result.counts.modified == 1
    modified = next(f for f in result.findings if f.kind == FindingKind.RECORD_MODIFIED)
    assert modified.affected_columns == ("flag",)


@pytest.mark.parametrize(
    ("before", "after", "expected_invalid_rows", "reason"),
    [
        (
            pl.DataFrame({"id": [1, None, 2, 2, 2]}, schema={"id": pl.Int64}),
            pl.DataFrame({"id": [1, 2, 3]}, schema={"id": pl.Int64}),
            4,
            "null and duplicate keys",
        ),
        (
            pl.DataFrame({"id": [1, 2]}, schema={"id": pl.Int64}),
            pl.DataFrame({"id": [1.0, 2.0]}, schema={"id": pl.Float64}),
            0,
            "mismatched key types",
        ),
    ],
)
def test_invalid_keys_never_produce_guessed_record_diffs(
    before: pl.DataFrame,
    after: pl.DataFrame,
    expected_invalid_rows: int,
    reason: str,
) -> None:
    result = compare(before, after, keys=("id",))

    assert result.status == RecordComparisonStatus.INVALID_KEYS
    assert result.counts.invalid_key_rows == expected_invalid_rows
    assert result.counts.added == result.counts.removed == result.counts.modified == 0
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.kind == FindingKind.INVALID_COMPARISON_KEY
    assert finding.provenance is not None
    assert finding.evidence.details["reasons"] or expected_invalid_rows
    assert reason


def test_missing_key_column_is_reported_as_invalid() -> None:
    result = compare(
        pl.DataFrame({"id": [1], "value": ["x"]}),
        pl.DataFrame({"value": ["x"]}),
        keys=("id",),
    )

    assert result.status == RecordComparisonStatus.INVALID_KEYS
    assert result.findings[0].evidence.details["reasons"][0]["reason"] == "missing_key_columns"


def test_keyless_multiset_counts_occurrences_nulls_and_text_case_exactly() -> None:
    before = pl.DataFrame(
        {"left": ["a", "a", "b"], "right": [None, None, "X"]},
        schema={"left": pl.String, "right": pl.String},
    )
    after = pl.DataFrame(
        {"left": ["b", "a", "b", "c"], "right": ["x", None, "x", None]},
        schema={"left": pl.String, "right": pl.String},
    )

    result = compare(before, after)

    assert result.mode == RecordComparisonMode.MULTISET
    assert result.status == RecordComparisonStatus.COMPARABLE
    assert result.counts.added == 3
    assert result.counts.removed == 2
    assert result.counts.modified == 0
    assert not any(f.kind == FindingKind.RECORD_MODIFIED for f in result.findings)


def test_keyless_multiset_is_row_order_independent_and_examples_are_bounded() -> None:
    before = pl.DataFrame({"value": ["c", "a", "a", "b"]})
    after = pl.DataFrame({"value": ["d", "b", "a", "a", "a"]})

    first = compare(before, after, example_limit=1)
    second = compare(before.reverse(), after.reverse(), example_limit=1)

    assert first == second
    assert first.counts.added == 2
    assert first.counts.removed == 1
    assert all(len(f.evidence.examples) <= 1 for f in first.findings)
    assert [f.finding_id for f in first.findings] == [f.finding_id for f in second.findings]


def test_keyless_comparison_with_no_common_comparable_columns_is_explicit() -> None:
    before = pl.DataFrame({"changed": [1, 2]}, schema={"changed": pl.Int64})
    after = pl.DataFrame({"changed": ["1", "2"]}, schema={"changed": pl.String})

    result = compare(before, after)

    assert result.status == RecordComparisonStatus.NON_COMPARABLE
    assert result.comparable_columns == ()
    assert result.counts.added == result.counts.removed == 0
    assert result.findings[0].kind == FindingKind.RECORDS_NOT_COMPARABLE


def test_keyed_membership_remains_comparable_without_shared_attributes() -> None:
    before = pl.DataFrame({"id": [1, 2], "old": ["a", "b"]})
    after = pl.DataFrame({"id": [1, 3], "new": ["a", "c"]})

    result = compare(before, after, keys=("id",))

    assert result.status == RecordComparisonStatus.COMPARABLE
    assert result.comparable_columns == ()
    assert result.counts.added == result.counts.removed == 1
    assert result.counts.modified == 0


def test_user_columns_cannot_collide_with_internal_join_aliases() -> None:
    reserved = "__traceiq_internal__before_present"
    before = pl.DataFrame({reserved: [1], "__traceiq_internal__before_0": ["old"]})
    after = pl.DataFrame({reserved: [1], "__traceiq_internal__before_0": ["new"]})

    keyed = compare(before, after, keys=(reserved,))
    keyless = compare(before, after)

    assert keyed.counts.modified == 1
    assert keyless.counts.added == keyless.counts.removed == 1


@pytest.mark.parametrize(
    ("baseline_dtype", "candidate_dtype", "baseline_values", "candidate_values", "added"),
    [
        (pl.Int32, pl.Int64, [1, 2], [2, 3], 1),
        (pl.UInt32, pl.UInt64, [1], [1, 2**63 + 7], 1),
    ],
)
def test_same_logical_integer_keys_with_different_widths_match_safely(
    baseline_dtype: pl.DataType,
    candidate_dtype: pl.DataType,
    baseline_values: list[int],
    candidate_values: list[int],
    added: int,
) -> None:
    before = pl.DataFrame({"id": baseline_values}, schema={"id": baseline_dtype})
    after = pl.DataFrame({"id": candidate_values}, schema={"id": candidate_dtype})

    result = compare(before, after, keys=("id",))

    assert result.status == RecordComparisonStatus.COMPARABLE
    assert result.counts.added == added
    assert result.counts.removed == len(baseline_values) + 1 - len(candidate_values)


def test_decimal_keys_with_different_precision_and_scale_match() -> None:
    before = pl.DataFrame(
        {"id": [Decimal("1.20"), Decimal("2.30")]},
        schema={"id": pl.Decimal(precision=12, scale=2)},
    )
    after = pl.DataFrame(
        {"id": [Decimal("2.300"), Decimal("1.200")]},
        schema={"id": pl.Decimal(precision=18, scale=3)},
    )

    result = compare(before, after, keys=("id",))

    assert result.status == RecordComparisonStatus.COMPARABLE
    assert result.counts.added == result.counts.removed == result.counts.modified == 0


def test_float_keys_with_different_widths_match_on_exact_upcast_values() -> None:
    before = pl.DataFrame({"id": [1.0, 2.5]}, schema={"id": pl.Float32})
    after = pl.DataFrame({"id": [2.5, 1.0]}, schema={"id": pl.Float64})

    result = compare(before, after, keys=("id",))

    assert result.status == RecordComparisonStatus.COMPARABLE
    assert result.counts.added == result.counts.removed == result.counts.modified == 0


def test_unaligned_physical_attributes_are_reported_instead_of_silently_skipped() -> None:
    before = pl.DataFrame(
        {"id": [1], "stamp": [datetime(2024, 1, 1, tzinfo=UTC)]},
        schema={"id": pl.Int64, "stamp": pl.Datetime("us", time_zone="UTC")},
    )
    after = pl.DataFrame(
        {
            "id": [1],
            "stamp": [datetime(2024, 1, 1, 1, tzinfo=timezone(timedelta(hours=1)))],
        },
        schema={"id": pl.Int64, "stamp": pl.Datetime("ns", time_zone="Etc/GMT-1")},
    )

    keyed = compare(before, after, keys=("id",))
    keyless = compare(before.drop("id"), after.drop("id"))

    warning = next(f for f in keyed.findings if f.kind == FindingKind.RECORD_COLUMNS_UNCOMPARABLE)
    assert warning.severity.value == "warning"
    assert warning.affected_columns == ("stamp",)
    assert (
        warning.evidence.details["columns"][0]["baseline_type"]
        != warning.evidence.details["columns"][0]["candidate_type"]
    )
    assert keyless.status == RecordComparisonStatus.NON_COMPARABLE
    assert keyless.findings[0].kind == FindingKind.RECORDS_NOT_COMPARABLE
    assert keyless.findings[0].affected_columns == ("stamp",)


def test_finding_examples_are_json_compatible_for_decimal_bytes_and_nan() -> None:
    before = pl.DataFrame(
        {"id": [1], "value": [float("nan")], "payload": [b"\xff"]},
        schema={"id": pl.Int64, "value": pl.Float64, "payload": pl.Binary},
    )
    after = pl.DataFrame(
        {"id": [2], "value": [1.5], "payload": [b"\x00"]},
        schema={"id": pl.Int64, "value": pl.Float64, "payload": pl.Binary},
    )

    result = compare(before, after, keys=("id",))
    removed = next(f for f in result.findings if f.kind == FindingKind.RECORDS_REMOVED)
    example = removed.evidence.examples[0]
    assert example["record"]["payload"] == {"encoding": "base64", "data": "/w=="}
    assert example["record"]["value"] == {"$traceiq_float": "NaN"}
    json.dumps(example, allow_nan=False)


def test_bad_key_option_is_invalid_and_not_silently_accepted() -> None:
    result = compare_records(
        pl.DataFrame({"id": [1]}).lazy(),
        pl.DataFrame({"id": [1]}).lazy(),
        ComparisonOptions(key_columns=("id", "id")),
        baseline_dataset_id=BASELINE_ID,
        candidate_dataset_id=CANDIDATE_ID,
    )
    assert result.status == RecordComparisonStatus.INVALID_KEYS
    assert (
        result.findings[0].evidence.details["reasons"][0]["reason"] == "duplicate_key_column_names"
    )
