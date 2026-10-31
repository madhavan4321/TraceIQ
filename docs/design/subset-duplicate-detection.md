# Future design: subset-based duplicate detection

## Scope

The current `DatasetProfile.duplicate_row_count` remains a full-row count: for each
identical complete row, count excess occurrences (`frequency - 1`). This behavior
must not change. A future opt-in mode could detect duplicates over caller-selected
business columns, such as `customer_id` or `(invoice_id, line_number)`.

## Proposed contract

Add an explicit duplicate-detection option rather than overloading the existing
count. `columns=None` means full-row semantics; a non-empty tuple means group rows
by exactly those case-sensitive column names. Reject empty tuples, repeated or
missing column names, and never silently fall back to full-row comparison.

Return a separate structured profile, for example:

```text
DuplicateProfile
  scope: full_row | selected_columns
  columns: tuple[str, ...]
  excess_count: int
  duplicate_group_count: int
  affected_row_count: int
  examples: bounded tuple of {values, multiplicity}
```

Keep `DatasetProfile.duplicate_row_count` as the legacy full-row metric. The
additional result can be attached to a later profile/report model without changing
the meaning of existing reports. Reports should state the selected columns and the
same excess-occurrence semantics used by full-row counting.

## Execution approach

Use an exact Polars `group_by(selected_columns).agg(pl.len())` against the staged
Parquet scan, then aggregate `max(frequency - 1, 0)`, duplicate-group count, and
affected rows. Sort groups by canonical column values before collecting bounded
examples. Do not use hashes as the sole equality test because collisions could
produce false duplicates. Nulls should group together, matching current full-row
behavior; document how NaN values group and test it explicitly.

## Trade-offs and safeguards

- Selected-column grouping can reveal business-key collisions that full-row
  grouping cannot, but broad or low-cardinality selections can produce large groups.
- Exact grouping can require memory proportional to distinct selected values. Keep
  streaming execution, use staged Parquet for repeated analysis, and benchmark
  high-cardinality and low-cardinality selections separately.
- A subset may intentionally omit distinguishing attributes. Report both the
  duplicate key values and multiplicity; do not label those rows as identical full
  records.
- Make the mode opt-in and keep full-row counts separately visible to preserve
  compatibility and avoid changing existing findings.
- Apply configured example limits; counts remain exact even when examples are
  sampled.

## Test plan

Cover default full-row behavior, one-column and composite business keys, null and
NaN values, missing/repeated/case-different column names, row-order independence,
excess and affected-row counts, deterministic bounded examples, report JSON
round-trip, and memory behavior across cardinalities. Verify existing full-row
results remain unchanged when subset detection is not requested.
