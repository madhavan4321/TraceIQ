# TraceIQ Sample Dataset Pack

Synthetic CSV fixtures for testing TraceIQ ingestion, profiling, schema drift,
keyed record comparison, keyless multiset comparison, and parser errors.

All records are fictional. These files contain no personal data.

## Scenarios

| Folder | Files | Intended checks |
|---|---|---|
| `01_customer_records` | baseline, candidate | Keyed comparison: modified, removed, and added records |
| `02_schema_drift` | baseline, candidate | Removed `legacy_code`, added `sales_channel`, and `order_total` type change |
| `03_keyless_multiset` | baseline, candidate | Order-independent multiset comparison; one removed occurrence and one added occurrence |
| `04_invalid_duplicate_keys` | baseline, candidate | Duplicate comparison key must make keyed comparison non-comparable |
| `05_invalid_null_keys` | baseline, candidate | Blank/null comparison key must be rejected |
| `06_nulls_and_duplicates` | baseline, candidate | Null profiling, full-row duplicate profiling, null equality, and modified value |
| `07_case_sensitive` | baseline, candidate | Case-sensitive key/value behavior |
| `08_parser_errors` | malformed, invalid_utf8 | Malformed CSV and invalid UTF-8 error handling |

## Suggested CLI checks (once the CLI exists)

```bash
traceiq inspect examples/test-data/01_customer_records/baseline.csv

traceiq compare examples/test-data/01_customer_records/baseline.csv examples/test-data/01_customer_records/candidate.csv --key customer_id

traceiq compare examples/test-data/03_keyless_multiset/baseline.csv examples/test-data/03_keyless_multiset/candidate.csv

traceiq compare examples/test-data/04_invalid_duplicate_keys/baseline.csv examples/test-data/04_invalid_duplicate_keys/candidate.csv --key customer_id
```

## Expected high-level outcomes

- `01_customer_records`: 1 added record, 1 removed record, 2 modified records.
- `02_schema_drift`: 1 added column, 1 removed column, 1 changed logical type (depending on inference settings).
- `03_keyless_multiset`: 1 added row occurrence and 1 removed row occurrence.
- `04_invalid_duplicate_keys`: comparison is non-comparable due to duplicate keys.
- `05_invalid_null_keys`: comparison is non-comparable due to a null/blank key.
- `06_nulls_and_duplicates`: baseline and candidate have zero excess duplicate full rows because `record_id` differs on otherwise similar rows; candidate has one additional record and one changed amount.
- `07_case_sensitive`: `ab-01` and `AB-01` are distinct values.
- `08_parser_errors`: ingestion should return typed, actionable errors rather than silently dropping data.

## Notes

- Empty CSV fields are used to represent null-like values. The exact null interpretation
  should follow TraceIQ's documented CSV null policy.
- CSV type inference can vary with inference settings. Tests should assert intended
  logical behavior, not rely on accidental inference from a tiny sample.
- The schema-drift fixture intentionally changes `order_total` from numeric-looking
  values to quoted numeric strings. Verify the actual inferred logical types before
  asserting a specific compatibility classification.
