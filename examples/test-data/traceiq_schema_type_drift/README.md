# Deterministic schema type-drift fixture

This fixture isolates a logical type change under TraceIQ's current Polars CSV
inference settings. `amount` is numeric in `baseline.csv`; the non-numeric
`not_recorded` value in the candidate makes the entire candidate column infer as
string. The expected profiles are `float` and `string`, respectively, and schema
comparison should report one changed column.

The integration test verifies those inferred types before asserting the schema
finding. The fixture is separate from the eight original sample scenarios.
