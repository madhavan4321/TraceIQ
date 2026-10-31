# TraceIQ contributor guidance

- Keep data analysis and report construction in the independently installable `traceiq-core` package.
- CLI and API layers must call the shared core and must not duplicate investigation logic.
- Keep analysis result models independent of terminal, HTTP, or web presentation formatting.
- Preserve deterministic comparison contracts: row order is ignored, logical types are normalized, and null or duplicate comparison keys are reported as invalid.
- Findings must include structured evidence, severity, affected columns, and relevant counts. Bound all examples using the configured report limit.
- Use type hints, clear domain errors, and tests for behavior introduced in each milestone.
- Do not implement ingestion, comparison algorithms, CLI, API, or frontend until the corresponding milestone is approved.
