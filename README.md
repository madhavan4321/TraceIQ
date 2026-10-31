# TraceIQ

TraceIQ is an open-source data forensics and drift intelligence platform. Its Python core will provide shared investigation contracts and analysis services for the CLI and REST API; the React web application will use the API.

## Current milestone

This repository currently contains the shared Python package scaffold and stable Pydantic model contracts. Dataset ingestion, analysis algorithms, CLI, API, and web application have not been implemented yet.

## Development setup

Requires Python 3.13 or newer. From the repository root, create a virtual environment and install the core with development tools:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Install optional interface dependencies when those milestones are added with `pip install -e ".[cli]"` or `pip install -e ".[api]"`.

## Design notes

- The core distribution is named `traceiq-core` and imports as `traceiq`; interfaces will depend on this installable package rather than copy business logic.
- Public report models carry analysis facts and structured evidence, not terminal or web formatting instructions.
- Comparison contracts specify ignored row order, logical type normalization, and invalid reporting for null or duplicate keys.
- Findings carry severity, affected columns, counts, and evidence. Example payloads are bounded by a configurable per-finding limit.
- Local snapshots/reports and SQLite metadata will be added in later milestones.
