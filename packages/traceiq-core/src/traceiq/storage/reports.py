"""Atomic SQLite persistence for structured investigation reports."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from traceiq.models import InvestigationReport
from traceiq.reporting.json import report_from_json, report_to_json


class ReportStoreError(RuntimeError):
    """Base error raised by the report store."""


class DuplicateReportIdError(ReportStoreError):
    """Raised when attempting to insert a report ID that already exists."""


class ReportNotFoundError(ReportStoreError):
    """Raised when a requested report ID does not exist."""


class ReportStore:
    """Persist reports as deterministic JSON payloads in a local SQLite database."""

    def __init__(self, database_path: str | Path) -> None:
        if str(database_path) == ":memory:":
            raise ValueError("ReportStore requires a persistent SQLite database path")
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, timeout=30.0)

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS investigation_reports (
                        report_id TEXT PRIMARY KEY COLLATE BINARY,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise ReportStoreError(f"Could not initialize report database: {exc}") from exc

    def save(self, report: InvestigationReport) -> None:
        """Atomically insert a report; duplicate IDs are rejected and never overwritten."""
        payload = report_to_json(report)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO investigation_reports(report_id, created_at, payload_json) "
                    "VALUES (?, ?, ?)",
                    (report.report_id, report.created_at.isoformat(), payload),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateReportIdError(f"Report ID {report.report_id!r} already exists") from exc
        except sqlite3.Error as exc:
            raise ReportStoreError(f"Could not save report {report.report_id!r}: {exc}") from exc

    def get(self, report_id: str) -> InvestigationReport:
        """Retrieve and validate a stored report by its exact case-sensitive ID."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT payload_json FROM investigation_reports WHERE report_id = ?",
                    (report_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ReportStoreError(f"Could not retrieve report {report_id!r}: {exc}") from exc
        if row is None:
            raise ReportNotFoundError(f"Report ID {report_id!r} was not found")
        try:
            return report_from_json(row[0])
        except Exception as exc:
            raise ReportStoreError(f"Stored report {report_id!r} is invalid: {exc}") from exc
