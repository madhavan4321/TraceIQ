"""Independent QA for source snapshot integrity and CSV staging execution."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import polars as pl
import pytest

from traceiq.ingestion import CsvIngestionError, stage_csv_to_parquet_with_metadata
from traceiq.ingestion.csv import CsvSourceChangedError
from traceiq.models import (
    AnalysisStage,
    AnalysisStageStatus,
    ContentHashStatus,
    DatasetSource,
    InvestigationStatus,
    SourceInvestigationRequest,
)
from traceiq.reporting.assembly import run_investigation_from_sources


def _change_source_mtime_during_capture(
    source_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open
    changed = False

    class MtimeChangingReader:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self.wrapped.__exit__(exc_type, exc_value, traceback)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def read(self, size=-1):
            nonlocal changed
            chunk = self.wrapped.read(size)
            if chunk and not changed:
                changed = True
                metadata = source_path.stat()
                os.utime(
                    source_path,
                    ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 2_000_000_000),
                )
            return chunk

    def open_with_mtime_change(path: Path, mode="r", *args, **kwargs):
        opened = original_open(path, mode, *args, **kwargs)
        if path == source_path and mode == "rb":
            return MtimeChangingReader(opened)
        return opened

    monkeypatch.setattr(Path, "open", open_with_mtime_change)


def test_path_investigation_parses_the_bytes_it_hashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_bytes = b"id,value\n1,original\n"
    replacement_bytes = b"id,value\n2,replaced\n3,replaced\n"
    source = tmp_path / "mutable.csv"
    source.write_bytes(source_bytes)
    original_scan_csv = pl.scan_csv
    scan_count = 0

    def replace_source_before_parse(path, *args, **kwargs):
        nonlocal scan_count
        scan_count += 1
        if scan_count == 1:
            source.write_bytes(replacement_bytes)
        return original_scan_csv(path, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_csv", replace_source_before_parse)
    report = run_investigation_from_sources(
        SourceInvestigationRequest(
            operation="inspect",
            datasets=(DatasetSource(path=source, dataset_id="mutable-source"),),
        )
    )

    assert source.read_bytes() == replacement_bytes
    assert report.datasets[0].content_sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert report.dataset_profiles[0].profile.row_count == 1


def test_metadata_change_during_capture_fails_without_parquet_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "changed-during-capture.csv"
    source.write_bytes(b"id,value\n1,original\n")
    destination = tmp_path / "must-not-exist.parquet"
    _change_source_mtime_during_capture(source, monkeypatch)

    with pytest.raises(CsvSourceChangedError):
        stage_csv_to_parquet_with_metadata(source, destination)

    assert not destination.exists()


def test_capture_change_becomes_failed_profile_and_skipped_compare_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    baseline.write_bytes(b"id,value\n1,a\n")
    candidate.write_bytes(b"id,value\n2,b\n")
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    _change_source_mtime_during_capture(candidate, monkeypatch)

    report = run_investigation_from_sources(
        SourceInvestigationRequest(
            operation="compare",
            datasets=(
                DatasetSource(path=baseline, dataset_id="baseline"),
                DatasetSource(path=candidate, dataset_id="candidate"),
            ),
        )
    )

    assert report.analysis_status == InvestigationStatus.PARTIAL
    assert report.datasets[1].content_sha256 == candidate_hash
    assert report.datasets[1].content_hash_status == ContentHashStatus.AVAILABLE
    status_by_key = {(stage.stage, stage.dataset_id): stage for stage in report.stages}
    assert status_by_key[(AnalysisStage.PROFILE, "baseline")].status == (
        AnalysisStageStatus.COMPLETED
    )
    changed_stage = status_by_key[(AnalysisStage.PROFILE, "candidate")]
    assert changed_stage.status == AnalysisStageStatus.FAILED
    assert changed_stage.error_code == "CsvSourceChangedError"
    assert status_by_key[(AnalysisStage.SCHEMA, None)].status == AnalysisStageStatus.SKIPPED
    assert status_by_key[(AnalysisStage.RECORDS, None)].status == AnalysisStageStatus.SKIPPED


def test_expected_hash_is_checked_before_parquet_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_bytes = b"id,value\n1,stable\n"
    source = tmp_path / "stable.csv"
    source.write_bytes(source_bytes)
    sink_calls: list[Path] = []
    original_sink = pl.LazyFrame.sink_parquet

    def count_sink(frame, destination, *args, **kwargs):
        sink_calls.append(Path(destination))
        return original_sink(frame, destination, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", count_sink)
    expected_hash = hashlib.sha256(source_bytes).hexdigest()
    staged = stage_csv_to_parquet_with_metadata(
        source,
        tmp_path / "valid.parquet",
        expected_sha256=expected_hash,
    )

    assert staged.content_sha256 == expected_hash
    assert len(sink_calls) == 1

    with pytest.raises(CsvIngestionError, match="content hash"):
        stage_csv_to_parquet_with_metadata(
            source,
            tmp_path / "mismatch.parquet",
            expected_sha256="0" * 64,
        )
    assert len(sink_calls) == 1


def test_compare_executes_one_parquet_sink_for_each_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = (tmp_path / "baseline.csv", tmp_path / "candidate.csv")
    sources[0].write_bytes(b"id,value\n1,a\n")
    sources[1].write_bytes(b"id,value\n1,b\n")
    sink_destinations: list[Path] = []
    original_sink = pl.LazyFrame.sink_parquet

    def count_sink(frame, destination, *args, **kwargs):
        sink_destinations.append(Path(destination))
        return original_sink(frame, destination, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", count_sink)
    report = run_investigation_from_sources(
        SourceInvestigationRequest(
            operation="compare",
            datasets=tuple(DatasetSource(path=path) for path in sources),
            comparison_options={"key_columns": ("id",)},
        )
    )

    assert report.analysis_status == InvestigationStatus.COMPLETE
    assert len(sink_destinations) == len(sources)
    assert len(set(sink_destinations)) == len(sources)


def test_permission_denied_source_produces_partial_compare_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed.csv"
    denied = tmp_path / "denied.csv"
    allowed.write_bytes(b"id,value\n1,a\n")
    denied.write_bytes(b"id,value\n2,b\n")
    original_open = Path.open

    def deny_source_read(path: Path, mode="r", *args, **kwargs):
        if path == denied and mode == "rb":
            raise PermissionError(13, "Permission denied", str(path))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_source_read)
    report = run_investigation_from_sources(
        SourceInvestigationRequest(
            operation="compare",
            datasets=(
                DatasetSource(path=allowed, dataset_id="allowed"),
                DatasetSource(path=denied, dataset_id="denied"),
            ),
        )
    )

    assert report.analysis_status == InvestigationStatus.PARTIAL
    assert report.datasets[0].content_hash_status == ContentHashStatus.AVAILABLE
    assert report.datasets[1].content_sha256 is None
    assert report.datasets[1].content_hash_status == ContentHashStatus.INCOMPLETE
    status_by_key = {(stage.stage, stage.dataset_id): stage for stage in report.stages}
    assert status_by_key[(AnalysisStage.PROFILE, "allowed")].status == (
        AnalysisStageStatus.COMPLETED
    )
    denied_stage = status_by_key[(AnalysisStage.PROFILE, "denied")]
    assert denied_stage.status == AnalysisStageStatus.FAILED
    assert "permission denied" in denied_stage.message.casefold()
    assert status_by_key[(AnalysisStage.SCHEMA, None)].status == AnalysisStageStatus.SKIPPED
    assert status_by_key[(AnalysisStage.RECORDS, None)].status == AnalysisStageStatus.SKIPPED
