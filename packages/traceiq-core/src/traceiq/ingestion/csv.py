"""CSV scanning and dataset profiling backed by Polars lazy queries."""

from __future__ import annotations

import codecs
import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from traceiq.models import ColumnProfile, DatasetProfile
from traceiq.types import normalize_polars_dtype


class CsvIngestionError(ValueError):
    """Base error for CSV input that TraceIQ cannot inspect."""

    def __init__(
        self,
        message: str,
        *,
        content_sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.content_sha256 = content_sha256
        self.size_bytes = size_bytes


class CsvEncodingError(CsvIngestionError):
    """Raised when the source is not valid UTF-8 text."""


class MalformedCsvError(CsvIngestionError):
    """Raised when CSV structure or values cannot be parsed safely."""


class CsvSourceChangedError(CsvIngestionError):
    """Raised when the CSV source changes or cannot be checked during capture."""


@dataclass(frozen=True, slots=True)
class CsvIngestionOptions:
    """Options that affect deterministic CSV parsing and type inference."""

    delimiter: str = ","
    infer_schema_length: int = 1_000
    try_parse_dates: bool = True

    def __post_init__(self) -> None:
        if len(self.delimiter) != 1 or self.delimiter in '\r\n\0"':
            raise ValueError("delimiter must be one non-newline, non-quote character")
        if self.infer_schema_length < 1:
            raise ValueError("infer_schema_length must be positive")


@dataclass(frozen=True, slots=True)
class StagedCsv:
    """A staged Parquet file and metadata computed while validating its CSV source."""

    path: Path
    content_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _SourceSignature:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _validate_utf8(path: Path) -> tuple[str, int]:
    """Validate UTF-8 and hash a source incrementally without materializing it."""
    _ensure_regular_file(path)
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    digest = hashlib.sha256()
    size_bytes = 0
    invalid_encoding = False
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
                if not invalid_encoding:
                    try:
                        decoder.decode(chunk)
                    except UnicodeDecodeError:
                        # Continue hashing the complete source so a readable but invalid
                        # CSV still has a trustworthy content identity.
                        invalid_encoding = True
    except PermissionError as exc:
        raise CsvIngestionError(f"Permission denied reading CSV file {path}: {exc}") from exc
    except OSError as exc:
        raise CsvIngestionError(f"Could not read CSV file {path}: {exc}") from exc
    if not invalid_encoding:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            invalid_encoding = True
    content_sha256 = digest.hexdigest()
    if invalid_encoding:
        raise CsvEncodingError(
            f"CSV is not valid UTF-8: {path}",
            content_sha256=content_sha256,
            size_bytes=size_bytes,
        )
    return content_sha256, size_bytes


def _ensure_regular_file(path: Path) -> None:
    _source_signature(path)


def _source_signature(path: Path) -> _SourceSignature:
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise CsvIngestionError(f"CSV file does not exist: {path}") from exc
    except PermissionError as exc:
        raise CsvIngestionError(f"Permission denied accessing CSV file {path}: {exc}") from exc
    except OSError as exc:
        raise CsvIngestionError(f"Could not inspect CSV file {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CsvIngestionError(f"CSV path is not a regular file: {path}")
    return _signature_from_stat(metadata)


def _signature_from_stat(metadata: os.stat_result) -> _SourceSignature:
    return _SourceSignature(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _same_file_state(left: _SourceSignature, right: _SourceSignature) -> bool:
    """Compare identity and mutable content metadata across stat/fstat APIs."""
    return (
        left.device == right.device
        and left.inode == right.inode
        and left.mode == right.mode
        and left.size == right.size
        and left.mtime_ns == right.mtime_ns
    )


def _capture_csv_snapshot(source_path: Path, snapshot_path: Path) -> tuple[str, int]:
    """Copy the opened source while hashing and validating the exact copied bytes."""
    path_before_open = _source_signature(source_path)
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    digest = hashlib.sha256()
    size_bytes = 0
    invalid_encoding = False
    try:
        with source_path.open("rb") as source, snapshot_path.open("wb") as snapshot:
            try:
                opened_before_capture = _signature_from_stat(os.fstat(source.fileno()))
            except OSError as exc:
                raise CsvSourceChangedError(
                    f"Could not inspect opened CSV source before capture: {source_path}"
                ) from exc
            if not _same_file_state(opened_before_capture, path_before_open):
                raise CsvSourceChangedError(
                    f"CSV source changed while it was being opened: {source_path}"
                )
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
                snapshot.write(chunk)
                if not invalid_encoding:
                    try:
                        decoder.decode(chunk)
                    except UnicodeDecodeError:
                        # Continue copying and hashing all bytes for a complete identity.
                        invalid_encoding = True
            try:
                opened_after_capture = _signature_from_stat(os.fstat(source.fileno()))
            except OSError as exc:
                raise CsvSourceChangedError(
                    f"Could not verify CSV source stability after capture: {source_path}",
                    content_sha256=digest.hexdigest(),
                    size_bytes=size_bytes,
                ) from exc
    except FileNotFoundError as exc:
        raise CsvIngestionError(f"CSV file does not exist: {source_path}") from exc
    except PermissionError as exc:
        raise CsvIngestionError(
            f"Permission denied while capturing CSV file {source_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise CsvIngestionError(f"Could not capture CSV file {source_path}: {exc}") from exc
    content_sha256 = digest.hexdigest()
    try:
        path_after_capture = _source_signature(source_path)
    except CsvIngestionError as exc:
        raise CsvSourceChangedError(
            f"Could not verify that CSV source remained stable after capture: {source_path}",
            content_sha256=content_sha256,
            size_bytes=size_bytes,
        ) from exc
    if (
        opened_before_capture != opened_after_capture
        or path_before_open != path_after_capture
        or not _same_file_state(opened_after_capture, path_after_capture)
    ):
        raise CsvSourceChangedError(
            f"CSV source changed during capture: {source_path}",
            content_sha256=content_sha256,
            size_bytes=size_bytes,
        )
    if not invalid_encoding:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            invalid_encoding = True
    if invalid_encoding:
        raise CsvEncodingError(
            f"CSV is not valid UTF-8: {source_path}",
            content_sha256=content_sha256,
            size_bytes=size_bytes,
        )
    return content_sha256, size_bytes


def _collect(lazy_frame: pl.LazyFrame, source: str | Path) -> pl.DataFrame:
    try:
        return lazy_frame.collect(engine="streaming")
    except (pl.exceptions.PolarsError, OSError) as exc:
        raise MalformedCsvError(f"Could not evaluate dataset {source}: {exc}") from exc


def _scan_csv_with_metadata(
    source: str | Path,
    options: CsvIngestionOptions | None = None,
    *,
    expected_sha256: str | None = None,
) -> tuple[pl.LazyFrame, str, int]:
    """Return a validated lazy scan and metadata from its single validation pass."""
    path = Path(source)
    _ensure_regular_file(path)
    actual_sha256, size_bytes = _validate_utf8(path)
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
        raise CsvIngestionError(
            f"CSV content hash does not match dataset metadata: {path}",
            content_sha256=actual_sha256,
            size_bytes=size_bytes,
        )
    frame = _create_csv_scan(
        path,
        options,
        content_sha256=actual_sha256,
        size_bytes=size_bytes,
    )
    return frame, actual_sha256, size_bytes


def _create_csv_scan(
    path: Path,
    options: CsvIngestionOptions | None,
    *,
    content_sha256: str,
    size_bytes: int,
) -> pl.LazyFrame:
    """Create a Polars scan and attach complete source metadata to parse errors."""
    settings = options or CsvIngestionOptions()

    try:
        frame = pl.scan_csv(
            path,
            separator=settings.delimiter,
            encoding="utf8",
            infer_schema_length=settings.infer_schema_length,
            try_parse_dates=settings.try_parse_dates,
            ignore_errors=False,
            truncate_ragged_lines=False,
            raise_if_empty=True,
        )
    except pl.exceptions.NoDataError as exc:
        raise MalformedCsvError(
            f"CSV file has no header or data: {path}",
            content_sha256=content_sha256,
            size_bytes=size_bytes,
        ) from exc
    except (pl.exceptions.PolarsError, OSError) as exc:
        raise MalformedCsvError(
            f"Could not read CSV structure from {path}: {exc}",
            content_sha256=content_sha256,
            size_bytes=size_bytes,
        ) from exc
    return frame


def scan_csv(
    source: str | Path,
    options: CsvIngestionOptions | None = None,
    *,
    expected_sha256: str | None = None,
) -> pl.LazyFrame:
    """Hash-preflight a source, then return a lazy scan over the original path.

    ``expected_sha256`` validates the file before this lazy plan is returned; it does not
    bind later collection to those same bytes. Callers that need hash-to-data integrity
    must use :func:`stage_csv_to_parquet_with_metadata`, which parses a captured snapshot.
    """
    frame, _, _ = _scan_csv_with_metadata(
        source, options, expected_sha256=expected_sha256
    )
    return frame


def profile_lazy_frame(
    frame: pl.LazyFrame,
    *,
    source: str | Path = "<lazy frame>",
) -> DatasetProfile:
    """Profile a lazy frame with streaming aggregations, not Python row materialization.

    Null values compare equal for duplicate-row accounting. The count is the number of
    excess occurrences (sum of multiplicity minus one), so row order has no effect.
    """
    try:
        schema = frame.collect_schema()
    except pl.exceptions.NoDataError as exc:
        raise MalformedCsvError(f"CSV file has no header or data: {source}") from exc
    except (pl.exceptions.PolarsError, OSError) as exc:
        raise MalformedCsvError(f"Could not read dataset structure from {source}: {exc}") from exc

    column_names = list(schema.names())
    expressions: list[pl.Expr] = [pl.len().alias("__traceiq_row_count")]
    expressions.extend(
        pl.col(name).is_null().sum().alias(f"__traceiq_null_{index}")
        for index, name in enumerate(column_names)
    )
    totals = _collect(frame.select(expressions), source)
    row_count = int(totals.item(0, "__traceiq_row_count"))

    columns: list[ColumnProfile] = []
    for index, name in enumerate(column_names):
        null_count = int(totals.item(0, f"__traceiq_null_{index}"))
        null_percentage = 100.0 * null_count / row_count if row_count else 0.0
        columns.append(
            ColumnProfile(
                name=name,
                logical_type=normalize_polars_dtype(schema[name]),
                null_count=null_count,
                null_percentage=null_percentage,
            )
        )

    duplicate_row_count = 0
    if column_names and row_count:
        multiplicities = frame.group_by(column_names).agg(pl.len().alias("__traceiq_count"))
        duplicate_total = _collect(
            multiplicities.select(
                (pl.col("__traceiq_count") - 1).sum().alias("__traceiq_duplicate_rows")
            ),
            source,
        )
        value = duplicate_total.item(0, "__traceiq_duplicate_rows")
        duplicate_row_count = int(value or 0)

    return DatasetProfile(
        row_count=row_count,
        column_count=len(column_names),
        columns=tuple(columns),
        duplicate_row_count=duplicate_row_count,
    )


def profile_csv(
    source: str | Path,
    options: CsvIngestionOptions | None = None,
) -> DatasetProfile:
    """Read and profile a CSV, streaming rows through Polars aggregations."""
    path = Path(source)
    return profile_lazy_frame(scan_csv(path, options), source=path)


def stage_csv_to_parquet(
    source: str | Path,
    destination: str | Path,
    options: CsvIngestionOptions | None = None,
    *,
    expected_sha256: str | None = None,
) -> Path:
    """Parse a CSV once into temporary Parquet for repeated lazy analysis scans.

    This performs an incremental UTF-8/hash validation pass, then trades temporary disk
    space and one Parquet write for avoiding repeated CSV parsing. The streaming sink
    bounds Python-side materialization; downstream group/join operators may still need
    memory proportional to distinct keys or rows.
    """
    return stage_csv_to_parquet_with_metadata(
        source,
        destination,
        options,
        expected_sha256=expected_sha256,
    ).path


def stage_csv_to_parquet_with_metadata(
    source: str | Path,
    destination: str | Path,
    options: CsvIngestionOptions | None = None,
    *,
    expected_sha256: str | None = None,
) -> StagedCsv:
    """Stage one CSV and return its computed hash and byte size without pre-hashing it."""
    output_path = Path(destination)
    if output_path.exists():
        raise CsvIngestionError(f"Staging destination already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(source)
    with tempfile.TemporaryDirectory(prefix="traceiq-csv-snapshot-") as temp_dir:
        snapshot_path = Path(temp_dir) / "source.csv"
        content_sha256, size_bytes = _capture_csv_snapshot(source_path, snapshot_path)
        if expected_sha256 is not None and content_sha256.lower() != expected_sha256.lower():
            raise CsvIngestionError(
                f"CSV content hash does not match dataset metadata: {source_path}",
                content_sha256=content_sha256,
                size_bytes=size_bytes,
            )
        frame = _create_csv_scan(
            snapshot_path,
            options,
            content_sha256=content_sha256,
            size_bytes=size_bytes,
        )
        try:
            frame.sink_parquet(output_path, compression="zstd", engine="streaming")
        except (pl.exceptions.PolarsError, OSError) as exc:
            output_path.unlink(missing_ok=True)
            raise MalformedCsvError(
                f"Could not stage CSV file {source_path}: {exc}",
                content_sha256=content_sha256,
                size_bytes=size_bytes,
            ) from exc
    return StagedCsv(
        path=output_path,
        content_sha256=content_sha256,
        size_bytes=size_bytes,
    )
