import hashlib
from pathlib import Path

import polars as pl
import pytest

from traceiq.ingestion import (
    CsvIngestionError,
    CsvIngestionOptions,
    profile_csv,
    profile_lazy_frame,
    stage_csv_to_parquet,
    stage_csv_to_parquet_with_metadata,
)
from traceiq.ingestion import csv as csv_module
from traceiq.ingestion.csv import CsvEncodingError, CsvSourceChangedError, MalformedCsvError
from traceiq.models import LogicalDataType


def write_csv(tmp_path: Path, content: bytes, name: str = "input.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def track_snapshot_tempdirs(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    original_temporary_directory = csv_module.tempfile.TemporaryDirectory
    created: list[Path] = []

    def track(*args, **kwargs):
        temporary_directory = original_temporary_directory(*args, **kwargs)
        created.append(Path(temporary_directory.name))
        return temporary_directory

    monkeypatch.setattr(csv_module.tempfile, "TemporaryDirectory", track)
    return created


def test_profile_counts_shape_types_nulls_and_duplicate_excess_rows(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        b"id,amount,active\n1,2.5,true\n2,,false\n2,,false\n3,4.0,true\n",
    )

    profile = profile_csv(path)

    assert profile.row_count == 4
    assert profile.column_count == 3
    assert profile.duplicate_row_count == 1
    assert [column.name for column in profile.columns] == ["id", "amount", "active"]
    assert [column.logical_type for column in profile.columns] == [
        LogicalDataType.INTEGER,
        LogicalDataType.FLOAT,
        LogicalDataType.BOOLEAN,
    ]
    assert profile.columns[1].null_count == 2
    assert profile.columns[1].null_percentage == 50.0


def test_duplicate_count_is_independent_of_row_order_and_counts_excess(tmp_path: Path) -> None:
    first = write_csv(tmp_path, b"x\na\na\nb\na\n", "first.csv")
    second = write_csv(tmp_path, b"x\nb\na\na\na\n", "second.csv")

    assert profile_csv(first).duplicate_row_count == 2
    assert profile_csv(second).duplicate_row_count == 2


def test_null_values_are_equal_when_detecting_duplicate_rows(tmp_path: Path) -> None:
    path = write_csv(tmp_path, b"x,y\n,1\n,1\na,2\n")

    assert profile_csv(path).duplicate_row_count == 1


def test_header_only_csv_reports_zero_rows_and_zero_null_percentages(tmp_path: Path) -> None:
    path = write_csv(tmp_path, b"id,name\n")

    profile = profile_csv(path)

    assert profile.row_count == 0
    assert profile.column_count == 2
    assert [column.null_percentage for column in profile.columns] == [0.0, 0.0]
    assert profile.duplicate_row_count == 0


def test_utf8_bom_is_accepted(tmp_path: Path) -> None:
    path = write_csv(tmp_path, b"\xef\xbb\xbfname,age\nAda,36\n")

    profile = profile_csv(path)

    assert [column.name for column in profile.columns] == ["name", "age"]


def test_invalid_utf8_is_reported_without_lossy_decoding(tmp_path: Path) -> None:
    path = write_csv(tmp_path, b"name\nAda\xff\n")

    with pytest.raises(CsvEncodingError, match="not valid UTF-8"):
        profile_csv(path)


def test_ragged_csv_is_reported_as_malformed(tmp_path: Path) -> None:
    path = write_csv(tmp_path, b"a,b\n1,2\n3,4,5\n")

    with pytest.raises(MalformedCsvError):
        profile_csv(path)


def test_value_outside_inferred_schema_is_not_silently_discarded(tmp_path: Path) -> None:
    rows = ["value", *(str(index) for index in range(1_001)), "not-an-integer"]
    path = write_csv(tmp_path, ("\n".join(rows) + "\n").encode())

    with pytest.raises(MalformedCsvError):
        profile_csv(path)


def test_missing_file_raises_ingestion_error(tmp_path: Path) -> None:
    with pytest.raises(CsvIngestionError, match="does not exist"):
        profile_csv(tmp_path / "missing.csv")


def test_directory_path_is_reported_as_not_a_regular_file(tmp_path: Path) -> None:
    with pytest.raises(CsvIngestionError, match="not a regular file"):
        profile_csv(tmp_path)


def test_permission_error_is_not_misreported_as_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_csv(tmp_path, b"id\n1\n")
    original_open = Path.open

    def deny_source_open(path: Path, *args, **kwargs):
        if path == source:
            raise PermissionError("access denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_source_open)
    with pytest.raises(CsvIngestionError, match="Permission denied reading CSV file"):
        profile_csv(source)


def test_csv_source_metadata_change_during_capture_fails_closed_and_cleans_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"id,value\n1,captured\n"
    source = write_csv(tmp_path, content)
    snapshot_dirs = track_snapshot_tempdirs(monkeypatch)
    original_signature = csv_module._source_signature
    source_stat_calls = 0

    def change_final_source_signature(path: Path):
        nonlocal source_stat_calls
        signature = original_signature(path)
        if path == source:
            source_stat_calls += 1
            if source_stat_calls == 2:
                return signature.__class__(
                    device=signature.device,
                    inode=signature.inode,
                    mode=signature.mode,
                    size=signature.size,
                    mtime_ns=signature.mtime_ns + 1,
                    ctime_ns=signature.ctime_ns,
                )
        return signature

    monkeypatch.setattr(csv_module, "_source_signature", change_final_source_signature)
    with pytest.raises(CsvSourceChangedError, match="changed during capture") as error:
        stage_csv_to_parquet_with_metadata(source, tmp_path / "changed.parquet")

    assert error.value.content_sha256 == hashlib.sha256(content).hexdigest()
    assert error.value.size_bytes == len(content)
    assert source_stat_calls == 2
    assert len(snapshot_dirs) == 1
    assert not snapshot_dirs[0].exists()


def test_snapshot_temp_is_cleaned_after_encoding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_csv(tmp_path, b"name\nAda\xff\n")
    snapshot_dirs = track_snapshot_tempdirs(monkeypatch)

    with pytest.raises(CsvEncodingError):
        stage_csv_to_parquet_with_metadata(source, tmp_path / "invalid.parquet")

    assert len(snapshot_dirs) == 1
    assert not snapshot_dirs[0].exists()


def test_snapshot_temp_is_cleaned_after_expected_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"id,value\n1,a\n"
    source = write_csv(tmp_path, content)
    snapshot_dirs = track_snapshot_tempdirs(monkeypatch)

    with pytest.raises(CsvIngestionError, match="content hash") as error:
        stage_csv_to_parquet_with_metadata(
            source,
            tmp_path / "wrong-hash.parquet",
            expected_sha256="0" * 64,
        )

    assert error.value.content_sha256 == hashlib.sha256(content).hexdigest()
    assert error.value.size_bytes == len(content)
    assert len(snapshot_dirs) == 1
    assert not snapshot_dirs[0].exists()


def test_snapshot_temp_is_cleaned_when_source_cannot_be_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_csv(tmp_path, b"id,value\n1,a\n")
    snapshot_dirs = track_snapshot_tempdirs(monkeypatch)
    original_open = Path.open

    def deny_source_read(path: Path, mode="r", *args, **kwargs):
        if path == source and mode == "rb":
            raise PermissionError("access denied")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_source_read)
    with pytest.raises(CsvIngestionError, match="Permission denied while capturing") as error:
        stage_csv_to_parquet_with_metadata(source, tmp_path / "unreadable.parquet")

    assert error.value.content_sha256 is None
    assert error.value.size_bytes is None
    assert len(snapshot_dirs) == 1
    assert not snapshot_dirs[0].exists()


def test_snapshot_temp_is_cleaned_after_parquet_sink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_csv(tmp_path, b"id,value\n1,a\n")
    snapshot_dirs = track_snapshot_tempdirs(monkeypatch)

    def fail_sink(self, *args, **kwargs):
        raise pl.exceptions.ComputeError("simulated parse failure")

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fail_sink)
    with pytest.raises(MalformedCsvError, match="Could not stage CSV"):
        stage_csv_to_parquet_with_metadata(source, tmp_path / "failed.parquet")

    assert len(snapshot_dirs) == 1
    assert not snapshot_dirs[0].exists()
    assert not (tmp_path / "failed.parquet").exists()


def test_invalid_csv_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="delimiter"):
        CsvIngestionOptions(delimiter="::")
    with pytest.raises(ValueError, match="infer_schema_length"):
        CsvIngestionOptions(infer_schema_length=0)


def test_parquet_staging_supports_lazy_profile_and_refuses_overwrite(tmp_path: Path) -> None:
    source = write_csv(tmp_path, b"id,value\n1,a\n2,b\n")
    staged = tmp_path / "staged.parquet"

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    stage_csv_to_parquet(source, staged, expected_sha256=digest)

    profile = profile_lazy_frame(pl.scan_parquet(staged), source=staged)
    assert profile.row_count == 2
    assert profile.column_count == 2
    with pytest.raises(CsvIngestionError, match="already exists"):
        stage_csv_to_parquet(source, staged)


def test_metadata_staging_returns_source_hash_and_exact_byte_size(tmp_path: Path) -> None:
    content = b"id,value\n1,a\n2,b\n"
    source = write_csv(tmp_path, content)

    result = stage_csv_to_parquet_with_metadata(source, tmp_path / "metadata.parquet")

    assert result.path == tmp_path / "metadata.parquet"
    assert result.content_sha256 == hashlib.sha256(content).hexdigest()
    assert result.size_bytes == len(content)


def test_invalid_utf8_error_retains_hash_when_all_source_bytes_were_read(
    tmp_path: Path,
) -> None:
    content = b"name\nAda\xff\n"
    source = write_csv(tmp_path, content)

    with pytest.raises(CsvEncodingError) as error:
        stage_csv_to_parquet_with_metadata(source, tmp_path / "invalid.parquet")

    assert error.value.content_sha256 == hashlib.sha256(content).hexdigest()
    assert error.value.size_bytes == len(content)


def test_parquet_staging_rejects_a_mismatched_dataset_hash(tmp_path: Path) -> None:
    source = write_csv(tmp_path, b"id\n1\n")

    with pytest.raises(CsvIngestionError, match="content hash"):
        stage_csv_to_parquet(
            source,
            tmp_path / "staged.parquet",
            expected_sha256="0" * 64,
        )
