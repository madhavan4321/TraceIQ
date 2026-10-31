"""Normalize Polars physical dtypes into TraceIQ's stable logical vocabulary."""

from __future__ import annotations

import polars as pl

from traceiq.models import LogicalDataType


def normalize_polars_dtype(dtype: pl.DataType) -> LogicalDataType:
    """Map a Polars dtype to a stable logical type, collapsing physical-width details.

    Polars integer widths normalize to ``integer`` and float widths to ``float``.
    Nested element definitions and datetime units/time zones are intentionally not
    represented by the current logical vocabulary. Unsupported Polars types map to
    ``unknown`` instead of being guessed.
    """
    base_type = dtype.base_type()

    if base_type == pl.Null:
        return LogicalDataType.NULL
    if base_type == pl.Boolean:
        return LogicalDataType.BOOLEAN
    integer_types = _available_dtype_classes(
        pl, "Int8", "Int16", "Int32", "Int64", "Int128", "UInt8", "UInt16", "UInt32", "UInt64"
    )
    if base_type in integer_types:
        return LogicalDataType.INTEGER
    if base_type in _available_dtype_classes(pl, "Float32", "Float64"):
        return LogicalDataType.FLOAT
    if base_type == pl.Decimal:
        return LogicalDataType.DECIMAL
    if base_type in _available_dtype_classes(pl, "String", "Utf8"):
        return LogicalDataType.STRING
    if base_type == pl.Date:
        return LogicalDataType.DATE
    if base_type == pl.Datetime:
        return LogicalDataType.DATETIME
    if base_type == pl.Time:
        return LogicalDataType.TIME
    if base_type == pl.Duration:
        return LogicalDataType.DURATION
    if base_type == pl.Binary:
        return LogicalDataType.BINARY
    if base_type in _available_dtype_classes(pl, "Categorical", "Enum"):
        return LogicalDataType.CATEGORICAL
    if base_type in _available_dtype_classes(pl, "List", "Array"):
        return LogicalDataType.LIST
    if base_type == pl.Struct:
        return LogicalDataType.STRUCT
    return LogicalDataType.UNKNOWN


def _available_dtype_classes(module: object, *names: str) -> tuple[object, ...]:
    """Return present Polars dtype classes, allowing older supported Polars releases."""
    return tuple(value for name in names if (value := getattr(module, name, None)) is not None)
