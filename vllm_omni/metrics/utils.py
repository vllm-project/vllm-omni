from collections.abc import Callable
from dataclasses import fields
from typing import Any

from prettytable import PrettyTable


def _build_field_defs(
    cls: type,
    exclude: set[str],
    transforms: dict[str, tuple[str, Callable]] | None = None,
) -> list[tuple[str, Callable[[Any], Any]]]:
    """Auto-generate field definitions from a dataclass.

    Args:
        cls: The dataclass type to extract fields from.
        exclude: Set of field names to exclude from output.
        transforms: Optional mapping of field transformations.
            Format: {original_name: (display_name, transform_fn)}

    Returns:
        List of (display_name, getter_fn) tuples for table generation.
    """
    transforms = transforms or {}
    result = []
    for f in fields(cls):
        if f.name in exclude:
            continue
        if f.name in transforms:
            display_name, transform_fn = transforms[f.name]
            # Capture variables in closure to avoid late binding issues
            result.append((display_name, lambda e, fn=transform_fn, n=f.name: fn(getattr(e, n))))
        else:
            result.append((f.name, lambda e, n=f.name: getattr(e, n)))
    return result


def _build_row(evt: Any, field_defs: list[tuple[str, Callable]]) -> dict[str, Any]:
    """Build a row dict from an event object using field definitions.

    Args:
        evt:  The event object (dataclass instance).
        field_defs: List of (field_name, getter_fn) tuples.

    Returns:
        Dict mapping field names to their values.
    """
    return {name: getter(evt) for name, getter in field_defs}


def _get_field_names(field_defs: list[tuple[str, Callable]]) -> list[str]:
    """Extract field names from field definitions.

    Args:
        field_defs: List of (field_name, getter_fn) tuples.

    Returns:
        List of field names.
    """
    return [name for name, _ in field_defs]


def _format_table(
    title: str,
    data: dict[str, Any] | list[dict[str, Any]],
    value_fields: list[str],
    column_key: str | None = None,
    column_prefix: str = "",
) -> str:
    """Format a table for display.

    Supports two modes:
    1. Single-column mode:  data is a dict, displays as Field | Value
    2. Multi-column mode: data is a list of dicts, displays as Field | col1 | col2 | ...

    Args:
        title:  Table title.
        data: Either a single dict (single-column) or list of dicts (multi-column).
        value_fields: List of field names to display as rows.
        column_key: Key in each dict used as column header (required for multi-column mode).
        column_prefix: Optional prefix for column headers (multi-column mode only).

    Returns:
        Formatted table string.
    """
    if not data:
        return f"[{title}] <empty>"

    def _format_value(value: Any) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:,.3f}"
        if isinstance(value, list):
            return ", ".join(str(f"{v:,.3f}") for v in value)
        return str(value)

    table = PrettyTable()

    # Single-column mode:  data is a dict
    if isinstance(data, dict):
        table.field_names = ["Field", "Value"]
        table.align["Field"] = "l"
        table.align["Value"] = "r"
        for field in value_fields:
            if field in data:
                if isinstance(data[field], dict):
                    for sub_key, sub_value in data[field].items():
                        table.add_row([f"{sub_key}", _format_value(sub_value)])
                else:
                    table.add_row([field, _format_value(data[field])])

    # Multi-column mode: data is a list of dicts
    else:
        if column_key is None:
            raise ValueError("column_key is required for multi-column mode")
        col_headers = [f"{column_prefix}{row.get(column_key, '?')}" for row in data]
        table.field_names = ["Field"] + col_headers
        table.align["Field"] = "l"
        for col in col_headers:
            table.align[col] = "r"
        for field in value_fields:
            row_values = [_format_value(r.get(field, "")) for r in data]
            table.add_row([field] + row_values)

    return "\n".join([f"[{title}]", table.get_string()])


def count_tokens_from_outputs(engine_outputs: list[Any]) -> int:
    total = 0
    for _ro in engine_outputs:
        try:
            outs = getattr(_ro, "outputs", None)
            if outs and len(outs) > 0:
                tokens = getattr(outs[0], "token_ids", None)
                if tokens is not None:
                    total += len(tokens)
        except Exception:
            # Ignore any issues with individual outputs to keep token counting best-effort.
            pass
    return total
