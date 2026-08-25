from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields
from typing import Any

from prettytable import PrettyTable


def coerce_positive_int_scalar(value: object) -> int | None:
    """Coerce a value to a positive int without importing tensor libs.

    Meant to pull positive integers such as sample rate out of the shapes
    that show up across omni stage outputs / configs:

    - plain ``int``: ``44100``
    - ``torch.Tensor`` / numpy scalar: ``tensor(44100)`` (via ``.item()``)
    - ``list`` / ``tuple`` wrap: ``[tensor(44100)]`` or ``[44100]``
    - Mapping field values: ``{"sr": tensor(44100)}`` → coerce the field
      value (key lookup is done by the caller, e.g.
      :func:`resolve_int_by_sequential_keys`)

    Returns ``None`` when the value is missing, unparsable, or not positive.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            coerced = coerce_positive_int_scalar(item)
            if coerced is not None:
                return coerced
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def resolve_int_by_sequential_keys(
    source: Mapping[str, object] | object | None,
    keys: Sequence[str],
) -> int | None:
    """Return the first positive int found by trying ``keys`` in order.

    For each key, looks up ``source[key]`` when ``source`` is a Mapping, otherwise
    ``getattr(source, key, None)``. Values are coerced via :func:`coerce_positive_int_scalar`.
    Returns ``None`` when ``source`` is empty/missing or no key yields a usable int.
    """
    if not source:
        return None
    for key in keys:
        raw = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
        value = coerce_positive_int_scalar(raw)
        if value is not None:
            return value
    return None


def extract_mm_output(mm_source: object) -> Mapping[str, Any]:
    """Return the first non-empty multimodal_output Mapping on ``mm_source``.

    Lookup order:
      1. ``mm_source.multimodal_output`` — top-level attribute / property
      2. ``mm_source.outputs[0].multimodal_output`` — CompletionOutput nesting
         (typical AR audio path)

    Accepts both plain ``dict`` and ``MultimodalPayload`` (a ``Mapping``).
    Returns ``{}`` when neither location yields a non-empty Mapping.

    Args:
        mm_source: Duck-typed container with optional ``multimodal_output`` and/or
            ``outputs`` (e.g. vLLM ``RequestOutput`` or ``OmniRequestOutput``).

    Returns:
        The first non-empty multimodal Mapping found, or an empty ``dict``.
    """
    mm = getattr(mm_source, "multimodal_output", None)
    if isinstance(mm, Mapping) and mm:
        return mm
    outs = getattr(mm_source, "outputs", None)
    if outs:
        nested = getattr(outs[0], "multimodal_output", None)
        if isinstance(nested, Mapping) and nested:
            return nested
    return {}


def iter_mm_outputs(mm_source: object) -> list[Mapping[str, Any]]:
    """Collect all non-empty multimodal_output Mappings from ``mm_source``.

    Lookup order:
      1. ``mm_source.multimodal_output`` — top-level attribute / property
      2. each ``mm_source.outputs[i].multimodal_output`` — every CompletionOutput
         entry that carries a non-empty multimodal Mapping

    Accepts both plain ``dict`` and ``MultimodalPayload`` (a ``Mapping``).
    Used by stage-pool metrics aggregation that needs to visit every mm payload.

    Args:
        mm_source: Duck-typed container with optional ``multimodal_output`` and/or
            ``outputs`` (e.g. vLLM ``RequestOutput`` or ``OmniRequestOutput``).

    Returns:
        A list of non-empty multimodal Mappings in discovery order. Empty when
        none are present.
    """
    multimodal_outputs: list[Mapping[str, Any]] = []
    outer_mm = getattr(mm_source, "multimodal_output", None)
    if isinstance(outer_mm, Mapping) and outer_mm:
        multimodal_outputs.append(outer_mm)
    for output in getattr(mm_source, "outputs", None) or []:
        inner_mm = getattr(output, "multimodal_output", None)
        if isinstance(inner_mm, Mapping) and inner_mm:
            multimodal_outputs.append(inner_mm)
    return multimodal_outputs


def count_audio_chunk_frames(audio_chunk: object) -> int:
    """Count frames (samples) in one audio tensor / chunk.

    Audio chunks are concatenated on dim=-1 in the output processor, so the
    frame/sample axis is the last dim (e.g. ``[channels, frames]``). Keep this
    aligned with serving_chat.py: audio tensors are consumed as ``(T,)``,
    ``(C, T)``, or ``(B, C, T)``. Flattening would corrupt multi-channel audio.

    Args:
        audio_chunk: A single audio tensor/array-like, or a scalar-like value.

    Returns:
        Frame count for this chunk. Uses ``shape[-1]`` when shaped; otherwise
        ``len(...)``; scalars / unlenable values count as ``1``.
    """
    shape = getattr(audio_chunk, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    try:
        return len(audio_chunk)  # type: ignore[arg-type]
    except TypeError:
        return 1


def count_audio_frames(mm_out: Mapping[str, Any]) -> int:
    """Sum frame counts over all audio chunks in ``mm_out["audio"]`` or with other related keys.

    For multi-dim tensors (e.g. shape ``[channels, samples]``) the last axis is
    the sample dim; for 1-D tensors the only axis is the sample dim; scalars
    count as 1. Missing or empty ``audio`` yields ``0``.

    Args:
        mm_out: A multimodal_output Mapping (plain ``dict`` or
            ``MultimodalPayload``) that may contain an ``audio`` or related key whose value
            is one chunk or a list of chunks.

    Returns:
        Total audio frames (samples) across all chunks.
    """
    audio_chunks = None
    if isinstance(mm_out, Mapping):
        for key in ("audio", "model_outputs"):
            audio_chunks = mm_out.get(key)
            if audio_chunks is not None:
                break
    if audio_chunks is None:
        return 0
    chunks = audio_chunks if isinstance(audio_chunks, list) else [audio_chunks]
    return sum(count_audio_chunk_frames(chunk) for chunk in chunks)


def count_image_pixels(value: object) -> int:
    """Count pixels in one image value, or sum over a nested list/tuple.

    Accepts PIL-like objects (``size=(W, H)``), tensors / arrays with a
    ``shape`` attribute, and nested ``list`` / ``tuple`` containers.

    Shape heuristics (aligned with StagePool image metrics):

    - ``ndim >= 4`` (e.g. ``BCHW``): ``B * H * W`` via
      ``dims[0] * dims[-2] * dims[-1]``
    - ``ndim == 3`` and ``dims[0] in (1, 3, 4)``: CHW → ``H * W``
    - ``ndim == 3`` and ``dims[-1] in (1, 3, 4)``: HWC → ``H * W``
    - otherwise: ``dims[-2] * dims[-1]``

    Returns ``0`` when ``value`` is missing or cannot be interpreted.
    """
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(count_image_pixels(item) for item in value)

    size = getattr(value, "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        try:
            return int(size[0]) * int(size[1])
        except (TypeError, ValueError):
            return 0

    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return 0
    dims = [int(dim) for dim in shape]
    if len(dims) >= 4:
        return dims[0] * dims[-2] * dims[-1]
    if len(dims) == 3 and dims[0] in (1, 3, 4):
        return dims[1] * dims[2]
    if len(dims) == 3 and dims[-1] in (1, 3, 4):
        return dims[0] * dims[1]
    return dims[-2] * dims[-1]


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
            if not value:
                return ""
            try:
                avg = sum(float(v) for v in value) / len(value)
                return f"{avg:,.3f} (n={len(value)})"
            except (TypeError, ValueError):
                return ", ".join(str(v) for v in value)
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
        # PrettyTable requires unique field names.  When the same column key
        # appears more than once (e.g. two events with the same stage_id),
        # deduplicate by appending _2, _3, … to the later occurrences.
        if len(col_headers) != len(set(col_headers)):
            seen: dict[str, int] = {}
            deduped: list[str] = []
            for h in col_headers:
                n = seen.get(h, 0)
                seen[h] = n + 1
                deduped.append(h if n == 0 else f"{h}_{n + 1}")
            col_headers = deduped
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
                for output in outs:
                    # In DELTA mode token_ids only contains the latest chunk.
                    # Omni's output processor attaches the cumulative sequence
                    # for inter-stage routing and accurate metrics.
                    tokens = getattr(output, "cumulative_token_ids", None)
                    if tokens is None:
                        tokens = getattr(output, "token_ids", None)
                    if tokens is not None:
                        total += len(tokens)
        except Exception:
            # Ignore any issues with individual outputs to keep token counting best-effort.
            pass
    return total
