#!/usr/bin/env python3
"""
Generate a nightly Excel performance report from JSON results.

"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter

LOGGER = logging.getLogger(__name__)

GREY_BLOCK_FILL = PatternFill(start_color="D3D3D3", fill_type="solid")

# Benchmark metric columns: grey the latest row's cell when value changed vs previous date.
BENCHMARK_COLUMNS: tuple[str, ...] = (
    "num_prompts",
    "request_rate",
    "burstiness",
    "max_concurrency",
    "duration",
    "completed",
    "failed",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p99_e2el_ms",
    "mean_audio_rtf",
    "p99_audio_rtf",
    "mean_audio_duration_s",
    "p99_audio_duration_s",
)
# Columns that get float coercion and number format in Excel. Excludes request_rate ("inf" str)
# and max_concurrency (null); leave those as-is. If they become float in the future, they are
# still written correctly from JSON without coercion here.
NUMERIC_FORMAT_COLUMNS: tuple[str, ...] = tuple(
    c for c in BENCHMARK_COLUMNS if c not in ("request_rate", "max_concurrency")
)

_COLUMNS_FILENAME = "nightly_perf_summary_columns.txt"
DEFAULT_INPUT_DIR = os.getenv("DEFAULT_INPUT_DIR") if os.getenv("DEFAULT_INPUT_DIR") else "tests"
DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR") if os.getenv("DEFAULT_OUTPUT_DIR") else "tests"


def _load_summary_columns(script_dir: str) -> list[str]:
    """Load summary column names from a file next to this script; fallback to default if missing."""
    path = os.path.join(script_dir, _COLUMNS_FILENAME)
    default = [
        "date",
        "endpoint_type",
        "backend",
        "model_id",
        "tokenizer_id",
        "num_prompts",
        "request_rate",
        "burstiness",
        "max_concurrency",
        "duration",
        "completed",
        "failed",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "p99_itl_ms",
        "mean_e2el_ms",
        "p99_e2el_ms",
        "mean_audio_rtf",
        "p99_audio_rtf",
        "mean_audio_duration_s",
        "p99_audio_duration_s",
        "commit_sha",
        "build_id",
        "build_url",
        "source_file",
    ]
    if not os.path.isfile(path):
        return default
    columns: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                columns.append(s)
    return columns if columns else default


def _vllm_omni_root() -> str:
    """Resolve vllm-omni repo root: directory that contains a 'tests' subdir (and usually 'tools')."""
    path = os.path.dirname(os.path.abspath(__file__))
    while path and path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, "tests")):
            return path
        path = os.path.dirname(path)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _default_input_dir() -> str:
    """Default: vllm-omni root / DEFAULT_INPUT_DIR (where performance JSON files live)."""
    root = _vllm_omni_root()
    return os.path.join(root, DEFAULT_INPUT_DIR)


def _default_output_file() -> str:
    """Default: vllm-omni root / DEFAULT_OUTPUT_DIR / nightly_perf_<timestamp>.xlsx."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return os.path.join(_vllm_omni_root(), DEFAULT_OUTPUT_DIR, f"nightly_perf_{ts}.xlsx")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Read performance JSON files from vllm-omni/tests/ and generate an Excel report."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=_default_input_dir(),
        help="Directory containing performance JSON files; default is <vllm-omni-root>/DEFAULT_INPUT_DIR.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=_default_output_file(),
        help="Output path of the Excel report; \
            default is <vllm-omni-root>/DEFAULT_OUTPUT_DIR/nightly_perf_<timestamp>.xlsx.",
    )
    parser.add_argument(
        "--commit-sha",
        type=str,
        default=None,
        help="Optional commit SHA; defaults to environment variable BUILDKITE_COMMIT if unset.",
    )
    parser.add_argument(
        "--build-id",
        type=str,
        default=None,
        help="Optional build ID; defaults to environment variable BUILDKITE_BUILD_ID if unset.",
    )
    parser.add_argument(
        "--build-url",
        type=str,
        default=None,
        help="Optional build URL; defaults to environment variable BUILDKITE_BUILD_URL if unset.",
    )
    return parser.parse_args()


def _load_json_file(path: str) -> dict[str, Any] | None:
    """Safely load a single JSON file; return None and log a warning on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("failed to load json '%s': %s", path, exc)
        return None

    if not isinstance(data, dict):
        LOGGER.warning("json root in '%s' is not an object, skip", path)
        return None

    return data


def _iter_json_records(input_dir: str) -> Iterable[dict[str, Any]]:
    """Iterate over JSON files in the input directory and yield normalized records.
    commit_sha/build_id/build_url are not set here; they are applied later only to
    rows with the latest date (see _apply_build_metadata_to_latest_only).
    """
    if not os.path.isdir(input_dir):
        LOGGER.warning("input dir '%s' does not exist or is not a directory", input_dir)
        return

    for entry in sorted(os.listdir(input_dir)):
        if not entry.endswith(".json"):
            continue
        full_path = os.path.join(input_dir, entry)
        if not os.path.isfile(full_path):
            continue

        data = _load_json_file(full_path)
        if data is None:
            continue

        record: dict[str, Any] = dict(data)
        record.setdefault("date", datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
        record["source_file"] = os.path.basename(full_path)
        yield record


def _collect_records(input_dir: str) -> list[dict[str, Any]]:
    """Collect all JSON records into a list."""
    records: list[dict[str, Any]] = []
    for record in _iter_json_records(input_dir):
        records.append(record)
    return records


def _apply_build_metadata_to_latest_only(
    records: Sequence[dict[str, Any]],
    commit_sha: str | None,
    build_id: str | None,
    build_url: str | None,
) -> None:
    """Set commit_sha, build_id, build_url only on rows with the latest date.
    Other rows get None so that build info is not duplicated for older benchmark data.
    """
    if not records:
        return
    max_date = max((r.get("date") or "") for r in records)
    for r in records:
        if (r.get("date") or "") == max_date:
            r["commit_sha"] = commit_sha
            r["build_id"] = build_id
            r["build_url"] = build_url
        else:
            r["commit_sha"] = None
            r["build_id"] = None
            r["build_url"] = None


def _sort_records_for_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort so that same model_id is grouped, newest date first within each group."""
    by_date_desc = sorted(records, key=lambda r: (r.get("date") or ""), reverse=True)
    return sorted(by_date_desc, key=lambda r: (r.get("model_id") or ""))


def _values_differ(a: Any, b: Any) -> bool:
    """Return True if two values are considered different; avoid direct float equality."""
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return False
        if a != a or b != b:
            return True
        return abs(a - b) > 1e-9
    return a != b


def _apply_benchmark_change_highlight(
    ws,
    summary_columns: Sequence[str],
    records: Sequence[dict[str, Any]],
) -> None:
    """Grey cells in the latest row of each model when a benchmark metric changed vs previous date."""
    if not records:
        return
    col_to_index = {c: i + 1 for i, c in enumerate(summary_columns)}
    # Walk by model_id blocks (records already sorted by model_id, date desc).
    i = 0
    while i < len(records):
        model_id = records[i].get("model_id")
        block_start = i
        while i < len(records) and records[i].get("model_id") == model_id:
            i += 1
        block_end = i
        newest_idx = block_start
        prev_idx = block_start + 1 if block_start + 1 < block_end else None
        if prev_idx is None:
            continue
        excel_row = newest_idx + 2
        for col in BENCHMARK_COLUMNS:
            if col not in col_to_index:
                continue
            cur_val = records[newest_idx].get(col)
            prev_val = records[prev_idx].get(col)
            if _values_differ(cur_val, prev_val):
                ws.cell(row=excel_row, column=col_to_index[col]).fill = GREY_BLOCK_FILL


def _build_raw_columns(
    records: Sequence[dict[str, Any]],
    summary_columns: Sequence[str],
) -> list[str]:
    """Infer the column set for the raw sheet based on all records."""
    keys: set[str] = set()
    for record in records:
        keys.update(record.keys())
    # Ensure summary columns appear first; remaining columns sorted alphabetically.
    ordered_keys: list[str] = []
    for key in summary_columns:
        if key in keys:
            ordered_keys.append(key)
            keys.discard(key)
    ordered_keys.extend(sorted(keys))
    return ordered_keys


def _to_float_if_numeric(value: Any) -> Any:
    """Coerce to float when possible so Excel treats as number; avoid #### from narrow columns."""
    if value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, int) else value
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return value


def _write_sheet(
    ws,
    columns: Sequence[str],
    rows: Iterable[dict[str, Any]],
    numeric_columns: Sequence[str] = (),
) -> None:
    """Write column names and row data into the given worksheet."""
    numeric_set = set(numeric_columns)
    ws.append(list(columns))
    for record in rows:
        row_values = []
        for col in columns:
            v = record.get(col)
            if col in numeric_set:
                v = _to_float_if_numeric(v)
            row_values.append(v)
        ws.append(row_values)


def _format_benchmark_columns(
    ws,
    columns: Sequence[str],
    num_data_rows: int,
) -> None:
    """Set number format and column width for numeric benchmark columns so values display (no ####)."""
    numeric_set = set(NUMERIC_FORMAT_COLUMNS)
    for c, col_name in enumerate(columns):
        if col_name not in numeric_set:
            continue
        col_letter = get_column_letter(c + 1)
        ws.column_dimensions[col_letter].width = 14
        for r in range(2, 2 + num_data_rows):
            cell = ws.cell(row=r, column=c + 1)
            if cell.value is not None and isinstance(cell.value, (int, float)):
                cell.number_format = "0.0000"
            elif isinstance(cell.value, str):
                try:
                    cell.value = float(cell.value)
                    cell.number_format = "0.0000"
                except (ValueError, TypeError):
                    pass


def _ensure_parent_dir(path: str) -> None:
    """Ensure that the parent directory of the output file exists."""
    parent = os.path.dirname(os.path.abspath(path))
    if not parent:
        return
    os.makedirs(parent, exist_ok=True)


def generate_excel_report(
    input_dir: str,
    output_file: str,
    commit_sha: str | None,
    build_id: str | None,
    build_url: str | None,
) -> None:
    """Main logic: load JSON records and generate an Excel report."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_columns = _load_summary_columns(script_dir)

    records = _collect_records(input_dir)
    if not records:
        LOGGER.warning("no valid json records found under '%s'", input_dir)

    sorted_records = _sort_records_for_summary(records)
    _apply_build_metadata_to_latest_only(sorted_records, commit_sha, build_id, build_url)

    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = "summary"

    _write_sheet(ws_summary, summary_columns, sorted_records, numeric_columns=NUMERIC_FORMAT_COLUMNS)
    _format_benchmark_columns(ws_summary, summary_columns, len(sorted_records))
    _apply_benchmark_change_highlight(ws_summary, summary_columns, sorted_records)

    if sorted_records:
        raw_columns = _build_raw_columns(sorted_records, summary_columns)
        ws_raw = wb.create_sheet(title="raw")
        _write_sheet(ws_raw, raw_columns, sorted_records)

    _ensure_parent_dir(output_file)
    wb.save(output_file)
    LOGGER.info("excel report saved to '%s'", output_file)


def main() -> None:
    """Command-line entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    commit_sha = args.commit_sha or os.getenv("BUILDKITE_COMMIT")
    build_id = args.build_id or os.getenv("BUILDKITE_BUILD_ID")
    build_url = args.build_url or os.getenv("BUILDKITE_BUILD_URL")

    generate_excel_report(
        input_dir=args.input_dir,
        output_file=args.output_file,
        commit_sha=commit_sha,
        build_id=build_id,
        build_url=build_url,
    )


if __name__ == "__main__":
    main()
