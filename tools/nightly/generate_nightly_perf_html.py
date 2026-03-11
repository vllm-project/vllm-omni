#!/usr/bin/env python3
"""
Generate a nightly HTML performance report from JSON results.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger(__name__)

_RESULT_JSON_PREFIX = "result_test_"
# Fallback to 'tests' when env vars are not set, to match CI_nightly_perf.md defaults.
DEFAULT_INPUT_DIR = os.getenv("DEFAULT_INPUT_DIR") if os.getenv("DEFAULT_INPUT_DIR") else "tests"
DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR") if os.getenv("DEFAULT_OUTPUT_DIR") else "tests"


def _vllm_omni_root() -> str:
    path = os.path.dirname(os.path.abspath(__file__))
    while path and path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, "tests")):
            return path
        path = os.path.dirname(path)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _default_input_dir() -> str:
    root = _vllm_omni_root()
    return os.path.join(root, DEFAULT_INPUT_DIR)


def _default_output_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return os.path.join(_vllm_omni_root(), DEFAULT_OUTPUT_DIR, f"nightly_perf_{ts}.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read performance JSON files from vllm-omni/tests/ and generate an HTML report."
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
        help="Output path of the HTML report; \
            default is <vllm-omni-root>/DEFAULT_OUTPUT_DIR/nightly_perf_<timestamp>.html.",
    )
    return parser.parse_args()


def _load_json_file(path: str) -> dict[str, Any] | None:
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


def _parse_from_filename(filename: str) -> dict[str, Any]:
    name, ext = os.path.splitext(filename)
    if ext != ".json" or not name.startswith(_RESULT_JSON_PREFIX):
        return {}

    core = name[len(_RESULT_JSON_PREFIX) :]
    parts = core.split("_")
    if len(parts) < 5:
        LOGGER.warning("filename '%s' does not match expected pattern, skip parsing test metadata", filename)
        return {}

    timestamp = parts[-1]
    num_prompts_str = parts[-2]
    max_concurrency_str = parts[-3]
    dataset_name = parts[-4]
    test_name = "_".join(parts[:-4]) if parts[:-4] else ""

    parsed: dict[str, Any] = {}

    if len(timestamp) >= 15:
        parsed["date"] = timestamp

    if dataset_name in ("random", "random-mm"):
        parsed["dataset_name"] = dataset_name

    try:
        parsed["num_prompts"] = int(num_prompts_str)
    except (TypeError, ValueError):
        pass

    try:
        parsed["max_concurrency"] = int(max_concurrency_str)
    except (TypeError, ValueError):
        pass

    if test_name:
        parsed["test_name"] = test_name

    return parsed


def _iter_json_records(input_dir: str) -> Iterable[dict[str, Any]]:
    if not os.path.isdir(input_dir):
        LOGGER.warning("input dir '%s' does not exist or is not a directory", input_dir)
        return

    for entry in sorted(os.listdir(input_dir)):
        if not entry.endswith(".json"):
            continue
        if not entry.startswith(_RESULT_JSON_PREFIX):
            LOGGER.warning("skip non-result json file '%s'", entry)
            continue
        full_path = os.path.join(input_dir, entry)
        if not os.path.isfile(full_path):
            continue

        data = _load_json_file(full_path)
        if data is None:
            continue

        record: dict[str, Any] = dict(data)
        filename_meta = _parse_from_filename(os.path.basename(full_path))

        if "date" not in record or not record["date"]:
            if "date" in filename_meta:
                record["date"] = filename_meta["date"]
            else:
                record["date"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        if "num_prompts" not in record or record["num_prompts"] is None:
            if "num_prompts" in filename_meta:
                record["num_prompts"] = filename_meta["num_prompts"]

        if "max_concurrency" not in record or record["max_concurrency"] is None:
            if "max_concurrency" in filename_meta:
                record["max_concurrency"] = filename_meta["max_concurrency"]

        if "test_name" not in record or not record.get("test_name"):
            if "test_name" in filename_meta:
                record["test_name"] = filename_meta["test_name"]

        if "dataset_name" not in record or not record.get("dataset_name"):
            if "dataset_name" in filename_meta:
                record["dataset_name"] = filename_meta["dataset_name"]

        record["source_file"] = os.path.basename(full_path)
        yield record


def _collect_records(input_dir: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _iter_json_records(input_dir):
        records.append(record)
    return records


def _sort_records_for_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_date_desc = sorted(records, key=lambda r: (r.get("date") or ""), reverse=True)
    return sorted(
        by_date_desc,
        key=lambda r: (
            r.get("model_id") or "",
            r.get("test_name") or "",
            r.get("dataset_name") or "",
            r.get("max_concurrency") or 0,
            r.get("num_prompts") or 0,
        ),
    )


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if not parent:
        return
    os.makedirs(parent, exist_ok=True)


def _html_escape(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
    )


def _build_html_table(columns: Sequence[str], records: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("<table>")
    lines.append("  <thead>")
    header_cells = "".join(f"<th>{_html_escape(col)}</th>" for col in columns)
    lines.append(f"    <tr>{header_cells}</tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")
    for record in records:
        row_cells = []
        for col in columns:
            row_cells.append(f"<td>{_html_escape(record.get(col))}</td>")
        lines.append(f"    <tr>{''.join(row_cells)}</tr>")
    lines.append("  </tbody>")
    lines.append("</table>")
    return "\n".join(lines)


def _build_html_document(columns: Sequence[str], records: Sequence[dict[str, Any]]) -> str:
    table_html = _build_html_table(columns, records)
    styles = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  padding: 16px;
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
}
thead th {
  position: sticky;
  top: 0;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ccc;
  padding: 4px 6px;
  text-align: left;
}
tbody tr:nth-child(odd) {
  background-color: #fafafa;
}
tbody td {
  border-bottom: 1px solid #eee;
  padding: 4px 6px;
}
tbody td.numeric {
  text-align: right;
}
"""
    html = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8" />',
        "  <title>Nightly Performance Report</title>",
        f"  <style>{styles}</style>",
        "</head>",
        "<body>",
        "  <h1>Nightly Performance Report</h1>",
        table_html,
        "</body>",
        "</html>",
    ]
    return "\n".join(html)


def generate_html_report(input_dir: str, output_file: str) -> None:
    records = _collect_records(input_dir)
    if not records:
        LOGGER.warning("no valid json records found under '%s'", input_dir)

    sorted_records = _sort_records_for_summary(records)

    columns: list[str] = [
        "date",
        "endpoint_type",
        "backend",
        "model_id",
        "tokenizer_id",
        "test_name",
        "dataset_name",
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
    ]

    html_content = _build_html_document(columns, sorted_records)
    _ensure_parent_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    LOGGER.info("html report saved to '%s'", output_file)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    generate_html_report(input_dir=args.input_dir, output_file=args.output_file)


if __name__ == "__main__":
    main()
