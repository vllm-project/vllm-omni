#!/usr/bin/env python3
"""Summarize latest-day baseline comparisons from kanban chart history JSON files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from kanban_repo_config import KANBAN_REPO_URL  # noqa: E402
from laptop_path_defaults import (  # noqa: E402
    DEFAULT_KANBAN_REPO_ROOT_DISPLAY,
    resolve_kanban_repo_root,
)

DEFAULT_ASSETS_DIR = (
    Path(os.environ.get("KANBAN_ASSETS_DIR", "").strip()).resolve()
    if os.environ.get("KANBAN_ASSETS_DIR", "").strip()
    else None
)
DEFAULT_REPO_ROOT = resolve_kanban_repo_root()

LOWER_BETTER_HINTS = (
    "latency",
    "ttfp",
    "ttft",
    "tpot",
    "rtl",
    "rtf",
    "memory",
    "e2e",
    "duration",
)
HIGHER_BETTER_HINTS = ("throughput", "qps", "tps")
NORMAL_DEGRADATION_THRESHOLD_PCT = 6.0


def _empty_perf_status_counts() -> dict[str, int]:
    return {"pass": 0, "normal": 0, "fail": 0, "n/a": 0}


@dataclass
class PerfRow:
    """One metric comparison row."""

    model: str
    model_type: str
    config_key: str
    config_view: str
    test_name: str
    metric: str
    latest: float
    baseline: float
    vs_baseline_pct: float | None
    status: str
    direction: str
    date: str


@dataclass
class KanbanSourceMeta:
    assets_dir: str
    repo_root: str
    current_branch: str
    upstream_remote: str
    upstream_branch: str
    warnings: list[str]


@dataclass
class HistoryPayload:
    records: list[dict[str, Any]]
    record_groups: list[list[dict[str, Any]]]
    meta: dict[str, Any]


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        try:
            return float(txt)
        except ValueError:
            return None
    return None


def _date_key(rec: dict[str, Any]) -> str | None:
    raw = str(rec.get("date") or "").strip()
    if len(raw) >= 10:
        return raw[:10]
    return None


def _timestamp_key(rec: dict[str, Any]) -> str:
    for key in ("sort_timestamp", "date", "timestamp_key"):
        value = rec.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _metric_direction(metric: str) -> str:
    m = metric.lower()
    if any(h in m for h in HIGHER_BETTER_HINTS):
        return "higher_better"
    if any(h in m for h in LOWER_BETTER_HINTS):
        return "lower_better"
    return "unknown"


def _baseline_compare_status(vs_baseline_pct: float | None, direction: str) -> str:
    if direction == "unknown" or vs_baseline_pct is None:
        return "n/a"
    if vs_baseline_pct >= 0:
        return "pass"
    if vs_baseline_pct >= -NORMAL_DEGRADATION_THRESHOLD_PCT:
        return "normal"
    return "fail"


def _history_group_key(
    group: dict[str, Any],
    group_fields: list[str],
    fallback: str,
) -> str:
    key_obj = group.get("key")
    if isinstance(key_obj, dict):
        fields = group_fields or sorted(str(k) for k in key_obj)
        values = ["" if key_obj.get(field) is None else str(key_obj.get(field)) for field in fields]
        return " | ".join(values)
    config_key = str(group.get("config_key") or "").strip()
    return config_key or fallback


def _parse_history_payload(path: Path) -> HistoryPayload:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recs = payload.get("records")
    records = [r for r in recs if isinstance(r, dict)] if isinstance(recs, list) else []

    group_fields_raw = payload.get("group_fields")
    group_fields = [str(v) for v in group_fields_raw] if isinstance(group_fields_raw, list) else []
    record_groups: list[list[dict[str, Any]]] = []
    groups_raw = payload.get("groups")
    if isinstance(groups_raw, list):
        for idx, group in enumerate(groups_raw):
            if not isinstance(group, dict):
                continue
            group_recs_raw = group.get("records")
            if not isinstance(group_recs_raw, list):
                continue
            group_key = _history_group_key(group, group_fields, f"{path.name}#{idx}")
            group_records: list[dict[str, Any]] = []
            for rec in group_recs_raw:
                if not isinstance(rec, dict):
                    continue
                normalized = dict(rec)
                normalized["_kanban_group_key"] = f"{path.name}::{group_key}"
                if group_key:
                    normalized["config_key"] = group_key
                group_records.append(normalized)
            if group_records:
                record_groups.append(group_records)

    return HistoryPayload(
        records=records,
        record_groups=record_groups,
        meta={
            "file": path.name,
            "generated_at": str(payload.get("generated_at") or ""),
            "record_count": len(records),
            "group_count": len(record_groups),
            "group_fields": group_fields,
        },
    )


def _latest_day(records: list[dict[str, Any]]) -> str | None:
    days = sorted({d for d in (_date_key(r) for r in records) if d})
    if not days:
        return None
    return days[-1]


def _pick_latest_records_in_day(records: list[dict[str, Any]], latest_day: str) -> list[dict[str, Any]]:
    in_day = [r for r in records if _date_key(r) == latest_day]
    grouped: dict[str, dict[str, Any]] = {}
    for rec in in_day:
        group_key = str(rec.get("_kanban_group_key") or rec.get("config_key") or rec.get("source_file") or id(rec))
        prev = grouped.get(group_key)
        if prev is None or _timestamp_key(rec) > _timestamp_key(prev):
            grouped[group_key] = rec
    return list(grouped.values())


def _pick_latest_records_from_groups(
    record_groups: list[list[dict[str, Any]]],
    latest_day: str,
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for group_records in record_groups:
        in_day = [r for r in group_records if _date_key(r) == latest_day]
        if not in_day:
            continue
        picked.append(max(in_day, key=_timestamp_key))
    return picked


def _pick_latest_records_per_test(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the latest record per (model_id, test_name) bucket, further
    deduped by config_key so concurrent runs of the same config don't
    double-count.

    The day-bounded helpers (``_pick_latest_records_in_day`` /
    ``_pick_latest_records_from_groups``) assume *one* ``*_history.json``
    = *one* model whose tests are all re-run on the same day. When a
    single history file bundles multiple test cases under one model (e.g.
    ``hunyuan_image3_history.json`` containing ``test_hunyuan_image3_*``,
    ``test_hunyuan_image_tp2_*``, ``test_hunyuan_image_tp4``) and they
    have not all been re-run on the same day, the day-bounded helper
    drops every test whose latest record is older than the file's overall
    freshest day. This helper generalizes "one file = one latest day" to
    "one test = one latest record" so each test surfaces its freshest
    baseline comparison independently.
    """
    latest_by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        test_key = (str(r.get("model_id") or ""), str(r.get("test_name") or ""))
        gk = str(r.get("config_key") or r.get("source_file") or id(r))
        inner = latest_by_key.setdefault(test_key, {})
        prev = inner.get(gk)
        if prev is None or _timestamp_key(r) > _timestamp_key(prev):
            inner[gk] = r
    out: list[dict[str, Any]] = []
    for inner in latest_by_key.values():
        out.extend(inner.values())
    return out


def _history_summary(metas: list[dict[str, Any]], used_group_payloads: bool) -> dict[str, Any]:
    generated = sorted({str(m.get("generated_at") or "") for m in metas if m.get("generated_at")})
    files = [str(m.get("file") or "") for m in metas if m.get("file")]
    return {
        "files": files,
        "generated_at": generated[-1] if generated else "",
        "record_count": sum(int(m.get("record_count") or 0) for m in metas),
        "group_count": sum(int(m.get("group_count") or 0) for m in metas),
        "selection": "groups" if used_group_payloads else "records",
    }


def _iter_metric_pairs(rec: dict[str, Any]) -> list[tuple[str, float, float]]:
    pairs: list[tuple[str, float, float]] = []

    for key, value in rec.items():
        if not key.startswith("baseline_"):
            continue
        metric = key[len("baseline_") :]
        baseline = _as_float(value)
        latest = _as_float(rec.get(metric))
        if baseline is None or latest is None:
            continue
        pairs.append((metric, latest, baseline))

    baseline_obj = rec.get("baseline")
    if isinstance(baseline_obj, dict):
        for metric, base_v in baseline_obj.items():
            baseline = _as_float(base_v)
            latest = _as_float(rec.get(metric))
            if baseline is None or latest is None:
                continue
            if any(metric == existing[0] for existing in pairs):
                continue
            pairs.append((str(metric), latest, baseline))
    return pairs


def _model_type(model: str, test_name: str) -> str:
    model_l = model.lower()
    test_l = test_name.lower()
    if "qwen3-omni" in model_l or "qwen3_omni" in test_l:
        return "qwen3_omni"
    if "qwen3-tts" in model_l or "qwen3_tts" in test_l:
        return "qwen3_tts"
    if "wan2.2" in model_l or "wan22" in model_l or "wan22" in test_l:
        return "wan22"
    if "qwen-image-edit-2509" in model_l or "qwen_image_edit_2509" in test_l:
        return "qwen_image_edit_2509"
    if "qwen-image-edit" in model_l or "qwen_image_edit" in test_l:
        return "qwen_image_edit"
    if "qwen-image-layered" in model_l or "qwen_image_layered" in test_l:
        return "qwen_image_layered"
    if "qwen-image" in model_l or "qwen_image" in test_l:
        return "qwen_image"
    return "other"


def _config_view(rec: dict[str, Any], model_type: str) -> str:
    def _v(key: str) -> str:
        if key == "qps":
            return _request_rate_view(rec)
        value = rec.get(key)
        if value is None:
            return ""
        return str(value).strip()

    def _request_rate_view(item: dict[str, Any]) -> str:
        for key in ("qps", "request_rate", "request-rate"):
            value = item.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if not text or text.lower() in ("inf", "infinity"):
                continue
            return text
        return ""

    fields_by_type: dict[str, list[tuple[str, str]]] = {
        "qwen3_omni": [
            ("data", "dataset_name"),
            ("in", "random_input_len"),
            ("out", "random_output_len"),
            ("c", "max_concurrency"),
            ("n", "num_prompts"),
            ("qbs", "qps"),
            ("profile", "omni_metrics_profile"),
        ],
        "qwen3_tts": [
            ("data", "dataset_name"),
            ("c", "max_concurrency"),
            ("n", "num_prompts"),
            ("qbs", "qps"),
        ],
        "wan22": [
            ("bench", "benchmark_name"),
            ("data", "dataset_name"),
            ("c", "max_concurrency"),
            ("n", "num_prompts"),
            ("qbs", "qps"),
        ],
        "qwen_image": [
            ("bench", "benchmark_name"),
            ("data", "dataset_name"),
            ("c", "max_concurrency"),
            ("n", "num_prompts"),
            ("qbs", "qps"),
        ],
        "qwen_image_layered": [
            ("bench", "benchmark_name"),
            ("data", "dataset_name"),
            ("c", "max_concurrency"),
            ("n", "num_prompts"),
            ("qbs", "qps"),
        ],
        "qwen_image_edit": [
            ("bench", "benchmark_name"),
            ("data", "dataset_name"),
            ("c", "max_concurrency"),
            ("n", "num_prompts"),
            ("qbs", "qps"),
        ],
        "qwen_image_edit_2509": [
            ("bench", "benchmark_name"),
            ("data", "dataset_name"),
            ("c", "max_concurrency"),
            ("n", "num_prompts"),
            ("qbs", "qps"),
        ],
        "other": [
            ("c", "max_concurrency"),
            ("n", "num_prompts"),
            ("qbs", "qps"),
        ],
    }
    pairs: list[str] = []
    for label, key in fields_by_type.get(model_type, fields_by_type["other"]):
        value = _v(key)
        if value:
            pairs.append(f"{label}={value}")
    return ", ".join(pairs) if pairs else "-"


def _build_perf_rows(records: list[dict[str, Any]]) -> tuple[list[PerfRow], int]:
    """Build perf rows; skip records lacking both ``model_id`` and ``title``.

    Returns ``(rows, skipped_unidentified_count)``. The skipped count is
    surfaced as a warning in the rendered report so stale Buildkite perf
    JSONs (all-null fields) no longer pollute the table with a bogus
    ``unknown`` model group.
    """
    rows: list[PerfRow] = []
    skipped_unidentified = 0
    for rec in records:
        model_id = rec.get("model_id")
        title = rec.get("title")
        # Skip stale records that lack both model_id and title (e.g. an empty
        # Buildkite perf JSON whose fields are all null/0). These would otherwise
        # surface in the Buildkite Performance baseline comparison section as a
        # bogus "unknown" model group.
        if not model_id and not title:
            skipped_unidentified += 1
            continue
        model = str(model_id or title)
        config_key = str(rec.get("config_key") or rec.get("source_file") or "")
        test_name = str(rec.get("test_name") or "")
        m_type = _model_type(model, test_name)
        cfg_view = _config_view(rec, m_type)
        date_value = str(rec.get("date") or "")
        for metric, latest, baseline in _iter_metric_pairs(rec):
            direction = _metric_direction(metric)
            if abs(baseline) < 1e-12:
                vs_pct = None
                status = "n/a"
            else:
                raw_pct = (latest - baseline) / baseline * 100.0
                if direction == "lower_better":
                    vs_pct = -raw_pct
                else:
                    vs_pct = raw_pct
                status = _baseline_compare_status(vs_pct, direction)
            rows.append(
                PerfRow(
                    model=model,
                    model_type=m_type,
                    config_key=config_key,
                    config_view=cfg_view,
                    test_name=test_name,
                    metric=metric,
                    latest=latest,
                    baseline=baseline,
                    vs_baseline_pct=vs_pct,
                    status=status,
                    direction=direction,
                    date=date_value,
                )
            )
    return rows, skipped_unidentified


def _run_cmd(argv: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _git_upstream_info(repo_root: Path) -> tuple[str, str]:
    code, out, _ = _run_cmd(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        cwd=repo_root,
    )
    if code != 0 or "/" not in out:
        return "", ""
    remote, _, branch = out.partition("/")
    return remote.strip(), branch.strip()


def _check_kanban_source(
    repo_root: Path | None,
    expected_remote: str | None,
    expected_branch: str | None,
) -> KanbanSourceMeta:
    warnings: list[str] = []
    if repo_root is None:
        return KanbanSourceMeta("", "", "", "", "", warnings)
    if not repo_root.exists():
        warnings.append(f"kanban repo root not found: {repo_root}")
        return KanbanSourceMeta("", str(repo_root), "", "", "", warnings)

    code, branch, err = _run_cmd(
        ["git", "branch", "--show-current"],
        cwd=repo_root,
    )
    current_branch = branch if code == 0 else ""
    if code != 0:
        warnings.append(f"failed to read kanban current branch: {err}")

    upstream_remote, upstream_branch = _git_upstream_info(repo_root)
    if expected_branch and current_branch and current_branch != expected_branch:
        warnings.append(f"kanban branch mismatch: current={current_branch}, expected={expected_branch}")
    if expected_remote and upstream_remote and upstream_remote != expected_remote:
        warnings.append(f"kanban upstream remote mismatch: current={upstream_remote}, expected={expected_remote}")
    if expected_branch and upstream_branch and upstream_branch != expected_branch:
        warnings.append(f"kanban upstream branch mismatch: current={upstream_branch}, expected={expected_branch}")
    return KanbanSourceMeta(
        assets_dir=str((repo_root / "docs/assets/charts").resolve()),
        repo_root=str(repo_root),
        current_branch=current_branch,
        upstream_remote=upstream_remote,
        upstream_branch=upstream_branch,
        warnings=warnings,
    )


def _resolve_assets_dir(
    assets_dir: Path | None,
    repo_root: Path | None,
) -> Path | None:
    if assets_dir is not None:
        return assets_dir.resolve()
    if repo_root is not None:
        return (repo_root / "docs/assets/charts").resolve()
    return None


def build_assets_perf_summary(
    assets_dir: Path | None,
    history_files: list[str] | None = None,
    *,
    kanban_repo_root: Path | None = None,
    expected_remote: str | None = None,
    expected_branch: str | None = None,
) -> dict[str, Any]:
    """Return baseline comparison summary using each history file's own latest day.

    Models on different schedules (e.g. nightly vs weekly) are kept independent:
    each ``*_history.json`` contributes rows from its own latest dated record,
    so a model whose freshest record is older than another model's freshest
    record still appears. ``latest_day`` on the returned summary is the absolute
    latest day across all files (kept for backward compat); ``latest_day_per_file``
    exposes the per-file date actually used.
    """
    src_meta = _check_kanban_source(
        kanban_repo_root,
        expected_remote=expected_remote,
        expected_branch=expected_branch,
    )
    warnings = list(src_meta.warnings)

    resolved_assets_dir = _resolve_assets_dir(assets_dir, kanban_repo_root)
    if resolved_assets_dir is None:
        return {
            "status": "missing",
            "assets_dir": "",
            "latest_day": "",
            "rows": [],
            "summary": _empty_perf_status_counts(),
            "history": {},
            "message": "assets dir is missing: provide --assets-dir or --kanban-repo-root",
            "source": src_meta.__dict__,
        }

    paths: list[Path]
    if history_files:
        paths = [resolved_assets_dir / name for name in history_files]
    else:
        paths = sorted(resolved_assets_dir.glob("*_history.json"))
    existing = [p for p in paths if p.exists() and p.is_file()]
    if not existing:
        return {
            "status": "missing",
            "assets_dir": str(resolved_assets_dir),
            "latest_day": "",
            "rows": [],
            "summary": _empty_perf_status_counts(),
            "history": {},
            "message": "No *_history.json files found.",
            "warnings": warnings,
            "source": src_meta.__dict__,
        }

    parsed_payloads: list[tuple[Path, HistoryPayload]] = []
    history_metas: list[dict[str, Any]] = []
    for path in existing:
        parsed = _parse_history_payload(path)
        parsed_payloads.append((path, parsed))
        history_metas.append(parsed.meta)
    used_group_payloads = any(payload.record_groups for _, payload in parsed_payloads)
    history = _history_summary(history_metas, used_group_payloads)

    # Pick each history file's own latest day so models on different schedules
    # (e.g. nightly vs weekly) all surface their latest baseline comparison.
    # Using a single global latest_day hides every model whose freshest record
    # is older than the single freshest record in any other file.
    file_latest_days: dict[str, str] = {}
    absolute_latest_day = ""
    all_records: list[dict[str, Any]] = []

    for path, payload in parsed_payloads:
        if used_group_payloads:
            if not payload.record_groups:
                continue
            file_records = [r for grp in payload.record_groups for r in grp]
        else:
            if not payload.records:
                continue
            file_records = payload.records
        file_day = _latest_day(file_records)
        if file_day:
            file_latest_days[path.name] = file_day
            if file_day > absolute_latest_day:
                absolute_latest_day = file_day
        all_records.extend(file_records)

    # Pick the latest record per (model_id, test_name) instead of per
    # *_history.json. A single history file may bundle multiple test cases
    # under one model; the day-bounded selection would drop any test whose
    # latest run is older than the file's single freshest day.
    day_records = _pick_latest_records_per_test(all_records)
    if not absolute_latest_day and day_records:
        absolute_latest_day = max(
            (d for d in (_date_key(r) for r in day_records) if d),
            default="",
        )

    if not absolute_latest_day:
        return {
            "status": "empty",
            "assets_dir": str(resolved_assets_dir),
            "latest_day": "",
            "rows": [],
            "summary": _empty_perf_status_counts(),
            "history": history,
            "message": "No dated records found in history files.",
            "warnings": warnings,
            "source": src_meta.__dict__,
            "latest_day_per_file": file_latest_days,
        }
    rows, skipped_unidentified = _build_perf_rows(day_records)
    if skipped_unidentified:
        warnings.append(
            f"Skipped {skipped_unidentified} stale perf record(s) lacking both "
            f"`model_id` and `title` (would have surfaced as a bogus 'unknown' "
            f"model group). Re-run `sync_buildkite_raw_model_results.py` + "
            f"`generate_charts.py` to refresh `docs/assets/charts/*_history.json`."
        )
    # Keep only models that have baseline-backed rows.
    rows = [row for row in rows if row.baseline is not None]
    rows.sort(key=lambda x: (x.model, x.config_key, x.metric))

    summary = _empty_perf_status_counts()
    for row in rows:
        summary[row.status] = summary.get(row.status, 0) + 1

    return {
        "status": "ok" if rows else "empty",
        "assets_dir": str(resolved_assets_dir),
        "latest_day": absolute_latest_day,
        "latest_day_per_file": file_latest_days,
        "rows": [
            {
                "model": r.model,
                "model_type": r.model_type,
                "config_key": r.config_key,
                "config_view": r.config_view,
                "test_name": r.test_name,
                "metric": r.metric,
                "latest": r.latest,
                "baseline": r.baseline,
                "vs_baseline_pct": r.vs_baseline_pct,
                "status": r.status,
                "direction": r.direction,
                "date": r.date,
            }
            for r in rows
        ],
        "summary": summary,
        "history": history,
        "message": "",
        "warnings": warnings,
        "source": src_meta.__dict__,
    }


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def _as_markdown(summary: dict[str, Any]) -> str:
    if summary.get("status") != "ok":
        return f"*{summary.get('message') or 'No performance rows available.'}*"

    def _md_cell(value: Any) -> str:
        return str(value or "").replace("|", "/").replace("\n", " ")

    rows = summary.get("rows", [])
    header = "| Model | Type | Config | Test | Metric | latest | baseline | vs baseline | Status |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    body = [
        "| "
        + " | ".join(
            [
                _md_cell(item.get("model") or ""),
                _md_cell(item.get("model_type") or ""),
                _md_cell(item.get("config_view") or ""),
                _md_cell(item.get("test_name") or ""),
                _md_cell(item.get("metric") or ""),
                _fmt_number(_as_float(item.get("latest"))),
                _fmt_number(_as_float(item.get("baseline"))),
                _fmt_pct(_as_float(item.get("vs_baseline_pct"))),
                _md_cell(item.get("status") or ""),
            ]
        )
        + " |"
        for item in rows
    ]
    src = summary.get("source") or {}
    lines = [
        f"- latest day: `{summary.get('latest_day')}`\n"
        f"- pass/normal/fail/n-a: `{summary['summary'].get('pass', 0)}` / "
        f"`{summary['summary'].get('normal', 0)}` / "
        f"`{summary['summary'].get('fail', 0)}` / `{summary['summary'].get('n/a', 0)}`\n",
        f"- assets dir: `{summary.get('assets_dir') or ''}`",
    ]
    if src.get("repo_root"):
        lines.append(f"- kanban repo: `{src.get('repo_root')}`")
    if src.get("current_branch"):
        lines.append(f"- current branch: `{src.get('current_branch')}`")
    if src.get("upstream_remote") or src.get("upstream_branch"):
        lines.append(f"- upstream: `{src.get('upstream_remote') or ''}/{src.get('upstream_branch') or ''}`")
    hist = summary.get("history") or {}
    if hist:
        lines.append(
            f"- history: `{len(hist.get('files') or [])}` files, "
            f"`{hist.get('group_count', 0)}` groups, selection `{hist.get('selection') or ''}`"
        )
        if hist.get("generated_at"):
            lines.append(f"- history generated: `{hist.get('generated_at')}`")
    for warning in summary.get("warnings") or []:
        lines.append(f"- warning: {warning}")
    lines.append("")
    title = "\n".join(lines) + "\n"
    return "\n".join([title, header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize latest-day baseline comparisons from kanban assets history."
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help="Path to docs/assets/charts.",
    )
    parser.add_argument(
        "--kanban-repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help=(
            f"Local clone root of {KANBAN_REPO_URL} (default: $KANBAN_REPO_ROOT or {DEFAULT_KANBAN_REPO_ROOT_DISPLAY})."
        ),
    )
    parser.add_argument(
        "--expected-remote",
        default=None,
        help="Expected upstream remote name for kanban repo, e.g. upstream.",
    )
    parser.add_argument(
        "--expected-branch",
        default=None,
        help="Expected branch name for kanban repo, e.g. main.",
    )
    parser.add_argument(
        "--history-file",
        action="append",
        default=[],
        help="Specific history JSON filename to include. Repeatable.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format.",
    )
    args = parser.parse_args()

    summary = build_assets_perf_summary(
        assets_dir=args.assets_dir.resolve() if args.assets_dir else None,
        history_files=args.history_file or None,
        kanban_repo_root=args.kanban_repo_root.resolve() if args.kanban_repo_root else None,
        expected_remote=(args.expected_remote or "").strip() or None,
        expected_branch=(args.expected_branch or "").strip() or None,
    )
    if args.format == "markdown":
        print(_as_markdown(summary))
        return
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
