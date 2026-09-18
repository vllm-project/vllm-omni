"""End-to-end + structural tests for the nightly "All major regressions"
filter (Model + Hardware) and the new visible Hardware column.

These tests load the skill's report modules with ``importlib`` (the same
pattern used by ``tests/test_generate_nightly_perf_html.py``) so we can
exercise the real renderer against a synthetic kanban history JSON and
inspect the generated HTML/Markdown without hitting the network.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = (
    Path(os.environ.get("NIGHTLY_REPORT_SKILL_DIR", "")).resolve()
    if os.environ.get("NIGHTLY_REPORT_SKILL_DIR")
    else REPO_ROOT / ".claude/skills/vllm-omni-test-report/scripts"
)


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nightly_module() -> ModuleType:
    return _load("nightly_local_log_report", SKILL_SCRIPTS / "nightly_local_log_report.py")


@pytest.fixture(scope="module")
def kanban_module() -> ModuleType:
    return _load("kanban_assets_perf_summary", SKILL_SCRIPTS / "kanban_assets_perf_summary.py")


def _write_history(assets_dir: Path, name: str, records: list[dict]) -> Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "Test",
        "generated_at": "2026-07-31T08:03:35.510929",
        "group_fields": ["test_name", "hardware", "model_id", "max_concurrency", "num_prompts"],
        "records": records,
    }
    path = assets_dir / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _record(
    *,
    model_id: str,
    test_name: str,
    hardware: str | None,
    hardware_legacy: str | None = None,
    metric: str = "e2e_latency_ms",
    latest: float = 120.0,
    baseline: float = 100.0,
    date: str = "2026-07-30 18:00:00",
) -> dict:
    rec: dict = {
        "model_id": model_id,
        "test_name": test_name,
        "date": date,
        "max_concurrency": 1,
        "num_prompts": 4,
        f"baseline_{metric}": baseline,
        metric: latest,
        "config_key": f"{test_name} | {hardware or hardware_legacy or 'unknown'} | 1 | 4",
        "timestamp_key": date.replace(" ", "-").replace(":", ""),
        "sort_timestamp": date,
    }
    if hardware is not None:
        rec["hardware"] = hardware
    if hardware_legacy is not None:
        rec["Hardware"] = hardware_legacy
    return rec


def test_perf_row_includes_hardware_when_present(kanban_module: ModuleType, tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _write_history(
        assets,
        "qwen_image_history.json",
        [
            _record(model_id="model-a", test_name="test_a_h100", hardware="H100"),
            _record(model_id="model-a", test_name="test_a_h100_2", hardware="H100"),
            _record(model_id="model-b", test_name="test_b_h200", hardware="H200"),
            _record(
                model_id="model-c",
                test_name="test_c_910",
                hardware=None,
                hardware_legacy="910",
            ),
        ],
    )
    summary = kanban_module.build_assets_perf_summary(assets_dir=assets)
    assert summary["status"] == "ok"
    rows = summary["rows"]
    assert len(rows) == 4
    assert {row["hardware"] for row in rows} == {"H100", "H200", "910"}


def test_perf_row_normalizes_missing_hardware_to_empty_string(kanban_module: ModuleType, tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _write_history(
        assets,
        "qwen_image_history.json",
        [
            _record(model_id="model-a", test_name="test_a_none", hardware=None),
            _record(model_id="model-b", test_name="test_b_blank", hardware=""),
        ],
    )
    summary = kanban_module.build_assets_perf_summary(assets_dir=assets)
    rows = summary["rows"]
    assert {row["hardware"] for row in rows} == {""}


def test_focus_table_emits_hardware_filter_and_data_attribute(nightly_module: ModuleType) -> None:
    items = [
        nightly_module.NightlyFocusItem(
            source="Buildkite",
            model="model-a",
            model_type="qwen_image",
            hardware="H100",
            config="c=1 | n=4",
            test="test_a",
            metric="e2e_latency_ms",
            latest=120.0,
            baseline=100.0,
            vs_baseline_pct=-20.0,
            status="fail",
            consec_fail_days=1,
        ),
        nightly_module.NightlyFocusItem(
            source="Local",
            model="model-b",
            model_type="qwen_image",
            hardware="H200",
            config="c=1 | n=4",
            test="test_b",
            metric="e2e_latency_ms",
            latest=120.0,
            baseline=100.0,
            vs_baseline_pct=-20.0,
            status="fail",
            consec_fail_days=0,
        ),
        nightly_module.NightlyFocusItem(
            source="Local",
            model="model-c",
            model_type="other",
            hardware="910",
            config="c=1 | n=4",
            test="test_c",
            metric="e2e_latency_ms",
            latest=120.0,
            baseline=100.0,
            vs_baseline_pct=-20.0,
            status="fail",
            consec_fail_days=0,
        ),
        nightly_module.NightlyFocusItem(
            source="Buildkite",
            model="model-d",
            model_type="qwen_image",
            hardware="",
            config="c=1 | n=4",
            test="test_d",
            metric="e2e_latency_ms",
            latest=120.0,
            baseline=100.0,
            vs_baseline_pct=-20.0,
            status="fail",
            consec_fail_days=0,
        ),
    ]
    html = nightly_module._render_focus_perf_table_html(items)

    # Fieldset wiring.
    assert 'class="focus-model-filter"' in html
    assert 'class="focus-hardware-filter"' in html
    assert 'data-filter-key="hardware" value="H100"' in html
    assert 'data-filter-key="hardware" value="H200"' in html
    assert 'data-filter-key="hardware" value="910"' in html
    assert 'data-filter-key="hardware" value="unknown"' in html
    # Visible "Unknown" label for the missing-hardware bucket.
    assert "Unknown" in html

    # Header now exposes the new column.
    assert "<th>Hardware</th>" in html
    headers = nightly_module._FOCUS_TABLE_HEADERS
    assert headers[2] == "Hardware"
    assert len(headers) == 12

    # Every row carries a data-hardware attribute.
    assert 'data-hardware="H100"' in html
    assert 'data-hardware="H200"' in html
    assert 'data-hardware="910"' in html
    assert 'data-hardware="unknown"' in html

    # Empty state plumbing still present.
    assert "data-perf-empty" in html


def test_focus_table_markdown_includes_hardware_column(nightly_module: ModuleType) -> None:
    items = [
        nightly_module.NightlyFocusItem(
            source="Buildkite",
            model="model-a",
            model_type="qwen_image",
            hardware="H100",
            config="c=1",
            test="test_a",
            metric="e2e_latency_ms",
            latest=120.0,
            baseline=100.0,
            vs_baseline_pct=-20.0,
            status="fail",
            consec_fail_days=0,
        ),
        nightly_module.NightlyFocusItem(
            source="Local",
            model="model-b",
            model_type="other",
            hardware="",
            config="c=1",
            test="test_b",
            metric="e2e_latency_ms",
            latest=120.0,
            baseline=100.0,
            vs_baseline_pct=-20.0,
            status="fail",
            consec_fail_days=0,
        ),
    ]
    rows = nightly_module._focus_perf_table_rows(items)
    assert [row[2] for row in rows] == ["H100", "unknown"]
    headers = nightly_module._FOCUS_TABLE_HEADERS
    assert headers[0] == "Source"
    assert headers[1] == "Model"
    assert headers[2] == "Hardware"
    assert headers[3] == "Type"


def test_apply_perf_filters_js_contract_preserved(nightly_module: ModuleType) -> None:
    """The focus table still wires up the inline applyPerfFilters JS."""
    html = nightly_module._render_focus_perf_table_html(
        [
            nightly_module.NightlyFocusItem(
                source="Buildkite",
                model="m",
                model_type="other",
                hardware="H100",
                config="c=1",
                test="t",
                metric="e2e_latency_ms",
                latest=1.0,
                baseline=1.0,
                vs_baseline_pct=None,
                status="n/a",
                consec_fail_days=0,
            )
        ]
    )
    # The model and hardware checkboxes are rendered with the
    # data-filter-key attribute that the JS reads.
    assert 'data-filter-key="model"' in html
    assert 'data-filter-key="hardware"' in html
    # The body row carries the data-perf-row attribute the JS scans.
    assert 'data-perf-row="1"' in html
    # The empty-state paragraph is still present so the JS can flip it.
    assert "data-perf-empty" in html


def test_focus_item_sort_key_includes_hardware(nightly_module: ModuleType) -> None:
    """Hardware becomes a deterministic tiebreaker in the sort key."""
    h100 = nightly_module.NightlyFocusItem(
        source="Local",
        model="m",
        model_type="other",
        hardware="H100",
        config="c=1",
        test="t",
        metric="e2e_latency_ms",
        latest=1.0,
        baseline=1.0,
        vs_baseline_pct=-20.0,
        status="fail",
        consec_fail_days=0,
    )
    h200 = nightly_module.NightlyFocusItem(
        source="Local",
        model="m",
        model_type="other",
        hardware="H200",
        config="c=1",
        test="t",
        metric="e2e_latency_ms",
        latest=1.0,
        baseline=1.0,
        vs_baseline_pct=-20.0,
        status="fail",
        consec_fail_days=0,
    )
    key_h100 = nightly_module._focus_item_sort_key(h100)
    key_h200 = nightly_module._focus_item_sort_key(h200)
    # Hardware is the trailing element of the sort key tuple.
    assert key_h100[-1] == "H100"
    assert key_h200[-1] == "H200"
    # "H100" < "H200" in standard string ordering.
    assert key_h100 < key_h200
    # Missing hardware normalizes to "unknown" and sorts after the known
    # buckets.
    unknown_item = nightly_module.NightlyFocusItem(
        source="Local",
        model="m",
        model_type="other",
        hardware="",
        config="c=1",
        test="t",
        metric="e2e_latency_ms",
        latest=1.0,
        baseline=1.0,
        vs_baseline_pct=-20.0,
        status="fail",
        consec_fail_days=0,
    )
    key_unknown = nightly_module._focus_item_sort_key(unknown_item)
    assert key_unknown[-1] == "unknown"
    assert key_unknown > key_h200


def test_norm_focus_hardware(nightly_module: ModuleType) -> None:
    assert nightly_module._norm_focus_hardware("H100") == "H100"
    assert nightly_module._norm_focus_hardware("  H200  ") == "H200"
    assert nightly_module._norm_focus_hardware("") == "unknown"
    assert nightly_module._norm_focus_hardware(None) == "unknown"
    assert nightly_module._norm_focus_hardware("910") == "910"
