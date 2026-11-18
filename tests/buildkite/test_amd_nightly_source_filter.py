# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_FILTER_PATH = Path(".buildkite/amd/scripts/filter_test_suite.py")
AMD_COMBINER_PATH = Path(".buildkite/amd/scripts/combine_test_suites.py")
AMD_NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
TINY_DIFFUSION_LABELS = {
    "ROCm · Tiny Diffusion Multi-GPU · 2-GPU",
    "ROCm · Tiny Diffusion Multi-GPU · 4-GPU",
}


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FILTER = _load_module("filter_test_suite", AMD_FILTER_PATH)
COMBINER = _load_module("combine_test_suites_for_filter", AMD_COMBINER_PATH)


def _load_pipeline(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _leaf_steps(doc: dict) -> list[dict]:
    leaves: list[dict] = []

    def walk(steps: list) -> None:
        for step in steps:
            nested = step.get("steps")
            if isinstance(nested, list):
                walk(nested)
            else:
                leaves.append(step)

    walk(doc.get("steps", []))
    return leaves


def _labels(doc: dict) -> set[str]:
    return {step["label"] for step in _leaf_steps(doc)}


def _small_nightly_pipeline() -> dict:
    return {
        "env": {"SHARED": "value"},
        "steps": [
            {
                "group": "Nightly group",
                "steps": [
                    {
                        "label": "job-a",
                        "grade": "NonBlocking",
                        "source_file_dependencies": ["src/a.py"],
                        "commands": ["pytest tests/a.py"],
                    },
                    {
                        "label": "job-b",
                        "grade": "NonBlocking",
                        "source_file_dependencies": ["src/b.py"],
                        "commands": ["pytest tests/b.py"],
                    },
                ],
            }
        ],
    }


def _filter_pipeline_edit(current: dict, base: dict | None, *other_files: str) -> dict:
    return FILTER.filter_amd_nightly_suite(
        current,
        [AMD_NIGHTLY_PIPELINE.as_posix(), *other_files],
        pipeline_path=AMD_NIGHTLY_PIPELINE,
        current_pipeline_doc=current,
        base_pipeline_doc=base,
    )


def test_every_amd_nightly_job_declares_source_dependencies() -> None:
    leaves = _leaf_steps(_load_pipeline(AMD_NIGHTLY_PIPELINE))
    assert leaves, "AMD nightly pipeline must contain jobs"
    assert all(step.get("source_file_dependencies") for step in leaves)


def test_current_pipeline_compares_cleanly_against_itself() -> None:
    current = _load_pipeline(AMD_NIGHTLY_PIPELINE)

    filtered = _filter_pipeline_edit(current, copy.deepcopy(current))

    assert filtered["steps"] == []


def test_diffusion_only_pr_keeps_only_matching_amd_nightly_jobs() -> None:
    filtered = FILTER.filter_amd_nightly_suite(
        _load_pipeline(AMD_NIGHTLY_PIPELINE),
        ["tests/model_tests/diffusion/case_filtering.py"],
        pipeline_path=AMD_NIGHTLY_PIPELINE,
    )

    assert _labels(filtered) == TINY_DIFFUSION_LABELS
    assert all("source_file_dependencies" not in step for step in _leaf_steps(filtered))


def test_unrelated_pr_change_omits_all_amd_nightly_jobs() -> None:
    filtered = FILTER.filter_amd_nightly_suite(
        _load_pipeline(AMD_NIGHTLY_PIPELINE),
        ["README.md"],
        pipeline_path=AMD_NIGHTLY_PIPELINE,
    )
    assert filtered["steps"] == []


@pytest.mark.parametrize(
    "changed_file",
    (
        "vllm_omni/model_executor/stage_input_processors/voxtral_tts.py",
        "vllm_omni/entrypoints/openai/tts_adapters/base.py",
        "vllm_omni/engine/stage_runtime.py",
    ),
)
def test_voxtral_runtime_changes_keep_voxtral_perf_nightly(changed_file: str) -> None:
    filtered = FILTER.filter_amd_nightly_suite(
        _load_pipeline(AMD_NIGHTLY_PIPELINE),
        [changed_file],
        pipeline_path=AMD_NIGHTLY_PIPELINE,
    )

    assert "Voxtral TTS Perf" in _labels(filtered)


@pytest.mark.parametrize(
    "changed_file",
    (
        "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py",
        "vllm_omni/model_executor/stage_input_processors/qwen3_tts.py",
        "vllm_omni/deploy/qwen3_tts.yaml",
    ),
)
def test_qwen3_tts_runtime_changes_keep_qwen3_tts_perf_nightly(changed_file: str) -> None:
    filtered = FILTER.filter_amd_nightly_suite(
        _load_pipeline(AMD_NIGHTLY_PIPELINE),
        [changed_file],
        pipeline_path=AMD_NIGHTLY_PIPELINE,
    )

    assert "Qwen3-TTS Perf" in _labels(filtered)


@pytest.mark.parametrize(
    "changed_file",
    (
        "vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py",
        "vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py",
        "vllm_omni/deploy/minicpmo_4_5.yaml",
    ),
)
def test_minicpmo_runtime_changes_keep_minicpmo_perf_nightly(changed_file: str) -> None:
    filtered = FILTER.filter_amd_nightly_suite(
        _load_pipeline(AMD_NIGHTLY_PIPELINE),
        [changed_file],
        pipeline_path=AMD_NIGHTLY_PIPELINE,
    )

    assert "MiniCPM-o 4.5 Perf" in _labels(filtered)


def test_missing_base_and_shared_filter_changes_keep_full_amd_nightly_suite() -> None:
    expected_labels = _labels(_load_pipeline(AMD_NIGHTLY_PIPELINE))
    for changed_file in (
        AMD_NIGHTLY_PIPELINE.as_posix(),
        AMD_FILTER_PATH.as_posix(),
        ".buildkite/common/ci_source_file_dependencies.yml",
    ):
        filtered = FILTER.filter_amd_nightly_suite(
            _load_pipeline(AMD_NIGHTLY_PIPELINE),
            [changed_file],
            pipeline_path=AMD_NIGHTLY_PIPELINE,
        )
        assert _labels(filtered) == expected_labels


def test_pipeline_leaf_edit_selects_only_changed_job() -> None:
    base = _small_nightly_pipeline()
    current = copy.deepcopy(base)
    current["steps"][0]["steps"][0]["grade"] = "Blocking"

    filtered = _filter_pipeline_edit(current, base)

    assert _labels(filtered) == {"job-a"}
    assert all("source_file_dependencies" not in step for step in _leaf_steps(filtered))


def test_pipeline_leaf_edits_union_with_source_selected_jobs() -> None:
    base = _small_nightly_pipeline()
    current = copy.deepcopy(base)
    current["steps"][0]["steps"][0]["timeout_in_minutes"] = 30

    filtered = _filter_pipeline_edit(current, base, "src/b.py")

    assert _labels(filtered) == {"job-a", "job-b"}


def test_pipeline_multiple_leaf_edits_select_only_changed_jobs() -> None:
    base = _small_nightly_pipeline()
    current = copy.deepcopy(base)
    current["steps"][0]["steps"][0]["grade"] = "Blocking"
    current["steps"][0]["steps"][1]["timeout_in_minutes"] = 30
    current["steps"][0]["steps"].append(
        {
            "label": "job-c",
            "source_file_dependencies": ["src/c.py"],
            "commands": ["pytest tests/c.py"],
        }
    )

    filtered = _filter_pipeline_edit(current, base)

    assert _labels(filtered) == {"job-a", "job-b", "job-c"}


def test_pipeline_rename_selects_new_label_and_deleted_only_selects_no_job() -> None:
    base = _small_nightly_pipeline()
    renamed = copy.deepcopy(base)
    renamed["steps"][0]["steps"][0]["label"] = "job-a-renamed"
    assert _labels(_filter_pipeline_edit(renamed, base)) == {"job-a-renamed"}

    deleted = copy.deepcopy(base)
    deleted["steps"][0]["steps"].pop()
    assert _filter_pipeline_edit(deleted, base)["steps"] == []


def test_unchanged_parsed_pipeline_selects_no_nightly_jobs() -> None:
    current = _small_nightly_pipeline()

    filtered = _filter_pipeline_edit(current, copy.deepcopy(current))

    assert filtered["steps"] == []


@pytest.mark.parametrize(
    "base",
    [
        None,
        {"steps": "not-a-list"},
        {
            "steps": [
                {"label": "duplicate", "commands": ["true"]},
                {"label": "duplicate", "commands": ["true"]},
            ]
        },
    ],
    ids=["missing", "malformed", "duplicate-label"],
)
def test_unsafe_base_pipeline_comparison_keeps_full_suite(base: dict | None) -> None:
    current = _small_nightly_pipeline()

    filtered = _filter_pipeline_edit(current, base)

    assert _labels(filtered) == {"job-a", "job-b"}


@pytest.mark.parametrize("change", ["shared", "group"])
def test_shared_or_group_pipeline_change_keeps_full_suite(change: str) -> None:
    base = _small_nightly_pipeline()
    current = copy.deepcopy(base)
    if change == "shared":
        current["env"]["SHARED"] = "updated"
    else:
        current["steps"][0]["group"] = "Updated nightly group"

    filtered = _filter_pipeline_edit(current, base)

    assert _labels(filtered) == {"job-a", "job-b"}


def test_fallback_plus_matching_model_change_still_filters() -> None:
    filtered = FILTER.filter_amd_nightly_suite(
        _load_pipeline(AMD_NIGHTLY_PIPELINE),
        [AMD_FILTER_PATH.as_posix(), "tests/model_tests/diffusion/case_filtering.py"],
        pipeline_path=AMD_NIGHTLY_PIPELINE,
    )
    assert _labels(filtered) == TINY_DIFFUSION_LABELS


def test_combined_suites_filter_only_nightly_and_preserve_ready_steps() -> None:
    combined = COMBINER.combine_test_suites(
        (
            f"READY_TESTS:{AMD_READY_PIPELINE}",
            f"NIGHTLY_TESTS:{AMD_NIGHTLY_PIPELINE}",
        )
    )
    ready_before = copy.deepcopy(combined["steps"][0])

    filtered = FILTER.filter_amd_nightly_suite(
        combined,
        ["tests/model_tests/diffusion/case_filtering.py"],
        pipeline_path=AMD_NIGHTLY_PIPELINE,
        group_name="NIGHTLY_TESTS",
    )

    assert filtered["steps"][0] == ready_before
    nightly = next(group for group in filtered["steps"] if group["group"] == "NIGHTLY_TESTS")
    assert _labels({"steps": nightly["steps"]}) == TINY_DIFFUSION_LABELS


def test_combined_pipeline_edit_preserves_ready_and_targets_nightly() -> None:
    base = _small_nightly_pipeline()
    current = copy.deepcopy(base)
    current["steps"][0]["steps"][1]["grade"] = "Blocking"
    ready = {
        "group": "READY_TESTS",
        "steps": [{"label": "ready-job", "commands": ["pytest tests/ready.py"]}],
    }
    combined = {
        "steps": [
            copy.deepcopy(ready),
            {"group": "NIGHTLY_TESTS", "steps": copy.deepcopy(current["steps"])},
        ]
    }

    filtered = FILTER.filter_amd_nightly_suite(
        combined,
        [AMD_NIGHTLY_PIPELINE.as_posix()],
        pipeline_path=AMD_NIGHTLY_PIPELINE,
        group_name="NIGHTLY_TESTS",
        current_pipeline_doc=current,
        base_pipeline_doc=base,
    )

    assert filtered["steps"][0] == ready
    nightly = filtered["steps"][1]
    assert nightly["group"] == "NIGHTLY_TESTS"
    assert _labels({"steps": nightly["steps"]}) == {"job-b"}


@pytest.mark.parametrize(
    ("pull_request", "debug_test_yaml", "scheduled_nightly", "expected"),
    [
        ("123", "", False, True),
        ("false", "", True, False),
        ("123", "nightly", False, False),
        ("false", "", False, False),
    ],
)
def test_only_label_selected_pr_nightlies_are_filtered(
    pull_request: str,
    debug_test_yaml: str,
    scheduled_nightly: bool,
    expected: bool,
) -> None:
    assert (
        FILTER.should_filter_pr_nightly(
            pull_request=pull_request,
            debug_test_yaml=debug_test_yaml,
            scheduled_nightly=scheduled_nightly,
        )
        is expected
    )


def test_unknown_dependency_key_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown source_file_dependencies"):
        FILTER.filter_amd_nightly_suite(
            {
                "steps": [
                    {
                        "label": "Unknown dependency",
                        "source_file_dependencies": "not_a_real_key",
                        "commands": ["true"],
                    }
                ]
            },
            ["README.md"],
            pipeline_path=AMD_NIGHTLY_PIPELINE,
        )
