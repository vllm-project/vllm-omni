# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".buildkite" / "common" / "scripts"))

from skip_ci import resolve_ci_decision  # noqa: E402
from upload_pipeline import (  # noqa: E402
    NIGHTLY_LABEL_IF,
    _changed_files_for_source_filter,
    _expand_mirror_hardwares,
    _get_mirror_hw_selector,
    _load_bootstrap_steps,
    _load_source_file_dependencies,
    _render_bootstrap_pipeline,
    _render_test_pipeline,
    _resolve_source_file_dependencies,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CUDA_BOOTSTRAP_STEPS = Path(".buildkite/cuda/bootstrap-upload-steps.yml")
BOOTSTRAP_STEPS_TEMPLATE = """steps:
  - key: image-build
  - key: upload-ready-pipeline
  - key: upload-merge-pipeline
  - key: upload-nightly-pipeline
  - key: upload-weekly-pipeline
"""


def _render(changed_files: list[str]) -> str:
    decision = resolve_ci_decision(changed_files)
    return _render_bootstrap_pipeline(
        BOOTSTRAP_STEPS_TEMPLATE,
        decision=decision,
        path=CUDA_BOOTSTRAP_STEPS,
    )


def test_bootstrap_if_injected_by_step_key() -> None:
    rendered = _render_bootstrap_pipeline(
        BOOTSTRAP_STEPS_TEMPLATE,
        decision=resolve_ci_decision([]),
        path=Path(".buildkite/npu/bootstrap-upload-steps.yml"),
    )
    doc = yaml.safe_load(rendered)
    by_key = {step["key"]: step for step in doc["steps"]}
    assert "image-build" in by_key
    # Unconditional image has no ``if`` (Buildkite rejects YAML bool if: true).
    assert "if" not in by_key["image-build"]
    assert isinstance(by_key["upload-ready-pipeline"]["if"], str)
    assert isinstance(by_key["upload-nightly-pipeline"]["if"], str)


def test_bootstrap_steps_loaded_from_file() -> None:
    steps = _load_bootstrap_steps(CUDA_BOOTSTRAP_STEPS)
    assert "key: image-build" in steps
    assert "key: upload-ready-pipeline" in steps
    assert "placeholder:" not in steps


def test_docs_only_allows_main_scheduled_nightly_weekly_only() -> None:
    """skip_all: no PR labels; main + NIGHTLY=1 / WEEKLY=1 / NON_CRITICAL=1 still gates scheduled CI."""
    rendered = _render(["docs/foo.md"])
    assert "key: image-build" in rendered
    assert "key: upload-nightly-pipeline" in rendered
    assert "key: upload-weekly-pipeline" in rendered
    # Scheduled main+WEEKLY=1 uploads L2/L3 with --e2e; NIGHTLY still gates L4 only.
    doc = yaml.safe_load(rendered)
    by_key = {step["key"]: step for step in doc["steps"]}
    assert "NIGHTLY" not in by_key["upload-ready-pipeline"]["if"]
    assert 'build.branch == "main"' in by_key["upload-ready-pipeline"]["if"]
    assert 'build.env("WEEKLY") == "1"' in by_key["upload-ready-pipeline"]["if"]
    assert "NIGHTLY" not in by_key["upload-merge-pipeline"]["if"]
    assert 'build.branch == "main"' in by_key["upload-merge-pipeline"]["if"]
    assert 'build.env("WEEKLY") == "1"' in by_key["upload-merge-pipeline"]["if"]
    assert 'build.env("NIGHTLY") == "1"' in by_key["upload-nightly-pipeline"]["if"]
    assert 'build.env("WEEKLY") == "1"' in rendered
    assert 'build.env("NON_CRITICAL") == "1"' in rendered
    assert "nightly-test" not in rendered
    assert "weekly-test" not in rendered
    assert "merge-test" not in rendered
    assert 'labels includes "ready"' not in rendered
    assert "if: false" not in rendered


def test_npu_docs_only_does_not_upload_ready_on_nightly() -> None:
    """NPU skip_all: scheduled NIGHTLY still uploads L4, not L2 ready."""
    rendered = _render_bootstrap_pipeline(
        BOOTSTRAP_STEPS_TEMPLATE,
        decision=resolve_ci_decision(["docs/foo.md"]),
        path=Path(".buildkite/npu/bootstrap-upload-steps.yml"),
    )
    doc = yaml.safe_load(rendered)
    by_key = {step["key"]: step for step in doc["steps"]}
    assert "upload-ready-pipeline" not in by_key
    assert 'build.env("NIGHTLY") == "1"' in by_key["upload-nightly-pipeline"]["if"]


def test_nightly_label_if_is_only_nightly_test() -> None:
    assert 'labels includes "nightly-test"' in NIGHTLY_LABEL_IF
    forbidden = (
        'includes "omni-test"',
        'includes "tts-test"',
        'includes "diffusion-x2iat-test"',
        'includes "diffusion-x2v-test"',
    )
    for needle in forbidden:
        assert needle not in NIGHTLY_LABEL_IF
    for path in (
        Path(".buildkite/cuda/test-nightly.yml"),
        Path(".buildkite/npu/test-npu-nightly.yml"),
        Path(".buildkite/common/scripts/upload_pipeline.py"),
    ):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path} still references {needle}"


def test_yaml_gated_l45_only_does_not_unconditionally_build_image() -> None:
    rendered = _render([".buildkite/cuda/test-nightly.yml"])
    assert "if: true" not in rendered
    assert 'build.pull_request.labels includes "nightly-test"' in rendered
    assert 'build.pull_request.labels includes "weekly-test"' in rendered
    # L2/L3 upload steps are unconditionally disabled → omitted from pipeline
    assert "key: upload-ready-pipeline" not in rendered
    assert "key: upload-merge-pipeline" not in rendered
    assert "key: upload-weekly-pipeline" in rendered


def test_yaml_gated_l2_still_enables_image_via_ready_base() -> None:
    rendered = _render([".buildkite/cuda/test-ready.yml"])
    assert 'build.pull_request.labels includes "ready"' in rendered
    assert "if: true" not in rendered


def test_mirror_hardwares_l4_1_expands_to_agents_and_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    doc = {
        "steps": [
            {
                "label": "Simple Test",
                "mirror_hardwares": "l4_1",
                "commands": ["pytest -sv tests/example"],
            },
        ],
    }
    rendered = _render_test_pipeline(doc, changed_files=None)
    step = rendered["steps"][0]
    assert "mirror_hardwares" not in step
    assert step["agents"]["queue"] == "l4-k8s"
    assert step["retry"] == {
        "automatic": [
            {"exit_status": -1, "limit": 1},
            {"exit_status": 128, "limit": 1},
            {"signal_reason": "agent_stop", "limit": 1},
            {"signal_reason": "agent_refused", "limit": 1},
        ],
    }
    container = step["plugins"][0]["kubernetes"]["podSpec"]["containers"][0]
    assert container["image"].endswith("$BUILDKITE_COMMIT")
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1


def test_mirror_hardwares_l4_preserves_explicit_retry() -> None:
    step = _expand_mirror_hardwares(
        {
            "label": "opt-out",
            "mirror_hardwares": "l4_1",
            "retry": {"automatic": [{"exit_status": 255, "limit": 2}]},
        },
    )
    assert step["retry"] == {"automatic": [{"exit_status": 255, "limit": 2}]}


def test_mirror_hardwares_conflicts_with_explicit_agents() -> None:
    with pytest.raises(ValueError, match="agents/plugins/image"):
        _expand_mirror_hardwares(
            {"label": "bad", "mirror_hardwares": "l4_1", "agents": {"queue": "gpu_1_queue"}},
        )


def test_mirror_hardwares_a2b3_npu_4_expands_agents_image_and_plugins() -> None:
    doc = {
        "steps": [
            {
                "label": "NPU X2V Test",
                "mirror_hardwares": "a2b3_npu_4",
                "commands": ["pytest -sv tests/example"],
            },
        ],
    }
    rendered = _render_test_pipeline(doc, changed_files=None)
    step = rendered["steps"][0]
    assert "mirror_hardwares" not in step
    assert step["agents"]["queue"] == "ascend-a2b3"
    assert step["agents"]["resource_class"] == "npu-4"
    assert step["image"].endswith("${BUILDKITE_COMMIT}")
    assert step["plugins"][0]["kubernetes"]["podSpecPatch"]["imagePullSecrets"] == [
        {"name": "swr-secret"},
    ]


def _gpu_limit(step: dict) -> int:
    return step["plugins"][0]["kubernetes"]["podSpec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"]


def test_mirror_hardwares_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a preset string"):
        _expand_mirror_hardwares(
            {"label": "bad", "mirror_hardwares": {"default": "h100_2", "b200": "b200_2"}},
        )


@pytest.mark.parametrize("hardware", [2, "2", "h100_99", "not_a_preset"])
def test_mirror_hardwares_unknown_name_is_rejected(hardware: int | str) -> None:
    with pytest.raises(ValueError, match="unknown mirror_hardwares"):
        _expand_mirror_hardwares({"label": "unknown preset", "mirror_hardwares": hardware})


def test_mirror_hardwares_string_ignores_pytest_cards_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares(
        {
            "label": "forced",
            "commands": ['pytest -sv tests/e2e -m "full_model and L4 and cards_4"'],
            "mirror_hardwares": "h100_1",
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "mithril-h100-pool"
    assert _gpu_limit(step) == 1


def test_mirror_hardwares_b200_omits_cuda_strings_and_remaps_inferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "b200")
    assert _expand_mirror_hardwares({"label": "h100", "mirror_hardwares": "h100_4"}) is None
    assert _expand_mirror_hardwares({"label": "l4", "mirror_hardwares": "l4_1"}) is None
    b200 = _expand_mirror_hardwares({"label": "b200", "mirror_hardwares": "b200_2"})
    assert b200 is not None and b200["agents"]["queue"] == "b200-k8s"
    npu = _expand_mirror_hardwares({"label": "npu", "mirror_hardwares": "a2b3_npu_4"})
    assert npu is not None and npu["agents"]["queue"] == "ascend-a2b3"

    rendered = _render_test_pipeline(
        {
            "steps": [
                {
                    "group": ":card_index_dividers: Mixed",
                    "steps": [
                        {"label": "H100 only", "mirror_hardwares": "h100_4"},
                        {
                            "label": "Count remap",
                            "commands": [
                                'pytest -sv tests/e2e -m "full_model and H100 and B200 and omni and cards_2"',
                            ],
                        },
                    ],
                },
                {
                    "group": ":card_index_dividers: H100 only group",
                    "steps": [{"label": "Skip me", "mirror_hardwares": "h100_1"}],
                },
            ],
        },
        changed_files=None,
    )
    groups = [step.get("group") for step in rendered["steps"]]
    assert ":card_index_dividers: H100 only group" not in groups
    mixed = next(step for step in rendered["steps"] if step.get("group") == ":card_index_dividers: Mixed")
    assert [child["label"] for child in mixed["steps"]] == ["Count remap"]
    assert mixed["steps"][0]["agents"]["queue"] == "b200-k8s"


@pytest.mark.parametrize(
    ("selector", "expr", "queue", "gpus"),
    [
        ("", "H100 and B200 and cards_2", "mithril-h100-pool", 2),
        ("b200", "H100 and B200 and cards_2", "b200-k8s", 2),
        ("", "H100 or L4 and cards_4", "mithril-h100-pool", 4),
        ("", "L4 and B200 and cards_4", "l4-k8s", 4),
        ("", "H100 and cards_2 and cards_3", "mithril-h100-pool", 3),
        ("", "H100 and not cards_1", "mithril-h100-pool", 4),
        ("b200", "H100 and B200 and not cards_1", "b200-k8s", 4),
    ],
)
def test_mirror_hardwares_inferred_from_marks(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    expr: str,
    queue: str,
    gpus: int,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: selector)
    step = _expand_mirror_hardwares({"label": "job", "commands": [f'pytest -sv tests/e2e -m "{expr}"']})
    assert step is not None
    assert step["agents"]["queue"] == queue
    assert _gpu_limit(step) == gpus


@pytest.mark.parametrize(
    ("selector", "expr"),
    [
        ("b200", "H100 and cards_2"),
        ("", "B200 and cards_2"),
        ("", "full_model and cards_2"),
    ],
)
def test_mirror_hardwares_inferred_skips_unmatched_chip(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    expr: str,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: selector)
    assert _expand_mirror_hardwares({"label": "job", "commands": [f'pytest -sv tests/e2e -m "{expr}"']}) is None


def test_mirror_hardwares_inferred_missing_preset_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    with pytest.raises(ValueError, match="has no preset 'h100_8'"):
        _expand_mirror_hardwares(
            {"label": "H100 8-gpu", "commands": ['pytest -sv tests/e2e -m "H100 and cards_8"']},
        )


def test_cpu_step_without_mirror_hardwares_is_unchanged() -> None:
    step = {"label": "CPU report", "commands": ["echo ok"], "agents": {"queue": "cpu_queue_premerge"}}
    assert _expand_mirror_hardwares(step) is step


@pytest.mark.parametrize(("raw", "expected"), [("", ""), ("  ", ""), ("b200", "b200"), ("B200", "b200")])
def test_mirror_hw_selector_empty_or_b200(monkeypatch: pytest.MonkeyPatch, raw: str, expected: str) -> None:
    monkeypatch.setenv("MIRROR_HW", raw)
    assert _get_mirror_hw_selector() == expected


def test_mirror_hw_typo_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIRROR_HW", "b20o")
    with pytest.raises(ValueError, match=r"unsupported MIRROR_HW='b20o'"):
        _get_mirror_hw_selector()
    with pytest.raises(ValueError, match=r"unsupported MIRROR_HW='b20o'"):
        _render_test_pipeline(
            {
                "steps": [
                    {"label": "CPU report", "commands": ["echo ok"]},
                    {"label": "H100 string", "mirror_hardwares": "h100_4"},
                ],
            },
            changed_files=None,
        )


def _surviving_labels(
    doc: dict,
    changed_files: list[str],
    *,
    pipeline_path: Path | None = None,
) -> set[str]:
    rendered = _render_test_pipeline(
        doc,
        changed_files=changed_files,
        pipeline_path=pipeline_path,
    )
    labels: set[str] = set()

    def walk(steps: list | None) -> None:
        for step in steps or []:
            if not isinstance(step, dict):
                continue
            if "label" in step:
                labels.add(step["label"])
            walk(step.get("steps"))

    walk(rendered.get("steps"))
    return labels


def _iter_steps(doc: dict):
    def walk(steps: list | None):
        for step in steps or []:
            if not isinstance(step, dict):
                continue
            yield step
            yield from walk(step.get("steps"))

    yield from walk(doc.get("steps"))


# Synthetic coverage-style job: shared inputs that change what the split measures.
_COVERAGE_SHARED_INPUTS_DOC = {
    "steps": [
        {
            "label": "Coverage Pilot",
            "source_file_dependencies": [
                "tests/e2e/online_serving/test_example.py",
                ".buildkite/common/scripts/run_cov_split.sh",
                "pyproject.toml",
            ],
            "commands": [".buildkite/common/scripts/run_cov_split.sh --model-id example"],
        },
        {
            "label": "Unrelated Model Test",
            "source_file_dependencies": [
                "tests/e2e/online_serving/test_other.py",
            ],
            "commands": ["pytest -sv tests/e2e/online_serving/test_other.py"],
        },
    ],
}


@pytest.mark.parametrize(
    "changed_file",
    [
        ".buildkite/common/scripts/run_cov_split.sh",
        "pyproject.toml",
    ],
)
def test_coverage_shared_inputs_select_dependent_job(changed_file: str) -> None:
    """Jobs that list coverage shared inputs must stay selected when those files change."""
    labels = _surviving_labels(_COVERAGE_SHARED_INPUTS_DOC, [changed_file])
    assert "Coverage Pilot" in labels
    assert "Unrelated Model Test" not in labels


def test_coverage_shared_inputs_ignored_for_unrelated_change() -> None:
    labels = _surviving_labels(
        _COVERAGE_SHARED_INPUTS_DOC,
        ["vllm_omni/entrypoints/openai/serving_chat.py"],
    )
    assert "Coverage Pilot" not in labels
    assert "Unrelated Model Test" not in labels


def _pipeline_dep_keys(path: Path) -> set[str]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys: set[str] = set()

    def walk(steps: list | None) -> None:
        for step in steps or []:
            if not isinstance(step, dict):
                continue
            deps = step.get("source_file_dependencies")
            if isinstance(deps, str) and "/" not in deps:
                keys.add(deps)
            elif isinstance(deps, list) and deps and all("/" not in item for item in deps):
                keys.update(deps)
            walk(step.get("steps"))

    walk((doc or {}).get("steps"))
    return keys


def test_pipeline_source_file_dependency_keys_are_registered() -> None:
    _load_source_file_dependencies.cache_clear()
    registry = _load_source_file_dependencies()
    used = (
        _pipeline_dep_keys(Path(".buildkite/cuda/test-ready.yml"))
        | _pipeline_dep_keys(Path(".buildkite/cuda/test-merge.yml"))
        | _pipeline_dep_keys(Path(".buildkite/cuda/test-nightly.yml"))
        | _pipeline_dep_keys(Path(".buildkite/cuda/test-weekly.yml"))
        | _pipeline_dep_keys(Path(".buildkite/npu/test-npu-nightly.yml"))
    )
    missing = used - set(registry)
    assert not missing, f"unregistered source_file_dependencies keys: {sorted(missing)}"


def test_source_file_dependencies_key_expands_from_registry() -> None:
    doc = {
        "steps": [
            {
                "label": "Diffusion · Qwen Image Test",
                "source_file_dependencies": "diffusion_qwen_image_function",
                "commands": ["pytest"],
            },
        ],
    }
    assert "Diffusion · Qwen Image Test" in _surviving_labels(
        doc,
        ["vllm_omni/diffusion/models/qwen_image/transformer.py"],
    )
    assert "Diffusion · Qwen Image Test" not in _surviving_labels(doc, ["vllm_omni/unrelated.py"])


def test_registry_lists_pytest_targets() -> None:
    _load_source_file_dependencies.cache_clear()
    resolved = _resolve_source_file_dependencies(
        {
            "label": "Diffusion · Wan22 Test",
            "source_file_dependencies": "diffusion_wan22_function",
            "commands": [
                "pytest -s -v tests/e2e/offline_inference/test_wan22_t2v.py "
                "tests/e2e/online_serving/test_wan22_t2v.py -m 'advanced_model'",
            ],
        },
    )
    assert resolved is not None
    assert "tests/e2e/offline_inference/test_wan22_t2v.py" in resolved
    assert "tests/e2e/online_serving/test_wan22_t2v.py" in resolved
    assert "vllm_omni/diffusion/models/wan2_2/" in resolved


def test_coverage_key_lists_offline_online_scripts() -> None:
    resolved = _resolve_source_file_dependencies(
        {
            "label": "TTS · Qwen3-TTS Base Test",
            "source_file_dependencies": "tts_qwen3_tts_cov",
            "commands": [
                ".buildkite/common/scripts/run_cov_split.sh \\\n"
                "  --offline tests/e2e/offline_inference/test_qwen3_tts_base.py \\\n"
                "  --online tests/e2e/online_serving/test_qwen3_tts_base.py",
            ],
        },
    )
    assert resolved is not None
    assert "tests/e2e/offline_inference/test_qwen3_tts_base.py" in resolved
    assert "tests/e2e/online_serving/test_qwen3_tts_base.py" in resolved
    assert ".buildkite/common/scripts/run_cov_split.sh" in resolved
    assert "pyproject.toml" in resolved


def test_source_file_dependencies_list_of_keys_concatenates() -> None:
    resolved = _resolve_source_file_dependencies(
        {
            "label": "composed",
            "source_file_dependencies": ["omni_qwen3_omni_function", "tts_qwen3_tts_function"],
        },
    )
    assert resolved is not None
    assert "vllm_omni/model_executor/models/qwen3_omni/" in resolved
    assert "vllm_omni/model_executor/models/qwen3_tts/" in resolved
    assert resolved.count("vllm_omni/model_executor/models/common/snake_activation.py") == 1


def test_unknown_source_file_dependencies_key() -> None:
    with pytest.raises(ValueError, match="unknown source_file_dependencies"):
        _render_test_pipeline(
            {"steps": [{"label": "bad", "source_file_dependencies": "not_a_real_key"}]},
            changed_files=None,
        )


def test_source_file_dependencies_rejects_mixed_keys_and_paths() -> None:
    with pytest.raises(ValueError, match="mixes registry keys and path prefixes"):
        _resolve_source_file_dependencies(
            {
                "label": "bad",
                "source_file_dependencies": ["omni_qwen3_omni_function", "tests/e2e/online_serving/test_qwen3_omni.py"],
            },
        )


# Synthetic pipeline: selection depends only on listed deps, not on live job names.
_SOURCE_FILTER_DOC = {
    "steps": [
        {
            "group": "E2E Tests",
            "if": 'build.env("NON_CRITICAL") == "1"',
            "steps": [
                {
                    "label": "Dedicated E2E",
                    "commands": ["pytest -sv tests/e2e/online_serving/test_magi2.py"],
                },
            ],
        },
        {
            "label": "Omni Sweep",
            "source_file_dependencies": ["tests/e2e/online_serving/test_qwen3_omni.py"],
            "commands": ["pytest -sv tests/e2e/ -m omni"],
        },
        {
            "label": "Z-Image Function",
            "source_file_dependencies": [
                "tests/e2e/online_serving/test_zimage_expansion.py",
                "vllm_omni/diffusion/models/z_image/",
            ],
            "mirror_hardwares": "l4_4",
            "commands": ["pytest -sv tests/e2e/online_serving/test_zimage_expansion.py"],
        },
        {
            "label": "Tiny Model",
            "source_file_dependencies": ["tests/e2e/online_serving/test_tiny.py"],
            "commands": ["pytest -sv tests/e2e/ -m tiny"],
        },
        {
            "label": "Wan Function",
            "source_file_dependencies": [
                "tests/e2e/offline_inference/test_wan22_t2v.py",
                "tests/e2e/online_serving/test_wan22_t2v.py",
                "vllm_omni/diffusion/models/wan2_2/",
            ],
            "commands": ["pytest -sv tests/e2e/offline_inference/test_wan22_t2v.py"],
        },
        {
            "label": "Wan Perf",
            "source_file_dependencies": ["vllm_omni/diffusion/models/wan2_2/"],
            "commands": ["pytest -sv tests/dfx/perf/scripts/run_benchmark.py"],
        },
        {
            "label": "Doc Test",
            "source_file_dependencies": [
                "tests/examples/offline_inference/test_text_to_image.py",
                "tests/examples/online_serving/test_text_to_image.py",
                "vllm_omni/diffusion/models/qwen_image/",
                "vllm_omni/diffusion/models/z_image/",
            ],
            "commands": ["pytest -sv tests/examples/*/test_text_to_image.py"],
        },
    ],
}


def test_source_filter_ignores_unrelated_e2e_file() -> None:
    """A sweep command is not a dependency; only listed paths select a keyed job."""
    labels = _surviving_labels(_SOURCE_FILTER_DOC, ["tests/e2e/online_serving/test_magi2.py"])
    assert labels == {"Dedicated E2E"}


def test_source_filter_selects_job_by_listed_script_not_sibling() -> None:
    labels = _surviving_labels(_SOURCE_FILTER_DOC, ["tests/e2e/online_serving/test_zimage_expansion.py"])
    assert labels == {"Dedicated E2E", "Z-Image Function"}


def test_source_filter_selects_every_job_sharing_a_path() -> None:
    labels = _surviving_labels(_SOURCE_FILTER_DOC, ["vllm_omni/diffusion/models/wan2_2/transformer.py"])
    assert labels == {"Dedicated E2E", "Wan Function", "Wan Perf"}


def test_source_filter_selects_composed_doc_job_by_example_or_model() -> None:
    for changed in (
        "tests/examples/online_serving/test_text_to_image.py",
        "vllm_omni/diffusion/models/z_image/transformer.py",
        "vllm_omni/diffusion/models/qwen_image/foo.py",
    ):
        labels = _surviving_labels(_SOURCE_FILTER_DOC, [changed])
        assert "Doc Test" in labels, changed
        assert "Tiny Model" not in labels
        assert "Omni Sweep" not in labels
    assert "Doc Test" not in _surviving_labels(_SOURCE_FILTER_DOC, ["tests/e2e/online_serving/test_magi2.py"])


def test_source_filter_strips_deps_and_expands_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    rendered = _render_test_pipeline(
        _SOURCE_FILTER_DOC,
        changed_files=["tests/e2e/online_serving/test_zimage_expansion.py"],
    )
    dumped = yaml.safe_dump(rendered)
    assert "source_file_dependencies" not in dumped
    assert "mirror_hardwares" not in dumped
    z_image = next(step for step in _iter_steps(rendered) if step.get("label") == "Z-Image Function")
    assert z_image["agents"]["queue"] == "l4-k8s"


def test_source_filter_disabled_on_main_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Ctx:
        changed_files = ["vllm_omni/unrelated.py"]

    monkeypatch.setenv("BUILDKITE_BRANCH", "main")
    assert _changed_files_for_source_filter(_Ctx(), force_all=False, e2e_only=False) is None

    monkeypatch.setenv("BUILDKITE_BRANCH", "feat/source-filter")
    assert _changed_files_for_source_filter(_Ctx(), force_all=False, e2e_only=False) == [
        "vllm_omni/unrelated.py",
    ]
    monkeypatch.setenv("BUILDKITE_BRANCH", "main")
    assert _changed_files_for_source_filter(_Ctx(), force_all=True, e2e_only=False) is None


def test_source_filter_fallback_is_fallback_when_no_job_key_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback only when no listed prefix matched; a matching prefix still wins."""
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    monkeypatch.setenv("BUILDKITE_BRANCH", "feat/nightly-yaml")
    pipeline_yaml = Path(".buildkite/npu/test-npu-nightly.yml")
    shared_paths = _load_source_file_dependencies()["source_filter_fallback"]
    assert ".buildkite/common/scripts/upload_pipeline.py" in shared_paths

    # Synthetic steps only — do not pin live Buildkite job labels.
    doc = {
        "steps": [
            {"key": "ungated", "commands": ["true"]},
            {
                "key": "gated_a",
                "source_file_dependencies": ["pkg/model_a/"],
                "commands": ["true"],
            },
            {
                "key": "gated_b",
                "source_file_dependencies": ["pkg/model_b/"],
                "commands": ["true"],
            },
        ],
    }

    def surviving_keys(changed_files: list[str]) -> set[str]:
        rendered = _render_test_pipeline(
            doc,
            changed_files=changed_files,
            pipeline_path=pipeline_yaml,
        )
        return {step["key"] for step in _iter_steps(rendered) if isinstance(step.get("key"), str)}

    # Only pipeline YAML / shared uploader → no listed prefix match → keep every step.
    for changed in [pipeline_yaml.as_posix(), *shared_paths]:
        assert surviving_keys([changed]) == {"ungated", "gated_a", "gated_b"}, changed

    # A different pipeline YAML is not a fallback for this upload.
    assert surviving_keys([".buildkite/cuda/test-nightly.yml"]) == {"ungated"}

    # A matching source prefix wins over fallback files in the same diff.
    assert surviving_keys(
        [
            "pkg/model_a/transformer.py",
            ".buildkite/common/scripts/upload_pipeline.py",
        ],
    ) == {"ungated", "gated_a"}
