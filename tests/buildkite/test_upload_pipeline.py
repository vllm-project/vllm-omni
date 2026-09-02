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


def test_mirror_hardwares_l4_1_expands_to_agents_and_plugins() -> None:
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
    assert step["agents"]["queue"] == "gpu_1_queue"
    assert step["plugins"][0]["docker#v5.2.0"]["image"].endswith("$BUILDKITE_COMMIT")


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


def _surviving_labels(doc: dict, changed_files: list[str]) -> set[str]:
    rendered = _render_test_pipeline(doc, changed_files=changed_files)
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
                "source_file_dependencies": "diffusion_qwen_image",
                "commands": ["pytest"],
            },
        ],
    }
    assert "Diffusion · Qwen Image Test" in _surviving_labels(
        doc,
        ["vllm_omni/diffusion/models/qwen_image/transformer.py"],
    )
    assert "Diffusion · Qwen Image Test" not in _surviving_labels(doc, ["vllm_omni/unrelated.py"])


def test_pytest_targets_are_merged_from_commands() -> None:
    resolved = _resolve_source_file_dependencies(
        {
            "label": "Diffusion · Wan22 Test",
            "source_file_dependencies": "diffusion_wan22",
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
    assert "tests/e2e/online_serving/test_wan22_t2v.py" not in _load_source_file_dependencies()["diffusion_wan22"]


def test_run_cov_split_offline_online_are_extracted() -> None:
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
            "source_file_dependencies": ["omni_qwen3_omni", "tts_qwen3_tts"],
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
                "source_file_dependencies": ["omni_qwen3_omni", "tests/e2e/online_serving/test_qwen3_omni.py"],
            },
        )


def test_ready_yaml_key_filter_selects_matching_e2e_job() -> None:
    doc = yaml.safe_load(Path(".buildkite/cuda/test-ready.yml").read_text(encoding="utf-8"))
    labels = _surviving_labels(doc, ["vllm_omni/diffusion/models/qwen_image/foo.py"])
    assert "Diffusion · Qwen Image Test" in labels
    assert "Omni · Qwen3-Omni Test" not in labels
    rendered = _render_test_pipeline(doc, changed_files=["vllm_omni/diffusion/models/qwen_image/foo.py"])
    dumped = yaml.safe_dump(rendered)
    assert "source_file_dependencies" not in dumped
    assert "mirror_hardwares" not in dumped


def test_ready_yaml_pytest_target_selects_job() -> None:
    doc = yaml.safe_load(Path(".buildkite/cuda/test-ready.yml").read_text(encoding="utf-8"))
    labels = _surviving_labels(doc, ["tests/e2e/online_serving/test_wan22_t2v.py"])
    assert "Diffusion · Wan22 Test" in labels
    assert "Diffusion · Qwen Image Test" not in labels


def test_merge_yaml_pytest_offline_target_selects_shared_key_job() -> None:
    doc = yaml.safe_load(Path(".buildkite/cuda/test-merge.yml").read_text(encoding="utf-8"))
    labels = _surviving_labels(doc, ["tests/e2e/offline_inference/test_wan22_t2v.py"])
    assert "Diffusion · Wan22 Test" in labels
    assert "Diffusion · Qwen Image Test" not in labels


def test_nightly_yaml_source_key_selects_matching_job() -> None:
    doc = yaml.safe_load(Path(".buildkite/cuda/test-nightly.yml").read_text(encoding="utf-8"))
    labels = _surviving_labels(doc, ["vllm_omni/diffusion/models/wan2_2/transformer.py"])
    assert any("Wan2.2 T2V Function Test" in label for label in labels)
    assert not any("Single-GPU" in label and "Qwen-Image" in label for label in labels)


def test_npu_nightly_yaml_source_key_selects_matching_job() -> None:
    _load_source_file_dependencies.cache_clear()
    doc = yaml.safe_load(Path(".buildkite/npu/test-npu-nightly.yml").read_text(encoding="utf-8"))
    labels = _surviving_labels(doc, ["vllm_omni/diffusion/models/wan2_2/transformer.py"])
    assert any("Diffusion X2V · Function Test" in label for label in labels)
    assert any("Diffusion X2V · Perf Test" in label for label in labels)
    assert not any("MiniCPM" in label for label in labels)
    assert not any("HunyuanImage3" in label for label in labels)
    rendered = _render_test_pipeline(
        doc,
        changed_files=["vllm_omni/diffusion/models/wan2_2/transformer.py"],
    )
    dumped = yaml.safe_dump(rendered)
    assert "source_file_dependencies" not in dumped
    assert "mirror_hardwares" not in dumped


def test_weekly_yaml_source_key_selects_matching_job() -> None:
    _load_source_file_dependencies.cache_clear()
    doc = yaml.safe_load(Path(".buildkite/cuda/test-weekly.yml").read_text(encoding="utf-8"))
    labels = _surviving_labels(doc, ["vllm_omni/diffusion/models/wan2_2/transformer.py"])
    assert "Reliability Test · wan22" in labels
    assert "Reliability Test · Invalid parameters · H100 · Single-GPU" in labels
    assert "Reliability Test · Invalid parameters · H100 · 2-GPU" in labels
    assert "Reliability Test · qwen3-omni" not in labels
    assert "Reliability Test · Invalid parameters · L4" not in labels
    assert not any("Perf Test · vLLM Text" in label for label in labels)
    rendered = _render_test_pipeline(
        doc,
        changed_files=["vllm_omni/diffusion/models/wan2_2/transformer.py"],
    )
    assert "source_file_dependencies" not in yaml.safe_dump(rendered)


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
