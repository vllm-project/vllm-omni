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
    _expand_mirror_hardwares,
    _load_bootstrap_steps,
    _render_bootstrap_pipeline,
    _render_test_pipeline,
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


def test_mirror_hardwares_mapping_uses_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares(
        {
            "label": "Nightly Omni",
            "mirror_hardwares": {"default": "h100_2", "b200": "b200_2"},
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "mithril-h100-pool"
    assert step["plugins"][0]["kubernetes"]["podSpec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 2


def test_mirror_hardwares_mapping_selects_b200_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "b200")
    step = _expand_mirror_hardwares(
        {
            "label": "Nightly Omni",
            "mirror_hardwares": {"default": "h100_2", "b200": "b200_2"},
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "b200-k8s"
    assert step["plugins"][0]["kubernetes"]["podSpec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 2


def test_mirror_hardwares_mapping_missing_selector_skips_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "b200")
    assert (
        _expand_mirror_hardwares(
            {
                "label": "H100 only",
                "mirror_hardwares": {"default": "h100_4"},
            },
        )
        is None
    )

    doc = {
        "steps": [
            {
                "group": ":card_index_dividers: Mixed",
                "steps": [
                    {"label": "H100 only", "mirror_hardwares": {"default": "h100_4"}},
                    {
                        "label": "Both",
                        "mirror_hardwares": {"default": "h100_2", "b200": "b200_2"},
                    },
                    {"label": "String H100", "mirror_hardwares": "h100_4"},
                ],
            },
            {
                "group": ":card_index_dividers: H100 only group",
                "steps": [
                    {"label": "Skip me", "mirror_hardwares": {"default": "h100_1"}},
                ],
            },
        ],
    }
    rendered = _render_test_pipeline(doc, changed_files=None)
    groups = [step.get("group") for step in rendered["steps"]]
    assert ":card_index_dividers: H100 only group" not in groups
    mixed = next(step for step in rendered["steps"] if step.get("group") == ":card_index_dividers: Mixed")
    assert [child["label"] for child in mixed["steps"]] == ["Both"]
    assert mixed["steps"][0]["agents"]["queue"] == "b200-k8s"


def test_mirror_hardwares_cuda_string_skips_when_selector_is_other_chip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "b200")
    assert _expand_mirror_hardwares({"label": "h100 string", "mirror_hardwares": "h100_4"}) is None
    assert _expand_mirror_hardwares({"label": "l4 string", "mirror_hardwares": "l4_1"}) is None
    b200 = _expand_mirror_hardwares({"label": "b200 string", "mirror_hardwares": "b200_2"})
    assert b200 is not None
    assert b200["agents"]["queue"] == "b200-k8s"
    npu = _expand_mirror_hardwares({"label": "npu string", "mirror_hardwares": "a2b3_npu_4"})
    assert npu is not None
    assert npu["agents"]["queue"] == "ascend-a2b3"


def test_mirror_hardwares_cuda_string_kept_when_selector_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares({"label": "h100 string", "mirror_hardwares": "h100_4"})
    assert step is not None
    assert step["agents"]["queue"] == "mithril-h100-pool"


def test_mirror_hardwares_mapping_requires_default() -> None:
    with pytest.raises(ValueError, match="must include a 'default' key"):
        _expand_mirror_hardwares(
            {"label": "bad", "mirror_hardwares": {"b200": "b200_2"}},
        )


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
