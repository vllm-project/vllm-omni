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
    _get_mirror_hw_selector,
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

    doc = {
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
                    {"label": "String H100", "mirror_hardwares": "h100_4"},
                ],
            },
            {
                "group": ":card_index_dividers: H100 only group",
                "steps": [
                    {"label": "Skip me", "mirror_hardwares": "h100_1"},
                ],
            },
        ],
    }
    rendered = _render_test_pipeline(doc, changed_files=None)
    groups = [step.get("group") for step in rendered["steps"]]
    assert ":card_index_dividers: H100 only group" not in groups
    mixed = next(step for step in rendered["steps"] if step.get("group") == ":card_index_dividers: Mixed")
    assert [child["label"] for child in mixed["steps"]] == ["Count remap"]
    assert mixed["steps"][0]["agents"]["queue"] == "b200-k8s"


def test_mirror_hardwares_cuda_string_kept_when_selector_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares({"label": "h100 string", "mirror_hardwares": "h100_4"})
    assert step is not None
    assert step["agents"]["queue"] == "mithril-h100-pool"


def test_mirror_hardwares_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a preset string"):
        _expand_mirror_hardwares(
            {"label": "bad", "mirror_hardwares": {"default": "h100_2", "b200": "b200_2"}},
        )


@pytest.mark.parametrize("hardware", [2, "2", "h100_99", "not_a_preset"])
def test_mirror_hardwares_unknown_name_is_rejected(hardware: int | str) -> None:
    with pytest.raises(ValueError, match="unknown mirror_hardwares"):
        _expand_mirror_hardwares(
            {
                "label": "unknown preset",
                "mirror_hardwares": hardware,
            },
        )


def test_mirror_hardwares_string_ignores_pytest_cards_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert step["plugins"][0]["kubernetes"]["podSpec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 1


def test_mirror_hardwares_inferred_from_sku_and_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares(
        {
            "label": "Omni Function",
            "commands": [
                'pytest -sv tests/e2e -m "full_model and H100 and B200 and omni and cards_2"',
            ],
        },
    )
    assert step is not None
    assert "mirror_hardwares" not in step
    assert step["agents"]["queue"] == "mithril-h100-pool"
    assert step["plugins"][0]["kubernetes"]["podSpec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 2


def test_mirror_hardwares_inferred_skips_when_h100_only_and_mirror_hw_b200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "b200")
    assert (
        _expand_mirror_hardwares(
            {
                "label": "Omni Function",
                "commands": ['pytest -sv tests/e2e -m "full_model and H100 and omni and cards_2"'],
            },
        )
        is None
    )


def test_mirror_hardwares_inferred_skips_when_unset_and_only_b200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    assert (
        _expand_mirror_hardwares(
            {
                "label": "B200 only",
                "commands": ['pytest -sv tests/e2e -m "B200 and cards_2"'],
            },
        )
        is None
    )


def test_mirror_hardwares_inferred_skips_cuda_sku_without_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    assert (
        _expand_mirror_hardwares(
            {
                "label": "H200 only",
                "commands": ['pytest -sv tests/e2e -m "H200 and cards_2"'],
            },
        )
        is None
    )


def test_mirror_hardwares_inferred_cards_8_rejected_without_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    with pytest.raises(ValueError, match="has no preset 'h100_8'"):
        _expand_mirror_hardwares(
            {
                "label": "H100 8-gpu",
                "commands": ['pytest -sv tests/e2e -m "H100 and cards_8"'],
            },
        )


def test_mirror_hardwares_inferred_skips_when_cards_without_h100_or_l4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    assert (
        _expand_mirror_hardwares(
            {
                "label": "no sku",
                "commands": ['pytest -sv tests/e2e -m "full_model and cards_2"'],
            },
        )
        is None
    )


def test_mirror_hardwares_inferred_follows_mirror_hw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "b200")
    step = _expand_mirror_hardwares(
        {
            "label": "Omni Function",
            "commands": [
                'pytest -sv tests/e2e -m "full_model and H100 and B200 and omni and cards_2"',
            ],
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "b200-k8s"


def test_mirror_hardwares_inferred_skips_when_mirror_hw_not_in_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "b200")
    assert (
        _expand_mirror_hardwares(
            {
                "label": "H100 or L4",
                "commands": ['pytest -sv tests/e2e -m "H100 or L4 and cards_4"'],
            },
        )
        is None
    )


def _gpu_limit(step: dict) -> int:
    return step["plugins"][0]["kubernetes"]["podSpec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"]


def test_mirror_hardwares_inferred_not_cards_1_uses_l4_highest_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares(
        {
            "label": "L4 remaining",
            "commands": [
                'pytest -sv tests/e2e -m "full_model and diffusion and L4 and not cards_1"',
            ],
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "gpu_4_queue"


def test_mirror_hardwares_inferred_not_cards_1_uses_h100_highest_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares(
        {
            "label": "H100 remaining",
            "commands": ['pytest -sv tests/e2e -m "full_model and H100 and not cards_1"'],
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "mithril-h100-pool"
    assert _gpu_limit(step) == 4


def test_mirror_hardwares_inferred_not_cards_1_follows_mirror_hw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "b200")
    step = _expand_mirror_hardwares(
        {
            "label": "Omni remaining",
            "commands": [
                'pytest -sv tests/e2e -m "full_model and H100 and B200 and not cards_1"',
            ],
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "b200-k8s"
    assert _gpu_limit(step) == 4


def test_mirror_hardwares_inferred_multiple_cards_uses_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares(
        {
            "label": "H100 mixed cards",
            "commands": ['pytest -sv tests/e2e -m "H100 and cards_2 and cards_3"'],
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "mithril-h100-pool"
    assert _gpu_limit(step) == 3


def test_mirror_hardwares_inferred_uses_h100_when_h100_and_l4_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares(
        {
            "label": "mixed",
            "commands": ['pytest -sv tests/e2e -m "H100 or L4 and cards_4"'],
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "mithril-h100-pool"
    assert _gpu_limit(step) == 4


def test_mirror_hardwares_inferred_uses_l4_when_l4_and_b200_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares(
        {
            "label": "L4 and B200",
            "commands": ['pytest -sv tests/e2e -m "L4 and B200 and cards_4"'],
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "gpu_4_queue"


def test_mirror_hardwares_inferred_l4_and_b200_follows_mirror_hw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "b200")
    step = _expand_mirror_hardwares(
        {
            "label": "L4 and B200",
            "commands": ['pytest -sv tests/e2e -m "L4 and B200 and cards_4"'],
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "b200-k8s"
    assert _gpu_limit(step) == 4


def test_mirror_hardwares_inferred_positive_cards_beats_not_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares(
        {
            "label": "H100 2-gpu",
            "commands": ['pytest -sv tests/e2e -m "H100 and cards_2 and not cards_1"'],
        },
    )
    assert step is not None
    assert _gpu_limit(step) == 2


def test_cpu_step_without_mirror_hardwares_is_unchanged() -> None:
    step = {"label": "CPU report", "commands": ["echo ok"], "agents": {"queue": "cpu_queue_premerge"}}
    assert _expand_mirror_hardwares(step) is step


def test_mirror_hardwares_inferred_ignores_not_h100(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("upload_pipeline._get_mirror_hw_selector", lambda: "")
    step = _expand_mirror_hardwares(
        {
            "label": "L4 only",
            "commands": ['pytest -sv tests/e2e -m "full_model and L4 and not H100 and cards_4"'],
        },
    )
    assert step is not None
    assert step["agents"]["queue"] == "gpu_4_queue"


@pytest.mark.parametrize("selector", ["", "   "])
def test_mirror_hw_empty_selector_uses_default(monkeypatch: pytest.MonkeyPatch, selector: str) -> None:
    monkeypatch.setenv("MIRROR_HW", selector)
    assert _get_mirror_hw_selector() == ""


@pytest.mark.parametrize("selector", ["b200", "B200"])
def test_mirror_hw_known_selector_is_accepted(monkeypatch: pytest.MonkeyPatch, selector: str) -> None:
    monkeypatch.setenv("MIRROR_HW", selector)
    assert _get_mirror_hw_selector() == "b200"


@pytest.mark.parametrize("selector", ["h100", "l4", "h200", "b20o"])
def test_mirror_hw_unknown_selector_fails_closed(monkeypatch: pytest.MonkeyPatch, selector: str) -> None:
    """Only empty or b200 are allowed; anything else must fail the upload."""
    monkeypatch.setenv("MIRROR_HW", selector)
    with pytest.raises(ValueError, match=rf"unsupported MIRROR_HW='{selector}'"):
        _get_mirror_hw_selector()


def test_mirror_hw_typo_fails_before_skipping_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typos must fail the upload, not silently drop CUDA steps (leave CPU report only)."""
    monkeypatch.setenv("MIRROR_HW", "b20o")
    doc = {
        "steps": [
            {"label": "CPU report", "commands": ["echo ok"]},
            {
                "label": "Nightly Omni",
                "commands": ['pytest -sv tests/e2e -m "full_model and H100 and omni and cards_2"'],
            },
            {"label": "H100 string", "mirror_hardwares": "h100_4"},
        ],
    }
    with pytest.raises(ValueError, match=r"unsupported MIRROR_HW='b20o'"):
        _render_test_pipeline(doc, changed_files=None)


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
