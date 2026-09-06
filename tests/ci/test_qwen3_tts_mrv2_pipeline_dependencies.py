from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
JOB_LABEL = "TTS · Qwen3-TTS MRv2 Test"
BUILDKITE_SCRIPTS = ROOT / ".buildkite" / "common" / "scripts"
sys.path.insert(0, str(BUILDKITE_SCRIPTS))

from upload_pipeline import _render_test_pipeline  # noqa: E402


def _labels(steps):
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        if label := step.get("label"):
            yield str(label)
        yield from _labels(step.get("steps"))


@pytest.mark.parametrize(
    "changed_file",
    [
        "vllm_omni/config/omni_config.py",
        "vllm_omni/core/sched/omni_ar_scheduler.py",
        "vllm_omni/engine/orchestrator.py",
        "vllm_omni/outputs/output_processor.py",
        "vllm_omni/worker/gpu_generation_worker.py",
    ],
)
def test_mrv2_gpu_job_covers_shared_runtime_dependencies(changed_file: str) -> None:
    pipeline = yaml.safe_load((ROOT / ".buildkite" / "cuda" / "test-merge.yml").read_text())

    rendered = _render_test_pipeline(pipeline, [changed_file])

    assert JOB_LABEL in set(_labels(rendered.get("steps")))
