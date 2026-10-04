# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 GPU jobs survive diff filtering and default/B200 mirror selection."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".buildkite/common/scripts"))

from upload_pipeline import _render_test_pipeline  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("tier", ["ready", "merge", "nightly"])
@pytest.mark.parametrize("mirror", ["", "b200"])
@pytest.mark.parametrize(
    "changed", ["vllm_omni/model_executor/models/zonos2/zonos2_talker.py", "vllm_omni/worker/gpu_model_runner.py"]
)
def test_zonos2_job_selected_for_model_and_shared_runner_changes(tier, mirror, changed, monkeypatch):
    monkeypatch.setenv("MIRROR_HW", mirror)
    path = ROOT / f".buildkite/cuda/test-{tier}.yml"
    doc = yaml.safe_load(path.read_text())
    rendered = _render_test_pipeline(doc, [changed], pipeline_path=path)
    jobs = [step for step in rendered["steps"] if "ZONOS2" in step.get("label", "")]
    assert len(jobs) == 1
    expected = "b200-k8s" if mirror else ("l4-k8s" if tier == "ready" else "h100")
    assert expected in jobs[0]["agents"]["queue"]
    command = "\n".join(jobs[0]["commands"])
    assert "cards_1" in command
    if tier != "ready":
        assert command.index("prepare_zonos2_ci.py") < command.index("pytest -sv")
        assert "asset.env" in command
