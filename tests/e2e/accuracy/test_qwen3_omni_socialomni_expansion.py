# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run the SocialOmni mini set against a real Qwen3-Omni thinker server.

Set VLLM_SOCIALOMNI_DATASET_ROOT to the extracted dataset to opt in. The model
defaults to Qwen/Qwen3-Omni-30B-A3B-Instruct; VLLM_SOCIALOMNI_MODEL can select a
local checkpoint. Requires two H100-class GPUs and the benchmark dependencies.
External judges are intentionally not configured, so quality stays incomplete.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

DATASET_ROOT = os.environ.get("VLLM_SOCIALOMNI_DATASET_ROOT")
pytestmark = [
    pytest.mark.full_model,
    pytest.mark.omni,
    pytest.mark.benchmark,
    pytest.mark.skipif(not DATASET_ROOT, reason="Set VLLM_SOCIALOMNI_DATASET_ROOT to run SocialOmni"),
]

SERVER_PARAMS = OmniServerParams(
    model=os.environ.get("VLLM_SOCIALOMNI_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
    stage_config_path=get_deploy_config_path("qwen3_omni_moe_thinking.yaml"),
    server_args=["--max-model-len", "65536"],
)


@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", [SERVER_PARAMS], indirect=True)
def test_socialomni_mini_without_judges(omni_server, tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.socialomni.evaluate",
            "--dataset-root",
            str(DATASET_ROOT),
            "--model",
            omni_server.model,
            "--base-url",
            f"http://{omni_server.host}:{omni_server.port}",
            "--level",
            "both",
            "--mini",
            "--warmup",
            "0",
            "--prefix-cache-dir",
            str(tmp_path / "prefixes"),
            "--output-dir",
            str(tmp_path / "results"),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    paths = list((tmp_path / "results").glob("socialomni-*.json"))
    assert len(paths) == 1, result.stdout + result.stderr
    output = json.loads(paths[0].read_text())
    assert output["failures"] == []
    level1 = output["per_sample"]["level1"]
    level2 = output["per_sample"]["level2"]
    assert len(level1) == len(level2) == 2
    assert all(record["predicted_answer"] in {"A", "B", "C", "D"} for record in level1)
    assert {record["gold_when"] for record in level2} == {"YES", "NO"}
    assert all(record["predicted_when"] in {"YES", "NO"} for record in level2)
    response = next(record for record in level2 if record["gold_when"] == "YES")
    assert response["gold_response_success"]
    assert response["gold_response"].strip()
    summary = output["summary"]
    assert summary["level1"]["speed"]["successful_requests"] == 2
    assert summary["level2"]["speed"]["model"]["successful_requests"] == 3
    assert summary["status"] == "incomplete"
    assert summary["level2"]["metrics"]["quality"] is None
    assert summary["level2"]["metrics"]["judge_status"] == {
        "complete": False,
        "eligible_responses": 1,
        "completed_scores": 0,
        "required_scores": 3,
    }
