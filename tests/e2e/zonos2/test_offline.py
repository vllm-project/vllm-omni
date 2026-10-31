# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P5-01/03 real-weight offline audio and four concurrent request acceptance."""

import json
import os
import uuid
from pathlib import Path

import pytest

from tests.e2e.zonos2.process import run_offline
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.tts]


def output_path(tmp_path, name):
    root = Path(os.environ.get("ZONOS2_TEST_OUTPUT_DIR", str(tmp_path)))
    return root / f"{name}-{uuid.uuid4().hex[:8]}"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
def test_real_offline_audio(tmp_path):
    directory = output_path(tmp_path, "offline")
    run_offline(directory, concurrent=False)
    report = json.loads((directory / "summary.json").read_text())
    assert report["real_weights"] and report["cases"][0]["sample_rate"] == 44100


@pytest.mark.full_model
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
def test_real_concurrent_audio_isolation(tmp_path):
    directory = output_path(tmp_path, "concurrent")
    run_offline(directory, concurrent=True)
    report = json.loads((directory / "summary.json").read_text())
    assert report["concurrency"] == 4
    assert len(report["cases"]) == 12
    assert len({case["lifecycle"]["seed"] for case in report["cases"]}) == 4
