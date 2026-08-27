# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json

import pytest

from vllm_omni.benchmarks.duplex.omni_duplex_eval_clock import normalize_response_items, split_text, validate_clock
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import (
    canonical_task_type,
    family_for_split,
    load_samples,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_dataset_mapping_and_manifest(tmp_path):
    manifest = tmp_path / "samples.json"
    manifest.write_text(json.dumps([{"id": "a", "split": "RTD_OCR", "video": "clip.mp4"}]), encoding="utf-8")
    sample = load_samples(manifest, media_root=tmp_path)[0]
    assert sample.family == "rtd"
    assert sample.video == str(tmp_path / "clip.mp4")
    assert family_for_split("PR_event_reminder") == "pr"
    assert canonical_task_type("post-event-reminder") == "post_event_reminder"


def test_response_aliases_and_clock_guard():
    assert split_text("One. Two!") == ["One.", "Two!"]
    assert normalize_response_items({"chunks": [{"text": "x", "current_time": 1}]}) == [
        {"sentence": "x", "start": 1.0, "end": 1.0}
    ]
    with pytest.raises(ValueError, match="clock=invalid"):
        validate_clock({"clock": "invalid"})
    validate_clock({"clock": "invalid"}, allow_invalid=True)
