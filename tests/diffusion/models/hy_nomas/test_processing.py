# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from vllm_omni.diffusion.models.hy_nomas.processing_hy_nomas import action_conditioning

pytestmark = [pytest.mark.local_model, pytest.mark.cpu, pytest.mark.diffusion]


class TokenizerStub:
    prompt_template_video = {"crop_start": 3}

    def text2tokens(self, prompts, *, max_length, **kwargs):
        ids = [list(text.encode())[:max_length] for text in prompts]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(1, len(ids[0]))}


@pytest.mark.parametrize("structured", ["caption", "empty", "list", "mapping", "single", "overlap", "missing_time"])
def test_action_masks_match_reference_helpers(structured):
    reference = pytest.importorskip("hymm.pipeline.base")
    model = reference.BasePipeline.__new__(reference.BasePipeline)
    nn.Module.__init__(model)
    model.args = OmegaConf.create({"text_len": 1280, "inference": {"video_fps": 24}})
    model.text_encoder = TokenizerStub()
    model.t_ratio = 4
    actions: list[dict[str, str | float]] = [
        {"start_time": "00:00.5", "end_time": "00:01.0", "action_description": "walk"},
        {"start_time": "00:01.0", "end_time": "00:01.5", "action_description": "turn"},
    ]
    if structured == "single":
        actions = [{"start_time": 0, "end_time": 10, "action_description": "walk"}]
    elif structured == "overlap":
        actions[1]["start_time"] = 0.5
    elif structured == "missing_time":
        del actions[0]["start_time"]
    contained_actions = (
        {f"Action{index}": action for index, action in enumerate(actions)} if structured == "mapping" else actions
    )
    prompt = json.dumps({"segments": [{"description": "A forest", "contained_actions": contained_actions}]})
    if structured == "caption":
        prompt = "A forest beside a river."
    elif structured == "empty":
        prompt = json.dumps({"segments": []})
    if structured == "overlap":
        with pytest.raises(ValueError, match="Overlapping"):
            model._compute_action_video_windows(prompt, 0, 12)
        with pytest.raises(ValueError, match="overlap"):
            action_conditioning(prompt, model.text_encoder, 1280, 12)
        return
    start = model._compute_action_start(prompt)
    offsets = model._compute_action_text_offsets(prompt)
    windows = model._compute_action_video_windows(prompt, 0, 12)
    expected = (start, offsets, windows) if offsets is not None and windows is not None else (start, None, None)
    assert action_conditioning(prompt, model.text_encoder, 1280, 12) == expected
