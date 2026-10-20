# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selected text weights must be released before LingBot's video stages."""

from collections import OrderedDict
from contextlib import contextmanager

import pytest
import torch
from transformers import BatchEncoding
from transformers.modeling_outputs import BaseModelOutput

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.lingbot_world.pipeline import LingBotWorldCausalDMDPipeline
from vllm_omni.diffusion.offloader import sequential_backend

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("encode_fails", [False, True])
def test_selected_text_encoder_is_released_before_following_video_work(monkeypatch, encode_fails):
    events = []

    def tokenizer(*args, **kwargs):
        return BatchEncoding(
            {"input_ids": torch.arange(512).view(1, 512), "attention_mask": torch.ones(1, 512, dtype=torch.long)}
        )

    def encoder(*args, **kwargs):
        events.append("encode")
        if encode_fails:
            raise RuntimeError("injected encoding failure")
        return BaseModelOutput(last_hidden_state=torch.ones(1, 512, 8))

    # Exercise the real method without downloading or constructing a model.
    pipeline = object.__new__(LingBotWorldCausalDMDPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.od_config = OmniDiffusionConfig(
        diffusion_offload_config={"mode": "module", "components": ["text_encoder"], "pin_memory": False}
    )
    pipeline._prompt_embeds_cache = OrderedDict()
    pipeline.tokenizer = tokenizer
    pipeline.text_encoder = encoder

    @contextmanager
    def component_scope(component):
        assert component is pipeline.text_encoder
        events.append("load")
        try:
            yield
        finally:
            events.append("release")

    monkeypatch.setattr(sequential_backend, "sequential_offload_component", component_scope)

    def encode():
        return pipeline.encode_prompt("scene", max_sequence_length=512, dtype=torch.float32)

    if encode_fails:
        with pytest.raises(RuntimeError, match="injected encoding failure"):
            encode()
    else:
        result = encode()
        assert encode() is result
    assert events == ["load", "encode", "release"]
