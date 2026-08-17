# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.diffusion_disagg import diffusion_stage_handoff

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_denoise_to_decode_handoff_forwards_only_current_payload():
    latents = torch.ones(1, 4, 2, 2, 2)
    handle = {
        "key": "req:stage_payload",
        "from_stage": "1",
        "to_stage": "2",
        "metadata": {"edge": "latent"},
        "payload_keys": ["latents"],
    }
    source_output = SimpleNamespace(
        _custom_output={
            "latents": latents,
            "_stage_payload_transfer": handle,
        }
    )
    prompt = {
        "prompt": "a prompt",
        "prompt_embeds": torch.zeros(1, 2, 3),
        "negative_prompt_embeds": torch.zeros(1, 2, 3),
        "height": 384,
        "width": 384,
        "seed": 123,
    }

    result = diffusion_stage_handoff([source_output], [prompt])[0]

    assert result["prompt"] == "a prompt"
    assert result["height"] == 384
    assert result["width"] == 384
    assert result["seed"] == 123
    torch.testing.assert_close(result["latents"], latents)
    assert result["_stage_payload_transfer"] == handle
    assert "prompt_embeds" not in result
    assert "negative_prompt_embeds" not in result
