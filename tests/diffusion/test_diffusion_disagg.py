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


@pytest.mark.parametrize("transferred", [False, True])
def test_wan_conditioning_handoff_preserves_current_payload_and_controls(transferred):
    payload = {
        "prompt_embeds": torch.zeros(2, 3),
        "negative_prompt_embeds": torch.ones(2, 3),
        "wan_image_condition": torch.zeros(1, 4, 1, 2, 4),
        "wan_conditioning_metadata": {"version": 1, "has_image": True, "height": 16, "width": 32},
    }
    handle = {"key": "req-0_0_0", "payload_keys": list(payload)}
    custom = {"_stage_payload_transfer": handle} if transferred else payload
    controls = {"seed": 123, "output_type": "latent", "num_outputs_per_prompt": 2, "max_sequence_length": 256}
    prompt = {"prompt": "a cat", "multi_modal_data": {"image": object()}, **controls}
    result = diffusion_stage_handoff([SimpleNamespace(custom_output=custom)], [prompt])[0]

    for key, value in controls.items():
        assert result[key] == value
    assert "multi_modal_data" not in result
    if transferred:
        assert result["_stage_payload_transfer"] is handle
        assert not set(payload).intersection(result)
    else:
        for key, value in payload.items():
            assert result[key] is value

    # The next edge carries G's result, never E's stale conditioning/handles.
    decoded_prompt = diffusion_stage_handoff(
        [SimpleNamespace(custom_output={"latents": torch.zeros(2, 4, 2, 2, 4)})], [result]
    )[0]
    assert not set(payload).intersection(decoded_prompt)
    assert "_stage_payload_transfer" not in decoded_prompt
    assert decoded_prompt["output_type"] == "latent"
