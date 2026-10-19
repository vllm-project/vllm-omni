# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for Breeze's decode-only embedding preprocessing."""

import pytest
import torch

from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze import BreezeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _embedding_talker(dtype):
    model = BreezeForConditionalGeneration.__new__(BreezeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.num_codebooks, model.hidden_size = 16, 8
    model.register_buffer("offsets", torch.arange(model.num_codebooks) * 7)
    model.depth_decoder = torch.nn.Module()
    model.depth_decoder.embed_tokens = torch.nn.Embedding(16 * 7, 8, dtype=dtype)
    with torch.no_grad():
        model.depth_decoder.embed_tokens.weight.copy_(
            torch.randn(16 * 7, 8, generator=torch.Generator().manual_seed(17))
        )
    return model


def _decode_info(request_id, current, seed):
    return {
        "global_request_id": [request_id],
        "breeze_prompt": {"role": "cond", "guidance_scale": 1.0},
        "breeze_sampling": {"temperature": 0.9, "top_k": 4, "top_p": 1.0, "repetition_penalty": 1.1},
        "breeze_state": {
            "current": current,
            "history": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "generator": torch.Generator().manual_seed(seed),
        },
        "_omni_is_prefill": False,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_decode_batch_matches_scalar_after_reordering_shrinking_and_cfg(dtype):
    model = _embedding_talker(dtype)
    shared_frame = torch.arange(16, dtype=torch.long).reshape(1, 16) % 7
    infos = [
        _decode_info("unconditional", shared_frame, 1),
        _decode_info("other", (shared_frame + 1) % 7, 2),
        _decode_info("conditioned", shared_frame, 3),
        _decode_info("last", (shared_frame + 3) % 7, 4),
    ]
    infos[0]["breeze_prompt"] = {"role": "uncond", "guidance_scale": 4.0}
    infos[2]["breeze_prompt"]["guidance_scale"] = 4.0
    saved = []
    for indices in ([0, 1, 2, 3], [2, 0, 1], [1]):
        selected = [infos[index] for index in indices]
        # Placeholder IDs deliberately differ from the RVQ frame IDs.
        input_ids = torch.arange(len(selected), dtype=torch.long) + 100
        expected = torch.cat(
            [
                model.preprocess(input_ids=input_ids[row : row + 1], input_embeds=None, **info)[1]
                for row, info in enumerate(selected)
            ]
        )
        state_before = [
            (
                info["breeze_state"]["current"].clone(),
                info["breeze_state"]["history"].clone(),
                info["breeze_state"]["generator"].get_state(),
            )
            for info in infos
        ]
        output_ids, embeds, updates = model.preprocess_decode_batch(input_ids=input_ids, req_infos=selected)
        assert output_ids is input_ids
        torch.testing.assert_close(embeds, expected, rtol=0, atol=0)
        assert updates == [{} for _ in selected]
        for info, (current, history, rng) in zip(infos, state_before, strict=True):
            state = info["breeze_state"]
            assert torch.equal(state["current"], current)
            assert torch.equal(state["history"], history)
            assert torch.equal(state["generator"].get_state(), rng)
        saved.append((embeds, embeds.clone()))
        # Simulate the next generated frame. CFG branches share the new frame.
        next_frame = (infos[2]["breeze_state"]["current"] + 2) % 7
        infos[0]["breeze_state"]["current"] = next_frame
        infos[2]["breeze_state"]["current"] = next_frame
        infos[1]["breeze_state"]["current"] = (infos[1]["breeze_state"]["current"] + 1) % 7
    assert infos[0]["breeze_state"]["current"] is infos[2]["breeze_state"]["current"]
    for output, snapshot in saved:
        assert torch.equal(output, snapshot)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_batch_empty_preserves_embedding_shape_and_dtype(dtype):
    model = _embedding_talker(dtype)
    ids = torch.empty(0, dtype=torch.long)
    output_ids, embeds, updates = model.preprocess_decode_batch(input_ids=ids, req_infos=[])
    assert output_ids is ids
    assert embeds.shape == (0, 8)
    assert embeds.dtype == dtype
    assert embeds.device == model.depth_decoder.embed_tokens.weight.device
    assert updates == []


@pytest.mark.parametrize("invalid", ["count", "state", "frame_shape", "prefill"])
def test_decode_batch_rejects_invalid_request_state(invalid):
    model = _embedding_talker(torch.float32)
    info = _decode_info("request", torch.zeros(1, 16, dtype=torch.long), 42)
    ids = torch.tensor([0])
    if invalid == "count":
        ids = torch.tensor([0, 1])
    elif invalid == "state":
        del info["breeze_state"]
    elif invalid == "frame_shape":
        info["breeze_state"]["current"] = torch.zeros(2, 16, dtype=torch.long)
    else:
        info["_omni_is_prefill"] = True
    with pytest.raises(ValueError, match="Breeze decode"):
        model.preprocess_decode_batch(input_ids=ids, req_infos=[info])
