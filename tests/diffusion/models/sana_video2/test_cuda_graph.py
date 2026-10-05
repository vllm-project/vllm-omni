# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.sana_video2.transformer_sana_video2 import (
    SanaVideo2TransformerConfig,
    SanaVideo2TransformerModel,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA"),
]


def tiny_model(dtype, in_channels=4):
    torch.manual_seed(42)
    config = SanaVideo2TransformerConfig(
        in_channels=in_channels,
        hidden_size=48,
        depth=3,
        num_heads=4,
        caption_channels=12,
        linear_head_dim=12,
        softmax_head_dim=12,
        attn_res_block_size=2,
        mlp_ratio=2,
    )
    model = SanaVideo2TransformerModel(config).to(device="cuda", dtype=dtype).eval()
    # Learned depth projections must exercise the aggregation, not just averaging.
    with torch.no_grad():
        for projection in (model.attn_res.attn_proj, model.attn_res.mlp_proj, model.attn_res.final_proj):
            projection.weight.normal_(std=0.1)
    return model


def inputs_for(model, batch=1, ti2v=False, frames=2, seed=1):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return dict(
        hidden_states=torch.randn(batch, 4, frames, 2, 3, device="cuda", generator=generator),
        timestep=(
            torch.full((batch, 1, frames, 1, 1), 900.0, device="cuda")
            if ti2v
            else torch.full((batch,), 900.0, device="cuda")
        ),
        encoder_hidden_states=torch.randn(batch, 300, 12, device="cuda", dtype=model.dtype, generator=generator),
        encoder_attention_mask=torch.arange(300, device="cuda")[None].expand(batch, -1) < 173,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,ti2v", [(1, False), (2, False), (1, True), (2, True)])
def test_capture_replays_fresh_inputs_without_overwriting_previous_output(dtype, batch, ti2v):
    from vllm_omni.diffusion.models.sana_video2.cuda_graph import SanaVideo2CudaGraphRunner

    model = tiny_model(dtype)
    first = inputs_for(model, batch, ti2v)
    second = inputs_for(model, batch, ti2v, seed=2)
    second["timestep"].fill_(300)
    if ti2v:
        first["timestep"][:, :, 0] = 0
        second["timestep"][:, :, 0] = 0
    second["encoder_attention_mask"] = ~second["encoder_attention_mask"]
    expected = [model(**kwargs) for kwargs in (first, second)]
    runner = SanaVideo2CudaGraphRunner(model)
    runner.capture(**first)
    # This hook sees real Python execution; replay should bypass it.
    calls = []
    handle = model.x_embedder.register_forward_hook(lambda *args: calls.append(1))
    try:
        result = runner(**first)
        following = runner(**second)
        again = runner(**first)
    finally:
        handle.remove()
    assert calls == []
    for actual, reference in ((result, expected[0]), (following, expected[1]), (again, expected[0])):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("changed_input", ["shape", "dtype"])
def test_uncaptured_input_runs_eager_and_capture_is_idempotent(changed_input):
    from vllm_omni.diffusion.models.sana_video2.cuda_graph import SanaVideo2CudaGraphRunner

    model = tiny_model(torch.float32)
    first = inputs_for(model)
    other = inputs_for(model, frames=3 if changed_input == "shape" else 2)
    if changed_input == "dtype":
        other["encoder_hidden_states"] = other["encoder_hidden_states"].bfloat16()
    runner = SanaVideo2CudaGraphRunner(model)
    runner.capture(**first)
    calls = []
    handle = model.x_embedder.register_forward_hook(lambda *args: calls.append(1))
    try:
        runner.capture(**first)
        assert calls == []
        expected = model(**other)
        calls.clear()
        assert torch.equal(runner(**other), expected)
        assert calls == [1]
        assert len(runner.entries) == 1
        calls.clear()
        runner(**first)
        assert calls == []
    finally:
        handle.remove()


def test_replay_rejects_new_all_masked_prompt():
    from vllm_omni.diffusion.models.sana_video2.cuda_graph import SanaVideo2CudaGraphRunner

    model = tiny_model(torch.float32)
    kwargs = inputs_for(model)
    runner = SanaVideo2CudaGraphRunner(model)
    runner.capture(**kwargs)
    kwargs["encoder_attention_mask"].fill_(False)
    with pytest.raises(ValueError, match="unmasked"):
        runner(**kwargs)


def test_multiple_captured_shapes_and_optional_mask_keep_independent_state():
    from vllm_omni.diffusion.models.sana_video2.cuda_graph import SanaVideo2CudaGraphRunner

    model = tiny_model(torch.bfloat16)
    first = inputs_for(model)
    other = inputs_for(model, batch=2, ti2v=True, frames=3)
    other["encoder_attention_mask"] = None
    expected = [model(**kwargs) for kwargs in (first, other)]
    runner = SanaVideo2CudaGraphRunner(model)
    runner.capture(**first)
    runner.capture(**other)
    calls = []
    handle = model.x_embedder.register_forward_hook(lambda *args: calls.append(1))
    try:
        outputs = [runner(**kwargs) for kwargs in (first, other, first)]
    finally:
        handle.remove()
    assert calls == []
    for output, reference in zip(outputs, (expected[0], expected[1], expected[0]), strict=True):
        assert torch.equal(output, reference)
