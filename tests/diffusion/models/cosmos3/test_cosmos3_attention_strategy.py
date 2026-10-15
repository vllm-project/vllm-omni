# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-forward capture, compilation and replay with CPU or real CUDA providers."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.attention_strategy import (
    check_strategy_capture_and_reuse,
    configure_strategy_test,
    move_strategy_inputs,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.parametrize("model_name", ["cosmos3", "cosmos3_sound"])
@pytest.mark.parametrize(
    ("device", "compiler_backend"),
    [
        pytest.param("cpu", "eager", marks=pytest.mark.cpu),
        pytest.param("cpu", "inductor", marks=pytest.mark.cpu),
        pytest.param("cuda", "inductor", marks=pytest.mark.cuda),
    ],
)
@torch.inference_mode()
def test_cosmos3_forward_capture_and_reuse(monkeypatch, model_name, device, compiler_backend):
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3 as cosmos

    common, steps = configure_strategy_test(monkeypatch, device, "cosmos3.gen")
    monkeypatch.setattr(cosmos, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(cosmos, "_get_ulysses_state", lambda: (1, 0, None))
    with set_current_diffusion_config(SimpleNamespace(**common)):
        from tests.diffusion.models.cosmos3.test_cosmos3_transformer import _tiny_cosmos3_config

        model_config = _tiny_cosmos3_config(num_hidden_layers=2)
        if device == "cuda":
            model_config.update(
                hidden_size=256, head_dim=128, intermediate_size=512, rope_scaling={"mrope_section": [24, 20, 20]}
            )
        cfg = SimpleNamespace(**common, tf_model_config=model_config)
        model = cosmos.Cosmos3VFMTransformer(
            cfg,
            **({"sound_gen": True, "sound_dim": 3, "sound_latent_fps": 24.0} if model_name == "cosmos3_sound" else {}),
        )
        inputs = dict(
            hidden_states=torch.randn(1, 2, 1, 16, 16) if device == "cuda" else torch.randn(1, 2, 1, 2, 2),
            timestep=torch.ones(1),
            text_ids=torch.zeros(1, 3, dtype=torch.long),
            text_mask=torch.ones(1, 3, dtype=torch.long),
            video_shape=(1, 16, 16) if device == "cuda" else (1, 2, 2),
        )
    if model_name == "cosmos3_sound":
        inputs["sound_latents"] = torch.randn(1, 3, 4)
    inputs = move_strategy_inputs(inputs, device, common["dtype"])
    model.to(device=device)
    if device == "cuda":
        model.to(dtype=common["dtype"])
        model.post_load_weights()
    model.eval()
    for p in model.parameters():
        p.fill_(1) if p.ndim == 1 else p.normal_(std=0.02)
    tolerances = {"atol": 0.005, "rtol": 0.005} if device == "cuda" else {}
    if model_name == "cosmos3_sound" and device == "cuda":
        # Two BF16 rounding units between eager and Inductor, also reproduced
        # before the host-packing cleanup at the same elements.
        tolerances["rtol"] = 0.016
    graphs, compiled_count = check_strategy_capture_and_reuse(
        model, inputs, device, compiler_backend, steps, tolerances
    )
    # Cosmos3's conditioning tensors must be inputs, not captured module caches.
    model.reset_cache()
    changed = {**inputs, "text_ids": torch.ones_like(inputs["text_ids"])}
    args, prepared = model.prepare_attention_strategy_inputs((), changed)
    expected = model.forward_with_attention_layout(*args, attention_layout=1, **prepared)
    with override_forward_context(ForwardContext(denoise_step_idx=1, total_denoise_steps=len(steps))):
        torch.testing.assert_close(model(**changed), expected, **tolerances)
    assert len(graphs) == compiled_count
    torch._dynamo.reset()
