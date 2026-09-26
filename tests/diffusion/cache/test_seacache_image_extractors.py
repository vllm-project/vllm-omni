# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.extractors import (
    extract_flux2_seacache_context,
    extract_flux_seacache_context,
    extract_qwen_seacache_context,
)
from vllm_omni.diffusion.cache.seacache.hook import apply_sea_cache_hook

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _StepMetadata:
    step: int = 0
    sigma: float = 0.8
    num_steps: int = 4


@dataclass
class _TensorParallelGroup:
    world_size: int = 1
    rank_in_group: int = 0


@pytest.fixture
def cpu_model_runtime(monkeypatch):
    """Use real tiny transformers and CPU attention without distributed workers."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import set_forward_context

    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", _TensorParallelGroup)
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.layer.get_attn_backend_for_role",
        lambda *args, **kwargs: (SDPABackend, None),
    )
    # Torch SDPA supports CPU; the platform dispatcher does not.
    monkeypatch.setattr(SDPAImpl, "forward", SDPAImpl.forward_cuda)
    od_config = OmniDiffusionConfig()
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
    ):
        yield od_config


def _image_ids(height: int, width: int, axes: int, time: int = 0) -> torch.Tensor:
    ids = torch.zeros(height * width, axes)
    ids[:, 0] = time
    ids[:, 1:3] = torch.cartesian_prod(torch.arange(height), torch.arange(width))
    return ids


@pytest.fixture(params=["flux", "flux2", "klein", "qwen", "qwen_edit"])
def image_model(request, cpu_model_runtime):
    torch.manual_seed(7)
    family = request.param
    common = dict(num_layers=1, num_attention_heads=2, attention_head_dim=16, joint_attention_dim=16, in_channels=8)
    if family == "flux":
        from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel

        module = FluxTransformer2DModel(
            **common, num_single_layers=1, pooled_projection_dim=8, axes_dims_rope=(4, 4, 8)
        )
        adapter = extract_flux_seacache_context
        inputs = {
            "img_ids": _image_ids(2, 3, 3),
            "txt_ids": torch.zeros(3, 3),
            "pooled_projections": torch.randn(1, 8),
            "guidance": torch.tensor([3.5]),
        }
        image_tokens = 6
    elif family in ("flux2", "klein"):
        if family == "flux2":
            from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Transformer2DModel
        else:
            from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import Flux2Transformer2DModel

        module = Flux2Transformer2DModel(**common, num_single_layers=1, axes_dims_rope=(4, 4, 4, 4))
        adapter = extract_flux2_seacache_context
        inputs = {
            "img_ids": torch.cat([_image_ids(2, 3, 4), _image_ids(1, 2, 4, time=10)]).unsqueeze(0),
            "txt_ids": torch.zeros(1, 3, 4),
            "guidance": torch.tensor([3.5]) if family == "flux2" else None,
        }
        image_tokens = 8
    else:
        from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel

        module = QwenImageTransformer2DModel(
            cpu_model_runtime,
            **common,
            out_channels=2,
            axes_dims_rope=(4, 6, 6),
            zero_cond_t=family == "qwen_edit",
        )
        adapter = extract_qwen_seacache_context
        shapes = [(1, 2, 3)] if family == "qwen" else [(1, 2, 3), (1, 1, 2)]
        inputs = {
            "img_shapes": [shapes],
            "txt_seq_lens": [3],
            "encoder_hidden_states_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        image_tokens = sum(frames * height * width for frames, height, width in shapes)

    # vLLM linear weights are allocated without initialization until load_weights.
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.05)
    # Keep rotary operations on CPU even when the host has an accelerator.
    from vllm_omni.diffusion.layers.rope import RotaryEmbedding

    for submodule in module.modules():
        if isinstance(submodule, RotaryEmbedding):
            submodule._forward_method = submodule.forward_native
    module.eval()
    inputs.update(
        hidden_states=torch.randn(1, image_tokens, 8),
        encoder_hidden_states=torch.randn(1, 3, 16),
        timestep=torch.tensor([0.8]),
        return_dict=False,
    )
    return module, adapter, _qwen_batch_inputs(inputs) if family == "qwen" else inputs


def test_modulated_indicator_matches_native_attention_and_preserves_full_execution(image_model, mocker):
    module, adapter, inputs = image_model
    attention = mocker.spy(module.transformer_blocks[0].attn, "forward")
    with torch.inference_mode():
        expected = module(**inputs)[0]
        native_feature = attention.call_args.kwargs["hidden_states"].clone()
        if adapter is extract_flux2_seacache_context:
            ctx = adapter(
                module,
                inputs["hidden_states"],
                inputs["encoder_hidden_states"],
                inputs["timestep"],
                inputs["img_ids"],
                inputs["txt_ids"],
                inputs["guidance"],
                None,
                False,
            )
        else:
            ctx = adapter(module, **inputs)
        assert ctx is not None
        ctx.validate()
        features = ctx.extra_states["sea_cache_latents"]
        assert len(features) == 1
        batch_size, image_tokens = inputs["hidden_states"].shape[:2]
        assert features[0].shape == (batch_size, module.inner_dim, 1, 2, 3)
        torch.testing.assert_close(
            features[0][:, :, 0].flatten(2).transpose(1, 2), native_feature[:, :6], rtol=0, atol=0
        )
        assert ctx.hidden_states.shape == (batch_size, image_tokens, module.inner_dim)
        execution_input = ctx.hidden_states.clone()
        full_output = ctx.run_transformer_blocks()[0]
        assert full_output.shape == ctx.hidden_states.shape
        torch.testing.assert_close(ctx.hidden_states, execution_input, rtol=0, atol=0)
        actual = ctx.postprocess(full_output)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _apply_image_hook(module, adapter, metadata, threshold=1e6):
    return apply_sea_cache_hook(
        module,
        SeaCacheConfig(threshold=threshold, power_exp=2.0, max_consecutive_cached=0),
        current_step_callback=lambda: metadata.step,
        current_sigma_callback=lambda: metadata.sigma,
        num_inference_steps_callback=lambda: metadata.num_steps,
        extractor_fn=adapter,
    )


@pytest.mark.parametrize("threshold, full_count, skip_count", [(0.0, 4, 0), (1e6, 2, 2)])
def test_image_cache_execution(image_model, threshold, full_count, skip_count, mocker):
    module, adapter, inputs = image_model
    metadata = _StepMetadata()
    step_inputs = [
        (sigma, {**inputs, "timestep": torch.full_like(inputs["timestep"], sigma)}) for sigma in [0.8, 0.6, 0.4, 0.2]
    ]
    with torch.inference_mode():
        expected = [module(**kwargs)[0] for _, kwargs in step_inputs] if threshold == 0 else None
        block = mocker.spy(module.transformer_blocks[0], "forward")
        hook = _apply_image_hook(module, adapter, metadata, threshold)
        with hook.cache_context("cond"):
            for step, (sigma, kwargs) in enumerate(step_inputs):
                metadata.step, metadata.sigma = step, sigma
                output = module(**kwargs)[0]
                assert output.shape == inputs["hidden_states"].shape
                assert torch.isfinite(output).all()
                if expected is not None:
                    torch.testing.assert_close(output, expected[step], rtol=0, atol=0)
            state = hook.state_manager.get_state()
            batch_size = inputs["hidden_states"].shape[0]
            assert len(state.previous_indicator) == batch_size
            assert all(residual.shape[0] == batch_size for _, residual in state.history)
    assert hook.full_count == block.call_count == full_count
    assert hook.skip_count == skip_count


@pytest.mark.parametrize("image_model", ["flux2"], indirect=True)
def test_missing_noisy_target_disables_the_indicator(image_model):
    module, adapter, inputs = image_model
    inputs["img_ids"][..., 0] = 10
    with torch.inference_mode():
        ctx = adapter(module, **inputs)
        assert ctx is not None
        assert ctx.extra_states["sea_cache_latents"] == []


def _qwen_batch_inputs(inputs, batch_size=2):
    return {
        **inputs,
        "hidden_states": inputs["hidden_states"].repeat(batch_size, 1, 1),
        "encoder_hidden_states": inputs["encoder_hidden_states"].repeat(batch_size, 1, 1),
        "encoder_hidden_states_mask": inputs["encoder_hidden_states_mask"].repeat(batch_size, 1),
        "img_shapes": inputs["img_shapes"] * batch_size,
        "txt_seq_lens": inputs["txt_seq_lens"] * batch_size,
        "timestep": inputs["timestep"].repeat(batch_size),
    }


@pytest.mark.parametrize("image_model", ["qwen_edit"], indirect=True)
@pytest.mark.parametrize("return_dict", [False, True])
def test_qwen_edit_batch_fallback_preserves_forward_and_clears_history(image_model, return_dict, mocker):
    module, adapter, inputs = image_model
    batched_inputs = {**_qwen_batch_inputs(inputs), "return_dict": return_dict}
    metadata = _StepMetadata(num_steps=5)
    resumed_inputs = {**inputs, "timestep": torch.tensor([0.4])}
    with torch.inference_mode():
        expected_batch = module(**batched_inputs)
        expected_resumed = module(**resumed_inputs)[0]
        original_forward = mocker.spy(module, "forward")
        hook = _apply_image_hook(module, adapter, metadata)
        # Populate both CFG branches; an unsupported call must invalidate both.
        for context in ("cond", "uncond"):
            with hook.cache_context(context):
                module(**inputs)
        assert set(hook.state_manager._states) == {"cond", "uncond"}
        assert hook.full_count == 2

        metadata.step, metadata.sigma = 1, 0.6
        positional_hidden = batched_inputs.pop("hidden_states")
        lookup = mocker.patch(
            "vllm_omni.diffusion.cache.seacache.extractors.get_extractor",
            side_effect=AssertionError("Unsupported batch must bypass shared extraction"),
        )
        with hook.cache_context("cond"):
            actual_batch = module(positional_hidden, **batched_inputs)
            assert hook.state_manager._states == {}
        mocker.stop(lookup)
        original_forward.assert_called_once_with(positional_hidden, **batched_inputs)
        assert type(actual_batch) is type(expected_batch)
        torch.testing.assert_close(actual_batch[0], expected_batch[0], rtol=0, atol=0)
        assert hook.full_count == 2
        assert hook.skip_count == 0

        # A supported batch resumes with a full pass, even at a cacheable step.
        metadata.step, metadata.sigma = 2, 0.4
        with hook.cache_context("cond"):
            resumed = module(**resumed_inputs)[0]
            torch.testing.assert_close(resumed, expected_resumed, rtol=0, atol=0)
        assert hook.full_count == 3
        assert hook.skip_count == 0
        original_forward.assert_called_once()
