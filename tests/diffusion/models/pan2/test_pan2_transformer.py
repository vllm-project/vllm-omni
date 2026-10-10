# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the native PAN2 transformer: diffusers parity, weight loading and the SP plan."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_NUM_HEADS = 2
_HEAD_DIM = 16
_TINY_CONFIG = {
    "patch_size": (1, 2, 2),
    "in_channels": 9,
    "out_channels": 4,
    "num_attention_heads": _NUM_HEADS,
    "attention_head_dim": _HEAD_DIM,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "mlp_ratio": 2.0,
    "context_mlp_ratio": 1.0,
    "refiner_mlp_ratio": 1.0,
    "text_embed_dim": 16,
    "rope_axes_dim": (4, 6, 6),
}


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    """The native transformer uses vLLM parallel linear layers and the vLLM-Omni attention layer, which need a
    single-process tensor-parallel group, a vLLM config and a diffusion forward context."""
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29511")
    init_distributed_environment()
    initialize_model_parallel()
    od_config = OmniDiffusionConfig.from_kwargs(
        model="test",
        dtype=torch.float32,
        parallel_config=DiffusionParallelConfig(),
        diffusion_attention_backend="TORCH_SDPA",
    )
    with (
        # Run every vLLM custom op through its PyTorch-native path so the model stays on CPU.
        set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["none"]))),
        set_forward_context(omni_diffusion_config=od_config),
        set_current_diffusion_config(od_config),
    ):
        yield od_config
    destroy_distributed_env()


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    return {
        "hidden_states": torch.randn(1, 9, 2, 8, 8, generator=generator),
        "timestep": torch.tensor([500.0]),
        "encoder_hidden_states": torch.randn(1, 7, 16, generator=generator),
    }


def _model(od_config):
    from vllm_omni.diffusion.layers.custom_op import CustomOp
    from vllm_omni.diffusion.models.pan2 import PAN2Transformer3DModel

    model = PAN2Transformer3DModel(od_config=od_config, **_TINY_CONFIG).eval()
    # vLLM-Omni custom ops bind their accelerator kernel at construction; run them natively on CPU.
    for module in model.modules():
        if isinstance(module, CustomOp):
            module._forward_method = module.forward_native
    return model


def test_tiny_transformer_matches_diffusers(_runtime):
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "PAN2Transformer3DModel"):
        pytest.skip("The installed diffusers does not ship PAN2Transformer3DModel yet.")

    torch.manual_seed(7)
    reference = diffusers.PAN2Transformer3DModel(**_TINY_CONFIG).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.2)

    model = _model(_runtime)
    loaded = model.load_weights(reference.state_dict().items())
    assert loaded == set(dict(model.named_parameters()))

    with torch.no_grad():
        expected = reference(**_inputs(), return_dict=False)[0]
        actual = model(**_inputs())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_load_weights_fuses_qkv_projections(_runtime):
    model = _model(_runtime)
    block = model.transformer_blocks[0]
    inner_dim = _NUM_HEADS * _HEAD_DIM
    q, k, v = (torch.full((inner_dim, inner_dim), float(i)) for i in range(3))

    loaded = model.load_weights(
        [
            ("transformer_blocks.0.attn.to_q.weight", q),
            ("transformer_blocks.0.attn.to_k.weight", k),
            ("transformer_blocks.0.attn.to_v.weight", v),
            ("context_refiner.0.attn.to_q.weight", q),
        ]
    )

    assert loaded == {"transformer_blocks.0.attn.to_qkv.weight", "context_refiner.0.attn.to_q.weight"}
    torch.testing.assert_close(block.attn.to_qkv.weight, torch.cat([q, k, v]))
    torch.testing.assert_close(model.context_refiner[0].attn.to_q.weight, q)


def test_sp_plan_targets_existing_modules(_runtime):
    model = _model(_runtime)
    module_names = {name for name, _ in model.named_modules()}
    assert set(model._sp_plan) == {"rope", "proj_out"}
    assert set(model._sp_plan) <= module_names


def test_metadata_is_not_attention_mask_free():
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

    # With parallel_config.mask_sp_padding the transformer masks the SP padding, so it must not default to the
    # mask-free TRTLLM_ATTN backend.
    assert get_diffusion_model_metadata("PAN2ModularPipeline").attention_mask_free is False


def test_output_shape_and_unpatchify(_runtime):
    model = _model(_runtime)
    torch.manual_seed(3)
    with torch.no_grad():
        # vLLM parallel linear layers allocate their weights uninitialized.
        for parameter in model.parameters():
            parameter.normal_(std=0.2)
        output = model(**_inputs())
    assert output.shape == (1, _TINY_CONFIG["out_channels"], 2, 8, 8)
    assert torch.isfinite(output).all()


def _randomized_model(od_config, seed: int):
    model = _model(od_config)
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.2)
    return model


def test_fused_video_qk_norm_rope_matches_eager_path(_runtime, monkeypatch):
    from vllm_omni.diffusion.models.pan2 import pan2_transformer

    model = _randomized_model(_runtime, seed=5)
    attention = model.transformer_blocks[0].attn
    cos, sin = model.rope(_inputs()["hidden_states"])
    generator = torch.Generator().manual_seed(9)
    query = torch.randn(1, cos.shape[0], _NUM_HEADS, _HEAD_DIM, generator=generator)
    key = torch.randn(1, cos.shape[0], _NUM_HEADS, _HEAD_DIM, generator=generator)

    expected = attention._video_qk_norm_rope(query, key, (cos, sin, None))
    # Route through the fused op's reference implementation to check PAN2's packed cos | sin table and pairing.
    monkeypatch.setattr(pan2_transformer, "_fused_cuda_supported", lambda *_args, **_kwargs: True)
    actual = attention._video_qk_norm_rope(query, key, (cos, sin, torch.cat((cos, sin), dim=-1)))

    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_batched_samples_match_separate_forwards(_runtime):
    """Samples in one batch, each with its own timestep, must not attend to each other."""
    model = _randomized_model(_runtime, seed=13)
    generator = torch.Generator().manual_seed(17)
    hidden_states = torch.randn(2, 9, 2, 8, 8, generator=generator)
    timestep = torch.tensor([500.0, 250.0])
    encoder_hidden_states = torch.randn(2, 7, 16, generator=generator)

    with torch.no_grad():
        batched = model(hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
        separate = torch.cat(
            [
                model(
                    hidden_states=hidden_states[i : i + 1],
                    timestep=timestep[i : i + 1],
                    encoder_hidden_states=encoder_hidden_states[i : i + 1],
                )
                for i in range(2)
            ]
        )
    torch.testing.assert_close(batched, separate, rtol=1e-5, atol=1e-5)
