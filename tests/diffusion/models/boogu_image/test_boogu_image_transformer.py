# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import logging
import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# Small mock dimensions (head_dim = HIDDEN_SIZE / NUM_HEADS = 16 must equal
# sum(AXES_DIM_ROPE)).
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
AXES_DIM_ROPE = (8, 4, 4)
AXES_LENS = (32, 16, 16)
MULTIPLE_OF = 32
NORM_EPS = 1e-5


@pytest.fixture(autouse=True)
def _init_distributed(request):
    """Initialize the minimal single-rank distributed environment required by
    the vLLM parallel linear layers (tensor-parallel group must exist)."""
    if request.node.name.startswith("test_real_rotary_"):
        yield
        return

    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29512")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _force_default_gemm(monkeypatch):
    """Force CPU-compatible GEMM dispatch for tests using CPU tensors.

    vLLM's dispatch_unquantized_gemm() selects the backend by platform, not by
    tensor device; CPU test tensors can crash on non-default backends."""
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


@pytest.fixture(autouse=True)
def _force_torch_sdpa():
    """Pin TORCH_SDPA so CPU shape tests do not pick CUDA-only backends (FA3)."""
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig

    od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    with set_current_diffusion_config(od_config):
        yield


def _randomize_parameters(module: torch.nn.Module) -> None:
    """Fill parameters with small random values.

    vLLM parallel linears allocate weights with `torch.empty` (real weights
    arrive via `load_weights`), so uninitialized memory must be overwritten
    before a forward pass."""
    with torch.no_grad():
        for param in module.parameters():
            param.uniform_(-0.02, 0.02)


def _identity_rotary_emb(batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-valued rotary frequencies encoding a zero rotation."""
    return (
        torch.ones(batch_size, seq_len, HEAD_DIM),
        torch.zeros(batch_size, seq_len, HEAD_DIM),
    )


def _tiny_tf_model_config(**overrides):
    config = {
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": 4,
        "num_double_stream_layers": 2,
        "num_refiner_layers": 2,
        "num_attention_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "multiple_of": MULTIPLE_OF,
        "norm_eps": NORM_EPS,
        "axes_dim_rope": list(AXES_DIM_ROPE),
        "axes_lens": list(AXES_LENS),
        "instruction_feature_configs": {
            "instruction_feat_dim": 32,
            "reduce_type": "mean",
            "num_instruction_feature_layers": 1,
        },
        "prompt_tuning_configs": {"use_prompt_tuning": False},
        "timestep_scale": 1.0,
    }
    config.update(overrides)
    return config


def _tiny_od_config(**overrides):
    from vllm_omni.diffusion.data import TransformerConfig

    return SimpleNamespace(
        tf_model_config=TransformerConfig.from_dict(_tiny_tf_model_config(**overrides)),
        dtype=torch.float32,
    )


def test_boogu_image_transformer_import():
    from vllm_omni.diffusion.models.boogu_image import BooguImageTransformer2DModel

    assert BooguImageTransformer2DModel is not None


def test_real_rotary_emb_matches_complex_reference():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import apply_rotary_emb

    torch.manual_seed(0)
    x = torch.randn(2, 7, NUM_HEADS, HEAD_DIM)
    angles = torch.randn(2, 7, HEAD_DIM // 2)
    rotary_emb = (
        angles.cos().repeat_interleave(2, dim=-1),
        angles.sin().repeat_interleave(2, dim=-1),
    )

    actual = apply_rotary_emb(x, rotary_emb)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], HEAD_DIM // 2, 2))
    freqs_cis = torch.polar(torch.ones_like(angles), angles).unsqueeze(2)
    expected = torch.view_as_real(x_complex * freqs_cis).flatten(3).type_as(x)

    torch.testing.assert_close(actual, expected)


def test_real_rotary_frequency_tables_are_real_and_repeated():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageDoubleStreamRotaryPosEmbed,
    )

    freqs_real = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_real(AXES_DIM_ROPE, AXES_LENS, theta=10000)

    assert len(freqs_real) == len(AXES_DIM_ROPE)
    for (freqs_cos, freqs_sin), axis_dim, axis_len in zip(freqs_real, AXES_DIM_ROPE, AXES_LENS):
        assert freqs_cos.shape == (axis_len, axis_dim)
        assert freqs_sin.shape == (axis_len, axis_dim)
        assert not freqs_cos.is_complex()
        assert not freqs_sin.is_complex()
        torch.testing.assert_close(freqs_cos[..., ::2], freqs_cos[..., 1::2])
        torch.testing.assert_close(freqs_sin[..., ::2], freqs_sin[..., 1::2])


def test_real_rotary_frequency_tables_use_float32_on_npu(monkeypatch):
    from vllm_omni.diffusion.models.boogu_image import boogu_image_transformer

    npu_platform = SimpleNamespace(is_npu=lambda: True, supports_float64=lambda: True)
    monkeypatch.setattr(boogu_image_transformer, "current_omni_platform", npu_platform)

    freqs_real = boogu_image_transformer.BooguImageDoubleStreamRotaryPosEmbed.get_freqs_real(
        AXES_DIM_ROPE, AXES_LENS, theta=10000
    )

    assert all(freqs.dtype == torch.float32 for pair in freqs_real for freqs in pair)


def test_real_rotary_frequency_tables_use_float32_on_mps(monkeypatch):
    from vllm_omni.diffusion.models.boogu_image import boogu_image_transformer

    platform = SimpleNamespace(
        is_npu=lambda: False,
        supports_float64=lambda: True,
    )
    monkeypatch.setattr(boogu_image_transformer, "current_omni_platform", platform)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    freqs_real = boogu_image_transformer.BooguImageDoubleStreamRotaryPosEmbed.get_freqs_real(
        AXES_DIM_ROPE, AXES_LENS, theta=10000
    )

    assert all(freqs.dtype == torch.float32 for pair in freqs_real for freqs in pair)


def test_single_stream_block_shape():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageTransformerBlock,
    )

    block = BooguImageTransformerBlock(
        dim=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        multiple_of=MULTIPLE_OF,
        ffn_dim_multiplier=None,
        norm_eps=NORM_EPS,
        modulation=True,
    )
    _randomize_parameters(block)

    batch_size, seq_len = 1, 16
    hidden_states = torch.randn(batch_size, seq_len, HIDDEN_SIZE)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    rotary_emb = _identity_rotary_emb(batch_size, seq_len)
    temb = torch.randn(batch_size, min(HIDDEN_SIZE, 1024))

    out = block(hidden_states, attention_mask, rotary_emb, temb)
    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()

    with pytest.raises(ValueError, match="temb"):
        block(hidden_states, attention_mask, rotary_emb, None)


def test_single_stream_block_no_modulation():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageTransformerBlock,
    )

    block = BooguImageTransformerBlock(
        dim=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        multiple_of=MULTIPLE_OF,
        ffn_dim_multiplier=None,
        norm_eps=NORM_EPS,
        modulation=False,
    )
    _randomize_parameters(block)

    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, HIDDEN_SIZE)
    out = block(hidden_states, None, _identity_rotary_emb(batch_size, seq_len))
    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()


def test_double_stream_block_shape():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageDoubleStreamTransformerBlock,
    )

    block = BooguImageDoubleStreamTransformerBlock(
        dim=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        multiple_of=MULTIPLE_OF,
        ffn_dim_multiplier=None,
        norm_eps=NORM_EPS,
        modulation=True,
    )
    _randomize_parameters(block)

    batch_size = 1
    img_len, instruct_len = 16, 8
    total_len = img_len + instruct_len

    img_hidden_states = torch.randn(batch_size, img_len, HIDDEN_SIZE)
    instruct_hidden_states = torch.randn(batch_size, instruct_len, HIDDEN_SIZE)
    img_attention_mask = torch.ones(batch_size, img_len, dtype=torch.bool)
    joint_attention_mask = torch.ones(batch_size, total_len, dtype=torch.bool)
    image_rotary_emb = _identity_rotary_emb(batch_size, img_len)
    rotary_emb = _identity_rotary_emb(batch_size, total_len)
    temb = torch.randn(batch_size, min(HIDDEN_SIZE, 1024))

    img_out, instruct_out = block(
        img_hidden_states,
        instruct_hidden_states,
        img_attention_mask,
        joint_attention_mask,
        image_rotary_emb,
        rotary_emb,
        temb=temb,
        encoder_seq_lengths=[instruct_len] * batch_size,
        seq_lengths=[total_len] * batch_size,
    )

    assert img_out.shape == img_hidden_states.shape
    assert instruct_out.shape == instruct_hidden_states.shape
    assert torch.isfinite(img_out).all()
    assert torch.isfinite(instruct_out).all()


def test_transformer_instantiates():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageTransformer2DModel,
    )

    model = BooguImageTransformer2DModel(od_config=_tiny_od_config())

    assert BooguImageTransformer2DModel._repeated_blocks == [
        "BooguImageTransformerBlock",
        "BooguImageContextRefinerTransformerBlock",
        "BooguImageSingleStreamTransformerBlock",
    ]
    assert BooguImageTransformer2DModel._layerwise_offload_blocks_attrs == [
        "single_stream_layers",
        "double_stream_layers",
    ]

    assert len(model.noise_refiner) == 2
    assert len(model.ref_image_refiner) == 2
    assert len(model.context_refiner) == 2
    assert len(model.double_stream_layers) == 2
    assert len(model.single_stream_layers) == 2  # num_layers - num_double_stream_layers
    assert model.image_index_embedding.shape == (5, HIDDEN_SIZE)
    # patch_size^2 * in_channels -> hidden_size
    assert model.x_embedder.in_features == 2 * 2 * 4
    assert model.x_embedder.out_features == HIDDEN_SIZE


def test_transformer_preprocesses_multiple_instruction_feature_layers():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageTransformer2DModel,
    )

    instruction_feature_configs = {
        "instruction_feat_dim": 32,
        "reduce_type": "concat",
        "num_instruction_feature_layers": 2,
    }
    model = BooguImageTransformer2DModel(
        od_config=_tiny_od_config(instruction_feature_configs=instruction_feature_configs)
    )
    hidden_states = [torch.randn(1, 8, 32), torch.randn(1, 8, 32)]

    processed = model.preprocess_instruction_hidden_states(hidden_states)

    assert model.preprocessed_instruction_feat_dim == 64
    assert processed.shape == (1, 8, 64)
    assert torch.equal(processed, torch.cat(hidden_states, dim=-1))


def test_transformer_validates_rope_dims():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageTransformer2DModel,
    )

    with pytest.raises(ValueError, match="axes_dim_rope"):
        BooguImageTransformer2DModel(od_config=_tiny_od_config(axes_dim_rope=[8, 8, 8]))


def test_transformer_rejects_prompt_tuning():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageTransformer2DModel,
    )

    with pytest.raises(NotImplementedError, match="[Pp]rompt tuning"):
        BooguImageTransformer2DModel(od_config=_tiny_od_config(prompt_tuning_configs={"use_prompt_tuning": True}))


def _native_to_checkpoint_name(name: str) -> str:
    """Inverse of ``load_weights`` remapping: native param name -> diffusers name.

    - ``.to_out.<suffix>`` -> ``.to_out.0.<suffix>`` (diffusers ModuleList wrap).
    - promoted joint-attention projections move back under ``.processor.``.
    """
    if ".to_out." in name:
        name = name.replace(".to_out.", ".to_out.0.")
    for proj in (
        "img_to_q",
        "img_to_k",
        "img_to_v",
        "instruct_to_q",
        "instruct_to_k",
        "instruct_to_v",
        "instruct_out",
        "img_out",
    ):
        token = f".img_instruct_attn.{proj}."
        if token in name:
            name = name.replace(token, f".img_instruct_attn.processor.{proj}.")
            break
    return name


def test_transformer_load_weights_round_trip():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageTransformer2DModel,
    )

    model = BooguImageTransformer2DModel(od_config=_tiny_od_config())
    native_params = dict(model.named_parameters())

    # Build synthetic diffusers-named weights (one per native parameter).
    checkpoint_weights = {}
    for native_name, param in native_params.items():
        checkpoint_weights[_native_to_checkpoint_name(native_name)] = torch.randn_like(param)

    # The remapping must be a bijection over the parameter set.
    assert len(checkpoint_weights) == len(native_params)

    loaded = model.load_weights(list(checkpoint_weights.items()))

    # No missing / unexpected parameters.
    assert loaded == set(native_params.keys())

    # Values landed on the right parameters (TP=1: weight_loader copies verbatim).
    reloaded = dict(model.named_parameters())
    for native_name in native_params:
        expected = checkpoint_weights[_native_to_checkpoint_name(native_name)]
        assert torch.allclose(reloaded[native_name], expected)


def test_transformer_load_weights_warns_for_unexpected_and_unloaded():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageTransformer2DModel,
    )

    # vllm_omni's logger hierarchy hangs off vLLM's `vllm` logger, which sets
    # propagate=False, so records never reach the root logger that pytest's
    # caplog listens on. Capture at the emitting logger instead.
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    log = logging.getLogger("vllm_omni.diffusion.models.boogu_image.boogu_image_transformer")
    handler = _Capture(level=logging.WARNING)
    log.addHandler(handler)
    try:
        model = BooguImageTransformer2DModel(od_config=_tiny_od_config())
        native_params = dict(model.named_parameters())
        loaded_name = next(iter(native_params))
        checkpoint_name = _native_to_checkpoint_name(loaded_name)

        loaded = model.load_weights(
            [
                (checkpoint_name, torch.randn_like(native_params[loaded_name])),
                ("unexpected.weight", torch.ones(1)),
            ]
        )
    finally:
        log.removeHandler(handler)

    text = "\n".join(messages)
    assert loaded == {loaded_name}
    assert "Skipping unexpected checkpoint weight unexpected.weight" in text
    assert "Model parameters not loaded from checkpoint" in text
    assert next(name for name in native_params if name != loaded_name) in text


def test_transformer_forward_t2i_shape():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageDoubleStreamRotaryPosEmbed,
        BooguImageTransformer2DModel,
    )

    model = BooguImageTransformer2DModel(od_config=_tiny_od_config())
    _randomize_parameters(model)
    model.eval()

    batch_size = 1
    in_channels = 4
    latent_h = latent_w = 8  # multiples of patch_size (2)
    instruct_len = 8
    instruction_feat_dim = 32  # matches _tiny_tf_model_config

    latents = torch.randn(batch_size, in_channels, latent_h, latent_w)
    timestep = torch.full((batch_size,), 0.5)
    instruction_hidden_states = torch.randn(batch_size, instruct_len, instruction_feat_dim)
    instruction_attention_mask = torch.ones(batch_size, instruct_len, dtype=torch.bool)
    freqs_real = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_real(model.axes_dim_rope, model.axes_lens, theta=10000)

    with torch.no_grad():
        out = model(latents, timestep, instruction_hidden_states, freqs_real, instruction_attention_mask)

    assert out.shape == (batch_size, model.out_channels, latent_h, latent_w)
    assert torch.isfinite(out).all()


def test_transformer_forward_ti2i_shape():
    """Editing path: a non-empty ``ref_image_hidden_states`` exercises the
    reference-image patch embedder + refiner and must not change the output
    shape (the output tracks the noise-latent dimensions)."""
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageDoubleStreamRotaryPosEmbed,
        BooguImageTransformer2DModel,
    )

    model = BooguImageTransformer2DModel(od_config=_tiny_od_config())
    _randomize_parameters(model)
    model.eval()

    batch_size = 1
    in_channels = 4
    latent_h = latent_w = 8
    ref_h, ref_w = 6, 10  # a differently-sized reference latent
    instruct_len = 8
    instruction_feat_dim = 32

    latents = torch.randn(batch_size, in_channels, latent_h, latent_w)
    timestep = torch.full((batch_size,), 0.5)
    instruction_hidden_states = torch.randn(batch_size, instruct_len, instruction_feat_dim)
    instruction_attention_mask = torch.ones(batch_size, instruct_len, dtype=torch.bool)
    freqs_real = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_real(model.axes_dim_rope, model.axes_lens, theta=10000)

    # One sample, one reference image (Boogu editing supports a single ref).
    ref_image_hidden_states = [[torch.randn(in_channels, ref_h, ref_w)]]

    with torch.no_grad():
        out = model(
            latents,
            timestep,
            instruction_hidden_states,
            freqs_real,
            instruction_attention_mask,
            ref_image_hidden_states=ref_image_hidden_states,
        )

    assert out.shape == (batch_size, model.out_channels, latent_h, latent_w)
    assert torch.isfinite(out).all()
