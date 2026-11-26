# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FLUX.1 attention on the fused QK RMSNorm + RoPE ops (double and single blocks)."""

import os

import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1), pytest.mark.diffusion]

_HEADS, _HEAD_DIM = 24, 128
_DIM = _HEADS * _HEAD_DIM


@pytest.fixture
def _dist_env():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    # vLLM 0.28+: parallel-state init, CustomOp construction and the linear
    # layers' forward all read the current vLLM config.
    config = VllmConfig()
    with set_current_vllm_config(config):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        try:
            initialize_model_parallel()
            yield config
        finally:
            cleanup_dist_env_and_memory()


def _rope(seq_len: int):
    torch.manual_seed(seq_len)
    angles = torch.randn(seq_len, _HEAD_DIM // 2, device="cuda")
    return torch.cos(angles), torch.sin(angles)


def test_table_skipped_on_cpu():
    """No table (no allocation) where the fused kernel cannot run."""
    from vllm_omni.diffusion.models.flux.flux_transformer import _with_qk_norm_rope_table

    cos, sin = torch.randn(40, 64), torch.randn(40, 64)
    x = torch.zeros(2, 40, _DIM, dtype=torch.bfloat16)
    assert _with_qk_norm_rope_table({"other": 1}, (cos, sin), x) == {"other": 1}


@pytest.mark.skipif(not current_platform.is_cuda() or not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_table_and_kwargs():
    from vllm_omni.diffusion.models.flux.flux_transformer import QK_NORM_ROPE_TABLE_KEY, _with_qk_norm_rope_table

    cos, sin = torch.randn(2, 40, 64, device="cuda"), torch.randn(2, 40, 64, device="cuda")
    x = torch.zeros(2, 40, _DIM, dtype=torch.bfloat16, device="cuda")
    kw = _with_qk_norm_rope_table({"other": 1}, (cos[0], sin[0]), x)
    assert kw["other"] == 1 and kw[QK_NORM_ROPE_TABLE_KEY].shape == (80, _HEAD_DIM)
    assert kw[QK_NORM_ROPE_TABLE_KEY].dtype == torch.bfloat16
    assert _with_qk_norm_rope_table(None, (cos[0], sin[0]), x.half()) is None  # non-bf16 activations


@pytest.mark.skipif(not current_platform.is_cuda() or not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("double_stream", [True, False])
def test_flux_attention_fused_matches_eager(_dist_env, double_stream):
    from vllm_omni.diffusion.models.flux.flux_transformer import (
        QK_NORM_ROPE_TABLE_KEY,
        FluxAttention,
        pack_qk_norm_rope_table,
    )

    torch.manual_seed(3)
    with torch.device("cuda"):
        attn = FluxAttention(
            query_dim=_DIM,
            heads=_HEADS,
            dim_head=_HEAD_DIM,
            added_kv_proj_dim=_DIM if double_stream else None,
            eps=1e-6,
        )
    attn = attn.to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_((0.02 * torch.randn(p.shape, device="cuda")).to(p.dtype) if p.dim() > 1 else p)
        for n in (attn.norm_q, attn.norm_k) + ((attn.norm_added_q, attn.norm_added_k) if double_stream else ()):
            n.weight.uniform_(0.5, 1.5)
    batch, txt, img = 2, 64, 256
    hidden = torch.randn(batch, img if double_stream else txt + img, _DIM, device="cuda", dtype=torch.bfloat16)
    encoder = torch.randn(batch, txt, _DIM, device="cuda", dtype=torch.bfloat16) if double_stream else None
    cos, sin = _rope(txt + img)
    table = pack_qk_norm_rope_table(cos, sin, batch, dtype=torch.bfloat16)

    with torch.no_grad():
        eager = attn(hidden, encoder_hidden_states=encoder, image_rotary_emb=(cos, sin))
        fused = attn(
            hidden, encoder_hidden_states=encoder, image_rotary_emb=(cos, sin), **{QK_NORM_ROPE_TABLE_KEY: table}
        )
    for e, f in zip(eager if double_stream else (eager,), fused if double_stream else (fused,)):
        # This integration check accepts the available eager RoPE provider.
        # Exact CUDA-order parity is covered by the fused-op tests; different
        # providers and compiled RoPE can have different rounding behavior.
        torch.testing.assert_close(f, e, atol=0.05, rtol=0.05)


@pytest.fixture
def _flux_model_inputs(_dist_env):
    from vllm.config.kernel import IrOpPriorityConfig

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel

    config = OmniDiffusionConfig(
        model=None,
        enforce_eager=True,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    with (
        set_current_diffusion_config(config),
        set_forward_context(vllm_config=_dist_env, omni_diffusion_config=config),
        IrOpPriorityConfig(rms_norm=["vllm_c", "native"]).set_priority(),
        torch.no_grad(),
    ):
        with torch.device("cuda"):
            model = (
                FluxTransformer2DModel(
                    num_layers=1,
                    num_single_layers=1,
                    num_attention_heads=2,
                    joint_attention_dim=64,
                    pooled_projection_dim=32,
                )
                .to(torch.bfloat16)
                .eval()
            )
        generator = torch.Generator(device="cuda").manual_seed(42)
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(0, 0.02, generator=generator)
            elif name.endswith("weight"):
                param.uniform_(0.5, 1.5, generator=generator)
            else:
                param.zero_()

        def random_tensor(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator)

        batch, txt, img = 2, 16, 64
        txt_ids = torch.zeros(txt, 3, device="cuda")
        txt_ids[:, 0] = torch.arange(txt, device="cuda")
        img_ids = torch.zeros(img, 3, device="cuda")
        positions = torch.arange(img, device="cuda")
        img_ids[:, 1], img_ids[:, 2] = positions // 8, positions % 8
        yield (
            model,
            dict(
                hidden_states=random_tensor(batch, img, 64),
                encoder_hidden_states=random_tensor(batch, txt, 64),
                pooled_projections=random_tensor(batch, 32),
                timestep=torch.full((batch,), 0.5, device="cuda"),
                guidance=torch.full((batch,), 3.5, device="cuda"),
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            ),
        )


@pytest.mark.skipif(not current_platform.is_cuda() or not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_flux_model_fused_matches_cuda_eager_bitwise(_flux_model_inputs, monkeypatch, mocker):
    from vllm import ir

    from vllm_omni.diffusion.models.flux import flux_transformer

    model, inputs = _flux_model_inputs
    attentions = (model.transformer_blocks[0].attn, model.single_transformer_blocks[0].attn)
    if any(attn.rope.apply_rotary_emb_vllm_flash_attn is None for attn in attentions):
        pytest.skip("Bitwise reference requires the vLLM FlashAttention RoPE provider")
    query = inputs["hidden_states"].new_empty((2, 64, 2, _HEAD_DIM))
    assert ir.ops.rms_norm.dispatch(query, attentions[0].norm_q.weight, 1e-6, None).provider == "vllm_c"
    rotations = [mocker.spy(attn.rope, "apply_rotary_emb_vllm_flash_attn") for attn in attentions]
    joint = mocker.spy(flux_transformer, "fused_joint_qkv_norm_rope")
    single = mocker.spy(flux_transformer, "fused_qk_norm_rope")

    with monkeypatch.context() as patch:
        patch.setattr(flux_transformer, "pack_qk_norm_rope_table", lambda *args, **kwargs: None)
        (eager,) = model(**inputs)
    joint.assert_not_called()
    single.assert_not_called()
    assert [rotation.call_count for rotation in rotations] == [2, 2]

    (fused,) = model(**inputs)
    for op in (joint, single):
        op.assert_called_once()
        assert op.call_args.kwargs["cuda_rms_norm"] is True
    assert [rotation.call_count for rotation in rotations] == [2, 2]
    assert torch.isfinite(eager).all()
    torch.testing.assert_close(fused.view(torch.int16), eager.view(torch.int16), rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda() or not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_flux_teacache_fuses_only_when_running_blocks(_flux_model_inputs, mocker):
    from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
    from vllm_omni.diffusion.cache.teacache.hook import apply_teacache_hook
    from vllm_omni.diffusion.models.flux import flux_transformer

    model, inputs = _flux_model_inputs
    (regular,) = model(**inputs)
    apply_teacache_hook(model, TeaCacheConfig(transformer_type=type(model).__name__, rel_l1_thresh=1.0))
    pack = mocker.spy(flux_transformer, "pack_qk_norm_rope_table")
    joint = mocker.spy(flux_transformer, "fused_joint_qkv_norm_rope")
    single = mocker.spy(flux_transformer, "fused_qk_norm_rope")

    (miss,) = model(**inputs)
    torch.testing.assert_close(miss.view(torch.int16), regular.view(torch.int16), rtol=0, atol=0)
    for op in (pack, joint, single):
        op.assert_called_once()
        op.reset_mock()

    (hit,) = model(**inputs)
    assert torch.isfinite(hit).all()
    for op in (pack, joint, single):
        op.assert_not_called()
