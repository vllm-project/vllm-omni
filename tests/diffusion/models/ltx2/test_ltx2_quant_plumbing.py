# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only structural tests for LTX-2 quantization plumbing.

Verifies that ``quant_config`` and ``prefix`` are threaded through every
linear-bearing layer in the LTX-2 transformer hierarchy, and that the
LTX-2-specific ``disable_kv_quant`` policy is applied only on text
cross-attention. Modeled after
``tests/diffusion/models/flux/test_flux_prefix_propagation.py``.

What's intentionally NOT here:
- Forward passes / quantization kernels — owned by
  ``tests/diffusion/quantization/test_quantization_fp8.py`` (e2e on CUDA)
  and ``test_quantization_quality.py`` (LPIPS gate).
- ``OmniDiffusionConfig`` string→config resolution — owned by
  ``tests/diffusion/quantization/test_int8_config.py``.
- Per-class signature checks — implied by the construction tests below
  (any missing ``quant_config``/``prefix`` kwarg would TypeError at build).
"""

import inspect
import os

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Minimal dimensions for a constructible LTX-2 block.
_DIM = 64
_HEADS = 2
_HEAD_DIM = _DIM // _HEADS
_AUDIO_DIM = 32
_AUDIO_HEADS = 2
_AUDIO_HEAD_DIM = _AUDIO_DIM // _AUDIO_HEADS
_CROSS_ATTN_DIM = 48
_AUDIO_CROSS_ATTN_DIM = 48


@pytest.fixture(autouse=True)
def _init_distributed():
    """vLLM parallel linears require a TP group; TP=1 suffices for plumbing."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29504")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


def _param_names(module) -> set[str]:
    return {name for name, _ in module.named_parameters()}


def _build_block(prefix: str = "transformer_blocks.0"):
    from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformerBlock

    return LTX2VideoTransformerBlock(
        dim=_DIM,
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        cross_attention_dim=_CROSS_ATTN_DIM,
        audio_dim=_AUDIO_DIM,
        audio_num_attention_heads=_AUDIO_HEADS,
        audio_attention_head_dim=_AUDIO_HEAD_DIM,
        audio_cross_attention_dim=_AUDIO_CROSS_ATTN_DIM,
        quant_config=None,
        prefix=prefix,
    )


# ---------------------------------------------------------------------------
# Class-attribute checks: things the construction tests don't cover.
# ---------------------------------------------------------------------------


def test_packed_modules_mapping_declared():
    """vLLM's quant weight loader fuses to_q/to_k/to_v scales into to_qkv."""
    from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel

    assert LTX2VideoTransformer3DModel.packed_modules_mapping == {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }


def test_create_transformer_from_config_accepts_quant_config():
    """The pipeline-level factory must forward quant_config to the model."""
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import create_transformer_from_config

    params = inspect.signature(create_transformer_from_config).parameters
    assert "quant_config" in params
    assert params["quant_config"].default is None


@pytest.mark.parametrize(
    "module_path,cls_name",
    [
        ("vllm_omni.diffusion.models.ltx2.pipeline_ltx2", "LTX2Pipeline"),
        ("vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3", "LTX23Pipeline"),
    ],
)
def test_pipeline_reads_od_config_quantization(module_path, cls_name):
    """Both top-level LTX-2 pipelines must read od_config.quantization_config
    and pass it to create_transformer_from_config. Fragile (string match on
    source), but catches the regression where the plumbing silently reverts
    to BF16 — no construction-based test would catch that."""
    import importlib

    cls = getattr(importlib.import_module(module_path), cls_name)
    src = inspect.getsource(cls.__init__)
    assert 'getattr(self.od_config, "quantization_config", None)' in src
    assert "create_transformer_from_config(transformer_config, quant_config=quant_config)" in src


# ---------------------------------------------------------------------------
# Construction-based: prefix propagates to real linear-layer parameter names.
# Same pattern as tests/diffusion/models/flux/test_flux_prefix_propagation.py.
# ---------------------------------------------------------------------------


def test_block_prefix_propagates_through_self_attention():
    """attn1 / audio_attn1 use fused QKV; output projection is to_out.0."""
    block = _build_block()
    params = _param_names(block)

    for branch in ("attn1", "audio_attn1"):
        assert any(name.startswith(f"{branch}.to_qkv.") for name in params), (
            f"{branch}.to_qkv.* missing — fused QKV prefix did not propagate."
        )
        assert any(name.startswith(f"{branch}.to_out.0.") for name in params), (
            f"{branch}.to_out.0.* missing — output-projection prefix did not propagate."
        )


def test_block_prefix_propagates_through_cross_attention():
    """attn2 / audio_attn2 / audio_to_video_attn / video_to_audio_attn use
    separate to_q/k/v (NOT fused) plus to_out.0."""
    block = _build_block()
    params = _param_names(block)

    for branch in ("attn2", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"):
        for sub in ("to_q", "to_k", "to_v", "to_out.0"):
            assert any(name.startswith(f"{branch}.{sub}.") for name in params), (
                f"{branch}.{sub}.* missing in cross-attention parameter names."
            )
        assert not any(name.startswith(f"{branch}.to_qkv.") for name in params), (
            f"{branch} unexpectedly produced to_qkv.* — cross-attn should use separate to_q/k/v."
        )


def test_block_prefix_propagates_through_feedforward():
    """ff / audio_ff use net.0 (GELU proj, ColumnParallel) and net.2 (RowParallel)."""
    block = _build_block()
    params = _param_names(block)

    for branch in ("ff", "audio_ff"):
        assert any(name.startswith(f"{branch}.net.0.") for name in params), (
            f"{branch}.net.0.* missing — FFN GELU prefix did not propagate."
        )
        assert any(name.startswith(f"{branch}.net.2.") for name in params), (
            f"{branch}.net.2.* missing — FFN output prefix did not propagate."
        )


def test_text_cross_attention_runtime_disables_kv_quant():
    """attn2 / audio_attn2 carry disable_kv_quant=True on their inner Attention
    (short text-encoder KV; FP8 KV quant degrades quality with no perf win).
    Self-attn and audio<->video cross-attn keep KV quant enabled."""
    block = _build_block()

    assert block.attn2.attn._disable_kv_quant is True
    assert block.audio_attn2.attn._disable_kv_quant is True

    assert block.attn1.attn._disable_kv_quant is False
    assert block.audio_attn1.attn._disable_kv_quant is False
    assert block.audio_to_video_attn.attn._disable_kv_quant is False
    assert block.video_to_audio_attn.attn._disable_kv_quant is False
