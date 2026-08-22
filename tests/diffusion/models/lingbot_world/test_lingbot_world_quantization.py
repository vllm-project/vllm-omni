# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization contract for LingBot World, against real vLLM machinery.

The other LingBot tests run with vLLM's linear layers stubbed out, which is
right for shape and plumbing but cannot verify the two facts the quantized
path actually depends on: how an exclusion list is matched against a prefix,
and what a fused QKV does with per-projection scales. Both are vLLM's
behaviour, not ours, and getting either wrong is silent.

Neither needs a GPU: ``ModelOptNvFp4Config`` resolves its methods and creates
its parameters on CPU.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _distributed():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="gloo"
        )
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _bfloat16_default():
    """Quantized linears expect a half-precision default, restored afterwards.

    ``torch.set_default_dtype`` is process-global: leaving it set makes every
    later test in the session build bfloat16 tensors where it expected float32.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _nvfp4_config(exclude_modules):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    return ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        exclude_modules=list(exclude_modules),
        group_size=16,
    )


def test_wildcard_exclusions_do_not_survive_a_namespaced_prefix():
    """Why the pipeline hands the transformer ``prefix=""``.

    ModelOpt checkpoints spell their exclusion list from the transformer's own
    root -- ``blocks.N...``, ``head``, ``patch_embedding``. vLLM matches a
    plain pattern by substring, so a namespaced prefix survives that case by
    luck, but a wildcard pattern is matched with ``fullmatch`` and does not.

    The public LingBot export happens to enumerate every module by plain name
    *in addition* to its wildcards, so both roots work on it today. The
    unnamespaced root is the one that does not depend on that accident, and it
    is what every other diffusion transformer in tree already uses.

    A missed exclusion is not an error at construction: the layer is built
    quantized and fails much later on a scale tensor it has no parameter for.
    """
    config = _nvfp4_config(["blocks.0*", "c2ws_hidden_states_layer1"])

    assert config.is_layer_excluded("blocks.0.self_attn.q") is True
    assert config.is_layer_excluded("c2ws_hidden_states_layer1") is True

    # A plain pattern still matches under a namespace, by substring...
    assert config.is_layer_excluded("transformer.c2ws_hidden_states_layer1") is True
    # ...but a wildcard one does not.
    assert config.is_layer_excluded("transformer.blocks.0.self_attn.q") is False


def test_a_fused_qkv_collapses_per_projection_weight_scales():
    """Why self-attention stops fusing Q/K/V once a quant config is present.

    A fused layer holds one ``weight_scale_2`` slot per shard but resolves them
    to a single scalar by taking the maximum. Each projection's FP8 block
    scales were normalised by its *own* global scale -- in the public LingBot
    NVFP4 export all three saturate at 448.0 -- so applying another
    projection's global scale rescales that projection's entire output by the
    ratio between them. Those ratios reach 3.5x in that checkpoint.

    vLLM warns about this rather than refusing, so nothing downstream would
    stop a fused build from producing confidently wrong frames.
    """
    from vllm.model_executor.layers.linear import QKVParallelLinear

    layer = QKVParallelLinear(512, 64, 8, bias=True, quant_config=_nvfp4_config([]), prefix="blocks.5.self_attn.qkv")
    params = dict(layer.named_parameters())

    # One slot per shard, exactly as a per-projection checkpoint supplies.
    assert params["weight_scale_2"].shape == (3,)
    assert params["input_scale"].shape == (3,)

    globals_ = {"q": 4.79562e-05, "k": 8.53766e-05, "v": 5.4859e-05}  # real block-5 values
    for shard, value in globals_.items():
        params["weight"].weight_loader(params["weight"], torch.full((512, 256), 0x44, dtype=torch.uint8), shard)
        params["weight_scale"].weight_loader(
            params["weight_scale"], torch.full((512, 32), 448.0).to(torch.float8_e4m3fn), shard
        )
        # float32 explicitly: the default dtype here is bfloat16, whose 8-bit
        # mantissa would round these scales enough to blur the comparison.
        params["weight_scale_2"].weight_loader(
            params["weight_scale_2"], torch.tensor(value, dtype=torch.float32), shard
        )
        params["input_scale"].weight_loader(params["input_scale"], torch.tensor(3.0e-3, dtype=torch.float32), shard)
        params["bias"].weight_loader(params["bias"], torch.zeros(512, dtype=torch.bfloat16), shard)

    layer.quant_method.process_weights_after_loading(layer)

    # The three globals became one, and it is the largest.
    assert float(layer.weight_global_scale) == pytest.approx(max(globals_.values()))
    # Which means q and v would dequantize this much too large.
    assert max(globals_.values()) / globals_["q"] == pytest.approx(1.78, abs=0.01)
    assert max(globals_.values()) / globals_["v"] == pytest.approx(1.56, abs=0.01)


def test_the_lingbot_transformer_does_not_build_a_fused_qkv_when_quantized():
    """The consequence of the two facts above, asserted on the real model."""
    from vllm_omni.diffusion.models.lingbot_world.transformer import (
        CausalLingBotWorldTransformer3DModel,
    )

    model = CausalLingBotWorldTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=64,
        in_channels=36,
        out_channels=2,
        text_dim=6,
        freq_dim=4,
        ffn_dim=8,
        num_layers=1,
        rope_max_seq_len=16,
        sink_size=1,
        num_frames_per_block=1,
        sliding_window_num_frames=3,
        quant_config=_nvfp4_config(["blocks.0*", "head*"]),
        prefix="",
    )

    attention = model.blocks[0].self_attn
    assert attention.fused_qkv is False
    assert {"q", "k", "v"} <= set(dict(attention.named_children()))
    assert "qkv" not in dict(attention.named_children())
