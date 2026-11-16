# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate real CUDA MXFP8 loading, fused shards, bias and diffusion shapes."""

import pytest
import torch
from torch.nn import functional as F
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.model_loader.reload.layerwise import finalize_layerwise_processing
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.mxfp8_cuda import CUDAMxfp8OnlineLinearMethod

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "B200"})]


@pytest.fixture
def cuda_mxfp8():
    capability = current_platform.get_device_capability()
    if not current_platform.is_cuda() or capability is None or capability.to_int() not in (100, 103, 107):
        pytest.skip("Requires SM100, SM103 or SM107")
    config = build_quant_config("mxfp8", ignored_layers=["skip"])
    od_config = OmniDiffusionConfig(model="test", dtype=torch.bfloat16, quantization_config=config)
    vllm_config = create_diffusion_vllm_config(torch.device("cuda:0"), od_config)
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="nccl",
        )
        with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16), torch.device("cuda:0"):
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            yield config
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(17, 128), (2, 9, 128)])
@pytest.mark.parametrize("bias", [False, True])
def test_cuda_mxfp8_online_linear(cuda_mxfp8, dtype, shape, bias):
    torch.manual_seed(42)
    layer = ReplicatedLinear(
        128, 256, bias=bias, params_dtype=dtype, quant_config=cuda_mxfp8, prefix="linear", disable_tp=True
    )
    assert isinstance(layer.quant_method, CUDAMxfp8OnlineLinearMethod)
    assert layer.weight.is_meta
    weights = torch.randn(256, 128, dtype=dtype) / 128**0.5
    # A strided last dimension exercises the activation's contiguous packing.
    inputs = torch.randn(*shape[:-1], shape[-1] * 2, dtype=dtype)[..., ::2]
    bias_data = torch.randn(256, dtype=dtype) if bias else None
    if bias:
        # Checkpoint iteration may deliver bias before the weight wrapper has
        # registered it. Exercise the same finalization as the Omni loader.
        layer.bias.weight_loader(layer.bias, bias_data)
    expected = F.linear(inputs, weights, bias_data).float()
    layer.weight.weight_loader(layer.weight, weights)
    finalize_layerwise_processing(layer, model_config=None)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.uint8
    assert type(layer.quant_method.kernel).__name__ == "FlashInferCutedslMxfp8LinearKernel"
    actual = layer(inputs)[0]
    assert actual.shape == (*shape[:-1], 256)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    relative_mse = (actual.float() - expected).square().mean() / expected.square().mean()
    assert relative_mse.item() < 0.003
    # The loader and post-load traversal may both call finalization.
    first_weight = layer.weight
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight is first_weight
    ignored = ReplicatedLinear(128, 256, quant_config=cuda_mxfp8, prefix="skip", disable_tp=True)
    assert isinstance(ignored.quant_method, UnquantizedLinearMethod)


@pytest.mark.parametrize("projection", ["qkv", "merged", "row"])
def test_cuda_mxfp8_fused_checkpoint_loading(cuda_mxfp8, projection):
    torch.manual_seed(43)
    kwargs = dict(bias=False, params_dtype=torch.bfloat16, quant_config=cuda_mxfp8, disable_tp=True)
    shards: list[str | int | None]
    if projection == "qkv":
        layer = QKVParallelLinear(128, 64, 2, **kwargs)
        shards = ["q", "k", "v"]
    elif projection == "merged":
        layer = MergedColumnParallelLinear(128, [128, 128], **kwargs)
        shards = [0, 1]
    else:
        layer = RowParallelLinear(128, 128, **kwargs)
        shards = [None]
    parts = []
    for index, shard in enumerate(shards):
        weights = torch.randn(128, 128, dtype=torch.bfloat16) / 128**0.5
        parts.append(weights)
        if shard is None:
            layer.weight.weight_loader(layer.weight, weights)
        else:
            layer.weight.weight_loader(layer.weight, weights, shard)
        if index + 1 < len(shards):
            assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.dtype == torch.float8_e4m3fn
    inputs = torch.randn(2, 17, 128, dtype=torch.bfloat16)
    expected = F.linear(inputs, torch.cat(parts)).float()
    actual = layer(inputs)[0].float()
    assert actual.shape == expected.shape
    assert ((actual - expected).square().mean() / expected.square().mean()).item() < 0.003


def test_cuda_mxfp8_invalid_partition(cuda_mxfp8):
    with pytest.raises(ValueError, match="divisible by 32"):
        ReplicatedLinear(33, 128, quant_config=cuda_mxfp8, prefix="linear", disable_tp=True)
