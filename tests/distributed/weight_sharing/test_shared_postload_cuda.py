# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real MoE finalization/forward on shared views; IPC transport is tested separately."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.distributed.weight_sharing.cuda_helpers import require_cuda
from tests.helpers.mark import hardware_marks

require_cuda(1)
pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.fixture
def moe_environment(tmp_path):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    config = VllmConfig(device_config=DeviceConfig(device="cuda"))
    config.kernel_config.moe_backend = "triton"
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method=(tmp_path / "rendezvous").as_uri()
        )
        initialize_model_parallel()
        init_workspace_manager(torch.device("cuda"))
        try:
            yield config
        finally:
            reset_workspace_manager()
            cleanup_dist_env_and_memory()


def test_shared_moe_initializes_kernel_without_changing_weights(moe_environment):
    from vllm.forward_context import set_forward_context
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import HunYuanSparseMoeBlock
    from vllm_omni.distributed.weight_sharing.adapters.hunyuan_image3 import (
        attach_imported_weights,
        collect_shared_bindings,
    )

    config = SimpleNamespace(
        hidden_size=128,
        num_experts=4,
        moe_topk=2,
        intermediate_size=128,
        moe_intermediate_size=None,
        use_mixed_mlp_moe=0,
    )

    def backbone(device, prefix):
        with set_default_torch_dtype(torch.bfloat16), torch.device(device):
            model = nn.Module()
            layer = nn.Module()
            layer.mlp = HunYuanSparseMoeBlock(config, prefix=prefix, shared_router_dtype=torch.float32)
            model.layers = nn.ModuleList([layer])
            return model

    torch.manual_seed(0)
    provider = backbone("cuda", "provider.mlp")
    with torch.no_grad():
        for param in provider.parameters():
            param.normal_(std=0.02)
    loader = object.__new__(DiffusersPipelineLoader)
    loader._process_weights_after_loading(provider, torch.device("cuda"))
    bindings = collect_shared_bindings(provider)
    # CPU snapshots detect in-place writes without allocating another GPU weight copy.
    before = {name: tensor.detach().cpu().clone() for name, tensor in bindings.items()}
    consumer = backbone("meta", "consumer.mlp")
    attach_imported_weights(consumer, bindings)
    method = consumer.layers[0].mlp.experts.routed_experts.quant_method
    assert method.moe_kernel is None
    assert method.supports_pre_processed_weights
    loader._process_weights_after_loading(
        consumer,
        torch.device("cuda"),
        preprocessed_module_names=frozenset(name.rsplit(".", 1)[0] for name in bindings),
    )
    assert method.moe_kernel is not None
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad(), set_forward_context(None, moe_environment):
        expected = provider.layers[0].mlp(x)
        actual = consumer.layers[0].mlp(x)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    for name, original in bindings.items():
        attached = consumer.get_parameter(name)
        assert attached.data_ptr() == original.data_ptr()
        assert attached.stride() == original.stride()
        torch.testing.assert_close(original.cpu(), before[name], rtol=0, atol=0)
