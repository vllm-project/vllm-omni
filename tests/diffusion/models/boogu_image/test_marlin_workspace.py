# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contracts for Boogu's local workspace; CUDA kernels are not executed."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.quantization.online import fp8 as online_fp8

from vllm_omni.diffusion.models.boogu_image import marlin_workspace as workspace

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def cpu_kernel(monkeypatch):
    # Kernel construction normally probes GPU support. Keep real class identities
    # for the binder's exact-type checks while replacing only that CPU boundary.
    def initialize(self, config, names):
        self.config = config
        self.layer_param_names = names
        self.marlin_input_dtype = None

    monkeypatch.setattr(workspace.MarlinFP8ScaledMMLinearKernel, "__init__", initialize)


def make_online_marlin_layer():
    layer = nn.Linear(4, 3, bias=False)
    layer.input_size_per_partition = 4
    layer.output_size_per_partition = 3
    method = object.__new__(online_fp8.Fp8PerBlockOnlineLinearMethod)
    method.fp8_linear = workspace.MarlinFP8ScaledMMLinearKernel(
        SimpleNamespace(), ["weight", "weight_scale_inv", "input_scale", "input_scale_ub"]
    )
    method.fp8_linear.marlin_input_dtype = torch.bfloat16
    layer.quant_method = method
    return layer


def test_bind_preserves_configuration_and_is_idempotent(cpu_kernel):
    layer = make_online_marlin_layer()
    selected = layer.quant_method.fp8_linear
    workspace.bind_boogu_marlin_workspaces(nn.ModuleList([layer]))
    local = layer.quant_method.fp8_linear

    assert type(local) is workspace._BooguMarlinFP8Kernel
    assert local.config is selected.config
    assert local.layer_param_names is selected.layer_param_names
    assert local.marlin_input_dtype is selected.marlin_input_dtype
    assert not hasattr(layer, workspace._WORKSPACE_NAME)  # Allocation waits for post-load.

    workspace.bind_boogu_marlin_workspaces(nn.ModuleList([layer]))
    assert layer.quant_method.fp8_linear is local


def test_bind_preserves_per_tensor_method(cpu_kernel):
    layer = make_online_marlin_layer()
    selected = layer.quant_method.fp8_linear
    layer.quant_method = object.__new__(online_fp8.Fp8PerTensorOnlineLinearMethod)
    layer.quant_method.fp8_linear = selected

    workspace.bind_boogu_marlin_workspaces(nn.ModuleList([layer]))

    assert layer.quant_method.fp8_linear is selected


def test_bind_preserves_non_marlin_kernel(cpu_kernel):
    layer = make_online_marlin_layer()
    selected = layer.quant_method.fp8_linear = object()

    workspace.bind_boogu_marlin_workspaces(nn.ModuleList([layer]))

    assert layer.quant_method.fp8_linear is selected


def test_postload_allocates_workspace_on_final_weight_device(cpu_kernel, monkeypatch, mocker):
    layers = [make_online_marlin_layer(), make_online_marlin_layer()]
    for layer in layers:
        layer.weight = nn.Parameter(torch.empty(3, 4, device="meta"), requires_grad=False)
    workspace.bind_boogu_marlin_workspaces(nn.ModuleList(layers))

    def parent_process(self, layer):
        assert not hasattr(layer, workspace._WORKSPACE_NAME)
        assert layer.weight.device.type == "meta"
        layer.weight = nn.Parameter(torch.ones(3, 4, dtype=torch.bfloat16), requires_grad=False)

    monkeypatch.setattr(workspace.MarlinFP8ScaledMMLinearKernel, "process_weights_after_loading", parent_process)
    allocator = mocker.patch.object(
        workspace, "marlin_make_workspace_new", side_effect=lambda device, blocks: torch.zeros(16, dtype=torch.int32)
    )
    buffers = []
    for layer in layers:
        layer.quant_method.fp8_linear.process_weights_after_loading(layer)
        buffer = getattr(layer, workspace._WORKSPACE_NAME)
        buffers.append(buffer)
        assert dict(layer.named_buffers())[workspace._WORKSPACE_NAME] is buffer
        assert workspace._WORKSPACE_NAME not in layer.state_dict()

    assert buffers[0].data_ptr() != buffers[1].data_ptr()
    assert allocator.call_args_list == [mocker.call(torch.device("cpu"), workspace.MARLIN_MAX_BLOCKS_PER_SM)] * 2


@pytest.mark.parametrize(("inverse_scale", "with_bias"), [(True, True), (False, False)])
def test_apply_forwards_explicit_workspace_scales_shapes_dtype_and_bias(cpu_kernel, mocker, inverse_scale, with_bias):
    layer = make_online_marlin_layer()
    workspace.bind_boogu_marlin_workspaces(nn.ModuleList([layer]))
    layer.weight_scale = torch.ones(1, 1)
    layer.weight_scale_inv = torch.full((1, 1), 2.0) if inverse_scale else None
    buffer = torch.zeros(16, dtype=torch.int32)
    layer.register_buffer(workspace._WORKSPACE_NAME, buffer, persistent=False)
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    bias = torch.ones(3, dtype=torch.bfloat16) if with_bias else None
    output = torch.empty(2, 3, dtype=torch.bfloat16)
    apply = mocker.patch.object(workspace, "apply_fp8_marlin_linear", return_value=output)

    result = layer.quant_method.fp8_linear.apply_weights(layer, x, bias)

    assert result is output
    apply.assert_called_once_with(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale_inv if inverse_scale else layer.weight_scale,
        workspace=buffer,
        size_n=3,
        size_k=4,
        input_dtype=torch.bfloat16,
        bias=bias,
    )
