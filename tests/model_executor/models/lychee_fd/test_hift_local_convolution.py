# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Torch contract and process isolation for HiFT-owned convolution dispatch."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.flashcosyvoice.modules.hifigan_components.convolution import (
    Conv1d,
    ConvTranspose1d,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def numerical_profile():
    """Snapshot process-wide math flags without changing or initializing CUDA."""
    return {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


@pytest.mark.parametrize("path", ["default", "local_aten"])
@pytest.mark.parametrize(
    "options,unbatched",
    [
        ({"kernel_size": 3, "padding": 2, "dilation": 2, "groups": 2, "bias": False}, False),
        ({"kernel_size": 3, "padding": 1, "padding_mode": "reflect"}, False),
        ({"kernel_size": 3, "padding": 1, "padding_mode": "replicate"}, True),
        ({"kernel_size": 3, "padding": 1, "padding_mode": "circular"}, False),
        ({"kernel_size": 4, "padding": "same"}, False),
        ({"kernel_size": 3, "padding": "valid", "stride": 2}, True),
    ],
)
def test_conv1d_preserves_torch_padding_group_dilation_and_unbatched_semantics(path, options, unbatched):
    reference = nn.Conv1d(4, 6, dtype=torch.float64, **options)
    local = Conv1d(4, 6, dtype=torch.float64, **options)
    local.load_state_dict(reference.state_dict(), strict=True)
    shape = (4, 9) if unbatched else (1, 4, 9)
    inputs = torch.linspace(-0.4, 0.6, 36, dtype=torch.float64).reshape(shape)
    profile = numerical_profile()
    expected = reference(inputs)
    actual = local(inputs) if path == "default" else local._deterministic_conv_forward(inputs, local.weight, local.bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert numerical_profile() == profile
    assert local.state_dict().keys() == reference.state_dict().keys()


@pytest.mark.parametrize("path", ["default", "local_aten"])
@pytest.mark.parametrize("dynamic_output", [False, True])
@pytest.mark.parametrize(
    "options,unbatched",
    [
        ({"kernel_size": 7, "stride": 3, "padding": 2, "output_padding": 1, "groups": 2}, False),
        ({"kernel_size": 3, "stride": 2, "padding": 1, "dilation": 2, "bias": False}, True),
        ({"kernel_size": 16, "stride": 8, "padding": 4}, False),
        ({"kernel_size": 11, "stride": 5, "padding": 3}, True),
    ],
)
def test_transposed_conv_preserves_torch_output_size_and_other_parameters(path, dynamic_output, options, unbatched):
    reference = nn.ConvTranspose1d(4, 6, dtype=torch.float64, **options)
    local = ConvTranspose1d(4, 6, dtype=torch.float64, **options)
    local.load_state_dict(reference.state_dict(), strict=True)
    shape = (4, 9) if unbatched else (1, 4, 9)
    inputs = torch.linspace(-0.4, 0.6, 36, dtype=torch.float64).reshape(shape)
    output_size = None
    if dynamic_output:
        output_size = list(reference(inputs).shape)
        output_size[-1] = (
            (inputs.shape[-1] - 1) * reference.stride[0]
            - 2 * reference.padding[0]
            + reference.dilation[0] * (reference.kernel_size[0] - 1)
            + reference.stride[0]
        )
    profile = numerical_profile()
    expected = reference(inputs, output_size)
    actual = local(inputs, output_size) if path == "default" else local._deterministic_forward(inputs, output_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert numerical_profile() == profile
    assert local.state_dict().keys() == reference.state_dict().keys()


@pytest.mark.parametrize("path", ["default", "local_aten"])
@pytest.mark.parametrize("base_class,local_class", [(nn.Conv1d, Conv1d), (nn.ConvTranspose1d, ConvTranspose1d)])
def test_weight_norm_checkpoint_keys_and_values_remain_compatible(path, base_class, local_class):
    reference = weight_norm(base_class(4, 6, 3, padding=1)).eval()
    local = weight_norm(local_class(4, 6, 3, padding=1)).eval()
    local.load_state_dict(reference.state_dict(), strict=True)
    assert local.state_dict().keys() == reference.state_dict().keys()
    for name, parameter in reference.state_dict().items():
        torch.testing.assert_close(local.state_dict()[name], parameter, rtol=0, atol=0)
    inputs = torch.linspace(-0.4, 0.6, 36).reshape(1, 4, 9)
    expected = reference(inputs)
    if path == "default":
        actual = local(inputs)
    elif isinstance(local, ConvTranspose1d):
        actual = local._deterministic_forward(inputs)
    else:
        actual = local._deterministic_conv_forward(inputs, local.weight, local.bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("transposed", [False, True])
def test_cpu_and_complex_inputs_delegate_the_default_torch_implementation(transposed):
    class CPUConv(Conv1d):
        def _deterministic_conv_forward(self, *args):
            raise AssertionError("CPU must delegate Torch's default implementation")

    class CPUTranspose(ConvTranspose1d):
        def _deterministic_forward(self, *args):
            raise AssertionError("CPU must delegate Torch's default implementation")

    reference_class: type[nn.Module] = nn.ConvTranspose1d if transposed else nn.Conv1d
    local_class: type[nn.Module] = CPUTranspose if transposed else CPUConv
    reference = reference_class(2, 2, 3, padding=1, dtype=torch.complex64)
    local = local_class(2, 2, 3, padding=1, dtype=torch.complex64)
    local.load_state_dict(reference.state_dict(), strict=True)
    inputs = torch.ones(1, 2, 5, dtype=torch.complex64) * (0.5 + 0.1j)
    torch.testing.assert_close(local(inputs), reference(inputs), rtol=0, atol=0)


def test_all_85_hift_convolutions_use_model_owned_classes_without_extra_state():
    from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.flashcosyvoice.modules.hifigan import HiFTGenerator
    from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.flashcosyvoice.modules.hifigan_components.f0_math import (
        F0Conv1d,
    )
    from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.flashcosyvoice.modules.hifigan_components.preconv_math import (
        PreconvConv1d,
    )

    model = HiFTGenerator().eval()
    convolutions = [module for module in model.modules() if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d))]
    assert len(convolutions) == 85
    assert sum(isinstance(module, Conv1d) for module in convolutions) == 82
    assert sum(isinstance(module, ConvTranspose1d) for module in convolutions) == 3
    f0_convolutions = [module for module in convolutions if isinstance(module, F0Conv1d)]
    assert len(f0_convolutions) == 5
    assert set(f0_convolutions) == {module for module in model.f0_predictor.modules() if isinstance(module, nn.Conv1d)}
    assert all(module._conv_forward.__module__.endswith("hifigan_components.f0_math") for module in f0_convolutions)
    preconv = [module for module in convolutions if isinstance(module, PreconvConv1d)]
    assert preconv == [model.conv_pre]
    assert preconv[0]._conv_forward.__module__.endswith("hifigan_components.preconv_math")
    other_convolutions = [module for module in convolutions if not isinstance(module, (F0Conv1d, PreconvConv1d))]
    assert len(other_convolutions) == 79
    assert all(
        (module.forward if isinstance(module, ConvTranspose1d) else module._conv_forward).__module__.endswith(
            "hifigan_components.convolution"
        )
        for module in other_convolutions
    )
    state = model.state_dict()
    assert len(state) == 328 and sum(value.numel() for value in state.values()) == 20821295
    assert "conv_pre.parametrizations.weight.original0" in state
    assert "f0_predictor.condnet.0.parametrizations.weight.original1" in state
    assert all(value.dtype == torch.float32 for value in state.values())
