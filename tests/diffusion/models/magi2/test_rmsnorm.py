# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import patch

import pytest
import torch

from tests.diffusion.models.magi2.test_native_preview import _initialize_tiny_model, _tiny_config
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.layers import ModalityDispatcher, MultiModalityRMSNorm
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2PreviewTransformer, Modality

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def _reference(module, tensor, modality_dispatcher=None):
    normalized = tensor.float()
    normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + module.eps)
    if module.num_modality == 1:
        result = normalized * (module.weight.view(module.num_patterns, module.dim) + 1.0)
    else:
        if modality_dispatcher is None:
            raise ValueError("modality_dispatcher is required for multimodal RMSNorm")
        parts = modality_dispatcher.dispatch(normalized)
        weights = module.weight.view(module.num_modality, module.num_patterns, module.dim)
        result = modality_dispatcher.undispatch(*(part * (weights[i] + 1.0) for i, part in enumerate(parts)))
    return result.to(module.out_dtype or tensor.dtype)


def _fixture(dtype, output_dtype, modalities=1, patterns=1, width=128, count=7, device="cpu", seed=91):
    generator = torch.Generator().manual_seed(seed)
    module = MultiModalityRMSNorm(width, num_modality=modalities, num_patterns=patterns, out_dtype=output_dtype)
    with torch.no_grad():
        module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * 0.1)
    module = module.to(device)
    tensor = torch.randn(count, patterns, width * 2, generator=generator).to(device=device, dtype=dtype)[..., ::2]
    # One modality can be empty; order matches the packed dispatch contract.
    counts = [count] if modalities == 1 else [count // 2, 0, count - count // 2]
    mapping = torch.tensor(
        [index for index, size in enumerate(counts) for _ in range(size)], dtype=torch.int64, device=device
    )
    return module, tensor, ModalityDispatcher(mapping, modalities)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "dtype,output_dtype",
    [(torch.float32, None), (torch.bfloat16, None), (torch.float16, None), (torch.bfloat16, torch.float32)],
)
@pytest.mark.parametrize("modalities,patterns,count", [(1, 1, 7), (3, 1, 7), (3, 4, 7), (3, 4, 0)])
def test_reference_parity(dtype, output_dtype, modalities, patterns, count):
    module, tensor, dispatch = _fixture(dtype, output_dtype, modalities, patterns, count=count)
    original_tensor, original_weight = tensor.clone(), module.weight.detach().clone()
    actual = module(tensor, dispatch)
    torch.testing.assert_close(actual, _reference(module, tensor, dispatch), rtol=1e-6, atol=1e-6)
    assert actual.dtype == (output_dtype or dtype)
    torch.testing.assert_close(tensor, original_tensor, rtol=0, atol=0)
    torch.testing.assert_close(module.weight, original_weight, rtol=0, atol=0)


@pytest.mark.cpu
def test_operator_gets_fp32_without_affine_weight():
    module, tensor, dispatch = _fixture(torch.bfloat16, torch.float32, 3, 4)
    with patch.object(torch.nn.functional, "rms_norm", wraps=torch.nn.functional.rms_norm) as operator:
        module(tensor, dispatch)
    operator.assert_called_once()
    normalized_input, shape, weight, eps = operator.call_args.args
    assert normalized_input.dtype == torch.float32
    assert shape == (128,) and weight is None and eps == module.eps


@pytest.mark.cpu
@pytest.mark.parametrize("case", ["scalar", "broadcast", "zero_width", "zero_values", "nan_inf", "no_dispatcher"])
def test_special_values_and_shape_contracts(case):
    module, tensor, dispatch = _fixture(torch.float32, None)
    if case == "scalar":
        tensor = torch.tensor(2.0)
    elif case == "broadcast":
        tensor = tensor[..., :1]
    elif case == "zero_width":
        module, tensor, dispatch = _fixture(torch.float32, None, width=0)
    elif case == "zero_values":
        tensor.zero_()
    elif case == "nan_inf":
        tensor[0, 0, 0], tensor[1, 0, 0] = torch.nan, torch.inf
    else:
        module.num_modality = 3
        with pytest.raises(ValueError, match="modality_dispatcher is required"):
            module(tensor)
        return
    torch.testing.assert_close(
        module(tensor, dispatch), _reference(module, tensor, dispatch), rtol=1e-6, atol=1e-6, equal_nan=True
    )


@pytest.mark.cpu
def test_gradients_and_checkpoint_scale_updates():
    module, tensor, dispatch = _fixture(torch.float32, None, 3, 4)
    tensor = tensor.clone().requires_grad_()
    expected = _reference(module, tensor, dispatch)
    actual = module(tensor, dispatch)
    for grad, ref in zip(
        torch.autograd.grad(actual.square().sum(), (tensor, module.weight)),
        torch.autograd.grad(expected.square().sum(), (tensor, module.weight)),
    ):
        torch.testing.assert_close(grad, ref, rtol=2e-5, atol=2e-5)
    keys = list(module.state_dict())
    with torch.no_grad():
        module.weight.add_(0.03)
    torch.testing.assert_close(module(tensor, dispatch), _reference(module, tensor, dispatch), rtol=1e-6, atol=1e-6)
    clone = MultiModalityRMSNorm(128, num_modality=3, num_patterns=4)
    clone.load_state_dict(module.state_dict())
    assert keys == ["weight"]
    torch.testing.assert_close(clone(tensor, dispatch), module(tensor, dispatch), rtol=0, atol=0)


@pytest.mark.cpu
def test_low_precision_weights_keep_offset_rounding_order():
    module, tensor, dispatch = _fixture(torch.bfloat16, torch.float32, 3, 4)
    module = module.bfloat16()
    torch.testing.assert_close(module(tensor, dispatch), _reference(module, tensor, dispatch), rtol=1e-6, atol=1e-6)


@pytest.mark.cpu
def test_compile_frontend_and_weight_changes():
    module, tensor, dispatch = _fixture(torch.bfloat16, torch.float32, 3, 4)
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(module, backend=backend, fullgraph=True)
    try:
        for _ in range(2):
            torch.testing.assert_close(
                compiled(tensor, dispatch), _reference(module, tensor, dispatch), rtol=1e-6, atol=1e-6
            )
            with torch.no_grad():
                module.weight.add_(0.125)
        assert graphs
    finally:
        torch._dynamo.reset()


@pytest.mark.cpu
def test_tiny_model_parity():
    model = Magi2PreviewTransformer(_tiny_config(num_layers=2)).eval()
    _initialize_tiny_model(model, seed=7)
    generator = torch.Generator().manual_seed(11)
    packed, coords = torch.randn(6, 4, generator=generator), torch.ones(6, 9)
    modalities = torch.tensor(
        [Modality.VIDEO, Modality.VIDEO, Modality.AUDIO, Modality.AUDIO, Modality.TEXT, Modality.TEXT]
    )
    cu = torch.tensor([0, 6], dtype=torch.int32)
    with torch.no_grad(), patch.object(MultiModalityRMSNorm, "forward", _reference):
        expected = model(packed, coords, modalities, VarlenHandler(cu, cu, 6, 6))
    with torch.no_grad(), patch.object(torch.nn.functional, "rms_norm", wraps=torch.nn.functional.rms_norm) as operator:
        actual = model(packed, coords, modalities, VarlenHandler(cu, cu, 6, 6))
    assert operator.call_count > 0
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.musa
@pytest.mark.parametrize(
    "dtype,output_dtype",
    [(torch.float32, None), (torch.bfloat16, None), (torch.float16, None), (torch.bfloat16, torch.float32)],
)
def test_musa_rmsnorm_matches_reference(dtype, output_dtype):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    for seed in (91, 92, 93):
        for modalities, patterns, count, width in ((1, 1, 7, 128), (3, 4, 7, 1536), (3, 4, 0, 128), (3, 1, 9, 768)):
            module, tensor, dispatch = _fixture(
                dtype, output_dtype, modalities, patterns, width, count, device="musa", seed=seed
            )
            if patterns == 1:
                tensor = tensor.expand(-1, 6, -1)  # One pattern shared by Q/K heads.
            with (
                torch.no_grad(),
                patch.object(torch.nn.functional, "rms_norm", wraps=torch.nn.functional.rms_norm) as operator,
            ):
                actual = module(tensor, dispatch)
            operator.assert_called_once()
            assert operator.call_args.args[0].dtype == torch.float32
            with torch.no_grad():
                expected = _reference(module, tensor, dispatch)
                module.out_dtype = torch.float32
                before_cast = module(tensor, dispatch)
                reference_fp32 = _reference(module, tensor, dispatch)
            torch.testing.assert_close(before_cast, reference_fp32, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(actual, before_cast.to(actual.dtype), rtol=0, atol=0)
            torch.testing.assert_close(expected, reference_fp32.to(expected.dtype), rtol=0, atol=0)
            if actual.dtype == torch.float32:
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            else:
                # FP32 reduction differences can cross a low-precision rounding
                # midpoint. Bound that final rounding to one ULP, in addition
                # to the independent FP32 and cast-order gates above.
                reference_cpu = expected.cpu()
                up = torch.nextafter(reference_cpu, torch.full_like(reference_cpu, float("inf"))).float()
                down = torch.nextafter(reference_cpu, torch.full_like(reference_cpu, -float("inf"))).float()
                ulp = torch.maximum(up - reference_cpu.float(), reference_cpu.float() - down)
                assert torch.all((actual.cpu().float() - reference_cpu.float()).abs() <= ulp)
