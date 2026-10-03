# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.diffusion.models.magi2.test_native_preview import _initialize_tiny_model, _tiny_config
from tests.diffusion.models.magi2.test_pipeline_magi2 import _pipeline, _request
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.magi2 import attention
from vllm_omni.diffusion.models.magi2 import mh_moe as moe
from vllm_omni.diffusion.models.magi2 import pipeline_magi2 as pipeline
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2PreviewTransformer, Modality

pytestmark = [pytest.mark.diffusion, pytest.mark.local_model]


def fake_platform(device_type: str, available: bool = True):
    return SimpleNamespace(
        device_type=device_type,
        is_available=lambda: available,
        is_cuda=lambda: device_type == "cuda",
        is_musa=lambda: device_type == "musa",
    )


def config():
    return OmniDiffusionConfig(
        model="/unused/checkpoint",
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2, ulysses_degree=2),
    )


class ResolvedDeviceError(Exception):
    pass


def stop_before_weights(monkeypatch):
    resolver = Mock(side_effect=ResolvedDeviceError)
    monkeypatch.setattr(pipeline, "_resolve_checkpoint_root", resolver)
    return resolver


@pytest.mark.cpu
@pytest.mark.parametrize("device_type", ["cuda", "musa"])
def test_constructor_uses_selected_device(monkeypatch, device_type):
    monkeypatch.setattr(pipeline, "current_omni_platform", fake_platform(device_type))
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)
    resolver = stop_before_weights(monkeypatch)
    pipe = pipeline.Magi2Pipeline.__new__(pipeline.Magi2Pipeline)
    with pytest.raises(ResolvedDeviceError):
        pipe.__init__(od_config=config())
    assert pipe.device_str == f"{device_type}:2"
    resolver.assert_called_once()


@pytest.mark.cpu
@pytest.mark.parametrize("device_type,available", [("cuda", False), ("musa", False), ("cpu", True), ("xpu", True)])
def test_constructor_rejects_unsupported_or_unavailable(monkeypatch, device_type, available):
    monkeypatch.setattr(pipeline, "current_omni_platform", fake_platform(device_type, available))
    resolver = stop_before_weights(monkeypatch)
    with pytest.raises(RuntimeError, match="available CUDA or MUSA"):
        pipeline.Magi2Pipeline(od_config=config())
    resolver.assert_not_called()


@pytest.mark.cpu
@pytest.mark.parametrize("device_type", ["cuda", "musa"])
@pytest.mark.parametrize("profiling", [False, True])
def test_forward_sync_and_memory_monitor(monkeypatch, device_type, profiling):
    monkeypatch.setattr(pipeline, "current_omni_platform", fake_platform(device_type))
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)
    synchronize = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    monitor = Mock(peak_bytes=9 * 1024**2)
    factory = Mock(return_value=monitor)
    monkeypatch.setattr(pipeline, "_PeakReservedMonitor", factory)
    pipe, runtime = _pipeline()
    pipe._profiler_lock = Lock()
    pipe._stage_durations = {}
    pipe.enable_diffusion_pipeline_profiler = profiling
    result = pipe(_request("A fox walks through snow"))
    assert len(runtime.calls) == 1
    assert synchronize.call_count == (2 if profiling else 1)
    if profiling:
        factory.assert_called_once_with(2)
        monitor.start.assert_called_once()
        monitor.stop.assert_called_once()
        assert result.peak_memory_mb == 9
    else:
        factory.assert_not_called()
        assert result.peak_memory_mb == 0


@pytest.mark.cpu
def test_musa_monitor_stops_if_inference_fails(monkeypatch):
    monkeypatch.setattr(pipeline, "current_omni_platform", fake_platform("musa"))
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    monitor = Mock(peak_bytes=0)
    monkeypatch.setattr(pipeline, "_PeakReservedMonitor", Mock(return_value=monitor))
    pipe, runtime = _pipeline()
    pipe.enable_diffusion_pipeline_profiler = True
    runtime.evaluate = Mock(side_effect=RuntimeError("inference failed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        pipe(_request("A fox walks through snow"))
    monitor.stop.assert_called_once()


def require_musa():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")


@pytest.mark.cpu
@pytest.mark.parametrize("device_type,is_cuda", [("cuda", True), ("musa", True), ("cuda", False)])
def test_bundled_flash_attention_requires_cuda_platform(monkeypatch, device_type, is_cuda):
    monkeypatch.setattr(attention, "current_omni_platform", fake_platform(device_type))
    # Compatibility layers can expose is_cuda on non-CUDA tensors.
    q = SimpleNamespace(is_cuda=is_cuda, shape=(2, 1, 4), device=torch.device("cpu"))
    k = v = torch.empty(2, 1, 4)
    cu = torch.tensor([0, 2], dtype=torch.int32)
    varlen = VarlenHandler(cu, cu, 2, 2)
    expected = torch.empty_like(k)
    reference = Mock(return_value=expected)
    flash = Mock(return_value=(expected, torch.zeros(1, 2)))
    resolver = Mock(return_value=3)
    monkeypatch.setattr(attention, "torch_varlen_attention_with_sink", reference)
    monkeypatch.setattr(attention, "vllm_flash_attn_varlen_with_lse", flash)
    monkeypatch.setattr(attention, "_resolve_flash_attn_version", resolver)
    assert attention.packed_attention_with_sink(q, k, v, varlen) is expected
    if device_type == "cuda" and is_cuda:
        flash.assert_called_once()
        resolver.assert_called_once()
        reference.assert_not_called()
    else:
        reference.assert_called_once()
        flash.assert_not_called()
        resolver.assert_not_called()


@pytest.mark.cpu
@pytest.mark.parametrize("platform,is_cuda", [("musa", True), ("cuda", True), ("rocm", True), ("musa", False)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("requested", [None, "0", "1"])
def test_moe_bf16_musa_uses_non_atomic_output(monkeypatch, platform, is_cuda, dtype, requested):
    monkeypatch.setattr(moe, "current_omni_platform", fake_platform(platform))
    if requested is None:
        monkeypatch.delenv("MAGI2_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("MAGI2_DETERMINISTIC", requested)
    layer = moe.Magi2MultiHeadMoE(moe.Magi2MultiHeadMoEConfig(4, 1, 2, 1, 8, dtype))
    x = SimpleNamespace(is_cuda=is_cuda, dtype=dtype)
    probabilities, indices = torch.ones(1, 1, 1), torch.zeros(1, 1, 1, dtype=torch.long)
    monkeypatch.setattr(layer, "_route", Mock(return_value=(probabilities, indices)))
    monkeypatch.setattr(moe, "global_sort_routes", Mock(return_value=(object(), object(), object())))
    expected = object()
    triton_path, torch_path = Mock(return_value=expected), Mock(return_value=expected)
    monkeypatch.setattr(moe, "triton_mh_moe_forward", triton_path)
    monkeypatch.setattr(moe, "torch_mh_moe_forward", torch_path)
    assert layer._local_forward(x) is expected
    if is_cuda:
        triton_path.assert_called_once()
        expected_non_atomic = requested == "1" or (platform == "musa" and dtype == torch.bfloat16)
        assert triton_path.call_args.kwargs == {"deterministic": expected_non_atomic}
        torch_path.assert_not_called()
    else:
        torch_path.assert_called_once()
        triton_path.assert_not_called()


@pytest.mark.musa
@pytest.mark.parametrize("tokens,heads", [(2, 3), (129, 1)])
def test_actual_musa_bf16_moe_default_dispatch(monkeypatch, tokens, heads):
    """Non-atomic dispatch/kernel smoke with exact nonzero output, not checkpoint accuracy."""
    require_musa()
    assert moe.current_omni_platform.is_musa()
    monkeypatch.delenv("MAGI2_DETERMINISTIC", raising=False)
    layer = moe.Magi2MultiHeadMoE(moe.Magi2MultiHeadMoEConfig(heads * 256, heads, 8, 6, 1280, torch.bfloat16))
    with torch.no_grad():
        layer.gate.zero_()
        layer.W_gate.fill_(1)
        layer.W_up.fill_(1)
        layer.W_down.fill_(3 / 2**17)
    layer.to("musa").eval()
    original_triton = moe.triton_mh_moe_forward
    selected_modes = []

    def checked_triton(*args, **kwargs):
        selected_modes.append(kwargs.get("deterministic"))
        assert kwargs.get("deterministic") is True
        return original_triton(*args, **kwargs)

    monkeypatch.setattr(moe, "triton_mh_moe_forward", checked_triton)
    x = torch.ones(tokens, heads, 256, device="musa", dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = layer._local_forward(x)
    assert selected_modes == [True]
    assert actual.shape == x.shape and actual.dtype == x.dtype
    # Clamped SwiGLU7 stores 56 in BF16. Six equal routes each store 35/128,
    # so their BF16 sum is exactly 105/64, independently of addition order.
    expected = torch.full_like(actual.cpu(), 105 / 64)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.musa
def test_actual_musa_constructor_selects_device(monkeypatch):
    require_musa()
    assert pipeline.current_omni_platform.is_musa()
    stop_before_weights(monkeypatch)
    pipe = pipeline.Magi2Pipeline.__new__(pipeline.Magi2Pipeline)
    with pytest.raises(ResolvedDeviceError):
        pipe.__init__(od_config=config())
    assert pipe.device_str == f"musa:{torch.accelerator.current_device_index()}"


@pytest.mark.musa
def test_request_seed_reaches_musa_rng():
    require_musa()
    pipeline._seed_request(1927)
    first = torch.randn(1024, device="musa")
    pipeline._seed_request(1927)
    second = torch.randn(1024, device="musa")
    pipeline._seed_request(1928)
    different = torch.randn(1024, device="musa")
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, different)


@pytest.mark.musa
def test_musa_tiny_native_transformer_matches_cpu():
    """Component smoke, not a full checkpoint/video or optimized FA3 test."""
    require_musa()
    cpu_model = Magi2PreviewTransformer(_tiny_config())
    _initialize_tiny_model(cpu_model, 119)
    gpu_model = Magi2PreviewTransformer(_tiny_config())
    gpu_model.load_state_dict(cpu_model.state_dict())
    gpu_model.to("musa")
    generator = torch.Generator().manual_seed(13)
    packed = torch.randn(6, 4, generator=generator)
    coords = torch.ones(6, 9)
    modalities = torch.tensor(
        [Modality.VIDEO, Modality.VIDEO, Modality.AUDIO, Modality.AUDIO, Modality.TEXT, Modality.TEXT]
    )
    cu = torch.tensor([0, 6], dtype=torch.int32)
    cpu_varlen = VarlenHandler(cu, cu, 6, 6)
    gpu_varlen = VarlenHandler(cu.to("musa"), cu.to("musa"), 6, 6)
    with torch.inference_mode():
        expected = cpu_model(packed, coords, modalities, cpu_varlen)
        actual = gpu_model(packed.to("musa"), coords.to("musa"), modalities.to("musa"), gpu_varlen)
    torch.musa.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)


@pytest.mark.musa
def test_actual_musa_forward_instrumentation():
    require_musa()
    pipe, runtime = _pipeline()
    pipe._profiler_lock = Lock()
    pipe._stage_durations = {}
    pipe.enable_diffusion_pipeline_profiler = True
    allocation = torch.ones(1024, device="musa")
    result = pipe(_request("A fox walks through snow"))
    assert len(runtime.calls) == 1
    assert result.peak_memory_mb > 0
    assert allocation.sum().item() == 1024
