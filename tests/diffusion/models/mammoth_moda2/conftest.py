# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest


@pytest.fixture(autouse=True)
def explicit_cpu_attention_backend(monkeypatch, request):
    """Let L1 run without GPU discovery; do not replace attention arithmetic.

    The unspecified platform has neither a backend selector nor a forward
    dispatch branch. Resolve explicit names to their real implementation and
    use SDPA's existing broadcast-key entrypoint on CPU tensors. No projection,
    mask, GQA, RoPE or kernel arithmetic is replaced. GPU tests retain both
    production platform selection and forward dispatch.
    """
    from vllm_omni.diffusion.attention import selector
    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
    from vllm_omni.platforms import current_omni_platform

    if request.node.get_closest_marker("cpu") is None or current_omni_platform.device_type != "cpu":
        return

    def resolve_explicit_backend(name, head_size):
        assert name is not None, "CPU contracts must pin an explicit backend"
        return DiffusionAttentionBackendEnum[name].get_class()

    monkeypatch.setattr(selector, "_cached_get_backend_cls", resolve_explicit_backend)
    monkeypatch.setattr(SDPAImpl, "forward", SDPAImpl.forward_cuda)


@pytest.fixture
def mock_tp1(monkeypatch):
    """Allow standalone linear-layer tests without a distributed process group."""
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)


@pytest.fixture(autouse=True)
def force_default_gemm(monkeypatch, request):
    """Force CPU-compatible GEMM dispatch for tests using CPU tensors."""
    if request.node.get_closest_marker("cpu") is None:
        return

    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
