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
