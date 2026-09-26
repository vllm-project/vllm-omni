# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def sage_backend(monkeypatch):
    package = types.ModuleType("sageattention")
    package.sageattn = lambda *args, **kwargs: pytest.fail("unexpected Sage kernel call")
    monkeypatch.setitem(sys.modules, "sageattention", package)
    module_name = "vllm_omni.diffusion.attention.backends.sage_attn"
    sys.modules.pop(module_name, None)
    try:
        yield importlib.import_module(module_name), package
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("head_size", [64, 96], ids=["dense", "padded_output"])
def test_sage_attention_dispatcher_is_opaque_to_compile(sage_backend, tmp_path, head_size):
    backend, package = sage_backend
    marker = tmp_path / "kernel"
    marker.write_text("loaded")
    calls = []

    def kernel(q, k, v, *, tensor_layout, is_causal, sm_scale):
        # File I/O must not be traced, just like Sage's architecture queries
        # and pybind quantization kernels. A padded result exercises strides.
        assert marker.read_text() == "loaded"
        calls.append((tensor_layout, is_causal, sm_scale))
        output = q + k + v
        if head_size == 96:
            output = torch.nn.functional.pad(output, (0, 32))[..., :head_size]
        return output

    package.sageattn = kernel
    scale = 0.17
    impl = backend.SageAttentionImpl(4, head_size, scale, causal=False)
    q, k, v = [torch.randn(1, 12, 4, head_size) for _ in range(3)]
    expected = impl.forward_cuda(q, k, v)
    compiled = torch.compile(impl.forward_cuda, fullgraph=True)
    actual = compiled(q, k, v)
    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()
    assert calls == [("NHD", False, scale)] * 2
    if head_size == 96:
        torch.library.opcheck(backend._sage_attention_op.default, (q, k, v, False, scale))


def test_sage_attention_rejects_mask_instead_of_ignoring_it(monkeypatch):
    fake_package = types.ModuleType("sageattention")
    fake_package.sageattn = lambda *args, **kwargs: pytest.fail("unexpected Sage kernel call")
    monkeypatch.setitem(sys.modules, "sageattention", fake_package)
    module_name = "vllm_omni.diffusion.attention.backends.sage_attn"
    sys.modules.pop(module_name, None)
    try:
        backend_module = importlib.import_module(module_name)
        impl = backend_module.SageAttentionImpl(
            num_heads=4,
            head_size=64,
            softmax_scale=1.0 / 8.0,
            causal=False,
        )
        query = torch.randn(1, 2, 4, 64)
        metadata = AttentionMetadata(attn_mask=torch.ones(1, 2, dtype=torch.bool))

        with pytest.raises(ValueError, match="does not support attn_mask"):
            impl.forward_cuda(query, query, query, metadata)
    finally:
        sys.modules.pop(module_name, None)
