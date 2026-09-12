# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""V2 dispatch, caller integration and strict CUDA rounding acceptance tests."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.diffusion.layers import test_gated_residual_adaln as original_tests
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import gated_residual_adaln as fusion

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def experimental(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_QWEN_ADALN_V2", "1")


@pytest.mark.cpu
@pytest.mark.parametrize(
    "case",
    [
        "short",
        "image",
        "batch",
        "fp32",
        "row_slice",
        "offset",
        "grad",
        "empty",
        "fp16",
        "channels",
        "token_mod",
        "mixed_dtype",
        "disabled",
        "norm2_only",
    ],
)
def test_new_sites_dispatch(monkeypatch, case):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((grid, kwargs))

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(fusion, "HAS_TRITON", True)
    monkeypatch.setattr(fusion.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(fusion, "triton", SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()))
    monkeypatch.setattr(fusion, "_has_measured_speedup", lambda *args: False)
    monkeypatch.setattr(fusion, "_gated_residual_cast_kernel", Kernel(), raising=False)
    monkeypatch.setattr(fusion, "_cast_modulate_kernel", Kernel(), raising=False)
    if case in ("disabled", "norm2_only"):
        monkeypatch.setenv("VLLM_OMNI_QWEN_ADALN_V2", "0" if case == "disabled" else "norm2")
    with FakeTensorMode():
        b, s, d = 2 if case == "batch" else 1, 0 if case == "empty" else 4096 if case == "image" else 29, 3072
        dtype = torch.float32 if case == "fp32" else torch.float16 if case == "fp16" else torch.bfloat16
        x = torch.empty(b, s, d, dtype=dtype, requires_grad=case == "grad")
        branch = torch.empty(b, s, 3 * d, dtype=dtype)[..., d : 2 * d]
        if case == "row_slice":
            x = torch.empty(b, 2 * s, d, dtype=dtype)[:, ::2]
        if case == "offset":
            x = torch.empty(b, s + 1, d, dtype=dtype)[:, 1:]
        gate, scale, shift = [t.unsqueeze(1) for t in torch.empty(b, 6 * d, dtype=dtype).chunk(6, -1)[:3]]
        if case == "channels":
            x, branch, gate, scale, shift = [t[..., ::2] for t in (x, branch, gate, scale, shift)]
        if case == "token_mod":
            gate = gate.expand(b, s, d)
            scale, shift = scale.expand_as(gate), shift.expand_as(gate)
        if case == "mixed_dtype":
            gate, scale, shift = gate.double(), scale.double(), shift.double()
        norm1 = fusion.try_fused_native_adaln(x, scale, shift, 1e-6)
        norm2 = fusion.try_fused_gated_residual_adaln(x, branch, gate, scale, shift, 1e-6)
        final = fusion.try_fused_gated_residual(x, branch, gate)
        eligible = case in ("short", "image", "batch", "fp32", "row_slice", "offset")
        assert (norm1 is not None) == eligible
        assert (final is not None) == eligible
        assert (norm2 is not None) == (eligible or case == "norm2_only")
        assert len(calls) == (4 if eligible else 2 if case == "norm2_only" else 0)
        if eligible:
            assert norm1.shape == final.shape == x.shape
            assert norm1.is_contiguous() and final.is_contiguous()
            # The short text case has 12 tiles/row rather than one 4096-wide
            # program. This asserts the proposed launch change is reached.
            if case == "short":
                assert all(grid == (29 * 12,) for grid, _ in calls)


@pytest.mark.cpu
@pytest.mark.parametrize("case", ["t2i", "indexed", "zero_cond"])
def test_real_block_forward_sites_and_fallback(case):
    # Execute the repository's actual forward and modulation methods, with
    # tiny stand-ins for attention/MLPs. No vLLM runtime or CUDA is simulated.
    # This isolates wiring (stream, gate, norm eps, fallback) from kernel math.
    path = Path(fusion.__file__).parents[1] / "models/qwen_image/qwen_image_transformer.py"
    tree = ast.parse(path.read_text())
    block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenImageTransformerBlock")
    methods: list[ast.stmt] = [
        n for n in block.body if isinstance(n, ast.FunctionDef) and n.name in ("forward", "_modulate")
    ]
    namespace = {"torch": torch, "Any": Any}
    namespace.update(
        dict.fromkeys(
            ("try_paired_native_adaln", "try_paired_gated_residual_adaln", "try_paired_gated_residual"),
            lambda *args: None,
        )
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    calls = []

    class Norm:
        eps = 1e-6

        def __call__(self, x, scale, shift):
            return F.layer_norm(x.float(), (x.shape[-1],), eps=self.eps).to(x.dtype) * (1 + scale) + shift

    obj = SimpleNamespace(
        zero_cond_t=case == "zero_cond",
        img_mod=lambda x: x,
        txt_mod=lambda x: x,
        img_norm1=Norm(),
        img_norm2=Norm(),
        txt_norm1=Norm(),
        txt_norm2=Norm(),
        img_mlp=lambda x: x * 0.25,
        txt_mlp=lambda x: x * 0.5,
        attn=lambda **kw: (kw["hidden_states"] * 0.3, kw["encoder_hidden_states"] * 0.4),
    )
    obj._modulate = namespace["_modulate"].__get__(obj)
    forward = namespace["forward"].__get__(obj)
    d = 8
    image = torch.randn(1 if case != "zero_cond" else 2, 3, d)
    text = torch.randn(2 if case == "indexed" else 1, 2, d)
    temb = torch.randn(1 if case == "t2i" else 2, 6 * d)
    index = torch.tensor([[0, 1, 0]]) if case == "indexed" else None
    kwargs = dict(
        hidden_states=image,
        encoder_hidden_states=text,
        encoder_hidden_states_mask=None,
        temb=temb,
        image_rotary_emb=(None, None),
        modulate_index=index,
    )
    names = ("try_fused_native_adaln", "try_fused_gated_residual_adaln", "try_fused_gated_residual")
    namespace.update(dict.fromkeys(names, lambda *args: None))
    expected = forward(**kwargs)

    def norm1(x, scale, shift, eps):
        calls.append("norm1")
        assert eps == 1e-6
        return Norm()(x, scale, shift)

    def norm2(x, branch, gate, scale, shift, eps):
        calls.append("norm2")
        assert eps == 1e-6
        r = x + gate * branch
        return r, Norm()(r, scale, shift)

    def residual(x, branch, gate):
        calls.append("final")
        return x + gate * branch

    namespace.update(zip(names, (norm1, norm2, residual)))
    actual = forward(**kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert calls == (["norm1", "norm1", "norm2", "final", "norm2", "final"] if case == "t2i" else [])


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "b,s,d",
    [
        (1, 1, 257),
        (2, 3, 8192),
        (1, 12, 3072),
        (1, 29, 3072),
        (2, 63, 3072),
        (1, 64, 3072),
        (1, 4096, 3072),
        (2, 4608, 3072),
    ],
)
def test_all_sites_exact(dtype, b, s, d):
    inputs = original_tests._inputs(b, s, d, dtype, "cuda")
    r, branch, gate, scale, shift = inputs
    original_tests._checked_call(inputs)
    before = [x.clone() for x in inputs]
    expected = F.layer_norm(r.float(), (d,), eps=1e-6).to(dtype) * (1 + scale) + shift
    actual = fusion.try_fused_native_adaln(r, scale, shift, 1e-6)
    final = fusion.try_fused_gated_residual(r, branch, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(final, r + gate * branch, rtol=0, atol=0, equal_nan=True)
    assert actual.data_ptr() != final.data_ptr()
    for x, original in zip(inputs, before):
        torch.testing.assert_close(x, original, atol=0, rtol=0, equal_nan=True)
        assert actual.data_ptr() != x.data_ptr() and final.data_ptr() != x.data_ptr()


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "case", ["zero_gate", "minus_one_scale", "constant", "tiny_variance", "small", "large", "nan", "inf"]
)
def test_v2_numerical_boundaries(dtype, case):
    original_tests.test_numerical_boundaries(dtype, case, None)


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("seed", [142, 143, 144])
@pytest.mark.parametrize("seq_len", [12, 29, 4096])
def test_v2_previous_rounding_regression(seed, seq_len):
    original_tests.test_qwen_layer_norm_rounding(seed, seq_len, None)


@hardware_test(res={"cuda": "L4"})
def test_all_sites_compile_and_graph():
    inputs = original_tests._inputs(2, 29, 3072, torch.bfloat16, "cuda")

    def chain(r, branch, gate, scale, shift):
        norm1 = fusion.try_fused_native_adaln(r, scale, shift, 1e-6)
        r, norm2 = fusion.try_fused_gated_residual_adaln(r, branch, gate, scale, shift, 1e-6)
        final = fusion.try_fused_gated_residual(r, branch, gate)
        return norm1, r, norm2, final

    reference = chain(*inputs)
    torch.testing.assert_close(torch.compile(chain, fullgraph=True)(*inputs), reference, atol=0, rtol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            chain(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = chain(*inputs)
    inputs[0].add_(0.5)
    graph.replay()
    torch.testing.assert_close(result, chain(*inputs), atol=0, rtol=0)
