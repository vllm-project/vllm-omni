# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pair dispatch/caller contracts on CPU and exact CUDA acceptance cases."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import gated_residual_adaln as single
from vllm_omni.diffusion.layers import paired_adaln as pair

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _inputs(b, s, d, dtype, device, stride=1):
    x = torch.randn(b, s * stride + 1, d, device=device, dtype=dtype)[:, 1::stride]
    branch = torch.randn(b, s, 3 * d, device=device, dtype=dtype)[..., d : 2 * d]
    modulation = torch.randn(b, 6 * d, device=device, dtype=dtype)
    gate, scale, shift = (t.unsqueeze(1) for t in modulation.chunk(6, -1)[:3])
    return x, branch, gate, scale, shift


def _norm(x, scale, shift, eps):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype) * (1 + scale) + shift


def _reference(left, right):
    x0, b0, g0, s0, t0 = left
    x1, b1, g1, s1, t1 = right
    r0, r1 = x0 + g0 * b0, x1 + g1 * b1
    return (
        (_norm(x0, s0, t0, 1e-6), _norm(x1, s1, t1, 1e-4)),
        (r0, _norm(r0, s0, t0, 1e-6), r1, _norm(r1, s1, t1, 1e-4)),
        (r0, r1),
    )


def _call(left, right):
    x0, b0, g0, s0, t0 = left
    x1, b1, g1, s1, t1 = right
    return (
        pair.try_paired_native_adaln(x0, x1, s0, t0, s1, t1, 1e-6, 1e-4),
        pair.try_paired_gated_residual_adaln(left, right, 1e-6, 1e-4),
        pair.try_paired_gated_residual(x0, b0, g0, x1, b1, g1),
    )


@pytest.fixture
def cpu_launches(monkeypatch):
    """Execute host wrappers; replace CUDA launches with explicit CPU oracles.

    This checks shapes, strides, epsilon, storage and dispatch. It does not
    interpret PTX or claim numerical validation of a CUDA kernel.
    """
    events = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                events.append((self.name, grid))
                if self.name == "cast":
                    x0, x1, out0, out1 = args[:4]
                    out0.copy_(x0.float())
                    out1.copy_(x1.float())
                elif self.name == "residual":
                    for offset in (0, 5):
                        x, branch, gate, out, norm = args[offset : offset + 5]
                        out.copy_(x + gate * branch)
                        if args[-1]:
                            norm.copy_(out.float())
                else:
                    for offset in (0, 4):
                        normalized, scale, shift, out = args[offset : offset + 4]
                        out.copy_(normalized.to(out.dtype) * (1 + scale) + shift)

            return launch

    monkeypatch.setenv("VLLM_OMNI_QWEN_ADALN_PAIR", "1")
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(single, "HAS_TRITON", True)
    monkeypatch.setattr(single.current_platform, "is_cuda", lambda: True)
    for name, attr in (
        ("cast", "_pair_cast_input_kernel"),
        ("residual", "_pair_residual_kernel"),
        ("modulate", "_pair_modulate_kernel"),
    ):
        monkeypatch.setattr(pair, attr, Kernel(name), raising=False)
    native = pair._native_norm_boundary

    def norm(x, eps):
        events.append(("native_norm", (tuple(x.shape), eps, x.dtype)))
        return native(x, eps)

    monkeypatch.setattr(pair, "_native_norm_boundary", norm)
    return events


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,stride", [(1, 1), (2, 2)])
def test_pair_wrappers_native_shapes_and_storage(cpu_launches, dtype, batch, stride):
    left = _inputs(batch, 67, 257, dtype, "cpu", stride)
    right = _inputs(batch, 3, 129, dtype, "cpu", stride)
    before = [t.clone() for t in (*left, *right)]
    with torch.inference_mode():
        expected, actual = _reference(left, right), _call(left, right)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert [event[1] for event in cpu_launches if event[0] == "native_norm"] == [
        ((batch, 67, 257), 1e-6, torch.float32),
        ((batch, 3, 129), 1e-4, torch.float32),
        ((batch, 67, 257), 1e-6, torch.float32),
        ((batch, 3, 129), 1e-4, torch.float32),
    ]
    launches = [event for event in cpu_launches if event[0] != "native_norm"]
    assert [name for name, _ in launches] == (["cast"] if dtype == torch.bfloat16 else []) + [
        "modulate",
        "residual",
        "modulate",
        "residual",
    ]
    expected_grid = batch * 67 + batch * 3
    assert all(grid == (expected_grid,) for _, grid in launches)
    outputs = [t for group in actual for t in group]
    assert len({t.data_ptr() for t in outputs}) == len(outputs)
    assert all(t.is_contiguous() for t in outputs)
    assert all(t.data_ptr() not in {x.data_ptr() for x in (*left, *right)} for t in outputs)
    for value, snapshot in zip((*left, *right), before):
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("reason", ["disabled", "grad", "empty", "fp16", "mixed", "token_mod", "channel_stride"])
def test_pair_atomic_fallback(cpu_launches, monkeypatch, reason):
    left = _inputs(1, 3, 16, torch.bfloat16, "cpu")
    right = list(_inputs(1, 2, 16, torch.bfloat16, "cpu"))
    if reason == "disabled":
        monkeypatch.delenv("VLLM_OMNI_QWEN_ADALN_PAIR")
    elif reason == "empty":
        right[0], right[1] = right[0][:, :0], right[1][:, :0]
    elif reason in ("fp16", "mixed"):
        right = [x.to(torch.float16 if reason == "fp16" else torch.float32) for x in right]
    elif reason == "token_mod":
        right[2:] = [x.expand(1, 2, 16) for x in right[2:]]
    elif reason == "channel_stride":
        right = [x[..., ::2] for x in right]
    with torch.set_grad_enabled(reason == "grad"):
        assert _call(left, tuple(right)) == (None, None, None)
    assert not cpu_launches


@pytest.mark.cpu
@pytest.mark.parametrize("mode", ["paired", "final_fallback", "norm2_fallback", "indexed", "zero_cond"])
def test_real_forward_stream_order(mode):
    path = Path(pair.__file__).parents[1] / "models/qwen_image/qwen_image_transformer.py"
    tree = ast.parse(path.read_text())
    block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenImageTransformerBlock")
    methods: list[ast.stmt] = [
        n for n in block.body if isinstance(n, ast.FunctionDef) and n.name in ("forward", "_modulate")
    ]
    namespace = {"torch": torch, "Any": Any}
    names = (
        "try_fused_native_adaln",
        "try_fused_gated_residual_adaln",
        "try_fused_gated_residual",
        "try_paired_native_adaln",
        "try_paired_gated_residual_adaln",
        "try_paired_gated_residual",
    )
    namespace.update(dict.fromkeys(names, lambda *args: None))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    events = []

    class Norm:
        def __init__(self, eps):
            self.eps = eps

        def __call__(self, x, scale, shift):
            return _norm(x, scale, shift, self.eps)

    def mlp(x, name):
        events.append(name)
        return x * (0.25 if name == "image_mlp" else 0.5)

    obj = SimpleNamespace(
        zero_cond_t=mode == "zero_cond",
        img_mod=lambda x: x,
        txt_mod=lambda x: x,
        img_norm1=Norm(1e-6),
        img_norm2=Norm(1e-5),
        txt_norm1=Norm(1e-4),
        txt_norm2=Norm(1e-3),
        img_mlp=lambda x: mlp(x, "image_mlp"),
        txt_mlp=lambda x: mlp(x, "text_mlp"),
        attn=lambda **kw: (kw["hidden_states"] * 0.3, kw["encoder_hidden_states"] * 0.4),
    )
    obj._modulate = namespace["_modulate"].__get__(obj)
    forward = namespace["forward"].__get__(obj)
    kwargs = dict(
        hidden_states=torch.randn(2 if mode == "zero_cond" else 1, 3, 16),
        encoder_hidden_states=torch.randn(2 if mode == "indexed" else 1, 2, 16),
        temb=torch.randn(2 if mode in ("indexed", "zero_cond") else 1, 96),
        encoder_hidden_states_mask=None,
        image_rotary_emb=(None, None),
        modulate_index=torch.tensor([[0, 1, 0]]) if mode == "indexed" else None,
    )
    expected = forward(**kwargs)
    events.clear()

    def norm1(x0, x1, s0, t0, s1, t1, eps0, eps1):
        events.append("pair_norm1")
        assert (eps0, eps1) == (1e-6, 1e-4)
        return _norm(x0, s0, t0, eps0), _norm(x1, s1, t1, eps1)

    def norm2(left, right, eps0, eps1):
        events.append("pair_norm2")
        if mode == "norm2_fallback":
            return None
        assert (eps0, eps1) == (1e-5, 1e-3)
        x0, b0, g0, s0, t0 = left
        x1, b1, g1, s1, t1 = right
        r0, r1 = x0 + g0 * b0, x1 + g1 * b1
        return r0, _norm(r0, s0, t0, eps0), r1, _norm(r1, s1, t1, eps1)

    def final(x0, b0, g0, x1, b1, g1):
        events.append("pair_final")
        return None if mode == "final_fallback" else (x0 + g0 * b0, x1 + g1 * b1)

    namespace.update(zip(names[-3:], (norm1, norm2, final)))
    actual = forward(**kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert [e for e in events if e.endswith("_mlp")] == ["image_mlp", "text_mlp"]
    assert [e for e in events if e.startswith("pair_")] == (
        []
        if mode in ("indexed", "zero_cond")
        else ["pair_norm1", "pair_norm2"] + ([] if mode == "norm2_fallback" else ["pair_final"])
    )


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "image,text,hidden,batch,stride",
    [
        (1, 1, 257, 1, 1),
        (67, 3, 129, 2, 2),
        (4096, 12, 3072, 1, 1),
        (4096, 29, 3072, 1, 2),
    ],
)
def test_pair_cuda_exact(monkeypatch, dtype, image, text, hidden, batch, stride):
    monkeypatch.setenv("VLLM_OMNI_QWEN_ADALN_PAIR", "1")
    left = _inputs(batch, image, hidden, dtype, "cuda", stride)
    right = _inputs(batch, text, hidden, dtype, "cuda", stride)
    snapshots = [x.clone() for x in (*left, *right)]
    with torch.inference_mode():
        torch.testing.assert_close(_call(left, right), _reference(left, right), atol=0, rtol=0, equal_nan=True)
    for x, snapshot in zip((*left, *right), snapshots):
        torch.testing.assert_close(x, snapshot, atol=0, rtol=0)


@hardware_test(res={"cuda": "L4"})
def test_pair_compile_and_graph(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_QWEN_ADALN_PAIR", "1")
    left = _inputs(2, 67, 257, torch.bfloat16, "cuda", 2)
    right = _inputs(2, 3, 129, torch.bfloat16, "cuda", 2)
    with torch.inference_mode():
        expected = _reference(left, right)
        torch.testing.assert_close(torch.compile(_call, fullgraph=True)(left, right), expected, atol=0, rtol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _call(left, right)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = _call(left, right)
        left[0].add_(0.5)
        right[0].sub_(0.25)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(actual, _reference(left, right), atol=0, rtol=0)


@hardware_test(res={"cuda": "L4"})
def test_pair_compile_with_changing_stream_lengths(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_QWEN_ADALN_PAIR", "1")
    compiled = torch.compile(_call, fullgraph=True, dynamic=True)
    with torch.inference_mode():
        for image, text in ((4096, 12), (4096, 29), (129, 29)):
            left = _inputs(1, image, 3072, torch.bfloat16, "cuda", 2)
            right = _inputs(1, text, 3072, torch.bfloat16, "cuda", 2)
            torch.testing.assert_close(compiled(left, right), _reference(left, right), atol=0, rtol=0)
