# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The one-kernel Sinkhorn iterations must reproduce the compiled native mHC post-processing bit for bit."""

import contextlib
import re
from functools import partial

import pytest
import torch
from vllm.triton_utils import tl, triton

from vllm_omni.diffusion.layers.mhc import MHCPostResidual
from vllm_omni.diffusion.models.magi2 import layers, mhc_sinkhorn

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

if mhc_sinkhorn.HAS_TRITON:

    @triton.jit
    def _copy_kernel(src_ptr, dst_ptr, block: tl.constexpr):
        offsets = tl.arange(0, block)
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets))


STREAMS, HIDDEN = 4, 3072
ITERATIONS, EPSILON = 20, 1e-12
SCALE = 1.0 / (STREAMS * HIDDEN) ** 0.5
_BITS = {torch.float32: torch.int32, torch.bfloat16: torch.int16}


def _native_iterations(matrix, iterations, epsilon):
    """The iteration loop of ``sinkhorn_knopp``."""
    for _ in range(iterations):
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + epsilon)
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + epsilon)
    return matrix


def _left_fold_iterations(matrix, iterations, epsilon):
    """The kernel's arithmetic on [tokens] columns: ``((m0 + m1) + m2) + m3 + epsilon``, then a division."""
    m = [[matrix[:, i, j] for j in range(STREAMS)] for i in range(STREAMS)]
    for _ in range(iterations):
        cols = [(((m[0][j] + m[1][j]) + m[2][j]) + m[3][j]) + epsilon for j in range(STREAMS)]
        m = [[m[i][j] / cols[j] for j in range(STREAMS)] for i in range(STREAMS)]
        rows = [(((m[i][0] + m[i][1]) + m[i][2]) + m[i][3]) + epsilon for i in range(STREAMS)]
        m = [[m[i][j] / rows[i] for j in range(STREAMS)] for i in range(STREAMS)]
    return torch.stack([torch.stack(row, dim=-1) for row in m], dim=-2)


def _pairwise_iterations(matrix, iterations, epsilon):
    """The same loop with the four-element sums paired as ``(m0 + m1) + (m2 + m3)``."""
    for _ in range(iterations):
        m = matrix
        matrix = m / ((m[:, :2].sum(-2, keepdim=True) + m[:, 2:].sum(-2, keepdim=True)) + epsilon)
        m = matrix
        matrix = m / ((m[..., :2].sum(-1, keepdim=True) + m[..., 2:].sum(-1, keepdim=True)) + epsilon)
    return matrix


def _native_post_residual(fused, alpha_post, bias_post, alpha_residual, bias_residual, out_dtype):
    _, post_logits, residual_logits = torch.split(fused, (STREAMS, STREAMS, STREAMS**2), dim=-1)
    return MHCPostResidual.forward_native(
        post_logits,
        residual_logits.view(-1, STREAMS, STREAMS),
        alpha_post,
        bias_post,
        alpha_residual,
        bias_residual,
        scale=SCALE,
        iterations=ITERATIONS,
        epsilon=EPSILON,
        out_dtype=out_dtype,
    )


def _fused_post_residual(fused, alpha_post, bias_post, alpha_residual, bias_residual, out_dtype):
    _, post_logits, residual_logits = torch.split(fused, (STREAMS, STREAMS, STREAMS**2), dim=-1)
    return layers._mhc_post_residual_fused_sinkhorn(
        post_logits,
        residual_logits.view(-1, STREAMS, STREAMS),
        alpha_post,
        bias_post,
        alpha_residual,
        bias_residual,
        scale=SCALE,
        iterations=ITERATIONS,
        epsilon=EPSILON,
        out_dtype=out_dtype,
    )


def _inputs(tokens, spread=1.0, seed=0, device="cpu"):
    """Fused [tokens, 24] logits and the post/residual alpha and bias, as in MHCHandler.compute_logits."""
    generator = torch.Generator().manual_seed(seed)
    fused = torch.randn(tokens, 2 * STREAMS + STREAMS**2, generator=generator) * (40.0 * spread)
    params = (
        torch.rand(1, generator=generator) + 0.5,
        torch.randn(STREAMS, generator=generator),
        torch.rand(1, generator=generator) + 0.5,
        torch.randn(STREAMS, STREAMS, generator=generator) * spread,
    )
    return tuple(tensor.to(device) for tensor in (fused, *params))


def _matrices(tokens, seed=0):
    """Normalized exponentials plus rows whose sums depend on the addition order."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, STREAMS, STREAMS, generator=generator) * 4
    matrix = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    tiny = 2.0**-24
    special = torch.tensor(
        [
            [[1.0, tiny, tiny, tiny], [tiny, tiny, tiny, tiny], [tiny, 1.0, tiny, tiny], [tiny, tiny, 1.0, tiny]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 1e-40, 3e-39, 0.5], [1e-45, 1.0, 0.25, 1e-30], [0.75, 1e-38, 1.0, 1e-20], [1.0, 1.0, 1.0, 1.0]],
        ]
    )
    return torch.cat((special, matrix))


@contextlib.contextmanager
def _restored_triton_language():
    """Undo the interpreter's patches, which Triton 3.2 leaves on ``triton.language`` for good.

    Left in place, they break every later Triton compile in the process, including the
    compile workers it forks.
    """
    targets = (tl, tl.core, tl.math, tl.core.tensor, tl.core.dtype)
    saved = [(target, dict(vars(target))) for target in targets]
    try:
        yield
    finally:
        for target, attrs in saved:
            for name in set(vars(target)) - set(attrs):
                delattr(target, name)
            for name, value in attrs.items():
                if vars(target).get(name) is not value:
                    setattr(target, name, value)


def _interpreter():
    """The Triton interpreter, if this Triton build runs kernels on CPU tensors."""
    if not mhc_sinkhorn.HAS_TRITON:
        pytest.skip("requires Triton")
    interpreter = pytest.importorskip("triton.runtime.interpreter")
    src, dst = torch.arange(4, dtype=torch.float32), torch.zeros(4)
    try:
        with _restored_triton_language():
            interpreter.InterpretedFunction(_copy_kernel.fn)[(1,)](src, dst, block=4)
    except Exception as error:
        pytest.skip(f"the Triton interpreter cannot run here: {error!r}")
    if not torch.equal(src, dst):
        pytest.skip("this Triton build's interpreter does not run kernels on CPU tensors")
    return interpreter


def _interpret(matrix, iterations, epsilon):
    interpreter = _interpreter()
    out = torch.empty_like(matrix)
    tokens = matrix.shape[0]
    grid = ((tokens + mhc_sinkhorn._BLOCK - 1) // mhc_sinkhorn._BLOCK,)
    with _restored_triton_language():
        interpreter.InterpretedFunction(mhc_sinkhorn._mhc_sinkhorn_kernel.fn)[grid](
            matrix, out, tokens, iterations, epsilon=epsilon, block=mhc_sinkhorn._BLOCK
        )
    return out


def _assert_bitwise_equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(_BITS[actual.dtype]), expected.contiguous().view(_BITS[expected.dtype]))


def _requires_triton():
    if not mhc_sinkhorn.HAS_TRITON:
        pytest.skip("requires Triton")


def _compiled_native_iterations(matrix, iterations):
    torch._dynamo.reset()
    compiled = torch.compile(_native_iterations, dynamic=False, fullgraph=True)(matrix, iterations, EPSILON)
    torch._dynamo.reset()
    return compiled


@pytest.mark.cpu
@pytest.mark.parametrize("tokens", [1, 65, 3702])
@pytest.mark.parametrize("iterations", [0, 1, ITERATIONS])
def test_inductor_unrolls_the_sums_as_left_folds(tokens, iterations):
    matrix = _matrices(tokens)
    compiled = _compiled_native_iterations(matrix, iterations)
    _assert_bitwise_equal(_left_fold_iterations(matrix, iterations, EPSILON), compiled)
    if iterations:
        # The ordered rows tell the folds apart, so a reordered sum cannot pass.
        assert not torch.equal(_pairwise_iterations(matrix, iterations, EPSILON), compiled)


@pytest.mark.cpu
@pytest.mark.parametrize("tokens", [1, 65, 3702])
@pytest.mark.parametrize("iterations", [0, 1, ITERATIONS])
def test_kernel_matches_compiled_native_iterations(tokens, iterations):
    matrix = _matrices(tokens)
    _assert_bitwise_equal(_interpret(matrix, iterations, EPSILON), _compiled_native_iterations(matrix, iterations))


@pytest.mark.cpu
def test_kernel_propagates_nan_like_the_native_iterations():
    matrix = _matrices(5)
    matrix[1, 2, 3] = float("nan")
    actual = _interpret(matrix, ITERATIONS, EPSILON)
    expected = _compiled_native_iterations(matrix, ITERATIONS)
    assert torch.equal(actual.isnan(), expected.isnan()) and bool(actual[1].isnan().all())
    _assert_bitwise_equal(actual[[0, 2, 3, 4]], expected[[0, 2, 3, 4]])


@pytest.mark.cpu
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("spread", [0.01, 1.0, 50.0])
def test_fused_helper_is_forward_native_around_the_iterations(monkeypatch, out_dtype, spread):
    # With the native loop in place of the kernel, the helper must be op-for-op forward_native.
    monkeypatch.setattr(layers, "mhc_sinkhorn_iterations", _native_iterations)
    fused, alpha_post, bias_post, alpha_residual, bias_residual = _inputs(37, spread=spread)
    expected = _native_post_residual(fused, alpha_post, bias_post, alpha_residual, bias_residual, out_dtype)
    actual = _fused_post_residual(fused, alpha_post, bias_post, alpha_residual, bias_residual, out_dtype)
    for got, want in zip(actual, expected):
        _assert_bitwise_equal(got, want)


@pytest.mark.cpu
def test_sinkhorn_op_checks_its_input():
    _requires_triton()
    empty = mhc_sinkhorn.mhc_sinkhorn_iterations(torch.empty(0, 4, 4), ITERATIONS, EPSILON)
    assert empty.shape == (0, 4, 4) and empty.dtype == torch.float32
    for bad in (torch.ones(2, 3, 3), torch.ones(2, 4, 4, dtype=torch.float16), torch.ones(16, 4)):
        with pytest.raises(ValueError, match="expected FP32"):
            mhc_sinkhorn.mhc_sinkhorn_iterations(bad, ITERATIONS, EPSILON)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "musa,streams,triton,compiling,requires_grad,expected",
    [
        (True, 4, True, True, False, "fused"),
        (False, 4, True, True, False, "native"),
        (True, 2, True, True, False, "native"),
        (True, 4, False, True, False, "native"),
        (True, 4, True, True, True, "native"),
        (True, 4, True, False, False, "custom_op"),
    ],
)
def test_compiled_post_residual_selection(monkeypatch, musa, streams, triton, compiling, requires_grad, expected):
    monkeypatch.setattr(layers.current_omni_platform, "is_musa", lambda: musa)
    monkeypatch.setattr(layers, "HAS_TRITON", triton)
    handler = layers.MHCHandler(streams, 8)
    assert handler.fused_sinkhorn == (musa and streams == 4 and triton)
    calls = []

    def record(name):
        def prepare(*args, **kwargs):
            calls.append(name)
            return args[0], args[1]

        return prepare

    class CustomOp:
        __call__ = staticmethod(record("custom_op"))
        forward_native = staticmethod(record("native"))

    monkeypatch.setattr(layers, "_mhc_post_residual_fused_sinkhorn", record("fused"))
    monkeypatch.setattr(layers, "_mhc_post_residual", CustomOp())
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    alpha = torch.ones(1, requires_grad=requires_grad)
    post = (alpha, torch.zeros(streams), torch.zeros(3, streams))
    residual = (torch.ones(1), torch.zeros(streams, streams), torch.zeros(3, streams, streams))
    handler.compute_post_residual(post, residual, out_dtype=torch.float32)
    assert calls == [expected]
    if requires_grad:
        with torch.no_grad():
            handler.compute_post_residual(post, residual, out_dtype=torch.float32)
        assert calls == [expected, "fused"]


@pytest.fixture(autouse=True)
def _deterministic_inductor(monkeypatch):
    # Reduction configs are otherwise benchmarked per compile, which can change the bits.
    # Dynamo resets ``deterministic`` after every traced frame; the config filter stays on.
    monkeypatch.setattr(torch._inductor.config, "deterministic", True)
    monkeypatch.setattr(torch._inductor.config.test_configs, "force_filter_reduction_configs", True)


@pytest.mark.cpu
def test_epsilon_and_block_are_compile_time_constants():
    # Inductor declares a runtime Python float argument of a user kernel as FP64.
    _requires_triton()
    params = mhc_sinkhorn._mhc_sinkhorn_kernel.params
    assert [param.name for param in params if param.is_constexpr] == ["epsilon", "block"]


def _musa_available():
    return hasattr(torch, "musa") and torch.musa.is_available()


def _launches(code):
    return re.findall(r"^\s+(\w+)\.run\(", code, re.M)


@pytest.mark.musa
@pytest.mark.parametrize("tokens,spread", [(3702, 1.0), (3702, 0.01), (3702, 50.0), (3651, 1.0), (129, 50.0), (1, 1.0)])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_real_musa_compiled_post_residual_is_bitwise_unchanged(tokens, spread, out_dtype):
    if not _musa_available():
        pytest.skip("requires a MUSA device")
    inputs = _inputs(tokens, spread=spread, seed=tokens, device="musa")
    torch._dynamo.reset()
    with torch.no_grad():
        expected = torch.compile(partial(_native_post_residual, out_dtype=out_dtype), dynamic=False, fullgraph=True)(
            *inputs
        )
        actual = torch.compile(partial(_fused_post_residual, out_dtype=out_dtype), dynamic=False, fullgraph=True)(
            *inputs
        )
    for got, want in zip(actual, expected):
        _assert_bitwise_equal(got.cpu(), want.cpu())
    torch._dynamo.reset()


@pytest.mark.musa
@pytest.mark.parametrize("tokens", [3702, 3651])
def test_real_musa_compiled_hyper_connection_is_bitwise_unchanged(tokens):
    if not _musa_available():
        pytest.skip("requires a MUSA device")
    fused_handler = layers.MHCHandler(STREAMS, HIDDEN)
    native_handler = layers.MHCHandler(STREAMS, HIDDEN)
    native_handler.fused_sinkhorn = False
    assert fused_handler.fused_sinkhorn

    def connect(handler, streams, output, fused, alpha_post, bias_post, alpha_residual, bias_residual):
        _, post_logits, residual_logits = torch.split(fused, (STREAMS, STREAMS, STREAMS**2), dim=-1)
        post, residual = handler.compute_post_residual(
            (alpha_post, bias_post, post_logits),
            (alpha_residual, bias_residual, residual_logits.view(-1, STREAMS, STREAMS)),
            out_dtype=streams.dtype,
        )
        return handler.hyper_connect(streams, output, post, residual)

    generator = torch.Generator().manual_seed(tokens)
    streams = torch.randn(tokens, STREAMS, HIDDEN, generator=generator).to(device="musa", dtype=torch.bfloat16)
    output = torch.randn(tokens, HIDDEN, generator=generator).to(device="musa", dtype=torch.bfloat16)
    inputs = _inputs(tokens, seed=tokens + 1, device="musa")
    torch._dynamo.reset()
    with torch.no_grad():
        expected = torch.compile(partial(connect, native_handler), dynamic=False, fullgraph=True)(
            streams, output, *inputs
        )
        actual = torch.compile(partial(connect, fused_handler), dynamic=False, fullgraph=True)(streams, output, *inputs)
    _assert_bitwise_equal(actual.cpu(), expected.cpu())
    torch._dynamo.reset()


@pytest.mark.musa
def test_real_musa_compiled_post_residual_launches_one_sinkhorn_kernel():
    if not _musa_available():
        pytest.skip("requires a MUSA device")
    from torch._inductor.utils import run_and_get_code

    inputs = _inputs(3702, device="musa")
    launches = {}
    for name, function in (("native", _native_post_residual), ("fused", _fused_post_residual)):
        torch._dynamo.reset()
        with torch.no_grad():
            compiled = torch.compile(partial(function, out_dtype=torch.float32), dynamic=False, fullgraph=True)
            _, codes = run_and_get_code(compiled, *inputs)
        launches[name] = [launch for code in codes for launch in _launches(code)]
    torch._dynamo.reset()
    assert len(launches["native"]) >= 2 * ITERATIONS
    assert len(launches["fused"]) <= 3
    assert sum("_mhc_sinkhorn_kernel" in launch for launch in launches["fused"]) == 1
