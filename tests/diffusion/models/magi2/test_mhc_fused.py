# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exactness and dispatch coverage for the fused mHC Sinkhorn kernel."""

from __future__ import annotations

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.models.magi2 import mhc_fused
from vllm_omni.diffusion.models.magi2.layers import MHCHandler
from vllm_omni.diffusion.models.magi2.mhc_fused import mhc_sinkhorn_matrix, sinkhorn_knopp

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_MATMUL_SCALE = 1.0 / (4 * 3072) ** 0.5
_ITERATIONS = 20
_EPSILON = 1e-12


def _clear_runtime_failure_cache():
    mhc_fused._FAILED_RUNTIME_KEYS.clear()


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor.contiguous().view(torch.int32)
    return tensor.contiguous().view(torch.int16)


def _reference(
    logits: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return sinkhorn_knopp(
        alpha * _MATMUL_SCALE * logits.float() + bias.unsqueeze(0).float(),
        _ITERATIONS,
        _EPSILON,
    ).to(out_dtype)


def _make_inputs(
    tokens: int,
    num_streams: int,
    device: str,
    *,
    seed: int = 0,
    scale: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = (
        torch.randn(tokens, num_streams, num_streams, device=device, dtype=torch.float32, generator=generator) * scale
    )
    alpha = torch.randn(1, device=device, dtype=torch.float32, generator=generator)
    bias = torch.randn(num_streams, num_streams, device=device, dtype=torch.float32, generator=generator)
    return logits, alpha, bias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_streams", [2, 4])
@pytest.mark.parametrize(
    "tokens",
    [1, 2, 3, 7, 15, 16, 17, 31, 255, 256, 1024, 4096, 65537],
)
def test_fused_mhc_sinkhorn_is_bit_exact(tokens, num_streams, out_dtype):
    _clear_runtime_failure_cache()
    logits, alpha, bias = _make_inputs(tokens, num_streams, "cuda", seed=tokens * 31 + num_streams)
    # Call the launcher directly: a kernel that fails to compile or launch
    # must fail the test instead of silently passing via eager fallback.
    fused = mhc_fused._launch_fused_mhc_sinkhorn(
        logits,
        alpha,
        bias,
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=out_dtype,
    )
    reference = _reference(logits, alpha, bias, out_dtype)
    assert fused.dtype == out_dtype
    assert torch.equal(_bits(fused), _bits(reference))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("magnitude", ["tiny", "large", "mixed"])
def test_fused_mhc_sinkhorn_special_values(out_dtype, magnitude):
    _clear_runtime_failure_cache()
    tokens, num_streams = 64, 4
    logits, alpha, bias = _make_inputs(tokens, num_streams, "cuda", seed=7)

    if magnitude == "tiny":
        logits[0] = 0.0
        logits[1] = -0.0
        logits[2] = 5e-45  # smallest fp32 subnormal
        logits[3] = 1e-38
    elif magnitude == "large":
        logits[0] = 80.0  # exp overflow boundary at fp32 after amax shift
        logits[1] = -100.0
        logits[2] = 3.4e38
        logits[3] = -3.4e38
    else:
        special = [
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            -0.0,
            1e-40,
            1e30,
            -1e30,
        ]
        for index, value in enumerate(special):
            logits[index, index % num_streams, (index * 3) % num_streams] = value
        bias[0, 0] = float("inf")
        bias[1, 1] = float("nan")

    # Direct launcher call: kernel failures must fail, not fall back.
    fused = mhc_fused._launch_fused_mhc_sinkhorn(
        logits,
        alpha,
        bias,
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=out_dtype,
    )
    reference = _reference(logits, alpha, bias, out_dtype)
    # NaN payloads: compare bit patterns, not value equality.
    assert torch.equal(_bits(fused), _bits(reference))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_mhc_sinkhorn_dispatches_supported_inputs(monkeypatch):
    _clear_runtime_failure_cache()
    launches = []
    real_launch = mhc_fused._launch_fused_mhc_sinkhorn

    def counting_launch(*args, **kwargs):
        launches.append(1)
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(mhc_fused, "_launch_fused_mhc_sinkhorn", counting_launch)
    logits, alpha, bias = _make_inputs(128, 4, "cuda")
    mhc_sinkhorn_matrix(
        logits,
        alpha,
        bias,
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=torch.bfloat16,
    )
    assert len(launches) == 1
    # The launch must have succeeded: a caught failure would flip this cache.
    assert not mhc_fused._FAILED_RUNTIME_KEYS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_mhc_sinkhorn_accepts_compute_logits_layout(monkeypatch):
    """compute_logits() hands compute_post_residual() a strided view.

    The residual slice of the [T, N*(N+2)] projection keeps stride
    (N*(N+2), N, 1); the fused path must launch on it directly.
    Regression: a full-tensor contiguity guard rejected every T > 1
    production call.
    """
    _clear_runtime_failure_cache()
    launches = []
    real_launch = mhc_fused._launch_fused_mhc_sinkhorn

    def counting_launch(*args, **kwargs):
        launches.append(1)
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(mhc_fused, "_launch_fused_mhc_sinkhorn", counting_launch)
    tokens, num_streams = 97, 4
    logits, alpha, bias = _make_inputs(tokens, num_streams, "cuda")
    # Rebuild the exact layout compute_logits() produces: [T, N*(N+2)]
    # projection, residual slice last, viewed back to [T, N, N].
    projection = torch.zeros(tokens, num_streams * (num_streams + 2), device="cuda", dtype=torch.float32)
    projection[:, 2 * num_streams :].copy_(logits.reshape(tokens, -1))
    residual_view = projection[:, 2 * num_streams :].view(-1, num_streams, num_streams)
    assert residual_view.stride() == (num_streams * (num_streams + 2), num_streams, 1)
    assert not residual_view.is_contiguous()

    fused = mhc_sinkhorn_matrix(
        residual_view,
        alpha,
        bias,
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=torch.bfloat16,
    )
    assert len(launches) == 1
    assert not mhc_fused._FAILED_RUNTIME_KEYS
    reference = _reference(logits, alpha, bias, torch.bfloat16)
    assert torch.equal(_bits(fused), _bits(reference))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_mhc_sinkhorn_unsupported_streams_falls_back(monkeypatch):
    """num_streams outside the fused set (e.g. 8) keeps the eager formula."""
    _clear_runtime_failure_cache()
    launches = []
    monkeypatch.setattr(
        mhc_fused,
        "_launch_fused_mhc_sinkhorn",
        lambda *a, **k: launches.append(1) or torch.empty(0),
    )
    logits, alpha, bias = _make_inputs(64, 8, "cuda")
    result = mhc_sinkhorn_matrix(
        logits,
        alpha,
        bias,
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=torch.bfloat16,
    )
    assert launches == []
    assert torch.equal(_bits(result), _bits(_reference(logits, alpha, bias, torch.bfloat16)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize(
    "unsupported",
    ["non_contiguous", "wrong_dtype", "odd_streams", "requires_grad", "empty"],
)
def test_fused_mhc_sinkhorn_falls_back_for_unsupported_inputs(monkeypatch, unsupported):
    _clear_runtime_failure_cache()
    launches = []
    monkeypatch.setattr(
        mhc_fused,
        "_launch_fused_mhc_sinkhorn",
        lambda *a, **k: launches.append(1) or torch.empty(0),
    )
    logits, alpha, bias = _make_inputs(32, 4, "cuda")
    if unsupported == "non_contiguous":
        # Row-gapped view: hidden dims dense, rows separated by a stride.
        storage = torch.empty(32, 4, 8, device="cuda", dtype=torch.float32)
        logits = torch.as_strided(storage, (32, 4, 4), (32, 8, 1))
        logits.copy_(_make_inputs(32, 4, "cuda")[0])
    elif unsupported == "wrong_dtype":
        logits = logits.to(torch.bfloat16)
    elif unsupported == "odd_streams":
        logits = torch.randn(32, 3, 3, device="cuda")
        bias = torch.randn(3, 3, device="cuda")
    elif unsupported == "requires_grad":
        logits = logits.detach().requires_grad_()
    elif unsupported == "empty":
        logits = logits[:0]

    result = mhc_sinkhorn_matrix(
        logits,
        alpha,
        bias,
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=torch.bfloat16,
    )
    assert launches == []
    assert result.shape[:2] == (logits.shape[0], logits.shape[1])


@pytest.mark.cpu
def test_fused_mhc_sinkhorn_cpu_input_matches_eager_formula():
    logits, alpha, bias = _make_inputs(33, 4, "cpu")
    result = mhc_sinkhorn_matrix(
        logits,
        alpha,
        bias,
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=torch.float32,
    )
    reference = _reference(logits, alpha, bias, torch.float32)
    assert torch.equal(_bits(result), _bits(reference))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_mhc_sinkhorn_falls_back_when_grad_is_required():
    _clear_runtime_failure_cache()
    logits, alpha, bias = _make_inputs(16, 4, "cuda")
    logits = logits.requires_grad_()
    result = mhc_sinkhorn_matrix(
        logits,
        alpha,
        bias,
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=torch.float32,
    )
    assert result.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("grad_context", [torch.inference_mode, torch.no_grad])
def test_fused_mhc_sinkhorn_accepts_parameters_without_active_autograd(grad_context, monkeypatch):
    """Model parameters keep requires_grad=True; inference contexts must fuse.

    ``load_weights`` never clears the nn.Parameter flags, and neither does
    inference_mode()/no_grad(), so eligibility keys on active autograd, not
    on the flags themselves.
    """
    _clear_runtime_failure_cache()
    launches = []
    real_launch = mhc_fused._launch_fused_mhc_sinkhorn

    def counting_launch(*args, **kwargs):
        launches.append(1)
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(mhc_fused, "_launch_fused_mhc_sinkhorn", counting_launch)
    logits, alpha, bias = _make_inputs(128, 4, "cuda")
    alpha_param = torch.nn.Parameter(alpha.detach().clone())
    bias_param = torch.nn.Parameter(bias.detach().clone())
    kwargs = dict(
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=torch.bfloat16,
    )

    with grad_context():
        fused = mhc_sinkhorn_matrix(logits, alpha_param, bias_param, **kwargs)
    assert len(launches) == 1
    assert not mhc_fused._FAILED_RUNTIME_KEYS
    assert not fused.requires_grad
    reference = _reference(logits, alpha_param.detach(), bias_param.detach(), torch.bfloat16)
    assert torch.equal(_bits(fused), _bits(reference))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_mhc_sinkhorn_fuses_without_grad_tracing_when_grad_enabled(monkeypatch):
    """Grad-enabled calls on inputs that need no tracing still take the kernel."""
    _clear_runtime_failure_cache()
    launches = []
    real_launch = mhc_fused._launch_fused_mhc_sinkhorn

    def counting_launch(*args, **kwargs):
        launches.append(1)
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(mhc_fused, "_launch_fused_mhc_sinkhorn", counting_launch)
    logits, alpha, bias = _make_inputs(64, 4, "cuda")
    assert torch.is_grad_enabled()
    mhc_sinkhorn_matrix(
        logits,
        alpha,
        bias,
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=torch.bfloat16,
    )
    assert len(launches) == 1
    assert not mhc_fused._FAILED_RUNTIME_KEYS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_mhc_sinkhorn_caches_runtime_failure(monkeypatch):
    _clear_runtime_failure_cache()
    launches = []

    def failing_launch(*args, **kwargs):
        launches.append(1)
        raise RuntimeError("simulated launch failure")

    monkeypatch.setattr(mhc_fused, "_launch_fused_mhc_sinkhorn", failing_launch)
    logits, alpha, bias = _make_inputs(64, 4, "cuda")
    kwargs = dict(
        matmul_scale=_MATMUL_SCALE,
        iterations=_ITERATIONS,
        epsilon=_EPSILON,
        out_dtype=torch.bfloat16,
    )
    first = mhc_sinkhorn_matrix(logits, alpha, bias, **kwargs)
    second = mhc_sinkhorn_matrix(logits, alpha, bias, **kwargs)
    assert len(launches) == 1  # second call served by the failure cache
    reference = _reference(logits, alpha, bias, torch.bfloat16)
    assert torch.equal(_bits(first), _bits(reference))
    assert torch.equal(_bits(second), _bits(reference))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_mhc_sinkhorn_propagates_out_of_memory(monkeypatch):
    """A transient CUDA OOM must propagate, not permanently disable fusion."""
    _clear_runtime_failure_cache()

    def oom_launch(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(mhc_fused, "_launch_fused_mhc_sinkhorn", oom_launch)
    logits, alpha, bias = _make_inputs(64, 4, "cuda")
    with pytest.raises(torch.OutOfMemoryError):
        mhc_sinkhorn_matrix(
            logits,
            alpha,
            bias,
            matmul_scale=_MATMUL_SCALE,
            iterations=_ITERATIONS,
            epsilon=_EPSILON,
            out_dtype=torch.bfloat16,
        )
    # The failure cache stays empty: fusion remains eligible once memory frees.
    assert not mhc_fused._FAILED_RUNTIME_KEYS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_streams", [2, 4])
def test_mhc_handler_compute_post_residual_is_bit_exact(out_dtype, num_streams):
    _clear_runtime_failure_cache()
    hidden = 64
    tokens = 97
    handler = MHCHandler(num_streams, hidden, dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(num_streams * 17)
    post_logits = torch.randn(tokens, num_streams, device="cuda", generator=generator)
    # Residual logits in the exact strided layout compute_logits() produces.
    projection = torch.randn(tokens, num_streams * (num_streams + 2), device="cuda", generator=generator)
    residual_logits = projection[:, 2 * num_streams :].view(-1, num_streams, num_streams)
    assert not residual_logits.is_contiguous()
    alpha_post = torch.randn(1, device="cuda", generator=generator)
    bias_post = torch.randn(num_streams, device="cuda", generator=generator)
    alpha_res = torch.randn(1, device="cuda", generator=generator)
    bias_res = torch.randn(num_streams, num_streams, device="cuda", generator=generator)

    fused_post, fused_residual = handler.compute_post_residual(
        (alpha_post, bias_post, post_logits),
        (alpha_res, bias_res, residual_logits),
        out_dtype=out_dtype,
    )

    # Force the eager path for the reference.
    _clear_runtime_failure_cache()
    mhc_fused._FAILED_RUNTIME_KEYS.add((fused_residual.device.index, out_dtype))
    eager_post, eager_residual = handler.compute_post_residual(
        (alpha_post, bias_post, post_logits),
        (alpha_res, bias_res, residual_logits),
        out_dtype=out_dtype,
    )

    assert torch.equal(_bits(fused_post), _bits(eager_post))
    assert torch.equal(_bits(fused_residual), _bits(eager_residual))
