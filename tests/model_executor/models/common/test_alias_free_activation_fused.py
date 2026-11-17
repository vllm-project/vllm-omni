# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Correctness and dispatch tests for the fused alias-free Snake Triton kernel.

The kernel replaces the eager ``AliasFreeActivation1d`` path (replicate-pad
-> depthwise conv_transpose1d upsample -> SnakeBeta -> replicate-pad ->
depthwise strided conv1d downsample) with two Triton kernels. Only the tap
summation order differs, so the fused output must match eager within the
eager path's own rounding.

The first group of tests calls the kernel directly with the exact argument
signature the dispatch uses, so they validate the math independently of
routing; the eager reference is the real module forward (which itself uses
SnakeBeta's Triton path on CUDA) — the shipping composition, not a
hand-rolled substitute. The second group verifies the dispatch in
``AliasFreeActivation1d.forward``: who routes to the kernel, who stays
eager, and what happens when the kernel fails.
"""

import copy

import pytest
import torch

import vllm_omni.model_executor.models.common.alias_free_activation as fir
from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.common.alias_free_activation import (
    AliasFreeActivation1d,
)
from vllm_omni.model_executor.models.common.ops.alias_free_triton import alias_free_snake
from vllm_omni.model_executor.models.common.snake_activation import Snake, SnakeBeta

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels"),
]

DEVICE = torch.device("cuda")


def _make_module(channels, activation="beta", dtype=torch.float32):
    act = SnakeBeta(channels) if activation == "beta" else Snake(channels)
    with torch.no_grad():
        act.alpha.uniform_(-0.5, 0.5)
        if hasattr(act, "beta"):
            act.beta.uniform_(-0.5, 0.5)
    module = AliasFreeActivation1d(activation=act).to(device=DEVICE, dtype=dtype).eval()
    return module


def _fused(module, x):
    """Call the fused kernel with the exact arguments the dispatch will pass."""
    up, down, act = module.upsample, module.downsample, module.act
    if not act._cached:
        act.precompute_exp_cache()
    return alias_free_snake(
        x,
        up.filter.reshape(-1),
        down.filter.reshape(-1),
        act._exp_alpha,
        act._inv_beta,
        up.ratio,
        up.pad,
        up.pad_left,
        down.stride,
        down.pad_left,
        down.pad_right,
    )


# Boundary lengths only: the kernel is shape-generic, so the block-size
# edges (63/64/65), the degenerate length 1 and one long odd length cover
# every code path; channels and batch are leading-dim variations of the
# same path (batch=2 stays because CFG runs the vocoder at batch 2).
@pytest.mark.parametrize("length", [1, 63, 64, 65, 1025])
@pytest.mark.parametrize("channels", [24])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("activation", ["beta", "snake"])
def test_fused_matches_eager(batch, channels, length, activation):
    """Fused kernel vs the module's eager path in fp32.

    The only difference is tap summation order (measured floor ~5e-7, so
    atol=1e-5 is 20x headroom; a wrong tap/pad/crop is O(1)). bf16 is gated
    by ``test_bf16_no_worse_than_eager_rounding`` instead: fused-vs-eager
    assert_close is the wrong criterion there, because near-zero outputs
    formed by cancellation carry eager's own stage-boundary roundings (up to
    ~1e-2 absolute, measured) while the fused path rounds once.
    """
    torch.manual_seed(1000 + channels)
    module = _make_module(channels, activation, torch.float32)
    x = torch.randn(batch, channels, length, device=DEVICE)
    with torch.no_grad():
        eager = module(x)
        fused = _fused(module, x)
    assert fused.dtype == torch.float32
    torch.testing.assert_close(fused, eager, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("length", [1, 64, 1025])
@pytest.mark.parametrize("channels", [24])
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("activation", ["beta", "snake"])
def test_bf16_no_worse_than_eager_rounding(batch, channels, length, activation):
    """Fused bf16 must not lose to eager bf16 against an fp32 reference.

    Both paths see the same bf16-quantized weights; the reference computes
    the same module in fp32. Eager rounds at every stage boundary (upsample
    conv, snake, downsample conv), fused computes fp32-in-register and
    rounds once at the store, so the fused error must stay within eager's
    error plus one bf16 rounding of the signal scale.
    """
    torch.manual_seed(2000 + channels)
    module = _make_module(channels, activation, torch.bfloat16)
    reference = copy.deepcopy(module)
    reference.act._exp_alpha = None
    reference.act._inv_beta = None
    reference = reference.float()

    x_bf16 = torch.randn(batch, channels, length, device=DEVICE, dtype=torch.bfloat16)
    x_fp32 = x_bf16.float()  # exact upcast of the same values
    with torch.no_grad():
        raw = _fused(module, x_bf16)
        fused = raw.float()
        eager = module(x_bf16).float()
        expected = reference(x_fp32)

    assert raw.dtype == torch.bfloat16, "kernel must preserve the input dtype"
    err_fused = (fused - expected).abs().max().item()
    err_eager = (eager - expected).abs().max().item()
    scale = expected.abs().max().item()
    assert err_fused <= err_eager + 1e-2 * scale, (
        f"fused bf16 error {err_fused:.3e} exceeds eager {err_eager:.3e} (scale {scale:.3e})"
    )


@pytest.mark.parametrize("kind", ["zero", "constant", "impulse"])
def test_boundary_signals(kind):
    """Edge samples are where a clamp-as-replicate-pad bug lives."""
    torch.manual_seed(3)
    module = _make_module(2, "beta", torch.float32)
    x = torch.zeros(1, 2, 105, device=DEVICE)
    if kind == "constant":
        x.fill_(1.0)
    elif kind == "impulse":
        # positions 0, 1, pad-1, T-pad, T-1 with pad=5, T=105
        x[..., [0, 1, 4, 100, 104]] = 1.0
    with torch.no_grad():
        eager = module(x)
        fused = _fused(module, x)
    torch.testing.assert_close(fused, eager, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_determinism(dtype):
    torch.manual_seed(4)
    module = _make_module(24, "beta", dtype)
    x = torch.randn(2, 24, 1000, device=DEVICE, dtype=dtype)
    with torch.no_grad():
        first = _fused(module, x)
        second = _fused(module, x)
    assert torch.equal(first, second)


def _make_tiny_vocoder():
    """A miniature of the real Qwen2.5-Omni BigVGAN vocoder (same classes,
    same block layout, seeded weights): 13 alias-free instances composed
    with trained-style convolutions."""
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniBigVGANConfig,
    )

    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
        Qwen2_5OmniToken2WavBigVGANModel,
    )

    config = Qwen2_5OmniBigVGANConfig(
        upsample_rates=(2, 2),
        upsample_kernel_sizes=(4, 4),
        upsample_initial_channel=64,
        mel_dim=80,
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
    )
    torch.manual_seed(11)
    return Qwen2_5OmniToken2WavBigVGANModel(config).to(device=DEVICE, dtype=torch.float32).eval()


@pytest.mark.parametrize("t_mel", [64, 256])
def test_whole_vocoder_smoke(t_mel, monkeypatch):
    """Composed diff through a real (tiny) vocoder, via the dispatch: measured
    ~8e-5 through the full 109-instance network, so atol=1e-3 has 10x
    headroom."""
    model = _make_tiny_vocoder()
    mel = torch.randn(1, 80, t_mel, device=DEVICE) - 5.0
    with torch.no_grad():
        fused = model(mel)
        monkeypatch.setattr(AliasFreeActivation1d, "_fused_disabled", True)
        stock = model(mel)
    torch.testing.assert_close(fused, stock, rtol=0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Dispatch: who routes to the kernel, who stays eager
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("up_ratio", "up_kernel", "down_ratio", "down_kernel", "eligible"),
    [
        (2, 12, 2, 12, True),  # the geometry every shared BigVGAN family uses
        (2, 16, 2, 16, True),
        (2, 11, 2, 11, True),  # odd taps: the polyphase form still hits every k
        (3, 15, 3, 15, True),
        (4, 8, 4, 8, False),  # crop does not cover the filter's reach
        (2, 12, 3, 12, False),  # downsample stride != upsample ratio
    ],
)
def test_eligibility_predicate(monkeypatch, up_ratio, up_kernel, down_ratio, down_kernel, eligible):
    calls = []
    real = fir.alias_free_snake

    def recorder(*args, **kwargs):
        calls.append(args[0].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(fir, "alias_free_snake", recorder)
    torch.manual_seed(7)
    act = SnakeBeta(4)
    with torch.no_grad():
        act.alpha.uniform_(-0.5, 0.5)
        act.beta.uniform_(-0.5, 0.5)
    module = AliasFreeActivation1d(act, up_ratio, down_ratio, up_kernel, down_kernel).to(DEVICE).eval()
    assert module._fused_eligible is eligible

    x = torch.randn(1, 4, 130, device=DEVICE)
    with torch.no_grad():
        dispatched = module(x)
        monkeypatch.setattr(AliasFreeActivation1d, "_fused_disabled", True)
        eager = module(x)
    assert bool(calls) is eligible
    if eligible:
        torch.testing.assert_close(dispatched, eager, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_whole_module_dispatch(monkeypatch, dtype):
    """The dispatch must produce exactly the kernel's output (same call,
    deterministic kernel) and preserve the input dtype; fp32 additionally
    matches eager at the T1.1 bound (bf16 vs eager is gated by
    ``test_bf16_no_worse_than_eager_rounding``)."""
    torch.manual_seed(8)
    module = _make_module(24, "beta", dtype)
    x = torch.randn(2, 24, 517, device=DEVICE, dtype=dtype)
    with torch.no_grad():
        dispatched = module(x)
        direct = _fused(module, x)
        monkeypatch.setattr(AliasFreeActivation1d, "_fused_disabled", True)
        eager = module(x)
    assert dispatched.dtype == dtype
    assert torch.equal(dispatched, direct)
    if dtype == torch.float32:
        torch.testing.assert_close(dispatched, eager, rtol=0.0, atol=1e-5)


def test_dispatch_requires_cuda_and_no_grad(monkeypatch):
    """CUDA + no-grad routes to the kernel; grad-enabled stays eager."""
    module = _make_module(8, "beta", torch.float32)
    x = torch.randn(1, 8, 300, device=DEVICE)
    calls = []
    real = fir.alias_free_snake

    def recorder(*args, **kwargs):
        calls.append(args[0].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(fir, "alias_free_snake", recorder)
    with torch.no_grad():
        module(x)
    assert len(calls) == 1
    with torch.enable_grad():
        module(x)
    assert len(calls) == 1  # no second call: grad-enabled kept the eager path


def test_custom_activation_stays_eager(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("a non-Snake activation must not reach the fused kernel")

    monkeypatch.setattr(fir, "alias_free_snake", unexpected)
    module = AliasFreeActivation1d(torch.nn.PReLU(8)).to(DEVICE).eval()
    x = torch.randn(1, 8, 300, device=DEVICE)
    with torch.no_grad():
        out = module(x)
    assert out.shape == x.shape


def test_triton_unavailable_stays_eager(monkeypatch):
    monkeypatch.setattr(fir, "HAS_TRITON", False)

    def unexpected(*args, **kwargs):
        pytest.fail("with Triton unavailable the module must stay eager")

    monkeypatch.setattr(fir, "alias_free_snake", unexpected)
    module = _make_module(8, "beta", torch.float32)
    x = torch.randn(1, 8, 300, device=DEVICE)
    with torch.no_grad():
        out = module(x)
        monkeypatch.setattr(AliasFreeActivation1d, "_fused_disabled", True)
        eager = module(x)
    torch.testing.assert_close(out, eager, rtol=0.0, atol=1e-5)


def test_npu_platform_stays_eager(monkeypatch):
    module = _make_module(8, "beta", torch.float32)  # built before the platform patch
    monkeypatch.setattr(fir.current_omni_platform, "is_npu", lambda: True)

    def unexpected(*args, **kwargs):
        pytest.fail("NPU must keep the replication-pad/chunked-conv path")

    monkeypatch.setattr(fir, "alias_free_snake", unexpected)
    x = torch.randn(1, 8, 300, device=DEVICE)
    with torch.no_grad():
        out = module(x)
    assert out.shape == x.shape


def test_fused_failure_warns_once_and_falls_back(monkeypatch):
    module = _make_module(8, "beta", torch.float32)
    x = torch.randn(1, 8, 300, device=DEVICE)
    warnings = []
    monkeypatch.setattr(fir.logger, "warning", lambda *args, **kwargs: warnings.append(args))

    def boom(*args, **kwargs):
        raise RuntimeError("kernel boom")

    monkeypatch.setattr(fir, "alias_free_snake", boom)
    monkeypatch.setattr(AliasFreeActivation1d, "_fused_disabled", False)
    with torch.no_grad():
        first = module(x)
        second = module(x)
    assert len(warnings) == 1  # one warning, then the latch silences it

    monkeypatch.setattr(AliasFreeActivation1d, "_fused_disabled", True)
    with torch.no_grad():
        eager = module(x)
    torch.testing.assert_close(first, eager, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(second, eager, rtol=0.0, atol=1e-5)


def test_non_contiguous_input(monkeypatch):
    module = _make_module(8, "beta", torch.float32)
    x = torch.randn(300, 1, 8, device=DEVICE).permute(1, 2, 0)  # [1, 8, 300]
    assert not x.is_contiguous()
    with torch.no_grad():
        dispatched = module(x)
        monkeypatch.setattr(AliasFreeActivation1d, "_fused_disabled", True)
        eager = module(x)
    torch.testing.assert_close(dispatched, eager, rtol=0.0, atol=1e-5)


def test_cuda_graph_capture():
    """The op must capture and replay: AUK's compiled decode buckets and any
    future BigVGAN graph depend on it."""
    torch.manual_seed(5)
    module = _make_module(8, "beta", torch.float32)
    x = torch.randn(1, 8, 500, device=DEVICE)

    with torch.no_grad():
        static_x = x.clone()
        # warm up on a side stream, then capture
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _fused(module, static_x)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = _fused(module, static_x)

        static_x.copy_(torch.randn_like(x))
        graph.replay()
        replayed = static_out.clone()
        direct = _fused(module, static_x)

    assert torch.equal(replayed, direct)


def test_torch_compile_traces_op():
    """The torch.library.triton_op registration must let torch.compile trace
    the kernel without a graph break (fullgraph=True raises on any break)."""
    torch.manual_seed(6)
    module = _make_module(8, "beta", torch.float32)
    x = torch.randn(1, 8, 500, device=DEVICE)
    up, down, act = module.upsample, module.downsample, module.act
    if not act._cached:
        act.precompute_exp_cache()
    up_taps = up.filter.reshape(-1).contiguous()
    down_taps = down.filter.reshape(-1).contiguous()

    def fn(x):
        return alias_free_snake(
            x,
            up_taps,
            down_taps,
            act._exp_alpha,
            act._inv_beta,
            up.ratio,
            up.pad,
            up.pad_left,
            down.stride,
            down.pad_left,
            down.pad_right,
        )

    compiled = torch.compile(fn, fullgraph=True)
    with torch.no_grad():
        out = compiled(x)
        ref = fn(x)
    assert torch.equal(out, ref)
