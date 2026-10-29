# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Correctness tests for the fused alias-free Snake Triton kernel.

The kernel replaces the eager ``AliasFreeActivation1d`` path (replicate-pad
-> depthwise conv_transpose1d upsample -> SnakeBeta -> replicate-pad ->
depthwise strided conv1d downsample) with two Triton kernels. Only the tap
summation order differs, so the fused output must match eager within the
eager path's own rounding.

These tests call the kernel directly with the exact argument signature the
dispatch uses, so they validate the math independently of routing. The eager
reference is the real module forward (which itself uses SnakeBeta's Triton
path on CUDA) — the shipping composition, not a hand-rolled substitute.
"""

import copy

import pytest
import torch

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


@pytest.mark.parametrize("length", [1, 2, 3, 7, 63, 64, 65, 1025])
@pytest.mark.parametrize("channels", [1, 3, 24, 768])
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


@pytest.mark.parametrize("length", [1, 2, 3, 7, 63, 64, 65, 1025])
@pytest.mark.parametrize("channels", [1, 3, 24, 768])
@pytest.mark.parametrize("batch", [1, 2])
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
def test_whole_vocoder_smoke(t_mel):
    """Composed diff through a real (tiny) vocoder: measured ~8e-5 through
    the full 109-instance network, so atol=1e-3 has 10x headroom."""
    model = _make_tiny_vocoder()
    mel = torch.randn(1, 80, t_mel, device=DEVICE) - 5.0

    fused_forwards = {}

    def make_fused_forward(m):
        up, down, act = m.upsample, m.downsample, m.act
        if not act._cached:
            act.precompute_exp_cache()
        up_taps = up.filter.reshape(-1).contiguous()
        down_taps = down.filter.reshape(-1).contiguous()
        return lambda x: alias_free_snake(
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

    try:
        with torch.no_grad():
            stock = model(mel)
            for m in model.modules():
                if isinstance(m, AliasFreeActivation1d):
                    fused_forwards[m] = m.forward
                    m.forward = make_fused_forward(m)
            fused = model(mel)
    finally:
        for m, forward in fused_forwards.items():
            m.forward = forward

    torch.testing.assert_close(fused, stock, rtol=0.0, atol=1e-3)


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
