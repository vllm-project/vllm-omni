# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-graph replay tests for the BigVGAN vocoder wrapper.

The wrapper captures the GPU vocode (``mel [1, C, T] -> wav [1, 1, T*hop]``)
for fixed mel-frame buckets and replays with zero-pad + trim. Exact-size
hits replay the same kernels on the same static buffers as the capture run,
so they must be deterministic and match a fresh eager vocode within eager's
own conv-nondeterminism floor. Padded hits differ from eager only inside the
conv receptive field of the tail (the zero-padding vs replicate-padding
boundary), which these tests characterize rather than deny.

The lazy-warmup tests deliberately trigger capture from inside
``torch.inference_mode()``: the wrapper must create its static buffers as
normal tensors (not inference tensors) so later non-inference callers can
still replay.
"""

import pytest
import torch

import vllm_omni.model_executor.models.common.ops.bigvgan_cuda_graph as bg
from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.common.alias_free_activation import (
    AliasFreeActivation1d,
)
from vllm_omni.model_executor.models.common.ops.bigvgan_cuda_graph import (
    BigVGANCUDAGraphWrapper,
)

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA graphs"),
]

DEVICE = torch.device("cuda")
HOP = 4  # tiny vocoder: upsample_rates (2, 2)


def _make_tiny_vocoder():
    """A miniature of the real Qwen2.5-Omni BigVGAN vocoder (same classes,
    same block layout, seeded weights); 13 alias-free instances composed
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


def _make_wrapper(model, sizes):
    return BigVGANCUDAGraphWrapper(
        model,
        capture_fn=model.vocode,
        num_mels=80,
        upsample_factor=HOP,
        capture_sizes=list(sizes),
    )


def _mel(t, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, 80, t, device=DEVICE) - 5.0


# ---------------------------------------------------------------------------
# Replay correctness
# ---------------------------------------------------------------------------


def test_replay_deterministic_and_matches_eager_exact():
    """G1/G2/G4: exact-size replay is deterministic and matches a fresh eager
    vocode within eager's own conv nondeterminism floor (the capture warm run
    may settle cuDNN autotune differently from a fresh call)."""
    model = _make_tiny_vocoder()
    wrapper = _make_wrapper(model, (64,))
    wrapper.warmup(DEVICE, torch.float32)
    mel = _mel(64, seed=2)

    first = wrapper(mel)
    snapshot = first.clone()
    second = wrapper(mel)
    assert wrapper.last_call_info["mode"] == "graph"
    assert wrapper.last_call_info["padding"] == 0
    assert torch.equal(first, snapshot) and torch.equal(first, second)
    assert first.shape == (1, 1, 64 * HOP)

    with torch.no_grad():
        eager = model.vocode(mel)
    torch.testing.assert_close(first, eager, rtol=0.0, atol=1e-4)


def test_padded_replay_trims_and_bounds_tail_leakage():
    """G3: padded replay keeps the exact output length, matches eager outside
    the conv receptive field of the padding boundary, and stays bounded (the
    vocoder clamps to [-1, 1]) inside it."""
    model = _make_tiny_vocoder()
    wrapper = _make_wrapper(model, (128,))
    wrapper.warmup(DEVICE, torch.float32)
    t = 100
    mel = _mel(t, seed=3)

    out = wrapper(mel)
    assert wrapper.last_call_info["padding"] == 28
    assert out.shape[-1] == t * HOP

    with torch.no_grad():
        eager = model.vocode(mel)
    clean_frames = t - 32  # well beyond the tiny model's receptive field
    torch.testing.assert_close(out[..., : clean_frames * HOP], eager[..., : clean_frames * HOP], rtol=0.0, atol=1e-4)
    assert (out - eager).abs().max() < 0.25


def test_shape_above_largest_bucket_falls_back_eager():
    """G5: no containing bucket -> eager passthrough through capture_fn."""
    model = _make_tiny_vocoder()
    wrapper = _make_wrapper(model, (64,))
    wrapper.warmup(DEVICE, torch.float32)
    mel = _mel(100, seed=4)

    out = wrapper(mel)
    assert wrapper.last_call_info["mode"] == "eager"
    with torch.no_grad():
        eager = model.vocode(mel)
    torch.testing.assert_close(out, eager, rtol=0.0, atol=0.0)


def test_batch2_stays_eager():
    """G6: graphs are batch-1 only."""
    model = _make_tiny_vocoder()
    wrapper = _make_wrapper(model, (64,))
    wrapper.warmup(DEVICE, torch.float32)
    torch.manual_seed(5)
    mel = torch.randn(2, 80, 64, device=DEVICE) - 5.0

    out = wrapper(mel)
    assert wrapper.last_call_info["mode"] == "eager"
    assert out.shape == (2, 1, 64 * HOP)
    with torch.no_grad():
        eager = model.vocode(mel)
    torch.testing.assert_close(out, eager, rtol=0.0, atol=0.0)


def test_nested_capture_routes_eager():
    """G7: replaying a graph while an outer capture is active is illegal, so
    the wrapper must route eager (the vocode itself is capture-safe, so the
    outer capture still succeeds)."""
    model = _make_tiny_vocoder()
    wrapper = _make_wrapper(model, (64,))
    wrapper.warmup(DEVICE, torch.float32)
    mel = _mel(64, seed=6)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        wrapper(mel)
    torch.cuda.current_stream().wait_stream(side)

    outer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(outer):
        wrapper(mel)
    assert wrapper.last_call_info["mode"] == "eager"


def test_capture_failure_skips_size_and_warns(monkeypatch):
    """G8: one failed bucket logs one warning and falls back to eager for that
    size only; the other buckets still replay."""
    model = _make_tiny_vocoder()
    original = model.vocode
    calls = {"n": 0}

    def flaky(mel):
        if mel.shape[-1] == 128:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
        return original(mel)

    monkeypatch.setattr(model, "vocode", flaky)
    warnings = []
    monkeypatch.setattr(
        bg.logger,
        "warning",
        lambda msg, *a, **k: warnings.append(msg % a if a else msg),
        raising=True,
    )

    wrapper = _make_wrapper(model, (64, 128))
    wrapper.warmup(DEVICE, torch.float32)
    assert 64 in wrapper.graphs and 128 not in wrapper.graphs
    assert len([w for w in warnings if "128" in str(w)]) == 1

    # The failing size still produces correct output via eager.
    out = wrapper(_mel(128, seed=7))
    assert wrapper.last_call_info["mode"] == "eager"
    with torch.no_grad():
        eager = original(_mel(128, seed=7))
    torch.testing.assert_close(out, eager, rtol=0.0, atol=0.0)
    # And the healthy bucket still replays.
    wrapper(_mel(64, seed=8))
    assert wrapper.last_call_info["mode"] == "graph"


def test_streaming_widths_are_exact_hits():
    """G10: the derived streaming widths (first chunk = chunk_size, middle
    chunk = chunk_size + 2*future_size) must be exact-shape hits with zero
    padding; a tail width pads to the nearest bucket."""
    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
        _vocoder_graph_capture_sizes,
    )

    sizes = _vocoder_graph_capture_sizes(chunk_size=72, future_size=20, ladder=[128, 256])
    assert sizes == [72, 112, 128, 256]

    model = _make_tiny_vocoder()
    wrapper = _make_wrapper(model, sizes)
    wrapper.warmup(DEVICE, torch.float32)

    for t in (72, 112):
        wrapper(_mel(t, seed=t))
        assert wrapper.last_call_info["mode"] == "graph"
        assert wrapper.last_call_info["padding"] == 0

    wrapper(_mel(47, seed=9))
    assert wrapper.last_call_info["bucket"] == 72
    assert wrapper.last_call_info["padding"] == 25


def test_graph_contains_fused_or_eager_per_capture_time(monkeypatch):
    """G11: the graph records whichever alias-free path is active at capture
    time. Default capture (fused dispatch on) matches fused eager; capture
    with the fused kernel disabled matches the eager-eager path at the fused
    kernel's own noise floor."""
    model = _make_tiny_vocoder()
    mel = _mel(64, seed=10)

    wrapper = _make_wrapper(model, (64,))
    wrapper.warmup(DEVICE, torch.float32)
    fused_graph = wrapper(mel)
    with torch.no_grad():
        fused_eager = model.vocode(mel)
    torch.testing.assert_close(fused_graph, fused_eager, rtol=0.0, atol=1e-4)

    monkeypatch.setattr(AliasFreeActivation1d, "_fused_disabled", True)
    wrapper2 = _make_wrapper(model, (64,))
    wrapper2.warmup(DEVICE, torch.float32)
    eager_graph = wrapper2(mel)
    with torch.no_grad():
        eager_eager = model.vocode(mel)
    torch.testing.assert_close(eager_graph, eager_eager, rtol=0.0, atol=1e-4)
    # Cross-path: graph-of-eager vs fused-eager differs only at the fused
    # kernel's tap-reorder floor (measured ~1e-3 through this tiny vocoder).
    torch.testing.assert_close(eager_graph, fused_eager, rtol=0.0, atol=5e-3)


def test_warmup_is_idempotent():
    """G12: warming up twice does not duplicate graphs."""
    model = _make_tiny_vocoder()
    wrapper = _make_wrapper(model, (64, 128))
    wrapper.warmup(DEVICE, torch.float32)
    wrapper.warmup(DEVICE, torch.float32)
    assert len(wrapper.graphs) == 2


def test_cpu_input_stays_eager():
    """G13: a CPU-resident model never captures; calls route eager."""
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
    model = Qwen2_5OmniToken2WavBigVGANModel(config).eval()
    wrapper = _make_wrapper(model, (64,))
    wrapper.warmup(torch.device("cpu"), torch.float32)
    assert not wrapper._warmed_up and not wrapper.graphs
    torch.manual_seed(11)
    mel = torch.randn(1, 80, 64) - 5.0

    out = wrapper(mel)
    assert wrapper.last_call_info["mode"] == "eager"
    with torch.no_grad():
        eager = model.vocode(mel)
    torch.testing.assert_close(out, eager, rtol=0.0, atol=0.0)


def test_lazy_warmup_inside_inference_mode():
    """G15: first call triggers lazy warmup from inside ``inference_mode``;
    static buffers must be normal tensors so a later non-inference caller
    can still replay and get identical output."""
    model = _make_tiny_vocoder()
    wrapper = _make_wrapper(model, (64,))
    mel = _mel(64, seed=12)

    with torch.inference_mode():
        out1 = wrapper(mel)
        assert wrapper.last_call_info["mode"] == "graph"
    out2 = wrapper(mel)
    assert torch.equal(out1, out2)
    with torch.no_grad():
        eager = model.vocode(mel)
    torch.testing.assert_close(out2, eager, rtol=0.0, atol=1e-4)


def test_cudnn_benchmark_on_during_warmup_and_restored():
    """The capture warm runs turn on cuDNN autotune (heuristic algos are
    ~2x slower for these convs); the process-global flag must be on during
    capture and restored afterwards, even when a bucket fails."""
    model = _make_tiny_vocoder()
    original = model.vocode

    def flaky(mel):
        if mel.shape[-1] == 128:
            raise RuntimeError("boom")
        return original(mel)

    model.vocode = flaky
    wrapper = _make_wrapper(model, (64, 128))
    seen = []
    real_capture = wrapper._capture

    def spy(size, device, dtype):
        seen.append(torch.backends.cudnn.benchmark)
        return real_capture(size, device, dtype)

    wrapper._capture = spy

    prev = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        wrapper.warmup(DEVICE, torch.float32)
        assert seen == [True, True]  # on for both buckets, including the failing one
        assert torch.backends.cudnn.benchmark is False  # restored
        assert 64 in wrapper.graphs and 128 not in wrapper.graphs
    finally:
        torch.backends.cudnn.benchmark = prev
        model.vocode = original


def test_dtype_mismatch_stays_eager():
    """A mel whose dtype differs from the captured static buffers routes
    eager (routing check: the fp32 test model cannot vocode bf16 at all, so
    the eager callable is replaced by a recorder)."""
    model = _make_tiny_vocoder()
    wrapper = _make_wrapper(model, (64,))
    wrapper.warmup(DEVICE, torch.float32)

    wrapper.capture_fn = lambda mel: mel  # recorder: routing is the subject
    mel_bf16 = _mel(64, seed=13).to(torch.bfloat16)
    out = wrapper(mel_bf16)
    assert wrapper.last_call_info["mode"] == "eager"
    assert wrapper.last_call_info["reason"] == "warmup_failed_or_mismatched"
    assert out is mel_bf16
