# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AuK codec's decode fast paths keep the reference numerics.

Five paths are checked against the plain eager decode: the Snake activation
with precomputed exponent caches, the per-channel FIR filter cache, the
per-length CUDA graph tier, the torch.compile bucket tier (both fall back to
eager off CUDA) and the tiled decode of long clips.
"""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.auk.auk_vae import AuKVAE, LowPass, SnakeBeta, Upsample
from vllm_omni.diffusion.models.auk.vae_cudagraph import AuKVAEDecodeGraph, plan_tiles
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]


@pytest.fixture(autouse=True)
def _bound_cpu_decode_threads(request):
    if request.node.get_closest_marker("cpu") is None:
        yield
        return
    # These narrow convolutions run in shared CI pods. A host-sized Torch
    # pool spends more time coordinating workers than decoding small clips.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    print(f"AuK CPU decode threads: {previous_threads} -> {torch.get_num_threads()}")
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def _small_vae() -> AuKVAE:
    torch.manual_seed(3)
    # Six halvings of the initial width must leave at least two channels.
    vae = AuKVAE(
        upsample_initial_channel=128,
        downsample_channels=(4, 4, 8, 8, 8, 16, 16),
        latent_dim=8,
    ).eval()
    # Give the activations non-trivial parameters so the paths are exercised.
    with torch.no_grad():
        for module in vae.modules():
            if isinstance(module, SnakeBeta):
                module.alpha.normal_(0.0, 0.3)
                module.beta.normal_(0.0, 0.3)
    vae.remove_weight_norm()
    return vae


def _reference_snake(module: SnakeBeta, x: torch.Tensor) -> torch.Tensor:
    """The formula as the reference implementation writes it."""
    alpha = module.alpha.unsqueeze(0).unsqueeze(-1)
    beta = module.beta.unsqueeze(0).unsqueeze(-1)
    if module.alpha_logscale:
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
    return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


def _assert_plain_graph_matches_eager(replay: torch.Tensor, eager: torch.Tensor) -> None:
    assert replay.shape == eager.shape and replay.dtype == eager.dtype
    assert torch.isfinite(replay).all() and torch.isfinite(eager).all()
    error = (replay - eager).abs()
    rms = error.square().mean().sqrt().item()
    print(f"AuK plain graph: max_abs={error.max().item():.9g}, rms={rms:.9g}")
    if current_omni_platform.is_rocm():
        # MI300 build 13040 measured max_abs <= 1.64e-7 and RMS <= 4.13e-8.
        # Bound both peak and aggregate FP32 error; near-zero samples must
        # not hide behind a relative tolerance. NVIDIA retains bit parity.
        torch.testing.assert_close(replay, eager, atol=1e-6, rtol=0.0)
        assert rms <= 1e-7, rms
    else:
        assert torch.equal(replay, eager)


@pytest.mark.cpu
@torch.inference_mode()
def test_window_padding_decodes_to_raw_zero() -> None:
    """Bucket padding is the normalized latent whose denormalized value is zero."""
    vae = _small_vae()
    with torch.no_grad():
        vae.global_mean.normal_()
        vae.global_log_std.uniform_(0.5, 2.0)
    pad = AuKVAEDecodeGraph(vae)._pad_for(torch.device("cpu"))
    raw = pad * torch.sqrt(vae.global_log_std) + vae.global_mean
    torch.testing.assert_close(raw, torch.zeros_like(raw), atol=1e-6, rtol=0.0)


@pytest.mark.cpu
@torch.inference_mode()
def test_eager_snake_matches_the_reference_formula() -> None:
    module = SnakeBeta(6, alpha_logscale=True)
    with torch.no_grad():
        module.alpha.normal_()
        module.beta.normal_()
    module.precompute_exp_cache()
    x = torch.randn(2, 6, 17)

    assert torch.equal(module(x), _reference_snake(module, x))


@pytest.mark.cpu
@torch.inference_mode()
def test_filter_cache_leaves_the_filters_and_output_unchanged() -> None:
    lowpass = LowPass(cutoff=0.25, half_width=0.3, stride=2, kernel_size=12, causal=True)
    upsample = Upsample(ratio=2, kernel_size=12)
    x = torch.randn(1, 5, 40)

    for module in (lowpass, upsample):
        module.cache_filters = False
        plain = module(x)
        module.cache_filters = True
        cached = module(x)
        assert torch.equal(cached, plain)
        assert module._expanded is not None and module._expanded.shape == (5, 1, 12)
        assert module._expanded.is_contiguous()
        # A different channel count rebuilds the cache rather than reusing it.
        module(torch.randn(1, 3, 40))
        assert module._expanded.shape == (3, 1, 12)


@pytest.mark.cpu
@torch.inference_mode()
def test_decode_fast_paths_reproduce_the_plain_decode() -> None:
    vae = _small_vae()
    latents = torch.randn(1, 6, vae.latent_dim)

    vae.set_decode_fast_paths(cached_filters=False)
    plain = vae.decode(latents)
    assert plain.shape == (1, 6 * vae.hop_size)

    vae.set_decode_fast_paths(cached_filters=True)
    fast = vae.decode(latents)
    assert torch.equal(fast, plain)
    assert all(module._cached for module in vae.modules() if isinstance(module, SnakeBeta))


@pytest.mark.cpu
@torch.inference_mode()
def test_graph_wrapper_falls_back_to_eager_off_cuda(mocker) -> None:
    vae = _small_vae()
    wrapper = AuKVAEDecodeGraph(vae, frame_alignment=64, compile_shapes=(8,))
    capture_spy = mocker.spy(wrapper, "_capture")
    compile_spy = mocker.spy(torch, "compile")
    latents = torch.randn(1, 6, vae.latent_dim)

    wrapper.warmup(torch.device("cpu"))
    assert torch.equal(wrapper(latents), vae.decode(latents))
    assert wrapper.last_mode == "eager"
    capture_spy.assert_not_called()
    compile_spy.assert_not_called()
    assert not wrapper._cache and not wrapper._compiled


@pytest.mark.cpu
def test_compiled_bucket_is_the_smallest_captured_one_that_fits() -> None:
    wrapper = AuKVAEDecodeGraph(_small_vae(), compile_shapes=(16, 8, 8))
    assert wrapper.compile_shapes == [8, 16]
    # Nothing captured yet: every length goes to the per-length graph tier.
    assert wrapper.compiled_bucket(6) is None
    wrapper._compiled = {8: object(), 16: object()}  # type: ignore[dict-item]
    assert [wrapper.compiled_bucket(frames) for frames in (6, 8, 9, 16, 17)] == [8, 8, 16, 16, None]
    # A bucket whose capture failed is skipped, not padded to.
    wrapper._compiled = {16: object()}  # type: ignore[dict-item]
    assert wrapper.compiled_bucket(6) == 16


@pytest.mark.cpu
def test_plan_tiles_covers_the_clip_once_with_full_context() -> None:
    assert plan_tiles(40, 64, 10, 3) == [(0, 40, 0, 40)]
    windows = plan_tiles(1000, 512, 53, 9)
    assert windows == [(0, 512, 0, 503), (450, 512, 503, 953), (488, 512, 953, 1000)]
    # Every emitted frame sits at least the context away from a window edge
    # that is not the clip's own edge, and the emitted ranges tile the clip.
    emitted = 0
    for start, width, emit_start, emit_end in windows:
        assert emit_start == emitted and start + width <= 1000
        assert start == 0 or emit_start - start >= 53
        assert start + width == 1000 or start + width - emit_end >= 9
        emitted = emit_end
    assert emitted == 1000
    # With the compiled buckets known, the last window shrinks to the smallest
    # one that holds the remainder plus the left context: 600 frames cost a
    # 512 and a 256 window instead of two 512s.
    assert plan_tiles(600, 512, 53, 9, sizes=(128, 256, 512)) == [(0, 512, 0, 503), (344, 256, 503, 600)]
    assert plan_tiles(1000, 512, 53, 9, sizes=(128, 256, 512))[-1] == (872, 128, 953, 1000)
    narrow = plan_tiles(200, 64, 53, 9)
    assert len(narrow) == 69
    assert [emit for _, _, emit, _ in narrow[:3]] == [0, 55, 57]
    assert all(a[3] == b[2] for a, b in zip(narrow, narrow[1:]))
    assert narrow[-1][3] == 200
    with pytest.raises(ValueError, match="must exceed"):
        plan_tiles(100, 60, 53, 9)


@pytest.mark.cpu
@torch.inference_mode()
def test_decode_context_bounds_the_measured_receptive_field() -> None:
    vae = _small_vae()
    left, right = vae.decode_context_frames()
    # The production geometry: causal stack, conv_pre and the upsamplers look ahead a little.
    assert (left, right) == (53, 9)
    frames, start, end = 240, 70, 190
    latents = torch.randn(1, frames, vae.latent_dim)
    whole = vae.decode(latents)[:, start * vae.hop_size : end * vae.hop_size]
    window = vae.decode(latents[:, start:end])
    per_frame = (window - whole).abs().reshape(-1, vae.hop_size).amax(dim=1)
    differing = (per_frame > 1e-5).nonzero().flatten().tolist()
    # Only frames within the context of a fake edge may differ, and some do:
    # the bound is tight enough that the halo is not wasted.
    assert differing and all(index < left or end - start - index <= right for index in differing)


@pytest.mark.cpu
@torch.inference_mode()
def test_tiled_decode_matches_the_whole_decode() -> None:
    vae = _small_vae()
    # Off the accelerator every window decodes eagerly, so this isolates the stitching.
    wrapper = AuKVAEDecodeGraph(vae, compile_shapes=(128,), tile_frames=128)
    assert wrapper.tile_frames == 128 and wrapper.context_frames == (53, 9)
    for frames in (128, 129, 200, 331):
        latents = torch.randn(1, frames, vae.latent_dim)
        whole = vae.decode(latents)
        tiled = wrapper(latents)
        assert tiled.shape == whole.shape
        torch.testing.assert_close(tiled, whole, atol=1e-5, rtol=0.0)
        assert wrapper.last_mode == ("eager" if frames == 128 else "tiled")
    # The streaming form yields the same audio in order.
    latents = torch.randn(1, 200, vae.latent_dim)
    pieces = [(start, chunk.clone()) for start, chunk in wrapper.decode_tiles(latents)]
    # A 128-frame tile avoids repeatedly decoding the same 62-frame halo.
    assert [start for start, _ in pieces] == [emit for _, _, emit, _ in plan_tiles(200, 128, 53, 9)]
    assert [start for start, _ in pieces][:3] == [0, 119, 185]
    torch.testing.assert_close(
        torch.cat([chunk for _, chunk in pieces], dim=1), vae.decode(latents), atol=1e-5, rtol=0.0
    )
    # tile_frames=0 turns tiling off; a tile smaller than the context is refused.
    assert AuKVAEDecodeGraph(vae, tile_frames=0).tile_frames == 0
    with pytest.raises(ValueError, match="must exceed"):
        AuKVAEDecodeGraph(vae, tile_frames=60)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_graph_replay_matches_eager_per_length() -> None:
    vae = _small_vae().to("cuda")
    wrapper = AuKVAEDecodeGraph(vae, max_graphs=2)
    pools = []
    saved_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frames in (6, 9, 6, 12, 6):
        latents = torch.randn(1, frames, vae.latent_dim, device="cuda")
        eager = vae.decode(latents)
        replay = wrapper(latents)
        _assert_plain_graph_matches_eager(replay, eager)
        # Returned waveforms own their storage and survive later replays
        # and retirement of the generation that produced them.
        for previous, snapshot in saved_outputs:
            assert torch.equal(previous, snapshot)
        saved_outputs.append((replay, replay.clone()))
        pools.append(wrapper._plain_pool)
    # The third distinct length found the cache full, so the whole generation
    # (6 and 9) was retired together with its pool rather than one graph at a
    # time; 12 and the re-captured 6 share the new pool.
    assert list(wrapper._cache) == [12, 6]
    assert pools[0] is pools[2] and pools[3] is pools[4] and pools[2] is not pools[3]


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_bucketed_graph_only_disturbs_the_tail() -> None:
    vae = _small_vae().to("cuda")
    wrapper = AuKVAEDecodeGraph(vae, frame_alignment=8)
    latents = torch.randn(1, 6, vae.latent_dim, device="cuda")
    eager = vae.decode(latents)
    replay = wrapper(latents)
    assert replay.shape == eager.shape and list(wrapper._cache) == [8]
    # The padding decodes to raw zero, which is what the eager conv_pre pads
    # with, but it still leaks in through the alias-free upsamplers, whose
    # lookahead accumulates through the stack, so a bucketed replay is close
    # to but not identical with the eager decode. That is why frame_alignment
    # defaults to 1.
    assert torch.isfinite(replay).all()
    assert not torch.equal(replay, eager)
    torch.testing.assert_close(replay, eager, atol=0.1, rtol=0.0)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile + CUDA graph capture requires CUDA")
@torch.inference_mode()
def test_compiled_buckets_replay_within_fusion_tolerance_and_leave_longer_clips_to_plain_graphs() -> None:
    vae = _small_vae().to("cuda")
    wrapper = AuKVAEDecodeGraph(vae, compile_shapes=(8,))
    wrapper.warmup(torch.device("cuda"))
    assert list(wrapper._compiled) == [8]

    exact = torch.randn(1, 8, vae.latent_dim, device="cuda")
    eager = vae.decode(exact)
    replay = wrapper(exact)
    assert wrapper.last_mode == "compiled"
    assert replay.shape == eager.shape
    # Same formula, different fusion order: not bit-identical, but close.
    torch.testing.assert_close(replay, eager, atol=1e-4, rtol=0.0)

    short = torch.randn(1, 5, vae.latent_dim, device="cuda")
    padded = wrapper(short)
    assert wrapper.last_mode == "compiled" and padded.shape == (1, 5 * vae.hop_size)
    torch.testing.assert_close(padded, vae.decode(short), atol=0.1, rtol=0.0)

    # An 8-frame bucket cannot hold the decoder context, so tiling is off and
    # a longer clip gets its own plain graph.
    assert wrapper.tile_frames == 0
    long = torch.randn(1, 12, vae.latent_dim, device="cuda")
    replay, eager = wrapper(long), vae.decode(long)
    _assert_plain_graph_matches_eager(replay, eager)
    assert wrapper.last_mode == "graph" and list(wrapper._cache) == [12]


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile + CUDA graph capture requires CUDA")
@torch.inference_mode()
def test_tiles_replay_the_compiled_bucket_for_long_clips() -> None:
    vae = _small_vae().to("cuda")
    wrapper = AuKVAEDecodeGraph(vae, compile_shapes=(72,))
    wrapper.warmup(torch.device("cuda"))
    assert wrapper.tile_frames == 72 and list(wrapper._compiled) == [72]

    latents = torch.randn(1, 150, vae.latent_dim, device="cuda")
    tiled = wrapper(latents)
    assert wrapper.last_mode == "tiled" and not wrapper._cache
    assert tiled.shape == (1, 150 * vae.hop_size)
    torch.testing.assert_close(tiled, vae.decode(latents), atol=1e-4, rtol=0.0)
