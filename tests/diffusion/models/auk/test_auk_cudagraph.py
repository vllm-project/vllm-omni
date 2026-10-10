# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-request eager and CUDA-graph AuK DiT sampling equivalence."""

import pytest
import torch

from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer, sample_latents
from vllm_omni.diffusion.models.auk.cudagraph_wrapper import AuKCUDAGraphWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_dit(device: str) -> AuKTransformer:
    torch.manual_seed(12)
    return (
        AuKTransformer(
            dim=32,
            heads=2,
            dim_head=16,
            ff_mult=2,
            latent_dim=4,
            text_hidden_dim=8,
            num_layers=2,
            num_single_layers=2,
        )
        .eval()
        .to(device)
    )


def _sample_inputs(device: str, offset: float = 0.0, ref_frames: int = 4) -> dict[str, torch.Tensor]:
    """``ref_frames=0`` is the text-only (instruct TTS) request shape."""
    generator = torch.Generator(device=device).manual_seed(42)
    return {
        "text": torch.randn(1, 7, 8, generator=generator, device=device) + offset,
        "c_mask": torch.ones(1, 7, dtype=torch.bool, device=device),
        "ref": (torch.randn(1, 4, 4, generator=generator, device=device) - offset)[:, :ref_frames],
        "ref_mask": torch.ones(1, ref_frames, dtype=torch.bool, device=device),
    }


@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_single_request_graph_wrapper_cpu_falls_back_to_eager(cfg_strength: float, mocker) -> None:
    dit = _make_dit("cpu")
    wrapper = AuKCUDAGraphWrapper(dit)
    prepare_spy = mocker.spy(wrapper, "_prepare")
    step_spy = mocker.spy(wrapper, "_step")
    capture_spy = mocker.spy(wrapper, "_capture")
    inputs = _sample_inputs("cpu")
    common = dict(
        **inputs,
        gen_frames=9,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=cfg_strength,
    )
    eager = sample_latents(dit, **common, generator=torch.Generator().manual_seed(7))
    graph = sample_latents(dit, **common, generator=torch.Generator().manual_seed(7), sampler=wrapper)
    torch.testing.assert_close(graph, eager)

    # The conditioning is prepared once per request and reused by every step.
    assert prepare_spy.call_count == 1
    # _prepare(x, x_mask, text, c_mask, ref, ref_mask, uses_cfg, timesteps)
    assert prepare_spy.call_args.args[6] is (cfg_strength >= 1e-5)
    # The time grid reaches prepare, so the adaLN modulations are computed once.
    assert prepare_spy.call_args.args[7].numel() == 2
    assert step_spy.call_count == 2
    capture_spy.assert_not_called()
    assert not wrapper._cache


def test_graph_inputs_use_bounded_length_buckets() -> None:
    wrapper = AuKCUDAGraphWrapper(_make_dit("cpu"))
    x = torch.ones(1, 65, 4)
    text = torch.ones(1, 65, 8)
    c_mask = torch.ones(1, 65, dtype=torch.bool)
    ref = torch.ones(1, 51, 4)
    ref_mask = torch.ones(1, 51, dtype=torch.bool)

    padded = wrapper._bucket_inputs(x, text, c_mask, ref, ref_mask)

    assert padded[0].shape == (1, 96, 4)
    assert padded[1].shape == (1, 96)
    assert padded[2].shape == (1, 96, 8)
    assert padded[3].shape == (1, 96)
    assert padded[4].shape == (1, 100, 4)
    assert padded[5].shape == (1, 100)
    assert [mask.sum().item() for mask in (padded[1], padded[3], padded[5])] == [65, 65, 51]
    assert wrapper._key(padded[0], padded[2], padded[4], False) == (1, 96, 96, 100, False, 0)
    assert wrapper._key(padded[0], padded[2], padded[4], False, 32) == (1, 96, 96, 100, False, 32)
    assert wrapper.max_graphs == 32


def test_full_graph_cache_retires_as_one_generation() -> None:
    wrapper = AuKCUDAGraphWrapper(_make_dit("cpu"), max_graphs=2)
    first = (1, 64, 64, 50, False, 0)
    second = (2, 128, 64, 50, False, 2)
    wrapper._cache[first] = object()

    wrapper._retire_graph_generation_if_full()
    assert list(wrapper._cache) == [first]

    wrapper._loop_cache[second] = object()  # type: ignore[assignment]
    wrapper._retire_graph_generation_if_full()
    assert not wrapper._cache and not wrapper._loop_cache


@torch.inference_mode()
@pytest.mark.parametrize("ref_frames", [0, 4])
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_bucket_padding_preserves_real_frame_outputs(cfg_strength: float, ref_frames: int) -> None:
    dit = _make_dit("cpu")
    wrapper = AuKCUDAGraphWrapper(dit)
    inputs = _sample_inputs("cpu", ref_frames=ref_frames)
    x = torch.randn(1, 9, 4)
    timestep = torch.tensor(0.4)
    cfg = torch.tensor(cfg_strength)
    uses_cfg = cfg_strength >= 1e-5
    text, c_mask, ref, ref_mask = inputs["text"], inputs["c_mask"], inputs["ref"], inputs["ref_mask"]

    ctx = wrapper._prepare(x, None, text, c_mask, ref, ref_mask, uses_cfg)
    eager = wrapper._step(x, timestep, ctx, cfg)

    bucketed = wrapper._bucket_inputs(x, text, c_mask, ref, ref_mask)
    padded_ctx = wrapper._prepare(*bucketed, uses_cfg)
    padded = wrapper._step(bucketed[0], timestep, padded_ctx, cfg)

    torch.testing.assert_close(padded[:, : x.shape[1]], eager)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("ref_frames", [0, 4])
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_single_request_graph_replay_matches_eager_and_updates_inputs(cfg_strength: float, ref_frames: int) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit)

    for offset in (0.0, 0.25):
        inputs = _sample_inputs("cuda", offset, ref_frames=ref_frames)
        common = dict(
            **inputs,
            gen_frames=9,
            t_grid=[0.0, 0.4, 1.0],
            cfg_strength=cfg_strength,
        )
        eager = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7))
        graph = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper)
        torch.testing.assert_close(graph, eager, atol=3e-6, rtol=3e-5)

        bucketed = wrapper._bucket_inputs(
            torch.empty(1, 9, 4, device="cuda"),
            inputs["text"],
            inputs["c_mask"],
            inputs["ref"],
            inputs["ref_mask"],
        )
        # sample_latents passes the grid, so the key carries its step count (two steps here).
        key = wrapper._key(bucketed[0], bucketed[2], bucketed[4], cfg_strength >= 1e-5, 2)
        assert key in wrapper._loop_cache

        # The static context was refreshed for this request's conditioning.
        entry = wrapper._loop_cache[key]
        grid = torch.tensor([0.0, 0.4], device="cuda")
        fresh = wrapper._prepare(*bucketed, cfg_strength >= 1e-5, grid)
        for static, want in zip(entry.static_ctx.tensors(), fresh.tensors(), strict=True):
            if want is None:
                assert static is None
            else:
                torch.testing.assert_close(static, want)

    assert len(wrapper._loop_cache) == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_graph_capture_failure_falls_back_to_eager(mocker) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit)
    capture = mocker.patch.object(wrapper, "_capture_loop", side_effect=RuntimeError("capture failed"))
    inputs = _sample_inputs("cuda")
    common = dict(
        **inputs,
        gen_frames=9,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=0.0,
    )

    # First attempt: capture fails, warning logged, falls back to eager execution.
    res1 = sample_latents(
        dit,
        **common,
        generator=torch.Generator(device="cuda").manual_seed(7),
        sampler=wrapper,
    )
    assert res1 is not None

    # Second attempt: key is uncapturable, bypasses capture immediately and runs eagerly.
    res2 = sample_latents(
        dit,
        **common,
        generator=torch.Generator(device="cuda").manual_seed(7),
        sampler=wrapper,
    )
    assert res2 is not None

    assert wrapper.enabled
    assert capture.call_count == 1
    assert not wrapper._loop_cache


def test_pad_batch_time_pads_batch_and_time() -> None:
    x = torch.ones(2, 3, 4)
    padded = AuKCUDAGraphWrapper._pad_batch_time(x, 4, 5)
    assert padded.shape == (4, 5, 4)
    assert torch.equal(padded[:2, :3], x)
    assert torch.equal(padded[2:], torch.zeros(2, 5, 4))
    mask = torch.ones(2, 3, dtype=torch.bool)
    padded_mask = AuKCUDAGraphWrapper._pad_batch_time(mask, 4, 5, mask=True)
    assert padded_mask.shape == (4, 5)
    assert padded_mask.dtype == torch.bool
    assert not padded_mask[2:].any() and not padded_mask[:, 3:].any()


def test_larger_loop_entry_reuses_near_batch_and_rejects_tiny_batch() -> None:
    wrapper = AuKCUDAGraphWrapper(_make_dit("cpu"), enabled=False)
    fake128 = object()
    fake8 = object()
    wrapper._loop_cache[(128, 160, 32, 0, True, 4)] = fake128  # type: ignore[assignment]
    wrapper._loop_cache[(8, 160, 32, 0, True, 4)] = fake8  # type: ignore[assignment]

    _, entry = wrapper._larger_loop_entry(127, 160, 32, 0, True, 4)
    assert entry is fake128
    _, entry = wrapper._larger_loop_entry(1, 160, 32, 0, True, 4)
    assert entry is None
    _, entry = wrapper._larger_loop_entry(7, 160, 32, 0, True, 4)
    assert entry is fake8
    _, entry = wrapper._larger_loop_entry(127, 160, 32, 0, True, 32)
    assert entry is None
    # Reusing a longer conditioning sequence adds work and changes bf16 rounding.
    _, entry = wrapper._larger_loop_entry(7, 160, 16, 0, True, 4)
    assert entry is None


@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_batch_sampling_preserves_each_request_seed(cfg_strength: float) -> None:
    dit = _make_dit("cpu")
    first, second = _sample_inputs("cpu"), _sample_inputs("cpu", 0.25)
    common = dict(gen_frames=9, t_grid=[0.0, 0.4, 1.0], cfg_strength=cfg_strength)
    expected = torch.cat(
        [
            sample_latents(dit, **inputs, **common, generator=torch.Generator().manual_seed(seed))
            for inputs, seed in zip((first, second), (7, 11))
        ]
    )
    inputs = {name: torch.cat([first[name], second[name]]) for name in first}
    actual = sample_latents(
        dit,
        **inputs,
        **common,
        generator=[torch.Generator().manual_seed(seed) for seed in (7, 11)],
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="one noise generator per request"):
        sample_latents(dit, **inputs, **common, generator=[torch.Generator().manual_seed(7)])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loop_replay_updates_schedule_and_reuses_larger_batch(cfg_strength: float, dtype: torch.dtype) -> None:
    dit = _make_dit("cuda").to(dtype=dtype)
    wrapper = AuKCUDAGraphWrapper(dit)
    inputs = _sample_inputs("cuda")
    inputs = {name: tensor.to(dtype) if tensor.is_floating_point() else tensor for name, tensor in inputs.items()}
    batched = {
        name: torch.cat([tensor, tensor + 0.25 if tensor.is_floating_point() else tensor])
        for name, tensor in inputs.items()
    }
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        sample_latents(
            dit,
            **batched,
            gen_frames=9,
            t_grid=[0.0, 0.4, 1.0],
            cfg_strength=cfg_strength,
            dtype=torch.float32,
            sampler=wrapper,
            generator=[torch.Generator(device="cuda").manual_seed(seed) for seed in (7, 11)],
        )
        for grid in ([0.0, 0.2, 1.0], [0.0, 0.7, 1.0]):
            common = dict(**inputs, gen_frames=9, t_grid=grid, cfg_strength=cfg_strength, dtype=torch.float32)
            expected = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7))
            actual = sample_latents(
                dit, **common, sampler=wrapper, generator=torch.Generator(device="cuda").manual_seed(7)
            )
            torch.testing.assert_close(
                actual,
                expected,
                # Padding changes GEMM/SDPA shapes and their bf16 rounding.
                # FP32 keeps the strict graph-replay tolerance above.
                atol=2e-2 if dtype == torch.bfloat16 else 3e-6,
                rtol=2e-2 if dtype == torch.bfloat16 else 3e-5,
            )
    assert len(wrapper._loop_cache) == 1
    assert actual.dtype == torch.float32


@torch.inference_mode()
def test_text_only_eager_batch_with_unequal_lengths_matches_serial() -> None:
    """Masking must prevent padded text positions from leaking into text-only attention."""
    dit = _make_dit("cpu")
    text1 = torch.randn(1, 7, 8)
    text2 = torch.randn(1, 3, 8)
    ref_empty = torch.zeros(1, 0, 4)
    ref_mask_empty = torch.zeros(1, 0, dtype=torch.bool)

    # 1. Serial execution of the short request
    common = dict(
        gen_frames=9,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=2.0,
        dtype=torch.float32,
        ref=ref_empty,
        ref_mask=ref_mask_empty,
    )
    expected_serial = sample_latents(
        dit,
        text=text2,
        c_mask=torch.ones(1, 3, dtype=torch.bool),
        generator=torch.Generator().manual_seed(11),
        **common,
    )

    # 2. Batched execution alongside the longer request
    padded_text = torch.zeros(2, 7, 8)
    padded_text[0] = text1[0]
    padded_text[1, :3] = text2[0]
    c_mask = torch.tensor([[True] * 7, [True] * 3 + [False] * 4])
    batched_ref = torch.zeros(2, 0, 4)
    batched_ref_mask = torch.zeros(2, 0, dtype=torch.bool)

    generators = [torch.Generator().manual_seed(7), torch.Generator().manual_seed(11)]
    actual_batch = sample_latents(
        dit,
        text=padded_text,
        c_mask=c_mask,
        ref=batched_ref,
        ref_mask=batched_ref_mask,
        generator=generators,
        gen_frames=9,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=2.0,
        dtype=torch.float32,
    )

    # Short request in the batch must match its serial execution with tight tolerance
    actual_short = actual_batch[1:2]
    torch.testing.assert_close(actual_short, expected_serial, atol=1e-5, rtol=1e-5)
