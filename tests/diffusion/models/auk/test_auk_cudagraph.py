# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-request eager and CUDA-graph AuK DiT sampling equivalence."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer, sample_latents
from vllm_omni.diffusion.models.auk.cudagraph_wrapper import AuKCUDAGraphWrapper

pytestmark = [pytest.mark.core_model]


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


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
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


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_graph_capture_failure_falls_back_to_eager(mocker) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit)
    capture_loop = mocker.patch.object(wrapper, "_capture_loop", side_effect=RuntimeError("loop capture failed"))
    capture_single = mocker.patch.object(wrapper, "_capture", side_effect=RuntimeError("single step capture failed"))
    inputs = _sample_inputs("cuda")
    common = dict(
        **inputs,
        gen_frames=9,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=0.0,
    )

    # First attempt: capture fails, warning logged, falls back to genuine eager execution
    # (without attempting single-step capture which would re-raise).
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
    assert capture_loop.call_count == 1
    capture_single.assert_not_called()
    assert not wrapper._loop_cache

    # Mixed-length batch fallback: verify _run_eager_loop receives unbucketed x_mask (shape [2, 9])
    # and matches serial execution for the shorter row.
    eager_loop_spy = mocker.spy(wrapper, "_run_eager_loop")
    batched_inputs = {name: torch.cat([tensor, tensor]) for name, tensor in inputs.items()}
    x_mask_mixed = torch.tensor([[True] * 6 + [False] * 3, [True] * 9], device="cuda")
    serial_short = sample_latents(
        dit,
        **inputs,
        gen_frames=9,
        x_mask=x_mask_mixed[:1],
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=0.0,
        generator=torch.Generator(device="cuda").manual_seed(17),
    )
    fallback_batch = sample_latents(
        dit,
        **batched_inputs,
        gen_frames=9,
        x_mask=x_mask_mixed,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=0.0,
        generator=[
            torch.Generator(device="cuda").manual_seed(17),
            torch.Generator(device="cuda").manual_seed(23),
        ],
        sampler=wrapper,
    )
    assert eager_loop_spy.call_args.kwargs["x_mask"].shape == (2, 9)
    torch.testing.assert_close(fallback_batch[:1, :6], serial_short[:, :6], atol=1e-5, rtol=1e-5)


@pytest.mark.cpu
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


@pytest.mark.cpu
def test_fit_loop_inputs_preserves_valid_c_mask_on_padded_rows() -> None:
    from types import SimpleNamespace

    wrapper = AuKCUDAGraphWrapper(_make_dit("cpu"), enabled=False)
    fake_entry = SimpleNamespace(
        static_x_in=torch.zeros(4, 9, 4),
        static_ctx=SimpleNamespace(c=torch.zeros(4, 8, 8), prompt=torch.zeros(4, 4, 4)),
    )
    x = torch.randn(2, 6, 4)
    x_mask = torch.ones(2, 6, dtype=torch.bool)
    text = torch.randn(2, 5, 8)
    c_mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    ref = torch.randn(2, 3, 4)
    ref_mask = torch.ones(2, 3, dtype=torch.bool)

    _, _, _, fitted_c_mask, _, _ = wrapper._fit_loop_inputs(
        fake_entry,
        x,
        x_mask,
        text,
        c_mask,
        ref,
        ref_mask,  # type: ignore[arg-type]
    )
    assert fitted_c_mask.shape == (4, 8)
    assert torch.equal(fitted_c_mask[:2, :5], c_mask)
    assert not fitted_c_mask[:2, 5:].any()
    assert fitted_c_mask[2:, 0].all()
    assert not fitted_c_mask[2:, 1:].any()


@pytest.mark.cpu
def test_is_oom_capture_error_classifies_oom_and_non_oom() -> None:
    oom_exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
    wrapped = RuntimeError("capture aborted")
    wrapped.__cause__ = oom_exc
    assert AuKCUDAGraphWrapper._is_oom_capture_error(oom_exc) is True
    assert AuKCUDAGraphWrapper._is_oom_capture_error(RuntimeError("CUDA out of memory.")) is True
    assert AuKCUDAGraphWrapper._is_oom_capture_error(wrapped) is True
    assert AuKCUDAGraphWrapper._is_oom_capture_error(RuntimeError("unsupported op")) is False


@pytest.mark.cpu
def test_retire_graph_generation_preserves_uncapturable_and_advances_generation() -> None:
    wrapper = AuKCUDAGraphWrapper(_make_dit("cpu"), enabled=False, max_graphs=1)
    bad_key = (2, 32, 16, 0, False, 4)
    oom_key = (4, 32, 16, 0, False, 4)
    wrapper._uncapturable_loop_keys.add(bad_key)
    wrapper._oom_loop_failed_gen[oom_key] = wrapper._cache_generation
    wrapper._loop_cache[(1, 32, 16, 0, False, 4)] = object()  # type: ignore[assignment]
    prev_gen = wrapper._cache_generation

    wrapper._retire_graph_generation_if_full()

    assert wrapper._loop_cache == {}
    assert bad_key in wrapper._uncapturable_loop_keys
    assert wrapper._cache_generation == prev_gen + 1
    assert wrapper.retirements == 1
    assert wrapper._oom_loop_failed_gen[oom_key] < wrapper._cache_generation


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_oom_loop_capture_retries_after_retirement_then_pins(mocker) -> None:
    dit = _make_dit("cuda")
    inputs = _sample_inputs("cuda")
    common = dict(**inputs, gen_frames=9, t_grid=[0.0, 0.4, 1.0], cfg_strength=0.0)

    # 1. Transient OOM waits strictly for cache retirement (does not auto-expire after repeated calls).
    wrapper = AuKCUDAGraphWrapper(dit, max_graphs=1)
    real_capture_loop = wrapper._capture_loop
    outcomes_transient: list[object] = [
        torch.cuda.OutOfMemoryError("CUDA out of memory during capture"),
        "real",
    ]
    attempts = 0

    def flaky_capture(*args, **kwargs):
        nonlocal attempts
        outcome = outcomes_transient[attempts]
        attempts += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return real_capture_loop(*args, **kwargs)

    mocker.patch.object(wrapper, "_capture_loop", side_effect=flaky_capture)
    out1 = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper)
    assert out1 is not None and attempts == 1
    assert not wrapper._uncapturable_loop_keys
    assert len(wrapper._oom_loop_failed_gen) == 1
    assert wrapper.eager_loop_fallbacks == 1

    # Even after >16 calls with no cache retirement, capture is not retried in the same generation.
    for _ in range(20):
        out_wait = sample_latents(
            dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper
        )
        assert out_wait is not None
    assert attempts == 1
    assert wrapper.eager_loop_fallbacks == 21

    # Retiring the generation advances _cache_generation so capture retries and succeeds.
    wrapper._loop_cache[("dummy",)] = object()  # type: ignore[assignment]
    wrapper._retire_graph_generation_if_full()
    assert wrapper.retirements == 1
    out3 = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper)
    assert out3 is not None and attempts == 2
    assert wrapper.loop_captures == 1
    assert len(wrapper._loop_cache) == 1
    assert wrapper._oom_loop_failed_gen == {}
    assert wrapper._oom_loop_retry_counts == {}

    # 2. Repeated OOM on a fresh wrapper: retries once after retirement, then pins permanently.
    wrapper2 = AuKCUDAGraphWrapper(dit, max_graphs=1)
    attempts2 = 0

    def always_oom(*args, **kwargs):
        nonlocal attempts2
        attempts2 += 1
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    mocker.patch.object(wrapper2, "_capture_loop", side_effect=always_oom)
    sample_a = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper2)
    assert sample_a is not None and attempts2 == 1
    wrapper2._loop_cache[("dummy",)] = object()  # type: ignore[assignment]
    wrapper2._retire_graph_generation_if_full()
    sample_b = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper2)
    assert sample_b is not None and attempts2 == 2
    assert len(wrapper2._uncapturable_loop_keys) == 1
    assert wrapper2._oom_loop_failed_gen == {}

    # Even after another retirement, the OOM-exhausted key stays permanently eager.
    wrapper2._loop_cache[("dummy",)] = object()  # type: ignore[assignment]
    wrapper2._retire_graph_generation_if_full()
    sample_c = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper2)
    assert sample_c is not None and attempts2 == 2

    # 3. loop_graph=False skips whole-loop capture cleanly even on CUDA.
    wrapper_noloop = AuKCUDAGraphWrapper(dit, loop_graph=False)
    sample_loop_spy = mocker.spy(wrapper_noloop, "sample_loop")
    out_single_step = sample_latents(
        dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper_noloop
    )
    assert out_single_step is not None
    sample_loop_spy.assert_not_called()
    assert not wrapper_noloop._loop_cache and len(wrapper_noloop._cache) == 1


@pytest.mark.cpu
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
    # Boundary checks on unbucketed batch size: 33 (bucket 64) must not pad to 128 (~4x),
    # while 65 (<= 128/2) and 4 (<= 8/2) may reuse the larger cached graph.
    _, entry = wrapper._larger_loop_entry(33, 160, 32, 0, True, 4)
    assert entry is None
    _, entry = wrapper._larger_loop_entry(65, 160, 32, 0, True, 4)
    assert entry is fake128
    _, entry = wrapper._larger_loop_entry(3, 160, 32, 0, True, 4)
    assert entry is None
    _, entry = wrapper._larger_loop_entry(4, 160, 32, 0, True, 4)
    assert entry is fake8


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_loop_capture_oom_releases_allocations_before_eager_and_recovers_from_prepare_oom(mocker) -> None:
    import weakref

    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit, max_graphs=1)
    inputs = _sample_inputs("cuda")
    common = dict(**inputs, gen_frames=9, t_grid=[0.0, 0.4, 1.0], cfg_strength=0.0)

    # Track when transient objects held in _prepare / _capture_loop frames are released relative to _run_eager_loop.
    transient_ref: weakref.ref | None = None
    transient_alive_at_eager: list[bool] = []
    real_run_eager_loop = wrapper._run_eager_loop

    class _Sentinel:
        pass

    def oom_in_capture(*args, **kwargs):
        nonlocal transient_ref
        sentinel = _Sentinel()
        transient_ref = weakref.ref(sentinel)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory inside _capture_loop")

    def spy_run_eager_loop(*args, **kwargs):
        transient_alive_at_eager.append(transient_ref is not None and transient_ref() is not None)
        return real_run_eager_loop(*args, **kwargs)

    mocker.patch.object(wrapper, "_capture_loop", side_effect=oom_in_capture)
    mocker.patch.object(wrapper, "_run_eager_loop", side_effect=spy_run_eager_loop)
    out = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper)
    assert out is not None
    assert transient_alive_at_eager == [False]
    assert len(wrapper._oom_loop_failed_gen) == 1
    assert not wrapper._uncapturable_loop_keys

    # Simulate OOM during _prepare (during bucket/prepare setup) -> recovers into failed_gen tracking + eager fallback.
    wrapper2 = AuKCUDAGraphWrapper(dit, max_graphs=1)
    real_prepare = wrapper2._prepare
    prepare_calls = 0

    def oom_once_in_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory inside _prepare")
        return real_prepare(*args, **kwargs)

    mocker.patch.object(wrapper2, "_prepare", side_effect=oom_once_in_prepare)
    out2 = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper2)
    assert out2 is not None
    assert prepare_calls == 2
    assert len(wrapper2._oom_loop_failed_gen) == 1
    assert not wrapper2._uncapturable_loop_keys


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile + CUDA graph capture requires CUDA")
@torch.inference_mode()
def test_full_compiled_dit_step_avoids_per_step_recompilation_in_loops() -> None:
    dit = _make_dit("cuda")
    inputs = _sample_inputs("cuda")
    t_grid = [i / 12.0 for i in range(13)]
    common = dict(**inputs, gen_frames=9, t_grid=t_grid, cfg_strength=0.0)

    shifted_grid = [0.0] + [(i - 0.1) / 12.0 for i in range(1, 12)] + [1.0]
    expected = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7))
    shifted_expected = sample_latents(
        dit, **{**common, "t_grid": shifted_grid}, generator=torch.Generator(device="cuda").manual_seed(7)
    )
    try:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        dit.step = torch.compile(dit.step, dynamic=False)

        graph_wrapper = AuKCUDAGraphWrapper(dit, enabled=True)
        graph_out = sample_latents(
            dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=graph_wrapper
        )
        torch.testing.assert_close(graph_out, expected, atol=1e-4, rtol=1e-4)
        assert dict(torch._dynamo.utils.counters.get("stats", {})).get("unique_graphs") == 1

        # A second loop replay with a shifted 12-step grid must trigger 0 new compiles.
        torch._dynamo.utils.counters.clear()
        shifted_out = sample_latents(
            dit,
            **{**common, "t_grid": shifted_grid},
            generator=torch.Generator(device="cuda").manual_seed(7),
            sampler=graph_wrapper,
        )
        torch.testing.assert_close(shifted_out, shifted_expected, atol=1e-4, rtol=1e-4)
        assert dict(torch._dynamo.utils.counters.get("frames", {})).get("total", 0) == 0

        # Unbucketed 12-step _run_eager_loop compiles once for its shape, then _sample_latents reuses it.
        torch._dynamo.utils.counters.clear()
        eager_wrapper = AuKCUDAGraphWrapper(dit, enabled=False)
        eager_out = sample_latents(
            dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=eager_wrapper
        )
        torch.testing.assert_close(eager_out, expected, atol=1e-4, rtol=1e-4)
        assert dict(torch._dynamo.utils.counters.get("stats", {})).get("unique_graphs") == 1

        torch._dynamo.utils.counters.clear()
        direct_out = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7))
        torch.testing.assert_close(direct_out, expected, atol=1e-4, rtol=1e-4)
        assert dict(torch._dynamo.utils.counters.get("frames", {})).get("total", 0) == 0
    finally:
        torch._dynamo.reset()


@pytest.mark.cpu
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
    with pytest.raises(ValueError, match="when batch_size > 1"):
        sample_latents(dit, **inputs, **common, generator=torch.Generator().manual_seed(7))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
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


@pytest.mark.cpu
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


@pytest.mark.cpu
@torch.inference_mode()
def test_batched_unequal_target_lengths_matches_serial() -> None:
    """Masking on target latents ensures shorter requests in a batch match their serial generation."""
    dit = _make_dit("cpu")
    text = torch.randn(2, 7, 8)
    c_mask = torch.ones(2, 7, dtype=torch.bool)
    ref = torch.randn(2, 4, 4)
    ref_mask = torch.ones(2, 4, dtype=torch.bool)

    # 1. Serial run of request 0 with target_bucket=9, masked to 6 valid frames
    gen0 = torch.Generator().manual_seed(42)
    x_mask0 = torch.tensor([[True] * 6 + [False] * 3])
    expected_serial0 = sample_latents(
        dit,
        text=text[0:1],
        c_mask=c_mask[0:1],
        ref=ref[0:1],
        ref_mask=ref_mask[0:1],
        gen_frames=9,
        x_mask=x_mask0,
        generator=gen0,
        t_grid=[0.0, 0.5, 1.0],
        cfg_strength=2.0,
        dtype=torch.float32,
    )

    # 2. Batched run with target_bucket=9, where row 0 has length 6 and row 1 has length 9
    gen0_batch = torch.Generator().manual_seed(42)
    gen1_batch = torch.Generator().manual_seed(99)
    x_mask = torch.tensor([[True] * 6 + [False] * 3, [True] * 9])

    actual_batch = sample_latents(
        dit,
        text=text,
        c_mask=c_mask,
        ref=ref,
        ref_mask=ref_mask,
        gen_frames=9,
        x_mask=x_mask,
        generator=[gen0_batch, gen1_batch],
        t_grid=[0.0, 0.5, 1.0],
        cfg_strength=2.0,
        dtype=torch.float32,
    )

    # Valid frames of row 0 must match serial generation exactly
    torch.testing.assert_close(actual_batch[0:1, :6], expected_serial0[:, :6], atol=1e-5, rtol=1e-5)
    # The serial and batched calls with the same generator and bucket draw identical initial noise
    torch.testing.assert_close(actual_batch[0:1], expected_serial0, atol=1e-5, rtol=1e-5)


@pytest.mark.cpu
@torch.inference_mode()
def test_sample_latents_noise_frames_matches_unpadded_serial() -> None:
    """``noise_frames`` consumes the exact unpadded RNG count per request before padding."""
    dit = _make_dit("cpu")
    text = torch.randn(2, 7, 8)
    c_mask = torch.ones(2, 7, dtype=torch.bool)
    ref = torch.randn(2, 4, 4)
    ref_mask = torch.ones(2, 4, dtype=torch.bool)

    gen0_serial = torch.Generator().manual_seed(42)
    expected_serial0 = sample_latents(
        dit,
        text=text[0:1],
        c_mask=c_mask[0:1],
        ref=ref[0:1],
        ref_mask=ref_mask[0:1],
        gen_frames=6,
        generator=gen0_serial,
        t_grid=[0.0, 0.5, 1.0],
        cfg_strength=2.0,
        dtype=torch.float32,
    )

    gen0_batch = torch.Generator().manual_seed(42)
    gen1_batch = torch.Generator().manual_seed(99)
    x_mask = torch.tensor([[True] * 6 + [False] * 3, [True] * 9])
    actual_batch = sample_latents(
        dit,
        text=text,
        c_mask=c_mask,
        ref=ref,
        ref_mask=ref_mask,
        gen_frames=9,
        x_mask=x_mask,
        noise_frames=[6, 9],
        generator=[gen0_batch, gen1_batch],
        t_grid=[0.0, 0.5, 1.0],
        cfg_strength=2.0,
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual_batch[0:1, :6], expected_serial0, atol=1e-5, rtol=1e-5)
    assert torch.equal(gen0_batch.get_state(), gen0_serial.get_state())

    with pytest.raises(ValueError, match="one noise frame count per request"):
        sample_latents(
            dit,
            text=text,
            c_mask=c_mask,
            ref=ref,
            ref_mask=ref_mask,
            gen_frames=9,
            noise_frames=[6],
            generator=[gen0_batch, gen1_batch],
        )
    with pytest.raises(ValueError, match="noise_frames must lie in"):
        sample_latents(
            dit,
            text=text,
            c_mask=c_mask,
            ref=ref,
            ref_mask=ref_mask,
            gen_frames=9,
            noise_frames=[10, 9],
            generator=[gen0_batch, gen1_batch],
        )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loop_graph_replay_matches_eager_loop_on_identical_bucket(cfg_strength: float, dtype: torch.dtype) -> None:
    """With exact bucketed shapes (no padding), loop graph replay is bit-exact against eager even in BF16."""
    dit = _make_dit("cuda").to(dtype=dtype)
    wrapper = AuKCUDAGraphWrapper(dit)
    in0 = _sample_inputs("cuda", 0.0, ref_frames=4)
    in1 = _sample_inputs("cuda", 0.25, ref_frames=4)
    batched = {
        k: torch.cat(
            [
                v.to(dtype) if v.is_floating_point() else v,
                in1[k].to(dtype) if in1[k].is_floating_point() else in1[k],
            ]
        )
        for k, v in in0.items()
    }
    batched["text"] = torch.nn.functional.pad(batched["text"], (0, 0, 0, 32 - batched["text"].shape[1]))
    batched["c_mask"] = torch.nn.functional.pad(batched["c_mask"], (0, 32 - batched["c_mask"].shape[1]), value=False)
    batched["ref"] = torch.nn.functional.pad(batched["ref"], (0, 0, 0, 50 - batched["ref"].shape[1]))
    batched["ref_mask"] = torch.nn.functional.pad(
        batched["ref_mask"], (0, 50 - batched["ref_mask"].shape[1]), value=False
    )

    common = dict(
        **batched,
        gen_frames=32,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=cfg_strength,
        dtype=torch.float32,
        sampler=wrapper,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        # Capture on first call, replay on second, then compare against _run_eager_loop on identical shapes.
        sample_latents(
            dit,
            **common,
            generator=[torch.Generator(device="cuda").manual_seed(1), torch.Generator(device="cuda").manual_seed(2)],
        )
        assert wrapper.loop_captures == 1
        graph_out = sample_latents(
            dit,
            **common,
            generator=[torch.Generator(device="cuda").manual_seed(7), torch.Generator(device="cuda").manual_seed(11)],
        )
        wrapper.enabled = False
        eager_out = sample_latents(
            dit,
            **common,
            generator=[torch.Generator(device="cuda").manual_seed(7), torch.Generator(device="cuda").manual_seed(11)],
        )
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


@pytest.mark.cpu
@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_mixed_empty_and_nonempty_ref_batch_matches_serial_cpu(cfg_strength: float) -> None:
    """Mixed zero-reference and voice-cloned rows in one batch match their serial runs."""
    dit = _make_dit("cpu")
    in_empty = _sample_inputs("cpu", 0.0, ref_frames=0)
    in_ref = _sample_inputs("cpu", 0.25, ref_frames=4)
    batched = {
        "text": torch.cat([in_empty["text"], in_ref["text"]]),
        "c_mask": torch.cat([in_empty["c_mask"], in_ref["c_mask"]]),
        "ref": torch.cat([torch.zeros(1, 4, 4), in_ref["ref"]]),
        "ref_mask": torch.cat([torch.zeros(1, 4, dtype=torch.bool), in_ref["ref_mask"]]),
    }
    common = dict(gen_frames=9, t_grid=[0.0, 0.4, 1.0], cfg_strength=cfg_strength, dtype=torch.float32)
    serial0 = sample_latents(dit, **in_empty, **common, generator=torch.Generator().manual_seed(7))
    serial1 = sample_latents(dit, **in_ref, **common, generator=torch.Generator().manual_seed(11))
    batched_out = sample_latents(
        dit,
        **batched,
        **common,
        generator=[torch.Generator().manual_seed(7), torch.Generator().manual_seed(11)],
    )
    torch.testing.assert_close(batched_out[0:1], serial0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batched_out[1:2], serial1, atol=1e-5, rtol=1e-5)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_empty_and_nonempty_ref_loop_graph_matches_serial_cuda(cfg_strength: float, dtype: torch.dtype) -> None:
    """Whole-loop CUDA graph preserves serial parity when text-only and zero-shot TTS rows share a batch."""
    dit = _make_dit("cuda").to(dtype=dtype)
    wrapper = AuKCUDAGraphWrapper(dit)
    in_empty = _sample_inputs("cuda", 0.0, ref_frames=0)
    in_ref = _sample_inputs("cuda", 0.25, ref_frames=4)
    in_empty = {k: v.to(dtype) if v.is_floating_point() else v for k, v in in_empty.items()}
    in_ref = {k: v.to(dtype) if v.is_floating_point() else v for k, v in in_ref.items()}
    batched = {
        "text": torch.cat([in_empty["text"], in_ref["text"]]),
        "c_mask": torch.cat([in_empty["c_mask"], in_ref["c_mask"]]),
        "ref": torch.cat([torch.zeros(1, 4, 4, device="cuda", dtype=dtype), in_ref["ref"]]),
        "ref_mask": torch.cat([torch.zeros(1, 4, dtype=torch.bool, device="cuda"), in_ref["ref_mask"]]),
    }
    common = dict(gen_frames=9, t_grid=[0.0, 0.4, 1.0], cfg_strength=cfg_strength, dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        serial0 = sample_latents(dit, **in_empty, **common, generator=torch.Generator(device="cuda").manual_seed(7))
        serial1 = sample_latents(dit, **in_ref, **common, generator=torch.Generator(device="cuda").manual_seed(11))
        batched_out = sample_latents(
            dit,
            **batched,
            **common,
            sampler=wrapper,
            generator=[torch.Generator(device="cuda").manual_seed(7), torch.Generator(device="cuda").manual_seed(11)],
        )
    atol = 3e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(batched_out[0:1], serial0, atol=atol, rtol=rtol)
    torch.testing.assert_close(batched_out[1:2], serial1, atol=atol, rtol=rtol)
    assert len(wrapper._loop_cache) == 1
