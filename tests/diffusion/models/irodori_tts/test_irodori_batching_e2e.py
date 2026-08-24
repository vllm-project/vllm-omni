# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end validation that Irodori-TTS really batches multiple requests.

``test_irodori_packed_batching.py`` proves the packed batch is row-independent
against a stub DiT. These tests drive the real model through the offline Python
entrypoint and check the properties only the full stack can show.

What is exact, and asserted as such:

* a one-request wave is **bit-identical** to a serial render — the batching code
  path itself introduces nothing;
* identical requests inside one wave are **bit-identical to each other** — no
  cross-request contamination through shared padding or shared context K/V;
* every render has the exact requested sample count;
* each request in a wave still resolves to *its own* audio, not a neighbour's.

What is *not* exact, and is deliberately not asserted tightly: a wave of two or
more requests changes the numerics relative to a serial render. Packed
attention runs in bfloat16 at every parameter dtype, so a different wave size
retiles the fused GEMM/attention and shifts results at the last bits; the
rectified-flow ODE then amplifies that over the denoise steps. Measured on an
RTX 5060 Ti for a 4 s render, relative RMSE against the serial render is
0.03-0.55 depending on the text, and it does not shrink at 40 steps or with
float32 parameters. The audio is a different valid sample, not a corrupted one.
``_report_similarity`` prints the numbers so a reviewer can see the drift; only
a loose ceiling is enforced, to catch outright garbage.

``test_batching_scales_with_diffusion_batch_size`` adds the throughput view:
the same eight concurrent requests at ``diffusion_batch_size`` 1/2/4/8, with
batch 1 as the serial reference. It asserts only that batching helps and prints
the curve, because the curve is structurally far below ``B``x — see that test's
own docstring for the two ceilings.

``diffusion_batch_size`` is what admits more than one request into a denoise
step (it becomes ``OmniDiffusionConfig.max_num_seqs``, which the step scheduler
reads as ``max_num_running_reqs``). It is a Python-only knob today: the
``vllm serve`` path pins it to 1, so these tests use ``AsyncOmni`` rather than
the HTTP server.

Request setup batches too: every request admitted in one scheduler step goes
through ``prepare_encode_batch``. That is exact for a request that pins
``seconds``, and moves a *predicted* duration by at most one codec frame --
``test_predicted_duration_stays_within_one_frame_across_wave_sizes`` is the
guard on that bound, and ``irodori_batch_prepare_encode=False`` turns the
behaviour off for deployments that need predicted lengths to be reproducible.

Requires a GPU and the ``irodori-tts`` extra. Point at an already-downloaded
checkpoint directory to avoid a Hub round-trip::

    VLLM_OMNI_IRODORI_TEST_MODEL=/path/to/Irodori-TTS-v4-Small \
        pytest -q tests/diffusion/models/irodori_tts/test_irodori_batching_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import time

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, pytest.mark.tts]

MODEL = os.environ.get("VLLM_OMNI_IRODORI_TEST_MODEL", "Aratako/Irodori-TTS-v4-Small")
SAMPLE_RATE = 48_000
# Kept small: these tests check batching semantics, not audio quality.
NUM_STEPS = 8
CAPTION = "落ち着いた女性の声"

# Distinct text *and* distinct seed per request, so every request has a unique
# waveform and a swap between requests is detectable.
TEXTS = (
    "こんにちは。これは音声合成のテストです。",
    "今日はとても良い天気ですね。",
    "バッチ処理の検証を行っています。",
    "音声の品質を確認してください。",
    "先週の会議の議事録を共有します。",
    "駅前の新しい店に行ってみました。",
    "明日の予定を確認させてください。",
    "資料の準備はもう終わりましたか。",
)
SEEDS = (1729, 4104, 13832, 20683, 39312, 40033, 21952, 32832)
# The correctness tests only need enough requests to detect a crossed pair.
IDENTITY_REQUESTS = 4

# Batch-size scaling sweep. 40 steps rather than NUM_STEPS on purpose: the
# denoise loop dominates a long wave, and at 8 steps the per-request remainder
# is most of the wall clock, so a short schedule saturates before the curve says
# anything about the batched part. ``prepare_encode`` batches across an
# admission group; ``post_decode`` deliberately does not — a batched DACVAE
# decode measures no faster than a serial one, because it is pure compute.
SWEEP_STEPS = 40
SWEEP_REQUESTS = len(TEXTS)
SWEEP_BATCH_SIZES = (1, 2, 4, 8)
SWEEP_WARMUP_WAVES = 2
SWEEP_TIMED_WAVES = 3
MIN_SWEEP_SPEEDUP = 1.25

# Loose ceiling: a wave render must still be recognisably the same request, but
# see the module docstring for why this cannot be tight.
MAX_WAVE_RELATIVE_RMSE = 1.0
MIN_WAVE_CORRELATION = 0.5

# DACVAE hop at 48 kHz: one codec frame, 40 ms. Batched request setup encodes a
# whole admission group in one pass, which moves the encoder output by a couple
# of bf16 ulp; a *predicted* duration can therefore cross a rounding boundary.
# Measured worst drift on an RTX 5060 Ti is 0.37 of a frame, so one frame of
# slack is the honest bound, and pinning ``seconds`` avoids the effect entirely.
CODEC_HOP_SAMPLES = 1920


def _sampling(seed: int, seconds: float, steps: int) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        seed=seed,
        num_inference_steps=steps,
        extra_args={"seconds": seconds},
    )


def _sampling_predicted(seed: int, steps: int) -> OmniDiffusionSamplingParams:
    """Let the duration predictor choose the length instead of pinning it."""
    return OmniDiffusionSamplingParams(seed=seed, num_inference_steps=steps, extra_args={})


def _extract_audio(output) -> np.ndarray:
    payload = getattr(output, "multimodal_output", None)
    if payload is None:
        request_output = getattr(output, "request_output", None)
        payload = getattr(request_output, "multimodal_output", None)
    if not isinstance(payload, dict) or "audio" not in payload:
        raise AssertionError("Irodori produced no audio output.")
    audio = np.asarray(payload["audio"], dtype=np.float32).squeeze()
    sample_rate = int(payload.get("audio_sample_rate", payload.get("sr", 0)))
    assert sample_rate == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {sample_rate}"
    assert audio.ndim == 1 and audio.size, f"unexpected audio shape {audio.shape}"
    assert np.isfinite(audio).all(), "audio contains non-finite samples"
    return audio


def _similarity(generated: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    assert generated.shape == reference.shape, f"shape drift {generated.shape} vs {reference.shape}"
    relative_rmse = float(np.sqrt(np.mean((generated - reference) ** 2)) / max(np.sqrt(np.mean(reference**2)), 1e-8))
    generated_centered = generated - generated.mean()
    reference_centered = reference - reference.mean()
    correlation = float(
        np.dot(generated_centered, reference_centered)
        / max(np.linalg.norm(generated_centered) * np.linalg.norm(reference_centered), 1e-8)
    )
    return relative_rmse, correlation


def _assert_sample_counts(audios, plan) -> None:
    for audio, (index, seconds) in zip(audios, plan, strict=True):
        expected = int(round(seconds * SAMPLE_RATE))
        assert audio.shape[0] == expected, f"request {index}: expected {expected} samples, got {audio.shape[0]}"


def _report_similarity(label: str, wave, serial, plan) -> None:
    """Print wave-vs-serial drift and enforce only a loose ceiling."""
    for position, (index, seconds) in enumerate(plan):
        relative_rmse, correlation = _similarity(wave[position], serial[position])
        print(f"   [{label}] request {index} ({seconds}s): rel_rmse={relative_rmse:.6f} corr={correlation:.8f}")
        assert relative_rmse <= MAX_WAVE_RELATIVE_RMSE and correlation >= MIN_WAVE_CORRELATION, (
            f"request {index} diverged beyond the wave-numerics ceiling: "
            f"rel_rmse={relative_rmse:.6f} corr={correlation:.8f}"
        )


def _assert_identity_preserved(wave, serial, plan) -> None:
    """Each wave render must resemble its own serial render more than any other."""
    for position, (index, _) in enumerate(plan):
        scores = {}
        for other, (other_index, _) in enumerate(plan):
            if serial[other].shape != wave[position].shape:
                continue
            scores[other_index] = _similarity(wave[position], serial[other])[1]
        best = max(scores, key=scores.get)
        assert best == index, (
            f"wave request {index} best-matches serial request {best} "
            f"(corr {scores[best]:.6f} vs own {scores[index]:.6f}); requests are being crossed"
        )


async def _render(omni: AsyncOmni, index: int, seconds: float, tag: str, steps: int = NUM_STEPS) -> np.ndarray:
    last = None
    async for output in omni.generate(
        prompt={"input": TEXTS[index], "caption": CAPTION},
        request_id=f"irodori-{tag}-{index}-{time.monotonic_ns()}",
        sampling_params_list=[_sampling(SEEDS[index], seconds, steps)],
    ):
        last = output
    assert last is not None, f"no output for request {index}"
    return _extract_audio(last)


async def _render_predicted(omni: AsyncOmni, index: int, tag: str) -> np.ndarray:
    """Render without pinning ``seconds`` so the duration predictor runs."""
    last = None
    async for output in omni.generate(
        prompt={"input": TEXTS[index], "caption": CAPTION},
        request_id=f"irodori-{tag}-{index}-{time.monotonic_ns()}",
        sampling_params_list=[_sampling_predicted(SEEDS[index], NUM_STEPS)],
    ):
        last = output
    assert last is not None, f"no output for request {index}"
    return _extract_audio(last)


async def _render_serial(omni: AsyncOmni, plan) -> tuple[list[np.ndarray], float]:
    """One request in flight at a time — the batch-size-1 reference."""
    started = time.perf_counter()
    audios = [await _render(omni, index, seconds, "serial") for index, seconds in plan]
    return audios, time.perf_counter() - started


async def _render_wave(omni: AsyncOmni, plan, steps: int = NUM_STEPS) -> tuple[list[np.ndarray], float]:
    """All requests in flight together — the step scheduler co-schedules them."""
    started = time.perf_counter()
    audios = await asyncio.gather(*(_render(omni, index, seconds, "wave", steps) for index, seconds in plan))
    return list(audios), time.perf_counter() - started


def _engine(batch_size: int) -> AsyncOmni:
    return AsyncOmni(
        model=MODEL,
        model_class_name="IrodoriTTSPipeline",
        max_num_seqs=batch_size,
        diffusion_batch_size=batch_size,
        step_execution=True,
    )


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_single_request_wave_is_bit_identical_to_serial():
    """With one request in flight the batching path must change nothing."""

    async def _inner():
        omni = _engine(4)
        try:
            serial = await _render(omni, 0, 4.0, "serial")
            wave, _ = await _render_wave(omni, [(0, 4.0)])
        finally:
            omni.shutdown()
        np.testing.assert_array_equal(wave[0], serial)

    asyncio.run(_inner())


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_identical_requests_in_one_wave_are_bit_identical():
    """No cross-request contamination: same input in a wave gives same output."""
    plan = [(0, 4.0)] * 4

    async def _inner():
        omni = _engine(len(plan))
        try:
            wave, _ = await _render_wave(omni, plan)
        finally:
            omni.shutdown()
        _assert_sample_counts(wave, plan)
        for position in range(1, len(wave)):
            np.testing.assert_array_equal(wave[position], wave[0])

    asyncio.run(_inner())


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_homogeneous_wave_preserves_request_identity():
    """Four same-length, different-content requests must not be crossed."""
    plan = [(index, 4.0) for index in range(IDENTITY_REQUESTS)]

    async def _inner():
        omni = _engine(len(plan))
        try:
            serial, serial_s = await _render_serial(omni, plan)
            wave, wave_s = await _render_wave(omni, plan)
        finally:
            omni.shutdown()
        print(f"\n[homogeneous] serial={serial_s:.2f}s wave={wave_s:.2f}s (unwarmed; see the scaling sweep)")
        _assert_sample_counts(wave, plan)
        _assert_identity_preserved(wave, serial, plan)
        _report_similarity("homogeneous", wave, serial, plan)

    asyncio.run(_inner())


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_mixed_length_wave_preserves_request_identity():
    """Mixed durations exercise ragged/padded grouping rather than one bucket."""
    plan = [(0, 4.0), (1, 10.0), (2, 4.0), (3, 10.0)]

    async def _inner():
        omni = _engine(len(plan))
        try:
            serial, serial_s = await _render_serial(omni, plan)
            wave, wave_s = await _render_wave(omni, plan)
        finally:
            omni.shutdown()
        print(f"\n[mixed-length] serial={serial_s:.2f}s wave={wave_s:.2f}s (unwarmed; see the scaling sweep)")
        _assert_sample_counts(wave, plan)
        _assert_identity_preserved(wave, serial, plan)
        _report_similarity("mixed-length", wave, serial, plan)

    asyncio.run(_inner())


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_predicted_duration_stays_within_one_frame_across_wave_sizes():
    """A predicted length must not depend on what else is in flight.

    Requests that pin ``seconds`` get an exact sample count, asserted
    elsewhere. Requests that let the model predict their length go through
    ``prepare_encode_batch``, whose batched encoder output differs from the
    serial one at the last bits — enough to move ``round()`` by one codec
    frame, and no further. This is the guard on "no further"; a regression that
    genuinely crossed requests would blow past a single frame.
    """
    indices = list(range(IDENTITY_REQUESTS))

    async def _inner():
        omni = _engine(len(indices))
        try:
            serial = [await _render_predicted(omni, index, "serial") for index in indices]
            wave = await asyncio.gather(*(_render_predicted(omni, index, "wave") for index in indices))
        finally:
            omni.shutdown()
        return serial, list(wave)

    serial, wave = asyncio.run(_inner())
    print()
    for index, (serial_audio, wave_audio) in enumerate(zip(serial, wave, strict=True)):
        drift = abs(int(serial_audio.shape[0]) - int(wave_audio.shape[0]))
        print(
            f"   [predicted] request {index}: serial={serial_audio.shape[0]} "
            f"wave={wave_audio.shape[0]} drift={drift} samples "
            f"({drift / CODEC_HOP_SAMPLES:.2f} codec frames)"
        )
        assert drift <= CODEC_HOP_SAMPLES, (
            f"request {index} predicted duration moved {drift} samples "
            f"({drift / CODEC_HOP_SAMPLES:.2f} codec frames) between a serial render and a wave; "
            f"at most one frame is expected from batched request setup"
        )


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_batching_scales_with_diffusion_batch_size():
    """Sweep ``diffusion_batch_size`` over 1/2/4/8 against a fixed workload.

    The same eight requests are submitted concurrently every time; only the
    number the step scheduler may admit per denoise step changes. ``batch=1`` is
    therefore the serial reference, and ``T(1)/T(B)`` is the honest speedup.

    This is the regression guard for the failure mode where every layer of the
    batching stack is present and correct but the scheduler still admits one
    request per denoise step, making concurrency a no-op.

    Only ``batch > 1`` beating ``batch = 1`` is asserted, plus a floor on the
    best point. The curve is expected to be well under ``B``x and to flatten,
    and that is arithmetic rather than a defect: at batch 8 the fused DiT step
    already runs near the device's compute limit, so a wider batch buys back
    launch and Python overhead, not work the GPU was not already doing.
    ``post_decode`` is per-request compute the batch width cannot touch. See
    the printed table.

    Every batch size is warmed on the wave it is later timed on — regional
    ``torch.compile`` compiles each batch shape on first use, and charging that
    to the wave inverts the result.
    """
    plan = [(index, 4.0) for index in range(SWEEP_REQUESTS)]

    async def _measure(batch_size: int) -> float:
        omni = _engine(batch_size)
        try:
            for _ in range(SWEEP_WARMUP_WAVES):
                await _render_wave(omni, plan, steps=SWEEP_STEPS)
            timings = []
            for _ in range(SWEEP_TIMED_WAVES):
                audios, elapsed = await _render_wave(omni, plan, steps=SWEEP_STEPS)
                assert len(audios) == SWEEP_REQUESTS
                _assert_sample_counts(audios, plan)
                timings.append(elapsed)
        finally:
            omni.shutdown()
        return min(timings)

    async def _inner() -> dict[int, float]:
        return {batch_size: await _measure(batch_size) for batch_size in SWEEP_BATCH_SIZES}

    wall = asyncio.run(_inner())
    reference = wall[1]

    print(f"\n[sweep] {SWEEP_REQUESTS} requests x 4.0s audio, {SWEEP_STEPS} denoise steps")
    print(f"[sweep] {'batch':>6}{'wall(s)':>10}{'req/s':>9}{'speedup':>10}{'efficiency':>12}")
    for batch_size in SWEEP_BATCH_SIZES:
        seconds = wall[batch_size]
        speedup = reference / seconds
        print(
            f"[sweep] {batch_size:>6}{seconds:>10.3f}{SWEEP_REQUESTS / seconds:>9.2f}"
            f"{speedup:>9.2f}x{speedup / batch_size:>11.0%}"
        )

    for batch_size in SWEEP_BATCH_SIZES[1:]:
        assert wall[batch_size] < reference, (
            f"diffusion_batch_size={batch_size} ({wall[batch_size]:.3f}s) was not faster than "
            f"diffusion_batch_size=1 ({reference:.3f}s) for {SWEEP_REQUESTS} concurrent requests; "
            f"requests are most likely executing one per denoise step"
        )
    best = reference / min(wall.values())
    assert best >= MIN_SWEEP_SPEEDUP, (
        f"best batched speedup was only {best:.2f}x over serial, below the {MIN_SWEEP_SPEEDUP:.2f}x floor"
    )
