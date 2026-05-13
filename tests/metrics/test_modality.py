from __future__ import annotations

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.modality import (
    OmniModalityMetrics,
    observe_audio_first_packet,
    observe_modality_at_finalize,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_MODEL = "test-modality-model"


@pytest.fixture(scope="module")
def mod() -> OmniModalityMetrics:
    return OmniModalityMetrics(model_name=_MODEL)


# Each test uses a distinct (stage, replica) so counter accumulation
# across tests doesn't couple assertions.
_AUDIO_STAGE = ("talker", "0")
_IMAGE_STAGE = ("diffusion", "0")
_VIDEO_STAGE = ("diffusion", "1")


def _sample_value(output: str, line_prefix: str) -> float | None:
    for line in output.splitlines():
        if line.startswith(line_prefix):
            return float(line.split()[-1])
    return None


# ---------------------------------------------------------------------------
# Family registration
# ---------------------------------------------------------------------------


_EXPECTED_FAMILIES = [
    defs.AUDIO_TTFP_SECONDS,
    defs.AUDIO_DURATION_SECONDS,
    defs.AUDIO_RTF_METRIC,
    defs.AUDIO_FRAMES_METRIC,
    defs.IMAGE_TTFP_SECONDS,
    defs.IMAGE_NUM_METRIC,
    defs.IMAGE_GENERATION_TIME_SECONDS,
    defs.VIDEO_GENERATION_TIME_SECONDS,
]


class TestRegistration:
    def test_all_eight_families_present(self, mod: OmniModalityMetrics) -> None:
        # Trigger at least one observation per family so the registry exposes them.
        mod.observe_audio_ttfp("s", "r", 0.1)
        mod.observe_audio_duration("s", "r", 1.0)
        mod.observe_audio_rtf("s", "r", 0.5)
        mod.inc_audio_frames("s", "r", 1)
        mod.observe_image_ttfp("s", "r", 0.2)
        mod.inc_image_num("s", "r", 1)
        mod.observe_image_generation_time("s", "r", 0.5)
        mod.observe_video_generation_time("s", "r", 1.0)

        out = generate_latest(REGISTRY).decode()
        for name in _EXPECTED_FAMILIES:
            assert f"# HELP {name}" in out, f"missing family: {name}"

    def test_video_duration_and_rtf_intentionally_absent(self) -> None:
        # Phase 3 deliberately drops these — see modality.py docstring.
        out = generate_latest(REGISTRY).decode()
        assert defs.VIDEO_DURATION_SECONDS not in out
        assert defs.VIDEO_RTF_METRIC not in out


# ---------------------------------------------------------------------------
# Audio observe API
# ---------------------------------------------------------------------------


class TestAudio:
    def test_audio_ttfp_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_ttfp", "0"
        mod.observe_audio_ttfp(stage, replica, 0.42)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.AUDIO_TTFP_SECONDS}_count{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 1.0

    def test_audio_duration_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_dur", "0"
        mod.observe_audio_duration(stage, replica, 3.5)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.AUDIO_DURATION_SECONDS}_sum{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 3.5

    def test_audio_rtf_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_rtf", "0"
        mod.observe_audio_rtf(stage, replica, 0.45)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.AUDIO_RTF_METRIC}_sum{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 0.45

    def test_audio_frames_inc(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_frames", "0"
        mod.inc_audio_frames(stage, replica, 240)
        mod.inc_audio_frames(stage, replica, 60)
        out = generate_latest(REGISTRY).decode()
        # Counter family auto-suffixes with _total in the exposed name.
        prefix = f'{defs.AUDIO_FRAMES_METRIC}_total{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 300.0

    def test_audio_frames_zero_or_negative_skipped(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_zero", "0"
        mod.inc_audio_frames(stage, replica, 0)
        mod.inc_audio_frames(stage, replica, -5)
        # No observation → no series exposed for this (stage, replica) yet.
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.AUDIO_FRAMES_METRIC}_total{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) is None


# ---------------------------------------------------------------------------
# Image observe API
# ---------------------------------------------------------------------------


class TestImage:
    def test_image_ttfp_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "diffusion_ttfp", "0"
        mod.observe_image_ttfp(stage, replica, 1.5)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.IMAGE_TTFP_SECONDS}_count{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 1.0

    def test_image_num_inc(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "diffusion_num", "0"
        mod.inc_image_num(stage, replica, 4)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.IMAGE_NUM_METRIC}_total{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 4.0

    def test_image_generation_time_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "diffusion_gen", "0"
        mod.observe_image_generation_time(stage, replica, 2.7)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.IMAGE_GENERATION_TIME_SECONDS}_sum{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 2.7


# ---------------------------------------------------------------------------
# Video observe API
# ---------------------------------------------------------------------------


class TestVideo:
    def test_video_generation_time_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "diffusion_video", "0"
        mod.observe_video_generation_time(stage, replica, 5.2)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.VIDEO_GENERATION_TIME_SECONDS}_sum{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 5.2


# ---------------------------------------------------------------------------
# Bucket selection (RTF uses RTF_BUCKETS, others use SECONDS_BUCKETS)
# ---------------------------------------------------------------------------


class _StubModMetrics:
    """Records every observe/inc call so the routing logic can be asserted."""

    def __init__(self):
        self.calls: list[tuple] = []

    def inc_audio_frames(self, s, r, n):
        self.calls.append(("inc_audio_frames", s, r, n))

    def observe_audio_duration(self, s, r, d):
        self.calls.append(("observe_audio_duration", s, r, d))

    def observe_audio_rtf(self, s, r, rtf):
        self.calls.append(("observe_audio_rtf", s, r, rtf))

    def inc_image_num(self, s, r, n):
        self.calls.append(("inc_image_num", s, r, n))

    def observe_image_generation_time(self, s, r, t):
        self.calls.append(("observe_image_generation_time", s, r, t))

    def observe_image_ttfp(self, s, r, t):
        self.calls.append(("observe_image_ttfp", s, r, t))

    def observe_video_generation_time(self, s, r, t):
        self.calls.append(("observe_video_generation_time", s, r, t))


class _Bag:
    """Tiny attribute bag for stage_metrics / engine_outputs stubs."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestObserveModalityAtFinalize:
    def test_audio_path_full(self):
        stub = _StubModMetrics()
        stage_metrics = _Bag(stage_gen_time_ms=500.0, audio_generated_frames=24000)
        engine_outputs = _Bag(multimodal_output={"audio_sample_rate": 24000})

        observe_modality_at_finalize(
            stub,
            output_type="audio",
            stage_id=1,
            replica_id=0,
            stage_metrics=stage_metrics,
            engine_outputs=engine_outputs,
            request_arrival_ts=100.0,
            finalize_ts=100.5,
        )
        # 24000 frames / 24000 Hz = 1.0s duration; gen 0.5s → rtf 0.5
        assert ("inc_audio_frames", "1", "0", 24000) in stub.calls
        assert ("observe_audio_duration", "1", "0", 1.0) in stub.calls
        assert ("observe_audio_rtf", "1", "0", 0.5) in stub.calls

    def test_audio_path_zero_frames_skips_duration_and_rtf(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="audio",
            stage_id=1,
            replica_id=0,
            stage_metrics=_Bag(stage_gen_time_ms=300.0, audio_generated_frames=0),
            engine_outputs=_Bag(multimodal_output={}),
            request_arrival_ts=100.0,
            finalize_ts=100.3,
        )
        # inc with 0 still called (Counter side gates internally to no-op)
        assert ("inc_audio_frames", "1", "0", 0) in stub.calls
        # but no duration / rtf because duration_s == 0
        assert not any(c[0] == "observe_audio_duration" for c in stub.calls)
        assert not any(c[0] == "observe_audio_rtf" for c in stub.calls)

    def test_audio_uses_resolved_sample_rate_from_multimodal_output(self):
        stub = _StubModMetrics()
        # Non-default 16 kHz from talker config
        observe_modality_at_finalize(
            stub,
            output_type="audio",
            stage_id=1,
            replica_id=0,
            stage_metrics=_Bag(stage_gen_time_ms=1000.0, audio_generated_frames=16000),
            engine_outputs=_Bag(multimodal_output={"sample_rate": 16000}),
            request_arrival_ts=0.0,
            finalize_ts=1.0,
        )
        # 16000 / 16000 = 1.0s
        assert ("observe_audio_duration", "1", "0", 1.0) in stub.calls

    def test_image_path_uses_finalize_minus_arrival_for_ttfp(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="image",
            stage_id=2,
            replica_id=1,
            stage_metrics=_Bag(stage_gen_time_ms=2000.0),
            engine_outputs=_Bag(images=["img1", "img2", "img3"]),
            request_arrival_ts=10.0,
            finalize_ts=12.5,
        )
        assert ("inc_image_num", "2", "1", 3) in stub.calls
        assert ("observe_image_generation_time", "2", "1", 2.0) in stub.calls
        assert ("observe_image_ttfp", "2", "1", 2.5) in stub.calls

    def test_image_ttfp_clamped_to_zero_on_clock_skew(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="image",
            stage_id=2,
            replica_id=0,
            stage_metrics=_Bag(stage_gen_time_ms=1000.0),
            engine_outputs=_Bag(images=["img"]),
            request_arrival_ts=100.0,
            finalize_ts=99.5,  # finalize earlier than arrival (impossible but defensive)
        )
        assert ("observe_image_ttfp", "2", "0", 0.0) in stub.calls

    def test_video_path_only_emits_generation_time(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="video",
            stage_id=2,
            replica_id=0,
            stage_metrics=_Bag(stage_gen_time_ms=5200.0),
            engine_outputs=_Bag(),
            request_arrival_ts=0.0,
            finalize_ts=5.3,
        )
        assert stub.calls == [("observe_video_generation_time", "2", "0", 5.2)]

    def test_text_path_no_calls(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="text",
            stage_id=0,
            replica_id=0,
            stage_metrics=_Bag(stage_gen_time_ms=100.0),
            engine_outputs=_Bag(),
            request_arrival_ts=0.0,
            finalize_ts=0.1,
        )
        assert stub.calls == []

    def test_replica_id_none_skipped(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="audio",
            stage_id=1,
            replica_id=None,  # error path: orchestrator emitted without replica_id
            stage_metrics=_Bag(stage_gen_time_ms=500.0, audio_generated_frames=240),
            engine_outputs=_Bag(multimodal_output={}),
            request_arrival_ts=0.0,
            finalize_ts=0.5,
        )
        assert stub.calls == []

    def test_stage_metrics_none_skipped(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="audio",
            stage_id=1,
            replica_id=0,
            stage_metrics=None,
            engine_outputs=_Bag(multimodal_output={}),
            request_arrival_ts=0.0,
            finalize_ts=0.5,
        )
        assert stub.calls == []


class TestObserveAudioFirstPacket:
    def test_observes_with_valid_inputs(self):
        stub = _StubModMetrics()
        # Patch in audio_ttfp to the stub for routing assertion.
        stub.observe_audio_ttfp = lambda s, r, t: stub.calls.append(("observe_audio_ttfp", s, r, t))

        observe_audio_first_packet(
            stub,
            stage_id=1,
            replica_id=0,
            arrival_ts=100.0,
            now_ts=100.42,
        )
        assert stub.calls == [("observe_audio_ttfp", "1", "0", pytest.approx(0.42))]

    def test_replica_none_skipped(self):
        stub = _StubModMetrics()
        stub.observe_audio_ttfp = lambda s, r, t: stub.calls.append(("observe_audio_ttfp", s, r, t))
        observe_audio_first_packet(
            stub, stage_id=1, replica_id=None, arrival_ts=100.0, now_ts=100.5
        )
        assert stub.calls == []

    def test_arrival_ts_zero_skipped(self):
        # Defensive: req_state.request_arrival_ts == 0.0 means async_omni
        # never populated it (e.g. some fast path). Skip rather than emit
        # garbage TTFP measured against epoch.
        stub = _StubModMetrics()
        stub.observe_audio_ttfp = lambda s, r, t: stub.calls.append(("observe_audio_ttfp", s, r, t))
        observe_audio_first_packet(
            stub, stage_id=1, replica_id=0, arrival_ts=0.0, now_ts=100.5
        )
        assert stub.calls == []

    def test_clock_skew_clamped_to_zero(self):
        stub = _StubModMetrics()
        stub.observe_audio_ttfp = lambda s, r, t: stub.calls.append(("observe_audio_ttfp", s, r, t))
        observe_audio_first_packet(
            stub, stage_id=1, replica_id=0, arrival_ts=100.5, now_ts=100.0
        )
        assert stub.calls == [("observe_audio_ttfp", "1", "0", 0.0)]


class TestBucketSelection:
    def test_audio_rtf_uses_rtf_buckets(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_buckets", "0"
        mod.observe_audio_rtf(stage, replica, 0.5)
        out = generate_latest(REGISTRY).decode()
        # RTF_BUCKETS includes 0.9 and 1.25 — distinctive boundaries vs SECONDS_BUCKETS.
        # Check that at least one RTF-distinctive bucket label appears.
        rtf_marker = f'{defs.AUDIO_RTF_METRIC}_bucket{{le="0.9"'
        assert rtf_marker in out, "audio_rtf should use RTF_BUCKETS containing le=0.9"

    def test_audio_ttfp_uses_seconds_buckets(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_seconds", "0"
        mod.observe_audio_ttfp(stage, replica, 0.1)
        out = generate_latest(REGISTRY).decode()
        # SECONDS_BUCKETS includes 0.05 — not in RTF_BUCKETS.
        sec_marker = f'{defs.AUDIO_TTFP_SECONDS}_bucket{{le="0.05"'
        assert sec_marker in out, "audio_ttfp should use SECONDS_BUCKETS containing le=0.05"
