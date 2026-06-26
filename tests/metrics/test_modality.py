from __future__ import annotations

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.modality import (
    OmniModalityMetrics,
    observe_audio_first_packet,
    observe_audio_streaming_finalize,
    observe_modality_at_finalize,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_MODEL = "test-modality-model"


@pytest.fixture(scope="module")
def mod() -> OmniModalityMetrics:
    return OmniModalityMetrics(model_name=_MODEL)


def _sample_value(output: str, line_prefix: str) -> float | None:
    for line in output.splitlines():
        if line.startswith(line_prefix):
            return float(line.split()[-1])
    return None


# ---------------------------------------------------------------------------
# Family registration — 7 audio families served from modality.py
# ---------------------------------------------------------------------------


_EXPECTED_FAMILIES = [
    defs.AUDIO_TTFP_S,
    defs.AUDIO_DURATION_S,
    defs.AUDIO_RTF_METRIC,
    defs.AUDIO_FRAMES_METRIC,
    defs.AUDIO_UNDERRUN_S,
    defs.AUDIO_CONTINUITY_OK_METRIC,
    defs.AUDIO_SKIPPED_REQUESTS_METRIC,
    defs.DIFFUSION_EXEC_S,
    defs.DIFFUSION_PREPROCESS_S,
    defs.DIFFUSION_POSTPROCESS_S,
]


class TestRegistration:
    def test_all_locked_families_present(self, mod: OmniModalityMetrics) -> None:
        # Trigger at least one observation per family so the registry exposes them.
        mod.observe_audio_ttfp("s", "r", 0.1)
        mod.observe_audio_duration("s", "r", 1.0)
        mod.observe_audio_rtf("s", "r", 0.5)
        mod.inc_audio_frames("s", "r", 1)
        mod.observe_audio_underrun("s", "r", 0.01)
        mod.inc_audio_continuity_ok("s", "r", 100)
        mod.inc_audio_skipped("s", "r", "malformed_codec")
        mod.observe_diffusion_exec("s", "r", 0.5)
        mod.observe_diffusion_preprocess("s", "r", 0.01)
        mod.observe_diffusion_postprocess("s", "r", 0.2)

        out = generate_latest(REGISTRY).decode()
        for name in _EXPECTED_FAMILIES:
            assert f"# HELP {name}" in out, f"missing family: {name}"


# ---------------------------------------------------------------------------
# Audio observe API
# ---------------------------------------------------------------------------


class TestAudio:
    def test_audio_ttfp_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_ttfp", "0"
        mod.observe_audio_ttfp(stage, replica, 0.42)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.AUDIO_TTFP_S}_count{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 1.0

    def test_audio_duration_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_dur", "0"
        mod.observe_audio_duration(stage, replica, 3.5)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.AUDIO_DURATION_S}_sum{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
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
        prefix = f'{defs.AUDIO_FRAMES_METRIC}_total{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 300.0

    def test_audio_frames_zero_or_negative_skipped(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_zero", "0"
        mod.inc_audio_frames(stage, replica, 0)
        mod.inc_audio_frames(stage, replica, -5)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.AUDIO_FRAMES_METRIC}_total{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) is None

    def test_audio_underrun_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_under", "0"
        mod.observe_audio_underrun(stage, replica, 0.05)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.AUDIO_UNDERRUN_S}_sum{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 0.05

    def test_audio_underrun_negative_clamped_to_zero(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_under_neg", "0"
        mod.observe_audio_underrun(stage, replica, -0.1)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.AUDIO_UNDERRUN_S}_sum{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 0.0

    def test_audio_continuity_ok_carries_threshold_label(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_cont", "0"
        mod.inc_audio_continuity_ok(stage, replica, threshold_ms=100)
        mod.inc_audio_continuity_ok(stage, replica, threshold_ms=200)
        out = generate_latest(REGISTRY).decode()
        # Counter family auto-suffixes _total at exposition.
        v100 = _sample_value(
            out,
            f'{defs.AUDIO_CONTINUITY_OK_METRIC}_total{{model_name="{_MODEL}",'
            f'replica="{replica}",stage="{stage}",threshold_ms="100"}}',
        )
        v200 = _sample_value(
            out,
            f'{defs.AUDIO_CONTINUITY_OK_METRIC}_total{{model_name="{_MODEL}",'
            f'replica="{replica}",stage="{stage}",threshold_ms="200"}}',
        )
        assert v100 == 1.0
        assert v200 == 1.0

    def test_audio_skipped_carries_reason_label(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_skip", "0"
        mod.inc_audio_skipped(stage, replica, "malformed_codec")
        mod.inc_audio_skipped(stage, replica, "")  # → "unknown"
        out = generate_latest(REGISTRY).decode()
        good = _sample_value(
            out,
            f'{defs.AUDIO_SKIPPED_REQUESTS_METRIC}_total{{model_name="{_MODEL}",'
            f'reason="malformed_codec",replica="{replica}",stage="{stage}"}}',
        )
        unknown = _sample_value(
            out,
            f'{defs.AUDIO_SKIPPED_REQUESTS_METRIC}_total{{model_name="{_MODEL}",'
            f'reason="unknown",replica="{replica}",stage="{stage}"}}',
        )
        assert good == 1.0
        assert unknown == 1.0


# ---------------------------------------------------------------------------
# Diffusion observe API
# ---------------------------------------------------------------------------


class TestDiffusion:
    def test_diffusion_exec_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "dit_exec", "0"
        mod.observe_diffusion_exec(stage, replica, 1.25)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.DIFFUSION_EXEC_S}_sum{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 1.25

    def test_diffusion_preprocess_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "dit_pre", "0"
        mod.observe_diffusion_preprocess(stage, replica, 0.03)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.DIFFUSION_PREPROCESS_S}_sum{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == pytest.approx(0.03)

    def test_diffusion_postprocess_observed(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "dit_post", "0"
        mod.observe_diffusion_postprocess(stage, replica, 0.8)
        out = generate_latest(REGISTRY).decode()
        prefix = f'{defs.DIFFUSION_POSTPROCESS_S}_sum{{model_name="{_MODEL}",replica="{replica}",stage="{stage}"}}'
        assert _sample_value(out, prefix) == 0.8

    def test_diffusion_uses_seconds_fast_buckets(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "dit_buckets", "0"
        mod.observe_diffusion_exec(stage, replica, 0.005)
        out = generate_latest(REGISTRY).decode()
        fast_marker = f'{defs.DIFFUSION_EXEC_S}_bucket{{le="0.001"'
        assert fast_marker in out, "diffusion_exec_s should use SECONDS_FAST_BUCKETS containing le=0.001"

    def test_log_stats_false_skips(self) -> None:
        silent = OmniModalityMetrics(model_name="silent-model", log_stats=False)
        silent.observe_diffusion_exec("s", "r", 1.0)
        silent.observe_diffusion_preprocess("s", "r", 0.01)
        silent.observe_diffusion_postprocess("s", "r", 0.5)
        out = generate_latest(REGISTRY).decode()
        assert f'model_name="silent-model"' not in out


# ---------------------------------------------------------------------------
# Stub-driven routing tests for the finalize / streaming helpers.
# ---------------------------------------------------------------------------


class _StubModMetrics:
    def __init__(self):
        self.calls: list[tuple] = []

    def inc_audio_frames(self, s, r, n):
        self.calls.append(("inc_audio_frames", s, r, n))

    def observe_audio_duration(self, s, r, d):
        self.calls.append(("observe_audio_duration", s, r, d))

    def observe_audio_rtf(self, s, r, rtf):
        self.calls.append(("observe_audio_rtf", s, r, rtf))

    def observe_audio_underrun(self, s, r, u):
        self.calls.append(("observe_audio_underrun", s, r, u))

    def inc_audio_continuity_ok(self, s, r, threshold_ms):
        self.calls.append(("inc_audio_continuity_ok", s, r, threshold_ms))

    def inc_audio_skipped(self, s, r, reason):
        self.calls.append(("inc_audio_skipped", s, r, reason))

    def observe_diffusion_exec(self, s, r, v):
        self.calls.append(("observe_diffusion_exec", s, r, v))

    def observe_diffusion_preprocess(self, s, r, v):
        self.calls.append(("observe_diffusion_preprocess", s, r, v))

    def observe_diffusion_postprocess(self, s, r, v):
        self.calls.append(("observe_diffusion_postprocess", s, r, v))


class _Bag:
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
        )
        # 24000 frames / 24000 Hz = 1.0s duration; gen 0.5s → rtf 0.5
        assert ("inc_audio_frames", "1", "0", 24000) in stub.calls
        assert ("observe_audio_duration", "1", "0", 1.0) in stub.calls
        assert ("observe_audio_rtf", "1", "0", 0.5) in stub.calls
        # Continuity/underrun NOT emitted from finalize — they come from the
        # streaming hook because they need the per-chunk arrival timeline.
        assert not any(c[0] == "observe_audio_underrun" for c in stub.calls)
        assert not any(c[0] == "inc_audio_continuity_ok" for c in stub.calls)

    def test_audio_path_zero_frames_emits_skipped(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="audio",
            stage_id=1,
            replica_id=0,
            stage_metrics=_Bag(stage_gen_time_ms=300.0, audio_generated_frames=0),
            engine_outputs=_Bag(multimodal_output={}),
        )
        assert ("inc_audio_frames", "1", "0", 0) in stub.calls
        assert not any(c[0] == "observe_audio_duration" for c in stub.calls)
        assert not any(c[0] == "observe_audio_rtf" for c in stub.calls)
        assert ("inc_audio_skipped", "1", "0", "no_audio_data") in stub.calls

    def test_audio_uses_resolved_sample_rate_from_multimodal_output(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="audio",
            stage_id=1,
            replica_id=0,
            stage_metrics=_Bag(stage_gen_time_ms=1000.0, audio_generated_frames=16000),
            engine_outputs=_Bag(multimodal_output={"sample_rate": 16000}),
        )
        assert ("observe_audio_duration", "1", "0", 1.0) in stub.calls

    def test_diffusion_path_full(self):
        stub = _StubModMetrics()
        stage_metrics = _Bag(
            stage_gen_time_ms=1000.0,
            diffusion_metrics={
                "preprocess_time_s": 0.05,
                "diffusion_engine_exec_time_s": 1.2,
                "postprocess_time_s": 0.3,
                "diffusion_engine_total_time_s": 1.55,
                "image_num": 1,
                "resolution": 1024,
            },
        )
        observe_modality_at_finalize(
            stub,
            output_type="image",
            stage_id=2,
            replica_id=0,
            stage_metrics=stage_metrics,
            engine_outputs=_Bag(),
        )
        assert ("observe_diffusion_preprocess", "2", "0", 0.05) in stub.calls
        assert ("observe_diffusion_exec", "2", "0", 1.2) in stub.calls
        assert ("observe_diffusion_postprocess", "2", "0", 0.3) in stub.calls

    def test_diffusion_metrics_none_skipped(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="image",
            stage_id=2,
            replica_id=0,
            stage_metrics=_Bag(stage_gen_time_ms=500.0, diffusion_metrics=None),
            engine_outputs=_Bag(),
        )
        assert not any(c[0].startswith("observe_diffusion") for c in stub.calls)

    def test_diffusion_zero_values_observed(self):
        stub = _StubModMetrics()
        stage_metrics = _Bag(
            stage_gen_time_ms=500.0,
            diffusion_metrics={
                "preprocess_time_s": 0.0,
                "diffusion_engine_exec_time_s": 0.0,
                "postprocess_time_s": 0.0,
            },
        )
        observe_modality_at_finalize(
            stub,
            output_type="image",
            stage_id=2,
            replica_id=0,
            stage_metrics=stage_metrics,
            engine_outputs=_Bag(),
        )
        assert ("observe_diffusion_preprocess", "2", "0", 0.0) in stub.calls
        assert ("observe_diffusion_exec", "2", "0", 0.0) in stub.calls
        assert ("observe_diffusion_postprocess", "2", "0", 0.0) in stub.calls

    def test_non_audio_output_type_skips_audio_but_still_checks_diffusion(self):
        stub = _StubModMetrics()
        for output_type in ("text", "image", "video"):
            observe_modality_at_finalize(
                stub,
                output_type=output_type,
                stage_id=0,
                replica_id=0,
                stage_metrics=_Bag(stage_gen_time_ms=100.0),
                engine_outputs=_Bag(),
            )
        assert not any(c[0].startswith("observe_audio") or c[0].startswith("inc_audio") for c in stub.calls)

    def test_replica_id_none_skipped(self):
        stub = _StubModMetrics()
        observe_modality_at_finalize(
            stub,
            output_type="audio",
            stage_id=1,
            replica_id=None,
            stage_metrics=_Bag(stage_gen_time_ms=500.0, audio_generated_frames=240),
            engine_outputs=_Bag(multimodal_output={}),
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
        )
        assert stub.calls == []


class TestObserveAudioFirstPacket:
    def test_observes_with_valid_inputs(self):
        stub = _StubModMetrics()
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
        observe_audio_first_packet(stub, stage_id=1, replica_id=None, arrival_ts=100.0, now_ts=100.5)
        assert stub.calls == []

    def test_arrival_ts_zero_skipped(self):
        stub = _StubModMetrics()
        stub.observe_audio_ttfp = lambda s, r, t: stub.calls.append(("observe_audio_ttfp", s, r, t))
        observe_audio_first_packet(stub, stage_id=1, replica_id=0, arrival_ts=0.0, now_ts=100.5)
        assert stub.calls == []

    def test_clock_skew_clamped_to_zero(self):
        stub = _StubModMetrics()
        stub.observe_audio_ttfp = lambda s, r, t: stub.calls.append(("observe_audio_ttfp", s, r, t))
        observe_audio_first_packet(stub, stage_id=1, replica_id=0, arrival_ts=100.5, now_ts=100.0)
        assert stub.calls == [("observe_audio_ttfp", "1", "0", 0.0)]


class TestObserveAudioStreamingFinalize:
    def test_continuous_run_emits_underrun_and_continuity_ok(self):
        stub = _StubModMetrics()
        # Three back-to-back chunks of 0.5s each at 24 kHz s16le mono:
        # 24000 Hz * 2 bytes * 0.5s = 24000 bytes per chunk.
        arrivals = [0.5, 1.0, 1.5]
        bytes_ = [24000, 24000, 24000]
        observe_audio_streaming_finalize(
            stub,
            stage_id=1,
            replica_id=0,
            chunk_arrival_times_s=arrivals,
            chunk_bytes=bytes_,
            sample_rate=24000,
            threshold_s=0.1,
        )
        # 0 underrun -> observe with 0.0, continuity_ok incremented at threshold 100.
        assert ("observe_audio_underrun", "1", "0", 0.0) in stub.calls
        assert ("inc_audio_continuity_ok", "1", "0", 100) in stub.calls

    def test_late_chunk_emits_nonzero_underrun_and_no_continuity_inc(self):
        stub = _StubModMetrics()
        # Two chunks but second arrives 1s late; underrun > 0.1s threshold.
        arrivals = [0.0, 2.0]
        bytes_ = [24000, 24000]
        observe_audio_streaming_finalize(
            stub,
            stage_id=1,
            replica_id=0,
            chunk_arrival_times_s=arrivals,
            chunk_bytes=bytes_,
            sample_rate=24000,
            threshold_s=0.1,
        )
        underrun_calls = [c for c in stub.calls if c[0] == "observe_audio_underrun"]
        assert underrun_calls
        assert underrun_calls[0][-1] > 0.1
        assert not any(c[0] == "inc_audio_continuity_ok" for c in stub.calls)

    def test_empty_arrivals_skipped(self):
        stub = _StubModMetrics()
        observe_audio_streaming_finalize(
            stub,
            stage_id=1,
            replica_id=0,
            chunk_arrival_times_s=[],
            chunk_bytes=[],
            sample_rate=24000,
        )
        assert stub.calls == []

    def test_replica_none_skipped(self):
        stub = _StubModMetrics()
        observe_audio_streaming_finalize(
            stub,
            stage_id=1,
            replica_id=None,
            chunk_arrival_times_s=[0.5],
            chunk_bytes=[24000],
            sample_rate=24000,
        )
        assert stub.calls == []


class TestBucketSelection:
    def test_audio_rtf_uses_rtf_buckets(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_buckets", "0"
        mod.observe_audio_rtf(stage, replica, 0.5)
        out = generate_latest(REGISTRY).decode()
        rtf_marker = f'{defs.AUDIO_RTF_METRIC}_bucket{{le="0.9"'
        assert rtf_marker in out, "audio_rtf should use RTF_BUCKETS containing le=0.9"

    def test_audio_ttfp_uses_seconds_buckets(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_seconds", "0"
        mod.observe_audio_ttfp(stage, replica, 0.1)
        out = generate_latest(REGISTRY).decode()
        sec_marker = f'{defs.AUDIO_TTFP_S}_bucket{{le="0.05"'
        assert sec_marker in out, "audio_ttfp should use SECONDS_BUCKETS containing le=0.05"

    def test_audio_underrun_uses_seconds_fast_buckets(self, mod: OmniModalityMetrics) -> None:
        stage, replica = "talker_fast", "0"
        mod.observe_audio_underrun(stage, replica, 0.005)
        out = generate_latest(REGISTRY).decode()
        # SECONDS_FAST_BUCKETS includes le=0.001 which is absent from SECONDS_BUCKETS.
        fast_marker = f'{defs.AUDIO_UNDERRUN_S}_bucket{{le="0.001"'
        assert fast_marker in out, "audio_underrun_s should use SECONDS_FAST_BUCKETS containing le=0.001"
