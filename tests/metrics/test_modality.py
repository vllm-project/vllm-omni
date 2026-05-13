from __future__ import annotations

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.modality import OmniModalityMetrics

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
