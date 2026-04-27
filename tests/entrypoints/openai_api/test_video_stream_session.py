# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VideoStreamSession (Phase 2 + 3)."""

from __future__ import annotations

import pytest

from tests.entrypoints.openai_api.conftest_video import (
    make_gradient_jpeg,
    make_jpeg,
)
from vllm_omni.entrypoints.openai.video_stream_session import (
    VideoStreamConfig,
    VideoStreamSession,
)

# ---------------------------------------------------------------------------
# VideoStreamConfig
# ---------------------------------------------------------------------------


class TestVideoStreamConfig:
    def test_defaults(self):
        cfg = VideoStreamConfig()
        assert cfg.max_frames == 64
        assert cfg.num_sample_frames == 16
        assert cfg.evs_enabled is True
        assert cfg.evs_threshold == 0.95

    def test_from_dict(self):
        cfg = VideoStreamConfig.from_dict(
            {
                "model": "test-model",
                "max_frames": 32,
                "evs_threshold": 0.90,
                "unknown_field": "ignored",
            }
        )
        assert cfg.model == "test-model"
        assert cfg.max_frames == 32
        assert cfg.evs_threshold == 0.90

    def test_from_dict_empty(self):
        cfg = VideoStreamConfig.from_dict({})
        assert cfg.model == ""
        assert cfg.max_frames == 64

    def test_from_dict_invalid_type(self):
        with pytest.raises(TypeError, match="max_frames.*expected int.*got str"):
            VideoStreamConfig.from_dict({"max_frames": "potato"})

    def test_from_dict_invalid_bool(self):
        with pytest.raises(TypeError, match="evs_enabled.*expected bool"):
            VideoStreamConfig.from_dict({"evs_enabled": "yes"})

    def test_from_dict_invalid_modalities(self):
        with pytest.raises(TypeError, match="modalities.*expected list"):
            VideoStreamConfig.from_dict({"modalities": 42})

    def test_from_dict_evs_threshold_int_accepted(self):
        """JSON doesn't distinguish int/float — int 1 is a valid threshold."""
        cfg = VideoStreamConfig.from_dict({"evs_threshold": 1})
        assert cfg.evs_threshold == 1


# ---------------------------------------------------------------------------
# Frame buffer & sliding window (uses deque)
# ---------------------------------------------------------------------------


class TestFrameBuffer:
    def test_add_frame_basic(self):
        cfg = VideoStreamConfig(evs_enabled=False, max_frames=10)
        session = VideoStreamSession(cfg)
        assert session.add_frame(make_jpeg()) is True
        assert session.frame_count == 1

    def test_add_frame_too_large(self):
        cfg = VideoStreamConfig(evs_enabled=False, max_frames=10)
        session = VideoStreamSession(cfg)
        huge = b"\xff" * (10 * 1024 * 1024 + 1)  # just over 10 MB
        with pytest.raises(ValueError, match="Frame too large"):
            session.add_frame(huge)

    def test_sliding_window(self):
        cfg = VideoStreamConfig(evs_enabled=False, max_frames=3)
        session = VideoStreamSession(cfg)
        for i in range(5):
            session.add_frame(make_jpeg(r=i * 50))
        assert session.frame_count == 3

    def test_sliding_window_keeps_newest(self):
        cfg = VideoStreamConfig(evs_enabled=False, max_frames=2)
        session = VideoStreamSession(cfg)
        f1 = make_jpeg(10, 10, 10)
        f2 = make_jpeg(20, 20, 20)
        f3 = make_jpeg(30, 30, 30)
        session.add_frame(f1)
        session.add_frame(f2)
        session.add_frame(f3)
        sampled = session.sample_frames()
        assert len(sampled) == 2
        assert sampled[0] == f2
        assert sampled[1] == f3


# ---------------------------------------------------------------------------
# EVS integration
# ---------------------------------------------------------------------------


class TestEVSIntegration:
    def test_evs_drops_identical_frames(self):
        cfg = VideoStreamConfig(evs_enabled=True, evs_threshold=0.90)
        session = VideoStreamSession(cfg)
        frame = make_jpeg(100, 100, 100)
        assert session.add_frame(frame) is True
        assert session.add_frame(frame) is False
        assert session.frame_count == 1

    def test_evs_keeps_different_frames(self):
        cfg = VideoStreamConfig(evs_enabled=True, evs_threshold=0.95)
        session = VideoStreamSession(cfg)
        for i in range(5):
            assert session.add_frame(make_gradient_jpeg(seed=i)) is True
        assert session.frame_count == 5

    def test_evs_disabled(self):
        cfg = VideoStreamConfig(evs_enabled=False)
        session = VideoStreamSession(cfg)
        frame = make_jpeg()
        assert session.add_frame(frame) is True
        assert session.add_frame(frame) is True
        assert session.frame_count == 2

    def test_evs_stats(self):
        cfg = VideoStreamConfig(evs_enabled=True, evs_threshold=0.90)
        session = VideoStreamSession(cfg)
        frame = make_jpeg()
        session.add_frame(frame)
        session.add_frame(frame)
        stats = session.evs_stats
        assert stats is not None
        assert stats["retained_count"] == 1
        assert stats["dropped_count"] == 1

    def test_evs_stats_none_when_disabled(self):
        cfg = VideoStreamConfig(evs_enabled=False)
        session = VideoStreamSession(cfg)
        assert session.evs_stats is None


# ---------------------------------------------------------------------------
# Uniform sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sample_exact(self):
        cfg = VideoStreamConfig(evs_enabled=False, num_sample_frames=3, max_frames=10)
        session = VideoStreamSession(cfg)
        frames = [make_gradient_jpeg(i) for i in range(3)]
        for f in frames:
            session.add_frame(f)
        sampled = session.sample_frames()
        assert len(sampled) == 3
        assert sampled == frames

    def test_sample_fewer_than_requested(self):
        cfg = VideoStreamConfig(evs_enabled=False, num_sample_frames=10, max_frames=64)
        session = VideoStreamSession(cfg)
        for i in range(3):
            session.add_frame(make_gradient_jpeg(i))
        assert len(session.sample_frames()) == 3

    def test_sample_uniform(self):
        cfg = VideoStreamConfig(evs_enabled=False, num_sample_frames=4, max_frames=64)
        session = VideoStreamSession(cfg)
        for i in range(10):
            session.add_frame(make_gradient_jpeg(i))
        sampled = session.sample_frames()
        assert len(sampled) == 4
        expected = [session._frames[i] for i in [0, 3, 6, 9]]
        assert sampled == expected

    def test_sample_empty(self):
        cfg = VideoStreamConfig(evs_enabled=False)
        session = VideoStreamSession(cfg)
        assert session.sample_frames() == []

    def test_sample_single_frame_from_multi_frame_buffer(self):
        cfg = VideoStreamConfig(evs_enabled=False, num_sample_frames=1, max_frames=64)
        session = VideoStreamSession(cfg)
        first = make_gradient_jpeg(0)
        second = make_gradient_jpeg(1)
        session.add_frame(first)
        session.add_frame(second)

        assert session.sample_frames() == [second]


# ---------------------------------------------------------------------------
# Audio buffer (Phase 3)
# ---------------------------------------------------------------------------


class TestAudioBuffer:
    def test_add_audio_chunk(self):
        session = VideoStreamSession(VideoStreamConfig())
        assert session.has_audio is False
        session.add_audio_chunk(b"\x00" * 100)
        assert session.has_audio is True

    def test_clear_audio(self):
        session = VideoStreamSession(VideoStreamConfig())
        session.add_audio_chunk(b"\x00" * 100)
        session.clear_audio()
        assert session.has_audio is False


# ---------------------------------------------------------------------------
# build_chat_request
# ---------------------------------------------------------------------------


class TestBuildChatRequest:
    def test_video_only_request(self):
        cfg = VideoStreamConfig(model="test-model", evs_enabled=False, num_sample_frames=4)
        session = VideoStreamSession(cfg)
        for i in range(4):
            session.add_frame(make_gradient_jpeg(i))

        request = session.build_chat_request("Describe this scene.")
        assert request.model == "test-model"
        assert request.stream is True

        content = request.messages[0]["content"]
        assert len(content) == 5  # 4 image_url + 1 text
        image_parts = [p for p in content if p["type"] == "image_url"]
        text_parts = [p for p in content if p["type"] == "text"]
        assert len(image_parts) == 4
        assert len(text_parts) == 1
        assert text_parts[0]["text"] == "Describe this scene."

        mm_kw = getattr(request, "mm_processor_kwargs", None)
        assert mm_kw is None or not mm_kw.get("use_audio_in_video", False)

    def test_video_plus_audio_request(self):
        cfg = VideoStreamConfig(model="test-model", evs_enabled=False, num_sample_frames=2)
        session = VideoStreamSession(cfg)
        session.add_frame(make_gradient_jpeg(0))
        session.add_frame(make_gradient_jpeg(1))
        session.add_audio_chunk(b"\x00" * 3200)

        request = session.build_chat_request("What is being said?")

        content = request.messages[0]["content"]
        assert len(content) == 4  # 2 image_url + 1 audio_url + 1 text
        audio_parts = [p for p in content if p["type"] == "audio_url"]
        assert len(audio_parts) == 1
        # RFC 3551: audio/L16 for linear 16-bit PCM
        assert audio_parts[0]["audio_url"]["url"].startswith("data:audio/L16;rate=16000;base64,")

        mm_kw = getattr(request, "mm_processor_kwargs", None) or {}
        assert mm_kw.get("use_audio_in_video") is True

    def test_image_url_is_valid_base64(self):
        cfg = VideoStreamConfig(evs_enabled=False, num_sample_frames=1)
        session = VideoStreamSession(cfg)
        session.add_frame(make_jpeg(200, 100, 50))
        request = session.build_chat_request("test")
        content = request.messages[0]["content"]
        img_url = content[0]["image_url"]["url"]
        assert img_url.startswith("data:image/jpeg;base64,")
        import base64

        b64_data = img_url.split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0

    def test_clear_audio_after_query(self):
        session = VideoStreamSession(VideoStreamConfig(evs_enabled=False))
        session.add_frame(make_jpeg())
        session.add_audio_chunk(b"\x00" * 100)
        session.build_chat_request("test")
        session.clear_audio()
        assert session.has_audio is False
        assert session.frame_count == 1
