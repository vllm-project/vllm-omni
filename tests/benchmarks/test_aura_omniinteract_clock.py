# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from vllm_omni.benchmarks.omniinteract import _session_capabilities, _wants_annotation_video_clock
from vllm_omni.model_executor.models.aura_omni.benchmarks.omniinteract_clock import (
    stream_annotation_clock,
    wants_annotation_video_clock,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Events:
    def __init__(self, events):
        self.events = events


class _Client:
    def __init__(self, capabilities):
        self.events = _Events([{"type": "session.created", "session": {"capabilities": capabilities}}])
        self._client = None


def test_annotation_clock_follows_capabilities_not_model_name():
    aura = {"required_input_modalities": ["video"], "supports_turn_commit_only": True}
    minicpm = {"required_input_modalities": ["audio"], "supports_turn_commit_only": False}
    audio_commit = {"required_input_modalities": ["audio"], "supports_turn_commit_only": True}
    assert wants_annotation_video_clock(aura)
    assert not wants_annotation_video_clock(minicpm)
    assert not wants_annotation_video_clock(audio_commit)
    assert not wants_annotation_video_clock(None)
    assert _wants_annotation_video_clock(_Client(minicpm)) is False
    assert _wants_annotation_video_clock(_Client(aura)) is True
    assert _session_capabilities(_Client(aura))["required_input_modalities"] == ["video"]


def test_annotation_clock_commits_only_the_due_wav(monkeypatch, tmp_path: Path):
    video = tmp_path / "videos" / "0002.mp4"
    video.parent.mkdir()
    video.write_bytes(b"")
    sent = []

    class Client:
        async def send(self, event):
            sent.append(event)

        async def commit(self):
            sent.append({"type": "commit"})

    class Playback:
        async def acknowledge(self, client):
            del client

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.aura_omni.benchmarks.omniinteract_clock.load_annotation_utterances",
        lambda path: [{"at_sec": 0.0, "pcm": b"\x01\x00" * 160, "sent": False, "qa_index": 0}],
    )
    chunks, frames, _, _ = asyncio.run(
        stream_annotation_clock(
            Client(),
            ["frame-a", "frame-b"],
            Playback(),
            video_path=video,
            fps=2.0,
        )
    )
    assert (chunks, frames) == (2, 2)
    assert sent[0]["is_speech"] is True
    assert sent[0]["video_frames"] == ["frame-a"]
    assert sent[1] == {"type": "commit"}
    # The trailing silent frame is one vision turn, committed after the clip.
    assert sent[2]["is_speech"] is False
    assert sent[2]["video_frames"] == ["frame-b"]
    assert sent[3] == {"type": "commit"}
    assert sent.count({"type": "commit"}) == 2


def test_annotation_clock_commits_silent_frames_in_pairs(monkeypatch, tmp_path: Path):
    video = tmp_path / "videos" / "0002.mp4"
    video.parent.mkdir()
    video.write_bytes(b"")
    sent = []

    class Client:
        async def send(self, event):
            sent.append(event)

        async def commit(self):
            sent.append({"type": "commit"})

    class Playback:
        async def acknowledge(self, client):
            del client

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.aura_omni.benchmarks.omniinteract_clock.load_annotation_utterances",
        lambda path: [{"at_sec": 100.0, "pcm": b"\x01\x00" * 160, "sent": False, "qa_index": 0}],
    )
    asyncio.run(
        stream_annotation_clock(
            Client(),
            ["f1", "f2", "f3", "f4"],
            Playback(),
            video_path=video,
            fps=2.0,
        )
    )
    vision = [event for event in sent if event.get("is_speech") is False]
    assert [event["video_frames"] for event in vision] == [["f1", "f2"], ["f3", "f4"]]
    assert sent.count({"type": "commit"}) == 2


def test_annotation_clock_holds_vision_while_a_response_is_open(monkeypatch, tmp_path: Path):
    video = tmp_path / "videos" / "0002.mp4"
    video.parent.mkdir()
    video.write_bytes(b"")
    sent = []

    class Events:
        def __init__(self):
            self.events = [{"type": "response.created", "response": {"id": "resp-open"}}]

    class Client:
        def __init__(self):
            self.events = Events()

        async def send(self, event):
            sent.append(event)

        async def commit(self):
            sent.append({"type": "commit"})

    class Playback:
        async def acknowledge(self, client):
            del client

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.aura_omni.benchmarks.omniinteract_clock.load_annotation_utterances",
        lambda path: [{"at_sec": 100.0, "pcm": b"\x01\x00" * 160, "sent": False, "qa_index": 0}],
    )
    asyncio.run(
        stream_annotation_clock(
            Client(),
            ["f1", "f2", "f3", "f4"],
            Playback(),
            video_path=video,
            fps=2.0,
        )
    )
    assert sent == []
