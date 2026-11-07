# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.stage_input_processors import streaming_text_frontend as frontend_module
from vllm_omni.model_executor.stage_input_processors.streaming_text_commitment import CommitmentTextFrontend
from vllm_omni.model_executor.stage_input_processors.streaming_text_frontend import (
    StreamingTextFrontendConfig,
    get_streaming_text_frontend_factory,
    register_streaming_text_frontend,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The total is 2026 dollars. Thank you", ("The total is 2026 dollars.", " Thank you")),
        ("等等。．５！￥2026元。", ("等等。", "．５！", "￥2026元。")),
        ("There are 3\nMore things.", ("There are 3\n", "More things.")),
        ("Use e.g. 3.14 kg. Wait... Done!", ("Use e.g. 3.14 kg.", " Wait...", " Done!")),
    ],
)
def test_commitment_frontend_segments_are_packet_boundary_independent(text, expected):
    for packets in [(text,), tuple(text)] + [(text[:i], text[i:]) for i in range(len(text) + 1)]:
        frontend = CommitmentTextFrontend()
        segments = tuple(segment for packet in packets for segment in frontend.feed(packet)) + frontend.finish()
        assert segments == expected
        assert "".join(segments) == text


def test_commitment_frontend_emits_at_boundary_then_flushes_raw_suffix():
    frontend = CommitmentTextFrontend()
    assert frontend.feed("The total is 2026") == ()
    assert frontend.feed(" dollars. ") == ("The total is 2026 dollars.",)
    assert frontend.feed("Thank you") == ()
    assert frontend.finish() == (" Thank you",)
    with pytest.raises(RuntimeError, match="finished"):
        frontend.finish()
    with pytest.raises(RuntimeError, match="finished"):
        frontend.feed("again")


def test_commitment_frontend_failed_feed_does_not_release_partial_segments():
    frontend = CommitmentTextFrontend(max_pending_chars=3)
    assert frontend.feed("prefix ") == ()
    with pytest.raises(ValueError, match="max_pending_chars=3"):
        frontend.feed("ready! 2026")
    with pytest.raises(RuntimeError, match="failed"):
        frontend.finish()


def test_commitment_frontend_skips_whitespace_only_utterance():
    frontend = CommitmentTextFrontend()
    assert frontend.feed(" \n\t") == ()
    assert frontend.finish() == ()


def test_builtin_factory_creates_fresh_frontends_with_the_requested_limit():
    factory = get_streaming_text_frontend_factory("zh_en_special_v1")
    config = StreamingTextFrontendConfig(language="English", max_pending_chars=3)
    first, second = factory(config), factory(config)
    assert first is not second
    assert first.feed("abc") == ()
    assert second.finish() == ()
    with pytest.raises(ValueError, match="max_pending_chars=3"):
        first.feed("d")


def test_frontend_registry_rejects_unknown_and_duplicate_profiles(monkeypatch):
    monkeypatch.setattr(frontend_module, "_FRONTEND_FACTORIES", {})
    with pytest.raises(ValueError, match="unsupported commitment profile"):
        get_streaming_text_frontend_factory("custom_v1")

    def factory(config):
        return CommitmentTextFrontend(max_pending_chars=config.max_pending_chars)

    register_streaming_text_frontend("custom_v1", factory)
    assert get_streaming_text_frontend_factory("custom_v1") is factory
    with pytest.raises(ValueError, match="already registered"):
        register_streaming_text_frontend("custom_v1", factory)
