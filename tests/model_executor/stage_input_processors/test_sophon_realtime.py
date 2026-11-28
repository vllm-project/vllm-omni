# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Thinker -> talker text streaming for Sophon-Realtime."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors import sophon_realtime as sr

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------- speakable_text
@pytest.mark.parametrize(
    ("text", "final", "expected"),
    [
        ("Sure. <tool_call>{...}</tool_call>", False, "Sure. "),
        ("Let me check <tool", False, "Let me check "),
        ("Let me check <tool", True, "Let me check <tool"),
        ("No tools here", False, "No tools here"),
    ],
)
def test_speakable_text(text, final, expected):
    assert sr.speakable_text(text, final) == expected


# ---------------------------------------------------------------- clean_for_speech
@pytest.mark.parametrize(
    ("text", "final", "expected"),
    [
        ("Hello **world**", True, "Hello world"),
        ("Ha [laughter] that's funny", True, "Ha [laughter] that's funny"),
        ("Note (an aside) here", True, "Note  here"),
        ("[unknown_tag] ok", True, " ok"),
        ("<strong>really</strong> good", True, "<strong>really</strong> good"),
        # <strong> closes at the next punctuation mark
        ("<strong>really, good", True, "<strong>really</strong>, good"),
        # an open span is closed when the thinker finishes
        ("<laughter>so funny", True, "<laughter>so funny</laughter>"),
        # a bracket that may still close is held back while streaming
        ("wait [laugh", False, "wait "),
        # ... and its text is spoken if it never closes
        ("wait [laugh", True, "wait "),
        ("x → y", True, "x , so  y"),
    ],
)
def test_clean_for_speech(text, final, expected):
    assert sr.clean_for_speech(text, final) == expected


def test_clean_for_speech_long_bracket_is_spoken():
    text = "(" + "a" * 50 + ") done"
    assert sr.clean_for_speech(text, True) == "a" * 50 + ") done"


def test_clean_for_speech_only_grows():
    reply = "Hi there, <strong>this</strong> is [laughter] a test (aside) of *streaming* text."
    previous = ""
    for n in range(1, len(reply) + 1):
        cur = sr.clean_for_speech(reply[:n], False)
        assert cur.startswith(previous), (n, previous, cur)
        previous = cur
    assert sr.clean_for_speech(reply, True).startswith(previous)


def test_strip_sound_tags():
    assert sr.strip_sound_tags("a [laughter] b <strong>c</strong>") == "a  b c"


# ---------------------------------------------------------------- streaming
class _CharTokenizer:
    """One token per character."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def encode(self, text, allowed_special="all"):
        return [ord(c) for c in text]


class _Voices:
    def __init__(self):
        self.calls = []

    def load(self, name, lang, *, clips=True):
        self.calls.append((name, lang, clips))
        return "chelsie_zh", {
            "speech_token": torch.zeros(1, 20, dtype=torch.int32),
            "speech_feat": torch.zeros(1, 40, 80),
            "embedding": torch.zeros(1, 192),
        }


def _manager():
    tok = _CharTokenizer()
    return SimpleNamespace(
        _sophon_realtime_thinker_tokenizer=tok,
        _sophon_realtime_talker_tokenizer=tok,
        _sophon_realtime_voices=_Voices(),
    )


def _request(text: str):
    return SimpleNamespace(
        external_req_id="r0",
        output_token_ids=[ord(c) for c in text],
        additional_information=None,
        sampling_params=None,
    )


def _ids(payload):
    return list(payload.ids.output)


def test_flash_waits_for_lead_and_holds_last_token():
    tm = _manager()
    assert sr.thinker2talker_flash_async_chunk(tm, None, _request("ab")) is None  # 1 stable < lead
    p = sr.thinker2talker_flash_async_chunk(tm, None, _request("abcd"))
    assert "".join(map(chr, _ids(p))) == "abc"  # last token held back
    assert sr.thinker2talker_flash_async_chunk(tm, None, _request("abcd")) is None  # nothing new
    p = sr.thinker2talker_flash_async_chunk(tm, None, _request("abcde"), is_finished=True)
    assert "".join(map(chr, _ids(p))) == "abcde"
    assert bool(p.meta.finished)
    assert "r0" not in tm._sophon_realtime_streams


def test_flash_never_speaks_tool_calls():
    tm = _manager()
    p = sr.thinker2talker_flash_async_chunk(tm, None, _request("Sure<tool_call>{}"), is_finished=True)
    assert "".join(map(chr, _ids(p))) == "Sure"


def test_tag_sends_every_step_once_started():
    tm = _manager()
    assert sr.thinker2talker_tag_async_chunk(tm, None, _request("abc")) is None  # below lead
    first = sr.thinker2talker_tag_async_chunk(tm, None, _request("abcdefg"))
    again = sr.thinker2talker_tag_async_chunk(tm, None, _request("abcdefg"))  # heartbeat
    assert _ids(first) == _ids(again)
    # the voice prompt goes with the first chunk only, and never waits for a clip
    assert first.embed is not None and tuple(first.embed.speech_token.shape) == (1, 20)
    assert int(first.embed.speech_token_len.reshape(-1)[0]) == 20
    assert again.embed is None
    assert tm._sophon_realtime_voices.calls == [("", "zh", False)]
    done = sr.thinker2talker_tag_async_chunk(tm, None, _request("abcdefg"), is_finished=True)
    assert "".join(map(chr, _ids(done))) == "abcdefg"


@pytest.mark.parametrize("func", [sr.thinker2talker_tag_async_chunk, sr.thinker2talker_flash_async_chunk])
def test_processors_ask_for_every_step(func):
    # Without it the scheduler calls a token-only processor only at the end of the reply,
    # and the talker starts after the thinker has finished.
    assert getattr(func, "requires_token_updates", False) is True
