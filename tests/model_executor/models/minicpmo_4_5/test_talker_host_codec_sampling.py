# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the NPU host-side codec sampler.

``make_omni_output``'s NPU branch has no coverage in CI, which let a
positional-argument drift slip through once (``max_tokens`` received
``eos_window_masked`` and every NPU request finished without emitting a
codec token). These tests call the sampler the way the production call
site does -- keywords for every scalar -- and pin the emit/limit
transition plus the per-request state release.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_CODEC_VOCAB = 16
_EOS_ID = 8


def _make_talker() -> tts.MiniCPMO45OmniTTSForConditionalGeneration:
    """Bypass ``__init__``: the sampler only needs the codec knob defaults."""
    talker = object.__new__(tts.MiniCPMO45OmniTTSForConditionalGeneration)
    talker._codec_seed = 0
    talker._codec_eos_id = _EOS_ID
    talker._codec_temperature = 0.0
    talker._codec_top_k = -1
    talker._codec_top_p = 1.0
    talker._codec_repetition_penalty = 1.0
    return talker


def _sample(talker, *, request_id: str, step: int, state: dict, max_tokens: int = 3):
    return talker._sample_audio_code_host(
        hidden_state=torch.zeros(1, 4),
        history=torch.empty(0, dtype=torch.long),
        request_id=request_id,
        request_state=state,
        step=step,
        min_tokens=0,
        max_tokens=max_tokens,
        eos_window_masked=False,
        lg_dev=torch.zeros(_CODEC_VOCAB),
    )


def test_host_sampler_emits_first_frame_and_honors_max_tokens() -> None:
    """First frame emits a code; ``max_tokens`` still ends the request."""
    talker = _make_talker()
    state = {"codec_temperature": 0.0, "codec_seed": 0}

    host_id, finished, emit = _sample(talker, request_id="req-1", step=0, state=state)
    # A positional drift (max_tokens <- eos_window_masked) would finish here.
    assert emit and not finished
    assert host_id != _EOS_ID

    # max_tokens=3: frames 0 and 1 emit, frame 2 reaches the limit.
    host_id, finished, emit = _sample(talker, request_id="req-1", step=1, state=state)
    assert emit and not finished
    assert host_id != _EOS_ID

    host_id, finished, emit = _sample(talker, request_id="req-1", step=2, state=state)
    assert not emit
    assert finished


def test_host_sampler_reads_per_request_knobs() -> None:
    """Per-request knobs come from ``request_state``, not the defaults."""
    talker = _make_talker()
    state = {"codec_temperature": 0.0, "codec_seed": 0, "codec_stop_token_ids": [_CODEC_VOCAB - 1]}
    logits = torch.zeros(_CODEC_VOCAB)
    logits[_CODEC_VOCAB - 1] = 100.0  # the per-request stop id would win
    host_id, finished, emit = talker._sample_audio_code_host(
        hidden_state=torch.zeros(1, 4),
        history=torch.empty(0, dtype=torch.long),
        request_id="req-stop",
        request_state=state,
        step=0,
        min_tokens=0,
        max_tokens=8,
        eos_window_masked=False,
        lg_dev=logits,
    )
    assert host_id == _CODEC_VOCAB - 1
    assert finished and not emit


def test_host_sampler_state_released_on_cleanup() -> None:
    """``_flush_deferred_cleanup`` drops the host sampler's per-request state."""
    talker = _make_talker()
    _sample(talker, request_id="req-2", step=0, state={"codec_temperature": 0.0})
    assert "req-2" in talker._request_host_codec_states

    talker._deferred_cleanup_ids = {"req-2"}
    talker._flush_deferred_cleanup()
    assert "req-2" not in talker._request_host_codec_states


@pytest.mark.parametrize("new_turn", [False, True])
def test_native_codec_sampler_resumes_after_condition_eos(new_turn):
    """A native phrase can use several conditions under the same request ID."""
    from types import SimpleNamespace

    talker = _make_talker()
    talker._request_audio_states = {}
    talker._request_condition_states = {}
    talker.emb_text = SimpleNamespace(weight=torch.zeros(1))
    talker._build_condition_embeddings = lambda *a, **k: torch.zeros(3, 4)
    talker._build_streaming_recompute_embeddings = lambda cond, **k: cond
    first_state = {"codec_temperature": 0.0}
    rid = "duplex-resumable"
    assert _sample(talker, request_id=rid, step=0, state=first_state, max_tokens=2)[2]
    assert _sample(talker, request_id=rid, step=1, state=first_state, max_tokens=2)[1]
    old = talker._request_host_codec_states[rid]
    draw_stream = old["generator"]
    assert old["history"] == [0]
    from vllm_omni.engine.duplex.intermediate import set_tts_handoff

    info = {
        "request_id": rid,
        "native_duplex": True,
        "meta": {"turn_start": new_turn, "turn_end": True, "turn_text_ends_sentence": True},
    }
    set_tts_handoff(info, torch.tensor([3]), torch.zeros(1, 4))
    _, _, scratch = talker.preprocess(torch.zeros(3, dtype=torch.long), input_embeds=None, **info)
    _, finished, emit = _sample(talker, request_id=rid, step=0, state=scratch["audio_state"])
    assert emit and not finished
    assert old["generator"] is draw_stream
    assert old["history"] == ([0] if new_turn else [0, 0])
