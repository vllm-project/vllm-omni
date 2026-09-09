# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Qwen3-TTS leading-silence ban (#4966, #5048).

``compute_logits`` suppresses the checkpoint-derived silence codec vocabulary
for the first ``silence_ban_frames`` decode frames of a request, and only for
requests running Base voice-clone in x-vector-only mode. ICL onsets are
legitimate audio and must not be masked.

Two per-request gates guard the mask, and both index by batch position:

* the decode step, from ``sampling_metadata.output_token_ids[i]`` -- the
  request's generated-token history, so the window closes on its own and a
  preempted request cannot re-enter it;
* the generation mode, recorded at prefill in ``preprocess_batch`` and looked
  up through the batch req_id order the runner supplies via
  ``set_batch_req_ids``.

These tests drive the methods directly against a talker built with ``__new__``:
no config, no checkpoint, no device. Only the attributes the masking path reads
are populated, and ``logits_processor`` is stubbed to return a fixed tensor.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import resolve_x_vector_only
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 32
_SILENCE_IDS = (3, 7, 11)


def _make_talker(*, ban_frames: int, logits: torch.Tensor, modes=None):
    """Talker stub exposing only what ``compute_logits`` touches.

    ``modes`` is one entry per batch row: ``True`` for x-vector-only, ``False``
    for ICL, ``None`` for a request whose mode was never recorded. Defaults to
    every row in scope, so a test that says nothing about mode exercises the
    step gate alone.
    """
    talker = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)

    silence_mask = torch.zeros((_VOCAB,), dtype=torch.bool)
    for token in _SILENCE_IDS:
        silence_mask[token] = True

    if modes is None:
        modes = [True] * logits.shape[0]
    req_ids = [f"req{i}" for i in range(len(modes))]

    object.__setattr__(talker, "_silence_ban_frames", ban_frames)
    object.__setattr__(talker, "_silence_mask", silence_mask)
    object.__setattr__(talker, "_codec_disallowed_mask", torch.zeros((_VOCAB,), dtype=torch.bool))
    object.__setattr__(talker, "_batch_req_ids", req_ids)
    object.__setattr__(
        talker,
        "_req_x_vector_only",
        {rid: mode for rid, mode in zip(req_ids, modes, strict=True) if mode is not None},
    )
    object.__setattr__(talker, "lm_head", None)
    object.__setattr__(talker, "logits_processor", lambda _head, _hidden: logits.clone())
    return talker


def _metadata(*steps: int):
    """sampling_metadata whose output_token_ids encode one step per request."""
    return SimpleNamespace(output_token_ids=[[0] * step for step in steps])


def _banned(row: torch.Tensor) -> set[int]:
    return {int(i) for i in torch.nonzero(torch.isinf(row) & (row < 0)).flatten()}


# -------------------- the step gate --------------------


def test_ban_disabled_leaves_logits_unchanged() -> None:
    logits = torch.zeros((1, _VOCAB))
    talker = _make_talker(ban_frames=0, logits=logits)
    out = talker.compute_logits(torch.zeros((1, 4)), _metadata(0))
    assert _banned(out[0]) == set(), "ban_frames=0 must be a no-op"


@pytest.mark.parametrize("step", [0, 1, 2])
def test_early_steps_mask_only_the_silence_vocabulary(step: int) -> None:
    logits = torch.zeros((1, _VOCAB))
    talker = _make_talker(ban_frames=3, logits=logits)
    out = talker.compute_logits(torch.zeros((1, 4)), _metadata(step))
    assert _banned(out[0]) == set(_SILENCE_IDS), f"step {step} must mask exactly the silence ids"


@pytest.mark.parametrize("step", [3, 4, 50])
def test_steps_past_the_window_are_untouched(step: int) -> None:
    logits = torch.zeros((1, _VOCAB))
    talker = _make_talker(ban_frames=3, logits=logits)
    out = talker.compute_logits(torch.zeros((1, 4)), _metadata(step))
    assert _banned(out[0]) == set(), f"step {step} is past the window; nothing may be masked"


def test_mixed_step_batch_masks_only_the_early_request() -> None:
    logits = torch.zeros((2, _VOCAB))
    talker = _make_talker(ban_frames=3, logits=logits)
    out = talker.compute_logits(torch.zeros((2, 4)), _metadata(0, 3))
    assert _banned(out[0]) == set(_SILENCE_IDS), "the step-0 request must be masked"
    assert _banned(out[1]) == set(), "the step-3 request must not be masked"


# -------------------- the mode gate --------------------


def test_icl_request_is_never_masked() -> None:
    logits = torch.zeros((1, _VOCAB))
    talker = _make_talker(ban_frames=3, logits=logits, modes=[False])
    out = talker.compute_logits(torch.zeros((1, 4)), _metadata(0))
    assert _banned(out[0]) == set(), "an ICL onset is real audio and must not be suppressed"


def test_mixed_mode_batch_masks_only_the_x_vector_request() -> None:
    logits = torch.zeros((2, _VOCAB))
    talker = _make_talker(ban_frames=3, logits=logits, modes=[True, False])
    out = talker.compute_logits(torch.zeros((2, 4)), _metadata(0, 0))
    assert _banned(out[0]) == set(_SILENCE_IDS), "the x-vector-only request must be masked"
    assert _banned(out[1]) == set(), "the ICL request in the same batch must not be"


def test_unrecorded_mode_is_left_alone() -> None:
    logits = torch.zeros((2, _VOCAB))
    talker = _make_talker(ban_frames=3, logits=logits, modes=[True, None])
    out = talker.compute_logits(torch.zeros((2, 4)), _metadata(0, 0))
    assert _banned(out[0]) == set(_SILENCE_IDS)
    assert _banned(out[1]) == set(), "a request that cannot be classified must not be masked"


def test_stale_batch_req_ids_skip_the_ban() -> None:
    logits = torch.zeros((2, _VOCAB))
    talker = _make_talker(ban_frames=3, logits=logits)
    # One req_id for a two-row batch: the recorded order no longer indexes the
    # logits, so masking by position would hit the wrong request.
    object.__setattr__(talker, "_batch_req_ids", ["req0"])
    out = talker.compute_logits(torch.zeros((2, 4)), _metadata(0, 0))
    assert _banned(out[0]) == set(), "a req_id/logits length mismatch must disable the ban"
    assert _banned(out[1]) == set()


# -------------------- metadata plumbing --------------------


def test_missing_sampling_metadata_does_not_crash() -> None:
    logits = torch.zeros((1, _VOCAB))
    talker = _make_talker(ban_frames=3, logits=logits)
    out = talker.compute_logits(torch.zeros((1, 4)), None)
    assert _banned(out[0]) == set()


@pytest.mark.parametrize(
    "metadata",
    [SimpleNamespace(output_token_ids=None), SimpleNamespace(), SimpleNamespace(output_token_ids=[])],
)
def test_absent_output_token_ids_warns_without_masking(metadata) -> None:
    logits = torch.zeros((1, _VOCAB))
    talker = _make_talker(ban_frames=3, logits=logits)
    out = talker.compute_logits(torch.zeros((1, 4)), metadata)
    assert _banned(out[0]) == set(), "the ban must skip, not crash, when history is unavailable"


def test_batch_length_mismatch_skips_the_ban() -> None:
    logits = torch.zeros((2, _VOCAB))
    talker = _make_talker(ban_frames=3, logits=logits)
    out = talker.compute_logits(torch.zeros((2, 4)), _metadata(0))
    assert _banned(out[0]) == set(), "a runner shape mismatch must disable the ban, not misindex it"
    assert _banned(out[1]) == set()


# -------------------- mode recording and eviction --------------------


def _mode_talker(ban_frames: int = 3):
    talker = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    object.__setattr__(talker, "_silence_ban_frames", ban_frames)
    object.__setattr__(talker, "_req_x_vector_only", {})
    object.__setattr__(talker, "_batch_req_ids", [])
    object.__setattr__(talker, "_prompt_builder", SimpleNamespace(preprocess_batch=lambda **_kwargs: None))
    return talker


def _info(*, task_type: str = "Base", xvec: bool = True, icl: bool | None = None) -> dict:
    info: dict = {"task_type": [task_type], "x_vector_only_mode": [xvec]}
    if icl is not None:
        info["voice_clone_prompt"] = [{"icl_mode": icl}]
    return info


def test_preprocess_batch_records_the_generation_mode() -> None:
    talker = _mode_talker()
    buffer = {
        "req0": _info(xvec=True),
        "req1": _info(xvec=False),
        "req2": _info(task_type="CustomVoice"),
    }
    talker.preprocess_batch(req_ids=list(buffer), model_intermediate_buffer=buffer, device=torch.device("cpu"))
    assert talker._req_x_vector_only == {"req0": True, "req1": False, "req2": None}


def test_recorded_mode_matches_the_prompt_builder_resolution() -> None:
    """The recording path must agree with the resolution the prefill path uses;
    a divergence here would mask requests the builder treats as in-context."""
    infos = [
        _info(xvec=True),
        _info(xvec=False),
        _info(xvec=True, icl=True),
        _info(xvec=False, icl=False),
        _info(task_type="VoiceDesign"),
    ]
    talker = _mode_talker()
    buffer = {f"req{i}": info for i, info in enumerate(infos)}
    talker.preprocess_batch(req_ids=list(buffer), model_intermediate_buffer=buffer, device=torch.device("cpu"))
    for req_id, info in buffer.items():
        assert talker._req_x_vector_only[req_id] == resolve_x_vector_only(info)


def test_preprocess_batch_records_nothing_when_the_ban_is_off() -> None:
    talker = _mode_talker(ban_frames=0)
    buffer = {"req0": _info(xvec=True)}
    talker.preprocess_batch(req_ids=["req0"], model_intermediate_buffer=buffer, device=torch.device("cpu"))
    assert talker._req_x_vector_only == {}


def test_set_batch_req_ids_evicts_departed_requests() -> None:
    talker = _mode_talker()
    object.__setattr__(talker, "_req_x_vector_only", {"req0": True, "req1": False, "req2": True})
    talker.set_batch_req_ids(["req2", "req0"])
    assert talker._batch_req_ids == ["req2", "req0"], "batch order must be preserved, not sorted"
    assert talker._req_x_vector_only == {"req0": True, "req2": True}, "req1 has left the batch"


# -------------------- _init_silence_mask sanity checks --------------------
#
# The silence vocabulary is derived from the checkpoint at load time rather than
# hardcoded. A derivation that comes back empty, oversized, or out of range means
# the encode did not produce silence codes, so the ban disables itself instead of
# masking a guess.

_TALKER_VOCAB = 2048


def _make_init_talker(*, codes, ban_frames: int = 3, vocab: int = _TALKER_VOCAB):
    """Talker stub exposing only what ``_init_silence_mask`` touches.

    ``codes`` is either the list ``_encode_ref_audio_batch`` should return or an
    exception instance it should raise.
    """
    talker = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)

    def _encode(_wavs, _sr, *, device):
        if isinstance(codes, Exception):
            raise codes
        return codes

    def _parameters():
        return iter([torch.zeros(1)])

    object.__setattr__(talker, "_silence_ban_frames", ban_frames)
    object.__setattr__(talker, "_silence_mask", torch.zeros((vocab,), dtype=torch.bool))
    object.__setattr__(talker, "talker_config", SimpleNamespace(vocab_size=vocab))
    object.__setattr__(talker, "_encode_ref_audio_batch", _encode)
    object.__setattr__(talker, "parameters", _parameters)
    return talker


def _codes_from(ids) -> list[torch.Tensor]:
    """One [T, Q] codec tensor whose codebook-0 column holds ``ids``."""
    cb0 = torch.tensor(list(ids), dtype=torch.long).unsqueeze(1)
    return [torch.cat([cb0, torch.zeros_like(cb0)], dim=1)]


def test_plausible_derivation_populates_the_mask() -> None:
    talker = _make_init_talker(codes=_codes_from([4, 4, 9, 17]))
    talker._init_silence_mask()
    assert talker._silence_ban_frames == 3
    assert set(torch.nonzero(talker._silence_mask).flatten().tolist()) == {4, 9, 17}


def test_empty_derivation_disables_the_ban() -> None:
    talker = _make_init_talker(codes=[])
    talker._init_silence_mask()
    assert talker._silence_ban_frames == 0, "an all-False mask must not stay enabled"
    assert not talker._silence_mask.any()


def test_oversized_derivation_disables_the_ban() -> None:
    limit = max(64, _TALKER_VOCAB // 20)
    talker = _make_init_talker(codes=_codes_from(range(limit + 1)))
    talker._init_silence_mask()
    assert talker._silence_ban_frames == 0, "a speech-sized vocabulary must not be masked"
    assert not talker._silence_mask.any()


def test_out_of_range_derivation_disables_the_ban() -> None:
    talker = _make_init_talker(codes=_codes_from([4, 9, _TALKER_VOCAB + 1]))
    talker._init_silence_mask()
    assert talker._silence_ban_frames == 0, "a vocabulary-size disagreement must not be half-applied"
    assert not talker._silence_mask.any()


def test_encoder_failure_disables_the_ban() -> None:
    talker = _make_init_talker(codes=RuntimeError("encoder unavailable"))
    talker._init_silence_mask()
    assert talker._silence_ban_frames == 0
    assert not talker._silence_mask.any()


def test_disabled_ban_skips_derivation_entirely() -> None:
    talker = _make_init_talker(codes=RuntimeError("must not be called"), ban_frames=0)
    talker._init_silence_mask()
    assert talker._silence_ban_frames == 0
