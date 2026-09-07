# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for Gepard's prompt layout, streaming-decode window and config.

The e2e test needs a GPU, NeMo and the checkpoint, so it only runs nightly.
These pin the parts that are pure logic and break silently: which frames each
decode covers, which samples survive the lookback trim, the prompt layout, and
that the packaged deploy config sits where the example looks for it.

The codec is replaced by a shift-invariant stand-in whose output depends only
on the codes, which makes "chunked emissions concatenate to one whole-history
decode" an exact assertion — the contract lookback exists to provide.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import vllm_omni
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig
from vllm_omni.model_executor.models.gepard.gepard_talker import (
    CHUNK_FRAMES,
    FIRST_CHUNK_FRAMES,
    GepardTalkerForConditionalGeneration,
    _GepardState,
)
from vllm_omni.model_executor.models.gepard.nanocodec import (
    LOOKBACK_FRAMES,
    NanoCodec,
    apply_end_of_speech_tail,
)
from vllm_omni.model_executor.models.gepard.prompt import build_gepard_prompt_ids

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SAMPLES_PER_FRAME = 1024
NUM_HEADS = 32
SAMPLE_RATE = 22050
# Text-token count at or above which the layout is left unrepeated, read from
# the checkpoint defaults so the boundary cases follow the config.
APPLY_BELOW = GepardConfig().text_repetition_apply_below


class _ShiftInvariantCodec:
    """Frame tagged ``v`` -> samples ``v*SPF .. v*SPF+SPF-1``, wherever it sits."""

    def __init__(self) -> None:
        self.windows: list[int] = []  # frame count of every decode call

    def decode_from_codes(self, codes: torch.Tensor, codes_len: torch.Tensor):
        assert codes.shape[0] == 1, "streaming decode is per-request"
        assert codes.shape[1] == NUM_HEADS, f"expected {NUM_HEADS} code channels"
        tags = codes[0, 0, :].to(torch.int64)  # (T,)
        self.windows.append(int(tags.shape[0]))
        offsets = torch.arange(SAMPLES_PER_FRAME, dtype=torch.int64)
        audio = (tags.unsqueeze(1) * SAMPLES_PER_FRAME + offsets).reshape(1, -1)
        return audio.to(torch.float32), codes_len * SAMPLES_PER_FRAME


def _make_codec() -> tuple[NanoCodec, _ShiftInvariantCodec]:
    codec = NanoCodec(codec_id="test/fake-codec", sample_rate=SAMPLE_RATE)
    fake = _ShiftInvariantCodec()
    codec._codec = fake  # bypasses load(); NeMo is never imported
    assert codec.is_loaded
    return codec, fake


def _make_talker(codec: NanoCodec) -> GepardTalkerForConditionalGeneration:
    """A real talker with only the audio-path attributes populated.

    ``__init__`` needs a VllmConfig and builds the backbone; the methods under
    test touch only these four attributes.
    """
    talker = GepardTalkerForConditionalGeneration.__new__(GepardTalkerForConditionalGeneration)
    nn.Module.__init__(talker)  # sets up _parameters/_buffers/_modules only
    talker._codec = codec
    talker._audio_queue = []
    talker._active_states = {}
    talker._deferred_cleanup_ids = set()
    return talker


def _frame(tag: int) -> torch.Tensor:
    """One committed frame: 32 per-dimension codes, all carrying the tag."""
    return torch.full((NUM_HEADS,), tag, dtype=torch.long)


def _expected_samples(num_frames: int) -> torch.Tensor:
    offsets = torch.arange(SAMPLES_PER_FRAME, dtype=torch.int64)
    tags = torch.arange(num_frames, dtype=torch.int64).unsqueeze(1)
    return (tags * SAMPLES_PER_FRAME + offsets).reshape(-1).to(torch.float32)


def _drain(talker) -> torch.Tensor:
    """Concatenate and clear the queued deltas, as make_omni_output does."""
    if not talker._audio_queue:
        return torch.empty(0)
    out = torch.cat([a.reshape(-1) for _, a in talker._audio_queue])
    talker._audio_queue.clear()
    return out


def _generate(talker, state, num_frames: int, *, stop_at_end: bool) -> None:
    """Drive the commit loop the way forward() does, one frame per step."""
    for tag in range(num_frames):
        state.frames.append(_frame(tag))
        state.frame_count += 1
        state.past_first_step = True
        talker._emit_audio(state, is_final=stop_at_end and tag == num_frames - 1)


# Counts straddling both cadence thresholds, where an off-by-one hides.
@pytest.mark.parametrize(
    "num_frames",
    [
        1,
        2,
        FIRST_CHUNK_FRAMES - 1,
        FIRST_CHUNK_FRAMES,
        FIRST_CHUNK_FRAMES + 1,
        CHUNK_FRAMES - 1,
        CHUNK_FRAMES,
        CHUNK_FRAMES + 1,
        FIRST_CHUNK_FRAMES + CHUNK_FRAMES,
        79,
        200,
    ],
)
def test_chunked_emissions_equal_a_whole_history_decode(num_frames: int) -> None:
    codec, _fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")

    _generate(talker, state, num_frames, stop_at_end=True)

    torch.testing.assert_close(_drain(talker), _expected_samples(num_frames))
    assert state.emitted_frames == num_frames


def test_every_frame_is_emitted_exactly_once() -> None:
    """The trim must not drop or duplicate samples at a chunk boundary."""
    codec, _fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")
    num_frames = FIRST_CHUNK_FRAMES + 2 * CHUNK_FRAMES + 3

    _generate(talker, state, num_frames, stop_at_end=True)
    audio = _drain(talker)

    assert audio.numel() == num_frames * SAMPLES_PER_FRAME
    # Tags are globally unique, so a duplicated or dropped frame shows up here.
    assert len(torch.unique(audio)) == audio.numel()


def test_decode_windows_carry_lookback_context() -> None:
    """Every decode after the first must re-decode LOOKBACK_FRAMES of context."""
    codec, fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")
    num_frames = FIRST_CHUNK_FRAMES + 2 * CHUNK_FRAMES

    _generate(talker, state, num_frames, stop_at_end=True)

    assert len(fake.windows) >= 3, "expected a first chunk plus steady-state chunks"
    assert fake.windows[0] == FIRST_CHUNK_FRAMES, "the first chunk has no left context"
    for window in fake.windows[1:]:
        assert window == CHUNK_FRAMES + LOOKBACK_FRAMES


def test_codec_runs_once_per_chunk_not_once_per_frame() -> None:
    codec, fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")
    num_frames = FIRST_CHUNK_FRAMES + 3 * CHUNK_FRAMES

    _generate(talker, state, num_frames, stop_at_end=True)

    assert len(fake.windows) == 4
    assert sum(fake.windows) < num_frames * 2  # far below one decode per frame


def test_final_emit_drains_a_partial_chunk() -> None:
    """A request can end mid-chunk (max_tokens, STOP); the frames committed
    since the last cadence boundary must still reach the queue.

    Arithmetic only: this drives ``_emit_audio`` directly, so it says nothing
    about *whether* the engine reaches it on the request's last step. That
    wiring lives in ``test_gepard_wiring.py`` — a bug there leaves this green.
    """
    codec, _fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")
    num_frames = FIRST_CHUNK_FRAMES + CHUNK_FRAMES + 5

    _generate(talker, state, num_frames, stop_at_end=False)
    assert state.emitted_frames < num_frames, "a partial chunk should still be pending"

    talker._emit_audio(state, is_final=True)

    torch.testing.assert_close(_drain(talker), _expected_samples(num_frames))
    assert state.emitted_frames == num_frames


def test_emitting_with_nothing_pending_is_a_no_op() -> None:
    codec, fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")

    _generate(talker, state, 4, stop_at_end=True)
    calls_after_flush = len(fake.windows)
    talker._audio_queue.clear()

    talker._emit_audio(state, is_final=True)

    assert len(fake.windows) == calls_after_flush, "codec ran with no new frames"
    assert talker._audio_queue == []


def test_concurrent_requests_keep_separate_windows() -> None:
    codec, _fake = _make_codec()
    talker = _make_talker(codec)
    states = [_GepardState(request_id=f"req-{i}") for i in range(2)]

    for state in states:
        _generate(talker, state, FIRST_CHUNK_FRAMES, stop_at_end=True)

    by_req: dict[str, list[torch.Tensor]] = {}
    for req_id, audio in talker._audio_queue:
        by_req.setdefault(req_id, []).append(audio.reshape(-1))
    assert set(by_req) == {"req-0", "req-1"}
    for chunks in by_req.values():
        torch.testing.assert_close(torch.cat(chunks), _expected_samples(FIRST_CHUNK_FRAMES))


def test_end_of_speech_tail_fades_and_pads() -> None:
    audio = torch.ones(SAMPLE_RATE)  # 1 s of full-scale signal
    out = apply_end_of_speech_tail(audio, SAMPLE_RATE, fade_ms=10.0, silence_ms=20.0)

    fade = int(SAMPLE_RATE * 10.0 / 1000.0)
    pad = int(SAMPLE_RATE * 20.0 / 1000.0)
    assert out.numel() == SAMPLE_RATE + pad
    assert torch.all(out[-pad:] == 0.0), "trailing pad must be silent"
    assert out[SAMPLE_RATE - fade - 1] == 1.0, "the fade must not reach back past fade_ms"
    # Monotonically ramped to zero — the pad alone would leave the click.
    ramp = out[SAMPLE_RATE - fade : SAMPLE_RATE]
    assert torch.all(ramp[:-1] >= ramp[1:])
    assert ramp[-1] == 0.0


def test_end_of_speech_tail_is_a_no_op_by_default() -> None:
    audio = torch.ones(128)
    out = apply_end_of_speech_tail(audio, SAMPLE_RATE, fade_ms=0.0, silence_ms=0.0)
    assert out is audio


def _repeat_count(ids: list[int], cfg: GepardConfig) -> int:
    """How many ``[SOT ... EOT]`` copies the layout carries."""
    return ids.count(cfg.start_of_text)


def test_gepard_prompt_layout() -> None:
    """The prompt carries the speaker slots and exactly one SOS, at the end."""
    cfg = GepardConfig()
    ids = build_gepard_prompt_ids([1, 2, 3], config=cfg)

    slots = [cfg.speaker_token_base + i for i in range(cfg.num_speaker_prefix)]
    assert ids[: cfg.num_speaker_prefix] == slots
    assert ids.count(cfg.start_of_speech) == 1, "only the canonical copy may carry SOS"
    assert ids[-1] == cfg.start_of_speech, "the first frame is sampled from the SOS position"


def test_short_text_repeats_to_reach_the_target_token_mass() -> None:
    """A short text is repeated until its region holds ~target_text_tokens."""
    cfg = GepardConfig()
    text = [1, 2, 3, 4]  # target 16 / 4 -> 4 copies
    ids = build_gepard_prompt_ids(text, config=cfg)

    assert _repeat_count(ids, cfg) == 4
    assert ids.count(cfg.start_of_speech) == 1, "only the last copy opens the speech region"
    assert ids[-1] == cfg.start_of_speech


@pytest.mark.parametrize(
    ("num_text_tokens", "repeated"),
    [
        (APPLY_BELOW - 1, True),
        (APPLY_BELOW, False),
        (APPLY_BELOW + 1, False),
    ],
)
def test_apply_below_is_the_repetition_boundary(num_text_tokens: int, repeated: bool) -> None:
    """Both sides of the threshold: at apply_below the text is left alone.

    The layout has to match training here or WER collapses, and the boundary
    is where an off-by-one would be invisible in a single happy-path clip.
    """
    cfg = GepardConfig()
    ids = build_gepard_prompt_ids(list(range(1, num_text_tokens + 1)), config=cfg)

    assert (_repeat_count(ids, cfg) > 1) is repeated


def test_max_repeats_caps_a_one_token_text() -> None:
    """target/1 would ask for 16 copies; the cap keeps the prefill bounded."""
    cfg = GepardConfig()
    ids = build_gepard_prompt_ids([1], config=cfg)

    assert _repeat_count(ids, cfg) == cfg.text_repetition_max_repeats


def test_repetition_thresholds_are_read_from_the_config() -> None:
    """The numbers come from the checkpoint sidecar, not from literals.

    A checkpoint whose text_repetition block differs must change the layout;
    duplicating the values in the builder would silently ignore it.
    """
    cfg = GepardConfig(text_repetition={"enabled": True, "target_text_tokens": 9, "apply_below": 8, "max_repeats": 2})
    assert _repeat_count(build_gepard_prompt_ids([1, 2, 3], config=cfg), cfg) == 2  # ceil(9/3)=3, capped at 2
    assert _repeat_count(build_gepard_prompt_ids(list(range(8)), config=cfg), cfg) == 1  # at apply_below


def test_repetition_can_be_disabled() -> None:
    cfg = GepardConfig(text_repetition={"enabled": False})
    ids = build_gepard_prompt_ids([1], config=cfg)

    assert _repeat_count(ids, cfg) == 1
    assert ids[-1] == cfg.start_of_speech


def test_empty_text_is_rejected() -> None:
    """An empty text would still assemble a valid layout and be voiced.

    ``[slots, SOT, EOT, SOS]`` parses, prefills and generates: the failure is
    audible, not raised, which is the worst shape for a caller to debug.
    """
    with pytest.raises(ValueError, match="at least one text token"):
        build_gepard_prompt_ids([], config=GepardConfig())


def test_packaged_deploy_config_is_where_the_example_looks_for_it() -> None:
    """The example resolves its default off ``vllm_omni.__file__``; tests reach
    the same YAML through ``tests.helpers.stage_config``, so nothing covered the
    packaged location until this.

    ``pipeline: gepard`` is the load-bearing key: the checkpoint self-identifies
    as qwen3_5_text, so without the pin the architectures fallback routes it to
    the diffusion registry.

    This asserts the location, not the example's constant — importing the
    example would pull ``Omni`` and NeMo into a CPU test. Running the quick
    start verbatim stays a release check.
    """
    path = Path(vllm_omni.__file__).resolve().parent / "deploy" / "gepard.yaml"
    assert path.is_file(), f"packaged deploy config missing at {path}"

    deploy = load_deploy_config(path)
    assert deploy.pipeline == "gepard"
    assert deploy.stages, "deploy config declares no stages"
