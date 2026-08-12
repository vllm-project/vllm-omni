# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for the Gepard-1.0 single-stage native-AR TTS.

Zero-shot only (the model's default learned voice). These tests need a GPU and
the NeMo NanoCodec, and are spread across CI tiers by level marker -- see the
comment above ``pytestmark``. The window arithmetic behind the streaming decode
is covered on CPU in ``tests/model_executor/models/test_gepard_window.py``.
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path
from typing import NamedTuple

import pytest
import soundfile as sf
import torch
from transformers import AutoTokenizer

from tests.helpers.mark import hardware_test
from tests.helpers.media import convert_audio_file_to_text
from tests.helpers.runtime import OmniRunner  # noqa: F401  (indirect fixture)
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni import Omni
from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig
from vllm_omni.model_executor.models.gepard.prompt import build_gepard_prompt_ids

# The codec decoder is an optional dependency; skip rather than fail the
# engine inside the worker process when it is absent. The CI image does not
# carry it today, so this skip is why the nightly and weekly TTS sweeps
# report nothing for this model, and why test-merge.yml has no per-file step
# for it: pytest exits 5 when a file-scoped run collects nothing, which would
# turn the merge gate red rather than pass it quietly. The install recipe is
# in examples/offline_inference/text_to_speech/README.md.
pytest.importorskip("nemo.collections.tts.models")

MODEL_NAME = "nineninesix/gepard-1.0"
STAGE_CONFIG = get_deploy_config_path("gepard.yaml")
SAMPLE_RATE = 22050
# NanoCodec upsamples one FSQ frame to exactly this many samples; the advertised
# 21.5 fps is 22050 / 1024 rounded.
SAMPLES_PER_FRAME = 1024


@functools.cache
def _config() -> GepardConfig:
    """The checkpoint's own layout, not the class defaults.

    The worker builds its config from the ``gepard_config.json`` sidecar, so a
    prompt assembled from the defaults would silently use a different layout
    the day the sidecar moves — and a mismatched layout degrades the speech
    rather than failing, which is the hardest way for this to be found.
    """
    return GepardConfig.from_checkpoint(MODEL_NAME)


# Prompt -> a word that appears in no other prompt. Concurrency is only
# testable if a clip can be traced back to the request that asked for it.
#
# These are not interchangeable, and swapping one is not free. Batched decoding
# perturbs every request's sampling trajectory — bf16 matmuls are not row-wise
# reproducible, and a request's codes here diverge from its solo run by frame 8,
# long before any neighbour finishes — and on some trajectories this
# checkpoint's stop head never fires, so the request runs to the token budget.
# "The weather is sunny today." sat in the first slot until it was measured at
# 0/10 runaways alone and 8/10 in this batch. Its replacement was screened over
# 10 batched runs with no runaway on any row. Re-screen the whole set, not just
# the row being changed: one different prompt changes all four trajectories.
_DISTINGUISHABLE_PROMPTS = {
    "He drinks coffee every morning.": "coffee",
    "Machine learning is interesting.": "learning",
    "Please close the window before leaving.": "window",
    "My favorite color is purple.": "purple",
}

# Longer than text_repetition.apply_below tokens, so the layout is voiced once
# instead of repeated. Every other prompt here is short enough to be repeated,
# so without this row the repeats == 1 branch never runs outside the CPU tests.
_LONG_TEXT = (
    "The morning train was late again, so I walked the last two miles along the "
    "river and reached the office just before nine."
)

_OMNI_RUNNER_PARAM = (
    MODEL_NAME,
    STAGE_CONFIG,
    {"trust_remote_code": True},
)

# These tests only ask whether a common word is present, which needs far less
# ASR than the shared helper's ``small`` default. A miss re-runs that one clip
# through the stronger model -- the escalation the shared speech assertion uses
# -- so a weak-ASR mishear still cannot flake the gate while the average clip
# pays for the fast one. Whisper pads every clip to 30 s whatever its length, so
# the model is the only knob that moves the cost here.
_ASR_MODEL = "base"
_ASR_ESCALATION_MODEL = "small"

# Transcription dominates this file's runtime on a single-GPU runner, where
# Whisper falls back to CPU. Two of the four concurrent clips carry the content
# check; the pairwise waveform comparison covers all four for the cheaper
# failure -- one request's audio delivered for another.
_TRANSCRIBED_CLIPS = 2

# No level marker at module scope: each test below carries the one that matches
# how often its claim needs re-checking, so the tiers do not all pay for all of
# it. L3 (advanced_model, every merge) gets the one canonical path; L4
# (full_model, nightly) gets the layout variants; L5 (slow, weekly) gets the
# concurrency batch, the most expensive test here and the one covering code that
# changes least. There is deliberately no L2 (core_model) row: under
# ``load_format: dummy`` the stop head is random, so every request runs to
# gepard.yaml's ``max_tokens`` -- 1000 frames, ~46 s of audio per request
# through NanoCodec with ``enforce_eager``, far past what the L2 budget is for.
pytestmark = [
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


def _build_request(text: str) -> dict:
    """Assemble the [speaker slots, SOT, text, EOT, SOS] prompt."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompt_token_ids = build_gepard_prompt_ids(
        tokenizer(text, add_special_tokens=False)["input_ids"],
        config=_config(),
    )
    return {
        "prompt_token_ids": prompt_token_ids,
        "additional_information": {"text": [text]},
    }


def _extract_audio(mm) -> torch.Tensor | None:
    # make_omni_output queues the waveform under "model_outputs"; the
    # consolidation path renames it to "audio". Explicit None checks — a
    # multi-element tensor has no truth value.
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if isinstance(audio, list):
        audio = audio[0] if len(audio) else None
    return audio


class _Clip(NamedTuple):
    wav: torch.Tensor
    # Output tokens that are not the STOP sentinel. compute_logits emits exactly
    # one token per sampled step — the committed frame's head0, which lives in
    # [0, stop_token), or STOP — so this counts the frames the codec was handed
    # using only what the engine returned, never the model's internals.
    committed_frames: int


def _synthesize_all(omni: Omni, texts: list[str]) -> list[_Clip]:
    """Submit every text in one call and return one clip per request, in
    submission order — ``clips[i]`` is what the engine produced for ``texts[i]``.

    The engine does not deliver them that way. ``Omni.generate`` yields each
    request as it finishes and never sorts, unlike vLLM's ``LLM.generate``, so
    they arrive in COMPLETION order. Restoring the pairing here is what lets a
    caller assert that a clip says what *its own* prompt asked for; without it
    the only testable claim is that the batch as a whole looks plausible, and a
    stable swap between two requests satisfies that.

    The pairing comes from the request id, which ``Omni.generate`` builds as
    ``f"{i}_{uuid4()}"`` over the submitted prompts (``entrypoints/omni.py``).

    Nothing here authors a SamplingParams: ``OmniRunner.generate`` forwards the
    engine's own stage defaults. A freshly built one would replace those
    defaults rather than merge over them, dropping the pipeline's
    stop_token_ids and leaving every request to run to ``max_tokens``.
    """
    outputs = omni.generate([_build_request(t) for t in texts])
    assert len(outputs) == len(texts), f"expected {len(texts)} outputs, got {len(outputs)}"

    by_index: dict[int, _Clip] = {}
    for stage_outputs in outputs:
        index = int(str(stage_outputs.request_id).split("_", 1)[0])
        # OmniRequestOutput.request_output is a single RequestOutput, not a list.
        req_output = stage_outputs.request_output
        assert req_output is not None, "request produced no output"
        waveform = None
        frames = 0
        for out in req_output.outputs:
            # Explicit None check: token_ids is not necessarily a plain list,
            # and `x or []` would ask an array for a truth value.
            token_ids = out.token_ids if out.token_ids is not None else []
            # The sentinel compute_logits emits instead of a head0 to end a
            # request. It sits just past head0's range, so a token equal to it
            # is never a committed frame.
            frames += sum(1 for t in token_ids if int(t) != _config().stop_token)
            mm = out.multimodal_output
            if mm is None:
                continue
            audio = _extract_audio(mm)
            if audio is None:
                continue
            sr_t = mm.get("sr")
            if hasattr(sr_t, "item"):
                assert int(sr_t.item()) == SAMPLE_RATE
            waveform = audio.cpu().float()
        assert waveform is not None, "no audio output produced"
        by_index[index] = _Clip(wav=waveform, committed_frames=frames)

    assert sorted(by_index) == list(range(len(texts))), (
        f"request ids did not cover every submitted prompt exactly once: got {sorted(by_index)}"
    )
    return [by_index[i] for i in range(len(texts))]


def _synthesize(omni: Omni, text: str) -> _Clip:
    return _synthesize_all(omni, [text])[0]


def _assert_clip_is_sane(clip: _Clip) -> None:
    """The checks every clip has to pass, whatever produced it."""
    wav = clip.wav
    assert wav.dim() == 1, f"expected mono 1-D, got shape {tuple(wav.shape)}"
    assert wav.numel() > 0, "empty waveform"
    assert torch.isfinite(wav).all(), "non-finite samples"

    # Frame conservation, reconciled against the engine's own token count rather
    # than merely checked for divisibility. This is the assertion that catches a
    # payload carrying non-audio data (per-step hidden states folded in under the
    # same output key) or a dropped tail: both leave the sample count and the
    # frame count disagreeing, and both pass every other check here.
    assert wav.numel() == clip.committed_frames * SAMPLES_PER_FRAME, (
        f"{wav.numel()} samples for {clip.committed_frames} committed frames — "
        f"expected exactly {clip.committed_frames * SAMPLES_PER_FRAME}"
    )

    # Decoded speech sits inside the waveform range; anything far outside it is
    # not codec output.
    assert float(wav.abs().max()) <= 1.5, f"peak {float(wav.abs().max()):.2f} is out of range"


def _assert_stopped_on_its_own(clip: _Clip) -> None:
    """The request must stop on its own rather than run to ``max_tokens``.

    Separate from the structural checks because it reads the stop head's
    judgement, not the payload's shape: under ``load_format: dummy`` that head
    is random and never fires, so every clip runs to the token cap.
    """
    seconds = clip.wav.numel() / SAMPLE_RATE
    assert 0.5 < seconds < 30.0, f"implausible duration {seconds:.2f}s — did STOP fire?"


def _transcribe(wav: torch.Tensor, tmp_dir: str, name: str, model_size: str = _ASR_MODEL) -> str:
    path = str(Path(tmp_dir) / f"{name}.wav")
    sf.write(path, wav.numpy(), SAMPLE_RATE)
    return convert_audio_file_to_text(path, model_size=model_size, language="en").lower()


def _transcribe_for(wav: torch.Tensor, tmp_dir: str, name: str, accepts) -> str:
    """Transcribe with the fast model, retrying once with the stronger one.

    ``accepts`` decides whether the fast transcript is usable. Escalating only
    on a miss keeps the average clip at ``_ASR_MODEL`` while a genuine model
    artifact still fails the caller's assertion, since the stronger ASR
    mistranscribes it too.
    """
    transcript = _transcribe(wav, tmp_dir, name)
    if accepts(transcript):
        return transcript
    return _transcribe(wav, tmp_dir, name, model_size=_ASR_ESCALATION_MODEL)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize(
    ("text", "keyword"),
    [
        # The canonical path, and the only row on the merge gate: one request,
        # one transcript, the cheapest check that reads the audio as speech.
        pytest.param(
            "Hello, this is Gepard speaking.",
            "hello",
            marks=pytest.mark.advanced_model,
            id="canonical",
        ),
        # No keyword: this row asserts the same thing about the audio as the one
        # above it, so it carries the structural and stopping checks over a
        # second text and leaves the transcript to the row that already makes
        # that claim. A second text is worth a nightly run, not a merge one.
        pytest.param(
            "The quick brown fox jumps over the lazy dog.",
            None,
            marks=pytest.mark.full_model,
            id="second_text",
        ),
    ],
)
def test_gepard_offline_zero_shot(omni_runner, run_level: str, text: str, keyword: str | None) -> None:
    """Zero-shot synthesis produces a finite mono waveform at 22.05 kHz.

    The transcript check is the one assertion here that reads the audio as
    speech rather than as a tensor. It stands in for a parity comparison, which
    needs a reference implementation this PR cannot run — noise, a repeated
    loop or the wrong clip all pass every numeric check above it.
    """
    clip = _synthesize(omni_runner, text)
    _assert_clip_is_sane(clip)

    if run_level not in {"advanced_model", "full_model"}:
        return  # dummy weights: the audio is structural only, so do not read it

    _assert_stopped_on_its_own(clip)
    if keyword is None:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        transcript = _transcribe_for(clip.wav, tmp_dir, "zero_shot", lambda t: keyword in t)
    assert keyword in transcript, f"expected {keyword!r} in transcript, got {transcript!r}"


@pytest.mark.full_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_gepard_offline_long_text_skips_repetition(omni_runner, run_level: str) -> None:
    """The other branch of the prompt layout, end to end.

    A text at or above ``apply_below`` tokens is voiced once; shorter ones are
    repeated to carry enough mass against the speaker prefix. The layout
    assertion keeps this row honest — if a tokenizer change pushed this text
    back under the threshold it would silently retest the repeated path.
    """
    prompt_ids = _build_request(_LONG_TEXT)["prompt_token_ids"]
    assert prompt_ids.count(_config().start_of_text) == 1, "_LONG_TEXT is no longer above apply_below"

    clip = _synthesize(omni_runner, _LONG_TEXT)
    _assert_clip_is_sane(clip)

    if run_level not in {"advanced_model", "full_model"}:
        return

    _assert_stopped_on_its_own(clip)
    with tempfile.TemporaryDirectory() as tmp_dir:
        transcript = _transcribe_for(clip.wav, tmp_dir, "long_text", lambda t: "river" in t)
    assert "river" in transcript, f"expected 'river' in transcript, got {transcript!r}"


@pytest.mark.slow
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_gepard_offline_concurrent_requests_stay_isolated(omni_runner, run_level: str) -> None:
    """Four requests in one batch, each holding its own generation state.

    The prompts are deliberately unlike each other: cross-talk between the
    per-request frame history, the codec window or the output routing shows up
    as a transcript matching the wrong prompt, which similar prompts would
    hide. Reading that requires knowing which prompt each clip belongs to, so
    ``_synthesize_all`` restores submission order from the request ids first.
    ``max_num_seqs`` is 4 in the deploy config, so these run together rather
    than back to back. All four are submitted for that reason even though only
    ``_TRANSCRIBED_CLIPS`` of them are read back as speech.
    """
    texts = list(_DISTINGUISHABLE_PROMPTS)
    clips = _synthesize_all(omni_runner, texts)

    for clip in clips:
        _assert_clip_is_sane(clip)

    if run_level not in {"advanced_model", "full_model"}:
        return  # dummy weights: the audio is structural only, so do not read it

    for clip in clips:
        _assert_stopped_on_its_own(clip)

    # Covers all four clips, not just the transcribed ones: delivering one
    # request's audio for another leaves two byte-identical waveforms, and this
    # costs nothing next to an ASR pass.
    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            assert not torch.equal(clips[i].wav, clips[j].wav), (
                f"clips {i} and {j} are the same waveform — one request's audio was delivered for another"
            )

    # Each clip must say what ITS OWN prompt asked for. Asserting only that
    # every clip says one of the four would pass a stable permutation, which is
    # exactly what a per-request state or routing mix-up produces.
    keywords = set(_DISTINGUISHABLE_PROMPTS.values())
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, clip in enumerate(clips[:_TRANSCRIBED_CLIPS]):
            expected = _DISTINGUISHABLE_PROMPTS[texts[i]]
            # Escalate on the expected word being absent, not on "no keyword at
            # all": a clip carrying a neighbour's word is precisely the failure
            # under test, and must not be waved through by the fast ASR.
            transcript = _transcribe_for(clip.wav, tmp_dir, f"concurrent_{i}", lambda t: expected in t)
            present = sorted(k for k in keywords if k in transcript)
            assert present == [expected], (
                f"clip for {texts[i]!r} transcribed as {transcript!r}: expected {expected!r} and "
                f"no other prompt's keyword, found {present} — requests bled into each other"
            )
