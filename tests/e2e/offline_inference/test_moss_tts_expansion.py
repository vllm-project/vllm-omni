# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for the MOSS-TTS two-stage pipeline (delay model).

Tests the MossTTSDelayModel path using MOSS-VoiceGenerator (1.7B) — the
smallest MossTTSDelayModel variant — so the test runs on a single A10G or L4
without requiring an 80 GB GPU.

Uses the standard omni_runner + pytestmark pattern (one module-scoped engine
per file, skill invariant I4) — the same harness as the sibling realtime file
test_moss_tts_realtime.py.  Do NOT add a second engine fixture here; that
triggers mid-module teardown races.
"""

from __future__ import annotations

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.audio_output import audio_from_mm, collect_audio_from_outputs
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni import Omni

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# MOSS-VoiceGenerator (MossTTSDelayModel, 1.7B) — smallest delay variant, fits an L4.
MODEL = "OpenMOSS-Team/MOSS-VoiceGenerator"

SAMPLE_RATE = 24_000  # All MOSS-TTS full variants output 24 kHz

# Voice description for the MOSS-VoiceGenerator (``voice_generator`` variant):
# it conditions on text (what to say) + instruction (how the voice sounds), not
# a reference clip.
_DELAY_INSTRUCTION = "A clear, natural speaking voice."

# Stage 0 (talker) samples audio tokens; the upstream recipe is stochastic.
_STAGE0_PARAMS = SamplingParams(
    temperature=1.7,
    top_p=0.8,
    top_k=25,
    max_tokens=512,  # short for tests (~20 s at 24 kHz, ~12.5 tokens/s)
    seed=42,
    detokenize=False,
)
# Stage 1 (codec) decodes those tokens to a waveform — a deterministic decode,
# not a sampling step. Give it greedy params instead of replicating the talker's
# (which would add sampling noise and cap the decode at 512 tokens); mirrors
# ``test_moss_tts_realtime.py``.
_STAGE1_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=65536,
    seed=42,
    detokenize=False,
)


def _per_stage_sampling(num_stages: int) -> list[SamplingParams]:
    """Per-stage sampling list: talker params for stage 0, greedy for the rest."""
    return [_STAGE0_PARAMS] + [_STAGE1_PARAMS] * max(0, num_stages - 1)


# ---------------------------------------------------------------------------
# Deploy config
# ---------------------------------------------------------------------------


def _get_test_config() -> str:
    """Derive a CI-friendly config from moss_voice_generator.yaml.

    Reduces Stage 0 gpu_memory_utilization from 0.60 → 0.45 to leave headroom
    for Stage 1 (0.12) on a shared L4/A10G.  Prefix caching stays disabled (as
    in the base YAML) so two identical-seed runs are bit-reproducible — the
    cached-prefill path can perturb logits enough to shift sampling/EOS, which
    ``test_moss_tts_delay_deterministic`` requires.  ``max_num_seqs`` is left at
    the YAML default so ``test_moss_tts_delay_batch`` still schedules both
    requests in one batch.
    """
    return modify_stage_config(
        get_deploy_config_path("moss_voice_generator.yaml"),
        updates={
            "stages": {
                0: {
                    "gpu_memory_utilization": 0.45,
                    "enable_prefix_caching": False,
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# pytestmark — one engine for the whole module
# ---------------------------------------------------------------------------

# Selected by the nightly "TTS · Function Test" (-m "full_model and L4 and tts").
# Without the tts + run-level marks these tests are collected but never run.
pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
    pytest.mark.parametrize(
        "omni_runner",
        [(MODEL, _get_test_config(), {"trust_remote_code": True})],
        indirect=True,
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_audio(omni: Omni, request: dict, sampling: list | None = None) -> tuple[torch.Tensor, int]:
    """Run one request; return (waveform_cpu, sample_rate).

    ``sampling`` is the PER-STAGE sampling-params list (its length must equal
    num_stages), not per request. MOSS-TTS is a two-stage pipeline (talker +
    codec), so a single SamplingParams is rejected — when omitted,
    ``_per_stage_sampling`` supplies stochastic talker params plus a greedy
    codec decode.
    """
    if sampling is None:
        sampling = _per_stage_sampling(omni.num_stages)
    return collect_audio_from_outputs(omni.generate(request, sampling), SAMPLE_RATE)


# ---------------------------------------------------------------------------
# MossTTSDelayModel — MOSS-VoiceGenerator 1.7B
# (tests the delay-pattern generation path on a small GPU)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def delay_processor():
    """Upstream MossTTSProcessor used to build conditioned prompts.

    The delay talker conditions on ``prompt_token_ids`` (tokenised text) and
    ``codes.ref`` (the unified audio-code grid) — NOT a bare assistant prompt
    with the text stuffed in ``additional_information`` (nothing tokenises that
    offline, so the talker would ignore it and every request would produce the
    same audio). Mirror the online speech server: run the processor.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)


def _build_voicegen_request(processor, text: str, instruction: str = _DELAY_INSTRUCTION, seed: int = 42) -> dict:
    """Build a MOSS-VoiceGenerator request the way ``serving_speech`` does.

    ``build_user_message`` + ``processor(..., mode="generation")`` yield a
    unified ``(L, 1+n_vq)`` grid; column 0 is the text/special token stream and
    columns ``1..n_vq`` are the delay-pattern audio-code grid. We forward column
    0 as ``prompt_token_ids`` and the rest as ``codes.ref`` — so different text
    produces different prompts (hence different audio), which is what makes the
    batch regression test a genuine guard rather than an RNG-noise artifact.
    """
    msg = processor.build_user_message(text=text, instruction=instruction)
    batch = processor(conversations=[[msg]], mode="generation")
    unified = batch["input_ids"][0]  # (L, 1 + n_vq)
    return {
        "prompt_token_ids": unified[:, 0].tolist(),
        "additional_information": {
            "codes": {"ref": unified[:, 1:].contiguous().to(torch.int64)},
            "seed": [seed],
        },
    }


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_english(omni_runner: OmniRunner, delay_processor) -> None:
    """MossTTSDelayModel: English text produces non-empty 24 kHz audio."""
    req = _build_voicegen_request(delay_processor, "Hello, this is a MOSS-TTS generation test.")
    audio, sr = _collect_audio(omni_runner.omni, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_chinese(omni_runner: OmniRunner, delay_processor) -> None:
    """MossTTSDelayModel: Chinese input produces non-empty audio."""
    req = _build_voicegen_request(delay_processor, "你好，这是语音合成测试。")
    audio, sr = _collect_audio(omni_runner.omni, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_deterministic(omni_runner: OmniRunner, delay_processor) -> None:
    """MossTTSDelayModel: same seed yields identical waveforms.

    Audio codes are sampled with ``torch.multinomial`` outside vLLM's seeded
    sampler; the talker re-seeds a per-(seed, frame) generator from the request
    seed so two same-seed runs reproduce exactly (without this they diverge).
    """
    # Build a FRESH request per run: the engine consumes ``codes.ref`` /
    # ``audio_state`` from ``additional_information`` in place, so reusing one
    # dict would let the first run's mutations leak into the second.
    audio1, _ = _collect_audio(
        omni_runner.omni, _build_voicegen_request(delay_processor, "Reproducibility check.", seed=99)
    )
    audio2, _ = _collect_audio(
        omni_runner.omni, _build_voicegen_request(delay_processor, "Reproducibility check.", seed=99)
    )

    assert audio1.shape == audio2.shape, "Shapes differ across identical seeds"
    assert torch.allclose(audio1, audio2, atol=1e-4), "Waveforms differ across identical seeds"


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_batch(omni_runner: OmniRunner, delay_processor) -> None:
    """MossTTSDelayModel: two concurrent requests with different text return
    non-empty AND distinct audio.

    This is the regression guard for the batch concurrency bugs: when audio
    codes are broadcast (``audio_codes_list[0]`` for every request) or hidden
    states are sliced with a uniform ``rows_per_req`` division, request 1
    receives request 0's audio. Asserting the waveforms differ catches that.

    The requests must be built through the processor (``_build_voicegen_request``)
    so the differing text actually reaches the talker as ``prompt_token_ids`` —
    a bare assistant prompt would leave every request identical and the bug
    invisible. Audio sampling is seeded (deterministic), so the distinctness
    here reflects genuine per-request content, not RNG noise.

    Coverage limitation — this does NOT exercise the mixed prefill+decode case.
    Both requests are submitted together via ``generate(requests, ...)``, so the
    scheduler prefills then decodes them in lockstep: every step is either
    all-prefill or all-decode, and each request contributes the same number of
    rows. The uniform ``rows_per_req`` fallback in ``_request_row_spans`` happens
    to be exact there, so it would still pass even if ``query_start_loc`` recovery
    were broken. The genuinely dangerous case — request A in decode (1 row) while
    request B prefills (N rows) in the SAME batch, where a uniform split
    misattributes A's hidden states — requires staggered submission (start A,
    let it reach decode, then submit B). The synchronous ``Omni.generate()`` batch
    API cannot inject a request into an in-flight batch, so that path is covered
    only indirectly by the ``query_start_loc`` recovery in ``_request_row_spans``
    and its unit-level reasoning, not by this end-to-end test.
    """
    requests = [
        _build_voicegen_request(delay_processor, "First sentence."),
        _build_voicegen_request(
            delay_processor, "A completely different second utterance, much longer than the first."
        ),
    ]
    # generate() returns a flat list of OmniRequestOutput, one or more per
    # request as audio streams. Each carries .request_id ("<idx>_<uuid>") and
    # exposes .multimodal_output directly. Group audio chunks per request index
    # so a per-request mix-up (the broadcast / row-misalignment bugs) is
    # observable as identical waveforms across the two requests.
    # sampling_params_list is PER STAGE (length == num_stages), independent of
    # the number of requests; the two prompts above are the per-request batch.
    sampling = _per_stage_sampling(omni_runner.omni.num_stages)
    chunks_by_req: dict[int, list[torch.Tensor]] = {}
    for omni_out in omni_runner.omni.generate(requests, sampling):
        idx = int(omni_out.request_id.split("_", 1)[0])
        chunk = audio_from_mm(omni_out.multimodal_output)
        if chunk is not None and chunk.numel() > 0:
            chunks_by_req.setdefault(idx, []).append(chunk)

    assert set(chunks_by_req) == {0, 1}, f"Expected audio for requests 0 and 1, got {sorted(chunks_by_req)}"
    results = [torch.cat(chunks_by_req[i], dim=0) for i in (0, 1)]
    for i, audio in enumerate(results):
        assert audio.numel() > 0, f"Audio[{i}] is empty"

    # Different text must not yield identical audio. Identical output here is
    # the signature of the broadcast / row-misalignment concurrency bugs.
    a, b = results
    if a.numel() == b.numel():
        assert not torch.equal(a, b), "Both requests returned identical audio (batch concurrency bug)"
