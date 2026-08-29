# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference test for MOSS-TTS-v1.5 (MossTTSDelay-8B).

MOSS-TTS-v1.5 is a continued-training upgrade of MOSS-TTS 1.0 with the SAME
``MossTTSDelay`` architecture and API, so it reuses the existing talker/codec
path unchanged — this test just pins that the 8B v1.5 checkpoint loads and
produces audio.

The model is 8B, so this lives in its own file (one module-scoped engine per
file, per skill invariant I4) and is gated behind an 80 GB GPU; the small-GPU
MossTTSDelay coverage stays in ``test_moss_tts.py`` (1.7B VoiceGenerator).
"""

from __future__ import annotations

import os
import urllib.request

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.audio_output import collect_audio_from_outputs
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni import Omni

SAMPLE_RATE = 24_000
REF_AUDIO_URL = "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/assets/audio/reference_zh_1.wav"
MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"

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


def _get_test_config() -> str:
    """Derive a single-GPU, CI-friendly config from moss_tts.yaml.

    MOSS-TTS-v1.5 shares the 8B MossTTSDelay architecture (n_vq=32) and reuses
    moss_tts.yaml.  That base config targets a dedicated card (Stage 0
    gpu_memory_utilization 0.85); on a single shared GPU that lets the talker
    claim almost the whole card and Stage 1 (codec) OOMs at load — the two-stage
    single-GPU OOM fixed for the other variants in PR #4100 (issue #4087).  Cap
    Stage 0 at 0.45 so both stages (talker + codec, both pinned to device "0")
    fit one GPU.  Prefix caching stays disabled (as in the base YAML).
    """
    return modify_stage_config(
        get_deploy_config_path("moss_tts.yaml"),
        updates={
            "stages": {
                0: {
                    "gpu_memory_utilization": 0.45,
                    "enable_prefix_caching": False,
                },
            },
        },
    )


# H100-gated (8B). Picked up by the nightly "TTS · Function Test with H100"
# job (-m "full_model and H100 and tts"); the L4 MOSS-TTS tests land in the
# sibling L4 tts lane instead.
pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
    pytest.mark.parametrize(
        "omni_runner",
        [(MODEL, _get_test_config(), {"stage_init_timeout": 600, "trust_remote_code": True})],
        indirect=True,
    ),
]


@pytest.fixture(scope="session")
def ref_audio_path(tmp_path_factory) -> str:
    """Download the upstream reference clip once per session.

    Set ``MOSS_TTS_SKIP_ON_NET_FAIL=1`` to skip in air-gapped environments.
    """
    target = tmp_path_factory.mktemp("moss_tts_v15_ref") / "zh_1.wav"
    try:
        with urllib.request.urlopen(REF_AUDIO_URL, timeout=30) as resp:
            target.write_bytes(resp.read())
    except Exception as exc:  # noqa: BLE001
        msg = f"Cannot fetch reference clip {REF_AUDIO_URL}: {exc}"
        if os.environ.get("MOSS_TTS_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)
    if not target.exists() or target.stat().st_size == 0:
        pytest.fail(f"Reference clip empty after download: {target}")
    return str(target)


@pytest.fixture(scope="module")
def v15_processor():
    """Upstream MossTTSProcessor used to build voice-clone prompts.

    The delay talker conditions on ``prompt_token_ids`` (tokenised text) and
    ``codes.ref`` (the unified audio-code grid) — a bare assistant prompt with
    the text/ref stuffed into ``additional_information`` is never tokenised
    offline, so the talker would ignore it and emit no conditioned audio. Mirror
    ``examples/offline_inference/text_to_speech/moss_tts/end2end.py``: run the
    processor (which also carries the CPU audio tokenizer used to encode the
    reference clip).
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)


def _load_ref_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load and resample the reference clip to (1, T) float32 at ``target_sr``."""
    import soundfile as sf

    data, sr = sf.read(path, always_2d=True)  # (T, C)
    wav = torch.from_numpy(data.T.astype("float32"))  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav  # (1, T)


def _build_request(processor, text: str, ref_wav: torch.Tensor, seed: int = 42) -> dict:
    """Build a MOSS-TTS-v1.5 voice_clone request the way ``serving_speech`` /
    ``end2end.py`` do.

    Encode the reference clip to ``reference`` codes, then build the unified
    ``(L, 1+n_vq)`` grid via the processor: column 0 is the text/special-token
    stream forwarded as ``prompt_token_ids``; columns ``1..n_vq`` are the
    delay-pattern audio-code grid forwarded as ``codes.ref``.
    """
    wav_2d = ref_wav if ref_wav.dim() == 2 else ref_wav.unsqueeze(0)
    codes_list = processor.encode_audios_from_wav([wav_2d], sampling_rate=24000, n_vq=processor.model_config.n_vq)
    user_msg = processor.build_user_message(text=text, reference=[codes_list[0]])
    batch = processor(conversations=[[user_msg]], mode="generation")
    unified = batch["input_ids"][0]  # (L, 1 + n_vq)
    return {
        "prompt_token_ids": unified[:, 0].tolist(),
        "additional_information": {
            "codes": {"ref": unified[:, 1:].contiguous().to(torch.int64)},
            "seed": [seed],
        },
    }


def _collect_audio(omni: Omni, request: dict) -> tuple[torch.Tensor, int]:
    """Run one request; return (waveform_cpu, sample_rate).

    Sampling params are PER STAGE — MOSS-TTS-v1.5 is a two-stage pipeline
    (talker + codec), so pair the stochastic talker params with a greedy codec
    decode via ``_per_stage_sampling``.
    """
    sampling = _per_stage_sampling(omni.num_stages)
    return collect_audio_from_outputs(omni.generate(request, sampling), SAMPLE_RATE)


@hardware_test(res={"cuda": "H100"})
def test_moss_tts_v15_voice_clone(omni_runner: OmniRunner, v15_processor, ref_audio_path: str) -> None:
    """MOSS-TTS-v1.5 (8B): voice_clone produces non-empty 24 kHz audio."""
    ref_wav = _load_ref_audio(ref_audio_path)
    req = _build_request(v15_processor, "Hello, this is a MOSS-TTS v1.5 voice cloning test.", ref_wav)
    audio, sr = _collect_audio(omni_runner.omni, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"
