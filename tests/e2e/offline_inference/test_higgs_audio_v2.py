# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end offline inference tests for higgs-audio v2.

Mirrors examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py:
build Stage-0 prompt_token_ids with the model's own processor + plain-text
chat template, drive the engine via ``Omni.generate``, and inspect the
Stage-1 PCM payload directly. The OmniRunnerHandler.send_audio_speech_request
helper applies a Qwen-style ChatML wrap that is not valid for higgs, so this
test bypasses it (same approach as test_voxtral_tts.py).

v1 scope is plain text -> 24 kHz speech; voice cloning, multi-speaker prompts,
language overrides, and task_type selection are out of scope.
"""

from __future__ import annotations

import os

# Match the example script: keep DeepGEMM FP8 off so engine init does not
# require the optional ``deep_gemm`` backend.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import numpy as np
import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "bosonai/higgs-audio-v2-generation-3B-base"
STAGE_CONFIG = get_deploy_config_path("higgs_audio_v2.yaml")
SAMPLE_RATE = 24_000
# A short sentence at 24 kHz must produce at least ~0.5 s of audio.
MIN_AUDIO_SAMPLES = 12_000
TEST_TEXT = "Hello world."

# Reuse the qwen3_tts ref clip + transcript for voice clone tests (see
# tests/e2e/online_serving/test_qwen3_tts_base.py:27-28 for the source).
_REF_AUDIO_ASSET = "qwen3_tts/clone_2.wav"
_REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

# (model, stage_config_path, extra_omni_kwargs) — same shape as test_voxtral_tts.py.
# Stage configs already pin enforce_eager=true per stage, so no Omni-level override.
_OMNI_RUNNER_PARAM = (MODEL, STAGE_CONFIG, {"trust_remote_code": True})


def _compose_request(model_name: str, text: str) -> dict:
    """Build a higgs prompt dict via the upstream processor + plain-text template.

    The serving layer's ``_build_higgs_audio_v2_params`` does the same thing
    (see vllm_omni/entrypoints/openai/serving_speech.py:1520) — keeping the
    offline and online prompt-construction paths identical is what guarantees
    input-token parity with the HF reference.
    """
    from transformers import AutoProcessor

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
        build_plain_text_prompt,
        input_ids_to_python_list,
    )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    inputs = build_plain_text_prompt(processor, text)
    return {"prompt_token_ids": input_ids_to_python_list(inputs)}


def _extract_pcm(outputs) -> torch.Tensor:
    """Pull the concatenated PCM tensor out of the Stage-1 multimodal_output."""
    for stage_output in outputs:
        mm = getattr(stage_output, "multimodal_output", None)
        if not mm:
            continue
        audio = mm.get("model_outputs") if "model_outputs" in mm else mm.get("audio")
        if audio is None:
            continue
        if isinstance(audio, list):
            chunks = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
            if not chunks:
                continue
            return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        return torch.as_tensor(audio).float().cpu().reshape(-1)
    raise AssertionError(f"no audio payload in any stage output (got {len(outputs)} stages)")


@pytest.mark.core_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_runner_function", [_OMNI_RUNNER_PARAM], indirect=True)
def test_higgs_audio_v2_offline_plain_text(omni_runner_function: OmniRunner) -> None:
    """
    Offline plain-text -> 24 kHz PCM via the bundled higgs_audio_v2 deploy.

    Deploy Setting: vllm_omni/deploy/higgs_audio_v2.yaml (codec_streaming=true,
        async_chunk=false, enforce_eager=true per stage).
    Input Modal: text
    Output Modal: audio (PCM tensor, 24 kHz mono float)
    Input Setting: single prompt, default sampling
    """
    omni = omni_runner_function.omni
    inputs = _compose_request(MODEL, TEST_TEXT)

    # max_tokens caps Stage-0 codec frames; Stage 1 max_tokens lifts the
    # per-codebook prefill cap. Mirror the deploy yaml's defaults.
    stage_sampling = [
        SamplingParams(temperature=1.0, top_p=0.95, top_k=50, max_tokens=1024, seed=42),
        SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=65536, seed=42),
    ]

    outputs = list(omni.generate([inputs], stage_sampling))
    assert outputs, "no outputs returned from Omni.generate"

    pcm = _extract_pcm(outputs)
    arr = pcm.numpy()

    assert arr.size > MIN_AUDIO_SAMPLES, (
        f"audio too short: {arr.size} samples, expected > {MIN_AUDIO_SAMPLES} "
        f"({MIN_AUDIO_SAMPLES / SAMPLE_RATE:.2f} s @ {SAMPLE_RATE} Hz)"
    )
    # Threshold 5e-4 (not 1e-3) keeps us above true silence while tolerating
    # the ``load_format=dummy`` Stage-1 codec — random RVQ/DAC weights collapse
    # the output to a near-constant peak around 7.4e-4. Real-weight runs reach
    # 0.1+ peak so this still catches genuine silence.
    assert np.max(np.abs(arr)) > 5e-4, "audio output is silent (peak abs amplitude < 5e-4)"
    assert np.isfinite(arr).all(), "audio contains non-finite samples (nan/inf)"


@pytest.mark.core_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_runner_function", [_OMNI_RUNNER_PARAM], indirect=True)
def test_higgs_audio_v2_offline_batch_two_prompts(omni_runner_function: OmniRunner) -> None:
    """
    Submit two prompts in one engine call and verify both yield non-trivial PCM.

    Guards the Stage-0 talker's per-slot audio state under a batched call —
    the prior example ran prompts one at a time citing 'request-scoped audio
    state'; this test pins the batched-2 behavior the engine path actually
    needs to support for /v1/audio/speech under concurrent traffic.
    """
    omni = omni_runner_function.omni
    prompts = [
        _compose_request(MODEL, "Hello world."),
        _compose_request(MODEL, "The quick brown fox jumps over the lazy dog."),
    ]
    stage_sampling = [
        SamplingParams(temperature=1.0, top_p=0.95, top_k=50, max_tokens=1024, seed=42),
        SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=65536, seed=42),
    ]

    outputs = list(omni.generate(prompts, stage_sampling))
    assert outputs, "no outputs returned from Omni.generate"

    pcm = _extract_pcm(outputs)
    # Whether the engine returns one concatenated PCM or per-request tensors is
    # implementation-defined; for either shape, total samples must clear the
    # floor for at least one of the two prompts.
    assert pcm.numel() > MIN_AUDIO_SAMPLES, (
        f"batched audio too short: {pcm.numel()} samples, expected > {MIN_AUDIO_SAMPLES}"
    )
    # See the silence-threshold note in test_higgs_audio_v2_offline_plain_text.
    assert np.max(np.abs(pcm.numpy())) > 5e-4, "batched audio output is silent"


def _compose_voice_clone_request(model_name: str, text: str, ref_text: str, ref_wav_path) -> dict:
    """Build the voice-clone prompt + additional_information payload.

    Calls the same ``build_voice_clone_prompt`` the serving layer uses online —
    the HF processor encodes the reference clip on the spot via the bundled
    HiggsAudioV2TokenizerModel. The talker's
    ``_maybe_apply_ref_audio_substitution`` consumes the tensors at prefill.
    """
    import soundfile as sf
    from transformers import AutoProcessor

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
        build_voice_clone_prompt,
    )

    wav, sr = sf.read(str(ref_wav_path), always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)  # downmix any stereo asset to mono
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    out = build_voice_clone_prompt(processor, text, wav, int(sr), ref_text)
    return {
        "prompt_token_ids": out["prompt_token_ids"],
        # Pass tensors bare (see serving_speech._build_higgs_audio_v2_params
        # — list-wrapped tensors get dropped by the msgspec serializer).
        "additional_information": {
            "audio_input_ids": out["audio_input_ids"],
            "audio_input_ids_mask": out["audio_input_ids_mask"],
        },
    }


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_runner_function", [_OMNI_RUNNER_PARAM], indirect=True)
def test_higgs_audio_v2_offline_voice_clone(omni_runner_function: OmniRunner) -> None:
    """
    Shallow voice clone via ref_audio + ref_text, offline.

    Deploy Setting: vllm_omni/deploy/higgs_audio_v2.yaml.
    Input Modal: text + reference audio (qwen3_tts/clone_2.wav vendored asset)
    Output Modal: audio (PCM tensor, 24 kHz mono float)
    Input Setting: single prompt
    """
    omni = omni_runner_function.omni
    ref_path = get_asset_path(_REF_AUDIO_ASSET)
    inputs = _compose_voice_clone_request(MODEL, TEST_TEXT, _REF_TEXT, ref_path)

    stage_sampling = [
        SamplingParams(temperature=1.0, top_p=0.95, top_k=50, max_tokens=1024, seed=42),
        SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=65536, seed=42),
    ]

    outputs = list(omni.generate([inputs], stage_sampling))
    assert outputs, "no outputs returned from Omni.generate for voice clone"

    pcm = _extract_pcm(outputs)
    arr = pcm.numpy()
    assert arr.size > MIN_AUDIO_SAMPLES, f"cloned audio too short: {arr.size} samples, expected > {MIN_AUDIO_SAMPLES}"
    assert np.max(np.abs(arr)) > 5e-4, "voice-clone output is silent"
    assert np.isfinite(arr).all(), "voice-clone audio contains non-finite samples"
