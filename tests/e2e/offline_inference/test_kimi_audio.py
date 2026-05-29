# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for Kimi-Audio offline inference.

Covers the two task modes the example exercises end-to-end:

* ``audio2text``  — audio in, text out (stage 1 receives no audio
  codes, so its decode loop is a no-op).
* ``audio2audio`` — audio in, text + spoken audio out (two-stage,
  async-chunk off so this fits on one GPU).

Multi-turn and text2audio share the audio2audio pipeline; we don't
drive those from CI.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media import MediaConnector

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner

KIMI_MODEL = os.environ.get("KIMI_AUDIO_MODEL", "zhangj1an/kimi_audio_7b_random")
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
STAGE_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "kimi_audio.yaml")

OUTPUT_SAMPLE_RATE = 24000
SEED = 42

# Mirrors the chat template that ``KimiAudioForConditionalGeneration``
# expects. ``ctd`` is the audio-output conditioning marker; without it
# the audio head emits off-distribution codec tokens.
_AUDIO_PLACEHOLDER = "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"


def _build_prompt(question: str, output_type: str) -> str:
    ct_token = "<|im_kimia_speech_ctd_id|>" if output_type == "both" else ""
    return (
        f"<|im_kimia_user_msg_start|>{_AUDIO_PLACEHOLDER}{question}{ct_token}"
        f"<|im_msg_end|><|im_kimia_assistant_msg_start|>"
    )


def _load_audio(url: str, sample_rate: int = 16000) -> np.ndarray:
    """Fetch + resample to the rate Kimi-Audio's whisper expects.

    Falls back to vLLM's bundled ``mary_had_lamb`` asset if the network
    fetch fails (so tests still run in air-gapped CI runners)."""
    try:
        connector = MediaConnector(allowed_local_media_path="/")
        audio, src_sr = connector.fetch_audio(url)
    except Exception:
        audio, src_sr = AudioAsset("mary_had_lamb").audio_and_sample_rate
    if int(src_sr) != sample_rate:
        from vllm.multimodal.audio import resample_audio_scipy

        audio = resample_audio_scipy(audio.astype("float32"), orig_sr=int(src_sr), target_sr=sample_rate)
    return audio


def _audio2text_query() -> dict[str, Any]:
    audio = _load_audio("https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/test_audios/asr_example.wav")
    return {
        "prompt": _build_prompt("请将音频内容转换为文字。", output_type="text"),
        "multi_modal_data": {"audio": [audio]},
    }


def _audio2audio_query() -> dict[str, Any]:
    audio = _load_audio("https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/test_audios/qa_example.wav")
    return {
        "prompt": _build_prompt("", output_type="both"),
        "multi_modal_data": {"audio": [audio]},
    }


def _audio2text_sampling_params() -> list[SamplingParams]:
    """audio2text YAML defines two stages but stage 1 receives no audio
    codes — pass a placeholder so the orchestrator's per-stage list
    lines up."""
    thinker = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=512,
        seed=SEED,
        detokenize=True,
        # 151667 = <|im_kimia_text_eos|> — needed for text-only
        # audio2text because the audio-EOD path that normally halts
        # audio2audio never fires.
        stop_token_ids=[151644, 151645, 151667],
    )
    placeholder = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        seed=SEED,
        detokenize=False,
    )
    return [thinker, placeholder]


def _audio2audio_sampling_params() -> list[SamplingParams]:
    thinker = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=5,
        max_tokens=2048,
        seed=SEED,
        detokenize=True,
        # Stop only on text-EOS / msg_end. compute_logits boosts 151644
        # once the MIMO head emits its own EOD; 151667 is intentionally
        # absent (text head fires it before audio is done).
        stop_token_ids=[151644, 151645],
    )
    code2wav = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=16384,
        seed=SEED,
        detokenize=False,
    )
    return [thinker, code2wav]


def _extract_audio_tensor(multimodal_output: dict[str, Any]) -> torch.Tensor:
    # Explicit None-check (don't use ``or``) — values can be tensors,
    # whose truthiness raises for multi-element shapes.
    audio = multimodal_output.get("audio")
    if audio is None:
        audio = multimodal_output.get("model_outputs")
    if audio is None:
        return torch.zeros((0,), dtype=torch.float32)
    if isinstance(audio, list):
        parts = [torch.as_tensor(a).float().reshape(-1) for a in audio if a is not None]
        return torch.cat(parts, dim=-1) if parts else torch.zeros((0,), dtype=torch.float32)
    return torch.as_tensor(audio).float().reshape(-1)


def _final_outputs(stage_outputs):
    text_out = None
    audio_tensor = torch.zeros((0,), dtype=torch.float32)
    for stage in stage_outputs:
        rout = getattr(stage, "request_output", None)
        if rout is None:
            continue
        if getattr(stage, "final_output_type", None) == "text":
            completions = getattr(rout, "outputs", None) or []
            if completions:
                text_out = getattr(completions[0], "text", None)
        elif getattr(stage, "final_output_type", None) == "audio":
            completions = getattr(rout, "outputs", None) or []
            for c in completions:
                mm = getattr(c, "multimodal_output", None)
                if isinstance(mm, dict):
                    audio_tensor = _extract_audio_tensor(mm)
                    break
    return text_out, audio_tensor


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_kimi_audio_audio2text(tmp_path: Path):
    with OmniRunner(KIMI_MODEL, stage_configs_path=STAGE_CONFIG) as runner:
        outputs = list(runner.omni.generate([_audio2text_query()], _audio2text_sampling_params()))

    assert outputs, "no outputs returned from audio2text run"
    text, _ = _final_outputs(outputs)
    assert text is not None and len(text.strip()) > 0, f"empty audio2text transcript: {text!r}"


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_kimi_audio_audio2audio(tmp_path: Path):
    with OmniRunner(KIMI_MODEL, stage_configs_path=STAGE_CONFIG) as runner:
        outputs = list(runner.omni.generate([_audio2audio_query()], _audio2audio_sampling_params()))

    assert outputs, "no outputs returned from audio2audio run"
    text, audio = _final_outputs(outputs)

    assert text is not None and len(text.strip()) > 0, f"empty audio2audio text: {text!r}"
    assert audio.numel() > OUTPUT_SAMPLE_RATE // 2, f"audio2audio output too short: {audio.numel()} samples"

    duration_s = audio.shape[0] / OUTPUT_SAMPLE_RATE
    assert 0.5 < duration_s < 60.0, f"audio2audio duration out of range: {duration_s:.2f}s"

    audio_np = audio.numpy()
    rms = float(np.sqrt(np.mean(np.square(audio_np))))
    peak = float(np.abs(audio_np).max())
    assert rms > 1e-3, f"audio2audio RMS too low ({rms:.4g}) — likely silence"
    # ``peak`` is bounded loosely so the random-weight CI fixture (zhangj1an/kimi_audio_7b_random)
    # passes; with the real checkpoint the trained vocoder normally keeps peaks under ~0.5.
    assert peak < 5.0, f"audio2audio peak {peak:.3f} unreasonably large — likely numerical blowup"

    # Smoke-write so a maintainer can listen at the failure site.
    out_path = tmp_path / "audio2audio_output.wav"
    sf.write(str(out_path), audio_np, OUTPUT_SAMPLE_RATE)
    assert out_path.stat().st_size > 1000
