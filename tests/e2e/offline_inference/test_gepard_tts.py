# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for the Gepard-1.0 single-stage native-AR TTS.

Zero-shot only (the model's default learned voice). These tests need
a GPU and the NeMo NanoCodec (pytest.mark.slow / tts). The window arithmetic
behind the streaming decode is covered on CPU in
``tests/model_executor/models/test_gepard_window.py``.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner  # noqa: F401  (indirect fixture)
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni import Omni
from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig
from vllm_omni.model_executor.models.gepard.prompt import build_gepard_prompt_ids

# The codec decoder is an optional dependency; skip rather than fail the
# engine inside the worker process when it is absent.
pytest.importorskip("nemo.collections.tts.models")

MODEL_NAME = "nineninesix/gepard-1.0"
STAGE_CONFIG = get_deploy_config_path("gepard.yaml")
SAMPLE_RATE = 22050
# NanoCodec upsamples one FSQ frame to exactly this many samples; the advertised
# 21.5 fps is 22050 / 1024 rounded.
SAMPLES_PER_FRAME = 1024

_OMNI_RUNNER_PARAM = (
    MODEL_NAME,
    STAGE_CONFIG,
    {"trust_remote_code": True},
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


def _build_request(text: str) -> dict:
    """Assemble the [speaker slots, SOT, text, EOT, SOS] prompt."""
    cfg = GepardConfig()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompt_token_ids = build_gepard_prompt_ids(
        tokenizer(text, add_special_tokens=False)["input_ids"],
        start_of_text=cfg.start_of_text,
        end_of_text=cfg.end_of_text,
        start_of_speech=cfg.start_of_speech,
        speaker_token_base=cfg.speaker_token_base,
        num_speaker_prefix=cfg.num_speaker_prefix,
    )
    return {
        "prompt_token_ids": prompt_token_ids,
        "additional_information": {"text": [text], "max_new_frames": [1000]},
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


def _synthesize(omni: Omni, text: str) -> torch.Tensor:
    """Run one request and return its waveform.

    No SamplingParams on purpose: a caller-supplied object replaces the stage
    defaults rather than merging over them, dropping the pipeline's
    stop_token_ids.
    """
    waveform = None
    for stage_outputs in omni.generate(_build_request(text)):
        # OmniRequestOutput.request_output is a single RequestOutput, not a list.
        req_output = stage_outputs.request_output
        if req_output is None:
            continue
        for out in req_output.outputs:
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
    return waveform


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize(
    "text",
    [
        "Hello, this is Gepard speaking.",
        "The quick brown fox jumps over the lazy dog.",
    ],
)
def test_gepard_offline_zero_shot(omni_runner, text: str) -> None:
    """Zero-shot synthesis produces a finite mono waveform at 22.05 kHz."""
    wav = _synthesize(omni_runner, text)

    assert wav.dim() == 1, f"expected mono 1-D, got shape {tuple(wav.shape)}"
    assert wav.numel() > 0, "empty waveform"
    assert torch.isfinite(wav).all(), "non-finite samples"

    # Every sample must come from the codec, which emits whole frames. A
    # payload that also carries non-audio data (e.g. per-step hidden states
    # folded in under the same output key) breaks this immediately.
    assert wav.numel() % SAMPLES_PER_FRAME == 0, (
        f"{wav.numel()} samples is not a whole number of {SAMPLES_PER_FRAME}-sample frames"
    )

    # Decoded speech sits inside the waveform range; anything far outside it is
    # not codec output.
    assert float(wav.abs().max()) <= 1.5, f"peak {float(wav.abs().max()):.2f} is out of range"

    # The request must stop on its own rather than run to max_tokens: a short
    # line is a few seconds of speech, not the ~190 s the 4096-token cap allows.
    seconds = wav.numel() / SAMPLE_RATE
    assert 0.5 < seconds < 30.0, f"implausible duration {seconds:.2f}s — did STOP fire?"
