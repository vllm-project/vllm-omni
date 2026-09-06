# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E tests for VibeVoice direct Omni inference on the default TP=1 topology."""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import pytest
import soundfile as sf
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.entrypoints.openai.tts_adapters.vibevoice import VibeVoiceTTSAdapter
from vllm_omni.model_executor.models.vibevoice.pipeline import VIBEVOICE_VALID_TOKEN_IDS

_MODEL = os.getenv("VIBEVOICE_TEST_MODEL", "microsoft/VibeVoice-1.5B")
_TOKENIZER = os.getenv("VIBEVOICE_TEST_TOKENIZER", "Qwen/Qwen2.5-1.5B")
_MODEL_REVISION = os.getenv("VIBEVOICE_TEST_MODEL_REVISION")
_TOKENIZER_REVISION = os.getenv("VIBEVOICE_TEST_TOKENIZER_REVISION")
_DEPLOY_CONFIG_OVERRIDE = os.getenv("VIBEVOICE_TEST_DEPLOY_CONFIG")
_DEPLOY_CONFIG = (
    str(Path(_DEPLOY_CONFIG_OVERRIDE).expanduser().resolve())
    if _DEPLOY_CONFIG_OVERRIDE
    else get_deploy_config_path("vibevoice.yaml")
)
_SAMPLE_RATE = 24_000
_REFERENCE_PATH = get_asset_path("qwen3_tts/clone_2.wav")
_RUNNER_ARGS = {
    "tokenizer": _TOKENIZER,
    "trust_remote_code": False,
}
if _MODEL_REVISION:
    _RUNNER_ARGS["revision"] = _MODEL_REVISION
if _TOKENIZER_REVISION:
    _RUNNER_ARGS["tokenizer_revision"] = _TOKENIZER_REVISION
_OMNI_RUNNER_PARAM = (_MODEL, _DEPLOY_CONFIG, _RUNNER_ARGS)

pytestmark = pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)


@pytest.fixture(scope="module", autouse=True)
def _require_real_weights(run_level: str) -> None:
    if run_level not in {"advanced_model", "full_model"}:
        pytest.skip("VibeVoice offline E2E requires --run-level advanced_model (or full_model)")


@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(
        _TOKENIZER,
        revision=_TOKENIZER_REVISION,
        trust_remote_code=False,
    )


def _make_prompt(text: str, request_index: int) -> dict:
    tokenizer = _tokenizer()
    waveform, sample_rate = sf.read(_REFERENCE_PATH, dtype="float32")
    rendered = VibeVoiceTTSAdapter._render_prompt([(0, text)], num_speakers=1)
    return {
        "prompt": rendered,
        "prompt_token_ids": tokenizer.encode(rendered, add_special_tokens=False),
        "multi_modal_data": {"audio": [(waveform, sample_rate)]},
        "multi_modal_uuids": {"audio": [f"vibevoice-offline-{request_index}:audio:0"]},
    }


def _sampling_params(max_tokens: int = 128) -> list[SamplingParams]:
    return [
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            allowed_token_ids=list(VIBEVOICE_VALID_TOKEN_IDS),
            stop_token_ids=[151643],
            detokenize=False,
        )
    ]


def _extract_audio(multimodal_output: Mapping[str, object]) -> tuple[torch.Tensor, int]:
    assert isinstance(multimodal_output, Mapping)
    audio = multimodal_output.get("audio")
    if audio is None:
        audio = multimodal_output.get("model_outputs")
    assert audio is not None

    raw_chunks = audio if isinstance(audio, list | tuple) else [audio]
    chunks = [torch.as_tensor(item).detach().cpu().reshape(-1) for item in raw_chunks if item is not None]
    assert chunks
    assert all(chunk.dtype == torch.float32 for chunk in chunks)
    waveform = torch.cat(chunks)

    sample_rate = multimodal_output.get("sr")
    if sample_rate is None:
        sample_rate = multimodal_output.get("sample_rate")
    if isinstance(sample_rate, list | tuple):
        assert sample_rate
        sample_rate = sample_rate[-1]
    if isinstance(sample_rate, torch.Tensor):
        assert sample_rate.numel() == 1
        sample_rate = sample_rate.item()
    assert isinstance(sample_rate, int | float)
    return waveform, int(sample_rate)


def _assert_valid_audio(output) -> None:
    audio, sample_rate = _extract_audio(output.outputs[0].multimodal_output)
    assert audio.dtype == torch.float32
    assert audio.ndim == 1
    assert sample_rate == _SAMPLE_RATE
    assert audio.numel() > 0
    assert audio.numel() % 3_200 == 0
    assert torch.isfinite(audio).all()
    assert 0.1 < audio.numel() / sample_rate < 60.0


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_vibevoice_reference_speech_001(omni_runner: OmniRunner) -> None:
    outputs = omni_runner.omni.generate(
        [_make_prompt("Hello, this is a direct Omni test.", 0)],
        sampling_params_list=_sampling_params(),
        use_tqdm=False,
    )
    assert len(outputs) == 1
    _assert_valid_audio(outputs[0])


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_vibevoice_prefill_decode_mixed_batch_002(omni_runner: OmniRunner) -> None:
    long_text = (
        "This deliberately longer sentence stays in decode while shorter "
        "requests enter prefill, exercising mixed prefill and decode scheduling."
    )
    texts = [long_text, "Hello one.", "Hello two.", "Hello three."]
    outputs = omni_runner.omni.generate(
        [_make_prompt(text, index) for index, text in enumerate(texts)],
        sampling_params_list=_sampling_params(),
        use_tqdm=False,
    )
    assert len(outputs) == len(texts)
    for output in outputs:
        _assert_valid_audio(output)
