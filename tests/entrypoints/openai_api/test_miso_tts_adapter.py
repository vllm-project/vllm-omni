# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Miso TTS serving adapter tests; these do not need the optional torchtune extra."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import TTS_ADAPTER_REGISTRY, all_tts_stage_keys
from vllm_omni.entrypoints.openai.tts_adapters.miso_tts import MisoTTSAdapter
from vllm_omni.transformers_utils.configs.miso_tts import MISO_DEFAULT_MAX_FRAMES, MISO_MAX_SEQ_LEN

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


def _adapter() -> MisoTTSAdapter:
    server = SimpleNamespace(_tts_tokenizer=_WordTokenizer())
    return MisoTTSAdapter(SimpleNamespace(server=server, engine_client=None))


def _request(text: str = "hello there", **kwargs) -> OpenAICreateSpeechRequest:
    return OpenAICreateSpeechRequest(input=text, **kwargs)


def test_adapter_registered() -> None:
    assert TTS_ADAPTER_REGISTRY["miso_tts"] is MisoTTSAdapter
    assert "miso_tts" in all_tts_stage_keys()


def test_validate_accepts_prompt_just_inside_budget() -> None:
    # Speaker tag + words + BOS/EOS.
    words = MISO_MAX_SEQ_LEN - MISO_DEFAULT_MAX_FRAMES - 4
    assert _adapter().validate(_request(" ".join(["w"] * words))) is None


def test_validate_rejects_prompt_at_budget() -> None:
    words = MISO_MAX_SEQ_LEN - MISO_DEFAULT_MAX_FRAMES - 3
    assert "too long" in _adapter().validate(_request(" ".join(["w"] * words)))


@pytest.mark.parametrize("max_new_tokens", [MISO_MAX_SEQ_LEN - 1, MISO_MAX_SEQ_LEN, 4096])
def test_validate_rejects_frames_without_prompt_room(max_new_tokens: int) -> None:
    assert _adapter().validate(_request(max_new_tokens=max_new_tokens)) is not None


@pytest.mark.parametrize(
    "extra_params",
    [
        {"temperature": 0},
        {"top_k": 0},
        {"top_k": 2052},
        {"top_k": 1.5},
        {"top_p": 0.9},
    ],
)
def test_validate_rejects_unusable_sampling(extra_params: dict) -> None:
    assert _adapter().validate(_request(extra_params=extra_params)) is not None


@pytest.mark.parametrize("value", [-1.0, float("inf"), True])
def test_invalid_temperature_is_rejected_by_request_schema(value: object) -> None:
    with pytest.raises(ValidationError, match=r"extra_params\.temperature"):
        _request(extra_params={"temperature": value})


def test_build_forwards_seed_and_sampling() -> None:
    request = _request(voice="3", seed=0, max_new_tokens=50, extra_params={"temperature": 0.7, "top_k": 20})
    adapter = _adapter()
    assert adapter.validate(request) is None

    prepared = asyncio.run(adapter.build(request, [], False))

    assert prepared.prompt["additional_information"] == {
        "text": ["hello there"],
        "speaker": [3],
        "max_generation_frames": [50],
        "seed": [0],
        "temperature": [0.7],
        "topk": [20],
    }
