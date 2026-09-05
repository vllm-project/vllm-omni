# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64
import io
import os

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "ICTNLP/LLaMA-Omni2-0.5B"
DEPLOY_CONFIG = get_deploy_config_path("llama_omni2.yaml")

test_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            server_args=["--no-trust-remote-code"],
            stage_init_timeout=0,
        ),
        id="default",
    )
]


def _messages(*, include_audio: bool) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": "Answer with exactly one word: OK",
        }
    ]
    if include_audio:
        audio = generate_synthetic_audio(1, 1, sample_rate=16000)
        content.insert(
            0,
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio['base64']}"},
            },
        )
    return [{"role": "user", "content": content}]


def _assert_wav_chunk(encoded: str) -> int:
    waveform, sample_rate = sf.read(
        io.BytesIO(base64.b64decode(encoded)),
        dtype="float32",
    )
    assert sample_rate == 24000
    assert waveform.size > 0
    assert np.isfinite(waveform).all()
    return int(waveform.size)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_chat_stream_text_audio_and_speech_input(
    omni_server,
    openai_client,
) -> None:
    stream = openai_client.client.chat.completions.create(
        model=omni_server.model,
        messages=_messages(include_audio=True),
        modalities=["text", "audio"],
        stream=True,
    )

    text_deltas: list[str] = []
    audio_chunks: list[str] = []
    finish_reasons: list[str] = []
    for chunk in stream:
        assert chunk.object == "chat.completion.chunk"
        assert chunk.modality in {"text", "audio"}
        for choice in chunk.choices:
            content = choice.delta.content
            if content:
                if chunk.modality == "text":
                    text_deltas.append(content)
                else:
                    audio_chunks.append(content)
            if choice.finish_reason is not None:
                finish_reasons.append(choice.finish_reason)

    assert "".join(text_deltas).strip()
    assert len(audio_chunks) > 1
    assert sum(_assert_wav_chunk(chunk) for chunk in audio_chunks) > 0
    assert finish_reasons == ["stop"]


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_completions_rejected(omni_server, openai_client) -> None:
    responses = openai_client.send_completions_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": "Answer with exactly one word: OK",
                "max_tokens": 4,
            }
        },
        err_code=400,
    )

    assert not responses[0].success
