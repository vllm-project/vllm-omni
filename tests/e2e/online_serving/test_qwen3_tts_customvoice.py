# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
E2E Online tests for Qwen3-TTS model with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
actual model inference, not mocks.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import io
import json

import pytest
import requests
import soundfile as sf
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from tests.helpers.assertions import assert_audio_speech_response
from tests.helpers.client import OmniResponse
from tests.helpers.mark import hardware_test
from tests.helpers.media import concat_audio
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import (
    get_deploy_config_path,
    get_deploy_config_stage,
    modify_stage_config,
)
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_run import decode_output
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters.base import conditioning_cache_salt
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

_DEFAULT_STAGE_CONFIG = get_deploy_config_path("qwen3_tts.yaml")
_STAGE_CONFIG = modify_stage_config(
    _DEFAULT_STAGE_CONFIG,
    updates={"stages": {0: {"default_sampling_params.max_tokens": 500}}},
)


def get_prompt(prompt_type="text"):
    """Text prompt for text-to-audio tests (same as test_qwen3_omni - beijing test case)."""
    prompts = {
        "text": "Beijing, China's capital, blends ancient wonders like the Great Wall with modern marvels. This vibrant metropolis offers rich culture, delicious Peking duck, and endless exploration opportunities.",
    }
    return prompts.get(prompt_type, prompts["text"])


def get_max_batch_size(size_type="few"):
    """Batch size for concurrent requests (same as test_qwen3_omni)."""
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=_STAGE_CONFIG,
            server_args=["--trust-remote-code"],
        ),
        id="async_chunk",
    )
]

default_tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=_DEFAULT_STAGE_CONFIG,
            server_args=["--trust-remote-code"],
        ),
        id="async_chunk",
    )
]

# Exercise the throughput path in the existing L4 lane with smaller capacities.
# The production H200 profile retains its own admission and graph buckets.
_FAST_PATH_STAGE_CONFIG = modify_stage_config(
    get_deploy_config_path("qwen3_tts_high_concurrency_mrv2_single_gpu.yaml"),
    updates={
        # The temporary overlay lives outside deploy/, so anchor its parent.
        "base_config": get_deploy_config_path("qwen3_tts_high_concurrency_mrv2.yaml"),
        "cuda_mps": False,
        "connectors.connector_of_shared_memory.extra.decode_batch_max_size": 2,
        "connectors.connector_of_shared_memory.extra.decode_cudagraph_batch_sizes": [1, 2],
        "stages": {
            0: {
                "max_num_seqs": 4,
                "max_num_batched_tokens": 128,
                "compilation_config.cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
                "compilation_config.max_cudagraph_capture_size": 128,
            },
            1: {"max_num_seqs": 4},
        },
    },
)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server",
    [OmniServerParams(model=MODEL, stage_config_path=_FAST_PATH_STAGE_CONFIG, server_args=["--trust-remote-code"])],
    indirect=True,
)
def test_cached_predictor_streaming_audio(omni_server, online_client) -> None:
    """Cover the optimized predictor, first audio and codec path with real weights at L3."""
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "The quick brown fox jumps over the lazy dog.",
            "stream": True,
            "stream_format": "audio",
            "response_format": "wav",
            "task_type": "CustomVoice",
            "voice": "vivian",
        },
        request_num=4,
    )


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", default_tts_server_params, indirect=True)
def test_default_cuda_graph_startup(omni_server) -> None:
    """Verify both stages start with the shipped CUDA Graph configuration.

    The fixture reaching this test is the smoke assertion: it waits for the
    server to become ready after both stages finish model initialization and
    CUDA Graph capture. The regression covered here exited during stage 1
    capture, before fixture setup could complete.
    """
    for stage_id in (0, 1):
        stage = get_deploy_config_stage("qwen3_tts.yaml", stage_id=stage_id)
        assert stage.get("enforce_eager", False) is False

    assert omni_server.proc is not None
    assert omni_server.proc.poll() is None


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, online_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: few requests
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "response_format": "wav",
        "task_type": "CustomVoice",
        "voice": "vivian",
    }

    online_client.send_audio_speech_request(request_config, request_num=get_max_batch_size())


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_002(omni_server, online_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=True
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "stream_format": "audio",
        "response_format": "wav",
        "task_type": "CustomVoice",
        "voice": "vivian",
    }

    online_client.send_audio_speech_request(request_config)


### Tests for /v1/run
def _stage0_input(text: str, speaker: str) -> dict:
    """Build the talker's input as the speech endpoint does."""
    tts_params = {"text": [text], "task_type": ["CustomVoice"], "language": ["Auto"], "speaker": [speaker]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
    with open(hf_hub_download(MODEL, "config.json")) as f:
        talker_config = json.load(f)["talker_config"]
    prompt_len = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=tts_params,
        task_type="CustomVoice",
        tokenize_prompt=lambda prompt: tokenizer(prompt, padding=False)["input_ids"],
        codec_language_id=talker_config.get("codec_language_id"),
        spk_is_dialect=talker_config.get("spk_is_dialect"),
    )
    cache_salt = conditioning_cache_salt(OpenAICreateSpeechRequest(input=text, voice=speaker), tts_params)
    return {"prompt_token_ids": [1] * prompt_len, "additional_information": tts_params, "cache_salt": cache_salt}


def _output_to_wav(output: str) -> bytes:
    """Decode a /v1/run final output into WAV bytes."""
    audio_output, audio_key = OmniOpenAIServingSpeech._extract_audio_output(decode_output(output))
    wav = io.BytesIO()
    sf.write(wav, concat_audio(audio_output[audio_key]), int(audio_output["sr"]), format="WAV")
    return wav.getvalue()


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server",
    # /v1/run rejects async_chunk, so we need to run this test without it.
    [
        OmniServerParams(
            model=MODEL, stage_config_path=_STAGE_CONFIG, server_args=["--trust-remote-code", "--no-async-chunk"]
        )
    ],
    indirect=True,
)
def test_run_stage_chain_returns_speech(omni_server, run_level: str) -> None:
    """Ensure calling /v1/run once per stage, posting each response as the next request, returns the speech."""
    request_config = {"input": get_prompt(), "voice": "vivian", "response_format": "wav"}
    url = f"http://{omni_server.host}:{omni_server.port}/v1/run"

    first = requests.post(
        url, json={"stage_input": _stage0_input(request_config["input"], request_config["voice"])}, timeout=300
    )
    first.raise_for_status()
    next_request = first.json()
    assert next_request["stage_id"] == 1

    final = requests.post(url, json=next_request, timeout=300)
    final.raise_for_status()
    response = OmniResponse(success=True, audio_bytes=_output_to_wav(final.json()["output"]), audio_format="audio/wav")
    assert_audio_speech_response(response, request_config, run_level)
