# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end offline inference tests for Voxtral TTS model.

Tests both synchronous (Omni) and async streaming (AsyncOmni) paths.

Equivalent to running:
    python3 examples/offline_inference/text_to_speech/voxtral_tts/end2end.py \
        --model mistralai/Voxtral-4B-TTS-2603 \
        --voice casual_female \
        --text "Hello, how are you?" \
        --write-audio
"""

import asyncio
import os
import uuid

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from pathlib import Path

import numpy as np
import pytest
import torch
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.entrypoints.async_omni import AsyncOmni

MODEL = "mistralai/Voxtral-4B-TTS-2603"
STAGE_CONFIG = get_deploy_config_path("voxtral_tts.yaml")
SAMPLE_RATE = 24000
# Minimum expected audio samples for a short sentence (~0.04s of 24kHz audio)
MIN_AUDIO_SAMPLES = 1000
VOICE = "casual_female"
TEST_TEXT = "Hello, how are you?"


def _compose_request(model_name: str, text: str, voice: str) -> dict:
    """Build the TTS input dict using mistral tokenizer."""
    if Path(model_name).is_dir():
        mistral_tokenizer = MistralTokenizer.from_file(str(Path(model_name) / "tekken.json"))
    else:
        mistral_tokenizer = MistralTokenizer.from_hf_hub(model_name)
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer

    tokenized = instruct_tokenizer.encode_speech_request(SpeechRequest(input=text, voice=voice))
    return {
        "prompt_token_ids": tokenized.tokens,
        "additional_information": {"voice": [voice]},
    }


def _resolve_stage_config(run_level: str) -> str:
    """Resolve stage config: strip load_format for advanced_model (real weights)."""
    if run_level == "advanced_model":
        return modify_stage_config(
            STAGE_CONFIG,
            deletes={
                "stages": {
                    0: ["load_format"],
                    1: ["load_format"],
                }
            },
        )
    return STAGE_CONFIG


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_voxtral_tts_offline_basic(run_level):
    """Test basic Voxtral TTS offline inference with a voice preset."""
    stage_config = _resolve_stage_config(run_level)

    with OmniRunner(
        MODEL,
        stage_configs_path=stage_config,
        enforce_eager=True,
    ) as runner:
        omni = runner.omni
        inputs = _compose_request(MODEL, TEST_TEXT, VOICE)

        sampling_params = SamplingParams(max_tokens=2500)
        sampling_params_list = [sampling_params, sampling_params]

        outputs = list(omni.generate([inputs], sampling_params_list))

        assert len(outputs) > 0, "No outputs generated"

        # Find audio output from the final stage
        audio_data = None
        for o in outputs:
            mm = getattr(o, "multimodal_output", None)
            if mm and "audio" in mm:
                audio_data = mm["audio"]
                break

        assert audio_data is not None, "No audio output found in any stage output"

        # Concatenate audio chunks if returned as a list of tensors
        if isinstance(audio_data, list):
            audio_tensor = torch.cat(audio_data)
        elif isinstance(audio_data, torch.Tensor):
            audio_tensor = audio_data
        else:
            audio_tensor = torch.tensor(audio_data)

        audio_array = audio_tensor.float().cpu().numpy()

        assert len(audio_array) > MIN_AUDIO_SAMPLES, (
            f"Audio too short: {len(audio_array)} samples, expected > {MIN_AUDIO_SAMPLES}"
        )

        # Verify audio isn't all zeros / silence
        assert np.max(np.abs(audio_array)) > 0.01, "Audio appears to be silence"


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_voxtral_tts_offline_streaming(run_level):
    """Test AsyncOmni streaming inference for Voxtral TTS."""

    async def _run():
        stage_config = _resolve_stage_config(run_level)

        async_omni = AsyncOmni(
            model=MODEL,
            stage_configs_path=stage_config,
            stage_init_timeout=300,
            enforce_eager=True,
        )

        try:
            inputs = _compose_request(MODEL, TEST_TEXT, VOICE)

            sampling_params = SamplingParams(max_tokens=2500)

            all_audio_chunks = []
            accumulated_samples = 0
            chunk_idx = 0
            async for stage_output in async_omni.generate(
                prompt=inputs,
                request_id=str(uuid.uuid4()),
                sampling_params_list=[sampling_params, sampling_params],
            ):
                mm = getattr(stage_output, "multimodal_output", None)
                if not mm or "audio" not in mm:
                    continue

                audio_chunk = mm["audio"]
                finished = stage_output.finished

                if isinstance(audio_chunk, torch.Tensor):
                    if finished:
                        # Last chunk may return whole audio instead of
                        # a delta — cut already-accumulated samples.
                        audio_np = audio_chunk[accumulated_samples:].float().detach().cpu().numpy()
                    else:
                        audio_np = audio_chunk.float().detach().cpu().numpy()
                elif isinstance(audio_chunk, list):
                    audio_np = audio_chunk[chunk_idx].float().detach().cpu().numpy()
                else:
                    audio_np = audio_chunk

                accumulated_samples += len(audio_np)
                all_audio_chunks.append(audio_np)
                chunk_idx += 1

            assert len(all_audio_chunks) > 0, "No audio chunks received from streaming"

            audio_array = np.concatenate(all_audio_chunks)

            assert len(audio_array) > MIN_AUDIO_SAMPLES, (
                f"Audio too short: {len(audio_array)} samples, expected > {MIN_AUDIO_SAMPLES}"
            )

            # Verify audio isn't all zeros / silence
            assert np.max(np.abs(audio_array)) > 0.01, "Audio appears to be silence"

        finally:
            async_omni.shutdown()

    asyncio.run(_run())
