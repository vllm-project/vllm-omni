# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Float32 reference-audio byte transport shared by MRV1 and MRV2."""

from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import _encode_raw_audio, build_voxcpm2_prompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("ref_text", [None, "reference text"])
def test_binary_reference_audio_preserves_prefill_length_and_samples(ref_text) -> None:
    class Tokenizer:
        bos_token_id = 1

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [1, 2, 3]

    samples = np.arange(24, dtype=np.float32)
    kwargs = dict(
        hf_config=SimpleNamespace(audio_vae_config={"sample_rate": 16000, "encoder_rates": [2]}, patch_size=2),
        tokenizer=Tokenizer(),
        split_map={},
        text="hello",
        ref_sr=16000,
        ref_text=ref_text,
    )
    list_prompt = build_voxcpm2_prompt(**kwargs, ref_audio=samples.tolist())
    binary_prompt = build_voxcpm2_prompt(**kwargs, ref_audio=samples.tobytes())
    assert len(binary_prompt["prompt_token_ids"]) == len(list_prompt["prompt_token_ids"])
    key = "reference_audio" if ref_text is None else "prompt_audio"
    assert binary_prompt["additional_information"][key][0][0] == samples.tobytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("uploaded", [False, True])
async def test_serving_adapter_transports_reference_as_float32_bytes(uploaded) -> None:
    from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2 import VoxCPM2Adapter

    class Tokenizer:
        bos_token_id = 1

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [1, 2]

    class Server:
        async def _resolve_ref_audio_array(self, uri: str):
            return np.arange(8, dtype=np.float32), 16000, "cache-key"

    hf_config = SimpleNamespace(audio_vae_config={"sample_rate": 16000, "encoder_rates": [2]}, patch_size=2)
    ctx = SimpleNamespace(
        server=Server(), engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))
    )
    adapter = VoxCPM2Adapter(ctx)
    adapter._tokenizer = Tokenizer()
    adapter._encode = lambda _text: []
    request = SimpleNamespace(
        ref_audio=None if uploaded else "data:audio/test", input="hello", ref_text=None, voice=None
    )
    uploaded_ref = (np.arange(8, dtype=np.float32), 16000) if uploaded else None
    prompt = await adapter._build_prompt(request, uploaded_ref=uploaded_ref)
    raw = prompt["additional_information"]["reference_audio"][0][0]
    assert isinstance(raw, bytes)
    np.testing.assert_array_equal(np.frombuffer(raw, dtype=np.float32), np.arange(8, dtype=np.float32))
    transported = msgspec.msgpack.decode(msgspec.msgpack.encode(prompt))
    assert transported["additional_information"]["reference_audio"][0][0] == raw


@pytest.mark.parametrize("padding_mode", ["left", "right"])
def test_binary_audio_encode_matches_sample_list(padding_mode) -> None:
    class EchoVAE(nn.Module):
        latent_dim = 1

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def encode(self, audio, sample_rate):
            return audio.unsqueeze(1)

    tts = SimpleNamespace(audio_vae=EchoVAE(), _encode_sample_rate=16000, patch_size=2, chunk_size=2)
    samples = np.linspace(-1, 1, 7, dtype=np.float32)
    expected = _encode_raw_audio(tts, samples.tolist(), 16000, padding_mode=padding_mode)
    actual = _encode_raw_audio(tts, samples.tobytes(), 16000, padding_mode=padding_mode)
    torch.testing.assert_close(actual, expected)


def test_binary_audio_rejects_incomplete_float32_sample() -> None:
    with pytest.raises(ValueError, match="float32 samples"):
        _encode_raw_audio(None, b"abc", 16000)
