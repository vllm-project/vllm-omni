# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.ming.audio_lengths import iter_chunk_lengths
from vllm_omni.model_executor.models.common.ming.audio_projector import Ming3AudioProjector
from vllm_omni.model_executor.models.common.ming.whisper_encoder import WhisperAudioEncoder
from vllm_omni.model_executor.models.ming3_audio.ming3_audio import BailingMoeV3AudioForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model(normalize: bool) -> BailingMoeV3AudioForConditionalGeneration:
    # Exercise the real audio path without constructing the language model.
    model = BailingMoeV3AudioForConditionalGeneration.__new__(BailingMoeV3AudioForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(audio_config=SimpleNamespace(whisper_chunk_size=64), norm_query_embeds=normalize)
    model.audio_encoder = WhisperAudioEncoder(n_mels=8, n_ctx=32, n_state=16, n_head=2, n_layer=1, use_flash_attn=False)
    model.audio_projector = Ming3AudioProjector(
        audio_dim=16, llm_dim=12, kernels=[3, 3], strides=[2, 2], paddings=[1, 1]
    )
    return model.eval()


def _unbatched_embeddings(model, features, feature_length):
    features = features[:feature_length]
    lengths = iter_chunk_lengths(feature_length, model.config.audio_config.whisper_chunk_size)
    padded = nn.utils.rnn.pad_sequence(list(features.split(lengths)), batch_first=True)
    encoded, encoded_lengths = model.audio_encoder.forward_padded(padded, list(lengths))
    projected = model.audio_projector(encoded)
    output = torch.cat(
        [projected[i, : model.audio_projector.output_length(length)] for i, length in enumerate(encoded_lengths)]
    )
    return F.normalize(output, dim=-1) if model.config.norm_query_embeds else output


@pytest.mark.parametrize("feature_length", [1, 63, 64, 65, 2048, 2049, 4097])
@pytest.mark.parametrize("normalize", [False, True])
@torch.inference_mode()
def test_audio_encoding_bounds_batches_and_preserves_embeddings(feature_length, normalize, monkeypatch):
    torch.manual_seed(0)
    model = _model(normalize)
    # Unused frames must not change either the padding width or the output.
    features = torch.randn(feature_length + 7, 8)
    expected = _unbatched_embeddings(model, features, feature_length)
    calls = []
    forward = model.audio_encoder.forward_padded

    def record(batch, lengths):
        calls.append((tuple(batch.shape), list(lengths)))
        return forward(batch, lengths)

    monkeypatch.setattr(model.audio_encoder, "forward_padded", record)
    actual = model._encode_audio_item(features, feature_length)

    assert max(shape[0] for shape, _ in calls) <= 32
    assert all(shape[1] == min(feature_length, 64) for shape, _ in calls)
    assert [length for _, lengths in calls for length in lengths] == list(iter_chunk_lengths(feature_length, 64))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
