# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep prompt tokenization coverage on OmniVoice's diffusion request path."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import (
    OmniVoicePipeline,
    _tokenize_with_nonverbal_tags,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("audio_mode", ["none", "uncached", "cached"])
def test_prepare_request_tokenizes_text_once_with_reference_audio(audio_mode):
    pipeline = object.__new__(OmniVoicePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.sample_rate = 24000
    pipeline.config = SimpleNamespace(
        num_audio_codebook=8, audio_mask_id=99, frame_rate=25, audio_chunk_duration=15.0, audio_chunk_threshold=30.0
    )
    pipeline.duration_estimator = SimpleNamespace(estimate_duration=Mock(return_value=2))
    pipeline.tokenizer = SimpleNamespace(
        encode=Mock(side_effect=[SimpleNamespace(ids=[1, 2]), SimpleNamespace(ids=[5, 6, 7])])
    )
    pipeline.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(sample_rate=24000, hop_length=240))
    ref_tokens = torch.full((8, 4), 8, dtype=torch.long)
    pipeline._encode_ref_audio = Mock(return_value=ref_tokens)
    pipeline._speaker_cache = Mock()
    pipeline._speaker_cache.get.return_value = (
        {"ref_audio_tokens": ref_tokens, "reference_rms": 0.2} if audio_mode == "cached" else None
    )
    prompt = {
        "text": "hello world",
        "mm_processor_kwargs": {"ref_text": "reference"},
    }
    if audio_mode != "none":
        prompt["multi_modal_data"] = {"audio": [(np.full(2400, 0.2, dtype=np.float32), 24000)]}
        prompt["voice_name"] = "speaker"

    prepared = pipeline._prepare_request_input(prompt, {"seed": 42})

    assert pipeline.tokenizer.encode.call_count == 2  # One style and one text encoding.
    expected_ref_text = "reference" if audio_mode == "none" else "reference."
    assert pipeline.tokenizer.encode.call_args.args == (f"<|text_start|>{expected_ref_text} hello world<|text_end|>",)
    torch.testing.assert_close(prepared.input_ids[:5], torch.tensor([1, 2, 5, 6, 7]).unsqueeze(1).repeat(1, 8))
    assert prepared.target_len == 2
    assert prepared.seed == 42
    assert not prepared.audio_mask[:5].any()
    assert prepared.audio_mask[5:].all()
    assert prepared.cond_len == (7 if audio_mode == "none" else 11)
    if audio_mode == "uncached":
        pipeline._encode_ref_audio.assert_called_once()
        audio, sr = pipeline._encode_ref_audio.call_args.args
        assert tuple(audio.shape) == (1, 2400)
        assert sr == 24000
        pipeline._speaker_cache.put.assert_called_once()
    else:
        pipeline._encode_ref_audio.assert_not_called()
    if audio_mode != "none":
        torch.testing.assert_close(prepared.input_ids[5:9], ref_tokens.T)


def test_nonverbal_tags_are_tokenized_independently_without_repeating_segments():
    tokenizer = SimpleNamespace(
        encode=Mock(side_effect=[SimpleNamespace(ids=[1, 2]), SimpleNamespace(ids=[3]), SimpleNamespace(ids=[4])])
    )

    assert _tokenize_with_nonverbal_tags("hello[laughter]world", tokenizer) == [1, 2, 3, 4]
    assert [call.args for call in tokenizer.encode.call_args_list] == [("hello",), ("[laughter]",), ("world",)]
