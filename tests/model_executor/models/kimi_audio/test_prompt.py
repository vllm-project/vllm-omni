# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PromptManager reference outputs with fixed encoder results, no model weights.

Fixtures were produced by the official get_prompt/tokenize_message methods at
the recorded revision, using checkpoint text BPE and synthetic audio features.
They pin the layout independently of the implementation under test.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tiktoken
import torch
from transformers import PretrainedConfig

from vllm_omni.model_executor.models.kimi_audio.prompt import (
    KimiAudioEncodedAudio,
    KimiAudioPromptBuilder,
    KimiAudioSpecialTokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]

REFERENCE = json.loads((Path(__file__).parent / "fixtures/prompt_reference.json").read_text(encoding="utf-8"))
INPUT_CONFIG = REFERENCE["input_config"]
FEATURE_SIZE = INPUT_CONFIG["continuous_feature_size"]


@pytest.fixture
def builder():
    return KimiAudioPromptBuilder(
        encode_text=REFERENCE["text_tokens"].__getitem__,
        special_tokens=KimiAudioSpecialTokens.from_vocab(REFERENCE["special_tokens"]),
        **INPUT_CONFIG,
    )


def test_bind_existing_tokenizer_without_interpreting_markers_in_text():
    # Real byte-level BPE, without downloading a checkpoint. Only the owning
    # vLLM object is substituted; encoding and marker lookup execute normally.
    encoding = tiktoken.Encoding(
        name="kimi-input-test",
        pat_str=r"[\s\S]",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens=REFERENCE["special_tokens"],
    )
    config = PretrainedConfig(
        kimia_token_offset=INPUT_CONFIG["audio_token_offset"],
        vocab_size=INPUT_CONFIG["audio_token_offset"] + INPUT_CONFIG["audio_vocab_size"],
        kimia_mimo_audiodelaytokens=INPUT_CONFIG["audio_delay"],
        kimia_adaptor_input_dim=FEATURE_SIZE,
    )
    bound = KimiAudioPromptBuilder.from_tokenizer(SimpleNamespace(_tokenizer=encoding), config)
    text = "你好 <|im_kimia_user_msg_start|> [BOS]"
    prompt = bound.build([{"role": "user", "message_type": "text", "content": text}])

    assert prompt.text_token_ids[1:-2] == list(text.encode("utf-8"))
    assert bound.tokens == KimiAudioSpecialTokens.from_vocab(REFERENCE["special_tokens"])
    assert bound.encode_text.__self__ is encoding


def encoded_inputs(messages):
    inputs = {}
    for i, message in enumerate(messages):
        if message["message_type"] not in ("audio", "audio-text"):
            continue
        continuous = message["message_type"] == "audio"
        source = message["content"] if continuous else message["content"][0]
        spec = REFERENCE["audio_inputs"][source]
        features = torch.full((len(spec["codes"]), FEATURE_SIZE), spec["feature_value"]) if continuous else None
        inputs[i] = KimiAudioEncodedAudio(spec["codes"], features)
    return inputs


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_matches_official_prompt_manager(builder, case):
    messages = copy.deepcopy(case["messages"])
    inputs = encoded_inputs(messages)
    result = builder.build(
        messages,
        audio_inputs=inputs,
        output_type=case["output_type"],
        add_assistant_start_msg=case["add_assistant_start_msg"],
    )

    assert result.text_token_ids == case["text_token_ids"]
    assert result.audio_token_ids == case["audio_token_ids"]
    assert [i for i, value in enumerate(result.is_continuous_mask) if value] == case["continuous_positions"]
    assert len(result.text_token_ids) == len(result.audio_token_ids) == len(result.is_continuous_mask)
    assert sum(result.is_continuous_mask) == sum(len(feature) for feature in result.continuous_features)
    assert [float(feature[0, 0]) for feature in result.continuous_features] == case["feature_values"]
    for actual, expected in zip(
        result.continuous_features, (x for x in inputs.values() if x.continuous_features is not None)
    ):
        torch.testing.assert_close(actual, expected.continuous_features)
    assert messages == case["messages"]


def test_requests_do_not_share_prompt_buffers(builder):
    case = next(case for case in REFERENCE["cases"] if case["name"] == "audio_text_output")
    inputs = encoded_inputs(case["messages"])
    first = builder.build(case["messages"], audio_inputs=inputs)
    first.audio_token_ids.clear()
    first.is_continuous_mask.clear()
    first.continuous_features.clear()
    second = builder.build(case["messages"], audio_inputs=inputs)
    assert second.audio_token_ids == case["audio_token_ids"]
    assert sum(second.is_continuous_mask) == 3
    assert len(second.continuous_features) == 1
