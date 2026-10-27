# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression coverage for Qwen2.5-Omni thinker-to-talker conditioning."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
    thinker2talker_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_AUDIO_TOKEN_ID = 50
_IMAGE_TOKEN_ID = 51
_VIDEO_TOKEN_ID = 52
_TEXT_EOS_TOKEN_ID = 60
_TEXT_PAD_TOKEN_ID = 61


@dataclass(frozen=True)
class _Status:
    name: str


@dataclass
class _Request:
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    status: _Status
    request_id: str = "request-0"
    sampling_params: None = None
    additional_information: None = None

    @property
    def all_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.output_token_ids]


@dataclass(frozen=True)
class _ThinkerConfig:
    audio_token_index: int = _AUDIO_TOKEN_ID
    image_token_index: int = _IMAGE_TOKEN_ID
    video_token_index: int = _VIDEO_TOKEN_ID


@dataclass(frozen=True)
class _TalkerConfig:
    hidden_size: int = 2
    tts_codec_mask_token_id: int = 70
    tts_codec_pad_token_id: int = 71
    tts_codec_start_token_id: int = 72


class _Talker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = _TalkerConfig()


def _bare_talker_model() -> Qwen2_5OmniForConditionalGeneration:
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    nn.Module.__init__(model)

    talker = _Talker()
    model.model_stage = "talker"
    model.model = talker
    model.talker = talker
    model.thinker_config = _ThinkerConfig()
    model.talker_config = talker.config
    model.thinker_embedding = nn.Embedding(64, 2)
    with torch.no_grad():
        token_values = torch.arange(64, dtype=torch.float32)
        model.thinker_embedding.weight.copy_(torch.stack([token_values, token_values + 0.5], dim=1))

    model.embed_text_bos_token = model.thinker_embedding(torch.tensor([59]))
    model.embed_text_spk_tokens = {"Chelsie": model.embed_text_bos_token}
    model.embed_text_eos_token = model.thinker_embedding(torch.tensor([_TEXT_EOS_TOKEN_ID]))
    model.embed_text_pad_token = model.thinker_embedding(torch.tensor([_TEXT_PAD_TOKEN_ID]))
    model.embed_codec_pad_token = torch.tensor([[100.0, 101.0]])
    model.embed_codec_bos_token = torch.tensor([[200.0, 201.0]])
    return model


def _assert_conditioning(status_name: str, output_token_ids: list[int], expected_output_ids: list[int]) -> None:
    model = _bare_talker_model()
    prompt_token_ids = [1, _AUDIO_TOKEN_ID, _IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID]
    request = _Request(
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        status=_Status(status_name),
    )
    # Generation captures the prompt plus one decode row for every sampled
    # token except the final token that closes the request.
    hidden_row_count = len(request.all_token_ids) - 1
    hidden_states = torch.arange(hidden_row_count * 2, dtype=torch.float32).reshape(-1, 2) + 10
    payload = thinker2talker_full_payload(
        transfer_manager=None,
        pooling_output={"hidden": hidden_states},
        request=request,
    )

    assert payload is not None
    assert payload["ids"]["output"] == expected_output_ids

    prompt_length = len(prompt_token_ids) + 2
    input_ids, input_embeds, updates = model.thinker_to_talker_process(
        input_ids=torch.zeros(prompt_length, dtype=torch.long),
        input_embeds=torch.zeros(prompt_length, 2),
        payload=payload,
    )

    prompt_hidden = payload["embed"]["prefill"]
    prompt_token_embeds = model.thinker_embedding(torch.tensor(prompt_token_ids))
    prompt_token_embeds = prompt_token_embeds.masked_fill(
        torch.tensor([False, True, True, True]).unsqueeze(-1),
        0,
    )
    expected_prompt_conditioning = prompt_hidden + prompt_token_embeds

    reply_hidden = payload["hidden_states"]["output"]
    reply_token_embeds = model.thinker_embedding(torch.tensor(expected_output_ids))
    expected_reply_conditioning = reply_hidden + reply_token_embeds
    expected_input_embeds = torch.cat(
        [
            expected_prompt_conditioning,
            model.embed_text_bos_token + model.embed_codec_pad_token,
            expected_reply_conditioning[:1] + model.embed_codec_bos_token,
        ]
    )
    expected_reply_queue = torch.cat(
        [
            expected_reply_conditioning[1:],
            model.embed_text_eos_token,
            model.embed_text_pad_token,
        ]
    )

    assert input_ids.tolist() == [
        *([model.talker_config.tts_codec_mask_token_id] * len(prompt_token_ids)),
        model.talker_config.tts_codec_pad_token_id,
        model.talker_config.tts_codec_start_token_id,
    ]
    torch.testing.assert_close(input_embeds, expected_input_embeds)
    torch.testing.assert_close(updates["embed"]["thinker_reply"], expected_reply_queue)


def test_natural_completion_replaces_stop_row_with_text_eos_and_pad() -> None:
    _assert_conditioning(
        status_name="FINISHED_STOPPED",
        output_token_ids=[2, 3, 63],
        expected_output_ids=[2, 3],
    )


def test_length_capped_completion_appends_text_eos_and_pad() -> None:
    _assert_conditioning(
        status_name="FINISHED_LENGTH_CAPPED",
        output_token_ids=[2, 3, 4],
        expected_output_ids=[2, 3],
    )
