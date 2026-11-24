# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.minimax_music3 import MiniMaxMusic3Adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_prompt_leaves_room_for_one_frame_and_delivery():
    engine = Mock()
    engine.model_config.max_model_len = 10
    adapter = MiniMaxMusic3Adapter(SpeechServingContext(server=Mock(), engine_client=engine))
    tokenizer = Mock()
    adapter._cached_tokenizer = tokenizer
    request = OpenAICreateSpeechRequest(input="lyrics", instructions="piano")

    tokenizer.encode.return_value = list(range(8))
    prepared = await adapter.build(request, [], False)
    assert prepared.prompt["prompt_token_ids"] == list(range(8))

    tokenizer.encode.return_value = list(range(9))
    with pytest.raises(ValueError, match="the maximum is 8"):
        await adapter.build(request, [], False)
