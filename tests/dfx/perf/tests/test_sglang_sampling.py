# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract checks for the comparison image's patched SGLang environment."""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# The benchmark client environment deliberately does not install SGLang.
pytest.importorskip("sglang_omni")


@pytest.mark.parametrize("ignore_eos", [None, False, True])
def test_chat_ignore_eos_reaches_thinker_sampler(ignore_eos: bool | None) -> None:
    import torch
    from sglang_omni.client.client import build_params
    from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
    from sglang_omni.models.qwen3_omni.request_builders import build_sglang_thinker_request
    from sglang_omni.serve.openai_api import build_chat_generate_request
    from sglang_omni.serve.protocol import ChatCompletionRequest

    body = {
        "messages": [{"role": "user", "content": "Read this aloud."}],
        "modalities": ["text", "audio"],
        "max_tokens": 100,
        "temperature": 0.0,
    }
    if ignore_eos is not None:
        body["ignore_eos"] = ignore_eos
    request = build_chat_generate_request(ChatCompletionRequest.model_validate(body))
    params = build_params(request)
    assert params["ignore_eos"] is (ignore_eos is True)
    state = Qwen3OmniPipelineState(
        prompt={
            "prompt_text": "Read this aloud.",
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.ones(3),
        }
    )
    thinker = build_sglang_thinker_request(state, params=params, tokenizer=None, vocab_size=256)
    assert thinker.req.sampling_params.ignore_eos is (ignore_eos is True)
    assert thinker.req.sampling_params.max_new_tokens == 100
