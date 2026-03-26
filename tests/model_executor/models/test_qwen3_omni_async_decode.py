# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration


def _make_model() -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.talker = SimpleNamespace(
        num_code_groups=2,
        text_projection=lambda x: x + 10,
    )
    model.tts_eos_embed = torch.tensor([[100.0, 101.0]], dtype=torch.bfloat16)
    model.tts_pad_embed = torch.tensor([[200.0, 201.0]], dtype=torch.bfloat16)
    return model


def test_talker_preprocess_decode_starts_from_prefill_consumed_boundary():
    model = _make_model()
    observed = {}

    def fake_decode(input_ids, input_embeds, update_dict, **info_dict):
        observed["num_processed_tokens"] = info_dict["num_processed_tokens"]
        update_dict["num_processed_tokens_delta"] = 1
        return torch.zeros((1, 2), dtype=torch.bfloat16), torch.ones((1, 2), dtype=torch.bfloat16), update_dict

    model.talker_preprocess_decode = fake_decode

    _, _, update_dict = model.talker_preprocess(
        input_ids=torch.tensor([1], dtype=torch.long),
        input_embeds=torch.ones((1, 2), dtype=torch.bfloat16),
        decode_flag=False,
        num_processed_tokens=0,
        prefill_consumed_text_tokens=1,
    )

    assert observed["num_processed_tokens"] == 1
    assert update_dict["decode_flag"] is True
    assert update_dict["num_processed_tokens"] == 2


def test_async_decode_terminal_steps_do_not_advance_processed_tokens():
    model = _make_model()
    device = torch.device("cpu")

    update_dict = {}
    text_step = model._thinker_decode_to_talker_decode(
        {
            "cached_thinker_decode_embeddings": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
            "num_processed_tokens": 1,
            "thinker_output_token_ids": [11, 12],
        },
        device,
        update_dict,
    )

    assert torch.equal(text_step, model.tts_eos_embed)
    assert update_dict["finished_flag"] is True
    assert update_dict["num_processed_tokens_delta"] == 0

    update_dict = {}
    text_step = model._thinker_decode_to_talker_decode(
        {
            "cached_thinker_decode_embeddings": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
            "num_processed_tokens": 1,
            "thinker_output_token_ids": [11, 12],
            "finished_flag": True,
        },
        device,
        update_dict,
    )

    assert torch.equal(text_step, model.tts_pad_embed)
    assert update_dict["num_processed_tokens_delta"] == 0


def test_async_decode_consumes_cached_embedding_and_appends_new_runtime_embed():
    model = _make_model()
    device = torch.device("cpu")
    update_dict = {}

    text_step = model._thinker_decode_to_talker_decode(
        {
            "cached_thinker_decode_embeddings": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
            "thinker_decode_embeddings": torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
            "num_processed_tokens": 0,
            "thinker_output_token_ids": [11, 12, 13],
        },
        device,
        update_dict,
    )

    assert torch.equal(text_step, torch.tensor([11.0, 12.0], dtype=torch.bfloat16))
    assert torch.equal(
        update_dict["cached_thinker_decode_embeddings"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
    )
    assert update_dict["num_processed_tokens_delta"] == 1


def test_async_decode_consumes_runtime_embedding_when_cache_is_empty():
    model = _make_model()
    device = torch.device("cpu")
    update_dict = {}

    text_step = model._thinker_decode_to_talker_decode(
        {
            "thinker_decode_embeddings": torch.tensor([[5.0, 6.0]], dtype=torch.bfloat16),
            "num_processed_tokens": 0,
            "thinker_output_token_ids": [11, 12, 13],
        },
        device,
        update_dict,
    )

    assert torch.equal(text_step, torch.tensor([15.0, 16.0], dtype=torch.bfloat16))
    assert update_dict["thinker_decode_embeddings"] is None
    assert update_dict["num_processed_tokens_delta"] == 1
