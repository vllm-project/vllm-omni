# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for MammothModa2 prefix caching and the AR-to-DiT bridge."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache
from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2dit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _InputBatch:
    def __init__(self, block_ids: torch.Tensor, num_computed_tokens: int):
        block_table = SimpleNamespace(cpu=block_ids)
        block_group = SimpleNamespace(block_table=block_table)
        self.block_table = _BlockTable(block_table, block_group)
        self.req_ids = ["request-0"]
        self.req_id_to_index = {"request-0": 0}
        self.num_computed_tokens_cpu = torch.tensor([num_computed_tokens])


class _BlockTable:
    def __init__(self, block_table, block_group):
        self.block_tables = [block_table]
        self._block_group = block_group

    def __getitem__(self, index):
        assert index == 0
        return self._block_group


def _ar_output(prompt_token_ids: list[int], generated_token_ids: list[int], hidden_states: torch.Tensor):
    completion = SimpleNamespace(
        cumulative_token_ids=generated_token_ids,
        multimodal_output={"latent": hidden_states},
    )
    return SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=prompt_token_ids,
        outputs=[completion],
    )


def test_prefix_cache_miss_hit_preserves_ar_to_dit_alignment():
    block_size = 4
    hidden_size = 3
    cached_tokens = 8
    cache = OmniTensorPrefixCache(
        num_blocks=8,
        block_size=block_size,
        hidden_size=hidden_size,
        hs_dtype=torch.float32,
    )

    cached_hidden = torch.arange(cached_tokens * hidden_size, dtype=torch.float32).reshape(cached_tokens, hidden_size)
    cached_slots = torch.arange(2 * block_size, 4 * block_size)
    cache.update_omni_tensor_prefix_cache(
        hidden_states=cached_hidden,
        multimodal_outputs=None,
        num_tokens_unpadded=cached_tokens,
        slot_mapping=cached_slots,
    )

    new_hidden = torch.arange(12, dtype=torch.float32).reshape(4, hidden_size) + 100
    miss_hidden = torch.cat([cached_hidden, new_hidden], dim=0)

    input_batch = _InputBatch(
        block_ids=torch.tensor([[2, 3]], dtype=torch.long),
        num_computed_tokens=cached_tokens,
    )
    cache.add_prefix_cached_new_req_id("request-0")
    hit_hidden = cache.get_merged_hidden_states(
        query_start_loc=torch.tensor([0]),
        input_batch=input_batch,
        hidden_states=new_hidden,
        num_scheduled_tokens={"request-0": len(new_hidden)},
    )["request-0"]

    prompt_token_ids = list(range(10))
    # ar2dit intentionally drops the final generated token because no hidden
    # state is produced for it.
    generated_token_ids = [20, 21, 22]
    prompt = {"additional_information": {"image_height": [256], "image_width": [256]}}
    miss = ar2dit([_ar_output(prompt_token_ids, generated_token_ids, miss_hidden)], prompt)[0]
    hit = ar2dit([_ar_output(prompt_token_ids, generated_token_ids, hit_hidden)], prompt)[0]

    miss_info = miss["additional_information"]
    hit_info = hit["additional_information"]
    assert hit_info["full_token_ids"] == prompt_token_ids + generated_token_ids[:-1]
    assert hit_info["answer_start_index"] == [len(prompt_token_ids)]
    assert torch.equal(hit_info["full_hidden_states"], miss_info["full_hidden_states"])
    assert hit_info["full_hidden_states"].shape[0] == len(hit_info["full_token_ids"])
