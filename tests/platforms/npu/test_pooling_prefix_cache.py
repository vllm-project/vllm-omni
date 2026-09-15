# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pooling logits must not enter the hidden-state prefix cache."""

import pytest
import torch

pytest.importorskip("vllm_ascend")

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.v1.worker.gpu_input_batch import InputBatch

from tests.helpers.mark import hardware_marks
from vllm_omni.core.prefix_cache import OmniTensorPrefixCache
from vllm_omni.platforms.npu.worker import npu_ar_model_runner

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A2"}, num_cards=1)]

NUM_TOKENS = 4
HIDDEN_SIZE = 1024
CLASSIFY_NUM = 5000


@pytest.mark.parametrize("is_pooling", [True, False])
@pytest.mark.parametrize("cache_enabled", [True, False])
def test_pooling_output_does_not_update_hidden_state_cache(mocker, is_pooling, cache_enabled):
    group = mocker.Mock(spec=GroupCoordinator)
    group.is_last_rank = True
    mocker.patch.object(npu_ar_model_runner, "get_pp_group", return_value=group)
    cache = OmniTensorPrefixCache(1, NUM_TOKENS, HIDDEN_SIZE, torch.float32) if cache_enabled else None
    if cache is not None:
        cache.hidden_states_cache.zero_()
    runner = object.__new__(npu_ar_model_runner.NPUARModelRunner)
    runner.is_pooling_model = is_pooling
    runner.omni_prefix_cache = cache
    runner.input_batch = mocker.Mock(spec=InputBatch)
    block_table = mocker.Mock()
    block_table.slot_mapping.gpu = torch.arange(NUM_TOKENS, device="npu")
    runner.input_batch.block_table = [block_table]
    mocker.patch.object(runner, "_model_needs_full_prefix_hidden_states", return_value=True)
    mocker.patch.object(runner, "_deferred_prefix_cache_mm_keys", return_value=set())
    width = CLASSIFY_NUM if is_pooling else HIDDEN_SIZE
    output = torch.ones(NUM_TOKENS, width, device="npu")
    runner._maybe_update_prefix_cache(output, {}, NUM_TOKENS, NUM_TOKENS)
    torch.testing.assert_close(output.cpu(), torch.ones(NUM_TOKENS, width))
    if cache is not None:
        expected = (
            torch.zeros_like(cache.hidden_states_cache) if is_pooling else torch.ones_like(cache.hidden_states_cache)
        )
        torch.testing.assert_close(cache.hidden_states_cache, expected)
