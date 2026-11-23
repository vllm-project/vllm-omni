# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject invalid codec requests before slot or device mutations."""

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.platforms.npu.layers.incremental_codec import NPUIncrementalCodec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_decoder_conversion_releases_incremental_graphs(mocker):
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder

    decoder = _make_decoder()
    arena = mocker.Mock()
    decoder._npu_incremental = arena
    decoder.to(dtype=torch.bfloat16)
    arena.close.assert_called_once_with()
    assert decoder._npu_incremental is None


@pytest.mark.parametrize(
    "case", ["shared_cache", "shared_slot", "free_slot", "bad_slot", "length", "codebooks", "capacity"]
)
def test_invalid_requests_cannot_partially_mutate_state(case):
    arena = object.__new__(NPUIncrementalCodec)
    arena.decoder = Mock()
    arena.decoder.config.num_quantizers = 2
    arena.slots = 4
    arena.free = [2, 3]
    arena.alloc = Mock(side_effect=AssertionError("must validate before allocation"))
    arena.reset = Mock(side_effect=AssertionError("must validate before reset"))
    codes = torch.zeros(2, 2, 4, dtype=torch.long)
    caches = [{"npu_slot": 0}, {"npu_slot": 1}]
    lengths = [4, 4]
    if case == "shared_cache":
        caches = [caches[0], caches[0]]
    elif case == "shared_slot":
        caches[1]["npu_slot"] = 0
    elif case == "free_slot":
        caches[1]["npu_slot"] = 2
    elif case == "bad_slot":
        caches[1]["npu_slot"] = 4
    elif case == "length":
        lengths[1] = 5
    elif case == "codebooks":
        codes = codes[:, :1]
    elif case == "capacity":
        caches = [{}, {}]
        arena.free = [3]
    before = [dict(cache) for cache in caches]
    with pytest.raises((ValueError, RuntimeError)):
        arena.decode_requests(codes, lengths, caches)
    assert caches == before
    arena.alloc.assert_not_called()
    arena.reset.assert_not_called()
