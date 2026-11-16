# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration as Talker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Stream:
    def __init__(self, slots: int) -> None:
        self.state = torch.zeros(slots, 3)

    def _slot_state(self, slot: int) -> list[torch.Tensor]:
        return [self.state[slot]]


def _talker(cache_size: int) -> SimpleNamespace:
    stream = _Stream(4)
    primed: list[int] = []

    def prime_groups(stream_, primes):
        for slot, ref in primes:
            primed.append(slot)
            stream_.state[slot] = ref.sum().float()

    talker = SimpleNamespace(
        stream_decoder=stream,
        stream_prime_cache=OrderedDict(),
        stream_prime_cache_size=cache_size,
        _stream_prime_key=Talker._stream_prime_key,
        _prime_stream_groups=prime_groups,
        primed=primed,
    )
    return talker


def test_same_reference_is_restored_from_cache():
    talker = _talker(cache_size=8)
    ref = torch.arange(6, dtype=torch.long).reshape(3, 2)
    Talker.prime_stream_decoder(talker, [(0, ref)])
    Talker.prime_stream_decoder(talker, [(1, ref)])
    assert talker.primed == [0]
    assert torch.equal(talker.stream_decoder.state[1], talker.stream_decoder.state[0])


def test_different_reference_is_primed():
    talker = _talker(cache_size=8)
    Talker.prime_stream_decoder(talker, [(0, torch.ones(3, 2, dtype=torch.long))])
    Talker.prime_stream_decoder(talker, [(1, torch.full((3, 2), 2, dtype=torch.long))])
    assert talker.primed == [0, 1]
    assert talker.stream_decoder.state[1, 0] == 12


def test_cache_size_zero_always_primes():
    talker = _talker(cache_size=0)
    ref = torch.ones(3, 2, dtype=torch.long)
    Talker.prime_stream_decoder(talker, [(0, ref)])
    Talker.prime_stream_decoder(talker, [(1, ref)])
    assert talker.primed == [0, 1]
    assert not talker.stream_prime_cache


def test_cached_state_is_a_copy_of_the_slot():
    talker = _talker(cache_size=8)
    ref = torch.ones(3, 2, dtype=torch.long)
    Talker.prime_stream_decoder(talker, [(0, ref)])
    talker.stream_decoder.state[0] = -1  # the slot keeps decoding after priming
    Talker.prime_stream_decoder(talker, [(2, ref)])
    assert talker.stream_decoder.state[2, 0] == 6
