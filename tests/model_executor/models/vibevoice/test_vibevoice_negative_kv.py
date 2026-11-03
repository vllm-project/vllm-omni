# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU lifecycle/metadata contracts; Qwen and FA3 computation are not exercised."""

import pytest
import torch
from vllm.config import VllmConfig
from vllm.forward_context import create_forward_context, get_forward_context, override_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2Model
from vllm.v1.attention.backend import AttentionMetadataBuilder

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.vibevoice.negative_kv import VibeVoiceNegativeKV
from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def pool(mocker):
    config = VllmConfig()
    config.model_config = mocker.Mock(
        spec=OmniModelConfig,
        max_model_len=32,
        hf_config=VibeVoiceConfig(
            text_config={"hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1}
        ),
    )
    config.cache_config.block_size = 16
    config.scheduler_config.max_num_seqs = 3
    attention = mocker.Mock(spec=Attention, layer_name="test.attention", kv_cache=torch.zeros(1))
    layer = mocker.Mock(spec=Qwen2DecoderLayer)
    layer.self_attn = mocker.Mock(spec=Qwen2Attention, attn=attention)
    language_model = mocker.Mock(spec=Qwen2Model, layers=[layer])
    language_model.side_effect = lambda **kwargs: kwargs["inputs_embeds"]
    pool = VibeVoiceNegativeKV(language_model=language_model, vllm_config=config)
    # Bypass CUDA-only allocation, not the constructor or lifecycle/forward_step logic.
    pool._device, pool._dtype = torch.device("cpu"), torch.bfloat16
    pool._kv_caches = [torch.zeros(6, 1, 16, 8, dtype=pool._dtype)]
    pool._block_tables = torch.arange(6, dtype=torch.int32).reshape(3, 2)
    pool._metadata_builder = mocker.Mock(spec=AttentionMetadataBuilder)
    return pool


def test_slot_release_reuse_and_segment_reset(pool):
    row = torch.ones(1, 4, dtype=torch.bfloat16)
    for request_id in ("a", "b", "c"):
        pool.acquire(request_id)
    pool.acquire("a")  # Acquiring the same owner twice must not consume a slot.
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.acquire("d")
    pool.forward_step(["a", "b"], [row, row])
    pool.reset_segment("a")
    assert pool.sequence_length("a") == 0
    assert pool.sequence_length("b") == 1
    assert pool._by_request == {"a": 0, "b": 1, "c": 2}

    pool.release("a")
    pool.release("a")
    pool.acquire("d")
    assert pool._by_request == {"b": 1, "c": 2, "d": 0}
    assert pool.sequence_length("d") == 0
    pool.forward_step(["d"], [row])
    assert pool._language_model.call_args.kwargs["positions"].tolist() == [0]
    assert pool.sequence_length("b") == 1


def test_reordered_rows_cross_pages_and_reject_overflow(pool):
    row = torch.ones(1, 4, dtype=torch.bfloat16)
    pool.acquire("a")
    pool.acquire("b")
    for _ in range(16):
        pool.forward_step(["a"], [row])
    original_kv = pool._attention_layers[0].kv_cache
    outer = create_forward_context(None, pool._vllm_config)
    with override_forward_context(outer):
        outputs = pool.forward_step(["b", "a"], [2 * row, row])
        assert get_forward_context() is outer
    assert pool._attention_layers[0].kv_cache is original_kv
    common = pool._metadata_builder.build.call_args.args[1]
    assert common.positions.tolist() == [0, 16]
    assert common.seq_lens.tolist() == [1, 17]
    assert common.slot_mapping.tolist() == [32, 16]
    assert common.block_table_tensor.tolist() == [[2, 3], [0, 1]]
    # Returned rows must own their data rather than alias the backbone output.
    pool._language_model.call_args.kwargs["inputs_embeds"].zero_()
    torch.testing.assert_close(outputs[0], 2 * row, atol=0, rtol=0)
    torch.testing.assert_close(outputs[1], row, atol=0, rtol=0)

    for _ in range(15):
        pool.forward_step(["a"], [row])
    calls = pool._language_model.call_count
    with pytest.raises(RuntimeError, match="exceeded capacity"):
        pool.forward_step(["b", "a"], [row, row])
    assert pool._language_model.call_count == calls
    assert pool.sequence_length("a") == 32
    assert pool.sequence_length("b") == 1


@pytest.mark.parametrize("failure", ["metadata", "forward"])
def test_failure_restores_bindings_and_invalidates_only_touched_owners(pool, failure):
    row = torch.ones(1, 4, dtype=torch.bfloat16)
    for request_id in ("a", "b", "c"):
        pool.acquire(request_id)
    pool.forward_step(["c"], [row])
    attention = pool._attention_layers[0]
    original_kv = attention.kv_cache
    outer = create_forward_context(None, pool._vllm_config)

    def fail_forward(**kwargs):
        assert get_forward_context() is not outer
        assert attention.kv_cache is pool._kv_caches[0]
        attention.kv_cache[0].fill_(1)  # Simulate a partial write before failure.
        raise RuntimeError("injected failure")

    if failure == "metadata":
        pool._metadata_builder.build.side_effect = RuntimeError("injected failure")
    else:
        pool._language_model.side_effect = fail_forward
    with override_forward_context(outer):
        with pytest.raises(RuntimeError, match="injected failure"):
            pool.forward_step(["b", "a"], [row, row])
        assert get_forward_context() is outer
    assert attention.kv_cache is original_kv
    assert pool._by_request == {"c": 2}
    assert pool.sequence_length("c") == 1
    assert pool._slots[0].owner is None and pool._slots[0].length == 0
    assert pool._slots[1].owner is None and pool._slots[1].length == 0
    pool.acquire("d")
    assert pool._by_request["d"] == 0 and pool.sequence_length("d") == 0
