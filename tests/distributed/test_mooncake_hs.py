# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU correctness checks for stored Thinker captures and positional assembly."""

import json
import struct
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save as save_tensors

from vllm_omni.distributed.kv_transfer.mooncake.hs import (
    HSBlockSpec,
    RequestHS,
    decode_hs,
    encode_hs,
    hs_key,
    prefix_digest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def captures(start, end, dtype=torch.float32):
    values = torch.arange(start * 3, end * 3, dtype=dtype).reshape(end - start, 3)
    return {"embedding": values, "hidden": values + 100}


def spec(dtype=torch.float32):
    return HSBlockSpec("model-revision/layout", prefix_digest(b"prefix-hash"), 4, 8, 24, 3, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_exact_roundtrip(dtype):
    original = captures(4, 8, dtype)
    restored = decode_hs(encode_hs(spec(dtype), original), spec(dtype))
    for name in original:
        assert torch.equal(restored[name], original[name])
        assert restored[name].dtype == dtype
        assert restored[name].shape == (4, 3)


@pytest.mark.parametrize(
    "changes",
    [
        {"namespace": "other-model"},
        {"prefix_hash": prefix_digest(b"changed-media")},
        {"start": 0, "end": 4},
        {"hidden_layer": 23},
        {"dtype": torch.float16},
        {"hidden_size": 6},
    ],
)
def test_read_rejects_incompatible_identity_or_layout(changes):
    payload = encode_hs(spec(), captures(4, 8))
    with pytest.raises(ValueError):
        decode_hs(payload, replace(spec(), **changes))


@pytest.mark.parametrize("payload", [b"", b"garbage", struct.pack("<Q", 2**63), struct.pack("<Q", 4) + b"null"])
def test_rejects_malformed_object(payload):
    with pytest.raises(ValueError):
        decode_hs(payload, spec())


@pytest.mark.parametrize("delta", [-1, 1])
def test_rejects_truncated_or_extended_payload(delta):
    payload = encode_hs(spec(), captures(4, 8))
    payload = payload[:-1] if delta < 0 else payload + b"x"
    with pytest.raises(ValueError):
        decode_hs(payload, spec())


def test_rejects_wrong_tensor_names():
    tensors = captures(4, 8)
    tensors["other"] = tensors.pop("hidden")
    payload = save_tensors(tensors, metadata=spec().metadata())
    with pytest.raises(ValueError, match="capture names"):
        decode_hs(payload, spec())


def test_rejects_invalid_offsets():
    payload = encode_hs(spec(), captures(4, 8))
    size = struct.unpack_from("<Q", payload)[0]
    header = json.loads(payload[8 : 8 + size])
    header["hidden"]["data_offsets"] = header["embedding"]["data_offsets"]
    changed = json.dumps(header).encode()
    damaged = struct.pack("<Q", len(changed)) + changed + payload[8 + size :]
    with pytest.raises(ValueError, match="tensor payload"):
        decode_hs(damaged, spec())


def test_prefix_key_separates_namespace_and_hash_and_accepts_rpc_views():
    assert hs_key("model-a", b"prefix") != hs_key("model-b", b"prefix")
    assert hs_key("model-a", b"prefix") != hs_key("model-a", b"different-media")
    assert prefix_digest(b"12") == prefix_digest(memoryview(b"12")) == prefix_digest(bytearray(b"12"))


def test_capture_validation_rejects_misalignment():
    bad = captures(4, 8)
    bad["hidden"] = bad["hidden"][:3]
    with pytest.raises(ValueError, match="matching shapes"):
        encode_hs(spec(), bad)


def test_chunked_prefill_saves_complete_blocks_and_excludes_decode():
    state = RequestHS(prompt_tokens=10, block_size=4, max_bytes=4096)
    first = state.append(0, captures(0, 3))
    assert torch.equal(first["embedding"], captures(0, 3)["embedding"])
    assert state.pop_blocks() == []
    state.append(3, captures(3, 7))
    blocks = state.pop_blocks()
    assert [(start, end) for start, end, _ in blocks] == [(0, 4)]
    assert torch.equal(blocks[0][2]["hidden"], captures(0, 4)["hidden"])
    output = state.append(7, captures(7, 12))
    blocks = state.pop_blocks()
    assert [(start, end) for start, end, _ in blocks] == [(4, 8)]
    assert torch.equal(blocks[0][2]["embedding"], captures(4, 8)["embedding"])
    assert torch.equal(output["hidden"], captures(7, 12)["hidden"])
    assert state.buffered_bytes == 0
    state.append(12, captures(12, 13))
    assert state.pop_blocks() == []


def test_partial_hit_prepends_once_and_only_saves_new_blocks():
    state = RequestHS(prompt_tokens=12, block_size=4, max_bytes=4096)
    state.restore(captures(0, 4))
    first = state.append(4, captures(4, 8))
    assert torch.equal(first["embedding"], captures(0, 8)["embedding"])
    second = state.append(8, captures(8, 13))
    assert torch.equal(second["hidden"], captures(8, 13)["hidden"])
    assert [(start, end) for start, end, _ in state.pop_blocks()] == [(4, 8), (8, 12)]


def test_final_token_replay_uses_new_row_without_duplicate():
    state = RequestHS(prompt_tokens=8, block_size=4, max_bytes=4096)
    state.restore(captures(0, 8))
    new = {name: value + 1000 for name, value in captures(7, 9).items()}
    output = state.append(7, new)
    for name in output:
        assert output[name].shape == (9, 3)
        assert torch.equal(output[name][:7], captures(0, 7)[name])
        assert torch.equal(output[name][7:], new[name])
    assert state.pop_blocks() == []


@pytest.mark.parametrize("start", [3, 5])
def test_rejects_gap_or_duplicate_after_first_step(start):
    state = RequestHS(8, 4, 4096)
    state.append(0, captures(0, 4))
    with pytest.raises(ValueError, match="gap or duplicate"):
        state.append(start, captures(start, start + 1))


def test_rejects_gap_between_restore_and_compute():
    state = RequestHS(12, 4, 4096)
    state.restore(captures(0, 4))
    with pytest.raises(ValueError, match="after the restored prefix"):
        state.append(5, captures(5, 8))


def test_cannot_restore_twice_or_after_output():
    state = RequestHS(12, 4, 4096)
    state.restore(captures(0, 4))
    with pytest.raises(ValueError, match="already installed"):
        state.restore(captures(0, 4))
    state.append(4, captures(4, 8))
    with pytest.raises(ValueError, match="already installed"):
        state.restore(captures(0, 4))


def test_save_budget_exhaustion_preserves_response_rows():
    state = RequestHS(12, 4, max_bytes=4 * 3 * 4 * 2)
    state.append(0, captures(0, 4))
    assert state.buffered_bytes == state.max_bytes
    output = state.append(4, captures(4, 8))
    assert not state.save_enabled
    assert torch.equal(output["embedding"], captures(4, 8)["embedding"])
    assert [(start, end) for start, end, _ in state.pop_blocks()] == [(0, 4)]
    assert state.buffered_bytes == 0


def test_restored_prefix_must_fit_memory_budget():
    state = RequestHS(12, 4, max_bytes=4)
    with pytest.raises(ValueError, match="memory limit"):
        state.restore(captures(0, 4))


def test_saved_blocks_do_not_alias_mutable_capture_buffers():
    state = RequestHS(8, 4, 4096)
    current = captures(0, 5)
    state.append(0, current)
    for tensor in current.values():
        tensor.fill_(-1)
    state.append(5, captures(5, 8))
    blocks = state.pop_blocks()
    for start, end, block in blocks:
        for name in block:
            assert torch.equal(block[name], captures(start, end)[name])


def test_partial_block_reserves_full_capacity_without_copy_on_completion():
    state = RequestHS(8, 4, max_bytes=4 * 3 * 4 * 2)
    state.append(0, captures(0, 1))
    assert state.buffered_bytes == state.max_bytes
    assert state.pop_blocks() == []
    state.append(1, captures(1, 4))
    assert state.buffered_bytes == state.max_bytes
    first = state.pop_blocks()
    assert torch.equal(first[0][2]["embedding"], captures(0, 4)["embedding"])
    assert state.buffered_bytes == 0
    state.append(4, captures(4, 8))
    assert state.save_enabled
    assert state.buffered_bytes == state.max_bytes


def test_save_budget_smaller_than_one_block_does_not_affect_output():
    state = RequestHS(4, 4, max_bytes=1)
    output = state.append(0, captures(0, 4))
    assert torch.equal(output["hidden"], captures(0, 4)["hidden"])
    assert not state.save_enabled
    assert state.buffered_bytes == 0
    assert state.pop_blocks() == []
