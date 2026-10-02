# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.engine.duplex.intermediate import (
    drop_static_append_configs,
    get_stream_request_key,
    get_tts_handoff,
    pack_transport_tensor,
    set_ref_audio,
    set_tts_handoff,
    unpack_transport_tensor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_get_stream_request_key_requires_stable_identifier():
    with pytest.raises(ValueError, match="stable request id"):
        get_stream_request_key({"ids": {"tts": [1, 2, 3]}})


def test_get_stream_request_key_accepts_global_request_id():
    assert get_stream_request_key({"global_request_id": ["duplex-sid-stage0"]}) == "duplex-sid-stage0"


# ── Tensor transport through model_intermediate_buffer ───────────────


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret as integers so equality is bit-exact (NaN, -0.0)."""
    integer = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()]
    return tensor.contiguous().view(integer)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.randn(7, 16),
        torch.randn(5, 8).to(torch.bfloat16),
        torch.arange(12, dtype=torch.int64),
        torch.tensor(3.5),
        torch.empty(0, 16),
        torch.randn(8, 6).t(),  # non-contiguous
        torch.tensor([float("nan"), -0.0, float("inf"), 1e-45]),
    ],
)
def test_packed_tensor_round_trip_is_bit_exact(tensor):
    packed = pack_transport_tensor(tensor)
    assert isinstance(packed["data"], bytes)
    restored = unpack_transport_tensor(packed)
    assert restored.dtype == tensor.dtype
    assert restored.shape == tensor.shape
    assert torch.equal(_bits(restored), _bits(tensor))
    # Writable, independent storage: the Talker may mutate it freely.
    if restored.numel():
        restored.reshape(-1)[0] = 0
        assert torch.equal(_bits(unpack_transport_tensor(packed)), _bits(tensor))


def test_non_tensor_values_pass_through_unchanged():
    rows = [[0.25, -0.5]]
    assert pack_transport_tensor(rows) is rows
    assert unpack_transport_tensor(rows) is rows
    assert unpack_transport_tensor(None) is None
    plain = {"dtype": "float32", "shape": [1], "data": b"\0\0\0\0"}
    assert unpack_transport_tensor(plain) is plain


def test_drop_static_append_configs_keeps_runtime_fields_and_payload():
    prompt = {
        "prompt_token_ids": [0, 0],
        "model_intermediate_buffer": {
            "duplex": {
                "payload": {"audio": "unit"},
                "scheduler_token_budget": 2,
                "session_config": {"conversation": ["old"]},
                "runtime_config": {"ref_audio_data": "large"},
            }
        },
    }

    assert drop_static_append_configs(prompt) is prompt
    duplex = prompt["model_intermediate_buffer"]["duplex"]
    assert duplex == {"payload": {"audio": "unit"}, "scheduler_token_budget": 2}


def test_packed_handoff_matches_the_legacy_list_transport():
    """The old path shipped ``tolist()`` and rebuilt with ``as_tensor``."""
    hidden = torch.randn(9, 32)
    legacy = torch.as_tensor(hidden.tolist(), dtype=torch.float32)
    buffer: dict = {}
    set_tts_handoff(buffer, [1, 2, 3], hidden)
    token_ids, restored = get_tts_handoff(buffer)
    assert token_ids == [1, 2, 3]
    assert torch.equal(_bits(restored), _bits(legacy))


def test_legacy_list_handoff_is_still_readable():
    buffer = {"ids": {"tts": [5]}, "hidden_states": {"tts": [[0.5, 1.5]]}}
    assert get_tts_handoff(buffer) == ([5], [[0.5, 1.5]])
    assert get_tts_handoff({"tts_token_ids": [4], "tts_hidden_states": [[2.0]]}) == ([4], [[2.0]])


def test_packed_handoff_survives_engine_core_request_msgpack():
    """Round-trip the exact orchestrator -> Stage-1 wire format."""
    from vllm import SamplingParams
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    from vllm_omni.engine import OmniEngineCoreRequest
    from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens

    hidden = torch.randn(33, 64)
    ref = torch.randn(4000)
    buffer: dict = {"global_request_id": ["req"], "meta": {"next_stage_prompt_len": 35}}
    set_tts_handoff(buffer, list(range(33)), hidden)
    set_ref_audio(buffer, ref, 16000)
    request = build_engine_core_request_from_tokens(
        request_id="req",
        prompt={"prompt_token_ids": [0] * 35, "model_intermediate_buffer": buffer},
        params=SamplingParams(max_tokens=4),
    )

    decoded = MsgpackDecoder(OmniEngineCoreRequest).decode(MsgpackEncoder().encode(request))

    info = decoded.model_intermediate_buffer
    token_ids, restored = get_tts_handoff(info)
    assert token_ids == list(range(33))
    assert torch.equal(_bits(restored), _bits(hidden))
    assert torch.equal(_bits(unpack_transport_tensor(info["codes"]["ref"])), _bits(ref))
    assert info["meta"] == {"next_stage_prompt_len": 35, "ref_audio_sr": 16000}
