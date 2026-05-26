# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm_omni.distributed.omni_connectors.utils import qwen3_transfer_packet as pkt
from vllm_omni.model_executor.stage_input_processors import qwen3_omni as q3


def _sample_thinker_payload() -> dict:
    return {
        "embed": {
            "prefill": torch.ones(2, 4),
            "tts_bos": torch.zeros(1, 4),
        },
        "hidden_states": {"output": torch.full((2, 4), 2.0)},
        "ids": {"all": [1, 2, 3], "prompt": [1, 2]},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
        "next_stage_prompt_len": 7,
        "speaker": "alice",
    }


def test_split_and_reconstruct_roundtrip() -> None:
    payload = _sample_thinker_payload()
    tensor_puts, sidecar = pkt.split_thinker_to_talker_full_payload(
        payload,
        request_id="req-1",
        external_req_id="ext-1",
        from_stage_id=0,
        to_stage_id=1,
        chunk_id=0,
    )
    assert sidecar["sidecar_put_key"] == "ext-1_0_0"
    assert len(tensor_puts) == 3
    store = {key: tensor for key, tensor in tensor_puts}
    rebuilt = pkt.reconstruct_thinker_to_talker_full_payload(
        sidecar,
        lambda transfer_key: store.get(transfer_key),
    )
    assert rebuilt["ids"] == payload["ids"]
    assert rebuilt["next_stage_prompt_len"] == 7
    assert rebuilt["speaker"] == "alice"
    assert torch.equal(rebuilt["embed"]["prefill"], payload["embed"]["prefill"])
    assert torch.equal(rebuilt["hidden_states"]["output"], payload["hidden_states"]["output"])
    assert bool(rebuilt["meta"]["finished"].item()) is True


def test_reconstruct_clones_managed_buffer_before_release() -> None:
    from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
        ManagedBuffer,
    )

    class DummyAllocator:
        def __init__(self) -> None:
            self.freed: list[tuple[int, int]] = []

        def free(self, offset: int, size: int) -> None:
            self.freed.append((offset, size))

    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    pool = source.view(torch.uint8).clone()
    allocator = DummyAllocator()
    buffer = ManagedBuffer(allocator, 0, pool.numel(), pool)
    sidecar = {
        "packet_version": pkt.PACKET_VERSION,
        "payload_kind": pkt.PAYLOAD_KIND_THINKER_TO_TALKER_FULL,
        "tensor_entries": [
            {
                "name": "embed.prefill",
                "dtype": str(source.dtype),
                "shape": list(source.shape),
                "transfer_key": "tensor-key",
            }
        ],
        "metadata": {},
    }

    rebuilt = pkt.reconstruct_qwen3_full_payload(sidecar, lambda _key: buffer)

    assert allocator.freed == [(0, pool.numel())]
    assert torch.equal(rebuilt["embed"]["prefill"], source)

    # Simulate the MTE receive pool reusing the released buffer for the next
    # tensor.  The reconstructed payload must not alias that pool memory.
    pool.fill_(0)
    assert torch.equal(rebuilt["embed"]["prefill"], source)


def test_should_use_packet_path_gate() -> None:
    assert pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=True,
        supports_raw_data=True,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=0,
        to_stage_id=1,
    )
    assert not pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=False,
        supports_raw_data=True,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=0,
        to_stage_id=1,
    )
    assert pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=False,
        supports_raw_data=True,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=0,
        to_stage_id=1,
        transfer_mode=pkt.MODE_NON_ASYNC_FULL_PAYLOAD,
    )
    assert not pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=True,
        supports_raw_data=False,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=0,
        to_stage_id=1,
    )
    assert not pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=True,
        supports_raw_data=True,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=0,
        to_stage_id=2,
    )
    assert pkt.should_use_thinker_to_talker_packet_path(
        async_chunk=False,
        supports_raw_data=True,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        from_stage_id=1,
        to_stage_id=2,
        transfer_mode=pkt.MODE_NON_ASYNC_FULL_PAYLOAD,
    )


def test_payload_has_packet_tensors() -> None:
    assert pkt.payload_has_packet_tensors(_sample_thinker_payload())
    assert pkt.payload_has_packet_tensors({"codes": {"audio": [1, 2, 3]}})
    assert not pkt.payload_has_packet_tensors({"ids": {"all": [1]}, "meta": {"finished": True}})


def test_talker_to_code2wav_split_and_reconstruct_roundtrip() -> None:
    payload = {
        "codes": {"audio": [1, 2, 3, 4]},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool), "left_context_size": 25},
    }
    tensor_puts, sidecar = pkt.split_qwen3_full_payload(
        payload,
        request_id="req-2",
        external_req_id="ext-2",
        from_stage_id=1,
        to_stage_id=2,
        chunk_id=0,
        mode=pkt.MODE_NON_ASYNC_FULL_PAYLOAD,
    )
    assert sidecar["payload_kind"] == pkt.PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL
    assert len(tensor_puts) == 1
    store = {key: tensor for key, tensor in tensor_puts}
    rebuilt = pkt.reconstruct_qwen3_full_payload(
        sidecar,
        lambda transfer_key: store.get(transfer_key),
    )
    assert "codes" in rebuilt and "audio" in rebuilt["codes"]
    assert rebuilt["code_predictor_codes"] == [1, 2, 3, 4]


def test_thinker2talker_full_payload_processor_roundtrip() -> None:
    request = SimpleNamespace(
        request_id="thinker",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3],
        all_token_ids=[151644, 872, 3],
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(3, 2),
        "hidden_states.layer_24": torch.full((3, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }
    payload = q3.thinker2talker_full_payload(None, pooling_output, request)
    assert payload is not None

    tensor_puts, sidecar = pkt.split_thinker_to_talker_full_payload(
        payload,
        request_id="thinker",
        external_req_id="thinker",
        from_stage_id=0,
        to_stage_id=1,
        chunk_id=0,
    )
    store = {key: tensor for key, tensor in tensor_puts}
    rebuilt = pkt.reconstruct_thinker_to_talker_full_payload(sidecar, store.get)
    assert rebuilt["ids"]["all"] == payload["ids"]["all"]
    assert rebuilt["embed"]["prefill"].shape == payload["embed"]["prefill"].shape
    assert rebuilt["hidden_states"]["output"].shape == payload["hidden_states"]["output"].shape
    assert rebuilt["next_stage_prompt_len"] == payload["next_stage_prompt_len"]
