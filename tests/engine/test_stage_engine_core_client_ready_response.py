import msgspec
import pytest

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ready_response_uses_strict_vllm_025_schema(monkeypatch):
    def fail_upstream(self, payload):
        raise msgspec.ValidationError("Object missing required field `block_size`")

    monkeypatch.setattr(
        "vllm_omni.engine.stage_engine_core_client.MPClient._apply_ready_response",
        fail_upstream,
    )

    client = object.__new__(StageEngineCoreClient)
    payload = msgspec.msgpack.encode(
        {
            "max_model_len": 32768,
            "num_gpu_blocks": 11,
            "dp_stats_address": "tcp://127.0.0.1:1234",
            "kv_cache_config": {"num_blocks": 1},
        }
    )

    with pytest.raises(msgspec.ValidationError, match="block_size"):
        client._apply_ready_response(payload)
