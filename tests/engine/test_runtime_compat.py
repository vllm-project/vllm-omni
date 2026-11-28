# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Legacy/current vLLM contracts and full H3 text-prefix output replay."""

import pytest
import torch
from vllm.config import ModelConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.core.prefix_cache.interface import StageCacheOutputs
from vllm_omni.distributed.omni_connectors.utils.config import connector_hash_factors
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.model_executor.models.minimax_h3.encoder import MiniMaxH3Encoder
from vllm_omni.outputs import ModelRunnerOutput, OmniModelRunnerOutput
from vllm_omni.request import OmniRequest
from vllm_omni.worker_v2.prefix_cache import MRv2PrefixCacheMixin, PrefixCacheRewrite

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("method", ["compute_hash", "compile_factors"])
def test_cache_api_delegates_without_mutating_connector_configuration(monkeypatch, method):
    config = object.__new__(OmniModelConfig)
    config.stage_connector_config = {"name": "SharedMemoryConnector", "extra": {"shm_name": "runtime-id"}}
    config.omni_kv_config = {"connector_config": config.stage_connector_config.copy()}
    original = config.stage_connector_config.copy()
    seen = []

    def base(normalized):
        seen.append(normalized)
        return (
            "cache-key"
            if method == "compute_hash"
            else {"cache": "factor", "scheduler_window_policy": "runtime.policy"}
        )

    monkeypatch.setattr(ModelConfig, method, base, raising=False)
    result = getattr(config, method)()
    assert result == ("cache-key" if method == "compute_hash" else {"cache": "factor"})
    assert seen[0] is not config
    assert seen[0].stage_connector_config == connector_hash_factors(original)
    assert seen[0].omni_kv_config["connector_config"] == connector_hash_factors(original)
    assert config.stage_connector_config == original
    assert config.omni_kv_config["connector_config"] == original


def core_request():
    return OmniEngineCoreRequest.from_request(
        EngineCoreRequest(
            request_id="compat-request",
            prompt_token_ids=[1, 2],
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
    )


def test_legacy_request_without_frontend_hashes_constructs():
    result = OmniRequest.from_engine_core_request(core_request(), block_hasher=None)
    assert result.request_id == "compat-request" and result.prompt_token_ids == [1, 2]
    assert result.block_hashes == []


@pytest.mark.parametrize("hashes", [None, [], ["frontend-block"]])
def test_only_explicit_frontend_hashes_are_forwarded(hashes):
    class RecordingRequest(OmniRequest):
        def __init__(self, **kwargs):
            self.received = kwargs

    result = RecordingRequest.from_engine_core_request(core_request(), None, hashes)
    if hashes is None:
        assert "block_hashes" not in result.received
    else:
        assert result.received["block_hashes"] is hashes


def test_connector_only_output_constructs():
    token = object()
    output = OmniModelRunnerOutput.with_kv_conn_output_only(token)
    assert output.req_ids == [] and output.kv_connector_output is token


def test_legacy_index_tracks_output_order(monkeypatch):
    monkeypatch.setattr(
        ModelRunnerOutput, "__dataclass_fields__", {**ModelRunnerOutput.__dataclass_fields__, "req_id_to_index": None}
    )

    class LegacyOutput(OmniModelRunnerOutput):
        def __init__(self, *, req_id_to_index, **kwargs):
            self.received_index = req_id_to_index

    output = LegacyOutput._from_components(req_ids=["second", "first"], sampled_token_ids=[[], []])
    assert output.received_index == {"second": 0, "first": 1}


def test_new_output_layout_does_not_receive_legacy_field(monkeypatch):
    fields = {k: v for k, v in ModelRunnerOutput.__dataclass_fields__.items() if k != "req_id_to_index"}
    monkeypatch.setattr(ModelRunnerOutput, "__dataclass_fields__", fields)

    class RecordingOutput(OmniModelRunnerOutput):
        def __init__(self, **kwargs):
            self.received = kwargs

    output = RecordingOutput._from_components(req_ids=["a"], sampled_token_ids=[[]])
    assert "req_id_to_index" not in output.received


def test_h3_enables_multimodal_prefix_output_replay():
    runner = MRv2PrefixCacheMixin()
    runner._snapshot_prefix_cache_model_policy(MiniMaxH3Encoder)
    assert not runner._omni_cache_policy.native_kv_cache_only
    # The named hidden_states.output payload is cached; duplicating it as
    # generic "hidden" would waste storage and add the payload twice.
    assert not runner._omni_cache_policy.needs_full_hidden_states


def test_text_prefix_and_role_labels_replace_the_scheduled_tail():
    full = torch.arange(32).reshape(8, 4).to(torch.bfloat16)
    tags = torch.tensor([[1], [1], [0], [0], [1], [0], [1], [1]])
    tail = {"hidden_states.output": full[6:].clone(), "meta.token_role_ids": tags[6:].clone()}

    class Cache:
        def materialize(self, step_id, req_ids, *, include_uncached):
            assert (step_id, req_ids, include_uncached) == (1, ["request"], False)
            return StageCacheOutputs(
                hidden_states=None,
                mm_outputs={
                    "hidden_states.output": {"request": full},
                    "meta.token_role_ids": {"request": tags},
                },
            )

    output = OmniModelRunnerOutput._from_components(
        req_ids=["request"],
        sampled_token_ids=[[]],
        inter_stage_outputs=[tail],
        pooler_output=[tail],
        multimodal_outputs=[tail],
    )
    PrefixCacheRewrite(Cache(), 1, ((0, "request"),)).apply(output)
    assert torch.equal(tail["hidden_states.output"], full)
    assert torch.equal(tail["meta.token_role_ids"], tags)
    assert output.full_payload_prefix_cache_keys["request"] == frozenset(tail)


def test_nonmember_encoder_rank_joins_creation_without_coordinator(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    pipeline = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._dit_rank = 12
    calls = []
    monkeypatch.setattr(torch.distributed, "new_group", lambda ranks, **kw: calls.append((ranks, kw["backend"])))

    def reject_nonmember(**kwargs):
        pytest.fail("a rank outside encoder TP must not construct its coordinator")

    monkeypatch.setattr(pipeline_module, "init_world_group", reject_nonmember)
    group = pipeline._build_text_encoder_group(8)
    assert calls == [(list(range(8)), pipeline_module.current_omni_platform.dist_backend), (list(range(8)), "gloo")]
    assert group.world_size == 1


@pytest.mark.parametrize("need_recv,role", [(False, "receiver"), (True, None)])
def test_shared_memory_receiver_is_visible_before_first_payload(monkeypatch, tmp_path, need_recv, role):
    from vllm_omni.distributed.omni_connectors import kv_transfer_manager as transfer
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector

    monkeypatch.setattr(SharedMemoryConnector, "shm_directory", str(tmp_path))
    monkeypatch.setattr(transfer, "get_local_tp_rank", lambda: 0)
    monkeypatch.setattr(transfer, "get_tp_world_size", lambda: 1)
    sender = SharedMemoryConnector({"stage_id": 0, "wakeup_scope": "runtime-test"})
    manager = transfer.OmniKVTransferManager(
        transfer.OmniKVCacheConfig(
            stage_id=1,
            need_recv_cache=need_recv,
            connector_config={"type": "SharedMemoryConnector", "role": role, "wakeup_scope": "runtime-test"},
        )
    )
    try:
        assert sender._peer_visible("1")
        assert manager.connector.stage_id == 1
    finally:
        manager.close()
        sender.close()
