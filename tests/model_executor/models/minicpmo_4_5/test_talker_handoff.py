# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Thinker->Talker handoff over the stage connector.

Producer put -> request carries a marker -> consumer get -> the Talker sees a
tensor; abort/TTL cleanup; and the default list path when nobody opts in.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.model_executor.models.minicpmo_4_5 import talker_handoff as th
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.talker_handoff import (
    HANDOFF_MARKER,
    HANDOFF_OPTION,
    TalkerHandoffProducer,
    get_talker_handoff_producer,
    handoff_connector_spec,
    is_handoff_marker,
    resolve_talker_handoff,
)
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import llm2tts

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _talker_model_config(*, enabled: bool = True, **extra) -> SimpleNamespace:
    """A stage-1 model config the way the orchestrator client and worker see it."""
    connector_extra = {"stage_id": 1, "role": "sender", "codec_chunk_frames": 25, **extra}
    if enabled:
        connector_extra[HANDOFF_OPTION] = True
    return SimpleNamespace(stage_connector_config={"name": "SharedMemoryConnector", "extra": connector_extra})


@pytest.fixture(autouse=True)
def _isolated_producers():
    saved = dict(th._producers)
    th._producers.clear()
    yield
    for producer in th._producers.values():
        producer.close()
    th._producers.clear()
    th._producers.update(saved)


@pytest.fixture()
def producer():
    instance = TalkerHandoffProducer(SharedMemoryConnector({}), ttl_s=600)
    yield instance
    instance.close()


@pytest.fixture()
def consumer():
    connector = SharedMemoryConnector({})
    yield connector
    connector.close()


def _info(marker) -> dict:
    return {"ids": {"tts": [11, 12]}, "hidden_states": {"tts": marker}, "meta": {}}


class TestProducerConsumer:
    def test_marker_resolves_to_the_put_tensor_and_is_cached_in_place(self, producer, consumer):
        hidden = torch.randn(7, 8)
        marker = producer.put("req-1", hidden)

        assert is_handoff_marker(marker)
        assert marker[HANDOFF_MARKER]["key"].startswith("mcpo45_handoff_req-1_")
        info = _info(marker)
        resolved = resolve_talker_handoff(info, consumer)

        assert torch.equal(resolved, hidden)
        # Later prefill chunks read the tensor from the runner buffer.
        assert info["hidden_states"]["tts"] is resolved
        assert resolve_talker_handoff(info, consumer) is None

    def test_a_claimed_handoff_cannot_be_read_twice(self, producer, consumer):
        marker = producer.put("req-1", torch.ones(2, 3))
        resolve_talker_handoff(_info(marker), consumer)

        with pytest.raises(ValueError, match="already consumed"):
            resolve_talker_handoff(_info(marker), consumer)

    def test_reputs_for_one_request_get_distinct_keys(self, producer, consumer):
        first = producer.put("req/1", torch.zeros(1, 2))
        second = producer.put("req/1", torch.ones(1, 2))

        assert first[HANDOFF_MARKER]["key"] != second[HANDOFF_MARKER]["key"]
        assert "/" not in first[HANDOFF_MARKER]["key"]
        assert torch.equal(resolve_talker_handoff(_info(first), consumer), torch.zeros(1, 2))
        assert torch.equal(resolve_talker_handoff(_info(second), consumer), torch.ones(1, 2))

    def test_cleanup_by_request_unlinks_only_that_requests_handoffs(self, producer, consumer):
        aborted = [producer.put("req-a", torch.zeros(2, 2)) for _ in range(2)]
        alive = producer.put("req-b", torch.ones(2, 2))

        producer.cleanup("req-a")

        assert producer.pending_keys == [alive[HANDOFF_MARKER]["key"]]
        for marker in aborted:
            with pytest.raises(ValueError, match="unavailable"):
                resolve_talker_handoff(_info(marker), consumer)
        assert torch.equal(resolve_talker_handoff(_info(alive), consumer), torch.ones(2, 2))

    def test_unconsumed_handoff_expires_after_ttl(self, consumer, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(th, "time", SimpleNamespace(monotonic=lambda: now[0]))
        producer = TalkerHandoffProducer(SharedMemoryConnector({}), ttl_s=10)
        try:
            stale = producer.put("req-1", torch.zeros(1, 1))
            now[0] += 5
            fresh = producer.put("req-1", torch.ones(1, 1))
            now[0] += 6  # stale is 11 s old, fresh 6 s
            latest = producer.put("req-2", torch.full((1, 1), 2.0))

            assert producer.pending_keys == [fresh[HANDOFF_MARKER]["key"], latest[HANDOFF_MARKER]["key"]]
            with pytest.raises(ValueError, match="expired"):
                resolve_talker_handoff(_info(stale), consumer)
            assert torch.equal(resolve_talker_handoff(_info(fresh), consumer), torch.ones(1, 1))
        finally:
            producer.close()

    def test_close_unlinks_everything_left(self, consumer):
        producer = TalkerHandoffProducer(SharedMemoryConnector({}), ttl_s=600)
        marker = producer.put("req-1", torch.zeros(3))

        producer.close()

        assert consumer.get("0", "1", marker[HANDOFF_MARKER]["key"]) is None

    def test_put_failure_returns_none_for_the_list_fallback(self):
        class _Refusing(SharedMemoryConnector):
            def put(self, *args, **kwargs):
                return False, 0, None

        producer = TalkerHandoffProducer(_Refusing({}), ttl_s=600)
        try:
            assert producer.put("req-1", torch.zeros(2)) is None
            assert producer.pending_keys == []
        finally:
            producer.close()

    def test_consumed_keys_are_dropped_from_pending_on_expiry(self, producer, consumer, monkeypatch):
        marker = producer.put("req-1", torch.zeros(1))
        resolve_talker_handoff(_info(marker), consumer)
        # Expiry of an already-claimed key must be a no-op, not an error.
        monkeypatch.setattr(producer, "_ttl_s", 0.0)
        producer.put("req-2", torch.zeros(1))
        assert len(producer.pending_keys) == 1


class TestSpec:
    def test_spec_requires_the_option(self):
        assert handoff_connector_spec(_talker_model_config(enabled=False)) is None
        assert handoff_connector_spec(None) is None
        assert handoff_connector_spec(SimpleNamespace(stage_connector_config=None)) is None

        name, extra = handoff_connector_spec(_talker_model_config())
        assert name == "SharedMemoryConnector"
        assert extra["codec_chunk_frames"] == 25

    def test_producer_is_shared_per_connector_spec(self):
        first = get_talker_handoff_producer(_talker_model_config())
        second = get_talker_handoff_producer(_talker_model_config())
        other = get_talker_handoff_producer(_talker_model_config(inline_tensor_bytes=1))

        assert first is second
        assert other is not first
        assert get_talker_handoff_producer(_talker_model_config(enabled=False)) is None
        assert get_talker_handoff_producer(None) is None


def _thinker_output(latent: torch.Tensor, *, prompt_ids, output_ids, request_id="req-1"):
    completion = SimpleNamespace(token_ids=output_ids, text="hello", multimodal_output={"latent": latent})
    return SimpleNamespace(request_id=request_id, prompt_token_ids=prompt_ids, outputs=[completion])


class TestLlm2ttsTransport:
    def test_default_path_keeps_the_list_handoff(self, monkeypatch):
        monkeypatch.setattr(th, "create_handoff_connector", lambda spec: pytest.fail("no connector on default path"))
        latent = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        source = _thinker_output(latent, prompt_ids=[101, 102], output_ids=[11, 12])

        plain = llm2tts([source], prompt=[{}])[0]["model_intermediate_buffer"]
        not_opted_in = llm2tts([source], prompt=[{}], target_model_config=_talker_model_config(enabled=False))[0][
            "model_intermediate_buffer"
        ]

        for info in (plain, not_opted_in):
            assert info["hidden_states"]["tts"] == latent[2:4].tolist()
            assert info["ids"]["tts"] == [11, 12]
            assert info["meta"]["next_stage_prompt_len"] == 4

    def test_opted_in_handoff_rides_the_connector(self, consumer):
        latent = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        ref_waveform = torch.tensor([0.1, 0.2, 0.3])
        source = _thinker_output(latent, prompt_ids=[101, 102], output_ids=[11, 12])

        converted = llm2tts(
            [source],
            prompt=[{"multi_modal_data": {"audio": (ref_waveform, 22050)}}],
            target_model_config=_talker_model_config(),
        )[0]

        info = converted["model_intermediate_buffer"]
        marker = info["hidden_states"]["tts"]
        assert is_handoff_marker(marker)
        assert info["ids"]["tts"] == [11, 12]
        # The scheduler prompt is sized from the tensor, not the marker.
        assert converted["prompt_token_ids"] == [0, 0, 0, 0]
        assert info["meta"]["next_stage_prompt_len"] == 4
        # The small reference waveform keeps riding the request as a list.
        assert info["codes"]["ref"] == ref_waveform.tolist()
        assert info["meta"]["ref_audio_sr"] == 22050

        assert torch.equal(resolve_talker_handoff(info, consumer), latent[2:4])
        assert get_talker_handoff_producer(_talker_model_config()).pending_keys == [marker[HANDOFF_MARKER]["key"]]

    def test_put_failure_falls_back_to_the_list_handoff(self, monkeypatch):
        class _Refusing(SharedMemoryConnector):
            def put(self, *args, **kwargs):
                return False, 0, None

        monkeypatch.setattr(th, "create_handoff_connector", lambda spec: _Refusing({}))
        latent = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        source = _thinker_output(latent, prompt_ids=[101, 102], output_ids=[11, 12])

        info = llm2tts([source], prompt=[{}], target_model_config=_talker_model_config())[0][
            "model_intermediate_buffer"
        ]

        assert info["hidden_states"]["tts"] == latent[2:4].tolist()


def _talker(spec) -> MiniCPMO45OmniTTSForConditionalGeneration:
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._handoff_spec = spec
    model._handoff_connector = None
    return model


class TestTalkerModel:
    def test_talker_resolves_marker_from_its_own_connector(self, producer):
        hidden = torch.randn(3, 4)
        info = _info(producer.put("req-1", hidden))
        model = _talker(handoff_connector_spec(_talker_model_config()))

        resolved = model._resolve_connector_handoff(info)

        assert torch.equal(resolved, hidden)
        assert info["hidden_states"]["tts"] is resolved
        assert isinstance(model._handoff_connector, SharedMemoryConnector)
        model._handoff_connector.close()

    def test_talker_without_the_option_rejects_a_marker(self, producer):
        info = _info(producer.put("req-1", torch.zeros(1, 4)))
        model = _talker(None)

        with pytest.raises(ValueError, match="thinker_talker_handoff"):
            model._resolve_connector_handoff(info)
