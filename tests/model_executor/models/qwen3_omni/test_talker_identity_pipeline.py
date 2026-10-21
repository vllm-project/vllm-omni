# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Producer -> connector -> consumer finalization with independently owned configs."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from vllm.config import DeviceConfig, LoadConfig, VllmConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.core.sched.input_finalization import compose_conditioning_cache_salt
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.data_entry_keys import validate_payload
from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
    _OmniConnectorPayloadTransportMixin,
)
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder, OmniMsgpackEncoder
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.inputs.processed_media import ProcessedMediaIdentity, ProcessedMediaProvenance
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from vllm_omni.model_executor.models.qwen3_omni.talker_identity import (
    TalkerInputIdentity,
    resolve_talker_speaker,
    stage_conditioning_namespace,
)
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_full_payload
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _config(stage, *, enabled=False):
    hf = Qwen3OmniMoeConfig(talker_config={"speaker_id": {"Ethan": 7, "Chelsie": 8}, "accept_hidden_layer": 18})
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=f"actual-{stage}-weights",
            model_weights=None,
            revision="a" * 40,
            code_revision=None,
            hf_config=hf,
            dtype=torch.bfloat16,
            model_stage=stage,
            model_arch="Qwen3OmniMoeForConditionalGeneration",
            async_chunk=False,
            seed=0,
        ),
        load_config=SimpleNamespace(load_format="safetensors", model_loader_extra_config={}),
        cache_config=SimpleNamespace(enable_prefix_caching=enabled),
        compute_hash=lambda: "resolved-runtime-graph",
    )


def _source(config, *, media=True):
    hf = config.model_config.hf_config
    token = hf.thinker_config.image_token_id if media else 11
    prompt = [
        hf.im_start_token_id,
        hf.system_token_id,
        198,
        10,
        hf.im_start_token_id,
        hf.user_token_id,
        198,
        token,
        hf.im_start_token_id,
        hf.assistant_token_id,
        198,
    ]
    features = [MultiModalFeatureSpec(None, "image", "routing-label", PlaceholderRange(7, 1), "routing-label")]
    proof = ProcessedMediaProvenance(
        tuple(prompt), (ProcessedMediaIdentity("image", "b" * 64, "c" * 64, 7, 1),), ("routing-label",)
    )
    return SimpleNamespace(
        request_id="producer-A",
        prompt_token_ids=prompt,
        all_token_ids=prompt + [50, 51, 52],
        prompt_embeds=None,
        mm_features=features if media else [],
        processed_media_provenance=proof if media else None,
        lora_request=None,
    )


def _payload(config, source, transfer=None):
    if transfer is None:
        transfer = SimpleNamespace(vllm_config=config, _get_model_config=lambda: config.model_config)
    rows = len(source.all_token_ids)
    embeddings = torch.arange(rows * 4, dtype=torch.bfloat16).reshape(rows, 4)
    hidden_layer = config.model_config.hf_config.talker_config.accept_hidden_layer
    output = {
        "hidden_states.layer_0": embeddings,
        f"hidden_states.layer_{hidden_layer}": embeddings + 40,
        "embed.tts_pad": torch.full((1, 1, 4), 50, dtype=torch.bfloat16),
        "embed.tts_bos": torch.full((1, 1, 4), 60, dtype=torch.bfloat16),
        "embed.tts_eos": torch.full((1, 1, 4), 70, dtype=torch.bfloat16),
    }
    return thinker2talker_full_payload(transfer, output, source)


def _request(payload, *, req_id="consumer-A", speaker=None):
    init_none_hash(sha256)
    return OmniRequest(
        request_id=req_id,
        prompt_token_ids=[0] * payload["meta"]["next_stage_prompt_len"],
        sampling_params=SamplingParams(max_tokens=4, temperature=0),
        pooling_params=None,
        block_hasher=get_request_block_hasher(4, sha256),
        cache_salt="original-caller",
        additional_information=serialize_additional_information({"speaker": speaker}) if speaker is not None else None,
    )


def _finalize(config, payload, request=None):
    validate_payload(payload)
    received = OmniMsgpackDecoder().decode(OmniMsgpackEncoder().encode(payload))
    metadata = _OmniConnectorPayloadTransportMixin._extract_scheduling_metadata(received)
    identity = TalkerInputIdentity(config)
    coordinator = OmniSchedulingCoordinator(stage_id=1, conditioning_finalizer=identity.finalize)
    request = _request(payload) if request is None else request
    coordinator.update_request_metadata({request.request_id: request}, {request.request_id: metadata})
    return request, coordinator, metadata


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_full_payload_notice_finalizes_ids_and_conditioning_together(cache_enabled):
    source_config = _config("thinker")
    target_config = _config("talker", enabled=cache_enabled)
    payload = _payload(source_config, _source(source_config))
    request = _request(payload)
    request._omni_input_finalized = False

    request, _coordinator, metadata = _finalize(target_config, payload, request)

    assert {"next_stage_prompt_ids", "next_stage_prompt_len", "next_stage_source_digest"} <= metadata.keys()
    assert len(metadata["next_stage_prompt_ids"]) == metadata["next_stage_prompt_len"]
    assert request._omni_input_finalized
    assert request.prompt_token_ids == metadata["next_stage_prompt_ids"]
    assert request._omni_conditioning_digest is not None
    assert request.block_hashes


def test_source_and_consumer_use_their_own_configs_and_preserve_caller_salt_once(mocker):
    source_config, target_config = _config("thinker"), _config("talker")
    mocker.spy(source_config, "compute_hash")
    mocker.spy(target_config, "compute_hash")
    source = _source(source_config)
    transfer = SimpleNamespace(vllm_config=source_config, _get_model_config=lambda: source_config.model_config)
    payload = _payload(source_config, source, transfer)
    assert "next_stage_source_digest" in payload["meta"]
    assert "next_stage_conditioning_digest" not in payload["meta"]
    request, coordinator, metadata = _finalize(target_config, payload)
    assert request.cache_salt == compose_conditioning_cache_salt("original-caller", request._omni_conditioning_digest)
    assert request._omni_original_cache_salt == "original-caller"
    assert request.prompt_token_ids == payload["meta"]["next_stage_prompt_ids"]
    assert request.block_hashes
    request._block_hasher = mocker.Mock(side_effect=AssertionError("duplicate notice rehashed request"))
    coordinator.update_request_metadata({request.request_id: request}, {request.request_id: metadata})
    source.request_id = "producer-B"
    # Routing-only UUIDs can change without changing actual media identity.
    source.mm_features[0] = replace(source.mm_features[0], identifier="other-label", mm_hash="other-label")
    source.processed_media_provenance = replace(source.processed_media_provenance, routing_hashes=("other-label",))
    second = _payload(source_config, source, transfer)
    same, _, _ = _finalize(target_config, second, _request(second, req_id="consumer-B"))
    assert same.block_hashes == request.block_hashes
    assert source_config.compute_hash.call_count == 1
    assert target_config.compute_hash.call_count == 2  # one per coordinator, never per notice


@pytest.mark.parametrize(
    "change",
    [
        "source_model",
        "target_model",
        "source_revision",
        "target_revision",
        "source_dtype",
        "target_dtype",
        "projection",
        "loader",
        "source_adapter",
        "target_adapter",
        "voice",
        "context",
        "suffix",
        "media",
        "preprocessing",
    ],
)
def test_changed_dependency_cannot_cross_the_real_finalized_hash_chain(change):
    source_config, target_config = _config("thinker"), _config("talker")
    source = _source(source_config)
    original, _, _ = _finalize(target_config, _payload(source_config, source))
    if change.startswith("source_") or change.startswith("target_"):
        owner, field = change.split("_", 1)
        config = source_config if owner == "source" else target_config
        if field != "adapter":
            setattr(config.model_config, field, torch.float16 if field == "dtype" else "changed")
        elif owner == "source":
            source.lora_request = LoRARequest("loaded-source-adapter", 1, "/immutable/adapter-A")
    elif change == "projection":
        target_config.model_config.hf_config.talker_config.text_config.hidden_size += 8
    elif change == "loader":
        target_config.load_config.model_loader_extra_config["variant"] = "changed"
    elif change == "context":
        source.prompt_token_ids[3] += 1
        source.all_token_ids[3] += 1
        source.processed_media_provenance = replace(
            source.processed_media_provenance, prompt_token_ids=tuple(source.prompt_token_ids)
        )
    elif change == "suffix":
        source.all_token_ids[-1] += 1
    elif change in {"media", "preprocessing"}:
        field = "content_digest" if change == "media" else "preprocessing_digest"
        item = replace(source.processed_media_provenance.media[0], **{field: "d" * 64})
        source.processed_media_provenance = replace(source.processed_media_provenance, media=(item,))
    payload = _payload(source_config, source)
    request = _request(payload, speaker="Chelsie" if change == "voice" else None)
    if change == "target_adapter":
        request.lora_request = LoRARequest("loaded-target-adapter", 2, "/immutable/adapter-B")
    changed, _, _ = _finalize(target_config, payload, request)
    assert changed.block_hashes != original.block_hashes


@pytest.mark.parametrize(
    "voice, expected",
    [(None, 7), ("", 7), ([], 7), ([None], 7), ("unknown", 7), (" ETHAN ", 7), ([" cHeLsIe "], 8), (42, 7)],
)
def test_identity_resolves_voice_exactly_as_model_execution(voice, expected):
    identity = TalkerInputIdentity(_config("talker"))
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.tts_text_spk_token_ids = identity.speakers
    model.default_tts_text_spk_type = identity.default_speaker
    assert model._get_text_spk_token_id(voice) == expected
    assert resolve_talker_speaker(voice, identity.speakers, identity.default_speaker) == expected


def test_identity_default_voice_is_independent_of_checkpoint_key_order():
    config = _config("talker")
    config.model_config.hf_config.talker_config.speaker_id = {"Chelsie": 8, "Ethan": 7}
    identity = TalkerInputIdentity(config)
    assert identity.default_speaker == "ethan"
    assert resolve_talker_speaker(None, identity.speakers, identity.default_speaker) == 7


def test_payload_voice_overrides_request_voice_and_missing_payload_voice_uses_request():
    source_config, target_config = _config("thinker"), _config("talker")
    payload = _payload(source_config, _source(source_config, media=False))
    expected, _, _ = _finalize(target_config, payload, _request(payload, speaker=["Chelsie"]))
    received = deepcopy(payload)
    received["speaker"] = " cHeLsIe "
    actual, _, _ = _finalize(target_config, received, _request(received, speaker=["Ethan"]))
    assert actual.block_hashes == expected.block_hashes
    received["speaker"] = None
    default, _, _ = _finalize(target_config, received, _request(received, speaker=["Chelsie"]))
    ethan, _, _ = _finalize(target_config, payload)
    assert default.block_hashes == ethan.block_hashes


@pytest.mark.parametrize("kind", ["missing", "forged", "stale", "embeddings", "reload"])
def test_unverified_source_is_not_accepted_in_cache_enabled_identity_mode(kind):
    source_config, target_config = _config("thinker"), _config("talker", enabled=True)
    source = _source(source_config)
    if kind == "missing":
        source.processed_media_provenance = None
    elif kind == "forged":
        source.processed_media_provenance = {"trusted": True}
    elif kind == "stale":
        source.processed_media_provenance = replace(source.processed_media_provenance, prompt_token_ids=(1,))
    elif kind == "embeddings":
        source.prompt_embeds = torch.ones(len(source.prompt_token_ids), 4)
    else:
        source.lora_request = LoRARequest("changed-in-place", 1, "/adapter", load_inplace=True)
    payload = _payload(source_config, source)
    assert "next_stage_source_digest" not in payload["meta"]
    assert payload["meta"]["next_stage_source_identity_error"]
    request = _request(payload)
    old_ids, old_hashes = list(request.prompt_token_ids), list(request.block_hashes)
    with pytest.raises(ValueError, match="verified complete source"):
        _finalize(target_config, payload, request)
    assert request.prompt_token_ids == old_ids and request.block_hashes == old_hashes
    assert request.cache_salt == "original-caller"
    assert not getattr(request, "_omni_input_finalized", False)


def test_conflicting_target_digest_is_rejected_before_installation():
    source_config, target_config = _config("thinker"), _config("talker")
    payload = _payload(source_config, _source(source_config))
    payload["meta"]["next_stage_conditioning_digest"] = "f" * 64
    request = _request(payload)
    with pytest.raises(ValueError, match="actual Talker namespace"):
        _finalize(target_config, payload, request)
    assert request.cache_salt == "original-caller"
    assert not getattr(request, "_omni_input_finalized", False)


def test_namespace_includes_resolved_weight_source_and_dummy_seed():
    config = _config("thinker")
    original = stage_conditioning_namespace(config)
    config.model_config.model_weights = "other-resolved-weight-source"
    assert stage_conditioning_namespace(config) != original
    config.load_config.load_format = "dummy"
    dummy = stage_conditioning_namespace(config)
    config.model_config.seed += 1
    assert stage_conditioning_namespace(config) != dummy


@pytest.mark.parametrize("stage", ["thinker", "talker"])
def test_namespace_with_real_offline_resolved_runtime_config(tmp_path, stage):
    hf = Qwen3OmniMoeConfig()
    hf.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
    hf.save_pretrained(tmp_path)
    model = OmniModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        model_stage=stage,
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        hf_config_name=f"{stage}_config",
        dtype="bfloat16",
        max_model_len=128,
        enforce_eager=True,
    )
    config = VllmConfig(
        model_config=model, device_config=DeviceConfig(device="cpu"), load_config=LoadConfig(load_format="dummy")
    )
    original = stage_conditioning_namespace(config)
    graph_hash = config.compute_hash()
    assert stage_conditioning_namespace(config) == original
    config.load_config.model_loader_extra_config = {"variant": "different-weights"}
    # Loading options do not enter the upstream graph hash; our static
    # conditioning namespace must still separate these configurations.
    assert config.compute_hash() == graph_hash
    assert stage_conditioning_namespace(config) != original


@pytest.mark.parametrize("architecture_source", ["override", "resolved", "hf"])
def test_scheduler_installs_identity_using_actual_target_configuration(architecture_source):
    config = _config("talker")
    config.model_config.stage_id = 1
    config.model_config.requires_full_payload_input = True
    if architecture_source != "override":
        config.model_config.model_arch = None
        owner = config.model_config if architecture_source == "resolved" else config.model_config.hf_config
        owner.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
    scheduler = OmniSchedulerMixin()
    scheduler.vllm_config = config
    scheduler._init_omni_io_scheduling_state()
    callback = scheduler.input_coordinator._conditioning_finalizer
    assert isinstance(callback.__self__, TalkerInputIdentity)
    assert callback.__self__.model_namespace == stage_conditioning_namespace(config)


def test_late_length_notice_does_not_rehash_or_discard_verified_conditioning(mocker):
    source_config, target_config = _config("thinker"), _config("talker", enabled=True)
    payload = _payload(source_config, _source(source_config))
    request, coordinator, _ = _finalize(target_config, payload)
    before = (request.cache_salt, list(request.block_hashes), request._omni_conditioning_digest)
    request._block_hasher = mocker.Mock(side_effect=AssertionError("late notice rehashed input"))
    coordinator.update_request_metadata(
        {request.request_id: request}, {request.request_id: {"next_stage_prompt_len": request.num_prompt_tokens}}
    )
    assert (request.cache_salt, request.block_hashes, request._omni_conditioning_digest) == before
    with pytest.raises(ValueError, match="conflicting finalized prompt length"):
        coordinator.update_request_metadata(
            {request.request_id: request},
            {request.request_id: {"next_stage_prompt_len": request.num_prompt_tokens + 1}},
        )


def test_length_notice_cannot_finalize_an_unverified_input():
    source_config, target_config = _config("thinker"), _config("talker", enabled=True)
    payload = _payload(source_config, _source(source_config))
    request = _request(payload)
    identity = TalkerInputIdentity(target_config)
    coordinator = OmniSchedulingCoordinator(conditioning_finalizer=identity.finalize)
    with pytest.raises(ValueError, match="verified complete source"):
        coordinator.update_request_metadata(
            {request.request_id: request},
            {request.request_id: {"next_stage_prompt_len": request.num_prompt_tokens + 1}},
        )
    assert not getattr(request, "_omni_input_finalized", False)
    assert request.cache_salt == "original-caller"
