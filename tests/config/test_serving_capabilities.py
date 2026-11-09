# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model contracts are explicit and owned by deployable pipeline definitions."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from transformers import Qwen3OmniMoeConfig

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import StageConfig
from vllm_omni.diffusion.capability_resolvers import MINIMAX_H3_CAPABILITIES, resolve_minimax_h3_capabilities
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS
from vllm_omni.model_executor.models.audex.pipeline import (
    AUDEX_THINKER_ONLY_PIPELINE,
    AUDEX_TTA_PIPELINE,
    AUDEX_TTS_PIPELINE,
)
from vllm_omni.model_executor.models.mammoth_moda2.pipeline import MAMMOTH_MODA2_PIPELINE
from vllm_omni.model_executor.models.minimax_music3.pipeline import MINIMAX_MUSIC3_PIPELINE
from vllm_omni.serving_capabilities import (
    OPERATIONS,
    TEXT_TO_IMAGE,
    OperationCapability,
    ServingCapabilities,
    declare_capabilities,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _profile(capabilities, operation, profile_id):
    return next(profile for profile in capabilities.get(operation).profiles if profile.id == profile_id)


def _assert_owned_declaration(capabilities):
    assert capabilities is not None
    assert {operation.id for operation in capabilities.operations} == set(OPERATIONS)
    assert capabilities.sources
    for source in capabilities.sources:
        assert not Path(source).is_absolute()
        assert (REPO_ROOT / source).is_file(), source
    for operation in capabilities.operations:
        assert operation.reason_code
        assert all(profile.reason_code for profile in operation.profiles)


@pytest.mark.parametrize("model_type", sorted(OMNI_PIPELINES))
def test_all_registered_native_entrypoints_declare_contracts(model_type):
    hf_config = Qwen3OmniMoeConfig() if model_type == "qwen3_omni_moe" else None
    pipeline = resolve_pipeline_config(model_type, hf_config)
    assert pipeline is not None
    _assert_owned_declaration(pipeline.serving_capabilities)


@pytest.mark.parametrize("architecture", sorted(_DIFFUSION_MODELS))
def test_all_diffusion_registry_aliases_reach_a_declaration(architecture):
    _assert_owned_declaration(get_diffusion_model_metadata(architecture).serving_capabilities)


@pytest.mark.parametrize(
    ("architecture", "operation", "profile_id", "expected"),
    [
        ("Cosmos3OmniDiffusersPipeline", "image_generation", "text_to_image", "supported"),
        ("Cosmos3OmniDiffusersPipeline", "video_generation", "image_to_video", "supported"),
        ("QwenImagePipeline", "image_edit", "image_to_image", "unsupported"),
        ("QwenImageEditPipeline", "image_generation", "text_to_image", "unsupported"),
        ("QwenImageEditPipeline", "image_edit", "image_to_image", "supported"),
        ("WanImageToVideoPipeline", "video_generation", "text_to_video", "unsupported"),
        ("WanImageToVideoPipeline", "video_generation", "image_to_video", "supported"),
        ("WanPipeline", "video_generation", "image_to_video", "unknown"),
        ("StableAudioPipeline", "speech", "text_to_speech", "unsupported"),
        ("StableAudioPipeline", "audio_generation", "text_to_audio", "supported"),
        ("GlmImagePipeline", "image_generation", "text_to_image", "unsupported"),
        ("Sam3Pipeline", "image_edit", "image_to_image", "unsupported"),
        ("DiffusersAdapterPipeline", "image_generation", "text_to_image", "unknown"),
    ],
)
def test_contracts_do_not_follow_default_output_type(architecture, operation, profile_id, expected):
    declaration = get_diffusion_model_metadata(architecture).serving_capabilities
    assert _profile(declaration, operation, profile_id).support == expected


def test_image_and_audio_outputs_do_not_imply_editing_or_tts():
    assert MAMMOTH_MODA2_PIPELINE.serving_capabilities.get("image_edit").support == "unsupported"
    music = MINIMAX_MUSIC3_PIPELINE.serving_capabilities
    assert _profile(music, "speech", "lyrics_to_music").support == "supported"
    assert _profile(music, "speech", "text_to_speech").support == "unsupported"
    tta = AUDEX_TTA_PIPELINE.serving_capabilities
    assert _profile(tta, "speech", "caption_to_audio").support == "supported"
    assert _profile(tta, "speech", "text_to_speech").support == "unsupported"
    assert tta.get("audio_generation").support == "unsupported"


def test_declarations_preserve_existing_metadata_overrides_and_defaults():
    # Discovery must not change task/handler initialization through unrelated
    # metadata fields, including entries that previously resolved via aliases.
    assert get_diffusion_model_metadata("LTX2TwoStagePipeline").final_output_type == "video"
    assert get_diffusion_model_metadata("BooguImageTurboPipeline").max_multimodal_image_inputs == 1
    assert get_diffusion_model_metadata("LingBotWorldCausalDMDPipeline").final_output_type == "video"
    assert get_diffusion_model_metadata("Wan22VACEPipeline").final_output_type is None
    assert (
        get_diffusion_model_metadata("Wan22VACEPipeline").serving_capabilities.get("video_generation").support
        == "supported"
    )


def test_selected_pipeline_variant_controls_speech_support():
    assert AUDEX_THINKER_ONLY_PIPELINE.serving_capabilities.get("speech").support == "unsupported"
    assert AUDEX_TTS_PIPELINE.serving_capabilities.get("speech").support == "supported"
    for enabled in (False, True):
        config = Qwen3OmniMoeConfig(enable_audio_output=enabled)
        pipeline = resolve_pipeline_config("qwen3_omni_moe", config)
        declaration = pipeline.serving_capabilities
        expected = "supported" if enabled else "unsupported"
        assert _profile(declaration, "chat", "audio_response").support == expected
        # Both Omni variants use chat; the speech adapter rejects talker-only TTS.
        assert declaration.get("speech").support == "unsupported"


def test_missing_plugin_declaration_is_unknown_and_partial_stays_partial():
    assert get_diffusion_model_metadata("ThirdPartyPipeline").serving_capabilities is None
    partial = ServingCapabilities((OperationCapability("image_generation", "supported", (TEXT_TO_IMAGE,)),))
    assert partial.get("image_generation").support == "supported"
    assert partial.get("image_edit").support == "unknown"
    assert partial.get("image_edit").reason_code == "declaration_missing"


def test_declaration_rejects_contradictory_ownership():
    with pytest.raises(ValueError, match="belongs to"):
        ServingCapabilities((OperationCapability("chat", "supported", (TEXT_TO_IMAGE,)),))
    with pytest.raises(ValueError, match="Unsupported operation"):
        ServingCapabilities((OperationCapability("image_generation", "unsupported", (TEXT_TO_IMAGE,)),))
    with pytest.raises(ValueError, match="unique IDs"):
        declare_capabilities(TEXT_TO_IMAGE, replace(TEXT_TO_IMAGE), sources=())


@pytest.mark.parametrize(
    "task_type,excluded",
    [
        (None, set()),
        ("auto", set()),
        ("combined", set()),
        ("fl2va", {"video_to_video"}),
        ("t2va", {"video_to_video"}),
        ("ref2va", {"text_to_video"}),
    ],
)
def test_h3_partition_exclusions_do_not_invent_worker_release_tasks(task_type, excluded):
    declaration = resolve_minimax_h3_capabilities(
        MINIMAX_H3_CAPABILITIES, [StageConfig(0, "diffusion", yaml_engine_args={"task_type": task_type}).to_omegaconf()]
    )
    for profile in declaration.get("video_generation").profiles:
        assert profile.support == ("unsupported" if profile.id in excluded else "unknown")
        assert profile.reason_code == (
            "startup_partition_excludes_profile" if profile.id in excluded else "worker_checkpoint_tasks_unavailable"
        )
        assert profile.outputs == ("video", "audio")


def test_cosmos_common_modes_and_worker_only_scheduler_limits():
    declaration = get_diffusion_model_metadata("Cosmos3OmniDiffusersPipeline").serving_capabilities
    assert _profile(declaration, "image_generation", "text_to_image").support == "supported"
    assert _profile(declaration, "video_generation", "image_to_video").support == "supported"
    for profile_id in ("text_to_video", "video_to_video"):
        profile = _profile(declaration, "video_generation", profile_id)
        assert profile.support == "unknown"
        assert profile.reason_code == "worker_scheduler_mode_unavailable"
    assert dict(_profile(declaration, "video_generation", "image_to_video").constraints)["optional_outputs"] == (
        "audio",
    )


@pytest.mark.parametrize(
    "architecture,profile_id",
    [
        ("LTX2Pipeline", "text_to_video"),
        ("LTX2DistilledTwoStagePipeline", "image_to_video"),
        ("WanS2VPipeline", "image_to_video"),
    ],
)
def test_joint_video_audio_outputs_are_not_reduced_to_container_type(architecture, profile_id):
    declaration = get_diffusion_model_metadata(architecture).serving_capabilities
    assert _profile(declaration, "video_generation", profile_id).outputs == ("video", "audio")


def test_startup_refiner_is_optional_serializable_and_uses_resolved_stages():
    from vllm_omni.config.stage_config import PipelineConfig
    from vllm_omni.engine.omni_engine_base import OmniEngineBase
    from vllm_omni.model_executor.models.minimax_h3.pipeline import MINIMAX_H3_PIPELINE

    engine = object.__new__(OmniEngineBase)
    engine.stage_configs = [StageConfig(0, "diffusion", yaml_engine_args={"task_type": "ref2va"}).to_omegaconf()]
    engine._set_pipeline_runtime_config(MINIMAX_H3_PIPELINE, None)
    assert _profile(engine.serving_capabilities, "video_generation", "text_to_video").support == "unsupported"
    assert _profile(MINIMAX_H3_PIPELINE.serving_capabilities, "video_generation", "text_to_video").support == "unknown"
    assert json.loads(json.dumps(asdict(MINIMAX_H3_PIPELINE)))["serving_capabilities_resolver"]
    plugin = PipelineConfig(model_type="third_party", model_arch="ThirdParty", stages=MINIMAX_H3_PIPELINE.stages)
    engine._set_pipeline_runtime_config(plugin, None)
    assert all(engine.serving_capabilities.get(operation).support == "unknown" for operation in OPERATIONS)


@pytest.mark.parametrize(
    "model,profile_id,inputs",
    [
        ("OpenMOSS-Team/MOSS-TTS", "text_to_speech", ("text",)),
        ("OpenMOSS-Team/MOSS-SoundEffect", "caption_to_audio", ("ambient_sound",)),
        ("OpenMOSS-Team/MOSS-VoiceGenerator", "text_to_speech", ("text", "instructions")),
    ],
)
def test_moss_shared_topology_preserves_distinct_speech_contracts(model, profile_id, inputs):
    from vllm_omni.model_executor.models.moss_tts.pipeline import MOSS_TTS_PIPELINE, resolve_moss_capabilities

    declaration = resolve_moss_capabilities(
        MOSS_TTS_PIPELINE.serving_capabilities,
        [StageConfig(0, "moss_tts", yaml_engine_args={"model": model}).to_omegaconf()],
    )
    profile = _profile(declaration, "speech", profile_id)
    assert profile.support == "supported" and profile.required_inputs == inputs
    if profile_id == "caption_to_audio":
        assert _profile(declaration, "speech", "text_to_speech").support == "unsupported"


@pytest.mark.parametrize(
    "model_type",
    ["aura_omni", "joyai_vl_interaction", "joyai_vl_interaction_asr", "ming_flash_omni", "bailing_moe_v3_audio"],
)
def test_downstream_talker_does_not_establish_standalone_speech_contract(model_type):
    declaration = resolve_pipeline_config(model_type).serving_capabilities
    assert declaration.get("speech").support == "unsupported"
    assert _profile(declaration, "chat", "audio_response").support == "supported"
