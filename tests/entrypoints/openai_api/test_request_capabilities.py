# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request facts use model owners, actual adapter paths and initialized data."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_capabilities import _initialized_image_geometry, _route_request
from vllm_omni.entrypoints.openai.tts_adapters.audex import AudexAdapter
from vllm_omni.entrypoints.openai.tts_adapters.audex_tta import AudexTTAAdapter
from vllm_omni.entrypoints.openai.tts_adapters.base import DEFAULT_TTS_LANGUAGES, SpeechServingContext, TTSCapabilities
from vllm_omni.entrypoints.openai.tts_adapters.cosyvoice3 import CosyVoice3Adapter
from vllm_omni.entrypoints.openai.tts_adapters.gepard import GepardAdapter
from vllm_omni.entrypoints.openai.tts_adapters.kokoro import KokoroAdapter
from vllm_omni.entrypoints.openai.tts_adapters.minimax_music3 import MiniMaxMusic3Adapter
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter
from vllm_omni.request_capabilities import (
    ImageGeometry,
    InputConstraint,
    ParameterConstraint,
    RequestConstraints,
    RuntimeServingMetadata,
)
from vllm_omni.serving_capabilities import VIDEO_TO_VIDEO

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _profile(architecture, operation, profile_id):
    declaration = get_diffusion_model_metadata(architecture).serving_capabilities
    return next(profile for profile in declaration.get(operation).profiles if profile.id == profile_id)


def _route(profile, path, **overrides):
    options = dict(
        stage_count=1, state=SimpleNamespace(), geometry=ImageGeometry(), image_limit=None, native_diffusion=True
    )
    options.update(overrides)
    return _route_request(profile, path, **options)


def test_request_facts_preserve_pipeline_dataclass_json_serialization():
    declaration = get_diffusion_model_metadata("Flux2KleinPipeline").serving_capabilities
    encoded = json.loads(json.dumps(asdict(declaration)))
    assert encoded["operations"][2]["profiles"][0]["request"]["inputs"]["image"]["max_count"] == 4
    with pytest.raises(ValidationError):
        InputConstraint(min_count=2, max_count=1)
    with pytest.raises(ValidationError):
        ImageGeometry(state="known")


def test_image_model_count_is_narrowed_by_deployment_and_geometry_stays_unknown():
    profile = _profile("Flux2KleinPipeline", "image_edit", "image_to_image")
    effective = _route(
        profile,
        "/v1/images/edits",
        image_limit=1,
        state=SimpleNamespace(args=SimpleNamespace(max_generated_image_size=500000)),
    )
    assert profile.request.inputs["image"].max_count == 4
    assert effective.inputs["image"].max_count == 1
    assert effective.max_output_pixels == 500000
    assert effective.mask.geometry.state == "unknown"
    assert effective.mask.source_resize.max_pixels == 1048576
    assert effective.parameters["width"].maximum is None


def test_adapter_backend_and_multistage_have_distinct_native_mask_contracts():
    profile = _profile("Flux2KleinPipeline", "image_edit", "image_to_image")
    external = _route(profile, "/v1/images/edits", native_diffusion=False)
    assert external.inputs["mask_image"].support == "unknown"
    assert external.mask is None
    multistage = _route(profile, "/v1/images/edits", stage_count=2)
    assert multistage.inputs["mask_image"].support == "unsupported"


def test_video_transport_does_not_claim_offline_last_frame_or_widen_fixed_frames():
    wan = _profile("WanImageToVideoPipeline", "video_generation", "image_to_video")
    assert wan.request.inputs["last_frame"].support == "supported"
    assert _route(wan, "/v1/videos").inputs["last_frame"].support == "unsupported"
    magi = _route(_profile("Magi2Pipeline", "video_generation", "text_to_video"), "/v1/videos/sync")
    assert magi.parameters["num_frames"].enum == (125,)
    assert magi.parameters["seconds"].enum == (10.0,)
    assert magi.parameters["fps"].enum == (12.5,)


@pytest.mark.parametrize("architecture", ["WanImageToVideoPipeline", "Wan22I2VPipeline", "WanI2VDMD2Pipeline"])
def test_wan_first_frame_describes_existing_required_image(architecture):
    profile = _profile(architecture, "video_generation", "image_to_video")
    assert profile.required_inputs == ("text", "image")
    request = _route(profile, "/v1/videos")
    assert request.inputs["image"].min_count == 1
    assert request.inputs["first_frame"] == InputConstraint(support="supported", min_count=0, max_count=1)


@pytest.mark.parametrize("architecture", ["WanVACEPipeline", "Wan22VACEPipeline"])
@pytest.mark.parametrize("profile_id", ["text_to_video", "image_to_video", "video_to_video"])
def test_vace_reference_inputs_do_not_claim_fixed_first_frame(architecture, profile_id):
    profile = _profile(architecture, "video_generation", profile_id)
    request = _route(profile, "/v1/videos")
    assert profile.support == "supported"
    assert request.inputs["first_frame"] == InputConstraint(support="unsupported", max_count=0)
    for role in profile.required_inputs:
        assert request.inputs[role] == InputConstraint(support="supported", min_count=1)


@pytest.mark.parametrize("flag", [True, False, None])
@pytest.mark.parametrize("support", ["supported", "unknown"])
def test_video_edit_roles_use_handler_policy_without_model_name(flag, support):
    roles = ("source_video", "video_noise_mask", "audio_noise_mask")
    profile = replace(
        VIDEO_TO_VIDEO,
        support=support,
        request=RequestConstraints(
            inputs={role: InputConstraint(support="supported", min_count=0, max_count=1) for role in roles}
        ),
    )
    state = SimpleNamespace(openai_serving_video=SimpleNamespace(supports_latent_mask_editing=flag))
    request = _route(profile, "/v1/videos", state=state)
    for role in roles:
        expected = (
            InputConstraint(support="unsupported", max_count=0) if flag is False else profile.request.inputs[role]
        )
        assert request.inputs[role] == expected
    assert profile.support == support


@pytest.mark.parametrize("flag", [True, False, None])
def test_unknown_video_without_request_keeps_only_proven_route_facts(flag):
    profile = replace(VIDEO_TO_VIDEO, support="unknown")
    state = SimpleNamespace(openai_serving_video=SimpleNamespace(supports_latent_mask_editing=flag))
    request = _route(profile, "/v1/videos", state=state)
    assert request.inputs["last_frame"].support == "unsupported"
    assert request.parameters["width"].minimum == 1
    for role in ("source_video", "video_noise_mask", "audio_noise_mask"):
        if flag is False:
            assert request.inputs[role].support == "unsupported"
        else:
            assert role not in request.inputs


@pytest.mark.parametrize("image_support,maximum", [("unknown", 3), ("supported", 1), ("unsupported", 0)])
def test_unknown_realtime_video_keeps_input_evidence_and_transport_limits(image_support, maximum):
    profile = replace(
        VIDEO_TO_VIDEO,
        support="unknown",
        request=RequestConstraints(
            inputs={
                "image": InputConstraint(support=image_support, min_count=0, max_count=maximum),
                "first_frame": InputConstraint(support="unknown", max_count=1),
                "source_video": InputConstraint(support="supported", min_count=0, max_count=1),
            }
        ),
    )
    request = _route(profile, "/v1/realtime/video")
    assert request.inputs["image"].support == image_support
    assert request.inputs["image"].max_count == min(maximum, 1)
    assert request.inputs["first_frame"].support == "unknown"
    for role in ("video", "audio", "last_frame", "source_video", "video_noise_mask", "audio_noise_mask"):
        assert request.inputs[role] == InputConstraint(support="unsupported", max_count=0)


def test_unknown_nonvideo_projection_is_unchanged():
    profile = replace(_profile("Flux2KleinPipeline", "image_edit", "image_to_image"), support="unknown")
    assert _route(profile, "/v1/images/edits") is None


def test_flux3_profiles_distinguish_optional_chat_images_and_required_edits():
    chat = _profile("Flux3Pipeline", "chat", "image_response")
    chat_request = _route(chat, "/v1/chat/completions")
    assert chat_request.inputs["image"].min_count == 0
    assert chat_request.inputs["image"].max_count == 10
    assert chat_request.parameters["num_outputs_per_prompt"].enum == (1,)
    edit = _profile("Flux3Pipeline", "image_edit", "image_to_image")
    edit_request = _route(edit, "/v1/images/edits", image_limit=2)
    assert edit.request.inputs["image"].max_count == 10
    assert edit_request.inputs["image"].min_count == 1
    assert edit_request.inputs["image"].max_count == 2
    assert edit_request.inputs["mask_image"].support == "unsupported"
    assert edit_request.inputs["reference_image"].support == "unsupported"
    assert edit_request.parameters["n"].enum == (1,)
    generation = _route(_profile("Flux3Pipeline", "image_generation", "text_to_image"), "/v1/images/generations")
    assert generation.inputs["image"].support == "unsupported"
    for request in (chat_request, edit_request, generation):
        assert request.parameters["width"].step == 16
        assert request.parameters["height"].step == 16
        # The loaded package chooses CFG or flow-map sampling. Discovery
        # cannot infer acceptance of these overrides from a model name.
        assert "guidance_scale" not in request.parameters
        assert "negative_prompt" not in request.parameters


def test_geometry_requires_agreement_of_all_current_initialized_replicas():
    def replica(multiple):
        return SimpleNamespace(
            serving_metadata=RuntimeServingMetadata(image_geometry=ImageGeometry(state="known", multiple_of=multiple))
        )

    pool = SimpleNamespace(stage_client=SimpleNamespace(final_output_type="image"), clients=[replica(24)])
    engine = SimpleNamespace(stage_pools=[pool])
    assert _initialized_image_geometry(engine).multiple_of == 24
    pool.clients.append(replica(32))
    assert _initialized_image_geometry(engine).state == "unknown"
    pool.clients[1] = SimpleNamespace()
    assert _initialized_image_geometry(engine).state == "unknown"
    pool.clients[1] = replica(24)
    assert _initialized_image_geometry(engine).multiple_of == 24


@pytest.mark.parametrize("adapter_type", [MiniMaxMusic3Adapter, Yue2Adapter])
def test_music_adapter_declares_lyrics_and_excludes_reference_conditioning(adapter_type):
    adapter = object.__new__(adapter_type)
    request = adapter.get_request_constraints()
    assert request.inputs["instructions"].min_count == 1
    assert request.inputs["reference_audio"].support == "unsupported"
    assert request.parameters["speed"].enum == (1.0,)
    assert request.parameters["language"].support == "unsupported"


def test_cosyvoice_requires_voice_or_reference_audio_and_transcript():
    request = object.__new__(CosyVoice3Adapter).get_request_constraints()
    assert request.required_any_of == (("voice", "reference_audio"), ("voice", "reference_text"))
    assert "language" not in request.parameters


def test_kokoro_uses_loaded_voices_and_does_not_promote_generic_language_defaults():
    adapter = object.__new__(KokoroAdapter)
    adapter.capabilities = TTSCapabilities(supported_speakers=frozenset({"af_heart"}))
    request = adapter.get_request_constraints()
    assert set(request.parameters["voice"].enum) == {"af_heart", "default", ""}
    assert "language" not in request.parameters


@pytest.mark.parametrize("variant", ["Base", "CustomVoice", "VoiceDesign"])
def test_qwen_deployed_variant_owns_reference_requirements_and_validated_languages(variant):
    adapter = object.__new__(Qwen3TTSAdapter)
    adapter.capabilities = TTSCapabilities()
    adapter.ctx = SimpleNamespace(
        server=SimpleNamespace(
            engine_client=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=SimpleNamespace(tts_model_type=variant))
            )
        )
    )
    request = adapter.get_request_constraints()
    assert request.parameters["task_type"].enum == (variant,)
    assert set(request.parameters["language"].enum) == DEFAULT_TTS_LANGUAGES
    if variant == "Base":
        assert request.required_any_of == (("voice", "reference_audio", "speaker_embedding"),)
    if variant == "VoiceDesign":
        assert request.inputs["instructions"].min_count == 1


@pytest.mark.parametrize("variant", [None, "tts", "realtime", "local", "ttsd", "sound_effect", "voice_generator"])
def test_moss_requirements_follow_initialized_variant_without_loading_processors(variant):
    from vllm_omni.entrypoints.openai.tts_adapters.moss_tts import MossTTSAdapter

    adapter = object.__new__(MossTTSAdapter)
    adapter._moss_variant = variant
    request = adapter.get_request_constraints()
    if variant == "sound_effect":
        assert request.inputs["ambient_sound"].min_count == 1
        assert request.inputs["text"].min_count == 0
    elif variant == "voice_generator":
        assert request.inputs["instructions"].min_count == 1
    else:
        assert request.required_any_of == (("voice", "reference_audio"),)
    if variant == "ttsd":
        assert request.inputs["reference_audio_2"].max_count == 1


@pytest.mark.parametrize("encoder_loaded,expected", [(None, "unknown"), (False, "unsupported"), (True, "supported")])
def test_voxtral_reference_encoder_is_never_discovered_by_query_rpc(encoder_loaded, expected):
    from vllm_omni.entrypoints.openai.tts_adapters.voxtral import VoxtralTTSAdapter

    adapter = object.__new__(VoxtralTTSAdapter)
    adapter._encoder_loaded = encoder_loaded
    request = adapter.get_request_constraints()
    assert request.inputs["reference_audio"].support == expected


def test_kokoro_default_alias_requires_its_actual_target_voice():
    adapter = object.__new__(KokoroAdapter)
    adapter.capabilities = TTSCapabilities(supported_speakers=frozenset({"bf_emma"}))
    assert "default" not in adapter.get_request_constraints().parameters["voice"].enum


@pytest.mark.parametrize(
    "adapter_type,parameter,value,normalization",
    [
        (Qwen3TTSAdapter, "language", "english", "title"),
        (KokoroAdapter, "voice", "AF_HEART", "lower"),
        (GepardAdapter, "voice", "DEFAULT", "lower"),
        (AudexAdapter, "voice", " DeFaUlT ", "strip_lower"),
        (AudexTTAAdapter, "voice", " DeFaUlT ", "strip_lower"),
        (MiniMaxMusic3Adapter, "voice", " DeFaUlT ", "strip_lower"),
        (Yue2Adapter, "voice", " DeFaUlT ", "strip_lower"),
    ],
)
def test_speech_enum_matching_follows_actual_adapter_validation(
    monkeypatch, adapter_type, parameter, value, normalization
):
    adapter = object.__new__(adapter_type)
    adapter.capabilities = TTSCapabilities(supported_speakers=frozenset({"af_heart"}), default_speaker="af_heart")
    if adapter_type is Qwen3TTSAdapter:
        adapter.ctx = SpeechServingContext(
            server=SimpleNamespace(
                engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace())),
                uploaded_speakers={},
                _get_available_speakers=lambda: {"af_heart"},
            )
        )
    if adapter_type is KokoroAdapter:
        monkeypatch.setattr("vllm_omni.entrypoints.openai.tts_adapters.kokoro.missing_g2p_package", lambda _: None)
    kwargs = {parameter: value}
    if adapter_type in (MiniMaxMusic3Adapter, Yue2Adapter):
        kwargs["instructions"] = "acoustic folk"
    request = OpenAICreateSpeechRequest(input="hello", **kwargs)
    assert adapter.validate(request) is None
    constraint = adapter.get_request_constraints().parameters[parameter]
    assert constraint.enum_normalization == normalization


@pytest.mark.parametrize("adapter_type", [AudexAdapter, AudexTTAAdapter, MiniMaxMusic3Adapter, Yue2Adapter])
@pytest.mark.parametrize("voice", [None, "", "  ", "default", " DeFaUlT ", "invalid"])
def test_placeholder_voice_enum_includes_the_accepted_empty_value(adapter_type, voice):
    adapter = object.__new__(adapter_type)
    kwargs = {"instructions": "acoustic folk"} if adapter_type in (MiniMaxMusic3Adapter, Yue2Adapter) else {}
    accepted = adapter.validate(OpenAICreateSpeechRequest(input="hello", voice=voice, **kwargs)) is None
    assert accepted == (voice != "invalid")
    constraint = adapter.get_request_constraints().parameters["voice"]
    assert constraint.enum_normalization == "strip_lower"
    assert accepted == ((voice or "").strip().lower() in constraint.enum)


@pytest.mark.parametrize(
    "fields",
    [
        {"type": "number", "enum": (1,)},
        {"type": "string", "enum": None},
        {"type": "string", "enum": (1,)},
    ],
)
def test_enum_normalization_requires_a_string_enum(fields):
    with pytest.raises(ValidationError):
        ParameterConstraint(enum_normalization="lower", **fields)


def test_enum_normalization_is_optional_and_serializes_without_changing_enum():
    exact = ParameterConstraint(type="string", enum=("CustomVoice",))
    assert "enum_normalization" not in exact.model_dump(exclude_none=True)
    normalized = ParameterConstraint(type="string", enum=("English",), enum_normalization="title")
    assert normalized.model_dump()["enum"] == ["English"]
    assert normalized.model_dump()["enum_normalization"] == "title"
    with pytest.raises(ValidationError):
        ParameterConstraint.model_validate({"type": "string", "enum": ["English"], "enum_normalization": "future"})


@pytest.mark.parametrize("adapter_type", [KokoroAdapter, GepardAdapter])
@pytest.mark.parametrize("voice", [None, "", "  ", "default", "DEFAULT", " default ", "AF_HEART", "invalid"])
def test_lowercase_voice_enums_preserve_each_adapters_empty_and_whitespace_policy(monkeypatch, adapter_type, voice):
    adapter = object.__new__(adapter_type)
    adapter.capabilities = TTSCapabilities(supported_speakers=frozenset({"af_heart"}))
    if adapter_type is KokoroAdapter:
        monkeypatch.setattr("vllm_omni.entrypoints.openai.tts_adapters.kokoro.missing_g2p_package", lambda _: None)
    accepted = adapter.validate(OpenAICreateSpeechRequest(input="hello", voice=voice)) is None
    constraint = adapter.get_request_constraints().parameters["voice"]
    assert constraint.enum_normalization == "lower"
    assert accepted == (voice is None or voice.lower() in constraint.enum)
