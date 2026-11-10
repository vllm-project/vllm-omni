# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request facts use model owners, actual adapter paths and initialized data."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.entrypoints.openai.serving_capabilities import _initialized_image_geometry, _route_request
from vllm_omni.entrypoints.openai.tts_adapters.base import DEFAULT_TTS_LANGUAGES, TTSCapabilities
from vllm_omni.entrypoints.openai.tts_adapters.cosyvoice3 import CosyVoice3Adapter
from vllm_omni.entrypoints.openai.tts_adapters.kokoro import KokoroAdapter
from vllm_omni.entrypoints.openai.tts_adapters.minimax_music3 import MiniMaxMusic3Adapter
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter
from vllm_omni.request_capabilities import ImageGeometry, InputConstraint, RuntimeServingMetadata

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
    assert set(request.parameters["voice"].enum) == {"af_heart", "default"}
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
