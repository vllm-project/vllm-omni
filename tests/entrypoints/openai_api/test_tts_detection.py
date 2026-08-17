# SPDX-License-Identifier: Apache-2.0
"""TTS model detection: registry-driven stage -> model-type resolution.

``serving_speech.py`` used to carry a hand-written 20-branch ladder mapping
``model_stage``/``model_arch`` to a model-type string, duplicating the
``stage_keys`` every adapter already declared. Detection is now derived from the
adapters. ``test_detection_matches_legacy_ladder`` pins the new implementation
against a verbatim copy of the ladder it replaced, over every input the ladder
could distinguish, so the migration is checkable rather than reviewable-by-eye.
"""

import itertools

import pytest

from vllm_omni.entrypoints.openai.tts_adapters import (
    TTS_ADAPTER_REGISTRY,
    all_tts_stage_keys,
    detect_tts_model_type,
    iter_tts_detectors,
    resolve_adapter,
    tts_entry_stage_archs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# --------------------------------------------------------------------------
# The legacy ladder, copied verbatim from serving_speech.py at 67c54777b
# (the commit this refactor is based on), with the module constants inlined.
# Do not "clean up" this function: its value is being a frozen oracle.
# --------------------------------------------------------------------------
_LEGACY_MING_TTS_MODEL_ARCHS = {"MingTTSForConditionalGeneration"}
_LEGACY_VOXTRAL_TTS_MODEL_STAGES = {"audio_generation"}
_LEGACY_QWEN3_TTS_MODEL_STAGES = {"qwen3_tts"}
_LEGACY_FISH_TTS_MODEL_STAGES = {"fish_speech_slow_ar"}
_LEGACY_COSYVOICE3_TTS_MODEL_STAGES = {"cosyvoice3_talker"}
_LEGACY_OMNIVOICE_TTS_MODEL_STAGES = {"omnivoice_generator"}
_LEGACY_COVO_AUDIO_MODEL_STAGES = {"fused_thinker_talker"}
_LEGACY_VOXCPM2_TTS_MODEL_STAGES = {"latent_generator"}
_LEGACY_MING_TTS_MODEL_STAGES = {"ming_tts"}
_LEGACY_MOSS_TTS_MODEL_STAGES = {"moss_tts_nano"}
_LEGACY_MOSS_TTS_FULL_MODEL_STAGES = {"moss_tts", "moss_tts_codec"}
_LEGACY_MOSS_TTS_LOCAL_MODEL_STAGES = {"moss_tts_local", "moss_tts_local_codec"}
_LEGACY_HIGGS_AUDIO_V2_TTS_MODEL_STAGES = {"higgs_audio_v2"}
_LEGACY_HIGGS_V3_TTS_MODEL_STAGES = {"higgs_audio_v3"}
_LEGACY_GLM_TTS_MODEL_STAGES = {"glm_tts"}
_LEGACY_STEP_AUDIO2_TTS_MODEL_STAGES = {"step_audio2_thinker"}
_LEGACY_INDEXTTS2_TTS_MODEL_STAGES = {"indextts2_talker"}
_LEGACY_AUDEX_TTS_MODEL_STAGES = {"audex_thinker", "audex_omni"}
_LEGACY_AUDEX_TTA_MODEL_STAGES = {"audex_tta_thinker"}

_LEGACY_TTS_MODEL_STAGES = (
    _LEGACY_VOXTRAL_TTS_MODEL_STAGES
    | _LEGACY_QWEN3_TTS_MODEL_STAGES
    | _LEGACY_FISH_TTS_MODEL_STAGES
    | _LEGACY_COSYVOICE3_TTS_MODEL_STAGES
    | _LEGACY_OMNIVOICE_TTS_MODEL_STAGES
    | _LEGACY_HIGGS_AUDIO_V2_TTS_MODEL_STAGES
    | _LEGACY_HIGGS_V3_TTS_MODEL_STAGES
    | _LEGACY_COVO_AUDIO_MODEL_STAGES
    | _LEGACY_VOXCPM2_TTS_MODEL_STAGES
    | _LEGACY_MING_TTS_MODEL_STAGES
    | _LEGACY_MOSS_TTS_MODEL_STAGES
    | _LEGACY_MOSS_TTS_FULL_MODEL_STAGES
    | _LEGACY_MOSS_TTS_LOCAL_MODEL_STAGES
    | _LEGACY_GLM_TTS_MODEL_STAGES
    | _LEGACY_STEP_AUDIO2_TTS_MODEL_STAGES
    | _LEGACY_INDEXTTS2_TTS_MODEL_STAGES
    | _LEGACY_AUDEX_TTS_MODEL_STAGES
    | _LEGACY_AUDEX_TTA_MODEL_STAGES
)


def _legacy_detect(model_stage, model_arch):
    if model_arch == "VoxCPM2TalkerForConditionalGeneration":
        return "voxcpm2"
    if model_stage in _LEGACY_QWEN3_TTS_MODEL_STAGES:
        return "qwen3_tts"
    if model_stage in _LEGACY_VOXTRAL_TTS_MODEL_STAGES:
        return "voxtral_tts"
    if model_stage in _LEGACY_FISH_TTS_MODEL_STAGES:
        return "fish_tts"
    if model_stage in _LEGACY_COSYVOICE3_TTS_MODEL_STAGES:
        return "cosyvoice3"
    if model_stage in _LEGACY_OMNIVOICE_TTS_MODEL_STAGES:
        return "omnivoice"
    if model_stage in _LEGACY_COVO_AUDIO_MODEL_STAGES:
        if model_arch and "CovoAudio" in model_arch:
            return "covo_audio"
    if model_stage in _LEGACY_VOXCPM2_TTS_MODEL_STAGES:
        return "voxcpm2"
    if model_stage in _LEGACY_MING_TTS_MODEL_STAGES:
        return "ming_flash_omni_tts"
    if model_arch in _LEGACY_MING_TTS_MODEL_ARCHS:
        return "ming_tts"
    if model_stage in _LEGACY_MOSS_TTS_MODEL_STAGES:
        return "moss_tts_nano"
    if model_stage in _LEGACY_MOSS_TTS_FULL_MODEL_STAGES:
        return "moss_tts"
    if model_stage in _LEGACY_MOSS_TTS_LOCAL_MODEL_STAGES:
        return "moss_tts"
    if model_stage in _LEGACY_HIGGS_AUDIO_V2_TTS_MODEL_STAGES:
        return "higgs_audio_v2"
    if model_stage in _LEGACY_HIGGS_V3_TTS_MODEL_STAGES:
        return "higgs_audio_v3"
    if model_stage in _LEGACY_GLM_TTS_MODEL_STAGES:
        return "glm_tts"
    if model_stage in _LEGACY_STEP_AUDIO2_TTS_MODEL_STAGES:
        return "step_audio2"
    if model_stage in _LEGACY_INDEXTTS2_TTS_MODEL_STAGES:
        return "indextts2"
    if model_stage in _LEGACY_AUDEX_TTS_MODEL_STAGES:
        return "audex"
    if model_stage in _LEGACY_AUDEX_TTA_MODEL_STAGES:
        return "audex_tta"
    return None


# Every ``model_stage`` value any in-tree pipeline emits, plus any stage key an
# adapter claims, so the parametrization below covers the real deployment space
# rather than a hand-picked subset. Refresh the pipeline half with:
#   grep -rhoE 'model_stage="[^"]+"' --include='pipeline*.py' vllm_omni/model_executor/ \
#     | sed 's/model_stage="//;s/"$//' | sort -u
# ``test_pipeline_stage_list_is_complete`` enforces the adapter half.
_PIPELINE_STAGES = [
    "AR",
    "ar",
    "asr",
    "audex_code2wav",
    "audex_omni",
    "audex_thinker",
    "audex_tta_thinker",
    "audex_xcodec",
    "audio_generation",
    "audio_tokenizer",
    "audio_vae",
    "aura",
    "code2wav",
    "cosyvoice3_code2wav",
    "cosyvoice3_talker",
    "dac_decoder",
    "diffusion",
    "dit",
    "fish_speech_slow_ar",
    "fused_thinker_talker",
    "glm_tts",
    "glm_tts_dit",
    "higgs_audio_v2",
    "higgs_audio_v3",
    "indextts2_5_talker",
    "indextts2_s2mel_decoder",
    "indextts2_talker",
    "latent_generator",
    "llm",
    "ming_tts",
    "moss_tts",
    "moss_tts_codec",
    "moss_tts_local",
    "moss_tts_local_codec",
    "moss_tts_nano",
    # Claimed by the OmniVoice adapter but emitted by no pipeline: OmniVoice is
    # served through the diffusion path (``OmniOpenAIServingSpeech.for_diffusion``
    # sets ``_tts_model_type`` directly and leaves ``_tts_stage`` None), so
    # detection is never consulted for it. The key is vestigial in the adapter
    # exactly as it was in the deleted ladder; kept here so detection is still
    # exercised on it.
    "omnivoice_generator",
    "qwen3_tts",
    "step_audio2_thinker",
    "talker",
    "thinker",
    "token2audio",
    "token2image",
    "token2text",
    "token2wav",
    "tts",
]

_STAGES = [*_PIPELINE_STAGES, None, "vae", "not_a_real_stage"]
_ARCHS = [
    None,
    "VoxCPM2TalkerForConditionalGeneration",
    "MingTTSForConditionalGeneration",
    "CovoAudioForConditionalGeneration",
    "MyCovoAudioThing",
    "Qwen3TTSForConditionalGeneration",
]

#: Architectures that no adapter claims. Paired with any stage key these are
#: inert, so the full cross product is meaningful for them.
_UNCLAIMED_ARCHS = [None, "Qwen3TTSForConditionalGeneration"]

#: Architectures a detector matches on. Pairing one of these with *another*
#: model's stage key describes a deployment no pipeline can produce (the
#: architecture comes from the checkpoint, the stage key from the pipeline
#: definition), and the two implementations deliberately disagree there —
#: see ``test_arch_matching_is_a_fallback_not_an_override``.
_CLAIMED_ARCH_CASES = [
    # Ming dense: its AR stage is the generic ``llm``.
    ("llm", "MingTTSForConditionalGeneration"),
    ("audio_vae", "MingTTSForConditionalGeneration"),
    (None, "MingTTSForConditionalGeneration"),
    # Ming flash-omni owns the ``ming_tts`` stage key and must win it.
    ("ming_tts", "MingTTSForConditionalGeneration"),
    # VoxCPM2: talker architecture plus its own stage key.
    ("latent_generator", "VoxCPM2TalkerForConditionalGeneration"),
    (None, "VoxCPM2TalkerForConditionalGeneration"),
    # CoVo shares ``fused_thinker_talker`` with non-CoVo fused deployments.
    ("fused_thinker_talker", "CovoAudioForConditionalGeneration"),
    ("fused_thinker_talker", "MyCovoAudioThing"),
    ("fused_thinker_talker", None),
    ("fused_thinker_talker", "Qwen3TTSForConditionalGeneration"),
]

# The frozen ladder can only be compared over stages it knew about. New
# adapter-owned stages are covered by their adapter tests and by the registry
# completeness checks below.
_NEW_ADAPTER_STAGE_KEYS = all_tts_stage_keys() - _LEGACY_TTS_MODEL_STAGES
_LEGACY_REACHABLE_STAGES = [stage for stage in _STAGES if stage not in _NEW_ADAPTER_STAGE_KEYS]
_REACHABLE = list(itertools.product(_LEGACY_REACHABLE_STAGES, _UNCLAIMED_ARCHS)) + _CLAIMED_ARCH_CASES


@pytest.mark.parametrize(("model_stage", "model_arch"), _REACHABLE)
def test_detection_matches_legacy_ladder(model_stage, model_arch):
    """Registry detection is identical to the ladder it replaced.

    Covers every stage the frozen ladder knew about, plus non-TTS pipeline
    stages and negatives, against every architecture no adapter claims. New
    adapter-owned stages are tested by their adapter suites. Pairing a claimed
    architecture with an unrelated model's stage key is excluded because no
    pipeline can produce it, and is covered separately by
    ``test_arch_matching_is_a_fallback_not_an_override``.
    """
    assert detect_tts_model_type(model_stage, model_arch) == _legacy_detect(model_stage, model_arch)


def test_arch_matching_is_a_fallback_not_an_override():
    """The one deliberate divergence from the ladder, pinned.

    The ladder tested Ming's architecture at a fixed position partway down, so a
    Ming-architecture model deployed under *another* model's stage key resolved
    to ``ming_tts`` if that key was checked below Ming, and to the other model if
    above. Detection now treats an architecture-only match as a fallback that
    runs after every adapter owning a real stage key, so the stage key always
    wins. That is both self-consistent and unreachable in practice: a
    checkpoint's architecture and a pipeline's stage key cannot be mixed across
    models.
    """
    assert detect_tts_model_type("audex_omni", "MingTTSForConditionalGeneration") == "audex"
    assert _legacy_detect("audex_omni", "MingTTSForConditionalGeneration") == "ming_tts"
    # Where the two real Ming deployments live, they agree.
    assert detect_tts_model_type("llm", "MingTTSForConditionalGeneration") == "ming_tts"
    assert detect_tts_model_type("ming_tts", "MingTTSForConditionalGeneration") == "ming_flash_omni_tts"


def test_stage_keys_cover_legacy_stage_set():
    """No stage key was dropped when the module constants were deleted."""
    assert _LEGACY_TTS_MODEL_STAGES <= all_tts_stage_keys()


def test_pipeline_stage_list_is_complete():
    """``_PIPELINE_STAGES`` must still list every stage key adapters claim.

    Guards the parametrization above from silently narrowing: a new adapter
    whose stage key is missing here would be detected but never exercised.
    Regenerate the list from the pipelines (see the comment beside it) when a
    new model lands.
    """
    missing = all_tts_stage_keys() - set(_PIPELINE_STAGES)
    assert not missing, f"stage keys claimed by adapters but absent from _PIPELINE_STAGES: {sorted(missing)}"


def test_entry_stage_archs_is_ming_only():
    """Only Ming dense identifies its entry stage by architecture.

    Widening this set would change which stage ``_find_tts_stage`` selects in
    mixed deployments — notably VoxCPM2, which declares ``model_archs`` but is
    found by its ``latent_generator`` stage key.
    """
    assert tts_entry_stage_archs() == frozenset({"MingTTSForConditionalGeneration"})


def test_every_detected_type_is_adapter_backed():
    """Detection may only name a model that has a registered adapter."""
    for stage, arch in itertools.product(_STAGES, _ARCHS):  # full cross product on purpose
        detected = detect_tts_model_type(stage, arch)
        if detected is None:
            continue
        assert detected in TTS_ADAPTER_REGISTRY, (
            f"detection produced undeclared model type {detected!r} for stage={stage!r} arch={arch!r}"
        )


def test_no_ambiguous_detectors_at_equal_priority():
    """Two detectors that can match the same deployment must be ordered.

    This is what makes ``detect_tts_model_type`` independent of adapter import
    order: any real overlap has to be resolved by an explicit
    ``detect_priority``, not by whichever module happened to register first.
    """
    detectors = iter_tts_detectors()
    for left, right in itertools.combinations(detectors, 2):
        if left.detect_priority != right.detect_priority:
            continue
        for stage, arch in itertools.product(_STAGES, _ARCHS):
            assert not (left.matches(stage, arch) and right.matches(stage, arch)), (
                f"{left.name!r} and {right.name!r} both match stage={stage!r} arch={arch!r} "
                f"at priority {left.detect_priority}; give one an explicit detect_priority"
            )


def test_detection_order_is_total_and_stable():
    """Resolution order must not depend on registry insertion order."""
    keys = [(d.detect_priority, d.name) for d in iter_tts_detectors()]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_adapters_declare_at_least_one_match_rule():
    """An adapter with neither stage keys nor architectures is unreachable."""
    for name, cls in TTS_ADAPTER_REGISTRY.items():
        assert cls.stage_keys or cls.model_archs, f"adapter {name!r} declares no stage_keys and no model_archs"


def test_audex_omni_requires_speech_decoder():
    """The Audex omni thinker only serves speech alongside its decoder."""
    audex = resolve_adapter("audex")
    assert audex.stage_serves_speech("audex_omni", frozenset({"audex_omni", "audex_code2wav"})) is True
    assert audex.stage_serves_speech("audex_omni", frozenset({"audex_omni"})) is False
    # The dedicated TTS thinker is unconditional.
    assert audex.stage_serves_speech("audex_thinker", frozenset({"audex_thinker"})) is True


def test_default_stage_serves_speech_is_unconditional():
    for name, cls in TTS_ADAPTER_REGISTRY.items():
        if name == "audex":
            continue
        for key in cls.stage_keys:
            assert cls.stage_serves_speech(key, frozenset({key})) is True
