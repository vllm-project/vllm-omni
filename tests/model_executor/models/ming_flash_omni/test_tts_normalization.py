# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.model_executor.models.ming_flash_omni import text_processing
from vllm_omni.model_executor.models.ming_flash_omni.tts_request import resolve_generation_params
from vllm_omni.model_executor.stage_input_processors.ming_talker import (
    build_ming_talker_additional_info,
    build_ming_talker_inputs,
)
from vllm_omni.transformers_utils.configs.ming_talker import MingFlashOmniTalkerConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _profile_config() -> MingFlashOmniTalkerConfig:
    root = Path(__file__).resolve().parents[4]
    deploy = load_deploy_config(root / "vllm_omni/deploy/ming3_audio_flash.yaml")
    overrides = deploy.stages[1].engine_extras["hf_overrides"]
    return MingFlashOmniTalkerConfig.from_dict({"normalize_text": False}, **overrides)


def test_selected_profile_defaults_reach_talker_without_request_overrides():
    config = _profile_config()
    info = build_ming_talker_additional_info("金额12.5元。", None)
    params = resolve_generation_params(info, config)
    assert params.normalize_text is True
    assert params.tn_mode == "upstream"
    assert "tn_mode" not in info
    assert resolve_generation_params(info, MingFlashOmniTalkerConfig()).tn_mode == "legacy"


@pytest.mark.parametrize(
    ("profile", "low_latency"),
    [("ming3_audio_flash.yaml", True), ("ming3_audio_tiny.yaml", False)],
)
def test_ming3_profiles_select_their_audio_collection_policy(profile, low_latency):
    root = Path(__file__).resolve().parents[4]
    deploy = load_deploy_config(root / "vllm_omni/deploy" / profile)
    overrides = deploy.stages[1].engine_extras["hf_overrides"]
    assert overrides["low_latency_streaming"] is low_latency
    assert MingFlashOmniTalkerConfig.from_dict({}, **overrides).low_latency_streaming is low_latency
    # Shared Ming2 and strict checkpoint loading retain their polling default.
    assert MingFlashOmniTalkerConfig().low_latency_streaming is False
    assert MingFlashOmniTalkerConfig(strict_weight_loading=True).low_latency_streaming is False


@pytest.mark.parametrize("enabled", [False, True])
def test_talker_forwards_collection_policy_to_each_streaming_job(mocker, enabled):
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.ming_flash_omni import ming_flash_omni_talker as talker

    model = mocker.Mock()
    model._stream_jobs = {}
    model.device = torch.device("cpu")
    model.config = MingFlashOmniTalkerConfig(low_latency_streaming=enabled)
    model.audio_vae.config.sample_rate = 24000
    create = mocker.patch.object(talker, "MingStreamingJob")
    create.return_value.next_audio.return_value = (torch.ones(1, 1, 4), False)
    talker.MingFlashOmniTalkerForConditionalGeneration._stream_forward(
        model, "request", None, None, {}, SimpleNamespace(seed=None)
    )
    assert create.call_args.kwargs == {"low_latency_streaming": enabled}


def test_explicit_legacy_and_disabled_normalization_override_profile():
    info = build_ming_talker_additional_info(
        "原始文字。", {"normalize_text": False, "tn_mode": "legacy", "stream_decode": False, "seed": 123}
    )
    params = resolve_generation_params(info, _profile_config())
    assert params.normalize_text is False
    assert params.tn_mode == "legacy"
    assert params.stream_decode is False
    assert params.seed == 123


def test_invalid_normalization_mode_is_rejected():
    with pytest.raises(ValueError, match="tn_mode"):
        resolve_generation_params({"tn_mode": "unsupported"}, _profile_config())
    with pytest.raises(ValueError, match="tn_mode"):
        MingFlashOmniTalkerConfig(tn_mode="unsupported")


def test_sync_language_only_text_does_not_leak_the_language_name():
    from types import SimpleNamespace

    outputs = [SimpleNamespace(outputs=[SimpleNamespace(text="  Chinese\t")])]
    prompts = build_ming_talker_inputs(outputs)
    assert prompts[0]["additional_information"]["text"] == ""


def test_upstream_mode_does_not_silently_fall_back_without_checkpoint_normalizer():
    from vllm_omni.model_executor.models.ming_flash_omni.native_text_frontend import normalize_native_fragment

    with pytest.raises(ValueError, match="requires a normalizer loaded from the Talker checkpoint"):
        text_processing.segment_and_normalize("日期2026年10月16日。", tn_mode="upstream")
    with pytest.raises(ValueError, match="requires a normalizer loaded from the Talker checkpoint"):
        normalize_native_fragment("日期2026年10月16日。", tn_mode="upstream")


def test_markup_control_is_independent_and_preserves_legacy_defaults():
    legacy = MingFlashOmniTalkerConfig()
    converted = MingFlashOmniTalkerConfig(scrub_markup=True)
    assert resolve_generation_params({}, legacy).scrub_markup is False
    assert resolve_generation_params({"normalize_text": False}, converted).scrub_markup is True
    assert resolve_generation_params({"scrub_markup": False}, converted).scrub_markup is False
    info = build_ming_talker_additional_info("text", {"scrub_markup": False})
    assert info["scrub_markup"] is False
    with pytest.raises(ValueError, match="scrub_markup"):
        resolve_generation_params({"scrub_markup": "false"}, converted)


@pytest.mark.parametrize("normalize", [False, True])
def test_native_fragments_are_normalized_once_without_receiver_resegmentation(mocker, normalize):
    from functools import partial
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.ming_flash_omni import ming_flash_omni_talker as talker

    model = mocker.Mock()
    model._tts_segments = partial(talker.MingFlashOmniTalkerForConditionalGeneration._tts_segments, model)
    model._project_spk_emb.return_value = None
    model.audio_generator.generate_latents.return_value = []
    model.audio_generator.duration_capped_steps.return_value = 4
    normalizer = mocker.patch.object(talker, "normalize_native_fragment", side_effect=lambda text, *args, **kw: text)
    resegment = mocker.patch.object(talker, "segment_and_normalize")
    build_input = mocker.patch.object(talker, "build_tts_input", return_value=(torch.empty(0), None))
    voice = SimpleNamespace(
        spk_emb=None, already_projected=False, prompt_text=None, prompt_wav_emb=None, prompt_wav_lat=None
    )
    params = resolve_generation_params({"normalize_text": normalize}, MingFlashOmniTalkerConfig())
    result = talker.MingFlashOmniTalkerForConditionalGeneration._generate_latents(
        model,
        input_ids=torch.tensor([0]),
        inputs_embeds=None,
        text="原始文本",
        params=params,
        voice=voice,
        native_fragments=["第一段。", "第二段。"],
    )
    assert result == []
    assert normalizer.call_count == (2 if normalize else 0)
    assert [call.kwargs["text"] for call in build_input.call_args_list] == ["第一段。", "第二段。"]
    resegment.assert_not_called()
    model.reset_mock()
    assert (
        talker.MingFlashOmniTalkerForConditionalGeneration._generate_latents(
            model,
            input_ids=torch.tensor([0]),
            inputs_embeds=None,
            text="",
            params=params,
            voice=voice,
            native_fragments=[],
        )
        == []
    )
    model.audio_generator.generate_latents.assert_not_called()
    model.model.get_input_embeddings.assert_not_called()
